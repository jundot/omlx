# SPDX-License-Identifier: Apache-2.0
"""BaseEngine-compatible proxy for an isolated MLX distributed job."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx

from ..cluster.deployment import ClusterDeployment
from ..cluster.launch import DistributedJobSupervisor, DistributedLaunchError
from ..cluster.liveness import check_peers, describe_failure
from ..reasoning_effort import _fallback_candidate, _normalized_input
from .base import GenerationOutput
from .batched import BatchedEngine

logger = logging.getLogger(__name__)

# How long one per-rank marker health read stays authoritative. Every request
# preflights the cluster, so this bounds both the added latency (one SSH read
# per peer, paid once per window) and how long a half-dead cluster can keep
# answering 200s before requests start failing cleanly (#2708).
_PEER_HEALTH_TTL = 10.0


def _reasoning_effort_retry_payloads(
    payload: dict[str, Any], detail: str
) -> list[dict[str, Any]]:
    """Payload variants to retry after rank-zero rejects ``reasoning_effort``.

    Local engines render the chat template in-process, so they can try the
    requested value, catch a template error, and fall back
    (``apply_chat_template_with_reasoning_effort_fallback`` in
    ``reasoning_effort.py``). The distributed engine cannot: only rank-zero's
    private mlx-lm server renders the template, and it already told us — via
    this failed response — which value it rejected. This mirrors the same
    alias-then-drop fallback ladder reactively, at the HTTP boundary.

    Returns at most three payloads (normalized value, alias fallback, then
    reasoning_effort dropped entirely), so a client that always sends an
    unsupported value can never turn into an unbounded retry loop. Returns
    ``[]`` when the failure is not about reasoning_effort, or there is
    nothing to retry.
    """

    if "reasoning effort" not in detail.lower():
        return []
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        return []
    value = chat_template_kwargs.get("reasoning_effort")
    if value is None:
        return []

    variants: list[dict[str, Any]] = []

    def _variant(effort: Any) -> dict[str, Any]:
        retry = dict(payload)
        retry["chat_template_kwargs"] = {
            **chat_template_kwargs,
            "reasoning_effort": effort,
        }
        return retry

    # Local engines normalize ("High" -> "high") before their first render
    # attempt (reasoning_effort.py), so the normalized tier must come first
    # here too or the same request behaves differently on a cluster.
    normalized = _normalized_input(value)
    if normalized != value:
        variants.append(_variant(normalized))
    candidate = _fallback_candidate(normalized)
    if candidate is not None and candidate != normalized:
        variants.append(_variant(candidate))
    logger.info(
        "rank-zero rejected reasoning_effort=%r; retrying with %s, then "
        "without it",
        value,
        [
            var["chat_template_kwargs"]["reasoning_effort"]
            for var in variants
        ],
    )

    dropped_kwargs = {
        key: val
        for key, val in chat_template_kwargs.items()
        if key != "reasoning_effort"
    }
    dropped = dict(payload)
    if dropped_kwargs:
        dropped["chat_template_kwargs"] = dropped_kwargs
    else:
        dropped.pop("chat_template_kwargs", None)
    variants.append(dropped)
    return variants


class DistributedInferenceError(RuntimeError):
    """A bounded error surfaced when the private rank-zero backend fails."""


class DistributedBatchedEngine(BatchedEngine):
    """Keep oMLX's API/tokenizer layer while proxying model work to MLX ranks."""

    # The coordinator does not own a local Scheduler: each rank process
    # creates and enforces its own prefill guard from the signed deployment.
    # This marker prevents ProcessMemoryEnforcer from reporting that expected
    # topology as a broken wrapper chain.
    _prefill_memory_guard_managed_externally = True

    def __init__(
        self,
        deployment: ClusterDeployment,
        *,
        stream_interval: int = 1,
        enable_thinking: bool | None = None,
        model_settings: Any | None = None,
        python_executable: str | None = None,
        cwd: Path | None = None,
        load_timeout: float = 1800.0,
        request_read_timeout: float | None = None,
    ) -> None:
        if request_read_timeout is None:
            raw = os.environ.get("OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT", "300.0")
            try:
                request_read_timeout = float(raw)
            except ValueError:
                raise ValueError(
                    "OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT must be a number, "
                    f"got {raw!r}"
                ) from None
        if not math.isfinite(request_read_timeout) or request_read_timeout <= 0:
            raise ValueError(
                "distributed request read timeout must be a finite positive "
                f"number, got {request_read_timeout!r}"
            )
        super().__init__(
            model_name=deployment.model,
            trust_remote_code=deployment.trust_remote_code,
            stream_interval=stream_interval,
            enable_thinking=enable_thinking,
            model_settings=model_settings,
        )
        launch_kwargs: dict[str, Any] = {
            "cwd": cwd,
            "load_timeout": load_timeout,
        }
        if python_executable is not None:
            launch_kwargs["python_executable"] = python_executable
        self.deployment = deployment
        self._request_read_timeout = request_read_timeout
        self._supervisor = DistributedJobSupervisor(
            deployment,
            **launch_kwargs,
        )
        self._client: httpx.AsyncClient | None = None
        self._model_type: str | None = None
        self._active_requests = 0
        self._active_lock = asyncio.Lock()
        self._peer_health: tuple[float, bool, str] | None = None
        self._peer_health_lock = asyncio.Lock()

    def _new_client(self, endpoint: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=endpoint,
            # MLX-LM emits SSE keepalives while a long prompt is processing, so
            # a read timeout is an inactivity bound rather than a total request
            # deadline. A rank stalled in a collective no longer holds the
            # public oMLX request open forever.
            timeout=httpx.Timeout(
                connect=10.0,
                read=self._request_read_timeout,
                write=30.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=32,
                max_keepalive_connections=8,
            ),
        )

    @property
    def model_type(self) -> str | None:
        return self._model_type

    @property
    def prefix_cache_enabled(self) -> bool:
        # MLX-LM owns a rank-local LRU prompt cache. It is intentionally not
        # exposed as oMLX's block-aware cache because those formats differ.
        return False

    def cluster_status(self) -> dict[str, Any]:
        """Return the bounded launcher/rank state used by the admin UI."""

        return self._supervisor.status().to_dict()

    async def start(self) -> None:
        if self._loaded:
            return
        self._validate_model_settings()

        # Tokenizer/config metadata stays in the oMLX process. No model weights
        # are loaded here.
        from mlx_lm.utils import _download, load_config, load_tokenizer

        metadata_path = await asyncio.to_thread(
            _download,
            self.deployment.model,
            allow_patterns=[
                "*.json",
                "*.py",
                "tokenizer.model",
                "*.tiktoken",
                "tiktoken.model",
                "*.txt",
                "*.jsonl",
                "*.jinja",
            ],
        )
        config = await asyncio.to_thread(load_config, metadata_path)
        self._model_type = config.get("model_type")
        self._tokenizer = await asyncio.to_thread(
            load_tokenizer,
            metadata_path,
            {"trust_remote_code": self.deployment.trust_remote_code},
            config.get("eos_token_id"),
        )

        try:
            await asyncio.to_thread(self._supervisor.start)
        except Exception:
            self._tokenizer = None
            self._model_type = None
            raise
        endpoint = self._supervisor.endpoint
        if endpoint is None:
            await asyncio.to_thread(self._supervisor.stop)
            raise DistributedLaunchError("distributed endpoint was not created")
        self._client = self._new_client(endpoint)
        self._loaded = True
        logger.info(
            "Distributed engine ready: model=%s ranks=%d backend=%s plan=%s",
            self.deployment.model,
            self.deployment.world_size,
            self.deployment.backend,
            self.deployment.plan_hash[:16],
        )

    def _validate_model_settings(self) -> None:
        settings = self._model_settings
        if settings is None:
            return
        incompatible = [
            name
            for name in (
                "dflash_enabled",
                "specprefill_enabled",
                "mtp_enabled",
                "vlm_mtp_enabled",
                "turboquant_kv_enabled",
            )
            if bool(getattr(settings, name, False))
        ]
        if incompatible:
            raise ValueError(
                "distributed inference cannot be combined with "
                + ", ".join(incompatible)
            )

    async def stop(self) -> None:
        client, self._client = self._client, None
        try:
            if client is not None:
                await client.aclose()
        finally:
            try:
                await asyncio.to_thread(self._supervisor.stop)
            finally:
                self._tokenizer = None
                self._model_type = None
                self._loaded = False
        logger.info("Distributed engine stopped: %s", self.deployment.deployment_id)

    def _ensure_available(self) -> httpx.AsyncClient:
        client = self._client
        status = self._supervisor.status()
        if not self._loaded or client is None:
            raise DistributedInferenceError("distributed engine is not loaded")
        if status.returncode is not None:
            detail = status.failure_reason or ""
            tail = " · ".join(
                line.strip() for line in status.stderr_tail[-3:] if line.strip()
            )[:1000]
            if tail and tail not in detail:
                detail = f"{detail} · Worker log: {tail}" if detail else tail
            suffix = f": {detail}" if detail else ""
            raise DistributedInferenceError(
                f"distributed job exited with code {status.returncode}{suffix}"
            )
        return client

    def _read_timeout_error(self, *, stream: bool) -> DistributedInferenceError:
        """Reconcile an inactive private HTTP request with launcher state."""

        status = self._supervisor.status()
        kind = "stream" if stream else "request"
        if status.returncode is not None:
            detail = f"distributed job exited with code {status.returncode}"
        elif status.failure_reason:
            detail = status.failure_reason
        else:
            detail = (
                f"no rank-zero data for {self._request_read_timeout:g}s "
                f"while the cluster was {status.phase}"
            )
        return DistributedInferenceError(
            f"rank-zero inference {kind} timed out: {detail}"
        )

    async def _transport_failure_error(
        self,
        exc: httpx.HTTPError,
        *,
        stream: bool,
    ) -> DistributedInferenceError:
        """Turn a closed private socket back into the rank failure that caused it.

        The HTTP connection usually disappears a few milliseconds before
        mlx.launch publishes the peer-lost event or exit code. Give the reader
        thread a bounded moment to catch up, then surface that evidence instead
        of the useless transport class name ``RemoteProtocolError``.
        """

        status = self._supervisor.status()
        for _ in range(5):
            if (
                status.failure_reason
                or status.returncode is not None
                or status.stderr_tail
            ):
                break
            await asyncio.sleep(0.05)
            status = self._supervisor.status()
        if status.failure_reason:
            detail = status.failure_reason
            tail = " · ".join(
                line.strip() for line in status.stderr_tail[-3:] if line.strip()
            )[:1000]
            if tail and tail not in detail:
                detail += f" · Worker log: {tail}"
        elif status.returncode is not None:
            tail = " · ".join(
                line.strip() for line in status.stderr_tail[-3:] if line.strip()
            )
            detail = f"distributed job exited with code {status.returncode}"
            if tail:
                detail += f": {tail[:1000]}"
        else:
            detail = (
                f"rank-zero connection closed while the cluster was "
                f"{status.phase} ({type(exc).__name__})"
            )
        kind = "stream" if stream else "request"
        return DistributedInferenceError(f"rank-zero inference {kind} failed: {detail}")

    @staticmethod
    def _validate_request_features(kwargs: dict[str, Any]) -> None:
        if kwargs.get("compiled_grammar") is not None:
            raise ValueError(
                "guided grammar is not yet supported by distributed inference"
            )
        if kwargs.get("logit_bias"):
            raise ValueError("logit_bias is not yet supported by distributed inference")
        if kwargs.get("specprefill") is True:
            raise ValueError("SpecPrefill is not supported by distributed inference")

    def _completion_payload(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        presence_penalty: float,
        stop: list[str] | None,
        stream: bool,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        self._validate_request_features(kwargs)
        if (
            kwargs.get("seed") is not None
            and self.deployment.execution.sampling_rank_only
        ):
            raise ValueError(
                "seeded single-request generation is incompatible with the "
                "experimental sampling-rank-only output path"
            )
        payload: dict[str, Any] = {
            "model": "default_model",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "xtc_probability": kwargs.get("xtc_probability", 0.0),
            "xtc_threshold": kwargs.get("xtc_threshold", 0.1),
            "stream": stream,
        }
        repetition_context_size = kwargs.get("repetition_context_size")
        if repetition_context_size is not None:
            # Widens mlx-lm's look-back window for the repetition penalty
            # (default 20 tokens). Verbatim loop units longer than that
            # window never overlap their own penalty context, so the
            # penalty is inert no matter its value.
            payload["repetition_context_size"] = repetition_context_size
        if stop:
            payload["stop"] = stop
        if kwargs.get("seed") is not None:
            payload["seed"] = kwargs["seed"]
        chat_template_kwargs = dict(kwargs.get("chat_template_kwargs") or {})
        # MLX-LM's private server reads thinking budgets from chat_template_kwargs
        # on the request body, not from a top-level field. Fold it in so the rank
        # sees it and can build its budget processor per request.
        if kwargs.get("thinking_budget") is not None:
            chat_template_kwargs["thinking_budget"] = kwargs["thinking_budget"]
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _chat_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        presence_penalty: float,
        stop: list[str] | None,
        stream: bool,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a real chat request for the private rank-zero server.

        Distributed chat used to render the prompt in the coordinator and send
        it through ``/v1/completions``. MLX-LM can detect a tool boundary on
        that route, but text-completion choices have nowhere to carry the
        parsed ``tool_calls`` array. Keeping the request as chat all the way to
        rank zero preserves oMLX's model protocol adapter and the structured
        response.
        """

        self._validate_request_features(kwargs)
        if (
            kwargs.get("seed") is not None
            and self.deployment.execution.sampling_rank_only
        ):
            raise ValueError(
                "seeded single-request generation is incompatible with the "
                "experimental sampling-rank-only output path"
            )
        payload: dict[str, Any] = {
            "model": "default_model",
            "messages": self._backend_chat_messages(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "xtc_probability": kwargs.get("xtc_probability", 0.0),
            "xtc_threshold": kwargs.get("xtc_threshold", 0.1),
            "stream": stream,
        }
        repetition_context_size = kwargs.get("repetition_context_size")
        if repetition_context_size is not None:
            # See _completion_payload: widens the penalty look-back window.
            payload["repetition_context_size"] = repetition_context_size
        if tools:
            payload["tools"] = tools
        if stop:
            payload["stop"] = stop
        if kwargs.get("seed") is not None:
            payload["seed"] = kwargs["seed"]
        chat_template_kwargs = dict(kwargs.get("chat_template_kwargs") or {})
        # MLX-LM's private server reads thinking budgets from chat_template_kwargs
        # on the request body, not from a top-level field. Fold it in so the rank
        # sees it and can build its budget processor per request.
        if kwargs.get("thinking_budget") is not None:
            chat_template_kwargs["thinking_budget"] = kwargs["thinking_budget"]
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _backend_chat_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Bridge oMLX-native history to MLX-LM's HTTP input contract.

        oMLX prepares native chat templates by parsing historical
        ``function.arguments`` JSON into dictionaries. MLX-LM's HTTP server
        performs that same conversion itself and unconditionally calls
        ``json.loads``. Serialize only that boundary field back to the OpenAI
        wire format so the private server parses it exactly once.
        """

        prepared: list[dict[str, Any]] = []
        for message in messages:
            copied = dict(message)
            raw_calls = message.get("tool_calls")
            if isinstance(raw_calls, list):
                copied_calls: list[Any] = []
                for raw_call in raw_calls:
                    if not isinstance(raw_call, dict):
                        copied_calls.append(raw_call)
                        continue
                    call = dict(raw_call)
                    raw_function = raw_call.get("function")
                    if isinstance(raw_function, dict):
                        function = dict(raw_function)
                        arguments = function.get("arguments")
                        if isinstance(arguments, dict):
                            function["arguments"] = json.dumps(
                                arguments,
                                ensure_ascii=False,
                            )
                        call["function"] = function
                    copied_calls.append(call)
                copied["tool_calls"] = copied_calls
            prepared.append(copied)
        return prepared

    @staticmethod
    def _normalize_backend_tool_calls(
        tool_calls: Any,
    ) -> list[dict[str, Any]] | None:
        """Convert private OpenAI tool calls to oMLX parser-call dictionaries."""

        if not isinstance(tool_calls, list):
            return None
        normalized: list[dict[str, Any]] = []
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            arguments = function.get("arguments", "{}")
            if not isinstance(name, str) or not name:
                continue
            if not isinstance(arguments, str):
                try:
                    arguments = json.dumps(arguments, ensure_ascii=False)
                except (TypeError, ValueError):
                    continue
            normalized.append(
                {
                    "id": item.get("id"),
                    "name": name,
                    "arguments": arguments,
                }
            )
        return normalized or None

    @staticmethod
    def _join_reasoning_and_content(reasoning: Any, content: Any) -> str:
        """Carry private structured reasoning through GenerationOutput safely."""

        reasoning_text = reasoning if isinstance(reasoning, str) else ""
        content_text = content if isinstance(content, str) else ""
        if reasoning_text:
            return f"<think>{reasoning_text}</think>{content_text}"
        return content_text

    async def _enter_request(self) -> None:
        async with self._active_lock:
            self._active_requests += 1

    async def _leave_request(self) -> None:
        async with self._active_lock:
            self._active_requests = max(0, self._active_requests - 1)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> GenerationOutput:
        """Proxy chat as chat so rank-zero protocol parsing is not discarded."""

        # Partial assistant prefills require oMLX's local template renderer;
        # MLX-LM's chat endpoint always adds a new generation prompt.
        if kwargs.get("is_partial"):
            return await super().chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                tools=tools,
                **kwargs,
            )
        if not self._loaded:
            await self.start()
        client = self._ensure_available()
        payload = self._chat_payload(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=kwargs.pop("stop", None),
            stream=False,
            kwargs=kwargs,
        )
        await self._enter_request()
        started_at = time.monotonic()
        try:
            response = await client.post("/v1/chat/completions", json=payload)
            if response.status_code >= 400:
                detail = self._backend_error_detail(response)
                for retry_payload in _reasoning_effort_retry_payloads(payload, detail):
                    response = await client.post(
                        "/v1/chat/completions", json=retry_payload
                    )
                    if response.status_code < 400:
                        break
            self._raise_for_backend(response)
            body = response.json()
        except httpx.ReadTimeout as exc:
            raise self._read_timeout_error(stream=False) from exc
        except httpx.HTTPError as exc:
            raise await self._transport_failure_error(
                exc,
                stream=False,
            ) from exc
        except json.JSONDecodeError as exc:
            raise DistributedInferenceError(
                "rank-zero backend returned invalid chat JSON"
            ) from exc
        finally:
            await self._leave_request()

        try:
            choice = body["choices"][0]
            message = choice["message"]
            usage = body["usage"]
            details = usage.get("prompt_tokens_details") or {}
            tool_calls = self._normalize_backend_tool_calls(message.get("tool_calls"))
            text = self._join_reasoning_and_content(
                message.get("reasoning") or message.get("reasoning_content"),
                message.get("content"),
            )
            return GenerationOutput(
                text=text,
                new_text=text,
                prompt_tokens=int(usage["prompt_tokens"]),
                completion_tokens=int(usage["completion_tokens"]),
                finish_reason=(
                    "tool_calls"
                    if tool_calls
                    else (choice.get("finish_reason") or "stop")
                ),
                tool_calls=tool_calls,
                cached_tokens=int(details.get("cached_tokens", 0)),
                generated_at=started_at,
                generated_until=time.monotonic(),
            )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise DistributedInferenceError(
                "rank-zero backend returned an invalid chat completion"
            ) from exc

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationOutput]:
        """Stream private chat while retaining reasoning and structured calls."""

        if kwargs.get("is_partial"):
            async for output in super().stream_chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                tools=tools,
                **kwargs,
            ):
                yield output
            return
        if not self._loaded:
            await self.start()
        client = self._ensure_available()
        payload = self._chat_payload(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=kwargs.pop("stop", None),
            stream=True,
            kwargs=kwargs,
        )
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        finish_reason: str | None = None
        full_text = ""
        first_token_at: float | None = None
        request_started_at = time.monotonic()
        reasoning_open = False
        backend_tool_calls: dict[int, dict[str, Any]] = {}

        await self._enter_request()
        try:
            # A client that always sends an unsupported reasoning_effort must
            # never turn this into an unbounded retry loop: `attempts` is
            # extended (by at most two entries) only once, from the FIRST
            # failure's detail, so this terminates in at most three tries.
            attempts = [payload]
            attempt_index = 0
            while True:
                attempt_payload = attempts[attempt_index]
                async with client.stream(
                    "POST", "/v1/chat/completions", json=attempt_payload
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        if attempt_index == 0:
                            detail = self._backend_error_detail(response)
                            attempts.extend(
                                _reasoning_effort_retry_payloads(
                                    attempt_payload, detail
                                )
                            )
                        if attempt_index + 1 < len(attempts):
                            attempt_index += 1
                            continue
                        self._raise_for_backend(response)
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise DistributedInferenceError(
                                "rank-zero backend emitted invalid chat SSE JSON"
                            ) from exc
                        if not isinstance(event, dict):
                            raise DistributedInferenceError(
                                "rank-zero backend emitted an invalid chat SSE event"
                            )
                        usage = event.get("usage") or {}
                        if usage:
                            if not isinstance(usage, dict):
                                raise DistributedInferenceError(
                                    "rank-zero backend emitted invalid chat usage"
                                )
                            prompt_tokens = int(
                                usage.get("prompt_tokens", prompt_tokens)
                            )
                            completion_tokens = int(
                                usage.get("completion_tokens", completion_tokens)
                            )
                            details = usage.get("prompt_tokens_details") or {}
                            if not isinstance(details, dict):
                                raise DistributedInferenceError(
                                    "rank-zero backend emitted invalid "
                                    "chat token details"
                                )
                            cached_tokens = int(details.get("cached_tokens", 0))
                        choices = event.get("choices") or []
                        if not isinstance(choices, list):
                            raise DistributedInferenceError(
                                "rank-zero backend emitted invalid chat choices"
                            )
                        if not choices:
                            continue
                        choice = choices[0]
                        if not isinstance(choice, dict):
                            raise DistributedInferenceError(
                                "rank-zero backend emitted an invalid chat choice"
                            )
                        delta = choice.get("delta") or {}
                        if not isinstance(delta, dict):
                            raise DistributedInferenceError(
                                "rank-zero backend emitted an invalid chat delta"
                            )

                        raw_tool_calls = delta.get("tool_calls") or []
                        if not isinstance(raw_tool_calls, list):
                            raise DistributedInferenceError(
                                "rank-zero backend emitted invalid chat tool calls"
                            )
                        for raw_call in raw_tool_calls:
                            if not isinstance(raw_call, dict):
                                continue
                            index = raw_call.get("index", len(backend_tool_calls))
                            if not isinstance(index, int):
                                continue
                            target = backend_tool_calls.setdefault(
                                index,
                                {
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                },
                            )
                            if raw_call.get("id"):
                                target["id"] = raw_call["id"]
                            function = raw_call.get("function") or {}
                            if isinstance(function, dict):
                                if isinstance(function.get("name"), str):
                                    target["function"]["name"] += function["name"]
                                if isinstance(function.get("arguments"), str):
                                    target["function"]["arguments"] += (
                                        function["arguments"]
                                    )

                        new_text = ""
                        reasoning = delta.get("reasoning") or delta.get(
                            "reasoning_content"
                        )
                        if isinstance(reasoning, str) and reasoning:
                            if not reasoning_open:
                                new_text += "<think>"
                                reasoning_open = True
                            new_text += reasoning
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            if reasoning_open:
                                new_text += "</think>"
                                reasoning_open = False
                            new_text += content
                        if raw_tool_calls and reasoning_open:
                            new_text += "</think>"
                            reasoning_open = False

                        reason = choice.get("finish_reason")
                        if reason is not None:
                            finish_reason = reason
                        if new_text:
                            now = time.monotonic()
                            if first_token_at is None:
                                first_token_at = now
                            full_text += new_text
                            completion_tokens += 1
                            yield GenerationOutput(
                                text=full_text,
                                new_text=new_text,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                finish_reason=None,
                                finished=False,
                                cached_tokens=cached_tokens,
                                generated_at=now,
                                generated_until=now,
                                first_token_at=first_token_at,
                            )
                    break
        except (TypeError, ValueError) as exc:
            raise DistributedInferenceError(
                "rank-zero backend emitted invalid chat token counts"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise self._read_timeout_error(stream=True) from exc
        except httpx.HTTPError as exc:
            raise await self._transport_failure_error(
                exc,
                stream=True,
            ) from exc
        finally:
            await self._leave_request()

        pending_final_text = ""
        if reasoning_open:
            pending_final_text = "</think>"
            full_text += pending_final_text
        tool_calls = self._normalize_backend_tool_calls(
            [backend_tool_calls[index] for index in sorted(backend_tool_calls)]
        )
        finished_at = time.monotonic()
        await self._record_strategy_benchmark(
            prompt_tokens=prompt_tokens,
            uncached_prompt_tokens=max(0, prompt_tokens - cached_tokens),
            completion_tokens=completion_tokens,
            started_at=request_started_at,
            first_token_at=first_token_at,
            finished_at=finished_at,
        )
        yield GenerationOutput(
            text=full_text,
            new_text=pending_final_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=("tool_calls" if tool_calls else (finish_reason or "stop")),
            finished=True,
            tool_calls=tool_calls,
            cached_tokens=cached_tokens,
            generated_at=first_token_at,
            generated_until=finished_at,
            first_token_at=first_token_at,
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerationOutput:
        if not self._loaded:
            await self.start()
        client = self._ensure_available()
        payload = self._completion_payload(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=False,
            kwargs=kwargs,
        )
        await self._enter_request()
        started_at = time.monotonic()
        try:
            response = await client.post("/v1/completions", json=payload)
            if response.status_code >= 400:
                detail = self._backend_error_detail(response)
                for retry_payload in _reasoning_effort_retry_payloads(payload, detail):
                    response = await client.post(
                        "/v1/completions", json=retry_payload
                    )
                    if response.status_code < 400:
                        break
            self._raise_for_backend(response)
            body = response.json()
        except httpx.ReadTimeout as exc:
            raise self._read_timeout_error(stream=False) from exc
        except httpx.HTTPError as exc:
            raise await self._transport_failure_error(
                exc,
                stream=False,
            ) from exc
        except json.JSONDecodeError as exc:
            raise DistributedInferenceError(
                "rank-zero backend returned invalid completion JSON"
            ) from exc
        finally:
            await self._leave_request()

        try:
            choice = body["choices"][0]
            usage = body["usage"]
            details = usage.get("prompt_tokens_details") or {}
            text = choice.get("text") or ""
            return GenerationOutput(
                text=text,
                new_text=text,
                prompt_tokens=int(usage["prompt_tokens"]),
                completion_tokens=int(usage["completion_tokens"]),
                finish_reason=choice.get("finish_reason") or "stop",
                cached_tokens=int(details.get("cached_tokens", 0)),
                generated_at=started_at,
                generated_until=time.monotonic(),
            )
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise DistributedInferenceError(
                "rank-zero backend returned an invalid completion"
            ) from exc

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationOutput]:
        if not self._loaded:
            await self.start()
        client = self._ensure_available()
        payload = self._completion_payload(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            stream=True,
            kwargs=kwargs,
        )
        prompt_tokens = len(self._tokenizer.encode(prompt))
        completion_tokens = 0
        cached_tokens = 0
        finish_reason: str | None = None
        pending_final_text = ""
        full_text = ""
        first_token_at: float | None = None
        request_started_at = time.monotonic()

        await self._enter_request()
        try:
            # See stream_chat for the retry-bound rationale.
            attempts = [payload]
            attempt_index = 0
            while True:
                attempt_payload = attempts[attempt_index]
                async with client.stream(
                    "POST", "/v1/completions", json=attempt_payload
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        if attempt_index == 0:
                            detail = self._backend_error_detail(response)
                            attempts.extend(
                                _reasoning_effort_retry_payloads(
                                    attempt_payload, detail
                                )
                            )
                        if attempt_index + 1 < len(attempts):
                            attempt_index += 1
                            continue
                        self._raise_for_backend(response)
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise DistributedInferenceError(
                                "rank-zero backend emitted invalid SSE JSON"
                            ) from exc
                        if not isinstance(event, dict):
                            raise DistributedInferenceError(
                                "rank-zero backend emitted an invalid SSE event"
                            )
                        usage = event.get("usage") or {}
                        if usage:
                            if not isinstance(usage, dict):
                                raise DistributedInferenceError(
                                    "rank-zero backend emitted invalid usage"
                                )
                            prompt_tokens = int(
                                usage.get("prompt_tokens", prompt_tokens)
                            )
                            completion_tokens = int(
                                usage.get("completion_tokens", completion_tokens)
                            )
                            details = usage.get("prompt_tokens_details") or {}
                            if not isinstance(details, dict):
                                raise DistributedInferenceError(
                                    "rank-zero backend emitted invalid token details"
                                )
                            cached_tokens = int(details.get("cached_tokens", 0))
                        choices = event.get("choices") or []
                        if not isinstance(choices, list):
                            raise DistributedInferenceError(
                                "rank-zero backend emitted invalid choices"
                            )
                        if not choices:
                            continue
                        choice = choices[0]
                        if not isinstance(choice, dict):
                            raise DistributedInferenceError(
                                "rank-zero backend emitted an invalid choice"
                            )
                        new_text = choice.get("text") or ""
                        reason = choice.get("finish_reason")
                        if reason is not None:
                            finish_reason = reason
                            pending_final_text += new_text
                            continue
                        if new_text:
                            now = time.monotonic()
                            if first_token_at is None:
                                first_token_at = now
                            full_text += new_text
                            # MLX-LM streams one generated response at a time but
                            # sends exact usage only in its terminal SSE event.
                            # Keep oMLX's live counters advancing, then replace
                            # them with the exact backend count at completion.
                            completion_tokens += 1
                            yield GenerationOutput(
                                text=full_text,
                                new_text=new_text,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                finish_reason=None,
                                finished=False,
                                cached_tokens=cached_tokens,
                                generated_at=now,
                                generated_until=now,
                                first_token_at=first_token_at,
                            )
                    break
        except (TypeError, ValueError) as exc:
            raise DistributedInferenceError(
                "rank-zero backend emitted invalid token counts"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise self._read_timeout_error(stream=True) from exc
        except httpx.HTTPError as exc:
            raise await self._transport_failure_error(
                exc,
                stream=True,
            ) from exc
        finally:
            await self._leave_request()

        finished_at = time.monotonic()
        await self._record_strategy_benchmark(
            prompt_tokens=prompt_tokens,
            uncached_prompt_tokens=max(0, prompt_tokens - cached_tokens),
            completion_tokens=completion_tokens,
            started_at=request_started_at,
            first_token_at=first_token_at,
            finished_at=finished_at,
        )
        full_text += pending_final_text
        yield GenerationOutput(
            text=full_text,
            new_text=pending_final_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason or "stop",
            finished=True,
            cached_tokens=cached_tokens,
            generated_at=first_token_at,
            generated_until=finished_at,
            first_token_at=first_token_at,
        )

    async def _record_strategy_benchmark(
        self,
        *,
        prompt_tokens: int,
        uncached_prompt_tokens: int,
        completion_tokens: int,
        started_at: float,
        first_token_at: float | None,
        finished_at: float,
    ) -> None:
        """Persist only requests that can measure both prefill and decode.

        The one-token readiness canary intentionally cannot enter this path:
        using it would teach Automatic that every strategy has zero decode
        throughput and make the result depend on which one happened to launch
        most recently.
        """

        if (
            prompt_tokens < 16
            or uncached_prompt_tokens < 1
            or completion_tokens < 2
            or first_token_at is None
            or first_token_at <= started_at
            or finished_at <= first_token_at
        ):
            return
        ttft = first_token_at - started_at
        decode_tps = (completion_tokens - 1) / (finished_at - first_token_at)
        prefill_tps = uncached_prompt_tokens / ttft
        try:
            from ..cluster.strategy_benchmarks import (
                get_strategy_benchmark_store,
                new_benchmark,
            )

            benchmark = new_benchmark(
                model=self.deployment.model,
                node_ids=tuple(host.node_id for host in self.deployment.hosts),
                backend=self.deployment.backend,
                tensor_parallel_size=self.deployment.tensor_parallel_size,
                context_tokens=prompt_tokens,
                prompt_tokens_per_second=prefill_tps,
                decode_tokens_per_second=decode_tps,
                time_to_first_token_seconds=ttft,
            )
            await asyncio.to_thread(
                get_strategy_benchmark_store().record,
                benchmark,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            # Performance history is advisory; a failed disk write must never
            # fail a completed inference request.
            logger.warning("Could not save cluster strategy measurement: %s", exc)

    @staticmethod
    def _backend_error_detail(response: httpx.Response) -> str:
        detail = ""
        with suppress(json.JSONDecodeError, TypeError, ValueError):
            payload = response.json()
            if isinstance(payload, dict):
                detail = str(payload.get("error") or payload.get("detail") or "")
        return detail

    @classmethod
    def _raise_for_backend(cls, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        detail = cls._backend_error_detail(response)
        suffix = f": {detail[:500]}" if detail else ""
        raise DistributedInferenceError(
            f"rank-zero backend returned HTTP {response.status_code}{suffix}"
        )

    async def _require_healthy_cluster(self) -> None:
        """Refuse a request the cluster cannot serve, before the 200 commits.

        A streaming response commits its status line before the body
        generator runs, so any failure detected later reaches the client as
        an error frame inside a 200 — the empty-response class from #2708.
        Preflight is the last point a clean HTTP error is still possible.
        The supervisor read is free; the per-rank marker read costs one SSH
        round trip per peer and is cached for ``_PEER_HEALTH_TTL`` seconds.
        """

        status = self._supervisor.status()
        if status.returncode is not None:
            raise DistributedInferenceError(
                f"distributed job exited with code {status.returncode}"
            )
        if status.failure_reason:
            raise DistributedInferenceError(
                f"distributed cluster failure: {status.failure_reason}"
            )
        cached = self._peer_health
        if cached is None or time.monotonic() - cached[0] >= _PEER_HEALTH_TTL:
            async with self._peer_health_lock:
                cached = self._peer_health
                if cached is None or (
                    time.monotonic() - cached[0] >= _PEER_HEALTH_TTL
                ):
                    hosts_by_rank = {
                        rank: (host.node_id, host.ssh)
                        for rank, host in enumerate(self.deployment.hosts)
                    }
                    try:
                        health = await asyncio.to_thread(
                            check_peers,
                            hosts_by_rank,
                            deployment_id=self.deployment.deployment_id,
                            require_heartbeat=True,
                        )
                    except Exception as exc:  # noqa: BLE001 - probe plumbing
                        # A broken probe must not take down a serving
                        # cluster; the supervisor checks above still catch
                        # hard failures.
                        logger.warning("peer health probe failed: %s", exc)
                        cached = (time.monotonic(), True, "")
                    else:
                        healthy = all(item.healthy for item in health)
                        cached = (
                            time.monotonic(),
                            healthy,
                            "" if healthy else describe_failure(health),
                        )
                    self._peer_health = cached
        if not cached[1]:
            raise DistributedInferenceError(
                f"cluster is not serving: {cached[2]}"
            )

    async def preflight_chat(self, *args: Any, **kwargs: Any) -> None:
        self._validate_request_features(kwargs)
        await self._require_healthy_cluster()
        return None

    async def preflight_completion(self, *args: Any, **kwargs: Any) -> None:
        self._validate_request_features(kwargs)
        await self._require_healthy_cluster()
        return None

    def has_active_requests(self) -> bool:
        return self._active_requests > 0

    def get_stats(self) -> dict[str, Any]:
        return {
            "engine_type": "distributed_batched",
            "model_name": self._model_name,
            "loaded": self._loaded,
            "active_requests": self._active_requests,
            "cluster": self._supervisor.status().to_dict(),
            "execution": self.deployment.execution.to_dict(),
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        return None

    async def abort_all_requests(
        self,
        *,
        reason: str | None = None,
        error_code: str | None = None,
    ) -> int:
        # Closing the private client disconnects all rank-zero handlers. The
        # MLX-LM handler cancels their generation contexts in ``finally``.
        active = self._active_requests
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()
            endpoint = self._supervisor.endpoint
            if endpoint is not None and self._loaded:
                self._client = self._new_client(endpoint)
        return active
