# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible external endpoint client for admin benchmarks.

Shared by the throughput and accuracy benchmarks to run against a remote
/chat/completions endpoint instead of a local engine. Token counts always
come from the endpoint's usage payload — SSE chunks are never counted as
tokens because providers batch multiple tokens per chunk.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field, SecretStr, field_validator

logger = logging.getLogger(__name__)

# read=3600 covers both the largest between-chunk gap on streams (TTFT of a
# very long prefill on a slow remote) and the full-response wait for
# non-streaming accuracy calls. Benchmarks are supervised and cancellable,
# so a generous ceiling beats spurious failures; connect=15 still fails
# dead endpoints fast.
DEFAULT_TIMEOUT = httpx.Timeout(connect=15.0, read=3600.0, write=120.0, pool=30.0)

_ERROR_DETAIL_MAX_CHARS = 300

_REASONING_FIELD_NAMES = ("reasoning_content", "reasoning", "analysis")

# Thinking models spend reasoning tokens before message.content, and those
# tokens count toward max_tokens on OpenAI-compatible APIs, so a small cap
# truncates them before they can answer (#2309). max_tokens only bounds
# runaway generation; non-thinking endpoints still stop after a few tokens.
_PREFLIGHT_MAX_TOKENS = 4096

# Provider-specific request JSON must not override fields owned by the
# benchmark or authentication layer.
PROTECTED_EXTRA_BODY_FIELDS = frozenset({
    "model",
    "messages",
    "stream",
    "stream_options",
    "max_tokens",
    "temperature",
    "api_key",
    "authorization",
})


class ExternalEndpointConfig(BaseModel):
    """Connection settings for an external OpenAI-compatible endpoint.

    api_key is a SecretStr so the key never leaks through repr() or logs.
    """

    base_url: str
    api_key: SecretStr = SecretStr("")
    model: str
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("model must not be empty")
        return v

    @field_validator("extra_body", mode="before")
    @classmethod
    def validate_extra_body(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("extra_body must be a JSON object")
        blocked = sorted(
            str(key)
            for key in v
            if str(key).lower() in PROTECTED_EXTRA_BODY_FIELDS
        )
        if blocked:
            raise ValueError(
                "extra_body cannot override protected field(s): "
                + ", ".join(blocked)
            )
        return dict(v)


class ExternalEndpointError(Exception):
    """User-presentable failure talking to an external endpoint."""

    def __init__(self, message: str, status: str = "invalid_response"):
        super().__init__(message)
        self.status = status


@dataclass
class StreamStats:
    """Timing and token stats from one streamed chat completion."""

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    start_time: float
    first_content_time: float
    last_content_time: float
    end_time: float
    text: str


@dataclass
class ChatResult:
    """Non-streaming chat completion result."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = None
    status: str = "ok"
    reasoning_fields_present: tuple[str, ...] = ()
    reasoning_fields_nonempty: tuple[str, ...] = ()


def _extract_error_detail(body: str) -> str:
    """Pull a short human-readable message out of an error response body."""
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])[:_ERROR_DETAIL_MAX_CHARS]
            for key in ("message", "detail"):
                if data.get(key):
                    return str(data[key])[:_ERROR_DETAIL_MAX_CHARS]
    except ValueError:
        pass
    text = body.strip()
    if "<" in text and ">" in text:
        return f"unexpected non-JSON response ({len(body)} bytes)"
    return text[:_ERROR_DETAIL_MAX_CHARS] or "no response body"


class ExternalAPIClient:
    """Async client for an external OpenAI-compatible /chat/completions API.

    Provider-specific fields may be supplied through config.extra_body. The
    config validator prevents them from overriding benchmark-owned or
    authentication-related fields.
    """

    def __init__(
        self,
        config: ExternalEndpointConfig,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self._config = config
        self._chat_url = f"{config.base_url}/chat/completions"
        headers = {}
        key = config.api_key.get_secret_value()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        # transport is injectable for tests (httpx.MockTransport).
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            limits=httpx.Limits(max_connections=64),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_body(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: Optional[float],
        stream: bool,
    ) -> dict:
        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        body.update(self._config.extra_body)
        return body

    def _redact_secret(self, text: str) -> str:
        """Defensively remove the configured key from provider errors."""
        key = self._config.api_key.get_secret_value()
        if key:
            return text.replace(key, "[REDACTED]")
        return text

    def _map_transport_error(self, exc: httpx.HTTPError) -> ExternalEndpointError:
        base_url = self._config.base_url
        if isinstance(exc, httpx.TimeoutException):
            if isinstance(exc, httpx.ConnectTimeout):
                return ExternalEndpointError(
                    f"Timed out connecting to external endpoint {base_url}",
                    status="timeout",
                )
            return ExternalEndpointError(
                "External endpoint timed out while waiting for a response",
                status="timeout",
            )
        if isinstance(exc, httpx.ConnectError):
            return ExternalEndpointError(
                self._redact_secret(
                    f"Cannot connect to external endpoint {base_url}: {exc}"
                ),
                status="connection_error",
            )
        return ExternalEndpointError(
            self._redact_secret(
                f"External endpoint request failed: {type(exc).__name__}: {exc}"
            ),
            status="connection_error",
        )

    def _status_error(self, status: int, body_text: str) -> ExternalEndpointError:
        if status in (401, 403):
            return ExternalEndpointError(
                f"External endpoint rejected the API key (HTTP {status})",
                status="http_error",
            )
        detail = self._redact_secret(_extract_error_detail(body_text))
        return ExternalEndpointError(
            f"External endpoint returned HTTP {status}: {detail}",
            status="http_error",
        )

    @staticmethod
    def _reasoning_diagnostics(
        message: dict[str, Any],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        present: list[str] = []
        nonempty: list[str] = []
        for name in _REASONING_FIELD_NAMES:
            if name not in message:
                continue
            present.append(name)
            value = message.get(name)
            if isinstance(value, str):
                has_value = bool(value.strip())
            else:
                has_value = value is not None and bool(value)
            if has_value:
                nonempty.append(name)
        return tuple(present), tuple(nonempty)

    @staticmethod
    def _usage_int(usage: dict[str, Any], name: str) -> int:
        try:
            return int(usage.get(name) or 0)
        except (TypeError, ValueError):
            return 0

    async def chat_completion(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: Optional[float],
    ) -> ChatResult:
        """Send a non-streaming chat completion request."""
        body = self._build_body(messages, max_tokens, temperature, stream=False)
        try:
            response = await self._client.post(self._chat_url, json=body)
        except httpx.HTTPError as e:
            raise self._map_transport_error(e) from e
        if response.status_code != 200:
            raise self._status_error(response.status_code, response.text)
        try:
            data = response.json()
        except ValueError as e:
            raise ExternalEndpointError(
                "External endpoint returned a non-JSON response",
                status="invalid_response",
            ) from e

        try:
            if not isinstance(data, dict):
                raise TypeError("top-level response is not an object")
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                raise KeyError("choices[0]")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise TypeError("choices[0] is not an object")
            message = choice.get("message")
            if not isinstance(message, dict):
                raise KeyError("choices[0].message")
        except (KeyError, TypeError) as e:
            raise ExternalEndpointError(
                f"External endpoint returned an unexpected response shape: {e}",
                status="invalid_response",
            ) from e

        content = message.get("content")
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            raise ExternalEndpointError(
                "External endpoint returned non-text message.content",
                status="invalid_response",
            )

        finish_reason_value = choice.get("finish_reason")
        finish_reason = (
            str(finish_reason_value) if finish_reason_value is not None else None
        )
        reasoning_present, reasoning_nonempty = self._reasoning_diagnostics(message)
        if finish_reason == "length":
            status = "truncated"
        elif not text.strip():
            status = "empty_content"
        else:
            status = "ok"

        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        return ChatResult(
            text=text,
            prompt_tokens=self._usage_int(usage, "prompt_tokens"),
            completion_tokens=self._usage_int(usage, "completion_tokens"),
            finish_reason=finish_reason,
            status=status,
            reasoning_fields_present=reasoning_present,
            reasoning_fields_nonempty=reasoning_nonempty,
        )

    async def stream_chat_completion(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: Optional[float],
    ) -> StreamStats:
        """Send a streaming chat completion request and collect stats.

        Requires the endpoint to return usage via stream_options
        (include_usage); raises ExternalEndpointError otherwise because
        token counts cannot be measured accurately without it.
        """
        body = self._build_body(messages, max_tokens, temperature, stream=True)
        start_time = time.perf_counter()
        first_content_time: Optional[float] = None
        last_content_time: Optional[float] = None
        usage: Optional[dict] = None
        text_parts: list[str] = []

        try:
            async with self._client.stream(
                "POST", self._chat_url, json=body
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise self._status_error(
                        response.status_code,
                        error_body.decode("utf-8", errors="replace"),
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    chunk_usage = chunk.get("usage")
                    if chunk_usage:
                        usage = chunk_usage
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        text_parts.append(content)
                    if content or delta.get("reasoning_content"):
                        now = time.perf_counter()
                        if first_content_time is None:
                            first_content_time = now
                        last_content_time = now
        except httpx.HTTPError as e:
            raise self._map_transport_error(e) from e

        end_time = time.perf_counter()

        if (
            usage is None
            or usage.get("prompt_tokens") is None
            or usage.get("completion_tokens") is None
        ):
            raise ExternalEndpointError(
                "External endpoint does not support stream usage "
                "(stream_options.include_usage); cannot measure token counts"
            )
        if first_content_time is None:
            first_content_time = end_time
        if last_content_time is None:
            last_content_time = end_time
        details = usage.get("prompt_tokens_details") or {}
        return StreamStats(
            prompt_tokens=int(usage["prompt_tokens"]),
            completion_tokens=int(usage["completion_tokens"]),
            cached_tokens=int(details.get("cached_tokens") or 0),
            start_time=start_time,
            first_content_time=first_content_time,
            last_content_time=last_content_time,
            end_time=end_time,
            text="".join(text_parts),
        )


@dataclass
class _AdapterOutput:
    """Minimal GenerationOutput stand-in; eval code only reads .text."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    external_status: str = "ok"
    finish_reason: Optional[str] = None
    reasoning_fields_present: tuple[str, ...] = ()
    reasoning_fields_nonempty: tuple[str, ...] = ()
    error_message: str = ""


class ExternalChatAdapter:
    """Duck-typed engine for accuracy benchmarks against an external API.

    eval/base.py only touches engine.model_type and engine.chat(), so this
    adapter maps both onto ExternalAPIClient. Sampling comes from the
    constructor-injected profile: the temperature/penalty defaults that
    _eval_single injects via setdefault cannot be told apart from
    profile-supplied values, so all sampling kwargs are accepted and
    dropped here — "deterministic" sends temperature 0 and
    "model_settings" sends no sampling params (remote server defaults).
    """

    model_type = None
    is_external_api = True

    def __init__(self, client: ExternalAPIClient, sampling_profile: str):
        self._client = client
        self._sampling_profile = sampling_profile

    async def preflight(self) -> None:
        """Validate final-answer compatibility before a paid evaluation."""
        result = await self._client.chat_completion(
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=_PREFLIGHT_MAX_TOKENS,
            temperature=None,
        )
        logger.info(
            "External accuracy preflight: finish_reason=%r "
            "reasoning_fields_present=%s reasoning_fields_nonempty=%s",
            result.finish_reason,
            list(result.reasoning_fields_present),
            list(result.reasoning_fields_nonempty),
        )
        if result.status == "truncated":
            raise ExternalEndpointError(
                "External API preflight was truncated (finish_reason=length)",
                status="truncated",
            )
        if result.status == "empty_content":
            if result.reasoning_fields_nonempty:
                raise ExternalEndpointError(
                    "External API connected, but message.content is empty. "
                    "The model may still be in reasoning mode, or the endpoint "
                    "response format may be incompatible with oMLX.",
                    status="empty_content",
                )
            raise ExternalEndpointError(
                "External API connected, but preflight message.content is empty",
                status="empty_content",
            )
        # Some endpoints inline reasoning into message.content as <think>
        # blocks; drop them before checking, the same way eval scoring does.
        answer = re.sub(
            r"<think>.*?</think>", "", result.text, flags=re.DOTALL
        ).strip()
        # Accept a leading OK with trailing punctuation or extra words
        # (OK. / Okay / OK!) so a compliant endpoint is not rejected over
        # formatting. A genuinely wrong or negated reply (NOT OK) still fails.
        if not answer.upper().startswith("OK"):
            raise ExternalEndpointError(
                "External API preflight response did not start with OK in "
                "message.content",
                status="parse_error",
            )

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> _AdapterOutput:
        temperature = 0.0 if self._sampling_profile == "deterministic" else None
        try:
            result = await self._client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except ExternalEndpointError as exc:
            return _AdapterOutput(
                text="",
                external_status=exc.status,
                error_message=str(exc),
            )
        return _AdapterOutput(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            external_status=result.status,
            finish_reason=result.finish_reason,
            reasoning_fields_present=result.reasoning_fields_present,
            reasoning_fields_nonempty=result.reasoning_fields_nonempty,
        )
