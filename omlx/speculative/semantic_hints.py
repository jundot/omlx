# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral semantic tool hints for target-side verification.

The sidecar protocol deliberately transports *semantics*, never token ids or
provider-rendered tool-call text.  A target integration must render and verify
the resulting suffix against its own tokenizer before it can commit anything.

The live integration is deliberately narrow: one greedy request, a loopback
sidecar, and ordinary dense ``KVCache`` layers.  Everything else fails closed
to normal target decoding.  See
``docs/experimental/ooo_spec_regular_batched_boundary.md`` for the exact
boundary and exclusions.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import json
import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import urlsplit

import httpx
from jsonschema import SchemaError, ValidationError, validate


class SemanticHintError(Exception):
    """Base error for a semantic hint that must be ignored safely."""


class SemanticHintValidationError(SemanticHintError):
    """The sidecar response or selected tool schema was not acceptable."""


class SemanticHintUnavailableError(SemanticHintError):
    """The sidecar could not return a usable response in time."""


class TargetTemplateMismatchError(SemanticHintError):
    """The target template did not yield an append-only suffix."""


@dataclass(frozen=True)
class SemanticHintConfig:
    """Opt-in sidecar configuration.

    No endpoint is configured by default.  ``from_env`` intentionally has no
    settings-manager/admin integration: this experimental transport remains
    an explicit local operator opt-in while the live lane is experimental.
    """

    enabled: bool = False
    endpoint: str | None = None
    timeout_s: float = 0.25
    max_prompt_tokens: int = 4096
    max_suffix_tokens: int = 128

    @classmethod
    def from_env(cls) -> SemanticHintConfig:
        enabled = os.environ.get("OMLX_OOO_SPEC_ENABLED", "").strip().lower()
        endpoint = os.environ.get("OMLX_OOO_SPEC_ENDPOINT", "").strip() or None
        timeout_value = os.environ.get("OMLX_OOO_SPEC_TIMEOUT_S", "0.25")
        try:
            timeout_s = float(timeout_value)
        except ValueError:
            timeout_s = 0.25
        if timeout_s <= 0:
            timeout_s = 0.25
        try:
            max_prompt_tokens = int(
                os.environ.get("OMLX_OOO_SPEC_MAX_PROMPT_TOKENS", "4096")
            )
        except ValueError:
            max_prompt_tokens = 4096
        try:
            max_suffix_tokens = int(
                os.environ.get("OMLX_OOO_SPEC_MAX_SUFFIX_TOKENS", "128")
            )
        except ValueError:
            max_suffix_tokens = 128
        return cls(
            enabled=enabled in {"1", "true", "yes", "on"} and endpoint is not None,
            endpoint=endpoint,
            timeout_s=timeout_s,
            max_prompt_tokens=max(1, max_prompt_tokens),
            max_suffix_tokens=max(1, max_suffix_tokens),
        )

    @property
    def is_local(self) -> bool:
        """Whether the configured HTTP endpoint is explicitly loopback-only."""
        if not self.endpoint:
            return False
        try:
            parsed = urlsplit(self.endpoint)
        except ValueError:
            return False
        return parsed.scheme in {"http", "https"} and parsed.hostname in {
            "127.0.0.1",
            "::1",
            "localhost",
        }


@dataclass(frozen=True)
class SemanticToolCall:
    """A validated tool-call meaning, resolved against final server tools."""

    tool_index: int
    function_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class TargetRenderedHint:
    """A target-tokenized candidate suffix, still requiring target verification."""

    tool_call: SemanticToolCall
    assistant_message: dict[str, Any]
    suffix_text: str
    prompt_token_ids: list[int]
    suffix_token_ids: list[int]


@dataclass(frozen=True)
class SemanticHintCandidate:
    """Call-free immutable input that may become a one-shot request later.

    The API/engine layer prepares this object, but only the scheduler may turn
    it into a live mailbox after admission and request eligibility are known.
    Keeping the final tool list frozen here also guarantees that validation
    and provider resolution use the exact same server-owned schemas.
    """

    messages_json: str
    final_tools_json: str
    template_tools_json: str
    template_options_json: str
    config: SemanticHintConfig
    event_loop: asyncio.AbstractEventLoop | None = field(
        default=None, compare=False, repr=False
    )


@dataclass(frozen=True)
class SemanticHintRequestContext:
    """Admitted immutable chat/template snapshot plus its one-shot mailbox.

    JSON strings make the message, tool, and template inputs immutable across
    the event-loop and scheduler threads.  The scheduler thaws fresh values
    only after target prefill, then binds the rerendered base prompt to the
    live ``Request.prompt_token_ids`` before verification.
    """

    mailbox: SemanticHintMailbox
    messages_json: str
    template_tools_json: str
    template_options_json: str
    config: SemanticHintConfig

    def cancel(self) -> None:
        self.mailbox.cancel()

    def poll_once(self) -> SemanticToolCall | None:
        return self.mailbox.poll_once()


def _json_copy(value: Any, *, label: str) -> Any:
    """Return JSON-only data without exposing values in validation errors."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SemanticHintValidationError(f"{label} must be JSON-compatible") from exc


def _as_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        raise SemanticHintValidationError(f"{label} must be an object")
    return dict(value)


def normalize_semantic_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """Copy request messages into the JSON-only sidecar request contract."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        item = _as_mapping(message, label="message")
        role = item.get("role")
        if not isinstance(role, str) or not role:
            raise SemanticHintValidationError("message role must be a non-empty string")
        normalized.append(_json_copy(item, label="message"))
    return normalized


def normalize_final_tools(tools: Sequence[Any]) -> list[dict[str, Any]]:
    """Normalize the final merged OpenAI tools used for sidecar resolution.

    Only function tools are accepted.  The output is intentionally narrow and
    stable so the tool index always refers to this exact server-owned list.
    """
    normalized: list[dict[str, Any]] = []
    for tool in tools:
        item = _as_mapping(tool, label="tool")
        if item.get("type") != "function":
            raise SemanticHintValidationError("only function tools are supported")
        function = _as_mapping(item.get("function"), label="tool function")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise SemanticHintValidationError(
                "tool function name must be a non-empty string"
            )
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, Mapping):
            raise SemanticHintValidationError("tool parameters must be an object")
        _reject_schema_references(parameters)

        clean_function: dict[str, Any] = {
            "name": name,
            "parameters": _json_copy(dict(parameters), label="tool parameters"),
        }
        description = function.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise SemanticHintValidationError("tool description must be a string")
            clean_function["description"] = description
        strict = function.get("strict")
        if strict is not None:
            if not isinstance(strict, bool):
                raise SemanticHintValidationError("tool strict must be a boolean")
            clean_function["strict"] = strict
        normalized.append({"type": "function", "function": clean_function})
    return normalized


def _reject_schema_references(value: Any) -> None:
    """Reject every schema reference recursively so validation cannot retrieve.

    The semantic sidecar lane deliberately supports only self-contained tool
    schemas.  Rejecting internal references too keeps the contract simple and
    makes remote retrieval/SSRF impossible rather than dependent on validator
    registry defaults.
    """
    if isinstance(value, Mapping):
        if any(key in value for key in ("$ref", "$dynamicRef", "$recursiveRef")):
            raise SemanticHintValidationError(
                "tool parameters must not contain schema references"
            )
        for nested in value.values():
            _reject_schema_references(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_schema_references(nested)


def parse_semantic_hint_response(
    payload: Any, tools: Sequence[dict[str, Any]]
) -> SemanticToolCall:
    """Strictly validate a semantic-only sidecar response.

    A response may contain exactly ``tool_index`` and ``arguments``.  Errors
    intentionally avoid embedding argument values, because those may contain
    user data or secrets.
    """
    if not isinstance(payload, Mapping) or set(payload) != {"tool_index", "arguments"}:
        raise SemanticHintValidationError(
            "response must contain only tool_index and arguments"
        )
    tool_index = payload["tool_index"]
    if isinstance(tool_index, bool) or not isinstance(tool_index, int):
        raise SemanticHintValidationError("tool_index must be an integer")
    if tool_index < 0 or tool_index >= len(tools):
        raise SemanticHintValidationError("tool_index is outside the final tool list")
    arguments = payload["arguments"]
    if not isinstance(arguments, Mapping):
        raise SemanticHintValidationError("arguments must be an object")
    clean_arguments = _json_copy(dict(arguments), label="arguments")

    function = tools[tool_index].get("function")
    if not isinstance(function, Mapping):
        raise SemanticHintValidationError("selected tool is malformed")
    name = function.get("name")
    parameters = function.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(name, str) or not name or not isinstance(parameters, Mapping):
        raise SemanticHintValidationError("selected tool is malformed")
    _reject_schema_references(parameters)
    try:
        validate(instance=clean_arguments, schema=dict(parameters))
    except (ValidationError, SchemaError) as exc:
        raise SemanticHintValidationError(
            "arguments do not match the selected tool schema"
        ) from exc
    return SemanticToolCall(
        tool_index=tool_index,
        function_name=name,
        arguments=clean_arguments,
    )


class SemanticHintProvider:
    """HTTP sidecar client with an injectable transport for CPU-only tests."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float = 0.25,
        headers: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self._headers = dict(headers or {})
        self._transport = transport

    async def request_hint(
        self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> SemanticToolCall:
        """Request one validated semantic tool call without logging its arguments."""
        request_body = {
            "messages": _json_copy(list(messages), label="messages"),
            "tools": _json_copy(list(tools), label="tools"),
        }
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s),
                headers=self._headers,
                transport=self._transport,
                trust_env=False,
            ) as client:
                response = await client.post(self.endpoint, json=request_body)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            raise SemanticHintUnavailableError(
                "semantic sidecar request failed"
            ) from exc
        except ValueError as exc:
            raise SemanticHintValidationError(
                "semantic sidecar returned invalid JSON"
            ) from exc
        return parse_semantic_hint_response(payload, tools)


class SemanticHintMailbox:
    """One-shot, nonblocking result slot for a sidecar request.

    ``poll_once`` never awaits.  Timeouts, cancellations, provider failures,
    late responses, and malformed payloads all resolve to ``None``.
    """

    def __init__(
        self,
        task: (
            asyncio.Future[SemanticToolCall | None]
            | concurrent.futures.Future[SemanticToolCall | None]
        ),
    ) -> None:
        self._task = task
        self._polled = False

    @classmethod
    def start(
        cls,
        provider: SemanticHintProvider,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        *,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ) -> SemanticHintMailbox:
        async def _resolve() -> SemanticToolCall | None:
            try:
                return await asyncio.wait_for(
                    provider.request_hint(messages, tools),
                    timeout=provider.timeout_s,
                )
            except asyncio.CancelledError:
                raise
            except (SemanticHintError, TimeoutError):
                return None
            except Exception:
                # A sidecar must never destabilize target decoding.
                return None

        if event_loop is None:
            try:
                event_loop = asyncio.get_running_loop()
            except RuntimeError as exc:
                raise SemanticHintUnavailableError(
                    "semantic sidecar has no event loop"
                ) from exc
        if event_loop.is_closed() or not event_loop.is_running():
            raise SemanticHintUnavailableError(
                "semantic sidecar event loop is unavailable"
            )
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if event_loop is current_loop:
            return cls(event_loop.create_task(_resolve()))
        return cls(asyncio.run_coroutine_threadsafe(_resolve(), event_loop))

    @property
    def pending(self) -> bool:
        return not self._task.done()

    def poll_once(self) -> SemanticToolCall | None:
        """Claim the one-shot result, or ``None`` without waiting.

        The first poll is terminal even when the provider is still pending.
        This makes a late sidecar result inert instead of allowing it to alter
        a later decode step after the target has already chosen its baseline.
        """
        if self._polled:
            return None
        self._polled = True
        if not self._task.done() or self._task.cancelled():
            return None
        try:
            return self._task.result()
        except (asyncio.CancelledError, SemanticHintError):
            return None
        except Exception:
            return None

    def cancel(self) -> None:
        """Cancel an outstanding sidecar request; completed values remain inert."""
        if not self._task.done():
            if isinstance(self._task, asyncio.Future):
                loop = self._task.get_loop()
                with suppress(RuntimeError):
                    loop.call_soon_threadsafe(self._task.cancel)
            else:
                self._task.cancel()

    async def wait(self) -> SemanticToolCall | None:
        """Test/support helper; production scheduling should use ``poll_once``."""
        try:
            if isinstance(self._task, asyncio.Future):
                return await self._task
            return await asyncio.wrap_future(self._task)
        except (asyncio.CancelledError, SemanticHintError):
            return None


def start_semantic_hint_mailbox(
    messages: Sequence[Any],
    final_tools: Sequence[Any] | None,
    *,
    config: SemanticHintConfig | None = None,
    provider: SemanticHintProvider | None = None,
    event_loop: asyncio.AbstractEventLoop | None = None,
) -> SemanticHintMailbox | None:
    """Start an opt-in sidecar request after final tool merging.

    Invalid local input is treated exactly like an unavailable provider: callers
    receive no mailbox and continue ordinary target decoding.
    """
    config = config or SemanticHintConfig.from_env()
    if not config.enabled or not config.endpoint or not final_tools:
        return None
    try:
        normalized_messages = normalize_semantic_messages(messages)
        normalized_tools = normalize_final_tools(final_tools)
    except SemanticHintError:
        return None
    provider = provider or SemanticHintProvider(
        config.endpoint, timeout_s=config.timeout_s
    )
    return SemanticHintMailbox.start(
        provider,
        normalized_messages,
        normalized_tools,
        event_loop=event_loop,
    )


def prepare_semantic_hint_candidate(
    messages: Sequence[Any],
    final_tools: Sequence[Any] | None,
    template_tools: Sequence[Any] | None,
    template_options: Mapping[str, Any],
    *,
    config: SemanticHintConfig | None = None,
) -> SemanticHintCandidate | None:
    """Freeze exact chat inputs without starting a provider request."""
    config = config or SemanticHintConfig.from_env()
    if not config.enabled or not config.is_local or not final_tools:
        return None
    try:
        normalized_messages = normalize_semantic_messages(messages)
        normalized_final_tools = normalize_final_tools(final_tools)
        frozen_template_tools = _json_copy(
            list(template_tools or ()), label="template tools"
        )
        frozen_template_options = _json_copy(
            dict(template_options), label="template options"
        )
    except Exception:
        return None
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None
    return SemanticHintCandidate(
        messages_json=json.dumps(
            normalized_messages,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        final_tools_json=json.dumps(
            normalized_final_tools,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        template_tools_json=json.dumps(
            frozen_template_tools,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        template_options_json=json.dumps(
            frozen_template_options,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        config=config,
        event_loop=event_loop,
    )


def start_semantic_hint_request_context(
    candidate: SemanticHintCandidate,
    *,
    provider: SemanticHintProvider | None = None,
) -> SemanticHintRequestContext | None:
    """Start one mailbox only after scheduler admission eligibility passes."""
    if not candidate.config.enabled or not candidate.config.is_local:
        return None
    try:
        messages = json.loads(candidate.messages_json)
        final_tools = json.loads(candidate.final_tools_json)
        mailbox = start_semantic_hint_mailbox(
            messages,
            final_tools,
            config=candidate.config,
            provider=provider,
            event_loop=candidate.event_loop,
        )
        if mailbox is None:
            return None
        return SemanticHintRequestContext(
            mailbox=mailbox,
            messages_json=candidate.messages_json,
            template_tools_json=candidate.template_tools_json,
            template_options_json=candidate.template_options_json,
            config=candidate.config,
        )
    except Exception:
        if "mailbox" in locals() and mailbox is not None:
            mailbox.cancel()
        return None


def tool_call_message(
    hint: SemanticToolCall, *, call_id: str = "semantic_hint"
) -> dict[str, Any]:
    """Build a server-owned assistant tool-call message from validated semantics."""
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": hint.function_name,
                    "arguments": json.dumps(
                        hint.arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
        ],
    }


def render_target_hint(
    *,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None,
    hint: SemanticToolCall,
    render_chat: Callable[[list[dict[str, Any]], Sequence[dict[str, Any]] | None], str],
    tokenizer: Any,
) -> TargetRenderedHint:
    """Render a tool-call suffix using only the target's template/tokenizer.

    ``render_chat`` must use the exact target template mode that the eventual
    verifier uses.  If appending the server-owned assistant tool call changes
    the earlier prompt, the template is unsuitable for this MVP and callers
    must fall back.
    """
    base_messages = copy.deepcopy(list(messages))
    base_prompt = render_chat(base_messages, tools)
    assistant_message = tool_call_message(hint)
    augmented_messages = copy.deepcopy(base_messages)
    augmented_messages.append(assistant_message)
    rendered = render_chat(augmented_messages, tools)
    if not isinstance(base_prompt, str) or not isinstance(rendered, str):
        raise TargetTemplateMismatchError("target template must render text")
    if not rendered.startswith(base_prompt):
        raise TargetTemplateMismatchError(
            "target template did not produce an append-only suffix"
        )
    suffix_text = rendered[len(base_prompt) :]
    if not suffix_text:
        raise TargetTemplateMismatchError(
            "target template produced an empty tool-call suffix"
        )
    try:
        prompt_token_ids = list(tokenizer.encode(base_prompt))
        rendered_token_ids = list(tokenizer.encode(rendered))
    except Exception as exc:
        raise TargetTemplateMismatchError(
            "target tokenizer could not encode rendered prompt"
        ) from exc
    if rendered_token_ids[: len(prompt_token_ids)] != prompt_token_ids:
        raise TargetTemplateMismatchError(
            "target tokenizer did not preserve an append-only token suffix"
        )
    suffix_token_ids = rendered_token_ids[len(prompt_token_ids) :]
    if not suffix_token_ids or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in suffix_token_ids
    ):
        raise TargetTemplateMismatchError(
            "target tokenizer returned invalid suffix tokens"
        )
    return TargetRenderedHint(
        tool_call=hint,
        assistant_message=assistant_message,
        suffix_text=suffix_text,
        prompt_token_ids=prompt_token_ids,
        suffix_token_ids=suffix_token_ids,
    )


def _apply_frozen_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    template_tools: list[dict[str, Any]],
    template_options: dict[str, Any],
    *,
    augmented: bool,
) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TargetTemplateMismatchError(
            "target tokenizer has no chat-template renderer"
        )

    options = {"tokenize": False, **template_options}
    options["add_generation_prompt"] = not augmented
    options.pop("continue_final_message", None)
    if template_tools:
        options["tools"] = copy.deepcopy(template_tools)
    try:
        rendered = apply_chat_template(copy.deepcopy(messages), **options)
        if not isinstance(rendered, str):
            raise TargetTemplateMismatchError("target template must render text")
        return rendered
    except TypeError:
        # Match BatchedEngine._apply_chat_template's compatibility fallback:
        # request/model template kwargs, tools, and enable_thinking are removed.
        fallback = {
            "tokenize": False,
            "add_generation_prompt": not augmented,
        }
        try:
            rendered = apply_chat_template(copy.deepcopy(messages), **fallback)
            if not isinstance(rendered, str):
                raise TargetTemplateMismatchError("target template must render text")
            return cast(str, rendered)
        except Exception as exc:
            raise TargetTemplateMismatchError(
                "target chat template could not render the hint"
            ) from exc
    except Exception as exc:
        raise TargetTemplateMismatchError(
            "target chat template could not render the hint"
        ) from exc


def render_live_target_hint(
    *,
    context: SemanticHintRequestContext,
    hint: SemanticToolCall,
    tokenizer: Any,
    live_prompt_token_ids: Sequence[int],
) -> TargetRenderedHint:
    """Render and bind a hint to the exact prompt token ids being decoded."""
    try:
        messages = json.loads(context.messages_json)
        template_tools = json.loads(context.template_tools_json)
        template_options = json.loads(context.template_options_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TargetTemplateMismatchError(
            "semantic hint context could not be restored"
        ) from exc

    assistant_message = tool_call_message(hint)
    base_prompt = _apply_frozen_chat_template(
        tokenizer,
        messages,
        template_tools,
        template_options,
        augmented=False,
    )
    augmented_messages = copy.deepcopy(messages)
    augmented_messages.append(assistant_message)
    rendered = _apply_frozen_chat_template(
        tokenizer,
        augmented_messages,
        template_tools,
        template_options,
        augmented=True,
    )
    if not isinstance(base_prompt, str) or not isinstance(rendered, str):
        raise TargetTemplateMismatchError("target template must render text")
    if not rendered.startswith(base_prompt):
        raise TargetTemplateMismatchError(
            "target template did not produce an append-only suffix"
        )

    try:
        prompt_token_ids = list(tokenizer.encode(base_prompt))
        rendered_token_ids = list(tokenizer.encode(rendered))
    except Exception as exc:
        raise TargetTemplateMismatchError(
            "target tokenizer could not encode rendered prompt"
        ) from exc
    if prompt_token_ids != [int(token) for token in live_prompt_token_ids]:
        raise TargetTemplateMismatchError(
            "rendered target prompt does not match the live request tokens"
        )
    if rendered_token_ids[: len(prompt_token_ids)] != prompt_token_ids:
        raise TargetTemplateMismatchError(
            "target tokenizer did not preserve an append-only token suffix"
        )
    suffix_token_ids = rendered_token_ids[len(prompt_token_ids) :]
    if not suffix_token_ids or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in suffix_token_ids
    ):
        raise TargetTemplateMismatchError(
            "target tokenizer returned invalid suffix tokens"
        )
    return TargetRenderedHint(
        tool_call=hint,
        assistant_message=assistant_message,
        suffix_text=rendered[len(base_prompt) :],
        prompt_token_ids=prompt_token_ids,
        suffix_token_ids=suffix_token_ids,
    )
