# SPDX-License-Identifier: Apache-2.0
"""Conversion utilities for the OpenAI Responses API."""

import copy
import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..exceptions import InvalidRequestError
from .responses_models import (
    InputItem,
    InputTokensDetails,
    OutputContent,
    OutputItem,
    OutputTokensDetails,
    ReasoningSummaryPart,
    ResponseObject,
    ResponsesRequest,
    ResponsesTool,
    ResponseUsage,
)
from .shared_models import IDPrefix, generate_id
from .tool_bindings import ToolBindingRegistry, ensure_call_id

logger = logging.getLogger(__name__)


class ResponseStateError(RuntimeError):
    """Base error for persisted Responses API conversation state."""


class ResponseStateNotFoundError(ResponseStateError):
    """Raised when the requested response state does not exist."""


class ResponseStateCorruptError(ResponseStateError):
    """Raised when a stored response chain is incomplete or invalid."""


def _try_parse_json(s: str):
    """Try to parse a string as JSON dict/list, return original string on failure."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s or not (s.startswith("{") or s.startswith("[")):
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


_TOOL_OUTPUT_TEXT_TYPES = ("input_text", "text", "output_text")


def _extract_tool_output_text(
    output: List[Any],
    image_parts: Optional[List[Dict[str, Any]]],
) -> Optional[str]:
    """Extract text from a multimodal function_call_output list.

    Returns None when the list has no recognized content parts so the
    caller can fall back to JSON serialization. Image parts are appended
    to ``image_parts`` for VLM processing when provided; otherwise they
    become a placeholder so base64 payloads never reach the prompt.
    """
    recognized = any(
        isinstance(part, dict)
        and part.get("type") in (*_TOOL_OUTPUT_TEXT_TYPES, "input_image")
        for part in output
    )
    if not recognized:
        return None

    text_parts: List[str] = []
    for part in output:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict):
            part_type = part.get("type")
            if part_type in _TOOL_OUTPUT_TEXT_TYPES:
                text_parts.append(part.get("text", ""))
            elif part_type == "input_image":
                if image_parts is not None:
                    image_url = part.get("image_url", part.get("url", ""))
                    image_parts.append(
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": part.get("detail", "auto"),
                        }
                    )
                else:
                    text_parts.append("(see attached image)")
            else:
                text_parts.append(json.dumps(part))
        else:
            text_parts.append(str(part))
    return "\n".join(p for p in text_parts if p)


def _flush_pending_tool_images(
    messages: List[Dict[str, Any]],
    pending_images: List[Dict[str, Any]],
) -> None:
    """Flush tool-output images as a user message after a tool run.

    Emitted only once the consecutive function_call_output items end so
    tool messages stay contiguous for strict chat templates.
    """
    if pending_images:
        messages.append({"role": "user", "content": list(pending_images)})
        pending_images.clear()


def _flush_pending_tool_calls(
    messages: List[Dict[str, Any]],
    pending: List[Dict[str, Any]],
    min_merge_index: int = 0,
    pending_reasoning: str = "",
) -> str:
    """Flush accumulated tool calls into messages.

    If the last message is an assistant message without tool_calls, merge
    into it (avoids duplicate assistant turns that confuse chat templates).
    Otherwise create a new assistant message.

    When ``pending_reasoning`` is set, attach it as ``reasoning_content``
    on the synthesized assistant message so reasoning round-trips even
    when the spec sequence is reasoning → function_call → output (no
    intervening message item). Returns the passthrough reasoning when
    no tool calls were flushed, or "" when reasoning was consumed.
    """
    if not pending:
        return pending_reasoning
    if (
        messages
        and len(messages) - 1 >= min_merge_index
        and messages[-1].get("role") == "assistant"
        and "tool_calls" not in messages[-1]
    ):
        messages[-1]["tool_calls"] = list(pending)
        if pending_reasoning:
            messages[-1]["reasoning_content"] = pending_reasoning
    else:
        msg: Dict[str, Any] = {"role": "assistant", "tool_calls": list(pending)}
        if pending_reasoning:
            msg["reasoning_content"] = pending_reasoning
        messages.append(msg)
    pending.clear()
    return ""


def _consolidate_system_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Move all system messages to the front and merge them into one."""
    system_parts: List[str] = []
    non_system: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if content:
                system_parts.append(content)
        else:
            non_system.append(msg)

    if not system_parts:
        return messages

    return [{"role": "system", "content": "\n\n".join(system_parts)}] + non_system


# =============================================================================
# Capability Validation
# =============================================================================

# A "namespace" tool is a container of client-executed function tools; oMLX
# expands its members eagerly.
_CONTAINER_TOOL_TYPES = frozenset({"namespace"})
# ``tool_search`` asks the host to lazily load namespace members. oMLX expands
# every member eagerly, so the capability the declaration asks for is already
# present and dropping it is not a degradation: it is accepted silently.
_REDUNDANT_TOOL_TYPES = frozenset({"tool_search"})
# Hosted/server-executed tool types. oMLX has no executor for them and a chat
# template cannot emit their wire shapes, so they can never be exposed to the
# model. They are still ACCEPTED as declarations -- an unused declaration
# provably cannot change the model's behaviour -- and ``convert_responses_tools``
# records them so the caller is warned (see ``_unexposed_tool_label``) instead
# of the request failing. A declaration of any other unrecognised type follows
# the same path for the same reason; the set below only lets the warning mark
# the types the Responses schema defines as hosted.
_HOSTED_TOOL_TYPES = frozenset(
    {
        "local_shell",
        "custom",
        "mcp",
        "web_search",
        "web_search_preview",
        "file_search",
        "computer_use_preview",
        "computer",
        "code_interpreter",
        "image_generation",
    }
)

_SUPPORTED_MESSAGE_ROLES = frozenset({"user", "assistant", "system", "developer"})
_SUPPORTED_MESSAGE_PART_TYPES = frozenset(
    {"input_text", "text", "output_text", "input_image"}
)

# Input item types for a hosted capability. Their presence means the client is
# replaying a tool round trip oMLX never performed.
_UNSUPPORTED_INPUT_ITEM_TYPES = frozenset(
    {
        "computer_call",
        "computer_call_output",
        "web_search_call",
        "file_search_call",
        "mcp_call",
        "mcp_list_tools",
        "mcp_approval_request",
        "mcp_approval_response",
        "image_generation_call",
        "code_interpreter_call",
        "local_shell_call",
        "custom_tool_call",
        "custom_tool_call_output",
        "item_reference",
    }
)

# ``reasoning.encrypted_content`` asks for an opaque reasoning blob. oMLX has no
# server-side reasoning state to encrypt, and the reasoning itself still travels
# in the item's summary/content, so a chain built from prior turns is unaffected.
_DEGRADED_INCLUDE_VALUES = frozenset({"reasoning.encrypted_content"})


_WARNING_LABEL_UNSAFE = re.compile(r"[^A-Za-z0-9_.:/+-]")
_WARNING_LABEL_LIMIT = 80


def _warning_label_token(value: str) -> str:
    """Make a client-supplied identifier safe inside a quoted Warning header."""
    cleaned = _WARNING_LABEL_UNSAFE.sub("_", value)[:_WARNING_LABEL_LIMIT]
    return cleaned or "?"


def _unexposed_tool_label(tool_type: str, *, namespace: Optional[str] = None) -> str:
    """Name one accepted-but-unexposed tool declaration for the Warning header.

    Every character is constrained to a header-safe set so a ``type`` taken
    verbatim from the request body cannot inject a header, and the label says
    whether the type is a known hosted one, an unrecognised type, or a nested
    namespace, so the caller can tell a typo from an unimplemented capability.
    """
    if tool_type in _HOSTED_TOOL_TYPES:
        label = f"{_warning_label_token(tool_type)} (hosted)"
    elif tool_type in _CONTAINER_TOOL_TYPES:
        label = f"{_warning_label_token(tool_type)} (nested)"
    else:
        label = f"{_warning_label_token(tool_type)} (unknown type)"
    if namespace:
        label += f" in namespace {_warning_label_token(namespace)}"
    return label


def validate_responses_request(request: ResponsesRequest) -> None:
    """Reject request capabilities the endpoint cannot honour at all.

    Only capabilities that cannot be silently degraded are refused with a 400
    naming the field. Tool *declarations* are handled separately: a declaration
    the model cannot use is accepted but not exposed and reported through the
    Warning header (see ``convert_responses_tools``), while request modes that
    would change the meaning of the response (``truncation: "auto"``,
    ``tool_choice: "required"`` and so on) still fail loudly here.
    """
    tool_choice = request.tool_choice
    if isinstance(tool_choice, str):
        if tool_choice not in ("auto", "none"):
            detail = (
                "tool_choice='required' is not supported: no chat template "
                "oMLX serves can guarantee that the model emits a tool call."
                if tool_choice == "required"
                else f"tool_choice={tool_choice!r} is not a valid value."
            )
            raise InvalidRequestError(detail, field="tool_choice")
    elif isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            raise InvalidRequestError(
                "tool_choice for a named function is not supported: no chat "
                "template oMLX serves can guarantee that the model calls a "
                "specific function. Use 'auto' or 'none'.",
                field="tool_choice",
            )
        raise InvalidRequestError(
            "Only tool_choice 'auto' or 'none' is supported by /v1/responses.",
            field="tool_choice",
        )
    elif tool_choice is not None:
        raise InvalidRequestError(
            "tool_choice must be 'auto', 'none', or omitted.",
            field="tool_choice",
        )

    truncation = request.truncation
    if truncation is not None and truncation not in ("auto", "disabled"):
        raise InvalidRequestError(
            f"truncation={truncation!r} is not a valid value; use 'auto' or "
            "'disabled'.",
            field="truncation",
        )
    if truncation == "auto":
        raise InvalidRequestError(
            "truncation='auto' is not supported: oMLX does not trim a "
            "conversation to fit the context window, and a naive trim would "
            "break tool-call/tool-result and reasoning pairing. Use "
            "truncation='disabled' and shorten the input.",
            field="truncation",
        )

    if request.parallel_tool_calls is False:
        raise InvalidRequestError(
            "parallel_tool_calls=false is not supported: no chat template "
            "oMLX serves can guarantee a single tool call per turn, and "
            "dropping the model's extra calls would silently discard tool "
            "calls. Omit the field or set it to true.",
            field="parallel_tool_calls",
        )

    if request.background:
        raise InvalidRequestError(
            "background responses are not supported by /v1/responses.",
            field="background",
        )
    if request.conversation is not None:
        raise InvalidRequestError(
            "conversation objects are not supported by /v1/responses; use "
            "previous_response_id.",
            field="conversation",
        )
    if request.max_tool_calls is not None:
        raise InvalidRequestError(
            "max_tool_calls is not supported by /v1/responses.",
            field="max_tool_calls",
        )
    if request.top_logprobs is not None:
        raise InvalidRequestError(
            "top_logprobs is not supported by /v1/responses.",
            field="top_logprobs",
        )

    for include_value in request.include or []:
        if include_value not in _DEGRADED_INCLUDE_VALUES:
            raise InvalidRequestError(
                f"include={include_value!r} is not supported by "
                "/v1/responses; reasoning.encrypted_content is accepted as a "
                "no-op because reasoning is returned in the item itself.",
                field="include",
            )

    text_format = request.text.format if request.text else None
    if text_format is not None:
        if text_format.type not in ("text", "json_object", "json_schema"):
            raise InvalidRequestError(
                f"text.format.type={text_format.type!r} is not valid; use "
                "'text', 'json_object' or 'json_schema'.",
                field="text.format",
            )
        if text_format.type == "json_schema" and not text_format.schema_:
            raise InvalidRequestError(
                "text.format.type='json_schema' requires a 'schema' object.",
                field="text.format",
            )


# =============================================================================
# Input Conversion
# =============================================================================


def _reasoning_item_text(item: Any) -> str:
    """Read reasoning text from either published shape.

    ``summary[].text`` is OpenAI's hosted shape; ``content[].reasoning_text`` is
    the raw Responses dialect. Accepts an ``InputItem`` or a stored dict, and
    falls back to content only when no summary part carried text, so neither
    dialect's reasoning is silently lost. The two are written from the same
    string upstream, so joining both would duplicate it.
    """
    if isinstance(item, dict):
        summary = item.get("summary")
        content = item.get("content")
    else:
        summary = getattr(item, "summary", None)
        if not summary:
            summary = (getattr(item, "model_extra", None) or {}).get("summary")
        content = getattr(item, "content", None)

    parts: List[str] = []
    for block in summary or []:
        if isinstance(block, dict):
            text = block.get("text", "")
        else:
            text = getattr(block, "text", "")
        if text:
            parts.append(text)
    if parts:
        return "\n".join(parts)

    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "reasoning_text":
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts)


def convert_responses_input_to_messages(
    input_data: Optional[Union[str, List[InputItem]]],
    instructions: Optional[str] = None,
    previous_messages: Optional[List[Dict[str, Any]]] = None,
    consolidate_system_messages: bool = True,
    preserve_images: bool = False,
) -> List[Dict[str, Any]]:
    """Convert Responses API input to internal messages format.

    Args:
        input_data: String prompt or list of InputItem objects.
        instructions: System prompt (prepended as system message).
        previous_messages: Messages from previous_response_id chain.
        consolidate_system_messages: If True, merge all system/developer content
            into one leading system message for strict templates. Server code can
            set this to False and resolve placement after the target template is
            known.
        preserve_images: If True, images in function_call_output lists are
            preserved as a user message following the tool run so VLM engines
            can extract them. If False, they become a text placeholder.

    Returns:
        List of message dicts compatible with chat template.
    """
    messages: List[Dict[str, Any]] = []

    # Collect system/developer content to merge into a single system message
    # when strict-template compatibility mode is active. In deferred mode,
    # top-level instructions still form a leading system message, but input
    # system/developer items keep their original position until template
    # capability probing decides whether they can be preserved.
    system_parts: List[str] = []
    if instructions:
        system_parts.append(instructions)

    # Prepend previous response context
    if previous_messages:
        messages.extend(copy.deepcopy(previous_messages))
    current_message_start = len(messages)

    if input_data is None:
        if system_parts:
            messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
        return (
            _consolidate_system_messages(messages)
            if consolidate_system_messages
            else messages
        )

    if isinstance(input_data, str):
        if system_parts:
            messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
        messages.append({"role": "user", "content": input_data})
        return (
            _consolidate_system_messages(messages)
            if consolidate_system_messages
            else messages
        )

    # Process input items
    # Track pending tool calls for grouping into a single assistant message
    pending_tool_calls: List[Dict[str, Any]] = []
    # Track reasoning content to attach to the next assistant message
    pending_reasoning: str = ""
    # Track images extracted from function_call_output lists; flushed as a
    # user message once the consecutive tool-output run ends
    pending_tool_images: List[Dict[str, Any]] = []
    # Call ids emitted by function_call items, in order. An output that omits
    # call_id pairs with the oldest unmatched call instead of minting an
    # unrelated id, so a round trip the client left unlabelled still lines up.
    unmatched_call_ids: List[str] = []

    for item in input_data:
        # Resolve effective type: EasyInputMessage has no type field
        item_type = item.type
        if item_type is None and item.role is not None:
            item_type = "message"
        if item_type is None:
            raise InvalidRequestError(
                "Every input item needs a 'type' (or a 'role' for an "
                "EasyInputMessage).",
                field="input",
            )
        if item_type in _UNSUPPORTED_INPUT_ITEM_TYPES:
            raise InvalidRequestError(
                f"input item type {item_type!r} is not supported by "
                "/v1/responses. It carries a hosted tool result oMLX cannot "
                "produce; remove it or replay the turn as function_call / "
                "function_call_output items.",
                field="input",
            )
        if item_type not in (
            "message",
            "reasoning",
            "function_call",
            "function_call_output",
        ):
            raise InvalidRequestError(
                f"input item type {item_type!r} is not a valid Responses "
                "input type.",
                field="input",
            )

        if item_type != "function_call_output":
            _flush_pending_tool_images(messages, pending_tool_images)

        if item_type == "message":
            # Flush pending tool calls before a new message. Reasoning
            # passes through when no tool calls were flushed so it lands
            # on this message instead.
            pending_reasoning = _flush_pending_tool_calls(
                messages,
                pending_tool_calls,
                min_merge_index=current_message_start,
                pending_reasoning=pending_reasoning,
            )

            role = item.role or "user"
            if role not in _SUPPORTED_MESSAGE_ROLES:
                raise InvalidRequestError(
                    f"message role {role!r} is not supported by /v1/responses; "
                    "use user, assistant, system or developer.",
                    field="input",
                )
            # Map "developer" role to "system"
            if role == "developer":
                role = "system"

            content = item.content
            if isinstance(content, list):
                # Convert content parts - preserve images for VLM processing
                text_parts = []
                has_image = False
                converted_parts: List[Dict[str, Any]] = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type")
                        if part_type in ("input_text", "text", "output_text"):
                            text = part.get("text", "")
                            text_parts.append(text)
                            converted_parts.append({"type": "text", "text": text})
                        elif part_type == "input_image":
                            # Preserve image data for VLM engines
                            has_image = True
                            image_url = part.get("image_url", part.get("url", ""))
                            detail = part.get("detail", "auto")
                            converted_parts.append(
                                {
                                    "type": "input_image",
                                    "image_url": image_url,
                                    "detail": detail,
                                }
                            )
                        elif part_type not in _SUPPORTED_MESSAGE_PART_TYPES:
                            raise InvalidRequestError(
                                f"message content part type {part_type!r} is "
                                "not supported by /v1/responses; use "
                                "input_text or input_image.",
                                field="input",
                            )
                    elif isinstance(part, str):
                        text_parts.append(part)
                        converted_parts.append({"type": "text", "text": part})
                if has_image:
                    # Keep as content list so VLM can extract images
                    content = converted_parts
                else:
                    content = "\n".join(text_parts) if text_parts else ""

            # Merge system/developer messages into the single system block unless
            # the server is deferring placement until the template is known.
            if role == "system":
                if consolidate_system_messages:
                    system_parts.append(content or "")
                else:
                    messages.append({"role": "system", "content": content or ""})
            else:
                msg_dict: Dict[str, Any] = {"role": role, "content": content or ""}
                if role == "assistant" and pending_reasoning:
                    msg_dict["reasoning_content"] = pending_reasoning
                    pending_reasoning = ""
                messages.append(msg_dict)

        elif item_type == "reasoning":
            # Collect reasoning text to attach to the next assistant message
            # as reasoning_content. Clients that follow the raw dialect send
            # ``content[].reasoning_text`` instead of (or as well as) ``summary``;
            # reading only one of them silently loses the other's reasoning.
            reasoning_text = _reasoning_item_text(item)
            if reasoning_text:
                pending_reasoning = reasoning_text

        elif item_type == "function_call":
            # Assistant's tool call — accumulate for grouping
            call_id = ensure_call_id(item.call_id or item.id)
            unmatched_call_ids.append(call_id)
            namespace = getattr(item, "namespace", None)
            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.name or "",
                        "arguments": _try_parse_json(item.arguments or "{}"),
                        **({"namespace": namespace} if namespace else {}),
                    },
                }
            )

        elif item_type == "function_call_output":
            # Flush pending tool calls first. Any pending reasoning gets
            # attached to the synthesized assistant tool_calls message.
            pending_reasoning = _flush_pending_tool_calls(
                messages,
                pending_tool_calls,
                min_merge_index=current_message_start,
                pending_reasoning=pending_reasoning,
            )

            output_content = item.output or ""
            if isinstance(item.output, list):
                extracted = _extract_tool_output_text(
                    item.output,
                    pending_tool_images if preserve_images else None,
                )
                output_content = (
                    extracted if extracted is not None else json.dumps(item.output)
                )
            # An omitted call_id pairs with the oldest function_call still
            # waiting for its output, so client-omitted ids never reach the
            # template as two unrelated generated ids.
            explicit_call_id = item.call_id or item.id
            if explicit_call_id:
                call_id = ensure_call_id(explicit_call_id)
                if call_id in unmatched_call_ids:
                    unmatched_call_ids.remove(call_id)
            elif unmatched_call_ids:
                call_id = unmatched_call_ids.pop(0)
            else:
                call_id = ensure_call_id(None)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output_content,
                }
            )

    # Flush images from a trailing tool-output run, then any remaining
    # pending tool calls. If reasoning survived without a trailing
    # message, attach it to the synthesized tool_calls message.
    _flush_pending_tool_images(messages, pending_tool_images)
    _flush_pending_tool_calls(
        messages,
        pending_tool_calls,
        min_merge_index=current_message_start,
        pending_reasoning=pending_reasoning,
    )

    # Insert merged system message at position 0
    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

    return (
        _consolidate_system_messages(messages)
        if consolidate_system_messages
        else messages
    )


# =============================================================================
# Tool Conversion
# =============================================================================


def _as_responses_tool(tool: Any) -> Optional[ResponsesTool]:
    """Coerce one namespace member to a ResponsesTool, if it is shaped like one."""
    if isinstance(tool, ResponsesTool):
        return tool
    if isinstance(tool, dict):
        return ResponsesTool(**tool)
    return None


def _register_flat_tool(
    tool: ResponsesTool,
    registry: ToolBindingRegistry,
    unexposed: List[str],
) -> List[Dict[str, Any]]:
    """Register one top-level tool, or record it as accepted-but-unexposed."""
    if tool.type == "function":
        if not tool.name:
            raise InvalidRequestError(
                "A function tool requires a 'name'.",
                field="tools",
            )
        binding = registry.register(
            tool.name,
            description=tool.description,
            parameters=tool.parameters,
            strict=tool.strict,
        )
        return [binding.to_chat_tool()]
    if tool.type in _REDUNDANT_TOOL_TYPES:
        # tool_search only lazy-loads namespace members; they are all exposed
        # eagerly here, so the tool has nothing left to do and its absence
        # costs the model no capability.
        return []
    if tool.type in _CONTAINER_TOOL_TYPES:
        return _register_namespace_tool(tool, registry, unexposed)
    # A declaration oMLX cannot expose is accepted and named for the caller
    # rather than rejected: the model never sees it, so an unused declaration
    # cannot change the response, and a real Codex session (which always
    # declares web_search) must still complete (#3757).
    unexposed.append(_unexposed_tool_label(tool.type))
    return []


def _register_namespace_tool(
    tool: ResponsesTool,
    registry: ToolBindingRegistry,
    unexposed: List[str],
) -> List[Dict[str, Any]]:
    """Expand a namespace group into its member function tools.

    Namespace members are the one place ``type`` is not a top-level branch. A
    member the endpoint cannot expose is dropped and labelled with its group
    instead of failing, by the same declared-vs-used rule as a flat tool.
    """
    if not tool.name:
        raise InvalidRequestError(
            "A namespace tool requires a 'name'.",
            field="tools",
        )
    converted: List[Dict[str, Any]] = []
    for raw_member in getattr(tool, "tools", None) or []:
        member = _as_responses_tool(raw_member)
        if member is None:
            raise InvalidRequestError(
                f"Namespace {tool.name!r} contains a member that is not a "
                "tool object.",
                field="tools",
            )
        if member.type in _REDUNDANT_TOOL_TYPES:
            continue
        if member.type in _CONTAINER_TOOL_TYPES:
            # A nested namespace would need a second join level the client
            # cannot resolve, so it is never exposed; since nothing is exposed
            # for it, accepting and dropping the declaration cannot mislead
            # the model.
            unexposed.append(
                _unexposed_tool_label(member.type, namespace=tool.name)
            )
            continue
        if member.type != "function":
            unexposed.append(
                _unexposed_tool_label(member.type, namespace=tool.name)
            )
            continue
        if not member.name:
            raise InvalidRequestError(
                f"Namespace {tool.name!r} contains a function member without "
                "a 'name'.",
                field="tools",
            )
        binding = registry.register(
            member.name,
            namespace=tool.name,
            description=member.description,
            parameters=member.parameters,
            strict=member.strict,
        )
        converted.append(binding.to_chat_tool())
    return converted


def convert_responses_tools(
    tools: Optional[List[ResponsesTool]],
    registry: Optional[ToolBindingRegistry] = None,
    aliases: Optional[Dict[str, Tuple[str, str]]] = None,
    unexposed: Optional[List[str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Convert Responses API tools to Chat Completions tool definitions.

    Responses: {"type": "function", "name": "fn", "parameters": {...}}
    Chat Completions: {"type": "function", "function": {"name": "fn", ...}}

    A "namespace" tool is a container of ordinary client-executed function
    tools (the shape Codex uses for each MCP server), so its members are
    expanded under a joined wire name and recorded in ``registry`` so a
    resulting call can be returned with its namespace intact (#3371).

    Declaring a capability is separate from using it: a tool type oMLX cannot
    expose (a hosted/server-executed type, an unknown type, or a nested
    namespace) is accepted as a declaration and simply left out of the returned
    tool list, because the model never sees it and an unused declaration cannot
    change the response. Each such declaration is appended to ``unexposed`` (a
    human-readable label) so the caller can surface the degradation in a
    ``Warning`` header instead of dropping the tool silently. Only malformed
    declarations -- a function with no name, a namespace member that is not a
    tool object -- still raise.
    """
    if not tools:
        return None

    registry = registry if registry is not None else ToolBindingRegistry()
    sink = unexposed if unexposed is not None else []
    result: List[Dict[str, Any]] = []
    for tool in tools:
        result.extend(_register_flat_tool(tool, registry, sink))
    if aliases is not None:
        aliases.update(registry.aliases())
    return result if result else None


def split_namespace_tool_name(
    name: str,
    registry: Optional[ToolBindingRegistry] = None,
    aliases: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Tuple[Optional[str], str]:
    """Restore ``(namespace, name)`` for a call made by wire name.

    Flat tools are unaffected: they return ``(None, name)``.
    """
    if registry is not None:
        return registry.resolve(name)
    if aliases:
        entry = aliases.get(name)
        if entry:
            return entry
    return None, name


def apply_namespace_tool_aliases(
    messages: List[Dict[str, Any]],
    registry: Optional[ToolBindingRegistry] = None,
    aliases: Optional[Dict[str, Tuple[str, str]]] = None,
) -> None:
    """Map preserved history identities to the current request's tool names."""
    if registry is not None:
        registry.apply_to_messages(messages)
        return
    wire_names = {identity: wire for wire, identity in (aliases or {}).items()}
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            namespace = function.pop("namespace", None)
            if namespace:
                name = function["name"]
                fallback = f"{namespace.rstrip('_')}__{name.lstrip('_')}"
                function["name"] = wire_names.get((namespace, name), fallback)


# =============================================================================
# Response Building
# =============================================================================


def build_message_output_item(
    text: str,
    item_id: Optional[str] = None,
    status: str = "completed",
) -> OutputItem:
    """Build a message-type OutputItem."""
    return OutputItem(
        type="message",
        id=item_id or generate_id(IDPrefix.MESSAGE),
        status=status,
        role="assistant",
        content=[OutputContent(type="output_text", text=text)],
    )


def build_function_call_output_item(
    name: str,
    arguments: str,
    call_id: str,
    item_id: Optional[str] = None,
    status: str = "completed",
    namespace: Optional[str] = None,
) -> OutputItem:
    """Build a function_call-type OutputItem."""
    return OutputItem(
        type="function_call",
        id=item_id or generate_id(IDPrefix.FUNCTION_CALL),
        status=status,
        call_id=call_id,
        name=name,
        arguments=arguments,
        namespace=namespace,
    )


def build_reasoning_output_item(
    reasoning_text: str,
    item_id: Optional[str] = None,
    status: str = "completed",
) -> OutputItem:
    """Build a reasoning-type OutputItem carrying the full CoT.

    The text is published in both shapes the ecosystem reads: ``summary``, which
    OpenAI's own hosts emit, and a ``reasoning_text`` content part, which is what
    Responses-dialect clients look at (``@ai-sdk/open-responses`` reads
    ``item.content[].text`` and would otherwise drop the reasoning entirely).
    Emitting both is additive — a client that knows one shape ignores the other.
    """
    summary = [ReasoningSummaryPart(text=reasoning_text)] if reasoning_text else []
    content = (
        [OutputContent(type="reasoning_text", text=reasoning_text)]
        if reasoning_text
        else []
    )
    return OutputItem(
        type="reasoning",
        id=item_id or generate_id(IDPrefix.REASONING),
        status=status,
        summary=summary,
        content=content,
    )


def build_response_usage(
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
) -> ResponseUsage:
    """Build ResponseUsage from token counts."""
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=cached_tokens),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=reasoning_tokens),
    )


def build_response_object(
    request: ResponsesRequest,
    *,
    response_id: str,
    created_at: int,
    output_items: List[Any],
    usage: Optional[ResponseUsage],
    truncated: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    status: Optional[str] = None,
) -> ResponseObject:
    """Build the one response envelope both response paths serialize.

    A single builder keeps the non-streaming body and the streaming terminal
    (and opening) events from drifting apart in which request fields they echo.
    The streaming caller still serializes with ``exclude_none``, so its
    null-valued fields stay omitted: that predates this builder and the
    streaming integration tests pin it.

    ``status`` overrides the truncation-derived status so the streaming
    ``response.created`` / ``response.in_progress`` snapshots can come from the
    same builder as the terminal event.
    """
    return ResponseObject(
        id=response_id,
        created_at=created_at,
        model=request.model,
        status=status or ("incomplete" if truncated else "completed"),
        output=output_items,
        usage=usage,
        tools=request.tools or [],
        tool_choice=request.tool_choice or "auto",
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=request.max_output_tokens,
        previous_response_id=request.previous_response_id,
        incomplete_details={"reason": "max_output_tokens"} if truncated else None,
        instructions=request.instructions,
        store=request.store,
        parallel_tool_calls=request.parallel_tool_calls,
        reasoning=request.reasoning,
        text=request.text,
        metadata=request.metadata or {},
        truncation=request.truncation or "disabled",
    )


# =============================================================================
# SSE Event Formatting
# =============================================================================


def format_sse_event(event_type: str, data: Any) -> str:
    """Format a Responses API SSE event.

    Returns: "event: {type}\\ndata: {json}\\n\\n"
    """
    if isinstance(data, str):
        json_str = data
    elif hasattr(data, "model_dump"):
        json_str = json.dumps(data.model_dump(exclude_none=True))
    elif isinstance(data, dict):
        json_str = json.dumps(data)
    else:
        json_str = json.dumps(data)
    return f"event: {event_type}\ndata: {json_str}\n\n"


# =============================================================================
# Response Store (previous_response_id support)
# =============================================================================

MAX_STORED_RESPONSES = 1000


class ResponseStore:
    """Bounded persisted store for response state and public responses."""

    def __init__(
        self,
        max_size: int = MAX_STORED_RESPONSES,
        state_dir: Optional[Union[str, Path]] = None,
    ):
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size
        self._state_dir = Path(state_dir).expanduser().resolve() if state_dir else None
        if self._state_dir:
            self._state_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted_records()

    @property
    def state_dir(self) -> Optional[Path]:
        """Resolved directory used for persisted response state."""
        return self._state_dir

    def _record_path(self, response_id: str) -> Optional[Path]:
        if self._state_dir is None:
            return None
        return self._state_dir / f"{response_id}.json"

    def _normalize_record(
        self,
        response_id: str,
        response_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if "public_response" in response_data:
            record = copy.deepcopy(response_data)
            record.setdefault("response_id", response_id)
            record.setdefault(
                "created_at", record.get("public_response", {}).get("created_at", 0)
            )
            record.setdefault(
                "previous_response_id",
                record.get("public_response", {}).get("previous_response_id"),
            )
            record.setdefault("input_messages", [])
            record.setdefault(
                "output_messages",
                normalize_response_output_to_messages(
                    record.get("public_response", {}).get("output", [])
                ),
            )
            return record

        public_response = copy.deepcopy(response_data)
        public_response.setdefault("id", response_id)
        return {
            "response_id": response_id,
            "previous_response_id": public_response.get("previous_response_id"),
            "input_messages": [],
            "output_messages": normalize_response_output_to_messages(
                public_response.get("output", [])
            ),
            "public_response": public_response,
            "created_at": public_response.get("created_at", 0),
        }

    def _persist_record(self, record: Dict[str, Any]) -> None:
        path = self._record_path(record["response_id"])
        if path is None:
            return
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
        tmp_path.replace(path)

    def _remove_persisted_record(self, response_id: str) -> None:
        path = self._record_path(response_id)
        if path is None or not path.exists():
            return
        path.unlink()

    def _evict_oldest(self) -> None:
        while len(self._store) > self._max_size:
            response_id, _record = self._store.popitem(last=False)
            self._remove_persisted_record(response_id)

    def _load_persisted_records(self) -> None:
        assert self._state_dir is not None
        loaded: List[Dict[str, Any]] = []
        for path in sorted(self._state_dir.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                response_id = raw.get("response_id") or raw.get(
                    "public_response", {}
                ).get("id")
                if not response_id:
                    raise ValueError("missing response_id")
                loaded.append(self._normalize_record(response_id, raw))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Skipping corrupt response state file %s: %s", path, exc)

        loaded.sort(
            key=lambda record: (record.get("created_at", 0), record["response_id"])
        )
        for record in loaded:
            self._store[record["response_id"]] = record
        self._evict_oldest()

    def put(self, response_id: str, response_data: Dict[str, Any]) -> None:
        """Store response state, evicting oldest records if needed."""
        record = self._normalize_record(response_id, response_data)
        if response_id in self._store:
            self._store.move_to_end(response_id)
        self._store[response_id] = record
        self._persist_record(record)
        self._evict_oldest()

    def get_record(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response-state record."""
        data = self._store.get(response_id)
        if data is not None:
            self._store.move_to_end(response_id)
            return copy.deepcopy(data)
        return None

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the public response object for a stored record."""
        data = self.get_record(response_id)
        if data is None:
            return None
        return data.get("public_response")

    def resolve_chain_messages(self, response_id: str) -> List[Dict[str, Any]]:
        """Resolve the full previous_response_id chain into message history."""
        if response_id not in self._store:
            raise ResponseStateNotFoundError(f"Response state not found: {response_id}")

        chain: List[Dict[str, Any]] = []
        seen: set[str] = set()
        current_id: Optional[str] = response_id
        while current_id:
            if current_id in seen:
                raise ResponseStateCorruptError(
                    f"Cycle detected in previous_response_id chain at {current_id}"
                )
            seen.add(current_id)
            record = self._store.get(current_id)
            if record is None:
                raise ResponseStateCorruptError(
                    f"Missing ancestor response state: {current_id}"
                )
            self._store.move_to_end(current_id)
            chain.append(record)
            current_id = record.get("previous_response_id")

        chain.reverse()
        messages: List[Dict[str, Any]] = []
        for record in chain:
            messages.extend(copy.deepcopy(record.get("input_messages", [])))
            messages.extend(copy.deepcopy(record.get("output_messages", [])))
        return _consolidate_system_messages(messages)

    def delete(self, response_id: str) -> bool:
        """Delete a stored response. Returns True if found."""
        if response_id not in self._store:
            return False
        del self._store[response_id]
        self._remove_persisted_record(response_id)
        return True

    def __len__(self) -> int:
        return len(self._store)


# =============================================================================
# Previous Response Conversion
# =============================================================================


def normalize_response_output_to_messages(
    output_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert response output items to assistant/tool-call history messages."""
    messages: List[Dict[str, Any]] = []
    pending_tool_calls: List[Dict[str, Any]] = []
    pending_reasoning: str = ""

    for item in output_items:
        item_type = item.get("type")
        if item_type == "reasoning":
            pending_reasoning = _reasoning_item_text(item)
        elif item_type == "message":
            pending_reasoning = _flush_pending_tool_calls(
                messages,
                pending_tool_calls,
                pending_reasoning=pending_reasoning,
            )
            content_blocks = item.get("content", [])
            text_parts = []
            for block in content_blocks:
                if block.get("type") == "output_text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "refusal":
                    # A refusal is model-visible history too; dropping it would
                    # silently rewrite the turn the client already saw.
                    text_parts.append(block.get("refusal", ""))
            msg_dict: Dict[str, Any] = {
                "role": item.get("role", "assistant"),
                "content": "\n".join(p for p in text_parts if p),
            }
            if pending_reasoning:
                msg_dict["reasoning_content"] = pending_reasoning
                pending_reasoning = ""
            messages.append(msg_dict)
        elif item_type == "function_call":
            call_id = ensure_call_id(item.get("call_id"))
            namespace = item.get("namespace")
            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": _try_parse_json(item.get("arguments", "{}")),
                        **({"namespace": namespace} if namespace else {}),
                    },
                }
            )
        else:
            raise ResponseStateCorruptError(
                f"Stored response contains an unsupported output item type "
                f"{item_type!r}; the conversation state cannot be replayed."
            )

    _flush_pending_tool_calls(
        messages,
        pending_tool_calls,
        pending_reasoning=pending_reasoning,
    )
    return _consolidate_system_messages(messages)


def build_response_store_record(
    public_response: Dict[str, Any],
    input_messages: List[Dict[str, Any]],
    output_messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a persisted response-state record."""
    return {
        "response_id": public_response.get("id", ""),
        "previous_response_id": public_response.get("previous_response_id"),
        "input_messages": copy.deepcopy(input_messages),
        "output_messages": copy.deepcopy(output_messages),
        "public_response": copy.deepcopy(public_response),
        "created_at": public_response.get("created_at", 0),
    }
