# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Utility functions for text processing.
"""

import json
import re
from typing import Any, List

from .openai_models import Message

# Model families whose chat templates consume message.reasoning_content directly.
_NATIVE_REASONING_MODEL_TYPES = {"minimax_m3", "minimax_m3_vl"}


def uses_native_reasoning_content(
    model_name: str | None = None,
    *,
    config_model_type: str | None = None,
    engine_model_type: str | None = None,
    preserve_thinking_default: bool | None = None,
) -> bool:
    """Return whether history should keep reasoning in message fields."""
    if preserve_thinking_default is True:
        return True

    if config_model_type in _NATIVE_REASONING_MODEL_TYPES:
        return True
    if engine_model_type in _NATIVE_REASONING_MODEL_TYPES:
        return True

    lowered = (model_name or "").lower()
    return "minimax" in lowered and "m3" in lowered


# =============================================================================
# Partial Mode Detection
# =============================================================================


def detect_and_strip_partial(messages: list[dict]) -> bool:
    """Check if the final assistant message has partial=True; strip the field from all messages.

    Partial mode signals that the model should continue from the final assistant
    message rather than starting a new turn.  The ``partial`` key is not part of
    the chat-template contract, so it is always removed before the messages are
    passed to ``apply_chat_template``.

    Args:
        messages: List of message dicts (mutated in-place).

    Returns:
        True if the final message is an assistant message with ``partial=True``.
    """
    is_partial = (
        bool(messages)
        and messages[-1].get("role") == "assistant"
        and messages[-1].get("partial", False)
    )
    for msg in messages:
        msg.pop("partial", None)
    return is_partial


# =============================================================================
# Special Token Patterns
# =============================================================================

# Pattern to match special tokens that should be removed from output
SPECIAL_TOKENS_PATTERN = re.compile(
    r"<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|"
    r"<\|end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|"
    r"<\|image\|>|<\|audio\|>|"  # Gemma 4 VLM special tokens
    r"\[e~\[|\]~b\]|\]~!b\[|\]!p~\[|\]!d~\[|"  # MiniMax M3 special tokens
    r"</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]|"
    r"<eos>|<bos>|<end_of_turn>|<start_of_turn>"  # Gemma special tokens (fixes #1087)
)


def clean_special_tokens(text: str) -> str:
    """Clean model output by removing only special tokens.

    Preserves <think>...</think> blocks for downstream processing.

    Args:
        text: Raw model output

    Returns:
        Text with special tokens removed but think tags preserved
    """
    if not text:
        return text
    return SPECIAL_TOKENS_PATTERN.sub("", text).strip()


def remove_special_tokens_preserve_whitespace(text: str) -> str:
    """Remove special tokens without trimming surrounding whitespace."""
    if not text:
        return text
    return SPECIAL_TOKENS_PATTERN.sub("", text)


def clean_output_text(text: str) -> str:
    """Clean model output by removing special tokens and thinking blocks.

    Args:
        text: Raw model output

    Returns:
        Cleaned text with special tokens and <think> blocks removed
    """
    if not text:
        return text
    text = SPECIAL_TOKENS_PATTERN.sub("", text)
    from .thinking import extract_thinking

    _, content = extract_thinking(text)
    return content.strip()


# =============================================================================
# Text Content Extraction
# =============================================================================


def _extract_text_from_content_list(content: list) -> str:
    """Extract text parts from a content array, dropping non-text items.

    Handles content arrays from both OpenAI and Anthropic formats.
    Only items with type="text" are extracted; all others (tool_use,
    image, image_url, thinking, refusal, etc.) are silently dropped.
    """
    text_parts = []
    for item in content:
        # Convert Pydantic models to dict
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        elif hasattr(item, "dict"):
            item = item.dict()

        if isinstance(item, dict):
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        elif isinstance(item, str):
            # Direct string in content list
            text_parts.append(item)

    return "\n".join(text_parts) if text_parts else ""


def _extract_multimodal_content_list(content: list) -> list:
    """Extract text, image, and audio parts from a content array.

    Keeps text, image_url, and input_audio items for VLM processing.
    Other content types (tool_use, thinking, refusal, etc.) are dropped.
    """
    parts = []
    for item in content:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        elif hasattr(item, "dict"):
            item = item.dict()
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type in ("text", "input_text"):
                text = item.get("text") or item.get("content") or ""
                parts.append({"type": "text", "text": text})
            elif item_type == "image_url":
                image_url_value = item.get("image_url")
                url = None
                if isinstance(image_url_value, str):
                    url = image_url_value
                elif isinstance(image_url_value, dict):
                    url = image_url_value.get("url")
                if url:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                        }
                    )
            elif item_type == "input_image":
                image_url_value = item.get("image_url", item.get("input_image"))
                url = None
                if isinstance(image_url_value, str):
                    url = image_url_value
                elif isinstance(image_url_value, dict):
                    url = image_url_value.get("url")
                if url:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": url},
                        }
                    )
            elif item_type == "image":
                # Anthropic format: convert to OpenAI image_url format
                source = item.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{data}",
                            },
                        }
                    )
            elif item_type == "input_audio":
                # OpenAI audio format: pass through for engine-side decoding
                input_audio = item.get("input_audio")
                if input_audio and isinstance(input_audio, dict):
                    parts.append(
                        {
                            "type": "input_audio",
                            "input_audio": input_audio,
                        }
                    )
    return parts


# Roles eligible for merging when consecutive.
# System and tool messages are excluded: system messages have distinct semantics
# (e.g., JSON schema instructions), and tool messages carry tool_call_id.
_MERGEABLE_ROLES = {"user", "assistant"}
_PRESERVE_BOUNDARY_KEY = "_preserve_role_boundary"

# Match `role == "tool"` / `role == 'tool'` in a chat template.
_TOOL_ROLE_CHECK_RE = re.compile(r"==\s*['\"]tool['\"]")

_MID_SYSTEM_USER_MARKER = "__OMLX_MID_SYSTEM_PROBE_USER__"
_MID_SYSTEM_MARKER = "__OMLX_MID_SYSTEM_PROBE_SYSTEM__"
_MID_SYSTEM_ASSISTANT_MARKER = "__OMLX_MID_SYSTEM_PROBE_ASSISTANT__"
_MID_SYSTEM_PROBE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "omlx_probe_tool",
            "description": "oMLX chat-template probe tool.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
_MID_SYSTEM_PROBE_CACHE: dict[tuple[Any, ...], bool] = {}


def _chat_template_supports_tool_role(tokenizer: Any) -> bool:
    """Check whether the tokenizer's chat template renders tool messages natively.

    mlx-lm / mlx-vlm only set ``has_tool_calling`` when their marker-based
    ``_infer_tool_parser`` recognises the chat template (qwen3_coder, json_tools,
    gemma4, etc.). Templates that branch on ``role == "tool"`` and render
    ``tool_calls`` but don't match any known marker (Qwen3 VL variants, custom
    fine-tunes) get flattened to ``role: "user"`` — making the model treat tool
    output as user instructions and breaking multi-turn tool flows (#1290).

    Strict superset of ``has_tool_calling``: if the tokenizer already flags
    itself, return True immediately. Otherwise probe the chat_template string
    for both a ``role == "tool"`` equality and the ``tool_calls`` variable —
    both together keep false positives down (a stray ``"tool"`` literal in a
    comment isn't enough).
    """
    if getattr(tokenizer, "has_tool_calling", False):
        return True
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        return False
    if not _TOOL_ROLE_CHECK_RE.search(chat_template):
        return False
    return "tool_calls" in chat_template


def _freeze_template_value(value: Any) -> Any:
    """Convert chat-template kwargs into a hashable cache-key value."""
    if isinstance(value, dict):
        return tuple(
            sorted((str(k), _freeze_template_value(v)) for k, v in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_template_value(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_template_value(v) for v in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _mid_system_probe_cache_key(
    tokenizer: Any,
    *,
    has_tools: bool,
    chat_template_kwargs: dict[str, Any] | None,
    placement: str,
    is_partial: bool,
) -> tuple[Any, ...]:
    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, str):
        template_fingerprint: Any = hash(chat_template)
    else:
        template_fingerprint = repr(chat_template)
    return (
        id(tokenizer),
        template_fingerprint,
        has_tools,
        _freeze_template_value(chat_template_kwargs or {}),
        placement,
        is_partial,
    )


def _apply_mid_system_probe_template(
    tokenizer: Any,
    probe_messages: list[dict],
    *,
    has_tools: bool,
    chat_template_kwargs: dict[str, Any] | None,
    is_partial: bool,
) -> str:
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": not is_partial,
    }
    if is_partial:
        template_kwargs["continue_final_message"] = True
    if has_tools:
        template_kwargs["tools"] = _MID_SYSTEM_PROBE_TOOL
    if chat_template_kwargs:
        template_kwargs.update(chat_template_kwargs)

    try:
        rendered = tokenizer.apply_chat_template(probe_messages, **template_kwargs)
    except TypeError:
        if chat_template_kwargs:
            for key in chat_template_kwargs:
                template_kwargs.pop(key, None)
        template_kwargs.pop("tools", None)
        template_kwargs.pop("enable_thinking", None)
        rendered = tokenizer.apply_chat_template(probe_messages, **template_kwargs)

    if isinstance(rendered, str):
        return rendered
    if isinstance(rendered, list):
        return " ".join(str(token) for token in rendered)
    return str(rendered)


def chat_template_preserves_mid_system(
    tokenizer: Any | None,
    *,
    tools: list[dict] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    placement: str = "tail",
    is_partial: bool = False,
) -> bool:
    """Return whether the chat template renders a mid-system message in-place.

    This does not prove model-level semantics. It only verifies that the
    current tokenizer template keeps the system content after the preceding
    user turn instead of raising, dropping it, or moving it to the front.
    """
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return False
    if placement not in {"tail", "between"}:
        return False

    has_tools = bool(tools)
    cache_key = _mid_system_probe_cache_key(
        tokenizer,
        has_tools=has_tools,
        chat_template_kwargs=chat_template_kwargs,
        placement=placement,
        is_partial=is_partial,
    )
    cached = _MID_SYSTEM_PROBE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    probe_messages = [
        {"role": "user", "content": _MID_SYSTEM_USER_MARKER},
        {"role": "system", "content": _MID_SYSTEM_MARKER},
    ]
    if placement == "between":
        probe_messages.append(
            {"role": "assistant", "content": _MID_SYSTEM_ASSISTANT_MARKER}
        )

    try:
        rendered = _apply_mid_system_probe_template(
            tokenizer,
            probe_messages,
            has_tools=has_tools,
            chat_template_kwargs=chat_template_kwargs,
            is_partial=is_partial,
        )
    except Exception:
        _MID_SYSTEM_PROBE_CACHE[cache_key] = False
        return False

    user_idx = rendered.find(_MID_SYSTEM_USER_MARKER)
    system_idx = rendered.find(_MID_SYSTEM_MARKER)
    assistant_idx = rendered.find(_MID_SYSTEM_ASSISTANT_MARKER)

    supported = user_idx >= 0 and system_idx > user_idx
    if placement == "between":
        supported = supported and assistant_idx > system_idx

    _MID_SYSTEM_PROBE_CACHE[cache_key] = supported
    return supported


def _system_content_as_text(content: Any) -> str:
    if isinstance(content, list):
        return _extract_text_from_content_list(content)
    return content if isinstance(content, str) else str(content)


def _is_system_role(role: Any) -> bool:
    return role in {"system", "developer"}


def _merge_consecutive_system_messages(messages: list[dict]) -> list[dict]:
    """Merge adjacent system messages in-place without moving their position."""
    merged: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if not _is_system_role(msg.get("role")):
            merged.append(msg)
            i += 1
            continue

        parts: list[str] = []
        while i < len(messages) and _is_system_role(messages[i].get("role")):
            content = messages[i].get("content", "")
            if content:
                text = _system_content_as_text(content)
                if text:
                    parts.append(text)
            i += 1

        if parts:
            merged.append({"role": "system", "content": "\n\n".join(parts)})

    return merged


def _mid_system_placement_kinds(messages: list[dict]) -> set[str] | None:
    """Classify supported cache-preserving mid-system placements.

    Returns None when any non-leading system run has an unsupported position.
    """
    placements: set[str] = set()
    seen_non_system = False
    i = 0
    while i < len(messages):
        role = messages[i].get("role")
        if not _is_system_role(role):
            seen_non_system = True
            i += 1
            continue

        start = i
        while i < len(messages) and _is_system_role(messages[i].get("role")):
            i += 1

        if not seen_non_system:
            continue

        prev_role = messages[start - 1].get("role") if start > 0 else None
        next_role = messages[i].get("role") if i < len(messages) else None
        if prev_role != "user":
            return None
        if next_role is None:
            placements.add("tail")
        elif next_role == "assistant":
            placements.add("between")
        else:
            return None

    return placements


def has_nonleading_system_message(messages: list[dict]) -> bool:
    """Return True when a system message appears after a non-system turn."""
    seen_non_system = False
    for msg in messages:
        if _is_system_role(msg.get("role")):
            if seen_non_system:
                return True
        else:
            seen_non_system = True
    return False


def _is_text_only_content_list(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    for part in content:
        if not isinstance(part, dict):
            return False
        if part.get("type", "text") != "text":
            return False
        text = part.get("text", "")
        if text is not None and not isinstance(text, str):
            return False
    return True


def _is_safe_user_note_target(msg: dict | None) -> bool:
    if not msg or msg.get("role") != "user":
        return False
    if msg.get(_PRESERVE_BOUNDARY_KEY):
        return False
    if msg.get("tool_calls") or msg.get("tool_call_id") or msg.get("tool_responses"):
        return False
    content = msg.get("content", "")
    return (
        content is None
        or isinstance(content, str)
        or _is_text_only_content_list(content)
    )


def _message_has_tool_calls(msg: dict | None) -> bool:
    return bool(msg and msg.get("role") == "assistant" and msg.get("tool_calls"))


def _format_system_note(parts: list[str]) -> str:
    return "[System note]\n" + "\n\n".join(parts) + "\n[/System note]"


def _merge_note_text(existing: str, note: str, *, placement: str) -> str:
    if not existing:
        return note
    if placement == "prepend":
        return f"{note}\n\n{existing}"
    return f"{existing}\n\n{note}"


def _rewrite_user_content_with_note(
    msg: dict,
    note: str,
    *,
    placement: str,
) -> dict:
    rewritten = dict(msg)
    content = rewritten.get("content", "")
    if isinstance(content, list):
        parts = [dict(part) for part in content]
        if not parts:
            rewritten["content"] = [{"type": "text", "text": note}]
            return rewritten
        index = 0 if placement == "prepend" else len(parts) - 1
        existing = parts[index].get("text") or ""
        parts[index]["text"] = _merge_note_text(
            existing,
            note,
            placement=placement,
        )
        rewritten["content"] = parts
        return rewritten

    existing = content if isinstance(content, str) else ""
    rewritten["content"] = _merge_note_text(existing, note, placement=placement)
    return rewritten


def _downgrade_mid_system_to_user_notes(messages: list[dict]) -> list[dict] | None:
    """Move unsupported non-leading system runs into adjacent safe user text.

    This keeps the native chat template/tool rendering path intact while making
    volatile tail notes cache-friendly. It deliberately refuses tool-call
    boundaries and multimodal user content, where changing roles is too risky.
    """
    rewritten: list[dict] = []
    seen_non_system = False
    i = 0

    while i < len(messages):
        msg = messages[i]
        if not _is_system_role(msg.get("role")):
            rewritten.append(msg)
            seen_non_system = True
            i += 1
            continue

        start = i
        parts: list[str] = []
        while i < len(messages) and _is_system_role(messages[i].get("role")):
            content = messages[i].get("content", "")
            if content:
                text = _system_content_as_text(content)
                if text:
                    parts.append(text)
            i += 1

        if not seen_non_system:
            rewritten.extend(messages[start:i])
            continue
        if not parts:
            continue

        note = _format_system_note(parts)
        next_msg = messages[i] if i < len(messages) else None
        next_role = next_msg.get("role") if next_msg is not None else None

        if _is_safe_user_note_target(rewritten[-1] if rewritten else None) and (
            next_msg is None or next_role == "assistant"
        ):
            rewritten[-1] = _rewrite_user_content_with_note(
                rewritten[-1],
                note,
                placement="append",
            )
            continue

        if _is_safe_user_note_target(next_msg):
            if _message_has_tool_calls(rewritten[-1] if rewritten else None):
                return None
            rewritten.append(
                _rewrite_user_content_with_note(
                    next_msg,
                    note,
                    placement="prepend",
                )
            )
            seen_non_system = True
            i += 1
            continue

        return None

    return rewritten


def prepare_system_messages_for_template(
    messages: list[dict],
    tokenizer: Any | None,
    *,
    tools: list[dict] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    is_partial: bool = False,
    merge_consecutive_roles: bool = True,
    unsupported_mid_system_policy: str = "strict",
) -> list[dict]:
    """Preserve cache-friendly mid-system turns when the template supports them.

    Unsupported placements or templates fall back to the historical behavior:
    all system messages are consolidated at the front.
    """
    messages = [dict(msg) for msg in messages]
    if unsupported_mid_system_policy not in {"strict", "user_note_safe"}:
        unsupported_mid_system_policy = "strict"

    def strict_fallback() -> list[dict]:
        prepared = _consolidate_system_messages(messages)
        if merge_consecutive_roles:
            prepared = _merge_consecutive_roles(prepared)
        return prepared

    def unsupported_fallback() -> list[dict]:
        if unsupported_mid_system_policy == "user_note_safe":
            prepared = _downgrade_mid_system_to_user_notes(messages)
            if prepared is not None:
                # _downgrade preserves leading system blocks as-is; merge
                # consecutive system messages so strict templates (Qwen3.6+)
                # that require a single leading system message don't fail.
                prepared = _merge_consecutive_system_messages(prepared)
                if merge_consecutive_roles:
                    prepared = _merge_consecutive_roles(prepared)
                return prepared
        return strict_fallback()

    placements = _mid_system_placement_kinds(messages)
    if not placements:
        if placements is None:
            return unsupported_fallback()
        return _merge_consecutive_system_messages(messages)

    if is_partial:
        return strict_fallback()

    can_preserve = all(
        chat_template_preserves_mid_system(
            tokenizer,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
            placement=placement,
            is_partial=is_partial,
        )
        for placement in placements
    )
    if can_preserve:
        return _merge_consecutive_system_messages(messages)

    return unsupported_fallback()


def _drop_void_assistant_messages(messages: list[dict]) -> list[dict]:
    """Drop assistant messages that have no content and no tool_calls.

    Strict chat templates (e.g., Devstral/Mistral) raise an error when an
    assistant message has empty content and no tool_calls.  These void messages
    carry no information and can appear when a client echoes back a response
    that had only tool calls which were not preserved in its history.

    Messages with ``tool_responses`` (Gemma 4 format) or ``reasoning_content``
    (Qwen 3.6+ native reasoning field) are never dropped even when content is
    empty — they carry their own payload the template renders.
    """
    return [
        msg
        for msg in messages
        if not (
            msg.get("role") == "assistant"
            and not msg.get("content")
            and not msg.get("tool_calls")
            and not msg.get("tool_responses")
            and not msg.get("reasoning_content")
        )
    ]


def _consolidate_system_messages(messages: list[dict]) -> list[dict]:
    """Move all system messages to the front, merged into one.

    Models with strict chat templates (e.g., Qwen3.5) require the system
    message to appear first.  Clients may send system or developer messages
    mid-conversation, so we consolidate them defensively.
    """
    system_parts: list[str] = []
    non_system: list[dict] = []
    for msg in messages:
        if _is_system_role(msg.get("role")):
            content = msg.get("content", "")
            if content:
                if isinstance(content, list):
                    text = _extract_text_from_content_list(content)
                    if text:
                        system_parts.append(text)
                else:
                    system_parts.append(content)
        else:
            non_system.append(msg)

    if not system_parts:
        return messages

    merged_system = {"role": "system", "content": "\n\n".join(system_parts)}
    return [merged_system] + non_system


def _merge_consecutive_roles(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages with the same mergeable role.

    Models with strict chat templates (e.g., Gemma-3) enforce alternating
    user/assistant roles and reject consecutive same-role messages.
    OpenAI's API accepts these, so we merge them for compatibility.

    Args:
        messages: List of processed message dicts with 'role' and 'content'.

    Returns:
        New list with consecutive same-role messages merged using "\\n\\n".
    """
    if not messages:
        return messages

    merged: list[dict] = [messages[0].copy()]

    for msg in messages[1:]:
        prev = merged[-1]
        if (
            msg["role"] == prev["role"]
            and msg["role"] in _MERGEABLE_ROLES
            and not prev.get(_PRESERVE_BOUNDARY_KEY)
            and not msg.get(_PRESERVE_BOUNDARY_KEY)
        ):
            prev_content = prev.get("content", "")
            new_content = msg.get("content", "")
            if prev_content and new_content:
                prev_is_list = isinstance(prev_content, list)
                new_is_list = isinstance(new_content, list)
                if prev_is_list or new_is_list:
                    # Convert both to list form for safe concatenation
                    if not prev_is_list:
                        prev_content = [{"type": "text", "text": prev_content}]
                    if not new_is_list:
                        new_content = [{"type": "text", "text": new_content}]
                    prev["content"] = prev_content + new_content
                else:
                    prev["content"] = prev_content + "\n\n" + new_content
            elif new_content:
                prev["content"] = new_content
        else:
            merged.append(msg.copy())

    return merged


def _apply_reasoning_reconstruction(
    role: str,
    content: Any,
    reasoning: str | None,
    native: bool,
) -> tuple[Any, str | None]:
    """Reconstruct reasoning on a historical assistant message.

    External clients echo reasoning back via the OpenAI ``reasoning_content``
    field (or Anthropic ``thinking`` blocks).  Chat templates fall into two
    camps:

    * ``native=True`` — template understands ``message.reasoning_content``
      as a top-level field (Qwen 3.6+).  Content stays clean and reasoning
      travels separately.
    * ``native=False`` — template only parses ``<think>...</think>`` embedded
      in content.  Reasoning is inlined into content as a fallback.

    Returns ``(new_content, reasoning_out)`` where ``reasoning_out`` is the
    string to attach as a ``reasoning_content`` field, or ``None`` to skip.
    """
    if role != "assistant" or not reasoning:
        if role != "assistant" or not native:
            return content, None
        text = content if isinstance(content, str) else ""
        if isinstance(content, list):
            text = _extract_text_from_content_list(content)
        from .thinking import extract_thinking

        inline_reasoning, inline_content = extract_thinking(text)
        if inline_reasoning:
            return inline_content, inline_reasoning
        return content, None
    text = content if isinstance(content, str) else ""
    if isinstance(content, list):
        text = _extract_text_from_content_list(content)
    if native:
        return text, reasoning
    return f"<think>\n{reasoning}\n</think>\n\n{text}", None


def extract_text_content(
    messages: List[Message],
    max_tool_result_tokens: int | None = None,
    tokenizer: Any | None = None,
    native_reasoning_content: bool = False,
    consolidate_system_messages: bool = True,
) -> List[dict]:
    """
    Extract text content from OpenAI-format messages.

    Handles:
    - Simple text messages
    - Content arrays (extracts text parts only)
    - Tool call messages (assistant with tool_calls)
    - Tool response messages (role="tool")

    Args:
        messages: List of Message objects
        max_tool_result_tokens: Maximum token count for tool results.
        tokenizer: Tokenizer instance for token counting and truncation.
        native_reasoning_content: If True, pass ``reasoning_content`` through
            as a message-level field (Qwen 3.6+ templates).  If False, inline
            ``<think>...</think>`` into content as a fallback.
        consolidate_system_messages: If True, preserve historical strict-template
            behavior by moving system messages to the front. Server code can
            set this to False and call ``prepare_system_messages_for_template``
            after tools/template kwargs are known.

    Returns:
        List of {"role": str, "content": str}
    """
    processed_messages = []

    for msg in messages:
        role = msg.role
        content = msg.content

        # Reconstruct reasoning for historical assistant messages.  Native
        # mode passes reasoning as a separate field; fallback inlines it as
        # <think>...</think> in content.
        reasoning = getattr(msg, "reasoning_content", None)
        content, reasoning_out = _apply_reasoning_reconstruction(
            role, content, reasoning, native_reasoning_content
        )

        # Normalize "developer" role to "system" (OpenAI API compatibility)
        if role == "developer":
            role = "system"

        # Handle tool response messages (role="tool")
        if role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", None) or ""
            # Convert list content to string if needed
            if isinstance(content, list):
                tool_content = _extract_text_from_content_list(content)
            else:
                tool_content = content if content else ""
            # Apply truncation if configured
            if max_tool_result_tokens and tokenizer and tool_content:
                from .anthropic_utils import truncate_tool_result

                tool_content = truncate_tool_result(
                    tool_content, max_tool_result_tokens, tokenizer
                )
            # Preserve structured format for models with native tool calling
            # so the chat template renders tool results in the model's native format
            if _chat_template_supports_tool_role(tokenizer):
                processed_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_content,
                    }
                )
            else:
                processed_messages.append(
                    {
                        "role": "user",  # mlx-lm expects user/assistant roles
                        "content": f"[Tool Result ({tool_call_id})]: {tool_content}",
                        _PRESERVE_BOUNDARY_KEY: True,
                    }
                )
            continue

        # Handle assistant messages with tool_calls
        if role == "assistant" and hasattr(msg, "tool_calls") and msg.tool_calls:
            if isinstance(content, list):
                content = _extract_text_from_content_list(content)
            msg_dict = {"role": role, "content": content if content else ""}
            if reasoning_out is not None:
                msg_dict["reasoning_content"] = reasoning_out
            if getattr(msg, "name", None):
                msg_dict["name"] = msg.name
            if getattr(msg, "partial", False):
                msg_dict["partial"] = True

            # Preserve structured tool_calls for models with native tool calling
            # so the chat template renders them in the model's native format.
            # Without this, models mimic text-formatted tool calls from history
            # instead of generating their native parseable format.
            if _chat_template_supports_tool_role(tokenizer):
                tool_calls_list = []
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        tool_calls_list.append(
                            {
                                "id": tc.get("id", ""),
                                "function": {
                                    "name": func.get("name", ""),
                                    "arguments": _try_parse_json(
                                        func.get("arguments", "{}")
                                    ),
                                },
                            }
                        )
                    else:
                        args_str = (
                            getattr(tc.function, "arguments", "{}")
                            if hasattr(tc, "function")
                            else "{}"
                        )
                        tool_calls_list.append(
                            {
                                "id": getattr(tc, "id", ""),
                                "function": {
                                    "name": (
                                        getattr(tc.function, "name", "")
                                        if hasattr(tc, "function")
                                        else ""
                                    ),
                                    "arguments": _try_parse_json(args_str),
                                },
                            }
                        )
                msg_dict["tool_calls"] = tool_calls_list
            else:
                # Text fallback for models without native tool calling
                tool_calls_text = []
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        name = func.get("name", "unknown")
                        args = func.get("arguments", "{}")
                        tool_calls_text.append(f"[Calling tool: {name}({args})]")
                text = msg_dict["content"]
                if tool_calls_text:
                    text = (text + "\n" if text else "") + "\n".join(tool_calls_text)
                msg_dict["content"] = text
            msg_dict[_PRESERVE_BOUNDARY_KEY] = True

            processed_messages.append(msg_dict)
            continue

        # Build optional extra fields from the source message
        _extra: dict = {}
        if getattr(msg, "name", None):
            _extra["name"] = msg.name
        if getattr(msg, "partial", False):
            _extra["partial"] = True
        if reasoning_out is not None:
            _extra["reasoning_content"] = reasoning_out

        # Handle None content
        if content is None:
            processed_messages.append({"role": role, "content": "", **_extra})
            continue

        if isinstance(content, str):
            # Simple text message
            processed_messages.append({"role": role, "content": content, **_extra})
        elif isinstance(content, list):
            # Content array - extract text parts only
            combined_text = _extract_text_from_content_list(content)
            processed_messages.append(
                {"role": role, "content": combined_text, **_extra}
            )
        else:
            # Unknown format, try to convert
            processed_messages.append({"role": role, "content": str(content), **_extra})

    processed_messages = _drop_void_assistant_messages(processed_messages)
    if consolidate_system_messages:
        processed_messages = _consolidate_system_messages(processed_messages)
    return _merge_consecutive_roles(processed_messages)


def extract_multimodal_content(
    messages: List[Message],
    max_tool_result_tokens: int | None = None,
    tokenizer: Any | None = None,
    native_reasoning_content: bool = False,
    consolidate_system_messages: bool = True,
) -> List[dict]:
    """
    Extract content from messages, preserving image_url parts for VLM.

    Same as extract_text_content but keeps image_url content parts
    in their original list format for VLM processing.

    Args:
        messages: List of Message objects
        max_tool_result_tokens: Maximum token count for tool results.
        tokenizer: Tokenizer instance for token counting and truncation.
        native_reasoning_content: If True, pass ``reasoning_content`` through
            as a message-level field.  See ``extract_text_content``.
        consolidate_system_messages: See ``extract_text_content``.

    Returns:
        List of message dicts. Messages with images have content as list.
    """
    processed_messages = []

    for msg in messages:
        role = msg.role
        content = msg.content

        # Reconstruct reasoning (see extract_text_content).
        reasoning = getattr(msg, "reasoning_content", None)
        content, reasoning_out = _apply_reasoning_reconstruction(
            role, content, reasoning, native_reasoning_content
        )

        if role == "developer":
            role = "system"

        # Tool response messages - same as extract_text_content
        if role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", None) or ""
            # Convert list content to string if needed
            if isinstance(content, list):
                tool_content = _extract_text_from_content_list(content)
            else:
                tool_content = content if content else ""
            if max_tool_result_tokens and tokenizer and tool_content:
                from .anthropic_utils import truncate_tool_result

                tool_content = truncate_tool_result(
                    tool_content, max_tool_result_tokens, tokenizer
                )
            if _chat_template_supports_tool_role(tokenizer):
                processed_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_content,
                    }
                )
            else:
                processed_messages.append(
                    {
                        "role": "user",
                        "content": f"[Tool Result ({tool_call_id})]: {tool_content}",
                        _PRESERVE_BOUNDARY_KEY: True,
                    }
                )
            continue

        # Assistant with tool_calls - same as extract_text_content
        if role == "assistant" and hasattr(msg, "tool_calls") and msg.tool_calls:
            if isinstance(content, list):
                content = _extract_text_from_content_list(content)
            msg_dict = {"role": role, "content": content if content else ""}
            if reasoning_out is not None:
                msg_dict["reasoning_content"] = reasoning_out
            if getattr(msg, "name", None):
                msg_dict["name"] = msg.name
            if getattr(msg, "partial", False):
                msg_dict["partial"] = True

            if _chat_template_supports_tool_role(tokenizer):
                tool_calls_list = []
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        tool_calls_list.append(
                            {
                                "id": tc.get("id", ""),
                                "function": {
                                    "name": func.get("name", ""),
                                    "arguments": _try_parse_json(
                                        func.get("arguments", "{}")
                                    ),
                                },
                            }
                        )
                    else:
                        args_str = (
                            getattr(tc.function, "arguments", "{}")
                            if hasattr(tc, "function")
                            else "{}"
                        )
                        tool_calls_list.append(
                            {
                                "id": getattr(tc, "id", ""),
                                "function": {
                                    "name": (
                                        getattr(tc.function, "name", "")
                                        if hasattr(tc, "function")
                                        else ""
                                    ),
                                    "arguments": _try_parse_json(args_str),
                                },
                            }
                        )
                msg_dict["tool_calls"] = tool_calls_list
            else:
                tool_calls_text = []
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        name = func.get("name", "unknown")
                        args = func.get("arguments", "{}")
                        tool_calls_text.append(f"[Calling tool: {name}({args})]")
                text = msg_dict["content"]
                if tool_calls_text:
                    text = (text + "\n" if text else "") + "\n".join(tool_calls_text)
                msg_dict["content"] = text
            msg_dict[_PRESERVE_BOUNDARY_KEY] = True

            processed_messages.append(msg_dict)
            continue

        # Build optional extra fields from the source message
        _extra: dict = {}
        if getattr(msg, "name", None):
            _extra["name"] = msg.name
        if getattr(msg, "partial", False):
            _extra["partial"] = True
        if reasoning_out is not None:
            _extra["reasoning_content"] = reasoning_out

        if content is None:
            processed_messages.append({"role": role, "content": "", **_extra})
            continue

        if isinstance(content, str):
            processed_messages.append({"role": role, "content": content, **_extra})
        elif isinstance(content, list):
            # Preserve image_url and input_audio parts for VLM processing
            multimodal_parts = _extract_multimodal_content_list(content)
            multimodal_types = {"image_url", "input_audio"}
            has_multimodal = any(
                p.get("type") in multimodal_types for p in multimodal_parts
            )
            if has_multimodal:
                # Keep as content list for VLM engine
                processed_messages.append(
                    {"role": role, "content": multimodal_parts, **_extra}
                )
            else:
                # Text-only, flatten to string
                combined_text = _extract_text_from_content_list(content)
                processed_messages.append(
                    {"role": role, "content": combined_text, **_extra}
                )
        else:
            processed_messages.append({"role": role, "content": str(content), **_extra})

    processed_messages = _drop_void_assistant_messages(processed_messages)
    if consolidate_system_messages:
        processed_messages = _consolidate_system_messages(processed_messages)
    return processed_messages


# =============================================================================
# Harmony (gpt-oss) Message Extraction
# =============================================================================


def _try_parse_json(s: str):
    """
    Try to parse a string as JSON. Returns parsed dict/list if valid JSON,
    otherwise returns the original string.

    This is needed because Harmony chat_template uses |tojson filter,
    which would double-encode strings that are already JSON.
    """
    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s:
        return s
    # Quick check: must start with { or [ to be JSON object/array
    if not (s.startswith("{") or s.startswith("[")):
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return s


def _wrap_truncated_for_harmony(truncated_text: str) -> dict:
    """Wrap truncated tool result in a dict for Harmony |tojson compatibility.

    The Harmony chat_template applies |tojson to tool result content.
    When truncation breaks valid JSON, the content becomes a string, and
    |tojson would double-encode it (wrapping in quotes and escaping).
    This function wraps the truncated text in a dict so |tojson produces
    a clean JSON object instead.

    Args:
        truncated_text: Text with truncation notice appended.

    Returns:
        Dict with 'output' key containing the truncated content and
        'truncated' key with a human-readable summary.
    """
    match = re.search(
        r'\n\n<truncated total_tokens="(\d+)" shown_tokens="(\d+)" />\s*$',
        truncated_text,
    )
    if match:
        return {
            "output": truncated_text[: match.start()],
            "truncated": f"Showing {match.group(2)} of {match.group(1)} tokens",
        }
    return {"output": truncated_text}


def extract_harmony_messages(
    messages: list,
    max_tool_result_tokens: int | None = None,
    tokenizer: Any | None = None,
    consolidate_system_messages: bool = True,
) -> List[dict]:
    """
    Extract messages for Harmony (gpt-oss) models.

    Unlike extract_text_content(), this function preserves:
    - tool messages: role="tool" with tool_call_id (chat_template handles conversion)
    - assistant tool_calls: tool_calls field intact (chat_template handles conversion)

    The Harmony chat_template expects standard OpenAI format and converts:
    - role="tool" → <|start|>functions.{name} to=assistant<|channel|>commentary...
    - assistant.tool_calls → <|start|>assistant to=functions.{name}<|channel|>commentary...

    IMPORTANT: The chat_template uses |tojson filter on:
    - tool_call.arguments (line 299)
    - message.content for tool results (line 322)

    If these are already JSON strings, |tojson would double-encode them.
    So we parse JSON strings to dicts before passing to the template.

    Args:
        messages: List of Message objects
        max_tool_result_tokens: Maximum token count for tool results.
        tokenizer: Tokenizer instance for token counting and truncation.
        consolidate_system_messages: See ``extract_text_content``.

    Returns:
        List of message dicts with tool-related fields preserved
    """
    processed_messages = []

    # Normalize to plain dicts -- callers may pass Pydantic Message
    # objects (OpenAI path) or plain dicts (Anthropic path).
    raw: list[dict] = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            raw.append(msg.model_dump())
        elif isinstance(msg, dict):
            raw.append(dict(msg))
        else:
            d: dict = {
                "role": getattr(msg, "role", "user"),
                "content": getattr(msg, "content", ""),
            }
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id is not None:
                d["tool_call_id"] = tool_call_id
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls is not None:
                d["tool_calls"] = tool_calls
            raw.append(d)

    for msg in raw:
        role = msg.get("role", "user")
        content = msg.get("content")

        # Normalize "developer" role to "system" (OpenAI API compatibility)
        if role == "developer":
            role = "system"

        # Tool response messages - preserve role and tool_call_id
        # Parse content as JSON if possible (chat_template applies |tojson)
        if role == "tool":
            # Convert list content to string if needed
            if isinstance(content, list):
                tool_content = _extract_text_from_content_list(content)
            else:
                tool_content = content if content else ""
            if max_tool_result_tokens and tokenizer and tool_content:
                from .anthropic_utils import truncate_tool_result

                # Parse JSON BEFORE truncation for better line-boundary cuts.
                # Harmony chat_template applies |tojson to content, so content
                # must be a dict (not a string) to avoid double-encoding.
                parsed_json = _try_parse_json(tool_content)
                if isinstance(parsed_json, (dict, list)):
                    # Valid JSON - pretty-print for line-boundary truncation
                    pretty = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                    truncated = truncate_tool_result(
                        pretty, max_tool_result_tokens, tokenizer
                    )
                    if "<truncated " in truncated:
                        # Truncation broke JSON - wrap in dict for |tojson
                        parsed_content = _wrap_truncated_for_harmony(truncated)
                    else:
                        # Not truncated - use parsed dict/list
                        parsed_content = parsed_json
                else:
                    # Not JSON - truncate raw text, keep as string
                    parsed_content = truncate_tool_result(
                        tool_content, max_tool_result_tokens, tokenizer
                    )
            else:
                # No truncation configured - just parse JSON if possible
                parsed_content = _try_parse_json(tool_content)
            processed_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", "") or "",
                    "content": parsed_content,
                }
            )
            continue

        # Assistant messages - preserve tool_calls field
        if role == "assistant":
            msg_dict = {"role": role}

            # Handle content (may be string or list)
            if content is None:
                msg_dict["content"] = ""
            elif isinstance(content, str):
                msg_dict["content"] = content
            elif isinstance(content, list):
                # Extract text parts from content array
                msg_dict["content"] = _extract_text_from_content_list(content)
            else:
                msg_dict["content"] = str(content)

            # Preserve tool_calls field for chat_template
            # Parse arguments as JSON if possible (chat_template applies |tojson)
            if msg.get("tool_calls"):
                tool_calls_list = []
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        args_str = tc.get("function", {}).get("arguments", "{}")
                        tool_calls_list.append(
                            {
                                "id": tc.get("id", ""),
                                "function": {
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": _try_parse_json(args_str),
                                },
                            }
                        )
                    else:
                        # Pydantic model
                        args_str = (
                            getattr(tc.function, "arguments", "{}")
                            if hasattr(tc, "function")
                            else "{}"
                        )
                        tool_calls_list.append(
                            {
                                "id": getattr(tc, "id", ""),
                                "function": {
                                    "name": (
                                        getattr(tc.function, "name", "")
                                        if hasattr(tc, "function")
                                        else ""
                                    ),
                                    "arguments": _try_parse_json(args_str),
                                },
                            }
                        )
                msg_dict["tool_calls"] = tool_calls_list
                msg_dict[_PRESERVE_BOUNDARY_KEY] = True

            processed_messages.append(msg_dict)
            continue

        # Other messages (user, system, developer)
        if content is None:
            processed_messages.append({"role": role, "content": ""})
        elif isinstance(content, str):
            processed_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Extract text parts from content array
            processed_messages.append(
                {"role": role, "content": _extract_text_from_content_list(content)}
            )
        else:
            processed_messages.append({"role": role, "content": str(content)})

    processed_messages = _drop_void_assistant_messages(processed_messages)
    if consolidate_system_messages:
        processed_messages = _consolidate_system_messages(processed_messages)
    return _merge_consecutive_roles(processed_messages)
