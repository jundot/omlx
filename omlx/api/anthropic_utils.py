# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for Anthropic Messages API conversion.

Handles conversion between Anthropic API format and internal oMLX format.
"""

import base64
import json
import logging
import uuid
from typing import Any

from .anthropic_models import (
    AnthropicTool,
    AnthropicUsage,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolUse,
    MessagesRequest,
    MessagesResponse,
    SystemContent,
)
from .openai_models import ToolCall

_PRESERVE_ROLE_BOUNDARY = "_preserve_role_boundary"
logger = logging.getLogger(__name__)


def _decode_document_block(block_dict: dict[str, Any]) -> str:
    """Decode an Anthropic document content block to text.

    For text/plain documents, decodes base64 data and returns the text.
    For other media types (e.g. PDF), returns a placeholder message since
    oMLX does not provide document parsing.
    """
    source = block_dict.get("source", {})
    media_type = source.get("media_type", "")
    data = source.get("data", "")
    title = block_dict.get("title", "")

    if media_type == "text/plain" and data:
        try:
            decoded = base64.b64decode(data).decode("utf-8")
            label = f"[Document: {title}]\n" if title else ""
            return f"{label}{decoded}"
        except Exception:
            return f"[Document: {title or 'untitled'} — failed to decode]"

    label = title or "untitled"
    return (
        f"[Document: {label} ({media_type}) — "
        f"oMLX does not provide PDF parsing. Send as text instead.]"
    )


def _content_block_to_dict(block: Any) -> dict[str, Any] | None:
    """Normalize Anthropic content blocks to dicts."""
    if hasattr(block, "model_dump"):
        return block.model_dump()
    if isinstance(block, dict):
        return block
    return None


def _append_anthropic_image_part(image_parts: list[dict], block_dict: dict[str, Any]) -> None:
    """Convert Anthropic image blocks to OpenAI-style image_url parts."""
    source = block_dict.get("source", {})
    if source.get("type") == "base64":
        media_type = source.get("media_type", "image/jpeg")
        data = source.get("data", "")
        image_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{data}",
            },
        })
    elif source.get("type") == "url":
        image_parts.append({
            "type": "image_url",
            "image_url": {
                "url": source.get("url", ""),
            },
        })


def _extract_images_from_tool_result_content(
    content: Any, image_parts: list[dict]
) -> None:
    """Extract image blocks from tool result content for VLM processing."""
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                _append_anthropic_image_part(image_parts, item)
    elif isinstance(content, dict) and content.get("type") == "image":
        _append_anthropic_image_part(image_parts, content)


def _build_message_from_parts(
    role: str,
    text_parts: list[str],
    image_parts: list[dict],
) -> dict[str, Any] | None:
    """Build a single internal message from accumulated text/image parts."""
    if image_parts:
        content_parts = list(image_parts)
        if text_parts:
            content_parts.append({
                "type": "text",
                "text": "\n".join(text_parts),
            })
        return {"role": role, "content": content_parts}

    if text_parts:
        return {"role": role, "content": "\n".join(text_parts)}

    return None

# =============================================================================
# Message Conversion: Anthropic -> Internal
# =============================================================================


def convert_anthropic_to_internal(
    request: MessagesRequest,
    max_tool_result_tokens: int | None = None,
    tokenizer: Any | None = None,
    preserve_images: bool = False,
    native_reasoning_content: bool = False,
) -> list[dict[str, Any]]:
    """
    Convert Anthropic Messages API format to internal format.

    Handles:
    - System message from separate 'system' field
    - Content blocks to text
    - Tool results and tool uses in message history
    - Image blocks (when preserve_images=True for VLM)

    Args:
        request: Anthropic MessagesRequest
        max_tool_result_tokens: Maximum token count for tool results.
        tokenizer: Tokenizer instance for token counting and truncation.
        preserve_images: If True, preserve image blocks as OpenAI image_url
            format for VLM processing.
        native_reasoning_content: If True, attach Anthropic ``thinking`` blocks
            as a ``reasoning_content`` field on assistant messages (Qwen 3.6+
            templates).  If False, inline each block as ``<think>...</think>``
            in the message content as a fallback.

    Returns:
        List of {"role": str, "content": str or list}
    """
    processed_messages: list[dict[str, Any]] = []
    native_tool_calling = bool(tokenizer and getattr(tokenizer, "has_tool_calling", False))

    # Handle system message (Anthropic has separate 'system' field)
    if request.system:
        system_text = _extract_system_text(request.system)
        if system_text:
            processed_messages.append({"role": "system", "content": system_text})

    # Process messages
    for msg in request.messages:
        role = msg.role
        content = msg.content

        if isinstance(content, str):
            # Simple text message
            processed_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            if native_tool_calling:
                if role == "assistant":
                    text_parts: list[str] = []
                    image_parts: list[dict] = []
                    tool_calls: list[dict] = []
                    thinking_parts: list[str] = []
                    for block in content:
                        block_dict = _content_block_to_dict(block)
                        if block_dict is None:
                            continue
                        block_type = block_dict.get("type", "")
                        if block_type == "text":
                            text_parts.append(block_dict.get("text", ""))
                        elif block_type == "image" and preserve_images:
                            _append_anthropic_image_part(image_parts, block_dict)
                        elif block_type == "tool_use":
                            tool_input = block_dict.get("input", {})
                            if isinstance(tool_input, str):
                                try:
                                    tool_input = json.loads(tool_input)
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            tool_calls.append({
                                "id": block_dict.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                "function": {
                                    "name": block_dict.get("name", ""),
                                    "arguments": tool_input,
                                },
                            })
                        elif block_type == "thinking":
                            # Native mode: collect for reasoning_content field.
                            # Fallback: inline as <think>...</think> in source
                            # order (Anthropic emits thinking first, so appending
                            # preserves the natural ordering).
                            thinking_text = block_dict.get("thinking", "")
                            if thinking_text:
                                if native_reasoning_content:
                                    thinking_parts.append(thinking_text)
                                else:
                                    text_parts.append(f"<think>\n{thinking_text}\n</think>")
                        elif block_type == "document":
                            text_parts.append(_decode_document_block(block_dict))
                    msg_dict = _build_message_from_parts(role, text_parts, image_parts) or {
                        "role": role,
                        "content": "",
                    }
                    if thinking_parts:
                        msg_dict["reasoning_content"] = "\n".join(thinking_parts)
                    if tool_calls:
                        msg_dict["tool_calls"] = tool_calls
                        msg_dict[_PRESERVE_ROLE_BOUNDARY] = True
                    processed_messages.append(msg_dict)
                    continue

                if role == "user":
                    text_parts = []
                    image_parts = []
                    saw_tool_result = False
                    for block in content:
                        block_dict = _content_block_to_dict(block)
                        if block_dict is None:
                            continue
                        block_type = block_dict.get("type", "")
                        if block_type == "text":
                            text_parts.append(block_dict.get("text", ""))
                        elif block_type == "image" and preserve_images:
                            _append_anthropic_image_part(image_parts, block_dict)
                        elif block_type == "tool_result":
                            msg_dict = _build_message_from_parts(role, text_parts, image_parts)
                            if msg_dict:
                                processed_messages.append(msg_dict)
                            text_parts = []
                            image_parts = []
                            saw_tool_result = True
                            processed_messages.append({
                                "role": "tool",
                                "tool_call_id": block_dict.get("tool_use_id", ""),
                                "content": _extract_tool_result_content(
                                    block_dict.get("content", ""),
                                    max_tokens=max_tool_result_tokens,
                                    tokenizer=tokenizer,
                                ),
                            })
                            if preserve_images:
                                _extract_images_from_tool_result_content(
                                    block_dict.get("content", ""), image_parts
                                )
                        elif block_type == "thinking":
                            # User messages don't carry reasoning_content in the
                            # Qwen 3.6 template, so native mode simply drops these
                            # blocks.  Fallback keeps the legacy <think> inline
                            # behaviour in source order.
                            thinking_text = block_dict.get("thinking", "")
                            if thinking_text and not native_reasoning_content:
                                text_parts.append(f"<think>\n{thinking_text}\n</think>")
                        elif block_type == "document":
                            text_parts.append(_decode_document_block(block_dict))
                    msg_dict = _build_message_from_parts(role, text_parts, image_parts)
                    if msg_dict:
                        processed_messages.append(msg_dict)
                    elif not saw_tool_result:
                        processed_messages.append({"role": role, "content": ""})
                    continue

            # Content blocks list
            text_parts: list[str] = []
            image_parts: list[dict] = []
            thinking_parts: list[str] = []
            saw_tool_markup = False
            for block in content:
                block_dict = _content_block_to_dict(block)
                if block_dict is None:
                    continue

                block_type = block_dict.get("type", "")

                if block_type == "text":
                    text_parts.append(block_dict.get("text", ""))

                elif block_type == "image" and preserve_images:
                    _append_anthropic_image_part(image_parts, block_dict)

                elif block_type == "tool_use":
                    # Tool use in assistant message (model called a tool)
                    tool_name = block_dict.get("name", "")
                    tool_input = block_dict.get("input", {})
                    text_parts.append(
                        f"[Calling tool: {tool_name}({json.dumps(tool_input)})]"
                    )
                    saw_tool_markup = True

                elif block_type == "tool_result":
                    # Tool result in user message (user providing tool output)
                    tool_use_id = block_dict.get("tool_use_id", "")
                    result_content = _extract_tool_result_content(
                        block_dict.get("content", ""),
                        max_tokens=max_tool_result_tokens,
                        tokenizer=tokenizer,
                    )
                    is_error = block_dict.get("is_error", False)
                    prefix = "[Tool Error" if is_error else "[Tool Result"
                    text_parts.append(f"{prefix} ({tool_use_id})]: {result_content}")
                    saw_tool_markup = True
                    if preserve_images:
                        _extract_images_from_tool_result_content(
                            block_dict.get("content", ""), image_parts
                        )

                elif block_type == "thinking":
                    # Native mode: collect for reasoning_content (assistant only).
                    # Fallback: inline as <think>...</think> in source order.
                    thinking_text = block_dict.get("thinking", "")
                    if thinking_text:
                        if native_reasoning_content and role == "assistant":
                            thinking_parts.append(thinking_text)
                        elif not native_reasoning_content:
                            text_parts.append(f"<think>\n{thinking_text}\n</think>")

                elif block_type == "document":
                    text_parts.append(_decode_document_block(block_dict))

            msg_dict = _build_message_from_parts(role, text_parts, image_parts) or {
                "role": role,
                "content": "",
            }
            if thinking_parts:
                msg_dict["reasoning_content"] = "\n".join(thinking_parts)
            if saw_tool_markup:
                msg_dict[_PRESERVE_ROLE_BOUNDARY] = True
            processed_messages.append(msg_dict)
        else:
            # Unknown format
            processed_messages.append({"role": role, "content": str(content)})

    from .utils import _merge_consecutive_roles

    return _merge_consecutive_roles(processed_messages)


def convert_anthropic_to_internal_harmony(
    request: MessagesRequest,
    max_tool_result_tokens: int | None = None,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Convert Anthropic Messages API format to internal format for Harmony (gpt-oss) models.

    Unlike convert_anthropic_to_internal(), this function preserves:
    - tool_use blocks as assistant.tool_calls field
    - tool_result blocks as role="tool" messages

    The Harmony chat_template expects these fields to properly generate
    the Harmony format tool calling syntax.

    Args:
        request: Anthropic MessagesRequest

    Returns:
        List of message dicts with tool-related fields preserved
    """
    processed_messages: list[dict[str, Any]] = []

    # Handle system message (Anthropic has separate 'system' field)
    if request.system:
        system_text = _extract_system_text(request.system)
        if system_text:
            processed_messages.append({"role": "system", "content": system_text})

    # Process messages
    for msg in request.messages:
        role = msg.role
        content = msg.content

        if isinstance(content, str):
            # Simple text message
            processed_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Content blocks list - need to separate tool_use, tool_result, and text
            text_parts: list[str] = []
            tool_calls: list[dict] = []
            tool_results: list[dict] = []

            for block in content:
                # Handle both Pydantic models and dicts
                if hasattr(block, "model_dump"):
                    block_dict = block.model_dump()
                elif isinstance(block, dict):
                    block_dict = block
                else:
                    continue

                block_type = block_dict.get("type", "")

                if block_type == "text":
                    text_parts.append(block_dict.get("text", ""))

                elif block_type == "tool_use":
                    # Tool use in assistant message - preserve as tool_calls
                    tool_id = block_dict.get("id", f"call_{uuid.uuid4().hex[:8]}")
                    tool_name = block_dict.get("name", "")
                    tool_input = block_dict.get("input", {})
                    # input should be dict for chat_template |tojson
                    if isinstance(tool_input, str):
                        try:
                            tool_input = json.loads(tool_input)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    tool_calls.append({
                        "id": tool_id,
                        "function": {
                            "name": tool_name,
                            "arguments": tool_input,  # dict, not string
                        }
                    })

                elif block_type == "tool_result":
                    # Tool result - will be converted to role="tool" message
                    tool_use_id = block_dict.get("tool_use_id", "")
                    result_content = block_dict.get("content", "")

                    if isinstance(result_content, str):
                        # Try JSON parse BEFORE truncation so we can pretty-print
                        parsed_json = None
                        try:
                            parsed_json = json.loads(result_content)
                        except (json.JSONDecodeError, ValueError):
                            pass

                        if parsed_json is not None and max_tool_result_tokens and tokenizer:
                            # Valid JSON - pretty-print for better line-based truncation
                            pretty = json.dumps(parsed_json, indent=2, ensure_ascii=False)
                            truncated = truncate_tool_result(
                                pretty, max_tool_result_tokens, tokenizer
                            )
                            if "<truncated " in truncated:
                                # Truncation broke JSON - wrap in dict for
                                # Harmony |tojson compatibility
                                from .utils import _wrap_truncated_for_harmony

                                result_content = _wrap_truncated_for_harmony(
                                    truncated
                                )
                            else:
                                # Not truncated - pass as parsed object
                                result_content = parsed_json
                        elif parsed_json is not None:
                            # Valid JSON, no truncation configured - pass as parsed object
                            result_content = parsed_json
                        else:
                            # Not JSON - apply truncation to raw text
                            result_content = _extract_tool_result_content(
                                result_content,
                                max_tokens=max_tool_result_tokens,
                                tokenizer=tokenizer,
                            )
                    elif isinstance(result_content, list):
                        # Extract text from content blocks
                        extracted = _extract_tool_result_content(
                            result_content,
                            max_tokens=max_tool_result_tokens,
                            tokenizer=tokenizer,
                        )
                        # Only try json.loads if content was NOT truncated
                        if isinstance(extracted, str) and "<truncated " not in extracted:
                            try:
                                result_content = json.loads(extracted)
                            except (json.JSONDecodeError, ValueError):
                                result_content = extracted
                        elif isinstance(extracted, str) and "<truncated " in extracted:
                            # Check if pre-truncation content was JSON-like
                            content_part = extracted.split(
                                "\n\n<truncated"
                            )[0].strip()
                            if content_part and content_part[0] in "{[":
                                from .utils import _wrap_truncated_for_harmony

                                result_content = _wrap_truncated_for_harmony(
                                    extracted
                                )
                            else:
                                result_content = extracted
                        else:
                            result_content = extracted
                    tool_results.append({
                        "tool_use_id": tool_use_id,
                        "content": result_content,
                    })

                elif block_type == "thinking":
                    # Thinking blocks are ignored (reasoning content is not passed to model)
                    continue

                elif block_type == "document":
                    text_parts.append(_decode_document_block(block_dict))

            # Build message(s) based on what we found
            if role == "assistant":
                # Assistant message with potential tool_calls
                msg_dict = {"role": "assistant", "content": "\n".join(text_parts) if text_parts else ""}
                if tool_calls:
                    msg_dict["tool_calls"] = tool_calls
                processed_messages.append(msg_dict)
            elif role == "user":
                # User message - may contain tool_results
                # First add any text content
                if text_parts:
                    processed_messages.append({"role": "user", "content": "\n".join(text_parts)})

                # Add each tool_result as a separate role="tool" message
                for tr in tool_results:
                    processed_messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": tr["content"],  # dict or string
                    })
            else:
                # Other roles
                processed_messages.append({"role": role, "content": "\n".join(text_parts) if text_parts else ""})
        else:
            # Unknown format
            processed_messages.append({"role": role, "content": str(content)})

    from .utils import _merge_consecutive_roles

    return _merge_consecutive_roles(processed_messages)


# Prefix to filter out from system blocks (billing metadata that
# contains randomly changing values, breaking prefix cache).
_BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"


def _extract_system_text(system: str | list[SystemContent]) -> str:
    """Extract text from system field."""
    if isinstance(system, str):
        return system
    elif isinstance(system, list):
        text_parts = []
        for block in system:
            if hasattr(block, "text"):
                text = block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
            else:
                continue
            # Skip billing header blocks (contain random values that break prefix cache)
            if text.startswith(_BILLING_HEADER_PREFIX):
                continue
            text_parts.append(text)
        return "\n".join(text_parts)
    return ""


def truncate_tool_result(
    text: str,
    max_tokens: int,
    tokenizer: Any,
) -> str:
    """Truncate tool result text to fit within a token budget.

    Strategy:
    1. Encode the full text to count tokens.
    2. If within budget, return as-is.
    3. Decode tokens up to the budget to get an approximate character position.
    4. Search backwards for the last newline to truncate at a line boundary.
    5. Append a truncation notice as a separate XML tag.

    Args:
        text: The full tool result text.
        max_tokens: Maximum number of tokens allowed.
        tokenizer: Tokenizer with encode()/decode() methods.

    Returns:
        The (possibly truncated) text with notice appended.
    """
    token_ids = tokenizer.encode(text)
    total_tokens = len(token_ids)

    if total_tokens <= max_tokens:
        return text

    # Decode tokens up to budget to get approximate char position
    truncated_text = tokenizer.decode(token_ids[:max_tokens])

    # Find last newline for line-boundary truncation
    last_newline = truncated_text.rfind("\n")
    if last_newline > 0 and last_newline > len(truncated_text) * 0.5:
        # Only use line boundary if we don't lose more than 50% of content
        truncated_text = truncated_text[:last_newline]

    # Recount actual tokens after line-boundary adjustment
    shown_tokens = len(tokenizer.encode(truncated_text))

    logger.info(
        f"Tool result truncated: {total_tokens} -> {shown_tokens} tokens "
        f"({len(text)} -> {len(truncated_text)} chars)"
    )

    notice = (
        f"\n\n<truncated total_tokens=\"{total_tokens}\" "
        f"shown_tokens=\"{shown_tokens}\" />"
    )

    return truncated_text + notice


def _extract_tool_result_content(
    content: Any,
    max_tokens: int | None = None,
    tokenizer: Any | None = None,
) -> str:
    """Extract text from tool result content.

    Args:
        content: Raw tool result content (str, list, or dict).
        max_tokens: Maximum token count for the result. If exceeded, content is truncated.
        tokenizer: Tokenizer instance for token counting and truncation.

    Returns:
        Extracted text, potentially truncated if max_tokens is set.
    """
    if isinstance(content, str):
        result_text = content
    elif isinstance(content, list):
        # List of content blocks
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        result_text = "\n".join(text_parts)
    elif isinstance(content, dict):
        if content.get("type") == "text":
            result_text = content.get("text", "")
        else:
            result_text = json.dumps(content)
    else:
        result_text = str(content)

    # Truncate by token count if configured
    if max_tokens and tokenizer and result_text:
        result_text = truncate_tool_result(result_text, max_tokens, tokenizer)
    elif max_tokens is not None:
        logger.debug(
            f"Tool result skip truncation: max_tokens={max_tokens}, "
            f"has_tokenizer={tokenizer is not None}, "
            f"result_len={len(result_text) if result_text else 0}"
        )

    return result_text


# =============================================================================
# Tool Conversion: Anthropic -> Internal
# =============================================================================


def convert_anthropic_tools_to_internal(
    tools: list[AnthropicTool] | None,
) -> list[dict[str, Any]] | None:
    """
    Convert Anthropic tools to internal/OpenAI format.

    Anthropic: {"name": "...", "description": "...", "input_schema": {...}}
    Internal:  {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

    Args:
        tools: List of Anthropic tool definitions

    Returns:
        List of internal tool definitions, or None if no tools
    """
    if not tools:
        return None

    internal_tools = []
    for tool in tools:
        # Handle both Pydantic models and dicts
        if hasattr(tool, "model_dump"):
            tool_dict = tool.model_dump()
        elif isinstance(tool, dict):
            tool_dict = tool
        else:
            continue

        internal_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_dict.get("name", ""),
                    "description": tool_dict.get("description", ""),
                    "parameters": tool_dict.get("input_schema", {}),
                },
            }
        )

    return internal_tools if internal_tools else None


# =============================================================================
# Response Conversion: Internal -> Anthropic
# =============================================================================


def convert_internal_to_anthropic_response(
    text: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str | None,
    tool_calls: list[ToolCall] | None = None,
    thinking: str | None = None,
    cached_tokens: int = 0,
    prefix_cache_enabled: bool = False,
) -> MessagesResponse:
    """
    Convert internal output to Anthropic MessagesResponse.

    When ``prefix_cache_enabled`` is True the prompt count is split into
    Anthropic's disjoint usage fields so that
    input_tokens + cache_creation_input_tokens + cache_read_input_tokens
    equals prompt_tokens. Otherwise the response keeps the legacy shape
    (input_tokens = prompt_tokens, cache fields = 0).

    Args:
        text: Generated text content
        model: Model name
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        finish_reason: Internal finish reason ("stop", "length", "tool_calls")
        tool_calls: List of internal ToolCall objects
        thinking: Reasoning/thinking content from <think> blocks
        cached_tokens: Prompt tokens served from the prefix cache
        prefix_cache_enabled: Whether the engine runs automatic prefix caching

    Returns:
        Anthropic MessagesResponse
    """
    content: list[ContentBlockText | ContentBlockToolUse | ContentBlockThinking] = []

    # Add thinking content block before text if present
    if thinking and thinking.strip():
        content.append(ContentBlockThinking(
            type="thinking",
            thinking=thinking,
            signature="",
        ))

    # Add text content block if present and not empty
    if text and text.strip():
        content.append(ContentBlockText(type="text", text=text))

    # Add tool_use blocks if present
    if tool_calls:
        for tc in tool_calls:
            try:
                # Parse arguments from JSON string
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, AttributeError):
                args = {}

            content.append(
                ContentBlockToolUse(
                    type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                    input=args,
                )
            )

    # Ensure at least one content block
    if not content:
        content.append(ContentBlockText(type="text", text=""))

    # Map finish reason to stop reason
    stop_reason = map_finish_reason_to_stop_reason(finish_reason, bool(tool_calls))

    # When prefix caching is on, split prompt_tokens into the Anthropic
    # disjoint triple (input + creation + read == prompt_tokens).
    if prefix_cache_enabled:
        cache_read = max(0, min(cached_tokens, prompt_tokens))
        cache_creation = prompt_tokens - cache_read
        input_display = 0
    else:
        cache_read = 0
        cache_creation = 0
        input_display = prompt_tokens

    return MessagesResponse(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        type="message",
        role="assistant",
        model=model,
        content=content,
        stop_reason=stop_reason,
        usage=AnthropicUsage(
            input_tokens=input_display,
            output_tokens=completion_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        ),
    )


def map_finish_reason_to_stop_reason(
    finish_reason: str | None, has_tool_calls: bool
) -> str | None:
    """
    Map internal finish_reason to Anthropic stop_reason.

    Internal: "stop", "length", "tool_calls"
    Anthropic: "end_turn", "max_tokens", "stop_sequence", "tool_use"

    Args:
        finish_reason: Internal finish reason
        has_tool_calls: Whether the response contains tool calls

    Returns:
        Anthropic stop_reason
    """
    if has_tool_calls:
        return "tool_use"

    if finish_reason is None:
        return None

    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }

    return mapping.get(finish_reason, "end_turn")


# =============================================================================
# SSE Event Formatting
# =============================================================================


def format_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """
    Format an SSE event for Anthropic streaming.

    Anthropic uses: "event: {type}\\ndata: {json}\\n\\n"
    (Different from OpenAI which just uses "data: {json}\\n\\n")

    Args:
        event_type: Event type (message_start, content_block_delta, etc.)
        data: Event data to serialize as JSON

    Returns:
        Formatted SSE event string
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def create_message_start_event(
    message_id: str, model: str, input_tokens: int = 0
) -> str:
    """Create message_start SSE event."""
    return format_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        },
    )


def create_content_block_start_event(index: int, block_type: str, **kwargs) -> str:
    """Create content_block_start SSE event."""
    if block_type == "text":
        content_block = {"type": "text", "text": ""}
    elif block_type == "tool_use":
        content_block = {
            "type": "tool_use",
            "id": kwargs.get("id", ""),
            "name": kwargs.get("name", ""),
            "input": {},
        }
    elif block_type == "thinking":
        content_block = {"type": "thinking", "thinking": ""}
    else:
        content_block = {"type": block_type}

    return format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        },
    )


def create_thinking_delta_event(index: int, thinking: str) -> str:
    """Create content_block_delta SSE event for thinking content."""
    return format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": thinking},
        },
    )


def create_text_delta_event(index: int, text: str) -> str:
    """Create content_block_delta SSE event for text."""
    return format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        },
    )


def create_input_json_delta_event(index: int, partial_json: str) -> str:
    """Create content_block_delta SSE event for tool input JSON."""
    return format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        },
    )


def create_content_block_stop_event(index: int) -> str:
    """Create content_block_stop SSE event."""
    return format_sse_event(
        "content_block_stop",
        {
            "type": "content_block_stop",
            "index": index,
        },
    )


def create_message_delta_event(
    stop_reason: str | None,
    output_tokens: int,
    stop_sequence: str | None = None,
    input_tokens: int | None = None,
    cached_tokens: int = 0,
    prefix_cache_enabled: bool = False,
) -> str:
    """Create message_delta SSE event.

    When ``prefix_cache_enabled`` is True and ``input_tokens`` is given, the
    count is split into Anthropic's disjoint triple (input stays 0, creation
    and read carry the remainder). Otherwise the legacy shape is preserved.
    """
    usage: dict[str, int] = {"output_tokens": output_tokens}

    if prefix_cache_enabled and input_tokens is not None:
        cache_read = max(0, min(cached_tokens, input_tokens))
        usage["input_tokens"] = 0
        usage["cache_creation_input_tokens"] = input_tokens - cache_read
        usage["cache_read_input_tokens"] = cache_read
    elif input_tokens is not None:
        usage["input_tokens"] = input_tokens

    return format_sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": usage,
        },
    )


def create_message_stop_event() -> str:
    """Create message_stop SSE event."""
    return format_sse_event("message_stop", {"type": "message_stop"})


def create_ping_event() -> str:
    """Create ping SSE event."""
    return format_sse_event("ping", {"type": "ping"})


def create_error_event(error_type: str, message: str) -> str:
    """Create error SSE event."""
    return format_sse_event(
        "error",
        {
            "type": "error",
            "error": {"type": error_type, "message": message},
        },
    )
