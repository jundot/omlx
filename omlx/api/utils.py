# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Utility functions for text processing.
"""

import json
import re
from typing import Any, List

from .openai_models import Message


# =============================================================================
# Special Token Patterns
# =============================================================================

# Pattern to match special tokens that should be removed from output
SPECIAL_TOKENS_PATTERN = re.compile(
    r'<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|'
    r'<\|end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|'
    r'</s>|<s>|<pad>|\[PAD\]|\[SEP\]|\[CLS\]'
)

# Pattern to strip <think>...</think> reasoning blocks from output
THINKING_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)
# Also handle case where <think> tag is missing but </think> is present
# (some models emit thinking content before </think> without explicit <think> tag)
THINKING_TAIL_PATTERN = re.compile(r'^.*?</think>\s*', re.DOTALL)


def clean_output_text(text: str) -> str:
    """
    Clean model output by removing special tokens and thinking blocks.

    Args:
        text: Raw model output

    Returns:
        Cleaned text with special tokens and <think> blocks removed
    """
    if not text:
        return text
    text = SPECIAL_TOKENS_PATTERN.sub('', text)
    # Remove <think>...</think> blocks (reasoning content)
    text = THINKING_PATTERN.sub('', text)
    # Handle partial thinking: content before </think> with no <think> tag
    if '</think>' in text and '<think>' not in text:
        text = THINKING_TAIL_PATTERN.sub('', text)
    return text.strip()


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
        if hasattr(item, 'model_dump'):
            item = item.model_dump()
        elif hasattr(item, 'dict'):
            item = item.dict()
        if isinstance(item, dict) and item.get("type") == "text":
            text_parts.append(item.get("text", ""))
    return "\n".join(text_parts) if text_parts else ""


def extract_text_content(
    messages: List[Message],
    max_tool_result_tokens: int | None = None,
    tokenizer: Any | None = None,
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

    Returns:
        List of {"role": str, "content": str}
    """
    processed_messages = []

    for msg in messages:
        role = msg.role
        content = msg.content

        # Normalize "developer" role to "system" (OpenAI API compatibility)
        if role == "developer":
            role = "system"

        # Handle tool response messages (role="tool")
        if role == "tool":
            tool_call_id = getattr(msg, 'tool_call_id', None) or ''
            tool_content = content if content else ""
            # Apply truncation if configured
            if max_tool_result_tokens and tokenizer and tool_content:
                from .anthropic_utils import truncate_tool_result
                tool_content = truncate_tool_result(
                    tool_content, max_tool_result_tokens, tokenizer
                )
            # Preserve structured format for models with native tool calling
            # so the chat template renders tool results in the model's native format
            if getattr(tokenizer, 'has_tool_calling', False):
                processed_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": tool_content,
                })
            else:
                processed_messages.append({
                    "role": "user",  # mlx-lm expects user/assistant roles
                    "content": f"[Tool Result ({tool_call_id})]: {tool_content}"
                })
            continue

        # Handle assistant messages with tool_calls
        if role == "assistant" and hasattr(msg, 'tool_calls') and msg.tool_calls:
            if isinstance(content, list):
                content = _extract_text_from_content_list(content)
            msg_dict = {"role": role, "content": content if content else ""}

            # Preserve structured tool_calls for models with native tool calling
            # so the chat template renders them in the model's native format.
            # Without this, models mimic text-formatted tool calls from history
            # instead of generating their native parseable format.
            if getattr(tokenizer, 'has_tool_calling', False):
                tool_calls_list = []
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        tool_calls_list.append({
                            "id": tc.get("id", ""),
                            "function": {
                                "name": func.get("name", ""),
                                "arguments": _try_parse_json(
                                    func.get("arguments", "{}")
                                ),
                            }
                        })
                    else:
                        args_str = (
                            getattr(tc.function, 'arguments', '{}')
                            if hasattr(tc, 'function') else '{}'
                        )
                        tool_calls_list.append({
                            "id": getattr(tc, 'id', ''),
                            "function": {
                                "name": (
                                    getattr(tc.function, 'name', '')
                                    if hasattr(tc, 'function') else ''
                                ),
                                "arguments": _try_parse_json(args_str),
                            }
                        })
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

            processed_messages.append(msg_dict)
            continue

        # Handle None content
        if content is None:
            processed_messages.append({"role": role, "content": ""})
            continue

        if isinstance(content, str):
            # Simple text message
            processed_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Content array - extract text parts only
            combined_text = _extract_text_from_content_list(content)
            processed_messages.append({"role": role, "content": combined_text})
        else:
            # Unknown format, try to convert
            processed_messages.append({"role": role, "content": str(content)})

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
    if not (s.startswith('{') or s.startswith('[')):
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
    messages: List[Message],
    max_tool_result_tokens: int | None = None,
    tokenizer: Any | None = None,
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

    Returns:
        List of message dicts with tool-related fields preserved
    """
    processed_messages = []

    for msg in messages:
        role = msg.role
        content = msg.content

        # Normalize "developer" role to "system" (OpenAI API compatibility)
        if role == "developer":
            role = "system"

        # Tool response messages - preserve role and tool_call_id
        # Parse content as JSON if possible (chat_template applies |tojson)
        if role == "tool":
            tool_content = content if content else ""
            if max_tool_result_tokens and tokenizer and tool_content:
                from .anthropic_utils import truncate_tool_result

                # Parse JSON BEFORE truncation for better line-boundary cuts.
                # Harmony chat_template applies |tojson to content, so content
                # must be a dict (not a string) to avoid double-encoding.
                parsed_json = _try_parse_json(tool_content)
                if isinstance(parsed_json, (dict, list)):
                    # Valid JSON - pretty-print for line-boundary truncation
                    pretty = json.dumps(
                        parsed_json, indent=2, ensure_ascii=False
                    )
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
            processed_messages.append({
                "role": "tool",
                "tool_call_id": getattr(msg, 'tool_call_id', '') or '',
                "content": parsed_content,
            })
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
                text_parts = []
                for item in content:
                    if hasattr(item, 'model_dump'):
                        item = item.model_dump()
                    elif hasattr(item, 'dict'):
                        item = item.dict()
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                msg_dict["content"] = "\n".join(text_parts)
            else:
                msg_dict["content"] = str(content)

            # Preserve tool_calls field for chat_template
            # Parse arguments as JSON if possible (chat_template applies |tojson)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_calls_list = []
                for tc in msg.tool_calls:
                    if isinstance(tc, dict):
                        args_str = tc.get("function", {}).get("arguments", "{}")
                        tool_calls_list.append({
                            "id": tc.get("id", ""),
                            "function": {
                                "name": tc.get("function", {}).get("name", ""),
                                "arguments": _try_parse_json(args_str),
                            }
                        })
                    else:
                        # Pydantic model
                        args_str = getattr(tc.function, 'arguments', '{}') if hasattr(tc, 'function') else '{}'
                        tool_calls_list.append({
                            "id": getattr(tc, 'id', ''),
                            "function": {
                                "name": getattr(tc.function, 'name', '') if hasattr(tc, 'function') else '',
                                "arguments": _try_parse_json(args_str),
                            }
                        })
                msg_dict["tool_calls"] = tool_calls_list

            processed_messages.append(msg_dict)
            continue

        # Other messages (user, system, developer)
        if content is None:
            processed_messages.append({"role": role, "content": ""})
        elif isinstance(content, str):
            processed_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Extract text parts from content array
            text_parts = []
            for item in content:
                if hasattr(item, 'model_dump'):
                    item = item.model_dump()
                elif hasattr(item, 'dict'):
                    item = item.dict()
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            processed_messages.append({"role": role, "content": "\n".join(text_parts)})
        else:
            processed_messages.append({"role": role, "content": str(content)})

    return processed_messages
