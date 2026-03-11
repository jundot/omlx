# SPDX-License-Identifier: Apache-2.0
"""Conversion utilities for the OpenAI Responses API."""

import json
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

from .responses_models import (
    InputItem,
    OutputContent,
    OutputItem,
    ResponseObject,
    ResponsesTool,
    ResponseUsage,
)
from .shared_models import IDPrefix, generate_id


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


# =============================================================================
# Input Conversion
# =============================================================================


def convert_responses_input_to_messages(
    input_data: Optional[Union[str, List[InputItem]]],
    instructions: Optional[str] = None,
    previous_messages: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Convert Responses API input to internal messages format.

    Args:
        input_data: String prompt or list of InputItem objects.
        instructions: System prompt (prepended as system message).
        previous_messages: Messages from previous_response_id chain.

    Returns:
        List of message dicts compatible with chat template.
    """
    messages: List[Dict[str, Any]] = []

    # Collect all system/developer content to merge into a single system message.
    # Many chat templates (Qwen, Llama, etc.) only allow one system message
    # at position 0. Codex can send both `instructions` and developer-role
    # input items, so we merge them.
    system_parts: List[str] = []
    if instructions:
        system_parts.append(instructions)

    # Prepend previous response context
    if previous_messages:
        messages.extend(previous_messages)

    if input_data is None:
        if system_parts:
            messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
        return messages

    if isinstance(input_data, str):
        if system_parts:
            messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
        messages.append({"role": "user", "content": input_data})
        return messages

    # Process input items
    # Track pending tool calls for grouping into a single assistant message
    pending_tool_calls: List[Dict[str, Any]] = []

    for item in input_data:
        # Resolve effective type: EasyInputMessage has no type field
        item_type = item.type
        if item_type is None and item.role is not None:
            item_type = "message"

        if item_type == "message":
            # Flush pending tool calls before a new message
            if pending_tool_calls:
                messages.append({
                    "role": "assistant",
                    "tool_calls": pending_tool_calls,
                })
                pending_tool_calls = []

            role = item.role or "user"
            # Map "developer" role to "system"
            if role == "developer":
                role = "system"

            content = item.content
            if isinstance(content, list):
                # Convert content parts to text
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "input_text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "input_image":
                            # Pass through image content for VLM
                            text_parts.append("[image]")
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = "\n".join(text_parts) if text_parts else ""

            # Merge system/developer messages into the single system block
            if role == "system":
                system_parts.append(content or "")
            else:
                messages.append({"role": role, "content": content or ""})

        elif item.type == "function_call":
            # Assistant's tool call — accumulate for grouping
            call_id = item.call_id or item.id or f"call_{uuid.uuid4().hex[:8]}"
            pending_tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": item.name or "",
                    "arguments": _try_parse_json(item.arguments or "{}"),
                },
            })

        elif item.type == "function_call_output":
            # Flush pending tool calls first
            if pending_tool_calls:
                messages.append({
                    "role": "assistant",
                    "tool_calls": pending_tool_calls,
                })
                pending_tool_calls = []

            messages.append({
                "role": "tool",
                "tool_call_id": item.call_id or "",
                "content": item.output or "",
            })

    # Flush remaining pending tool calls
    if pending_tool_calls:
        messages.append({
            "role": "assistant",
            "tool_calls": pending_tool_calls,
        })

    # Insert merged system message at position 0
    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

    return messages


# =============================================================================
# Tool Conversion
# =============================================================================


def convert_responses_tools(
    tools: Optional[List[ResponsesTool]],
) -> Optional[List[Dict[str, Any]]]:
    """Convert Responses API flat tool format to Chat Completions nested format.

    Responses: {"type": "function", "name": "fn", "parameters": {...}}
    Chat Completions: {"type": "function", "function": {"name": "fn", "parameters": {...}}}

    Non-function tool types (local_shell, mcp, web_search, etc.) are skipped
    since they are not supported by local model chat templates.
    """
    if not tools:
        return None

    result = []
    for tool in tools:
        if tool.type == "function" and tool.name:
            func_def: Dict[str, Any] = {"name": tool.name}
            if tool.description:
                func_def["description"] = tool.description
            if tool.parameters:
                func_def["parameters"] = tool.parameters
            if tool.strict is not None:
                func_def["strict"] = tool.strict
            result.append({"type": "function", "function": func_def})
        # Non-function tools (local_shell, mcp, web_search, etc.) are
        # silently skipped — local models can't execute them.
    return result if result else None


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
) -> OutputItem:
    """Build a function_call-type OutputItem."""
    return OutputItem(
        type="function_call",
        id=item_id or generate_id(IDPrefix.FUNCTION_CALL),
        status=status,
        call_id=call_id,
        name=name,
        arguments=arguments,
    )


def build_response_usage(
    input_tokens: int, output_tokens: int
) -> ResponseUsage:
    """Build ResponseUsage from token counts."""
    return ResponseUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
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
    """Bounded in-memory store for responses, supporting previous_response_id."""

    def __init__(self, max_size: int = MAX_STORED_RESPONSES):
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_size = max_size

    def put(self, response_id: str, response_data: Dict[str, Any]) -> None:
        """Store a response, evicting oldest if at capacity."""
        if response_id in self._store:
            self._store.move_to_end(response_id)
            self._store[response_id] = response_data
            return
        if len(self._store) >= self._max_size:
            self._store.popitem(last=False)  # Remove oldest
        self._store[response_id] = response_data

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response."""
        data = self._store.get(response_id)
        if data is not None:
            self._store.move_to_end(response_id)
        return data

    def delete(self, response_id: str) -> bool:
        """Delete a stored response. Returns True if found."""
        if response_id in self._store:
            del self._store[response_id]
            return True
        return False

    def __len__(self) -> int:
        return len(self._store)


# =============================================================================
# Previous Response Conversion
# =============================================================================


def convert_stored_response_to_messages(
    response_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a stored ResponseObject dict back to messages for context chaining.

    This is used when previous_response_id is set: the previous response's
    output items are converted to assistant/tool messages.
    """
    messages: List[Dict[str, Any]] = []

    # If the stored response had input messages, we'd ideally re-include them.
    # But the Responses API expects the client to only send new input,
    # so we only convert the output items.
    output_items = response_data.get("output", [])
    for item in output_items:
        item_type = item.get("type")

        if item_type == "message":
            # Extract text from content blocks
            content_blocks = item.get("content", [])
            text_parts = []
            for block in content_blocks:
                if block.get("type") == "output_text":
                    text_parts.append(block.get("text", ""))
            text = "\n".join(text_parts)
            messages.append({
                "role": item.get("role", "assistant"),
                "content": text,
            })

        elif item_type == "function_call":
            call_id = item.get("call_id", f"call_{uuid.uuid4().hex[:8]}")
            messages.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": _try_parse_json(item.get("arguments", "{}")),
                    },
                }],
            })

    return messages
