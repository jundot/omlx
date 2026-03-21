# SPDX-License-Identifier: Apache-2.0
"""Conversion utilities for the OpenAI Responses API."""

import copy
import json
import logging
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .responses_models import (
    InputItem,
    OutputContent,
    OutputItem,
    ResponsesTool,
    ResponseUsage,
)
from .shared_models import IDPrefix, generate_id

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


def _flush_pending_tool_calls(
    messages: List[Dict[str, Any]],
    pending: List[Dict[str, Any]],
    min_merge_index: int = 0,
) -> None:
    """Flush accumulated tool calls into messages.

    If the last message is an assistant message without tool_calls, merge
    into it (avoids duplicate assistant turns that confuse chat templates).
    Otherwise create a new assistant message.
    """
    if not pending:
        return
    if (
        messages
        and len(messages) - 1 >= min_merge_index
        and messages[-1].get("role") == "assistant"
        and "tool_calls" not in messages[-1]
    ):
        messages[-1]["tool_calls"] = list(pending)
    else:
        messages.append({"role": "assistant", "tool_calls": list(pending)})
    pending.clear()


def _consolidate_system_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        messages.extend(copy.deepcopy(previous_messages))
    current_message_start = len(messages)

    if input_data is None:
        if system_parts:
            messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
        return _consolidate_system_messages(messages)

    if isinstance(input_data, str):
        if system_parts:
            messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
        messages.append({"role": "user", "content": input_data})
        return _consolidate_system_messages(messages)

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
            _flush_pending_tool_calls(
                messages, pending_tool_calls, min_merge_index=current_message_start
            )

            role = item.role or "user"
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
                        if part.get("type") in ("input_text", "text", "output_text"):
                            text = part.get("text", "")
                            text_parts.append(text)
                            converted_parts.append({"type": "text", "text": text})
                        elif part.get("type") == "input_image":
                            # Preserve image data for VLM engines
                            has_image = True
                            image_url = part.get("image_url", part.get("url", ""))
                            detail = part.get("detail", "auto")
                            converted_parts.append({
                                "type": "input_image",
                                "image_url": image_url,
                                "detail": detail,
                            })
                    elif isinstance(part, str):
                        text_parts.append(part)
                        converted_parts.append({"type": "text", "text": part})
                if has_image:
                    # Keep as content list so VLM can extract images
                    content = converted_parts
                else:
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
            _flush_pending_tool_calls(
                messages, pending_tool_calls, min_merge_index=current_message_start
            )

            messages.append({
                "role": "tool",
                "tool_call_id": item.call_id or "",
                "content": item.output or "",
            })

    # Flush remaining pending tool calls
    _flush_pending_tool_calls(
        messages, pending_tool_calls, min_merge_index=current_message_start
    )

    # Insert merged system message at position 0
    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

    return _consolidate_system_messages(messages)


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
            record.setdefault("created_at", record.get("public_response", {}).get("created_at", 0))
            record.setdefault("previous_response_id", record.get("public_response", {}).get("previous_response_id"))
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
                response_id = raw.get("response_id") or raw.get("public_response", {}).get("id")
                if not response_id:
                    raise ValueError("missing response_id")
                loaded.append(self._normalize_record(response_id, raw))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Skipping corrupt response state file %s: %s", path, exc)

        loaded.sort(key=lambda record: (record.get("created_at", 0), record["response_id"]))
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


def convert_stored_response_to_messages(
    response_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert a stored public response or state record back to messages."""
    if "output_messages" in response_data:
        return copy.deepcopy(response_data.get("output_messages", []))
    return normalize_response_output_to_messages(response_data.get("output", []))


def normalize_response_output_to_messages(
    output_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert response output items to assistant/tool-call history messages."""
    messages: List[Dict[str, Any]] = []
    pending_tool_calls: List[Dict[str, Any]] = []

    for item in output_items:
        item_type = item.get("type")
        if item_type == "message":
            _flush_pending_tool_calls(messages, pending_tool_calls)
            content_blocks = item.get("content", [])
            text_parts = []
            for block in content_blocks:
                if block.get("type") == "output_text":
                    text_parts.append(block.get("text", ""))
            messages.append({
                "role": item.get("role", "assistant"),
                "content": "\n".join(text_parts),
            })
        elif item_type == "function_call":
            call_id = item.get("call_id", f"call_{uuid.uuid4().hex[:8]}")
            pending_tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": _try_parse_json(item.get("arguments", "{}")),
                },
            })

    _flush_pending_tool_calls(messages, pending_tool_calls)
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
