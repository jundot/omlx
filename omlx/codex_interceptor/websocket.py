from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Iterable
from typing import Any

_MAX_RETAINED_RESPONSES = 64
_MAX_RETAINED_BYTES = 32 * 1024 * 1024


class SSEEventDecoder:
    """Incrementally decode JSON payloads from an SSE byte stream."""

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buffer += chunk.replace(b"\r\n", b"\n")
        events: list[bytes] = []
        while (idx := self._buffer.find(b"\n\n")) >= 0:
            block, self._buffer = self._buffer[:idx], self._buffer[idx + 2 :]
            data = b"\n".join(
                line[5:].lstrip()
                for line in block.splitlines()
                if line.startswith(b"data:")
            )
            if data and data != b"[DONE]":
                events.append(data)
        return events

    def finish(self) -> list[bytes]:
        return self.feed(b"\n\n")


class ResponsesWebSocketState:
    """Expand Codex's incremental WebSocket requests for stateless local HTTP."""

    def __init__(self) -> None:
        self._responses: dict[str, dict[str, Any]] = {}
        self._retained_bytes = 0

    @staticmethod
    def is_response_create(payload: Any) -> bool:
        return isinstance(payload, dict) and payload.get("type") == "response.create"

    @staticmethod
    def is_prewarm(payload: Any) -> bool:
        return (
            ResponsesWebSocketState.is_response_create(payload)
            and payload.get("generate") is False
        )

    def acknowledge_prewarm(
        self, payload: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        response_id = "resp_local_warm_" + uuid.uuid4().hex
        request = self._http_request(payload)
        # `request` is a fresh deep copy owned by this entry, so the context
        # can share its input items instead of copying them a second time.
        input_items = request.get("input")
        context = list(input_items) if isinstance(input_items, list) else []
        self._store(response_id, request, context)
        usage = {
            "input_tokens": 0,
            "input_tokens_details": None,
            "output_tokens": 0,
            "output_tokens_details": None,
            "total_tokens": 0,
        }
        return response_id, [
            {"type": "response.created", "response": {"id": response_id}},
            {
                "type": "response.completed",
                "response": {"id": response_id, "usage": usage},
            },
        ]

    def expand(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._http_request(payload)
        previous_id = payload.get("previous_response_id")
        previous = (
            self._responses.get(previous_id) if isinstance(previous_id, str) else None
        )
        if previous is None:
            return request

        # The retained request's "input" duplicates its "context", so skip it
        # here and deep-copy the context once below. Values from `request`
        # are fresh deep copies owned by this call and can move across as-is.
        expanded = {
            key: copy.deepcopy(value)
            for key, value in previous["request"].items()
            if key != "input"
        }
        for key, value in request.items():
            if key != "input":
                expanded[key] = value
        prior_context = previous.get("context")
        incremental = request.get("input")
        if isinstance(prior_context, list) and isinstance(incremental, list):
            # Retained items are deep-copied so later caller mutations of the
            # expanded request can never corrupt the stored conversation.
            expanded["input"] = copy.deepcopy(prior_context) + incremental
        else:
            expanded["input"] = incremental
        return expanded

    def remember(
        self, request: dict[str, Any], events: Iterable[dict[str, Any]]
    ) -> str | None:
        response_id: str | None = None
        completed_output: list[Any] | None = None
        done_output: list[Any] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("type") == "response.output_item.done" and "item" in event:
                done_output.append(copy.deepcopy(event["item"]))
            if event.get("type") == "response.completed":
                response = event.get("response")
                candidate = response.get("id") if isinstance(response, dict) else None
                if isinstance(candidate, str) and candidate:
                    response_id = candidate
                output = response.get("output") if isinstance(response, dict) else None
                if isinstance(output, list):
                    completed_output = copy.deepcopy(output)
        if response_id is None:
            return None
        # Some OpenAI-compatible servers emit only response.completed and omit
        # response.output_item.done entirely. Treat completed.response.output as
        # authoritative, supplementing it with any richer done items that are not
        # already present. Otherwise the next function_call_output is reconstructed
        # without its function_call and local chat templates reject the history.
        output_items = _merge_response_output(done_output, completed_output or [])
        stored_request = copy.deepcopy(request)
        input_items = stored_request.get("input")
        # The stored request and its context share the same input items: the
        # single deep copy above serves both, and expand() copies retained
        # state again before handing it back to callers.
        context = list(input_items) if isinstance(input_items, list) else []
        context.extend(output_items)
        self._store(
            response_id,
            stored_request,
            context,
            extra_size=_estimate_size(output_items),
        )
        return response_id

    def _store(
        self,
        response_id: str,
        request: dict[str, Any],
        context: list[Any],
        *,
        extra_size: int = 0,
    ) -> None:
        previous = self._responses.pop(response_id, None)
        if previous is not None:
            self._retained_bytes -= previous.get("size", 0)
        size = _estimate_size(request) + extra_size
        self._responses[response_id] = {
            "request": request,
            "context": context,
            "size": size,
        }
        self._retained_bytes += size
        # Cap retention by count and by estimated serialized bytes. The newest
        # entry is never evicted, even if it alone exceeds the byte ceiling.
        while len(self._responses) > _MAX_RETAINED_RESPONSES or (
            self._retained_bytes > _MAX_RETAINED_BYTES and len(self._responses) > 1
        ):
            oldest = next(iter(self._responses))
            entry = self._responses.pop(oldest, None)
            if entry is not None:
                self._retained_bytes -= entry.get("size", 0)

    @staticmethod
    def _http_request(payload: dict[str, Any]) -> dict[str, Any]:
        request = copy.deepcopy(payload)
        for key in ("type", "generate", "client_metadata", "previous_response_id"):
            request.pop(key, None)
        request["stream"] = True
        return request


def _estimate_size(value: Any) -> int:
    try:
        return len(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    except (TypeError, ValueError):
        return 0


def _merge_response_output(done: list[Any], completed: list[Any]) -> list[Any]:
    merged: list[Any] = []
    positions: dict[str, int] = {}

    def key(item: Any) -> str:
        if isinstance(item, dict):
            item_type = item.get("type")
            call_id = item.get("call_id")
            if (
                item_type
                in {
                    "function_call",
                    "custom_tool_call",
                    "computer_call",
                    "tool_search_call",
                }
                and isinstance(call_id, str)
                and call_id
            ):
                return f"call:{item_type}:{call_id}"
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id:
                return "id:" + item_id
            if isinstance(call_id, str) and call_id:
                return f"call:{item_type}:{call_id}"
        try:
            return "json:" + json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError):
            return "repr:" + repr(item)

    # Completed output is the final ordered response. Done events can contain
    # fuller per-item fields, so replace matching completed items with those.
    for item in completed:
        item_key = key(item)
        positions[item_key] = len(merged)
        merged.append(copy.deepcopy(item))
    for item in done:
        item_key = key(item)
        position = positions.get(item_key)
        if position is None:
            positions[item_key] = len(merged)
            merged.append(copy.deepcopy(item))
        else:
            merged[position] = copy.deepcopy(item)
    return merged


def decode_json_events(payloads: Iterable[bytes]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for payload in payloads:
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(decoded, dict):
            events.append(decoded)
    return events
