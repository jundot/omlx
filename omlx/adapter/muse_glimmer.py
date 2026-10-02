# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer channel-protocol output parsing.

Muse Glimmer generates a sequence of channel-scoped messages::

    [HEADER]<|message|>BODY(<|eom|>|<|eot|>)<|start|>[HEADER]<|message|>...

Generation begins right after the ``<|start|>assistant`` generation prompt,
so the first emitted text is a header remnant (e.g. `` to=self``). The
header's recipient classifies the body:

- ``to=self`` — reasoning; surfaced on the stream inside oMLX's
  ``<think>``/``</think>`` markers (minimax/inkling pattern: the API layer
  extracts ``reasoning_content`` from the block).
- ``to=user`` or no recipient — the visible answer.
- any other recipient — a tool call carrying an ATEM ``<atem:invoke>``
  block; suppressed while streaming and parsed at finalize. Per the
  model's own tool-definition prompt, the payload "is not expected to be
  valid XML and is parsed with regular expressions".

``<|eom|>`` ends a message with the turn continuing; ``<|eot|>`` and
``<|end_of_text|>`` end the turn (both are eos in generation_config).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..utils.tokenizer import create_streaming_detokenizer
from .output_parser import (
    OutputParserFinalizeResult,
    OutputParserTokenResult,
)

logger = logging.getLogger(__name__)

_MUSE_START = "<|start|>"
_MUSE_MESSAGE = "<|message|>"
_MUSE_EOM = "<|eom|>"
_MUSE_EOT = "<|eot|>"
_MUSE_END_OF_TEXT = "<|end_of_text|>"

_MUSE_MARKERS = (
    _MUSE_START,
    _MUSE_MESSAGE,
    _MUSE_EOM,
    _MUSE_EOT,
    _MUSE_END_OF_TEXT,
)

_RECIPIENT_RE = re.compile(r"\bto=([^\s<]+)")

# Headers are short ("assistant to=browser.search"). A head buffer growing
# past this without <|message|> means the model went off-protocol (e.g. a
# heavily quantized checkpoint emitting plain text); flush it as visible
# text so streaming keeps working instead of buffering to finalize.
_HEAD_FLUSH_LIMIT = 96

_INVOKE_RE = re.compile(
    r'<atem:invoke\s+name="([^"]+)"\s*>(.*?)(?:</atem:invoke>|\Z)',
    re.S,
)
_PARAM_RE = re.compile(
    r'<atem:parameter\s+name="([^"]+)"\s*>(.*?)(?:</atem:parameter>|\Z)',
    re.S,
)


class _MuseChannelSplitter:
    """Streaming splitter for the ``<|start|>``/``<|message|>`` protocol."""

    def __init__(self) -> None:
        self._buffer = ""
        self._channel: str | None = None  # None = reading a header
        self._head = ""
        self._think_open = False
        self.stopped = False

    def _partial_suffix_len(self, text: str) -> int:
        max_len = min(len(text), max(len(m) for m in _MUSE_MARKERS) - 1)
        for size in range(max_len, 0, -1):
            suffix = text[-size:]
            if any(m.startswith(suffix) for m in _MUSE_MARKERS):
                return size
        return 0

    def _emit_body(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        if self._channel in ("text", "thinking"):
            # Thinking flows to BOTH channels wrapped in <think> markers
            # (minimax pattern): the scheduler accumulates only
            # visible_text into request.output_text, and the API layer
            # extracts reasoning_content from the <think> block there.
            return text, text
        if self._channel == "tool":
            return "", ""
        # Header: buffer until <|message|> classifies it — but bail out to
        # the text channel if it grows implausibly long (off-protocol).
        self._head += text
        if len(self._head) > _HEAD_FLUSH_LIMIT:
            head, self._head = self._head, ""
            self._channel = "text"
            return head, head
        return "", ""

    def _close_channel(self) -> tuple[str, str]:
        stream = visible = ""
        if self._think_open:
            stream += "</think>"
            visible += "</think>"
            self._think_open = False
        self._channel = None
        self._head = ""
        return stream, visible

    def _handle_marker(self, marker: str) -> tuple[str, str]:
        stream = visible = ""
        if marker == _MUSE_MESSAGE:
            head, self._head = self._head, ""
            match = _RECIPIENT_RE.search(head)
            recipient = match.group(1) if match else None
            if recipient == "self":
                self._channel = "thinking"
                if not self._think_open:
                    stream += "<think>"
                    visible += "<think>"
                    self._think_open = True
            elif recipient is None or recipient == "user":
                if self._think_open:
                    # A text message after an unterminated thinking message
                    # still closes the visible thinking span.
                    stream += "</think>"
                    visible += "</think>"
                    self._think_open = False
                self._channel = "text"
            else:
                if self._think_open:
                    stream += "</think>"
                    visible += "</think>"
                    self._think_open = False
                self._channel = "tool"
        elif marker == _MUSE_EOM:
            s, v = self._close_channel()
            stream += s
            visible += v
        elif marker in (_MUSE_EOT, _MUSE_END_OF_TEXT):
            s, v = self._close_channel()
            stream += s
            visible += v
            self.stopped = True
        elif marker == _MUSE_START:
            # Start of the next message header. Reaching this from inside a
            # body (no <|eom|>) is off-protocol; close the channel anyway.
            s, v = self._close_channel()
            stream += s
            visible += v
        return stream, visible

    def feed(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        self._buffer += text
        stream = visible = ""
        while True:
            first_idx = -1
            first_marker = None
            for marker in _MUSE_MARKERS:
                idx = self._buffer.find(marker)
                if idx >= 0 and (first_idx < 0 or idx < first_idx):
                    first_idx = idx
                    first_marker = marker
            if first_marker is None:
                break
            s, v = self._emit_body(self._buffer[:first_idx])
            stream += s
            visible += v
            m_s, m_v = self._handle_marker(first_marker)
            stream += m_s
            visible += m_v
            self._buffer = self._buffer[first_idx + len(first_marker) :]

        keep = self._partial_suffix_len(self._buffer)
        ready = self._buffer[: len(self._buffer) - keep]
        self._buffer = self._buffer[len(self._buffer) - keep :]
        s, v = self._emit_body(ready)
        return stream + s, visible + v

    def finish(self) -> tuple[str, str]:
        stream = visible = ""
        s, v = self._emit_body(self._buffer)
        stream += s
        visible += v
        self._buffer = ""
        # An unclassified header at end-of-generation is flushed as text so
        # nothing the model produced is ever dropped.
        head, self._head = self._head, ""
        if head:
            stream += head
            visible += head
        if self._think_open:
            stream += "</think>"
            visible += "</think>"
            self._think_open = False
        return stream, visible


def _tool_param_schemas(tools: list[dict] | None) -> dict[str, dict[str, Any]]:
    """Map function name -> {param name -> JSON schema} from an OpenAI tools list."""
    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        parameters = fn.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties")
        if isinstance(properties, dict):
            schemas[name] = properties
    return schemas


def _coerce_param_value(raw: str, schema: dict[str, Any] | None) -> Any:
    """Reverse the chat template's parameter rendering.

    The template emits strings and numbers as-is (spaces not stripped),
    booleans as ``true``/``false``, ``None`` as ``null``, and lists/objects
    via ``tojson``. With a schema, string-typed parameters stay strings even
    when they look numeric; without one, JSON-decodable values are decoded.
    """
    schema_type = schema.get("type") if isinstance(schema, dict) else None
    if schema_type == "string":
        return raw
    if schema_type == "boolean":
        if raw.strip() == "true":
            return True
        if raw.strip() == "false":
            return False
    if schema_type in ("integer", "number"):
        try:
            return int(raw.strip()) if schema_type == "integer" else float(raw.strip())
        except ValueError:
            pass
    stripped = raw.strip()
    if stripped == "null":
        return None
    if stripped in ("true", "false"):
        return stripped == "true"
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return raw


def _extract_tool_calls(
    raw_text: str, tools: list[dict] | None
) -> list[dict[str, str]]:
    """Extract ATEM tool calls from tool-channel message bodies.

    Only bodies whose header recipient is neither ``self`` nor ``user`` are
    scanned, so ATEM examples the model may quote inside reasoning are never
    parsed as calls. The first generated message has no leading
    ``<|start|>assistant`` (the generation prompt supplied it), so one is
    prepended before matching. The header regex allows ``\\s*`` rather than
    ``\\s+`` because the streaming detokenizer strips the leading space off
    the first segment: a turn that opens directly with a tool call (common
    right after a tool-error result) arrives as ``to=bash...``, and the
    prepend then yields ``assistantto=bash...``.
    """
    schemas = _tool_param_schemas(tools)
    tool_calls: list[dict[str, str]] = []

    search_text = _MUSE_START + "assistant" + raw_text
    message_re = re.compile(
        re.escape(_MUSE_START)
        + r"assistant\s*to=([^\s<]+)\s*"
        + re.escape(_MUSE_MESSAGE)
        + r"(.*?)(?:"
        + re.escape(_MUSE_EOM)
        + r"|"
        + re.escape(_MUSE_EOT)
        + r"|"
        + re.escape(_MUSE_END_OF_TEXT)
        + r"|"
        + re.escape(_MUSE_START)
        + r"|\Z)",
        re.S,
    )
    for match in message_re.finditer(search_text):
        recipient, body = match.group(1), match.group(2)
        if recipient in ("self", "user"):
            continue
        for invoke in _INVOKE_RE.finditer(body):
            name = invoke.group(1)
            arguments: dict[str, Any] = {}
            for param in _PARAM_RE.finditer(invoke.group(2)):
                arguments[param.group(1)] = _coerce_param_value(
                    param.group(2), schemas.get(name, {}).get(param.group(1))
                )
            tool_calls.append(
                {
                    "name": name,
                    "arguments": json.dumps(
                        arguments, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
    return tool_calls


class MuseGlimmerOutputParserSession:
    """Parser session for Muse Glimmer channel output (thinking / text / tool)."""

    def __init__(
        self,
        tokenizer: Any,
        model_path: str | None = None,
        tools: list[dict] | None = None,
    ):
        self._tokenizer = tokenizer
        self._tools = tools
        self._raw_text = ""
        self._splitter = _MuseChannelSplitter()
        self._detokenizer = create_streaming_detokenizer(tokenizer, model_path)
        if self._detokenizer is not None:
            self._detokenizer.reset()

    def _decode_token(self, token_id: int) -> str:
        if self._detokenizer is not None:
            self._detokenizer.add_token(token_id)
            return self._detokenizer.last_segment
        try:
            return self._tokenizer.decode([token_id], skip_special_tokens=False)
        except TypeError:
            return self._tokenizer.decode([token_id])

    def process_token(self, token_id: int) -> OutputParserTokenResult:
        if self._splitter.stopped:
            return OutputParserTokenResult(is_stop=True, record_token=False)
        decoded_text = self._decode_token(token_id)
        self._raw_text += decoded_text
        stream_text, visible_text = self._splitter.feed(decoded_text)
        is_stop = self._splitter.stopped
        return OutputParserTokenResult(
            stream_text=stream_text,
            visible_text=visible_text,
            is_stop=is_stop,
            record_token=not is_stop,
        )

    def finalize(self) -> OutputParserFinalizeResult:
        stream_text = ""
        visible_text = ""
        if self._detokenizer is not None and not self._splitter.stopped:
            self._detokenizer.finalize()
            final_text = self._detokenizer.last_segment
            if final_text:
                self._raw_text += final_text
                s, v = self._splitter.feed(final_text)
                stream_text += s
                visible_text += v
        s, v = self._splitter.finish()
        stream_text += s
        visible_text += v

        tool_calls = _extract_tool_calls(self._raw_text, self._tools)
        return OutputParserFinalizeResult(
            stream_text=stream_text,
            visible_text=visible_text,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else None,
        )
