# SPDX-License-Identifier: Apache-2.0
"""LLM-jp-4 Harmony output parsing.

LLM-jp-4 models (``llm-jp/llm-jp-4-*`` and fine-tunes such as
``elyza/ELYZA-Thinking-1.0-llm-jp-4-32b-a3b``) render conversations in the
OpenAI Harmony format with their own tokenizer. The gpt-oss parser in
:mod:`omlx.adapter.harmony` matches token ids of openai-harmony's
``o200k_harmony`` encoding (``<|channel|>`` = 200005), while the LLM-jp-4
vocabulary puts the same markers at ids 2-13, so that parser never sees a
message boundary. This module parses the decoded text instead::

    <|channel|>analysis<|message|>REASONING<|end|>
    <|start|>assistant<|channel|>final<|message|>ANSWER<|return|>

    <|channel|>analysis<|message|>REASONING<|end|>
    <|start|>assistant to=functions.NAME<|channel|>commentary json
    <|message|>{"arg": 1}<|call|>

Generation starts after the ``<|start|>assistant`` generation prompt, so the
first header has no ``<|start|>``. ``analysis`` bodies stream inside oMLX's
``<think>`` markers, ``final`` bodies stream as content, and
``functions.*`` messages are hidden while streaming and returned as
``tool_calls`` at finalize.

``<|call|>`` and ``<|return|>`` are EOS tokens in the model's
generation_config, and the scheduler does not pass EOS tokens to parser
sessions, so a call usually arrives without its terminator. A trailing
``functions.*`` message whose body is a JSON object is therefore a call; one
closed by ``<|end|>`` is not, matching Harmony call semantics. The final
message ends the turn: the model sometimes closes its answer with ``<|end|>``
and goes on to write an imagined next turn, which is not parsed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .output_parser import OutputParserFinalizeResult, OutputParserTokenResult

logger = logging.getLogger(__name__)

_START = "<|start|>"
_CHANNEL = "<|channel|>"
_CONSTRAIN = "<|constrain|>"
_MESSAGE = "<|message|>"
_END = "<|end|>"
_RETURN = "<|return|>"
_CALL = "<|call|>"

LLMJP4_MARKERS = (_START, _CHANNEL, _CONSTRAIN, _MESSAGE, _END, _RETURN, _CALL)
LLMJP4_THINKING_END_TEXT = _END
LLMJP4_FINAL_HEADER_TEXT = _START + "assistant" + _CHANNEL + "final" + _MESSAGE

# Fixed ids of the Harmony markers in the LLM-jp-4 vocabulary. Detection
# requires all of them, so a tokenizer that only shares the marker strings
# keeps its own parser.
_LLMJP4_MARKER_IDS = {
    _RETURN: 2,
    _CONSTRAIN: 8,
    _CHANNEL: 9,
    _START: 10,
    _END: 11,
    _MESSAGE: 12,
    _CALL: 13,
}
LLMJP4_STOP_TOKEN_IDS = frozenset(
    (_LLMJP4_MARKER_IDS[_RETURN], _LLMJP4_MARKER_IDS[_CALL])
)

_RECIPIENT_RE = re.compile(r"\bto=functions\.([A-Za-z0-9_.\-]+)")
_CHANNEL_NAMES = ("analysis", "commentary", "final")
_OTHER_ROLES = ("user", "system", "developer")

# Headers are short ("assistant to=functions.x<|channel|>commentary json").
# A header that grows past this without <|message|> is off-protocol output;
# flush it as visible text instead of buffering until finalize.
_HEAD_FLUSH_LIMIT = 256


def is_llmjp4_tokenizer(tokenizer: Any) -> bool:
    """Return True when the tokenizer uses the LLM-jp-4 Harmony vocabulary."""
    try:
        return all(
            tokenizer.convert_tokens_to_ids(marker) == token_id
            for marker, token_id in _LLMJP4_MARKER_IDS.items()
        )
    except Exception:
        # Detection runs for every model; an unusual tokenizer must fall
        # through to the other parsers instead of failing the load.
        return False


def _classify_header(header: str) -> tuple[str, str | None]:
    """Return ``(kind, tool_name)`` for a Harmony message header.

    ``kind`` is ``"thinking"``, ``"final"``, ``"text"``, ``"tool"``, or
    ``"other_role"`` for a message the assistant does not author (an
    imagined next turn). Fine-tunes sometimes omit ``<|channel|>`` and write
    the channel as a bare word (``assistant analysis<|message|>``).
    """
    before, has_channel, after = header.partition(_CHANNEL)
    words = before.split()
    role = words[0] if words else ""
    if role in _OTHER_ROLES or role.startswith("functions."):
        return "other_role", None
    if has_channel:
        # "commentary<|constrain|>json" or "commentary json"
        channel_words = after.replace(_CONSTRAIN, " ").split()
        channel = channel_words[0] if channel_words else None
    else:
        channel = next((w for w in words if w in _CHANNEL_NAMES), None)

    recipient = _RECIPIENT_RE.search(header)
    if recipient and channel in (None, "commentary"):
        return "tool", recipient.group(1)
    if channel == "analysis":
        return "thinking", None
    if channel == "final":
        return "final", None
    return "text", None


def _looks_like_header(text: str) -> bool:
    """Return True for a header cut off before ``<|message|>``."""
    return (
        _CHANNEL in text
        or _CONSTRAIN in text
        or _RECIPIENT_RE.search(text) is not None
        or text.strip() in ("", "assistant")
    )


class _IncrementalDecoder:
    """Decode one token at a time with the tokenizer's own ``decode``.

    LLM-jp-4 decoding drops the leading space of a decoded string and of
    each segment after a Harmony marker. mlx-lm streams this tokenizer with
    ``NaiveStreamingDetokenizer``, which decodes every line on its own, so
    each streamed line lost one space of indentation. Keeping the previous
    chunk as left context (the prefix/read offsets used by vLLM and TGI)
    keeps the text identical to ``tokenizer.decode`` over the whole output,
    and holds back incomplete UTF-8 from byte-fallback tokens.
    """

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._tokens: list[int] = []
        self._prefix_offset = 0
        self._read_offset = 0

    def _decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        try:
            return self._tokenizer.decode(token_ids, skip_special_tokens=False)
        except TypeError:
            return self._tokenizer.decode(token_ids)

    def _delta(self, *, force: bool) -> str:
        prefix = self._decode(self._tokens[self._prefix_offset : self._read_offset])
        text = self._decode(self._tokens[self._prefix_offset :])
        if len(text) <= len(prefix) or (not force and text.endswith("\ufffd")):
            return ""
        self._prefix_offset = self._read_offset
        self._read_offset = len(self._tokens)
        return text[len(prefix) :]

    def add_token(self, token_id: int) -> str:
        self._tokens.append(token_id)
        return self._delta(force=False)

    def finalize(self) -> str:
        return self._delta(force=True)


class _Llmjp4ChannelSplitter:
    """Streaming splitter for LLM-jp-4 Harmony text."""

    def __init__(self) -> None:
        self._buffer = ""
        self._kind: str | None = None  # None = reading a header
        self._head = ""
        self._think_open = False
        self._tool_name: str | None = None
        self._tool_body = ""
        # (name, body, end marker or None) for each functions.* message.
        self.tool_messages: list[tuple[str, str, str | None]] = []
        self.stopped = False

    def _partial_suffix_len(self, text: str) -> int:
        max_len = min(len(text), max(len(m) for m in LLMJP4_MARKERS) - 1)
        for size in range(max_len, 0, -1):
            suffix = text[-size:]
            if any(marker.startswith(suffix) for marker in LLMJP4_MARKERS):
                return size
        return 0

    def _close_think(self) -> str:
        if not self._think_open:
            return ""
        self._think_open = False
        return "</think>"

    def _end_message(self, end: str | None) -> str:
        """Close the current message; return any visible closing text."""
        if self._kind == "tool" and self._tool_name:
            self.tool_messages.append((self._tool_name, self._tool_body, end))
        if self._kind == "final":
            # The final answer is the last message of a turn.
            self.stopped = True
        self._kind = None
        self._head = ""
        self._tool_name = None
        self._tool_body = ""
        return self._close_think()

    def _emit_body(self, text: str) -> str:
        if not text:
            return ""
        if self._kind == "tool":
            self._tool_body += text
            return ""
        if self._kind is not None:
            return text
        self._head += text
        if len(self._head) <= _HEAD_FLUSH_LIMIT:
            return ""
        head, self._head = self._head, ""
        self._kind = "text"
        return self._close_think() + head

    def _handle_marker(self, marker: str) -> str:
        if marker in (_CHANNEL, _CONSTRAIN):
            if self._kind is None:
                self._head += marker
            return ""
        if marker == _START:
            if self._kind is None:
                self._head = ""
                return ""
            # A body without an end marker is off-protocol; close it anyway.
            return self._end_message(None)
        if marker == _MESSAGE:
            if self._kind is not None:
                return ""
            head, self._head = self._head, ""
            kind, name = _classify_header(head)
            if kind == "other_role":
                self.stopped = True
                return self._close_think()
            self._kind = kind
            self._tool_name = name
            if kind != "thinking":
                return self._close_think()
            if self._think_open:
                return ""
            self._think_open = True
            return "<think>"

        # <|end|>, <|return|>, <|call|>
        text = self._end_message(marker)
        if marker in (_RETURN, _CALL):
            self.stopped = True
        return text

    def feed(self, text: str) -> str:
        if not text or self.stopped:
            return ""
        self._buffer += text
        out = ""
        while not self.stopped:
            first_idx = -1
            first_marker = None
            for marker in LLMJP4_MARKERS:
                idx = self._buffer.find(marker)
                if idx >= 0 and (first_idx < 0 or idx < first_idx):
                    first_idx = idx
                    first_marker = marker
            if first_marker is None:
                break
            out += self._emit_body(self._buffer[:first_idx])
            out += self._handle_marker(first_marker)
            self._buffer = self._buffer[first_idx + len(first_marker) :]

        if self.stopped:
            self._buffer = ""
            return out

        keep = self._partial_suffix_len(self._buffer)
        ready = self._buffer[: len(self._buffer) - keep]
        self._buffer = self._buffer[len(self._buffer) - keep :]
        return out + self._emit_body(ready)

    def finish(self) -> str:
        if self.stopped:
            return self._close_think()
        out = self._emit_body(self._buffer)
        self._buffer = ""
        if self._kind is not None:
            return out + self._end_message(None)
        # Flush an unclassified remainder unless it is a cut-off header, so
        # off-protocol text is never dropped.
        head, self._head = self._head, ""
        if head and not _looks_like_header(head):
            out += self._close_think() + head
        return out + self._close_think()


def _tool_calls_from_messages(
    messages: list[tuple[str, str, str | None]],
) -> list[dict[str, str]]:
    tool_calls: list[dict[str, str]] = []
    for name, body, end in messages:
        if end not in (_CALL, None):
            # Closed by <|end|> or <|return|>: not a Harmony call.
            continue
        arguments = body.strip() or "{}"
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if not isinstance(parsed, dict):
            if end == _CALL:
                logger.warning(
                    "Dropping LLM-jp-4 tool call %r: arguments are not a JSON "
                    "object: %r",
                    name,
                    arguments[:120],
                )
            # Without <|call|> this is usually a body cut off by max_tokens.
            continue
        tool_calls.append(
            {"name": name, "arguments": json.dumps(parsed, ensure_ascii=False)}
        )
    return tool_calls


class Llmjp4OutputParserSession:
    """Parser session for LLM-jp-4 Harmony output (thinking / text / tool)."""

    def __init__(self, tokenizer: Any):
        self._decoder = _IncrementalDecoder(tokenizer)
        self._splitter = _Llmjp4ChannelSplitter()

    def process_token(self, token_id: int) -> OutputParserTokenResult:
        if self._splitter.stopped:
            return OutputParserTokenResult(is_stop=True, record_token=False)
        text = self._splitter.feed(self._decoder.add_token(token_id))
        is_stop = self._splitter.stopped
        return OutputParserTokenResult(
            stream_text=text,
            visible_text=text,
            is_stop=is_stop,
            record_token=not is_stop,
        )

    def finalize(self) -> OutputParserFinalizeResult:
        text = ""
        if not self._splitter.stopped:
            text = self._splitter.feed(self._decoder.finalize())
        text += self._splitter.finish()

        tool_calls = _tool_calls_from_messages(self._splitter.tool_messages)
        return OutputParserFinalizeResult(
            stream_text=text,
            visible_text=text,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else None,
        )
