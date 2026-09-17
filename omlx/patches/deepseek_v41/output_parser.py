# SPDX-License-Identifier: MIT
"""Request-local V4.1 DSML framing with opaque reasoning content.

Reasoning must not land in ``visible_text`` (that becomes ``request.output_text``
and, after ThinkingParser recovery, assistant ``content``). ``stream_text`` may
still carry ``<think>``…``</think>`` so the API ThinkingParser can populate
``reasoning_content``.

DSML inside an explicit think span is example text, not a tool call. A real
call is the envelope after ``</think>``. If the model jumps to DSML without
closing think, finalize promotes a complete valid envelope.
"""

from ...adapter.output_parser import (
    OutputParserFinalizeResult,
    OutputParserTokenResult,
    _decode_output_token,
    create_streaming_detokenizer,
)
from .tool_parser import (
    find_tool_call_end,
    find_tool_call_start,
    parse_tool_call,
    tool_call_start,
)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class DeepSeekV41OutputParserSession:
    def __init__(self, tokenizer, model_path=None):
        self._tokenizer = tokenizer
        self._detokenizer = create_streaming_detokenizer(tokenizer, model_path)
        if self._detokenizer is not None:
            self._detokenizer.reset()
        self._buffer = ""
        self._state = "content"
        self._stopped = False
        self._calls = []
        self._active_start_marker = tool_call_start
        self._think_open_emitted = False

    def notify_prefilled_thought(self):
        self._state = "reasoning"
        # Prompt already opened <think>; do not inject a second opener into
        # stream_text (prefilled replay tests expect the raw generated span).
        self._think_open_emitted = True

    def _hold_suffix(self, markers, final: bool) -> int:
        if final:
            return 0
        held = 0
        for marker in markers:
            for size in range(1, min(len(marker), len(self._buffer) + 1)):
                if self._buffer.endswith(marker[:size]):
                    held = max(held, size)
        return held

    def _emit_think_open_if_needed(self, stream: list[str]) -> None:
        if not self._think_open_emitted:
            stream.append(_THINK_OPEN)
            self._think_open_emitted = True

    def _close_think(self, stream: list[str]) -> None:
        if self._think_open_emitted:
            stream.append(_THINK_CLOSE)
            self._think_open_emitted = False

    def _try_promote_trailing_tool(self, stream: list[str]) -> bool:
        """If unterminated thought ends on a valid DSML envelope, treat it as a call."""
        start_hit = find_tool_call_start(self._buffer)
        if start_hit is None:
            return False
        pos, marker = start_hit
        end = find_tool_call_end(self._buffer, pos)
        if end is None:
            return False
        end_idx, cutoff = end
        body = self._buffer[pos + len(marker) : end_idx]
        try:
            calls = parse_tool_call(body)
        except (ValueError, AssertionError):
            return False
        prefix = self._buffer[:pos]
        if prefix:
            self._emit_think_open_if_needed(stream)
            stream.append(prefix)
        self._close_think(stream)
        self._calls = calls
        self._stopped = True
        self._buffer = ""
        self._state = "content"
        return True

    def _consume(self, final=False):
        stream: list[str] = []
        visible: list[str] = []
        while self._buffer:
            if self._state == "tool":
                end = find_tool_call_end(self._buffer)
                if end is None:
                    if final:
                        # Incomplete envelope: keep it on the stream (tests +
                        # round-trip) but never in visible_text (Telegram).
                        if self._buffer:
                            stream.append(self._buffer)
                            self._buffer = ""
                        self._state = "content"
                    break
                end_idx, cutoff = end
                start_len = len(self._active_start_marker)
                block_body = self._buffer[start_len:end_idx]
                raw = self._buffer[:cutoff]
                try:
                    calls = parse_tool_call(block_body)
                except (ValueError, AssertionError):
                    stream.append(raw)
                    self._buffer = self._buffer[cutoff:]
                    self._state = "content"
                    continue
                self._calls = calls
                self._stopped = True
                self._buffer = ""
                break

            if self._state == "reasoning":
                close_idx = self._buffer.find(_THINK_CLOSE)
                if close_idx >= 0:
                    body = self._buffer[:close_idx]
                    self._emit_think_open_if_needed(stream)
                    if body:
                        stream.append(body)
                    self._close_think(stream)
                    self._buffer = self._buffer[close_idx + len(_THINK_CLOSE) :]
                    self._state = "content"
                    continue
                if final and self._try_promote_trailing_tool(stream):
                    break
                hold_markers = (_THINK_CLOSE,)
                held = self._hold_suffix(hold_markers, final)
                count = len(self._buffer) - held
                if count:
                    self._emit_think_open_if_needed(stream)
                    stream.append(self._buffer[:count])
                    self._buffer = self._buffer[count:]
                if final:
                    if self._buffer:
                        self._emit_think_open_if_needed(stream)
                        stream.append(self._buffer)
                        self._buffer = ""
                    self._close_think(stream)
                    self._state = "content"
                break

            think_idx = self._buffer.find(_THINK_OPEN)
            start_hit = find_tool_call_start(self._buffer)
            candidates = []
            if think_idx >= 0:
                candidates.append((think_idx, "think"))
            if start_hit is not None:
                candidates.append((start_hit[0], "tool"))
            if candidates:
                pos, kind = min(candidates, key=lambda item: item[0])
                preceding = self._buffer[:pos]
                if preceding:
                    stream.append(preceding)
                    visible.append(preceding)
                if kind == "think":
                    self._buffer = self._buffer[pos + len(_THINK_OPEN) :]
                    self._state = "reasoning"
                    self._think_open_emitted = False
                    continue
                self._active_start_marker = start_hit[1]
                self._buffer = self._buffer[pos:]
                self._state = "tool"
                continue

            hold_markers = (
                _THINK_OPEN,
                tool_call_start,
                "<｜DSML｜calls>",
                "<｜DSML｜tool_calls>",
                "<｜DSML｜ tool_calls>",
                "<｜DSML｜",
            )
            held = self._hold_suffix(hold_markers, final)
            count = len(self._buffer) - held
            if count:
                chunk = self._buffer[:count]
                stream.append(chunk)
                visible.append(chunk)
                self._buffer = self._buffer[count:]
            break

        if final and self._state == "reasoning" and not self._buffer:
            self._close_think(stream)
            self._state = "content"

        return "".join(stream), "".join(visible)

    def process_token(self, token_id):
        if self._stopped:
            return OutputParserTokenResult(is_stop=True, record_token=False)
        self._buffer += _decode_output_token(
            self._tokenizer, self._detokenizer, token_id
        )
        stream_text, visible_text = self._consume()
        return OutputParserTokenResult(
            stream_text=stream_text,
            visible_text=visible_text,
            is_stop=self._stopped,
            record_token=True,
        )

    def finalize(self):
        if self._detokenizer is not None and not self._stopped:
            self._detokenizer.finalize()
            self._buffer += self._detokenizer.last_segment
        stream_text, visible_text = self._consume(final=True)
        return OutputParserFinalizeResult(
            stream_text=stream_text,
            visible_text=visible_text,
            tool_calls=self._calls,
            finish_reason="tool_calls" if self._calls else None,
        )
