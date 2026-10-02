# SPDX-License-Identifier: MIT
"""Request-local V4.1 DSML framing with opaque reasoning content."""

from ...adapter.output_parser import (
    OutputParserFinalizeResult,
    OutputParserTokenResult,
    _decode_output_token,
    create_streaming_detokenizer,
)
from .tool_parser import parse_tool_call, tool_call_end, tool_call_start


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

    def notify_prefilled_thought(self):
        self._state = "reasoning"

    def _consume(self, final=False):
        output = []
        while self._buffer:
            if self._state == "tool":
                end = self._buffer.find(tool_call_end)
                if end < 0:
                    if final:
                        output.append(self._buffer)
                        self._buffer = ""
                    break
                cutoff = end + len(tool_call_end)
                block = self._buffer[:cutoff]
                try:
                    calls = parse_tool_call(block[len(tool_call_start) : end])
                except (ValueError, AssertionError):
                    # An invalid envelope cannot terminate the assistant turn.
                    output.append(block)
                    self._buffer = self._buffer[cutoff:]
                    self._state = "content"
                    continue
                self._calls = calls
                self._stopped = True
                self._buffer = ""
                break
            markers = (
                ("</think>",)
                if self._state == "reasoning"
                else ("<think>", tool_call_start)
            )
            positions = [(self._buffer.find(m), m) for m in markers]
            positions = [(pos, marker) for pos, marker in positions if pos >= 0]
            if positions:
                pos, marker = min(positions)
                output.append(self._buffer[:pos])
                if marker == tool_call_start:
                    self._buffer = self._buffer[pos:]
                    self._state = "tool"
                else:
                    output.append(marker)
                    self._buffer = self._buffer[pos + len(marker) :]
                    self._state = "content" if marker == "</think>" else "reasoning"
                continue
            held = 0
            if not final:
                for marker in markers:
                    for size in range(1, min(len(marker), len(self._buffer) + 1)):
                        if self._buffer.endswith(marker[:size]):
                            held = max(held, size)
            count = len(self._buffer) - held
            output.append(self._buffer[:count])
            self._buffer = self._buffer[count:]
            break
        return "".join(output)

    def process_token(self, token_id):
        if self._stopped:
            return OutputParserTokenResult(is_stop=True, record_token=False)
        self._buffer += _decode_output_token(
            self._tokenizer, self._detokenizer, token_id
        )
        text = self._consume()
        return OutputParserTokenResult(
            stream_text=text,
            visible_text=text,
            is_stop=self._stopped,
            record_token=True,
        )

    def finalize(self):
        if self._detokenizer is not None and not self._stopped:
            self._detokenizer.finalize()
            self._buffer += self._detokenizer.last_segment
        text = self._consume(final=True)
        return OutputParserFinalizeResult(
            stream_text=text,
            visible_text=text,
            tool_calls=self._calls,
            finish_reason="tool_calls" if self._calls else None,
        )
