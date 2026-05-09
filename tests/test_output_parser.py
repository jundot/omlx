# SPDX-License-Identifier: Apache-2.0
"""Tests for protocol-specific output parser sessions."""

from __future__ import annotations

from openai_harmony import load_harmony_encoding

from omlx.adapter.gemma4 import Gemma4OutputParserSession
from omlx.adapter.output_parser import detect_output_parser


class FakeDetokenizer:
    def __init__(self, decode_one):
        self._decode_one = decode_one
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token_id: int):
        self.last_segment = self._decode_one(token_id)

    def finalize(self):
        self.last_segment = ""


class GemmaTokenizer:
    def __init__(self, token_map: dict[int, str]):
        self._token_map = token_map

    @property
    def detokenizer(self):
        return FakeDetokenizer(lambda token_id: self._token_map[token_id])

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "".join(self._token_map[token_id] for token_id in token_ids)


class HarmonyTokenizer:
    def __init__(self, encoding):
        self._encoding = encoding

    def convert_tokens_to_ids(self, token: str) -> int:
        ids = self._encoding.encode(token, allowed_special="all")
        return ids[0] if ids else -1

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return self._encoding.decode(token_ids)

    @property
    def detokenizer(self):
        return FakeDetokenizer(lambda token_id: self._encoding.decode([token_id]))


class TestGemma4OutputParserSession:
    def test_normal_reasoning_block(self):
        token_map = {
            1: "<|channel>",
            2: "thought\n",
            3: "reasoning",
            4: "<channel|>",
            5: "final answer",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = Gemma4OutputParserSession(tokenizer)

        stream = []
        visible = []
        for token_id in [1, 2, 3, 4, 5]:
            result = session.process_token(token_id)
            stream.append(result.stream_text)
            visible.append(result.visible_text)
        final = session.finalize()
        stream.append(final.stream_text)
        visible.append(final.visible_text)

        full_stream = "".join(stream)
        full_visible = "".join(visible)

        assert full_stream == "<think>\nreasoning</think>\nfinal answer"
        assert full_visible == full_stream
        assert "<|channel>" not in full_stream
        assert "<channel|>" not in full_stream

    def test_empty_thought_block(self):
        token_map = {
            1: "<|channel>thought\n",
            2: "<channel|>",
            3: "answer",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = Gemma4OutputParserSession(tokenizer)

        parts = []
        for token_id in [1, 2, 3]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        assert "".join(parts) == "<think>\n</think>\nanswer"

    def test_partial_marker_across_tokens(self):
        token_map = {
            1: "<|chan",
            2: "nel>thought\nstep 1",
            3: " and step 2<chan",
            4: "nel|>",
            5: "done",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = Gemma4OutputParserSession(tokenizer)

        parts = []
        for token_id in [1, 2, 3, 4, 5]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        text = "".join(parts)
        assert text == "<think>\nstep 1 and step 2</think>\ndone"
        assert "<|channel>thought" not in text
        assert "<channel|>" not in text

    def test_suppresses_turn_end_marker(self):
        token_map = {
            1: "<|channel>thought\n",
            2: "reasoning",
            3: "<channel|>",
            4: "answer",
            5: "<turn|>",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = Gemma4OutputParserSession(tokenizer)

        parts = []
        for token_id in [1, 2, 3, 4, 5]:
            result = session.process_token(token_id)
            parts.append(result.stream_text)
            assert "<turn|>" not in result.stream_text
            assert "<turn|>" not in result.visible_text
        parts.append(session.finalize().stream_text)

        text = "".join(parts)
        assert text == "<think>\nreasoning</think>\nanswer"
        assert "<turn|>" not in text

    def test_stray_close_marker_outside_thought_dropped(self):
        """A bare ``<channel|>`` after the thought block already closed must
        not leak into visible content. Models occasionally emit one in long
        multi-turn contexts and the SDK rejects it as raw markup."""
        token_map = {
            1: "<|channel>thought\n",
            2: "reasoning",
            3: "<channel|>",
            4: "answer",
            5: "<channel|>",
            6: "more",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = Gemma4OutputParserSession(tokenizer)

        parts = []
        for token_id in [1, 2, 3, 4, 5, 6]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        text = "".join(parts)
        assert text == "<think>\nreasoning</think>\nanswermore"
        assert "<channel|>" not in text

    def test_stray_open_marker_inside_thought_dropped(self):
        """A nested ``<|channel>thought\\n`` while already inside a thought
        block must not re-emit ``<think>``. The block stays open until the
        first matching close marker."""
        token_map = {
            1: "<|channel>thought\n",
            2: "step 1",
            3: "<|channel>thought\n",
            4: "step 2",
            5: "<channel|>",
            6: "answer",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = Gemma4OutputParserSession(tokenizer)

        parts = []
        for token_id in [1, 2, 3, 4, 5, 6]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        text = "".join(parts)
        assert text == "<think>\nstep 1step 2</think>\nanswer"
        assert text.count("<think>\n") == 1
        assert text.count("</think>\n") == 1

    def test_tool_call_markers_pass_through(self):
        """Tool-call markup must reach the buffered output text untouched so
        ``parse_tool_calls`` can extract the call. ``ToolCallStreamFilter``
        downstream is responsible for removing it from stream deltas."""
        token_map = {
            1: "<|channel>thought\n",
            2: "calling",
            3: "<channel|>",
            4: "<|tool_call>",
            5: "call:bash{cmd:ls}",
            6: "<tool_call|>",
            7: "done",
        }
        tokenizer = GemmaTokenizer(token_map)
        session = Gemma4OutputParserSession(tokenizer)

        stream_parts = []
        visible_parts = []
        for token_id in [1, 2, 3, 4, 5, 6, 7]:
            result = session.process_token(token_id)
            stream_parts.append(result.stream_text)
            visible_parts.append(result.visible_text)
        final = session.finalize()
        stream_parts.append(final.stream_text)
        visible_parts.append(final.visible_text)

        stream_text = "".join(stream_parts)
        visible_text = "".join(visible_parts)
        assert stream_text == visible_text
        assert "<|tool_call>" in stream_text
        assert "<tool_call|>" in stream_text
        assert "call:bash{cmd:ls}" in stream_text


class TestOutputParserFactory:
    def test_detects_gemma4(self):
        tokenizer = GemmaTokenizer({1: "x"})
        factory = detect_output_parser(
            "google/gemma-4b",
            tokenizer,
            {"model_type": "gemma4"},
        )

        assert factory is not None
        assert factory.kind == "gemma4"

    def test_harmony_wrapper_regression(self):
        encoding = load_harmony_encoding("HarmonyGptOss")
        tokenizer = HarmonyTokenizer(encoding)
        factory = detect_output_parser(
            "gpt-oss-20b",
            tokenizer,
            {"model_type": "gpt_oss"},
        )

        assert factory is not None
        assert factory.kind == "harmony"

        session = factory.create_session(tokenizer)
        tokens = encoding.encode(
            "<|channel|>analysis<|message|>thinking<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Answer<|return|>",
            allowed_special="all",
        )

        stream = []
        visible = []
        saw_stop = False
        for token in tokens:
            result = session.process_token(token)
            stream.append(result.stream_text)
            visible.append(result.visible_text)
            saw_stop = saw_stop or result.is_stop
        final = session.finalize()
        stream.append(final.stream_text)
        visible.append(final.visible_text)

        assert saw_stop is True
        assert "<think>\n" in "".join(stream)
        assert "</think>\n" in "".join(stream)
        assert "".join(visible) == "Answer"

    def test_harmony_non_streaming_preserves_reasoning(self):
        """Non-streaming output_text retains analysis-channel reasoning."""
        from omlx.api.thinking import extract_thinking

        encoding = load_harmony_encoding("HarmonyGptOss")
        tokenizer = HarmonyTokenizer(encoding)
        factory = detect_output_parser(
            "gpt-oss-20b",
            tokenizer,
            {"model_type": "gpt_oss"},
        )
        session = factory.create_session(tokenizer)

        tokens = encoding.encode(
            "<|channel|>analysis<|message|>Let me think about this<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Four<|return|>",
            allowed_special="all",
        )

        visible_parts = []
        for token in tokens:
            result = session.process_token(token)
            visible_parts.append(result.visible_text)

        final = session.finalize()
        visible_parts.append(final.visible_text)

        # Mirror scheduler aggregation: prepend any parser-provided prefix
        # to the accumulated visible_text before exposing as output_text.
        prefix = getattr(final, "output_text_prefix", "")
        output_text = prefix + "".join(visible_parts)

        thinking, content = extract_thinking(output_text)
        assert thinking == "Let me think about this"
        assert content == "Four"
