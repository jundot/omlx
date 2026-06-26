# SPDX-License-Identifier: Apache-2.0
"""Tests for protocol-specific output parser sessions."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

from omlx.adapter.gemma4 import Gemma4OutputParserSession
from omlx.adapter.harmony import load_harmony_gpt_oss_encoding
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


class CohereTokenizer:
    def __init__(self, token_map: dict[int, str]):
        self._token_map = token_map

    @property
    def detokenizer(self):
        return FakeDetokenizer(lambda token_id: self._token_map[token_id])

    def decode(self, token_ids, skip_special_tokens: bool = True):
        return "".join(self._token_map[token_id] for token_id in token_ids)


class _FakeMelodyOptions:
    def cmd4(self):
        return self

    def stream_tool_actions(self):
        return self


class _FakeMelodyFilter:
    def __init__(self, options):
        self.options = options

    def write_decoded(self, decoded_text: str):
        if decoded_text.startswith("R:"):
            return SimpleNamespace(
                content=None,
                reasoning=decoded_text[2:],
                tool_calls=[],
            )
        if decoded_text.startswith("C:"):
            return SimpleNamespace(
                content=decoded_text[2:],
                reasoning=None,
                tool_calls=[],
            )
        if decoded_text.startswith("T1"):
            tool_call = SimpleNamespace(
                index=0,
                id="call_",
                name="look",
                arguments='{"q"',
            )
            return SimpleNamespace(content=None, reasoning=None, tool_calls=[tool_call])
        if decoded_text.startswith("T2"):
            tool_call = SimpleNamespace(
                index=0,
                id="1",
                name="up",
                arguments=':"x"}',
            )
            return SimpleNamespace(content=None, reasoning=None, tool_calls=[tool_call])
        return SimpleNamespace(content=None, reasoning=None, tool_calls=[])

    def flush_partials(self):
        return SimpleNamespace(content=None, reasoning=None, tool_calls=[])


def _install_fake_melody(monkeypatch):
    module = types.ModuleType("cohere_melody")
    module.PyFilter = _FakeMelodyFilter
    module.PyFilterOptions = _FakeMelodyOptions
    monkeypatch.setitem(sys.modules, "cohere_melody", module)


def _write_json(path, data):
    path.write_text(json.dumps(data))


def _spm_decoder():
    return {
        "type": "Sequence",
        "decoders": [
            {
                "type": "Replace",
                "pattern": {"String": "\u2581"},
                "content": " ",
            },
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }


class ByteFallbackTokenizer:
    clean_up_tokenization_spaces = False
    vocab = {
        "<pad>": 0,
        "<0xEC>": 1,
        "<0x9E>": 2,
        "<0xA0>": 3,
    }

    def decode(self, token_ids, skip_special_tokens: bool = True):
        table = {
            0: b"",
            1: bytes([0xEC]),
            2: bytes([0x9E]),
            3: bytes([0xA0]),
        }
        raw = b"".join(table[token_id] for token_id in token_ids)
        if not raw:
            return ""
        if raw == bytes([0xEC, 0x9E, 0xA0]):
            return "\uc7a0"
        return "\ufffd" * sum(1 for token_id in token_ids if token_id != 0)


class TestCohere2MoeOutputParserSession:
    def test_detects_cohere2_moe_from_model_config(self, monkeypatch):
        _install_fake_melody(monkeypatch)
        tokenizer = CohereTokenizer({1: "C:hello"})

        factory = detect_output_parser(
            "North-Mini-Code",
            tokenizer,
            {"model_type": "cohere2_moe"},
        )

        assert factory is not None
        assert factory.kind == "cohere2_moe"

    def test_streams_reasoning_as_think_block_and_visible_content(self, monkeypatch):
        _install_fake_melody(monkeypatch)
        tokenizer = CohereTokenizer(
            {
                1: "R:reasoning",
                2: "C:answer",
            }
        )
        factory = detect_output_parser(
            "North-Mini-Code",
            tokenizer,
            {"model_type": "cohere2_moe"},
        )
        session = factory.create_session(tokenizer)

        parts = []
        visible = []
        for token_id in [1, 2]:
            result = session.process_token(token_id)
            parts.append(result.stream_text)
            visible.append(result.visible_text)
        final = session.finalize()
        parts.append(final.stream_text)
        visible.append(final.visible_text)

        assert "".join(parts) == "<think>\nreasoning</think>\nanswer"
        assert "".join(visible) == "<think>\nreasoning</think>\nanswer"
        assert final.tool_calls == []
        assert final.finish_reason is None

    def test_accumulates_streamed_tool_call_deltas(self, monkeypatch):
        _install_fake_melody(monkeypatch)
        tokenizer = CohereTokenizer({1: "T1", 2: "T2"})
        factory = detect_output_parser(
            "North-Mini-Code",
            tokenizer,
            {"model_type": "cohere2_moe"},
        )
        session = factory.create_session(tokenizer)

        assert session.process_token(1).stream_text == ""
        assert session.process_token(2).stream_text == ""
        final = session.finalize()

        assert final.tool_calls == [
            {
                "id": "call_1",
                "name": "lookup",
                "arguments": '{"q":"x"}',
            }
        ]
        assert final.finish_reason == "tool_calls"

    def test_literal_newline_in_arguments_is_reescaped(self, monkeypatch):
        """Melody may stream literal control chars when the model emits them inside
        JSON string values (e.g. newlines inside code arguments).  finalize() must
        re-serialize the accumulated arguments so they are valid JSON."""
        # Build a fake Melody that returns arguments containing a literal newline
        # (U+000A) inside the JSON string value, as the real model sometimes does.
        literal_newline_args = '{"path":"f.py","code":"line1\nline2"}'  # literal \n

        class _FakeMelodyFilterLiteralNewline:
            def __init__(self, options):
                pass

            def write_decoded(self, decoded_text: str):
                if decoded_text == "TC":
                    tc = SimpleNamespace(
                        index=0,
                        id="call_1",
                        name="edit",
                        arguments=literal_newline_args,
                    )
                    return SimpleNamespace(content=None, reasoning=None, tool_calls=[tc])
                return SimpleNamespace(content=None, reasoning=None, tool_calls=[])

            def flush_partials(self):
                return SimpleNamespace(content=None, reasoning=None, tool_calls=[])

        import types, json as _json
        module = types.ModuleType("cohere_melody")
        module.PyFilter = _FakeMelodyFilterLiteralNewline
        module.PyFilterOptions = _FakeMelodyOptions
        monkeypatch.setitem(__import__("sys").modules, "cohere_melody", module)

        tokenizer = CohereTokenizer({"TC": "TC"})
        from omlx.adapter.output_parser import Cohere2MoeOutputParserSession
        session = Cohere2MoeOutputParserSession.__new__(Cohere2MoeOutputParserSession)
        session._tokenizer = tokenizer
        session._melody = _FakeMelodyFilterLiteralNewline(None)
        session._detokenizer = None
        session._thinking_started = False
        session._thinking_closed = False
        session._tool_calls = {}

        session.process_token("TC")
        final = session.finalize()

        assert len(final.tool_calls) == 1
        args_str = final.tool_calls[0]["arguments"]
        # Must be valid strict JSON (no literal control characters)
        parsed = _json.loads(args_str)
        assert parsed["code"] == "line1\nline2"
        # The literal newline must have been escaped
        assert "\n" not in args_str or "\\n" in args_str


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

    def test_spm_fallback_buffers_split_utf8(self, tmp_path):
        _write_json(tmp_path / "tokenizer.json", {"decoder": _spm_decoder()})
        session = Gemma4OutputParserSession(
            ByteFallbackTokenizer(),
            model_path=tmp_path,
        )

        parts = []
        for token_id in [1, 2, 3]:
            parts.append(session.process_token(token_id).stream_text)
        parts.append(session.finalize().stream_text)

        text = "".join(parts)
        assert text == "\uc7a0"
        assert "\ufffd" not in text


class TestOutputParserFactory:
    def test_detects_minimax_m3_by_config(self):
        tokenizer = CohereTokenizer({1: "x"})
        factory = detect_output_parser(
            "MiniMax-M3-4bit",
            tokenizer,
            {"model_type": "minimax_m3_vl"},
        )

        assert factory is not None
        assert factory.kind == "minimax_m3"

    def test_minimax_m3_parser_extracts_tool_calls(self, monkeypatch):
        module = types.ModuleType("mlx_vlm.tool_parsers.minimax_m3")

        def parse_tool_call(text):
            assert "lookup" in text
            return {"name": "lookup", "arguments": {"query": "mlx"}}

        module.parse_tool_call = parse_tool_call
        monkeypatch.setitem(sys.modules, "mlx_vlm.tool_parsers.minimax_m3", module)

        start = "]<]minimax[>[<tool_call>"
        end = "]<]minimax[>[</tool_call>"
        tokenizer = CohereTokenizer(
            {
                1: "before ",
                2: start,
                3: ']<]minimax[>[<invoke name="lookup">',
                4: "]<]minimax[>[</invoke>",
                5: end,
                6: " after",
            }
        )
        factory = detect_output_parser(
            "MiniMax-M3-4bit",
            tokenizer,
            {"model_type": "minimax_m3_vl"},
        )
        session = factory.create_session(tokenizer)

        visible = []
        stream = []
        for token_id in [1, 2, 3, 4, 5, 6]:
            result = session.process_token(token_id)
            stream.append(result.stream_text)
            visible.append(result.visible_text)
        final = session.finalize()

        assert "".join(stream) == "before  after"
        assert start not in "".join(stream)
        assert "".join(visible) + final.visible_text == "before  after"
        assert final.tool_calls == [{"name": "lookup", "arguments": '{"query":"mlx"}'}]
        assert final.finish_reason == "tool_calls"

    def test_minimax_m3_parser_normalizes_thinking_and_strips_eos(self):
        tokenizer = CohereTokenizer(
            {
                1: "<mm:think>",
                2: "reasoning",
                3: "</mm:think>",
                4: "Answer",
                5: "[e~[",
                6: "]!d~[",
            }
        )
        factory = detect_output_parser(
            "MiniMax-M3-4bit",
            tokenizer,
            {"model_type": "minimax_m3_vl"},
        )
        session = factory.create_session(tokenizer)

        stream = []
        visible = []
        stop_seen = False
        record_flags = []
        for token_id in [1, 2, 3, 4, 6, 5]:
            result = session.process_token(token_id)
            stream.append(result.stream_text)
            visible.append(result.visible_text)
            stop_seen = stop_seen or result.is_stop
            record_flags.append(result.record_token)
        final = session.finalize()
        stream.append(final.stream_text)
        visible.append(final.visible_text)

        assert "".join(stream) == "<think>reasoning</think>Answer"
        assert "".join(visible) == "<think>reasoning</think>Answer"
        assert stop_seen is True
        assert record_flags[-1] is False

    def test_minimax_m3_factory_exposes_native_thinking_markers(self):
        tokenizer = CohereTokenizer({})
        tokenizer.convert_tokens_to_ids = lambda text: {
            "[e~[": 200020,
            "<mm:think>": 200059,
            "</mm:think>": 200060,
        }.get(text, -1)
        tokenizer.unk_token_id = -1

        factory = detect_output_parser(
            "MiniMax-M3-4bit",
            tokenizer,
            {"model_type": "minimax_m3_vl"},
        )

        assert factory.thinking_start_text == "<mm:think>"
        assert factory.thinking_start_output_text == "<think>\n"
        assert factory.thinking_end_text == "</mm:think>"
        assert factory.stop_token_ids == {200020}

    def test_detects_gemma4(self):
        tokenizer = GemmaTokenizer({1: "x"})
        factory = detect_output_parser(
            "google/gemma-4b",
            tokenizer,
            {"model_type": "gemma4"},
        )

        assert factory is not None
        assert factory.kind == "gemma4"

    def test_detects_gemma4_unified_by_config(self):
        tokenizer = GemmaTokenizer({1: "x"})
        factory = detect_output_parser(
            "some-model",
            tokenizer,
            {"model_type": "gemma4_unified"},
        )

        assert factory is not None
        assert factory.kind == "gemma4"

    def test_harmony_wrapper_regression(self):
        encoding = load_harmony_gpt_oss_encoding()
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

        encoding = load_harmony_gpt_oss_encoding()
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
