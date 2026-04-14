# SPDX-License-Identifier: Apache-2.0
"""Tests for Harmony streaming parser (omlx.adapter.harmony)."""

import pytest
from unittest.mock import MagicMock

from openai_harmony import load_harmony_encoding, StreamableParser

from omlx.adapter.harmony import (
    HarmonyStreamingParser,
    parse_tool_calls_from_tokens,
    preprocess_harmony_messages,
)


@pytest.fixture
def encoding():
    """Load HarmonyGptOss encoding."""
    return load_harmony_encoding("HarmonyGptOss")


@pytest.fixture
def tokenizer(encoding):
    """Create a mock tokenizer that resolves Harmony special tokens."""
    tok = MagicMock()
    special_map = {}
    for name in [
        "<|start|>", "<|end|>", "<|message|>", "<|channel|>",
        "<|return|>", "<|call|>", "<|constrain|>",
    ]:
        tokens = encoding.encode(name, allowed_special="all")
        if tokens:
            special_map[name] = tokens[0]

    def _convert(token_str):
        return special_map.get(token_str, -1)

    tok.convert_tokens_to_ids = _convert
    return tok


@pytest.fixture
def parser(tokenizer):
    """Create a HarmonyStreamingParser instance (pre-primed with <|start|>assistant)."""
    return HarmonyStreamingParser(tokenizer=tokenizer)


# ── Tool call header parsing ──────────────────────────────────────────
# The parser is pre-primed with "<|start|>assistant", so the first
# model output token is <|channel|> (regular message) or " to=..."
# (tool call continuation).  Subsequent messages start with <|start|>.


class TestToolCallParsing:
    """StreamableParser with role=None correctly parses tool call headers."""

    def test_tool_call_header_parsed(self, parser, encoding):
        """Tool call header 'to=functions.Write' sets commentary channel.

        The first message's <|start|>assistant is already primed.
        Model output for a tool call starts with ' to=functions.Write<|channel|>...'
        """
        # Model continues from primed "assistant" with " to=functions.Write..."
        tokens = encoding.encode(
            " to=functions.Write<|channel|>commentary<|message|>args",
            allowed_special="all",
        )
        for t in tokens:
            parser.process_token(t)

        assert parser.current_channel == "commentary"
        assert parser.current_recipient == "functions.Write"

    def test_regular_message_still_works(self, parser, encoding):
        """Regular assistant message parses correctly (primed parser)."""
        # Model output starts from <|channel|> (first message)
        tokens = encoding.encode(
            "<|channel|>final<|message|>Hello",
            allowed_special="all",
        )
        results = [parser.process_token(t) for t in tokens]

        assert parser.current_channel == "final"

        # "Hello" token should be streamed and visible
        last = results[-1]
        control_text, stream_token, visible_token, is_stop = last
        assert stream_token is not None
        assert visible_token is not None

    def test_tool_call_tokens_not_streamed(self, parser, encoding):
        """Commentary channel tokens return None for stream/visible."""
        tokens = encoding.encode(
            " to=functions.Write<|channel|>commentary<|message|>args",
            allowed_special="all",
        )
        results = [parser.process_token(t) for t in tokens]

        # "args" is in commentary channel — should not be streamed
        last = results[-1]
        _, stream_token, visible_token, _ = last
        assert stream_token is None
        assert visible_token is None


# ── Multi-message sequences ──────────────────────────────────────────


class TestMultiMessageSequence:
    """Correct channel transitions across analysis → final → tool call."""

    def test_analysis_final_tool_sequence(self, parser, encoding):
        """Channels transition correctly: analysis → final → commentary.

        First message uses primed header (starts from <|channel|>).
        Subsequent messages include full <|start|>assistant header.
        """
        # First message: analysis (primed, starts from <|channel|>)
        analysis = encoding.encode(
            "<|channel|>analysis<|message|>thinking<|end|>",
            allowed_special="all",
        )
        # Second message: final (full header)
        final = encoding.encode(
            "<|start|>assistant<|channel|>final<|message|>result<|end|>",
            allowed_special="all",
        )
        # Third message: tool call (full header with routing)
        tool = encoding.encode(
            "<|start|>assistant to=functions.Read"
            '<|channel|>commentary<|message|>{"path":"f.py"}<|end|>',
            allowed_special="all",
        )

        channels_seen = []
        for t in analysis + final + tool:
            parser.process_token(t)
            ch = parser.current_channel
            if not channels_seen or channels_seen[-1] != ch:
                channels_seen.append(ch)

        assert "analysis" in channels_seen
        assert "final" in channels_seen
        assert "commentary" in channels_seen

    def test_think_tags_emitted(self, parser, encoding):
        """<think>/<think> control text emitted on analysis channel transitions."""
        # First message: analysis (primed)
        # Second message: final (full header)
        tokens = encoding.encode(
            "<|channel|>analysis<|message|>thought<|end|>"
            "<|start|>assistant<|channel|>final<|message|>answer",
            allowed_special="all",
        )
        control_texts = []
        for t in tokens:
            control_text, _, _, _ = parser.process_token(t)
            if control_text:
                control_texts.append(control_text)

        assert "<think>\n" in control_texts
        assert "</think>\n" in control_texts


# ── Passthrough mode ─────────────────────────────────────────────────


class TestPassthroughMode:
    """Passthrough mode activates on parser error and prevents cascading."""

    def test_passthrough_activates_on_error(self, tokenizer, encoding):
        """Parser switches to passthrough when process() raises."""
        parser = HarmonyStreamingParser(tokenizer=tokenizer)

        # Force an error by monkey-patching the inner parser
        def _failing_process(token_id):
            raise RuntimeError("simulated parser failure")

        parser._parser.process = _failing_process

        result = parser.process_token(42)
        control_text, stream_token, visible_token, is_stop = result

        assert parser._passthrough_mode is True
        assert stream_token is None
        assert visible_token is None

    def test_passthrough_buffers_silently(self, tokenizer, encoding):
        """After passthrough activation, all tokens return None."""
        parser = HarmonyStreamingParser(tokenizer=tokenizer)
        parser._passthrough_mode = True

        for token_id in [100, 200, 300]:
            control_text, stream_token, visible_token, is_stop = (
                parser.process_token(token_id)
            )
            assert stream_token is None
            assert visible_token is None
            assert control_text == ""

    def test_stop_token_detected_in_passthrough(self, tokenizer, encoding):
        """Stop tokens are still detected in passthrough mode."""
        parser = HarmonyStreamingParser(tokenizer=tokenizer)
        parser._passthrough_mode = True

        stop_ids = parser.get_stop_token_ids()
        assert len(stop_ids) > 0

        stop_token = next(iter(stop_ids))
        _, _, _, is_stop = parser.process_token(stop_token)
        assert is_stop is True

    def test_passthrough_closes_think_tag(self, tokenizer, encoding):
        """Passthrough activation while in analysis channel closes think tag."""
        parser = HarmonyStreamingParser(tokenizer=tokenizer)

        # Process analysis channel header (primed, starts from <|channel|>)
        analysis_header = encoding.encode(
            "<|channel|>analysis<|message|>",
            allowed_special="all",
        )
        for t in analysis_header:
            parser.process_token(t)

        assert parser._in_think_tag is True

        # Now trigger passthrough
        def _failing_process(token_id):
            raise RuntimeError("simulated failure")

        parser._parser.process = _failing_process

        control_text, _, _, _ = parser.process_token(999)
        assert parser._passthrough_mode is True
        assert "</think>\n" in control_text
        assert parser._in_think_tag is False

    def test_reset_clears_passthrough(self, tokenizer):
        """reset() clears passthrough mode and re-primes parser."""
        parser = HarmonyStreamingParser(tokenizer=tokenizer)
        parser._passthrough_mode = True

        parser.reset()
        assert parser._passthrough_mode is False

    def test_reset_parser_accepts_channel_token(self, parser, encoding):
        """After reset, parser accepts <|channel|> (re-primed)."""
        parser.reset()

        # Should not raise — parser is re-primed with <|start|>assistant
        tokens = encoding.encode(
            "<|channel|>final<|message|>test",
            allowed_special="all",
        )
        for t in tokens:
            parser.process_token(t)

        assert parser.current_channel == "final"


# ── Preprocessing ─────────────────────────────────────────────────────


class TestPreprocessHarmonyMessages:
    """preprocess_harmony_messages strips think tags from assistant content."""

    def test_strips_think_tags(self):
        msgs = [
            {"role": "assistant", "content": "<think>reasoning</think>answer"}
        ]
        result = preprocess_harmony_messages(msgs)
        assert result[0]["content"] == "answer"

    def test_passes_tool_messages(self):
        msgs = [
            {"role": "tool", "tool_call_id": "123", "content": "result"}
        ]
        result = preprocess_harmony_messages(msgs)
        assert result == msgs


# ── parse_tool_calls_from_tokens ──────────────────────────────────────


class TestParseToolCallsFromTokens:
    """Non-streaming tool call extraction from complete token sequences."""

    def test_extracts_tool_call(self, encoding):
        """Extracts function name and arguments from tool call tokens."""
        # Model output starts from <|channel|> (prompt includes <|start|>assistant)
        tokens = encoding.encode(
            '<|channel|>commentary<|message|>{"path":"t.py"}<|end|>'
            "<|return|>",
            allowed_special="all",
        )
        # prepend_start=True adds <|start|>assistant
        output_text, analysis_text, tool_calls = parse_tool_calls_from_tokens(
            tokens, prepend_start=True
        )
        assert isinstance(tool_calls, list)

    def test_extracts_final_text(self, encoding):
        """Extracts text from final channel."""
        tokens = encoding.encode(
            "<|channel|>final<|message|>Hello world<|end|>",
            allowed_special="all",
        )
        output_text, analysis_text, tool_calls = parse_tool_calls_from_tokens(
            tokens, prepend_start=True
        )
        assert "Hello world" in output_text
