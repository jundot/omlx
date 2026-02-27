# SPDX-License-Identifier: Apache-2.0
"""Tests for thinking/reasoning content parser."""

import pytest

from omlx.api.thinking import ThinkingParser, extract_thinking


class TestExtractThinking:
    """Tests for non-streaming extract_thinking()."""

    def test_basic_separation(self):
        """Standard <think>reasoning</think>answer case."""
        thinking, content = extract_thinking("<think>reasoning</think>Answer")
        assert thinking == "reasoning"
        assert content == "Answer"

    def test_no_thinking(self):
        """No think tags in text."""
        thinking, content = extract_thinking("Just a normal answer")
        assert thinking == ""
        assert content == "Just a normal answer"

    def test_empty_text(self):
        """Empty input."""
        thinking, content = extract_thinking("")
        assert thinking == ""
        assert content == ""

    def test_empty_think_block(self):
        """Empty <think></think> block."""
        thinking, content = extract_thinking("<think></think>Answer")
        assert thinking == ""
        assert content == "Answer"

    def test_think_only(self):
        """Only thinking content, no answer."""
        thinking, content = extract_thinking("<think>reasoning</think>")
        assert thinking == "reasoning"
        assert content == ""

    def test_multiline_thinking(self):
        """Thinking with newlines."""
        thinking, content = extract_thinking(
            "<think>\nLet me think...\nStep 1\nStep 2\n</think>Final answer"
        )
        assert "Let me think..." in thinking
        assert "Step 1" in thinking
        assert content == "Final answer"

    def test_partial_no_open_tag(self):
        """Content before </think> without <think> tag (scheduler prefix case)."""
        thinking, content = extract_thinking("reasoning content</think>Answer")
        assert thinking == "reasoning content"
        assert content == "Answer"

    def test_multiple_think_blocks(self):
        """Multiple think blocks should all be extracted."""
        thinking, content = extract_thinking(
            "<think>first</think>middle<think>second</think>end"
        )
        assert "first" in thinking
        assert "second" in thinking
        assert "middle" in content
        assert "end" in content

    def test_thinking_with_special_chars(self):
        """Thinking with special characters."""
        thinking, content = extract_thinking(
            "<think>9.9 > 9.11 because...</think>9.9 is greater."
        )
        assert "9.9 > 9.11" in thinking
        assert content == "9.9 is greater."

    def test_thinking_with_newline_prefix(self):
        """Thinking with newline after tag (scheduler format)."""
        thinking, content = extract_thinking(
            "<think>\nLet me reason...\n</think>\nFinal answer."
        )
        assert "Let me reason..." in thinking
        assert "Final answer." in content


class TestThinkingParser:
    """Tests for streaming ThinkingParser."""

    def test_basic_streaming(self):
        """Basic streaming with complete tags in one chunk."""
        parser = ThinkingParser()
        t, c = parser.feed("<think>reasoning</think>answer")
        assert t == "reasoning"
        assert c == "answer"

    def test_tag_split_across_chunks(self):
        """<think> tag split across two chunks."""
        parser = ThinkingParser()

        t1, c1 = parser.feed("<thi")
        assert t1 == ""
        assert c1 == ""  # Buffered

        t2, c2 = parser.feed("nk>reasoning</think>answer")
        assert t2 == "reasoning"
        assert c2 == "answer"

    def test_close_tag_split(self):
        """</think> tag split across chunks."""
        parser = ThinkingParser()

        t1, c1 = parser.feed("<think>reason")
        assert t1 == "reason"
        assert c1 == ""

        t2, c2 = parser.feed("ing</thi")
        assert t2 == "ing"
        assert c2 == ""  # </thi buffered

        t3, c3 = parser.feed("nk>answer")
        assert t3 == ""
        assert c3 == "answer"

    def test_no_thinking_content(self):
        """Regular content without think tags."""
        parser = ThinkingParser()
        t, c = parser.feed("Hello, world!")
        assert t == ""
        assert c == "Hello, world!"

    def test_empty_feed(self):
        """Empty string feed."""
        parser = ThinkingParser()
        t, c = parser.feed("")
        assert t == ""
        assert c == ""

    def test_thinking_only_stream(self):
        """Stream that contains only thinking."""
        parser = ThinkingParser()

        t1, c1 = parser.feed("<think>Let me think")
        assert t1 == "Let me think"
        assert c1 == ""

        t2, c2 = parser.feed(" more</think>")
        assert t2 == " more"
        assert c2 == ""

        t3, c3 = parser.finish()
        assert t3 == ""
        assert c3 == ""

    def test_finish_flushes_buffer(self):
        """finish() should flush partial tag buffer."""
        parser = ThinkingParser()

        t1, c1 = parser.feed("Hello <")
        assert c1 == "Hello "  # '<' buffered
        assert t1 == ""

        t2, c2 = parser.finish()
        assert c2 == "<"  # Flushed as content
        assert t2 == ""

    def test_not_a_tag(self):
        """< followed by non-tag content."""
        parser = ThinkingParser()
        t, c = parser.feed("a < b and c > d")
        assert t == ""
        assert c == "a < b and c > d"

    def test_multiple_chunks_progressive(self):
        """Progressive streaming: one or few chars at a time."""
        parser = ThinkingParser()
        full_text = "<think>reasoning</think>answer"

        all_thinking = []
        all_content = []
        for char in full_text:
            t, c = parser.feed(char)
            all_thinking.append(t)
            all_content.append(c)
        t, c = parser.finish()
        all_thinking.append(t)
        all_content.append(c)

        assert "".join(all_thinking) == "reasoning"
        assert "".join(all_content) == "answer"

    def test_transition_thinking_to_content(self):
        """Transition from thinking to content across chunks."""
        parser = ThinkingParser()

        t1, c1 = parser.feed("<think>step1")
        assert t1 == "step1"

        t2, c2 = parser.feed("</think>The answer is 42.")
        assert t2 == ""
        assert c2 == "The answer is 42."

    def test_angle_bracket_in_thinking(self):
        """Angle brackets inside thinking that are not tags."""
        parser = ThinkingParser()

        t1, c1 = parser.feed("<think>if x > 0 then y < 10")
        # The > and < characters should pass through since they don't form valid tags
        assert "x > 0" in t1 or "x > 0" in (t1 + parser._buffer)

    def test_real_world_qwen3_output(self):
        """Simulate real Qwen3.5 output pattern."""
        parser = ThinkingParser()

        chunks = [
            "<think>\n",
            "The user wants me to ",
            "calculate 2+2.\n",
            "Let me think...\n",
            "2+2 = 4\n",
            "</think>\n",
            "The answer is ",
            "**4**.",
        ]

        all_thinking = []
        all_content = []
        for chunk in chunks:
            t, c = parser.feed(chunk)
            all_thinking.append(t)
            all_content.append(c)
        t, c = parser.finish()
        all_thinking.append(t)
        all_content.append(c)

        thinking = "".join(all_thinking)
        content = "".join(all_content)

        assert "calculate 2+2" in thinking
        assert "2+2 = 4" in thinking
        assert "The answer is" in content
        assert "**4**" in content
        assert "<think>" not in thinking
        assert "</think>" not in content


class TestCleanSpecialTokens:
    """Tests for clean_special_tokens (preserves think tags)."""

    def test_preserves_think_tags(self):
        from omlx.api.utils import clean_special_tokens
        result = clean_special_tokens("<think>reasoning</think>Answer")
        assert "<think>reasoning</think>Answer" == result

    def test_removes_special_tokens(self):
        from omlx.api.utils import clean_special_tokens
        result = clean_special_tokens("<|im_end|>Hello<|endoftext|>")
        assert result == "Hello"

    def test_removes_special_preserves_think(self):
        from omlx.api.utils import clean_special_tokens
        result = clean_special_tokens(
            "<|im_start|><think>reasoning</think>Answer<|im_end|>"
        )
        assert "<think>reasoning</think>Answer" == result


class TestCleanOutputTextBackwardCompat:
    """Verify clean_output_text still strips thinking (backward compat)."""

    def test_still_removes_thinking(self):
        from omlx.api.utils import clean_output_text
        result = clean_output_text("<think>reasoning</think>Answer")
        assert result == "Answer"

    def test_still_removes_partial_think(self):
        from omlx.api.utils import clean_output_text
        result = clean_output_text("reasoning content</think>Answer")
        assert result == "Answer"

    def test_still_removes_special_tokens(self):
        from omlx.api.utils import clean_output_text
        result = clean_output_text("<|im_end|>Hello<|endoftext|>")
        assert result == "Hello"
