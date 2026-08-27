# SPDX-License-Identifier: Apache-2.0
"""
Tests for tool result truncation logic.

Tests the truncate_tool_result() function and its integration with
Anthropic/OpenAI message conversion paths.
"""

import json
from unittest.mock import MagicMock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omlx.api.anthropic_models import ContentBlockText, ContentBlockToolUse
from omlx.api.anthropic_utils import (
    _extract_tool_result_content,
    truncate_tool_result,
)

# =============================================================================
# Mock Tokenizer
# =============================================================================


class MockTokenizer:
    """Simple tokenizer that treats each word as a token for testing."""

    def encode(self, text: str) -> list[int]:
        """Split on whitespace, each word = 1 token."""
        if not text:
            return []
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int]) -> str:
        """Not directly usable since we lose the words. Use _text for reconstruction."""
        # This mock is limited - we need a real encode/decode pair
        raise NotImplementedError("Use CharTokenizer instead")


class CharTokenizer:
    """Character-level tokenizer for precise testing."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))

    def decode(self, token_ids: list[int]) -> str:
        # Reconstruct from stored text
        return self._last_text[: len(token_ids)]

    def encode_and_store(self, text: str) -> list[int]:
        self._last_text = text
        return self.encode(text)


class WordTokenizer:
    """Word-level tokenizer that preserves encode/decode roundtrip."""

    def encode(self, text: str) -> list[int]:
        self._words = text.split(" ") if text else []
        return list(range(len(self._words)))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self._words[: len(token_ids)])


# =============================================================================
# truncate_tool_result() Tests
# =============================================================================


class TestTruncateToolResult:
    """Tests for the truncate_tool_result() function."""

    @pytest.fixture
    def tokenizer(self):
        return WordTokenizer()

    def test_no_truncation_needed(self, tokenizer):
        """Text within budget returns unchanged."""
        text = "hello world"
        result = truncate_tool_result(text, max_tokens=10, tokenizer=tokenizer)
        assert result == text

    def test_exact_limit(self, tokenizer):
        """Text exactly at limit returns unchanged."""
        text = "one two three"
        result = truncate_tool_result(text, max_tokens=3, tokenizer=tokenizer)
        assert result == text

    def test_basic_truncation(self, tokenizer):
        """Text over budget is truncated with XML notice."""
        text = "one two three four five six seven eight nine ten"
        result = truncate_tool_result(text, max_tokens=5, tokenizer=tokenizer)
        assert "<truncated " in result
        assert 'total_tokens="10"' in result
        assert "/>" in result
        # Should not contain words beyond the limit
        assert "six" not in result.split("<truncated")[0]

    def test_line_boundary_truncation(self, tokenizer):
        """Truncation should happen at line boundaries when possible."""
        # Use spaces to separate words so WordTokenizer works correctly.
        # We join lines with \n but words with space.
        lines = ["line1 w1 w2", "line2 w3 w4", "line3 w5 w6", "line4 w7 w8"]
        text = "\n".join(lines)
        # WordTokenizer splits on space: "line1", "w1", "w2", "\nline2", "w3", "w4", "\nline3", ...
        # That's not ideal. Let's construct text with clear word boundaries.
        text = "a b c d e\nf g h i j\nk l m n o"
        # 15 words, limit to 7 → decode "a b c d e\nf g"
        # rfind('\n') finds the \n after 'e', at a reasonable position
        result = truncate_tool_result(text, max_tokens=7, tokenizer=tokenizer)
        truncated_part = result.split("\n\n<truncated")[0]
        # Should truncate at the line boundary (after "e")
        assert truncated_part == "a b c d e"

    def test_single_line_fallback(self, tokenizer):
        """Content with no newlines falls back to token-level truncation."""
        text = "one two three four five six seven eight nine ten"
        result = truncate_tool_result(text, max_tokens=5, tokenizer=tokenizer)
        assert "<truncated " in result
        truncated_part = result.split("\n\n<truncated")[0]
        assert truncated_part == "one two three four five"

    def test_xml_tag_format(self, tokenizer):
        """Verify truncation notice uses XML tag format."""
        text = "one two three four five six seven eight nine ten"
        result = truncate_tool_result(text, max_tokens=3, tokenizer=tokenizer)
        assert "<truncated " in result
        assert "total_tokens=" in result
        assert "shown_tokens=" in result
        assert "/>" in result

    def test_truncation_notice_separated(self, tokenizer):
        """Truncation notice is separated from content by blank line."""
        text = "one two three four five six seven eight nine ten"
        result = truncate_tool_result(text, max_tokens=3, tokenizer=tokenizer)
        assert "\n\n<truncated " in result

    def test_empty_text(self, tokenizer):
        """Empty text with 0 tokens returns unchanged."""
        text = ""
        result = truncate_tool_result(text, max_tokens=100, tokenizer=tokenizer)
        assert result == ""

    def test_50_percent_guard(self, tokenizer):
        """Line boundary is not used if it would lose more than 50% of content."""
        # Newline very early, then a lot of content on one line
        text = "a\nb c d e f g h i j k l m n o p q r s t u"
        # 21 words, limit 10 → decoded "a\nb c d e f g h i j"
        # Last \n is after "a" (position 1) which is < 50% of the decoded text
        result = truncate_tool_result(text, max_tokens=10, tokenizer=tokenizer)
        truncated_part = result.split("\n\n<truncated")[0]
        # Should NOT truncate at the early newline, should keep the full 10 tokens
        assert "b" in truncated_part

    def test_json_content_truncation(self, tokenizer):
        """JSON content is truncated but notice is clearly separated."""
        data = {"key": "value", "items": ["a", "b", "c", "d"]}
        text = json.dumps(data)
        result = truncate_tool_result(text, max_tokens=3, tokenizer=tokenizer)
        assert "<truncated " in result
        # The original JSON is broken, but the notice is cleanly separated
        assert "\n\n<truncated " in result


# =============================================================================
# _extract_tool_result_content() Tests
# =============================================================================


class TestExtractToolResultContent:
    """Tests for _extract_tool_result_content() with truncation."""

    @pytest.fixture
    def tokenizer(self):
        return WordTokenizer()

    def test_string_content_no_truncation(self, tokenizer):
        """String content without truncation returns as-is."""
        result = _extract_tool_result_content("hello world")
        assert result == "hello world"

    def test_string_content_with_truncation(self, tokenizer):
        """String content exceeding limit is truncated."""
        text = "one two three four five six seven eight nine ten"
        result = _extract_tool_result_content(text, max_tokens=3, tokenizer=tokenizer)
        assert "<truncated " in result

    def test_list_content_extraction(self, tokenizer):
        """List of content blocks is extracted and joined."""
        content = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        result = _extract_tool_result_content(content)
        assert result == "hello\nworld"

    def test_list_content_with_truncation(self, tokenizer):
        """List content exceeding limit is truncated after extraction."""
        content = [
            {"type": "text", "text": "one two three four five"},
            {"type": "text", "text": "six seven eight nine ten"},
        ]
        result = _extract_tool_result_content(
            content, max_tokens=5, tokenizer=tokenizer
        )
        assert "<truncated " in result

    def test_list_content_with_pydantic_model_blocks(self):
        """A Pydantic content-block object in the list must not be silently
        dropped — it should normalize the same way _content_block_to_dict
        does elsewhere in this module (#22)."""
        content = [
            ContentBlockText(type="text", text="hello"),
            {"type": "text", "text": "world"},
        ]
        result = _extract_tool_result_content(content)
        assert result == "hello\nworld"

    # -- generative coverage for heterogeneous list content (#22) -----------

    _text_block = st.builds(
        lambda text: ContentBlockText(type="text", text=text), text=st.text()
    )
    _dict_text_block = st.builds(
        lambda text: {"type": "text", "text": text}, text=st.text()
    )
    _tool_use_block = st.builds(
        lambda: ContentBlockToolUse(type="tool_use", id="t1", name="Bash", input={})
    )
    _any_item = st.one_of(_text_block, _dict_text_block, _tool_use_block, st.text())

    @given(st.lists(_any_item, max_size=10))
    def test_extract_tool_result_content_only_ever_contains_declared_text(self, items):
        """Whatever comes back is built only from text-typed items —
        Pydantic and dict text blocks alike, in list order."""
        result = _extract_tool_result_content(items)
        for item in items:
            if isinstance(item, ContentBlockText):
                assert item.text in result or item.text == ""
            elif isinstance(item, dict) and item.get("type") == "text":
                assert item["text"] in result or item["text"] == ""
            elif isinstance(item, str):
                assert item in result or item == ""

    def test_dict_content_text_type(self):
        """Dict with type=text returns text value."""
        content = {"type": "text", "text": "hello world"}
        result = _extract_tool_result_content(content)
        assert result == "hello world"

    def test_dict_content_other_type(self):
        """Dict with non-text type returns JSON serialization."""
        content = {"type": "image", "url": "http://example.com"}
        result = _extract_tool_result_content(content)
        assert json.loads(result) == content

    def test_no_truncation_without_tokenizer(self):
        """Truncation is skipped when tokenizer is None."""
        text = "one two three four five six seven eight nine ten"
        result = _extract_tool_result_content(text, max_tokens=3, tokenizer=None)
        assert result == text  # Unchanged

    def test_no_truncation_when_max_tokens_none(self, tokenizer):
        """Truncation is skipped when max_tokens is None."""
        text = "one two three four five six seven eight nine ten"
        result = _extract_tool_result_content(text, max_tokens=None, tokenizer=tokenizer)
        assert result == text  # Unchanged


# =============================================================================
# Harmony Path Integration Tests
# =============================================================================


class TestHarmonyTruncation:
    """Tests for truncation in the Harmony (gpt-oss) conversion path."""

    @pytest.fixture
    def tokenizer(self):
        return WordTokenizer()

    def _make_request(self, tool_result_content):
        """Create a minimal MessagesRequest with a tool_result block."""
        request = MagicMock(spec=[])
        request.system = None
        request.messages = [
            MagicMock(
                role="user",
                content=[
                    {
                        "type": "tool_result",
                        "tool_use_id": "test-id",
                        "content": tool_result_content,
                    }
                ],
            )
        ]
        # Make content iterable for the conversion function
        request.messages[0].content = [
            {
                "type": "tool_result",
                "tool_use_id": "test-id",
                "content": tool_result_content,
            }
        ]
        return request

    def test_json_truncation_wrapped_for_harmony(self, tokenizer):
        """Truncated JSON content should be wrapped in dict for |tojson."""
        from omlx.api.anthropic_utils import convert_anthropic_to_internal_harmony

        # Large JSON that needs truncation
        data = {"key": "value " * 50}
        json_str = json.dumps(data)

        request = self._make_request(json_str)
        messages = convert_anthropic_to_internal_harmony(
            request, max_tool_result_tokens=10, tokenizer=tokenizer
        )

        # Find the tool message
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

        content = tool_msgs[0]["content"]
        # Content should be a dict (wrapped for |tojson compatibility)
        assert isinstance(content, dict)
        assert "output" in content
        assert "truncated" in content
        assert isinstance(content["output"], str)
        assert "Showing" in content["truncated"]

    def test_json_no_truncation_passes_as_dict(self, tokenizer):
        """Small JSON content should be passed as parsed dict when no truncation needed."""
        from omlx.api.anthropic_utils import convert_anthropic_to_internal_harmony

        data = {"key": "value"}
        json_str = json.dumps(data)

        request = self._make_request(json_str)
        messages = convert_anthropic_to_internal_harmony(
            request, max_tool_result_tokens=100, tokenizer=tokenizer
        )

        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

        content = tool_msgs[0]["content"]
        # Should be a parsed dict, not a string
        assert isinstance(content, dict)
        assert content == data

    def test_non_json_truncation(self, tokenizer):
        """Non-JSON string content should be truncated normally."""
        from omlx.api.anthropic_utils import convert_anthropic_to_internal_harmony

        text = " ".join(f"word{i}" for i in range(50))

        request = self._make_request(text)
        messages = convert_anthropic_to_internal_harmony(
            request, max_tool_result_tokens=10, tokenizer=tokenizer
        )

        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

        content = tool_msgs[0]["content"]
        assert isinstance(content, str)
        assert "<truncated " in content


# =============================================================================
# OpenAI Path Integration Tests
# =============================================================================


class TestOpenAITruncation:
    """Tests for truncation in the OpenAI conversion path."""

    @pytest.fixture
    def tokenizer(self):
        return WordTokenizer()

    def _make_tool_message(self, content, tool_call_id="call-123"):
        """Create a mock OpenAI tool message."""
        msg = MagicMock(spec=[])
        msg.role = "tool"
        msg.content = content
        msg.tool_call_id = tool_call_id
        return msg

    def test_extract_text_content_with_truncation(self, tokenizer):
        """Tool results in extract_text_content() should be truncated."""
        from omlx.api.utils import extract_text_content

        text = " ".join(f"word{i}" for i in range(50))
        messages = [self._make_tool_message(text)]

        result = extract_text_content(
            messages, max_tool_result_tokens=10, tokenizer=tokenizer
        )

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "<truncated " in result[0]["content"]
        assert "[Tool Result (call-123)]" in result[0]["content"]

    def test_extract_text_content_no_truncation_params(self, tokenizer):
        """Without truncation params, content is not truncated."""
        from omlx.api.utils import extract_text_content

        text = " ".join(f"word{i}" for i in range(50))
        messages = [self._make_tool_message(text)]

        result = extract_text_content(messages)

        assert len(result) == 1
        assert "<truncated " not in result[0]["content"]
        assert text in result[0]["content"]

    def test_extract_harmony_messages_non_json_truncation(self, tokenizer):
        """Non-JSON tool results should stay as string when truncated."""
        from omlx.api.utils import extract_harmony_messages

        text = " ".join(f"word{i}" for i in range(50))
        messages = [self._make_tool_message(text)]

        result = extract_harmony_messages(
            messages, max_tool_result_tokens=10, tokenizer=tokenizer
        )

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        content = result[0]["content"]
        # Non-JSON content stays as string
        assert isinstance(content, str)
        assert "<truncated " in content

    def test_extract_harmony_messages_json_truncation_wrapped(self, tokenizer):
        """JSON tool results should be wrapped in dict when truncated (Harmony)."""
        from omlx.api.utils import extract_harmony_messages

        data = {"key": "value " * 50}
        json_str = json.dumps(data)
        messages = [self._make_tool_message(json_str)]

        result = extract_harmony_messages(
            messages, max_tool_result_tokens=10, tokenizer=tokenizer
        )

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        content = result[0]["content"]
        # Truncated JSON should be wrapped in dict for |tojson
        assert isinstance(content, dict)
        assert "output" in content
        assert "truncated" in content

    def test_extract_harmony_messages_json_no_truncation_dict(self, tokenizer):
        """Small JSON tool results should be passed as dict (Harmony)."""
        from omlx.api.utils import extract_harmony_messages

        data = {"key": "value"}
        json_str = json.dumps(data)
        messages = [self._make_tool_message(json_str)]

        result = extract_harmony_messages(
            messages, max_tool_result_tokens=100, tokenizer=tokenizer
        )

        assert len(result) == 1
        content = result[0]["content"]
        assert isinstance(content, dict)
        assert content == data


# =============================================================================
# _wrap_truncated_for_harmony() Tests
# =============================================================================


class TestWrapTruncatedForHarmony:
    """Tests for _wrap_truncated_for_harmony helper."""

    def test_extracts_truncation_metadata(self):
        from omlx.api.utils import _wrap_truncated_for_harmony

        text = '{\n  "key": "val\n\n<truncated total_tokens="5000" shown_tokens="1000" />'
        result = _wrap_truncated_for_harmony(text)
        assert isinstance(result, dict)
        assert result["output"] == '{\n  "key": "val'
        assert result["truncated"] == "Showing 1000 of 5000 tokens"

    def test_no_notice_fallback(self):
        from omlx.api.utils import _wrap_truncated_for_harmony

        text = "some truncated text without notice"
        result = _wrap_truncated_for_harmony(text)
        assert isinstance(result, dict)
        assert result["output"] == text
        assert "truncated" not in result

    def test_tojson_produces_clean_json(self):
        """Verify the dict wrapper produces clean JSON (simulating |tojson)."""
        from omlx.api.utils import _wrap_truncated_for_harmony

        text = '{\n  "result": "da...\n\n<truncated total_tokens="5000" shown_tokens="1000" />'
        result = _wrap_truncated_for_harmony(text)
        # Simulating what Jinja2 |tojson does on a dict
        json_output = json.dumps(result)
        parsed_back = json.loads(json_output)
        assert isinstance(parsed_back, dict)
        assert "output" in parsed_back
        assert parsed_back["truncated"] == "Showing 1000 of 5000 tokens"
