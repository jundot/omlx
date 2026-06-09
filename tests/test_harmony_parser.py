# SPDX-License-Identifier: Apache-2.0
"""
Tests for HarmonyStreamingParser.

Tests the streaming parser for gpt-oss models using Harmony format.
Uses real openai_harmony encoding for accurate testing.
"""

import pytest

from omlx.adapter.harmony import (
    HarmonyStreamingParser,
    load_harmony_gpt_oss_encoding,
    preprocess_harmony_messages,
)


class RealTokenizer:
    """Tokenizer wrapper using real openai_harmony encoding."""

    def __init__(self, enc):
        self._enc = enc

    def convert_tokens_to_ids(self, token: str) -> int:
        """Convert token string to ID using harmony encoding."""
        try:
            ids = self._enc.encode(token, allowed_special="all")
            return ids[0] if ids else -1
        except Exception:
            return -1


@pytest.fixture(scope="session")
def harmony_encoding():
    """Get the real openai_harmony encoding."""
    return load_harmony_gpt_oss_encoding()


@pytest.fixture
def tokenizer(harmony_encoding):
    """Create a tokenizer that uses real openai_harmony encoding."""
    return RealTokenizer(harmony_encoding)


@pytest.fixture
def parser(tokenizer):
    """Create a HarmonyStreamingParser with real tokenizer."""
    return HarmonyStreamingParser(tokenizer)


class TestHarmonyStreamingParser:
    """Tests for HarmonyStreamingParser."""

    def test_get_stop_token_ids(self, parser):
        """Test stop token ID extraction."""
        stop_ids = parser.get_stop_token_ids()
        # Real openai_harmony stop tokens for assistant actions
        assert 200002 in stop_ids  # <|return|>
        assert 200012 in stop_ids  # <|call|>
        assert len(stop_ids) == 2

    def test_final_channel_returns_token_ids(self, parser, harmony_encoding):
        """Test that final channel returns token IDs for both stream and visible."""
        # Encode a real harmony message sequence
        # Note: Model generates from <|channel|> since <|start|>assistant is in prompt
        tokens = harmony_encoding.encode(
            "<|channel|>final<|message|>Hello world<|end|>",
            allowed_special="all",
        )
        # tokens: [200005, 17196, 200008, 13225, 2375, 200007]
        #         channel  final message Hello   world  end

        results = []
        for token in tokens:
            result = parser.process_token(token)
            results.append(result)

        # Find content tokens (after <|message|>)
        message_idx = tokens.index(200008)  # <|message|>
        hello_token = tokens[message_idx + 1]  # Hello
        world_token = tokens[message_idx + 2]  # world

        # Content tokens should return both stream and visible
        hello_result = results[message_idx + 1]
        assert hello_result[1] == hello_token  # stream_token
        assert hello_result[2] == hello_token  # visible_token
        assert hello_result[3] is False  # not stop

        world_result = results[message_idx + 2]
        assert world_result[1] == world_token  # stream_token
        assert world_result[2] == world_token  # visible_token

    def test_analysis_channel_wrapped_in_think(self, parser, harmony_encoding):
        """Test that analysis channel outputs <think> control and stream tokens only."""
        tokens = harmony_encoding.encode(
            "<|channel|>analysis<|message|>thinking content<|end|>",
            allowed_special="all",
        )

        results = []
        for token in tokens:
            result = parser.process_token(token)
            results.append(result)

        # Collect all control text and stream/visible tokens
        control_parts = [r[0] for r in results if r[0]]
        stream_tokens = [r[1] for r in results if r[1] is not None]
        visible_tokens = [r[2] for r in results if r[2] is not None]

        # Should have <think> opening and </think> closing
        full_control = "".join(control_parts)
        assert "<think>" in full_control
        assert "</think>" in full_control

        # Analysis content should stream but not be visible
        assert len(stream_tokens) > 0  # Has stream tokens
        assert len(visible_tokens) == 0  # No visible tokens

    def test_return_token_is_stop(self, parser, harmony_encoding):
        """Test that <|return|> signals stop."""
        tokens = harmony_encoding.encode(
            "<|channel|>final<|message|>Hello<|return|>",
            allowed_special="all",
        )

        results = []
        for token in tokens:
            result = parser.process_token(token)
            results.append(result)

        # Last token (<|return|>) should signal stop
        assert results[-1][3] is True

    def test_finalize_closes_think_tag(self, parser, harmony_encoding):
        """Test that finalize() closes open think tags."""
        # Enter analysis channel without <|end|>
        tokens = harmony_encoding.encode(
            "<|channel|>analysis<|message|>thinking",
            allowed_special="all",
        )
        for token in tokens:
            parser.process_token(token)

        # Parser should be in think tag
        assert parser._in_think_tag is True

        # Finalize should close it
        final = parser.finalize()
        assert final == "</think>\n"
        assert parser._in_think_tag is False

    def test_finalize_no_output_when_no_open_tag(self, parser, harmony_encoding):
        """Test that finalize() returns empty when no open tags."""
        # Process a complete final channel message
        tokens = harmony_encoding.encode(
            "<|channel|>final<|message|>Hello<|return|>",
            allowed_special="all",
        )
        for token in tokens:
            parser.process_token(token)

        final = parser.finalize()
        assert final == ""

    def test_reset_clears_state(self, parser, harmony_encoding):
        """Test that reset() clears all parser state."""
        # Process some tokens to set state
        tokens = harmony_encoding.encode(
            "<|channel|>analysis<|message|>thinking",
            allowed_special="all",
        )
        for token in tokens:
            parser.process_token(token)

        # Verify state is set
        assert parser._in_think_tag is True
        assert parser.current_channel == "analysis"

        # Reset
        parser.reset()

        # Verify state is cleared
        assert parser._in_think_tag is False
        assert parser._prev_channel is None

    def test_multiple_messages_in_sequence(self, parser, harmony_encoding):
        """Test handling multiple messages in sequence."""
        # analysis -> final sequence
        tokens = harmony_encoding.encode(
            "<|channel|>analysis<|message|>thinking<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hello<|return|>",
            allowed_special="all",
        )

        control_parts = []
        stream_tokens = []
        visible_tokens = []

        for token in tokens:
            control, stream_token, visible_token, _ = parser.process_token(token)
            if control:
                control_parts.append(control)
            if stream_token is not None:
                stream_tokens.append(stream_token)
            if visible_token is not None:
                visible_tokens.append(visible_token)

        # Should have think tags
        full_control = "".join(control_parts)
        assert "<think>" in full_control
        assert "</think>" in full_control

        # Stream should have both analysis and final tokens
        assert len(stream_tokens) > 0

        # Visible should only have final channel tokens
        # Analysis tokens should not be in visible
        assert len(visible_tokens) > 0

    def test_channel_transition_closes_think_tag(self, parser, harmony_encoding):
        """Test that transitioning from analysis to final closes think tag."""
        # First: enter analysis channel
        tokens1 = harmony_encoding.encode(
            "<|channel|>analysis<|message|>thinking",
            allowed_special="all",
        )
        for token in tokens1:
            parser.process_token(token)

        assert parser._in_think_tag is True

        # Second: transition to final channel (via new message)
        tokens2 = harmony_encoding.encode(
            "<|end|><|start|>assistant<|channel|>final<|message|>",
            allowed_special="all",
        )

        control_parts = []
        for token in tokens2:
            control, _, _, _ = parser.process_token(token)
            if control:
                control_parts.append(control)

        # Should have closing think tag
        full_control = "".join(control_parts)
        assert "</think>" in full_control
        assert parser._in_think_tag is False


class TestHarmonyToolCalling:
    """Tests for Harmony tool call extraction."""

    def test_tool_call_extraction(self, tokenizer, harmony_encoding):
        """Test tool call extraction from commentary channel."""
        parser = HarmonyStreamingParser(tokenizer)

        # Simulate tool call message format
        # commentary channel with recipient functions.get_weather
        tokens = harmony_encoding.encode(
            '<|channel|>commentary to=functions.get_weather<|message|>{"location": "Seoul"}<|call|>',
            allowed_special="all",
        )

        for token in tokens:
            _, _, _, is_stop = parser.process_token(token)

        # Last token should be stop
        assert is_stop is True

        tool_calls = parser.get_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"
        assert "Seoul" in tool_calls[0]["arguments"]

    def test_no_tool_calls_for_final_channel(self, parser, harmony_encoding):
        """Test that final channel doesn't produce tool calls."""
        tokens = harmony_encoding.encode(
            "<|channel|>final<|message|>Hello<|return|>",
            allowed_special="all",
        )
        for token in tokens:
            parser.process_token(token)

        tool_calls = parser.get_tool_calls()
        assert len(tool_calls) == 0


class TestHarmonyEdgeCases:
    """Edge case tests for HarmonyStreamingParser."""

    def test_empty_content(self, parser, harmony_encoding):
        """Test handling of empty message content."""
        tokens = harmony_encoding.encode(
            "<|channel|>final<|message|><|end|>",
            allowed_special="all",
        )

        stream_tokens = []
        for token in tokens:
            _, stream_token, _, _ = parser.process_token(token)
            if stream_token is not None:
                stream_tokens.append(stream_token)

        # Should have no content tokens
        assert stream_tokens == []

    def test_unknown_channel_buffers_content(self, parser, harmony_encoding):
        """Test handling of unknown channel names."""
        # Note: openai_harmony may handle unknown channels differently
        tokens = harmony_encoding.encode(
            "<|channel|>someunknownchannel<|message|>content<|end|>",
            allowed_special="all",
        )

        results = []
        for token in tokens:
            control, stream_token, visible_token, _ = parser.process_token(token)
            results.append((stream_token, visible_token))

        # Unknown channel should not output to stream or visible
        content_outputs = [(s, v) for s, v in results if s is not None or v is not None]
        # Content from unknown channels should be buffered, not output
        assert len(content_outputs) == 0

    def test_consecutive_analysis_channels(self, parser, harmony_encoding):
        """Test multiple consecutive analysis channels."""
        tokens = harmony_encoding.encode(
            "<|channel|>analysis<|message|>first<|end|>"
            "<|start|>assistant<|channel|>analysis<|message|>second<|end|>",
            allowed_special="all",
        )

        control_parts = []
        for token in tokens:
            control, _, _, _ = parser.process_token(token)
            if control:
                control_parts.append(control)

        # Should have proper think tag wrapping for both
        assert control_parts.count("<think>\n") == 2
        assert control_parts.count("</think>\n") == 2


class TestHarmonyIntegration:
    """Integration-style tests for realistic Harmony sequences."""

    def test_typical_response_sequence(self, parser, harmony_encoding):
        """Test a typical Harmony response: analysis -> final."""
        tokens = harmony_encoding.encode(
            "<|channel|>analysis<|message|>Let me think...<|end|>"
            "<|start|>assistant<|channel|>final<|message|>Hello world<|return|>",
            allowed_special="all",
        )

        control_output = []
        stream_tokens = []
        visible_tokens = []

        for token in tokens:
            control, stream_token, visible_token, _ = parser.process_token(token)
            if control:
                control_output.append(control)
            if stream_token is not None:
                stream_tokens.append(stream_token)
            if visible_token is not None:
                visible_tokens.append(visible_token)

        # Control should have think tags
        full_control = "".join(control_output)
        assert "<think>" in full_control
        assert "</think>" in full_control

        # Stream should have analysis + final tokens
        assert len(stream_tokens) > 0

        # Visible should only have final tokens (analysis excluded)
        assert len(visible_tokens) > 0
        assert len(visible_tokens) < len(stream_tokens)

    def test_accumulate_visible_tokens(self, parser, harmony_encoding):
        """Test that visible tokens accumulate correctly for storage."""
        tokens = harmony_encoding.encode(
            "<|channel|>final<|message|>Hello world<|return|>",
            allowed_special="all",
        )

        visible_tokens = []
        for token in tokens:
            _, _, visible_token, _ = parser.process_token(token)
            if visible_token is not None:
                visible_tokens.append(visible_token)

        # Decode accumulated tokens
        output_text = harmony_encoding.decode(visible_tokens)
        assert "Hello" in output_text
        assert "world" in output_text


class TestHarmonyPreprocess:
    """Tests for preprocess_harmony_messages."""

    def test_strips_think_tags_from_assistant(self):
        """Test that <think> tags are stripped from assistant messages."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "<think>\nLet me think...\n</think>\nHi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        result = preprocess_harmony_messages(messages)

        assert result[0]["content"] == "Hello"  # user unchanged
        assert result[1]["content"] == "Hi there!"  # think stripped
        assert result[2]["content"] == "How are you?"  # user unchanged

    def test_preserves_user_messages(self):
        """Test that user messages are not modified."""
        messages = [
            {"role": "user", "content": "Tell me about <think> tags"},
        ]

        result = preprocess_harmony_messages(messages)
        assert result[0]["content"] == "Tell me about <think> tags"

    def test_handles_empty_content(self):
        """Test handling of empty or None content."""
        messages = [
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": None},
            {"role": "assistant"},
        ]

        result = preprocess_harmony_messages(messages)
        assert len(result) == 3

    def test_handles_no_think_tags(self):
        """Test that messages without think tags are unchanged."""
        messages = [
            {"role": "assistant", "content": "Just a normal response"},
        ]

        result = preprocess_harmony_messages(messages)
        assert result[0]["content"] == "Just a normal response"

    def test_strips_multiple_think_blocks(self):
        """Test stripping multiple <think> blocks."""
        messages = [
            {"role": "assistant", "content": "<think>first</think>Hello<think>second</think>World"},
        ]

        result = preprocess_harmony_messages(messages)
        assert result[0]["content"] == "HelloWorld"

    def test_preserves_tool_messages_unchanged(self):
        """Test that tool role messages are preserved unchanged for chat_template."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {"name": "get_weather", "arguments": '{"location": "Seoul"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": '{"temp": 20}'},
        ]

        result = preprocess_harmony_messages(messages)

        # Tool message should be preserved unchanged
        # Chat template expects role="tool" and uses last_tool_call.name from assistant
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_123"
        assert result[2]["content"] == '{"temp": 20}'


class TestHarmonyUTF8:
    """Tests for UTF-8 handling with Harmony parser.

    These tests verify that the parser returns token IDs (not decoded text)
    so the caller can use streaming detokenizer for proper UTF-8 handling.
    """

    def test_returns_token_ids_not_text(self, parser, harmony_encoding):
        """Verify that parser returns token IDs, not decoded text."""
        # Process a token in final channel
        tokens = harmony_encoding.encode(
            "<|channel|>final<|message|>Hello",
            allowed_special="all",
        )

        result = None
        for token in tokens:
            result = parser.process_token(token)

        # Last token should return token ID, not string
        control, stream_token, visible_token, is_stop = result
        assert isinstance(stream_token, int)
        assert isinstance(visible_token, int)

    def test_control_text_is_string(self, parser, harmony_encoding):
        """Verify that control text is always a string."""
        # Enter analysis channel to get <think> control
        tokens = harmony_encoding.encode(
            "<|channel|>analysis<|message|>",
            allowed_special="all",
        )

        control_found = None
        for token in tokens:
            result = parser.process_token(token)
            if result[0]:  # control_text
                control_found = result[0]

        assert isinstance(control_found, str)
        assert control_found == "<think>\n"

    def test_korean_text_handling(self, parser, harmony_encoding):
        """Test that Korean text is handled correctly via token IDs."""
        # Korean text should be tokenized and returned as token IDs
        tokens = harmony_encoding.encode(
            "<|channel|>final<|message|>안녕하세요<|return|>",
            allowed_special="all",
        )

        visible_tokens = []
        for token in tokens:
            _, _, visible_token, _ = parser.process_token(token)
            if visible_token is not None:
                visible_tokens.append(visible_token)

        # Should have token IDs (integers)
        assert all(isinstance(t, int) for t in visible_tokens)

        # Decoding should give Korean text
        decoded = harmony_encoding.decode(visible_tokens)
        assert "안녕" in decoded


class TestExtractHarmonyMessages:
    """Tests for extract_harmony_messages function."""

    def test_preserves_tool_role_and_tool_call_id(self):
        """Test that tool messages preserve role and tool_call_id."""
        from omlx.api.utils import extract_harmony_messages

        # Create mock messages
        class MockMessage:
            def __init__(self, role, content, tool_call_id=None, tool_calls=None):
                self.role = role
                self.content = content
                self.tool_call_id = tool_call_id
                self.tool_calls = tool_calls

        messages = [
            MockMessage("user", "What's the weather?"),
            MockMessage(
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": "call_123",
                        "function": {"name": "get_weather", "arguments": '{"location": "Seoul"}'},
                    }
                ],
            ),
            MockMessage("tool", '{"temp": 20}', tool_call_id="call_123"),
        ]

        result = extract_harmony_messages(messages)

        # Tool message should preserve role and tool_call_id
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "call_123"
        # Content is parsed to dict to avoid double-encoding by chat_template |tojson
        assert result[2]["content"] == {"temp": 20}

    def test_preserves_assistant_tool_calls(self):
        """Test that assistant messages preserve tool_calls field."""
        from omlx.api.utils import extract_harmony_messages

        class MockMessage:
            def __init__(self, role, content, tool_calls=None):
                self.role = role
                self.content = content
                self.tool_calls = tool_calls

        messages = [
            MockMessage(
                "assistant",
                "Let me check the weather.",
                tool_calls=[
                    {
                        "id": "call_456",
                        "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
                    }
                ],
            ),
        ]

        result = extract_harmony_messages(messages)

        # Assistant message should preserve tool_calls
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me check the weather."
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["id"] == "call_456"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        # Arguments are parsed to dict to avoid double-encoding by chat_template |tojson
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"location": "Tokyo"}

    def test_handles_content_array(self):
        """Test that content arrays are properly extracted."""
        from omlx.api.utils import extract_harmony_messages

        class MockMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        messages = [
            MockMessage("user", [
                {"type": "text", "text": "Hello"},
                {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                {"type": "text", "text": "World"},
            ]),
        ]

        result = extract_harmony_messages(messages)

        # Should extract only text parts
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello\nWorld"

    def test_handles_none_content(self):
        """Test that None content is handled correctly."""
        from omlx.api.utils import extract_harmony_messages

        class MockMessage:
            def __init__(self, role, content, tool_calls=None):
                self.role = role
                self.content = content
                self.tool_calls = tool_calls

        messages = [
            MockMessage("assistant", None, tool_calls=[
                {"id": "call_789", "function": {"name": "search", "arguments": '{}'}}
            ]),
        ]

        result = extract_harmony_messages(messages)

        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == ""
        assert "tool_calls" in result[0]

    def test_differs_from_extract_text_content(self):
        """Test that extract_harmony_messages differs from extract_text_content for tool messages."""
        from omlx.api.utils import extract_harmony_messages, extract_text_content

        class MockMessage:
            def __init__(self, role, content, tool_call_id=None, tool_calls=None):
                self.role = role
                self.content = content
                self.tool_call_id = tool_call_id
                self.tool_calls = tool_calls

        messages = [
            MockMessage("tool", '{"result": "success"}', tool_call_id="call_abc"),
        ]

        harmony_result = extract_harmony_messages(messages)
        text_result = extract_text_content(messages)

        # Harmony should preserve tool role
        assert harmony_result[0]["role"] == "tool"
        assert harmony_result[0]["tool_call_id"] == "call_abc"

        # Text content converts to user role with formatted text
        assert text_result[0]["role"] == "user"
        assert "[Tool Result" in text_result[0]["content"]

    def test_parses_json_tool_content_to_dict(self):
        """Test that JSON string content is parsed to dict to avoid double-encoding."""
        from omlx.api.utils import extract_harmony_messages

        class MockMessage:
            def __init__(self, role, content, tool_call_id=None):
                self.role = role
                self.content = content
                self.tool_call_id = tool_call_id

        messages = [
            MockMessage("tool", '{"temp": 20, "condition": "sunny"}', tool_call_id="call_123"),
        ]

        result = extract_harmony_messages(messages)

        # Content should be parsed to dict (for chat_template |tojson filter)
        assert result[0]["role"] == "tool"
        assert isinstance(result[0]["content"], dict)
        assert result[0]["content"]["temp"] == 20
        assert result[0]["content"]["condition"] == "sunny"

    def test_keeps_non_json_tool_content_as_string(self):
        """Test that non-JSON content remains as string."""
        from omlx.api.utils import extract_harmony_messages

        class MockMessage:
            def __init__(self, role, content, tool_call_id=None):
                self.role = role
                self.content = content
                self.tool_call_id = tool_call_id

        messages = [
            MockMessage("tool", "The weather is sunny and 20 degrees.", tool_call_id="call_123"),
        ]

        result = extract_harmony_messages(messages)

        # Content should remain as string
        assert result[0]["role"] == "tool"
        assert isinstance(result[0]["content"], str)
        assert result[0]["content"] == "The weather is sunny and 20 degrees."

    def test_parses_tool_call_arguments_to_dict(self):
        """Test that tool_call arguments are parsed to dict to avoid double-encoding."""
        from omlx.api.utils import extract_harmony_messages

        class MockMessage:
            def __init__(self, role, content, tool_calls=None):
                self.role = role
                self.content = content
                self.tool_calls = tool_calls

        messages = [
            MockMessage(
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": "call_456",
                        "function": {"name": "get_weather", "arguments": '{"location": "Seoul", "unit": "celsius"}'},
                    }
                ],
            ),
        ]

        result = extract_harmony_messages(messages)

        # Arguments should be parsed to dict (for chat_template |tojson filter)
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        arguments = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(arguments, dict)
        assert arguments["location"] == "Seoul"
        assert arguments["unit"] == "celsius"
