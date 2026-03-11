# SPDX-License-Identifier: Apache-2.0
"""
Tests for Anthropic API adapter.

Tests the AnthropicAdapter class for converting between Anthropic Messages API
format and internal oMLX format.
"""

import json
import pytest

from omlx.api.adapters.anthropic import AnthropicAdapter
from omlx.api.adapters.base import (
    BaseAdapter,
    InternalMessage,
    InternalRequest,
    InternalResponse,
    StreamChunk,
)
from omlx.api.anthropic_models import (
    AnthropicMessage,
    AnthropicTool,
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    MessagesRequest,
)


class TestAnthropicAdapter:
    """Tests for AnthropicAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create AnthropicAdapter instance."""
        return AnthropicAdapter()

    # =========================================================================
    # Adapter Name Tests
    # =========================================================================

    def test_adapter_name(self, adapter):
        """Test adapter name property."""
        assert adapter.name == "anthropic"

    def test_adapter_inherits_base(self, adapter):
        """Test adapter inherits from BaseAdapter."""
        assert isinstance(adapter, BaseAdapter)

    # =========================================================================
    # parse_request Tests - Basic Messages
    # =========================================================================

    def test_parse_request_simple_message(self, adapter):
        """Test parsing a simple Anthropic request."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role="user", content="Hello"),
            ],
        )

        internal = adapter.parse_request(request)

        assert isinstance(internal, InternalRequest)
        assert len(internal.messages) == 1
        assert internal.messages[0].role == "user"
        assert internal.messages[0].content == "Hello"
        assert internal.model == "claude-3-sonnet"
        assert internal.max_tokens == 1024

    def test_parse_request_multiple_messages(self, adapter):
        """Test parsing request with multiple messages."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role="user", content="Hello"),
                AnthropicMessage(role="assistant", content="Hi there!"),
                AnthropicMessage(role="user", content="How are you?"),
            ],
        )

        internal = adapter.parse_request(request)

        assert len(internal.messages) == 3
        assert internal.messages[0].role == "user"
        assert internal.messages[1].role == "assistant"
        assert internal.messages[2].role == "user"

    def test_parse_request_with_system(self, adapter):
        """Test parsing request with system message."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role="user", content="Hello"),
            ],
            system="You are a helpful assistant.",
        )

        internal = adapter.parse_request(request)

        # System message should be first
        assert len(internal.messages) == 2
        assert internal.messages[0].role == "system"
        assert internal.messages[0].content == "You are a helpful assistant."
        assert internal.messages[1].role == "user"

    # =========================================================================
    # parse_request Tests - Generation Parameters
    # =========================================================================

    def test_parse_request_with_temperature(self, adapter):
        """Test parsing request with temperature."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            temperature=0.5,
        )

        internal = adapter.parse_request(request)

        assert internal.temperature == 0.5

    def test_parse_request_with_zero_temperature(self, adapter):
        """Test parsing request with zero temperature."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            temperature=0.0,
        )

        internal = adapter.parse_request(request)

        assert internal.temperature == 0.0

    def test_parse_request_default_temperature(self, adapter):
        """Test parsing request without temperature uses default."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )

        internal = adapter.parse_request(request)

        assert internal.temperature == 1.0

    def test_parse_request_with_top_p(self, adapter):
        """Test parsing request with top_p."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            top_p=0.9,
        )

        internal = adapter.parse_request(request)

        assert internal.top_p == 0.9

    def test_parse_request_with_top_k(self, adapter):
        """Test parsing request with top_k."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            top_k=40,
        )

        internal = adapter.parse_request(request)

        assert internal.top_k == 40

    def test_parse_request_with_stream(self, adapter):
        """Test parsing request with stream=True."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            stream=True,
        )

        internal = adapter.parse_request(request)

        assert internal.stream is True

    def test_parse_request_with_stop_sequences(self, adapter):
        """Test parsing request with stop_sequences."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            stop_sequences=["STOP", "END"],
        )

        internal = adapter.parse_request(request)

        assert internal.stop == ["STOP", "END"]

    # =========================================================================
    # parse_request Tests - Tools
    # =========================================================================

    def test_parse_request_with_tools(self, adapter):
        """Test parsing request with tools."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            tools=[
                AnthropicTool(
                    name="get_weather",
                    description="Get weather info",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                        },
                    },
                )
            ],
        )

        internal = adapter.parse_request(request)

        assert internal.tools is not None
        assert len(internal.tools) == 1
        assert internal.tools[0]["function"]["name"] == "get_weather"
        assert internal.tools[0]["function"]["description"] == "Get weather info"

    def test_parse_request_generates_request_id(self, adapter):
        """Test that parse_request generates a request ID."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )

        internal = adapter.parse_request(request)

        assert internal.request_id is not None
        assert internal.request_id.startswith("msg_")

    # =========================================================================
    # format_response Tests
    # =========================================================================

    def test_format_response_basic(self, adapter):
        """Test formatting a basic response."""
        from omlx.api.anthropic_models import MessagesResponse

        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        response = InternalResponse(
            text="Hi there!",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
        )

        result = adapter.format_response(response, request)

        assert isinstance(result, MessagesResponse)
        assert result.type == "message"
        assert result.role == "assistant"
        assert result.model == "claude-3-sonnet"
        assert len(result.content) == 1
        assert result.content[0].type == "text"
        assert result.content[0].text == "Hi there!"
        assert result.stop_reason == "end_turn"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    def test_format_response_with_tool_calls(self, adapter):
        """Test formatting response with tool calls."""
        from omlx.api.openai_models import FunctionCall, ToolCall

        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        tool_calls = [
            ToolCall(
                id="toolu_123",
                type="function",
                function=FunctionCall(
                    name="get_weather",
                    arguments='{"location": "Tokyo"}',
                ),
            )
        ]
        response = InternalResponse(
            text="",
            finish_reason="tool_calls",
            tool_calls=tool_calls,
        )

        result = adapter.format_response(response, request)

        assert result.stop_reason == "tool_use"
        # Should have tool_use content block
        tool_use_blocks = [c for c in result.content if c.type == "tool_use"]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0].name == "get_weather"
        assert tool_use_blocks[0].input == {"location": "Tokyo"}

    def test_format_response_finish_reason_length(self, adapter):
        """Test formatting response with length finish reason."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        response = InternalResponse(
            text="Response truncated...",
            finish_reason="length",
        )

        result = adapter.format_response(response, request)

        assert result.stop_reason == "max_tokens"

    def test_format_response_empty_text(self, adapter):
        """Test formatting response with empty text."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        response = InternalResponse(text="")

        result = adapter.format_response(response, request)

        # Should still have at least one content block
        assert len(result.content) >= 1

    # =========================================================================
    # format_stream_chunk Tests
    # =========================================================================

    def test_format_stream_chunk_first(self, adapter):
        """Test formatting first stream chunk."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        chunk = StreamChunk(text="Hello", is_first=True)

        result = adapter.format_stream_chunk(chunk, request)

        # Should contain multiple SSE events
        assert "event: message_start" in result
        assert "event: content_block_start" in result
        assert "event: content_block_delta" in result

    def test_format_stream_chunk_middle(self, adapter):
        """Test formatting middle stream chunk."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        chunk = StreamChunk(text=" world", is_first=False, is_last=False)

        result = adapter.format_stream_chunk(chunk, request)

        assert "event: content_block_delta" in result
        assert "text_delta" in result
        assert " world" in result

    def test_format_stream_chunk_last(self, adapter):
        """Test formatting last stream chunk."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        chunk = StreamChunk(
            text="",
            finish_reason="stop",
            is_last=True,
            completion_tokens=10,
        )

        result = adapter.format_stream_chunk(chunk, request)

        assert "event: content_block_stop" in result
        assert "event: message_delta" in result
        assert "event: message_stop" in result

    def test_format_stream_chunk_with_tool_call_delta(self, adapter):
        """Test formatting stream chunk with tool call delta."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        tool_delta = {"name": "get_weather"}
        chunk = StreamChunk(tool_call_delta=tool_delta)

        result = adapter.format_stream_chunk(chunk, request)

        assert "event: content_block_delta" in result
        assert "input_json_delta" in result

    def test_format_stream_chunk_empty_no_events(self, adapter):
        """Test formatting empty stream chunk."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )
        chunk = StreamChunk(text="", is_first=False, is_last=False)

        result = adapter.format_stream_chunk(chunk, request)

        # Empty chunk should produce no events
        assert result == ""

    # =========================================================================
    # format_stream_end Tests
    # =========================================================================

    def test_format_stream_end(self, adapter):
        """Test formatting stream end marker."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )

        result = adapter.format_stream_end(request)

        # Anthropic doesn't use [DONE] marker
        assert result == ""

    # =========================================================================
    # create_error_response Tests
    # =========================================================================

    def test_create_error_response_default(self, adapter):
        """Test creating error response with defaults."""
        result = adapter.create_error_response("Something went wrong")

        assert result["type"] == "error"
        assert result["error"]["message"] == "Something went wrong"
        assert result["error"]["type"] == "api_error"

    def test_create_error_response_custom_type(self, adapter):
        """Test creating error response with custom type."""
        result = adapter.create_error_response(
            "Invalid request",
            error_type="invalid_request_error",
        )

        assert result["error"]["message"] == "Invalid request"
        assert result["error"]["type"] == "invalid_request_error"

    def test_create_error_response_authentication(self, adapter):
        """Test creating authentication error response."""
        result = adapter.create_error_response(
            "Invalid API key",
            error_type="authentication_error",
        )

        assert result["error"]["type"] == "authentication_error"
        assert result["error"]["message"] == "Invalid API key"

    # =========================================================================
    # format_error_event Tests
    # =========================================================================

    def test_format_error_event(self, adapter):
        """Test formatting error as SSE event."""
        result = adapter.format_error_event("Something went wrong")

        assert "event: error" in result
        assert "Something went wrong" in result

    def test_format_error_event_custom_type(self, adapter):
        """Test formatting error event with custom type."""
        result = adapter.format_error_event(
            "Invalid request",
            error_type="invalid_request_error",
        )

        assert "event: error" in result
        assert "invalid_request_error" in result


class TestAnthropicStreamingEvents:
    """Tests for Anthropic streaming event sequence."""

    @pytest.fixture
    def adapter(self):
        """Create AnthropicAdapter instance."""
        return AnthropicAdapter()

    def test_full_stream_sequence(self, adapter):
        """Test complete streaming event sequence."""
        request = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )

        # First chunk
        first = adapter.format_stream_chunk(
            StreamChunk(text="Hi", is_first=True),
            request,
        )

        # Middle chunk
        middle = adapter.format_stream_chunk(
            StreamChunk(text=" there"),
            request,
        )

        # Last chunk
        last = adapter.format_stream_chunk(
            StreamChunk(
                text="!",
                finish_reason="stop",
                is_last=True,
                completion_tokens=3,
            ),
            request,
        )

        # End marker
        end = adapter.format_stream_end(request)

        # Verify event sequence
        assert "message_start" in first
        assert "content_block_start" in first
        assert "content_block_delta" in first

        assert "content_block_delta" in middle

        assert "content_block_delta" in last
        assert "content_block_stop" in last
        assert "message_delta" in last
        assert "message_stop" in last

        assert end == ""  # No additional end marker

    def test_stream_preserves_model_name(self, adapter):
        """Test that streaming preserves model name."""
        request = MessagesRequest(
            model="claude-3-opus",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )

        result = adapter.format_stream_chunk(
            StreamChunk(text="Hi", is_first=True),
            request,
        )

        # Model should be in message_start event
        assert "claude-3-opus" in result


class TestAnthropicToolUseConversion:
    """Tests for tool_use block conversion in convert_anthropic_to_internal (issue #159)."""

    def test_tool_use_block_converted_to_calling_tool_format(self):
        """tool_use blocks should be converted to [Calling tool: ...] format, not [Tool call: ...]."""
        from omlx.api.anthropic_utils import convert_anthropic_to_internal
        from omlx.api.anthropic_models import MessagesRequest, AnthropicMessage

        request = MessagesRequest(
            model="test-model",
            max_tokens=1024,
            messages=[
                AnthropicMessage(role="user", content="What is the weather?"),
                AnthropicMessage(
                    role="assistant",
                    content=[
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "call_123",
                            "name": "get_weather",
                            "input": {"city": "Tokyo"},
                        },
                    ],
                ),
                AnthropicMessage(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_123",
                            "content": "Sunny, 25C",
                        },
                    ],
                ),
            ],
        )

        messages = convert_anthropic_to_internal(request)

        # Find the assistant message that should contain the tool call
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        content = assistant_msgs[0]["content"]

        # Must use [Calling tool: ...] not [Tool call: ...]
        assert "[Calling tool: get_weather(" in content
        assert "[Tool call:" not in content
