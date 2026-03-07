# SPDX-License-Identifier: Apache-2.0
"""
Tests for OpenAI API adapter.

Tests the OpenAIAdapter class and base adapter data structures for converting
between OpenAI API format and internal oMLX format.
"""

import json
import pytest

from omlx.api.adapters.base import (
    BaseAdapter,
    InternalMessage,
    InternalRequest,
    InternalResponse,
    StreamChunk,
)
from omlx.api.adapters.openai import OpenAIAdapter
from omlx.api.openai_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ContentPart,
    Message,
    ToolDefinition,
)


class TestInternalDataClasses:
    """Tests for internal data class structures."""

    # =========================================================================
    # InternalMessage Tests
    # =========================================================================

    def test_internal_message_basic(self):
        """Test InternalMessage with required fields only."""
        msg = InternalMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_internal_message_with_optional_fields(self):
        """Test InternalMessage with all optional fields."""
        tool_calls = [{"id": "call_123", "function": {"name": "test"}}]
        msg = InternalMessage(
            role="assistant",
            content="Response",
            name="assistant_name",
            tool_calls=tool_calls,
            tool_call_id="call_456",
        )
        assert msg.role == "assistant"
        assert msg.content == "Response"
        assert msg.name == "assistant_name"
        assert msg.tool_calls == tool_calls
        assert msg.tool_call_id == "call_456"

    def test_internal_message_roles(self):
        """Test InternalMessage with different roles."""
        for role in ["user", "assistant", "system", "tool"]:
            msg = InternalMessage(role=role, content=f"Content for {role}")
            assert msg.role == role

    # =========================================================================
    # InternalRequest Tests
    # =========================================================================

    def test_internal_request_minimal(self):
        """Test InternalRequest with minimal required fields."""
        messages = [InternalMessage(role="user", content="Hello")]
        req = InternalRequest(messages=messages)

        assert req.messages == messages
        assert req.max_tokens == 2048
        assert req.temperature == 1.0
        assert req.top_p == 1.0
        assert req.top_k == 0
        assert req.stream is False
        assert req.stop is None
        assert req.stop_token_ids is None
        assert req.tools is None
        assert req.tool_choice is None
        assert req.response_format is None
        assert req.model is None
        assert req.request_id is None

    def test_internal_request_with_all_fields(self):
        """Test InternalRequest with all fields set."""
        messages = [
            InternalMessage(role="system", content="Be helpful"),
            InternalMessage(role="user", content="Hello"),
        ]
        tools = [{"type": "function", "function": {"name": "test"}}]

        req = InternalRequest(
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            stream=True,
            stop=["STOP", "END"],
            stop_token_ids=[1, 2],
            tools=tools,
            tool_choice="auto",
            response_format={"type": "json_object"},
            model="test-model",
            request_id="req-123",
        )

        assert len(req.messages) == 2
        assert req.max_tokens == 1024
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.top_k == 50
        assert req.stream is True
        assert req.stop == ["STOP", "END"]
        assert req.stop_token_ids == [1, 2]
        assert req.tools == tools
        assert req.tool_choice == "auto"
        assert req.response_format == {"type": "json_object"}
        assert req.model == "test-model"
        assert req.request_id == "req-123"

    # =========================================================================
    # InternalResponse Tests
    # =========================================================================

    def test_internal_response_minimal(self):
        """Test InternalResponse with minimal fields."""
        resp = InternalResponse(text="Hello!")

        assert resp.text == "Hello!"
        assert resp.finish_reason is None
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0
        assert resp.tool_calls is None
        assert resp.request_id is None
        assert resp.model is None

    def test_internal_response_with_all_fields(self):
        """Test InternalResponse with all fields set."""
        tool_calls = [{"id": "call_123", "function": {"name": "get_weather"}}]

        resp = InternalResponse(
            text="Here is the weather.",
            finish_reason="tool_calls",
            prompt_tokens=100,
            completion_tokens=50,
            tool_calls=tool_calls,
            request_id="chatcmpl-abc123",
            model="test-model",
        )

        assert resp.text == "Here is the weather."
        assert resp.finish_reason == "tool_calls"
        assert resp.prompt_tokens == 100
        assert resp.completion_tokens == 50
        assert resp.tool_calls == tool_calls
        assert resp.request_id == "chatcmpl-abc123"
        assert resp.model == "test-model"

    # =========================================================================
    # StreamChunk Tests
    # =========================================================================

    def test_stream_chunk_default(self):
        """Test StreamChunk with defaults."""
        chunk = StreamChunk()

        assert chunk.text == ""
        assert chunk.finish_reason is None
        assert chunk.tool_call_delta is None
        assert chunk.is_first is False
        assert chunk.is_last is False
        assert chunk.prompt_tokens == 0
        assert chunk.completion_tokens == 0

    def test_stream_chunk_first_chunk(self):
        """Test StreamChunk as first chunk."""
        chunk = StreamChunk(text="Hello", is_first=True)

        assert chunk.text == "Hello"
        assert chunk.is_first is True
        assert chunk.is_last is False

    def test_stream_chunk_last_chunk(self):
        """Test StreamChunk as last chunk."""
        chunk = StreamChunk(
            text="",
            finish_reason="stop",
            is_last=True,
            prompt_tokens=100,
            completion_tokens=50,
        )

        assert chunk.text == ""
        assert chunk.finish_reason == "stop"
        assert chunk.is_last is True
        assert chunk.prompt_tokens == 100
        assert chunk.completion_tokens == 50

    def test_stream_chunk_with_tool_call_delta(self):
        """Test StreamChunk with tool call delta."""
        tool_delta = {"name": "get_weather", "arguments": '{"location":'}
        chunk = StreamChunk(tool_call_delta=tool_delta)

        assert chunk.tool_call_delta == tool_delta


class TestOpenAIAdapter:
    """Tests for OpenAIAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create OpenAIAdapter instance."""
        return OpenAIAdapter()

    # =========================================================================
    # Adapter Name Tests
    # =========================================================================

    def test_adapter_name(self, adapter):
        """Test adapter name property."""
        assert adapter.name == "openai"

    def test_adapter_inherits_base(self, adapter):
        """Test adapter inherits from BaseAdapter."""
        assert isinstance(adapter, BaseAdapter)

    # =========================================================================
    # parse_request Tests
    # =========================================================================

    def test_parse_request_simple_message(self, adapter):
        """Test parsing a simple chat request."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                Message(role="user", content="Hello"),
            ],
        )

        internal = adapter.parse_request(request)

        assert isinstance(internal, InternalRequest)
        assert len(internal.messages) == 1
        assert internal.messages[0].role == "user"
        assert internal.messages[0].content == "Hello"
        assert internal.model == "test-model"

    def test_parse_request_multiple_messages(self, adapter):
        """Test parsing request with multiple messages."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                Message(role="system", content="Be helpful"),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
                Message(role="user", content="How are you?"),
            ],
        )

        internal = adapter.parse_request(request)

        assert len(internal.messages) == 4
        assert internal.messages[0].role == "system"
        assert internal.messages[1].role == "user"
        assert internal.messages[2].role == "assistant"
        assert internal.messages[3].role == "user"

    def test_parse_request_with_temperature(self, adapter):
        """Test parsing request with temperature."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.5,
        )

        internal = adapter.parse_request(request)

        assert internal.temperature == 0.5

    def test_parse_request_with_zero_temperature(self, adapter):
        """Test parsing request with zero temperature."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            temperature=0.0,
        )

        internal = adapter.parse_request(request)

        assert internal.temperature == 0.0

    def test_parse_request_default_temperature(self, adapter):
        """Test parsing request without temperature uses default."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        internal = adapter.parse_request(request)

        assert internal.temperature == 1.0

    def test_parse_request_with_top_p(self, adapter):
        """Test parsing request with top_p."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            top_p=0.9,
        )

        internal = adapter.parse_request(request)

        assert internal.top_p == 0.9

    def test_parse_request_with_min_p(self, adapter):
        """Test parsing request with min_p."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            min_p=0.1,
        )

        internal = adapter.parse_request(request)

        assert internal.min_p == 0.1

    def test_parse_request_with_presence_penalty(self, adapter):
        """Test parsing request with presence_penalty."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            presence_penalty=0.5,
        )

        internal = adapter.parse_request(request)

        assert internal.presence_penalty == 0.5

    def test_parse_request_default_min_p_and_presence_penalty(self, adapter):
        """Test default min_p and presence_penalty values."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        internal = adapter.parse_request(request)

        assert internal.min_p == 0.0
        assert internal.presence_penalty == 0.0

    def test_parse_request_with_max_tokens(self, adapter):
        """Test parsing request with max_tokens."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            max_tokens=500,
        )

        internal = adapter.parse_request(request)

        assert internal.max_tokens == 500

    def test_parse_request_default_max_tokens(self, adapter):
        """Test parsing request without max_tokens uses default."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        internal = adapter.parse_request(request)

        assert internal.max_tokens == 2048

    def test_parse_request_with_stream_true(self, adapter):
        """Test parsing request with stream=True."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            stream=True,
        )

        internal = adapter.parse_request(request)

        assert internal.stream is True

    def test_parse_request_with_stream_false(self, adapter):
        """Test parsing request with stream=False."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            stream=False,
        )

        internal = adapter.parse_request(request)

        assert internal.stream is False

    def test_parse_request_with_stop_list(self, adapter):
        """Test parsing request with stop sequences as list."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            stop=["STOP", "END"],
        )

        internal = adapter.parse_request(request)

        assert internal.stop == ["STOP", "END"]

    def test_parse_request_with_tools(self, adapter):
        """Test parsing request with tools."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            tools=[
                ToolDefinition(
                    type="function",
                    function={
                        "name": "get_weather",
                        "description": "Get weather info",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                        },
                    },
                )
            ],
        )

        internal = adapter.parse_request(request)

        assert internal.tools is not None
        assert len(internal.tools) == 1
        assert internal.tools[0]["function"]["name"] == "get_weather"

    def test_parse_request_with_tool_choice(self, adapter):
        """Test parsing request with tool_choice."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            tool_choice="auto",
        )

        internal = adapter.parse_request(request)

        assert internal.tool_choice == "auto"

    def test_parse_request_with_response_format(self, adapter):
        """Test parsing request with response_format."""
        from omlx.api.openai_models import ResponseFormat

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            response_format=ResponseFormat(type="json_object"),
        )

        internal = adapter.parse_request(request)

        # response_format is passed through (can be dict or ResponseFormat)
        assert internal.response_format is not None

    def test_parse_request_generates_request_id(self, adapter):
        """Test that parse_request generates a request ID."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        internal = adapter.parse_request(request)

        assert internal.request_id is not None
        assert internal.request_id.startswith("chatcmpl-")

    def test_parse_request_with_content_array(self, adapter):
        """Test parsing request with content array."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                Message(
                    role="user",
                    content=[{"type": "text", "text": "Hello world"}],
                ),
            ],
        )

        internal = adapter.parse_request(request)

        assert len(internal.messages) == 1
        # Content should be extracted
        assert "Hello world" in internal.messages[0].content

    # =========================================================================
    # format_response Tests
    # =========================================================================

    def test_format_response_basic(self, adapter):
        """Test formatting a basic response."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        response = InternalResponse(
            text="Hi there!",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            request_id="chatcmpl-abc123",
        )

        result = adapter.format_response(response, request)

        assert isinstance(result, ChatCompletionResponse)
        assert result.model == "test-model"
        assert result.object == "chat.completion"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hi there!"
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15

    def test_format_response_with_tool_calls(self, adapter):
        """Test formatting response with tool calls."""
        from omlx.api.openai_models import FunctionCall, ToolCall

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        tool_calls = [
            ToolCall(
                id="call_abc123",
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

        assert result.choices[0].finish_reason == "tool_calls"
        assert result.choices[0].message.tool_calls == tool_calls

    def test_format_response_empty_text(self, adapter):
        """Test formatting response with empty text."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        response = InternalResponse(text="")

        result = adapter.format_response(response, request)

        # Empty text should result in None content
        assert result.choices[0].message.content is None

    def test_format_response_with_special_tokens(self, adapter):
        """Test formatting response cleans special tokens."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        response = InternalResponse(
            text="Hello<|im_end|>",
            finish_reason="stop",
        )

        result = adapter.format_response(response, request)

        assert result.choices[0].message.content == "Hello"

    def test_format_response_preserves_request_id(self, adapter):
        """Test formatting response preserves request ID."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        response = InternalResponse(
            text="Hi!",
            request_id="chatcmpl-custom123",
        )

        result = adapter.format_response(response, request)

        assert result.id == "chatcmpl-custom123"

    # =========================================================================
    # format_stream_chunk Tests
    # =========================================================================

    def test_format_stream_chunk_basic(self, adapter):
        """Test formatting a basic stream chunk."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        chunk = StreamChunk(text="Hello")

        result = adapter.format_stream_chunk(chunk, request)

        assert result.startswith("data: ")
        assert result.endswith("\n\n")

        # Parse the JSON
        json_str = result[6:-2]  # Remove "data: " prefix and "\n\n" suffix
        data = json.loads(json_str)

        assert data["object"] == "chat.completion.chunk"
        assert data["model"] == "test-model"
        assert data["choices"][0]["delta"]["content"] == "Hello"

    def test_format_stream_chunk_first(self, adapter):
        """Test formatting first stream chunk includes role."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        chunk = StreamChunk(text="Hi", is_first=True)

        result = adapter.format_stream_chunk(chunk, request)

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["choices"][0]["delta"]["role"] == "assistant"

    def test_format_stream_chunk_last_with_finish_reason(self, adapter):
        """Test formatting last stream chunk includes finish reason."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        chunk = StreamChunk(
            text="",
            finish_reason="stop",
            is_last=True,
            prompt_tokens=10,
            completion_tokens=5,
        )

        result = adapter.format_stream_chunk(chunk, request)

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["choices"][0]["finish_reason"] == "stop"
        # Note: ChatCompletionChunk may not have usage field in all implementations

    def test_format_stream_chunk_with_tool_call_delta(self, adapter):
        """Test formatting stream chunk with tool call delta."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )
        # tool_call_delta should be a list for OpenAI format
        tool_delta = [{"index": 0, "function": {"name": "get_weather"}}]
        chunk = StreamChunk(tool_call_delta=tool_delta)

        result = adapter.format_stream_chunk(chunk, request)

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["choices"][0]["delta"]["tool_calls"] == tool_delta

    # =========================================================================
    # format_stream_end Tests
    # =========================================================================

    def test_format_stream_end(self, adapter):
        """Test formatting stream end marker."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        result = adapter.format_stream_end(request)

        assert result == "data: [DONE]\n\n"

    # =========================================================================
    # create_error_response Tests
    # =========================================================================

    def test_create_error_response_default(self, adapter):
        """Test creating error response with defaults."""
        result = adapter.create_error_response("Something went wrong")

        assert result["error"]["message"] == "Something went wrong"
        assert result["error"]["type"] == "server_error"
        assert result["error"]["code"] == 500
        assert result["error"]["param"] is None

    def test_create_error_response_custom_type(self, adapter):
        """Test creating error response with custom type."""
        result = adapter.create_error_response(
            "Invalid request",
            error_type="invalid_request_error",
            status_code=400,
        )

        assert result["error"]["message"] == "Invalid request"
        assert result["error"]["type"] == "invalid_request_error"
        assert result["error"]["code"] == 400

    def test_create_error_response_not_found(self, adapter):
        """Test creating 404 error response."""
        result = adapter.create_error_response(
            "Model not found",
            error_type="not_found_error",
            status_code=404,
        )

        assert result["error"]["code"] == 404
        assert result["error"]["type"] == "not_found_error"
