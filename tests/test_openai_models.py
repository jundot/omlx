# SPDX-License-Identifier: Apache-2.0
"""
Tests for OpenAI API Pydantic models.

Tests the request and response models for OpenAI-compatible chat completions,
text completions, tool calling, and structured output.
"""

import json
import pytest
from pydantic import ValidationError

from omlx.api.openai_models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ContentPart,
    FunctionCall,
    Message,
    ModelInfo,
    ModelsResponse,
    ResponseFormat,
    ResponseFormatJsonSchema,
    ToolCall,
    ToolDefinition,
    Usage,
)


class TestContentPart:
    """Tests for ContentPart model."""

    def test_text_content_part(self):
        """Test creating text content part."""
        part = ContentPart(type="text", text="Hello world")

        assert part.type == "text"
        assert part.text == "Hello world"

    def test_text_content_part_empty_text(self):
        """Test creating text content part with empty text."""
        part = ContentPart(type="text", text="")

        assert part.type == "text"
        assert part.text == ""

    def test_content_part_none_text(self):
        """Test creating content part without text."""
        part = ContentPart(type="text")

        assert part.type == "text"
        assert part.text is None


class TestMessage:
    """Tests for Message model."""

    def test_simple_user_message(self):
        """Test creating simple user message."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_simple_assistant_message(self):
        """Test creating simple assistant message."""
        msg = Message(role="assistant", content="Hi there!")

        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_system_message(self):
        """Test creating system message."""
        msg = Message(role="system", content="You are a helpful assistant.")

        assert msg.role == "system"

    def test_message_with_content_array(self):
        """Test creating message with content array."""
        msg = Message(
            role="user",
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ],
        )

        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_message_with_content_parts(self):
        """Test creating message with ContentPart objects."""
        msg = Message(
            role="user",
            content=[
                ContentPart(type="text", text="Hello"),
            ],
        )

        assert isinstance(msg.content, list)
        assert len(msg.content) == 1

    def test_assistant_message_with_tool_calls(self):
        """Test creating assistant message with tool calls."""
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        )

        assert msg.role == "assistant"
        assert msg.content is None
        assert len(msg.tool_calls) == 1

    def test_tool_response_message(self):
        """Test creating tool response message."""
        msg = Message(
            role="tool",
            content='{"weather": "sunny"}',
            tool_call_id="call_abc123",
        )

        assert msg.role == "tool"
        assert msg.tool_call_id == "call_abc123"


class TestFunctionCallAndToolCall:
    """Tests for FunctionCall and ToolCall models."""

    def test_function_call(self):
        """Test creating function call."""
        fc = FunctionCall(name="get_weather", arguments='{"location": "Tokyo"}')

        assert fc.name == "get_weather"
        assert fc.arguments == '{"location": "Tokyo"}'

    def test_function_call_empty_arguments(self):
        """Test creating function call with empty arguments."""
        fc = FunctionCall(name="no_args", arguments="{}")

        assert fc.name == "no_args"
        assert fc.arguments == "{}"

    def test_tool_call(self):
        """Test creating tool call."""
        tc = ToolCall(
            id="call_abc123",
            type="function",
            function=FunctionCall(name="get_weather", arguments='{"location": "Tokyo"}'),
        )

        assert tc.id == "call_abc123"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"

    def test_tool_call_default_type(self):
        """Test that tool call defaults to function type."""
        tc = ToolCall(
            id="call_123",
            function=FunctionCall(name="test", arguments="{}"),
        )

        assert tc.type == "function"


class TestFunctionCallArgumentsValidation:
    """Tests for FunctionCall.arguments JSON-object validator.

    Native tool-calling chat templates iterate `arguments.items()`, which
    requires the echoed value to parse back into a dict. Malformed inputs
    must be rejected at the request boundary with a 422-friendly ValueError.
    """

    def test_accepts_empty_string_as_empty_object(self):
        """Some clients send ``""`` — normalize to ``"{}"`` instead of rejecting."""
        fc = FunctionCall(name="f", arguments="")
        assert fc.arguments == "{}"

    def test_accepts_whitespace_only_string(self):
        """Whitespace-only input normalizes to ``"{}"`` too."""
        fc = FunctionCall(name="f", arguments="   ")
        assert fc.arguments == "{}"

    def test_accepts_dict_and_serializes(self):
        """Some clients wrongly send a dict; coerce to JSON string instead of 422."""
        fc = FunctionCall(name="f", arguments={"a": 1, "b": "x"})
        assert json.loads(fc.arguments) == {"a": 1, "b": "x"}

    def test_rejects_invalid_json(self):
        """Truncated or otherwise malformed JSON must raise."""
        with pytest.raises(ValidationError) as exc:
            FunctionCall(name="f", arguments='{"a": "b')
        msg = str(exc.value)
        assert "valid JSON" in msg

    def test_rejects_python_repr_with_single_quotes(self):
        """``{'a': 1}`` is not valid JSON and must be rejected."""
        with pytest.raises(ValidationError):
            FunctionCall(name="f", arguments="{'a': 1}")

    def test_rejects_bare_string(self):
        """``"Tokyo"`` parses as a JSON string, not an object — reject."""
        with pytest.raises(ValidationError) as exc:
            FunctionCall(name="f", arguments='"Tokyo"')
        assert "JSON object" in str(exc.value)

    def test_rejects_json_array(self):
        """Arrays are valid JSON but not objects; reject."""
        with pytest.raises(ValidationError) as exc:
            FunctionCall(name="f", arguments="[1, 2, 3]")
        assert "JSON object" in str(exc.value)

    def test_rejects_plain_text(self):
        """Plain text that isn't JSON at all — reject."""
        with pytest.raises(ValidationError):
            FunctionCall(name="f", arguments="Tokyo")

    def test_rejects_numeric_type(self):
        """Non-string, non-dict types are rejected."""
        with pytest.raises(ValidationError):
            FunctionCall(name="f", arguments=123)


class TestMessageToolCallsArgumentsValidation:
    """Tests that Message.tool_calls validates nested arguments.

    Since tool_calls is typed as List[dict], Pydantic does not transitively
    run FunctionCall's own validator on the nested entries. A Message-level
    validator rejects malformed echoes so downstream template rendering is
    safe.
    """

    def test_valid_tool_calls_accepted(self):
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"a": 1}'},
                }
            ],
        )
        assert msg.tool_calls[0]["function"]["arguments"] == '{"a": 1}'

    def test_empty_arguments_normalized(self):
        msg = Message(
            role="assistant",
            tool_calls=[{"function": {"name": "f", "arguments": ""}}],
        )
        assert msg.tool_calls[0]["function"]["arguments"] == "{}"

    def test_dict_arguments_coerced(self):
        msg = Message(
            role="assistant",
            tool_calls=[{"function": {"name": "f", "arguments": {"a": 1}}}],
        )
        assert json.loads(msg.tool_calls[0]["function"]["arguments"]) == {"a": 1}

    def test_malformed_json_rejected(self):
        with pytest.raises(ValidationError):
            Message(
                role="assistant",
                tool_calls=[{"function": {"name": "f", "arguments": '{"a":'}}],
            )

    def test_python_repr_rejected(self):
        with pytest.raises(ValidationError):
            Message(
                role="assistant",
                tool_calls=[{"function": {"name": "f", "arguments": "{'a': 1}"}}],
            )

    def test_plain_text_rejected(self):
        with pytest.raises(ValidationError):
            Message(
                role="assistant",
                tool_calls=[{"function": {"name": "f", "arguments": "Tokyo"}}],
            )

    def test_missing_arguments_key_allowed(self):
        """Some payloads omit arguments entirely — don't invent a failure."""
        msg = Message(
            role="assistant",
            tool_calls=[{"function": {"name": "f"}}],
        )
        assert msg.tool_calls[0]["function"] == {"name": "f"}


class TestToolDefinition:
    """Tests for ToolDefinition model."""

    def test_tool_definition(self):
        """Test creating tool definition."""
        tool = ToolDefinition(
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

        assert tool.type == "function"
        assert tool.function["name"] == "get_weather"

    def test_tool_definition_default_type(self):
        """Test that tool definition defaults to function type."""
        tool = ToolDefinition(
            function={"name": "test", "parameters": {}},
        )

        assert tool.type == "function"


class TestResponseFormat:
    """Tests for ResponseFormat model."""

    def test_text_format(self):
        """Test text response format."""
        rf = ResponseFormat(type="text")

        assert rf.type == "text"
        assert rf.json_schema is None

    def test_json_object_format(self):
        """Test JSON object response format."""
        rf = ResponseFormat(type="json_object")

        assert rf.type == "json_object"

    def test_json_schema_format(self):
        """Test JSON schema response format."""
        rf = ResponseFormat(
            type="json_schema",
            json_schema=ResponseFormatJsonSchema(
                name="person",
                description="A person object",
                schema={"type": "object", "properties": {"name": {"type": "string"}}},
            ),
        )

        assert rf.type == "json_schema"
        assert rf.json_schema.name == "person"
        assert rf.json_schema.description == "A person object"

    def test_response_format_default_type(self):
        """Test response format defaults to text."""
        rf = ResponseFormat()

        assert rf.type == "text"


class TestChatCompletionRequest:
    """Tests for ChatCompletionRequest model."""

    def test_minimal_request(self):
        """Test creating minimal chat completion request."""
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Hello")],
        )

        assert req.model == "gpt-4"
        assert len(req.messages) == 1
        assert req.temperature is None
        assert req.top_p is None
        assert req.max_tokens is None
        assert req.stream is False
        assert req.tools is None

    def test_request_with_all_fields(self):
        """Test creating request with all fields."""
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                Message(role="system", content="Be helpful"),
                Message(role="user", content="Hello"),
            ],
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            stream=True,
            stop=["STOP"],
            tools=[
                ToolDefinition(
                    function={"name": "test", "parameters": {}},
                )
            ],
            tool_choice="auto",
            response_format=ResponseFormat(type="json_object"),
        )

        assert req.model == "gpt-4"
        assert len(req.messages) == 2
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.max_tokens == 1024
        assert req.stream is True
        assert req.stop == ["STOP"]
        assert len(req.tools) == 1
        assert req.tool_choice == "auto"

    def test_request_validation_requires_model(self):
        """Test that model is required."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[Message(role="user", content="Hello")],
            )

    def test_request_validation_requires_messages(self):
        """Test that messages is required."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="gpt-4")

    def test_request_serialization(self):
        """Test request can be serialized to JSON."""
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Hello")],
        )

        json_str = req.model_dump_json()
        data = json.loads(json_str)

        assert data["model"] == "gpt-4"
        assert data["messages"][0]["role"] == "user"

    def test_xtc_defaults_to_none(self):
        """Test XTC params default to None (not sent by client)."""
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Hello")],
        )
        assert req.xtc_probability is None
        assert req.xtc_threshold is None

    def test_xtc_accepted(self):
        """Test XTC params are accepted in request."""
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Hello")],
            xtc_probability=0.5,
            xtc_threshold=0.1,
        )
        assert req.xtc_probability == 0.5
        assert req.xtc_threshold == 0.1


class TestChatCompletionResponse:
    """Tests for ChatCompletionResponse model."""

    def test_basic_response(self):
        """Test creating basic chat completion response."""
        resp = ChatCompletionResponse(
            model="gpt-4",
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(content="Hello!"),
                    finish_reason="stop",
                )
            ],
        )

        assert resp.model == "gpt-4"
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.content == "Hello!"
        assert resp.choices[0].finish_reason == "stop"

    def test_response_with_usage(self):
        """Test creating response with usage stats."""
        resp = ChatCompletionResponse(
            model="gpt-4",
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(content="Hello!"),
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )

        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5
        assert resp.usage.total_tokens == 15

    def test_response_generates_id(self):
        """Test that response generates an ID."""
        resp = ChatCompletionResponse(
            model="gpt-4",
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(content="Hello!"),
                )
            ],
        )

        assert resp.id is not None
        assert resp.id.startswith("chatcmpl-")

    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        resp = ChatCompletionResponse(
            model="gpt-4",
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call_123",
                                function=FunctionCall(name="test", arguments="{}"),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )

        assert resp.choices[0].finish_reason == "tool_calls"
        assert resp.choices[0].message.tool_calls[0].id == "call_123"

    def test_response_serialization(self):
        """Test response can be serialized to JSON."""
        resp = ChatCompletionResponse(
            model="gpt-4",
            choices=[
                ChatCompletionChoice(
                    message=AssistantMessage(content="Hello!"),
                )
            ],
        )

        json_str = resp.model_dump_json()
        data = json.loads(json_str)

        assert data["model"] == "gpt-4"
        assert data["object"] == "chat.completion"


class TestChatCompletionChunk:
    """Tests for streaming chunk models."""

    def test_basic_chunk(self):
        """Test creating basic streaming chunk."""
        chunk = ChatCompletionChunk(
            model="gpt-4",
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(content="Hello"),
                )
            ],
        )

        assert chunk.model == "gpt-4"
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.content == "Hello"

    def test_chunk_with_role(self):
        """Test chunk with role in delta."""
        chunk = ChatCompletionChunk(
            model="gpt-4",
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(role="assistant"),
                )
            ],
        )

        assert chunk.choices[0].delta.role == "assistant"

    def test_chunk_with_finish_reason(self):
        """Test chunk with finish reason."""
        chunk = ChatCompletionChunk(
            model="gpt-4",
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="stop",
                )
            ],
        )

        assert chunk.choices[0].finish_reason == "stop"

    def test_chunk_generates_id(self):
        """Test that chunk generates an ID."""
        chunk = ChatCompletionChunk(
            model="gpt-4",
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(content="Hi"),
                )
            ],
        )

        assert chunk.id is not None
        assert chunk.id.startswith("chatcmpl-")


class TestCompletionModels:
    """Tests for text completion models."""

    def test_completion_request(self):
        """Test creating completion request."""
        req = CompletionRequest(
            model="gpt-3.5-turbo-instruct",
            prompt="Once upon a time",
        )

        assert req.model == "gpt-3.5-turbo-instruct"
        assert req.prompt == "Once upon a time"
        assert req.stream is False

    def test_completion_request_with_prompt_list(self):
        """Test completion request with prompt list."""
        req = CompletionRequest(
            model="gpt-3.5-turbo-instruct",
            prompt=["Hello", "World"],
        )

        assert isinstance(req.prompt, list)
        assert len(req.prompt) == 2

    def test_completion_request_xtc_defaults_to_none(self):
        """Test XTC params default to None on CompletionRequest."""
        req = CompletionRequest(
            model="gpt-3.5-turbo-instruct",
            prompt="Hello",
        )
        assert req.xtc_probability is None
        assert req.xtc_threshold is None

    def test_completion_request_xtc_accepted(self):
        """Test XTC params are accepted in CompletionRequest."""
        req = CompletionRequest(
            model="gpt-3.5-turbo-instruct",
            prompt="Hello",
            xtc_probability=0.3,
            xtc_threshold=0.2,
        )
        assert req.xtc_probability == 0.3
        assert req.xtc_threshold == 0.2

    def test_completion_response(self):
        """Test creating completion response."""
        resp = CompletionResponse(
            model="gpt-3.5-turbo-instruct",
            choices=[
                CompletionChoice(text=" there was a dragon"),
            ],
        )

        assert resp.model == "gpt-3.5-turbo-instruct"
        assert resp.object == "text_completion"
        assert resp.choices[0].text == " there was a dragon"


class TestUsage:
    """Tests for Usage model."""

    def test_usage_creation(self):
        """Test creating usage stats."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_usage_defaults(self):
        """Test usage with defaults."""
        usage = Usage()

        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestModelInfo:
    """Tests for ModelInfo model."""

    def test_model_info(self):
        """Test creating model info."""
        info = ModelInfo(id="gpt-4")

        assert info.id == "gpt-4"
        assert info.object == "model"
        assert info.owned_by == "omlx"

    def test_models_response(self):
        """Test creating models list response."""
        resp = ModelsResponse(
            data=[
                ModelInfo(id="gpt-4"),
                ModelInfo(id="gpt-3.5-turbo"),
            ],
        )

        assert resp.object == "list"
        assert len(resp.data) == 2
        assert resp.data[0].id == "gpt-4"


# =============================================================================
# Stop Field Coercion
# =============================================================================

class TestStopCoercion:
    """Tests for stop field string-to-list coercion (OpenAI compat)."""

    def test_chat_stop_string_coerced_to_list(self):
        """A bare string for stop should be wrapped in a list."""
        req = ChatCompletionRequest(
            model="m",
            messages=[Message(role="user", content="hi")],
            stop="<|endoftext|>",
        )
        assert req.stop == ["<|endoftext|>"]

    def test_chat_stop_list_unchanged(self):
        """A list value for stop should remain unchanged."""
        req = ChatCompletionRequest(
            model="m",
            messages=[Message(role="user", content="hi")],
            stop=["a", "b"],
        )
        assert req.stop == ["a", "b"]

    def test_chat_stop_none_unchanged(self):
        """None value for stop should remain None."""
        req = ChatCompletionRequest(
            model="m",
            messages=[Message(role="user", content="hi")],
        )
        assert req.stop is None

    def test_completion_stop_string_coerced_to_list(self):
        """CompletionRequest stop string should also be coerced."""
        req = CompletionRequest(
            model="m",
            prompt="hello",
            stop="eos",
        )
        assert req.stop == ["eos"]

    def test_completion_stop_list_unchanged(self):
        """CompletionRequest stop list should remain unchanged."""
        req = CompletionRequest(
            model="m",
            prompt="hello",
            stop=["a"],
        )
        assert req.stop == ["a"]
