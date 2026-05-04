# SPDX-License-Identifier: Apache-2.0
"""
Tests for Anthropic API Pydantic models.

Tests the request and response models for Anthropic Messages API,
including content blocks, tools, and streaming events.
"""

import json
import pytest
from pydantic import ValidationError

from omlx.api.anthropic_models import (
    AnthropicErrorDetail,
    AnthropicErrorResponse,
    AnthropicMessage,
    AnthropicTool,
    AnthropicUsage,
    ContentBlockDeltaEvent,
    ContentBlockDocument,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    ErrorEvent,
    InputJsonDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    MessagesRequest,
    MessagesResponse,
    PingEvent,
    SystemContent,
    TextDelta,
    ThinkingConfig,
    TokenCountRequest,
    TokenCountResponse,
    ToolChoice,
)


class TestContentBlocks:
    """Tests for content block models."""

    def test_content_block_text(self):
        """Test creating text content block."""
        block = ContentBlockText(text="Hello world")

        assert block.type == "text"
        assert block.text == "Hello world"

    def test_content_block_text_empty(self):
        """Test creating empty text content block."""
        block = ContentBlockText(text="")

        assert block.type == "text"
        assert block.text == ""

    def test_content_block_tool_use(self):
        """Test creating tool use content block."""
        block = ContentBlockToolUse(
            id="toolu_abc123",
            name="get_weather",
            input={"location": "Tokyo"},
        )

        assert block.type == "tool_use"
        assert block.id == "toolu_abc123"
        assert block.name == "get_weather"
        assert block.input == {"location": "Tokyo"}

    def test_content_block_tool_use_empty_input(self):
        """Test creating tool use with empty input."""
        block = ContentBlockToolUse(
            id="toolu_123",
            name="no_args_tool",
            input={},
        )

        assert block.input == {}

    def test_content_block_tool_result(self):
        """Test creating tool result content block."""
        block = ContentBlockToolResult(
            tool_use_id="toolu_abc123",
            content="The weather is sunny.",
        )

        assert block.type == "tool_result"
        assert block.tool_use_id == "toolu_abc123"
        assert block.content == "The weather is sunny."
        assert block.is_error is None

    def test_content_block_tool_result_with_error(self):
        """Test creating tool result with error."""
        block = ContentBlockToolResult(
            tool_use_id="toolu_abc123",
            content="Error: API unavailable",
            is_error=True,
        )

        assert block.is_error is True

    def test_content_block_tool_result_dict_content(self):
        """Test tool result with dict content."""
        block = ContentBlockToolResult(
            tool_use_id="toolu_123",
            content={"weather": "sunny", "temperature": 25},
        )

        assert isinstance(block.content, dict)

    def test_content_block_tool_result_list_content(self):
        """Test tool result with list content."""
        block = ContentBlockToolResult(
            tool_use_id="toolu_123",
            content=[{"type": "text", "text": "Result"}],
        )

        assert isinstance(block.content, list)

    def test_content_block_document_pdf(self):
        """Test creating document content block for PDF."""
        block = ContentBlockDocument(
            source={
                "type": "base64",
                "media_type": "application/pdf",
                "data": "JVBERi0xLjQ=",
            },
            title="test.pdf",
        )

        assert block.type == "document"
        assert block.source["media_type"] == "application/pdf"
        assert block.title == "test.pdf"

    def test_content_block_document_text(self):
        """Test creating document content block for plain text."""
        import base64

        text_data = base64.b64encode(b"Hello world").decode()
        block = ContentBlockDocument(
            source={
                "type": "base64",
                "media_type": "text/plain",
                "data": text_data,
            },
        )

        assert block.type == "document"
        assert block.title is None

    def test_content_block_document_in_message(self):
        """Test that document blocks are accepted in AnthropicMessage."""
        msg = AnthropicMessage(
            role="user",
            content=[
                ContentBlockText(text="Read this document:"),
                ContentBlockDocument(
                    source={
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "JVBERi0=",
                    },
                    title="manual.pdf",
                ),
            ],
        )

        assert len(msg.content) == 2
        assert msg.content[1].type == "document"


class TestSystemContent:
    """Tests for SystemContent model."""

    def test_system_content(self):
        """Test creating system content."""
        content = SystemContent(text="You are a helpful assistant.")

        assert content.type == "text"
        assert content.text == "You are a helpful assistant."
        assert content.cache_control is None

    def test_system_content_with_cache_control(self):
        """Test system content with cache control."""
        content = SystemContent(
            text="System prompt",
            cache_control={"type": "ephemeral"},
        )

        assert content.cache_control == {"type": "ephemeral"}


class TestAnthropicMessage:
    """Tests for AnthropicMessage model."""

    def test_user_message_string_content(self):
        """Test creating user message with string content."""
        msg = AnthropicMessage(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message_string_content(self):
        """Test creating assistant message with string content."""
        msg = AnthropicMessage(role="assistant", content="Hi there!")

        assert msg.role == "assistant"

    def test_user_message_content_blocks(self):
        """Test creating user message with content blocks."""
        msg = AnthropicMessage(
            role="user",
            content=[
                ContentBlockText(text="Hello"),
                ContentBlockText(text="World"),
            ],
        )

        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_message_with_tool_result(self):
        """Test creating user message with tool result."""
        msg = AnthropicMessage(
            role="user",
            content=[
                ContentBlockToolResult(
                    tool_use_id="toolu_123",
                    content="Result here",
                )
            ],
        )

        assert msg.role == "user"
        assert msg.content[0].type == "tool_result"

    def test_assistant_message_with_tool_use(self):
        """Test creating assistant message with tool use."""
        msg = AnthropicMessage(
            role="assistant",
            content=[
                ContentBlockText(text="Let me check the weather."),
                ContentBlockToolUse(
                    id="toolu_abc",
                    name="get_weather",
                    input={"location": "Tokyo"},
                ),
            ],
        )

        assert msg.role == "assistant"
        assert len(msg.content) == 2

    def test_message_role_validation(self):
        """Test that message role must be user or assistant."""
        # Valid roles
        AnthropicMessage(role="user", content="Hello")
        AnthropicMessage(role="assistant", content="Hi")

        # Invalid role
        with pytest.raises(ValidationError):
            AnthropicMessage(role="system", content="System")


class TestAnthropicTool:
    """Tests for AnthropicTool model."""

    def test_tool_definition(self):
        """Test creating tool definition."""
        tool = AnthropicTool(
            name="get_weather",
            description="Get current weather for a location",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        )

        assert tool.name == "get_weather"
        assert tool.description == "Get current weather for a location"
        assert tool.input_schema["type"] == "object"

    def test_tool_without_description(self):
        """Test tool without description."""
        tool = AnthropicTool(
            name="simple_tool",
            input_schema={"type": "object"},
        )

        assert tool.description is None

    def test_tool_with_cache_control(self):
        """Test tool with cache control."""
        tool = AnthropicTool(
            name="cached_tool",
            input_schema={"type": "object"},
            cache_control={"type": "ephemeral"},
        )

        assert tool.cache_control == {"type": "ephemeral"}

    def test_server_side_web_search_tool(self):
        """Server-side web_search tool: type-only, no input_schema."""
        tool = AnthropicTool(type="web_search_20250305", name="web_search")

        assert tool.type == "web_search_20250305"
        assert tool.name == "web_search"
        assert tool.input_schema is None

    def test_server_side_code_execution_tool(self):
        """Server-side code_execution tool validates without input_schema."""
        tool = AnthropicTool(type="code_execution_20250825", name="code_execution")

        assert tool.type == "code_execution_20250825"
        assert tool.input_schema is None

    def test_server_side_tool_with_extra_fields(self):
        """Server-side tools may carry extra Anthropic-defined fields."""
        tool = AnthropicTool(
            type="web_search_20250305",
            name="web_search",
            max_uses=5,
            allowed_domains=["docs.anthropic.com"],
        )

        assert tool.type == "web_search_20250305"
        # extra="allow" preserves unknown fields on the model
        dumped = tool.model_dump()
        assert dumped["max_uses"] == 5
        assert dumped["allowed_domains"] == ["docs.anthropic.com"]

    def test_tool_rejects_missing_schema_and_type(self):
        """Tool without input_schema or type is invalid."""
        with pytest.raises(ValidationError):
            AnthropicTool(name="broken")

    def test_tool_with_only_input_schema_still_works(self):
        """Regression: existing user-defined tool form still validates."""
        tool = AnthropicTool(name="user_tool", input_schema={"type": "object"})

        assert tool.input_schema == {"type": "object"}
        assert tool.type is None

    def test_messages_request_with_mixed_tools(self):
        """MessagesRequest accepts mix of user tool and server-side tool."""
        req = MessagesRequest(
            model="claude-haiku-local",
            max_tokens=128,
            messages=[AnthropicMessage(role="user", content="hi")],
            tools=[
                AnthropicTool(name="get_weather", input_schema={"type": "object"}),
                AnthropicTool(type="web_search_20250305", name="web_search"),
            ],
        )

        assert len(req.tools) == 2
        assert req.tools[0].input_schema == {"type": "object"}
        assert req.tools[1].type == "web_search_20250305"

    def test_token_count_request_with_server_side_tool(self):
        """TokenCountRequest also accepts server-side tool definitions."""
        req = TokenCountRequest(
            model="claude-haiku-local",
            messages=[AnthropicMessage(role="user", content="hi")],
            tools=[AnthropicTool(type="web_search_20250305", name="web_search")],
        )

        assert len(req.tools) == 1
        assert req.tools[0].type == "web_search_20250305"


class TestToolChoice:
    """Tests for ToolChoice model."""

    def test_tool_choice_auto(self):
        """Test auto tool choice."""
        choice = ToolChoice(type="auto")

        assert choice.type == "auto"
        assert choice.name is None

    def test_tool_choice_any(self):
        """Test any tool choice."""
        choice = ToolChoice(type="any")

        assert choice.type == "any"

    def test_tool_choice_specific(self):
        """Test specific tool choice."""
        choice = ToolChoice(type="tool", name="get_weather")

        assert choice.type == "tool"
        assert choice.name == "get_weather"


class TestThinkingConfig:
    """Tests for ThinkingConfig model."""

    def test_thinking_enabled(self):
        """Test enabled thinking config."""
        config = ThinkingConfig(type="enabled", budget_tokens=10000)

        assert config.type == "enabled"
        assert config.budget_tokens == 10000

    def test_thinking_disabled(self):
        """Test disabled thinking config."""
        config = ThinkingConfig(type="disabled")

        assert config.type == "disabled"

    def test_thinking_adaptive(self):
        """Test adaptive thinking config."""
        config = ThinkingConfig(type="adaptive")

        assert config.type == "adaptive"

    def test_thinking_default_type(self):
        """Test thinking config defaults to enabled."""
        config = ThinkingConfig()

        assert config.type == "enabled"


class TestMessagesRequest:
    """Tests for MessagesRequest model."""

    def test_minimal_request(self):
        """Test creating minimal messages request."""
        req = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
        )

        assert req.model == "claude-3-sonnet"
        assert req.max_tokens == 1024
        assert len(req.messages) == 1
        assert req.stream is False

    def test_request_with_system(self):
        """Test request with system message."""
        req = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            system="You are a helpful assistant.",
        )

        assert req.system == "You are a helpful assistant."

    def test_request_with_system_content_list(self):
        """Test request with system content list."""
        req = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            system=[
                SystemContent(text="You are helpful."),
                SystemContent(text="Be concise."),
            ],
        )

        assert isinstance(req.system, list)
        assert len(req.system) == 2

    def test_request_with_tools(self):
        """Test request with tools."""
        req = MessagesRequest(
            model="claude-3-sonnet",
            max_tokens=1024,
            messages=[AnthropicMessage(role="user", content="Hello")],
            tools=[
                AnthropicTool(
                    name="get_weather",
                    input_schema={"type": "object"},
                )
            ],
        )

        assert len(req.tools) == 1

    def test_request_with_all_parameters(self):
        """Test request with all parameters."""
        req = MessagesRequest(
            model="claude-3-opus",
            max_tokens=4096,
            messages=[AnthropicMessage(role="user", content="Hello")],
            system="Be helpful",
            stop_sequences=["STOP"],
            stream=True,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            metadata={"user_id": "123"},
            tools=[AnthropicTool(name="test", input_schema={})],
            tool_choice=ToolChoice(type="auto"),
            thinking=ThinkingConfig(budget_tokens=5000),
        )

        assert req.model == "claude-3-opus"
        assert req.max_tokens == 4096
        assert req.stop_sequences == ["STOP"]
        assert req.stream is True
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.top_k == 40
        assert req.metadata == {"user_id": "123"}

    def test_request_max_tokens_required(self):
        """Test that max_tokens is required."""
        with pytest.raises(ValidationError):
            MessagesRequest(
                model="claude-3-sonnet",
                messages=[AnthropicMessage(role="user", content="Hello")],
            )

    def test_request_model_required(self):
        """Test that model is required."""
        with pytest.raises(ValidationError):
            MessagesRequest(
                max_tokens=1024,
                messages=[AnthropicMessage(role="user", content="Hello")],
            )

    def test_request_messages_required(self):
        """Test that messages is required."""
        with pytest.raises(ValidationError):
            MessagesRequest(
                model="claude-3-sonnet",
                max_tokens=1024,
            )


class TestTokenCounting:
    """Tests for token counting models."""

    def test_token_count_request(self):
        """Test creating token count request."""
        req = TokenCountRequest(
            model="claude-3-sonnet",
            messages=[AnthropicMessage(role="user", content="Hello")],
        )

        assert req.model == "claude-3-sonnet"

    def test_token_count_response(self):
        """Test creating token count response."""
        resp = TokenCountResponse(input_tokens=100)

        assert resp.input_tokens == 100


class TestMessagesResponse:
    """Tests for MessagesResponse model."""

    def test_basic_response(self):
        """Test creating basic messages response."""
        resp = MessagesResponse(
            model="claude-3-sonnet",
            content=[ContentBlockText(text="Hello!")],
            stop_reason="end_turn",
        )

        assert resp.type == "message"
        assert resp.role == "assistant"
        assert resp.model == "claude-3-sonnet"
        assert len(resp.content) == 1
        assert resp.stop_reason == "end_turn"

    def test_response_generates_id(self):
        """Test that response generates an ID."""
        resp = MessagesResponse(
            model="claude-3-sonnet",
            content=[ContentBlockText(text="Hi")],
        )

        assert resp.id is not None
        assert resp.id.startswith("msg_")

    def test_response_with_usage(self):
        """Test response with usage stats."""
        resp = MessagesResponse(
            model="claude-3-sonnet",
            content=[ContentBlockText(text="Hello")],
            usage=AnthropicUsage(
                input_tokens=10,
                output_tokens=5,
            ),
        )

        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5

    def test_response_with_tool_use(self):
        """Test response with tool use."""
        resp = MessagesResponse(
            model="claude-3-sonnet",
            content=[
                ContentBlockText(text="Let me check."),
                ContentBlockToolUse(
                    id="toolu_123",
                    name="get_weather",
                    input={"location": "Tokyo"},
                ),
            ],
            stop_reason="tool_use",
        )

        assert resp.stop_reason == "tool_use"
        assert len(resp.content) == 2

    def test_response_stop_reasons(self):
        """Test different stop reasons."""
        stop_reasons = ["end_turn", "max_tokens", "stop_sequence", "tool_use"]

        for reason in stop_reasons:
            resp = MessagesResponse(
                model="claude-3-sonnet",
                content=[ContentBlockText(text="Hi")],
                stop_reason=reason,
            )
            assert resp.stop_reason == reason


class TestAnthropicUsage:
    """Tests for AnthropicUsage model."""

    def test_usage_creation(self):
        """Test creating usage stats."""
        usage = AnthropicUsage(
            input_tokens=100,
            output_tokens=50,
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0

    def test_usage_with_cache_tokens(self):
        """Test usage with cache tokens."""
        usage = AnthropicUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=20,
            cache_read_input_tokens=30,
        )

        assert usage.cache_creation_input_tokens == 20
        assert usage.cache_read_input_tokens == 30


class TestStreamingEvents:
    """Tests for streaming event models."""

    def test_message_start_event(self):
        """Test message start event."""
        event = MessageStartEvent(
            message={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
            },
        )

        assert event.type == "message_start"
        assert event.message["id"] == "msg_123"

    def test_content_block_start_event(self):
        """Test content block start event."""
        event = ContentBlockStartEvent(
            index=0,
            content_block={"type": "text", "text": ""},
        )

        assert event.type == "content_block_start"
        assert event.index == 0

    def test_text_delta(self):
        """Test text delta."""
        delta = TextDelta(text="Hello")

        assert delta.type == "text_delta"
        assert delta.text == "Hello"

    def test_input_json_delta(self):
        """Test input JSON delta."""
        delta = InputJsonDelta(partial_json='{"location":')

        assert delta.type == "input_json_delta"
        assert delta.partial_json == '{"location":'

    def test_content_block_delta_event_text(self):
        """Test content block delta event with text."""
        event = ContentBlockDeltaEvent(
            index=0,
            delta=TextDelta(text="Hello"),
        )

        assert event.type == "content_block_delta"
        assert event.index == 0

    def test_content_block_delta_event_json(self):
        """Test content block delta event with JSON."""
        event = ContentBlockDeltaEvent(
            index=0,
            delta=InputJsonDelta(partial_json='{"key":'),
        )

        assert event.type == "content_block_delta"

    def test_content_block_stop_event(self):
        """Test content block stop event."""
        event = ContentBlockStopEvent(index=0)

        assert event.type == "content_block_stop"
        assert event.index == 0

    def test_message_delta_event(self):
        """Test message delta event."""
        event = MessageDeltaEvent(
            delta={"stop_reason": "end_turn", "stop_sequence": None},
            usage={"output_tokens": 10},
        )

        assert event.type == "message_delta"
        assert event.delta["stop_reason"] == "end_turn"
        assert event.usage["output_tokens"] == 10

    def test_message_stop_event(self):
        """Test message stop event."""
        event = MessageStopEvent()

        assert event.type == "message_stop"

    def test_ping_event(self):
        """Test ping event."""
        event = PingEvent()

        assert event.type == "ping"

    def test_error_event(self):
        """Test error event."""
        event = ErrorEvent(
            error={
                "type": "api_error",
                "message": "Something went wrong",
            },
        )

        assert event.type == "error"
        assert event.error["type"] == "api_error"


class TestErrorModels:
    """Tests for error response models."""

    def test_error_detail(self):
        """Test error detail."""
        detail = AnthropicErrorDetail(
            type="invalid_request_error",
            message="Invalid API key",
        )

        assert detail.type == "invalid_request_error"
        assert detail.message == "Invalid API key"

    def test_error_response(self):
        """Test error response."""
        resp = AnthropicErrorResponse(
            error=AnthropicErrorDetail(
                type="api_error",
                message="Server error",
            ),
        )

        assert resp.type == "error"
        assert resp.error.type == "api_error"
        assert resp.error.message == "Server error"
