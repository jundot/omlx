# SPDX-License-Identifier: Apache-2.0
"""
End-to-end streaming tests for oMLX server.

Tests streaming response formats for OpenAI and Anthropic APIs
using mock AsyncIterator without loading actual models.
"""

import json
import pytest
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


@dataclass
class MockGenerationOutput:
    """Mock generation output for streaming tests."""

    text: str = ""
    tokens: List[int] = field(default_factory=list)
    prompt_tokens: int = 10
    completion_tokens: int = 0
    finish_reason: Optional[str] = None
    new_text: str = ""
    finished: bool = False
    tool_calls: Optional[List[Dict[str, Any]]] = None
    cached_tokens: int = 0


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self.eos_token_id = 2

    def encode(self, text: str) -> List[int]:
        return [100 + i for i, _ in enumerate(text.split())]

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        return f"<decoded:{len(tokens)} tokens>"

    def apply_chat_template(
        self, messages: List[Dict], tokenize: bool = False, **kwargs
    ) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


class MockBaseEngine:
    """Mock LLM engine with streaming support for testing."""

    def __init__(self, model_name: str = "test-model"):
        self._model_name = model_name
        self._tokenizer = MockTokenizer()
        self._model_type = "llama"
        # Configurable streaming responses
        self._stream_outputs: List[MockGenerationOutput] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model_type(self) -> Optional[str]:
        return self._model_type

    @property
    def prefix_cache_enabled(self) -> bool:
        return False

    def set_stream_outputs(self, outputs: List[MockGenerationOutput]):
        """Set custom streaming outputs for testing."""
        self._stream_outputs = outputs

    async def generate(self, prompt: str, **kwargs) -> MockGenerationOutput:
        return MockGenerationOutput(
            text="Generated response.",
            completion_tokens=5,
            finish_reason="stop",
            finished=True,
        )

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[MockGenerationOutput]:
        if self._stream_outputs:
            for output in self._stream_outputs:
                yield output
        else:
            yield MockGenerationOutput(
                text="Hello",
                new_text="Hello",
                completion_tokens=1,
                finished=False,
            )
            yield MockGenerationOutput(
                text="Hello world",
                new_text=" world",
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
            )

    def count_chat_tokens(self, messages: List[Dict], tools=None, chat_template_kwargs=None) -> int:
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)
        return len(self._tokenizer.encode(prompt))

    async def chat(self, messages: List[Dict], **kwargs) -> MockGenerationOutput:
        return MockGenerationOutput(
            text="Chat response.",
            completion_tokens=5,
            finish_reason="stop",
            finished=True,
        )

    async def stream_chat(self, messages: List[Dict], **kwargs) -> AsyncIterator[MockGenerationOutput]:
        if self._stream_outputs:
            for output in self._stream_outputs:
                yield output
        else:
            yield MockGenerationOutput(
                text="Hi",
                new_text="Hi",
                completion_tokens=1,
                finished=False,
            )
            yield MockGenerationOutput(
                text="Hi there",
                new_text=" there",
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
            )


class MockEnginePool:
    """Mock engine pool for testing."""

    def __init__(self, engine: Optional[MockBaseEngine] = None):
        self._engine = engine or MockBaseEngine()
        self._models = [
            {"id": "test-model", "loaded": True, "pinned": False, "size": 1000000}
        ]

    @property
    def model_count(self) -> int:
        return len(self._models)

    @property
    def loaded_model_count(self) -> int:
        return 1

    @property
    def max_model_memory(self) -> int:
        return 32 * 1024 * 1024 * 1024

    @property
    def current_model_memory(self) -> int:
        return 1000000

    def resolve_model_id(self, model_id_or_alias, settings_manager=None):
        return model_id_or_alias

    def get_model_ids(self) -> List[str]:
        return [m["id"] for m in self._models]

    def get_status(self) -> Dict[str, Any]:
        return {"models": self._models}

    async def get_engine(self, model_id: str):
        return self._engine


def parse_sse_events(response_text: str) -> List[Dict]:
    """Parse SSE events from response text."""
    events = []
    for line in response_text.strip().split("\n"):
        if line.startswith("data: "):
            data = line[6:]  # Remove "data: " prefix
            if data == "[DONE]":
                events.append({"done": True})
            else:
                try:
                    events.append(json.loads(data))
                except json.JSONDecodeError:
                    pass
    return events


class TestOpenAIStreamingFormat:
    """Tests for OpenAI streaming response format (SSE)."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        return MockBaseEngine()

    @pytest.fixture
    def mock_engine_pool(self, mock_engine):
        """Create mock engine pool."""
        return MockEnginePool(mock_engine)

    @pytest.fixture
    def client(self, mock_engine_pool):
        """Create test client with mocked state."""
        from fastapi.testclient import TestClient
        from omlx.server import app, _server_state

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model

        _server_state.engine_pool = mock_engine_pool
        _server_state.default_model = "test-model"

        yield TestClient(app)

        _server_state.engine_pool = original_pool
        _server_state.default_model = original_default

    @pytest.mark.slow
    @pytest.mark.integration
    def test_chat_completion_streaming_format(self, client):
        """Test that streaming chat completion returns SSE format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    @pytest.mark.slow
    @pytest.mark.integration
    def test_chat_completion_streaming_events(self, client):
        """Test streaming events structure."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        events = parse_sse_events(response.text)

        # Should have at least one content event and a [DONE] event
        assert len(events) >= 2

        # Check that last event is DONE
        assert events[-1].get("done") is True

        # Check structure of first chunk
        first_chunk = events[0]
        assert "id" in first_chunk
        assert first_chunk["object"] == "chat.completion.chunk"
        assert "choices" in first_chunk

    @pytest.mark.slow
    @pytest.mark.integration
    def test_chat_completion_streaming_role_in_first_chunk(self, client):
        """Test that first chunk contains role."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Test"}],
                "stream": True,
            },
        )

        events = parse_sse_events(response.text)
        non_done_events = [e for e in events if not e.get("done")]

        if non_done_events:
            first_chunk = non_done_events[0]
            # First chunk should have role in delta
            assert "choices" in first_chunk
            delta = first_chunk["choices"][0].get("delta", {})
            assert delta.get("role") == "assistant"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_completion_streaming_format(self, client):
        """Test that streaming completion returns SSE format."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello",
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    @pytest.mark.slow
    @pytest.mark.integration
    def test_completion_streaming_events(self, client):
        """Test completion streaming events structure."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Once upon a time",
                "stream": True,
            },
        )

        events = parse_sse_events(response.text)

        # Should have events and end with DONE
        assert len(events) >= 2
        assert events[-1].get("done") is True

        # Check first content event structure
        first_chunk = events[0]
        assert first_chunk["object"] == "text_completion"
        assert "choices" in first_chunk
        assert "text" in first_chunk["choices"][0]


class TestAnthropicStreamingFormat:
    """Tests for Anthropic streaming response format (SSE events)."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        return MockBaseEngine()

    @pytest.fixture
    def mock_engine_pool(self, mock_engine):
        """Create mock engine pool."""
        return MockEnginePool(mock_engine)

    @pytest.fixture
    def client(self, mock_engine_pool):
        """Create test client with mocked state."""
        from fastapi.testclient import TestClient
        from omlx.server import app, _server_state

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model

        _server_state.engine_pool = mock_engine_pool
        _server_state.default_model = "test-model"

        yield TestClient(app)

        _server_state.engine_pool = original_pool
        _server_state.default_model = original_default

    @pytest.mark.slow
    @pytest.mark.integration
    def test_anthropic_streaming_format(self, client):
        """Test that Anthropic streaming returns SSE format."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    @pytest.mark.slow
    @pytest.mark.integration
    def test_anthropic_streaming_event_order(self, client):
        """Test Anthropic streaming event order."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        events = parse_sse_events(response.text)
        non_done_events = [e for e in events if not e.get("done")]

        if len(non_done_events) >= 3:
            # Check event type order
            event_types = [e.get("type") for e in non_done_events]

            # Should start with message_start
            assert "message_start" in event_types

            # Should have content_block_start
            assert "content_block_start" in event_types

            # Should end with message_stop
            assert "message_stop" in event_types

    @pytest.mark.slow
    @pytest.mark.integration
    def test_anthropic_message_start_event(self, client):
        """Test Anthropic message_start event structure."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Test"}],
                "stream": True,
            },
        )

        events = parse_sse_events(response.text)
        message_start = None
        for event in events:
            if event.get("type") == "message_start":
                message_start = event
                break

        assert message_start is not None
        assert "message" in message_start

    @pytest.mark.slow
    @pytest.mark.integration
    def test_anthropic_message_delta_event(self, client):
        """Test Anthropic message_delta event has stop_reason."""
        response = client.post(
            "/v1/messages",
            json={
                "model": "test-model",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Quick test"}],
                "stream": True,
            },
        )

        events = parse_sse_events(response.text)
        message_delta = None
        for event in events:
            if event.get("type") == "message_delta":
                message_delta = event
                break

        if message_delta:
            assert "delta" in message_delta
            assert "usage" in message_delta


class TestStreamingHelperFunctions:
    """Tests for streaming helper functions in server module."""

    @pytest.mark.asyncio
    async def test_stream_completion_yields_sse(self):
        """Test stream_completion yields SSE formatted strings."""
        from omlx.server import stream_completion
        from omlx.api.openai_models import CompletionRequest

        engine = MockBaseEngine()
        request = CompletionRequest(model="test-model", prompt="Hello", stream=True)

        events = []
        async for event in stream_completion(engine, "Hello", request):
            events.append(event)

        # Should have content events and DONE
        assert len(events) >= 2
        assert events[-1] == "data: [DONE]\n\n"

        # Check SSE format
        for event in events[:-1]:
            assert event.startswith("data: ")
            assert event.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_stream_completion_json_content(self):
        """Test stream_completion events contain valid JSON."""
        from omlx.server import stream_completion
        from omlx.api.openai_models import CompletionRequest

        engine = MockBaseEngine()
        request = CompletionRequest(model="test-model", prompt="Test", stream=True)

        async for event in stream_completion(engine, "Test", request):
            if event != "data: [DONE]\n\n":
                json_str = event[6:-2]  # Remove "data: " and "\n\n"
                data = json.loads(json_str)
                assert "id" in data
                assert "model" in data
                assert "choices" in data

    @pytest.mark.asyncio
    async def test_stream_chat_completion_yields_sse(self):
        """Test stream_chat_completion yields SSE formatted strings."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine, messages, request, max_tokens=256, temperature=0.7, top_p=0.9, top_k=40
        ):
            events.append(event)

        assert len(events) >= 2
        assert events[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    async def test_stream_chat_completion_first_chunk_has_role(self):
        """Test first streaming chunk has assistant role."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            stream=True,
        )

        first_event = None
        messages = [{"role": "user", "content": "Hello"}]
        async for event in stream_chat_completion(
            engine, messages, request, max_tokens=256, temperature=0.7, top_p=0.9, top_k=40
        ):
            if event != "data: [DONE]\n\n":
                first_event = event
                break

        assert first_event is not None
        json_str = first_event[6:-2]
        data = json.loads(json_str)
        assert data["choices"][0]["delta"].get("role") == "assistant"

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_tools_streams_content_incrementally(self):
        """Tool availability must not force full buffering of normal text deltas."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="Hello",
                new_text="Hello",
                completion_tokens=1,
                finished=False,
            ),
            MockGenerationOutput(
                text="Hello world",
                new_text=" world",
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]
        content_deltas = [
            payload["choices"][0].get("delta", {}).get("content")
            for payload in payloads
            if payload.get("choices")
        ]
        content_deltas = [delta for delta in content_deltas if delta]

        # With two streamed model outputs, we expect two incremental content chunks,
        # not one buffered chunk emitted only at completion.
        assert content_deltas == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_chat_completion_sanitizes_tool_call_markup_inside_reasoning(self):
        """Reasoning deltas should keep prose while suppressing tool-call markup."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="<think>Need to inspect first.",
                new_text="<think>Need to inspect first.",
                completion_tokens=1,
                finished=False,
            ),
            MockGenerationOutput(
                text=(
                    "<think>Need to inspect first."
                    '<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>'
                    "Then continue.</think>"
                ),
                new_text=(
                    '<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>'
                    "Then continue.</think>"
                ),
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        reasoning_deltas = []
        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            reasoning = delta.get("reasoning_content")
            if reasoning:
                reasoning_deltas.append(reasoning)
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_reasoning = "".join(reasoning_deltas)
        assert "<tool_call>" not in streamed_reasoning
        assert "</tool_call>" not in streamed_reasoning
        assert "Need to inspect first." in streamed_reasoning
        assert "Then continue." in streamed_reasoning
        assert content_deltas == []
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_call_deltas[0]["function"]["arguments"]) == {"city": "SF"}
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_sanitizes_fragmented_reasoning_tool_call_markup(self):
        """Fragmented tool-call tags inside reasoning should never leak to the client."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="<think>Need to inspect ",
                new_text="<think>Need to inspect ",
                completion_tokens=1,
                finished=False,
            ),
            MockGenerationOutput(
                text="<tool_",
                new_text="<tool_",
                completion_tokens=2,
                finished=False,
            ),
            MockGenerationOutput(
                text='call>{"name":"get_weather","arguments":{"city":"SF"}}',
                new_text='call>{"name":"get_weather","arguments":{"city":"SF"}}',
                completion_tokens=3,
                finished=False,
            ),
            MockGenerationOutput(
                text="</tool_call></think>",
                new_text="</tool_call></think>",
                completion_tokens=4,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        reasoning_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            reasoning = delta.get("reasoning_content")
            if reasoning:
                reasoning_deltas.append(reasoning)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_reasoning = "".join(reasoning_deltas)
        assert "<tool_" not in streamed_reasoning
        assert "<tool_call>" not in streamed_reasoning
        assert "</tool_call>" not in streamed_reasoning
        assert streamed_reasoning.strip() == "Need to inspect"
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_call_deltas[0]["function"]["arguments"]) == {"city": "SF"}
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_anthropic_messages_sanitizes_tool_call_markup_inside_thinking(self):
        """Anthropic thinking blocks should hide raw tool-call markup and emit tool_use."""
        from omlx.server import stream_anthropic_messages
        from omlx.api.anthropic_models import MessagesRequest

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="<think>Need to inspect first.",
                new_text="<think>Need to inspect first.",
                completion_tokens=1,
                finished=False,
            ),
            MockGenerationOutput(
                text=(
                    "<think>Need to inspect first."
                    '<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>'
                    "Then continue.</think>"
                ),
                new_text=(
                    '<tool_call>{"name":"get_weather","arguments":{"city":"SF"}}</tool_call>'
                    "Then continue.</think>"
                ),
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        anthropic_tools = [{
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }]
        internal_tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = MessagesRequest(
            model="test-model",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            tools=anthropic_tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_anthropic_messages(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=internal_tools,
        ):
            events.append(event)

        parsed_events = []
        for event in events:
            for line in event.split("\n"):
                if line.startswith("data: "):
                    try:
                        parsed_events.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass

        thinking_deltas = []
        tool_use_blocks = []
        stop_reasons = []
        for event in parsed_events:
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "thinking_delta":
                    thinking_deltas.append(delta["thinking"])
            elif event.get("type") == "content_block_start":
                content_block = event.get("content_block", {})
                if content_block.get("type") == "tool_use":
                    tool_use_blocks.append(content_block)
            elif event.get("type") == "message_delta":
                stop_reason = event.get("delta", {}).get("stop_reason")
                if stop_reason:
                    stop_reasons.append(stop_reason)

        streamed_thinking = "".join(thinking_deltas)
        assert "<tool_call>" not in streamed_thinking
        assert "</tool_call>" not in streamed_thinking
        assert "Need to inspect first." in streamed_thinking
        assert "Then continue." in streamed_thinking
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "get_weather"
        assert "tool_use" in stop_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_tools_and_tool_calls_keeps_prior_content(self):
        """A tool_call finish should end the turn, not suppress already-generated text."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="Let me check that for you.",
                new_text="Let me check that for you.",
                completion_tokens=1,
                finished=False,
            ),
            MockGenerationOutput(
                text="Let me check that for you.",
                new_text="",
                completion_tokens=1,
                finished=True,
                finish_reason="tool_calls",
                tool_calls=[{
                    "name": "get_weather",
                    "arguments": "{\"city\":\"SF\"}",
                }],
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]
        content_deltas = []
        saw_tool_call_delta = False
        finish_reasons = []

        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                saw_tool_call_delta = True
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        assert content_deltas == ["Let me check that for you."]
        assert saw_tool_call_delta is True
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_tools_parsed_from_text_does_not_leak_markup(self):
        """Parsed-from-text tool calls must not appear in streamed content deltas."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        # Non-Harmony path: no output.tool_calls; tool calls parsed from inline text.
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="Let me check",
                new_text="Let me check",
                completion_tokens=1,
                finished=False,
                finish_reason=None,
                tool_calls=None,
            ),
            MockGenerationOutput(
                text=(
                    "Let me check that for you."
                    "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}</tool_call>"
                ),
                new_text=(
                    " that for you."
                    "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}</tool_call>"
                ),
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        content_event_indexes = []
        tool_call_event_indexes = []
        for idx, payload in enumerate(payloads):
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
                content_event_indexes.append(idx)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
                tool_call_event_indexes.append(idx)
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert "<tool_call>" not in streamed_content
        assert "</tool_call>" not in streamed_content
        assert content_deltas == ["Let me check", " that for you."]
        assert streamed_content == "Let me check that for you."
        assert streamed_content.count("Let me check that for you.") == 1
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_call_deltas[0]["function"]["arguments"]) == {"city": "SF"}
        assert content_event_indexes
        assert tool_call_event_indexes
        assert max(content_event_indexes) < min(tool_call_event_indexes)
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_tools_parsed_from_split_text_does_not_leak_partial_markup(self):
        """Split <tool_call> markup across chunks must not leak raw fragments in content deltas."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        # Non-Harmony path with fragmented tool-call markup across chunks.
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="Let me check that for you.<tool_",
                new_text="Let me check that for you.<tool_",
                completion_tokens=1,
                finished=False,
                finish_reason=None,
                tool_calls=None,
            ),
            MockGenerationOutput(
                text=(
                    "Let me check that for you."
                    "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}</to"
                ),
                new_text=(
                    "call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}</to"
                ),
                completion_tokens=2,
                finished=False,
                finish_reason=None,
                tool_calls=None,
            ),
            MockGenerationOutput(
                text=(
                    "Let me check that for you."
                    "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}</tool_call>"
                ),
                new_text="ol_call>",
                completion_tokens=3,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        content_event_indexes = []
        tool_call_event_indexes = []
        for idx, payload in enumerate(payloads):
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
                content_event_indexes.append(idx)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
                tool_call_event_indexes.append(idx)
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert "<tool_" not in streamed_content
        assert "</to" not in streamed_content
        assert "<tool_call>" not in streamed_content
        assert "</tool_call>" not in streamed_content
        assert content_deltas == ["Let me check that for you."]
        assert streamed_content == "Let me check that for you."
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_call_deltas[0]["function"]["arguments"]) == {"city": "SF"}
        assert content_event_indexes
        assert tool_call_event_indexes
        assert max(content_event_indexes) < min(tool_call_event_indexes)
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_tools_parsed_from_namespaced_text_does_not_leak_markup(self):
        """Supported namespaced tags must not leak into streamed content deltas."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="Let me check ",
                new_text="Let me check ",
                completion_tokens=1,
                finished=False,
                finish_reason=None,
                tool_calls=None,
            ),
            MockGenerationOutput(
                text=(
                    "Let me check "
                    "that for you."
                    "<minimax:tool_call>"
                    "<invoke name=\"get_weather\">"
                    "<parameter name=\"city\">\"SF\"</parameter>"
                    "</invoke>"
                    "</minimax:tool_call>"
                ),
                new_text=(
                    "that for you."
                    "<minimax:tool_call>"
                    "<invoke name=\"get_weather\">"
                    "<parameter name=\"city\">\"SF\"</parameter>"
                    "</invoke>"
                    "</minimax:tool_call>"
                ),
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        content_event_indexes = []
        tool_call_event_indexes = []
        for idx, payload in enumerate(payloads):
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
                content_event_indexes.append(idx)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
                tool_call_event_indexes.append(idx)
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert "<minimax:tool_call>" not in streamed_content
        assert "</minimax:tool_call>" not in streamed_content
        assert "<invoke" not in streamed_content
        assert "<parameter" not in streamed_content
        assert content_deltas == ["Let me check ", "that for you."]
        assert streamed_content == "Let me check that for you."
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_call_deltas[0]["function"]["arguments"]) == {"city": "SF"}
        assert content_event_indexes
        assert tool_call_event_indexes
        assert max(content_event_indexes) < min(tool_call_event_indexes)
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_tools_parsed_from_tokenizer_delimiters_does_not_leak_markup(self):
        """Split tokenizer delimiters must not leak into streamed content deltas."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.tokenizer.has_tool_calling = True
        engine.tokenizer.tool_call_start = "<|tool|>"
        engine.tokenizer.tool_call_end = "<|/tool|>"

        def tool_parser(payload: str, _tools):
            parsed = json.loads(payload)
            return {"name": parsed["name"], "arguments": parsed["arguments"]}

        engine.tokenizer.tool_parser = tool_parser
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="Let me check that for you.<|to",
                new_text="Let me check that for you.<|to",
                completion_tokens=1,
                finished=False,
                finish_reason=None,
                tool_calls=None,
            ),
            MockGenerationOutput(
                text=(
                    "Let me check that for you."
                    "<|tool|>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}<|/to"
                ),
                new_text=(
                    "ol|>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}<|/to"
                ),
                completion_tokens=2,
                finished=False,
                finish_reason=None,
                tool_calls=None,
            ),
            MockGenerationOutput(
                text=(
                    "Let me check that for you."
                    "<|tool|>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}<|/tool|>"
                ),
                new_text="ol|>",
                completion_tokens=3,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        content_event_indexes = []
        tool_call_event_indexes = []
        for idx, payload in enumerate(payloads):
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
                content_event_indexes.append(idx)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
                tool_call_event_indexes.append(idx)
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert "<|to" not in streamed_content
        assert "<|tool|>" not in streamed_content
        assert "<|/tool|>" not in streamed_content
        assert len(content_deltas) >= 1
        assert streamed_content == "Let me check that for you."
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_call_deltas[0]["function"]["arguments"]) == {"city": "SF"}
        assert content_event_indexes
        assert tool_call_event_indexes
        assert max(content_event_indexes) < min(tool_call_event_indexes)
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_suppresses_unmatched_tool_like_literal_suffix_under_clean_output_strict(self):
        """Clean-output strict contract suppresses unmatched tool-like suffixes."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="Use literal marker <tool_",
                new_text="Use literal marker <tool_",
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert "<tool_" not in streamed_content
        assert streamed_content == "Use literal marker "
        assert tool_call_deltas == []
        assert "stop" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_literal_bracket_marker_without_parse_is_preserved(self):
        """Literal bracket marker text should not be truncated when no tool call parses."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="Heads up: [Calling tool:",
                new_text="Heads up: [Calling tool:",
                completion_tokens=1,
                finished=False,
                finish_reason=None,
                tool_calls=None,
            ),
            MockGenerationOutput(
                text="Heads up: [Calling tool: maybe later]",
                new_text=" maybe later]",
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert streamed_content == "Heads up: [Calling tool: maybe later]"
        assert tool_call_deltas == []
        assert "stop" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_bracket_tool_call_parses_without_leak(self):
        """Valid bracket tool-call envelopes should not leak into streamed content."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="Let me check. [Calling tool:",
                new_text="Let me check. [Calling tool:",
                completion_tokens=1,
                finished=False,
                finish_reason=None,
                tool_calls=None,
            ),
            MockGenerationOutput(
                text=(
                    "Let me check. "
                    "[Calling tool: get_weather({\"city\":\"SF\"})]"
                ),
                new_text=' get_weather({"city":"SF"})]',
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert "[Calling tool:" not in streamed_content
        assert streamed_content == "Let me check. "
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_call_deltas[0]["function"]["arguments"]) == {"city": "SF"}
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_bracket_tool_call_then_visible_text_preserves_tail(self):
        """Tool envelope suppression must not truncate ordinary prose that follows it."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="Before [Calling tool:",
                new_text="Before [Calling tool:",
                completion_tokens=1,
                finished=False,
                finish_reason=None,
                tool_calls=None,
            ),
            MockGenerationOutput(
                text=(
                    "Before "
                    "[Calling tool: get_weather({\"city\":\"SF\"})]"
                    " After text"
                ),
                new_text=' get_weather({"city":"SF"})] After text',
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert "[Calling tool:" not in streamed_content
        assert streamed_content == "Before  After text"
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_call_deltas[0]["function"]["arguments"]) == {"city": "SF"}
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_bracket_hyphen_tool_name_parses_without_leak(self):
        """Bracket parser/filter should accept common hyphenated tool names."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="Let me check. [Calling tool:",
                new_text="Let me check. [Calling tool:",
                completion_tokens=1,
                finished=False,
                finish_reason=None,
                tool_calls=None,
            ),
            MockGenerationOutput(
                text=(
                    "Let me check. "
                    "[Calling tool: get-weather({\"city\":\"SF\"})]"
                ),
                new_text=' get-weather({"city":"SF"})]',
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get-weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert "[Calling tool:" not in streamed_content
        assert streamed_content == "Let me check. "
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["function"]["name"] == "get-weather"
        assert json.loads(tool_call_deltas[0]["function"]["arguments"]) == {"city": "SF"}
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_long_bracket_tool_call_does_not_leak_markup(self):
        """Long bracket envelopes should remain suppressed until complete."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        long_note = "x" * 320
        final_text = (
            "Before "
            f'[Calling tool: get_weather({{"note":"{long_note}"}})]'
            " After"
        )
        split_at = final_text.index("] After")

        engine.set_stream_outputs([
            MockGenerationOutput(
                text=final_text[:split_at],
                new_text=final_text[:split_at],
                completion_tokens=1,
                finished=False,
                finish_reason=None,
                tool_calls=None,
            ),
            MockGenerationOutput(
                text=final_text,
                new_text=final_text[split_at:],
                completion_tokens=2,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"note": {"type": "string"}},
                    "required": ["note"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert "[Calling tool:" not in streamed_content
        assert streamed_content == "Before  After"
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_call_deltas[0]["function"]["arguments"]) == {"note": long_note}
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_drops_unresolved_bracket_control_fragment_at_finish(self):
        """Unclosed bracket control fragments should not leak at stream end."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text='Before [Calling tool: get_weather({"city":"SF"}',
                new_text='Before [Calling tool: get_weather({"city":"SF"}',
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert "[Calling tool:" not in streamed_content
        assert streamed_content == "Before "
        assert tool_call_deltas == []
        assert "stop" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_preserves_literal_bracket_and_suppresses_later_parseable_envelope(self):
        """A literal early bracket marker must not block later valid bracket suppression."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        mixed = (
            "literal [Calling tool: maybe later] and then "
            '[Calling tool: get_weather({"city":"SF"})] done'
        )
        engine.set_stream_outputs([
            MockGenerationOutput(
                text=mixed,
                new_text=mixed,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert "literal [Calling tool: maybe later] and then " in streamed_content
        assert '[Calling tool: get_weather({"city":"SF"})]' not in streamed_content
        assert streamed_content == "literal [Calling tool: maybe later] and then  done"
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_call_deltas[0]["function"]["arguments"]) == {"city": "SF"}
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_sanitizes_unresolved_bracket_prefix_before_later_tool_call(self):
        """Unresolved early bracket prefixes should not leak even when a later bracket call parses."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        mixed = (
            "Before [Calling tool: unfinished and then "
            '[Calling tool: get_weather({"city":"NY"})] done'
        )
        engine.set_stream_outputs([
            MockGenerationOutput(
                text=mixed,
                new_text=mixed,
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert "[Calling tool:" not in streamed_content
        assert streamed_content == "Before  unfinished and then  done"
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_call_deltas[0]["function"]["arguments"]) == {"city": "NY"}
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_hyphen_namespaced_tool_call_parses_without_leak(self):
        """Hyphenated namespaced tool_call tags should parse into structured tool_calls."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text=(
                    "Let me check."
                    "<foo-bar:tool_call>"
                    "<invoke name=\"get_weather\">"
                    "<parameter name=\"city\">\"SF\"</parameter>"
                    "</invoke>"
                    "</foo-bar:tool_call>"
                ),
                new_text=(
                    "Let me check."
                    "<foo-bar:tool_call>"
                    "<invoke name=\"get_weather\">"
                    "<parameter name=\"city\">\"SF\"</parameter>"
                    "</invoke>"
                    "</foo-bar:tool_call>"
                ),
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert "<foo-bar:tool_call>" not in streamed_content
        assert "</foo-bar:tool_call>" not in streamed_content
        assert streamed_content == "Let me check."
        assert len(tool_call_deltas) == 1
        assert tool_call_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_call_deltas[0]["function"]["arguments"]) == {"city": "SF"}
        assert "tool_calls" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_preserves_non_tool_namespaced_like_suffix_literal(self):
        """Trailing namespaced-looking literals that are not tool_call tags should be preserved."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="Keep literal suffix <alpha:beta",
                new_text="Keep literal suffix <alpha:beta",
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert streamed_content == "Keep literal suffix <alpha:beta"
        assert tool_call_deltas == []
        assert "stop" in finish_reasons

    @pytest.mark.asyncio
    async def test_stream_chat_completion_preserves_non_tool_angle_identifier_suffix_literal(self):
        """Trailing '<identifier' literal should not be dropped as tool-control markup."""
        from omlx.server import stream_chat_completion
        from omlx.api.openai_models import ChatCompletionRequest, Message

        engine = MockBaseEngine()
        engine.set_stream_outputs([
            MockGenerationOutput(
                text="Use <alpha",
                new_text="Use <alpha",
                completion_tokens=1,
                finished=True,
                finish_reason="stop",
                tool_calls=None,
            ),
        ])

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hi")],
            stream=True,
            tools=tools,
        )

        events = []
        messages = [{"role": "user", "content": "Hi"}]
        async for event in stream_chat_completion(
            engine,
            messages,
            request,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            tools=tools,
        ):
            events.append(event)

        payloads = [
            json.loads(event[6:-2])
            for event in events
            if event.startswith("data: {")
        ]

        content_deltas = []
        tool_call_deltas = []
        finish_reasons = []
        for payload in payloads:
            choices = payload.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                content_deltas.append(content)
            if delta.get("tool_calls"):
                tool_call_deltas.extend(delta["tool_calls"])
            finish_reason = choice.get("finish_reason")
            if finish_reason:
                finish_reasons.append(finish_reason)

        streamed_content = "".join(content_deltas)
        assert streamed_content == "Use <alpha"
        assert tool_call_deltas == []
        assert "stop" in finish_reasons

class TestStreamingEdgeCases:
    """Tests for edge cases in streaming responses."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock engine."""
        return MockBaseEngine()

    @pytest.fixture
    def mock_engine_pool(self, mock_engine):
        """Create mock engine pool."""
        return MockEnginePool(mock_engine)

    @pytest.fixture
    def client(self, mock_engine_pool):
        """Create test client with mocked state."""
        from fastapi.testclient import TestClient
        from omlx.server import app, _server_state

        original_pool = _server_state.engine_pool
        original_default = _server_state.default_model

        _server_state.engine_pool = mock_engine_pool
        _server_state.default_model = "test-model"

        yield TestClient(app)

        _server_state.engine_pool = original_pool
        _server_state.default_model = original_default

    @pytest.mark.slow
    @pytest.mark.integration
    def test_streaming_with_empty_content(self, client, mock_engine):
        """Test streaming handles empty content chunks."""
        mock_engine.set_stream_outputs([
            MockGenerationOutput(text="", new_text="", finished=False),
            MockGenerationOutput(text="Hello", new_text="Hello", finished=False),
            MockGenerationOutput(
                text="Hello there",
                new_text=" there",
                finished=True,
                finish_reason="stop",
            ),
        ])

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Test"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        events = parse_sse_events(response.text)
        assert any(e.get("done") for e in events)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_streaming_finish_reason_propagation(self, client, mock_engine):
        """Test that finish_reason is propagated in streaming."""
        mock_engine.set_stream_outputs([
            MockGenerationOutput(text="Hi", new_text="Hi", finished=False),
            MockGenerationOutput(
                text="Hi!",
                new_text="!",
                finished=True,
                finish_reason="stop",
            ),
        ])

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        events = parse_sse_events(response.text)
        non_done_events = [e for e in events if not e.get("done")]

        # Find event with finish_reason
        finish_reasons = []
        for event in non_done_events:
            if "choices" in event:
                fr = event["choices"][0].get("finish_reason")
                if fr:
                    finish_reasons.append(fr)

        assert "stop" in finish_reasons

    @pytest.mark.slow
    @pytest.mark.integration
    def test_streaming_max_tokens_finish(self, client, mock_engine):
        """Test streaming with max_tokens finish reason."""
        mock_engine.set_stream_outputs([
            MockGenerationOutput(text="Long", new_text="Long", finished=False),
            MockGenerationOutput(
                text="Long text",
                new_text=" text",
                finished=True,
                finish_reason="length",
            ),
        ])

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Generate long text"}],
                "stream": True,
                "max_tokens": 5,
            },
        )

        events = parse_sse_events(response.text)
        non_done_events = [e for e in events if not e.get("done")]

        finish_reasons = []
        for event in non_done_events:
            if "choices" in event:
                fr = event["choices"][0].get("finish_reason")
                if fr:
                    finish_reasons.append(fr)

        assert "length" in finish_reasons
