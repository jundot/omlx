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
