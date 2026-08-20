# SPDX-License-Identifier: Apache-2.0
"""Tests for DS4 OpenAI chat-completions proxying."""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from omlx.api.anthropic_models import MessagesRequest as AnthropicMessagesRequest
from omlx.api.openai_models import ChatCompletionRequest, CompletionRequest, Message
from omlx.api.responses_models import ResponsesRequest
from omlx.engine.ds4 import (
    DS4ProcessEngine,
    DS4ProxyResponse,
    DS4StreamingProxyResponse,
)
from omlx.exceptions import ModelNotFoundError
from omlx.model_settings import ModelSettings
from omlx.server import ServerState, app
from omlx.server_metrics import get_server_metrics, reset_server_metrics
from omlx.settings import DS4_THINK_MAX_CONTEXT_TOKENS


class _SettingsManager:
    def __init__(self, settings: ModelSettings | None = None):
        self.settings = settings or ModelSettings()

    def get_settings(self, model_id: str) -> ModelSettings:
        return self.settings

    def get_all_settings(self) -> dict[str, ModelSettings]:
        return {"foo": self.settings}

    def get_exposed_profile_runtime_settings_for_request(self, model_id: str):
        return None


class _Pool:
    def __init__(self, engine: DS4ProcessEngine):
        self.engine = engine
        self.requested_model_ids: list[str] = []

    def resolve_model_id(self, model_id, settings_manager):
        if model_id in {"foo", "gpt-4o"}:
            return "foo"
        return model_id

    async def get_engine(self, model_id, **kwargs):
        self.requested_model_ids.append(model_id)
        if model_id != "foo":
            raise ModelNotFoundError(model_id, ["foo"])
        return self.engine

    async def release_engine(self, model_id):
        return None

    def get_entry(self, model_id):
        return None

    async def preload_pinned_models(self):
        return None

    async def check_ttl_expirations(self, *args, **kwargs):
        return []

    async def shutdown(self):
        return None


@dataclass
class _StreamingProxy:
    chunks: list[bytes]
    status_code: int = 200
    headers: dict[str, str] | None = None
    closed: bool = False

    def iter_bytes(self):
        yield from self.chunks

    def close(self):
        self.closed = True


class _RawResponseBody:
    def __init__(self, *, content: bytes = b"", chunks: list[bytes] | None = None):
        self.content = content
        self.chunks = chunks or []
        self._read_buffer = b"".join(self.chunks) if self.chunks else self.content
        self._read_offset = 0

    def read(self, amt=None, decode_content=True):
        if amt is None:
            data = self._read_buffer[self._read_offset :]
            self._read_offset = len(self._read_buffer)
            return data
        end = min(self._read_offset + amt, len(self._read_buffer))
        data = self._read_buffer[self._read_offset : end]
        self._read_offset = end
        return data

    def stream(self, amt=None, decode_content=True):
        yield from self.chunks


class _BufferingRawResponseBody(_RawResponseBody):
    def stream(self, amt=None, decode_content=True):
        if self.chunks:
            yield b"".join(self.chunks)
        elif self.content:
            yield self.content


class _RequestsResponse:
    def __init__(self, *, content: bytes = b"", chunks: list[bytes] | None = None):
        self.status_code = 200
        self.headers = {"Content-Type": "text/event-stream" if chunks else "application/json"}
        self.content = b"decoded-content-should-not-be-used"
        self.chunks = chunks or []
        self.raw = _RawResponseBody(content=content, chunks=chunks)
        self.closed = False

    def iter_content(self, chunk_size=None):
        yield from self.chunks

    def close(self):
        self.closed = True


class _FakeRequestsSession:
    instances: list[_FakeRequestsSession] = []
    next_response: _RequestsResponse | None = None
    post_started: threading.Event | None = None
    allow_post: threading.Event | None = None

    def __init__(self):
        self.trust_env = True
        self.closed = False
        self.calls: list[dict] = []
        self.trust_env_at_post: bool | None = None
        _FakeRequestsSession.instances.append(self)

    def post(self, url, *, json, stream, headers, timeout=None):
        self.trust_env_at_post = self.trust_env
        self.calls.append({"url": url, "json": json, "stream": stream, "headers": headers})
        if self.post_started is not None:
            self.post_started.set()
        if self.allow_post is not None:
            self.allow_post.wait(timeout=5.0)
        return self.next_response or _RequestsResponse(content=b"{}")

    def close(self):
        self.closed = True


class _FakeDS4Engine(DS4ProcessEngine):
    def __init__(self, tmp_path):
        gguf = tmp_path / "Foo.gguf"
        gguf.write_bytes(b"0" * 1000)
        super().__init__(model_id="foo", model_path=gguf, base_path=tmp_path)
        self.proxy_bodies: list[dict] = []
        self.stream_bodies: list[dict] = []
        self.completion_bodies: list[dict] = []
        self.completion_stream_bodies: list[dict] = []
        self.response_bodies: list[dict] = []
        self.response_stream_bodies: list[dict] = []
        self.anthropic_bodies: list[dict] = []
        self.anthropic_stream_bodies: list[dict] = []
        self.chat_response_body = b'{"ds4":true,"choices":[]}'
        self.completion_response_body = b'{"ds4_completion":true,"choices":[]}'
        self.responses_response_body = b'{"ds4_response":true,"output":[]}'
        self.anthropic_response_body = b'{"type":"message","content":[]}'
        self.min_context_requests: list[int] = []
        self.active_during_ensure_min_context: list[bool] = []
        self.active_during_chat_proxy: list[bool] = []
        self.chat_stream_chunks = [b"data: one\n\n", b"data: [DONE]\n\n"]
        self.completion_stream_chunks = [
            b"data: completion\n\n",
            b"data: [DONE]\n\n",
        ]
        self.responses_stream_chunks = [
            b"event: response.output_text.delta\n",
            b"data: {}\n\n",
        ]
        self.anthropic_stream_chunks = [
            b"event: content_block_delta\n",
            b"data: {}\n\n",
        ]

    async def ensure_min_context(self, min_tokens: int) -> bool:
        self.min_context_requests.append(min_tokens)
        self.active_during_ensure_min_context.append(self.has_active_requests())
        self.context_tokens = max(self.context_tokens or 0, min_tokens)
        return True

    async def proxy_chat_completion(self, body: dict):
        self.active_during_chat_proxy.append(self.has_active_requests())
        self.proxy_bodies.append(body)
        return DS4ProxyResponse(
            status_code=201,
            headers={"Content-Type": "application/json; charset=utf-8"},
            body=self.chat_response_body,
        )

    async def open_chat_completion_stream(self, body: dict):
        self.stream_bodies.append(body)
        return _StreamingProxy(
            chunks=self.chat_stream_chunks,
            headers={"Content-Type": "text/event-stream"},
        )

    async def proxy_completion(self, body: dict):
        self.completion_bodies.append(body)
        return DS4ProxyResponse(
            status_code=202,
            headers={"Content-Type": "application/json; charset=utf-8"},
            body=self.completion_response_body,
        )

    async def open_completion_stream(self, body: dict):
        self.completion_stream_bodies.append(body)
        return _StreamingProxy(
            chunks=self.completion_stream_chunks,
            headers={"Content-Type": "text/event-stream"},
        )

    async def proxy_response(self, body: dict):
        self.response_bodies.append(body)
        return DS4ProxyResponse(
            status_code=203,
            headers={"Content-Type": "application/json; charset=utf-8"},
            body=self.responses_response_body,
        )

    async def open_response_stream(self, body: dict):
        self.response_stream_bodies.append(body)
        return _StreamingProxy(
            chunks=self.responses_stream_chunks,
            headers={"Content-Type": "text/event-stream"},
        )

    async def proxy_anthropic_message(self, body: dict):
        self.anthropic_bodies.append(body)
        return DS4ProxyResponse(
            status_code=204,
            headers={"Content-Type": "application/json; charset=utf-8"},
            body=self.anthropic_response_body,
        )

    async def open_anthropic_message_stream(self, body: dict):
        self.anthropic_stream_bodies.append(body)
        return _StreamingProxy(
            chunks=self.anthropic_stream_chunks,
            headers={"Content-Type": "text/event-stream"},
        )


@contextmanager
def _client_with_engine(engine: _FakeDS4Engine, settings: ModelSettings | None = None):
    state = ServerState()
    state.engine_pool = _Pool(engine)
    state.settings_manager = _SettingsManager(settings)
    state.api_key = None
    with patch("omlx.server._server_state", state):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client, state.engine_pool


@pytest.mark.asyncio
async def test_ds4_process_engine_proxies_non_streaming_chat(monkeypatch, tmp_path):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = _RequestsResponse(content=b'{"ok":true}')
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    response = await engine.proxy_chat_completion({"model": "foo"})

    session = _FakeRequestsSession.instances[0]
    captured = session.calls[0]
    assert session.trust_env_at_post is False
    assert session.closed is True
    assert captured["url"] == "http://127.0.0.1:49152/v1/chat/completions"
    assert captured["json"] == {"model": "foo"}
    assert captured["headers"]["Accept-Encoding"] == "identity"
    assert captured["stream"] is True
    assert response.body == b'{"ok":true}'
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_ds4_process_engine_proxies_non_streaming_completion(monkeypatch, tmp_path):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = _RequestsResponse(content=b'{"text":"ok"}')
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    response = await engine.proxy_completion({"model": "foo", "prompt": "hello"})

    session = _FakeRequestsSession.instances[0]
    captured = session.calls[0]
    assert captured["url"] == "http://127.0.0.1:49152/v1/completions"
    assert captured["json"] == {"model": "foo", "prompt": "hello"}
    assert captured["stream"] is True
    assert response.body == b'{"text":"ok"}'
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_ds4_process_engine_proxies_non_streaming_response(monkeypatch, tmp_path):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = _RequestsResponse(content=b'{"output":[]}')
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    response = await engine.proxy_response({"model": "foo", "input": "hello"})

    session = _FakeRequestsSession.instances[0]
    captured = session.calls[0]
    assert captured["url"] == "http://127.0.0.1:49152/v1/responses"
    assert captured["json"] == {"model": "foo", "input": "hello"}
    assert captured["stream"] is True
    assert response.body == b'{"output":[]}'
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_ds4_process_engine_proxies_non_streaming_anthropic_message(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = _RequestsResponse(content=b'{"content":[]}')
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    response = await engine.proxy_anthropic_message(
        {"model": "foo", "messages": [], "max_tokens": 1}
    )

    session = _FakeRequestsSession.instances[0]
    captured = session.calls[0]
    assert captured["url"] == "http://127.0.0.1:49152/v1/messages"
    assert captured["json"] == {"model": "foo", "messages": [], "max_tokens": 1}
    assert captured["stream"] is True
    assert response.body == b'{"content":[]}'
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_ds4_process_engine_stream_tracks_active_until_consumed(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    backend_response = _RequestsResponse(chunks=[b"data: one\n\n", b"data: [DONE]\n\n"])
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = backend_response
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    response = await engine.open_chat_completion_stream({"model": "foo"})
    assert engine.has_active_requests() is True
    assert list(response.iter_bytes()) == [b"data: one\n\n", b"data: [DONE]\n\n"]
    session = _FakeRequestsSession.instances[0]
    assert session.trust_env_at_post is False
    assert session.calls[0]["stream"] is True
    assert backend_response.closed is True
    assert session.closed is True
    assert engine.has_active_requests() is False


def test_ds4_streaming_proxy_flushes_buffered_raw_sse_events():
    """SSE events are yielded individually even if urllib3 stream would buffer."""
    chunks = [b"data: one\n\n", b"data: two\n\n"]
    response = _RequestsResponse(chunks=chunks)
    response.raw = _BufferingRawResponseBody(chunks=chunks)
    closed = False

    def close():
        nonlocal closed
        closed = True

    proxy = DS4StreamingProxyResponse(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        response=response,
        on_close=close,
    )

    assert list(proxy.iter_bytes()) == chunks
    assert closed is True
    assert response.closed is True


@pytest.mark.asyncio
async def test_ds4_process_engine_completion_stream_uses_completions_endpoint(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    backend_response = _RequestsResponse(chunks=[b"data: completion\n\n"])
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = backend_response
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    response = await engine.open_completion_stream({"model": "foo", "prompt": "hello"})

    assert list(response.iter_bytes()) == [b"data: completion\n\n"]
    assert _FakeRequestsSession.instances[0].calls[0]["url"] == (
        "http://127.0.0.1:49152/v1/completions"
    )
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_ds4_process_engine_response_stream_uses_responses_endpoint(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    backend_response = _RequestsResponse(chunks=[b"event: response.done\n\n"])
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = backend_response
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    response = await engine.open_response_stream({"model": "foo", "input": "hello"})

    assert list(response.iter_bytes()) == [b"event: response.done\n\n"]
    assert _FakeRequestsSession.instances[0].calls[0]["url"] == (
        "http://127.0.0.1:49152/v1/responses"
    )
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_ds4_process_engine_anthropic_stream_uses_messages_endpoint(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    backend_response = _RequestsResponse(chunks=[b"event: message_stop\n\n"])
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = backend_response
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    response = await engine.open_anthropic_message_stream(
        {"model": "foo", "messages": [], "max_tokens": 1}
    )

    assert list(response.iter_bytes()) == [b"event: message_stop\n\n"]
    assert _FakeRequestsSession.instances[0].calls[0]["url"] == (
        "http://127.0.0.1:49152/v1/messages"
    )
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_ds4_multiple_streams_remain_active_until_all_thread_closes(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    _FakeRequestsSession.instances = []
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    first_backend_response = _RequestsResponse(chunks=[b"data: first\n\n"])
    _FakeRequestsSession.next_response = first_backend_response
    first = await engine.open_chat_completion_stream({"model": "foo"})
    second_backend_response = _RequestsResponse(chunks=[b"data: second\n\n"])
    _FakeRequestsSession.next_response = second_backend_response
    second = await engine.open_chat_completion_stream({"model": "foo"})

    assert engine.has_active_requests() is True
    await asyncio.to_thread(first.close)
    assert engine.has_active_requests() is True
    await asyncio.to_thread(second.close)
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_ds4_stream_response_close_releases_active_without_iteration(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    backend_response = _RequestsResponse(chunks=[b"data: never-read\n\n"])
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = backend_response
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    response = await engine.open_chat_completion_stream({"model": "foo"})
    assert engine.has_active_requests() is True
    response.close()

    assert list(response.iter_bytes()) == []
    assert backend_response.closed is True
    assert _FakeRequestsSession.instances[0].closed is True
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_ds4_non_streaming_cancellation_keeps_active_until_thread_finishes(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    read_started = threading.Event()
    allow_read = threading.Event()
    backend_response = _RequestsResponse(content=b'{"ok":true}')

    def blocking_read(decode_content=True):
        read_started.set()
        allow_read.wait(timeout=5.0)
        return b'{"ok":true}'

    backend_response.raw.read = blocking_read
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = backend_response
    _FakeRequestsSession.post_started = None
    _FakeRequestsSession.allow_post = None
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    task = asyncio.create_task(engine.proxy_chat_completion({"model": "foo"}))
    assert await asyncio.to_thread(read_started.wait, 2.0)
    assert engine.has_active_requests() is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.has_active_requests() is True

    allow_read.set()
    for _ in range(50):
        if not engine.has_active_requests():
            break
        await asyncio.sleep(0.02)

    assert engine.has_active_requests() is False
    assert backend_response.closed is True
    assert _FakeRequestsSession.instances[0].closed is True


@pytest.mark.asyncio
async def test_ds4_stream_open_cancellation_cleans_up_unclaimed_response(
    monkeypatch, tmp_path
):
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=tmp_path / "Foo.gguf",
        base_path=tmp_path,
    )
    engine.process = SimpleNamespace(
        is_running=True,
        port=49152,
        process=SimpleNamespace(pid=123),
        command=[],
        recent_log_text=lambda: "",
    )
    backend_response = _RequestsResponse(chunks=[b"data: later\n\n"])
    post_started = threading.Event()
    allow_post = threading.Event()
    _FakeRequestsSession.instances = []
    _FakeRequestsSession.next_response = backend_response
    _FakeRequestsSession.post_started = post_started
    _FakeRequestsSession.allow_post = allow_post
    monkeypatch.setattr("omlx.engine.ds4.requests.Session", _FakeRequestsSession)

    task = asyncio.create_task(engine.open_chat_completion_stream({"model": "foo"}))
    assert await asyncio.to_thread(post_started.wait, 2.0)
    assert engine.has_active_requests() is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert engine.has_active_requests() is True

    allow_post.set()
    for _ in range(50):
        if not engine.has_active_requests():
            break
        await asyncio.sleep(0.02)

    assert engine.has_active_requests() is False
    assert backend_response.closed is True
    assert _FakeRequestsSession.instances[0].closed is True
    _FakeRequestsSession.post_started = None
    _FakeRequestsSession.allow_post = None


def test_ds4_chat_non_streaming_proxies_raw_response_and_applies_defaults(tmp_path):
    engine = _FakeDS4Engine(tmp_path)
    settings = ModelSettings(
        temperature=0.25,
        top_p=0.5,
        top_k=7,
        repetition_penalty=1.2,
        presence_penalty=0.4,
        max_tokens=123,
        force_sampling=True,
    )

    with _client_with_engine(engine, settings) as (client, pool):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "foo",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.9,
                "top_p": 0.8,
            },
        )

    assert response.status_code == 201
    assert response.content == b'{"ds4":true,"choices":[]}'
    assert pool.requested_model_ids == ["foo"]
    body = engine.proxy_bodies[0]
    assert body["model"] == "foo"
    assert body["messages"] == [{"role": "user", "content": "hello", "partial": False}]
    assert body["temperature"] == 0.9
    assert body["top_p"] == 0.8
    assert body["top_k"] == 7
    assert body["max_tokens"] == 123
    assert "repetition_penalty" not in body
    assert "presence_penalty" not in body
    assert "frequency_penalty" not in body
    assert "xtc_probability" not in body
    assert "xtc_threshold" not in body


def test_ds4_explicit_max_chat_request_raises_backend_context(tmp_path):
    engine = _FakeDS4Engine(tmp_path)

    with _client_with_engine(engine) as (client, _pool):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "foo",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "max",
            },
        )

    assert response.status_code == 201
    assert engine.min_context_requests == [DS4_THINK_MAX_CONTEXT_TOKENS]
    assert engine.proxy_bodies[0]["reasoning_effort"] == "max"


def test_ds4_max_request_reserves_active_window_after_context_raise(tmp_path):
    engine = _FakeDS4Engine(tmp_path)

    with _client_with_engine(engine) as (client, _pool):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "foo",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "max",
            },
        )

    assert response.status_code == 201
    assert engine.active_during_ensure_min_context == [False]
    assert engine.active_during_chat_proxy == [True]
    assert engine.has_active_requests() is False


def test_ds4_usage_parser_handles_openai_responses_and_anthropic_shapes():
    from omlx.server import _ds4_usage_from_body

    assert _ds4_usage_from_body(
        b'{"usage":{"prompt_tokens":10,"completion_tokens":4,'
        b'"prompt_tokens_details":{"cached_tokens":3}}}'
    ) == (10, 4, 3)
    assert _ds4_usage_from_body(
        b'{"usage":{"input_tokens":12,"output_tokens":5,'
        b'"input_tokens_details":{"cached_tokens":2}}}'
    ) == (12, 5, 2)
    assert _ds4_usage_from_body(
        b'{"usage":{"input_tokens":8,"output_tokens":6,'
        b'"cache_read_input_tokens":4}}'
    ) == (8, 6, 4)
    assert _ds4_usage_from_body(
        b'{"usage":{"prompt_tokens":10,"completion_tokens":4,'
        b'"cached_tokens":9,"prompt_tokens_details":{"cached_tokens":3}}}'
    ) == (10, 4, 9)
    assert _ds4_usage_from_body(
        b'{"response":{"usage":{"input_tokens":13,"output_tokens":7,'
        b'"input_tokens_details":{"cached_tokens":5}}}}'
    ) == (13, 7, 5)
    assert _ds4_usage_from_body(
        b'{"message":{"usage":{"input_tokens":11,"output_tokens":0,'
        b'"cache_read_input_tokens":6}}}'
    ) == (11, 0, 6)
    assert _ds4_usage_from_body(b"not json") == (0, 0, 0)


def test_ds4_non_streaming_proxy_records_usage_metrics(tmp_path):
    reset_server_metrics()
    engine = _FakeDS4Engine(tmp_path)
    engine.chat_response_body = (
        b'{"ds4":true,"choices":[],"usage":{"prompt_tokens":10,'
        b'"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":3}}}'
    )

    try:
        with _client_with_engine(engine) as (client, _pool):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "foo",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert response.status_code == 201
        assert response.content == engine.chat_response_body
        snapshot = get_server_metrics().get_snapshot(model_id="foo")
        assert snapshot["total_requests"] == 1
        assert snapshot["total_prompt_tokens"] == 10
        assert snapshot["total_completion_tokens"] == 4
        assert snapshot["total_cached_tokens"] == 3
    finally:
        reset_server_metrics()


def test_ds4_streaming_proxy_tees_usage_metrics_without_changing_bytes(tmp_path):
    reset_server_metrics()
    engine = _FakeDS4Engine(tmp_path)
    engine.chat_stream_chunks = [
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":10,',
        b'"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":3}}}\n\n',
        b"data: [DONE]\n\n",
    ]
    expected_body = b"".join(engine.chat_stream_chunks)

    try:
        with _client_with_engine(engine) as (client, _pool):
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            ) as response:
                body = b"".join(response.iter_bytes())

        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"
        assert body == expected_body
        snapshot = get_server_metrics().get_snapshot(model_id="foo")
        assert snapshot["total_requests"] == 1
        assert snapshot["total_prompt_tokens"] == 10
        assert snapshot["total_completion_tokens"] == 4
        assert snapshot["total_cached_tokens"] == 3
    finally:
        reset_server_metrics()


def test_ds4_streaming_proxy_merges_anthropic_usage_events(tmp_path):
    reset_server_metrics()
    engine = _FakeDS4Engine(tmp_path)
    engine.anthropic_stream_chunks = [
        b'event: message_start\ndata: {"type":"message_start","message":',
        b'{"usage":{"input_tokens":11,"output_tokens":0,',
        b'"cache_read_input_tokens":6}}}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta",',
        b'"usage":{"output_tokens":5}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]
    expected_body = b"".join(engine.anthropic_stream_chunks)

    try:
        with _client_with_engine(engine) as (client, _pool):
            with client.stream(
                "POST",
                "/v1/messages",
                json={
                    "model": "foo",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            ) as response:
                body = b"".join(response.iter_bytes())

        assert response.status_code == 200
        assert body == expected_body
        snapshot = get_server_metrics().get_snapshot(model_id="foo")
        assert snapshot["total_requests"] == 1
        assert snapshot["total_prompt_tokens"] == 11
        assert snapshot["total_completion_tokens"] == 5
        assert snapshot["total_cached_tokens"] == 6
    finally:
        reset_server_metrics()


def test_ds4_completion_non_streaming_proxies_raw_response_and_applies_defaults(
    tmp_path,
):
    engine = _FakeDS4Engine(tmp_path)
    settings = ModelSettings(
        temperature=0.3,
        top_p=0.6,
        top_k=9,
        repetition_penalty=1.1,
        presence_penalty=0.2,
        max_tokens=77,
        force_sampling=True,
    )

    with _client_with_engine(engine, settings) as (client, pool):
        response = client.post(
            "/v1/completions",
            json={
                "model": "foo",
                "prompt": "complete this",
                "temperature": 0.95,
                "top_p": 0.85,
                "echo": True,
                "logprobs": 2,
                "suffix": " after",
                "best_of": 3,
                "logit_bias": {"42": -1},
                "reasoning_effort": "max",
            },
        )

    assert response.status_code == 202
    assert response.content == b'{"ds4_completion":true,"choices":[]}'
    assert pool.requested_model_ids == ["foo"]
    body = engine.completion_bodies[0]
    assert body["model"] == "foo"
    assert body["prompt"] == "complete this"
    assert body["echo"] is True
    assert body["logprobs"] == 2
    assert body["suffix"] == " after"
    assert body["best_of"] == 3
    assert body["logit_bias"] == {"42": -1}
    assert body["temperature"] == 0.95
    assert body["top_p"] == 0.85
    assert body["top_k"] == 9
    assert body["max_tokens"] == 77
    assert body["reasoning_effort"] == "max"
    assert engine.min_context_requests == [DS4_THINK_MAX_CONTEXT_TOKENS]
    assert "repetition_penalty" not in body
    assert "presence_penalty" not in body
    assert "frequency_penalty" not in body
    assert "xtc_probability" not in body
    assert "xtc_threshold" not in body


def test_ds4_completion_streaming_preserves_backend_sse_bytes(tmp_path):
    engine = _FakeDS4Engine(tmp_path)

    with _client_with_engine(engine) as (client, _pool):
        with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": "foo",
                "prompt": "complete this",
                "reasoning_effort": "high",
                "stream": True,
            },
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body == b"data: completion\n\ndata: [DONE]\n\n"
    assert engine.completion_stream_bodies[0]["model"] == "foo"
    assert engine.completion_stream_bodies[0]["reasoning_effort"] == "high"
    assert engine.completion_stream_bodies[0]["stream"] is True


def test_ds4_responses_non_streaming_proxies_raw_response_and_applies_defaults(
    tmp_path,
):
    engine = _FakeDS4Engine(tmp_path)
    settings = ModelSettings(
        temperature=0.35,
        top_p=0.65,
        max_tokens=88,
        force_sampling=True,
    )

    with _client_with_engine(engine, settings) as (client, pool):
        response = client.post(
            "/v1/responses",
            json={
                "model": "foo",
                "input": "answer this",
                "temperature": 0.97,
                "reasoning": {"summary": "auto", "effort": "max"},
                "max_tokens": 42,
                "parallel_tool_calls": True,
                "ds4_extension": {"passthrough": True},
            },
        )

    assert response.status_code == 203
    assert response.content == b'{"ds4_response":true,"output":[]}'
    assert pool.requested_model_ids == ["foo"]
    body = engine.response_bodies[0]
    assert body["model"] == "foo"
    assert body["input"] == "answer this"
    assert body["temperature"] == 0.97
    assert body["top_p"] == 0.65
    assert body["max_tokens"] == 42
    assert body["max_output_tokens"] == 88
    assert body["reasoning"] == {"summary": "auto", "effort": "max"}
    assert engine.min_context_requests == [DS4_THINK_MAX_CONTEXT_TOKENS]
    assert body["parallel_tool_calls"] is True
    assert body["ds4_extension"] == {"passthrough": True}


def test_ds4_responses_streaming_preserves_backend_sse_bytes(tmp_path):
    engine = _FakeDS4Engine(tmp_path)

    with _client_with_engine(engine) as (client, _pool):
        with client.stream(
            "POST",
            "/v1/responses",
            json={
                "model": "foo",
                "input": "answer this",
                "reasoning": {"effort": "high"},
                "stream": True,
            },
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body == b"event: response.output_text.delta\ndata: {}\n\n"
    assert engine.response_stream_bodies[0]["model"] == "foo"
    assert engine.response_stream_bodies[0]["reasoning"] == {"effort": "high"}
    assert engine.response_stream_bodies[0]["stream"] is True


def test_ds4_anthropic_non_streaming_proxies_raw_response_and_applies_defaults(
    tmp_path,
):
    engine = _FakeDS4Engine(tmp_path)
    settings = ModelSettings(
        temperature=0.45,
        top_p=0.75,
        top_k=11,
        max_tokens=66,
        force_sampling=True,
    )

    with _client_with_engine(engine, settings) as (client, pool):
        response = client.post(
            "/v1/messages",
            json={
                "model": "foo",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.91,
                "output_config": {"effort": "max"},
                "metadata": {"user_id": "u1"},
                "ds4_extension": {"passthrough": True},
            },
        )

    assert response.status_code == 204
    assert response.content == b'{"type":"message","content":[]}'
    assert pool.requested_model_ids == ["foo"]
    body = engine.anthropic_bodies[0]
    assert body["model"] == "foo"
    assert body["max_tokens"] == 66
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["temperature"] == 0.91
    assert body["top_p"] == 0.75
    assert body["top_k"] == 11
    assert body["output_config"] == {"effort": "max"}
    assert "reasoning_effort" not in body
    assert engine.min_context_requests == [DS4_THINK_MAX_CONTEXT_TOKENS]
    assert body["metadata"] == {"user_id": "u1"}
    assert body["ds4_extension"] == {"passthrough": True}


def test_ds4_anthropic_streaming_preserves_backend_sse_bytes(tmp_path):
    engine = _FakeDS4Engine(tmp_path)

    with _client_with_engine(engine) as (client, _pool):
        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "foo",
                "max_tokens": 12,
                "messages": [{"role": "user", "content": "hello"}],
                "output_config": {"effort": "high"},
                "stream": True,
            },
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body == b"event: content_block_delta\ndata: {}\n\n"
    assert engine.anthropic_stream_bodies[0]["model"] == "foo"
    assert engine.anthropic_stream_bodies[0]["output_config"] == {"effort": "high"}
    assert engine.anthropic_stream_bodies[0]["stream"] is True


@pytest.mark.asyncio
async def test_ds4_streaming_response_closes_proxy_when_send_start_fails():
    from omlx.server import _DS4StreamingResponse

    proxy = _StreamingProxy(
        chunks=[b"data: never-started\n\n"],
        headers={"Content-Type": "text/event-stream"},
    )
    response = _DS4StreamingResponse(proxy, media_type="text/event-stream")

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            raise RuntimeError("client disconnected before stream body")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
    }
    with pytest.raises(RuntimeError, match="client disconnected"):
        await response(scope, receive, send)

    assert proxy.closed is True


def test_ds4_chat_streaming_preserves_backend_sse_bytes(tmp_path):
    engine = _FakeDS4Engine(tmp_path)

    with _client_with_engine(engine) as (client, _pool):
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "foo",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "high",
                "stream": True,
            },
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body == b"data: one\n\ndata: [DONE]\n\n"
    assert engine.stream_bodies[0]["model"] == "foo"
    assert engine.stream_bodies[0]["reasoning_effort"] == "high"
    assert engine.stream_bodies[0]["stream"] is True


@pytest.mark.parametrize("removed_model_id", ["foo-" + "reasoner", "provider/foo"])
def test_removed_ds4_model_id_is_rejected_by_all_proxy_apis(
    tmp_path, removed_model_id
):
    engine = _FakeDS4Engine(tmp_path)
    requests = [
        (
            "/v1/chat/completions",
            {
                "model": removed_model_id,
                "messages": [{"role": "user", "content": "hello"}],
            },
        ),
        (
            "/v1/completions",
            {"model": removed_model_id, "prompt": "hello"},
        ),
        (
            "/v1/responses",
            {"model": removed_model_id, "input": "hello"},
        ),
        (
            "/v1/messages",
            {
                "model": removed_model_id,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hello"}],
            },
        ),
    ]

    with _client_with_engine(engine) as (client, _pool):
        responses = [client.post(endpoint, json=payload) for endpoint, payload in requests]

    assert [response.status_code for response in responses] == [404, 404, 404, 404]
    for response in responses[:3]:
        error = response.json()["error"]
        assert error["type"] == "not_found_error"
        assert error["param"] == "model"
        assert error["code"] == "model_not_found"
    assert responses[3].json() == {
        "type": "error",
        "error": {
            "type": "not_found_error",
            "message": (
                f"Model '{removed_model_id}' not found. Available models: foo"
            ),
        },
    }
    assert engine.proxy_bodies == []
    assert engine.completion_bodies == []
    assert engine.response_bodies == []
    assert engine.anthropic_bodies == []


def test_ds4_non_thinking_controls_are_forwarded_explicitly(tmp_path):
    engine = _FakeDS4Engine(tmp_path)

    with _client_with_engine(engine) as (client, _pool):
        client.post(
            "/v1/chat/completions",
            json={
                "model": "foo",
                "messages": [{"role": "user", "content": "hello"}],
                "reasoning_effort": "none",
            },
        )
        client.post(
            "/v1/completions",
            json={"model": "foo", "prompt": "hello", "reasoning_effort": "none"},
        )
        client.post(
            "/v1/responses",
            json={"model": "foo", "input": "hello", "reasoning": {"effort": "none"}},
        )
        client.post(
            "/v1/messages",
            json={
                "model": "foo",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hello"}],
                "thinking": {"type": "disabled"},
            },
        )

    assert engine.proxy_bodies[0]["reasoning_effort"] == "none"
    assert engine.completion_bodies[0]["reasoning_effort"] == "none"
    assert engine.response_bodies[0]["reasoning"] == {"effort": "none"}
    assert engine.anthropic_bodies[0]["thinking"] == {"type": "disabled"}
    assert engine.min_context_requests == []


def test_ds4_proxy_body_preserves_openai_schema_alias(tmp_path):
    from omlx.server import _build_ds4_chat_proxy_body

    engine = _FakeDS4Engine(tmp_path)
    state = ServerState()
    state.engine_pool = _Pool(engine)
    state.settings_manager = _SettingsManager()
    request = ChatCompletionRequest(
        model="foo",
        messages=[Message(role="user", content="json please")],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "result",
                "schema": {"type": "object"},
            },
        },
    )

    with patch("omlx.server._server_state", state):
        body = _build_ds4_chat_proxy_body(request, "foo")

    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert "schema_" not in body["response_format"]["json_schema"]


def test_ds4_chat_proxy_uses_resolved_engine_id_and_explicit_effort(tmp_path):
    from omlx.server import _build_ds4_chat_proxy_body

    engine = _FakeDS4Engine(tmp_path)
    state = ServerState()
    state.engine_pool = _Pool(engine)
    state.settings_manager = _SettingsManager(ModelSettings(model_alias="gpt-4o"))
    request = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="hello")],
        reasoning_effort="max",
    )

    with patch("omlx.server._server_state", state):
        body = _build_ds4_chat_proxy_body(request, "foo")

    assert body["model"] == "foo"
    assert body["reasoning_effort"] == "max"


def test_ds4_completion_proxy_uses_resolved_engine_id_and_explicit_effort(tmp_path):
    from omlx.server import _build_ds4_completion_proxy_body

    engine = _FakeDS4Engine(tmp_path)
    state = ServerState()
    state.engine_pool = _Pool(engine)
    state.settings_manager = _SettingsManager(ModelSettings(model_alias="gpt-4o"))
    request = CompletionRequest(
        model="gpt-4o",
        prompt="complete this",
        reasoning_effort="max",
    )

    with patch("omlx.server._server_state", state):
        body = _build_ds4_completion_proxy_body(request, "foo")

    assert body["model"] == "foo"
    assert body["prompt"] == "complete this"
    assert body["reasoning_effort"] == "max"


def test_ds4_responses_proxy_uses_resolved_engine_id_and_explicit_effort(tmp_path):
    from omlx.server import _build_ds4_responses_proxy_body

    engine = _FakeDS4Engine(tmp_path)
    state = ServerState()
    state.engine_pool = _Pool(engine)
    state.settings_manager = _SettingsManager(ModelSettings(model_alias="gpt-4o"))
    request = ResponsesRequest(
        model="gpt-4o",
        input="answer this",
        reasoning={"summary": "detailed", "effort": "max"},
    )

    with patch("omlx.server._server_state", state):
        body = _build_ds4_responses_proxy_body(request, "foo")

    assert body["model"] == "foo"
    assert body["input"] == "answer this"
    assert body["reasoning"] == {"summary": "detailed", "effort": "max"}


def test_ds4_anthropic_proxy_uses_resolved_engine_id_and_explicit_effort(tmp_path):
    from omlx.server import _build_ds4_anthropic_proxy_body

    engine = _FakeDS4Engine(tmp_path)
    state = ServerState()
    state.engine_pool = _Pool(engine)
    state.settings_manager = _SettingsManager(ModelSettings(model_alias="gpt-4o"))
    request = AnthropicMessagesRequest(
        model="gpt-4o",
        max_tokens=12,
        messages=[{"role": "user", "content": "hello"}],
        output_config={"effort": "max"},
        ds4_extension={"passthrough": True},
    )

    with patch("omlx.server._server_state", state):
        body = _build_ds4_anthropic_proxy_body(request, "foo")

    assert body["model"] == "foo"
    assert body["max_tokens"] == 12
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["output_config"] == {"effort": "max"}
    assert "reasoning_effort" not in body
    assert body["ds4_extension"] == {"passthrough": True}
