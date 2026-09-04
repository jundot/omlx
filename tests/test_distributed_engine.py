# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from omlx.cluster.deployment import ClusterDeployment, ClusterHost
from omlx.cluster.performance import execution_profile
from omlx.cluster.planner import PipelineAssignment
from omlx.cluster.strategy_benchmarks import configure_strategy_benchmark_store
from omlx.engine import distributed
from omlx.engine.distributed import (
    DistributedBatchedEngine,
    DistributedInferenceError,
)


def _deployment() -> ClusterDeployment:
    return ClusterDeployment(
        deployment_id="engine-test",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("peer", "peer.local", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("local", 0, 2, 4, 2, 0, 0, 4),
            PipelineAssignment("peer", 1, 0, 2, 2, 0, 0, 4),
        ),
        plan_hash="d" * 64,
    )


class _Tokenizer:
    @staticmethod
    def encode(text):
        return list(text.encode())

    @staticmethod
    def apply_chat_template(messages, **_kwargs):
        return "\n".join(str(message["content"]) for message in messages)


def _sse_response(*, json):
    import json as json_module

    choices = []
    for original in json["choices"]:
        choice = dict(original)
        if "message" in choice:
            choice["delta"] = choice.pop("message")
        choices.append(choice)
    events = [{"choices": choices}, {"choices": [], "usage": json.get("usage", {})}]
    content = (
        ": keepalive 1/1\n\n"
        + "".join(f"data: {json_module.dumps(event)}\n\n" for event in events)
        + "data: [DONE]\n\n"
    )
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=content
    )


def _ready_engine(handler) -> DistributedBatchedEngine:
    engine = DistributedBatchedEngine(_deployment())
    engine._loaded = True
    engine._tokenizer = _Tokenizer()
    engine._client = httpx.AsyncClient(
        base_url="http://127.0.0.1:1",
        transport=httpx.MockTransport(handler),
    )
    return engine


def _deepseek_v4_vision_config() -> dict:
    return {
        "model_type": "deepseek_v4",
        "vision_n_layers": 2,
        "vision_dim": 8,
        "vision_n_heads": 2,
        "vision_inter_dim": 16,
        "vision_patch_size": 2,
        "vision_rope_theta": 10_000.0,
        "vision_downsample_ratio": 2,
        "vision_max_n_token": 32,
        "vision_min_pixels": 16,
        "vision_max_wh_ratio": 8,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (_deepseek_v4_vision_config(), True),
        ({"model_type": "deepseek_v4"}, False),
        ({"model_type": "llama", "vision_config": None}, False),
    ],
)
async def test_multimodal_api_capability_requires_recognized_distributed_vision(
    monkeypatch,
    config,
    expected,
):
    import mlx_lm.utils as mlx_utils

    monkeypatch.setattr(mlx_utils, "_download", lambda *_args, **_kwargs: "/model")
    monkeypatch.setattr(mlx_utils, "load_config", lambda _path: config)
    monkeypatch.setattr(
        mlx_utils, "load_tokenizer", lambda *_args, **_kwargs: _Tokenizer()
    )
    engine = DistributedBatchedEngine(_deployment())
    monkeypatch.setattr(engine._supervisor, "start", lambda: None)
    monkeypatch.setattr(engine._supervisor, "stop", lambda: None)
    engine._supervisor.port = 32000

    assert engine.supports_multimodal_fallback is False
    await engine.start()
    try:
        assert engine.supports_multimodal_fallback is expected
    finally:
        await engine.stop()
    assert engine.supports_multimodal_fallback is False


def test_backend_chat_messages_serialize_native_tool_history_once():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather",
            "content": '{"temperature_c":18}',
        },
    ]

    prepared = DistributedBatchedEngine._backend_chat_messages(messages)

    assert prepared[0]["tool_calls"][0]["function"]["arguments"] == (
        '{"city": "Paris"}'
    )
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}
    assert prepared[1] == messages[1]


def test_distributed_vision_counts_only_text_message_parts(monkeypatch):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64"}},
                {"type": "text", "text": "describe this image"},
            ],
        }
    ]
    cleaned = [{"role": "user", "content": "describe this image"}]

    def extract(received):
        assert received is messages
        return cleaned, [object()], []

    monkeypatch.setattr("omlx.utils.image.extract_images_from_messages", extract)
    engine = DistributedBatchedEngine(_deployment())
    engine._tokenizer = _Tokenizer()
    engine._supports_multimodal_fallback = True

    count = engine.count_chat_tokens(messages)

    assert count == len("describe this image")
    assert isinstance(messages[0]["content"], list)


@pytest.mark.asyncio
async def test_private_rank_zero_client_has_finite_inactivity_timeouts():
    engine = DistributedBatchedEngine(_deployment(), request_read_timeout=12.5)
    client = engine._new_client("http://127.0.0.1:1")
    try:
        assert client.timeout.connect == 10.0
        assert client.timeout.read == 12.5
        assert client.timeout.write == 30.0
        assert client.timeout.pool == 10.0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_request_read_timeout_defaults_from_env_var(monkeypatch):
    monkeypatch.setenv("OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT", "600")
    engine = DistributedBatchedEngine(_deployment())
    client = engine._new_client("http://127.0.0.1:1")
    try:
        assert client.timeout.read == 600.0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_request_read_timeout_env_var_takes_backseat_to_explicit_arg(monkeypatch):
    monkeypatch.setenv("OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT", "600")
    engine = DistributedBatchedEngine(_deployment(), request_read_timeout=12.5)
    client = engine._new_client("http://127.0.0.1:1")
    try:
        assert client.timeout.read == 12.5
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_request_read_timeout_env_var_rejects_non_numeric(monkeypatch):
    monkeypatch.setenv("OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT", "not-a-number")
    with pytest.raises(ValueError, match="must be a number"):
        DistributedBatchedEngine(_deployment())


@pytest.mark.asyncio
async def test_request_read_timeout_rejects_non_finite_and_non_positive(monkeypatch):
    for bad in ("nan", "inf", "0", "-5"):
        monkeypatch.setenv("OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT", bad)
        with pytest.raises(ValueError, match="finite positive"):
            DistributedBatchedEngine(_deployment())

    monkeypatch.delenv("OMLX_DISTRIBUTED_REQUEST_READ_TIMEOUT")
    with pytest.raises(ValueError, match="finite positive"):
        DistributedBatchedEngine(_deployment(), request_read_timeout=float("nan"))
    with pytest.raises(ValueError, match="finite positive"):
        DistributedBatchedEngine(_deployment(), request_read_timeout=0.0)


def _stalled_engine():
    def handler(request):
        raise httpx.ReadTimeout("collective stalled", request=request)

    engine = _ready_engine(handler)
    status_calls = []

    def status():
        status_calls.append(True)
        return SimpleNamespace(
            returncode=None,
            failure_reason=None,
            phase="ready",
        )

    engine._supervisor.status = status
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)
    return engine, status_calls, stop_calls, engine._client


@pytest.mark.asyncio
async def test_distributed_generate_bounds_rank_zero_read_stalls():
    engine, status_calls, stop_calls, client = _stalled_engine()
    with pytest.raises(
        DistributedInferenceError,
        match="stream timed out.*no rank-zero data.*cluster was ready",
    ):
        await engine.generate("hello")

    assert len(status_calls) == 2, "availability must be rechecked after timeout"
    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_distributed_stream_bounds_rank_zero_read_stalls():
    engine, status_calls, stop_calls, client = _stalled_engine()
    with pytest.raises(
        DistributedInferenceError,
        match="stream timed out.*no rank-zero data.*cluster was ready",
    ):
        [output async for output in engine.stream_generate("hello")]

    assert len(status_calls) == 2, "availability must be rechecked after timeout"
    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_cancelled_distributed_request_stops_every_rank():
    started = asyncio.Event()

    async def handler(_request):
        started.set()
        await asyncio.Event().wait()

    engine = _ready_engine(handler)
    client = engine._client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)

    request = asyncio.create_task(engine.generate("hello"))
    await started.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False
    assert engine.has_active_requests() is False


class _PartialSSEStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield (
            b'data: {"choices":[{"text":"one","finish_reason":null}]}'
            b"\n\n"
        )
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_closing_partial_distributed_stream_stops_every_rank():
    def handler(_request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_PartialSSEStream(),
        )

    engine = _ready_engine(handler)
    client = engine._client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)
    stream = engine.stream_generate("hello")

    output = await anext(stream)
    assert output.new_text == "one"
    await stream.aclose()

    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_successful_distributed_stream_keeps_cluster_loaded():
    def handler(_request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"choices":[{"text":"ok","finish_reason":null}]}\n\n'
                'data: {"choices":[{"text":"","finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    engine = _ready_engine(handler)
    client = engine._client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)
    try:
        outputs = [output async for output in engine.stream_generate("hello")]
    finally:
        await client.aclose()

    assert "".join(output.new_text for output in outputs) == "ok"
    assert stop_calls == []
    assert engine._client is client
    assert engine._loaded is True


@pytest.mark.asyncio
async def test_fail_closed_engine_retains_metadata_and_preflight_restarts():
    engine = _ready_engine(lambda _request: httpx.Response(200))
    client = engine._client
    tokenizer = engine._tokenizer
    engine._model_type = "llama"
    engine._supports_multimodal_fallback = True
    engine._supervisor.stop = lambda: None

    await engine._teardown_failed_request(client, reason="test failure")

    assert engine._tokenizer is tokenizer
    assert engine._model_type == "llama"
    assert engine._supports_multimodal_fallback is True
    engine.start = AsyncMock()
    engine._require_healthy_cluster = AsyncMock()

    await engine.preflight_completion("hello")

    engine.start.assert_awaited_once_with()
    engine._require_healthy_cluster.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_failed_restart_preserves_retained_engine_metadata(monkeypatch):
    import mlx_lm.utils as mlx_utils

    engine = _ready_engine(lambda _request: httpx.Response(200))
    client = engine._client
    tokenizer = engine._tokenizer
    engine._model_type = "llama"
    engine._supports_multimodal_fallback = True
    engine._supervisor.stop = lambda: None
    await engine._teardown_failed_request(client, reason="test failure")

    monkeypatch.setattr(mlx_utils, "_download", lambda *_args, **_kwargs: "/model")
    monkeypatch.setattr(
        mlx_utils,
        "load_config",
        lambda _path: {"model_type": "replacement"},
    )
    def fail_start():
        raise RuntimeError("launch failed")

    # Reach the supervisor failure after loading replacement metadata, then
    # ensure that metadata is not published over the last pool-valid state.
    monkeypatch.setattr(mlx_utils, "load_tokenizer", lambda *_args, **_kwargs: object())
    engine._supervisor.start = fail_start

    with pytest.raises(RuntimeError, match="launch failed"):
        await engine.start()

    assert engine._tokenizer is tokenizer
    assert engine._model_type == "llama"
    assert engine._supports_multimodal_fallback is True


@pytest.mark.asyncio
async def test_cancelled_supervisor_start_finishes_and_stops_before_unlock(monkeypatch):
    import mlx_lm.utils as mlx_utils

    monkeypatch.setattr(mlx_utils, "_download", lambda *_args, **_kwargs: "/model")
    monkeypatch.setattr(mlx_utils, "load_config", lambda _path: {"model_type": "llama"})
    monkeypatch.setattr(
        mlx_utils, "load_tokenizer", lambda *_args, **_kwargs: _Tokenizer()
    )
    engine = DistributedBatchedEngine(_deployment())
    started = threading.Event()
    release = threading.Event()
    stop_calls = []

    def blocking_start():
        started.set()
        release.wait()

    engine._supervisor.start = blocking_start
    engine._supervisor.stop = lambda: stop_calls.append(True)
    task = asyncio.create_task(engine.start())
    assert await asyncio.to_thread(started.wait, 1)

    task.cancel()
    await asyncio.sleep(0)
    assert engine._lifecycle_lock.locked()
    assert task.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)

    assert stop_calls == [True]
    assert engine._lifecycle_lock.locked() is False
    assert engine._loaded is False
    assert engine._tokenizer is None


@pytest.mark.asyncio
async def test_cancelled_supervisor_stop_finishes_before_unlock():
    engine = _ready_engine(lambda _request: httpx.Response(200))
    started = threading.Event()
    release = threading.Event()

    def blocking_stop():
        started.set()
        release.wait()

    engine._supervisor.stop = blocking_stop
    task = asyncio.create_task(engine.stop())
    assert await asyncio.to_thread(started.wait, 1)

    task.cancel()
    await asyncio.sleep(0)
    assert engine._lifecycle_lock.locked()
    assert task.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)

    assert engine._lifecycle_lock.locked() is False
    assert engine._loaded is False
    assert engine._tokenizer is None


@pytest.mark.asyncio
async def test_cancelled_supervisor_operation_preserves_cancellation_on_failure():
    started = threading.Event()
    release = threading.Event()

    def failing_operation():
        started.set()
        release.wait(timeout=2)
        raise RuntimeError("teardown failed")

    task = asyncio.create_task(
        DistributedBatchedEngine._run_supervisor_operation(failing_operation)
    )
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await asyncio.wait_for(task, 1)
    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
async def test_repeated_stream_cancellation_drains_activity_during_teardown(chat):
    started = asyncio.Event()
    stopping = asyncio.Event()
    release = asyncio.Event()

    async def handler(_request):
        started.set()
        await asyncio.Event().wait()

    engine = _ready_engine(handler)
    client = engine._client

    async def stop_operation(_operation):
        stopping.set()
        await release.wait()

    engine._run_supervisor_operation = stop_operation

    async def consume():
        stream = (
            engine.stream_chat([{"role": "user", "content": "hello"}])
            if chat
            else engine.stream_generate("hello")
        )
        async for _output in stream:
            pass

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        await asyncio.wait_for(stopping.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert engine.has_active_requests() is False
        assert engine._lifecycle_lock.locked()
        assert engine._teardown_tasks
    finally:
        release.set()
        await asyncio.gather(*engine._teardown_tasks)
        await client.aclose()
    assert not engine._lifecycle_lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_cancelled_queued_vision_request_keeps_running_rank_work(chat, streaming):
    started = asyncio.Event()
    release = asyncio.Event()
    requests = []

    async def handler(request):
        requests.append(request)
        started.set()
        await release.wait()
        if json.loads(request.content)["stream"]:
            field = '"delta":{"content":"ok"}' if chat else '"text":"ok"'
            return httpx.Response(
                200,
                content=(
                    f'data: {{"choices":[{{{field},"finish_reason":"stop"}}]}}\n\n'
                    "data: [DONE]\n\n"
                ),
            )
        return _sse_response(
            json={
                "choices": [
                    {
                        "text": "ok",
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    engine = _ready_engine(handler)
    engine._supports_multimodal_fallback = True
    client = engine._client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)

    async def invoke():
        prompt = [{"role": "user", "content": "hello"}] if chat else "hello"
        if streaming:
            method = engine.stream_chat if chat else engine.stream_generate
            return [output async for output in method(prompt)]
        method = engine.chat if chat else engine.generate
        return await method(prompt)

    running = asyncio.create_task(invoke())
    queued = None
    try:
        await asyncio.wait_for(started.wait(), 1)
        queued = asyncio.create_task(invoke())
        await asyncio.sleep(0)
        assert engine._active_requests == 2
        assert len(requests) == 1, "queued work must not open a backend socket"
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert engine._active_requests == 1
        assert stop_calls == []
        assert engine._client is client
        release.set()
        await asyncio.wait_for(running, 1)
        assert not engine.has_active_requests()
        assert not engine._vision_request_lock.locked()
        assert stop_calls == []
    finally:
        release.set()
        await asyncio.gather(
            running, *([queued] if queued else []), return_exceptions=True
        )
        await client.aclose()


@pytest.mark.asyncio
async def test_queued_vision_request_cannot_use_replaced_deployment():
    engine = _ready_engine(lambda _request: httpx.Response(200))
    engine._supports_multimodal_fallback = True
    client = engine._client
    await engine._vision_request_lock.acquire()
    queued = asyncio.create_task(engine.generate("hello"))
    replacement = httpx.AsyncClient(base_url="http://127.0.0.1:2")
    try:
        await asyncio.sleep(0)
        assert engine._active_requests == 1
        engine._client = replacement
        engine._vision_request_lock.release()
        with pytest.raises(DistributedInferenceError, match="changed while.*queued"):
            await queued
        assert not engine.has_active_requests()
        assert not engine._vision_request_lock.locked()
        assert engine._client is replacement
        assert not replacement.is_closed
    finally:
        await client.aclose()
        await replacement.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True])
async def test_nonstream_request_uses_progress_to_outlive_read_timeout(chat):
    completed = asyncio.Event()
    seen = []

    async def serve(reader, writer):
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            seen.append(json.loads(await reader.readexactly(length)))
            writer.write(b"HTTP/1.0 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
            # Total request time exceeds the real HTTP client's inactivity
            # limit, but regular backend progress makes the request healthy.
            for _ in range(6):
                writer.write(b": keepalive 1/2\n\n")
                await writer.drain()
                await asyncio.sleep(0.05)
            choice = {"finish_reason": "stop"}
            choice.update({"message": {"content": "ok"}} if chat else {"text": "ok"})
            writer.write(
                _sse_response(
                    json={
                        "choices": [choice],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }
                ).content
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            completed.set()

    async with await asyncio.start_server(serve, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        engine = DistributedBatchedEngine(_deployment(), request_read_timeout=0.2)
        engine._loaded = True
        engine._tokenizer = _Tokenizer()
        engine._client = engine._new_client(f"http://127.0.0.1:{port}")
        client = engine._client
        stop_calls = []
        engine._supervisor.stop = lambda: stop_calls.append(True)
        try:
            call = (
                engine.chat([{"role": "user", "content": "hi"}])
                if chat
                else engine.generate("hi")
            )
            output = await asyncio.wait_for(call, 2)
            assert output.text == output.new_text == "ok"
            assert output.completion_tokens == 1
            assert seen[0]["stream"] is True
            assert stop_calls == []
            assert engine._loaded
        finally:
            await client.aclose()
            await asyncio.wait_for(completed.wait(), 1)


@pytest.mark.asyncio
async def test_concurrent_distributed_failures_stop_supervisor_once():
    engine = _ready_engine(lambda _request: httpx.Response(200))
    client = engine._client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)

    await asyncio.gather(
        engine._teardown_failed_request(client, reason="first failure"),
        engine._teardown_failed_request(client, reason="second failure"),
    )

    assert stop_calls == [True]
    assert engine._client is None
    assert engine._loaded is False


@pytest.mark.asyncio
async def test_stale_distributed_request_cannot_stop_replacement_job():
    engine = _ready_engine(lambda _request: httpx.Response(200))
    stale_client = engine._client
    replacement_client = httpx.AsyncClient(base_url="http://127.0.0.1:2")
    engine._client = replacement_client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)
    try:
        await engine._teardown_failed_request(stale_client, reason="stale failure")
        assert stop_calls == []
        assert engine._client is replacement_client
        assert engine._loaded is True
    finally:
        await stale_client.aclose()
        await replacement_client.aclose()


@pytest.mark.asyncio
async def test_abort_all_distributed_requests_hard_stops_cluster():
    engine = _ready_engine(lambda _request: httpx.Response(200))
    client = engine._client
    engine._active_requests = 3
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)

    aborted = await engine.abort_all_requests(reason="memory pressure")

    assert aborted == 3
    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False


def test_chat_payload_folds_thinking_budget_into_chat_template_kwargs():
    engine = DistributedBatchedEngine(_deployment())
    payload = engine._chat_payload(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs={
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "thinking_budget": 2048,
        },
    )
    assert payload["chat_template_kwargs"] == {
        "reasoning_effort": "low",
        "thinking_budget": 2048,
    }


def test_chat_payload_without_thinking_budget_leaves_template_kwargs_untouched():
    engine = DistributedBatchedEngine(_deployment())
    payload = engine._chat_payload(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs={"chat_template_kwargs": {"reasoning_effort": "low"}},
    )
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}


def test_completion_payload_folds_thinking_budget_into_chat_template_kwargs():
    engine = DistributedBatchedEngine(_deployment())
    payload = engine._completion_payload(
        prompt="hi",
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs={"thinking_budget": 512},
    )
    assert payload["chat_template_kwargs"] == {"thinking_budget": 512}


def test_payloads_forward_repetition_context_size_when_requested():
    engine = DistributedBatchedEngine(_deployment())
    kwargs = {"repetition_context_size": 128}
    chat = engine._chat_payload(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.1,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs=dict(kwargs),
    )
    completion = engine._completion_payload(
        prompt="hi",
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.1,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs=dict(kwargs),
    )
    assert chat["repetition_context_size"] == 128
    assert completion["repetition_context_size"] == 128


def test_payloads_omit_repetition_context_size_by_default():
    # The key must stay off the wire unless the client asked for it: ranks
    # running mlx-lm default the window to 20 tokens when it is absent.
    engine = DistributedBatchedEngine(_deployment())
    chat = engine._chat_payload(
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.1,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs={},
    )
    completion = engine._completion_payload(
        prompt="hi",
        max_tokens=64,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
        min_p=0.0,
        repetition_penalty=1.1,
        presence_penalty=0.0,
        stop=None,
        stream=False,
        kwargs={},
    )
    assert "repetition_context_size" not in chat
    assert "repetition_context_size" not in completion


def test_model_thinking_budget_is_supported_by_distributed_engine():
    engine = DistributedBatchedEngine(
        _deployment(),
        model_settings=SimpleNamespace(thinking_budget_enabled=True),
    )

    engine._validate_model_settings()


@pytest.mark.asyncio
async def test_distributed_generate_translates_backend_completion():
    def handler(request):
        body = json.loads(request.content)
        assert body["prompt"] == "Hello"
        assert body["stream"] is True
        return _sse_response(
            json={
                "choices": [{"text": " world", "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 1},
                },
            },
        )

    engine = _ready_engine(handler)
    try:
        output = await engine.generate("Hello", max_tokens=8)
    finally:
        await engine._client.aclose()

    assert output.text == " world"
    assert output.prompt_tokens == 1
    assert output.completion_tokens == 2
    assert output.cached_tokens == 1
    assert engine.has_active_requests() is False


@pytest.mark.asyncio
async def test_distributed_chat_preserves_rank_zero_tool_calls_and_reasoning():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]

    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["messages"] == [{"role": "user", "content": "Weather?"}]
        assert body["tools"] == tools
        assert body["stream"] is True
        return _sse_response(
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'll check.",
                            "reasoning": "A weather lookup is required.",
                            "tool_calls": [
                                {
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "Paris"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "prompt_tokens_details": {"cached_tokens": 3},
                },
            },
        )

    engine = _ready_engine(handler)
    try:
        output = await engine.chat(
            [{"role": "user", "content": "Weather?"}],
            tools=tools,
        )
    finally:
        await engine._client.aclose()

    assert output.text == ("<think>A weather lookup is required.</think>I'll check.")
    assert output.finish_reason == "tool_calls"
    assert output.tool_calls == [
        {
            "id": "call_weather",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
        }
    ]
    assert output.cached_tokens == 3


@pytest.mark.asyncio
async def test_distributed_stream_chat_preserves_structured_tool_calls():
    events = [
        {
            "choices": [
                {
                    "delta": {"role": "assistant", "reasoning": "Need lookup."},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Paris"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 5,
                "total_tokens": 17,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"

    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )

    engine = _ready_engine(handler)
    try:
        outputs = [
            output
            async for output in engine.stream_chat(
                [{"role": "user", "content": "Weather?"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )
        ]
    finally:
        await engine._client.aclose()

    assert outputs[0].new_text == "<think>Need lookup."
    assert outputs[-1].new_text == ""
    assert outputs[-1].text == "<think>Need lookup.</think>"
    assert outputs[-1].finish_reason == "tool_calls"
    assert outputs[-1].tool_calls == [
        {
            "id": "call_weather",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        }
    ]
    assert outputs[-1].prompt_tokens == 12
    assert outputs[-1].completion_tokens == 5
    assert outputs[-1].cached_tokens == 2


@pytest.mark.asyncio
async def test_distributed_stream_waits_for_usage_before_final_output():
    events = [
        {
            "choices": [
                {"text": "A", "finish_reason": None},
            ]
        },
        {
            "choices": [
                {"text": "B", "finish_reason": "length"},
            ]
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )

    engine = _ready_engine(handler)
    try:
        outputs = [output async for output in engine.stream_generate("test")]
    finally:
        await engine._client.aclose()

    assert [output.new_text for output in outputs] == ["A", "B"]
    assert outputs[0].finished is False
    assert outputs[0].completion_tokens == 1
    assert outputs[0].generated_at is not None
    assert outputs[0].generated_until == outputs[0].generated_at
    assert outputs[-1].finished is True
    assert outputs[-1].text == "AB"
    assert outputs[-1].finish_reason == "length"
    assert outputs[-1].prompt_tokens == 4
    assert outputs[-1].completion_tokens == 2
    assert outputs[-1].cached_tokens == 3
    assert outputs[-1].generated_at == outputs[0].generated_at


@pytest.mark.asyncio
async def test_stream_records_real_prefill_and_decode_for_automatic_choice(
    monkeypatch,
    tmp_path,
):
    from omlx.engine import distributed

    events = [
        {"choices": [{"text": "A", "finish_reason": None}]},
        {"choices": [{"text": "B", "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 32,
                "completion_tokens": 2,
                "total_tokens": 34,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"
    engine = _ready_engine(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )
    )
    store = configure_strategy_benchmark_store(tmp_path)
    ticks = iter((10.0, 12.0, 16.0))
    monkeypatch.setattr(
        distributed,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    try:
        [output async for output in engine.stream_generate("x" * 32)]
    finally:
        await engine._client.aclose()

    measurements = store.measurements(
        model="org/model",
        node_ids=("local", "peer"),
        backend="ring",
        target_context_tokens=1024,
    )
    assert measurements[1].prompt_tokens_per_second == 16.0
    assert measurements[1].decode_tokens_per_second == 0.25
    assert measurements[1].time_to_first_token_seconds == 2.0


@pytest.mark.asyncio
async def test_strategy_benchmark_buckets_total_context_but_rates_uncached_prefill(
    tmp_path, monkeypatch
):
    events = [
        {"choices": [{"text": "A", "finish_reason": None}]},
        {"choices": [{"text": "B", "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 8192,
                "completion_tokens": 2,
                "total_tokens": 8194,
                "prompt_tokens_details": {"cached_tokens": 7168},
            },
        },
    ]
    content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    content += "data: [DONE]\n\n"
    engine = _ready_engine(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=content,
        )
    )
    store = configure_strategy_benchmark_store(tmp_path)
    ticks = iter((10.0, 12.0, 16.0))
    monkeypatch.setattr(
        distributed,
        "time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    try:
        [output async for output in engine.stream_generate("x" * 8192)]
    finally:
        await engine._client.aclose()

    measurements = store.measurements(
        model="org/model",
        node_ids=("local", "peer"),
        backend="ring",
        target_context_tokens=8192,
    )
    assert measurements[1].context_tokens == 8192
    assert measurements[1].prompt_tokens_per_second == 512.0


@pytest.mark.asyncio
async def test_distributed_stream_rejects_malformed_usage():
    event = {
        "choices": [],
        "usage": {"prompt_tokens_details": "not-an-object"},
    }

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(event)}\n\n",
        )

    engine = _ready_engine(handler)
    client = engine._client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)
    with pytest.raises(
        DistributedInferenceError,
        match="invalid token details",
    ):
        [output async for output in engine.stream_generate("test")]

    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False


@pytest.mark.asyncio
async def test_distributed_engine_surfaces_bounded_backend_error():
    def handler(request):
        return httpx.Response(503, json={"error": "rank 1 failed"})

    engine = _ready_engine(handler)
    client = engine._client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)
    try:
        with pytest.raises(DistributedInferenceError, match="HTTP 503.*rank 1"):
            await engine.generate("hello")
    finally:
        await client.aclose()

    assert stop_calls == []
    assert engine._client is client
    assert engine._loaded is True


@pytest.mark.asyncio
async def test_distributed_stream_backend_error_keeps_cluster_loaded():
    engine = _ready_engine(
        lambda _request: httpx.Response(503, json={"error": "rank 1 busy"})
    )
    client = engine._client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)
    try:
        with pytest.raises(DistributedInferenceError, match="HTTP 503.*rank 1 busy"):
            [output async for output in engine.stream_generate("hello")]
    finally:
        await client.aclose()

    assert stop_calls == []
    assert engine._client is client
    assert engine._loaded is True


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
async def test_distributed_stream_early_eof_stops_every_rank(kind):
    if kind == "chat":
        event = {"choices": [{"delta": {"content": "one"}}]}
    else:
        event = {"choices": [{"text": "one", "finish_reason": None}]}

    engine = _ready_engine(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(event)}\n\n",
        )
    )
    client = engine._client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)

    stream = (
        engine.stream_chat([{"role": "user", "content": "hello"}])
        if kind == "chat"
        else engine.stream_generate("hello")
    )
    with pytest.raises(DistributedInferenceError, match=r"ended before \[DONE\]"):
        [output async for output in stream]

    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["completion", "chat"])
async def test_distributed_stream_done_without_terminal_event_stops_every_rank(kind):
    engine = _ready_engine(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="data: [DONE]\n\n",
        )
    )
    client = engine._client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)

    stream = (
        engine.stream_chat([{"role": "user", "content": "hello"}])
        if kind == "chat"
        else engine.stream_generate("hello")
    )
    with pytest.raises(DistributedInferenceError, match="before a terminal event"):
        [output async for output in stream]

    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False


@pytest.mark.asyncio
async def test_distributed_transport_error_surfaces_peer_failure_reason():
    def handler(request):
        raise httpx.RemoteProtocolError(
            "server disconnected",
            request=request,
        )

    engine = _ready_engine(handler)
    client = engine._client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)
    statuses = iter(
        (
            SimpleNamespace(returncode=None, failure_reason=None, phase="ready"),
            SimpleNamespace(
                returncode=1,
                failure_reason=(
                    "Studio stopped publishing its runtime heartbeat. "
                    "Check oMLX is running on that Mac."
                ),
                phase="failed",
                stderr_tail=(),
            ),
        )
    )
    engine._supervisor.status = lambda: next(statuses)
    with pytest.raises(
        DistributedInferenceError,
        match="Studio stopped publishing its runtime heartbeat",
    ):
        await engine.generate("hello")

    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False


@pytest.mark.asyncio
async def test_distributed_transport_error_reports_bounded_launcher_exit():
    def handler(request):
        raise httpx.RemoteProtocolError(
            "server disconnected",
            request=request,
        )

    engine = _ready_engine(handler)
    client = engine._client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)
    statuses = iter(
        (
            SimpleNamespace(returncode=None, failure_reason=None, phase="ready"),
            SimpleNamespace(
                returncode=1,
                failure_reason=None,
                phase="failed",
                stderr_tail=("rank 1 out of memory",),
            ),
        )
    )
    engine._supervisor.status = lambda: next(statuses)
    with pytest.raises(
        DistributedInferenceError,
        match="exited with code 1.*rank 1 out of memory",
    ):
        await engine.generate("hello")

    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False


@pytest.mark.asyncio
async def test_distributed_stream_transport_error_stops_every_rank():
    def handler(request):
        raise httpx.ConnectError("connection reset", request=request)

    engine = _ready_engine(handler)
    client = engine._client
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)
    engine._supervisor.status = lambda: SimpleNamespace(
        returncode=None,
        failure_reason=None,
        phase="ready",
        stderr_tail=(),
    )

    with pytest.raises(
        DistributedInferenceError,
        match="stream failed.*ConnectError",
    ):
        [output async for output in engine.stream_generate("hello")]

    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False


@pytest.mark.asyncio
async def test_cancel_during_transport_diagnosis_still_stops_every_rank():
    def handler(request):
        raise httpx.ConnectError("connection reset", request=request)

    engine = _ready_engine(handler)
    client = engine._client
    diagnosis_started = asyncio.Event()
    stop_calls = []
    engine._supervisor.stop = lambda: stop_calls.append(True)

    async def blocked_diagnosis(_exc, *, stream):
        assert stream is True
        diagnosis_started.set()
        await asyncio.Event().wait()

    engine._transport_failure_error = blocked_diagnosis
    task = asyncio.create_task(engine.generate("hello"))
    await diagnosis_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False


@pytest.mark.asyncio
async def test_distributed_engine_rejects_unimplemented_grammar():
    def handler(request):
        raise AssertionError("backend should not be called")

    engine = _ready_engine(handler)
    try:
        with pytest.raises(ValueError, match="guided grammar"):
            await engine.generate("hello", compiled_grammar=object())
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_experimental_token_only_output_rejects_seeded_single_request():
    deployment = replace(
        _deployment(),
        execution=replace(
            execution_profile("balanced"),
            sampling_rank_only=True,
        ),
    )
    engine = DistributedBatchedEngine(deployment)
    engine._loaded = True
    engine._tokenizer = _Tokenizer()
    engine._client = httpx.AsyncClient(
        base_url="http://127.0.0.1:1",
        transport=httpx.MockTransport(
            lambda request: pytest.fail("backend should not be called")
        ),
    )
    try:
        with pytest.raises(ValueError, match="sampling-rank-only"):
            await engine.generate("hello", seed=7)
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_distributed_preflight_rejects_features_before_stream_starts():
    engine = _ready_engine(lambda request: httpx.Response(500))
    try:
        # thinking_budget is now supported: it is forwarded to the rank inside
        # chat_template_kwargs instead of being rejected.
        with pytest.raises(ValueError, match="SpecPrefill"):
            await engine.preflight_chat(
                [{"role": "user", "content": "hello"}],
                specprefill=True,
            )
    finally:
        await engine._client.aclose()


# ---------------------------------------------------------------------------
# reasoning_effort fallback: the distributed engine cannot render the chat
# template itself (only rank-zero can), so an unsupported value must be
# retried against rank-zero's HTTP endpoint rather than caught locally the
# way the batched/vlm/dflash engines do.
# ---------------------------------------------------------------------------


def test_reasoning_effort_retry_payloads_maps_alias_first():
    from omlx.engine.distributed import _reasoning_effort_retry_payloads

    payload = {"chat_template_kwargs": {"reasoning_effort": "high"}}
    variants = _reasoning_effort_retry_payloads(
        payload, "Unexpected reasoning effort high. Supported types are xhigh."
    )
    assert len(variants) == 2
    assert variants[0]["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
    # Second tier drops the field entirely (template's own default).
    assert "reasoning_effort" not in variants[1].get("chat_template_kwargs", {})


def test_reasoning_effort_retry_payloads_drops_when_no_alias_helps():
    from omlx.engine.distributed import _reasoning_effort_retry_payloads

    # "xhigh" has no further fallback in _ALIAS_FALLBACKS beyond "max", but if
    # the alias candidate equals the normalized value there is nothing to
    # retry with as an alias -- only the drop tier applies. Use a value with a
    # real alias to prove the two-tier ordering, and a bogus value to prove
    # single-tier (drop only) when there's no useful candidate.
    payload = {"chat_template_kwargs": {"reasoning_effort": "not-a-real-level"}}
    variants = _reasoning_effort_retry_payloads(
        payload, "Unexpected reasoning effort not-a-real-level."
    )
    assert len(variants) == 1
    assert "reasoning_effort" not in variants[0].get("chat_template_kwargs", {})


def test_reasoning_effort_retry_payloads_ignores_unrelated_failures():
    from omlx.engine.distributed import _reasoning_effort_retry_payloads

    payload = {"chat_template_kwargs": {"reasoning_effort": "high"}}
    assert _reasoning_effort_retry_payloads(payload, "model not found") == []


def test_reasoning_effort_retry_payloads_ignores_when_not_requested():
    from omlx.engine.distributed import _reasoning_effort_retry_payloads

    payload = {"chat_template_kwargs": {}}
    assert (
        _reasoning_effort_retry_payloads(
            payload, "Unexpected reasoning effort high."
        )
        == []
    )


@pytest.mark.asyncio
async def test_distributed_chat_retries_unsupported_reasoning_effort():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        effort = body.get("chat_template_kwargs", {}).get("reasoning_effort")
        calls.append(effort)
        if effort == "high":
            return httpx.Response(
                404,
                json={
                    "error": "Unexpected reasoning effort high. Supported "
                    "types are xhigh (default), medium, and low."
                },
            )
        assert effort == "xhigh"
        return _sse_response(
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    engine = _ready_engine(handler)
    try:
        output = await engine.chat(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "high"},
        )
    finally:
        await engine._client.aclose()

    assert calls == ["high", "xhigh"]
    assert output.text == "ok"


@pytest.mark.asyncio
async def test_distributed_chat_tries_the_normalized_value_first():
    # Local engines normalize before the first render, so "High" succeeds
    # locally; the cluster path must land on the same value, not jump
    # straight to the alias tier.
    calls = []

    def handler(request):
        body = json.loads(request.content)
        effort = body.get("chat_template_kwargs", {}).get("reasoning_effort")
        calls.append(effort)
        if effort == "high":
            return _sse_response(
                json={
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )
        return httpx.Response(
            404,
            json={"error": "Unexpected reasoning effort High."},
        )

    engine = _ready_engine(handler)
    try:
        output = await engine.chat(
            [{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "High"},
        )
    finally:
        await engine._client.aclose()

    assert calls == ["High", "high"]
    assert output.text == "ok"


@pytest.mark.asyncio
async def test_distributed_generate_retries_unsupported_reasoning_effort():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        effort = body.get("chat_template_kwargs", {}).get("reasoning_effort")
        calls.append(effort)
        if effort == "minimal":
            return httpx.Response(
                404,
                json={"error": "Unexpected reasoning effort minimal."},
            )
        assert effort == "low"
        return _sse_response(
            json={
                "choices": [{"text": "ok", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    engine = _ready_engine(handler)
    try:
        output = await engine.generate(
            "hi", chat_template_kwargs={"reasoning_effort": "minimal"}
        )
    finally:
        await engine._client.aclose()

    assert calls == ["minimal", "low"]
    assert output.text == "ok"


@pytest.mark.asyncio
async def test_distributed_stream_chat_retries_unsupported_reasoning_effort():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        effort = body.get("chat_template_kwargs", {}).get("reasoning_effort")
        calls.append(effort)
        if effort == "high":
            return httpx.Response(
                404,
                json={"error": "Unexpected reasoning effort high."},
            )
        assert effort == "xhigh"
        lines = [
            'data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": null}]}',
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], '
            '"usage": {"prompt_tokens": 1, "completion_tokens": 1}}',
            "data: [DONE]",
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="\n".join(lines) + "\n",
        )

    engine = _ready_engine(handler)
    try:
        outputs = [
            output
            async for output in engine.stream_chat(
                [{"role": "user", "content": "hi"}],
                chat_template_kwargs={"reasoning_effort": "high"},
            )
        ]
    finally:
        await engine._client.aclose()

    assert calls == ["high", "xhigh"]
    assert "".join(o.new_text for o in outputs) == "ok"


@pytest.mark.asyncio
async def test_distributed_stream_generate_bounds_retries_and_gives_up():
    # Every attempt is rejected. "High" walks the full ladder — original,
    # normalized ("high"), alias ("xhigh"), dropped — exactly 4 requests,
    # then raise; never an unbounded loop.
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(
            404,
            json={"error": "Unexpected reasoning effort High."},
        )

    engine = _ready_engine(handler)
    try:
        with pytest.raises(DistributedInferenceError, match="HTTP 404"):
            async for _ in engine.stream_generate(
                "hi", chat_template_kwargs={"reasoning_effort": "High"}
            ):
                pass
    finally:
        await engine._client.aclose()

    assert len(calls) == 4


@pytest.mark.asyncio
async def test_distributed_chat_does_not_retry_unrelated_404():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404, json={"error": "model not found"})

    engine = _ready_engine(handler)
    try:
        with pytest.raises(DistributedInferenceError, match="model not found"):
            await engine.chat([{"role": "user", "content": "hi"}])
    finally:
        await engine._client.aclose()

    assert len(calls) == 1


def _healthy_supervisor_status():
    return SimpleNamespace(returncode=None, failure_reason=None)


@pytest.mark.asyncio
async def test_preflight_rejects_an_unhealthy_rank_before_streaming(monkeypatch):
    # The 200 commits before a streaming body runs, so preflight is the last
    # point a half-dead cluster can still become a clean HTTP error (#2708).
    engine = _ready_engine(lambda request: httpx.Response(200))
    client = engine._client
    stop_calls = []
    monkeypatch.setattr(engine._supervisor, "stop", lambda: stop_calls.append(True))
    monkeypatch.setattr(engine._supervisor, "status", _healthy_supervisor_status)
    monkeypatch.setattr(
        distributed,
        "check_peers",
        lambda hosts, **kwargs: (
            SimpleNamespace(healthy=True),
            SimpleNamespace(healthy=False),
        ),
    )
    monkeypatch.setattr(
        distributed,
        "describe_failure",
        lambda health: "rank 1 (peer) stopped heartbeating",
    )
    with pytest.raises(DistributedInferenceError, match="not serving"):
        await engine.preflight_chat([{"role": "user", "content": "hi"}])

    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False


@pytest.mark.asyncio
async def test_preflight_caches_the_peer_health_read(monkeypatch):
    engine = _ready_engine(lambda request: httpx.Response(200))
    monkeypatch.setattr(engine._supervisor, "status", _healthy_supervisor_status)
    calls = []

    def fake_check_peers(hosts, **kwargs):
        calls.append(hosts)
        return (SimpleNamespace(healthy=True),)

    monkeypatch.setattr(distributed, "check_peers", fake_check_peers)
    try:
        await engine.preflight_chat([{"role": "user", "content": "hi"}])
        await engine.preflight_completion("hi")
        assert len(calls) == 1  # second preflight served from the TTL cache
        assert calls[0] == {0: ("local", "127.0.0.1"), 1: ("peer", "peer.local")}
    finally:
        await engine._client.aclose()


@pytest.mark.asyncio
async def test_preflight_rejects_a_reported_failure_without_probing(monkeypatch):
    engine = _ready_engine(lambda request: httpx.Response(200))
    client = engine._client
    stop_calls = []
    monkeypatch.setattr(engine._supervisor, "stop", lambda: stop_calls.append(True))
    monkeypatch.setattr(
        engine._supervisor,
        "status",
        lambda: SimpleNamespace(
            returncode=None,
            failure_reason="rank 1 connection closed",
            stderr_tail=(),
        ),
    )
    probed = []
    monkeypatch.setattr(
        distributed, "check_peers", lambda *a, **k: probed.append(1) or ()
    )
    with pytest.raises(DistributedInferenceError, match="rank 1 connection"):
        await engine.preflight_chat([{"role": "user", "content": "hi"}])

    assert probed == []
    assert stop_calls == [True]
    assert client.is_closed
    assert engine._client is None
    assert engine._loaded is False


@pytest.mark.asyncio
async def test_preflight_fails_open_when_the_probe_itself_breaks(monkeypatch):
    # A broken probe must not take down a serving cluster; the supervisor
    # checks still catch hard failures.
    engine = _ready_engine(lambda request: httpx.Response(200))
    monkeypatch.setattr(engine._supervisor, "status", _healthy_supervisor_status)

    def broken_check_peers(hosts, **kwargs):
        raise OSError("ssh binary missing")

    monkeypatch.setattr(distributed, "check_peers", broken_check_peers)
    try:
        await engine.preflight_chat([{"role": "user", "content": "hi"}])
    finally:
        await engine._client.aclose()
