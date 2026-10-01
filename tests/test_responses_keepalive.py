# SPDX-License-Identifier: Apache-2.0
"""Responses route keepalives during a blocked model prefill (no model needed)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from omlx import server
from omlx.api.responses_models import ResponsesRequest
from omlx.engine.base import GenerationOutput
from omlx.settings import GlobalSettings


def _event(frame):
    return json.loads(frame.split("data: ", 1)[1])


@pytest.fixture
def responses_route(monkeypatch):
    ready = asyncio.Event()
    closed = asyncio.Event()
    started = asyncio.Event()
    requests = []
    finish = asyncio.Event()
    finish.set()
    output = GenerationOutput(
        text="Hello", new_text="Hello", prompt_tokens=10, completion_tokens=1
    )
    engine = SimpleNamespace(
        tokenizer=None,
        model_type="llama",
        start=AsyncMock(),
        count_chat_tokens=lambda *args, **kwargs: 10,
        preflight_chat=AsyncMock(),
        failure=None,
    )

    async def stream_chat(**kwargs):
        try:
            started.set()
            await ready.wait()
            if engine.failure:
                raise engine.failure
            yield output
            await finish.wait()
        finally:
            closed.set()

    engine.stream_chat = stream_chat
    monkeypatch.setattr(server, "get_engine_for_model", AsyncMock(return_value=engine))
    monkeypatch.setattr(server, "get_server_metrics", Mock(return_value=Mock()))
    monkeypatch.setattr(
        server._server_state,
        "engine_pool",
        SimpleNamespace(
            get_entry=lambda _: None, resolve_model_id=lambda model, settings: model
        ),
    )
    for name in ("settings_manager", "mcp_manager", "oq_manager"):
        monkeypatch.setattr(server._server_state, name, None)
    release = AsyncMock()
    monkeypatch.setattr(server._LLMEngineLease, "release", release)

    # Exercise the route's actual wrapper selection with short test intervals.
    wrap = server._with_sse_keepalive

    def fast_keepalive(generator, **kwargs):
        return wrap(generator, interval=0.01, disconnect_poll=0.01, **kwargs)

    monkeypatch.setattr(server, "_with_sse_keepalive", fast_keepalive)

    async def create(mode="chunk", **kwargs):
        settings = GlobalSettings()
        settings.server.sse_keepalive_mode = mode
        monkeypatch.setattr(
            server._server_state, "global_settings", settings if mode else None
        )
        request = ResponsesRequest(
            model="test-model", input="Hello", stream=True, store=False, **kwargs
        )
        http_request = SimpleNamespace(
            headers={}, scope={}, is_disconnected=AsyncMock(return_value=False)
        )
        requests.append(http_request)
        response = await server.create_response(request, http_request)
        return response.body_iterator

    return SimpleNamespace(
        create=create,
        ready=ready,
        closed=closed,
        output=output,
        engine=engine,
        release=release,
        finish=finish,
        started=started,
        requests=requests,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chunk", None])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
async def test_prefill_emits_response_events_before_model_output(
    responses_route, mode, finish_reason
):
    route = responses_route
    route.output.finish_reason = finish_reason
    stream = await route.create(mode)
    try:
        events = [_event(await anext(stream)), _event(await anext(stream))]
        assert [e["type"] for e in events] == [
            "response.created",
            "response.in_progress",
        ]
        # No model token can arrive until ready is set. Require multiple real
        # data events, rather than comments ignored by event-level idle timers.
        for _ in range(2):
            events.append(_event(await asyncio.wait_for(anext(stream), timeout=1)))
        assert not route.ready.is_set()
        initial = events[0]["response"]
        for event in events[1:]:
            assert event["type"] == "response.in_progress"
            assert event["response"] == initial
            assert event["response"]["output"] == []
        route.ready.set()
        events.extend([_event(frame) async for frame in stream])
    finally:
        await stream.aclose()

    assert [e["sequence_number"] for e in events] == list(range(1, len(events) + 1))
    terminal = events[-1]
    assert terminal["type"] == (
        "response.incomplete" if finish_reason == "length" else "response.completed"
    )
    assert terminal["response"]["id"] == initial["id"]
    assert terminal["response"]["model"] == "test-model"
    assert terminal["response"]["output"][0]["content"][0]["text"] == "Hello"
    assert terminal["response"]["usage"]["output_tokens"] == 1
    assert route.closed.is_set()
    route.release.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["comment", "off"])
async def test_responses_preserve_comment_and_off_modes(responses_route, mode):
    route = responses_route
    stream = await route.create(mode)
    timer = asyncio.get_running_loop().call_later(0.05, route.ready.set)
    try:
        frames = await asyncio.wait_for(_collect(stream), timeout=1)
    finally:
        timer.cancel()
        await stream.aclose()
    comments = [f for f in frames if f.startswith(":")]
    assert bool(comments) == (mode == "comment")
    events = [_event(f) for f in frames if not f.startswith(":")]
    assert sum(e["type"] == "response.in_progress" for e in events) == 1
    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    assert [e["sequence_number"] for e in events] == list(range(1, len(events) + 1))


async def _collect(stream):
    return [frame async for frame in stream]


@pytest.mark.asyncio
async def test_cancel_during_prefill_closes_engine_and_releases_lease(responses_route):
    route = responses_route
    stream = await route.create()
    try:
        await anext(stream)  # response.created
        await anext(stream)  # initial response.in_progress
        heartbeat = _event(await asyncio.wait_for(anext(stream), timeout=1))
        assert heartbeat["type"] == "response.in_progress"
        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    finally:
        await stream.aclose()
    assert route.closed.is_set()
    route.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_prefill_failure_after_keepalive_keeps_response_identity(responses_route):
    route = responses_route
    route.engine.failure = RuntimeError("prefill failed")
    stream = await route.create()
    try:
        events = [_event(await anext(stream)), _event(await anext(stream))]
        events.append(_event(await asyncio.wait_for(anext(stream), timeout=1)))
        route.ready.set()
        events.extend([_event(frame) async for frame in stream])
    finally:
        await stream.aclose()
    assert events[-1]["type"] == "response.failed"
    assert events[-1]["response"]["id"] == events[0]["response"]["id"]
    assert events[-1]["response"]["error"]["message"] == "prefill failed"
    assert [e["sequence_number"] for e in events] == list(range(1, len(events) + 1))
    assert route.closed.is_set()
    route.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_failure_during_disconnect_poll_cannot_overtake_keepalive(
    responses_route,
):
    route = responses_route
    route.engine.failure = RuntimeError("prefill failed during disconnect check")
    stream = await route.create()

    async def is_disconnected():
        if route.started.is_set():
            route.ready.set()
            # Let the in-flight __anext__ produce response.failed while the
            # wrapper is awaiting its disconnect poll, before its next tick.
            await asyncio.sleep(0)
        return False

    route.requests[0].is_disconnected = is_disconnected
    try:
        events = [
            _event(frame) for frame in await asyncio.wait_for(_collect(stream), 1)
        ]
    finally:
        await stream.aclose()
    assert events[-1]["type"] == "response.failed"
    assert [e["sequence_number"] for e in events] == list(range(1, len(events) + 1))


@pytest.mark.asyncio
async def test_no_empty_response_snapshot_after_output_starts(responses_route):
    route = responses_route
    route.finish.clear()
    stream = await route.create()
    pending = None
    try:
        await anext(stream)
        await anext(stream)
        assert (
            _event(await asyncio.wait_for(anext(stream), 1))["type"]
            == "response.in_progress"
        )
        route.ready.set()
        while _event(await anext(stream))["type"] != "response.output_text.delta":
            pass
        pending = asyncio.create_task(anext(stream))
        done, _ = await asyncio.wait({pending}, timeout=0.05)
        assert not done, "Empty response snapshot emitted after output started"
        route.finish.set()
        assert (
            _event(await asyncio.wait_for(pending, 1))["type"]
            == "response.output_text.done"
        )
        events = [_event(frame) async for frame in stream]
        assert all(e["type"] != "response.in_progress" for e in events)
        assert events[-1]["type"] == "response.completed"
    finally:
        route.finish.set()
        if pending is not None and not pending.done():
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await stream.aclose()


@pytest.mark.asyncio
async def test_concurrent_responses_have_independent_keepalive_sequences(
    responses_route,
):
    route = responses_route
    streams = [await route.create(), await route.create()]
    try:
        first_events = []
        for stream in streams:
            first_events.append(
                [_event(await anext(stream)), _event(await anext(stream))]
            )
        heartbeats = await asyncio.gather(
            *(asyncio.wait_for(anext(stream), 1) for stream in streams)
        )
        for events, frame in zip(first_events, heartbeats):
            events.append(_event(frame))
        assert (
            first_events[0][0]["response"]["id"] != first_events[1][0]["response"]["id"]
        )
        route.ready.set()
        tails = await asyncio.gather(*(_collect(stream) for stream in streams))
        for events, tail in zip(first_events, tails):
            events.extend(_event(frame) for frame in tail)
            assert [e["sequence_number"] for e in events] == list(
                range(1, len(events) + 1)
            )
            assert {e["response"]["id"] for e in events if "response" in e} == {
                events[0]["response"]["id"]
            }
    finally:
        for stream in streams:
            await stream.aclose()
