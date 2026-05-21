# SPDX-License-Identifier: Apache-2.0
"""Tests for the replay-on-subscribe SSE delivery model.

These pin three behaviors the previous single-consumer `asyncio.Queue`
implementation could not provide:

1. **Replay**: a subscriber that connects after events were emitted
   still sees the entire history.
2. **Multi-consumer**: two subscribers both see every event in order.
3. **Terminal close**: the stream closes cleanly after a terminal
   event (`done` / `upload_done` / `error`) without blocking on a
   timeout for a subsequent event that will never come.

The reader helper mirrors the loop the SSE endpoint uses in
`omlx/admin/routes.py`: snapshot events under the lock, release,
yield, repeat.
"""

import asyncio
from typing import Optional

import pytest

from omlx.admin.benchmark import (
    BenchmarkRequest,
    BenchmarkRun,
    _send_event as bench_send_event,
)
from omlx.admin.accuracy_benchmark import (
    AccuracyBenchmarkRequest,
    AccuracyBenchmarkRun,
    _send_event as acc_send_event,
)


# --- Test helpers -----------------------------------------------------------


async def _drain(
    run,
    *,
    max_events: Optional[int] = None,
    timeout: float = 1.0,
) -> list[dict]:
    """Read the run's event log replay-then-attach style.

    Matches the SSE endpoint loop: snapshot under `run.cond`, release,
    yield, repeat. Returns once the run is terminal, `max_events` is
    reached, or the wait times out.
    """
    seen = 0
    out: list[dict] = []
    while True:
        async with run.cond:
            while seen >= len(run.events) and not run.terminal:
                try:
                    await asyncio.wait_for(run.cond.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
            new = list(run.events[seen:])
            seen = len(run.events)
            done = run.terminal
        out.extend(new)
        if max_events is not None and len(out) >= max_events:
            break
        if done:
            break
        if not new:
            # Timeout with no new events and not terminal — bail rather
            # than spin.
            break
    return out


def _bench_run() -> BenchmarkRun:
    req = BenchmarkRequest(model_id="x", prompt_lengths=[1024])
    return BenchmarkRun(bench_id="b-1", request=req)


def _acc_run() -> AccuracyBenchmarkRun:
    req = AccuracyBenchmarkRequest(model_id="x", benchmarks={"mmlu": 10})
    return AccuracyBenchmarkRun(bench_id="a-1", request=req)


# --- BenchmarkRun -----------------------------------------------------------


class TestBenchmarkSSEReplay:
    @pytest.mark.asyncio
    async def test_replay_buffered_events_to_late_subscriber(self):
        run = _bench_run()
        await bench_send_event(run, {"type": "progress", "n": 1})
        await bench_send_event(run, {"type": "progress", "n": 2})
        await bench_send_event(run, {"type": "upload_done", "data": {}})
        # Subscriber connects AFTER all events.
        events = await _drain(run)
        assert events == [
            {"type": "progress", "n": 1},
            {"type": "progress", "n": 2},
            {"type": "upload_done", "data": {}},
        ]

    @pytest.mark.asyncio
    async def test_multiple_simultaneous_consumers(self):
        run = _bench_run()

        async def reader():
            return await _drain(run, max_events=3)

        async def producer():
            # Give readers a tick to subscribe before sending.
            await asyncio.sleep(0.01)
            await bench_send_event(run, {"type": "progress", "n": 1})
            await bench_send_event(run, {"type": "progress", "n": 2})
            await bench_send_event(run, {"type": "upload_done", "data": {}})

        r1, r2, _ = await asyncio.gather(reader(), reader(), producer())
        expected = [
            {"type": "progress", "n": 1},
            {"type": "progress", "n": 2},
            {"type": "upload_done", "data": {}},
        ]
        assert r1 == expected
        assert r2 == expected

    @pytest.mark.asyncio
    async def test_terminal_event_closes_stream_without_extra_wait(self):
        run = _bench_run()
        await bench_send_event(run, {"type": "progress", "n": 1})
        await bench_send_event(run, {"type": "upload_done", "data": {}})

        # Should return immediately — no timeout — because the run is
        # already terminal.
        start = asyncio.get_event_loop().time()
        events = await _drain(run, timeout=5.0)
        elapsed = asyncio.get_event_loop().time() - start

        assert events[-1]["type"] == "upload_done"
        assert elapsed < 0.5, "stream blocked after terminal event"

    @pytest.mark.asyncio
    async def test_error_is_also_terminal(self):
        run = _bench_run()
        await bench_send_event(run, {"type": "error", "message": "boom"})
        events = await _drain(run, timeout=5.0)
        assert events == [{"type": "error", "message": "boom"}]
        assert run.terminal is True

    @pytest.mark.asyncio
    async def test_late_subscriber_sees_replay_then_live_events(self):
        run = _bench_run()
        # Pre-existing buffered events
        await bench_send_event(run, {"type": "progress", "n": 1})

        async def producer():
            # Subscriber will be mid-replay when this fires.
            await asyncio.sleep(0.02)
            await bench_send_event(run, {"type": "progress", "n": 2})
            await bench_send_event(run, {"type": "upload_done", "data": {}})

        async def subscriber():
            return await _drain(run)

        events, _ = await asyncio.gather(subscriber(), producer())
        # No event lost between replay and live phases.
        assert events == [
            {"type": "progress", "n": 1},
            {"type": "progress", "n": 2},
            {"type": "upload_done", "data": {}},
        ]


# --- AccuracyBenchmarkRun ---------------------------------------------------


class TestAccuracyBenchmarkSSEReplay:
    """Same contract on the accuracy benchmark run dataclass."""

    @pytest.mark.asyncio
    async def test_replay_to_late_subscriber(self):
        run = _acc_run()
        await acc_send_event(run, {"type": "progress", "phase": "load"})
        await acc_send_event(run, {"type": "result", "data": {"score": 0.5}})
        await acc_send_event(run, {"type": "done"})
        events = await _drain(run)
        assert [e["type"] for e in events] == ["progress", "result", "done"]

    @pytest.mark.asyncio
    async def test_multiple_consumers_see_same_events(self):
        run = _acc_run()

        async def reader():
            return await _drain(run, max_events=2)

        async def producer():
            await asyncio.sleep(0.01)
            await acc_send_event(run, {"type": "progress", "phase": "eval"})
            await acc_send_event(run, {"type": "done"})

        r1, r2, _ = await asyncio.gather(reader(), reader(), producer())
        assert r1 == r2

    @pytest.mark.asyncio
    async def test_last_progress_still_tracked(self):
        # The queue/status REST endpoint relies on `last_progress` for
        # the reconnect hint. The SSE refactor must preserve that.
        run = _acc_run()
        await acc_send_event(run, {"type": "progress", "phase": "eval", "current": 5})
        assert run.last_progress == {
            "type": "progress",
            "phase": "eval",
            "current": 5,
        }
