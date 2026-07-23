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
    _benchmark_runs,
    _send_event as bench_send_event,
    get_active_run,
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


# --- Active-run discovery ---------------------------------------------------


class TestGetActiveRun:
    """A second subscriber (page refresh / new tab) needs a way to find
    the currently-running bench so it can attach to the SSE stream.
    `get_active_run()` is that discovery surface — it scans the run
    registry and returns the first one whose status is "running"."""

    @pytest.fixture(autouse=True)
    def _clear_registry(self):
        # Test-level isolation: the module-level _benchmark_runs registry
        # leaks between tests otherwise.
        _benchmark_runs.clear()
        yield
        _benchmark_runs.clear()

    def test_returns_none_when_no_runs(self):
        assert get_active_run() is None

    def test_returns_none_when_all_completed(self):
        r = _bench_run()
        r.status = "completed"
        _benchmark_runs[r.bench_id] = r
        assert get_active_run() is None

    def test_returns_the_running_run(self):
        finished = _bench_run()
        finished.status = "completed"
        _benchmark_runs[finished.bench_id] = finished

        running = BenchmarkRun(
            bench_id="b-2",
            request=BenchmarkRequest(model_id="x", prompt_lengths=[1024]),
        )
        running.status = "running"
        _benchmark_runs[running.bench_id] = running

        found = get_active_run()
        assert found is running
        assert found.status == "running"

    def test_only_returns_running_status_not_cancelled_or_error(self):
        for state in ("cancelled", "error"):
            r = BenchmarkRun(
                bench_id=f"b-{state}",
                request=BenchmarkRequest(model_id="x", prompt_lengths=[1024]),
            )
            r.status = state
            _benchmark_runs[r.bench_id] = r
        assert get_active_run() is None


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


# --- Route-level: /api/bench/active + 409 on concurrent start ---------------


class _FakeEntry:
    def __init__(self):
        self.engine_type = "batched"
        self.model_type = "llm"
        self.engine = None
        self.is_pinned = False
        self.is_loading = False
        self.model_path = "/fake"


class _FakePool:
    def __init__(self):
        self._entries = {"model-x": _FakeEntry()}

    def get_entry(self, model_id):
        return self._entries.get(model_id)


@pytest.fixture
def bench_client(monkeypatch):
    """FastAPI TestClient with auth stubbed and a fake engine pool wired in."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from omlx.admin import routes as admin_routes

    _benchmark_runs.clear()
    admin_routes._get_engine_pool = lambda: _FakePool()

    async def _fake_require_admin():
        return True

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    yield TestClient(app)
    _benchmark_runs.clear()


class TestActiveBenchEndpoint:
    def test_returns_not_running_when_idle(self, bench_client):
        r = bench_client.get("/admin/api/bench/active")
        assert r.status_code == 200
        assert r.json() == {"running": False, "bench_id": None, "model_id": None}

    def test_returns_running_run_payload(self, bench_client):
        run = BenchmarkRun(
            bench_id="bench-abc",
            request=BenchmarkRequest(model_id="model-x", prompt_lengths=[1024]),
        )
        run.status = "running"
        _benchmark_runs[run.bench_id] = run

        r = bench_client.get("/admin/api/bench/active")
        assert r.status_code == 200
        assert r.json() == {
            "running": True,
            "bench_id": "bench-abc",
            "model_id": "model-x",
            "force_lm_engine": False,
            "external": False,
        }


class TestConcurrentStartRejection:
    """Server refuses to start a second throughput bench while one is
    running — two concurrent runs on the same engine produce mutually-
    corrupted measurements, and there's no way to recover the data."""

    def test_start_409_when_already_running(self, bench_client):
        # Seed a running run in the registry.
        existing = BenchmarkRun(
            bench_id="bench-existing",
            request=BenchmarkRequest(model_id="model-x", prompt_lengths=[1024]),
        )
        existing.status = "running"
        _benchmark_runs[existing.bench_id] = existing

        r = bench_client.post("/admin/api/bench/start", json={
            "model_id": "model-x",
            "prompt_lengths": [1024],
        })
        assert r.status_code == 409
        body = r.json()
        assert "already running" in body["detail"].lower()
        assert "bench-existing" in body["detail"]
        # Confirm the registry still has only the original run — no
        # second one was spuriously created and abandoned.
        assert len(_benchmark_runs) == 1

    def test_start_allowed_when_previous_completed(self, bench_client, monkeypatch):
        # A completed prior run must not block a fresh start.
        finished = BenchmarkRun(
            bench_id="bench-finished",
            request=BenchmarkRequest(model_id="model-x", prompt_lengths=[1024]),
        )
        finished.status = "completed"
        _benchmark_runs[finished.bench_id] = finished

        # Stub out the async run_benchmark task so the request returns
        # immediately without actually executing a bench.
        from omlx.admin import benchmark as bench_module

        async def _noop(run, pool):
            return

        monkeypatch.setattr(bench_module, "run_benchmark", _noop)

        r = bench_client.post("/admin/api/bench/start", json={
            "model_id": "model-x",
            "prompt_lengths": [1024],
        })
        assert r.status_code == 200, r.text
