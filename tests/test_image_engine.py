# SPDX-License-Identifier: Apache-2.0
"""Deterministic worker ownership checks; no model or Metal execution."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

import omlx.engine.image as module


@pytest.fixture
def setup_engine(monkeypatch, tmp_path):
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(module, "get_mlx_executor", lambda: executor)
    monkeypatch.setattr(module.mx, "synchronize", lambda: None)
    monkeypatch.setattr(
        module,
        "inspect_bundle",
        lambda path: {"structural_status": "valid", "physical_weight_bytes": 100},
    )
    engine = module.ImageEngine(tmp_path)
    engine._fairness = Mock()
    monkeypatch.setattr(engine, "_admit", lambda: None)
    yield engine
    executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_generator_returns_and_releases_activity(setup_engine, monkeypatch):
    engine = setup_engine
    events = []

    def steps(*args, **kwargs):
        try:
            yield ("encoding", 1, 1)
            return b"png", {}
        finally:
            events.append("closed")

    monkeypatch.setattr(module, "generate_steps", steps)
    await engine.start()
    assert await engine.generate("x", 1) == (b"png", {})
    assert events == ["closed"] and not engine.has_active_requests()
    assert engine._fairness.settle.call_count == 2


@pytest.mark.asyncio
async def test_repeated_cancel_waits_for_worker_before_release(
    setup_engine, monkeypatch
):
    engine = setup_engine
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def steps(*args, **kwargs):
        try:
            entered.set()
            assert release.wait(5)
            yield ("denoising", 1, 8)
        finally:
            closed.set()

    monkeypatch.setattr(module, "generate_steps", steps)
    await engine.start()
    task = asyncio.create_task(engine.generate("x", 1))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and engine.has_active_requests() and not closed.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert closed.is_set() and not engine.has_active_requests()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_stop_and_second_request_do_not_overlap_worker(setup_engine, monkeypatch):
    engine = setup_engine
    entered = threading.Event()
    release = threading.Event()

    def steps(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        yield ("encoding", 1, 1)
        if kwargs["cancel"].is_set():
            raise InterruptedError("stopped")

    monkeypatch.setattr(module, "generate_steps", steps)
    await engine.start()
    task = asyncio.create_task(engine.generate("x", 1))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        with pytest.raises(RuntimeError, match="busy"):
            await engine.generate("x", 2)
        stop = asyncio.create_task(engine.stop())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not stop.done() and engine.has_active_requests()
        release.set()
        with pytest.raises(InterruptedError):
            await task
        await stop
        assert not engine.get_stats()["loaded"] and not engine.has_active_requests()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_admission_failure_never_enters_pipeline(setup_engine, monkeypatch):
    engine = setup_engine
    started = []

    def steps(*args, **kwargs):
        started.append(True)
        yield ("encoding", 1, 1)

    def reject():
        raise MemoryError("budget")

    monkeypatch.setattr(module, "generate_steps", steps)
    monkeypatch.setattr(engine, "_admit", reject)
    await engine.start()
    with pytest.raises(MemoryError):
        await engine.generate("x", 1)
    assert not started and not engine.has_active_requests()


@pytest.mark.asyncio
async def test_failed_start_not_loaded(setup_engine, monkeypatch):
    monkeypatch.setattr(
        module,
        "inspect_bundle",
        lambda path: {"structural_status": "invalid", "errors": ["missing shard"]},
    )
    with pytest.raises(ValueError):
        await setup_engine.start()
    assert not setup_engine.get_stats()["loaded"]
