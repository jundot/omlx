# SPDX-License-Identifier: Apache-2.0
"""Idle reporting and TTL start when an engine lease finishes."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from omlx.admin import routes
from omlx.engine_pool import EngineEntry, EnginePool


def _pool():
    pool = EnginePool()
    engine = MagicMock()
    engine.has_active_requests.return_value = False
    engine._engine = None
    engine.get_activity_snapshot.return_value = {"active_requests": 0, "activities": []}
    pool._entries["model"] = EngineEntry(
        model_id="model",
        model_path="/test/model",
        model_type="llm",
        engine_type="batched",
        estimated_size=1024,
        engine=engine,
        last_access=100.0,
        in_use=1,
    )
    pool._unload_engine = AsyncMock()
    return pool


def _activity(monkeypatch, pool):
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(routes, "_get_server_state", lambda: None)
    monkeypatch.setattr(routes, "_get_settings_manager", lambda: None)
    monkeypatch.setattr(routes, "_get_global_settings", lambda: None)
    monkeypatch.setattr(
        "omlx.prefill_progress.get_prefill_tracker",
        lambda: SimpleNamespace(get_model_progress=lambda _: []),
    )
    return routes._build_active_models_data()["models"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True])
async def test_release_records_completion_and_noop_release_preserves_it(
    monkeypatch, pending
):
    pool = _pool()
    entry = pool._entries["model"]
    if pending:
        entry.pending_unload_reason = "unload after request"
        # Exercise the locked release path without unloading the engine.
        entry.engine.has_active_requests.return_value = True
    monkeypatch.setattr("omlx.engine_pool.time.time", lambda: 700.0)
    await pool.release_engine("model")
    assert entry.in_use == 0
    assert entry.last_access == 700.0
    monkeypatch.setattr("omlx.engine_pool.time.time", lambda: 705.0)
    await pool.release_engine("model")
    await pool.release_engine("unknown")
    assert entry.last_access == 700.0


@pytest.mark.asyncio
async def test_long_request_gets_full_idle_ttl_after_release(monkeypatch):
    pool = _pool()
    monkeypatch.setattr("omlx.engine_pool.time.time", lambda: 700.0)
    await pool.release_engine("model")
    manager = SimpleNamespace(get_settings=lambda _: SimpleNamespace(ttl_seconds=60))
    monkeypatch.setattr("omlx.engine_pool.time.time", lambda: 701.0)
    assert await pool.check_ttl_expirations(manager) == []
    pool._unload_engine.assert_not_awaited()
    assert _activity(monkeypatch, pool)["idle_seconds"] == 1.0
    monkeypatch.setattr("omlx.engine_pool.time.time", lambda: 761.0)
    assert await pool.check_ttl_expirations(manager) == ["model"]
    pool._unload_engine.assert_awaited_once_with("model")


@pytest.mark.asyncio
async def test_cancelled_release_still_records_completion(monkeypatch):
    pool = _pool()
    entry = pool._entries["model"]
    entry.pending_unload_reason = "pending"
    entry.engine.has_active_requests.return_value = True
    await pool._lock.acquire()
    caller = asyncio.create_task(pool.release_engine("model"))
    try:
        await asyncio.sleep(0)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert entry.in_use == 1
        monkeypatch.setattr("omlx.engine_pool.time.time", lambda: 700.0)
    finally:
        pool._lock.release()
    await pool._drain_lease_release_tasks()
    assert entry.in_use == 0
    assert entry.last_access == 700.0


@pytest.mark.parametrize("busy", ["lease", "active", "waiting"])
def test_busy_model_reports_zero_idle_seconds(monkeypatch, busy):
    pool = _pool()
    entry = pool._entries["model"]
    entry.in_use = int(busy == "lease")
    if busy == "active":
        entry.engine.get_activity_snapshot.return_value = {
            "active_requests": 1,
            "activities": [],
        }
    elif busy == "waiting":
        scheduler = SimpleNamespace(
            snapshot_for_admin=lambda: {
                "running_by_id": {},
                "waiting": [
                    SimpleNamespace(
                        request_id="waiting", arrival_time=100.0, num_prompt_tokens=1
                    )
                ],
            }
        )
        entry.engine._engine = SimpleNamespace(
            engine=SimpleNamespace(
                _output_collectors={},
                scheduler=scheduler,
            )
        )
    monkeypatch.setattr("omlx.engine_pool.time.time", lambda: 700.0)
    model = _activity(monkeypatch, pool)
    assert model["idle_seconds"] == 0.0


@pytest.mark.asyncio
async def test_overlapping_requests_stay_busy_until_last_release(monkeypatch):
    pool = _pool()
    entry = pool._entries["model"]
    entry.in_use = 2
    monkeypatch.setattr("omlx.engine_pool.time.time", lambda: 700.0)
    await pool.release_engine("model")
    monkeypatch.setattr("omlx.engine_pool.time.time", lambda: 710.0)
    assert _activity(monkeypatch, pool)["idle_seconds"] == 0.0
    await pool.release_engine("model")
    monkeypatch.setattr("omlx.engine_pool.time.time", lambda: 711.0)
    assert _activity(monkeypatch, pool)["idle_seconds"] == 1.0
