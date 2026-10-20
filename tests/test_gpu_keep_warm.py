# SPDX-License-Identifier: Apache-2.0
"""The idle GPU keep-warm ticker runs only while a model is loaded and idle."""

import asyncio
import concurrent.futures
import time
from types import SimpleNamespace

import pytest

from omlx import engine_pool as ep
from omlx.settings import GlobalSettings, ServerSettings


def _entry(*, active: bool = False, loaded: bool = True, idle_for: float = 0.0):
    engine = SimpleNamespace(has_active_requests=lambda: active) if loaded else None
    return SimpleNamespace(engine=engine, in_use=0, last_access=time.time() - idle_for)


@pytest.fixture
def pool(monkeypatch):
    ticks = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(ep, "_touch_gpu", lambda: ticks.append(1))
    monkeypatch.setattr(ep, "get_mlx_executor", lambda: executor)
    monkeypatch.setattr(ep, "shutdown_mlx_executor", executor.shutdown)
    pool = ep.EnginePool()
    pool.ticks = ticks
    yield pool


async def _settle(pool, seconds=0.08):
    await asyncio.sleep(seconds)
    return len(pool.ticks)


@pytest.mark.asyncio
async def test_ticks_only_while_a_loaded_model_is_idle(pool):
    pool.configure_gpu_keep_warm(0.005)
    pool._entries["m"] = _entry()
    pool._ensure_gpu_keep_warm_task()
    assert await _settle(pool) > 0

    pool._entries["m"] = _entry(active=True)
    n = await _settle(pool, 0.02)
    assert await _settle(pool) == n  # generation keeps the GPU busy already

    pool._entries["m"] = _entry(loaded=False)
    n = await _settle(pool, 0.02)
    assert await _settle(pool) == n  # nothing resident: let the GPU sleep

    pool._entries["m"] = _entry()
    assert await _settle(pool) > n

    await pool._stop_gpu_keep_warm()
    assert pool._gpu_keep_warm_task is None
    n = len(pool.ticks)
    assert await _settle(pool) == n


@pytest.mark.asyncio
async def test_ticks_stop_after_the_idle_window(pool):
    window = ep._GPU_KEEP_WARM_IDLE_WINDOW_S
    pool.configure_gpu_keep_warm(0.005)
    pool._entries["m"] = _entry(idle_for=window + 1)
    pool._ensure_gpu_keep_warm_task()
    assert await _settle(pool) == 0  # idle too long: let the GPU sleep

    # A request that outlasts the window restarts it when it finishes.
    pool._entries["m"] = _entry(active=True, idle_for=window + 1)
    await _settle(pool, 0.02)
    pool._entries["m"] = _entry(idle_for=window + 1)
    assert await _settle(pool) > 0

    await pool._stop_gpu_keep_warm()


@pytest.mark.asyncio
async def test_disabled_interval_starts_no_task(pool):
    pool.configure_gpu_keep_warm(0)
    pool._entries["m"] = _entry()
    pool._ensure_gpu_keep_warm_task()
    assert pool._gpu_keep_warm_task is None
    assert await _settle(pool, 0.03) == 0


@pytest.mark.asyncio
async def test_reconfiguring_to_zero_cancels_and_shutdown_stops(pool):
    pool.configure_gpu_keep_warm(0.005)
    pool._entries["m"] = _entry()
    pool._ensure_gpu_keep_warm_task()
    task = pool._gpu_keep_warm_task
    assert task is not None
    pool.configure_gpu_keep_warm(0)
    await asyncio.sleep(0)
    assert task.cancelled() or task.done()
    assert pool._gpu_keep_warm_task is None

    pool.configure_gpu_keep_warm(0.005)
    pool._ensure_gpu_keep_warm_task()
    await pool.shutdown()
    assert pool._gpu_keep_warm_task is None


def test_settings_default_roundtrip_and_env(monkeypatch):
    assert ServerSettings().gpu_keep_warm_interval == 0.5
    assert ServerSettings.from_dict({}).gpu_keep_warm_interval == 0.5
    s = ServerSettings.from_dict({"gpu_keep_warm_interval": 0})
    assert s.gpu_keep_warm_interval == 0.0
    assert ServerSettings.from_dict(s.to_dict()).gpu_keep_warm_interval == 0.0

    settings = GlobalSettings()
    monkeypatch.setenv("OMLX_GPU_KEEP_WARM_INTERVAL", "2")
    settings._apply_env_overrides()
    assert settings.server.gpu_keep_warm_interval == 2.0
    monkeypatch.setenv("OMLX_GPU_KEEP_WARM_INTERVAL", "nope")
    settings._apply_env_overrides()
    assert settings.server.gpu_keep_warm_interval == 2.0
