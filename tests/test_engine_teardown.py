# SPDX-License-Identifier: Apache-2.0
"""Cache-preserving engine-close timeout and async lifetime regressions."""

import asyncio
import concurrent.futures
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import mlx.core as mx
import pytest

import omlx.cache.paged_ssd_cache as ssd
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.engine.base import _close_engine_core
from omlx.engine.batched import BatchedEngine
from omlx.engine.vlm import VLMBatchedEngine
from omlx.engine_core import EngineCore, _EngineTeardown
from omlx.model_registry import ModelOwnershipError, get_registry


def budget():
    with patch("omlx.engine_core.time.monotonic", return_value=100.0):
        return _EngineTeardown("test")


@pytest.mark.parametrize("last", [None, 99, 120, 129.9])
def test_no_recent_success_does_not_extend(last):
    guard = budget()
    guard.set_phase("primary_ssd", lambda: last)
    assert guard.check(159.9) is None
    assert "no recent SSD" in guard.check(160)
    assert not guard.extended


def test_one_shared_hard_cap_even_after_phase_changes():
    guard = budget()
    guard.set_phase("primary_ssd", lambda: 155)
    assert guard.check(160) is None
    assert guard.extended
    guard.set_phase("draft_ssd", lambda: 215)
    assert guard.check(219.9) is None
    assert "exceeded 120s" in guard.check(220)
    assert guard.deadline == 220


def test_mlx_phase_does_not_inherit_ssd_progress():
    guard = budget()
    guard.set_phase("primary_ssd", lambda: 159)
    guard.set_phase("mlx_reclaim")
    assert "no recent SSD" in guard.check(160)


def test_watchdog_logs_every_ten_seconds_with_current_phase(caplog):
    guard = budget()
    guard.set_phase("primary_ssd", lambda: 159.0)
    now = 100.0
    waits = []

    def wait(seconds):
        nonlocal now
        waits.append(seconds)
        now += seconds
        if now == 180:
            guard.set_phase("mlx_reclaim")
        return now >= 190

    with (
        patch.object(guard._done, "wait", side_effect=wait),
        patch("omlx.engine_core.time.monotonic", side_effect=lambda: now),
        caplog.at_level("INFO", logger="omlx.engine_core"),
    ):
        guard._watch()

    assert waits == [60, 10, 10, 10]
    assert [record.levelname for record in caplog.records] == [
        "WARNING",
        "INFO",
        "INFO",
    ]
    assert "elapsed 70s/120s, phase=primary_ssd" in caplog.records[1].message
    assert "last completed write 11.0s ago" in caplog.records[1].message
    assert "elapsed 80s/120s, phase=mlx_reclaim" in caplog.records[2].message
    assert "write" not in caplog.records[2].message


def test_progress_requires_the_same_waiting_thread(tmp_path):
    manager = PagedSSDCacheManager(tmp_path, max_size_bytes=1024**2)
    try:
        manager._persistence_last_success = 155
        thread_id = threading.get_ident()
        assert manager.persistence_progress(thread_id) is None
        with manager._persistence_io():
            assert manager.persistence_progress(thread_id) == 155
            assert manager.persistence_progress(thread_id + 1) is None
            manager._persistence_failed = True
            assert manager.persistence_progress(thread_id) is None
        assert manager.persistence_progress(thread_id) is None
    finally:
        manager.close()


@pytest.mark.parametrize("draft", [False, True])
def test_shutdown_preserves_all_blocks_for_reload(
    tmp_path, mock_model, mock_tokenizer, draft
):
    manager = PagedSSDCacheManager(
        tmp_path, max_size_bytes=1024**2, hot_cache_max_bytes=1024**2
    )
    # A queue smaller than the flush proves admission waits preserve blocks.
    manager._write_queue.maxsize = 2
    hashes = [i.to_bytes(32, "big") for i in range(12)]
    keys = mx.ones((1, 1, 4, 1))
    for block_hash in hashes:
        assert manager.save_block(block_hash, [(keys, keys)], token_count=4)
    engine = EngineCore(mock_model, mock_tokenizer)
    if draft:
        engine.scheduler._draft_paged_ssd_cache_manager = manager
    else:
        engine.scheduler.paged_ssd_cache_manager = manager
    try:
        engine.close()
        assert not manager._writer_thread.is_alive()
        assert manager.get_stats().ssd_write_drops == 0
        restarted = PagedSSDCacheManager(tmp_path, max_size_bytes=1024**2)
        try:
            for block_hash in hashes:
                cache = restarted.load_block(block_hash)
                assert cache is not None
            assert restarted._index.count == 12
        finally:
            restarted.close()
    finally:
        if not engine._closed:
            engine.close()


@pytest.mark.asyncio
async def test_async_close_keeps_loop_alive_and_waits_through_cancellation():
    entered, release = threading.Event(), threading.Event()

    class Core:
        def close(self):
            entered.set()
            assert release.wait(3)

    task = asyncio.create_task(_close_engine_core(Core()))
    for _ in range(100):
        if entered.is_set():
            break
        await asyncio.sleep(0.001)
    assert entered.is_set()
    task.cancel()
    await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    assert await task is True


def test_force_ownership_does_not_replace_closing_engine(mock_model, mock_tokenizer):
    engine = EngineCore(mock_model, mock_tokenizer)
    engine._closing = True
    try:
        with pytest.raises(ModelOwnershipError, match="still closing"):
            get_registry().acquire(mock_model, object(), "replacement", force=True)
    finally:
        engine._closing = False
        engine.close()


@pytest.mark.parametrize("progressing", [False, True])
def test_watchdog_terminates_stalled_or_over_budget_process(progressing):
    code = f"""
import time
from omlx.engine_core import _EngineTeardown
with _EngineTeardown("subprocess", 0.2) as guard:
    if {progressing!r}:
        guard.set_phase("primary_ssd", time.monotonic)
    time.sleep(10)
"""
    # Cold engine imports can exceed five seconds before the watchdog starts.
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 70
    if progressing:
        assert "extending teardown to 0.4s" in result.stderr
        assert "exceeded 0.4s" in result.stderr
    else:
        assert "no recent SSD persistence progress" in result.stderr
        assert "extending teardown" not in result.stderr


def test_completed_close_removes_watchdog_thread():
    with _EngineTeardown("joined") as guard:
        assert guard._thread.is_alive()
    assert not guard._thread.is_alive()


def test_active_store_worker_drains_before_reset_and_restores_prefix(
    tmp_path, mock_model, mock_tokenizer
):
    def caches():
        manager = PagedSSDCacheManager(
            tmp_path,
            max_size_bytes=1024**2,
            expected_block_size=4,
            expected_model_name="test-model",
            expected_num_layers=1,
        )
        paged = PagedCacheManager(
            block_size=4, max_blocks=100, initial_blocks=100, model_name="test-model"
        )
        paged.set_paged_ssd_cache_manager(manager)
        prefix = BlockAwarePrefixCache(
            model=mock_model,
            paged_cache_manager=paged,
            paged_ssd_cache_manager=manager,
        )
        return manager, paged, prefix

    manager, paged, prefix = caches()
    manager._write_queue.maxsize = 1
    engine = EngineCore(mock_model, mock_tokenizer)
    scheduler = engine.scheduler
    scheduler.paged_ssd_cache_manager = manager
    scheduler.paged_cache_manager = paged
    scheduler.block_aware_cache = prefix
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    scheduler._store_cache_executor = executor
    tokens = list(range(48))
    keys = mx.ones((1, 1, 48, 1))
    mx.eval(keys)
    cache = [
        {
            "state": (keys, keys),
            "meta_state": (48,),
            "class_name": "KVCache",
            "cache_type": "KVCache",
        }
    ]
    write = ssd._write_safetensors_no_mx
    entered, release = threading.Event(), threading.Event()

    def blocked_write(*args, **kwargs):
        entered.set()
        assert release.wait(10), "shutdown did not release the blocked writer"
        return write(*args, **kwargs)

    shutdown = scheduler.shutdown

    def release_and_shutdown():
        release.set()
        shutdown()

    with (
        patch.object(ssd, "_write_safetensors_no_mx", blocked_write),
        patch.object(scheduler, "shutdown", side_effect=release_and_shutdown),
    ):
        future = executor.submit(
            scheduler._async_store_cache_worker,
            "store",
            tokens,
            cache,
            None,
            None,
            None,
            None,
            None,
            False,
        )
        scheduler._inflight_store_futures["store"] = future
        try:
            assert entered.wait(10)
            assert not future.done()
            engine.close()
        finally:
            release.set()
            if not engine._closed:
                engine.close()
    assert future.done()
    assert not manager._writer_thread.is_alive()
    restored, _, prefix = caches()
    try:
        table, remaining = prefix.fetch_cache("reload", tokens + [999])
        assert table is not None
        assert table.num_tokens == 48
        assert remaining == [999]
    finally:
        restored.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper_class", [BatchedEngine, VLMBatchedEngine])
async def test_cancelled_wrapper_finishes_close_and_clears_references(wrapper_class):
    entered, release = threading.Event(), threading.Event()

    def close():
        entered.set()
        assert release.wait(3)

    wrapper = wrapper_class(model_name="test")
    wrapper._engine = SimpleNamespace(
        stop=AsyncMock(), engine=SimpleNamespace(close=close)
    )
    wrapper._vision_cache = None
    wrapper._diffusion_cancel_events = set()
    task = asyncio.create_task(wrapper.stop())
    for _ in range(100):
        if entered.is_set():
            break
        await asyncio.sleep(0.001)
    assert entered.is_set()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert wrapper._engine is None
    assert wrapper._loaded is False
