# SPDX-License-Identifier: Apache-2.0
"""Cache-store completions wake admission without moving MLX cleanup threads."""

import asyncio
import concurrent.futures
import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omlx.engine_core import EngineConfig, EngineCore
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerOutput, _StoreCacheGate


class ObservedEvent(asyncio.Event):
    def __init__(self):
        super().__init__()
        self.wait_entered = asyncio.Event()

    async def wait(self):
        self.wait_entered.set()
        return await super().wait()


def make_engine():
    """Use real engine-loop/deferred-cleanup methods with no model/GPU work."""
    engine = EngineCore.__new__(EngineCore)
    engine.config = EngineConfig(step_interval=60.0, decode_burst_max_steps=1)
    engine._loop = asyncio.get_running_loop()
    engine._wake_event = ObservedEvent()
    engine._running = True
    engine._mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    engine._last_reap = float("inf")
    engine._steps_executed = 0
    engine._output_collectors = {}
    engine._stream_states = {}

    scheduler = Scheduler.__new__(Scheduler)
    engine.scheduler = scheduler
    scheduler.model = SimpleNamespace()
    scheduler._stream = None
    scheduler._wake_callback = engine._wake_engine_loop
    scheduler._pending_async_removes = deque()
    scheduler._store_cache_gate = _StoreCacheGate(cap=1)
    scheduler._store_cache_gate.note_submitted()
    scheduler._inflight_store_futures = {}
    scheduler._inflight_store_info = {}
    scheduler._boundary_snapshot_store = None
    scheduler.requests = {}
    scheduler.uid_to_request_id = {1: "saved"}
    scheduler.request_id_to_uid = {"saved": 1}
    scheduler._remove_uid_from_active_batch = MagicMock()
    scheduler._clear_request_admission_bookkeeping = MagicMock()
    scheduler._schedule_deferred_metal_clear = MagicMock()
    scheduler._memory_admission_blocked_request_id = None
    scheduler.has_requests = lambda: True
    return engine


def submit_store(scheduler, future):
    """Enqueue via the production cancelled-prefill store path."""
    scheduler._store_cache_gate.note_done()
    scheduler._store_cache_executor = MagicMock()
    scheduler._store_cache_executor.submit.return_value = future
    scheduler._bypass_hot_cache_under_pressure = lambda: False
    request = SimpleNamespace(
        request_id="saved",
        vlm_extra_keys_for_cache=None,
        vlm_extra_key_token_start_for_cache=0,
        vlm_extra_key_ranges_for_cache=None,
    )
    return scheduler._submit_abort_store(request, ([1], [], None, None))


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["result", "error", "cancel"])
async def test_store_completion_wakes_throttled_admission_on_owner_thread(finish):
    engine = make_engine()
    scheduler = engine.scheduler
    future = concurrent.futures.Future()
    submit_store(scheduler, future)
    admitted = asyncio.Event()
    cleanup_threads = []
    scheduler._clear_request_admission_bookkeeping.side_effect = (
        lambda _rid: cleanup_threads.append(threading.get_ident())
    )

    def step():
        scheduler._drain_pending_async_removes()
        if scheduler._store_cache_gate.has_capacity:
            engine._loop.call_soon_threadsafe(admitted.set)
        return SchedulerOutput(has_work=False)

    scheduler.step = step
    with patch("omlx.scheduler._safe_sync_stream"):
        task = asyncio.create_task(engine._engine_loop())
        try:
            await asyncio.wait_for(engine._wake_event.wait_entered.wait(), timeout=1)
            assert not admitted.is_set()
            callback_threads = []

            def complete():
                callback_threads.append(threading.get_ident())
                if finish == "error":
                    future.set_exception(RuntimeError("synthetic store failure"))
                elif finish == "cancel":
                    future.cancel()
                else:
                    future.set_result(None)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
                worker.submit(complete).result(timeout=1)
            # The engine's polling timeout is 60 seconds: admission must be
            # driven by the completion notification, not the timeout.
            await asyncio.wait_for(admitted.wait(), timeout=1)
            assert len(cleanup_threads) == 1
            assert cleanup_threads[0] != callback_threads[0]
            assert scheduler._store_cache_gate.in_flight == 0
        finally:
            engine._running = False
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            engine._mlx_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_completion_during_step_is_not_cleared_before_throttled_wait():
    engine = make_engine()
    scheduler = engine.scheduler
    future = concurrent.futures.Future()
    submit_store(scheduler, future)
    admitted = asyncio.Event()
    steps = 0

    def step():
        nonlocal steps
        steps += 1
        scheduler._drain_pending_async_removes()
        if steps == 1:
            # Complete after the step's drain but before it returns no-work.
            # Clearing the wake event after dispatch loses this notification.
            future.set_result(None)
        elif scheduler._store_cache_gate.has_capacity:
            engine._loop.call_soon_threadsafe(admitted.set)
        return SchedulerOutput(has_work=False)

    scheduler.step = step
    with patch("omlx.scheduler._safe_sync_stream"):
        task = asyncio.create_task(engine._engine_loop())
        try:
            await asyncio.wait_for(admitted.wait(), timeout=1)
            assert steps == 2
            assert not scheduler._pending_async_removes
        finally:
            engine._running = False
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            engine._mlx_executor.shutdown(wait=True)


def test_already_completed_future_is_enqueued_before_notification():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._pending_async_removes = deque()
    pending_at_notification = []
    scheduler._wake_callback = lambda: pending_at_notification.append(
        list(scheduler._pending_async_removes)
    )
    future = concurrent.futures.Future()
    future.set_result(None)
    scheduler._defer_async_remove(1, "saved", future)
    assert pending_at_notification == [[(1, "saved", future)]]


def test_finished_request_store_notifies_without_releasing_its_row(
    mock_model, mock_tokenizer
):
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    callback = MagicMock()
    scheduler._wake_callback = callback
    scheduler.block_aware_cache = MagicMock()
    scheduler.paged_cache_manager = None
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.active_batch = None
    future = concurrent.futures.Future()
    scheduler._store_cache_executor = MagicMock()
    scheduler._store_cache_executor.submit.return_value = future
    scheduler._store_cache_gate = _StoreCacheGate(cap=1)
    request = Request(
        request_id="saved", prompt="synthetic", sampling_params=SamplingParams()
    )
    request.prompt_token_ids = [1, 2, 3, 4]
    request.num_prompt_tokens = 4
    request.output_token_ids = [5]
    request._extracted_cache = [{"state": "synthetic-cache"}]
    scheduler.running["saved"] = request
    scheduler.requests["saved"] = request
    scheduler.request_id_to_uid["saved"] = 1
    scheduler.uid_to_request_id[1] = "saved"
    with (
        patch("omlx.scheduler._safe_sync_stream"),
        patch("omlx.scheduler.mx.stream"),
        patch("omlx.scheduler.mx.eval"),
        patch.object(scheduler, "_get_boundary_store_override", return_value=None),
        patch.object(scheduler, "_detect_boundary_snapshot_need", return_value=False),
        patch.object(
            scheduler, "_model_has_unreconstructible_cache", return_value=False
        ),
        patch.object(scheduler, "_bypass_hot_cache_under_pressure", return_value=False),
    ):
        scheduler._cleanup_finished({"saved"})
    assert list(scheduler._pending_async_removes) == [(1, "saved", future)]
    callback.assert_not_called()
    future.set_result(None)
    callback.assert_called_once_with()
    # The callback must leave row removal and gate accounting for the owner.
    assert scheduler.request_id_to_uid["saved"] == 1
    assert scheduler._store_cache_gate.in_flight == 1
    with patch("omlx.scheduler._safe_sync_stream"):
        assert scheduler._drain_pending_async_removes()
    assert scheduler._store_cache_gate.in_flight == 0
    assert "saved" not in scheduler.requests


def test_completion_after_engine_stop_is_a_noop_until_owner_cleanup():
    scheduler = Scheduler.__new__(Scheduler)
    engine = EngineCore.__new__(EngineCore)
    engine._loop = None
    engine._wake_event = None
    scheduler._wake_callback = engine._wake_engine_loop
    scheduler._pending_async_removes = deque()
    future = concurrent.futures.Future()
    scheduler._defer_async_remove(1, "saved", future)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(future.set_result, None).result(timeout=1)
    assert list(scheduler._pending_async_removes) == [(1, "saved", future)]
