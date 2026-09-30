# SPDX-License-Identifier: Apache-2.0
"""Route admission must re-sample memory after the pool's eviction callback."""

import concurrent.futures
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from omlx.engine.batched import BatchedEngine
from omlx.engine.vlm import VLMBatchedEngine
from omlx.exceptions import PrefillMemoryExceededError
from omlx.scheduler import Scheduler, SchedulerConfig

_GB = 1024**3


def _scheduler(monkeypatch):
    model = MagicMock()
    model.layers = []
    model.config = SimpleNamespace(
        num_hidden_layers=32,
        num_key_value_heads=8,
        num_attention_heads=32,
        head_dim=128,
    )
    del model.make_cache
    tokenizer = MagicMock(eos_token_id=2)
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(paged_cache_block_size=0),
    )
    scheduler._prefill_memory_guard = True
    scheduler._memory_hard_limit_bytes = 100 * _GB
    scheduler._memory_abort_limit_bytes = 10**18
    scheduler._prefill_headroom_safety = 1.0
    scheduler._last_mlx_active_memory_bytes = 110 * _GB
    scheduler._last_mlx_active_memory_at = time.monotonic()
    # Isolate post-callback refresh from the existing idle-age refresh path.
    monkeypatch.setattr(scheduler, "route_preflight_usage_is_stale", lambda: False)
    monkeypatch.setattr("omlx.scheduler.mx.get_cache_memory", lambda: 0)
    monkeypatch.setattr("omlx.scheduler.unreleased_graphics_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(scheduler, "_hot_cache_cpu_bytes", lambda: 0)
    return scheduler


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_cls", [BatchedEngine, VLMBatchedEngine])
@pytest.mark.parametrize("callback_result", [True, False])
@pytest.mark.parametrize("remaining_gb", [80, 110])
async def test_post_callback_refresh_preserves_admission_guard(
    monkeypatch, engine_cls, callback_result, remaining_gb
):
    scheduler = _scheduler(monkeypatch)
    resident = {"bytes": 110 * _GB}
    sample_threads = []
    event_loop_thread = threading.get_ident()

    def active_memory():
        sample_threads.append(threading.get_ident())
        return resident["bytes"]

    monkeypatch.setattr("omlx.scheduler.mx.get_active_memory", active_memory)
    monkeypatch.setattr("omlx.scheduler.get_phys_footprint", lambda: resident["bytes"])
    requests = []

    async def evict(request):
        requests.append(request)
        # False can mean the pool already sees enough headroom; its sample
        # can disagree with the requesting scheduler's cached MLX reading.
        resident["bytes"] = remaining_gb * _GB
        return callback_result

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        engine = engine_cls("test-model", prefill_eviction_callback=evict)
        engine._engine = SimpleNamespace(engine=SimpleNamespace(_mlx_executor=executor))
        if remaining_gb == 110:
            with pytest.raises(PrefillMemoryExceededError) as exc:
                await engine._preflight_or_raise_with_eviction(
                    scheduler, num_prompt_tokens=2048, request_id="after-eviction"
                )
            assert exc.value.estimated_bytes > 110 * _GB
        else:
            await engine._preflight_or_raise_with_eviction(
                scheduler, num_prompt_tokens=2048, request_id="after-eviction"
            )

    assert len(requests) == 1
    assert requests[0].current_bytes == 110 * _GB
    assert scheduler._last_mlx_active_memory_bytes == remaining_gb * _GB
    assert len(sample_threads) == 1
    assert sample_threads[0] != event_loop_thread


@pytest.mark.asyncio
async def test_fitting_preflight_does_not_queue_executor_work(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    scheduler._last_mlx_active_memory_bytes = 80 * _GB
    monkeypatch.setattr("omlx.scheduler.get_phys_footprint", lambda: 80 * _GB)
    refresh = MagicMock()
    monkeypatch.setattr(scheduler, "refresh_route_preflight_usage", refresh)
    evict = AsyncMock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        engine = BatchedEngine("test-model", prefill_eviction_callback=evict)
        engine._engine = SimpleNamespace(engine=SimpleNamespace(_mlx_executor=executor))
        await engine._preflight_or_raise_with_eviction(
            scheduler, num_prompt_tokens=2048, request_id="already-fits"
        )
    evict.assert_not_awaited()
    refresh.assert_not_called()
