# SPDX-License-Identifier: Apache-2.0
"""The hybrid model's cache block as a panel setting (hybrid_cache_block).

Measured 05/09 on GLM-5.3-Flash oQ2e at 110k tokens: 1024 gave 181 tok/s (94 whole
chunks, zero cuts) against 163 with 512. The default 0 keeps today's behaviour.
"""
from types import SimpleNamespace

from omlx.scheduler import Scheduler, SchedulerConfig
from omlx.settings import SchedulerSettings


def test_default_is_zero_and_round_trips():
    assert SchedulerSettings().hybrid_cache_block == 0
    assert SchedulerSettings.from_dict({}).hybrid_cache_block == 0
    assert SchedulerSettings.from_dict({"hybrid_cache_block": 1024}).hybrid_cache_block == 1024
    assert SchedulerConfig().hybrid_cache_block == 0


def _scheduler_with(block):
    from mlx_lm.models.cache import ArraysCache

    ns = SimpleNamespace(
        config=SchedulerConfig(
            paged_ssd_cache_dir="/tmp/x",
            paged_cache_block_size=256,
            prefill_step_size=512,
            hybrid_cache_block=block,
        ),
        model=SimpleNamespace(make_cache=lambda: [ArraysCache(size=2)]),
        _detect_rotating_window_sizes=lambda: [],
        _cache_tree_has_arrays_cache=lambda c: isinstance(c, ArraysCache),
        _ARRAYS_CACHE_BLOCK_SIZE=Scheduler._ARRAYS_CACHE_BLOCK_SIZE,
        _qwen35_prefill_floor=0,
    )
    ns._enlarge_block_size_for_arrays_cache = (
        Scheduler._enlarge_block_size_for_arrays_cache.__get__(ns, Scheduler)
    )
    ns._enlarge_block_size_for_arrays_cache()
    return ns.config


def test_the_setting_drives_block_and_prefill_step():
    cfg = _scheduler_with(1024)
    assert cfg.paged_cache_block_size == 1024
    assert cfg.prefill_step_size == 1024


def test_zero_keeps_the_automatic_target():
    cfg = _scheduler_with(0)
    expected = max(Scheduler._ARRAYS_CACHE_BLOCK_SIZE, 512)
    assert cfg.paged_cache_block_size == expected
    assert cfg.prefill_step_size == 512  # untouched on the automatic path
