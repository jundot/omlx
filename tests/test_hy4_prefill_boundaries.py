# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Hy4 forwards spanning existing SSD blocks."""

from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.cache import CacheList, KVCache, RotatingKVCache

from omlx.patches.hy_v4.prefill_windows import prefill_windows, window_width
from omlx.scheduler import Scheduler


def test_wide_candidate_preserves_storage_grid_and_smaller_policy_limits():
    scheduler = SimpleNamespace(model=SimpleNamespace(model_type="hy_v4_flash"))
    cache = [RotatingKVCache(max_size=512, keep=0), CacheList(KVCache(), KVCache())]

    def plan(tokens, *, offset=0, enabled=True, allow=True):
        return Scheduler._prefill_boundary_candidate(
            scheduler, tokens, offset, 512, enabled, cache, allow_wide=allow
        )

    assert plan(4096) == (2048, True)
    assert plan(600) == (512, False)
    assert plan(2048, offset=100) == (412, False)
    assert plan(2048, allow=False) == (512, False)
    assert plan(4096, enabled=False) == (4096, False)
    assert Scheduler._whole_prefill_blocks(1620, 512) == (1536, True)
    assert Scheduler._whole_prefill_blocks(463, 512) == (463, False)


def test_intermediate_rotating_cache_contains_the_same_window():
    scheduler = SimpleNamespace(_stream=mx.default_stream(mx.gpu))
    stock = RotatingKVCache(max_size=512)
    wide = RotatingKVCache(max_size=512)
    keys = mx.arange(2048).reshape(1, 1, 2048, 1)
    values = -keys
    wide.update_and_fetch(keys, values)

    for boundary in (512, 1024, 1536):
        stock.update_and_fetch(
            keys[:, :, boundary - 512 : boundary],
            values[:, :, boundary - 512 : boundary],
        )
        rebuilt = Scheduler._hy4_rotating_window_at(scheduler, [wide], 2048, boundary)[
            0
        ]
        assert rebuilt.offset == boundary
        assert mx.array_equal(rebuilt.keys, stock.keys[:, :, -512:]).item()
        assert mx.array_equal(rebuilt.values, stock.values[:, :, -512:]).item()
    assert wide.offset == 2048


def test_arithmetic_window_context_is_local_to_the_forward():
    assert window_width() == 0
    with prefill_windows(512):
        assert window_width() == 512
    assert window_width() == 0
