# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DeepSeek V4.1 Engram QuantHotRowCache."""

from __future__ import annotations

import numpy as np

from omlx.patches.deepseek_v41.engram_cache import QuantHotRowCache


def test_lru_hit_miss_and_eviction():
    cache = QuantHotRowCache(capacity_rows=4, head_dim=256, block_size=32, policy="lru")
    assert cache.enabled
    keys = np.arange(4, dtype=np.int64)
    w = np.arange(4 * 256, dtype=np.uint8).reshape(4, 256)
    s = np.arange(4 * 8, dtype=np.uint8).reshape(4, 8)
    cache.put_many(keys, w, s)
    hit, hw, hs = cache.get_many_raw(keys)
    assert hit.all()
    assert np.array_equal(hw, w)
    assert np.array_equal(hs, s)
    # Insert 2 more → evict 2 LRU
    cache.put_many(
        np.array([10, 11], dtype=np.int64),
        np.zeros((2, 256), np.uint8),
        np.zeros((2, 8), np.uint8),
    )
    assert cache.evictions == 2
    hit2, _, _ = cache.get_many_raw(np.array([0, 1, 10, 11], dtype=np.int64))
    assert list(hit2) == [False, False, True, True]


def test_clock_eviction_runs():
    cache = QuantHotRowCache(capacity_rows=2, head_dim=64, block_size=32, policy="clock")
    cache.put_many(
        np.array([1, 2], dtype=np.int64),
        np.ones((2, 64), np.uint8),
        np.ones((2, 2), np.uint8),
    )
    cache.put_many(
        np.array([3], dtype=np.int64),
        np.zeros((1, 64), np.uint8),
        np.zeros((1, 2), np.uint8),
    )
    assert cache.evictions >= 1
    assert cache.size == 2


def test_disabled_when_capacity_zero():
    cache = QuantHotRowCache(0, 256, 32, "lru")
    assert not cache.enabled
    hit, _, _ = cache.get_many_raw(np.array([1], dtype=np.int64))
    assert not hit.any()
