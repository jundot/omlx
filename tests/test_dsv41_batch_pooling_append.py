# SPDX-License-Identifier: Apache-2.0
"""BatchPoolingCache append-only + generate_patch shared-pool identity."""

from __future__ import annotations

import importlib

import mlx.core as mx
import pytest

from omlx.patches.deepseek_v4.cache_extras import BatchPoolingCache, PoolingCache
from omlx.patches.deepseek_v4.generate_patch import apply_generate_patch


def _rows(start: int, count: int, D: int, B: int) -> mx.array:
    vals = mx.arange(start * D * B, (start + count) * D * B, dtype=mx.float32)
    return (vals.reshape(B, count, D) % 997) / 997.0


def test_batch_append_only_multi_step_matches_concatenate():
    """Index / ratio==1 pools never call accumulate_windows (_processed stays 0)."""
    B, D, ratio = 2, 8, 4
    cache = BatchPoolingCache(ratio, [0] * B)
    assert all(p == 0 for p in cache._processed)

    chunks = [3, 1, 5, 2, 8]
    start = 0
    expected = None
    for n in chunks:
        px = _rows(start, n, D, B)
        start += n
        got = cache.update_and_fetch(px)
        mx.eval(got)
        if expected is None:
            expected = px
        else:
            expected = mx.concatenate([expected, px], axis=1)
        mx.eval(expected)
        assert got.shape == expected.shape
        assert bool(mx.array_equal(got, expected))
        assert cache._pool_lengths == [expected.shape[1]] * B


def test_batch_accumulate_then_update_matches_single_pooling():
    """Equal-length KV path: BatchPoolingCache matches per-row PoolingCache."""
    B, L, D, ratio = 2, 32, 16, 4
    mx.random.seed(0)
    kv = mx.random.normal((B, L, D)).astype(mx.float32)
    gate = mx.random.normal((B, L, 1)).astype(mx.float32)

    singles = []
    for b in range(B):
        c = PoolingCache(ratio)
        r_kv, r_gate, _ = c.accumulate_windows(kv[b : b + 1], gate[b : b + 1], 0)
        if r_kv.shape[1] == 0:
            out = c.update_and_fetch(mx.zeros((1, 0, D), dtype=kv.dtype))
        else:
            px = r_kv.reshape(1, -1, ratio, D).mean(axis=2)
            out = c.update_and_fetch(px)
        mx.eval(out)
        singles.append(out)

    batch = BatchPoolingCache(ratio, [0] * B)
    r_kv, r_gate, r_base = batch.accumulate_windows(kv, gate, mx.array([0, 0]))
    if r_kv.shape[1] == 0:
        bout = batch.update_and_fetch(mx.zeros((B, 0, D), dtype=kv.dtype))
    else:
        px = r_kv.reshape(B, -1, ratio, D).mean(axis=2)
        bout = batch.update_and_fetch(px)
    mx.eval(bout)
    for b in range(B):
        assert bool(mx.array_equal(bout[b : b + 1], singles[b]))


def test_generate_patch_preserves_shared_pooling_identity():
    """CSA2 layers share one PoolingCache; batch convert must keep identity."""
    from mlx_lm.models.cache import CacheList, RotatingKVCache

    import mlx_lm.models.cache as cache_mod
    from omlx.patches.deepseek_v4 import cache_extras as extras

    if not hasattr(cache_mod, "PoolingCache"):
        cache_mod.PoolingCache = extras.PoolingCache
        cache_mod.BatchPoolingCache = extras.BatchPoolingCache
    apply_generate_patch()

    gen = importlib.import_module("mlx_lm.generate")

    shared = PoolingCache(4)
    caches = [
        CacheList(RotatingKVCache(max_size=128), shared),
        CacheList(RotatingKVCache(max_size=128), shared),
    ]

    class _M:
        layers = [None, None]

        def make_cache(self):
            return caches

    out = gen._make_cache(_M(), left_padding=[0, 0], max_kv_size=None)
    assert len(out) == 2
    a = out[0].caches[1]
    b = out[1].caches[1]
    assert isinstance(a, extras.BatchPoolingCache)
    assert a is b


def test_append_only_after_first_chunk_grows_lengths():
    """Regression: negative new_counts must not block later append-only updates."""
    cache = BatchPoolingCache(2, [0, 0])
    first = _rows(0, 10, 4, 2)
    got = cache.update_and_fetch(first)
    mx.eval(got)
    assert cache._pool_lengths == [10, 10]
    second = _rows(10, 1, 4, 2)
    got2 = cache.update_and_fetch(second)
    mx.eval(got2)
    assert cache._pool_lengths == [11, 11]
    assert got2.shape[1] == 11
