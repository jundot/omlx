# SPDX-License-Identifier: Apache-2.0
"""Regression tests for dedup-branch boundary-snapshot backfill.

Deduplicated blocks are never rewritten by store_cache, so a block first
stored without boundary-snapshot coverage keeps placeholder non-sliceable
payloads forever: every partial prefix match ending inside that region is
rejected and the request re-prefills from scratch, even though later stores
re-process the same tokens with fresh snapshots in hand. The backfill step
repairs such dedup'd placeholder blocks from the current store's boundary
snapshots, restoring partial-match walk-back at those boundaries.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omlx.cache.paged_cache import BlockTable, PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.pooling_delta import (
    POOLING_CACHE_DELTA_CLASS,
    compact_pooling_cache_snapshot,
)
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.cache.type_registry import CacheTypeRegistry

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

BLOCK_SIZE = 4
WINDOW = 4
POOL_RATIO = 4
POOL_DIM = 8

PLACEHOLDER_SHAPE = (1,)
REAL_ROTATING_SHAPE = (1, 2, WINDOW, 8)


class MockModel:
    def __init__(self, num_layers: int = 2):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        a = MagicMock()
        a.num_hidden_layers = self._num_layers
        return a


def _make_cache(tmp_path, num_layers=2):
    paged_cache = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=100,
        model_name="test-model",
        initial_blocks=100,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "ssd_cache",
        max_size_bytes=100 * 1024**2,
        hot_cache_max_bytes=10 * 1024**2,
        hot_cache_only=True,
    )
    cache = BlockAwarePrefixCache(
        model=MockModel(num_layers=num_layers),
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=ssd,
    )
    return cache, ssd


def _hybrid_cache_data(seq_len):
    """Gemma3-style hybrid: one sliceable KVCache + one rotating layer."""
    return [
        {
            "state": (
                mx.ones((1, 2, seq_len, 8)),
                mx.ones((1, 2, seq_len, 8)),
            ),
            "cache_type": "KVCache",
            "class_name": "KVCache",
            "meta_state": (str(seq_len),),
        },
        {
            "state": (
                mx.ones((1, 2, WINDOW, 8)),
                mx.ones((1, 2, WINDOW, 8)),
            ),
            "cache_type": "RotatingKVCache",
            "class_name": "RotatingKVCache",
            "meta_state": ("0", str(WINDOW), str(seq_len), str(WINDOW)),
        },
    ]


def _hybrid_snapshot(boundary_tc):
    """Full cache state at a block boundary (what prefill capture yields)."""
    return _hybrid_cache_data(boundary_tc)


def _rotating_layer_shape(ssd, block_hash):
    data, meta = ssd.load_block_with_metadata(block_hash)
    assert data is not None and meta is not None
    types = meta["layer_cache_types"]
    for i, type_name in enumerate(types):
        if CacheTypeRegistry.is_rotating_family(type_name):
            return tuple(data[i][0].shape)
    raise AssertionError("no rotating layer in block")


def _rotating_meta(ssd, block_hash):
    _, meta = ssd.load_block_with_metadata(block_hash)
    assert meta is not None
    return tuple(str(x) for x in meta["layer_meta_states"][1])


def _block_hash(cache, table, idx):
    block = cache.paged_cache.allocated_blocks[table.block_ids[idx]]
    assert block.block_hash is not None
    return block.block_hash


def _partial_table(cache, table, num_blocks, request_id):
    for block_id in table.block_ids[:num_blocks]:
        cache.paged_cache.allocated_blocks[block_id].ref_count += 1
    return BlockTable(
        request_id=request_id,
        block_ids=list(table.block_ids[:num_blocks]),
        num_tokens=num_blocks * BLOCK_SIZE,
    )


# --- V4 pooling fixtures (mirrors test_pooling_cache_delta.py) ---


def _pooling_layer(token_count: int, *, include_overlap_state: bool = False) -> dict:
    pooled_count = token_count // POOL_RATIO
    pooled = mx.arange(pooled_count * POOL_DIM, dtype=mx.float32).reshape(
        1, pooled_count, POOL_DIM
    )
    mx.eval(pooled)
    state = (None, None, pooled)
    if include_overlap_state:
        prev_win_kv = mx.arange(POOL_RATIO * POOL_DIM, dtype=mx.float32).reshape(
            1, 1, POOL_RATIO, POOL_DIM
        )
        prev_win_gate = prev_win_kv + 1000
        mx.eval(prev_win_kv, prev_win_gate)
        state = (*state, prev_win_kv, prev_win_gate)
    return {
        "state": [state],
        "meta_state": (["PoolingCache"], [POOL_RATIO]),
        "sub_class_names": ["PoolingCache"],
        "class_name": "CacheList",
        "cache_type": "CacheList",
    }


def _delta_pooling_layer(token_count: int) -> list[dict]:
    layers = [_pooling_layer(token_count, include_overlap_state=True)]
    compact_pooling_cache_snapshot(layers, token_count, BLOCK_SIZE)
    return layers


def _make_v4_cache(tmp_path):
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    paged = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=100,
        model_name="pooling-delta-test",
        initial_blocks=100,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "ssd",
        max_size_bytes=100 * 1024**2,
        hot_cache_max_bytes=10 * 1024**2,
        hot_cache_only=True,
        expected_model_name="pooling-delta-test",
    )
    cache = BlockAwarePrefixCache(
        model=MockModel(num_layers=1),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
    )
    return cache, ssd


def test_dedup_placeholder_rotating_block_backfilled(tmp_path):
    cache, ssd = _make_cache(tmp_path)
    tokens = list(range(3 * BLOCK_SIZE))

    # Relic simulation: stored without snapshots, interior blocks placeholder.
    t1 = cache.store_cache("relic", tokens, _hybrid_cache_data(len(tokens)))
    assert t1 is not None and len(t1.block_ids) == 3
    b0, b1 = _block_hash(cache, t1, 0), _block_hash(cache, t1, 1)
    assert _rotating_layer_shape(ssd, b0) == PLACEHOLDER_SHAPE
    assert _rotating_layer_shape(ssd, b1) == PLACEHOLDER_SHAPE

    # Partial match over the first 2 blocks rejects: no real rotating state.
    assert cache.reconstruct_cache(_partial_table(cache, t1, 2, "pre")) is None

    # Re-store the same tokens with full snapshot coverage (a diverging
    # request re-prefilled this region): dedup'd blocks get backfilled.
    snapshots = {
        tc: _hybrid_snapshot(tc)
        for tc in range(BLOCK_SIZE, len(tokens) + 1, BLOCK_SIZE)
    }
    t2 = cache.store_cache(
        "repair", tokens, _hybrid_cache_data(len(tokens)), boundary_snapshots=snapshots
    )
    assert t2 is not None

    assert _rotating_layer_shape(ssd, b0) == REAL_ROTATING_SHAPE
    assert _rotating_layer_shape(ssd, b1) == REAL_ROTATING_SHAPE
    # The rotating meta now carries the boundary offset, not the relic
    # end-of-sequence offset.
    assert _rotating_meta(ssd, b0)[2] == str(BLOCK_SIZE)
    assert _rotating_meta(ssd, b1)[2] == str(2 * BLOCK_SIZE)

    # The same partial match now restores.
    partial = _partial_table(cache, t1, 2, "post")
    assert cache.reconstruct_cache(partial) is not None


def test_partial_match_walks_back_to_backfilled_block(tmp_path):
    cache, ssd = _make_cache(tmp_path)
    tokens = list(range(3 * BLOCK_SIZE))
    t1 = cache.store_cache("relic", tokens, _hybrid_cache_data(len(tokens)))
    assert t1 is not None
    b0, b1 = _block_hash(cache, t1, 0), _block_hash(cache, t1, 1)

    # Snapshot only at the first boundary: b0 repaired, b1 stays placeholder.
    t2 = cache.store_cache(
        "repair",
        tokens,
        _hybrid_cache_data(len(tokens)),
        boundary_snapshots={BLOCK_SIZE: _hybrid_snapshot(BLOCK_SIZE)},
    )
    assert t2 is not None
    assert _rotating_layer_shape(ssd, b0) == REAL_ROTATING_SHAPE
    assert _rotating_layer_shape(ssd, b1) == PLACEHOLDER_SHAPE

    # Restore over blocks 0..1 walks back to the backfilled block.
    partial = _partial_table(cache, t1, 2, "walkback")
    result = cache.reconstruct_cache(partial)
    assert result is not None
    assert partial.num_tokens == BLOCK_SIZE


def test_dedup_placeholder_v4_delta_block_backfilled(tmp_path):
    cache, ssd = _make_v4_cache(tmp_path)
    tokens = list(range(3 * BLOCK_SIZE))

    t1 = cache.store_cache("relic", tokens, [_pooling_layer(len(tokens))])
    assert t1 is not None and len(t1.block_ids) == 3
    b0, b1 = _block_hash(cache, t1, 0), _block_hash(cache, t1, 1)
    data0, _ = ssd.load_block_with_metadata(b0)
    assert cache._is_placeholder_state(data0[0])

    assert cache.reconstruct_cache(_partial_table(cache, t1, 2, "pre")) is None

    snapshots = {
        tc: _delta_pooling_layer(tc)
        for tc in range(BLOCK_SIZE, len(tokens) + 1, BLOCK_SIZE)
    }
    t2 = cache.store_cache(
        "repair",
        tokens,
        [_pooling_layer(len(tokens))],
        boundary_snapshots=snapshots,
    )
    assert t2 is not None

    # Backfilled blocks carry the fresh delta form with per-block ranges.
    for block_idx, block_hash in enumerate([b0, b1]):
        block_data, _ = ssd.load_block_with_metadata(block_hash)
        marker = block_data[0][0]
        assert marker[0] == "__nstate__"
        assert marker[1] == POOLING_CACHE_DELTA_CLASS
        assert marker[2][5].tolist() == [block_idx, block_idx + 1]

    partial = _partial_table(cache, t1, 2, "post")
    restored = cache.reconstruct_cache(partial)
    assert restored is not None
    assert restored[0].caches[0].pooled.shape[1] == 2


def test_no_snapshot_leaves_dedup_unchanged(tmp_path):
    cache, ssd = _make_cache(tmp_path)
    tokens = list(range(3 * BLOCK_SIZE))
    t1 = cache.store_cache("relic", tokens, _hybrid_cache_data(len(tokens)))
    assert t1 is not None
    b0 = _block_hash(cache, t1, 0)

    t2 = cache.store_cache("again", tokens, _hybrid_cache_data(len(tokens)))
    assert t2 is not None
    assert _rotating_layer_shape(ssd, b0) == PLACEHOLDER_SHAPE
    assert not cache._backfill_checked_hashes


def test_backfill_inspects_each_hash_once_per_session(tmp_path):
    cache, ssd = _make_cache(tmp_path)
    tokens = list(range(3 * BLOCK_SIZE))
    cache.store_cache("relic", tokens, _hybrid_cache_data(len(tokens)))
    snapshots = {
        tc: _hybrid_snapshot(tc)
        for tc in range(BLOCK_SIZE, len(tokens) + 1, BLOCK_SIZE)
    }
    cache.store_cache(
        "repair", tokens, _hybrid_cache_data(len(tokens)), boundary_snapshots=snapshots
    )
    assert len(cache._backfill_checked_hashes) >= 2

    calls = {"n": 0}
    original = ssd.load_block_with_metadata

    def counting(block_hash):
        calls["n"] += 1
        return original(block_hash)

    ssd.load_block_with_metadata = counting
    cache.store_cache(
        "third", tokens, _hybrid_cache_data(len(tokens)), boundary_snapshots=snapshots
    )
    assert calls["n"] == 0
