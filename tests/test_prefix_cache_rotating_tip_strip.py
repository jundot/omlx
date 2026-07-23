# SPDX-License-Identifier: Apache-2.0
"""Regression guards for supersede-on-extend rotating tip stripping (#1795).

On sliding-window models every store of a growing conversation writes one
tip block carrying the full rotating state of all sliding layers (~205-410MB
fp16 on a gemma3-class 25-sliding-layer model). Restore only ever consumes
the newest such block, and the immediate previous tip serves as the
walk-back fallback — so the tip two generations back is dead weight. Before
the fix those blocks accumulated one per turn, filled the hot cache after
~10-20 turns, and LRU eviction broke the prefix chain (multi-turn cache hit
collapsed permanently to 0%).

The fix tracks tip lineage in store_cache and strips the grandparent tip's
rotating payload down to the standard placeholder, keeping sliceable
(KVCache) layer slices intact and keeping the hot-cache / shared-budget byte
counters consistent (rewrite goes through forget_block + save_block).

Tests:
- test_grandparent_tip_stripped_prev_tip_kept — 4-turn chain: tip two
  generations back is stripped, immediate previous tip stays intact.
- test_stripped_block_keeps_sliceable_layers — KVCache slice survives the
  strip; only the rotating layer becomes a placeholder.
- test_hot_cache_byte_counter_consistent — _hot_cache_total_bytes matches
  the recomputed entry sizes and decreases after the strip.
- test_shared_budget_counter_consistent — SharedHotCacheBudget accounting
  matches the owner's hot cache bytes after the strip.
- test_kvcache_only_model_unaffected — no rotating layers: no lineage
  tracking, no rewrites.
- test_prefill_ready_rotating_name_stripped — family detection covers the
  PrefillReadyRotatingKVCache subclass name used by warm-restored caches.
- test_registry_rotating_family — registry-level family matching.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omlx.cache._rotating_subclass import PrefillReadyRotatingKVCache
from omlx.cache.paged_cache import BlockTable, PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager, SharedHotCacheBudget
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


class MockModel:
    def __init__(self, num_layers: int = 2):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        a = MagicMock()
        a.num_hidden_layers = self._num_layers
        return a


def _make_cache(tmp_path, budget=None):
    """A prefix cache wired to a real hot-cache-only SSD manager."""
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
        hot_cache_budget=budget,
    )
    cache = BlockAwarePrefixCache(
        model=MockModel(num_layers=2),
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=ssd,
    )
    return cache, ssd


def _hybrid_cache_data(seq_len, rotating_type="RotatingKVCache"):
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
            "cache_type": rotating_type,
            "class_name": rotating_type,
            "meta_state": ("0", str(WINDOW), str(seq_len), str(WINDOW)),
        },
    ]


def _kvcache_only_data(seq_len):
    return [
        {
            "state": (
                mx.ones((1, 2, seq_len, 8)),
                mx.ones((1, 2, seq_len, 8)),
            ),
            "cache_type": "KVCache",
            "class_name": "KVCache",
            "meta_state": (str(seq_len),),
        }
        for _ in range(2)
    ]


def _store_turn(cache, turn, num_blocks, data_fn=_hybrid_cache_data):
    """Simulate one conversation turn: store a chain of num_blocks blocks.

    Each turn uses a fresh request id (as real multi-turn requests do);
    earlier blocks dedup against the existing chain and only the new tail
    is allocated and saved.
    """
    tokens = list(range(num_blocks * BLOCK_SIZE))
    table = cache.store_cache(f"turn-{turn}", tokens, data_fn(seq_len=len(tokens)))
    assert table is not None
    assert len(table.block_ids) == num_blocks
    return table


def _block_hash(cache, table, idx):
    block = cache.paged_cache.allocated_blocks[table.block_ids[idx]]
    assert block.block_hash is not None
    return block.block_hash


def _rotating_layer_shape(ssd, block_hash):
    data, meta = ssd.load_block_with_metadata(block_hash)
    assert data is not None and meta is not None
    types = meta["layer_cache_types"]
    for i, type_name in enumerate(types):
        if CacheTypeRegistry.is_rotating_family(type_name):
            return tuple(data[i][0].shape)
    raise AssertionError("no rotating layer in block")


def _kvcache_layer_shape(ssd, block_hash):
    data, meta = ssd.load_block_with_metadata(block_hash)
    assert data is not None and meta is not None
    types = meta["layer_cache_types"]
    for i, type_name in enumerate(types):
        if type_name == "KVCache":
            return tuple(data[i][0].shape)
    raise AssertionError("no KVCache layer in block")


PLACEHOLDER_SHAPE = (1,)
REAL_ROTATING_SHAPE = (1, 2, WINDOW, 8)


def test_grandparent_tip_stripped_prev_tip_kept(tmp_path):
    """Tip two generations back is stripped; previous tip stays intact."""
    cache, ssd = _make_cache(tmp_path)

    t1 = _store_turn(cache, 1, num_blocks=2)
    tip1 = _block_hash(cache, t1, -1)
    assert _rotating_layer_shape(ssd, tip1) == REAL_ROTATING_SHAPE

    t2 = _store_turn(cache, 2, num_blocks=3)
    tip2 = _block_hash(cache, t2, -1)
    # tip1 is now the immediate previous tip: kept as walk-back fallback.
    assert _rotating_layer_shape(ssd, tip1) == REAL_ROTATING_SHAPE
    assert _rotating_layer_shape(ssd, tip2) == REAL_ROTATING_SHAPE

    t3 = _store_turn(cache, 3, num_blocks=4)
    tip3 = _block_hash(cache, t3, -1)
    # tip1 is two generations back: rotating payload stripped.
    assert _rotating_layer_shape(ssd, tip1) == PLACEHOLDER_SHAPE
    assert _rotating_layer_shape(ssd, tip2) == REAL_ROTATING_SHAPE
    assert _rotating_layer_shape(ssd, tip3) == REAL_ROTATING_SHAPE

    t4 = _store_turn(cache, 4, num_blocks=5)
    tip4 = _block_hash(cache, t4, -1)
    # Steady state: exactly the two newest tips stay heavy.
    assert _rotating_layer_shape(ssd, tip2) == PLACEHOLDER_SHAPE
    assert _rotating_layer_shape(ssd, tip3) == REAL_ROTATING_SHAPE
    assert _rotating_layer_shape(ssd, tip4) == REAL_ROTATING_SHAPE


def test_stripped_block_keeps_sliceable_layers(tmp_path):
    """Only the rotating layer is stripped; the KVCache slice survives,
    and the stripped layer reads back as a standard placeholder."""
    cache, ssd = _make_cache(tmp_path)

    t1 = _store_turn(cache, 1, num_blocks=2)
    tip1 = _block_hash(cache, t1, -1)
    kv_shape_before = _kvcache_layer_shape(ssd, tip1)

    _store_turn(cache, 2, num_blocks=3)
    _store_turn(cache, 3, num_blocks=4)

    assert _rotating_layer_shape(ssd, tip1) == PLACEHOLDER_SHAPE
    assert _kvcache_layer_shape(ssd, tip1) == kv_shape_before

    # The stripped layer must look like the standard last-block-only
    # placeholder so restore's walk-back/reject path handles it natively.
    data, meta = ssd.load_block_with_metadata(tip1)
    rotating_idx = next(
        i
        for i, t in enumerate(meta["layer_cache_types"])
        if CacheTypeRegistry.is_rotating_family(t)
    )
    assert cache._is_placeholder_state(data[rotating_idx])


def test_hot_cache_byte_counter_consistent(tmp_path):
    """The stripped entry shrinks and the byte counter stays exact."""
    cache, ssd = _make_cache(tmp_path)

    t1 = _store_turn(cache, 1, num_blocks=2)
    tip1 = _block_hash(cache, t1, -1)
    _store_turn(cache, 2, num_blocks=3)
    tip1_size_before = ssd._hot_cache_entry_size(ssd._hot_cache[tip1])

    _store_turn(cache, 3, num_blocks=4)

    # The superseded tip's hot-cache entry must have shrunk in place.
    tip1_size_after = ssd._hot_cache_entry_size(ssd._hot_cache[tip1])
    assert tip1_size_after < tip1_size_before
    # And the counter must equal the recomputed sum of live entry sizes.
    expected = sum(
        ssd._hot_cache_entry_size(entry) for entry in ssd._hot_cache.values()
    )
    assert ssd._hot_cache_total_bytes == expected


def test_shared_budget_counter_consistent(tmp_path):
    """Shared budget accounting matches the owner's hot-cache bytes."""
    budget = SharedHotCacheBudget(max_bytes=10 * 1024**2)
    cache, ssd = _make_cache(tmp_path, budget=budget)

    _store_turn(cache, 1, num_blocks=2)
    _store_turn(cache, 2, num_blocks=3)
    _store_turn(cache, 3, num_blocks=4)

    assert budget.total_bytes == ssd._hot_cache_total_bytes
    expected = sum(
        ssd._hot_cache_entry_size(entry) for entry in ssd._hot_cache.values()
    )
    assert budget.total_bytes == expected


def test_kvcache_only_model_unaffected(tmp_path):
    """Models without rotating layers never track lineage or rewrite."""
    cache, ssd = _make_cache(tmp_path)

    t1 = _store_turn(cache, 1, num_blocks=2, data_fn=_kvcache_only_data)
    tip1 = _block_hash(cache, t1, -1)
    kv_before = _kvcache_layer_shape(ssd, tip1)

    _store_turn(cache, 2, num_blocks=3, data_fn=_kvcache_only_data)
    _store_turn(cache, 3, num_blocks=4, data_fn=_kvcache_only_data)

    assert cache._rotating_tip_lineage == {}
    assert _kvcache_layer_shape(ssd, tip1) == kv_before


def test_prefill_ready_rotating_name_stripped(tmp_path):
    """Warm-restored caches serialize as PrefillReadyRotatingKVCache; the
    family match must strip those tips too (exact-name comparison broke
    here historically)."""
    cache, ssd = _make_cache(tmp_path)

    def data_fn(seq_len):
        return _hybrid_cache_data(seq_len, rotating_type="PrefillReadyRotatingKVCache")

    t1 = _store_turn(cache, 1, num_blocks=2, data_fn=data_fn)
    tip1 = _block_hash(cache, t1, -1)
    assert _rotating_layer_shape(ssd, tip1) == REAL_ROTATING_SHAPE

    _store_turn(cache, 2, num_blocks=3, data_fn=data_fn)
    _store_turn(cache, 3, num_blocks=4, data_fn=data_fn)

    assert _rotating_layer_shape(ssd, tip1) == PLACEHOLDER_SHAPE


def test_prefill_ready_rotating_name_reconstructs_with_rotating_handler():
    """Restore must route PrefillReadyRotatingKVCache metadata through the
    strict rotating path, not the ArraysCache-style token_count path."""
    paged_cache = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=100,
        model_name="test-model",
        initial_blocks=100,
    )
    mock_ssd = MagicMock()
    cache = BlockAwarePrefixCache(
        model=MockModel(num_layers=2),
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=mock_ssd,
    )

    block = paged_cache.allocate_block()
    block.block_hash = b"hash0"
    block.token_count = BLOCK_SIZE
    block.ref_count = 2
    block_table = BlockTable(
        request_id="req-001",
        block_ids=[block.block_id],
        num_tokens=BLOCK_SIZE,
    )

    kv_slice = (
        mx.ones((1, 2, BLOCK_SIZE, 8)),
        mx.ones((1, 2, BLOCK_SIZE, 8)),
    )
    rotating_real = (
        mx.ones((1, 2, WINDOW, 8)),
        mx.ones((1, 2, WINDOW, 8)),
    )
    metadata = {
        "model_name": "test-model",
        "num_layers": 2,
        "layer_cache_types": ["KVCache", "PrefillReadyRotatingKVCache"],
        "layer_meta_states": [
            (str(BLOCK_SIZE),),
            ("0", str(WINDOW), str(BLOCK_SIZE), str(WINDOW)),
        ],
    }
    mock_ssd.load_block_with_metadata.return_value = (
        [kv_slice, rotating_real],
        metadata,
    )

    result = cache.reconstruct_cache(block_table)

    assert result is not None
    assert isinstance(result[1], PrefillReadyRotatingKVCache)
    assert result[1].max_size == WINDOW
    assert result[1].offset == BLOCK_SIZE


def test_prefill_ready_rotating_name_detects_window_padding():
    paged_cache = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=100,
        model_name="test-model",
        initial_blocks=100,
    )
    mock_ssd = MagicMock()
    cache = BlockAwarePrefixCache(
        model=MockModel(num_layers=2),
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=mock_ssd,
    )
    block = paged_cache.allocate_block()
    block.block_hash = b"hash0"

    mock_ssd.load_block_with_metadata.return_value = (
        None,
        {
            "layer_cache_types": ["KVCache", "PrefillReadyRotatingKVCache"],
            "layer_meta_states": [
                (str(BLOCK_SIZE),),
                ("0", str(WINDOW), str(BLOCK_SIZE), str(WINDOW)),
            ],
        },
    )

    result = cache._detect_window_padding_from_blocks([block.block_id])

    assert result is not None
    assert result.has_rotating_layers()
    assert result.get_max_window_size() == WINDOW


def test_registry_rotating_family():
    assert CacheTypeRegistry.is_rotating_family("RotatingKVCache")
    assert CacheTypeRegistry.is_rotating_family("PrefillReadyRotatingKVCache")
    assert CacheTypeRegistry.is_rotating_family("BufferedRotatingKVCache")
    assert CacheTypeRegistry.is_rotating_family("BatchRotatingKVCache")
    assert not CacheTypeRegistry.is_rotating_family("KVCache")
    assert not CacheTypeRegistry.is_rotating_family("TurboQuantKVCache")
    assert not CacheTypeRegistry.is_rotating_family("ArraysCache")
    assert not CacheTypeRegistry.is_rotating_family("SomethingElse")
