# SPDX-License-Identifier: Apache-2.0
"""Regression tests for block-delta PoolingCache persistence."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from omlx.cache.paged_cache import BlockTable, PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.pooling_delta import (
    POOLING_CACHE_DELTA_CLASS,
    POOLING_CACHE_DELTA_FORMAT_VERSION,
    compact_pooling_cache_snapshot,
)
from omlx.cache.prefix_cache import BlockAwarePrefixCache

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

BLOCK_SIZE = 4
POOL_RATIO = 4
POOL_DIM = 8


class _MockModel:
    def __init__(self):
        self.layers = [MagicMock()]

    @property
    def args(self):
        args = MagicMock()
        args.num_hidden_layers = 1
        return args


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


def _delta_pooling_layer(
    token_count: int, *, include_overlap_state: bool = False
) -> list[dict]:
    layers = [_pooling_layer(token_count, include_overlap_state=include_overlap_state)]
    compact_pooling_cache_snapshot(layers, token_count, BLOCK_SIZE)
    return layers


def _make_cache(tmp_path, *, hot_cache_only: bool = True):
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
        hot_cache_only=hot_cache_only,
        expected_model_name="pooling-delta-test",
    )
    return (
        BlockAwarePrefixCache(
            model=_MockModel(),
            paged_cache_manager=paged,
            paged_ssd_cache_manager=ssd,
        ),
        ssd,
    )


def _wait_for_pending_writes(ssd: PagedSSDCacheManager) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with ssd._pending_write_hashes_lock:
            if not ssd._pending_write_hashes:
                return
        time.sleep(0.01)
    pytest.fail("Timed out waiting for SSD cache writes")


def test_compaction_is_linear_and_preserves_absolute_ranges():
    full_rows = 0
    delta_rows = 0
    for boundary in range(BLOCK_SIZE, 101 * BLOCK_SIZE, BLOCK_SIZE):
        layers = [_pooling_layer(boundary)]
        full_rows += layers[0]["state"][0][2].shape[1]
        compact_pooling_cache_snapshot(layers, boundary, BLOCK_SIZE)
        state = layers[0]["state"][0]
        start, end = layers[0]["pooling_delta_ranges"]["0"]
        assert (start, end) == (
            (boundary - BLOCK_SIZE) // POOL_RATIO,
            boundary // POOL_RATIO,
        )
        assert state[2].shape[1] == end - start == 1
        delta_rows += state[2].shape[1]

    assert full_rows == 5050
    assert delta_rows == 100


def test_mismatched_pool_length_keeps_legacy_full_snapshot():
    layers = [_pooling_layer(2 * BLOCK_SIZE)]
    compact_pooling_cache_snapshot(layers, BLOCK_SIZE, BLOCK_SIZE)
    assert "pooling_delta_ranges" not in layers[0]
    assert layers[0]["state"][0][2].shape[1] == 2


def test_compaction_preserves_overlap_state():
    layers = [_pooling_layer(2 * BLOCK_SIZE, include_overlap_state=True)]
    original = layers[0]["state"][0]
    compact_pooling_cache_snapshot(layers, 2 * BLOCK_SIZE, BLOCK_SIZE)
    compacted = layers[0]["state"][0]

    assert len(compacted) == 5
    assert compacted[2].shape[1] == 1
    assert compacted[3] is original[3]
    assert compacted[4] is original[4]


def test_boundary_snapshot_metadata_roundtrip(tmp_path):
    from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore

    store = BoundarySnapshotSSDStore(tmp_path)
    layers = [_pooling_layer(2 * BLOCK_SIZE)]
    saved = store.save(
        "req-delta",
        2 * BLOCK_SIZE,
        [object()],
        lambda _: (layers, None),
        block_size=BLOCK_SIZE,
    )
    assert saved is True
    restored = store.load("req-delta", 2 * BLOCK_SIZE)
    assert restored is not None
    assert restored[0]["pooling_delta_ranges"] == {"0": [1, 2]}
    assert restored[0]["state"][0][2].shape[1] == 1
    store.shutdown()


def test_v4_delta_blocks_restore_full_and_partial_prefix(tmp_path):
    from omlx.cache.paged_ssd_cache import _signature_cachelist_subtypes

    cache, ssd = _make_cache(tmp_path)
    num_blocks = 3
    tokens = list(range(num_blocks * BLOCK_SIZE))
    snapshots = {
        boundary: _delta_pooling_layer(boundary, include_overlap_state=True)
        for boundary in range(BLOCK_SIZE, len(tokens) + 1, BLOCK_SIZE)
    }
    table = cache.store_cache(
        "req-delta",
        tokens,
        [_pooling_layer(len(tokens))],
        boundary_snapshots=snapshots,
    )
    assert table is not None
    assert len(table.block_ids) == num_blocks

    for block_idx, block_id in enumerate(table.block_ids):
        block = cache.paged_cache.allocated_blocks[block_id]
        block_data, metadata = ssd.load_block_with_metadata(block.block_hash)
        assert block_data is not None and metadata is not None
        hot_entry = ssd._hot_cache_get(block.block_hash)
        assert hot_entry is not None
        assert (
            hot_entry["file_metadata"]["omlx_cache_format_version"]
            == POOLING_CACHE_DELTA_FORMAT_VERSION
        )
        marker = block_data[0][0]
        assert marker[0] == "__nstate__"
        assert marker[1] == POOLING_CACHE_DELTA_CLASS
        assert len(marker[2]) == 6
        assert marker[2][2].shape[1] == 1
        assert marker[2][5].tolist() == [block_idx, block_idx + 1]
        assert _signature_cachelist_subtypes(metadata.get("cache_signature", "")) == {
            "0": ["PoolingCache:5"]
        }

    restored = cache.reconstruct_cache(table)
    assert restored is not None
    pooling = restored[0].caches[0]
    expected = _pooling_layer(len(tokens))["state"][0][2]
    assert pooling.pooled.shape == expected.shape
    assert mx.max(mx.abs(pooling.pooled - expected)).item() == 0.0
    assert pooling.prev_win_kv is not None
    assert pooling.prev_win_gate is not None

    for block_id in table.block_ids[:2]:
        cache.paged_cache.allocated_blocks[block_id].ref_count += 1
    partial = BlockTable(
        request_id="req-delta-partial",
        block_ids=list(table.block_ids[:2]),
        num_tokens=2 * BLOCK_SIZE,
    )
    partial_restored = cache.reconstruct_cache(partial)
    assert partial_restored is not None
    assert partial_restored[0].caches[0].pooled.shape[1] == 2

    for block_id in (table.block_ids[0], table.block_ids[2]):
        cache.paged_cache.allocated_blocks[block_id].ref_count += 1
    gapped = BlockTable(
        request_id="req-delta-gapped",
        block_ids=[table.block_ids[0], table.block_ids[2]],
        num_tokens=2 * BLOCK_SIZE,
    )
    assert cache.reconstruct_cache(gapped) is None


def test_missing_intermediate_snapshot_stops_before_pooling_delta_gap(tmp_path):
    from omlx.cache.paged_ssd_cache import _signature_cachelist_subtypes

    cache, ssd = _make_cache(tmp_path)
    tokens = list(range(3 * BLOCK_SIZE))
    final_boundary = 3 * BLOCK_SIZE
    snapshots = {
        BLOCK_SIZE: _delta_pooling_layer(BLOCK_SIZE, include_overlap_state=True),
        final_boundary: _delta_pooling_layer(
            final_boundary, include_overlap_state=True
        ),
    }

    table = cache.store_cache(
        "req-delta-gap",
        tokens,
        [_pooling_layer(len(tokens), include_overlap_state=True)],
        boundary_snapshots=snapshots,
    )

    assert table is not None
    assert table.num_tokens == BLOCK_SIZE
    assert len(table.block_ids) == 1
    block = cache.paged_cache.allocated_blocks[table.block_ids[0]]
    _, metadata = ssd.load_block_with_metadata(block.block_hash)
    assert metadata is not None
    assert _signature_cachelist_subtypes(metadata.get("cache_signature", "")) == {
        "0": ["PoolingCache:5"]
    }
    assert cache.reconstruct_cache(table) is not None


def test_legacy_full_block_can_anchor_v4_delta_chain(tmp_path):
    cache, ssd = _make_cache(tmp_path)

    first = cache.store_cache(
        "req-legacy-base",
        list(range(BLOCK_SIZE)),
        [_pooling_layer(BLOCK_SIZE)],
        boundary_snapshots={BLOCK_SIZE: [_pooling_layer(BLOCK_SIZE)]},
    )
    assert first is not None
    first_block = cache.paged_cache.allocated_blocks[first.block_ids[0]]
    _, first_metadata = ssd.load_block_with_metadata(first_block.block_hash)
    assert first_metadata is not None
    first_entry = ssd._hot_cache_get(first_block.block_hash)
    assert first_entry is not None
    assert first_entry["file_metadata"]["omlx_cache_format_version"] == "3"

    tokens = list(range(3 * BLOCK_SIZE))
    snapshots = {
        BLOCK_SIZE: _delta_pooling_layer(BLOCK_SIZE, include_overlap_state=True),
        2
        * BLOCK_SIZE: _delta_pooling_layer(2 * BLOCK_SIZE, include_overlap_state=True),
        3
        * BLOCK_SIZE: _delta_pooling_layer(3 * BLOCK_SIZE, include_overlap_state=True),
    }
    mixed = cache.store_cache(
        "req-mixed-chain",
        tokens,
        [_pooling_layer(len(tokens))],
        boundary_snapshots=snapshots,
    )
    assert mixed is not None

    restored = cache.reconstruct_cache(mixed)
    assert restored is not None
    expected = _pooling_layer(len(tokens))["state"][0][2]
    assert mx.max(mx.abs(restored[0].caches[0].pooled - expected)).item() == 0.0


def test_live_pooling_signature_records_state_arity():
    from omlx.cache.paged_ssd_cache import cachelist_subtypes_from_cache_list
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    from mlx_lm.models.cache import CacheList, PoolingCache

    live = [CacheList(PoolingCache(ratio=POOL_RATIO))]
    assert cachelist_subtypes_from_cache_list(live) == {"0": ["PoolingCache:5"]}


def test_legacy_pooling_state_arity_is_rejected(tmp_path):
    from omlx.cache.paged_ssd_cache import _signature_cachelist_subtypes

    cache, ssd = _make_cache(tmp_path)
    table = cache.store_cache(
        "req-legacy-arity",
        list(range(BLOCK_SIZE)),
        [_pooling_layer(BLOCK_SIZE)],
        boundary_snapshots={BLOCK_SIZE: [_pooling_layer(BLOCK_SIZE)]},
    )
    assert table is not None
    block = cache.paged_cache.allocated_blocks[table.block_ids[0]]
    _, metadata = ssd.load_block_with_metadata(block.block_hash)
    assert metadata is not None
    assert _signature_cachelist_subtypes(metadata.get("cache_signature", "")) == {
        "0": ["PoolingCache:3"]
    }

    changed = ssd.set_expected_layer_signature(
        ["CacheList"],
        cachelist_subtypes={"0": ["PoolingCache:5"]},
    )
    assert changed is True
    assert cache.reconstruct_cache(table) is None


@pytest.mark.parametrize("hot_cache_only", [True, False], ids=["hot-cache", "ssd"])
def test_stale_pooling_tail_is_replaced_in_one_refill(tmp_path, caplog, hot_cache_only):
    from omlx.cache.paged_ssd_cache import _signature_cachelist_subtypes

    cache, ssd = _make_cache(tmp_path, hot_cache_only=hot_cache_only)
    tokens = list(range(4 * BLOCK_SIZE))
    final_boundary = len(tokens)
    snapshots = {
        BLOCK_SIZE: _delta_pooling_layer(BLOCK_SIZE, include_overlap_state=True),
        2 * BLOCK_SIZE: _delta_pooling_layer(2 * BLOCK_SIZE),
        3 * BLOCK_SIZE: _delta_pooling_layer(3 * BLOCK_SIZE),
        final_boundary: _delta_pooling_layer(
            final_boundary, include_overlap_state=True
        ),
    }

    try:
        table = cache.store_cache(
            "req-stale-tail",
            tokens,
            [_pooling_layer(len(tokens), include_overlap_state=True)],
            boundary_snapshots=snapshots,
            hot_cache_write_back=hot_cache_only,
        )
        assert table is not None
        _wait_for_pending_writes(ssd)

        changed = ssd.set_expected_layer_signature(
            ["CacheList"],
            cachelist_subtypes={"0": ["PoolingCache:5"]},
        )
        assert changed is True

        # The first block is compatible and the second block is stale. The
        # restore path forgets that first mismatch, leaving the stale tail
        # present so the next store must actively replace it.
        truncated = cache.reconstruct_cache(table)
        assert truncated is not None
        assert table.num_tokens == BLOCK_SIZE
        assert len(table.block_ids) == 1
        assert "CacheList sub composition at layer 0" in caplog.text

        good_snapshots = {
            boundary: _delta_pooling_layer(boundary, include_overlap_state=True)
            for boundary in range(BLOCK_SIZE, len(tokens) + 1, BLOCK_SIZE)
        }
        repaired = cache.store_cache(
            "req-stale-tail-repair",
            tokens,
            [_pooling_layer(len(tokens), include_overlap_state=True)],
            boundary_snapshots=good_snapshots,
            hot_cache_write_back=hot_cache_only,
        )
        assert repaired is not None
        _wait_for_pending_writes(ssd)

        signatures = []
        for block_id in repaired.block_ids:
            block = cache.paged_cache.allocated_blocks[block_id]
            _, metadata = ssd.load_block_with_metadata(block.block_hash)
            assert metadata is not None
            signatures.append(
                _signature_cachelist_subtypes(metadata.get("cache_signature", ""))
            )
        assert signatures == [
            {"0": ["PoolingCache:5"]},
            {"0": ["PoolingCache:5"]},
            {"0": ["PoolingCache:5"]},
            {"0": ["PoolingCache:5"]},
        ]
        assert cache.reconstruct_cache(repaired) is not None
    finally:
        ssd.close()
