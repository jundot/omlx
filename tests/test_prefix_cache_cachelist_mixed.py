# SPDX-License-Identifier: Apache-2.0
"""Mixed CacheList(KVCache, ArraysCache) prefix/SSD round-trip guards.

Inkling-style models return ``CacheList(KVCache(), ArraysCache(4))`` for
every layer (hybrid attention + 4 short-conv slots). The store path
decides slicing per LAYER: ``all_sub_sliceable`` is False whenever any
sub-state's first element is not 4D (ArraysCache conv state is 3D), so
every block stores the FULL cumulative state of ALL subs at that block's
boundary (from boundary snapshots). The restore path decides per SUB:
only ArraysCache/Pooling/rotating subs take the last block, while a
KVCache sub is concatenated across blocks as if the blocks held per-block
slices. Concatenating cumulative snapshots duplicates the KV sequence
(4+8+12 tokens instead of 12) and corrupts positions.

Existing CacheList users never hit this: GLM/deepseek_v32/longcat are
KVCache+KVCache (all_sub_sliceable=True, real per-block slices stored),
DeepSeek-V4 is RotatingKVCache+PoolingCache (every sub takes last block).
qwen3.5/3.6 mix ArraysCache and KVCache at the LAYER level (bare caches,
no CacheList), which routes per-layer handlers and never enters the
CacheList branch.

These tests build production-shaped layer dicts (via CacheListHandler
extract, matching scheduler._extract_cache_states output — note: no
top-level ``sub_class_names`` key) and round-trip them through a real
hot-cache-only PagedSSDCacheManager.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omlx.cache.observability import BoundarySnapshotDiagnostics
from omlx.cache.paged_cache import BlockTable, PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.cache.type_registry import CacheTypeRegistry

try:
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

BLOCK_SIZE = 4
NUM_LAYERS = 1
# Inkling conv slots: k/v sconv operate on n_kv*head_dim channels,
# attn/mlp sconv on hidden — per-slot channel counts differ.
CONV_CHANNELS = (16, 16, 32, 32)


class MockModel:
    def __init__(self, num_layers: int = NUM_LAYERS):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        a = MagicMock()
        a.num_hidden_layers = self._num_layers
        return a


def _make_cache(tmp_path):
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
        expected_model_name="test-model",
    )
    cache = BlockAwarePrefixCache(
        model=MockModel(),
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=ssd,
    )
    return cache, ssd


def _position_kv(seq_len):
    """KV tensors whose value at position p equals p — duplication shows."""
    pos = mx.arange(seq_len, dtype=mx.float32).reshape(1, 1, seq_len, 1)
    keys = mx.broadcast_to(pos, (1, 2, seq_len, 8))
    values = keys + 1000.0
    return mx.contiguous(keys), mx.contiguous(values)


def _build_mixed_cachelist(seq_len, none_slots=()):
    """A real CacheList(KVCache, ArraysCache(4)) advanced to seq_len tokens.

    Conv slot i is filled with ``seq_len + i / 10`` so each boundary's
    snapshot is distinguishable; slots listed in none_slots stay None.
    """
    kv = KVCache()
    keys, values = _position_kv(seq_len)
    kv.update_and_fetch(keys, values)

    arrays = ArraysCache(size=4)
    for i, channels in enumerate(CONV_CHANNELS):
        if i in none_slots:
            continue
        arrays[i] = mx.full((1, 3, channels), seq_len + i / 10.0, dtype=mx.float32)

    cache_list = CacheList(kv, arrays)
    mx.eval([t for t in [keys, values] + list(arrays.cache) if t is not None])
    return cache_list


def _layer_dict(cache_list):
    """Production-shaped layer dict (scheduler._extract_cache_states)."""
    handler = CacheTypeRegistry.get_handler_by_class_name("CacheList")
    state_dict = handler.extract_state(cache_list)
    return {
        "state": list(state_dict["sub_states"]),
        "meta_state": (
            list(state_dict["sub_class_names"]),
            list(state_dict["sub_meta_states"]),
        ),
        "class_name": "CacheList",
        "cache_type": "CacheList",
    }


def _cache_data(seq_len, none_slots=()):
    return [_layer_dict(_build_mixed_cachelist(seq_len, none_slots))]


def _store_blocks(cache, num_blocks, request_id="req-mixed"):
    """Store num_blocks blocks with cumulative boundary snapshots."""
    tokens = list(range(num_blocks * BLOCK_SIZE))
    boundary_snapshots = {
        BLOCK_SIZE * (i + 1): _cache_data(BLOCK_SIZE * (i + 1))
        for i in range(num_blocks)
    }
    table = cache.store_cache(
        request_id,
        tokens,
        _cache_data(len(tokens)),
        boundary_snapshots=boundary_snapshots,
    )
    return table


def _assert_restored(result, expected_seq_len):
    """Restored layer must be a CacheList holding exactly expected_seq_len
    KV tokens (position-encoded) and the conv snapshot of that boundary."""
    assert result is not None
    assert len(result) == NUM_LAYERS
    restored = result[0]
    assert type(restored).__name__ == "CacheList"
    sub_caches = list(restored.caches)
    assert len(sub_caches) == 2

    kv = sub_caches[0]
    kv_state = kv.state
    keys = kv_state[0]
    assert keys.shape[2] == expected_seq_len, (
        f"restored KV holds {keys.shape[2]} tokens, "
        f"expected {expected_seq_len} (cumulative-snapshot duplication?)"
    )
    expected_keys, expected_values = _position_kv(expected_seq_len)
    assert mx.max(mx.abs(keys - expected_keys)).item() == 0.0
    assert mx.max(mx.abs(kv_state[1] - expected_values)).item() == 0.0

    arrays = sub_caches[1]
    slots = list(arrays.state)
    assert len(slots) == 4
    for i, (slot, channels) in enumerate(zip(slots, CONV_CHANNELS)):
        assert slot is not None
        assert slot.dtype == mx.float32
        assert tuple(slot.shape) == (1, 3, channels)
        assert (
            mx.max(mx.abs(slot - (expected_seq_len + i / 10.0))).item() == 0.0
        ), f"conv slot {i} does not match boundary {expected_seq_len} snapshot"


def test_single_block_roundtrip(tmp_path):
    """One block: last-block state stored and restored verbatim."""
    cache, _ = _make_cache(tmp_path)
    table = _store_blocks(cache, num_blocks=1, request_id="req-single")
    assert table is not None
    assert len(table.block_ids) == 1

    result = cache.reconstruct_cache(table)
    _assert_restored(result, expected_seq_len=BLOCK_SIZE)


def test_multiblock_restore_no_kv_duplication(tmp_path):
    """G1 core: 3 cumulative-snapshot blocks must restore to the LAST
    boundary's state (12 tokens), not the concatenation of all three
    cumulative KV snapshots (4+8+12 = 24 tokens)."""
    cache, _ = _make_cache(tmp_path)
    table = _store_blocks(cache, num_blocks=3)
    assert table is not None
    assert len(table.block_ids) == 3

    result = cache.reconstruct_cache(table)
    _assert_restored(result, expected_seq_len=3 * BLOCK_SIZE)


def test_partial_prefix_restores_matched_boundary(tmp_path):
    """Restoring only the first 2 of 3 blocks must yield block 2's
    cumulative boundary state (8 tokens) for ALL subs."""
    cache, _ = _make_cache(tmp_path)
    table = _store_blocks(cache, num_blocks=3, request_id="req-partial")
    assert table is not None
    assert len(table.block_ids) == 3

    for bid in table.block_ids[:2]:
        cache.paged_cache.allocated_blocks[bid].ref_count += 1
    partial = BlockTable(
        request_id="req-partial-restore",
        block_ids=list(table.block_ids[:2]),
        num_tokens=2 * BLOCK_SIZE,
    )
    result = cache.reconstruct_cache(partial)
    _assert_restored(result, expected_seq_len=2 * BLOCK_SIZE)


def test_block_signature_stamps_sub_composition(tmp_path):
    """Saved mixed-CacheList blocks stamp their sub composition (incl.
    ArraysCache slot count) into the compatibility signature."""
    import json

    from omlx.cache.paged_ssd_cache import _signature_cachelist_subtypes

    cache, ssd = _make_cache(tmp_path)
    table = _store_blocks(cache, num_blocks=1, request_id="req-sig")
    assert table is not None

    block = cache.paged_cache.allocated_blocks[table.block_ids[0]]
    _, meta = ssd.load_block_with_metadata(block.block_hash)
    assert meta is not None
    subtypes = _signature_cachelist_subtypes(meta.get("cache_signature", ""))
    assert subtypes == {"0": ["KVCache", "ArraysCache:4", "@pm"]}
    # The flat type list stays "CacheList" (dispatch strings unchanged).
    types = meta["layer_cache_types"]
    if isinstance(types, str):
        types = json.loads(types)
    assert list(types) == ["CacheList"]


def test_live_subtypes_descriptor_matches_block_stamp():
    """cachelist_subtypes_from_cache_list (expectation side) must produce
    the same descriptor the save path stamps from block payloads."""
    from omlx.cache.paged_ssd_cache import cachelist_subtypes_from_cache_list

    live = [_build_mixed_cachelist(seq_len=4)]
    assert cachelist_subtypes_from_cache_list(live) == {
        "0": ["KVCache", "ArraysCache:4", "@pm"]
    }
    # KVCache-only CacheList layers are not stamped (GLM/deepseek_v32
    # signatures stay byte-identical to the previous format).
    assert cachelist_subtypes_from_cache_list([CacheList(KVCache(), KVCache())]) is (
        None
    )


def test_stale_sub_composition_swept(tmp_path):
    """A stored block whose ArraysCache slot count disagrees with the live
    model expectation must be swept, not restored into an IndexError."""
    cache, ssd = _make_cache(tmp_path)
    table = _store_blocks(cache, num_blocks=1, request_id="req-stale")
    assert table is not None

    # Live model now expects 2 conv slots per layer (composition changed).
    changed = ssd.set_expected_layer_signature(
        ["CacheList"],
        cachelist_subtypes={"0": ["KVCache", "ArraysCache:2"]},
    )
    assert changed is True
    # Hot-cache-only managers keep no disk index to sweep; the per-block
    # signature gate in reconstruct_cache must reject the stale block.
    ssd.invalidate_stale_layer_signature()
    assert cache.reconstruct_cache(table) is None

    # Matching expectation keeps blocks restorable.
    cache2, ssd2 = _make_cache(tmp_path / "match")
    table2 = _store_blocks(cache2, num_blocks=1, request_id="req-match")
    changed = ssd2.set_expected_layer_signature(
        ["CacheList"],
        cachelist_subtypes={"0": ["KVCache", "ArraysCache:4", "@pm"]},
    )
    assert changed is True
    assert ssd2.invalidate_stale_layer_signature() == 0
    _assert_restored(cache2.reconstruct_cache(table2), expected_seq_len=BLOCK_SIZE)


def test_prefill_snapshot_decoupled_from_live_cache():
    """In-memory prefill boundary snapshots must capture the state AT the
    boundary. Storing the live cache objects aliased every boundary to
    the prefill's final state (KVCache mutates its buffer in place)."""
    from types import SimpleNamespace

    from omlx.scheduler import Scheduler

    live = _build_mixed_cachelist(seq_len=BLOCK_SIZE)

    stub = SimpleNamespace(
        block_aware_cache=object(),
        config=SimpleNamespace(paged_cache_block_size=BLOCK_SIZE),
        model=SimpleNamespace(),
        _cache_list_needs_boundary_snapshot=lambda cache: True,
        _boundary_cache_snapshots={},
        _boundary_snapshot_store=None,
        _boundary_snapshot_diagnostics=BoundarySnapshotDiagnostics(),
        _boundary_snapshot_required=False,
        _stream=mx.default_stream(mx.default_device()),
        _PREFILL_SNAPSHOT_MARKER=Scheduler._PREFILL_SNAPSHOT_MARKER,
    )
    stub._extract_cache_states = lambda caches: Scheduler._extract_cache_states(
        stub, caches
    )
    stub._extract_snapshot_cache_states = (
        lambda caches: Scheduler._extract_snapshot_cache_states(stub, caches)
    )
    stub._extract_prefill_snapshot_states = (
        lambda caches: Scheduler._extract_prefill_snapshot_states(stub, caches)
    )
    stub._prefill_snapshot_value = lambda caches: Scheduler._prefill_snapshot_value(
        stub, caches
    )
    stub._enable_mtp_boundary_alignment = (
        lambda: Scheduler._enable_mtp_boundary_alignment(stub)
    )
    stub._eval_snapshot_cache = lambda caches: None

    Scheduler._on_prefill_boundary_snapshot(stub, "req-alias", [live], BLOCK_SIZE)

    # Prefill continues: the live cache doubles its sequence and the conv
    # slots move on.
    keys, values = _position_kv(BLOCK_SIZE)
    live.caches[0].update_and_fetch(keys + 100.0, values + 100.0)
    for i, channels in enumerate(CONV_CHANNELS):
        live.caches[1][i] = mx.full((1, 3, channels), -1.0, dtype=mx.float32)

    stored = stub._boundary_cache_snapshots["req-alias"][BLOCK_SIZE]
    assert isinstance(stored, tuple)
    assert stored[0] == Scheduler._PREFILL_SNAPSHOT_MARKER
    extracted = stored[1]
    # Per-member filtering blanks the sliceable KV member — snapshots only
    # need the non-sliceable state; the store path slices KV from the live
    # cache. The conv slots remain the aliasing guard: they must hold the
    # boundary's values even after the live cache moves on.
    kv_state = extracted[0]["state"][0]
    assert kv_state == (), (
        "pm-eligible snapshot should blank the sliceable KV member, "
        f"got {kv_state!r}"
    )
    conv_slot0 = extracted[0]["state"][1][0]
    assert mx.max(mx.abs(conv_slot0 - BLOCK_SIZE)).item() == 0.0


def test_boundary_store_mixed_cachelist_roundtrip(tmp_path):
    """BoundarySnapshotSSDStore round-trips a mixed CacheList layer:
    nested shape, None conv slots, and fp32 dtype all preserved."""
    from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore

    store = BoundarySnapshotSSDStore(base_dir=tmp_path)

    keys, values = _position_kv(8)
    c0 = mx.full((1, 3, 16), 8.0, dtype=mx.float32)
    c2 = mx.full((1, 3, 32), 8.2, dtype=mx.float32)
    mx.eval(keys, values, c0, c2)

    extracted = [
        {
            "state": [(keys, values), [c0, None, c2, None]],
            "meta_state": (["KVCache", "ArraysCache"], [("8",), ()]),
            "class_name": "CacheList",
            "cache_type": "CacheList",
        }
    ]
    tensors_raw, metadata = store._serialize_extracted(
        extracted, request_id="req-bss", token_count=8
    )
    result = store._deserialize(tensors_raw, metadata)
    assert result is not None and len(result) == 1
    state = result[0]["state"]
    assert isinstance(state, list) and len(state) == 2
    kv_sub, arrays_sub = state[0], state[1]
    assert mx.max(mx.abs(kv_sub[0] - keys)).item() == 0.0
    assert arrays_sub[1] is None and arrays_sub[3] is None
    assert arrays_sub[0].dtype == mx.float32
    assert mx.max(mx.abs(arrays_sub[0] - c0)).item() == 0.0
    assert tuple(arrays_sub[2].shape) == (1, 3, 32)

    # Tensor-less CacheList layer (empty KV + untouched conv slots) is
    # recorded as state-less instead of a phantom "has_state" entry.
    empty = [
        {
            "state": [(), [None, None, None, None]],
            "meta_state": (["KVCache", "ArraysCache"], [(), ()]),
            "class_name": "CacheList",
            "cache_type": "CacheList",
        }
    ]
    tensors_raw2, metadata2 = store._serialize_extracted(
        empty, request_id="req-bss-empty", token_count=0
    )
    assert not tensors_raw2
    import json as _json

    info = _json.loads(metadata2["layer_info"])[0]
    assert info["has_state"] == "false"

    store.shutdown()


def test_arrays_cache_extract_none_guard():
    """Extract from an ArraysCache with untouched (None) slots — the state
    of a request aborted before its first forward — must not crash.
    filter/extend/merge already tolerate None slots; extract lacked the
    guard until the omlx patch."""
    from omlx.patches.arrays_cache_extract import (
        apply_arrays_cache_extract_guard,
    )

    assert apply_arrays_cache_extract_guard() is True

    ac = ArraysCache(size=4)
    ac[0] = mx.ones((2, 3, 8))
    out = ac.extract(1)
    assert out.cache[0].shape == (1, 3, 8)
    assert out.cache[1] is None
    assert out.cache[2] is None

    all_none = ArraysCache(size=4).extract(0)
    assert all(slot is None for slot in all_none.cache)


def test_none_conv_slots_roundtrip(tmp_path):
    """Untouched (None) ArraysCache slots survive the SSD round-trip as
    None instead of crashing or materializing placeholder tensors."""
    cache, _ = _make_cache(tmp_path)
    tokens = list(range(BLOCK_SIZE))
    table = cache.store_cache(
        "req-none-slots", tokens, _cache_data(BLOCK_SIZE, none_slots=(1, 3))
    )
    assert table is not None

    result = cache.reconstruct_cache(table)
    assert result is not None
    restored = result[0]
    assert type(restored).__name__ == "CacheList"
    slots = list(restored.caches[1].state)
    assert slots[1] is None
    assert slots[3] is None
    assert slots[0] is not None and slots[2] is not None
    kv_state = restored.caches[0].state
    assert kv_state[0].shape[2] == BLOCK_SIZE
