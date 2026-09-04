# SPDX-License-Identifier: Apache-2.0
"""Per-member CacheList block storage (``__cache_list_pm__``) guards.

Mixed CacheList layers (inkling-style ``CacheList(KVCache, ArraysCache)``)
previously stored the FULL cumulative state of every member in every block —
quadratic in context length on the allocator pool and the SSD (issue #2546).
Per-member storage slices the sliceable KV member per block and keeps only
the boundary snapshot's small non-sliceable state, restoring linear cost.

Guards here:
1. Stored blocks are per-block sized (KV member holds BLOCK_SIZE tokens,
   not the cumulative prefix) and round-trip positionally.
2. Legacy cumulative blocks still restore with last-block semantics.
3. A chain mixing legacy and per-member blocks is rejected, not corrupted.
4. Member-filtered snapshots (blanked KV member) still store correctly.

Harness mirrors test_prefix_cache_cachelist_mixed.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import omlx.cache.prefix_cache as prefix_cache_module
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache, cachelist_pm_member_plan
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
    pos = mx.arange(seq_len, dtype=mx.float32).reshape(1, 1, seq_len, 1)
    keys = mx.broadcast_to(pos, (1, 2, seq_len, 8))
    values = keys + 1000.0
    return mx.contiguous(keys), mx.contiguous(values)


def _build_mixed_cachelist(seq_len):
    kv = KVCache()
    keys, values = _position_kv(seq_len)
    kv.update_and_fetch(keys, values)

    arrays = ArraysCache(size=4)
    for i, channels in enumerate(CONV_CHANNELS):
        arrays[i] = mx.full((1, 3, channels), seq_len + i / 10.0, dtype=mx.float32)

    cache_list = CacheList(kv, arrays)
    mx.eval([t for t in [keys, values] + list(arrays.cache) if t is not None])
    return cache_list


def _layer_dict(cache_list, blank_kv=False):
    handler = CacheTypeRegistry.get_handler_by_class_name("CacheList")
    state_dict = handler.extract_state(cache_list)
    state = list(state_dict["sub_states"])
    if blank_kv:
        # Mirrors Scheduler._extract_snapshot_cache_states: sliceable
        # members blanked, boundary members kept.
        state[0] = ()
    return {
        "state": state,
        "meta_state": (
            list(state_dict["sub_class_names"]),
            list(state_dict["sub_meta_states"]),
        ),
        "class_name": "CacheList",
        "cache_type": "CacheList",
    }


def _cache_data(seq_len, blank_kv=False):
    return [_layer_dict(_build_mixed_cachelist(seq_len), blank_kv=blank_kv)]


def _boundary_snapshots(num_blocks, blank_kv=False):
    return {
        BLOCK_SIZE * (i + 1): _cache_data(BLOCK_SIZE * (i + 1), blank_kv=blank_kv)
        for i in range(num_blocks)
    }


def _store_blocks(cache, num_blocks, request_id="req-pm", blank_kv=False):
    tokens = list(range(num_blocks * BLOCK_SIZE))
    return cache.store_cache(
        request_id,
        tokens,
        _cache_data(len(tokens)),
        boundary_snapshots=_boundary_snapshots(num_blocks, blank_kv=blank_kv),
    )


def _assert_restored(result, expected_seq_len):
    assert result is not None
    restored = result[0]
    assert type(restored).__name__ == "CacheList"
    kv = list(restored.caches)[0]
    keys = kv.state[0]
    assert keys.shape[2] == expected_seq_len
    expected_keys, _ = _position_kv(expected_seq_len)
    assert mx.max(mx.abs(keys - expected_keys)).item() == 0.0
    arrays = list(restored.caches)[1]
    for i, channels in enumerate(CONV_CHANNELS):
        slot = list(arrays.state)[i]
        assert tuple(slot.shape) == (1, 3, channels)
        assert mx.max(mx.abs(slot - (expected_seq_len + i / 10.0))).item() == 0.0


def test_plan_helper_classification():
    cases = {
        "mixed kv+arrays": (["KVCache", "ArraysCache"], True),
        "kv only": (["KVCache", "KVCache"], False),
        "arrays only": (["ArraysCache"], False),
        "pooling member": (["KVCache", "PoolingCache"], False),
        "no names": ([], False),
    }
    live = _build_mixed_cachelist(BLOCK_SIZE)
    handler = CacheTypeRegistry.get_handler_by_class_name("CacheList")
    kv_state, arrays_state = handler.extract_state(live)["sub_states"]
    states_by_class = {
        "KVCache": kv_state,
        "ArraysCache": arrays_state,
        "PoolingCache": arrays_state,
    }
    for name, (classes, eligible) in cases.items():
        states = [states_by_class[c] for c in classes]
        plan = cachelist_pm_member_plan(classes, states)
        assert (plan is not None) == eligible, name
        if plan is not None:
            assert plan == ["slice", "boundary"]


def test_blocks_stored_per_member_sized(tmp_path):
    """The core assertion: block payloads hold per-block KV slices, not the
    cumulative prefix — and load re-tags them as per-member."""
    cache, ssd = _make_cache(tmp_path)
    num_blocks = 3
    table = _store_blocks(cache, num_blocks)
    assert table is not None
    assert len(table.block_ids) == num_blocks

    for idx, bid in enumerate(table.block_ids):
        block = cache.paged_cache.allocated_blocks[bid]
        payload, _meta = ssd.load_block_with_metadata(block.block_hash)
        assert payload is not None
        layer = payload[0]
        assert (
            isinstance(layer, tuple)
            and len(layer) == 2
            and layer[0] == "__cache_list_pm__"
        ), f"block {idx} not per-member tagged: {type(layer)}"
        subs = layer[1]
        kv_keys = subs[0][0]
        assert kv_keys.shape[2] == BLOCK_SIZE, (
            f"block {idx} KV member holds {kv_keys.shape[2]} tokens — "
            f"cumulative storage leaked back in"
        )


def test_pm_multiblock_roundtrip(tmp_path):
    cache, _ = _make_cache(tmp_path)
    table = _store_blocks(cache, num_blocks=3)
    _assert_restored(cache.reconstruct_cache(table), expected_seq_len=3 * BLOCK_SIZE)


def test_pm_partial_prefix_roundtrip(tmp_path):
    from omlx.cache.paged_cache import BlockTable

    cache, _ = _make_cache(tmp_path)
    table = _store_blocks(cache, num_blocks=3, request_id="req-part")
    for bid in table.block_ids[:2]:
        cache.paged_cache.allocated_blocks[bid].ref_count += 1
    partial = BlockTable(
        request_id="req-part-restore",
        block_ids=list(table.block_ids[:2]),
        num_tokens=2 * BLOCK_SIZE,
    )
    _assert_restored(cache.reconstruct_cache(partial), expected_seq_len=2 * BLOCK_SIZE)


def test_legacy_cumulative_blocks_still_restore(tmp_path, monkeypatch):
    """Blocks produced by the legacy cumulative path (pm plan ineligible)
    keep last-block restore semantics."""
    monkeypatch.setattr(
        prefix_cache_module, "cachelist_pm_member_plan", lambda *a, **k: None
    )
    cache, ssd = _make_cache(tmp_path)
    table = _store_blocks(cache, num_blocks=3, request_id="req-legacy")
    assert table is not None

    block = cache.paged_cache.allocated_blocks[table.block_ids[0]]
    payload, _ = ssd.load_block_with_metadata(block.block_hash)
    assert isinstance(payload[0], list), "legacy blocks must stay untagged"

    _assert_restored(cache.reconstruct_cache(table), expected_seq_len=3 * BLOCK_SIZE)


def test_mixed_format_chain_rejected(tmp_path, monkeypatch):
    """Legacy blocks + per-member blocks in one chain must reject (miss),
    never concatenate cumulative KV into a duplicated sequence."""
    cache, _ = _make_cache(tmp_path)

    monkeypatch.setattr(
        prefix_cache_module, "cachelist_pm_member_plan", lambda *a, **k: None
    )
    table = _store_blocks(cache, num_blocks=2, request_id="req-mix")
    assert table is not None
    monkeypatch.undo()

    # Extend the same request with two more blocks — now stored per-member.
    tokens = list(range(4 * BLOCK_SIZE))
    table = cache.store_cache(
        "req-mix",
        tokens,
        _cache_data(len(tokens)),
        boundary_snapshots=_boundary_snapshots(4),
    )
    assert table is not None
    assert len(table.block_ids) == 4

    assert cache.reconstruct_cache(table) is None


def test_filtered_snapshots_store_correctly(tmp_path):
    """Snapshots with blanked KV members (as produced by
    _extract_snapshot_cache_states) still yield correct per-member blocks —
    KV comes from the live cache, conv state from the snapshot."""
    cache, _ = _make_cache(tmp_path)
    table = _store_blocks(cache, num_blocks=3, request_id="req-blank", blank_kv=True)
    assert table is not None
    _assert_restored(cache.reconstruct_cache(table), expected_seq_len=3 * BLOCK_SIZE)


# ---------------------------------------------------------------------------
# #2550 review fixes
# ---------------------------------------------------------------------------


def test_missing_middle_snapshot_truncates_store(tmp_path):
    """A missing middle boundary snapshot must truncate the store at that
    boundary — never produce a pm/placeholder/pm chain that restores fewer
    KV tokens than block_table.num_tokens claims."""
    cache, _ = _make_cache(tmp_path)
    num_blocks = 3
    tokens = list(range(num_blocks * BLOCK_SIZE))
    snapshots = _boundary_snapshots(num_blocks)
    del snapshots[2 * BLOCK_SIZE]  # omit the middle boundary

    table = cache.store_cache(
        "req-gap",
        tokens,
        _cache_data(len(tokens)),
        boundary_snapshots=snapshots,
    )
    assert table is not None
    # Only the first block (boundary 4) is persisted.
    assert len(table.block_ids) == 1
    assert table.num_tokens == BLOCK_SIZE
    _assert_restored(cache.reconstruct_cache(table), expected_seq_len=BLOCK_SIZE)


def test_restore_rejects_short_kv_chain(tmp_path, monkeypatch):
    """Reviewer repro: pm / placeholder / pm chain. The placeholder block is
    skipped by the collector, so restored KV would be 8 tokens against
    num_tokens=12 — the length check must reject the cache."""
    real_plan = prefix_cache_module.cachelist_pm_member_plan
    calls = {"n": 0}

    def first_call_none(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return None  # defeats pm_layers_present -> no store truncation
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(
        prefix_cache_module, "cachelist_pm_member_plan", first_call_none
    )

    cache, _ = _make_cache(tmp_path)
    num_blocks = 3
    tokens = list(range(num_blocks * BLOCK_SIZE))
    snapshots = _boundary_snapshots(num_blocks)
    del snapshots[2 * BLOCK_SIZE]  # middle block becomes a placeholder

    table = cache.store_cache(
        "req-short",
        tokens,
        _cache_data(len(tokens)),
        boundary_snapshots=snapshots,
    )
    assert table is not None
    assert len(table.block_ids) == 3

    assert cache.reconstruct_cache(table) is None


def test_legacy_blocks_swept_on_pm_expectation(tmp_path, monkeypatch):
    """Upgrade path (#2550 review): pre-upgrade legacy blocks must be
    invalidated by the layout-aware signature, so a post-upgrade store
    cannot recreate a mixed chain via token-hash dedup."""
    from omlx.cache.paged_ssd_cache import cachelist_subtypes_from_cache_list

    # The live-model expectation now carries the layout token.
    live = [_build_mixed_cachelist(seq_len=4)]
    assert cachelist_subtypes_from_cache_list(live) == {
        "0": ["KVCache", "ArraysCache:4", "@pm"]
    }

    # Pre-upgrade store: legacy cumulative blocks (no @pm stamp).
    monkeypatch.setattr(
        prefix_cache_module, "cachelist_pm_member_plan", lambda *a, **k: None
    )
    cache, ssd = _make_cache(tmp_path)
    table = cache.store_cache(
        "req-old",
        list(range(2 * BLOCK_SIZE)),
        _cache_data(2 * BLOCK_SIZE),
        boundary_snapshots=_boundary_snapshots(2),
    )
    assert table is not None
    monkeypatch.undo()

    # "Restart on upgraded code": live expectation adopts the pm layout.
    changed = ssd.set_expected_layer_signature(
        ["CacheList"],
        cachelist_subtypes={"0": ["KVCache", "ArraysCache:4", "@pm"]},
    )
    assert changed is True
    ssd.invalidate_stale_layer_signature()

    # Legacy blocks are no longer restorable...
    assert cache.reconstruct_cache(table) is None

    # ...and a fresh store of the same tokens yields a pure pm chain that
    # restores — the mixed chain cannot be recreated.
    table2 = cache.store_cache(
        "req-new",
        list(range(2 * BLOCK_SIZE)),
        _cache_data(2 * BLOCK_SIZE),
        boundary_snapshots=_boundary_snapshots(2),
    )
    assert table2 is not None
    assert len(table2.block_ids) == 2
    _assert_restored(cache.reconstruct_cache(table2), expected_seq_len=2 * BLOCK_SIZE)


def test_decode_snapshot_fallback_filters_kv(tmp_path):
    """In-memory decode-snapshot fallback (#2550 review): the stored value
    must be pre-extracted with the KV member blanked, not the raw CacheList
    retaining the full KV prefix."""
    from types import SimpleNamespace

    from omlx.scheduler import Scheduler

    live = _build_mixed_cachelist(seq_len=BLOCK_SIZE)

    stub = SimpleNamespace(
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
    stub._eval_snapshot_cache = lambda caches: None

    value = Scheduler._decode_boundary_snapshot_value(
        stub, [live], BLOCK_SIZE, BLOCK_SIZE
    )

    assert isinstance(value, tuple)
    assert value[0] == Scheduler._PREFILL_SNAPSHOT_MARKER
    extracted = value[1]
    assert extracted[0]["state"][0] == (), "KV member must be blanked"
    conv_slot0 = extracted[0]["state"][1][0]
    assert mx.max(mx.abs(conv_slot0 - BLOCK_SIZE)).item() == 0.0


def test_store_refuses_blanked_member_source(tmp_path):
    """Parser-stop regression (#2550 follow-up): a store source whose KV
    member is still blanked (member-filtered snapshot promoted without
    refill) must refuse to store entirely. The legacy branch used to drop
    the blanked sub silently, and the short-payload blocks then poisoned
    the prefix for the whole session via token-hash dedup."""
    cache, ssd = _make_cache(tmp_path)
    num_blocks = 3
    tokens = list(range(num_blocks * BLOCK_SIZE))

    table = cache.store_cache(
        "req-blank-source",
        tokens,
        _cache_data(len(tokens), blank_kv=True),
        boundary_snapshots=_boundary_snapshots(num_blocks, blank_kv=True),
    )
    assert table is None

    # The refused store must leave nothing behind: a proper store of the
    # same tokens builds a clean pm chain that restores.
    table2 = _store_blocks(cache, num_blocks, request_id="req-proper")
    assert table2 is not None
    assert len(table2.block_ids) == num_blocks
    for bid in table2.block_ids:
        block = cache.paged_cache.allocated_blocks[bid]
        payload, _ = ssd.load_block_with_metadata(block.block_hash)
        layer = payload[0]
        assert (
            isinstance(layer, tuple) and layer[0] == "__cache_list_pm__"
        ), "refused store must not leave short-payload legacy blocks behind"
    _assert_restored(
        cache.reconstruct_cache(table2), expected_seq_len=num_blocks * BLOCK_SIZE
    )
