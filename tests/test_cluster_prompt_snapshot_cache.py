# SPDX-License-Identifier: Apache-2.0
"""SSD prompt-cache snapshots must round-trip non-sliceable state, keep KV as
one linear chain of slabs, and stay consistent across ranks that see the same
requests."""

import json
import threading
import time

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    CacheList,
    KVCache,
    RotatingKVCache,
    load_prompt_cache,
)

from omlx.cluster.performance import DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES
from omlx.cluster.prompt_snapshot_cache import (
    PoolingCacheDeltaSnapshot,
    SSDPromptSnapshotStore,
    agreed_boundary,
    candidate_boundaries,
)
from omlx.patches.deepseek_v4.cache_extras import PoolingCache

MODEL = ("model-path", None, None)
STEP = 2048


def _kv(layers=1, steps=2):
    """Populated KV caches: an empty cache has no state to serialise."""

    caches = [KVCache() for _ in range(layers)]
    for _ in range(steps):
        k = mx.random.normal((1, 2, 1, 4))
        v = mx.random.normal((1, 2, 1, 4))
        for cache in caches:
            cache.update_and_fetch(k, v)
    return caches


def _advance(caches, count):
    for _ in range(count):
        k = mx.random.normal((1, 2, 1, 4))
        v = mx.random.normal((1, 2, 1, 4))
        for cache in caches:
            if isinstance(cache, ArraysCache):
                cache[0] = k  # a recurrent state slot, overwritten each step
            else:
                cache.update_and_fetch(k, v)


def _feed(cache, count):
    cache.update_and_fetch(
        mx.random.normal((1, 2, count, 4)), mx.random.normal((1, 2, count, 4))
    )


def _rotating_and_gdn():
    """A sliding window plus a recurrent state: neither can be sliced."""

    rot = RotatingKVCache(max_size=8)
    gdn = ArraysCache(size=1)
    _advance([rot, gdn], 20)
    return [rot, gdn]


def _pooling(ratio=4, tokens=10, dim=8, with_prev=True):
    """A pooling cache driven through its own accumulate/pool surface."""

    cache = PoolingCache(ratio)
    kv = mx.random.normal((1, tokens, dim))
    gate = mx.random.normal((1, tokens, dim))
    ready_kv, ready_gate, _ = cache.accumulate_windows(kv, gate, 0)
    windows = ready_kv.shape[1] // ratio
    if windows > 0:
        cache.update_and_fetch(mx.random.normal((1, windows, dim)))
        if with_prev:
            cache.store_prev(
                ready_kv.reshape(1, windows, ratio, dim),
                ready_gate.reshape(1, windows, ratio, dim),
                0,
            )
    return cache


def _feed_pooling(cache, count, *, offset=0, dim=8, with_prev=True):
    """Advance one PoolingCache without replacing its append-only history."""

    kv = mx.random.normal((1, count, dim))
    gate = mx.random.normal((1, count, dim))
    ready_kv, ready_gate, _ = cache.accumulate_windows(kv, gate, offset)
    windows = ready_kv.shape[1] // cache.ratio
    if windows:
        cache.update_and_fetch(mx.random.normal((1, windows, dim)))
        if with_prev:
            cache.store_prev(
                ready_kv.reshape(1, windows, cache.ratio, dim),
                ready_gate.reshape(1, windows, cache.ratio, dim),
                0,
            )


def _assert_pooling_equal(restored, original):
    assert type(restored).__name__ == "PoolingCache"
    assert restored.ratio == original.ratio
    assert restored.remainder == original.remainder
    for got, want in zip(restored.state, original.state):
        assert (got is None) == (want is None)
        if want is not None:
            assert mx.array_equal(got, want)


def test_candidate_boundaries_are_aligned_and_longest_first():
    assert candidate_boundaries(5000, 2048) == (4096, 2048)
    assert candidate_boundaries(2048, 2048) == (2048,)
    assert candidate_boundaries(1000, 2048) == ()
    assert candidate_boundaries(0, 2048) == ()


def test_a_rotating_and_recurrent_state_round_trips(tmp_path):
    store = SSDPromptSnapshotStore(tmp_path, step=STEP)
    tokens = list(range(STEP))
    caches = _rotating_and_gdn()
    rot_state = caches[0].state
    gdn_state = caches[1].state

    assert store.put(MODEL, tokens, caches)
    restored = store.load(MODEL, tokens, STEP)

    assert restored is not None
    assert [type(c).__name__ for c in restored] == ["RotatingKVCache", "ArraysCache"]
    # The window offset and the recurrent slot survive the round trip.
    assert restored[0].offset == caches[0].offset
    assert mx.array_equal(restored[0].state[0], rot_state[0])
    assert mx.array_equal(restored[1].state[0], gdn_state[0])


def test_kv_segments_reassemble_across_the_chain(tmp_path):
    """The local paged policy ported: each file holds one step-sized slab and
    the chain concatenates back to the exact full KV."""

    store = SSDPromptSnapshotStore(tmp_path, step=4)
    tokens = list(range(12))
    kv = KVCache()
    for boundary in (4, 8, 12):
        _feed(kv, 4)
        assert store.put(MODEL, tokens[:boundary], [kv])

    restored = store.load(MODEL, tokens, 12)
    assert restored is not None
    assert type(restored[0]).__name__ == "KVCache"
    assert restored[0].offset == 12
    assert mx.array_equal(restored[0].state[0], kv.state[0])
    assert mx.array_equal(restored[0].state[1], kv.state[1])

    interior = store.load(MODEL, tokens, 8)
    assert interior is not None
    assert mx.array_equal(interior[0].state[0], kv.state[0][..., :8, :])

    # One slab per file, not one cumulative copy per boundary.
    sizes = [p.stat().st_size for p in tmp_path.glob("*.safetensors")]
    assert len(sizes) == 3
    assert max(sizes) < 2 * min(sizes)


def test_a_zero_width_value_cache_segments_cleanly(tmp_path):
    """GLM's MLA-style caches keep all data in the keys and a zero-width
    values half; the segment layout must carry and rebuild it exactly."""

    store = SSDPromptSnapshotStore(tmp_path, step=4)
    tokens = list(range(8))
    mla = KVCache()
    for boundary in (4, 8):
        mla.update_and_fetch(mx.random.normal((1, 2, 4, 4)), mx.zeros((1, 2, 4, 0)))
        assert store.put(MODEL, tokens[:boundary], [mla])

    restored = store.load(MODEL, tokens, 8)
    assert restored is not None
    assert restored[0].offset == 8
    assert mx.array_equal(restored[0].state[0], mla.state[0])
    assert restored[0].state[1].shape == (1, 2, 8, 0)


def test_a_hole_in_the_chain_hides_deeper_boundaries(tmp_path):
    store = SSDPromptSnapshotStore(tmp_path, step=4)
    tokens = list(range(12))
    kv = KVCache()
    for boundary in (4, 8, 12):
        _feed(kv, 4)
        assert store.put(MODEL, tokens[:boundary], [kv])

    middle_key = store._chain_keys(MODEL, tuple(tokens))[1]
    store._path(middle_key).unlink()

    assert store.present_boundaries(MODEL, tokens) == (4,)
    assert store.load(MODEL, tokens, 12) is None
    assert store.load(MODEL, tokens, 4) is not None


def test_branching_prompts_share_their_common_chain(tmp_path):
    store = SSDPromptSnapshotStore(tmp_path, step=4)
    trunk = list(range(8))
    branch = list(range(4)) + [99, 98, 97, 96]

    kv_a = KVCache()
    for boundary in (4, 8):
        _feed(kv_a, 4)
        assert store.put(MODEL, trunk[:boundary], [kv_a])

    kv_b = KVCache()
    _feed(kv_b, 8)
    # The shared first boundary is kept, not rewritten; only the divergent
    # second boundary adds a file.
    assert store.put(MODEL, branch[:4], [kv_b])
    assert store.put(MODEL, branch, [kv_b])

    assert len(store) == 3
    assert store.present_boundaries(MODEL, trunk) == (8, 4)
    assert store.present_boundaries(MODEL, branch) == (8, 4)


def test_non_sliceable_members_ride_the_deepest_file(tmp_path):
    store = SSDPromptSnapshotStore(tmp_path, step=4)
    tokens = list(range(8))
    kv = KVCache()
    rot = RotatingKVCache(max_size=6)
    for boundary in (4, 8):
        _feed(kv, 4)
        _advance([rot], 4)
        assert store.put(MODEL, tokens[:boundary], [kv, rot])

    restored = store.load(MODEL, tokens, 8)
    assert restored is not None
    assert mx.array_equal(restored[0].state[0], kv.state[0])
    assert restored[1].offset == rot.offset
    assert mx.array_equal(restored[1].state[0], rot.state[0])


def test_a_pooling_cache_round_trips_every_slot(tmp_path):
    """DeepSeek's pool cache: remainder rows, pooled rows and the overlap
    carry must all survive, or a partial hit diverges from the live cache."""

    store = SSDPromptSnapshotStore(tmp_path, step=STEP)
    tokens = list(range(STEP))
    original = _pooling(ratio=4, tokens=10)
    assert original.remainder == 2 and original.prev_win_kv is not None

    assert store.put(MODEL, tokens, [original])
    restored = store.load(MODEL, tokens, STEP)

    assert restored is not None
    _assert_pooling_equal(restored[0], original)


def test_pooling_deltas_grow_linearly_and_carry_absolute_ranges(tmp_path):
    """Each boundary stores one fixed-size pooled slab, not all prior rows."""

    step = 128
    boundaries = tuple(range(step, 6 * step + 1, step))
    tokens = list(range(boundaries[-1]))
    store = SSDPromptSnapshotStore(tmp_path, step=step)
    pool = PoolingCache(4)

    for boundary in boundaries:
        _feed_pooling(pool, step, offset=boundary - step, dim=256)
        assert store.put(MODEL, tokens[:boundary], [pool])
        key = store._chain_keys(MODEL, tuple(tokens[:boundary]))[-1]
        raw = load_prompt_cache(str(store._path(key)))[0]
        assert isinstance(raw, PoolingCacheDeltaSnapshot)
        assert (raw.source_start, raw.source_end) == (boundary - step, boundary)
        assert (raw.pool_start, raw.pool_end) == (
            (boundary - step) // pool.ratio,
            boundary // pool.ratio,
        )

    sizes = [path.stat().st_size for path in tmp_path.glob("*.safetensors")]
    assert len(sizes) == len(boundaries)
    # Header digit growth is tiny; cumulative snapshots would make the last
    # payload roughly six times the first.
    assert max(sizes) < 1.05 * min(sizes)


def test_pooling_delta_chain_rebuilds_remainders_and_empty_slots(tmp_path):
    """Non-aligned boundaries preserve both overlap carries and None slots."""

    step = 5
    tokens = list(range(3 * step))
    store = SSDPromptSnapshotStore(tmp_path, step=step)
    overlap = PoolingCache(4)
    simple = PoolingCache(8)

    for boundary in (5, 10, 15):
        _feed_pooling(overlap, step, offset=boundary - step, with_prev=True)
        _feed_pooling(simple, step, offset=boundary - step, with_prev=False)
        assert store.put(
            MODEL,
            tokens[:boundary],
            [CacheList(overlap, simple)],
        )

    restored = store.load(MODEL, tokens, len(tokens))
    assert restored is not None
    restored_overlap, restored_simple = restored[0].caches
    assert overlap.remainder == 3
    assert simple.remainder == 7
    _assert_pooling_equal(restored_overlap, overlap)
    _assert_pooling_equal(restored_simple, simple)
    assert restored_simple.prev_win_kv is None
    assert restored_simple.prev_win_gate is None


def test_pooling_delta_chain_survives_a_rank_restart(tmp_path):
    step = 5
    tokens = list(range(3 * step))
    first = SSDPromptSnapshotStore(
        tmp_path,
        step=step,
        persistent=True,
        write_behind=True,
        max_pending_writes=3,
        pending_max_bytes=1024 * 1024,
    )
    pool = PoolingCache(4)
    for boundary in (5, 10, 15):
        _feed_pooling(pool, step, offset=boundary - step)
        assert first.put(MODEL, tokens[:boundary], [pool])
    assert first.close(timeout=5)

    second = SSDPromptSnapshotStore(tmp_path, step=step, persistent=True)
    restored = second.load(MODEL, tokens, len(tokens))
    assert restored is not None
    _assert_pooling_equal(restored[0], pool)


def test_clear_drains_write_behind_and_resets_persistent_manifest(tmp_path):
    tokens = list(range(STEP))
    store = SSDPromptSnapshotStore(
        tmp_path,
        step=STEP,
        persistent=True,
        write_behind=True,
        pending_max_bytes=1024 * 1024,
    )
    assert store.put(MODEL, tokens, _kv())
    assert store.clear(timeout=5) == 1
    assert len(store) == 0
    assert store.nbytes == 0
    assert store.present_boundaries(MODEL, tokens) == ()
    assert store.close(timeout=5)

    reopened = SSDPromptSnapshotStore(tmp_path, step=STEP, persistent=True)
    assert len(reopened) == 0
    assert reopened.present_boundaries(MODEL, tokens) == ()


def test_corrupt_pooling_delta_range_fails_closed_and_unpoisons_manifest(tmp_path):
    step = 4
    tokens = list(range(2 * step))
    store = SSDPromptSnapshotStore(tmp_path, step=step, persistent=True)
    pool = PoolingCache(4)
    for boundary in (4, 8):
        _feed_pooling(pool, step, offset=boundary - step)
        assert store.put(MODEL, tokens[:boundary], [pool])

    deepest_key = store._chain_keys(MODEL, tuple(tokens))[-1]
    path = store._path(deepest_key)
    arrays, metadata = mx.load(str(path), return_metadata=True)
    # ``mx.load`` is lazy. Materialize the payload before overwriting its
    # backing file or save_safetensors can truncate the source first and then
    # fail while evaluating arrays that still map that now-empty file.
    mx.eval(*arrays.values())
    # pool_start for a top-level delta is metadata slot five. Turn [1,2)
    # into the overlapping [0,2) while leaving the tensor bytes untouched.
    assert metadata["0.0.5"] == "1"
    metadata["0.0.5"] = "0"
    mx.save_safetensors(str(path), arrays, metadata)

    assert store.load(MODEL, tokens, len(tokens)) is None
    assert len(store) == 0
    manifest = json.loads((tmp_path / "index.json").read_text())
    assert manifest["entries"] == []


def test_legacy_cumulative_pooling_snapshot_seeds_a_new_delta_chain(
    tmp_path, monkeypatch
):
    """An on-disk pre-upgrade boundary remains reusable after extension."""

    step = 5
    tokens = list(range(2 * step))
    store = SSDPromptSnapshotStore(tmp_path, step=step, persistent=True)
    pool = PoolingCache(4)
    _feed_pooling(pool, step)

    # Simulate the pre-delta writer, which used PoolingCacheSnapshot and put
    # the whole cumulative pooled tensor in each boundary file.
    with monkeypatch.context() as patch:
        patch.setattr(
            PoolingCacheDeltaSnapshot,
            "from_cache",
            classmethod(lambda _cls, _inner, **_kwargs: None),
        )
        assert store.put(MODEL, tokens[:step], [pool])
    legacy = store.load(MODEL, tokens[:step], step)
    assert legacy is not None
    _assert_pooling_equal(legacy[0], pool)

    _feed_pooling(pool, step, offset=step)
    assert store.put(MODEL, tokens, [pool])
    restarted = SSDPromptSnapshotStore(tmp_path, step=step, persistent=True)
    restored = restarted.load(MODEL, tokens, len(tokens))
    assert restored is not None
    _assert_pooling_equal(restored[0], pool)


def test_the_deepseek_layer_shape_round_trips(tmp_path):
    """The real DSA layout: CacheList(rotating, pool, pool) plus a plain
    rotating layer, with a boundary-typical empty remainder on one pool."""

    store = SSDPromptSnapshotStore(tmp_path, step=STEP)
    tokens = list(range(STEP))
    rot_member = RotatingKVCache(max_size=8)
    plain = RotatingKVCache(max_size=8)
    _advance([rot_member, plain], 20)
    pool_small = _pooling(ratio=4, tokens=10)
    pool_large = _pooling(ratio=128, tokens=256, with_prev=False)
    assert pool_large.remainder == 0  # buf and prev slots are all None
    caches = [CacheList(rot_member, pool_small, pool_large), plain]

    assert store.put(MODEL, tokens, caches)
    restored = store.load(MODEL, tokens, STEP)

    assert restored is not None
    assert [type(c).__name__ for c in restored] == ["CacheList", "RotatingKVCache"]
    members = restored[0].caches
    assert type(members[0]).__name__ == "RotatingKVCache"
    assert mx.array_equal(members[0].state[0], rot_member.state[0])
    _assert_pooling_equal(members[1], pool_small)
    _assert_pooling_equal(members[2], pool_large)
    assert mx.array_equal(restored[1].state[0], plain.state[0])
    # The live cache was wrapped, not rewritten.
    assert pool_small.prev_win_kv is not None


def test_an_arrays_cache_with_an_unwritten_slot_round_trips(tmp_path):
    """A recurrent cache may leave slots None until a layer first writes them;
    the stand-in must carry the mixed written/unwritten layout exactly."""

    store = SSDPromptSnapshotStore(tmp_path, step=STEP)
    gdn = ArraysCache(size=2)
    gdn[0] = mx.random.normal((1, 2, 4))  # slot 1 never written

    assert store.put(MODEL, list(range(STEP)), [gdn])
    restored = store.load(MODEL, list(range(STEP)), STEP)

    assert restored is not None
    assert type(restored[0]).__name__ == "ArraysCache"
    assert mx.array_equal(restored[0][0], gdn[0])
    assert restored[0][1] is None


def test_an_empty_pooling_cache_still_round_trips(tmp_path):
    """A member with no state yet must not shift later caches in the file."""

    store = SSDPromptSnapshotStore(tmp_path, step=STEP)
    trailing = _kv()[0]
    assert store.put(MODEL, list(range(STEP)), [PoolingCache(4), trailing])
    restored = store.load(MODEL, list(range(STEP)), STEP)

    assert restored is not None
    assert type(restored[0]).__name__ == "PoolingCache"
    assert restored[0].empty() and restored[0].ratio == 4
    assert mx.array_equal(restored[1].state[0], trailing.state[0])


def test_an_untouched_rotating_member_round_trips(tmp_path):
    """DeepSeek short context: a sparse branch below its engagement length
    keeps a rotating member whose state slices are zero-size, which
    safetensors rejects. The stand-in must carry it and every later cache."""

    store = SSDPromptSnapshotStore(tmp_path, step=STEP)
    idle = RotatingKVCache(max_size=8)
    idle.keys = mx.zeros((1, 2, 0, 4), dtype=mx.float16)
    idle.values = mx.zeros((1, 2, 0, 4), dtype=mx.float16)
    pool = _pooling(ratio=4, tokens=10)
    trailing = RotatingKVCache(max_size=8)
    _advance([trailing], 20)

    assert store.put(MODEL, list(range(STEP)), [CacheList(idle, pool), trailing])
    restored = store.load(MODEL, list(range(STEP)), STEP)

    assert restored is not None
    members = restored[0].caches
    assert type(members[0]).__name__ == "RotatingKVCache"
    assert members[0].offset == 0
    assert members[0].keys.shape == (1, 2, 0, 4)
    assert members[0].keys.dtype == mx.float16
    _assert_pooling_equal(members[1], pool)
    assert mx.array_equal(restored[1].state[0], trailing.state[0])


def test_a_new_store_reclaims_what_a_dead_process_left(tmp_path):
    """Snapshots are process-lifetime: digest filenames cannot be re-indexed
    without their token tuples, so a stale file would be invisible to hits yet
    still hold disk. A new store starts by clearing its directory."""

    (tmp_path / "deadbeef.safetensors").write_bytes(b"stale")
    (tmp_path / ".partial.safetensors").write_bytes(b"orphaned temp")
    store = SSDPromptSnapshotStore(tmp_path, step=STEP)

    assert list(tmp_path.iterdir()) == []
    assert store.put(MODEL, list(range(STEP)), _kv())  # still fully usable


def test_persistent_store_restores_its_chain_after_rank_restart(tmp_path):
    tokens = list(range(8))
    first = SSDPromptSnapshotStore(tmp_path, step=4, persistent=True)
    kv = KVCache()
    for boundary in (4, 8):
        _feed(kv, 4)
        assert first.put(MODEL, tokens[:boundary], [kv])

    manifest = tmp_path / "index.json"
    assert manifest.is_file()
    second = SSDPromptSnapshotStore(tmp_path, step=4, persistent=True)
    assert second.present_boundaries(MODEL, tokens) == (8, 4)
    restored = second.load(MODEL, tokens, 8)
    assert restored is not None
    assert mx.array_equal(restored[0].state[0], kv.state[0])


def test_invalid_persistent_manifest_fails_closed(tmp_path):
    (tmp_path / "deadbeef.safetensors").write_bytes(b"stale")
    (tmp_path / "index.json").write_text('{"version":1,"step":4,"entries":[{}]}')

    store = SSDPromptSnapshotStore(tmp_path, step=4, persistent=True)

    assert len(store) == 0
    assert not (tmp_path / "deadbeef.safetensors").exists()
    assert (tmp_path / "index.json").is_file()


def test_v1_persistent_manifest_migrates_without_losing_restore(tmp_path):
    tokens = list(range(4))
    first = SSDPromptSnapshotStore(tmp_path, step=4, persistent=True)
    assert first.put(MODEL, tokens, _kv())
    legacy = json.loads((tmp_path / "index.json").read_text())
    legacy["version"] = 1
    for row in legacy["entries"]:
        row.pop("capacity_charge_bytes", None)
    (tmp_path / "index.json").write_text(json.dumps(legacy))

    second = SSDPromptSnapshotStore(tmp_path, step=4, persistent=True)

    assert second.load(MODEL, tokens, 4) is not None
    migrated = json.loads((tmp_path / "index.json").read_text())
    assert migrated["version"] == 2
    assert migrated["entries"][0]["capacity_charge_bytes"] > 0


def test_an_unaligned_prompt_is_rejected(tmp_path):
    store = SSDPromptSnapshotStore(tmp_path, step=STEP)
    assert store.put(MODEL, list(range(STEP + 1)), _kv()) is False
    assert store.load(MODEL, list(range(STEP)), STEP - 1) is None


def test_a_prefix_of_different_tokens_is_not_a_hit(tmp_path):
    store = SSDPromptSnapshotStore(tmp_path, step=STEP)
    store.put(MODEL, list(range(STEP)), _kv())

    other = list(range(1, STEP + 1))  # same length, different tokens
    assert store.present_boundaries(MODEL, other) == ()
    assert store.load(MODEL, other, STEP) is None


def test_count_lru_eviction_is_deterministic(tmp_path):
    """Independent chains: the oldest files fall out first."""

    store = SSDPromptSnapshotStore(tmp_path, step=2, max_entries=2)
    prompts = ([0, 1], [2, 3], [4, 5], [6, 7])
    for prompt in prompts:
        assert store.put(MODEL, prompt, _kv())
    assert len(store) == 2
    assert store.load(MODEL, [4, 5], 2) is not None
    assert store.load(MODEL, [6, 7], 2) is not None
    assert store.load(MODEL, [0, 1], 2) is None


def test_touching_a_chain_saves_it_from_eviction(tmp_path):
    store = SSDPromptSnapshotStore(tmp_path, step=2, max_entries=2)
    store.put(MODEL, [0, 1], _kv())
    store.put(MODEL, [2, 3], _kv())
    assert store.load(MODEL, [0, 1], 2) is not None  # touch the oldest
    store.put(MODEL, [4, 5], _kv())  # evicts the now-oldest ([2, 3])
    assert store.load(MODEL, [0, 1], 2) is not None
    assert store.load(MODEL, [2, 3], 2) is None


def test_the_byte_budget_evicts_oldest_files(tmp_path):
    probe = SSDPromptSnapshotStore(tmp_path / "probe", step=2)
    assert probe.put(MODEL, [0, 1], _kv())
    file_size = probe.nbytes

    store = SSDPromptSnapshotStore(
        tmp_path / "capped", step=2, max_bytes=int(file_size * 2.5)
    )
    for prompt in ([0, 1], [2, 3], [4, 5]):
        assert store.put(MODEL, prompt, _kv())
    assert len(store) == 2
    assert store.nbytes <= file_size * 2.5
    assert store.load(MODEL, [0, 1], 2) is None


def test_default_disk_budget_is_finite_and_conservative(tmp_path):
    store = SSDPromptSnapshotStore(tmp_path, step=2)

    assert store.max_bytes == DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES
    assert store.max_bytes == 20 * 1024**3


def test_default_20_gib_budget_evicts_when_shared_charges_cross_it(tmp_path):
    charge = 12 * 1024**3
    store = SSDPromptSnapshotStore(
        tmp_path,
        step=2,
        write_behind=True,
        capacity_agreement=lambda _local: charge,
    )

    assert store.put(MODEL, [0, 1], _kv())
    assert store.put(MODEL, [2, 3], _kv())
    assert store.flush(timeout=5)

    assert len(store) == 1
    assert store.capacity_bytes == charge
    assert store.nbytes <= store.max_bytes
    assert store.evictions == 1
    assert store.load(MODEL, [0, 1], 2) is None
    assert store.load(MODEL, [2, 3], 2) is not None
    assert store.close(timeout=5)


def test_shared_capacity_charge_keeps_unequal_ranks_eviction_symmetric(tmp_path):
    """Different shard file sizes must still evict the same oldest key."""

    shared_charge = 1024 * 1024
    budget = 2 * shared_charge
    rank0 = SSDPromptSnapshotStore(
        tmp_path / "r0",
        step=2,
        max_bytes=budget,
        write_behind=True,
        capacity_agreement=lambda _local: shared_charge,
    )
    rank1 = SSDPromptSnapshotStore(
        tmp_path / "r1",
        step=2,
        max_bytes=budget,
        write_behind=True,
        capacity_agreement=lambda _local: shared_charge,
    )
    prompts = ([0, 1], [2, 3], [4, 5])
    for prompt in prompts:
        assert rank0.put(MODEL, prompt, _kv(layers=1))
        assert rank1.put(MODEL, prompt, _kv(layers=2))
    assert rank0.flush(timeout=5)
    assert rank1.flush(timeout=5)

    assert list(rank0._index) == list(rank1._index)
    assert rank0.present_boundaries(MODEL, list(prompts[0])) == ()
    assert rank1.present_boundaries(MODEL, list(prompts[0])) == ()
    assert rank0.capacity_bytes == rank1.capacity_bytes == budget
    assert rank0.nbytes <= rank0.max_bytes
    assert rank1.nbytes <= rank1.max_bytes
    assert rank0.evictions == rank1.evictions == 1
    assert rank0.close(timeout=5)
    assert rank1.close(timeout=5)


def test_two_ranks_keep_identical_keys_from_identical_requests(tmp_path):
    """Different layer slices, same keys: the emergent-consistency contract."""

    rank0 = SSDPromptSnapshotStore(tmp_path / "r0", step=STEP, max_entries=8)
    rank1 = SSDPromptSnapshotStore(tmp_path / "r1", step=STEP, max_entries=8)
    tokens = list(range(2 * STEP))
    # Rank 1's cache is a different shape (its own layer slice); the keys are
    # still keyed on tokens, so both stores agree on which boundaries exist.
    rank0.put(MODEL, tokens[:STEP], _kv())
    rank1.put(MODEL, tokens[:STEP], _kv(layers=2))
    rank0.put(MODEL, tokens, _kv())
    rank1.put(MODEL, tokens, _kv(layers=2))

    assert rank0.present_boundaries(MODEL, tokens) == rank1.present_boundaries(
        MODEL, tokens
    )


def test_agreed_boundary_takes_the_longest_unanimous():
    candidates = (6144, 4096, 2048)
    # world of 3: 2048 present on all, 4096 on two, 6144 on one.
    assert agreed_boundary(candidates, [1, 2, 3], world_size=3) == 2048
    # unanimous at the longest.
    assert agreed_boundary(candidates, [3, 3, 3], world_size=3) == 6144
    # nobody agrees.
    assert agreed_boundary(candidates, [1, 2, 2], world_size=3) == 0


def test_agreed_boundary_drops_a_rank_that_lost_its_write():
    """The write-failure guard: a missing snapshot on one rank blocks reuse."""

    candidates = (4096, 2048)
    # Rank A has both, rank B lost 4096: votes are A=[1,1], B=[0,1], sum=[1,2].
    assert agreed_boundary(candidates, [1, 2], world_size=2) == 2048


def test_an_unserialisable_cache_disables_the_store(tmp_path, monkeypatch):
    """A cache type save_prompt_cache rejects and no stand-in covers.

    Such a type never will serialise, so the store stops trying after the
    first failure instead of paying a doomed write on every boundary.
    """

    store = SSDPromptSnapshotStore(tmp_path, step=STEP)
    calls = []

    def _unserialisable(*_a, **_k):
        calls.append(1)
        raise ValueError("Metadata must be a dictionary with string keys")

    monkeypatch.setattr(
        "mlx_lm.models.cache.save_prompt_cache", _unserialisable, raising=True
    )
    assert store.put(MODEL, list(range(STEP)), _kv()) is False
    assert store.put(MODEL, list(range(2 * STEP)), _kv()) is False
    assert len(calls) == 1  # only the first was attempted
    assert len(store) == 0


def test_a_disk_error_keeps_the_store_live(tmp_path, monkeypatch):
    """A transient write failure must not permanently disable the store."""

    store = SSDPromptSnapshotStore(tmp_path, step=STEP)
    calls = []

    def _flaky(*_a, **_k):
        calls.append(1)
        raise OSError("no space left on device")

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", _flaky, raising=True)
    assert store.put(MODEL, list(range(STEP)), _kv()) is False
    assert store.put(MODEL, list(range(2 * STEP)), _kv()) is False
    assert len(calls) == 2  # each attempt was made


def test_a_failed_write_leaves_the_index_unchanged(tmp_path, monkeypatch):
    store = SSDPromptSnapshotStore(tmp_path, step=STEP)

    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr("mlx_lm.models.cache.save_prompt_cache", _boom, raising=True)
    assert store.put(MODEL, list(range(STEP)), _kv()) is False
    assert len(store) == 0
    assert store.present_boundaries(MODEL, list(range(STEP))) == ()
    # No half-written temp file is left behind.
    assert list(tmp_path.glob("*")) == []


def test_write_behind_freezes_mutable_rotating_state_at_the_boundary(
    tmp_path, monkeypatch
):
    """The worker must never observe a live cache after decode overwrites it."""

    started = threading.Event()
    release = threading.Event()
    original_save = mx.save_safetensors

    def _blocked_save(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original_save(*args, **kwargs)

    monkeypatch.setattr(mx, "save_safetensors", _blocked_save)
    store = SSDPromptSnapshotStore(
        tmp_path,
        step=4,
        persistent=True,
        write_behind=True,
        pending_max_bytes=1024 * 1024,
    )
    cache = RotatingKVCache(max_size=4)
    _advance([cache], 4)
    expected_state = [value.tolist() for value in cache.state]
    expected_meta = cache.meta_state

    assert store.put(MODEL, [0, 1, 2, 3], [cache])
    assert started.wait(timeout=5)
    _advance([cache], 1)  # overwrites the live full rotating buffer in place
    assert cache.meta_state != expected_meta
    release.set()
    assert store.flush(timeout=5)

    restored = store.load(MODEL, [0, 1, 2, 3], 4)
    assert restored is not None
    assert restored[0].meta_state == expected_meta
    assert [value.tolist() for value in restored[0].state] == expected_state
    assert store.close(timeout=5)


def test_write_behind_strictly_rejects_one_oversized_payload(tmp_path):
    store = SSDPromptSnapshotStore(
        tmp_path,
        step=2,
        write_behind=True,
        pending_max_bytes=1,
    )

    assert store.put(MODEL, [0, 1], _kv()) is False
    assert store.pending_count == 0
    assert store.pending_bytes == 0
    assert store.pending_peak_bytes == 0
    assert store.close(timeout=5)


def test_write_behind_count_saturation_drops_instead_of_blocking_prefill(
    tmp_path, monkeypatch
):
    started = threading.Event()
    release = threading.Event()
    original_save = mx.save_safetensors

    def _blocked_save(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original_save(*args, **kwargs)

    monkeypatch.setattr(mx, "save_safetensors", _blocked_save)
    store = SSDPromptSnapshotStore(
        tmp_path,
        step=2,
        write_behind=True,
        max_pending_writes=1,
        pending_max_bytes=1024 * 1024,
    )
    assert store.put(MODEL, [0, 1], _kv())
    assert started.wait(timeout=5)

    before = time.monotonic()
    assert store.put(MODEL, [2, 3], _kv()) is False
    assert time.monotonic() - before < 0.5
    assert store.pending_count == 1

    release.set()
    assert store.close(timeout=5)


def test_write_behind_close_is_bounded_when_the_writer_stalls(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    original_save = mx.save_safetensors

    def _blocked_save(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original_save(*args, **kwargs)

    monkeypatch.setattr(mx, "save_safetensors", _blocked_save)
    store = SSDPromptSnapshotStore(
        tmp_path,
        step=2,
        persistent=True,
        write_behind=True,
        pending_max_bytes=1024 * 1024,
    )
    assert store.put(MODEL, [0, 1], _kv())
    assert started.wait(timeout=5)

    before = time.monotonic()
    assert store.close(timeout=0.02) is False
    assert time.monotonic() - before < 0.5
    # A timed-out close leaves the directory alone while the daemon writer may
    # still own its atomic staging file.  A later retry finishes cleanly.
    assert tmp_path.is_dir()
    release.set()
    assert store.close(timeout=5)


def test_write_behind_fifo_keeps_successful_rank_key_order_symmetric(tmp_path):
    rank0 = SSDPromptSnapshotStore(
        tmp_path / "r0",
        step=2,
        write_behind=True,
        pending_max_bytes=1024 * 1024,
    )
    rank1 = SSDPromptSnapshotStore(
        tmp_path / "r1",
        step=2,
        write_behind=True,
        pending_max_bytes=1024 * 1024,
    )
    for tokens in ([0, 1], [2, 3]):
        assert rank0.put(MODEL, tokens, _kv(layers=1))
        assert rank1.put(MODEL, tokens, _kv(layers=2))
    assert rank0.flush(timeout=5)
    assert rank1.flush(timeout=5)
    assert list(rank0._index) == list(rank1._index)
    assert rank0.close(timeout=5)
    assert rank1.close(timeout=5)
