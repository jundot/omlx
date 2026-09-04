# SPDX-License-Identifier: Apache-2.0
"""SSD prompt-cache snapshots must round-trip non-sliceable state, keep KV as
one linear chain of slabs, and stay consistent across ranks that see the same
requests."""

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache

from omlx.cluster.prompt_snapshot_cache import (
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
