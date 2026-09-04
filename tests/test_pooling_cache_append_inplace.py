# SPDX-License-Identifier: Apache-2.0
"""Losslessness tests for the append-in-place PoolingCache rework.

The caches in ``omlx/patches/deepseek_v4/cache_extras.py`` used to rebuild
``self.pooled`` with ``mx.concatenate`` on every chunk; they now append into
a preallocated backing buffer with geometric regrowth and expose the logical
tensor as a view. These tests pin the exact old observable behavior:
contents, shapes, offset/size bookkeeping, snapshot/delta immunity, and
trim/rollback semantics.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches.deepseek_v4.cache_extras import (
    BatchPoolingCache,
    PoolingCache,
)


def _rows(start: int, count: int, D: int, B: int = 1) -> mx.array:
    """Deterministic distinct values per append so mis-ordering shows up."""
    vals = mx.arange(start * D * B, (start + count) * D * B, dtype=mx.float32)
    return (vals.reshape(B, count, D) % 997) / 997.0


class _RefSingle:
    """Old concatenate semantics for PoolingCache.update_and_fetch."""

    def __init__(self):
        self.pooled = None

    def update_and_fetch(self, px: mx.array):
        if px.shape[1] == 0:
            return self.pooled
        if self.pooled is None:
            self.pooled = px
        else:
            self.pooled = mx.concatenate([self.pooled, px], axis=1)
        return self.pooled


def _assert_same(actual, expected):
    mx.eval(actual, expected)
    assert actual.shape == expected.shape
    assert bool(mx.array_equal(actual, expected))


# ---------------------------------------------------------------------------
# PoolingCache (single sequence)
# ---------------------------------------------------------------------------


def test_single_varied_appends_match_concatenate_reference():
    cache = PoolingCache(4)
    ref = _RefSingle()
    # Include regrowth-forcing big appends, single rows, and zero-row calls.
    sizes = [1, 3, 2, 8, 1, 1, 16, 5, 0, 33, 2, 0, 1, 64]
    start = 0
    for n in sizes:
        px = _rows(start, n, 8)
        start += n
        got = cache.update_and_fetch(px)
        want = ref.update_and_fetch(px)
        if want is None:
            assert got.shape[1] == 0
            continue
        _assert_same(got, want)
        _assert_same(cache.pooled, want)
        assert cache.offset == want.shape[1]
        assert cache.size() == want.shape[1]
    # Geometric capacity: backing buffer never smaller than the logical view.
    assert cache._pool_buf.shape[1] >= cache._pool_len == ref.pooled.shape[1]


def test_single_capacity_regrowth_preserves_data():
    cache = PoolingCache(4)
    ref = _RefSingle()
    capacities = []
    for i in range(40):
        px = _rows(i * 2, 2, 4)
        cache.update_and_fetch(px)
        ref.update_and_fetch(px)
        mx.eval(cache.pooled)
        capacities.append(cache._pool_buf.shape[1])
    _assert_same(cache.pooled, ref.pooled)
    # Growth happened and was geometric (never grows by less than double).
    assert capacities[-1] >= 80
    for prev, cur in zip(capacities, capacities[1:]):
        assert cur == prev or cur >= 2 * prev


def test_single_snapshot_immune_to_later_appends():
    cache = PoolingCache(4)
    ref = _RefSingle()
    for i in range(6):
        px = _rows(i, 1 + (i % 3), 8)
        cache.update_and_fetch(px)
        ref.update_and_fetch(px)

    # Materialized snapshot of the logical region (what pooling_delta does:
    # slice + mx.contiguous, then the scheduler mx.evals the delta).
    prefix_len = ref.pooled.shape[1]
    snap = mx.contiguous(cache.pooled[:, :prefix_len])
    mx.eval(snap)

    # The state view, evaluated in place like the per-chunk fence does.
    state_view = cache.state[2]
    mx.eval(state_view)

    for i in range(10):
        cache.update_and_fetch(_rows(100 + i * 4, 4, 8))
    mx.eval(cache.pooled)

    _assert_same(snap, ref.pooled)
    # Rows below the snapshot length are never rewritten, so even the
    # evaluated view still reads the same values.
    _assert_same(state_view, ref.pooled)
    assert cache.pooled.shape[1] == prefix_len + 40


def test_single_state_setter_roundtrip():
    cache = PoolingCache(4)
    ref = _RefSingle()
    for i in range(5):
        px = _rows(i, 3, 8)
        cache.update_and_fetch(px)
        ref.update_and_fetch(px)

    restored = PoolingCache(4)
    restored.state = cache.state
    _assert_same(restored.pooled, ref.pooled)
    assert restored.offset == ref.pooled.shape[1]

    # Appends continue seamlessly after a restore (regrowth from exact fit).
    more = _rows(50, 7, 8)
    restored.update_and_fetch(more)
    ref.update_and_fetch(more)
    _assert_same(restored.pooled, ref.pooled)


def test_single_zero_row_append_on_empty_cache():
    cache = PoolingCache(4)
    got = cache.update_and_fetch(mx.zeros((1, 0, 8), dtype=mx.float32))
    assert got.shape == (1, 0, 8)
    assert cache.pooled is None
    assert cache.offset == 0
    assert cache.empty()


def test_single_trim_within_remainder_keeps_pooled():
    cache = PoolingCache(4)
    D1, D2 = 8, 8
    # Prompt of 5 tokens: completes one window, remainder 1.
    kv = _rows(0, 5, D1)
    gate = _rows(10, 5, D2)
    r_kv, r_gate, _ = cache.accumulate_windows(kv, gate, 0)
    assert r_kv.shape[1] == 4
    px = _rows(20, 1, 8)
    cache.update_and_fetch(px)
    mx.eval(cache.pooled)
    assert cache.remainder == 1
    assert cache.offset == 1

    assert cache.trim(1) == 1
    assert cache.remainder == 0
    # Pooled rows are untouched by a remainder trim.
    _assert_same(cache.pooled, px)


def test_single_undo_trim_restores_pre_update_rows():
    """MTP draft rejection: a decode-sized update that completed a window is
    rolled back through the one-update undo log; pooled must return to the
    exact pre-update logical contents."""
    from omlx.patches.mlx_lm_mtp import cache_rollback

    cache_rollback.set_undo_armed(True)
    try:
        cache = PoolingCache(4)
        ref = _RefSingle()
        for i in range(3):
            px = _rows(i * 2, 2, 8)
            cache.update_and_fetch(px)
            ref.update_and_fetch(px)
        mx.eval(cache.pooled)
        pre_update = mx.contiguous(cache.pooled)
        mx.eval(pre_update)

        # Decode-sized update (L=1) that completes a window: 3 tokens sat in
        # the remainder, so one more token produces a pooled row.
        cache.remainder = 3
        cache.buf_kv = mx.zeros((1, 4, 8))
        cache.buf_gate = mx.zeros((1, 4, 8))
        kv = _rows(60, 1, 8)
        gate = _rows(70, 1, 8)
        r_kv, r_gate, _ = cache.accumulate_windows(kv, gate, 24)
        assert r_kv.shape[1] == 4  # window completed
        new_row = _rows(80, 1, 8)
        cache.update_and_fetch(new_row)
        mx.eval(cache.pooled)
        assert cache.pooled.shape[1] == pre_update.shape[1] + 1

        assert cache.is_trimmable()
        assert cache.trim(1) == 1
        mx.eval(cache.pooled)
        _assert_same(cache.pooled, pre_update)
        assert cache.offset == pre_update.shape[1]

        # Appending after the rollback rewrites the trimmed slot; the
        # pre-update snapshot taken before must stay immune.
        cache.update_and_fetch(_rows(90, 2, 8))
        mx.eval(cache.pooled)
        _assert_same(pre_update, ref.pooled)
    finally:
        cache_rollback.set_undo_armed(False)


# ---------------------------------------------------------------------------
# BatchPoolingCache
# ---------------------------------------------------------------------------


def _old_batch_update(state, px, ratio):
    """Verbatim old (pre-rework) BatchPoolingCache.update_and_fetch semantics.

    ``state`` is a dict with keys pooled, pool_lengths, processed, remainder.
    Returns the new pooled tensor; mutates pool_lengths in place.
    """
    B, N, D = px.shape
    pooled = state["pooled"]
    pool_lengths = state["pool_lengths"]

    if N == 0:
        return pooled

    new_counts = [
        (state["processed"][i] - state["remainder"][i]) // ratio - pool_lengths[i]
        for i in range(B)
    ]
    max_new = max(new_counts)
    if max_new == 0:
        return pooled

    if B == 1:
        count = new_counts[0]
        current = pool_lengths[0]
        new_rows = px[:, :count]
        if pooled is None or current == 0:
            pooled = new_rows
        else:
            pooled = mx.concatenate([pooled[:, :current], new_rows], axis=1)
        pool_lengths[0] = current + count
        return pooled

    max_pool = max(pool_lengths) + max_new
    if pooled is None:
        pooled = mx.zeros((B, max_pool, D), dtype=px.dtype)
    elif pooled.shape[1] < max_pool:
        pad = mx.zeros((B, max_pool - pooled.shape[1], D), dtype=px.dtype)
        pooled = mx.concatenate([pooled, pad], axis=1)

    for i in range(B):
        nc = new_counts[i]
        if nc > 0:
            pl = pool_lengths[i]
            pooled[i, pl : pl + nc] = px[i, :nc]
            pool_lengths[i] = pl + nc
    return pooled


@pytest.mark.parametrize("B", [1, 2, 3])
def test_batch_varied_appends_match_old_semantics(B):
    ratio = 4
    cache = BatchPoolingCache(ratio, [0] * B)
    ref = {"pooled": None, "pool_lengths": [0] * B}

    # Each step: every row consumes `step + i` tokens (some completing
    # windows, some only filling remainders).
    step = 0
    for tokens in ([4, 8, 3, 12, 5, 16, 1, 7, 20, 2, 9, 6],):
        for t in tokens:
            L = t
            kv = _rows(step, L, 8, B)
            gate = _rows(1000 + step, L, 8, B)
            step += L
            cache.prepare(lengths=[L] * B)
            r_kv, r_gate, _ = cache.accumulate_windows(kv, gate, 0)
            n_rows = r_kv.shape[1] // ratio
            px = _rows(2000 + step, n_rows, 8, B) if n_rows else mx.zeros(
                (B, 0, 8), dtype=mx.float32
            )
            # Reference bookkeeping mirrors the real cache's fields.
            ref["processed"] = list(cache._processed)
            ref["remainder"] = list(cache.remainder)
            cache.update_and_fetch(px)
            ref["pooled"] = _old_batch_update(ref, px, ratio)
            ref["pool_lengths"] = list(cache._pool_lengths)
            if ref["pooled"] is None:
                assert cache.pooled is None or cache.pooled.shape[1] == 0
                continue
            _assert_same(cache.pooled, ref["pooled"])
            assert cache.pooled.shape[1] == ref["pooled"].shape[1]
            assert cache.size() == ref["pooled"].shape[1]


def test_batch_extent_overshoot_matches_old_shape():
    """Old physical shape could overshoot max(_pool_lengths) when the longest
    row was not the row completing windows (max(lengths)+max_new)."""
    ratio = 4
    B = 2
    cache = BatchPoolingCache(ratio, [0] * B)
    ref = {"pooled": None, "pool_lengths": [0] * B}

    # Step 1: row 0 completes 10 windows (40 tokens), row 1 completes 1
    # (4 valid tokens; per-row valid lengths come from prepare()).
    cache.prepare(lengths=[40, 4])
    kv = _rows(0, 40, 8, B)
    gate = _rows(100, 40, 8, B)
    cache.accumulate_windows(kv, gate, 0)
    ref["processed"] = list(cache._processed)
    ref["remainder"] = list(cache.remainder)
    px = _rows(200, 10, 8, B)  # row 0 -> 10 rows, row 1 -> 1 row
    cache.update_and_fetch(px)
    ref["pooled"] = _old_batch_update(ref, px, ratio)
    ref["pool_lengths"] = list(cache._pool_lengths)
    _assert_same(cache.pooled, ref["pooled"])
    assert cache._pool_lengths == [10, 1]

    # Step 2: row 0 completes nothing (3 tokens), row 1 completes 2 windows.
    # Old max_pool overshoots: max(lengths)=10 + max_new=2 -> 12 while the
    # new lengths are [10, 3].
    cache.prepare(lengths=[3, 8])
    kv = _rows(300, 8, 8, B)
    gate = _rows(400, 8, 8, B)
    cache.accumulate_windows(kv, gate, 0)
    ref["processed"] = list(cache._processed)
    ref["remainder"] = list(cache.remainder)
    px = _rows(500, 2, 8, B)
    cache.update_and_fetch(px)
    ref["pooled"] = _old_batch_update(ref, px, ratio)
    ref["pool_lengths"] = list(cache._pool_lengths)

    _assert_same(cache.pooled, ref["pooled"])
    assert ref["pooled"].shape[1] == 12  # overshoot really happened
    assert cache.pooled.shape[1] == 12
    assert cache._pool_lengths == [10, 3]
    # Overshoot columns stay zero-filled exactly like the old pad path.
    tail = cache.pooled[:, 10:]
    mx.eval(tail)
    assert float(mx.abs(tail).max()) == 0.0


def test_batch_snapshot_and_extract_immune_to_later_appends():
    ratio = 4
    cache = BatchPoolingCache(ratio, [0, 0])
    ref = {"pooled": None, "pool_lengths": [0] * 2}

    for step, t in enumerate([8, 8, 8, 8]):
        cache.prepare(lengths=[t] * 2)
        kv = _rows(step * 10, t, 8, 2)
        gate = _rows(500 + step * 10, t, 8, 2)
        cache.accumulate_windows(kv, gate, 0)
        ref["processed"] = list(cache._processed)
        ref["remainder"] = list(cache.remainder)
        px = _rows(900 + step * 4, 2, 8, 2)
        cache.update_and_fetch(px)
        ref["pooled"] = _old_batch_update(ref, px, ratio)
        ref["pool_lengths"] = list(cache._pool_lengths)

    mx.eval(cache.pooled)
    snap = mx.contiguous(cache.pooled)
    mx.eval(snap)
    extracted = cache.extract(1)
    mx.eval(extracted.pooled)

    # Keep appending (forces regrowth) and re-verify both snapshots.
    for step, t in enumerate([12, 8, 16]):
        cache.prepare(lengths=[t] * 2)
        kv = _rows(2000 + step * 10, t, 8, 2)
        gate = _rows(3000 + step * 10, t, 8, 2)
        cache.accumulate_windows(kv, gate, 0)
        ref["processed"] = list(cache._processed)
        ref["remainder"] = list(cache.remainder)
        n_rows = t // ratio
        px = _rows(4000 + step * 4, n_rows, 8, 2)
        cache.update_and_fetch(px)
        ref["pooled"] = _old_batch_update(ref, px, ratio)
        ref["pool_lengths"] = list(cache._pool_lengths)

    _assert_same(snap, ref["pooled"][:, : snap.shape[1]])
    # extract() holds row 1's first pl rows as an independent copy; each of
    # the 4 pre-snapshot steps completed 2 windows per row, so pl == 8.
    pl = 8
    _assert_same(extracted.pooled, ref["pooled"][1:2, :pl])
    assert isinstance(extracted, PoolingCache)
    assert extracted.offset == pl


def test_batch_truncate_pooled_tail_matches_old_slice():
    ratio = 4
    cache = BatchPoolingCache(ratio, [0, 0])
    cache.prepare(lengths=[8, 8])
    kv = _rows(0, 8, 8, 2)
    gate = _rows(100, 8, 8, 2)
    cache.accumulate_windows(kv, gate, 0)
    px = _rows(200, 2, 8, 2)
    cache.update_and_fetch(px)
    mx.eval(cache.pooled)
    assert cache.pooled.shape[1] == 2

    # Simulate a rejected speculative suffix on row 1 only: old code sliced
    # pooled to max(_pool_lengths).
    cache._pool_lengths[1] = 1
    cache._truncate_pooled_tail()
    assert cache.pooled.shape[1] == 2  # max length still 2 (row 0)
    cache._pool_lengths[0] = 1
    cache._truncate_pooled_tail()
    assert cache.pooled.shape[1] == 1
    _assert_same(cache.pooled, px[:, :1])


def test_batch_state_setter_and_filter():
    ratio = 4
    cache = BatchPoolingCache(ratio, [0, 0])
    cache.prepare(lengths=[8, 8])
    kv = _rows(0, 8, 8, 2)
    gate = _rows(100, 8, 8, 2)
    cache.accumulate_windows(kv, gate, 0)
    px = _rows(200, 2, 8, 2)
    cache.update_and_fetch(px)
    mx.eval(cache.pooled)

    restored = BatchPoolingCache(ratio, [0, 0])
    restored.state = cache.state
    _assert_same(restored.pooled, cache.pooled)

    filtered = BatchPoolingCache(ratio, [0, 0])
    filtered.state = cache.state
    filtered._pool_lengths = list(cache._pool_lengths)
    filtered.filter([1])
    _assert_same(filtered.pooled, cache.pooled[1:2])
    assert filtered._pool_lengths == [cache._pool_lengths[1]]


def test_merge_single_caches_preserves_contents():
    caches = []
    refs = []
    for b in range(3):
        c = PoolingCache(4)
        ref = _RefSingle()
        for i in range(b + 2):
            px = _rows(10 * b + i, 1 + i, 8)
            c.update_and_fetch(px)
            ref.update_and_fetch(px)
        mx.eval(c.pooled)
        caches.append(c)
        refs.append(ref.pooled)

    batch = PoolingCache.merge(caches)
    assert isinstance(batch, BatchPoolingCache)
    max_pool = max(r.shape[1] for r in refs)
    assert batch.pooled.shape == (3, max_pool, 8)
    for i, r in enumerate(refs):
        _assert_same(batch.pooled[i : i + 1, : r.shape[1]], r)
