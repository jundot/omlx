# SPDX-License-Identifier: Apache-2.0
"""SlotArena — the shared fixed-row residency primitive (V4-2a).

Covers the acquire/commit/rollback protocol, two-phase grow/compact,
frozen (verify) ordering and staged-prefetch bookkeeping on a minimal
host — the same semantics _ExpertSlots delegates to.
"""

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.expert_streaming.slot_cache import SlotArena


def _host(capacity, row=(2,)):
    """Minimal host: dict of proj -> {field: array} slot storage."""
    store = {"p": {"weight": mx.zeros((capacity, *row), dtype=mx.float32)}}
    arena = SlotArena(
        capacity,
        ("p",),
        lambda proj: store[proj],
        lambda proj, values: store[proj].update(values),
        lambda: mx.eval(store["p"]["weight"]),
    )
    return arena, store


def _payload(expert):
    return {"p": {"weight": mx.full((2,), float(expert + 1))}}


def _produce(fl):
    return [_payload(e) for e, _s, _v in fl]


def test_ensure_commits_and_rows_for():
    arena, store = _host(4)
    missed = arena.ensure_set({2, 0}, False, _produce)
    assert missed
    assert set(arena.book.slot_of) == {0, 2}
    np.testing.assert_array_equal(
        np.asarray(store["p"]["weight"][arena.book.slot_of[2]]),
        np.full((2,), 3.0),
    )
    rows = arena.rows_for([2, 0])
    assert sorted(rows.tolist()) == sorted(arena.book.slot_of.values())


def test_ensure_hit_costs_no_write():
    arena, store = _host(4)
    arena.ensure_set({1}, False, _produce)
    writes = []
    orig = arena.write_row
    arena.write_row = lambda s, p: (writes.append(s), orig(s, p))
    missed = arena.ensure_set({1}, False, lambda fl: [])
    assert not missed
    assert writes == []
    assert arena.book.hits == 1


def test_rollback_restores_victim_on_failed_fetch():
    arena, store = _host(2)
    arena.ensure_set({0, 1}, False, _produce)
    assert arena.book.free == []

    def boom(fl):
        raise RuntimeError("fetch failed")

    with pytest.raises(RuntimeError):
        arena.ensure_set({2, 3}, False, boom)
    # Reservations rolled back: the two victims keep their residency.
    assert set(arena.book.slot_of) == {0, 1}


def test_rollback_frees_interrupted_row():
    arena, store = _host(2)
    arena.ensure_set({0, 1}, False, _produce)
    calls = []

    def produce(fl):
        # Commit the first entry, then fail on the second's write.
        payloads = [_payload(e) for e, _s, _v in fl]
        orig = arena.write_row

        def flaky(slot, payload):
            calls.append(slot)
            if len(calls) == 2:
                raise RuntimeError("mid-commit")
            return orig(slot, payload)

        arena.write_row = flaky
        return payloads

    with pytest.raises(RuntimeError):
        arena.ensure_set({2, 3}, False, produce)
    # Only the fully-committed entry stuck. The interrupted row is
    # suspect (can't tell pre-write from mid-write failure): freed back
    # to the pool and its victim stays evicted — rollback's contract.
    assert len(arena.book.slot_of) == 1
    assert 2 in arena.book.slot_of or 3 in arena.book.slot_of
    assert len(arena.book.free) == arena.book.rooms - 1


def test_rollback_restores_original_recency_order():
    """A failed ensure puts evicted victims back at their ORIGINAL
    recency positions — the MRU-end approximation would leave the
    survivors artificially oldest, so the next victim pick would evict
    decode-hot rows first."""
    arena, _ = _host(4)
    arena.ensure_set({0, 1, 2, 3}, False, _produce)
    order_before = list(arena.book.slot_of)
    slots_before = dict(arena.book.slot_of)

    def boom(fl):
        raise RuntimeError("fetch failed")

    with pytest.raises(RuntimeError):
        arena.ensure_set({4, 5}, False, boom)
    # Exact restore: same residents, same rows, same recency order —
    # under the MRU-end approximation the survivors 2,3 would lead.
    assert list(arena.book.slot_of) == order_before
    assert arena.book.slot_of == slots_before
    assert arena.book.free == []
    assert arena.book.evictions == 0


def test_rollback_recency_order_with_partial_commit():
    """Mixed outcome: a committed entry keeps its MRU position, the
    interrupted entry's victim stays evicted, and the unstarted victim
    returns to its ORIGINAL position — not the end."""
    arena, _ = _host(4)
    arena.ensure_set({0, 1, 2, 3}, False, _produce)
    calls = []

    def produce(fl):
        payloads = [_payload(e) for e, _s, _v in fl]
        orig = arena.write_row

        def flaky(slot, payload):
            calls.append(slot)
            if len(calls) == 2:
                raise RuntimeError("mid-commit")
            return orig(slot, payload)

        arena.write_row = flaky
        return payloads

    with pytest.raises(RuntimeError):
        arena.ensure_set({4, 5, 6}, False, produce)
    # Fetch order evicts 0,1,2: expert 4 committed (row 3), expert 5's
    # row was mid-write (freed, victim 1 stays evicted), expert 6's row
    # was never written (victim 2 restored at its original position).
    # Survivor 3 keeps its slot; slot_of order is [2, 3, 4] — the
    # approximation would have produced [3, 4, 2].
    assert list(arena.book.slot_of) == [2, 3, 4]
    assert arena.book.slot_of[4] == 3
    assert arena.book.slot_of[2] == 1
    assert arena.book.slot_of[3] == 0
    assert 5 not in arena.book.slot_of and 6 not in arena.book.slot_of
    assert 0 not in arena.book.slot_of and 1 not in arena.book.slot_of
    assert arena.book.free == [2]
    assert arena.book.evictions == 2


def test_grow_extends_rows_and_preserves_residents():
    arena, store = _host(2)
    arena.ensure_set({0, 1}, False, _produce)
    arena.book.cap = 4
    # Demand 3 residents > rooms 2 -> forces grow inside ensure_set.
    arena.ensure_set({0, 1, 2}, False, _produce)
    assert arena.book.rooms >= 3
    assert store["p"]["weight"].shape[0] >= 3
    # Old rows preserved through the grow.
    np.testing.assert_array_equal(
        np.asarray(store["p"]["weight"][arena.book.slot_of[0]]),
        np.full((2,), 1.0),
    )


def test_compact_keeps_mru_and_remaps():
    arena, store = _host(4)
    arena.ensure_set({0, 1, 2, 3}, False, _produce)
    # Touch 2,3 so 0,1 are LRU-oldest.
    arena.book.touch(2)
    arena.book.touch(3)
    drop = arena.compact(2)
    assert drop == 2
    assert set(arena.book.slot_of) == {2, 3}
    assert sorted(arena.book.slot_of.values()) == [0, 1]
    np.testing.assert_array_equal(
        np.asarray(store["p"]["weight"][arena.book.slot_of[2]]),
        np.full((2,), 3.0),
    )


def test_frozen_commits_at_oldest_end():
    arena, store = _host(4)
    arena.ensure_set({0, 1}, False, _produce)
    # Frozen (verify) miss: lands at the LRU-oldest end.
    arena.ensure_set({9}, True, _produce)
    assert list(arena.book.slot_of)[0] == 9
    # Frozen hits do not reorder.
    order = list(arena.book.slot_of)
    arena.ensure_set({0}, True, lambda fl: [])
    assert list(arena.book.slot_of) == order


def test_staged_leftovers_drop_and_count():
    arena, _ = _host(4)
    arena.staged[7] = object()
    arena.staged[8] = object()
    arena.ensure_set({0}, False, _produce)
    assert arena.staged == {}
    assert arena.staged_drops == 2


def test_staged_leftover_futures_cancelled():
    """Dropped staged entries cancel their in-flight reads."""
    from concurrent.futures import Future

    arena, _ = _host(4)
    fut = Future()
    arena.staged[7] = fut
    arena.staged[8] = object()  # non-future payloads just drop
    arena.ensure_set({0}, False, _produce)
    assert fut.cancelled()
    assert arena.staged == {}
    assert arena.staged_drops == 2


def test_produce_short_payload_count_rolls_back():
    """A produce() returning fewer payloads than fetches must roll back
    every reservation — not silently commit a partial set."""
    arena, _ = _host(2)
    arena.ensure_set({0, 1}, False, _produce)
    assert arena.book.free == []

    def short(fl):
        return [_payload(e) for e, _s, _v in fl][:-1]  # one short

    with pytest.raises(RuntimeError, match="payloads"):
        arena.ensure_set({2, 3}, False, short)
    # Victims restored: the two reservations are undone.
    assert set(arena.book.slot_of) == {0, 1}
    assert arena.book.free == []


def test_grow_once_per_ensure():
    """An N-miss ensure pays one physical grow, not N realloc rounds."""
    arena, store = _host(2)
    arena.ensure_set({0, 1}, False, _produce)
    arena.book.cap = 8
    grows = []
    orig = arena.grow
    arena.grow = lambda need: (grows.append(need), orig(need))[1]
    # All residents demanded, 3 misses, no free rows -> one grow call.
    arena.ensure_set({0, 1, 2, 3, 4}, False, _produce)
    assert grows == [5]  # misses(3) - free(0) - victims(0) = grow to 5
    assert arena.book.rooms == 5


def test_grow_once_counts_evictable_victims():
    """The pre-pass subtracts evictable victims — only the true shortfall
    grows physically."""
    arena, store = _host(4)
    arena.ensure_set({0, 1, 2, 3}, False, _produce)
    arena.book.cap = 6
    grows = []
    orig = arena.grow
    arena.grow = lambda need: (grows.append(need), orig(need))[1]
    # {4,5} needed: 2 misses, 4 evictable victims -> no grow. Only as
    # many victims evict as misses demand (0,1), the rest stay resident.
    arena.ensure_set({4, 5}, False, _produce)
    assert grows == []
    assert {4, 5}.issubset(set(arena.book.slot_of))
    assert len(arena.book.slot_of) == 4
    assert arena.book.rooms == 4


def test_set_cap_clamps_to_rooms_max():
    arena, _ = _host(4)
    assert arena.rooms_max == 4
    # Below the bound applies; past the bound clamps.
    assert arena.set_cap(2) == 2
    assert arena.book.cap == 2
    assert arena.set_cap(99) == 4
    assert arena.book.cap == 4


def test_rooms_max_follows_demand_growth():
    """Demand-driven grow() re-bases rooms_max so a later set_cap clamps
    against rows that physically exist, not the construction bound."""
    arena, _ = _host(2)
    arena.ensure_set({0, 1}, False, _produce)
    arena.book.cap = 4
    arena.ensure_set({0, 1, 2, 3}, False, _produce)
    assert arena.book.rooms == 4
    assert arena.rooms_max == 4
    # A shrink can close it again; growth back to the paid bound reopens.
    arena.set_cap(2)
    assert arena.book.cap == 2
    arena.set_cap(10)
    assert arena.book.cap == 4
