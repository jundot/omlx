# SPDX-License-Identifier: Apache-2.0
"""Shared expert-slot bookkeeping for the per-expert cache tracks.

The two non-unified caches — DeepSeek V4.1's ``_ExpertSlots`` and the
legacy ``ExpertCache`` — share the same expert-id → row machinery: an
LRU-ordered ``slot_of`` dict, a free-row list, the rooms-vs-cap split
(physical rows vs governor ceiling), and the acquire/commit/rollback
protocol that keeps a fetch failure from orphaning a row or silently
dropping the evicted victim's residency. This module is that machinery.

The unified ``ExpertLRUCache`` keeps its own internals (per-projection
slots, the s3fifo policy, cross-layer budget): only the visit-stats
contract below is shared, which is the shape the governor duck-reads.

Locking stays with the cache: ``SlotBookkeeping`` itself is not
thread-safe; callers serialize under their own per-layer lock.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterable

import mlx.core as mx

from ._env import env_int

# OMLX_EXPERT_STREAMING_HOTPIN: pin the K hottest experts per book against
# eviction. Routing frequency is tracked on the demand set each ensure_set
# call; pins are recomputed per call and counters halve periodically so the
# protected set follows the current routing distribution instead of
# freezing history. 0 (default) keeps plain LRU victims.
_HOTPIN_K = env_int("OMLX_EXPERT_STREAMING_HOTPIN", 0)
_HOTPIN_DECAY_CALLS = 512


@dataclass
class DecodeVisitStats:
    """Governor-facing visit counters: decode layer-calls and misses.

    ``decode_layers``/``decode_layers_missed`` count layer-CALLS (a visit
    that missed ≥1 expert stalls once regardless of miss count — the
    layer waits for its slowest read). ``decode_misses_by_layer`` is the
    governor's per-layer targeting signal: the per-expert tracks bump it
    once per stalled visit; the unified cache bumps it per missed slot —
    either way it ranks which layers hurt, which is all the governor reads.
    """

    decode_layers: int = 0
    decode_layers_missed: int = 0
    decode_misses_by_layer: dict = field(default_factory=dict)

    def note_visit(self, layer_idx: int, missed: bool) -> None:
        self.decode_layers += 1
        if missed:
            self.decode_layers_missed += 1
            self.note_miss(layer_idx)

    def note_miss(self, layer_idx: int) -> None:
        """Bump the per-layer miss tally directly.

        ``note_visit`` counts one stalled layer-CALL; the unified cache
        instead bumps per missed SLOT (each ``_get_unlocked`` miss during
        decode) — either way ``decode_misses_by_layer`` stays the
        governor's targeting signal.
        """
        self.decode_misses_by_layer[layer_idx] = (
            self.decode_misses_by_layer.get(layer_idx, 0) + 1
        )

    def reset_visits(self) -> None:
        self.decode_layers = 0
        self.decode_layers_missed = 0
        self.decode_misses_by_layer.clear()


def working_set_step(cap: int, top_k: int) -> int:
    """Tokens per forward chunk so a chunk's routed rows fit in *cap*.

    Every token routes to at most ``top_k`` experts, so ``cap // top_k``
    tokens can never touch more than ``cap`` distinct experts — an O(1)
    bound with no device→host syncs. ``top_k <= cap`` is a caller
    precondition (enforced upstream: a cache smaller than the routing
    width cannot serve).
    """
    top_k = max(1, int(top_k))
    return max(1, int(cap) // top_k)


class SlotBookkeeping:
    """Expert-id → resident-row map with LRU eviction and fetch rollback.

    ``slot_of`` is insertion-ordered: first entry = eviction victim, last =
    most recently used. ``rooms`` counts the physical rows allocated in the
    slot tensors; ``cap`` is the governor-driven working ceiling — after a
    shrink, rooms can exceed cap (rows stay allocated but must not host
    residents). Growth past rooms is physical (the caller reallocs), then
    ``grew_to`` opens the new rows here.

    Acquire/commit/rollback: ``acquire`` reserves a row BEFORE the fetch
    payload exists, ``commit`` publishes residency AFTER it is in place,
    and ``rollback`` undoes a partially-run batch — restoring victims whose
    rows were never overwritten and freeing the row whose bytes are
    suspect.
    """

    def __init__(self, rooms: int, cap: int | None = None) -> None:
        self.slot_of: OrderedDict[int, int] = OrderedDict()
        self.rooms = int(rooms)
        self.cap = self.rooms if cap is None else int(cap)
        self.free: list[int] = list(range(self.rooms))
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.freq: dict[int, int] = {}
        self.pinned: frozenset[int] = frozenset()
        self._demand_calls = 0
        # Verify-phase telemetry (V41 megaplan F0): experts committed while
        # a verify scope was active and still resident — they sit at the
        # cold end by construction (frozen commits land to_oldest), so
        # evicting one is harmless churn while evicting anything else is
        # consuming decode residency. ``verify_misses``/``verify_evict_resident``
        # count demand misses served under verify and victims popped during
        # verify that were NOT verify-installed, respectively.
        self.verify_installed: set[int] = set()
        self.verify_misses = 0
        self.verify_evict_resident = 0

    @staticmethod
    def _as_set(ids: Iterable[int]) -> set:
        return ids if isinstance(ids, (set, frozenset)) else set(ids)

    def note_demand(self, needed: Iterable[int], pin_k: int) -> None:
        """Frequency-track the demanded set and refresh the pinned hot set.

        Each demanded expert gains one count per call (hit or miss — the
        routing signal is the demand, not residency). Counters halve every
        ``_HOTPIN_DECAY_CALLS`` calls so pins track the live distribution.
        ``pin_k`` is clamped to ``cap - len(needed)`` so a fully demanded
        book always leaves evictable rows for the working set itself.
        """
        needed_set = self._as_set(needed)
        for expert in needed_set:
            self.freq[expert] = self.freq.get(expert, 0) + 1
        self._demand_calls += 1
        if self._demand_calls >= _HOTPIN_DECAY_CALLS:
            self.freq = {
                expert: halved
                for expert, count in self.freq.items()
                if (halved := count >> 1)
            }
            self._demand_calls = 0
        eff = min(int(pin_k), max(0, self.cap - len(needed_set)))
        if eff <= 0 or not self.freq:
            self.pinned = frozenset()
            return
        self.pinned = frozenset(
            sorted(self.freq, key=self.freq.get, reverse=True)[:eff]
        )

    def __contains__(self, expert: int) -> bool:
        return expert in self.slot_of

    def __len__(self) -> int:
        return len(self.slot_of)

    def touch(self, expert: int) -> bool:
        """Move *expert* to the MRU end when resident; returns residency."""
        if expert not in self.slot_of:
            return False
        self.slot_of.move_to_end(expert)
        return True

    def _note_eviction(self, victim: int, verify: bool) -> None:
        """Bookkeep a popped victim: verify-resident accounting + set fix.

        Under verify, evicting an entry that was NOT itself installed by
        verify traffic consumes decode residency — counted separately so
        the verify path's damage is observable (and later prevented).
        """
        if verify and victim not in self.verify_installed:
            self.verify_evict_resident += 1
        self.verify_installed.discard(victim)

    def _pop_victim(self, candidates, verify: bool) -> tuple[int, int] | None:
        """Pop the first hit of *candidates* → ``(expert, row)`` or None.

        *candidates* is the caller's victim policy as a lazy generator
        over ``slot_of`` keys (scan order + predicate stay with the
        caller); this pops the row and bookkeeps the eviction.
        """
        victim = next(candidates, None)
        if victim is None:
            return None
        row = self.slot_of.pop(victim)
        self.evictions += 1
        self._note_eviction(victim, verify)
        return victim, row

    def evict_oldest_outside(
        self,
        needed: Iterable[int],
        on_evict: Callable[[int], None] | None = None,
        verify: bool = False,
    ) -> tuple[int, int] | None:
        """Pop the LRU-oldest expert not in *needed* → ``(expert, row)``."""
        needed_set = self._as_set(needed)
        evicted = self._pop_victim(
            (
                e
                for e in self.slot_of
                if e not in needed_set and e not in self.pinned
            ),
            verify,
        )
        if evicted is None:
            return None
        if on_evict is not None:
            on_evict(evicted[0])
        return evicted

    def trim_to_cap(
        self,
        needed: Iterable[int] = (),
        on_evict: Callable[[int], None] | None = None,
        verify: bool = False,
        target: int | None = None,
    ) -> int:
        """Evict LRU-oldest entries outside *needed* until within ``cap``.

        ``target`` overrides the ceiling: the verify scratch path trims
        to ``cap + scratch`` so its transient rows are not evicted by
        the post-shrink trim that exists for decode-phase occupancy.
        """
        ceiling = self.cap if target is None else int(target)
        trimmed = 0
        while len(self.slot_of) > ceiling:
            evicted = self.evict_oldest_outside(needed, on_evict, verify)
            if evicted is None:
                break
            self.free.append(evicted[1])
            trimmed += 1
        return trimmed

    def acquire(
        self,
        needed: Iterable[int],
        verify: bool = False,
        scratch: int = 0,
    ) -> tuple[int, int | None, bool]:
        """Reserve a row for a missing expert.

        Returns ``(slot, victim, needs_grow)``: a free row, an evicted
        victim's row, or — when every resident is needed — ``(rooms,
        None, True)`` telling the caller to grow physical storage first,
        then take the new row from ``free``.

        Under ``verify`` with ``scratch > 0`` (V41 megaplan F4) only
        verify-installed rows are fair victims — pre-verify residents
        are decode-hot and verify must not consume them. While rooms
        are below ``cap + scratch`` the caller grows instead; above the
        bound the policy degrades gracefully to ordinary LRU victims.
        """
        if self.free:
            return self.free.pop(), None, False
        needed_set = self._as_set(needed)
        if verify and scratch:
            # Oldest verify-installed victim first: slot_of is head=cold,
            # so scan from the tail and take the LAST match — the
            # verify-installed entry nearest the decode residents is the
            # oldest-installed one (LRU-fair inside the scratch region).
            evicted = self._pop_victim(
                (
                    e
                    for e in reversed(self.slot_of)
                    if e in self.verify_installed
                    and e not in needed_set
                    and e not in self.pinned
                ),
                verify,
            )
            if evicted is not None:
                return evicted[1], evicted[0], False
            if self.rooms < self.cap + scratch:
                return self.rooms, None, True
        evicted = self._pop_victim(
            (
                e
                for e in self.slot_of
                if e not in needed_set and e not in self.pinned
            ),
            verify,
        )
        if evicted is not None:
            return evicted[1], evicted[0], False
        return self.rooms, None, True

    def grew_to(self, new_rooms: int) -> None:
        """Open rows ``[rooms, new_rooms)`` after a physical grow."""
        new_rooms = int(new_rooms)
        if new_rooms > self.rooms:
            self.free.extend(range(self.rooms, new_rooms))
            self.rooms = new_rooms

    def commit(self, expert: int, slot: int, *, to_oldest: bool = False) -> None:
        """Publish ``expert -> slot`` and count the miss.

        ``to_oldest`` inserts at the eviction-candidate end — the DSv4.1
        verify path uses it so draft-only experts leave first without
        disturbing the decode-hot order.
        """
        self.slot_of[expert] = slot
        if to_oldest:
            self.slot_of.move_to_end(expert, last=False)
        self.misses += 1

    def rollback(
        self,
        fetch_list: list[tuple[int, int, int | None]],
        committed_through: int,
        pass3_started: bool,
        order_snapshot: list[int] | None = None,
    ) -> None:
        """Undo slot assignments for entries whose install did not finish.

        ``fetch_list`` entries are ``(expert, slot, victim)`` in commit
        order. Unstarted entries still hold the victim's bytes — restore
        it. The entry interrupted mid-commit has suspect bytes: its row
        goes back to ``free`` and the victim stays evicted.

        With ``order_snapshot`` — ``slot_of``'s key order captured just
        before the call's first victim pop — restored victims reinsert
        at their ORIGINAL recency positions: the map is rebuilt as the
        snapshot order minus the victims that stay evicted, with the
        committed entries appended in commit order (matching the end
        positions ``commit`` gave non-frozen entries; a frozen commit's
        to_oldest head placement is approximated — residency and rows
        stay exact either way, only that edge's recency differs).
        Without a snapshot the restore falls back to the MRU end —
        residency and rows stay consistent, only recency is approximate.
        """
        # Victims whose rows host committed experts stay evicted, as
        # does the mid-commit interruption's victim (suspect bytes).
        stay_evicted = {
            victim
            for _e, _s, victim in fetch_list[: committed_through + 1]
            if victim is not None
        }
        # Unstarted entries still hold the victim's bytes — restore it.
        # The entry interrupted mid-commit has suspect bytes: its row
        # goes back to ``free`` and the victim stays evicted.
        restored: dict[int, int] = {}
        for j in range(committed_through + 1, len(fetch_list)):
            _expert, slot, victim = fetch_list[j]
            if pass3_started and j == committed_through + 1:
                self.free.append(slot)
                if victim is not None:
                    stay_evicted.add(victim)
            elif victim is not None:
                restored[victim] = slot
                self.evictions -= 1
            else:
                self.free.append(slot)
        if order_snapshot is None:
            # No snapshot: restored victims reinsert at the MRU end —
            # residency and rows stay consistent, only recency is
            # approximate.
            for victim, slot in restored.items():
                self.slot_of[victim] = slot
            return
        # Rebuild the order: snapshot keys keep their positions — the
        # still-resident ones at their current rows, restored victims at
        # the rows they were popped from. Entries committed before the
        # failure postdate the snapshot and append in commit order.
        rebuilt: OrderedDict[int, int] = OrderedDict()
        for e in order_snapshot:
            if e in stay_evicted:
                continue
            if e in self.slot_of:
                rebuilt[e] = self.slot_of[e]
            elif e in restored:
                rebuilt[e] = restored[e]
        for e, s in self.slot_of.items():
            if e not in rebuilt:
                rebuilt[e] = s
        self.slot_of = rebuilt

    def release(self, expert: int) -> None:
        """Drop *expert*'s residency, returning its row to ``free``."""
        slot = self.slot_of.pop(expert, None)
        if slot is not None:
            self.verify_installed.discard(expert)
            self.free.append(slot)

    def rebuild(self, kept: list[int]) -> None:
        """Remap rows to 0..k-1 in *kept* order after a physical compact."""
        self.slot_of = OrderedDict(
            (expert, row) for row, expert in enumerate(kept)
        )
        self.verify_installed.intersection_update(kept)
        self.rooms = len(kept)
        self.free = []

    def reset(self) -> list[int]:
        """Drop all residency; returns the evicted expert ids."""
        evicted = list(self.slot_of)
        self.slot_of.clear()
        self.verify_installed.clear()
        self.free = list(range(self.rooms))
        return evicted


class SlotArena:
    """Fixed-row residency arena shared by the per-expert slot caches.

    The host module owns the stacked ``(rooms, *row_shape)`` arrays per
    projection field, bound into whatever module layout it uses —
    QuantizedProjection fields for DeepSeek V4.1, a plain bank for the
    unified streaming linears. The arena owns everything else
    about *where* residents live: the ``SlotBookkeeping`` map, the
    per-arena lock, the two-phase grow/compact of physical rows, the
    demand-set acquire/commit/rollback protocol, and the speculative
    prefetch side dict.

    Payload production — which bytes to read and how to decode them —
    stays with the caller through the ``produce`` callback to
    ``ensure_set``; the arena only knows rows. Because residents are
    bound into fixed rows once at admission, the consumer gathers by
    slot id (``rhs_indices``) and a cache hit costs zero assembly —
    the ~1 ms/projection ``mx.stack`` per call a bundle-dict
    representation would pay is avoided entirely.

    Host bindings supplied at construction:
      ``arrays(proj) -> {field: mx.array}``  currently bound rows
      ``bind(proj, {field: array})``       rebind grown/compacted storage
      ``eval_params()``                    mx.eval of host params post-rebind
    """

    def __init__(
        self,
        capacity: int,
        projections,
        arrays: Callable,
        bind: Callable,
        eval_params: Callable | None = None,
    ) -> None:
        self.book = SlotBookkeeping(capacity)
        self.projections = tuple(projections)
        self._arrays_of = arrays
        self._bind = bind
        self._eval_params = eval_params
        # Physical ceiling fixed at construction: the builder sized
        # ``capacity`` from a byte bound, so later cap retargets may only
        # move INSIDE the bound — never past it (that would commit
        # unbounded bank memory, the class of bug the ceiling exists to
        # prevent).
        self.rooms_max = self.book.rooms
        # Per-arena lock: residency mutation (and its fetches) serialize
        # per layer only — a global backing lock would hold ALL layers'
        # IO against the governor's resize.
        self.lock = threading.RLock()
        # Speculative prefetch side dict: expert -> future of the payload
        # a demand fetch returns. Entries are consumed by the producer's
        # join or dropped as mispredicts at the end of an ensure.
        self.staged = {}
        self.staged_hits = 0
        self.staged_drops = 0
        self.staged_submits = 0

    def write_row(self, slot: int, payload: dict) -> None:
        """Write one payload ``{proj: {field: array}}`` into row ``slot``."""
        for proj, fields in payload.items():
            arrays = self._arrays_of(proj)
            for fname, array in fields.items():
                arrays[fname][slot] = array

    def resident_bytes(self) -> int:
        """Bytes committed by this arena's bound banks.

        Sum of every bound array's ``nbytes`` across projections —
        physical rows are committed memory whether or not they host a
        resident (a grown-but-free row still occupies the bank), so the
        accounting is rooms x row-bytes, not len(slot_of). Deliberately
        unlocked: ``nbytes`` is a metadata read and a mid-rebind caller
        may see either generation — for byte accounting either is
        correct, and taking the arena lock from a cache-locked caller
        would add a lock order for no precision gain.
        """
        total = 0
        try:
            for proj in self.projections:
                for array in self._arrays_of(proj).values():
                    total += int(getattr(array, "nbytes", 0) or 0)
        except Exception:
            pass
        return total

    def set_cap(self, cap: int) -> int:
        """Retarget the residency ceiling; returns the applied cap.

        Clamped to the construction-time physical bound ``rooms_max`` —
        the byte ceiling the builder sized is the hard limit, so a
        governor grow can only re-open rows the bound already paid for.
        Residents above the new cap are not evicted here; the next
        ``ensure_set`` trims them via ``trim_to_cap`` (eviction order
        stays demand-scoped).
        """
        with self.lock:
            self.book.cap = max(1, min(int(cap), self.rooms_max))
            return self.book.cap

    def grow(self, need: int) -> None:
        """Extend physical rooms to ``need`` (copy rows, extend free).

        Two-phase: every projection's grown arrays are built and evaluated
        BEFORE any rebind, so an OOM mid-way leaves the host on the old
        consistent storage instead of mixed-row projections.
        """
        if self.book.rooms >= need:
            return
        with self.lock:
            grown_all = {}
            for proj in self.projections:
                current = self._arrays_of(proj)
                grown = {
                    fname: mx.zeros((need, *array.shape[1:]), dtype=array.dtype)
                    for fname, array in current.items()
                }
                for fname, array in current.items():
                    grown[fname][: self.book.rooms] = array
                grown_all[proj] = grown
            self._rebind_all(grown_all)
            self.book.grew_to(need)
            # Demand-driven growth re-bases the physical ceiling: rows
            # already committed are a paid cost, so a later set_cap must
            # clamp against what exists, not the construction-time size.
            self.rooms_max = max(self.rooms_max, self.book.rooms)

    def compact(self, keep: int, rooms: int | None = None) -> int:
        """Keep the most-recent ``keep`` entries, remap rows 0..k-1.

        Two-phase like ``grow``: stacks for ALL projections evaluate
        before the first rebind, so a mid-compact failure cannot serve a
        half-remapped layer.

        ``rooms`` optionally sets the physical row count after the remap
        (>= len(kept)): the transient prefill-cap path uses it to hand
        back grown-but-free rows at call end — dropping residents is not
        required to release empty storage.
        """
        with self.lock:
            order = list(self.book.slot_of)
            keep = max(0, min(int(keep), len(order)))
            if rooms is None:
                # No-op unless keep shrinks the live set — raising keep
                # is a ceiling change (lazy growth covers expansion);
                # empty rooms are the physical floor, not slack to
                # release.
                if len(order) <= keep:
                    return 0
                target = keep
            else:
                target = max(int(rooms), keep)
                # Explicit rooms shrink: release grown-but-free rows even
                # when no resident drops — but never below len(kept).
                if len(order) <= keep and self.book.rooms <= target:
                    return 0
            drop = len(order) - keep
            kept = order[drop:]
            stacked_all = {
                proj: {
                    fname: self._kept_rows(
                        array,
                        [self.book.slot_of[e] for e in kept],
                        target,
                    )
                    for fname, array in self._arrays_of(proj).items()
                }
                for proj in self.projections
            }
            self._rebind_all(stacked_all)
            self.book.rebuild(kept)
            if target > len(kept):
                self.book.free.extend(range(len(kept), target))
                self.book.rooms = target
            return drop

    def _rebind_all(self, new_banks: dict) -> None:
        """Evaluate staged banks for ALL projections, then rebind.

        Two-phase: the mx.eval of every built array precedes the first
        ``_bind``, so an OOM mid-way leaves the host on the old
        consistent storage instead of mixed-row projections.
        """
        mx.eval(*[a for bank in new_banks.values() for a in bank.values()])
        for proj in self.projections:
            self._bind(proj, new_banks[proj])
        if self._eval_params is not None:
            self._eval_params()

    @staticmethod
    def _kept_rows(array, rows, target):
        """Stack ``rows`` of ``array`` into a (target, *row_shape) bank."""
        kept = (
            mx.stack([array[row] for row in rows])
            if rows
            else mx.zeros((0, *array.shape[1:]), dtype=array.dtype)
        )
        if target <= kept.shape[0]:
            return kept
        grown = mx.zeros((target, *array.shape[1:]), dtype=array.dtype)
        grown[: kept.shape[0]] = kept
        return grown

    def ensure_set(
        self,
        needed,
        frozen: bool,
        produce: Callable,
        verify: bool = False,
        scratch: int = 0,
    ) -> bool:
        """Cover ``needed`` expert ids; returns True if any miss committed.

        Pass 1 reserves a row per missing expert (``acquire`` — free row,
        evicted victim's row, or physical grow). ``produce(fetch_list)``
        then runs the caller's IO+decode and returns payloads aligned
        with it. Pass 3 writes each payload into its row and publishes
        residency in fetch order — ``to_oldest`` under ``frozen`` so
        verify-path experts become first eviction candidates without
        disturbing the decode-hot order. A failure anywhere rolls back
        the uncommitted tail: untouched rows restore their victim, the
        interrupted row is freed.

        Physical growth runs at most once per ensure: a pre-pass predicts
        the row count the call can need (residents + misses − free rows −
        evictable victims) and grows straight to it, so an N-expert miss
        pays one realloc+rebind round instead of N.

        Leftover staged entries (predicted but not demanded) drop here —
        their futures are cancelled so an in-flight speculative read
        stops early instead of finishing into a dropped payload.

        ``verify`` (V41 megaplan F0 telemetry, F4 scratch): marks the call
        as draft-verify traffic — committed experts join
        ``book.verify_installed`` (they already land to_oldest under
        ``frozen``) and evictions of non-verify-installed residents count
        as ``verify_evict_resident``. With ``scratch > 0`` the admission
        bound becomes ``cap + scratch`` rows: a verify block's union fits
        in ONE ensure instead of re-splitting into one-token chunks, and
        ``acquire`` evicts only verify-installed rows — growing rooms up
        to the bound rather than consuming decode-hot residents. The
        scratch content is logically transient (cold-end residents) but
        persists between verify rounds for cross-iteration hits; the
        decode-phase ``trim_to_cap`` reclaims it on the next ordinary
        ensure, and the governor's compact drops it under pressure.
        """
        needed = set(needed)
        limit = self.book.cap + scratch if verify and scratch else self.book.cap
        if len(needed) > limit:
            raise ValueError("Expert working set exceeds resident capacity")
        fetch_list = []
        committed_through = -1
        started = False
        order_snapshot = None
        try:
            if _HOTPIN_K:
                self.book.note_demand(needed, _HOTPIN_K)
            # Post-shrink trim: evict oldest non-needed down to the
            # ceiling. Safe: needed <= cap < len implies a non-needed
            # entry exists. Under verify+scratch the ceiling is the
            # scratch bound, so the trim does not undo the transient
            # residency this call is allowed to hold.
            self.book.trim_to_cap(
                needed, verify=verify, target=limit
            )
            # Protect the entire working set, including hits after
            # misses.
            resident = 0
            for expert in needed:
                if expert in self.book:
                    self.book.hits += 1
                    resident += 1
                    if not frozen:
                        self.book.touch(expert)
            misses = len(needed) - resident
            if verify:
                self.book.verify_misses += misses
            # Grow-once pre-pass: the largest row count this call can
            # need is what the miss set requires after free rows and
            # evictable victims are spent. Pinned-but-not-needed rows are
            # NOT evictable, so demand may legitimately push rooms past
            # cap — occupancy over the ceiling is forced by the pins,
            # not by this bound. Under verify+scratch the victim pool is
            # the verify-installed set only (decode-hot residents are not
            # forfeit), and growth is bounded by the scratch ceiling.
            pool = self.book.verify_installed if verify and scratch else None
            victims_available = sum(
                1
                for e in self.book.slot_of
                if (pool is None or e in pool)
                and e not in needed
                and e not in self.book.pinned
            )
            grow_by = max(0, misses - len(self.book.free) - victims_available)
            if verify and scratch:
                grow_by = min(grow_by, max(0, limit - self.book.rooms))
            if grow_by:
                self.grow(self.book.rooms + grow_by)
            for expert in needed:
                if expert in self.book:
                    continue
                if order_snapshot is None and not self.book.free:
                    # acquire() evicts only once free rows are spent —
                    # capture the recency order just before this call's
                    # first victim pop so a rollback can reinsert victims
                    # at their ORIGINAL positions instead of the MRU end.
                    order_snapshot = list(self.book.slot_of)
                slot, victim, needs_grow = self.book.acquire(
                    needed, verify=verify, scratch=scratch
                )
                if needs_grow:
                    # Safety net only: the pre-pass above opens every row
                    # the demand can need, so this cannot fire without
                    # bookkeeping drift. Keep the one-row grow rather than
                    # corrupting the book.
                    self.grow(self.book.rooms + 1)
                    slot = self.book.free.pop()
                fetch_list.append((expert, slot, victim))
            payloads = produce(fetch_list)
            if len(payloads) != len(fetch_list):
                raise RuntimeError(
                    "arena produce returned %d payloads for %d fetches"
                    % (len(payloads), len(fetch_list))
                )
            started = True
            for i, ((expert, slot, _victim), payload) in enumerate(
                zip(fetch_list, payloads)
            ):
                self.write_row(slot, payload)
                self.book.commit(expert, slot, to_oldest=frozen)
                if verify:
                    self.book.verify_installed.add(expert)
                committed_through = i
        except Exception:
            self.book.rollback(
                fetch_list, committed_through, started, order_snapshot
            )
            raise
        if self.staged:
            staged = list(self.staged.values())
            self.staged_drops += len(staged)
            self.staged.clear()
            for fut in staged:
                try:
                    fut.cancel()
                except Exception:
                    pass
        return committed_through >= 0

    def rows_for(self, expert_ids):
        """Slot row indices for ``expert_ids`` (all must be resident)."""
        return mx.array(
            [self.book.slot_of[int(e)] for e in expert_ids],
            dtype=mx.int32,
        )
