# SPDX-License-Identifier: Apache-2.0
"""Bounded per-layer expert cache: stats, admission, and the LRU store.

``CacheStats`` is the counter block the governor duck-reads,
``_AdmissionWorker`` the detached copy+put path that keeps demand-miss
admission off the inference thread, ``_layer_index_*`` the shared
per-layer recency index policy subclasses reuse, and ``ExpertLRUCache``
the unified store itself (global byte budget split into per-layer caps,
phase-aware decode/prefill pairs, governor resize, the weak arena
registry and the live-cache registry ``_LIVE_STREAMING_CACHES`` that
``streaming_gate_state`` aggregates for the MTP gate).
``make_expert_cache`` picks the eviction policy; ``s3fifo`` lives in
``cache_policies`` and is imported lazily so policy modules can subclass
``ExpertLRUCache`` without a cycle.
"""

from __future__ import annotations

import logging
import queue as _queue
import threading
import weakref
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from ._env import env_str
from .slot_cache import DecodeVisitStats
from .speculation import SpeculationState

logger = logging.getLogger(__name__)


# Detached demand admission. Decode-miss rows are views into shared bank
# buffers — caching them retains the whole bank per entry (~9x the
# per-expert accounting on prefill-sized banks). The detached worker
# copies the row into a private np array off the demand path and
# batch-puts under one lock, so admission costs ~a queue push on the
# critical path. (were OMLX_EXPERT_STREAMING_ADMIT_Q / _ADMIT_BATCH)
_ADMIT_Q_MAX = 8192
_ADMIT_BATCH = 96


# Live-cache registry for cross-subsystem signals (the MTP gate in
# batch_generator). Weak so a dead engine's cache never lingers;
# `closed` marks explicit teardown.
_LIVE_STREAMING_CACHES: "weakref.WeakSet" = weakref.WeakSet()


def streaming_gate_state() -> "dict | None":
    """Aggregate signal for the MTP gate (batch_generator).

    Returns None when no live bounded streaming cache exists — models
    without expert streaming are never constrained. Otherwise the current
    decode hit rate (the verify path's RAM-serve share) and capacity.
    """
    hits = misses = capacity = 0
    active = False
    for c in list(_LIVE_STREAMING_CACHES):
        if getattr(c, "closed", False) or getattr(c, "capacity", 0) <= 0:
            continue
        st = getattr(c, "stats", None)
        if st is None:
            continue
        active = True
        capacity += int(c.capacity)
        hits += int(getattr(st, "decode_hits", 0))
        misses += int(getattr(st, "decode_misses", 0))
    if not active:
        return None
    return {
        "active": True,
        "capacity": capacity,
        "decode_hit_rate": hits / max(1, hits + misses),
    }


@dataclass
class CacheStats(DecodeVisitStats):
    # decode_layers / decode_layers_missed / decode_misses_by_layer are
    # inherited from DecodeVisitStats — the contract the governor reads
    # (and the same shape _V41CacheStats / _LegacyCacheStats expose).
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    # Demand-scoped counters. `hits`/`misses` count EVERY cache.get(),
    # which includes prefill lookups and the rolling path's
    # prefetch+ensure double split — a denominator over 2x the decode
    # demand that dilutes the hit-rate. These count exactly once per
    # projection, at the authoritative resolve boundary (_ensure_union
    # loop / _ensure_rolling), split by call shape.
    decode_hits: int = 0
    decode_misses: int = 0
    prefill_hits: int = 0
    prefill_misses: int = 0
    # Per-LAYER stall counters.
    # Every counter above counts *misses*, but a layer that misses 1 expert
    # and a layer that misses 40 stall for roughly the same time: the reads
    # of one layer are issued together and the layer waits for the slowest
    # of them, not for the sum. So miss count is a poor proxy for wall time
    # and the quantity worth minimising is the fraction of layer-calls that
    # missed AT ALL. These count layer-calls (one per _LayerLoadContext),
    # split by the same shape test as the demand counters above.
    prefill_layers: int = 0
    prefill_layers_missed: int = 0

    def reset(self) -> None:
        """Zero every counter in place.

        Callers like ``_count_demand`` grab a local reference to
        ``cache.stats`` and then increment it, so rebinding ``self.stats``
        to a fresh CacheStats would drop concurrent increments onto an
        orphaned object. Resetting in place keeps every holder pointed at
        the truth.
        """
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.decode_hits = 0
        self.decode_misses = 0
        self.prefill_hits = 0
        self.prefill_misses = 0
        self.prefill_layers = 0
        self.prefill_layers_missed = 0
        self.reset_visits()


class _AdmissionWorker:
    """Detached cache admission: copy + batch-put off the demand path.

    Demand-miss rows resolved by the ctx paths are views into shared bank
    buffers; caching the view pins the whole bank (up to ~9x the per-entry
    accounting). The worker copies each admitted row into a private np
    array — np only, never MLX — applies the 2nd-touch window filter, and
    flushes via ``put_many`` under a single lock acquisition. The demand
    path only pays a ``put_nowait``.
    """

    def __init__(self, cache: "ExpertLRUCache"):
        # Weakref, not strong: a worker that pins its cache keeps dead
        # caches live in the gate registry forever. When the cache dies
        # the queue stops being fed and the worker exits on the next item
        # it drains.
        self._cache_ref = weakref.ref(cache)
        self._q: _queue.Queue = _queue.Queue(maxsize=_ADMIT_Q_MAX)
        self._t = threading.Thread(
            target=self._run, name="omlx-expert-admit", daemon=True
        )
        self._t.start()

    def submit(self, key: tuple[int, int, str], row: Any) -> None:
        try:
            self._q.put_nowait((key, row))
        except _queue.Full:
            pass  # shed load: a dropped row is simply never cached

    def _put_batch(self, batch: list) -> None:
        cache = self._cache_ref()
        if cache is None:
            return
        items = []
        for key, row in batch:
            try:
                if not cache.admission_note(key):
                    continue
                items.append(
                    (
                        key,
                        tuple(None if x is None else np.array(x) for x in row),
                    )
                )
            except Exception:
                continue
        if not items:
            return
        try:
            cache.put_many(items)
        except Exception:
            logger.debug(
                "expert_streaming: admission put_many of %d rows failed",
                len(items),
                exc_info=True,
            )

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            batch = [item]
            while len(batch) < _ADMIT_BATCH:
                try:
                    nxt = self._q.get_nowait()
                except _queue.Empty:
                    break
                if nxt is None:
                    self._put_batch(batch)
                    return
                batch.append(nxt)
            self._put_batch(batch)

    def close(self) -> None:
        try:
            self._q.put_nowait(None)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-layer recency index: {layer: OrderedDict[key -> None]} ordered by
# recency. Module-level so policy subclasses (S3FIFO keeps one index per
# queue) share the same add/touch/drop bodies via ``index_map`` first arg.
# ---------------------------------------------------------------------------


def _layer_index_add(index_map: dict[int, OrderedDict], layer: int, key) -> None:
    order = index_map.get(layer)
    if order is None:
        order = index_map[layer] = OrderedDict()
    order[key] = None


def _layer_index_touch(index_map: dict[int, OrderedDict], layer: int, key) -> None:
    order = index_map.get(layer)
    if order is not None and key in order:
        order.move_to_end(key)


def _layer_index_drop(index_map: dict[int, OrderedDict], layer: int, key) -> None:
    order = index_map.get(layer)
    if order is not None:
        order.pop(key, None)


class ExpertLRUCache:
    """Per-layer LRU for expert slices (global budget split evenly).

    Each slot holds one expert's bundle for one layer (weight+scales+biases).
    Budget is split across MoE layers → per-layer capacity = budget // (layers*per_expert)
    approximated via total capacity, but eviction is per-layer to avoid cross-layer thrashing.
    `size`/`capacity` remain global totals for logging.
    """

    def __init__(
        self,
        budget_bytes: int,
        per_slot_bytes: int | None = None,
        num_layers: int | None = None,
        *,
        per_expert_bytes: int | None = None,
    ):
        # Concurrency: the cache is mutated from three threads -- the MLX
        # inference thread (get/put during forward), the warm-pool workers
        # (worker-side bundle puts), and the asyncio event loop (governor
        # resize at request boundaries). Every read of more than one
        # field, and every mutation of _store / _layer_counts /
        # _layer_orders / stats, happens under this lock. It is
        # re-entrant so the internal helpers can keep calling each other,
        # and it is never held across mx.eval or a blocking future wait.
        self._lock = threading.RLock()
        # ``per_expert_bytes`` is the deprecated name: one slot is ONE
        # projection's share of an expert, not a whole expert.
        if per_slot_bytes is None:
            per_slot_bytes = per_expert_bytes
        self.per_slot_bytes = int(per_slot_bytes or 0)
        self.num_layers = int(num_layers) if num_layers else 0
        if self.per_slot_bytes > 0:
            self.capacity = max(1, budget_bytes // self.per_slot_bytes) if budget_bytes > 0 else 0
        else:
            self.capacity = 0
        # per-layer stores to avoid global thrashing (layer 47 evicting layer 0)
        self._per_layer_cap = (
            max(1, self.capacity // self.num_layers)
            if self.num_layers > 0 and self.capacity > 0
            else self.capacity
        )
        # Dynamic budget (auto default): per-layer cap overrides for the
        # governor's targeted growth ({layer: slots}; empty = uniform).
        # Cleared by resize() back to uniform; survives clear() (policy,
        # not contents).
        self._per_layer_cap_over: dict[int, int] = {}
        # Phase-aware caps. The decode pair (capacity/_per_layer_cap) is
        # the dynamic one the governor tunes. Prefill streams through with
        # its own smaller pair so a big prompt cannot displace the decode
        # hot set; the warmer re-seeds + retain_hot after prefill repair
        # what churn remains. None = derive (global // 4, per-layer // 4).
        self._active_decode = True  # steady state; _LayerLoadContext corrects
        self._prefill_global_cap: int | None = None
        self._prefill_per_layer: int | None = None
        self._prefill_pinned = False
        self._derive_prefill_caps()
        self._store: OrderedDict[tuple[int, int, str], Any] = OrderedDict()
        # per-layer tracking for eviction
        self._layer_counts: Dict[int, int] = {}
        # Per-layer LRU index (key -> None, recency order) so the
        # per-layer victim is O(1) — without it `put` scans the whole
        # store on every insert once every layer sits at its cap.
        # Subclasses that override put (s3fifo) keep their own victim
        # selection and never read this, so a stale index can only affect
        # the base policy.
        self._layer_orders: dict[int, OrderedDict] = {}
        self.stats = CacheStats()
        # Per-conversion speculation state (set by the converter).
        self.spec_state: SpeculationState | None = None
        # Second-touch admission window: counts *demand sightings* (one
        # note per submitted miss) so a key admits from its second demand
        # under pressure. The worker applies it before copying; put_many
        # itself is unfiltered (seed/staged puts never touch this window).
        # The window scales with capacity — a fixed 1024-entry window is
        # noise next to a multi-GiB working set.
        self._admission_window = max(1024, min(self.capacity // 4, 16384))
        self._admit_counts: Dict[Tuple[int, int, str], int] = {}
        self._admit_order: deque[Tuple[int, int, str]] = deque()  # type: ignore[type-arg]
        # Live-cache registry: the MTP gate (batch_generator) reads the
        # aggregate signal — weakrefs so a closed engine's cache never
        # lingers past GC.
        try:
            _LIVE_STREAMING_CACHES.add(self)
        except Exception:
            pass
        self.closed = False
        # Detached admission worker. Owned per-cache so two engines
        # never share a worker; closed by ExpertBackingStore.close() via
        # _streaming_cache.
        self.admission: _AdmissionWorker | None = (
            _AdmissionWorker(self) if self.capacity > 0 else None
        )
        # Per-engine ctx fallback counters (per reason).
        self._ctx_fallbacks: dict[str, int] = {}
        # Per-key repeated-read-failure counter. Bumped from IO pool
        # workers, so it needs the same lock as everything else here.
        self._read_failures: dict[tuple[int, int, str], int] = {}
        # Arena registry (weak): the governor's resize / set_layer_caps
        # push the active per-layer cap into each layer's SlotArena so the
        # cache ceiling and the arena ceiling cannot diverge between engage
        # calls. Entries are (weakref(arena), layer_idx).
        self._arenas: list = []

    def note_read_failure(self, key: tuple[int, int, str]) -> int:
        """Atomically bump and return the failure count for *key*.

        Called straight from warm/IO pool workers, so the read-modify-
        write must run under the cache lock — an unlocked get/set loses
        concurrent increments on the same key.
        """
        with self._lock:
            n = int(self._read_failures.get(key, 0) or 0) + 1
            self._read_failures[key] = n
            return n

    def _count_ctx_fallback(self, reason: str) -> None:
        # Per-engine/per-conversion counters (module globals would mix
        # sessions in a persistent server). Reasons: read_failure (ctx
        # read produced nothing usable), bank_too_large (union declined
        # over _CTX_UNION_MAX_BYTES or the rolling bank cap),
        # tier_mismatch (bundles did not cover the demand set),
        # dict_backing (projections without a bank reader, so no context
        # was built at all), bank_read_l<i> and bank_promote_fail.
        # Read-modify-write from the inference thread and from IO workers;
        # the lock makes the increment atomic.
        with self._lock:
            self._ctx_fallbacks[reason] = self._ctx_fallbacks.get(reason, 0) + 1

    def ctx_fallback_stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._ctx_fallbacks)

    def __contains__(self, key: tuple[int, int, str]) -> bool:
        with self._lock:
            return key in self._store

    @property
    def per_expert_bytes(self) -> int:
        """Deprecated alias for ``per_slot_bytes``.

        One slot is one (layer, expert, projection-key) bundle — NOT a
        whole expert (gate+up+down are separate slots). Out-of-tree
        callers (moe_expert_offload's resident-size probe, older tests)
        still read/write this name; keep it routed at the new field.
        """
        return self.per_slot_bytes

    @per_expert_bytes.setter
    def per_expert_bytes(self, value) -> None:
        self.per_slot_bytes = int(value or 0)

    def peek(self, key: tuple[int, int, str]) -> Any | None:
        """Membership/value probe that never counts or promotes.

        Speculative paths (rolling prefetch, staged-read selection) use
        this so a peeked hit does not inflate stats.hits, mint a
        decode_misses_by_layer entry, or move recency — the demand touch
        remains the only admission/recency signal.
        """
        with self._lock:
            return self._store.get(key)

    def peek_many(self, keys: list) -> list:
        """Batch peek under one lock (same contract as get_many)."""
        with self._lock:
            store = self._store
            return [store.get(k) for k in keys]

    def resident_keys(self) -> set:
        """Snapshot of resident bundle keys under the cache lock.

        adaptive_topk's residency probe reads this instead of ``_store``
        directly: the base LRU holds everything in ``_store``, but the
        S3FIFO policy keeps probationary residents in ``_small`` — an
        unlocked ``_store`` read both races and misses them.
        """
        with self._lock:
            return set(self._store)

    def _register_arena(self, arena: Any, layer: int) -> None:
        """Register a layer's SlotArena for cap propagation."""
        with self._lock:
            # Compact dead weakrefs before appending: arenas die with
            # their GLU/engine, and an unpruned list grows one corpse
            # per model reload.
            self._arenas = [
                (ref, lyr) for ref, lyr in self._arenas if ref() is not None
            ]
            self._arenas.append((weakref.ref(arena), int(layer)))

    def _sync_arena_caps(self) -> None:
        """Push the active per-layer cap into every registered arena.

        Called after resize() / set_layer_caps() — the arena's book.cap
        follows the cache's per-layer ceiling so a governor shrink takes
        effect on the arena immediately, not at some later engage.
        """
        arenas = self._arenas
        for ref, layer in arenas:
            arena = ref()
            if arena is None:
                continue
            try:
                arena.set_cap(self._cap_for(layer))
            except Exception:
                logger.debug(
                    "expert_streaming: arena cap sync skipped for layer %d",
                    layer,
                    exc_info=True,
                )

    def _layer_of(self, key: tuple[int, int, str]) -> int:
        try:
            return int(key[0])
        except Exception:
            return -1

    # -- dynamic budget: phase + per-layer caps --------------------------

    def _derive_prefill_caps(self) -> None:
        """Derive the prefill pair from the decode pair (unless pinned)."""
        if self._prefill_pinned:
            return
        self._prefill_global_cap = max(32, self.capacity // 4)
        self._prefill_per_layer = max(1, self._per_layer_cap // 4) if self._per_layer_cap else 0

    def set_prefill_budget_slots(self, total_slots: int | None) -> None:
        """Pin (or unpin with None) the prefill pair from total slots."""
        with self._lock:
            if total_slots is None:
                self._prefill_pinned = False
                self._derive_prefill_caps()
                return
            total = max(0, int(total_slots))
            num = max(1, self.num_layers)
            self._prefill_global_cap = total
            self._prefill_per_layer = max(1, total // num) if total > 0 else 0
            self._prefill_pinned = True

    def note_phase(self, is_decode: bool) -> None:
        """Select the active cap pair for the calling layer-call.

        Called by _LayerLoadContext (and the arena engage path) once per
        layer-call from the call's real sequence length — the same
        ``_decode_call_shape`` test the decode/prefill stats split uses.
        Last-writer-wins is safe: the pairs only bound eviction, so a
        flap costs residency precision, never correctness.
        """
        with self._lock:
            self._active_decode = bool(is_decode)

    def _cap_for(self, layer: int) -> int:
        """Active per-layer admission cap (overrides apply to decode)."""
        if self._active_decode:
            over = self._per_layer_cap_over.get(layer)
            if over is not None:
                return over
            return self._per_layer_cap
        if self._prefill_per_layer:
            return self._prefill_per_layer
        return self._per_layer_cap

    def _global_cap_active(self) -> int:
        if not self._active_decode and self._prefill_global_cap is not None:
            return self._prefill_global_cap
        return self.capacity

    def set_layer_caps(self, mapping: dict[int, int] | None) -> None:
        """Replace the governor's per-layer cap overrides (None/{} clears)."""
        with self._lock:
            self._per_layer_cap_over = (
                {int(k): max(1, int(v)) for k, v in dict(mapping or {}).items()}
            )
        # Outside the cache lock on purpose: set_cap takes the per-layer
        # arena lock, and arena.locked paths never take the cache lock —
        # keeping this outside means no lock order exists to invert.
        self._sync_arena_caps()

    def layer_cap_overrides(self) -> dict[int, int]:
        with self._lock:
            return dict(self._per_layer_cap_over)

    # -- per-layer LRU index (O(1) victim selection) ------------------------

    def _dec_layer_count(self, layer: int) -> None:
        self._layer_counts[layer] = max(
            0, self._layer_counts.get(layer, 1) - 1
        )

    # Thin wrappers over the module-level index helpers (``_layer_orders``
    # is this policy's only index map).
    def _layer_index_add(self, key: tuple[int, int, str], layer: int) -> None:
        _layer_index_add(self._layer_orders, layer, key)

    def _layer_index_touch(self, key: tuple[int, int, str], layer: int) -> None:
        _layer_index_touch(self._layer_orders, layer, key)

    def _layer_index_drop(self, key: tuple[int, int, str], layer: int) -> None:
        _layer_index_drop(self._layer_orders, layer, key)

    def _evict_indexed_unlocked(self, layer: int, store, order) -> bool:
        """Evict the first entry of *order* (a layer's recency index) that
        is still live in *store*.

        Stale heads — keys a path removed without index maintenance —
        drop off until a live victim is found, so each check is O(1)
        instead of a full index copy. True when one went.
        """
        while order:
            victim = next(iter(order))
            del order[victim]
            if victim in store:
                store.pop(victim, None)
                self.stats.evictions += 1
                self._dec_layer_count(layer)
                return True
        return False

    def _evict_layer_scan_unlocked(self, layer: int, stores=None) -> bool:
        """Fallback per-layer victim: the first entry of *layer* found by
        scanning *stores* (default ``self._store``) in order.

        Reached when the index has no live entry for the layer —
        subclasses that keep their own stores never maintain
        ``_layer_orders``, so it can be stale or empty.
        """
        for store in (self._store,) if stores is None else stores:
            for k in list(store.keys()):
                if self._layer_of(k) == layer:
                    store.pop(k)
                    self.stats.evictions += 1
                    self._dec_layer_count(layer)
                    return True
        return False

    def _evict_layer_unlocked(self, layer: int) -> bool:
        """Evict the least-recently-used entry of `layer`; True if one went.

        O(1) via the per-layer index. Falls back to the linear scan when the
        index has no live entry for the layer (subclasses that keep their own
        stores never maintain the index, so it can go stale).
        """
        if self._evict_indexed_unlocked(
            layer, self._store, self._layer_orders.get(layer)
        ):
            return True
        return self._evict_layer_scan_unlocked(layer)

    def _enforce_layer_cap(self, layer: int) -> bool:
        """Evict one entry of *layer* when it sits at its per-layer cap.

        True when the layer is under cap or a victim went; False when the
        cap bound but the per-layer eviction found nothing — the caller's
        global-cap path is the fallback for that index/store
        disagreement.
        """
        if self.num_layers > 0:
            cap = self._cap_for(layer)
            if cap and self._layer_counts.get(layer, 0) >= cap:
                return self._evict_layer_unlocked(layer)
        return True

    def _rebuild_layer_index(self, *stores) -> tuple[dict[int, int], list]:
        """Recompute per-layer counts and recency indexes over live *stores*.

        Returns ``(counts, index_maps)`` — one fresh ``{layer:
        OrderedDict}`` index per store, in argument order. The S3FIFO
        subclass passes its two queues (each gets its own index map); the
        base policy passes ``_store`` alone.
        """
        counts: Dict[int, int] = {}
        index_maps = [OrderedDict() for _ in stores]
        for store, index_map in zip(stores, index_maps):
            for key in store:
                layer = self._layer_of(key)
                counts[layer] = counts.get(layer, 0) + 1
                _layer_index_add(index_map, layer, key)
        return counts, index_maps

    def get(self, key: tuple[int, int, str]) -> Any | None:
        with self._lock:
            return self._get_unlocked(key)

    def get_many(self, keys: list) -> list:
        """Batch get under one lock — per-key semantics of get(), minus
        one RLock acquire/release per expert (a decode split looks up
        ~60 ids x 3 projections x 48 layers per token)."""
        with self._lock:
            return [self._get_unlocked(k) for k in keys]

    def _get_unlocked(self, key: tuple[int, int, str]) -> Any | None:
        v = self._store.get(key)
        if v is not None:
            self._store.move_to_end(key)
            self._layer_index_touch(key, self._layer_of(key))
            self.stats.hits += 1
            return v
        self.stats.misses += 1
        if self._active_decode:
            self.stats.note_miss(self._layer_of(key))
        return None

    def admission_note(self, key: tuple[int, int, str]) -> bool:
        """2nd-touch filter for the detached admission worker.

        One call per submitted demand miss; admits from the second sighting
        inside the sliding window. Runs on the worker thread — the demand
        path only enqueues.

        Capacity-aware: when the layer and the global store both have
        room, the first sighting admits directly — the filter exists to
        stop scan pollution under pressure, not to gate free space.
        """
        with self._lock:
            lk = (int(key[0]), int(key[1]), str(key[2]))
            layer = self._layer_of(lk)
            cap = self._cap_for(layer)
            layer_full = bool(
                self.num_layers > 0
                and cap
                and self._layer_counts.get(layer, 0) >= cap
            )
            global_full = (
                self._resident_count_unlocked() >= self._global_cap_active()
            )
            if not layer_full and not global_full:
                return True
            c = self._admit_counts.get(lk, 0) + 1
            self._admit_counts[lk] = c
            self._admit_order.append(lk)
            if len(self._admit_order) > self._admission_window:
                old = self._admit_order.popleft()
                oc = self._admit_counts.get(old, 0) - 1
                if oc <= 0:
                    self._admit_counts.pop(old, None)
                else:
                    self._admit_counts[old] = oc
            return c >= 2

    def put(self, key: tuple[int, int, str], value: Any) -> None:
        with self._lock:
            self._put_unlocked(key, value)

    def put_many(self, items: list) -> None:
        """Batch admission under one lock (worker path, pre-filtered)."""
        with self._lock:
            for key, value in items:
                self._put_unlocked(key, value)

    def close(self) -> None:
        """Stop the detached admission worker. Idempotent."""
        self.closed = True
        adm = self.admission
        self.admission = None
        if adm is not None:
            try:
                adm.close()
            except Exception:
                pass

    def _put_unlocked(self, key: tuple[int, int, str], value: Any) -> None:
        if self.capacity <= 0:
            return
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        # per-layer cap enforcement (governor overrides + phase pair);
        # a cap hit that found no layer victim falls through to global.
        layer = self._layer_of(key)
        self._enforce_layer_cap(layer)
        # global cap (prefill pair while a prefill layer-call runs)
        while len(self._store) >= self._global_cap_active():
            old_k, _ = self._store.popitem(last=False)
            old_layer = self._layer_of(old_k)
            self._layer_index_drop(old_k, old_layer)
            self.stats.evictions += 1
            self._dec_layer_count(old_layer)
        self._store[key] = value
        self._layer_index_add(key, layer)
        self._layer_counts[layer] = self._layer_counts.get(layer, 0) + 1

    def clear(self) -> None:
        with self._lock:
            self._clear_unlocked()

    def _clear_unlocked(self) -> None:
        self._store.clear()
        self._layer_counts.clear()
        self._layer_orders.clear()
        self.stats.reset()

    @property
    def policy(self) -> str:
        return "lru"

    def retain_hot(self, hot_pairs: set) -> int:
        with self._lock:
            return self._retain_hot_unlocked(hot_pairs)

    def _retain_hot_unlocked(self, hot_pairs: set) -> int:
        """Keep only entries whose (layer_idx, expert_id) is in hot_pairs.

        The prefill demand path fills the cache with the *last* chunks'
        experts; the hotness seeder replaces those contents with the
        prompt-wide hot set. Rebuilds per-layer counts; returns the number
        of evicted entries.
        """
        if self.capacity <= 0 or not self._store:
            return 0
        evicted = 0
        for key in list(self._store.keys()):
            if (key[0], key[1]) not in hot_pairs:
                del self._store[key]
                evicted += 1
        # Rebuild the per-layer index unconditionally: entries can reach the
        # store without going through base put (governor resizes, subclass
        # paths), and a stale index silently degrades victim selection.
        self._layer_counts, (self._layer_orders,) = self._rebuild_layer_index(
            self._store
        )
        if evicted:
            self.stats.evictions += evicted
        return evicted

    # -- governor-driven resize -------------------------------------------

    def resize(self, capacity: int, per_layer_cap: int | None = None) -> None:
        """Atomically retarget the cache capacity (governor entry point).

        One locked retarget covers ``capacity``, ``_per_layer_cap``,
        ``_store`` and ``_layer_counts`` — the governor
        calls it from the asyncio event loop while the MLX thread may be
        inside get/put. Subclasses override ``_drain_to_unlocked`` /
        ``_evict_layer_unlocked`` to make "shrink to cap" mean the right
        thing for their own queues.
        """
        with self._lock:
            self._resize_unlocked(capacity, per_layer_cap)
        # Governor shrink/grow propagates to the per-layer arenas (the
        # arena bound follows the cache ceiling); outside the cache lock —
        # set_cap takes the arena lock, and no arena-locked path acquires
        # the cache lock, so there is no order to invert.
        self._sync_arena_caps()

    def _resize_unlocked(self, capacity: int, per_layer_cap: int | None = None) -> None:
        self.capacity = max(0, int(capacity))
        if per_layer_cap is not None:
            self._per_layer_cap = max(0, int(per_layer_cap))
        # Retargeting resets targeted overrides to uniform; the governor
        # re-applies them after resize when hunger persists. Prefill pair
        # re-derives unless explicitly pinned.
        self._per_layer_cap_over = {}
        self._derive_prefill_caps()
        self._drain_to_unlocked(self.capacity)
        if self.num_layers > 0 and self._per_layer_cap:
            for layer in list(self._layer_counts):
                while self._layer_counts.get(layer, 0) > self._per_layer_cap:
                    if not self._evict_layer_unlocked(layer):
                        break

    def _drain_to_unlocked(self, cap: int) -> None:
        """Drop entries until the store holds at most `cap` (LRU order)."""
        while len(self._store) > cap:
            old_k, _ = self._store.popitem(last=False)
            old_layer = self._layer_of(old_k)
            self._layer_index_drop(old_k, old_layer)
            self._layer_counts[old_layer] = max(0, self._layer_counts.get(old_layer, 1) - 1)
            self.stats.evictions += 1

    def _resident_count_unlocked(self) -> int:
        """Resident count under the already-held cache lock.

        The base policy's residents all live in ``_store``; policies with
        extra resident queues (S3FIFO's probationary ``_small``) override
        so occupancy-sensitive paths — admission_note's global fullness —
        see every resident, not just the main store's.
        """
        return len(self._store)

    @property
    def size(self) -> int:
        with self._lock:
            return self._resident_count_unlocked()

    def resident_bytes(self) -> int:
        """App-level bytes currently held resident by this cache.

        ``size`` slots (polymorphic: S3FIFO counts both queues) times the
        per-slot bytes the budget was sized with, PLUS every registered
        SlotArena's bound-bank bytes — arena residents live in fixed
        mx.array rows, never in ``_store``, so the slot count alone
        under-reports while the arena path is engaged. The scheduler's
        prefill tracker subtracts this heap's growth from the measured
        chunk delta so a budget-bounded LRU fill is not linearized as a
        per-token transient rate. Best-effort: never raises.
        """
        try:
            per_slot = int(getattr(self, "per_slot_bytes", 0) or 0)
            total = max(0, int(self.size)) * max(0, per_slot)
            with self._lock:
                arenas = list(self._arenas)
            for ref, _layer in arenas:
                arena = ref()
                if arena is not None:
                    total += int(arena.resident_bytes())
            return total
        except Exception:
            return 0


_CACHE_POLICY_ENV = (env_str("OMLX_EXPERT_STREAMING_CACHE", "lru") or "lru").lower()


def make_expert_cache(
    budget_bytes: int,
    per_slot: int,
    num_layers: int | None = None,
    policy: str | None = None,
) -> ExpertLRUCache:
    """Build the configured eviction policy (default LRU).

    ``policy`` (per-model setting) wins over the env default; None keeps
    OMLX_EXPERT_STREAMING_CACHE ("lru").
    """
    eff = (policy or _CACHE_POLICY_ENV or "lru").strip().lower()
    if eff == "s3fifo":
        from .cache_policies import S3FIFOExpertCache

        return S3FIFOExpertCache(budget_bytes, per_slot, num_layers=num_layers)
    return ExpertLRUCache(budget_bytes, per_slot, num_layers=num_layers)
