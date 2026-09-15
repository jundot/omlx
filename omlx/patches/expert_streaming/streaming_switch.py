# SPDX-License-Identifier: Apache-2.0
"""Streaming MoE switch layers with per-expert LRU cache.

Implements a drop-in replacement for SwitchLinear / QuantizedSwitchLinear and
SwitchGLU that keeps a bounded number of experts resident as mx.arrays and
faults the rest from the SSD-backed ExpertBackingStore (or an in-RAM dict for
tests).  The budget is a total byte budget across all MoE layers; the cache
is global per model.
"""

from __future__ import annotations

import inspect
import logging
import os
import queue as _queue
import threading
import weakref
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ._env import env_int
from .slot_cache import DecodeVisitStats, SlotArena
from .speculation import SpeculationState, _STAGED_MAX_IDS

logger = logging.getLogger(__name__)

_COALESCE_ENV = os.environ.get("OMLX_EXPERT_STREAMING_COALESCE", "") != "0"
# Suppress speculative staging when residency has no headroom
# (governor at floor or desperate-free band). 0 restores eager staging.
_STAGED_HEADROOM_ENV = (
    os.environ.get("OMLX_EXPERT_STREAMING_STAGED_HEADROOM", "1") != "0"
)
_BANK_MAX_BYTES = max(
    1,
    env_int("OMLX_EXPERT_STREAMING_BANK_MAX_BYTES", 256 * 1024**2),
)
# Chunk oversized demand banks instead of declining them. Prefill demand
# (~500/512 experts per projection, ~450 MB) exceeds _BANK_MAX_BYTES, so
# the bank is allocated once and filled by cap-bounded read_expert_into
# slices — the segments/promotion contract is unchanged.
# Bound each contiguous run inside a bank read. Uncapped, a dense
# (prefill) demand set is one run = one preadv on one worker — the device
# sits at single-stream speed (~1.5 GB/s) instead of the ~2.7 GB/s
# it reaches with several reads in flight. The cap is in bytes; each
# component converts it to an expert count. 0 disables (runs unbounded).
_BANK_RUN_MAX_BYTES = max(
    0,
    int(
        os.environ.get(
            "OMLX_EXPERT_STREAMING_BANK_RUN_MAX_BYTES", str(64 * 1024**2)
        )
    ),
)
_RUN_MAX = env_int("OMLX_EXPERT_STREAMING_RUN_MAX", 16, lo=1)
# An all-miss demand bank is promoted with a single mx.array instead of U
# per-expert mx arrays followed by mx.stack — bit-identical (gather_qmm
# receives the same bytes, dtype and shape) but it halves the Metal
# transient at the promotion point, where the U copies and the bank would
# briefly coexist. The layer-context path reads the bank as NumPy on an
# IO pool worker and promotes on the inference thread, so no MLX op is
# ever bound off-stream.
_LAYER_BARRIER_ENV = os.environ.get("OMLX_EXPERT_STREAMING_LAYER_BARRIER", "1") != "0"
# Hybrid decode fast path: routed calls at or below this many positions
# resolve through UNION mode (all projections in flight at once); larger
# calls keep rolling so prefill never holds all projections resident.
# 0 disables the hybrid (rolling everywhere).
# NOTE: this bounds UNION SELECTION only — it is a routed-row ceiling on
# union residency, not the decode/prefill phase test. Phase classification
# uses the call's real sequence length via _decode_call_shape (a 1-row
# prefill tail chunk is NOT decode).
_DECODE_UNION_MAX_ROWS = max(
    0, env_int("OMLX_EXPERT_STREAMING_DECODE_UNION_ROWS", 64)
)


def _decode_call_shape(positions: int, seq_len: int | None = None) -> bool:
    """True when this call shape is a decode (single-token) call.

    ``seq_len`` (indices.shape[-2] — tokens in the call) is authoritative
    when known: a decode token routes top_k rows (top_k > 1 means
    positions > 1 for one token), and a multi-token call is never decode —
    a 1-row prefill tail chunk or an n-token verify block must not accrue
    decode-layer stats or hold the decode caps. When ``seq_len`` is
    unknown (hand-rolled/test callers) the routed-row bound
    approximates the phase.
    """
    if seq_len is not None:
        return int(seq_len) <= 1
    return _DECODE_UNION_MAX_ROWS > 0 and int(positions) <= _DECODE_UNION_MAX_ROWS
# The union fast path declines (falls back to the per-expert
# resolution) when one layer call's bank set would exceed this many bytes.
# Decode-shaped calls never approach it; the cap only fences a misrouted
# prefill-shaped call out of union residency.
_CTX_UNION_MAX_BYTES = max(
    0,
    env_int("OMLX_EXPERT_STREAMING_CTX_UNION_MAX_BYTES", 1024**3),
)
# Batched cache lookup/writeback: get_many/put_many walk the index once
# under a single lock — ~60 lookups x 3 projections x 48 layers of RLock
# churn per decode token otherwise.
# Raw uint8 demand banks cannot be pooled: the LRU retains rows as views
# into them, so recycling a bank would corrupt cached experts (aliasing).
# The allocation cost (one np.empty per (key, tier) per layer call) is
# small next to the preadv payload.


def _layer_ctx_mode(positions: int) -> str:
    """Layer-context mode for one GLU call.

    'union' for small routed-row calls when the hybrid is enabled;
    'rolling' otherwise.

    ``positions`` is deliberate here: the union choice is a memory
    question (can every projection's bank be resident at once), which
    scales with routed rows — NOT the phase question, which is a
    sequence-length property handled by ``_decode_call_shape``.
    """
    if _DECODE_UNION_MAX_ROWS > 0 and int(positions) <= _DECODE_UNION_MAX_ROWS:
        return "union"
    return "rolling"
# How many *following* projections to read in the background while the
# current one is promoted/computed. 0 disables prefetch entirely.
#
# Default 3: this is the only knob that widens the I/O queue depth on the
# rolling path — read_expert_into issues its preadv calls strictly one at
# a time (see _RUN_IO_QD in shard_bank for the in-call depth), so AHEAD=1
# leaves the NVMe idle between reads. 3 keeps the following projections
# in flight at no measured memory cost.
_CTX_PREFETCH_AHEAD = max(
    0, env_int("OMLX_EXPERT_STREAMING_CTX_AHEAD", 3)
)
# Banks larger than this are never held speculatively; they are read on demand.
_CTX_PREFETCH_MAX_BYTES = max(
    0,
    int(
        os.environ.get(
            "OMLX_EXPERT_STREAMING_CTX_AHEAD_BYTES", str(512 * 1024**2)
        )
    ),
)
# Detached demand admission. Decode-miss rows are views into shared bank
# buffers — caching them retains the whole bank per entry (~9x the
# per-expert accounting on prefill-sized banks). The detached worker
# copies the row into a private np array off the demand path and
# batch-puts under one lock, so admission costs ~a queue push on the
# critical path.
_ADMIT_Q_MAX = env_int("OMLX_EXPERT_STREAMING_ADMIT_Q", 8192)
_ADMIT_BATCH = env_int("OMLX_EXPERT_STREAMING_ADMIT_BATCH", 96)

# Cross-layer speculation via F_RDADVISE (default on;
# OMLX_EXPERT_STREAMING_RA=0 disables). Each MoE layer advises the next
# layer's previous-token experts so the NVMe fetch overlaps compute.
# Hints only: nothing is copied into userspace and the LRU is untouched.
#
# The advisor targets the NEXT layer's banks (ids come from
# spec_state.prev_uniq_by_layer[next_layer]); the F_RDADVISE key must be
# the next layer's stacking key, resolved through the converted-linears
# registry (spec_state.linears_by_layer), never this linear's own key.
# The whole speculation state (history, registry, stats, pending futures)
# is PER CONVERSION — it hangs off the cache/backing and dies with them,
# so two engines never share state. The advisory is capped at
# _MAX_ADVISE_ROWS and deduped per layer call (_RemapPlan.advised_runs)
# so the projections of one layer issue each next-layer run at most once.
_RA_ENV = os.environ.get("OMLX_EXPERT_STREAMING_RA", "") != "0"
# Persistent slot-arena residency (opt-out). Misses are bound into fixed
# rows of a per-(layer, proj) bank at admission and gather_qmm indexes it
# by slot id, so a cache hit costs zero per-call assembly — the bundle
# path pays one mx.stack per projection per call. The ExpertLRUCache
# keeps owning admission/eviction policy via the per-layer caps it
# reports; the arena owns payload rows and the id->slot map.
# OMLX_EXPERT_STREAMING_ARENA=0 opts out; every unsupported condition
# (non-quantized proj, dict backing, hot/cold split, demand > arena
# bound) falls back to the bundle path.
_ARENA_ENV = os.environ.get("OMLX_EXPERT_STREAMING_ARENA", "1") != "0"
# Physical bound for one layer's arena across ALL its projections
# (weight+scales+biases rows). The cache's per-layer cap is a residency
# bound, not a byte budget — a governor-grown cap can otherwise commit
# tens of GiB of banks.
# A byte ceiling keeps the arena decode-shaped: demand sets larger than
# the row bound keep the bundle path (prefill is dense streaming — the
# bank-read path is already optimal there).
_ARENA_MAX_BYTES = max(
    1,
    env_int("OMLX_EXPERT_STREAMING_ARENA_MAX_MIB", 128),
) << 20
# Read jobs per projection inside arena produce. 1 = one coalesced bank
# read per projection (the union path's shape). >1 splits a projection's
# missing set into that many chunks — more pool parents, but each chunk
# plans a narrower run (breaks intra-projection coalescing). The run pool
# (_RUN_IO_QD=16 process-wide) stays the device queue-depth bound either
# way.
_ARENA_PRODUCE_JOBS = max(
    1,
    env_int("OMLX_EXPERT_STREAMING_ARENA_PROJ_JOBS", 3),
)
# Opt-in whole-layer readahead during prefill. A prefill-shaped call
# touches ~every expert of the next layer, so advising its full stacked
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

# Rows/positions above which a routed call is prefill-shaped rather than
# decode speculation: the advisory cap and the prefill-bypass cache guard
# share this boundary.
_PREFILL_SHAPE_MIN_ROWS = 64
# Hard cap on the advisory row set (rows above the boundary are
# prefill-shaped, not decode speculation).
_MAX_ADVISE_ROWS = _PREFILL_SHAPE_MIN_ROWS


# Layer-context fallbacks to per-expert resolution are counted per reason
# on the cache (per-engine/per-conversion — module-global counters would
# mix sessions in a persistent server). Reasons: read_failure (ctx read
# produced nothing usable), bank_too_large (union declined over
# _CTX_UNION_MAX_BYTES or the rolling loader's bank cap), tier_mismatch
# (bundles did not cover the demand set), dict_backing (projections
# without a bank reader, so no context was built at all).


# Parallel os.pread pool for the demand-set of one MoE layer call. Workers
# return raw numpy slices only — MLX promotion happens on the inference
# thread. QD8 sustains ~1.5 GB/s on the reference NVMe; QD16 plateaus near
# ~2.5 GB/s. OMLX_EXPERT_STREAMING_QD overrides.
#
# _EXPERT_IO_POOL is a process-wide SINGLETON with 16 workers shared
# across all concurrent parents: device depth is 16 total, not N*16, and
# per-call pools would oversubscribe the device. Prefetch shares this
# demand pool; the rolling path's extra depth comes from
# _CTX_PREFETCH_AHEAD — this pool stays 16.
_IO_QD = env_int("OMLX_EXPERT_STREAMING_QD", 16, lo=1)

_EXPERT_IO_POOL = ThreadPoolExecutor(
    max_workers=_IO_QD,
    thread_name_prefix="omlx-expert-io",
)

def _io_pool(linear: Any) -> ThreadPoolExecutor:
    """Read pool for a layer call: the process-wide QD16 singleton, or the
    per-linear override set by conversion for tests."""
    override = getattr(linear, "_io_pool_override", None)
    if override is not None:
        return override
    return _EXPERT_IO_POOL


def _accepts_kwarg(fn: Any, name: str) -> bool:
    """True when *fn* can be called with ``name=`` (one-time signature probe).

    On a bound method ``inspect.signature`` already hides ``self``.
    Uninspectable callables (C extensions, exotic mocks) assume the new
    signature — dispatch never retries, so a hook-internal TypeError
    surfaces once instead of triggering a second invocation with a
    different signature.
    """
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    if name in params:
        return True
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


# Per-depth executors for models whose per-model settings override the pool
# depth (autotune). One shared executor per distinct depth value — repeated
# conversions of models tuned to the same depth must not multiply idle
# worker threads. depth None → the env-default module pool above.
_IO_POOLS: Dict[int, ThreadPoolExecutor] = {}
_IO_POOLS_LOCK = threading.Lock()


def io_pool_for(depth: int | None) -> ThreadPoolExecutor:
    """Return the expert IO pool for a per-model depth override."""
    if depth is None:
        return _EXPERT_IO_POOL
    try:
        d = int(depth)
    except (TypeError, ValueError):
        return _EXPERT_IO_POOL
    if d < 1:
        return _EXPERT_IO_POOL
    d = min(64, d)
    with _IO_POOLS_LOCK:
        pool = _IO_POOLS.get(d)
        if pool is None:
            pool = ThreadPoolExecutor(
                max_workers=d, thread_name_prefix=f"omlx-expert-io-{d}"
            )
            _IO_POOLS[d] = pool
        return pool


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
        self.decode_layers = 0
        self.decode_layers_missed = 0
        self.prefill_layers = 0
        self.prefill_layers_missed = 0
        self.decode_misses_by_layer.clear()


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
        self.submitted = 0
        self.queue_drops = 0
        self.admitted = 0
        self.filtered = 0
        self._t = threading.Thread(
            target=self._run, name="omlx-expert-admit", daemon=True
        )
        self._t.start()

    def submit(self, key: tuple[int, int, str], row: Any) -> None:
        try:
            self._q.put_nowait((key, row))
            self.submitted += 1
        except _queue.Full:
            self.queue_drops += 1
            cache = self._cache_ref()
            if cache is not None:
                try:
                    cache.admission_drops += 1
                except Exception:
                    pass

    def _put_batch(self, batch: list) -> None:
        cache = self._cache_ref()
        if cache is None:
            return
        items = []
        for key, row in batch:
            try:
                if not cache.admission_note(key):
                    self.filtered += 1
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
        self.admitted += len(items)
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
        if self.num_layers > 0 and self.capacity > 0:
            per_layer = max(1, self.capacity // self.num_layers)
            # distribute remainder
            self._per_layer_cap = per_layer
        else:
            self._per_layer_cap = self.capacity
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
        self._admission_window = max(1024, min(self.capacity // 4, 16384)) if self.capacity > 0 else 1024
        self._admit_counts: Dict[Tuple[int, int, str], int] = {}
        self._admit_order: deque[Tuple[int, int, str]] = deque()  # type: ignore[type-arg]
        self.admission_drops = 0
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

    def _layer_index_add(self, key: tuple[int, int, str], layer: int) -> None:
        order = self._layer_orders.get(layer)
        if order is None:
            order = self._layer_orders[layer] = OrderedDict()
        order[key] = None

    def _layer_index_touch(self, key: tuple[int, int, str], layer: int) -> None:
        order = self._layer_orders.get(layer)
        if order is not None and key in order:
            order.move_to_end(key)

    def _layer_index_drop(self, key: tuple[int, int, str], layer: int) -> None:
        order = self._layer_orders.get(layer)
        if order is not None:
            order.pop(key, None)

    def _evict_layer_unlocked(self, layer: int) -> bool:
        """Evict the least-recently-used entry of `layer`; True if one went.

        O(1) via the per-layer index. Falls back to the linear scan when the
        index has no live entry for the layer (subclasses that keep their own
        stores never maintain the index, so it can go stale).
        """
        order = self._layer_orders.get(layer)
        if order:
            for victim in list(order):
                del order[victim]
                if victim in self._store:
                    self._store.pop(victim, None)
                    self.stats.evictions += 1
                    self._dec_layer_count(layer)
                    return True
        for k in list(self._store.keys()):
            if self._layer_of(k) == layer:
                self._store.pop(k)
                self.stats.evictions += 1
                self._dec_layer_count(layer)
                return True
        return False

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
        if key in self._store:
            self._store.move_to_end(key)
            self._layer_index_touch(key, self._layer_of(key))
            self.stats.hits += 1
            return self._store[key]
        self.stats.misses += 1
        if self._active_decode:
            try:
                _miss_layer = self._layer_of(key)
            except Exception:
                _miss_layer = -1
            _by_layer = self.stats.decode_misses_by_layer
            _by_layer[_miss_layer] = _by_layer.get(_miss_layer, 0) + 1
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
            global_full = len(self._store) >= self._global_cap_active()
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
        # per-layer cap enforcement (governor overrides + phase pair)
        layer = self._layer_of(key)
        _cap = self._cap_for(layer)
        if self.num_layers > 0 and _cap:
            cnt = self._layer_counts.get(layer, 0)
            # evict oldest entry of same layer if per-layer full (O(1) index)
            if cnt >= _cap:
                self._evict_layer_unlocked(layer)
                # if still over capacity due to rounding, fall through to global
        # global cap (prefill pair while a prefill layer-call runs)
        while len(self._store) >= self._global_cap_active():
            old_k, _ = self._store.popitem(last=False)
            self._layer_index_drop(old_k, self._layer_of(old_k))
            self.stats.evictions += 1
            self._dec_layer_count(self._layer_of(old_k))
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
        counts: Dict[int, int] = {}
        self._layer_orders = {}
        for key in self._store:
            layer = self._layer_of(key)
            counts[layer] = counts.get(layer, 0) + 1
            self._layer_index_add(key, layer)
        self._layer_counts = counts
        if evicted:
            self.stats.evictions += evicted
        return evicted

    # -- governor-driven resize -------------------------------------------

    def resize(self, capacity: int, per_layer_cap: int | None = None) -> None:
        """Atomically retarget the cache capacity (governor entry point).

        One locked retarget covers ``capacity``, ``_per_layer_cap``,
        ``_global_cap``, ``_store`` and ``_layer_counts`` — the governor
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
        self._global_cap = self.capacity
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

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def resident_bytes(self) -> int:
        """App-level bytes currently held resident by this cache.

        ``size`` slots (polymorphic: S3FIFO counts both queues) times the
        per-slot bytes the budget was sized with. The scheduler's prefill
        tracker subtracts this heap's growth from the measured chunk delta
        so a budget-bounded LRU fill is not linearized as a per-token
        transient rate. Best-effort: never raises.
        """
        try:
            per_slot = int(getattr(self, "per_slot_bytes", 0) or 0)
            return max(0, int(self.size)) * max(0, per_slot)
        except Exception:
            return 0


_CACHE_POLICY_ENV = os.environ.get("OMLX_EXPERT_STREAMING_CACHE", "lru").strip().lower()


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


def __getattr__(name: str):
    # Lazy re-exports: the policies subclass ExpertLRUCache (cache_policies
    # -> streaming_switch), so a top-level import here would cycle. Module
    # __getattr__ keeps `from .streaming_switch import S3FIFOExpertCache`
    # working for callers like the package __init__.
    if name in ("S3FIFOExpertCache",):
        from . import cache_policies

        return getattr(cache_policies, name)
    raise AttributeError(name)


# ---------------------------------------------------------------------------
# Helpers that mirror switch_layers.py
# ---------------------------------------------------------------------------

def promote_np_array(v: Any, dtype_str: str | None = None):
    """Single promotion rule for numpy -> MLX (QuantHandler registry).

    Centralizes the call sites that need the BF16-as-uint16 reinterpret:
    a new quantization adds one branch here. Handlers keyed by (stored
    numpy dtype, safetensors dtype string):
      (uint16, BF16) -> bit-exact reinterpret (matches mx.load; a
        shift->f32->astype path would flush subnormals via Metal FTZ and
        cost ~9x more on 4 MB slices).
      default -> mx.array copy on this thread.
    """
    if v is None:
        return None
    if isinstance(v, mx.array):
        return v
    try:
        if dtype_str == "BF16" and getattr(v, "dtype", None) == np.uint16:
            return mx.array(v).view(mx.bfloat16)
    except Exception:
        pass
    return mx.array(v)  # np.ndarray -> mx.array copy on this thread


def _inverse_permutation(order, inverse_scatter=False):
    if inverse_scatter:
        return mx.put_along_axis(
            mx.zeros_like(order), order, mx.arange(order.size, dtype=order.dtype), axis=0
        )
    return mx.argsort(order)


def _gather_sort(x, indices, inverse_scatter=False):
    *_, M = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = _inverse_permutation(order, inverse_scatter)
    lhs_indices = order // M
    x = x.flatten(0, -3)
    return x[lhs_indices], indices[order], inv_order


def _scatter_unsort(x, inv_order, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


# ---------------------------------------------------------------------------
# Shared per-layer routing plan (one host sync per MoE layer)
# ---------------------------------------------------------------------------

class _LayerLoadContext:
    """Shared quantized demand load for one MoE layer's projections.

    Scope: quantized only. The context is driven through hooks
    that exist solely on StreamingQuantizedSwitchLinear — bundle_key,
    _bank_bytes_for and _load_expert_bank_np(_full) — and it is only
    constructed when the owning GLU is quantized, so StreamingSwitchLinear
    (bf16) never participates. That is intentional: the bf16 path resolves
    one projection at a time inside its own __call__ and therefore has no
    cross-projection union for the context to collapse.

    Cache keys go through linear.bundle_key, so under the HOBBIT split
    hot and cold copies of one expert never alias, and the bank reads are
    tier-segmented (read_expert_into components are tier-homogeneous by
    contract).

    Two modes, selected per call by _layer_ctx_mode:

    rolling
        Each projection resolves its own bank on demand. At most
        _CTX_PREFETCH_AHEAD following projections are read on pool workers
        in the background, so the next bank is in flight while the current
        one is promoted and consumed on the GPU. Peak NumPy residency drops
        from the *union* of every projection (~3 banks) to ~1-2 banks.

    union
        One pool.map across every projection; all banks are resident until
        the last projection is consumed. Maximum I/O parallelism, highest
        RSS — used for decode-sized routed sets.

    Both modes preserve one shared routing plan per layer call, and reads
    run on IO-pool workers that never allocate MLX arrays.
    """

    def __init__(
        self,
        linears: list[Any],
        cache: ExpertLRUCache,
        mode: str | None = None,
        positions: int | None = None,
        seq_len: int | None = None,
    ):
        # The GLU picks the mode per call (union for decode-shaped,
        # rolling for prefill).
        self.mode = mode or "rolling"
        self.linears = linears
        self.cache = cache
        # Call shape from the constructor site (indices.size), NOT inferred
        # from mode: the env kill switch can force union onto prefill calls.
        # Drives the demand-scoped cache counters; None (callers without
        # shape info) simply does not count.
        self.positions = None if positions is None else int(positions)
        # Sequence length of the call (indices.shape[-2]): the multi-token
        # signal for the decode/prefill stats split and phase caps.
        # positions alone cannot tell a single-token decode apart from a
        # prefill row on top_k>1 models (1 token x top_k 8 = 8 routed
        # rows).
        self.seq_len = None if seq_len is None else int(seq_len)
        # Phase-aware caps: select the cache's active pair from the same
        # shape test as the decode/prefill stats split — real seq_len
        # when the caller supplied it (a multi-token verify or prefill
        # tail is never decode, even at few routed rows), the routed-row
        # bound only as the no-shape fallback (duck-typed: caches without
        # note_phase simply ignore it).
        try:
            _note = getattr(cache, "note_phase", None)
            if _note is not None and self.positions is not None:
                _note(_decode_call_shape(self.positions, self.seq_len))
        except Exception:
            logger.debug(
                "expert_streaming: ctx note_phase skipped",
                exc_info=True,
            )
        self.bundles: dict[int, dict[int, tuple]] = {}
        self.misses: dict[int, int] = {}
        self.failed = False
        # The raw contiguous NumPy banks behind bundles, kept so the
        # linear can promote a whole demand set with one mx.array per key
        # instead of U per-expert arrays plus a stack. Populated only when the
        # read covered the *entire* demand set (all-miss); bank_ids records
        # exactly which ids the bank holds, so a stale bank can never be
        # promoted against a demand set it does not describe.
        self.bank_raw: dict[int, Any] = {}
        self.bank_ids: dict[int, list[int]] = {}
        # rolling state
        self._order: dict[int, int] = {id(lin): i for i, lin in enumerate(linears)}
        self._futures: dict[int, Any] = {}
        self._resolved: set[int] = set()
        self._expert_ids: list[int] = []
        # union latch
        self._loaded = False
        # Completion-ordered union state: per-projection futures joined
        # lazily by ensure(), in compute order.

        # Per-layer stall accounting (flushed once by close()).
        self._closed = False
        # Union declined a demand set over _CTX_UNION_MAX_BYTES; the
        # linears fall back to the per-expert resolution.
        self.declined = False
        # Why the last resolve failed, when it did (read_failure vs
        # bank_too_large) so the fallback counter reports the true reason.
        self.fallback_reason: str | None = None

    # -- helpers ------------------------------------------------------------

    def _split(
        self,
        linear: Any,
        expert_ids: list[int],
        wait: bool = True,
        *,
        count: bool = True,
    ) -> tuple[dict, list[int]]:
        """Partition expert_ids into cached bundles and missing ids.

        After the LRU, consult the staged prefetch rows — a staged
        hit is a real hit (no new read; the demand touch is also the
        admission signal, so the row is put into the LRU here). ``wait``
        controls whether in-flight staged reads are joined (demand path)
        or skipped (the rolling prefetch's speculative split — it must not
        park the inference thread behind a just-submitted read).

        ``count=False`` is the non-counting probe the rolling prefetch
        uses: cache lookups go through ``peek``/``peek_many`` (no hit/miss
        counters, no recency move — the demand touch is the only admission
        signal) and staged entries are only probed via ``stage_pending``
        (never consumed, never joined). An in-flight staged read still
        counts as covered for READ purposes — its expert is excluded from
        ``missing`` so prefetch never double-reads it — but it does not
        appear in ``cached`` (the demand join owns the row).
        """
        cached: dict[int, tuple] = {}
        missing: list[int] = []
        covered: set[int] = set()  # in-flight staged: covered, not resolved
        spec = getattr(self.cache, "spec_state", None)
        if count:
            batch_get = getattr(self.cache, "get_many", None)
        else:
            batch_get = getattr(self.cache, "peek_many", None)
            if batch_get is None:
                batch_get = getattr(self.cache, "get_many", None)
        if batch_get is not None:
            keys = [linear.bundle_key(eid) for eid in expert_ids]
            staged_puts: list = []
            for eid, key, value in zip(
                expert_ids, keys, batch_get(keys)
            ):
                if value is None and spec is not None:
                    if count:
                        value = spec.stage_resolve(key, wait=wait)
                        if value is not None:
                            staged_puts.append((key, value))
                    elif spec.stage_pending(key):
                        covered.add(eid)
                if value is None:
                    if eid not in covered:
                        missing.append(eid)
                else:
                    cached[eid] = value
            # Staged hits promote into the LRU in one lock acquisition —
            # a per-hit put can trigger _drain_to eviction N times.
            if staged_puts:
                try:
                    self.cache.put_many(staged_puts)
                except Exception:
                    for key, value in staged_puts:
                        try:
                            self.cache.put(key, value)
                        except Exception:
                            logger.debug(
                                "expert_streaming: staged writeback put "
                                "failed",
                                exc_info=True,
                            )
            return cached, missing
        get = self.cache.get if count else getattr(
            self.cache, "peek", self.cache.get
        )
        for eid in expert_ids:
            key = linear.bundle_key(eid)
            value = get(key)
            if value is None and spec is not None:
                if count:
                    value = spec.stage_resolve(key, wait=wait)
                    if value is not None:
                        try:
                            self.cache.put(key, value)
                        except Exception:
                            logger.debug(
                                "expert_streaming: staged resolve writeback "
                                "put failed",
                                exc_info=True,
                            )
                elif spec.stage_pending(key):
                    covered.add(eid)
            if value is None:
                if eid not in covered:
                    missing.append(eid)
            else:
                cached[eid] = value
        return cached, missing

    def _count_demand(self, n_cached: int, n_missing: int) -> None:
        """Record one projection's resolve at the authoritative boundary.

        Called exactly once per projection per layer-call (the _ensure_union
        loop and _ensure_rolling) — never from _prefetch, whose non-counting
        split is recomputed at await time and would double-count. Uses the
        same ``_decode_call_shape`` test as note_phase so the decode bucket
        tracks real single-token calls. No shape info -> not counted.
        """
        if self.positions is None:
            return
        st = self.cache.stats
        if _decode_call_shape(self.positions, self.seq_len):
            st.decode_hits += n_cached
            st.decode_misses += n_missing
        else:
            st.prefill_hits += n_cached
            st.prefill_misses += n_missing

    def close(self) -> None:
        """Flush the per-layer stall counters. Idempotent; call once per layer.

        One _LayerLoadContext == one MoE layer-call, so this is the natural
        unit for the "stall is per layer, not per miss" accounting:
        within a layer the reads are issued together, so the layer waits
        for the slowest of them rather than for their sum.

        Deliberately called from a ``finally`` in the GLU: a layer that
        raised still resolved its projections, and a layer that never ran
        (declined union, bf16, no barrier) has no context to close.
        """
        if self._closed:
            return
        self._closed = True
        if self.positions is None:
            return
        any_miss = any(v > 0 for v in self.misses.values())
        st = self.cache.stats
        if _decode_call_shape(self.positions, self.seq_len):
            st.decode_layers += 1
            st.decode_layers_missed += 1 if any_miss else 0
        else:
            st.prefill_layers += 1
            st.prefill_layers_missed += 1 if any_miss else 0
        # Mid-request governor tick — the request-boundary observe()
        # leaves a whole decode at the initial budget. Throttled inside
        # tick() (~1 s); cheap compare otherwise.
        gov = getattr(self.cache, "governor", None)
        if gov is not None:
            try:
                gov.tick()
            except Exception:
                logger.debug(
                    "expert_streaming: governor tick skipped", exc_info=True
                )

    # -- rolling path -------------------------------------------------------

    def _prefetch(self, linear: Any) -> None:
        """Start background reads for the following projections, bounded.

        Bounded two ways: at most _CTX_PREFETCH_AHEAD submissions per call,
        and no single bank larger than _CTX_PREFETCH_MAX_BYTES is held
        speculatively (it is read on demand instead).
        """
        if _CTX_PREFETCH_AHEAD <= 0:
            return
        start = self._order.get(id(linear), -1)
        if start < 0:
            return
        submitted = 0
        for nxt in self.linears[start + 1 :]:
            if submitted >= _CTX_PREFETCH_AHEAD:
                break
            nid = id(nxt)
            if nid in self._resolved or nid in self._futures:
                continue
            # Speculative membership probe only: peek never counts hits or
            # moves recency, and in-flight staged reads are skipped as
            # covered (never consumed/joined — the demand split owns them).
            cached, missing = self._split(
                nxt, self._expert_ids, wait=False, count=False
            )
            self.bundles[nid] = cached
            self.misses[nid] = len(missing)
            if not missing and len(cached) == len(self._expert_ids):
                # Fully cached with real values in hand: nothing to read,
                # mark resolved so the linear short-circuits when it asks.
                # A staged-covered-but-unresolved expert is NOT here —
                # the demand split must still run to join it.
                self._resolved.add(nid)
                continue
            bank_bytes = int(nxt._tier_bank_bytes_for(missing))
            if bank_bytes > _CTX_PREFETCH_MAX_BYTES:
                continue
            # The worker reads the raw contiguous bank and hands back
            # (segments, rows) — NumPy only, so no MLX op is ever created
            # on a pool thread.
            self._futures[nid] = _io_pool(nxt).submit(
                nxt._load_expert_bank_np_full, missing
            )
            submitted += 1

    def _ensure_rolling(self, linear: Any, expert_ids: list[int]) -> None:
        lid = id(linear)
        if lid in self._resolved:
            return
        self._resolved.add(lid)
        if not self._expert_ids:
            self._expert_ids = list(expert_ids)
        ids = self._expert_ids

        # A prefetch may already be in flight; the split is recomputed
        # because the cache can change between submit and await. The FIRST
        # projection of the rolling path resolves synchronously here — its
        # read depth is bounded by read_expert_into's _RUN_IO_QD run pool;
        # only the FOLLOWING projections (prefetch) run on the shared pool.
        fut = self._futures.pop(lid, None)
        cached, missing = self._split(linear, ids)
        self.bundles[lid] = cached
        self.misses[lid] = len(missing)
        self._count_demand(len(cached), len(missing))

        if missing:
            if fut is not None:
                try:
                    got = fut.result()
                except Exception:
                    got = None
            else:
                got = linear._load_expert_bank_np_full(missing)
            rows = None if got is None else got[1]
            if rows is None or len(rows) != len(missing):
                self.failed = True
                self.fallback_reason = "read_failure"
                return
            self.bundles[lid].update(zip(missing, rows))
            # Single-promotion is only valid when the read covered the
            # *whole* demand set. A partial bank would have to be
            # concatenated with separately promoted cache hits, which
            # changes the layout contract, so it falls back per expert.
            if got[0] is not None and len(missing) == len(ids):
                self.bank_raw[lid] = got[0]
                self.bank_ids[lid] = list(missing)

        self._prefetch(linear)
    # -- union path ---------------------------------------------------------

    def _admit_rows(self, proj: Any, ids: list[int], rows: list) -> None:
        """Enqueue decode-miss rows for the detached admission worker.

        Prefill-shaped calls never admit — their demand is one-shot and
        the hotness seeder owns prefill residency.
        """
        # Decode-miss rows only: the demand admission is the decode working
        # set's maintenance path, so a verify chunk or a small prefill tail
        # (seq_len > 1, however few routed rows) must not admit into the LRU.
        if self.positions is not None and not _decode_call_shape(
            self.positions, self.seq_len
        ):
            return
        adm = getattr(self.cache, "admission", None)
        if adm is not None:
            for eid, row in zip(ids, rows):
                adm.submit(proj.bundle_key(eid), row)

    def _ensure_union(self, expert_ids: list[int]) -> None:
        if self._loaded:
            return
        self._loaded = True
        jobs: list[tuple[Any, list[int]]] = []
        for proj in self.linears:
            cached, missing = self._split(proj, expert_ids)
            self.bundles[id(proj)] = cached
            self.misses[id(proj)] = len(missing)
            self._count_demand(len(cached), len(missing))
            if missing:
                jobs.append((proj, missing))
        if not jobs:
            return
        pool = _io_pool(jobs[0][0])
        live = sum(proj._tier_bank_bytes_for(ids) for proj, ids in jobs)
        # A demand set too large for union residency declines so the
        # linears fall back per expert instead of holding the whole layer at
        # once (prefill-shaped calls never reach union, but a misrouted call
        # must not force union residency).
        if _CTX_UNION_MAX_BYTES > 0 and live > _CTX_UNION_MAX_BYTES:
            self.declined = True
            return
        results = list(pool.map(lambda job: job[0]._load_expert_bank_np(job[1]), jobs))
        for (proj, ids), rows in zip(jobs, results):
            if rows is None or len(rows) != len(ids):
                self.failed = True
                self.fallback_reason = "read_failure"
                return
            self.bundles[id(proj)].update(zip(ids, rows))
            self._admit_rows(proj, ids, rows)
    # -- public API ---------------------------------------------------------

    def ensure(self, linear: Any, expert_ids: list[int]) -> None:
        """Resolve linear's demand set for this layer call."""
        if self.mode == "union":
            self._ensure_union(expert_ids)
        else:
            self._ensure_rolling(linear, expert_ids)


@dataclass
class _RemapPlan:
    """Routing plan shared by every streaming linear of one MoE layer call.

    The first linear invoked in a layer builds the plan (mx.eval + host copy
    + np.unique + compact remap); the other projections (up/gate/down) reuse
    it — one sync per MoE layer instead of three.
    """

    indices_shape: Tuple[int, ...] = ()
    flat_np: Any = None
    uniq_list: list = field(default_factory=list)
    remapped: Any = None  # mx.array of compact ids, original indices shape
    positions: int = 0
    # Real token count (indices.shape[-2]) — the authoritative
    # decode-vs-prefill signal. Set by the GLU before the projections run;
    # _build_plan_into fills it from a 2-D indices array when unset. A
    # 1-token top_k=8 decode is 8 routed rows, and a 4-token verify can
    # have few rows — positions alone mislabels both.
    seq_len: Any = None
    uniq_np: Any = None  # host-side unique ids; promoted lazily for bias
    ctx: Any = None  # per-layer load context (quantized GLU)
    # (target_linear_id, run_first, run_count) already advised in this
    # layer call — the projections share one plan, so dedupe here.
    advised_runs: set = field(default_factory=set)
    # Arena mode (OMLX_EXPERT_STREAMING_ARENA): when the GLU binds the
    # demand set into persistent slot rows, arena_rhs is rhs_indices in
    # slot space and each linear reads the arena bank — no per-call
    # assembly.
    arena_rhs: Any = None


def _build_plan_into(plan: _RemapPlan, indices) -> None:
    """Populate a shared routing plan in place (called once per MoE layer)."""
    mx.eval(indices)
    # np.asarray, not np.array(copy=False): NumPy >=2.0 makes copy=False
    # mean "never copy" and an MLX array must copy device->host, so that
    # call raises. asarray(copy=None) is the direct host path.
    flat_np = np.asarray(indices).reshape(-1)
    uniq_np = np.unique(flat_np)
    uniq_list = uniq_np.tolist()
    # compact remap via searchsorted (uniq is sorted ascending):
    # vectorized C lookup
    remapped_np = np.searchsorted(uniq_np, flat_np).astype(np.int32)
    plan.indices_shape = tuple(indices.shape)
    # Only fill seq_len when the caller did not set it — the GLU stamps the
    # true token count before dispatch (a gather-sorted flat `idx` is 1-D
    # and would lose it here).
    if plan.seq_len is None:
        plan.seq_len = (
            int(indices.shape[-2])
            if getattr(indices, "ndim", 0) >= 2
            else None
        )
    plan.flat_np = flat_np
    plan.uniq_list = uniq_list
    plan.remapped = mx.array(remapped_np.reshape(indices.shape))
    plan.uniq_np = uniq_np
    plan.positions = int(flat_np.size)


# ---------------------------------------------------------------------------
# Streaming SwitchLinear variants
# ---------------------------------------------------------------------------

class StreamingSwitchLinear(nn.Module):
    """BF16 SwitchLinear with streaming cache."""

    def __init__(
        self,
        layer_idx: int,
        proj_name: str,
        stacked_key: str,
        num_experts: int,
        input_dims: int,
        output_dims: int,
        backing: Any,
        cache: ExpertLRUCache,
        bias: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.proj_name = proj_name
        self.stacked_key = stacked_key
        self.num_experts = num_experts
        self._input_dims = input_dims
        self._output_dims = output_dims
        self.backing = backing
        self.cache = cache
        # Bias per expert (small, keep resident)
        self._bias: mx.array | None = None
        self._has_bias = bias
        # Per-model IO overrides (expert_streaming_io_depth/coalesce settings).
        # Consumed by the quantized demand path; inert here. None → module
        # env defaults (_EXPERT_IO_POOL / _COALESCE_ENV).
        self._io_pool_override: Any = None
        self._coalesce_override: bool | None = None

    @property
    def input_dims(self) -> int:
        return self._input_dims

    @property
    def output_dims(self) -> int:
        return self._output_dims

    def _read_expert_weight(self, key, expert_id: int) -> mx.array:
        """Backing fetch + cache put for a demand miss — the caller already
        ran the ``get`` that produced the miss, so hit/miss telemetry
        derives from that same probe (no separate membership test that a
        concurrent eviction could falsify between probe and load)."""
        # Load slice from backing
        if hasattr(self.backing, "load_expert"):
            w = self.backing.load_expert(self.stacked_key, expert_id)
        else:
            # dict-backed for tests: backing is dict[(layer, proj)] -> mx.array[E,O,I]
            bank = self.backing[(self.layer_idx, self.proj_name)]  # type: ignore[index]
            w = bank[expert_id]
            # ensure mx.array
            if not isinstance(w, mx.array):
                w = mx.array(w)
        self.cache.put(key, w)
        return w

    def set_bias(self, bias: mx.array | None) -> None:
        self._bias = bias

    def __call__(self, x, indices, sorted_indices=False, plan: _RemapPlan | None = None):
        if plan is None:
            plan = _RemapPlan()
        if plan.flat_np is None:
            _build_plan_into(plan, indices)
        # Load each unique expert weight. Derive hit/miss from the get
        # result actually used for the load — a probe-then-load read can
        # go stale when a concurrent eviction lands between the two
        # (TOCTOU). get_many takes the cache lock once for the whole set.
        keys = [
            (self.layer_idx, int(eid), self.stacked_key)
            for eid in plan.uniq_list
        ]
        get_many = getattr(self.cache, "get_many", None)
        values = (
            get_many(keys)
            if get_many is not None
            else [self.cache.get(k) for k in keys]
        )
        mini_weights = []
        for eid, key, w in zip(plan.uniq_list, keys, values):
            if w is None:
                w = self._read_expert_weight(key, int(eid))
            mini_weights.append(w)
        # Stack into mini-bank (U, O, I)
        if len(mini_weights) == 1:
            mini_bank = mx.expand_dims(mini_weights[0], 0)
        else:
            mini_bank = mx.stack(mini_weights, axis=0)
        remapped = plan.remapped
        # Call gather_mm with mini-bank
        out = mx.gather_mm(x, mini_bank.swapaxes(-1, -2), rhs_indices=remapped, sorted_indices=sorted_indices)
        if self._bias is not None and self._has_bias:
            b_mini = mx.take(self._bias, mx.array(plan.uniq_np), axis=0)  # (U,O)
            out = out + mx.expand_dims(b_mini[remapped], -2)
        return out


class _ArenaBank:
    """Plain (non-Module) holder for a linear's persistent slot rows.

    Keeping the mx.array banks off the nn.Module attribute path stops them
    registering as parameters — the GLU's SlotArena owns them through the
    arrays/bind closures instead.
    """

    __slots__ = ("weight", "scales", "biases")

    def __init__(self) -> None:
        self.weight = None
        self.scales = None
        self.biases = None


class _ResolvedDemand(NamedTuple):
    """One projection's resolved expert rows for a layer call.

    Output of ``StreamingQuantizedSwitchLinear._resolve_demand``: the raw
    NumPy bundles keyed by expert id, the cache hit/miss split, the
    single-promoted bank when the whole demand set was one contiguous
    read.
    """

    bundles: Dict[int, tuple]
    banked: Any


class StreamingQuantizedSwitchLinear(nn.Module):
    """INT4/INT8 quantized SwitchLinear with streaming cache."""

    def __init__(
        self,
        layer_idx: int,
        proj_name: str,
        stacked_weight_key: str,
        stacked_scales_key: str,
        stacked_biases_key: str | None,
        num_experts: int,
        input_dims: int,
        output_dims: int,
        backing: Any,
        cache: ExpertLRUCache,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        has_bias: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.proj_name = proj_name
        self.stacked_weight_key = stacked_weight_key
        self.stacked_scales_key = stacked_scales_key
        self.stacked_biases_key = stacked_biases_key
        self.num_experts = num_experts
        self._input_dims = input_dims
        self._output_dims = output_dims
        self.backing = backing
        self.cache = cache
        self.group_size = group_size
        self.bits = bits
        self.mode = mode
        self._has_bias = has_bias
        self._bias: mx.array | None = None
        # Per-model IO overrides (expert_streaming_io_depth/coalesce settings).
        # None → module env defaults (_EXPERT_IO_POOL / _COALESCE_ENV).
        self._io_pool_override: Any = None
        self._coalesce_override: bool | None = None
        # HOBBIT hot/cold split: hot experts keep the ORIGINAL packing
        # (source bits/gs below); the rest compute at the cold tier
        # (self._cold_bits/_cold_gs from expert_cold/ metadata). Empty set
        # or None bits = uniform tier — the single-bank path.
        self._hot_experts: set | None = None
        self._cold_bits: int | None = None
        self._cold_gs: int | None = None
        self._split_active = False
        # The arena bank lives in a plain object so its mx.array rows stay
        # out of the module parameter tree.
        self._arena_bank = _ArenaBank()

    def set_hobbit_split(self, hot_experts, cold_bits: int, cold_gs: int) -> None:
        """Enable the dual-tier path for this linear (convert-time only)."""
        self._hot_experts = {int(e) for e in (hot_experts or [])}
        self._cold_bits = int(cold_bits)
        self._cold_gs = int(cold_gs)
        self._split_active = bool(
            self._hot_experts and self._cold_bits != self.bits
        )

    def _is_split_active(self) -> bool:
        # Conversion-time invariant — computed once at set_hobbit_split,
        # re-checked per expert on the bundle_key hot path.
        return self._split_active

    def _tier_of(self, expert_id: int) -> int:
        """0 = hot (source packing), 1 = cold (tier packing)."""
        return 0 if int(expert_id) in (self._hot_experts or ()) else 1

    @property
    def input_dims(self) -> int:
        return self._input_dims

    @property
    def output_dims(self) -> int:
        return self._output_dims

    def set_bias(self, bias: mx.array | None) -> None:
        self._bias = bias

    def _slice_dtypes_lazy(self):
        if not hasattr(self, "_slice_dtypes"):
            td = getattr(self.backing, "tensor_dtype", None)
            self._slice_dtypes = (
                td(self.stacked_scales_key) if td else None,
                td(self.stacked_biases_key) if td and self.stacked_biases_key else None,
            )
        return self._slice_dtypes

    def _promote_np(self, v, dtype_str: str | None = None):
        """Promote a cached/staged np.ndarray to mx.array on this thread."""
        return promote_np_array(v, dtype_str)
    def _slice_bytes(self, key: str) -> int:
        """Per-expert byte size of *key* (truthful: read from the backing reader).

        Tier-blind (cold-first); only used for sizing estimates — actual
        reads resolve per expert.
        """
        try:
            reader = self.backing._reader_for_key(key)
            return int(reader._rp_for(key).expert_bytes)
        except Exception:
            return 0

    def _per_slot_bytes(self) -> int:
        """Summed per-SLOT bytes across this projection's stacked tensors.

        One slot = one expert's slice of THIS projection's keys
        (weight+scales+biases) — the same unit the cache's per_slot_bytes
        budget counts. Renamed from ``_per_expert_bytes``: a whole expert
        spans every projection and is n_proj times this.
        """
        keys = [self.stacked_weight_key, self.stacked_scales_key]
        if self.stacked_biases_key:
            keys.append(self.stacked_biases_key)
        return sum(self._slice_bytes(k) for k in keys)

    def _tier_groups(self, ids: list[int]) -> dict[int, list[int]]:
        """Group ids by tier with ONE pass (split-aware segmentation).

        Shared by _tier_bank_bytes_for and the read path so reader
        resolution happens once per (key, tier-run) instead of once per
        expert. Without the split everything is tier 0.
        """
        groups: Dict[int, list[int]] = {}
        if not ids:
            return groups
        try:
            if not self._is_split_active():
                groups[0] = list(ids)
                return groups
            for eid in ids:
                groups.setdefault(self._tier_of(int(eid)), []).append(int(eid))
            return groups
        except Exception:
            return {0: list(ids)}

    def _tier_bank_bytes_for(self, ids: list[int]) -> int:
        """True raw-bank bytes for an id set under the HOBBIT split.

        Sums per tier group with the TIER's own reader width (hot = source
        packing, cold = tier packing). Without the split it reduces to
        _bank_bytes_for. Never raises: a resolution failure falls back to
        the cold-first estimate so the caps stay at least as strict for
        unknown layouts. One reader resolution per (key, tier) via
        _tier_groups.
        """
        if not ids:
            return 0
        try:
            if not self._is_split_active():
                return len(ids) * self._per_slot_bytes()
            keys = [self.stacked_weight_key, self.stacked_scales_key]
            if self.stacked_biases_key:
                keys.append(self.stacked_biases_key)
            total = 0
            for _t, g in self._tier_groups(ids).items():
                per_t = 0
                for key in keys:
                    reader = self.backing._reader_for_key(key, g[0])
                    per_t += int(reader._rp_for(key).expert_bytes)
                total += len(g) * per_t
            return total
        except Exception:
            return len(ids) * self._per_slot_bytes()


    def _read_expert_banks(self, expert_ids: list[int]):
        """Read a contiguous demand bank per (key, tier).

        Returns (segments, rows): segments is a list of (tier_ids, banks)
        with banks[i] a raw (n_tier, per_bytes) uint8 buffer per stacked
        key (weight, scales, bias); rows are the per-expert typed views in
        expert_ids order that the LRU caches. None when the backing cannot
        serve the demand set as banks (dict backing, unsupported layout,
        oversized demand set, or a tier-mixed component the backing rejects).
        """
        if not hasattr(self.backing, "read_expert_into") or not expert_ids:
            return None
        keys = [self.stacked_weight_key, self.stacked_scales_key]
        if self.stacked_biases_key:
            keys.append(self.stacked_biases_key)
        try:
            split = self._is_split_active()
            if split:
                groups: list[tuple[int, list[int]]] = []
                for t in (0, 1):
                    ids_t = [e for e in expert_ids if self._tier_of(e) == t]
                    if ids_t:
                        groups.append((t, ids_t))
            else:
                groups = [(0, list(expert_ids))]
            segments: list[tuple[list[int], list]] = []
            rows: list[tuple] = []
            total = 0
            for _t, ids_t in groups:
                per_bytes = []
                # Resolve each key's reader ONCE per tier-run (all ids
                # in ids_t share the tier by construction) and reuse it for
                # the slice views below.
                _readers = []
                for key in keys:
                    reader = self.backing._reader_for_key(key, ids_t[0])
                    _readers.append(reader)
                    per_bytes.append(reader._rp_for(key).expert_bytes)
                if any(size <= 0 for size in per_bytes):
                    return None
                total += len(ids_t) * sum(per_bytes)
                banks = [
                    np.empty((len(ids_t), size), dtype=np.uint8) for size in per_bytes
                ]
                if len(ids_t) * sum(per_bytes) > _BANK_MAX_BYTES:
                    # Read the bank in row windows bounded by the
                    # cap. Rows are views into `banks` either way, so peak
                    # residency is identical to the single-read shape; only
                    # the preadv side is chunked, which also bounds the temp
                    # run buffers inside read_expert_into.
                    rows_per_chunk = max(1, _BANK_MAX_BYTES // sum(per_bytes))
                    for lo in range(0, len(ids_t), rows_per_chunk):
                        hi = min(len(ids_t), lo + rows_per_chunk)
                        components = [(key, ids_t[lo:hi]) for key in keys]
                        views = [b[lo:hi] for b in banks]
                        if not self.backing.read_expert_into(
                            components,
                            views,
                            merge_gap=0,
                            max_run_bytes=_BANK_RUN_MAX_BYTES,
                        ):
                            return None
                else:
                    components = [(key, ids_t) for key in keys]
                    if not self.backing.read_expert_into(
                        components,
                        banks,
                        merge_gap=0,
                        max_run_bytes=_BANK_RUN_MAX_BYTES,
                    ):
                        return None
                segments.append((ids_t, banks))
                # Reinterpret rows with the already-resolved per-tier
                # readers (same packing for the whole run by contract) —
                # _slice_view would re-resolve per expert.
                _rps = [r._rp_for(k) for r, k in zip(_readers, keys)]
                for i in range(len(ids_t)):
                    w = np.frombuffer(banks[0][i], dtype=_rps[0].np_dtype).reshape(_rps[0].per_shape)
                    s = np.frombuffer(banks[1][i], dtype=_rps[1].np_dtype).reshape(_rps[1].per_shape)
                    b = (
                        np.frombuffer(banks[2][i], dtype=_rps[2].np_dtype).reshape(_rps[2].per_shape)
                        if self.stacked_biases_key
                        else None
                    )
                    rows.append((w, s, b))
            if split:
                # rows arrive tier-grouped; restore expert_ids order so the
                # caller can zip missing -> rows directly.
                flat = [e for _t, ids_t in groups for e in ids_t]
                by_id = {int(e): r for e, r in zip(flat, rows)}
                rows = [by_id[int(e)] for e in expert_ids]
            return segments, rows
        except Exception:
            # Bank-read failures fall back to per-expert loads — count
            # them (with the layer) instead of failing silently, so a
            # rotting backing shows up in the per-request summary.
            try:
                self.cache._count_ctx_fallback(f"bank_read_l{self.layer_idx}")
            except Exception:
                pass
            return None


    def _load_expert_bank_np(self, expert_ids: list[int]) -> list[tuple] | None:
        """Read a demand set into one raw NumPy bank per (key, tier).

        The backing performs coalesced contiguous reads into caller-owned
        banks; rows are then exposed as views for the existing LRU
        representation. Returning None preserves the per-expert fallback
        for dict backings and unsupported layouts.
        """
        got = self._read_expert_banks(expert_ids)
        return None if got is None else got[1]

    def _load_expert_bank_np_full(self, expert_ids: list[int]):
        """Like _load_expert_bank_np, but keeps the raw contiguous banks.

        Needed by the layer context: the NumPy read may happen on an
        IO pool worker, yet promoting those buffers to MLX must happen later
        on the inference thread (MLX ops may not be bound off-stream). Same
        failure contract as _load_expert_bank_np — None whenever the backing
        cannot serve the demand set as banks.
        """
        return self._read_expert_banks(expert_ids)

    def _promote_banks(self, segments: list) -> list | None:
        """Promote raw contiguous per-tier banks into one mx.array per key.

        Shared by _load_expert_bank_mx (read + promote together) and the
        layer context (read on a pool thread, promote here on the
        inference thread).

        Bit-identical to promoting U per-expert arrays and stacking them:
        each bank is reinterpreted with exactly the dtype and per-expert
        shape that _slice_view applies to a single row, so gather_qmm
        receives the same bytes, dtype and layout. Only the allocation count
        differs — one mx.array per key instead of U of them plus the stack.

        Returns a list aligned with segments: one (w_bank, s_bank, b_bank)
        triple per (tier_ids, banks) segment.
        """
        try:
            dt = self._slice_dtypes_lazy()
            promoted = []
            keys = [self.stacked_weight_key, self.stacked_scales_key]
            if self.stacked_biases_key:
                keys.append(self.stacked_biases_key)
            for ids_t, banks in segments:
                n = len(ids_t)
                one: list = []
                for i, key in enumerate(keys):
                    reader = self.backing._reader_for_key(key, ids_t[0])
                    rp = reader._rp_for(key)
                    typed = np.frombuffer(banks[i], dtype=rp.np_dtype).reshape(
                        n, *rp.per_shape
                    )
                    arr = promote_np_array(
                        typed,
                        dt[0] if i == 1 else (dt[1] if i == 2 else None),
                    )
                    # promote_np_array handles the mx.array fast path, but
                    # here the input is always numpy; fall back to a plain
                    # mx.array copy when the registry passed through
                    # (non-BF16 stored dtypes).
                    if not isinstance(arr, mx.array):
                        arr = mx.array(typed)
                    one.append(arr)
                while len(one) < 3:
                    one.append(None)
                promoted.append((one[0], one[1], one[2]))
            return promoted
        except Exception:
            # Same noisy-fallback contract as _read_expert_banks.
            try:
                self.cache._count_ctx_fallback("bank_promote_fail")
            except Exception:
                pass
            return None

    def _load_expert_bank_mx(self, expert_ids: list[int]):
        """Promote an all-miss demand bank in one shot (per tier).

        Returns (segments_promoted, rows) or None when the demand set cannot
        be served as banks. segments_promoted is a list of
        (tier_ids, (w_bank, s_bank, b_bank)); rows are the per-expert raw
        views the caller still has to seed into the LRU, so the hit-rate
        path is unaffected.

        Bit-identical to promoting U per-expert arrays and stacking them:
        each bank is reinterpreted with exactly the dtype and per-expert
        shape that _slice_view uses per row, so gather_qmm receives the same
        bytes in the same layout. Only the allocation count differs.
        """
        got = self._read_expert_banks(expert_ids)
        if got is None:
            return None
        segments, rows = got
        promoted = self._promote_banks(segments)
        if promoted is None:
            return None
        return [(ids_t, triple) for (ids_t, _banks), triple in zip(segments, promoted)], rows

    def _group_runs(
        self, sorted_ids: list[int], max_run: int | None = None
    ) -> list[tuple[int, int]]:
        """Split ascending expert ids into bounded contiguous runs.

        Under the HOBBIT split a run must NOT cross a tier boundary: the
        coalesced pread reads from ONE backing reader (resolved by the first
        id — source shard vs expert_cold/), so experts past the boundary
        would come back in the wrong packing. Runs therefore end at the
        first id whose tier differs from the run's first id. The bound comes
        from _RUN_MAX (env OMLX_EXPERT_STREAMING_RUN_MAX).

        Gap bridging is deliberately NOT wired: a run *could* be stretched to
        bridge a small hole of missing ids within the same tier (the caller
        drops rows outside its scatter set, so the extra bytes are harmless
        semantically), but reads lose more to the idle bytes than they gain
        from sequentiality. The capability stays in
        shard_bank.segment_runs(merge_gap=...); the runtime never opts in.
        """
        tier_of = self._tier_of if self._is_split_active() else None
        max_run = _RUN_MAX if max_run is None else max(1, int(max_run))
        merge_gap = 0
        from .shard_bank import segment_runs

        return segment_runs(
            sorted_ids,
            same=(lambda a, b: tier_of(a) == tier_of(b)) if tier_of is not None else None,
            merge_gap=merge_gap,
            max_run=max_run,
        )


    def bundle_key(self, expert_id: int):
        # Tier-suffixed under the HOBBIT split so a hot (source-packing)
        # bundle and a cold (tier-packing) bundle of the same expert can
        # coexist in the LRU without aliasing.
        tier = self._tier_of(expert_id) if self._is_split_active() else 0
        base = self.stacked_weight_key if tier == 0 else self.stacked_weight_key + "#c"
        return (self.layer_idx, expert_id, base)

    def _admit(self, key: tuple[int, int, str], row: Any) -> None:
        """Demand admission: enqueue for the detached worker copy+put."""
        adm = getattr(self.cache, "admission", None)
        if adm is not None:
            adm.submit(key, row)

    def _load_expert_np(self, expert_id: int) -> tuple | None:
        """Numpy-only load for the prefetch worker.

        Never touches the LRU and never allocates MLX arrays (worker threads
        must not bind MLX ops to a non-existent default stream). Returns None
        when the backing has no slice-level API or the read fails.
        """
        if not hasattr(self.backing, "load_expert_slice"):
            return None
        # Tier contract: the backing's hot set (same ids as this linear's
        # _hot_experts) routes hot ids to the source shards; everyone else
        # reads expert_cold/. The LRU key (bundle_key) keeps the two apart.
        try:
            w = self.backing.load_expert_slice(self.stacked_weight_key, expert_id)
            s = self.backing.load_expert_slice(self.stacked_scales_key, expert_id)
            b = None
            if self.stacked_biases_key:
                try:
                    b = self.backing.load_expert_slice(self.stacked_biases_key, expert_id)
                except Exception:
                    b = None
            return (w, s, b)
        except Exception as exc:
            # Repeated per-expert read failures warn once per key
            # (a rotting shard otherwise degrades silently to fallbacks).
            try:
                # Locked RMW on the cache (this runs on IO workers, so a
                # bare dict get/set would lose concurrent increments).
                key = self.bundle_key(int(expert_id))
                n = int(self.cache.note_read_failure(key))
                if n == 3:
                    logger.warning(
                        "expert_streaming: repeated read failure layer=%d expert=%d (%d fails): %s",
                        self.layer_idx, int(expert_id), n, str(exc)[:160],
                    )
            except Exception:
                pass
            return None

    def _load_expert_run_np(self, first_id: int, count: int) -> list[tuple] | None:
        """Numpy-only load of *count* consecutive experts in one pread per key.

        Returns None when the run read is unsupported/fails (caller falls
        back to per-expert loads). Runs exploit row-major contiguity: one
        sequential transfer instead of *count* scattered ones.
        """
        if not hasattr(self.backing, "load_expert_run"):
            return None
        try:
            ws = self.backing.load_expert_run(self.stacked_weight_key, first_id, count)
            ss = self.backing.load_expert_run(self.stacked_scales_key, first_id, count)
            bs: list | None = None
            if self.stacked_biases_key:
                try:
                    bs = self.backing.load_expert_run(self.stacked_biases_key, first_id, count)
                except Exception:
                    bs = None
            return [
                (w, s, bs[i] if bs is not None and i < len(bs) else None)
                for i, (w, s) in enumerate(zip(ws, ss))
            ]
        except Exception:
            return None

    def _bundle_cached_or_staged(self, expert_id: int):
        """Resolve a bundle without touching the disk (inference thread only).

        Returns the cached bundle (mx or raw np tuple) or None when the expert
        must be fetched from the backing store.
        """
        cached = self.cache.get(self.bundle_key(expert_id))
        # Only the bundle format (w, s, b) is ever written under the weight
        # key — every current writer (warmer seed, ctx writeback, bank
        # promote, the bundle loader itself) stores 3-tuples, and a raw
        # mx.array only appears under the non-quantized streamer's own key
        # namespace, which a quantized reader never resolves.
        if isinstance(cached, tuple) and len(cached) == 3:
            return cached  # type: ignore[return-value]
        # Staged prefetch rows (same 3-tuple contract, NumPy only).
        spec = self._spec_state()
        if spec is not None:
            try:
                row = spec.stage_resolve(self.bundle_key(expert_id))
            except Exception:
                row = None
            if isinstance(row, tuple) and len(row) == 3:
                try:
                    self.cache.put(self.bundle_key(expert_id), row)
                except Exception:
                    pass
                return row  # type: ignore[return-value]
        return None

    def _spec_state(self) -> SpeculationState | None:
        """Per-conversion speculation state.

        The converter hangs one instance on the cache and (when the
        backing is an object) on the backing; it dies with them, so two
        engines can never share ring bytes or routing history.
        """
        state = getattr(self.backing, "spec_state", None)
        if state is None:
            state = getattr(self.cache, "spec_state", None)
        return state

    def _advise_next_layer_prev_token(self, plan: _RemapPlan | None = None) -> None:
        """Speculate the NEXT layer's previous-token experts.

        spec_state.prev_uniq_by_layer holds layer N+1's expert ids; the
        advisory must therefore hit layer N+1's banks. The converted-
        linears registry resolves the next layer's real stacking keys (and
        its HOBBIT tier routing — backing.advise_expert_run segments runs
        per resolved reader, so hot/cold boundaries are respected
        automatically).

        Guards: the advisory is capped at _MAX_ADVISE_ROWS experts
        (prefill-shaped sets are skipped: they are dense demand, not decode
        speculation, and advising them would flood the device queue with
        speculative traffic), and each (target, run) fires at most once per
        layer call through plan.advised_runs.
        """
        if not _RA_ENV:
            return
        state = self._spec_state()
        if state is None or state.is_closed():
            return
        next_layer = self.layer_idx + 1
        targets = state.linears_by_layer.get(next_layer)
        if not targets:
            return
        advised_runs = plan.advised_runs if plan is not None else None
        # Prev-token routing is the advisory prediction (advisory ids,
        # never output).
        prev = state.prev_uniq_by_layer.get(next_layer)
        if not prev or len(prev) > _MAX_ADVISE_ROWS:
            return
        # When staging already covers this layer (recall gate open), the
        # staged READ warms the same pages for real — the hint is redundant.
        try:
            if state.stage_gate(next_layer):
                state.bump("advise_staged_skips", 1)
                return
        except Exception:
            logger.debug(
                "expert_streaming: stage_gate probe failed for layer %d",
                next_layer,
                exc_info=True,
            )
        try:
            sorted_prev = sorted(int(e) for e in prev)
            if not sorted_prev:
                return
            # TTL dedup — the same prev-set was just hinted last token;
            # F_RDADVISE is sticky, re-issuing it is pure overhead.
            _before = len(sorted_prev)
            sorted_prev = state.advise_fresh(next_layer, sorted_prev)
            state.bump("advise_ttl_drops", _before - len(sorted_prev))
            if not sorted_prev:
                return
            # k+1 overfetch — union the transition-table candidates
            # for the next layer's prev set into the advisory. Same caps
            # and dedup as the base set; hints only, never output.
            # Gated by measured precision of the last extras.
            try:
                _extra = (
                    state.predict_next(next_layer, sorted_prev)
                    if state.trans_overfetch_ok()
                    else []
                )
            except Exception:
                _extra = []
            if _extra:
                _have = set(sorted_prev)
                _added = []
                for _c in _extra:
                    if _c not in _have and len(sorted_prev) < _MAX_ADVISE_ROWS:
                        sorted_prev.append(int(_c))
                        _have.add(int(_c))
                        _added.append(int(_c))
                sorted_prev.sort()
                state.note_trans_extras(next_layer, _added)
                try:
                    state.bump("trans_overfetch", len(_added))
                except Exception:
                    pass
            # One shared segmentation with the demand path —
            # consecutive ids within one resolved reader become one run for
            # a single F_RDADVISE (tier boundaries break the run). Without
            # the hot/cold split every id resolves to the same reader, so
            # the runs are plain consecutive spans.
            from .shard_bank import segment_runs

            if self._is_split_active():
                rid_of = {
                    e: id(self.backing._reader_for_key(self.stacked_weight_key, e))
                    for e in sorted_prev
                }
                runs = segment_runs(
                    sorted_prev, same=lambda a, b: rid_of[a] == rid_of[b]
                )
            else:
                runs = segment_runs(sorted_prev)
            n_adv = n_bytes = n_segs = n_runs = n_fail = 0
            for target in targets:
                for first, count in runs:
                    if advised_runs is not None:
                        dedupe_key = (id(target), first, count)
                        if dedupe_key in advised_runs:
                            continue
                        advised_runs.add(dedupe_key)
                    try:
                        ok, adv_bytes, adv_segs = target.backing.advise_expert_run(
                            target.stacked_weight_key, first, count
                        )
                        if ok:
                            n_adv += count
                            n_bytes += adv_bytes
                            n_segs += adv_segs
                            n_runs += 1
                        else:
                            n_fail += 1
                    except Exception:
                        logger.debug(
                            "expert_streaming: advisory run failed for "
                            "layer %d",
                            next_layer,
                            exc_info=True,
                        )
            if n_adv:
                state.bump("advised", n_adv)
                state.bump("advised_experts", n_adv)
            if n_runs:
                state.bump("advised_runs", n_runs)
            if n_bytes:
                state.bump("advised_bytes", n_bytes)
            if n_segs:
                state.bump("advice_tier_segments", n_segs)
            if n_fail:
                state.bump("advice_failures", n_fail)
        except Exception:
            logger.debug(
                "expert_streaming: advisory skipped for layer %d",
                next_layer,
                exc_info=True,
            )

    def _stage_next_layer(self, plan: _RemapPlan | None = None) -> None:
        """Read the next layer's predicted demand into staging.

        Decode-shaped calls only: predicts layer L+1's experts as
        prev_uniq_by_layer[L+1] (last token's routing) + transition-table
        candidates, capped at _STAGED_MAX_IDS, and submits ONE coalesced
        bank read for this projection's counterpart linear. Rows land in
        spec_state.staged as NumPy — no MLX off-stream — and are consumed
        by _split/_bundle_cached_or_staged when layer L+1 resolves.

        Gated by stage_gate (per-layer EWMA recall of the prev-token
        prediction): layers whose routing diverges stop being prefetched
        after the evidence accumulates, so a bad predictor costs a bounded
        burst, not a permanent I/O tax.
        """
        state = self._spec_state()
        if state is None or not state.stage_enabled():
            return
        # Decode-shaped calls only, and the call shape is the real token
        # count: a multi-token verify/prefill with few routed rows must
        # not stage (positions is routed rows, seq_len is tokens).
        if plan is None or not _decode_call_shape(
            int(getattr(plan, "positions", 0)),
            getattr(plan, "seq_len", None),
        ):
            return
        # Headroom gate: speculative reads only pay when residency has
        # slack — floored capacity or free memory in the governor's
        # desperate band means mispredicts compete with demand fetches for
        # slots that cannot hold them.
        if _STAGED_HEADROOM_ENV and not self._stage_headroom():
            state.bump("staged_skips", 1)
            return
        next_layer = self.layer_idx + 1
        if not state.stage_gate(next_layer) or not state.stage_room():
            return
        try:
            targets = state.linears_by_layer.get(next_layer)
            if not targets:
                return
            proj = getattr(self, "proj_name", None)
            target = next(
                (l for l in targets if getattr(l, "proj_name", None) == proj),
                None,
            )
            if target is None:
                return
            # Prev-token routing is the advisory prediction.
            prev = state.prev_uniq_by_layer.get(next_layer)
            if not prev:
                return
            ids = list(dict.fromkeys(int(e) for e in prev))
            try:
                extra = state.predict_next(next_layer, ids)
            except Exception:
                extra = []
            for e in extra:
                if len(ids) >= _STAGED_MAX_IDS:
                    break
                if e not in ids:
                    ids.append(int(e))
            ids = ids[:_STAGED_MAX_IDS]
            # If the target layer can't hold even one token's working
            # set, every staged row is a guaranteed drop — the global
            # headroom gate misses this because governor pressure shrinks
            # per-layer caps before the global floor binds.
            try:
                _cap = self.cache._cap_for(next_layer)
            except Exception:
                _cap = 0
            if _cap and _cap < min(len(prev), _STAGED_MAX_IDS):
                state.bump("staged_skips", 1)
                return
            # Drop what the cache already serves; staging duplicates read.
            # Membership probe only — get() would mint hits/misses and
            # promote recency for rows no demand ever touched. peek_many
            # takes the cache lock once for the whole set.
            _peek = getattr(self.cache, "peek_many", None)
            if _peek is not None:
                _keys = [target.bundle_key(e) for e in ids]
                want = [e for e, v in zip(ids, _peek(_keys)) if v is None]
            else:
                want = [
                    e
                    for e in ids
                    if target.bundle_key(e) not in self.cache
                ]
            if not want:
                return
            want.sort()
            reader = target._load_expert_bank_np_full
            pool = _io_pool(target)
            # Re-staging replaces last token's staged set for this layer:
            # drop stale rows/futures keyed under next_layer BEFORE
            # registering.
            # stage_register skips ids already staged, so leaving stale
            # entries would both keep a wrong prediction alive and split
            # the new read around it. Keys are exact bundle keys
            # (layer, expert, kind); shared futures cancel only when no
            # sibling staged key still references them.
            for staged_keys in (state.staged, state.staged_futs):
                for stale in (
                    k for k in list(staged_keys) if k[0] == next_layer
                ):
                    state.stage_drop(stale)
            fut = pool.submit(reader, want)
            n = state.stage_register(fut, want, target)
            if n:
                state.bump("staged_submitted", n)
        except Exception:
            logger.debug(
                "expert_streaming: staging skipped for layer %d",
                getattr(self, "layer_idx", -1) + 1,
                exc_info=True,
            )

    def _stage_headroom(self) -> bool:
        """True when speculative staged reads have residency slack.

        Suppress staging when the shared cache capacity is at the
        governor floor or the last observed free memory was inside the
        desperate-clear band — under memory pressure staged rows get
        dropped before they are ever hit. Static caches (no governor)
        always have slack.
        """
        gov = getattr(self.cache, "governor", None)
        if gov is None:
            return True
        try:
            at_floor = getattr(gov, "at_floor", None)
            if callable(at_floor) and at_floor():
                return False
            desperate = getattr(gov, "in_desperate_band", None)
            if callable(desperate) and desperate():
                return False
        except Exception:
            return True
        return True

    def _load_expert_bundle(self, expert_id: int) -> tuple[mx.array, mx.array, mx.array | None]:
        key = self.bundle_key(expert_id)
        # Cache / staging resolution (shared with the parallel demand-set path)
        resolved = self._bundle_cached_or_staged(expert_id)
        if resolved is not None:
            return resolved  # type: ignore[return-value]
        # 3) synchronous load from backing
        if hasattr(self.backing, "load_expert_slice"):
            # Async-friendly: store plain np.ndarray slices in the cache and
            # promote them to mx.array on the inference thread at use time
            # (avoids cross-thread stream errors from MLX op allocation —
            # the prefetch worker must never allocate MLX arrays).
            w = self.backing.load_expert_slice(self.stacked_weight_key, expert_id)
            s = self.backing.load_expert_slice(self.stacked_scales_key, expert_id)
            b = None
            if self.stacked_biases_key:
                try:
                    b = self.backing.load_expert_slice(self.stacked_biases_key, expert_id)
                except Exception:
                    b = None
        elif hasattr(self.backing, "load_expert"):
            w = self.backing.load_expert(self.stacked_weight_key, expert_id)
            s = self.backing.load_expert(self.stacked_scales_key, expert_id)
            b = None
            if self.stacked_biases_key:
                try:
                    b = self.backing.load_expert(self.stacked_biases_key, expert_id)
                except Exception:
                    b = None
        else:
            # dict backing for tests
            w_bank = self.backing[(self.layer_idx, self.proj_name, "weight")]
            s_bank = self.backing[(self.layer_idx, self.proj_name, "scales")]
            b_bank = self.backing.get((self.layer_idx, self.proj_name, "biases"))
            w = w_bank[expert_id] if isinstance(w_bank[expert_id], mx.array) else mx.array(w_bank[expert_id])
            s = s_bank[expert_id] if isinstance(s_bank[expert_id], mx.array) else mx.array(s_bank[expert_id])
            b = None
            if b_bank is not None:
                bb = b_bank[expert_id]
                b = bb if isinstance(bb, mx.array) else mx.array(bb)
        bundle = (w, s, b)
        self.cache.put(key, bundle)  # type: ignore[arg-type]
        return bundle

    def __call__(self, x, indices, sorted_indices=False, plan: _RemapPlan | None = None):
        if plan is None:
            plan = _RemapPlan()
        # Speculation for layer+1, deduped per layer call
        # through plan.advised_runs (the GLU shares one plan).
        _spec_state = self._spec_state()
        if _RA_ENV and _spec_state is not None and _spec_state.prev_uniq_by_layer:
            try:
                self._advise_next_layer_prev_token(plan)
            except Exception:
                logger.debug(
                    "expert_streaming: advise failed for layer %d",
                    getattr(self, "layer_idx", -1),
                    exc_info=True,
                )
        # Staged decode prefetch — real reads into staging buffers
        # for the next layer's predicted demand (recall-gated; see
        # _stage_next_layer). Independent of the F_RDADVISE advisor.
        if _spec_state is not None:
            try:
                self._stage_next_layer(plan)
            except Exception:
                logger.debug(
                    "expert_streaming: stage_next failed for layer %d",
                    getattr(self, "layer_idx", -1),
                    exc_info=True,
                )
        built = plan.flat_np is None
        if built:
            _build_plan_into(plan, indices)
        # While the hotness seeder is active a prefill demand set is not
        # cached — seeding the LRU with prefill-only experts would evict the
        # decode working set.
        cache_result = not (
            getattr(self.cache, "prefill_bypass", False)
            and plan.positions > _PREFILL_SHAPE_MIN_ROWS
        )
        if getattr(plan, "arena_rhs", None) is not None:
            return self._arena_call(x, plan, sorted_indices)
        res = self._resolve_demand(plan, cache_result)
        out = self._assemble_banks(x, plan, res, sorted_indices)
        return self._finish(plan, out)

    def _arena_call(self, x, plan, sorted_indices):
        """Arena path: residents were bound into fixed slot rows at
        admission, so gather_qmm reads the persistent bank by slot index —
        a hit costs zero assembly (the bundle path re-stacks every call)."""
        bank = self._arena_bank
        out = mx.gather_qmm(
            x,
            bank.weight,
            bank.scales,
            bank.biases,
            rhs_indices=plan.arena_rhs,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        return self._finish(plan, out)

    def _resolve_demand(self, plan, cache_result) -> _ResolvedDemand:
        """Cover ``plan.uniq_list`` with raw ``(w, s, b)`` bundles.

        Order: the layer context (which prefetches projections in the
        background) -> the expert LRU -> one contiguous bank read for a
        full miss -> the coalesced/per-expert fallback.
        """
        bundles: Dict[int, tuple] = {}
        missing: list[int] = []
        context_bundles = None
        if plan.ctx is not None:
            # Resolve *this* projection through the layer context; the
            # context prefetches the next one in the background so banks
            # are not all resident at once.
            plan.ctx.ensure(self, plan.uniq_list)
            # Count every fallback to the per-expert resolution so runs
            # can prove the fast path engaged. The ctx records WHICH
            # reason when the read came back unusable.
            if plan.ctx.failed:
                context_bundles = None
                self.cache._count_ctx_fallback(
                    getattr(plan.ctx, "fallback_reason", None) or "read_failure"
                )
            elif getattr(plan.ctx, "declined", False):
                context_bundles = None
                self.cache._count_ctx_fallback("bank_too_large")
            else:
                context_bundles = plan.ctx.bundles.get(id(self))
                if context_bundles is not None and len(context_bundles) == len(plan.uniq_list):
                    bundles.update(context_bundles)
                else:
                    context_bundles = None
                    self.cache._count_ctx_fallback("tier_mismatch")
        if context_bundles is None:
            for eid in plan.uniq_list:
                eid = int(eid)
                b = self._bundle_cached_or_staged(eid)
                if b is not None:
                    bundles[eid] = b
                else:
                    missing.append(eid)
        banked = None
        if missing:
            # ascending expert id = ascending file offset within the stacked
            # bank (row-major) — sorted reads keep the NVMe's locality
            missing.sort()
            if (
                len(missing) == len(plan.uniq_list)
                and hasattr(self.backing, "read_expert_into")
            ):
                # Every demanded expert is a miss, so the demand set
                # is one contiguous bank per key (two segments under the
                # HOBBIT split) — promote each once instead of building U
                # per-expert mx arrays and stacking them.
                banked = self._load_expert_bank_mx(missing)
            if banked is not None:
                rows = banked[1]
                for eid, raw in zip(missing, rows):
                    bundles[eid] = raw
                    if cache_result:
                        self._admit(self.bundle_key(eid), raw)
            elif hasattr(self.backing, "load_expert_slice"):
                io_pool = _io_pool(self)
                # Bank-first path: read all missing experts
                # into one raw bank per (key, tier) on this thread, then
                # expose rows as views. Avoids one task/result allocation per
                # expert on dense demand sets and one reader resolution per
                # expert on every set.
                raws = self._load_expert_bank_np(missing)
                if raws is None:
                    coalesce_on = (
                        _COALESCE_ENV
                        if self._coalesce_override is None
                        else bool(self._coalesce_override)
                    )
                    raws = [None] * len(missing)
                    # Fallback: coalesce consecutive ids into
                    # single-pread runs (dense in long-prompt prefill; rare in
                    # decode, where runs are size 1 and the path degenerates
                    # to the per-expert fetch). map keeps a sliding window of
                    # 16 in flight (singleton pool), so the device queue stays
                    # full; batch drain/sawtooth is avoided without moving
                    # promotion off the inference thread.
                    runs = self._group_runs(missing)
                    if coalesce_on and len(runs) < len(missing):
                        results_by_run = list(
                            io_pool.map(
                                lambda r: (r, self._load_expert_run_np(r[0], r[1])),
                                runs,
                            )
                        )
                        idx_of = {eid: i for i, eid in enumerate(missing)}
                        leftover: list[int] = []
                        for (first, count), out in results_by_run:
                            if out is not None:
                                for j in range(count):
                                    eid = first + j
                                    if eid in idx_of:
                                        raws[idx_of[eid]] = out[j]
                                    # else: bridged gap row — read but
                                    # never promoted/used, so dropped here
                            else:
                                leftover.extend(
                                    e for e in range(first, first + count) if e in idx_of
                                )
                        if leftover:
                            for eid, raw in zip(
                                leftover, io_pool.map(self._load_expert_np, leftover)
                            ):
                                raws[idx_of[eid]] = raw
                    else:
                        raws = list(io_pool.map(self._load_expert_np, missing))
                for eid, raw in zip(missing, raws):
                    if raw is None:
                        bundles[eid] = self._load_expert_bundle(eid)
                        continue
                    # Raw np bundles in the LRU by design: Metal only holds
                    # the per-call stack. Caching promoted mx copies here
                    # double-holds the same weights in wired memory when the
                    # budget is positive, so wired residency would overshoot
                    # the budget outright.
                    bundles[eid] = raw
                    if cache_result:
                        # Tier-suffixed key (bundle_key): under the HOBBIT
                        # split a hot (source-packing) and cold
                        # (tier-packing) bundle of the same expert must
                        # never alias in the LRU — a mixed-width bundle
                        # served to the wrong tier crashes mx.stack.
                        self._admit(self.bundle_key(eid), raw)
            else:
                # dict-backed test doubles: sequential fallback
                for eid in missing:
                    bundles[eid] = self._load_expert_bundle(eid)
        return _ResolvedDemand(bundles=bundles, banked=banked)

    def _assemble_banks(self, x, plan, res: _ResolvedDemand, sorted_indices):
        """Promote the resolved rows into the bank(s) gather_qmm consumes."""
        dt = self._slice_dtypes_lazy()
        ctx_banks = None
        if res.banked is None and plan.ctx is not None:
            # The layer context read this projection's demand set
            # as one contiguous NumPy bank per key (possibly on an IO pool
            # worker). Promote it *here*, on the inference thread, so MLX
            # ops stay on-stream and the U per-expert mx arrays plus the
            # stack copy are both skipped. Guarded by bank_ids: only a bank
            # that describes exactly this demand set may be promoted, so a
            # stale or partial bank cannot silently mis-pair experts.
            segs = plan.ctx.bank_raw.get(id(self))
            if (
                segs is not None
                and plan.ctx.bank_ids.get(id(self)) == plan.uniq_list
            ):
                promoted = self._promote_banks(segs)
                if promoted is not None:
                    ctx_banks = [
                        (ids_t, triple)
                        for (ids_t, _banks), triple in zip(segs, promoted)
                    ]

        split = self._is_split_active()
        # Single-promoted per-tier banks. A segment whose ids
        # match EXACTLY the tier's demanded ids can feed gather_qmm directly
        # (no per-expert promote, no mx.stack) — bit-identical by
        # construction. Anything else falls back per tier.
        tier_single: dict[int, tuple] = {}
        segs_in = res.banked[0] if res.banked is not None else ctx_banks
        if segs_in:
            for ids_t, triple in segs_in:
                tier = self._tier_of(ids_t[0]) if split else 0
                tier_single[tier] = triple

        if split:
            return self._qmm_dual_tier(x, plan, res.bundles, tier_single, dt, sorted_indices)
        return self._qmm_uniform(x, plan, res.bundles, tier_single, dt, sorted_indices)

    def _qmm_uniform(self, x, plan, bundles, tier_single, dt, sorted_indices):
        """Uniform tier: one mini-bank and one gather_qmm."""
        if 0 in tier_single:
            w_bank, s_bank, b_bank = tier_single[0]
        else:
            mini_w, mini_s, mini_b = [], [], []
            has_b = False
            for eid in plan.uniq_list:
                w, s, b = bundles[int(eid)]
                w = self._promote_np(w)
                s = self._promote_np(s, dt[0])
                if b is not None:
                    has_b = True
                    b = self._promote_np(b, dt[1])
                mini_w.append(w)
                mini_s.append(s)
                if b is not None:
                    mini_b.append(b)
            # Bias consistency: a partial-bias bank (some bundles carry b,
            # some not) must fail loudly — silently stacking fewer biases
            # than weights would mis-pair rows or drop a declared bias.
            if mini_b and len(mini_b) != len(mini_w):
                raise RuntimeError(
                    "expert_streaming: inconsistent bias coverage in %s "
                    "(%d biases for %d weights)"
                    % (self.proj_name, len(mini_b), len(mini_w))
                )
            if len(mini_w) == 1:
                w_bank = mx.expand_dims(mini_w[0], 0)
                s_bank = mx.expand_dims(mini_s[0], 0)
                b_bank = mx.expand_dims(mini_b[0], 0) if has_b and mini_b else None
            else:
                w_bank = mx.stack(mini_w, axis=0)
                s_bank = mx.stack(mini_s, axis=0)
                b_bank = mx.stack(mini_b, axis=0) if has_b and mini_b else None
        return mx.gather_qmm(
            x,
            w_bank,
            s_bank,
            b_bank,
            rhs_indices=plan.remapped,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )

    def _qmm_dual_tier(self, x, plan, bundles, tier_single, dt, sorted_indices):
        """HOBBIT dual-tier assembly: one mini-bank per tier and a
        masked add — positions are mutually exclusive (each position
        consumes exactly one expert), so the two gather_qmm outputs
        partition the positions and zeros fill the rest."""
        # Per-tier bundle lists under the HOBBIT split: hot (source packing)
        # and cold (tier packing) widths differ (e.g. 8 vs 6 u32 cols per
        # row at gs 64), so a single stacked mini-bank is impossible — build
        # one per tier and combine the two gather_qmm outputs.
        tier_w = ([], [])  # hot, cold
        tier_s = ([], [])
        tier_b = ([], [])
        uniq = [int(e) for e in plan.uniq_list]
        hot_idx = [i for i, e in enumerate(uniq) if self._tier_of(e) == 0]
        cold_idx = [i for i, e in enumerate(uniq) if self._tier_of(e) == 1]
        for t, idxs in ((0, hot_idx), (1, cold_idx)):
            if t in tier_single:
                continue
            for i in idxs:
                w, s, b = bundles[uniq[i]]
                tier_w[t].append(self._promote_np(w))
                tier_s[t].append(self._promote_np(s, dt[0]))
                if b is not None:
                    tier_b[t].append(self._promote_np(b, dt[1]))

        def _stack_tier(t: int) -> tuple:
            """Per-expert stack for one tier (no matching segment)."""
            ws, ss, bs_ = tier_w[t], tier_s[t], tier_b[t]
            # Bias consistency: partial coverage within a tier must fail —
            # silently stacking fewer biases than weights mis-pairs rows.
            if bs_ and len(bs_) != len(ws):
                raise RuntimeError(
                    "expert_streaming: inconsistent bias coverage in %s "
                    "tier %d (%d biases for %d weights)"
                    % (self.proj_name, t, len(bs_), len(ws))
                )
            if len(ws) == 1:
                w_b = mx.expand_dims(ws[0], 0)
                s_b = mx.expand_dims(ss[0], 0)
                b_b = mx.expand_dims(bs_[0], 0) if bs_ else None
            else:
                w_b = mx.stack(ws, axis=0)
                s_b = mx.stack(ss, axis=0)
                b_b = mx.stack(bs_, axis=0) if bs_ else None
            return w_b, s_b, b_b

        flat_np = np.asarray(plan.flat_np).reshape(-1)
        out = None
        # Hot tier first (the masked add is elementwise-commutative so the
        # order is bit-exact).
        for t, idxs in ((0, hot_idx), (1, cold_idx)):
            if not idxs:
                continue
            if t in tier_single:
                w_b, s_b, b_b = tier_single[t]
            else:
                w_b, s_b, b_b = _stack_tier(t)
            bits_ = self.bits if t == 0 else self._cold_bits
            gs_ = self.group_size if t == 0 else self._cold_gs
            # expert-id -> rank within THIS tier's bank (flat ids here,
            # not compact uniq ranks); -1 where the other tier owns it.
            tier_map = np.full((self.num_experts,), -1, dtype=np.int32)
            for rank, i in enumerate(idxs):
                tier_map[uniq[i]] = rank
            tier_remapped_np = tier_map[flat_np].reshape(plan.indices_shape)
            # gather_qmm takes UNSIGNED row indices — -1 wraps to a huge
            # OOB index (garbage/nan) that the keep mask cannot undo
            # (nan * 0 = nan). Clamp the gather indices to 0 (any valid
            # rank: the row is zeroed by the keep mask below); the -1
            # survives only in keep_np, which is what selects the tier.
            gather_np = np.maximum(tier_remapped_np, 0)
            tier_remapped = mx.array(gather_np)
            tier_out = mx.gather_qmm(
                x,
                w_b,
                s_b,
                b_b,
                rhs_indices=tier_remapped,
                transpose=True,
                group_size=gs_,
                bits=bits_,
                mode=self.mode,
                sorted_indices=sorted_indices,
            )
            # Mask: keep only the positions this tier owns (-1 elsewhere).
            # gather_qmm inserts the indices' shape at dims 2.. so the
            # keep mask is the (index-shaped) validity, expanded over the
            # trailing (x_exp singleton, output) dims: [.., topk, 1, 1].
            keep_np = (tier_remapped_np >= 0).astype(np.float32)
            keep_shape = tuple(plan.indices_shape) + (1,) * (tier_out.ndim - len(plan.indices_shape))
            keep = mx.array(keep_np).reshape(keep_shape)
            tier_out = tier_out * keep
            out = tier_out if out is None else out + tier_out
        if out is None:
            # Degenerate: every unique expert hot (hot bank == full uniq
            # order) — identical to the uniform path.
            if 0 in tier_single:
                w_b, s_b, b_b = tier_single[0]
            else:
                w_b, s_b, b_b = _stack_tier(0)
            out = mx.gather_qmm(
                x, w_b, s_b, b_b, rhs_indices=plan.remapped,
                transpose=True, group_size=self.group_size, bits=self.bits,
                mode=self.mode, sorted_indices=sorted_indices,
            )
        return out

    def _finish(self, plan, out):
        """Shared tail: residual bias."""
        if self._bias is not None and self._has_bias:
            b_mini = mx.take(self._bias, mx.array(plan.uniq_np), axis=0)
            out = out + mx.expand_dims(b_mini[plan.remapped], -2)
        return out


class StreamingSwitchGLU(nn.Module):
    """Streaming SwitchGLU that delegates to streaming linears."""

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        layer_idx: int,
        backing: Any,
        cache: ExpertLRUCache,
        fused_gate_up: bool = False,
        inverse_scatter: bool = False,
        quantized: bool = False,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        activation: Any | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.inverse_scatter = inverse_scatter
        self.quantized = quantized
        # Original SwitchGLU activation (e.g. DeepSeek V4's LimitedSwiGLU with
        # swiglu_limit / fp32). None falls back to the stock mlx-lm swiglu.
        # Underscore attr keeps it out of the nn.Module parameter tree.
        self._activation = activation

        # Populated by the converter after construction; dims/quant
        # metadata live on the linears, not here.
        self._num_experts = num_experts
        self._backing = backing
        self._cache = cache

    @property
    def activation(self) -> Any:
        """Stock SwitchGLU surface for verify paths.

        The MTP target-verify path (mlx_vlm qwen3_5_moe
        ``_target_verify_switch_glu``) reaches into
        ``switch_mlp.activation`` directly. The captured callable lives
        in ``_activation`` to stay out of the parameter tree, so
        re-expose it here; fall back to the stock SwiGLU convention.
        """
        act = getattr(self, "_activation", None)
        if act is not None:
            return act
        from mlx_lm.models.switch_layers import SwiGLU

        return SwiGLU()

    def _apply_activation(self, x_up: Any, x_gate: Any) -> Any:
        act = getattr(self, "_activation", None)
        if act is not None:
            # Same call order as the original SwitchGLU: activation(up, gate)
            return act(x_up, x_gate)
        from mlx_lm.models.activations import swiglu

        return swiglu(x_gate, x_up)

    @staticmethod
    def _hook_conventions(hook):
        """Resolve this hook's call conventions once per hook instance.

        ``seq_len`` postdates some hook implementations; probing
        ``inspect.signature`` once and caching the verdict ON the hook
        (``_omlx_call_conv``) means per-call dispatch never retries — an
        ``except TypeError`` retry shim would re-invoke the hook with a
        different signature, so a TypeError raised INSIDE a new-signature
        hook body would execute its side effects twice.
        """
        conv = getattr(hook, "_omlx_call_conv", None)
        if conv is None:
            conv = (
                _accepts_kwarg(
                    getattr(hook, "on_layer_start", None), "seq_len"
                ),
                _accepts_kwarg(
                    getattr(hook, "on_layer_plan", None), "seq_len"
                ),
            )
            try:
                hook._omlx_call_conv = conv
            except Exception:
                pass  # unattr-able hook — re-probe per call; still no retry
        return conv

    def _notify_layer_start(self, hook, positions: int, seq_len) -> None:
        start_kw, _plan_kw = self._hook_conventions(hook)
        if start_kw:
            hook.on_layer_start(self.layer_idx, positions, seq_len=seq_len)
        else:
            hook.on_layer_start(self.layer_idx, positions)

    def _notify_layer_plan(self, hook, plan, seq_len) -> None:
        # Hotness signal: per-TOKEN usage over the routing plan
        # (bincount of the flat ids), computed only when a consumer
        # wants it — the readahead warmer keeps the uniq-list contract
        # and pays nothing. flat_np is already on the host (built by
        # _build_plan_into), so the bincount is a cheap vectorized pass.
        counts = (
            np.bincount(
                np.asarray(plan.flat_np).reshape(-1),
                minlength=self._num_experts,
            )
            if getattr(hook, "wants_usage_counts", False)
            else None
        )
        _start_kw, plan_kw = self._hook_conventions(hook)
        if plan_kw:
            hook.on_layer_plan(
                self.layer_idx,
                plan.uniq_list,
                plan.positions,
                counts,
                seq_len=seq_len,
            )
        else:
            hook.on_layer_plan(
                self.layer_idx, plan.uniq_list, plan.positions, counts
            )

    def _engage_arena_and_ctx(self, plan, idx, indices, has_fused, seq_len) -> None:
        # Arena mode: engage before the layer-ctx decision — when the
        # demand set fits the slot bank, gather_qmm reads slot rows and the
        # ctx/bundle machinery is skipped for this call. The _arena_failed
        # latch keeps a permanently failing engage from paying a per-token
        # exception: one warning, then the bundle path forever (the same
        # flag _arena_get sets when the arena cannot be built).
        if _ARENA_ENV and not getattr(self, "_arena_failed", False):
            try:
                self._arena_engage(plan, idx)
            except Exception:
                plan.arena_rhs = None
                self._arena_failed = True
                logger.warning(
                    "expert_streaming: arena engage failed on layer %d — "
                    "arena disabled for the rest of this engine's life",
                    getattr(self, "layer_idx", -1),
                    exc_info=True,
                )
        if self.quantized and _LAYER_BARRIER_ENV:
            projections = (
                [self.gate_up_proj, self.down_proj]
                if has_fused
                else [self.up_proj, self.gate_proj, self.down_proj]
            )
            if getattr(plan, "arena_rhs", None) is not None:
                pass  # arena owns residency this call — no layer ctx
            elif all(hasattr(proj, '_load_expert_bank_np') for proj in projections):
                # Hybrid: decode-shaped calls (<64 routed rows) read all
                # projections at once (union); prefill keeps rolling so
                # all banks are never resident simultaneously.
                ctx_mode = _layer_ctx_mode(int(indices.size))
                plan.ctx = _LayerLoadContext(
                    projections, self._cache, mode=ctx_mode,
                    positions=int(indices.size),
                    seq_len=seq_len,
                )
            else:
                # No bank reader on every projection (dict-backed test
                # doubles / bf16 mixes) — no context to resolve through.
                self._cache._count_ctx_fallback("dict_backing")

    def _run_projections(self, x_exp, idx, do_sort, has_fused, plan):
        if has_fused:
            x_gate_up = self.gate_up_proj(x_exp, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
            x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
            x_act = self._apply_activation(x_up, x_gate)
            return self.down_proj(x_act, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
        x_up = self.up_proj(x_exp, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
        x_gate = self.gate_proj(x_exp, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
        x_act = self._apply_activation(x_up, x_gate)
        return self.down_proj(x_act, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]

    def _record_prev(self, plan) -> None:
        # Remember this layer's routing for next token's speculation —
        # once per layer call. The projections share plan.uniq_list, so a
        # per-projection record would triple-count it.
        _spec_state = getattr(self._backing, "spec_state", None)
        if _spec_state is None:
            _spec_state = getattr(self._cache, "spec_state", None)
        if _spec_state is not None:
            try:
                _spec_state.record_prev(
                    self.layer_idx, plan.uniq_list, positions=plan.positions
                )
            except Exception:
                logger.debug(
                    "expert_streaming: record_prev skipped for layer %d",
                    getattr(self, "layer_idx", -1),
                    exc_info=True,
                )

    def _call_prologue(self, x, indices):
        """-> (has_fused, hook, seq_len).

        Call shape for hooks/phase: seq_len (indices.shape[-2]) is the
        token count — the authoritative decode-vs-prefill signal (a
        1-token top_k=8 decode is 8 routed rows, not "prefill").
        """
        # Mirror SwitchGLU.__call__ but route through streaming linears
        # Determine fused vs split by presence of gate_up_proj
        has_fused = hasattr(self, "gate_up_proj")
        # Opt-in warm/pin hook (warmer.py): fires previous-token reads for
        # the next layer before this layer's demand loads; decode-only.
        hook = getattr(self, "_warm_pins", None)
        _seq_len = (
            int(indices.shape[-2])
            if getattr(indices, "ndim", 0) >= 2
            else None
        )
        return has_fused, hook, _seq_len

    def _sort_call_inputs(self, x, indices):
        """Expand + gather-sort the call inputs -> (x_exp, idx, do_sort,
        inv_order)."""
        x_exp = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x_exp, idx, inv_order = _gather_sort(x_exp, indices, inverse_scatter=self.inverse_scatter)
        return x_exp, idx, do_sort, inv_order

    def _unsort_output(self, x_out, inv_order, do_sort, indices):
        """Scatter-unsort + squeeze — the reverse of _sort_call_inputs."""
        if do_sort:
            x_out = _scatter_unsort(x_out, inv_order, indices.shape)
        return x_out.squeeze(-2)

    def __call__(self, x, indices, scores=None, weighted_sum: bool = False):
        has_fused, hook, _seq_len = self._call_prologue(x, indices)
        if hook is not None:
            self._notify_layer_start(hook, int(indices.size), _seq_len)
        x_exp, idx, do_sort, inv_order = self._sort_call_inputs(x, indices)

        # One shared routing plan for the whole layer: the first linear
        # invoked builds it (single mx.eval + unique + remap), the rest reuse.
        # seq_len rides on the plan so every per-projection decode-shape
        # decision (staging gate, advisory) sees the token count rather
        # than routed rows.
        plan = _RemapPlan()
        plan.seq_len = _seq_len
        self._engage_arena_and_ctx(plan, idx, indices, has_fused, _seq_len)
        # The projection block is the whole lifetime of the layer context:
        # close() in a finally so a raising projection still flushes the
        # per-layer stall counters (a partial layer is still a stalled layer).
        _ctx = getattr(plan, "ctx", None)
        try:
            x_out = self._run_projections(x_exp, idx, do_sort, has_fused, plan)
        finally:
            if _ctx is not None:
                _ctx.close()

        self._record_prev(plan)
        if hook is not None:
            self._notify_layer_plan(hook, plan, _seq_len)

        # Weighted-sum fast path: the family's fused kernel, injected by the
        # converter from model_hooks (glm_moe_dsa.fast.glm_moe_weighted_sum,
        # which itself falls back to mx.fast). Plain scatter-unsort when no
        # kernel is registered — the caller's ndim contract then applies
        # the scores itself.
        if weighted_sum and scores is not None and do_sort:
            _ws_kernel = getattr(self, "_weighted_sum_kernel", None)
            if _ws_kernel is not None:
                try:
                    return _ws_kernel(x_out, inv_order, scores)
                except Exception:
                    # A real kernel failure must not be invisible — the
                    # unflatten path below is the fallback, not a fix.
                    logger.debug(
                        "expert_streaming: weighted-sum kernel failed for "
                        "layer %d — falling back",
                        getattr(self, "layer_idx", -1),
                        exc_info=True,
                    )

        out = self._unsort_output(x_out, inv_order, do_sort, indices)
        return out

    # ------------------------------------------------------------------
    # Arena mode (OMLX_EXPERT_STREAMING_ARENA)
    # ------------------------------------------------------------------

    def _arena_engage(self, plan: _RemapPlan, idx) -> bool:
        """Bind this call's demand set into persistent slot rows.

        On success plan.arena_rhs holds rhs_indices in slot space and every
        quantized linear of the layer reads its own arena bank — the bundle
        resolve + per-call mx.stack assembly is skipped entirely. Returns
        False (leaving plan.arena_rhs None) whenever the shape cannot be
        served: non-bank backing, hot/cold split active, or a demand set
        larger than the layer cap — those calls keep the bundle path.
        """
        projections = getattr(self, "_arena_projs", None)
        if projections is None:
            cand = (
                [self.gate_up_proj, self.down_proj]
                if hasattr(self, "gate_up_proj")
                else [self.up_proj, self.gate_proj, self.down_proj]
            )
            if not all(
                isinstance(p, StreamingQuantizedSwitchLinear)
                and not p._is_split_active()
                and hasattr(p.backing, "load_expert_slice")
                for p in cand
            ):
                self._arena_projs = False  # permanent shape: latch the decline
                return False
            self._arena_projs = projections = cand
        elif projections is False:
            return False
        if plan.flat_np is None:
            _build_plan_into(plan, idx)
        uniq = plan.uniq_list
        # Phase from the call's real sequence length: plan.seq_len is the
        # authoritative stamp (set pre-sort, from indices.shape[-2]); the
        # 1-D indices_shape of a sorted plan would drop it and a ≤64-row
        # multi-token call would misclassify as decode.
        seq_len = getattr(plan, "seq_len", None)
        if seq_len is None and len(plan.indices_shape) >= 2:
            seq_len = int(plan.indices_shape[-2])
        is_decode = _decode_call_shape(plan.positions, seq_len)
        # The ctx constructor issues this note in bundle mode; arena skips
        # the ctx, so issue it here — otherwise the active cap pair (and
        # the tighter prefill bound) lags the call's real phase.
        try:
            _note = getattr(self._cache, "note_phase", None)
            if _note is not None:
                _note(is_decode)
        except Exception:
            logger.debug(
                "expert_streaming: note_phase skipped for layer %d",
                getattr(self, "layer_idx", -1),
                exc_info=True,
            )
        # The cache keeps owning capacity: its per-layer cap bounds the
        # arena, and a demand set that cannot fit keeps the bundle path —
        # checked before the arena banks are ever allocated.
        try:
            cap = int(self._cache._cap_for(self.layer_idx))
        except Exception:
            cap = 0
        if not uniq or cap < 1 or len(uniq) > cap:
            return False
        arena = self._arena_get(projections, cap)
        if arena is None:
            return False
        # Keep book.cap synced to the active cap at engage time (a governor
        # action between engages is also pushed via cache._sync_arena_caps;
        # set_cap clamps to the build-time byte bound).
        try:
            arena.set_cap(cap)
        except Exception:
            logger.debug(
                "expert_streaming: arena set_cap skipped for layer %d",
                getattr(self, "layer_idx", -1),
                exc_info=True,
            )
        # The arena's own bound is physical (byte-ceilinged at build);
        # demand sets above it keep the bundle path.
        if len(uniq) > arena.book.cap:
            return False
        m0 = arena.book.misses
        with arena.lock:
            # frozen for non-decode calls: verify/prefill
            # experts commit at the eviction-candidate end and hits skip
            # LRU promotion, so a multi-token call churns its own cold-end
            # rows instead of sweeping the decode-hot set.
            arena.ensure_set(
                uniq,
                not is_decode,
                self._arena_produce(projections, int(plan.positions)),
            )
            plan.arena_rhs = arena.rows_for(uniq)[plan.remapped]
        arena_misses = arena.book.misses - m0
        # The layer-ctx close() feeds these counters in bundle mode; arena
        # skips the ctx, so report the same visit signal directly — the
        # governor's dynamic-budget loop stays informed either way. The
        # per-expert hit/miss pair mirrors _LayerLoadContext._count_demand:
        # covered = demanded − missed, so decode_hit_rate() and
        # streaming_gate_state() see arena-mode residency instead of a
        # permanent 0.0.
        st = getattr(self._cache, "stats", None)
        if st is not None:
            missed = arena_misses > 0
            n_demanded = len(uniq)
            n_missed = int(arena_misses)
            n_covered = max(0, n_demanded - n_missed)
            if is_decode:
                st.decode_layers += 1
                st.decode_layers_missed += 1 if missed else 0
                st.decode_hits += n_covered
                st.decode_misses += n_missed
                if missed:
                    d = st.decode_misses_by_layer
                    d[self.layer_idx] = d.get(self.layer_idx, 0) + 1
            else:
                st.prefill_layers += 1
                st.prefill_layers_missed += 1 if missed else 0
                st.prefill_hits += n_covered
                st.prefill_misses += n_missed
        gov = getattr(self._cache, "governor", None)
        if gov is not None:
            try:
                gov.tick()
            except Exception:
                logger.debug(
                    "expert_streaming: governor tick skipped", exc_info=True
                )
        return True

    def _arena_get(self, projections, cap):
        """Lazily build the per-layer SlotArena over the linears' banks.

        Row shape/dtype come from a promoted expert-0 slice so arena rows
        are byte-identical to what the bundle path would feed qmm.
        """
        arena = getattr(self, "_arena", None)
        if arena is not None:
            return arena
        if getattr(self, "_arena_failed", False):
            return None
        try:
            # Sample expert 0 per field: row shape/dtype and the bytes a
            # slot costs across ALL projections — the arena bound is a
            # byte ceiling, not the cache's residency cap.
            samples: dict = {}
            row_bytes = 0
            for lin in projections:
                dt = lin._slice_dtypes_lazy()
                fields: dict = {
                    "weight": lin._promote_np(
                        lin.backing.load_expert_slice(
                            lin.stacked_weight_key, 0
                        )
                    )
                }
                s0 = lin.backing.load_expert_slice(lin.stacked_scales_key, 0)
                if s0 is not None:
                    fields["scales"] = lin._promote_np(s0, dt[0])
                if lin.stacked_biases_key:
                    try:
                        b0 = lin.backing.load_expert_slice(
                            lin.stacked_biases_key, 0
                        )
                    except Exception:
                        b0 = None
                    if b0 is not None:
                        fields["biases"] = lin._promote_np(b0, dt[1])
                    elif (
                        getattr(lin.backing, "tensor_dtype", None) is not None
                        and lin.backing.tensor_dtype(lin.stacked_biases_key)
                        is not None
                    ):
                        # Bias consistency: the biases tensor exists but
                        # the sample read came back empty — running
                        # gather_qmm without it would silently serve
                        # un-biased numerics. Refuse the arena; the bundle
                        # path surfaces its own consistency error.
                        raise ValueError(
                            "arena sample: %s has biases tensor but read "
                            "returned none" % lin.proj_name
                        )
                samples[lin.proj_name] = fields
                row_bytes += sum(
                    int(getattr(a, "nbytes", 0) or 0)
                    for a in fields.values()
                )
            rooms0 = min(cap, self._num_experts or cap)
            if row_bytes > 0:
                # The byte ceiling is a hard bound: floor 1 row (a usable
                # arena); a larger floor would overshoot the env ceiling
                # whenever _ARENA_MAX_BYTES fits under that many rows.
                rooms0 = min(rooms0, max(1, _ARENA_MAX_BYTES // row_bytes))
            if rooms0 < 1:
                self._arena_failed = True
                return None
            for lin in projections:
                bank = lin._arena_bank
                for field, arr in samples[lin.proj_name].items():
                    setattr(
                        bank,
                        field,
                        mx.zeros((rooms0, *arr.shape), dtype=arr.dtype),
                    )
            lin_by_proj = {lin.proj_name: lin for lin in projections}

            def _arrays(proj):
                bank = lin_by_proj[proj]._arena_bank
                d = {"weight": bank.weight}
                if bank.scales is not None:
                    d["scales"] = bank.scales
                if bank.biases is not None:
                    d["biases"] = bank.biases
                return d

            def _bind(proj, values):
                bank = lin_by_proj[proj]._arena_bank
                bank.weight = values["weight"]
                bank.scales = values.get("scales")
                bank.biases = values.get("biases")

            def _eval_params():
                mx.eval(
                    *[
                        a
                        for lin in projections
                        for a in (
                            lin._arena_bank.weight,
                            lin._arena_bank.scales,
                            lin._arena_bank.biases,
                        )
                        if a is not None
                    ]
                )

            self._arena = SlotArena(
                rooms0,
                [lin.proj_name for lin in projections],
                _arrays,
                _bind,
                _eval_params,
            )
            # Governor resize/set_layer_caps propagate the cache's active
            # per-layer cap into the arena (weakly held; the GLU owns it).
            try:
                reg = getattr(self._cache, "_register_arena", None)
                if reg is not None:
                    reg(self._arena, self.layer_idx)
            except Exception:
                logger.debug(
                    "expert_streaming: arena registration skipped for "
                    "layer %d",
                    getattr(self, "layer_idx", -1),
                    exc_info=True,
                )
            return self._arena
        except Exception:
            self._arena_failed = True
            logger.debug(
                "expert_streaming: arena engage permanently disabled for "
                "layer %d",
                getattr(self, "layer_idx", -1),
                exc_info=True,
            )
            return None

    def _arena_produce(self, projections, positions: int = 0):
        """Payload producer for arena.ensure_set: staged-bundle join first
        (same contract as _bundle_cached_or_staged), then one coalesced
        bank read per linear for the still-missing ids.

        ``positions`` is the CALL's routed-row count — the pool regime
        input. The arena miss count must not drive it: a prefill that
        evicts few residents still wants the prefill pool."""
        linears = list(projections)

        def _fields(lin, bundle, dt):
            w, s, b = bundle
            out = {"weight": lin._promote_np(w)}
            if s is not None:
                out["scales"] = lin._promote_np(s, dt[0])
            if b is not None:
                out["biases"] = lin._promote_np(b, dt[1])
            return out

        def produce(fetch_list):
            ids = [int(e) for e, _s, _v in fetch_list]
            spec = linears[0]._spec_state()
            # Phase 1 — staged-bundle join per linear (cheap dict ops on
            # the inference thread; same contract as _bundle_cached_or_staged).
            per_lin = []  # (lin, dtypes, rows, missing[(row_idx, eid)])
            for lin in linears:
                dt = lin._slice_dtypes_lazy()
                rows = [None] * len(ids)
                missing = []
                for i, eid in enumerate(ids):
                    row = None
                    if spec is not None:
                        try:
                            r = spec.stage_resolve(lin.bundle_key(eid))
                            if isinstance(r, tuple) and len(r) == 3:
                                row = r
                        except Exception:
                            row = None
                    if row is not None:
                        rows[i] = _fields(lin, row, dt)
                    else:
                        missing.append((i, eid))
                per_lin.append((lin, dt, rows, missing))
            # Phase 2 — bank reads for the still-missing ids submitted to
            # the io pool TOGETHER (cross-projection parallelism — serial
            # per-projection reads cost ~3x the miss latency).
            # _ARENA_PRODUCE_JOBS>1 additionally splits each projection's
            # set into chunks.
            jobs = []  # (per_lin_idx, missing_chunk)
            for li, (_lin, _dt, _rows, missing) in enumerate(per_lin):
                if not missing:
                    continue
                if _ARENA_PRODUCE_JOBS > 1 and len(missing) > 1:
                    step = -(-len(missing) // _ARENA_PRODUCE_JOBS)
                    jobs.extend(
                        (li, missing[i : i + step])
                        for i in range(0, len(missing), step)
                    )
                else:
                    jobs.append((li, missing))
            if len(jobs) > 1:
                pool = _io_pool(linears[0])
                futs = [
                    pool.submit(
                        per_lin[li][0]._load_expert_bank_np,
                        [e for _, e in chunk],
                    )
                    for li, chunk in jobs
                ]
                results = [f.result() for f in futs]
            else:
                results = [
                    per_lin[jobs[0][0]][0]._load_expert_bank_np(
                        [e for _, e in jobs[0][1]]
                    )
                ] if jobs else []
            for (li, chunk), banks in zip(jobs, results):
                lin, dt, rows, _m = per_lin[li]
                if banks is None or len(banks) != len(chunk):
                    raise RuntimeError("arena produce: bank read failed")
                for j, b in enumerate(banks):
                    rows[chunk[j][0]] = _fields(lin, b, dt)
            return [
                {lin.proj_name: rows[i] for lin, _dt, rows, _m in per_lin}
                for i in range(len(ids))
            ]

        return produce
