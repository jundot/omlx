# SPDX-License-Identifier: Apache-2.0
"""Demand-side bank IO: the preadv pool and the per-layer load context.

The process-wide ``_EXPERT_IO_POOL`` (QD16 by default,
``OMLX_EXPERT_STREAMING_QD``) is the device-depth bound every layer call
shares; ``io_pool_for``/``_io_pool`` resolve per-model depth overrides.
``_decode_call_shape``/``_layer_ctx_mode`` are the call-shape heuristics
the context modes key on, ``promote_np_array`` the single numpy -> MLX
promotion rule, and ``_LayerLoadContext`` the shared demand load for one
MoE layer call — rolling or union — that keeps reads on IO workers and
promotion on the inference thread.
"""

from __future__ import annotations

import inspect
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import mlx.core as mx
import numpy as np

from ._env import env_int
from .expert_cache import ExpertLRUCache
from .shard_bank import np_to_mx

logger = logging.getLogger(__name__)


# Hybrid decode fast path: routed calls at or below this many positions
# resolve through UNION mode (all projections in flight at once); larger
# calls keep rolling so prefill never holds all projections resident.
# 0 disables the hybrid (rolling everywhere).
# NOTE: this bounds UNION SELECTION only — it is a routed-row ceiling on
# union residency, not the decode/prefill phase test. Phase classification
# uses the call's real sequence length via _decode_call_shape (a 1-row
# prefill tail chunk is NOT decode).
_DECODE_UNION_MAX_ROWS = env_int(
    "OMLX_EXPERT_STREAMING_DECODE_UNION_ROWS", 64, lo=0
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
_CTX_UNION_MAX_BYTES = env_int(
    "OMLX_EXPERT_STREAMING_CTX_UNION_MAX_BYTES", 1024**3, lo=0
)


def _layer_ctx_mode(positions: int) -> str:
    """Layer-context mode for one GLU call.

    'union' for small routed-row calls when the hybrid is enabled;
    'rolling' otherwise.

    ``positions`` is deliberate here: the union choice is a memory
    question (can every projection's bank be resident at once), which
    scales with routed rows — NOT the phase question, which is a
    sequence-length property handled by ``_decode_call_shape``.
    """
    return "union" if _decode_call_shape(positions) else "rolling"


# How many *following* projections to read in the background while the
# current one is promoted/computed. 0 disables prefetch entirely.
#
# Default 3: this is the only knob that widens the I/O queue depth on the
# rolling path — read_expert_into issues its preadv calls strictly one at
# a time (see _RUN_IO_QD in shard_bank for the in-call depth), so AHEAD=1
# leaves the NVMe idle between reads. 3 keeps the following projections
# in flight at no measured memory cost.
_CTX_PREFETCH_AHEAD = env_int("OMLX_EXPERT_STREAMING_CTX_AHEAD", 3, lo=0)
# Banks larger than this are never held speculatively; they are read on
# demand. Measured constant (was OMLX_EXPERT_STREAMING_CTX_AHEAD_BYTES).
_CTX_PREFETCH_MAX_BYTES = 512 * 1024**2


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


# ---------------------------------------------------------------------------
# Helpers that mirror switch_layers.py
# ---------------------------------------------------------------------------

def promote_np_array(v: Any, dtype_str: str | None = None):
    """Single promotion rule for numpy -> MLX (QuantHandler registry).

    Passthrough (None / already-mx.array) lives here; the ndarray ->
    mx.array conversion delegates to ``shard_bank.np_to_mx`` — the shared
    BF16-as-uint16 reinterpret + F8 decode (a shift->f32->astype path
    would flush subnormals via Metal FTZ and cost ~9x more on 4 MB
    slices).
    """
    if v is None:
        return None
    if isinstance(v, mx.array):
        return v
    if dtype_str == "BF16" and getattr(v, "dtype", None) != np.uint16:
        # A BF16 label on non-uint16 data is not raw bits — .view would
        # reinterpret at the wrong width, so degrade to a plain promote.
        dtype_str = None
    return np_to_mx(v, dtype_str)


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
        # get_many/peek_many exist on every cache (LRU and S3FIFO) — one
        # lock acquisition for the whole demand set.
        batch_get = self.cache.get_many if count else self.cache.peek_many
        keys = [linear.bundle_key(eid) for eid in expert_ids]
        staged_puts: list = []
        for eid, key, value in zip(expert_ids, keys, batch_get(keys)):
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
        for eid, row in zip(ids, rows):
            proj._admit(proj.bundle_key(eid), row)

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
