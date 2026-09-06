# SPDX-License-Identifier: Apache-2.0
"""Streaming MoE switch layers with per-expert LRU cache.

Implements a drop-in replacement for SwitchLinear / QuantizedSwitchLinear and
SwitchGLU that keeps a bounded number of experts resident as mx.arrays and
faults the rest from the SSD-backed ExpertBackingStore (or an in-RAM dict for
tests).  The budget is a total byte budget across all MoE layers; the cache
is global per model.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)
# Opt-in per-layer / per-projection Metal memory trace (Fase J prefill-memory
# work). Null-tracer by default: call sites cost one attribute lookup.
from .memtrace import memtrace  # noqa: E402


_PROFILE_ENV = os.environ.get("OMLX_EXPERT_STREAMING_PROFILE", "") == "1"
_COALESCE_ENV = os.environ.get("OMLX_EXPERT_STREAMING_COALESCE", "") != "0"
# Fase K F6 (port of faseJ f799067e / 0a4d3c7 / 4d7609b / 80bf9b9):
# bank sizing + single-promotion + rolling layer-context knobs.
_BANK_MAX_BYTES = max(
    1,
    int(os.environ.get("OMLX_EXPERT_STREAMING_BANK_MAX_BYTES", str(256 * 1024**2))),
)
_RUN_MAX = max(1, int(os.environ.get("OMLX_EXPERT_STREAMING_RUN_MAX", "16")))
# Fase K F7: bridge gaps of up to this many experts inside one same-tier
# coalesced run (gap bytes are read but never promoted/used). DEFAULT 0:
# three windows on this box measured a NET LOSS from bridging in BOTH
# regimes — single-tier 2k 34.0s bridged vs 31.4s unbridged (3 reps,
# mergeab/), and split-active 2k 55.0s vs 47.5s / 8k 107.2s vs 98.9s
# (split4/). The NVMe at QD16 already saturates without the holes; the
# bridge only adds idle bytes and longer per-layer waits. The env knob
# stays for slower/HDD backends where sequential reads may win.
_RUN_MERGE_GAP = max(0, int(os.environ.get("OMLX_EXPERT_STREAMING_RUN_MERGE_GAP", "0")))
# Etapa A1: promote an all-miss demand bank with a single mx.array instead of
# U per-expert mx arrays followed by mx.stack. Bit-identical — gather_qmm
# receives the same bytes, dtype and shape — but it halves the Metal transient
# at the promotion point, where the U copies and the bank briefly coexist.
# 0 restores the per-expert promote + stack path.
_BANK_PROMOTE_ENV = os.environ.get("OMLX_EXPERT_STREAMING_BANK_PROMOTE", "1") != "0"
# Etapa A1b: same single-promotion trick, but on the *layer-context* path,
# which is the one that actually runs when the Etapa B barrier is on (the
# default). The context reads the demand bank as NumPy on an IO pool worker
# and hands the raw buffers back; promoting them to MLX must happen here, on
# the inference thread, so no MLX op is ever bound off-stream. 0 restores the
# per-expert promote + stack path for A/B.
_BANK_PROMOTE_CTX_ENV = (
    os.environ.get("OMLX_EXPERT_STREAMING_BANK_PROMOTE_CTX", "1") != "0"
)
_LAYER_BARRIER_ENV = os.environ.get("OMLX_EXPERT_STREAMING_LAYER_BARRIER", "1") != "0"
# Etapa B: rolling per-projection bank load instead of the union load.
# 0 restores the legacy behaviour (every projection's NumPy bank resident at
# once) for A/B against the new pipelined path.
_CTX_ROLLING_ENV = os.environ.get("OMLX_EXPERT_STREAMING_CTX_ROLLING", "1") != "0"
# Fase 1 (hybrid decode fast path): routed calls at or below this many
# positions resolve through the UNION mode (all projections in flight at
# once — the measured-best decode shape on this box); larger calls keep
# rolling so prefill never holds all projections resident. 0 disables
# the hybrid (rolling everywhere), matching the pre-Fase-1 behavior.
_DECODE_UNION_MAX_ROWS = max(
    0, int(os.environ.get("OMLX_EXPERT_STREAMING_DECODE_UNION_ROWS", "64"))
)
# Fase L4B B3: dual-tier execution order. Default "" runs hot first
# (historical); "small-first" submits the tier with the FEWER positions
# first. The masked add is elementwise and commutative in IEEE fp, so the
# order is bit-exact; measured as a residency experiment (no gain on this
# box — kept as a diagnostic knob).
_DUAL_TIER_ORDER = os.environ.get(
    "OMLX_EXPERT_STREAMING_DUAL_TIER_ORDER", ""
)
# Fase L1: the union fast path declines (falls back to the legacy per-expert
# resolution) when one layer call's bank set would exceed this many bytes.
# Decode-shaped calls never approach it; the cap only fences a misrouted
# prefill-shaped call out of union residency.
_CTX_UNION_MAX_BYTES = max(
    0,
    int(os.environ.get("OMLX_EXPERT_STREAMING_CTX_UNION_MAX_BYTES", str(1024**3))),
)
# P2 note (bank pooling deliberately NOT done): raw uint8 demand banks
# cannot be pooled because the LRU retains rows as views into them —
# recycling a bank would corrupt cached experts (aliasing). The real
# allocation cost (one np.empty per (key, tier) per layer call) is small
# next to the preadv payload; the per-expert mx.array copies were already
# eliminated by single-promotion (_BANK_PROMOTE_ENV/_BANK_PROMOTE_CTX_ENV).
# (reserved knob name kept so a future safe pool does not collide.)


def _layer_ctx_mode(positions: int, *, quantized: bool, barrier: bool) -> str | None:
    """Layer-context mode for one GLU call (Fase 1).

    None -> no context (not quantized / barrier off). 'union' for
    decode-shaped calls when the hybrid is enabled; 'rolling' otherwise.
    The env kills or forces union via the global switch in the caller.
    """
    if not (quantized and barrier):
        return None
    if _DECODE_UNION_MAX_ROWS > 0 and int(positions) <= _DECODE_UNION_MAX_ROWS:
        return "union"
    return "rolling"
# How many *following* projections to read in the background while the
# current one is promoted/computed. 0 disables prefetch entirely.
#
# Default 3 (faseJ 072e19e2): this is the only knob that widens the I/O queue
# depth on the rolling path and depth is what the decode regression turned
# on. read_expert_into issues its preadv calls strictly one at a time (see
# _RUN_IO_QD in shard_bank for the in-call depth), so with AHEAD=1 the whole
# layer call had a queue depth of 1: the NVMe sat idle between reads and
# decode throughput tracked that idleness (CPU 41%, 0.46 GiB/s, 1.86 tok/s).
# Raising it to 3 keeps the following projections in flight: CPU 50%,
# 0.56 GiB/s, 2.22 tok/s (+19% decode, -9% TTFT) at no measured memory cost.
_CTX_PREFETCH_AHEAD = max(
    0, int(os.environ.get("OMLX_EXPERT_STREAMING_CTX_AHEAD", "3"))
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

# Prefill attribution diag: sync the GPU at every prefill-sized MoE GLU call
# and record the drain as a per-layer gpu bucket. Serializes CPU/GPU overlap
# (wall inflates), so use it for attribution only — never for latency claims.
_PREFILL_DIAG_ENV = os.environ.get("OMLX_EXPERT_STREAMING_PREFILL_DIAG", "") == "1"
# Routes above this count are treated as prefill-sized for the diag sync.
_PREFILL_DIAG_MIN_ROUTES = 512

# B5 admission filter (scan-resistant). When OMLX_EXPERT_STREAMING_ADMISSION=1,
# only experts seen >=2 times in the recent window enter the LRU. Disabled by
# default; operational default for this model/box is budget=0 (see docs).
# P3: the window scales with capacity (min 1024, up to 1/4 of slots) so the
# filter stays meaningful at real budgets — a fixed 1024-entry window is
# noise next to a 6 GiB working set. OMLX_EXPERT_STREAMING_ADMISSION_WINDOW
# overrides.
_ADMISSION_ENV = os.environ.get("OMLX_EXPERT_STREAMING_ADMISSION", "") == "1"
_ADMISSION_WINDOW_ENV = int(os.environ.get("OMLX_EXPERT_STREAMING_ADMISSION_WINDOW", "0") or 0)
_ADMISSION_WINDOW = 1024

# O2 cross-layer speculation (G2 F_RDADVISE). RA is default-on (like G2)
# and can be disabled with OMLX_EXPERT_STREAMING_RA=0. When enabled, each
# MoE layer advises the next layer's previous-token experts via F_RDADVISE
# so the NVMe fetch overlaps compute. Hints only: nothing is copied into
# userspace and the LRU is untouched.
#
# Fase K F1: the advisor targets the NEXT layer's banks (ids come from
# spec_state.prev_uniq_by_layer[next_layer]); the F_RDADVISE key must be
# the next layer's stacking key, resolved through the converted-linears
# registry (spec_state.linears_by_layer), never __self__'s key — the old
# port advised the CURRENT layer's byte range for the NEXT layer's ids
# (warmed the wrong bytes, and under the HOBBIT split applied the wrong
# hot-set routing). K1: the whole speculation state (history,
# registry, stats, pending futures) is PER CONVERSION — it hangs off the
# cache/backing and dies with them, so two engines never share state.
# Fase K F2: the advisory is guarded like the warmer G2 (_MAX_ADVISE_ROWS)
# and deduped per layer call (_RemapPlan.advised_runs) so the 3 projections
# of one layer issue each next-layer run at most once.
_RA_ENV = os.environ.get("OMLX_EXPERT_STREAMING_RA", "") != "0"

# Fase K F2: hard cap on the advisory row set, matching the warmer G2's
# _MAX_WARM_ROWS (rows > 64 are prefill-shaped, not decode speculation).
_MAX_ADVISE_ROWS = 64
# Fase K correction K1/K7: the O2 advisor's speculation state is PER
# CONVERSION (one instance per backing/cache pair), never module-global.
# Two engines (different checkpoints, same tensor keys) can never share
# routing history or linears registries: the state dies with
# the owning store, and close() drains the speculation workers the same
# way ExpertBackingStore.close drains its readers.
# FU1: transition-table overfetch. The (layer, expert) -> next-token expert
# distribution (EWMA, temporal: same layer, token t-1 -> t) feeds the RA
# advisor with one extra candidate per demanded expert (k+1 overfetch).
# Hints only (F_RDADVISE), never changes output. 0 disables.
_TRANSITION_ENV = os.environ.get("OMLX_EXPERT_STREAMING_TRANSITION", "1") != "0"
_TRANSITION_TOP = 8  # entries kept per (layer, expert) source
_TRANSITION_OVERFETCH = 1  # extra candidates per demanded expert


class SpeculationState:
    """Per-conversion O2 speculation state (Fase K K1/K7).

    Owns the routing history used by the next-layer advisor, the
    converted-linears registry, the advise stats, and the transition
    table. Hangs off ``cache.spec_state``
    (and off ``backing.spec_state`` when the backing is an object) so the
    demand path, the advisor, and backing.close() all share one instance.
    All mutations are lock-guarded. A closed state stops advising so a
    drained engine never serves another conversion's speculation.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.closed = False
        self.prev_uniq_by_layer: Dict[int, list[int]] = {}
        self.linears_by_layer: Dict[int, list[Any]] = {}
        self.stats = {
            "advised": 0,
            "advised_runs": 0,
            "advised_experts": 0,
            "advised_bytes": 0,
            "advice_failures": 0,
            "advice_tier_segments": 0,
        }
        # FU1: transition table (layer, expert) -> {next_expert: weight}.
        # Temporal only (same layer, token t-1 -> t); cross-layer same-token
        # transitions are not observable without a model-loop hook (PILOT
        # covers glm5_next). Bounded: TOP entries per source, pruned on write.
        self.trans: Dict[Tuple[int, int], Dict[int, float]] = {}
        self.trans_updates = 0

    # -- registry / history ----------------------------------------------

    def register_linears(self, layer_idx: int, linears: list[Any]) -> None:
        """Record one MoE layer's converted quantized linears (convert-time)."""
        kept = [l for l in linears if l is not None]
        if not kept:
            return
        with self.lock:
            self.linears_by_layer[int(layer_idx)] = kept

    def record_prev(self, layer_idx: int, ids: list[int]) -> None:
        """Remember this layer's routing for the next token's speculation."""
        now = [int(e) for e in ids]
        with self.lock:
            prev = self.prev_uniq_by_layer.get(int(layer_idx))
            if _TRANSITION_ENV and prev and now:
                # Credit temporal transitions prev -> now (EWMA without a
                # global decay pass: w = 1 + 0.9*w, normalized on read).
                for ep in prev:
                    row = self.trans.setdefault((int(layer_idx), int(ep)), {})
                    for en in now:
                        row[int(en)] = 1.0 + 0.9 * float(row.get(int(en), 0.0))
                    if len(row) > _TRANSITION_TOP:
                        for k in sorted(row, key=row.get)[: len(row) - _TRANSITION_TOP]:  # type: ignore[arg-type]
                            del row[k]
                self.trans_updates += 1
            self.prev_uniq_by_layer[int(layer_idx)] = now

    def to_payload(self) -> dict:
        """Serialize the transition table (fingerprint filled by caller)."""
        try:
            with self.lock:
                layers: dict[str, dict[str, dict[str, float]]] = {}
                for (layer, expert), row in self.trans.items():
                    layers.setdefault(str(int(layer)), {})[str(int(expert))] = {
                        str(int(k)): float(v) for k, v in row.items()
                    }
                return {"version": 1, "updates": int(self.trans_updates), "trans": layers}
        except Exception:
            return {"version": 1, "updates": 0, "trans": {}}

    def load_payload(self, payload: dict) -> int:
        """Load a transition payload; returns sources restored."""
        try:
            raw = (payload or {}).get("trans") or {}
            n = 0
            with self.lock:
                for layer_s, experts in raw.items():
                    for expert_s, row in (experts or {}).items():
                        clean = {
                            int(k): float(v)
                            for k, v in (row or {}).items()
                        }
                        if clean:
                            self.trans[(int(layer_s), int(expert_s))] = dict(
                                sorted(clean.items(), key=lambda kv: kv[1], reverse=True)[:_TRANSITION_TOP]
                            )
                            n += 1
                self.trans_updates += int((payload or {}).get("updates", 0) or 0)
            return n
        except Exception:
            return 0

    def predict_next(self, layer_idx: int, ids: list[int], k: int = _TRANSITION_OVERFETCH) -> list[int]:
        """FU1: top-k next-token candidates for this layer's demand set.

        Scores sum the transition rows of the demanded experts; the demanded
        ids themselves are excluded (the demand path already loads them).
        Returns at most k ids, possibly fewer (cold table). Lock-guarded.
        """
        if not _TRANSITION_ENV or k <= 0 or not ids:
            return []
        try:
            scores: Dict[int, float] = {}
            now = {int(e) for e in ids}
            with self.lock:
                for e in now:
                    row = self.trans.get((int(layer_idx), int(e)))
                    if not row:
                        continue
                    for cand, w in row.items():
                        if cand not in now:
                            scores[int(cand)] = scores.get(int(cand), 0.0) + float(w)
            ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
            return [int(c) for c in ranked[: max(1, int(k))]]
        except Exception:
            return []

    def bump(self, key: str, amount: int = 1) -> None:
        with self.lock:
            self.stats[key] = self.stats.get(key, 0) + amount

    def is_closed(self) -> bool:
        with self.lock:
            return self.closed

    def close(self) -> None:
        """Stop speculation: mark closed and clear the per-conversion state.

        Idempotent. The routing history and registry are cleared so a
        closed state can never serve another conversion's speculation.
        """
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.prev_uniq_by_layer.clear()
            self.linears_by_layer.clear()

# Fase L1: count every layer-context fallback to the legacy per-expert
# resolution, per reason, so a bench run can prove the fast path engaged.
# Fase M3: ctx fallback counters moved onto ExpertLRUCache (per-engine/
# per-conversion) — module-global counters would mix sessions in a
# persistent server. Reasons: read_failure (ctx read produced nothing
# usable), bank_too_large (union declined over _CTX_UNION_MAX_BYTES or the
# rolling loader's bank cap), tier_mismatch (bundles did not cover the
# demand set), dict_backing (projections without a bank reader, so no
# context was built at all).


# Routing trace (Fase I3): when OMLX_EXPERT_STREAMING_TRACE is set, append one
# JSONL row per MoE layer call ({call, layer, positions, uniq}) so
# bench/lrc_analysis.py can compute routing-consistency (SRP/SCH) offline.
_TRACE_PATH = os.environ.get("OMLX_EXPERT_STREAMING_TRACE", "") or None
_TRACE_FILE = None
_TRACE_CALL = 0


def _trace_row(layer_idx: int, uniq_list: list, positions: int) -> None:
    global _TRACE_FILE, _TRACE_CALL
    if _TRACE_FILE is None:
        _TRACE_FILE = open(_TRACE_PATH, "a", buffering=1)  # noqa: SIM115
    _TRACE_CALL += 1
    _TRACE_FILE.write(
        json.dumps(
            {
                "call": _TRACE_CALL,
                "layer": layer_idx,
                "positions": positions,
                "uniq": [int(e) for e in uniq_list],
            }
        )
        + "\n"
    )

# Parallel os.pread pool for the demand-set of one MoE layer call. Workers
# return raw numpy slices only — MLX promotion happens on the inference
# thread. QD8 sustains ~1.5 GB/s on the reference NVMe; QD16 plateaus near
# ~2.5 GB/s (+34% decode) — see E1. OMLX_EXPERT_STREAMING_QD overrides.
#
# B1 correction (Fase J): _EXPERT_IO_POOL is a process-wide SINGLETON with
# 16 workers shared across all concurrent parents. Device depth is 16
# total, not N*16. Do not "fix" this to per-call pools — that oversubscribes
# and regressed at QD32 (faseJ bench 0a4d3c7: 3.324/3.419 tok/s at QD16 vs
# 3.138 at QD32). The sweep value 16 is process-wide. (Fase K F6 port brings
# the rolling layer-context prefetch; see _CTX_PREFETCH_AHEAD — that knob
# raises depth on the rolling path, this pool stays 16.)
_EXPERT_IO_POOL = ThreadPoolExecutor(
    max_workers=max(1, int(os.environ.get("OMLX_EXPERT_STREAMING_QD", "") or 16)),
    thread_name_prefix="omlx-expert-io",
)

# Fase K F12 (opt-in): a separate pool for PREFILL-SHAPED calls.
# OMLX_EXPERT_STREAMING_PREFILL_QD=<workers>. The sweep evidence: QD16 is the
# decode optimum, QD24 measured the better 8k TTFT (85 s). Two bounded regime
# pools keep both numbers without oversubscribing either phase; 0 (default)
# keeps the single process-wide pool. Selection happens per layer call from
# the count of routed positions (> _PREFILL_REGIME_MIN_POSITIONS).
_PREFILL_QD_ENV = max(0, int(os.environ.get("OMLX_EXPERT_STREAMING_PREFILL_QD", "") or 0))
_PREFILL_REGIME_MIN_POSITIONS = 64
_PREFILL_IO_POOL: ThreadPoolExecutor | None = (
    ThreadPoolExecutor(
        max_workers=_PREFILL_QD_ENV,
        thread_name_prefix="omlx-expert-io-prefill",
    )
    if _PREFILL_QD_ENV > 0
    else None
)


def io_pool_for_positions(
    linear: Any, positions: int
) -> ThreadPoolExecutor:
    """Regime pool for a layer call: prefill-shaped calls may use the
    separate bounded pool; decode keeps the process-wide QD16 singleton."""
    override = getattr(linear, "_io_pool_override", None)
    if override is not None:
        return override
    if _PREFILL_IO_POOL is not None and positions > _PREFILL_REGIME_MIN_POSITIONS:
        return _PREFILL_IO_POOL
    return _EXPERT_IO_POOL

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
class LayerProfile:
    calls: int = 0
    gate_eval_s: float = 0.0
    unique_s: float = 0.0
    load_s: float = 0.0
    stack_s: float = 0.0
    gpu_s: float = 0.0
    load_hits: int = 0
    load_misses: int = 0
    experts_requested: int = 0
    positions: int = 0
    # load-source split: staging (prefetch) vs synchronous backing read
    staged_hits: int = 0
    staged_s: float = 0.0  # take + promote (np -> mx on this thread)
    sync_loads: int = 0
    sync_s: float = 0.0  # backing read (np copy) + promote


class ProfileAccumulator:
    """Per-layer stage timing for the streaming switch (Fase 0 instrumentation).

    Buckets per layer, per token:
      gate_eval  – mx.eval(indices) + device->host copy
      unique     – np.unique + id remap
      load       – _load_expert_bundle total (split hits/misses)
      stack      – mx.stack of mini-bank + gather graph build (lazy; kernel cost
                   shows up in GLU wall time)
    Wall time (full GLU __call__) is tracked separately so kernel cost can be
    derived as wall − ∑(linears buckets).
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.layers: Dict[int, LayerProfile] = {}
        self.wall_s: Dict[int, float] = {}
        self.predicted: Dict[int, set] = {}
        self.observed: Dict[int, set] = {}

    def record_predicted(self, idx: int, ids: Any) -> None:
        if not self.enabled:
            return
        self.predicted.setdefault(idx, set()).update(int(v) for v in ids)

    def record_observed(self, idx: int, ids: Any) -> None:
        if not self.enabled:
            return
        self.observed.setdefault(idx, set()).update(int(v) for v in ids)

    def add(
        self,
        idx: int,
        *,
        gate: float,
        unique: float,
        load: float,
        stack: float,
        hits: int,
        misses: int,
        experts: int,
        positions: int,
    ) -> None:
        if not self.enabled:
            return
        lp = self.layers.setdefault(idx, LayerProfile())
        lp.calls += 1
        lp.gate_eval_s += gate
        lp.unique_s += unique
        lp.load_s += load
        lp.stack_s += stack
        lp.load_hits += hits
        lp.load_misses += misses
        lp.experts_requested += experts
        lp.positions += positions

    def add_wall(self, idx: int, dt: float) -> None:
        if not self.enabled:
            return
        self.wall_s[idx] = self.wall_s.get(idx, 0.0) + dt

    def add_gpu(self, idx: int, dt: float) -> None:
        if not self.enabled:
            return
        lp = self.layers.setdefault(idx, LayerProfile())
        lp.gpu_s += dt

    def add_load_source(self, idx: int, *, staged: bool, dt: float) -> None:
        if not self.enabled:
            return
        lp = self.layers.setdefault(idx, LayerProfile())
        if staged:
            lp.staged_hits += 1
            lp.staged_s += dt
        else:
            lp.sync_loads += 1
            lp.sync_s += dt

    def report(self) -> dict:
        per: Dict[str, dict] = {}
        totals = LayerProfile()
        for idx in sorted(self.layers):
            lp = self.layers[idx]
            c = max(lp.calls, 1)
            per[str(idx)] = {
                "calls": lp.calls,
                "gate_eval_ms": lp.gate_eval_s / c * 1e3,
                "unique_ms": lp.unique_s / c * 1e3,
                "load_ms": lp.load_s / c * 1e3,
                "stack_ms": lp.stack_s / c * 1e3,
                "gpu_ms": lp.gpu_s / c * 1e3,
                "wall_ms": self.wall_s.get(idx, 0.0) / c * 1e3,
                "load_hits": lp.load_hits,
                "load_misses": lp.load_misses,
                "hit_rate": lp.load_hits / max(lp.load_hits + lp.load_misses, 1),
                "staged_hits": lp.staged_hits,
                "staged_ms_per_hit": lp.staged_s / max(lp.staged_hits, 1) * 1e3,
                "sync_loads": lp.sync_loads,
                "sync_ms_per_load": lp.sync_s / max(lp.sync_loads, 1) * 1e3,
                "experts_req_per_call": lp.experts_requested / c,
                "positions_per_call": lp.positions / c,
            }
            totals.calls += lp.calls
            totals.gate_eval_s += lp.gate_eval_s
            totals.unique_s += lp.unique_s
            totals.load_s += lp.load_s
            totals.stack_s += lp.stack_s
            totals.gpu_s += lp.gpu_s
            totals.load_hits += lp.load_hits
            totals.load_misses += lp.load_misses
            totals.experts_requested += lp.experts_requested
            totals.positions += lp.positions
            totals.staged_hits += lp.staged_hits
            totals.staged_s += lp.staged_s
            totals.sync_loads += lp.sync_loads
            totals.sync_s += lp.sync_s
        n = max(totals.calls, 1)
        tots = {
            "calls": totals.calls,
            "gate_eval_ms": totals.gate_eval_s / n * 1e3,
            "unique_ms": totals.unique_s / n * 1e3,
            "load_ms": totals.load_s / n * 1e3,
            "stack_ms": totals.stack_s / n * 1e3,
            "gpu_ms": totals.gpu_s / n * 1e3,
            "load_hits": totals.load_hits,
            "load_misses": totals.load_misses,
            "hit_rate_global": totals.load_hits / max(totals.load_hits + totals.load_misses, 1),
            "staged_hits": totals.staged_hits,
            "staged_ms_per_hit": totals.staged_s / max(totals.staged_hits, 1) * 1e3,
            "sync_loads": totals.sync_loads,
            "sync_ms_per_load": totals.sync_s / max(totals.sync_loads, 1) * 1e3,
            "wall_ms_per_call": sum(self.wall_s.values()) / n * 1e3,
            "layers": len(self.layers),
        }
        # Prediction accuracy: of the ids actually requested per layer, how
        # many had been predicted by the lookahead at least once
        pred_acc = {}
        pred_tot = obs_tot = hit_tot = 0
        for idx in sorted(set(self.predicted) | set(self.observed)):
            pr = self.predicted.get(idx, set())
            ob = self.observed.get(idx, set())
            hit = len(pr & ob)
            pred_tot += len(pr)
            obs_tot += len(ob)
            hit_tot += hit
            pred_acc[str(idx)] = {
                "predicted": len(pr),
                "observed": len(ob),
                "hit": hit,
                "recall": hit / max(len(ob), 1),
                "precision": hit / max(len(pr), 1),
            }
        return {
            "per_layer": per,
            "totals": tots,
            "prediction": pred_acc,
            "prediction_totals": {
                "predicted": pred_tot,
                "observed": obs_tot,
                "hit": hit_tot,
                "recall": hit_tot / max(obs_tot, 1),
            },
        }

try:
    from .shard_bank import ExpertBackingStore
except Exception:  # pragma: no cover
    ExpertBackingStore = Any  # type: ignore


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    def hit_rate(self) -> float:
        tot = self.hits + self.misses
        return self.hits / tot if tot else 0.0


class ExpertLRUCache:
    """Per-layer LRU for expert slices (global budget split evenly).

    Each slot holds one expert's bundle for one layer (weight+scales+biases).
    Budget is split across MoE layers → per-layer capacity = budget // (layers*per_expert)
    approximated via total capacity, but eviction is per-layer to avoid cross-layer thrashing.
    `size`/`capacity` remain global totals for logging.
    """

    def __init__(self, budget_bytes: int, per_expert_bytes: int, num_layers: int | None = None):
        self.budget_bytes = int(budget_bytes)
        self.per_expert_bytes = int(per_expert_bytes)
        self.num_layers = int(num_layers) if num_layers else 0
        if per_expert_bytes > 0:
            self.capacity = max(1, budget_bytes // per_expert_bytes) if budget_bytes > 0 else 0
        else:
            self.capacity = 0
        # per-layer stores to avoid global thrashing (layer 47 evicting layer 0)
        if self.num_layers > 0 and self.capacity > 0:
            per_layer = max(1, self.capacity // self.num_layers)
            # distribute remainder
            self._per_layer_cap = per_layer
            self._global_cap = self.capacity
        else:
            self._per_layer_cap = self.capacity
            self._global_cap = self.capacity
        self._store: OrderedDict[tuple[int, int, str], Any] = OrderedDict()
        # per-layer tracking for eviction
        self._layer_counts: Dict[int, int] = {}
        self.stats = CacheStats()
        self.profile = ProfileAccumulator(enabled=_PROFILE_ENV)
        # Fase K K1: per-conversion speculation state (set by the converter).
        self.spec_state: SpeculationState | None = None
        # B5 admission filter: scan-resistant frequency window (only when env set).
        # Fase K F9: the old capacity < 4096 cap meant the filter never engaged
        # at the 6 GiB budgets measured net-negative (capacity ~6847): a
        # 1024-entry window is trivial next to the read volume, so drop it.
        self._admission_enabled = bool(_ADMISSION_ENV and self.capacity > 0)
        # P3: window scales with capacity (fixed 1024 drowns at GiB budgets).
        if _ADMISSION_WINDOW_ENV > 0:
            self._admission_window = max(16, _ADMISSION_WINDOW_ENV)
        else:
            self._admission_window = max(1024, min(self.capacity // 4, 16384)) if self.capacity > 0 else 1024
        self._admission_counts: Dict[Tuple[int, int, str], int] = {}
        self._admission_order: deque[Tuple[int, int, str]] = deque()  # type: ignore[type-arg]
        self.admission_drops = 0
        # Fase M3: per-engine ctx fallback-to-legacy counters (per reason).
        self._ctx_fallbacks: dict[str, int] = {}

    def _count_ctx_fallback(self, reason: str) -> None:
        self._ctx_fallbacks[reason] = self._ctx_fallbacks.get(reason, 0) + 1
        if memtrace.enabled:
            memtrace.record("ctx.fallback", reason=reason)

    def ctx_fallback_stats(self) -> dict[str, int]:
        return dict(self._ctx_fallbacks)

    def _reset_ctx_fallback_stats(self) -> None:
        self._ctx_fallbacks.clear()

    def __contains__(self, key: tuple[int, int, str]) -> bool:
        return key in self._store

    def _layer_of(self, key: tuple[int, int, str]) -> int:
        try:
            return int(key[0])
        except Exception:
            return -1

    def get(self, key: tuple[int, int, str]) -> Any | None:
        if key in self._store:
            self._store.move_to_end(key)
            self.stats.hits += 1
            return self._store[key]
        self.stats.misses += 1
        return None

    def _admission_should_insert(self, key: tuple[int, int, str]) -> bool:
        if not self._admission_enabled:
            return True
        # P3: frequency keyed by the FULL bundle key (layer, expert,
        # stacked_key incl. HOBBIT tier suffix) — the old (layer, expert)
        # key let a hot gate_proj admit a cold up_proj of the same expert.
        lk = (int(key[0]), int(key[1]), str(key[2]))
        c = self._admission_counts.get(lk, 0) + 1
        self._admission_counts[lk] = c
        self._admission_order.append(lk)
        if len(self._admission_order) > self._admission_window:
            old = self._admission_order.popleft()
            oc = self._admission_counts.get(old, 0) - 1
            if oc <= 0:
                self._admission_counts.pop(old, None)
            else:
                self._admission_counts[old] = oc
        if c < 2:
            self.admission_drops += 1
            return False
        return True

    def put(self, key: tuple[int, int, str], value: Any) -> None:
        if self.capacity <= 0:
            return
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        if not self._admission_should_insert(key):
            return
        # per-layer cap enforcement
        layer = self._layer_of(key)
        if self.num_layers > 0 and self._per_layer_cap:
            cnt = self._layer_counts.get(layer, 0)
            # evict oldest entry of same layer if per-layer full
            if cnt >= self._per_layer_cap:
                # find oldest entry of this layer
                for k in list(self._store.keys()):
                    if self._layer_of(k) == layer:
                        self._store.pop(k)
                        self.stats.evictions += 1
                        self._layer_counts[layer] = max(0, self._layer_counts.get(layer, 1) - 1)
                        break
                # if still over capacity due to rounding, fall through to global
        # global cap
        while len(self._store) >= self.capacity:
            old_k, _ = self._store.popitem(last=False)
            self.stats.evictions += 1
            old_layer = self._layer_of(old_k)
            self._layer_counts[old_layer] = max(0, self._layer_counts.get(old_layer, 1) - 1)
        self._store[key] = value
        self._layer_counts[layer] = self._layer_counts.get(layer, 0) + 1

    def clear(self) -> None:
        self._store.clear()
        self._layer_counts.clear()
        self.stats = CacheStats()

    @property
    def policy(self) -> str:
        return "lru"

    def retain_hot(self, hot_pairs: set) -> int:
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
        if evicted:
            counts: Dict[int, int] = {}
            for key in self._store:
                layer = self._layer_of(key)
                counts[layer] = counts.get(layer, 0) + 1
            self._layer_counts = counts
            self.stats.evictions += evicted
        return evicted

    @property
    def size(self) -> int:
        return len(self._store)


_CACHE_POLICY_ENV = os.environ.get("OMLX_EXPERT_STREAMING_CACHE", "lru").strip().lower()


class S3FIFOExpertCache(ExpertLRUCache):
    """P2: S3-FIFO eviction behind the ExpertLRUCache interface.

    LRU keeps recency; MoE routing is skewed (heavy hitters + scan-like
    prefill demand), where S3-FIFO's scan resistance wins: a small FIFO
    filters one-hit wonders, the main queue holds reuse, and a ghost
    queue promotes re-referenced entries (2nd chance). Per-layer caps,
    admission filter, retain_hot, stats and the spec hooks are
    inherited unchanged — only the global eviction order differs.
    Select with OMLX_EXPERT_STREAMING_CACHE=s3fifo (default lru).
    """

    def __init__(self, budget_bytes: int, per_expert_bytes: int, num_layers: int | None = None):
        super().__init__(budget_bytes, per_expert_bytes, num_layers)
        self._small: OrderedDict = OrderedDict()
        self._ghost: OrderedDict = OrderedDict()
        cap = max(1, self.capacity)
        # A/B offline finding (traces jang4m/jang4s, 2026-09): a 10% small
        # FIFO holds ~300 slots against a ~1440-key decode working set, so
        # every small entry churns before its re-reference and main stays
        # empty (hit ~0 vs LRU ~0.43-0.75). Small must cover a full decode
        # token across all layers: per_layer_cap slots x num_layers. With
        # per-layer caps active this approaches capacity (documented limit
        # of S3-FIFO under per-layer quotas — see A/B note in docs).
        per_layer = max(1, self._per_layer_cap) if self.num_layers > 0 else 0
        working = per_layer * max(1, self.num_layers) if self.num_layers > 0 else cap // 10
        self._small_cap = max(1, min(cap - 1, max(cap // 10, working)))
        self._ghost_cap = max(self._small_cap, cap // 10)

    @property
    def policy(self) -> str:
        return "s3fifo"

    def _in(self, key: tuple[int, int, str]) -> int:
        if key in self._small:
            return 0
        if key in self._store:
            return 1
        return -1

    def __contains__(self, key: tuple[int, int, str]) -> bool:
        return key in self._small or key in self._store

    def get(self, key: tuple[int, int, str]) -> Any | None:
        if key in self._small:
            # Promote small -> main on re-reference (frequency signal).
            val = self._small.pop(key)
            self._store[key] = val
            self.stats.hits += 1
            return val
        if key in self._store:
            self._store.move_to_end(key)
            self.stats.hits += 1
            return self._store[key]
        self.stats.misses += 1
        return None

    def put(self, key: tuple[int, int, str], value: Any) -> None:
        if self.capacity <= 0:
            return
        if key in self._small:
            self._small[key] = value
            return
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = value
            return
        if not self._admission_should_insert(key):
            return
        layer = self._layer_of(key)
        if self.num_layers > 0 and self._per_layer_cap:
            cnt = self._layer_counts.get(layer, 0)
            if cnt >= self._per_layer_cap:
                if not self._evict_layer(layer):
                    self._evict_one_global()
        # Ghost hit -> main queue (2nd chance); else small queue.
        target_main = key in self._ghost
        if key in self._ghost:
            del self._ghost[key]
        total = len(self._small) + len(self._store)
        while total >= self.capacity:
            self._evict_one_global()
            total = len(self._small) + len(self._store)
        if target_main:
            self._store[key] = value
        else:
            if len(self._small) >= self._small_cap:
                old_k, _ = self._small.popitem(last=False)
                self.stats.evictions += 1
                self._layer_counts[self._layer_of(old_k)] = max(
                    0, self._layer_counts.get(self._layer_of(old_k), 1) - 1
                )
                self._ghost[old_k] = None
                while len(self._ghost) > self._ghost_cap:
                    self._ghost.popitem(last=False)
            self._small[key] = value
        self._layer_counts[layer] = self._layer_counts.get(layer, 0) + 1

    def _evict_layer(self, layer: int) -> bool:
        for store in (self._small, self._store):
            for k in list(store.keys()):
                if self._layer_of(k) == layer:
                    store.pop(k)
                    self.stats.evictions += 1
                    self._layer_counts[layer] = max(0, self._layer_counts.get(layer, 1) - 1)
                    return True
        return False

    def _evict_one_global(self) -> None:
        if len(self._small):
            old_k, _ = self._small.popitem(last=False)
            self._ghost[old_k] = None
            while len(self._ghost) > self._ghost_cap:
                self._ghost.popitem(last=False)
        elif len(self._store):
            old_k, _ = self._store.popitem(last=False)
        else:
            return
        self.stats.evictions += 1
        self._layer_counts[self._layer_of(old_k)] = max(
            0, self._layer_counts.get(self._layer_of(old_k), 1) - 1
        )

    def clear(self) -> None:
        super().clear()
        self._small.clear()
        self._ghost.clear()

    def retain_hot(self, hot_pairs: set) -> int:
        if self.capacity <= 0 or (not self._store and not self._small):
            return 0
        evicted = 0
        for store in (self._small, self._store):
            for key in list(store.keys()):
                if (key[0], key[1]) not in hot_pairs:
                    del store[key]
                    evicted += 1
        if evicted:
            counts: Dict[int, int] = {}
            for store in (self._small, self._store):
                for key in store:
                    layer = self._layer_of(key)
                    counts[layer] = counts.get(layer, 0) + 1
            self._layer_counts = counts
            self.stats.evictions += evicted
        return evicted

    @property
    def size(self) -> int:
        return len(self._small) + len(self._store)


def make_expert_cache(
    budget_bytes: int,
    per_slot: int,
    num_layers: int | None = None,
    policy: str | None = None,
) -> ExpertLRUCache:
    """P2: build the configured eviction policy (default LRU).

    ``policy`` (per-model setting) wins over the env default; None keeps
    OMLX_EXPERT_STREAMING_CACHE ("lru").
    """
    eff = (policy or _CACHE_POLICY_ENV or "lru").strip().lower()
    if eff == "s3fifo":
        return S3FIFOExpertCache(budget_bytes, per_slot, num_layers=num_layers)
    return ExpertLRUCache(budget_bytes, per_slot, num_layers=num_layers)


# ---------------------------------------------------------------------------
# Helpers that mirror switch_layers.py
# ---------------------------------------------------------------------------

def promote_np_array(v: Any, dtype_str: str | None = None):
    """P3: single promotion rule for numpy -> MLX (QuantHandler registry).

    Centralizes the two call sites that used to duplicate the BF16-as-uint16
    reinterpret (here and _promote_banks): a new quantization adds one
    branch here plus a row in the grade test — never a third copy.
    Handlers keyed by (stored numpy dtype, safetensors dtype string):
      (uint16, BF16) -> bit-exact reinterpret (matches mx.load; the old
        shift->f32->astype path flushed subnormals via Metal FTZ and cost
        ~9x more on 4 MB slices).
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

    Scope: quantized only (Fase J G4). The context is driven through hooks
    that exist solely on StreamingQuantizedSwitchLinear — bundle_key,
    _bank_bytes_for and _load_expert_bank_np(_full) — and it is only
    constructed when the owning GLU is quantized, so StreamingSwitchLinear
    (bf16) never participates. That is intentional: the bf16 path resolves
    one projection at a time inside its own __call__ and therefore has no
    cross-projection union for the context to collapse.

    Fase K adaptation: cache keys go through linear.bundle_key, so under
    the HOBBIT split hot and cold copies of one expert never alias, and the
    bank reads are tier-segmented (read_expert_into components are
    tier-homogeneous by contract).

    Two modes, selected by OMLX_EXPERT_STREAMING_CTX_ROLLING:

    rolling (default — Etapa B)
        Each projection resolves its own bank on demand. At most
        _CTX_PREFETCH_AHEAD following projections are read on pool workers
        in the background, so the next bank is in flight while the current
        one is promoted and consumed on the GPU. Peak NumPy residency drops
        from the *union* of every projection (~3 banks) to ~1-2 banks.

    union (legacy — set the env var to 0)
        One pool.map across every projection; all banks are resident until
        the last projection is consumed. Maximum I/O parallelism, highest RSS.

    Both modes preserve the C6 contract: one shared routing plan, and reads
    performed on IO-pool workers that never allocate MLX arrays.
    """

    def __init__(
        self, linears: list[Any], cache: ExpertLRUCache, mode: str | None = None
    ):
        # Fase 1: the GLU picks the mode per call (union for decode-shaped,
        # rolling for prefill); OMLX_EXPERT_STREAMING_CTX_ROLLING=0 remains
        # the global kill switch (forces union here).
        self.mode = mode or ("union" if not _CTX_ROLLING_ENV else "rolling")
        self.linears = linears
        self.cache = cache
        self.bundles: dict[int, dict[int, tuple]] = {}
        self.hits: dict[int, int] = {}
        self.misses: dict[int, int] = {}
        self.failed = False
        # Etapa A1b: the raw contiguous NumPy banks behind bundles, kept so
        # the linear can promote a whole demand set with one mx.array per key
        # instead of U per-expert arrays plus a stack. Populated only when the
        # read covered the *entire* demand set (all-miss); bank_ids records
        # exactly which ids the bank holds, so a stale bank can never be
        # promoted against a demand set it does not describe.
        self.bank_raw: dict[int, Any] = {}
        self.bank_ids: dict[int, list[int]] = {}
        # rolling state
        self._order: dict[int, int] = {id(lin): i for i, lin in enumerate(linears)}
        self._futures: dict[int, Any] = {}
        self._inflight: dict[int, int] = {}
        self._resolved: set[int] = set()
        self._expert_ids: list[int] = []
        # legacy union latch
        self._loaded = False
        # Fase L1: union declined a demand set over _CTX_UNION_MAX_BYTES; the
        # linears fall back to the legacy per-expert resolution.
        self.declined = False
        # Fase L1: why the last resolve failed, when it did (read_failure vs
        # bank_too_large) so the fallback counter reports the true reason.
        self.fallback_reason: str | None = None

    # -- helpers ------------------------------------------------------------

    def _split(self, linear: Any, expert_ids: list[int]) -> tuple[dict, list[int]]:
        """Partition expert_ids into cached bundles and missing ids."""
        cached: dict[int, tuple] = {}
        missing: list[int] = []
        for eid in expert_ids:
            key = linear.bundle_key(eid)
            value = self.cache.get(key)
            if value is None:
                missing.append(eid)
            else:
                cached[eid] = value
        return cached, missing

    @staticmethod
    def _pool_for(linear: Any, positions: int = 0):
        # Fase K F12/K4: regime by asked demand size (decode ~10 experts,
        # prefill ~hundreds). The caller passes the context's demand-set
        # size — the rolling path MUST route its prefetch submissions
        # through this delegate, not the fixed singleton (K4: before the
        # fix the 24-worker prefill pool never saw a single rolling task).
        return io_pool_for_positions(linear, positions)

    @property
    def _inflight_bytes(self) -> int:
        return sum(self._inflight.values())

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
            cached, missing = self._split(nxt, self._expert_ids)
            self.bundles[nid] = cached
            self.hits[nid] = len(cached)
            self.misses[nid] = len(missing)
            if not missing:
                # Fully cached: nothing to read, mark resolved so the linear
                # short-circuits when it asks.
                self._resolved.add(nid)
                continue
            bank_bytes = int(nxt._tier_bank_bytes_for(missing))
            if bank_bytes > _CTX_PREFETCH_MAX_BYTES:
                continue
            # Etapa A1b: ask for the raw contiguous banks as well when
            # single-promotion is on. The read is identical either way — only
            # what the worker hands back differs, and it stays NumPy, so no
            # MLX op is ever created on a pool thread.
            reader = (
                nxt._load_expert_bank_np_full
                if _BANK_PROMOTE_CTX_ENV
                else nxt._load_expert_bank_np
            )
            self._futures[nid] = self._pool_for(nxt, len(self._expert_ids)).submit(reader, missing)
            self._inflight[nid] = bank_bytes
            submitted += 1

    def _ensure_rolling(self, linear: Any, expert_ids: list[int]) -> None:
        lid = id(linear)
        if lid in self._resolved:
            return
        self._resolved.add(lid)
        if not self._expert_ids:
            self._expert_ids = list(expert_ids)
        ids = self._expert_ids

        # A prefetch may already be in flight; the split is recomputed because
        # the cache can change between submit and await. K4: the FIRST
        # projection of the rolling path resolves synchronously here — its
        # read depth is bounded by read_expert_into's _RUN_IO_QD run pool,
        # not by the regime pool; only the FOLLOWING projections (prefetch)
        # run on the regime pool. Sizing stays safe: the sync read is what
        # the demand path always paid.
        fut = self._futures.pop(lid, None)
        self._inflight.pop(lid, None)
        cached, missing = self._split(linear, ids)
        self.bundles[lid] = cached
        self.hits[lid] = len(cached)
        self.misses[lid] = len(missing)
        bank_bytes = int(linear._tier_bank_bytes_for(missing))

        if missing:
            if fut is not None:
                try:
                    got = fut.result()
                except Exception:
                    got = None
            else:
                got = (
                    linear._load_expert_bank_np_full(missing)
                    if _BANK_PROMOTE_CTX_ENV
                    else (None, linear._load_expert_bank_np(missing))
                )
            # A worker dispatched as bare rows (prefetch submitted before this
            # call, or with the knob off) yields a list; normalise to the
            # (segments, rows) shape so the consumer has one contract.
            if got is not None and not isinstance(got, tuple):
                got = (None, got)
            rows = None if got is None else got[1]
            if rows is None or len(rows) != len(missing):
                self.failed = True
                self.fallback_reason = (
                    "bank_too_large" if bank_bytes > _BANK_MAX_BYTES else "read_failure"
                )
                if memtrace.enabled:
                    memtrace.record(
                        "ctx.ensure.fail",
                        layer=linear.layer_idx,
                        proj=getattr(linear, "proj_name", "?"),
                        uniq=len(ids),
                        miss=len(missing),
                        reason=self.fallback_reason,
                    )
                return
            self.bundles[lid].update(zip(missing, rows))
            # Etapa A1b: single-promotion is only valid when the read covered
            # the *whole* demand set. A partial bank would have to be
            # concatenated with separately promoted cache hits, which changes
            # the layout contract, so it is left on the legacy path.
            if _BANK_PROMOTE_CTX_ENV and got[0] is not None and len(missing) == len(ids):
                self.bank_raw[lid] = got[0]
                self.bank_ids[lid] = list(missing)

        self._prefetch(linear)
        if memtrace.enabled:
            memtrace.record(
                "ctx.ensure.exit",
                layer=linear.layer_idx,
                proj=getattr(linear, "proj_name", "?"),
                ctx_mode=self.mode,
                positions=len(ids),
                uniq=len(ids),
                miss=len(missing),
                ctx_bank_bytes=int(linear._tier_bank_bytes_for(missing)),
                bank_bytes=int(linear._tier_bank_bytes_for(missing)),
                ctx_inflight_bytes=self._inflight_bytes,
                inflight_bytes=self._inflight_bytes,
                ctx_prefetch_count=len(self._futures),
                inflight=len(self._futures),
            )

    # -- legacy union path --------------------------------------------------

    def _ensure_union(self, linear: Any, expert_ids: list[int]) -> None:
        if self._loaded:
            return
        self._loaded = True
        layer = self.linears[0].layer_idx if self.linears else -1
        if memtrace.enabled:
            memtrace.record(
                "ctx.ensure.enter",
                layer=layer,
                n_proj=len(self.linears),
                uniq=len(expert_ids),
            )
        jobs: list[tuple[Any, list[int]]] = []
        for proj in self.linears:
            cached, missing = self._split(proj, expert_ids)
            self.bundles[id(proj)] = cached
            self.hits[id(proj)] = len(cached)
            self.misses[id(proj)] = len(missing)
            if missing:
                jobs.append((proj, missing))
        if not jobs:
            return
        pool = self._pool_for(jobs[0][0], len(expert_ids))
        live = sum(proj._tier_bank_bytes_for(ids) for proj, ids in jobs)
        # Fase L1: a demand set too large for union residency declines so the
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
                self.fallback_reason = (
                    "bank_too_large"
                    if any(
                        p._tier_bank_bytes_for(ids_) > _BANK_MAX_BYTES
                        for p, ids_ in jobs
                    )
                    else "read_failure"
                )
                return
            self.bundles[id(proj)].update(zip(ids, rows))
        if memtrace.enabled:
            memtrace.record(
                "ctx.ensure.exit",
                layer=layer,
                ctx_mode=self.mode,
                n_proj=len(self.linears),
                positions=len(expert_ids),
                uniq=len(expert_ids),
                n_loaded=len(jobs),
                miss_per_proj=[len(ids) for _, ids in jobs],
                ctx_bank_bytes=live,
                bank_bytes=live,
                ctx_inflight_bytes=0,
                ctx_prefetch_count=0,
            )

    # -- public API ---------------------------------------------------------

    def ensure(self, linear: Any, expert_ids: list[int]) -> None:
        """Resolve linear's demand set for this layer call."""
        if self.mode == "union":
            self._ensure_union(linear, expert_ids)
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
    gate_s: float = 0.0
    unique_s: float = 0.0
    uniq_mx: Any = None  # MLX unique expert IDs reused by bias gather
    ctx: Any = None  # Fase K F6: per-layer load context (quantized GLU)
    # Fase K F2: (target_linear_id, run_first, run_count) already advised in
    # this layer call — the 3 projections share one plan, so dedupe here.
    advised_runs: set = field(default_factory=set)


def _build_plan_into(plan: _RemapPlan, indices) -> None:
    """Populate a shared routing plan in place (called once per MoE layer)."""
    t0 = time.perf_counter()
    mx.eval(indices)
    try:
        flat_np = np.array(indices, copy=False).reshape(-1)
    except Exception:
        flat_list = indices.tolist()  # type: ignore[attr-defined]

        def _flatten(obj):
            if isinstance(obj, list):
                for v in obj:
                    yield from _flatten(v)
            else:
                yield obj

        flat_np = np.array(list(_flatten(flat_list)), dtype=np.int32)
    t1 = time.perf_counter()
    uniq_np = np.unique(flat_np)
    uniq_list = uniq_np.tolist()
    # compact remap via searchsorted (uniq is sorted ascending): vectorized C
    # lookup, replaces the per-element np.vectorize dict indirection
    remapped_np = np.searchsorted(uniq_np, flat_np).astype(np.int32)
    t2 = time.perf_counter()
    plan.indices_shape = tuple(indices.shape)
    plan.flat_np = flat_np
    plan.uniq_list = uniq_list
    plan.remapped = mx.array(remapped_np.reshape(indices.shape))
    plan.uniq_mx = mx.array(uniq_np)
    plan.positions = int(flat_np.size)
    plan.gate_s = t1 - t0
    plan.unique_s = t2 - t1


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

    def _load_expert_weight(self, expert_id: int) -> mx.array:
        key = (self.layer_idx, expert_id, self.stacked_key)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
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
        p = self.cache.profile
        if plan is None:
            plan = _RemapPlan()
        built = plan.flat_np is None
        if built:
            _build_plan_into(plan, indices)
        t2 = time.perf_counter()
        # Load each unique expert weight
        mini_weights = []
        t_load = 0.0
        hits = 0
        misses = 0
        for eid in plan.uniq_list:
            was_hit = (self.layer_idx, eid, self.stacked_key) in self.cache
            t_l = time.perf_counter()
            w = self._load_expert_weight(int(eid))
            t_load += time.perf_counter() - t_l
            if was_hit:
                hits += 1
            else:
                misses += 1
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
            b_mini = mx.stack([self._bias[int(e)] for e in plan.uniq_list], axis=0)  # (U,O)
            out = out + mx.expand_dims(b_mini[remapped], -2)
        t4 = time.perf_counter()
        p.record_observed(self.layer_idx, plan.uniq_list)
        p.add(
            self.layer_idx,
            gate=plan.gate_s if built else 0.0,
            unique=plan.unique_s if built else 0.0,
            load=t_load,
            stack=t4 - t2 - t_load,
            hits=hits,
            misses=misses,
            experts=len(plan.uniq_list),
            positions=plan.positions,
        )
        return out


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
        # HOBBIT hot/cold split (Fase I6): hot experts keep the ORIGINAL
        # packing (source bits/gs below); the rest compute at the cold tier
        # (self._cold_bits/_cold_gs from expert_cold/ metadata). Empty set or
        # None bits = uniform tier (I5) — the single-bank path.
        self._hot_experts: set | None = None
        self._cold_bits: int | None = None
        self._cold_gs: int | None = None

    def set_hobbit_split(self, hot_experts, cold_bits: int, cold_gs: int) -> None:
        """Enable the dual-tier path for this linear (convert-time only)."""
        self._hot_experts = {int(e) for e in (hot_experts or [])}
        self._cold_bits = int(cold_bits)
        self._cold_gs = int(cold_gs)

    def _is_split_active(self) -> bool:
        return (
            self._hot_experts is not None
            and len(self._hot_experts) > 0
            and self._cold_bits is not None
            and self._cold_bits != self.bits
        )

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

    def _per_expert_bytes(self) -> int:
        """Summed per-expert bytes across this projection's stacked tensors."""
        keys = [self.stacked_weight_key, self.stacked_scales_key]
        if self.stacked_biases_key:
            keys.append(self.stacked_biases_key)
        return sum(self._slice_bytes(k) for k in keys)

    def _bank_bytes_for(self, n_experts: int) -> int:
        """Bytes of raw NumPy bank needed to hold n_experts of this projection.

        Used by the layer-context memory bookkeeping and by the demand-set
        bank reader to size reads under the bank cap. Tier-blind (cold-first):
        use _tier_bank_bytes_for when the demand set is known — under the
        HOBBIT split hot experts carry the SOURCE packing width, which this
        cold-first measurement under-estimates (Fase K K6).
        """
        if n_experts <= 0:
            return 0
        return n_experts * self._per_expert_bytes()

    def _tier_groups(self, ids: list[int]) -> dict[int, list[int]]:
        """P2: group ids by tier with ONE pass (split-aware segmentation).

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
        """True raw-bank bytes for an id set under the HOBBIT split (K6).

        Sums per tier group with the TIER's own reader width (hot = source
        packing, cold = tier packing). Without the split it reduces to
        _bank_bytes_for. Never raises: a resolution failure falls back to
        the cold-first estimate so the caps stay at least as strict as
        their pre-K6 behavior for unknown layouts. P2: one reader
        resolution per (key, tier) via _tier_groups.
        """
        if not ids:
            return 0
        try:
            if not self._is_split_active():
                return len(ids) * self._per_expert_bytes()
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
            return len(ids) * self._per_expert_bytes()

    def _slice_view(self, key: str, buf: np.ndarray, expert_id: int) -> np.ndarray:
        """Reshape a raw uint8 expert buffer exactly as expert_slice would.

        Resolves the reader per expert id so a HOBBIT hot expert reinterprets
        with the SOURCE packing and a cold expert with the tier packing —
        mirrors _ShardReader.expert_slice so the promoted mx.array is
        bit-identical to the legacy per-slice path (C2 correctness).
        """
        reader = self.backing._reader_for_key(key, int(expert_id))
        rp = reader._rp_for(key)
        return np.frombuffer(buf, dtype=rp.np_dtype).reshape(rp.per_shape)

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
            # Fase K K5: bridge holes in the C2 read path when the env
            # asks for it (RUN_MERGE_GAP defaults to 0 — measured loss in
            # both regimes, split4/). Gap rows are read with the run; the
            # scatter never promotes them.
            merge_gap = _RUN_MERGE_GAP if split else 0
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
                # P2: resolve each key's reader ONCE per tier-run (all ids
                # in ids_t share the tier by construction) and reuse it for
                # the slice views below — was once per expert per key.
                _readers = []
                for key in keys:
                    reader = self.backing._reader_for_key(key, ids_t[0])
                    _readers.append(reader)
                    per_bytes.append(reader._rp_for(key).expert_bytes)
                if any(size <= 0 for size in per_bytes):
                    return None
                total += len(ids_t) * sum(per_bytes)
                if total > _BANK_MAX_BYTES:
                    return None
                banks = [
                    np.empty((len(ids_t), size), dtype=np.uint8) for size in per_bytes
                ]
                components = [(key, ids_t) for key in keys]
                if not self.backing.read_expert_into(
                    components, banks, merge_gap=merge_gap
                ):
                    return None
                segments.append((ids_t, banks))
                # P2: reinterpret rows with the already-resolved per-tier
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
        except Exception as exc:
            # P2: bank-read failures fall back to per-expert loads — count
            # them (with the layer) instead of failing silently, so a
            # rotting backing shows up in the per-request summary.
            try:
                self.cache._count_ctx_fallback(f"bank_read_l{self.layer_idx}")
                if memtrace.enabled:
                    memtrace.record("bank.read_fail", layer=self.layer_idx, n=len(expert_ids), err=str(exc)[:120])
            except Exception:
                pass
            return None

    def _load_expert_bank_np(self, expert_ids: list[int]) -> list[tuple] | None:
        """Read a demand set into one raw NumPy bank per (key, tier).

        The backing performs coalesced contiguous reads into caller-owned
        banks; rows are then exposed as views for the existing LRU
        representation. Returning None preserves the legacy per-expert
        fallback for dict backings and unsupported layouts.
        """
        got = self._read_expert_banks(expert_ids)
        return None if got is None else got[1]

    def _load_expert_bank_np_full(self, expert_ids: list[int]):
        """Like _load_expert_bank_np, but keeps the raw contiguous banks.

        Needed by the Etapa B layer context: the NumPy read may happen on an
        IO pool worker, yet promoting those buffers to MLX must happen later
        on the inference thread (MLX ops may not be bound off-stream). Same
        failure contract as _load_expert_bank_np — None whenever the backing
        cannot serve the demand set as banks.
        """
        return self._read_expert_banks(expert_ids)

    def _promote_banks(self, segments: list) -> list | None:
        """Promote raw contiguous per-tier banks into one mx.array per key.

        Shared by Etapa A1 (read + promote together) and Etapa A1b (read on
        a pool thread, promote here on the inference thread).

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
                    # here the input is always numpy; normalize dtypes the
                    # legacy way when the registry passed through (non-BF16
                    # stored dtypes keep their mx.array copy).
                    if not isinstance(arr, mx.array):
                        arr = mx.array(typed)
                    one.append(arr)
                while len(one) < 3:
                    one.append(None)
                promoted.append((one[0], one[1], one[2]))
            return promoted
        except Exception as exc:
            # P2: same noisy-fallback contract as _read_expert_banks.
            try:
                self.cache._count_ctx_fallback("bank_promote_fail")
                if memtrace.enabled:
                    memtrace.record("bank.promote_fail", layer=self.layer_idx, err=str(exc)[:120])
            except Exception:
                pass
            return None

    def _load_expert_bank_mx(self, expert_ids: list[int]):
        """Etapa A1: promote an all-miss demand bank in one shot (per tier).

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

        Fase K F7: since the caller neither promotes nor uses extra rows
        (rows outside the requested scatter set are dropped), a run may be
        stretched to BRIDGE a small gap (_RUN_MERGE_GAP, default 0 — the
        missing ids within the SAME tier — prefill demand under the HOBBIT
        split fragments into many single-expert runs; bridging turns them
        into longer sequential reads on the 40 Gbps NVMe at the cost of a
        few idle bytes. The gap experts are read but never promoted or used.
        Bridging is OFF by default: re-measurement showed a net cost in
        BOTH regimes (single-tier 2k 34.0s vs 31.4s; split-active 2k 55.0s
        vs 47.5s, 8k 107.2s vs 98.9s — mergeab/ and split4/ artifacts). It
        remains available via the env knob for backends where sequential
        reads win.
        """
        tier_of = self._tier_of if self._is_split_active() else None
        max_run = _RUN_MAX if max_run is None else max(1, int(max_run))
        merge_gap = _RUN_MERGE_GAP if tier_of is not None else 0
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
            # P2: repeated per-expert read failures warn once per key
            # (a rotting shard otherwise degrades silently to fallbacks).
            try:
                fails = getattr(self.cache, "_read_failures", None)
                if fails is None:
                    fails = {}
                    self.cache._read_failures = fails  # type: ignore[attr-defined]
                key = self.bundle_key(int(expert_id))
                n = int(fails.get(key, 0) or 0) + 1
                fails[key] = n
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
        key = self.bundle_key(expert_id)
        cached = self.cache.get(key)
        if cached is not None:
            # New format: bundle tuple stored under weight key
            if isinstance(cached, tuple) and len(cached) == 3:
                return cached  # type: ignore[return-value]
            # Legacy: companion keys (weight hit but scales separate) — upgrade to bundle
            if isinstance(cached, mx.array):
                s_key = (self.layer_idx, expert_id, self.stacked_scales_key)
                b_key = (self.layer_idx, expert_id, self.stacked_biases_key) if self.stacked_biases_key else None
                s = self.cache.get(s_key)
                b = self.cache.get(b_key) if b_key else None
                if s is not None:
                    bundle = (cached, s, b)
                    # Collapse 3 slots into 1 bundle slot (evict companions)
                    try:
                        self.cache._store.pop(s_key, None)  # type: ignore[attr-defined]
                        if b_key:
                            self.cache._store.pop(b_key, None)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    self.cache.put(key, bundle)  # type: ignore[arg-type]
                    return bundle  # type: ignore[return-value]
        return None

    def _spec_state(self) -> SpeculationState | None:
        """Per-conversion speculation state (Fase K K1).

        The converter hangs one instance on the cache and (when the
        backing is an object) on the backing; it dies with them, so two
        engines can never share ring bytes or routing history.
        """
        state = getattr(self.backing, "spec_state", None)
        if state is None:
            state = getattr(self.cache, "spec_state", None)
        return state

    def _advise_next_layer_prev_token(self, plan: _RemapPlan | None = None) -> None:
        """Speculate the NEXT layer's previous-token experts (Fase K F1/F2).

        spec_state.prev_uniq_by_layer holds layer N+1's expert ids; the
        advisory must therefore hit layer N+1's banks. The converted-
        linears registry resolves the next layer's real stacking keys (and
        its HOBBIT tier routing — backing.advise_expert_run segments runs
        per resolved reader, so hot/cold boundaries are respected
        automatically).

        F2 guards: the advisory is capped at _MAX_ADVISE_ROWS experts
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
        prev = state.prev_uniq_by_layer.get(next_layer)
        if not prev or len(prev) > _MAX_ADVISE_ROWS:
            return
        targets = state.linears_by_layer.get(next_layer)
        if not targets:
            return
        advised_runs = plan.advised_runs if plan is not None else None
        try:
            sorted_prev = sorted(int(e) for e in prev)
            if not sorted_prev:
                return
            # FU1: k+1 overfetch — union the transition-table candidates
            # for the next layer's prev set into the advisory. Same caps
            # and dedup as the base set; hints only, never output.
            try:
                _extra = state.predict_next(next_layer, sorted_prev)
            except Exception:
                _extra = []
            if _extra:
                _have = set(sorted_prev)
                for _c in _extra:
                    if _c not in _have and len(sorted_prev) < _MAX_ADVISE_ROWS:
                        sorted_prev.append(int(_c))
                        _have.add(int(_c))
                sorted_prev.sort()
                try:
                    state.bump("trans_overfetch", len(sorted_prev) - len(prev))
                except Exception:
                    pass
            # Fase K K2: one shared segmentation with the demand path —
            # consecutive ids within one resolved reader become one run for
            # a single F_RDADVISE (tier boundaries break the run).
            from .shard_bank import segment_runs

            rid_of = {
                e: id(self.backing._reader_for_key(self.stacked_weight_key, e))
                for e in sorted_prev
            }
            runs = segment_runs(
                sorted_prev, same=lambda a, b: rid_of[a] == rid_of[b]
            )
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
                            state.bump("advised", count)
                            state.bump("advised_experts", count)
                            state.bump("advised_runs", 1)
                            state.bump("advised_bytes", adv_bytes)
                            state.bump("advice_tier_segments", adv_segs)
                        else:
                            state.bump("advice_failures", 1)
                    except Exception:
                        pass
        except Exception:
            pass

    def _load_expert_bundle(self, expert_id: int) -> tuple[mx.array, mx.array, mx.array | None]:
        key = self.bundle_key(expert_id)
        # Cache / staging resolution (shared with the parallel demand-set path)
        resolved = self._bundle_cached_or_staged(expert_id)
        if resolved is not None:
            return resolved  # type: ignore[return-value]
        # 3) synchronous load from backing
        t_sy = time.perf_counter()
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
        if getattr(self.cache, "profile", None) is not None:
            self.cache.profile.add_load_source(
                self.layer_idx, staged=False, dt=time.perf_counter() - t_sy
            )
        return bundle

    def __call__(self, x, indices, sorted_indices=False, plan: _RemapPlan | None = None):
        p = self.cache.profile
        if plan is None:
            plan = _RemapPlan()
        # Fase K F1/F2: speculation for layer+1, deduped per layer call
        # through plan.advised_runs (the GLU shares one plan).
        _spec_state = self._spec_state()
        if _RA_ENV and _spec_state is not None and _spec_state.prev_uniq_by_layer:
            try:
                self._advise_next_layer_prev_token(plan)
            except Exception:
                pass
        built = plan.flat_np is None
        if built:
            _build_plan_into(plan, indices)
        # C4: while the hotness seeder is active a prefill demand set is not
        # cached — seeding the LRU with prefill-only experts would evict the
        # decode working set.
        cache_result = not (
            getattr(self.cache, "prefill_bypass", False) and plan.positions > 64
        )
        t2 = time.perf_counter()
        bundles: Dict[int, tuple] = {}
        mini_w, mini_s, mini_b = [], [], []
        has_b = False
        hits = 0
        misses = 0
        missing: list[int] = []
        t_res_start = time.perf_counter()
        context_bundles = None
        if plan.ctx is not None:
            # Fase K F6 (Etapa B): resolve *this* projection through the
            # layer context; the context prefetches the next one in the
            # background so banks are not all resident at once.
            plan.ctx.ensure(self, plan.uniq_list)
            # Fase L1: count every fallback to the legacy per-expert
            # resolution so runs can prove the fast path engaged. The ctx
            # records WHICH reason when the read came back unusable.
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
                hits = plan.ctx.hits.get(id(self), 0)
                misses = plan.ctx.misses.get(id(self), 0)
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
                    hits += 1
                else:
                    misses += 1
                    missing.append(eid)
        banked = None
        if missing:
            # ascending expert id = ascending file offset within the stacked
            # bank (row-major) — sorted reads keep the NVMe's locality
            missing.sort()
            if (
                _BANK_PROMOTE_ENV
                and len(missing) == len(plan.uniq_list)
                and hasattr(self.backing, "read_expert_into")
            ):
                # Etapa A1: every demanded expert is a miss, so the demand set
                # is one contiguous bank per key (two segments under the
                # HOBBIT split) — promote each once instead of building U
                # per-expert mx arrays and stacking them.
                banked = self._load_expert_bank_mx(missing)
            if banked is not None:
                rows = banked[1]
                dt_per = time.perf_counter() - t_res_start
                for eid, raw in zip(missing, rows):
                    bundles[eid] = raw
                    if cache_result:
                        self.cache.put(self.bundle_key(eid), raw)  # type: ignore[arg-type]
                    if p is not None:
                        p.add_load_source(
                            self.layer_idx, staged=False, dt=dt_per / len(missing)
                        )
            elif hasattr(self.backing, "load_expert_slice"):
                # Fase K F12: prefill-shaped calls may use the separate pool.
                io_pool = io_pool_for_positions(self, plan.positions)
                # Fase K F6 (C2 bank-first path): read all missing experts
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
                    # Legacy fallback: coalesce consecutive ids into
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
                                    # else: Fase K F7 bridge gap row — read
                                    # but never promoted/used, so dropped here
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
                dt_per = time.perf_counter() - t_res_start
                for eid, raw in zip(missing, raws):
                    if raw is None:
                        bundles[eid] = self._load_expert_bundle(eid)
                        continue
                    # Raw np bundles in the LRU by design: Metal only holds the
                    # per-call stack. Caching promoted mx copies here double-
                    # holds the same weights in wired memory when the budget
                    # is positive — measured: LRU(mx) + stacks summed 37GB
                    # Metal active on a 6GiB budget and the guard killed the
                    # second prefill outright (F2 post-mortem).
                    bundles[eid] = raw
                    if cache_result:
                        # Tier-suffixed key (bundle_key): under the HOBBIT
                        # split a hot (source-packing) and cold (tier-packing)
                        # bundle of the same expert must never alias in the
                        # LRU — the raw pread path staged unsuffixed keys and
                        # served a cold-packing bundle to a hot slot (mixed
                        # widths, mx.stack crash).
                        self.cache.put(self.bundle_key(eid), raw)  # type: ignore[arg-type]
                    if p is not None:
                        p.add_load_source(self.layer_idx, staged=False, dt=dt_per / len(missing))
            else:
                # dict-backed test doubles: sequential fallback
                for eid in missing:
                    bundles[eid] = self._load_expert_bundle(eid)
        t_load = time.perf_counter() - t_res_start

        if memtrace.enabled:
            memtrace.record(
                "linear.resolve",
                layer=self.layer_idx,
                proj=self.proj_name,
                uniq=len(plan.uniq_list),
                hits=hits,
                misses=misses,
                bank_bytes=self._bank_bytes_for(len(missing)),
                from_ctx=context_bundles is not None,
            )

        dt = self._slice_dtypes_lazy()
        ctx_banks = None
        if banked is None and plan.ctx is not None and _BANK_PROMOTE_CTX_ENV:
            # Etapa A1b: the layer context read this projection's demand set
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
        # Fase K F6: single-promoted per-tier banks. A segment whose ids
        # match EXACTLY the tier's demanded ids can feed gather_qmm directly
        # (no per-expert promote, no mx.stack) — bit-identical by
        # construction. Anything else falls back per tier.
        tier_single: dict[int, tuple] = {}
        if banked is not None:
            for ids_t, triple in banked[0]:
                tier = self._tier_of(ids_t[0]) if split else 0
                tier_single[tier] = triple
                has_b = has_b or triple[2] is not None
        elif ctx_banks is not None:
            for ids_t, triple in ctx_banks:
                tier = self._tier_of(ids_t[0]) if split else 0
                tier_single[tier] = triple
                has_b = has_b or triple[2] is not None
        # Per-tier bundle lists under the HOBBIT split: hot (source packing)
        # and cold (tier packing) widths differ (e.g. 8 vs 6 u32 cols per
        # row at gs 64), so a single stacked mini-bank is impossible — build
        # one per tier and combine the two gather_qmm outputs.
        tier_w = ([], [])  # hot, cold
        tier_s = ([], [])
        tier_b = ([], [])
        uniq: list[int] = []
        hot_idx: list[int] = []
        cold_idx: list[int] = []
        tier_demand: dict[int, list[int]] = {}
        if split:
            uniq = [int(e) for e in plan.uniq_list]
            hot_idx = [i for i, e in enumerate(uniq) if self._tier_of(e) == 0]
            cold_idx = [i for i, e in enumerate(uniq) if self._tier_of(e) == 1]
            tier_demand[0] = [uniq[i] for i in hot_idx]
            tier_demand[1] = [uniq[i] for i in cold_idx]
            # Fase L4A: per-tier byte attribution for the whole layer call.
            hot_req = [uniq[i] for i in hot_idx]
            cold_req = [uniq[i] for i in cold_idx]
            hot_bank_bytes = self._tier_bank_bytes_for(hot_req) if hot_req else 0
            cold_bank_bytes = self._tier_bank_bytes_for(cold_req) if cold_req else 0
            if memtrace.enabled:
                memtrace.record(
                    "dual_tier.enter",
                    layer=self.layer_idx,
                    proj=self.proj_name,
                    positions=len(plan.uniq_list),
                    hot_positions=len(hot_idx),
                    cold_positions=len(cold_idx),
                    hot_bank_bytes=hot_bank_bytes,
                    cold_bank_bytes=cold_bank_bytes,
                )
            for i, eid in enumerate(uniq):
                t = 0 if i in set(hot_idx) else 1  # hot_rank lookup
                if t in tier_single:
                    continue
                w, s, b = bundles[eid]
                tier_w[t].append(self._promote_np(w))
                tier_s[t].append(self._promote_np(s, dt[0]))
                if b is not None:
                    has_b = True
                    tier_b[t].append(self._promote_np(b, dt[1]))
        else:
            if 0 not in tier_single:
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

        if memtrace.enabled:
            # Sampled *before* the QMM runs: at this instant the U promoted
            # per-expert mx copies and the freshly stacked bank coexist,
            # which is the transient double-buffer that single-promotion
            # removes.
            memtrace.record(
                "linear.stack",
                layer=self.layer_idx,
                proj=self.proj_name,
                uniq=len(plan.uniq_list),
                bank_bytes=self._bank_bytes_for(len(plan.uniq_list)),
            )
        remapped = plan.remapped

        def _stack_tier(t: int, idxs: list[int]) -> tuple:
            """Legacy per-expert stack for one tier (no matching segment)."""
            ws, ss, bs_ = tier_w[t], tier_s[t], tier_b[t]
            if len(ws) == 1:
                w_b = mx.expand_dims(ws[0], 0)
                s_b = mx.expand_dims(ss[0], 0)
                b_b = mx.expand_dims(bs_[0], 0) if bs_ else None
            else:
                w_b = mx.stack(ws, axis=0)
                s_b = mx.stack(ss, axis=0)
                b_b = mx.stack(bs_, axis=0) if bs_ else None
            return w_b, s_b, b_b

        # HOBBIT dual-tier assembly (Fase I6): one mini-bank per tier and a
        # masked add — positions are mutually exclusive (each position
        # consumes exactly one expert), so the two gather_qmm outputs
        # partition the positions and zeros fill the rest.
        if split:
            flat_np = np.asarray(plan.flat_np).reshape(-1)
            out = None
            tier_order = [(0, hot_idx), (1, cold_idx)]
            if _DUAL_TIER_ORDER == "small-first":
                # Fase L4B B3: smaller tier first — reduces the coexistence
                # of the two banks in the lazy graph before the larger
                # tier's build; bit-exact (elementwise commutative add).
                tier_order.sort(key=lambda tv: len(tv[1]))
            for t, idxs in tier_order:
                if not idxs:
                    continue
                if t in tier_single:
                    w_b, s_b, b_b = tier_single[t]
                else:
                    w_b, s_b, b_b = _stack_tier(t, idxs)
                bits_ = self.bits if t == 0 else self._cold_bits
                gs_ = self.group_size if t == 0 else self._cold_gs
                if memtrace.enabled:
                    # Fase L4A: one bank_ready event per tier, so the trace
                    # can attribute the peak to hot or cold residency.
                    memtrace.record(
                        "dual_tier.%s.bank_ready" % ("hot" if t == 0 else "cold"),
                        layer=self.layer_idx,
                        proj=self.proj_name,
                        tier=t,
                        experts=len(idxs),
                        bank_bytes=self._tier_bank_bytes_for(idxs),
                    )
                # expert-id -> rank within THIS tier's bank (flat ids here,
                # not compact uniq ranks); -1 where the other tier owns it.
                tier_map = np.full((self.num_experts,), -1, dtype=np.int32)
                for rank, i in enumerate(idxs):
                    tier_map[uniq[i]] = rank
                tier_remapped_np = tier_map[flat_np].reshape(plan.indices_shape)
                if memtrace.enabled:
                    memtrace.record(
                        "dual_tier.%s.qmm_submitted" % ("hot" if t == 0 else "cold"),
                        layer=self.layer_idx,
                        proj=self.proj_name,
                        tier=t,
                        positions=int((tier_remapped_np >= 0).sum()),
                    )
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
                if memtrace.enabled:
                    memtrace.record(
                        "dual_tier.mask_ready",
                        layer=self.layer_idx,
                        proj=self.proj_name,
                        tier=t,
                    )
                tier_out = tier_out * keep
                if memtrace.enabled:
                    memtrace.record(
                        "dual_tier.add_submitted",
                        layer=self.layer_idx,
                        proj=self.proj_name,
                        tier=t,
                        first_add=(out is None),
                    )
                out = tier_out if out is None else out + tier_out
            if memtrace.enabled:
                memtrace.record(
                    "dual_tier.layer_exit",
                    layer=self.layer_idx,
                    proj=self.proj_name,
                    positions=len(plan.uniq_list),
                    hot_positions=len(hot_idx),
                    cold_positions=len(cold_idx),
                    hot_bank_bytes=hot_bank_bytes,
                    cold_bank_bytes=cold_bank_bytes,
                )
            if out is None:
                # Degenerate: every unique expert hot (hot bank == full uniq
                # order) — identical to the uniform path.
                if 0 in tier_single:
                    w_b, s_b, b_b = tier_single[0]
                else:
                    w_b, s_b, b_b = _stack_tier(0, hot_idx)
                out = mx.gather_qmm(
                    x, w_b, s_b, b_b, rhs_indices=plan.remapped,
                    transpose=True, group_size=self.group_size, bits=self.bits,
                    mode=self.mode, sorted_indices=sorted_indices,
                )
            if self._bias is not None and self._has_bias:
                b_mini = mx.take(self._bias, plan.uniq_mx, axis=0)
                out = out + mx.expand_dims(b_mini[plan.remapped], -2)
            t4 = time.perf_counter()
            p.record_observed(self.layer_idx, plan.uniq_list)
            p.add(
                self.layer_idx,
                gate=plan.gate_s if built else 0.0,
                unique=plan.unique_s if built else 0.0,
                load=t_load,
                stack=t4 - t2 - t_load,
                hits=hits,
                misses=misses,
                experts=len(plan.uniq_list),
                positions=plan.positions,
            )
            # O2: remember this layer's routing for next token's speculation
            if _spec_state is not None:
                _spec_state.record_prev(self.layer_idx, plan.uniq_list)
            return out

        if 0 in tier_single:
            w_bank, s_bank, b_bank = tier_single[0]
        else:
            if len(mini_w) == 1:
                w_bank = mx.expand_dims(mini_w[0], 0)
                s_bank = mx.expand_dims(mini_s[0], 0)
                b_bank = mx.expand_dims(mini_b[0], 0) if has_b and mini_b else None
            else:
                w_bank = mx.stack(mini_w, axis=0)
                s_bank = mx.stack(mini_s, axis=0)
                b_bank = mx.stack(mini_b, axis=0) if has_b and mini_b else None
        remapped = plan.remapped
        out = mx.gather_qmm(
            x,
            w_bank,
            s_bank,
            b_bank,
            rhs_indices=remapped,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            sorted_indices=sorted_indices,
        )
        if self._bias is not None and self._has_bias:
            b_mini = mx.take(self._bias, plan.uniq_mx, axis=0)
            out = out + mx.expand_dims(b_mini[remapped], -2)
        t4 = time.perf_counter()
        # O2: remember this layer's routing for next token's speculation
        if _spec_state is not None:
            _spec_state.record_prev(self.layer_idx, plan.uniq_list)
        p.record_observed(self.layer_idx, plan.uniq_list)
        p.add(
            self.layer_idx,
            gate=plan.gate_s if built else 0.0,
            unique=plan.unique_s if built else 0.0,
            load=t_load,
            stack=t4 - t2 - t_load,
            hits=hits,
            misses=misses,
            experts=len(plan.uniq_list),
            positions=plan.positions,
        )
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
        self.fused_gate_up = fused_gate_up
        self.inverse_scatter = inverse_scatter
        self.quantized = quantized
        # Original SwitchGLU activation (e.g. DeepSeek V4's LimitedSwiGLU with
        # swiglu_limit / fp32). None falls back to the stock mlx-lm swiglu.
        # Underscore attr keeps it out of the nn.Module parameter tree.
        self._activation = activation

        # We will be populated by the converter after construction
        # Placeholder attributes for introspection
        self._input_dims = input_dims
        self._hidden_dims = hidden_dims
        self._num_experts = num_experts
        self._backing = backing
        self._cache = cache
        self._group_size = group_size
        self._bits = bits
        self._mode = mode

        # Create streaming linears lazily; actual keys set by converter
        self._initialized = False

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

    def _ensure_initialized(self, template_glu: Any) -> None:
        if self._initialized:
            return
        # Called once after we have a template to copy projection config
        self._initialized = True

    def _apply_activation(self, x_up: Any, x_gate: Any) -> Any:
        act = getattr(self, "_activation", None)
        if act is not None:
            # Same call order as the original SwitchGLU: activation(up, gate)
            return act(x_up, x_gate)
        from mlx_lm.models.activations import swiglu

        return swiglu(x_gate, x_up)

    def __call__(self, x, indices, scores=None, weighted_sum: bool = False):
        # Mirror SwitchGLU.__call__ but route through streaming linears
        p = getattr(self, "_cache", None).profile if hasattr(self, "_cache") else None
        t_wall0 = time.perf_counter() if (p is not None and p.enabled) else None
        # Determine fused vs split by presence of gate_up_proj
        has_fused = hasattr(self, "gate_up_proj")
        # Opt-in warm/pin hook (warmer.py): fires previous-token reads for
        # the next layer before this layer's demand loads; decode-only.
        hook = getattr(self, "_warm_pins", None)
        if hook is not None:
            hook.on_layer_start(self.layer_idx, int(indices.size))
        x_exp = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x_exp, idx, inv_order = _gather_sort(x_exp, indices, inverse_scatter=self.inverse_scatter)

        # One shared routing plan for the whole layer: the first linear
        # invoked builds it (single mx.eval + unique + remap), the rest reuse.
        plan = _RemapPlan()
        # P3: the RA advisor's prev-token prediction for THIS layer is the
        # previous token's routing (recorded in spec_state by the last
        # token). Register it as 'predicted' so profile.report() measures
        # RA precision/recall with the same machinery as PILOT (observed
        # is recorded by every linear path below).
        try:
            _spec = getattr(getattr(self, "_cache", None), "spec_state", None)
            _prev = getattr(_spec, "prev_uniq_by_layer", None) if _spec is not None else None
            if p is not None and p.enabled and _prev:
                _pp = _prev.get(self.layer_idx)
                if _pp:
                    p.record_predicted(self.layer_idx, list(_pp))
        except Exception:
            pass
        if self.quantized and _LAYER_BARRIER_ENV:
            projections = (
                [self.gate_up_proj, self.down_proj]
                if has_fused
                else [self.up_proj, self.gate_proj, self.down_proj]
            )
            if all(hasattr(proj, '_load_expert_bank_np') for proj in projections):
                # Fase 1 hybrid: decode-shaped calls (<64 routed rows)
                # read all projections at once (union — the measured-best
                # decode shape); prefill keeps rolling so all banks are
                # never resident simultaneously.
                ctx_mode = _layer_ctx_mode(
                    int(indices.size),
                    quantized=self.quantized,
                    barrier=_LAYER_BARRIER_ENV,
                )
                plan.ctx = _LayerLoadContext(
                    projections, self._cache, mode=ctx_mode
                )
            else:
                # Fase L1: no bank reader on every projection (dict-backed
                # test doubles / bf16 mixes) — no context to resolve through.
                self._cache._count_ctx_fallback("dict_backing")
        if memtrace.enabled:
            memtrace.record(
                'glu.enter', layer=self.layer_idx, positions=int(indices.size)
            )

        if has_fused:
            x_gate_up = self.gate_up_proj(x_exp, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
            x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
            x_act = self._apply_activation(x_up, x_gate)
            x_out = self.down_proj(x_act, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
        else:
            x_up = self.up_proj(x_exp, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
            x_gate = self.gate_proj(x_exp, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]
            x_act = self._apply_activation(x_up, x_gate)
            x_out = self.down_proj(x_act, idx, sorted_indices=do_sort, plan=plan)  # type: ignore[attr-defined]

        if hook is not None:
            # Fase I6 hotness signal: per-TOKEN usage over the routing plan
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
            hook.on_layer_plan(self.layer_idx, plan.uniq_list, plan.positions, counts)
        if _TRACE_PATH is not None:
            _trace_row(self.layer_idx, plan.uniq_list, plan.positions)

        if (
            _PREFILL_DIAG_ENV
            and p is not None
            and p.enabled
            and int(indices.size) >= _PREFILL_DIAG_MIN_ROUTES
        ):
            # Force-eval the layer's graph (everything upstream is a lazy
            # dependency of x_out, so with a sync at every MoE GLU each eval
            # covers exactly one layer's segment: attention + dense + GLU
            # QMMs). CPU buckets measured inside the linears remain valid;
            # absolute wall inflates because the CPU/GPU overlap is gone.
            t_gpu0 = time.perf_counter()
            mx.eval(x_out)
            p.add_gpu(self.layer_idx, time.perf_counter() - t_gpu0)

        # Weighted-sum fast path: native ext via
        # omlx.custom_kernels.glm_moe_dsa.fast (falls back to mx.fast
        # inside the wrapper); plain scatter-unsort when unavailable.
        if weighted_sum and scores is not None and do_sort:
            try:
                from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast  # type: ignore

                if hasattr(glm_fast, "glm_moe_weighted_sum"):
                    return glm_fast.glm_moe_weighted_sum(x_out, inv_order, scores)
            except Exception:
                pass

        if do_sort:
            x_out = _scatter_unsort(x_out, inv_order, indices.shape)
        out = x_out.squeeze(-2)
        if t_wall0 is not None and p is not None:
            p.add_wall(self.layer_idx, time.perf_counter() - t_wall0)
        return out
