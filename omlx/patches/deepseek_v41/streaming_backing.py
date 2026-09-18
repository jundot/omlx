# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic dynamic budget over the V4.1 native expert adapter.

Split of responsibilities:

- Policy (model-agnostic): ``ExpertResidencyGovernor`` from
  ``expert_streaming.governor`` decides capacity from pressure (free RAM)
  and hunger (windowed decode stall). It only knows the ``cache``
  duck-type: ``capacity`` / ``resize`` / ``clear`` / ``set_layer_caps`` /
  ``stats``. No V4.1 knowledge, no kernels.
- Mechanics (model-specific): this backing translates those calls onto
  the partitioned per-layer ``_ExpertSlots`` storage (fixed mx arrays per
  layer, V4.1 packed layouts untouched).

Unit adaptation: V4.1 storage is partitioned per layer while the governor
reasons about one pool. The backing therefore presents capacity in
per-layer units — ``per_slot = bytes_per_expert * num_layers`` with
``num_layers=1`` at the governor — so every budget<->slots conversion is
exact and a global shrink can never starve one layer below its decode
working set. Targeting overrides (``set_layer_caps``) are per-layer
ceilings enforced lazily; ``capacity`` reports the uniform base.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..expert_streaming._env import env_bool, env_float
from ..expert_streaming.residency_adapter import PerLayerResidencyAdapter
from ..expert_streaming.slot_cache import DecodeVisitStats
from ..expert_streaming.staging import (
    PrevTokenPredictor,
    ewma,
    stage_headroom,
)

logger = logging.getLogger(__name__)

# Per-layer staging recall gate (parity with the generic path's
# stage_gate in expert_streaming.speculation): the prev-token predictor
# must have covered at least this share of observed demand recently
# (EWMA, decay 0.9) or stage_predicted is skipped — a layer whose
# routing diverged pays one bounded burst of staged reads and then
# shuts itself off. env_float's contract matches the old hand-rolled
# parse: an empty or malformed value keeps the default instead of
# holding the recall gate open ("" -> 0.0 would stage unconditionally).
_STAGED_MIN_RECALL = env_float("OMLX_V41_STAGE_MIN_RECALL", 0.3)
_STAGED_RECALL_DECAY = 0.9


def _verify_ra_enabled() -> bool:
    """Verify-fed cross-iteration readahead (megaplan F3) — advise only.

    Under DSpark essentially every trunk forward is a verify call, so
    the decode-phase prev-token predictor starves (the staged_hits=0
    bug). The verify predictor instead records each layer's routed
    UNION per verify round and advises the next layer's predicted set —
    F_RDADVISE only: a union is ~depth*top_k rows, far too large to
    materialize as staged payloads held across an iteration.
    OMLX_V41_VERIFY_RA=0 disables.
    """
    return env_bool("OMLX_V41_VERIFY_RA", True)


_VERIFY_MIN_RECALL = env_float("OMLX_V41_VERIFY_MIN_RECALL", 0.3)


def _verify_hunger_frac() -> float:
    """Share of verify traffic bridged into the hunger signal (F6).

    Verify misses stayed out of the governor deliberately: pre-F4 the
    verify working set (~depth*top_k) exceeded ``cap`` structurally, so
    verify stalls were hunger the budget could never satisfy. Scratch
    rows make the union servable now — but feeding verify 1:1 would
    still let a ~36-expert union dominate a 10-row decode budget. A
    fractional bridge (0.25 = every fourth verify visit counts as a
    decode visit, hit or miss) lets measurement pick the dose; the
    emitted visit keeps THIS visit's outcome so the stall ratio stays
    truthful. 0 = verify stays telemetry-only (default).
    OMLX_V41_VERIFY_HUNGER, 0..1.
    """
    return min(1.0, max(0.0, env_float("OMLX_V41_VERIFY_HUNGER", 0.0)))


@dataclass
class _V41CacheStats(DecodeVisitStats):
    """Cumulative decode-visit counters for governor windows.

    The shared contract class (``expert_streaming.slot_cache``) is the
    shape ``ExpertResidencyGovernor._window`` duck-reads — same fields the
    unified CacheStats exposes.

    Cadence note: V4.1 notes a visit per ``ensure()`` — one per decode
    CHUNK, and a B-row batched decode splits into ceil(B/step) chunks —
    while the generic cache notes one visit per layer-call. Governor
    windows therefore fill ~chunks-per-layer-call faster here; the stall
    signal stays a ratio so ``stall_target`` applies unchanged.

    Verify visits are counted separately (megaplan F0): real wall-time
    stalls, but deliberately excluded from the governor's hunger signal —
    under DSpark verify demand exceeds ``cap`` structurally, so feeding
    it would target growth the working set cannot satisfy.
    """

    verify_visits: int = 0
    verify_visits_missed: int = 0
    verify_misses_by_layer: dict = field(default_factory=dict)

    def note_verify_visit(self, layer_idx: int, missed: bool) -> None:
        self.verify_visits += 1
        if missed:
            self.verify_visits_missed += 1
            self.verify_misses_by_layer[layer_idx] = (
                self.verify_misses_by_layer.get(layer_idx, 0) + 1
            )


class V41StreamingBacking(PerLayerResidencyAdapter):
    """Governor-facing cache over V4.1's per-layer expert slots."""

    _GOVERNOR_LABEL = "V4.1"

    def __init__(
        self,
        plan,
        layers: list,
        *,
        dynamic: bool = True,
        max_budget_bytes: int | None = None,
        min_budget_bytes: int | None = None,
        stall_target: float | None = None,
        min_cap: int | None = None,
    ) -> None:
        self.plan = plan
        self.slots_of = {}
        for layer_idx, slots in layers:
            self.slots_of[int(layer_idx)] = slots
            slots.backing = self
            slots.layer = int(layer_idx)
        # Staging predictor: last decode-token routing per layer, the
        # MoE layer order used to find the "next" layer to stage, and the
        # per-layer prev-token recall EWMA that gates speculative reads.
        # prev_uniq/recall_ewma alias the shared predictor's tables so
        # direct inspection (and test writes) stays live.
        self._predictor = PrevTokenPredictor(
            decay=_STAGED_RECALL_DECAY, min_recall=_STAGED_MIN_RECALL
        )
        self.prev_uniq = self._predictor.prev_uniq_by_layer
        self.recall_ewma = self._predictor.recall_ewma
        self._order = sorted(self.slots_of)
        self.staged_skips = 0
        # Megaplan F0: skip causes split for diagnostics — a staged_hits=0
        # run needs to distinguish "no predictor yet", "recall gate" and
        # "headroom suppressed" without a debugger. staged_skips stays
        # the aggregate of recall+headroom skips (bench JSON compat).
        self.staged_skips_no_pred = 0
        self.staged_skips_recall = 0
        self.staged_skips_headroom = 0
        # F3 verify predictor: the routed UNION each layer saw during the
        # last completed verify round (verify_uniq) predicts its next
        # round — adjacent draft text routes similarly. verify_acc
        # accumulates the in-flight round's unions; a verify pass visits
        # layers in monotonically non-decreasing order (chunked ensures
        # repeat the same index), so an index DECREASE marks a new round
        # and promotes acc -> uniq. verify_recall EWMAs how much of each
        # new chunk the last round's union actually covered.
        self.verify_uniq: dict = {}
        self.verify_acc: dict = {}
        self._verify_last = -1
        self.verify_recall: dict = {}
        self.verify_ra_submits = 0
        self.verify_ra_no_pred = 0
        self.verify_ra_recall = 0
        self.verify_ra_headroom = 0
        # F6 fractional verify->hunger bridge accumulator (see
        # _verify_hunger_frac); crossing 1.0 emits one decode visit.
        self._verify_hunger_credit = 0.0
        # The adapter owns _lock/cache/overrides/stats/num_layers/
        # per_slot/base_cap/governor and the resize/clear/cap plumbing.
        super().__init__(
            self.slots_of,
            dynamic=dynamic,
            max_budget_bytes=max_budget_bytes,
            min_budget_bytes=min_budget_bytes,
            stall_target=stall_target,
            min_cap=min_cap,
        )

    # -- PerLayerResidencyAdapter hooks -------------------------------------
    def _iter_units(self):
        return self.slots_of.values()

    def _per_slot_bytes(self) -> int:
        # plan.full_bytes spans ALL layers while count is per-layer: derive
        # per-expert size from resident_bytes (capacity x layers) so the
        # governor's GiB budgets stay truthful. Actions are scale-invariant
        # (halve/double/+25%), but byte floors and log labels are not.
        per_expert = max(
            1,
            int(self.plan.resident_bytes)
            // max(1, int(self.plan.capacity) * max(1, len(self.slots_of))),
        )
        # Per-layer units (see module docstring).
        return per_expert * max(1, len(self.slots_of))

    def _initial_base_cap(self) -> int:
        return int(self.plan.capacity)

    def _floor_cap(self) -> int:
        # Never shrink a layer below one token's decode working set:
        # below n_activated_experts `ensure` raises "working set exceeds
        # resident capacity" mid-generation — a fixed floor smaller than
        # top_k breaks on top-k>8 models. The plan already guarantees
        # capacity >= n_activated.
        return max(1, int(getattr(self.plan, "n_activated", 8) or 8))

    def _apply_layer_caps(self) -> None:
        for layer_idx, slots in self.slots_of.items():
            eff = self._cap_for(layer_idx)
            # Lock order backing -> slots (same as clear); the cap write
            # and the compact must be one atomic step against a
            # concurrent ensure on this layer.
            with slots._slots_lock:
                slots.book.cap = eff
                self._compact(slots, eff)

    def _clear_units(self) -> None:
        for slots in self.slots_of.values():
            slots.clear_residency()

    def _make_stats(self) -> _V41CacheStats:
        return _V41CacheStats()

    # No ``streaming_guard_info``: the scheduler's prefill-bank transient
    # exists for the generic path's lazy mini-banks; V4.1 expert slots are
    # persistent pre-allocated buffers, so that term does not apply. The
    # scheduler reads it via getattr-with-default.
    def close(self) -> None:
        """Release the plan's shard readers and fetch pool.

        ``shutdown_expert_streaming`` reaches this on engine stop; the
        model's own ``close()`` reaches the same plan via
        ``_moe_offload_plan`` — ``plan.close()`` is idempotent, so both
        paths are safe and the shard fds cannot outlive the engine.
        """
        try:
            self.plan.close()
        except Exception:
            logger.debug("V4.1 plan close failed", exc_info=True)

    # -- stats -------------------------------------------------------------
    # ``note_visit`` comes from the adapter — the decode-visit counter +
    # governor tick contract is identical here.
    def note_verify_visit(self, layer_idx: int, missed: bool) -> None:
        """Count one verify-phase layer-call (telemetry only, F0).

        Deliberately does NOT tick the governor: under DSpark the verify
        working set (~depth*top_k) exceeds cap structurally, so verify
        misses would fake hunger the budget cannot satisfy. Real verify
        demand misses surface after F4 scratch rows make the set servable.
        """
        with self._lock:
            self.stats.note_verify_visit(layer_idx, missed)
        frac = _verify_hunger_frac()
        if frac <= 0.0:
            return
        with self._lock:
            self._verify_hunger_credit += frac
            if self._verify_hunger_credit < 1.0:
                return
            self._verify_hunger_credit -= 1.0
        # The bridge emitted a visit: join the decode stream with this
        # visit's outcome (misses feed hunger at the verify miss rate,
        # hits dilute it identically — the fraction scales BOTH).
        self.note_visit(layer_idx, missed)

    def note_verify_routing(self, layer_idx: int, experts) -> None:
        """Feed a verify chunk's routed set to the F3 predictor.

        Round detection: a verify pass visits its MoE layers in
        non-decreasing order, so a layer-index DECREASE means the round
        wrapped — the accumulated unions promote to the predictor and a
        fresh round begins. Recall measures the last round's union
        coverage of THIS chunk (the quantity the next-layer advisory
        relies on).
        """
        li = int(layer_idx)
        now = {int(e) for e in experts}
        if not now:
            return
        with self._lock:
            # A pass visits layers in non-decreasing order — repeated
            # indices are this layer's own chunks, a DECREASE is the
            # next round's first layer. Promote the accumulated unions.
            if li < self._verify_last:
                self.verify_uniq = self.verify_acc
                self.verify_acc = {}
            self._verify_last = li
            prev = self.verify_uniq.get(li)
            if prev:
                obs = len(prev & now) / len(now)
                self.verify_recall[li] = ewma(
                    self.verify_recall.get(li, 0.0),
                    obs,
                    _STAGED_RECALL_DECAY,
                )
            acc = self.verify_acc.get(li)
            if acc is None:
                acc = self.verify_acc[li] = set()
            acc.update(now)

    def _next_in_order(self, layer_idx, wrap=True):
        """The MoE layer index after *layer_idx* in ``_order``.

        ``wrap`` makes the last layer's "next" the first layer again —
        the cross-token/cross-round lead. Without it the last layer has
        no next. An unconvertible or unknown layer returns None —
        "don't advise".
        """
        if not self._order:
            return None
        try:
            pos = self._order.index(int(layer_idx))
        except (TypeError, ValueError):
            return None
        if pos + 1 >= len(self._order):
            return self._order[0] if wrap else None
        return self._order[pos + 1]

    def advise_verify_next(self, layer_idx: int) -> None:
        """Advise the next layer's predicted verify demand (F3).

        Wraps to layer 0 on purpose: the last layer's "next" is the NEXT
        verify round's first layer — the longest-lead case. Runs the
        same ``advise_demand`` filter as the exact-demand path (residents
        and staged rows are skipped), so a wrong prediction only costs
        page-cache bytes, never correctness or held memory.
        """
        if not _verify_ra_enabled():
            return
        nxt = self._next_in_order(layer_idx)
        if nxt is None:
            return
        with self._lock:
            pred = self.verify_uniq.get(nxt)
            recall = self.verify_recall.get(nxt, 0.0)
        if not pred:
            self.verify_ra_no_pred += 1
            return
        if recall < _VERIFY_MIN_RECALL:
            self.verify_ra_recall += 1
            return
        # F6: same headroom discipline as decode staging — the advisory
        # fetches real pages, so under floor-capacity or desperate-free
        # pressure it competes with THIS layer's demand reads. The
        # prediction is only early-fetch of true demand; skipping it
        # under pressure just reverts to demand-time reads.
        if not stage_headroom(self.governor):
            self.verify_ra_headroom += 1
            return
        try:
            if self.slots_of[nxt].advise_demand(pred):
                self.verify_ra_submits += 1
        except Exception:
            pass

    def note_routing(self, layer_idx: int, experts) -> None:
        """Record a decode token's routed set as this layer's predictor
        for the next token (same temporal-locality assumption the generic
        path's SpeculationState uses).

        Deliberate simplification vs the generic path: ``prev_uniq``
        keeps only last token's set — no (layer, expert) -> next-expert
        transition table. Each staged payload here is a full projection
        row, so a looser predictor would spend real reads; instead the
        observed prev-token recall EWMA below gates ``stage_next``.
        """
        self._predictor.record(layer_idx, experts)

    def stage_next(self, layer_idx: int) -> None:
        """Stage the NEXT MoE layer's predicted set.

        Called after a decode ensure — while layer ``layer_idx``'s MoE
        compute runs, the next layer's likely experts read on the plan's
        staging worker. Wraps to the first MoE layer so the last layer
        stages the next token's entry point. Failure is silent: staging
        is a hint, never on the demand path.
        """
        nxt = self._next_in_order(layer_idx)
        if nxt is None:
            # int(None)/int(non-numeric) escapes ValueError alone; an
            # unconvertible or unknown layer just means "don't stage".
            return
        pred = self._predictor.predict(nxt)
        if not pred:
            self.staged_skips_no_pred += 1
            return
        # Recall gate (generic stage_gate parity): a layer whose
        # prev-token prediction stopped covering demand does not earn
        # speculative reads — the EWMA shuts it off after a bounded burst.
        if not self._predictor.gate(nxt):
            self.staged_skips += 1
            self.staged_skips_recall += 1
            return
        if not stage_headroom(self.governor):
            self.staged_skips += 1
            self.staged_skips_headroom += 1
            return
        try:
            self.slots_of[nxt].stage_predicted(pred)
        except Exception:
            pass

    def advise_next_layer(self, layer_idx: int) -> None:
        """Whole-bank F_RDADVISE of the next MoE layer (megaplan F1b).

        Called at a saturated-coverage prefill call's entry: the next
        layer's demand is almost surely its whole bank, so advising it
        now overlaps that IO with this layer's compute+fetch. No wrap —
        the last layer's "next" would be the NEXT request's layer 0,
        whose demand is a different token's tiny set. Failure is silent:
        readahead is a hint, never the demand path.
        """
        nxt = self._next_in_order(layer_idx, wrap=False)
        if nxt is None:
            return
        try:
            self.slots_of[nxt].advise_bank()
        except Exception:
            pass

    # ``stage_headroom(governor)`` is the shared suppress check — at the
    # governor floor or inside the desperate-clear band, speculative
    # reads only compete with demand fetches.

    # -- storage -----------------------------------------------------------
    # Realloc/compact live on _ExpertSlots (layer-3 mechanics: packed
    # layouts, QuantizedProjection rebind); the backing only orchestrates
    # ceilings and counts evictions.
    def _compact(self, slots, keep: int) -> None:
        # compact() takes the layer's own lock (it also runs standalone in
        # the ensure path); eviction counts are per-layer.
        slots.compact(keep)

    def summary(self) -> dict:
        # The per-layer counters mutate under each slots._slots_lock, not
        # self._lock — summing under the backing lock can tear by a few
        # counts while an ensure is in flight. Accepted: this is a
        # reporting path, not accounting, and single-int reads are atomic.
        # hits/misses count demanded EXPERTS (SlotBookkeeping bumps once
        # per covered/missed id), not the generic cache's per-projection
        # puts — hit_rate denominators differ by ~3x vs the generic path.
        # The governor snapshot runs OUTSIDE the backing lock on purpose:
        # governor.summary() takes gov._lock while observe()/tick() hold
        # gov._lock across cache.resize() -> backing._lock. Taking the
        # backing lock first here would invert the mandated
        # governor->cache order and AB-BA deadlock against a mid-request
        # tick; the governor numbers don't need the backing lock anyway.
        gov = self.governor.summary() if self.governor else {}
        with self._lock:
            hits = sum(s.hits for s in self.slots_of.values())
            misses = sum(s.misses for s in self.slots_of.values())
            resident = sum(len(s.slot_of) for s in self.slots_of.values())
        total = hits + misses
        slots = list(self.slots_of.values())
        # Per-layer counters are a reporting contract — the key names
        # below are read by admin summary payloads and the bench JSON.
        counters = {
            name: sum(getattr(s, name, 0) for s in slots)
            for name in (
                "staged_hits",
                "staged_drops",
                "staged_failures",
                "staged_submits",
                "ra_calls",
                "ra_rows",
                "ra_bytes",
                "span_reads",
                "span_demand_rows",
                "span_phys_rows",
                "span_fallbacks",
            )
        }
        counters.update(
            {
                name: sum(getattr(s.book, name, 0) for s in slots)
                for name in ("verify_misses", "verify_evict_resident")
            }
        )
        stats = self.stats
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": (hits / total) if total else 0.0,
            "evictions": self.evictions,
            "resident": resident,
            "capacity_per_layer": self.base_cap,
            "layers": self.num_layers,
            **counters,
            "staged_skips": self.staged_skips,
            "staged_skips_no_pred": self.staged_skips_no_pred,
            "staged_skips_recall": self.staged_skips_recall,
            "staged_skips_headroom": self.staged_skips_headroom,
            "verify_visits": stats.verify_visits,
            "verify_visits_missed": stats.verify_visits_missed,
            "verify_ra_submits": self.verify_ra_submits,
            "verify_ra_no_pred": self.verify_ra_no_pred,
            "verify_ra_recall": self.verify_ra_recall,
            "verify_ra_headroom": self.verify_ra_headroom,
            "verify_recall": (
                sum(self.verify_recall.values()) / len(self.verify_recall)
                if self.verify_recall
                else 0.0
            ),
            "governor": gov,
        }
