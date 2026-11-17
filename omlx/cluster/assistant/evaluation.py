# SPDX-License-Identifier: Apache-2.0
"""Evaluate measured deployment constellations and pick the best.

The optimization loop (product decision 2026-11-13):
  rank (estimate from hardware) -> oMLX plans + starts -> MEASURE -> repeat over
  worth-testing constellations -> pick the best measured -> fine-tune.

This module is the "pick the best measured" step. It is pure: it takes the real
measurements of each constellation the loop deployed and selects by the user's
priority (quality / speed / context). Methodology ported from LocalInferenz
``console/planner.py`` (Pareto front over quality / critical stage / headroom /
measured rate; priority selection that PREFERS measured rows and never invents a
rate) — not its app or planner. oMLX's planner still owns every layer split;
this only compares finished, measured deployments.

A constellation with no measurement is never preferred over a measured one, and
an unmeasured rate is ``None`` — never zero, never an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .election import PlanCandidate

Priority = Literal["balanced", "speed", "ttft", "quality", "context"]


@dataclass(frozen=True)
class ConstellationResult:
    """One deployed-and-measured constellation (a rank order oMLX split+ran).

    ``decode_tok_per_s``/``prefill_tok_per_s``/``ttft_ms`` are the real cluster
    measurement (oMLX usage fields); any may be ``None`` when unreported.
    ``quality_score`` is a transparent heuristic (quantization/role), not a
    benchmark, until real eval scores exist (LocalInferenz planner.quality_score).
    """

    rank_order: tuple[str, ...]
    model_path: str
    context_tokens: int
    plan_signature: str
    decode_tok_per_s: float | None = None
    prefill_tok_per_s: float | None = None
    ttft_ms: float | None = None
    quality_score: float = 0.0
    feasible: bool = True
    # The running deployment this result measured (so the loop can stop losers).
    deployment_id: str = ""
    coordinator_node: str = ""
    operation_id: str = ""

    @property
    def node_count(self) -> int:
        return len(self.rank_order)

    @property
    def measured(self) -> bool:
        return self.decode_tok_per_s is not None and self.decode_tok_per_s > 0


def _dominates(left: ConstellationResult, right: ConstellationResult) -> bool:
    """Pareto dominance: at least as good everywhere, strictly better somewhere.

    Objectives: higher quality, higher measured decode rate, larger context,
    fewer nodes (less pipeline latency). An unmeasured rate never dominates a
    measured one (unknown is not fast).
    """

    def rate(row: ConstellationResult) -> float:
        return row.decode_tok_per_s if row.measured else -1.0

    ge = (
        left.quality_score >= right.quality_score
        and rate(left) >= rate(right)
        and left.context_tokens >= right.context_tokens
        and left.node_count <= right.node_count
    )
    gt = (
        left.quality_score > right.quality_score
        or rate(left) > rate(right)
        or left.context_tokens > right.context_tokens
        or left.node_count < right.node_count
    )
    return ge and gt


def pareto_front(
    results: list[ConstellationResult],
) -> list[ConstellationResult]:
    """The non-dominated feasible constellations."""

    feasible = [row for row in results if row.feasible]
    return [
        row
        for row in feasible
        if not any(_dominates(other, row) for other in feasible if other is not row)
    ]


def select_best(
    results: list[ConstellationResult], *, priority: Priority = "balanced"
) -> ConstellationResult | None:
    """Pick the best constellation for the user's priority.

    Measured constellations are always preferred over unmeasured ones. Among
    measured rows the priority decides; ties fall back to fewer nodes then more
    headroom-quality. Returns ``None`` when nothing is feasible.
    """

    feasible = [row for row in results if row.feasible]
    if not feasible:
        return None
    measured = [row for row in feasible if row.measured]
    pool = measured or feasible

    def speed_key(row: ConstellationResult) -> tuple:
        # Higher decode rate first; unmeasured sorts last.
        return (
            row.decode_tok_per_s if row.measured else -1.0,
            -row.node_count,
            row.quality_score,
        )

    if priority == "speed":
        return max(pool, key=speed_key)
    if priority == "ttft":
        with_ttft = [row for row in pool if row.ttft_ms is not None]
        if with_ttft:
            return min(with_ttft, key=lambda row: (row.ttft_ms, -row.node_count))
        return max(pool, key=speed_key)
    if priority == "quality":
        return max(
            pool,
            key=lambda row: (row.quality_score, *speed_key(row)),
        )
    if priority == "context":
        return max(
            pool,
            key=lambda row: (row.context_tokens, *speed_key(row)),
        )
    # balanced: Pareto front, then the best measured speed with fewest nodes.
    front = pareto_front(pool) or pool
    return max(front, key=speed_key)

def select_constellations_to_test(
    candidates: list[PlanCandidate], *, max_count: int = 3
) -> list[PlanCandidate]:
    """Pick up to ``max_count`` diverse, worth-testing constellations.

    The election already ranks candidates by its hardware estimate (best first).
    The search loop measures the best estimate first, then spends its remaining
    budget on DIVERSITY — a different node count is the most informative next
    test (the fewer-nodes vs more-context tradeoff), then a different coordinator,
    then a different rank order. Ties keep the election's ranking. This is the
    "rank beforehand, then measure the worth-testing constellations" step; oMLX's
    planner still owns each candidate's layer split.
    """

    feasible = [c for c in candidates if c.feasible and c.rank_order]
    if not feasible or max_count <= 0:
        return []
    selected: list[PlanCandidate] = [feasible[0]]
    rank_index = {id(c): i for i, c in enumerate(feasible)}

    def diversity(candidate: PlanCandidate) -> tuple[int, int, int]:
        counts = {len(s.rank_order) for s in selected}
        coordinators = {s.rank_order[0] for s in selected}
        orders = {s.rank_order for s in selected}
        return (
            0 if len(candidate.rank_order) in counts else 1,
            0 if candidate.rank_order[0] in coordinators else 1,
            0 if candidate.rank_order in orders else 1,
        )

    while len(selected) < max_count and len(selected) < len(feasible):
        remaining = [c for c in feasible if c not in selected]
        # Maximize diversity (new node count > new coordinator > new order),
        # tie-break by the election's original ranking (lower index first).
        best = max(
            remaining,
            key=lambda c: (diversity(c), -rank_index[id(c)]),
        )
        selected.append(best)
    return selected
