# SPDX-License-Identifier: Apache-2.0
"""The automatic constellation search loop.

Product decision 2026-11-13: fully automatic, up to 3 constellations. The loop:

  1. select diverse, worth-testing constellations from the election's estimate
     (``evaluation.select_constellations_to_test``);
  2. for each, the caller's ``evaluate`` seam applies + starts + MEASURES it
     (oMLX's planner owns the split; the existing apply/start/observe seams do
     the work) and returns a ``ConstellationResult``;
  3. pick the best MEASURED constellation (``evaluation.select_best``);
  4. stop the losing deployments so only the winner keeps serving.

This module owns the loop logic only. The hardware driving (apply/start/observe
and stop) is injected as seams, so the loop is unit-testable off-cluster and the
route layer wires the real seams. Constellations are evaluated one at a time,
respecting the cluster-wide operation lock; a candidate that fails to deploy is
recorded as infeasible/unmeasured and never preferred over a measured one.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .election import PlanCandidate
from .evaluation import (
    ConstellationResult,
    Priority,
    select_best,
    select_constellations_to_test,
)

# Apply + start + measure one candidate; returns its measured result. May be sync
# (tests) or async (the real apply/start/observe seams are async). Raising is
# treated as a failed deployment (recorded infeasible), never crashes the loop.
EvaluateSeam = Callable[[PlanCandidate], ConstellationResult | Awaitable[ConstellationResult]]
# Stop a losing deployment (best-effort); sync or async.
StopSeam = Callable[[ConstellationResult], None | Awaitable[None]]


@dataclass
class SearchOutcome:
    best: ConstellationResult | None
    results: list[ConstellationResult] = field(default_factory=list)
    tested: list[PlanCandidate] = field(default_factory=list)
    stopped: list[ConstellationResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


async def run_search(
    candidates: list[PlanCandidate],
    *,
    evaluate: EvaluateSeam,
    stop: StopSeam | None = None,
    max_count: int = 3,
    priority: Priority = "balanced",
) -> SearchOutcome:
    """Measure up to ``max_count`` constellations and keep the best measured.

    Each candidate is evaluated sequentially (the cluster-wide lock allows one
    apply/start at a time). A candidate whose evaluation raises is recorded as
    an infeasible result with the error, and the loop continues. After measuring,
    the best measured constellation is selected and the other measured
    deployments are stopped (best-effort; a stop failure is recorded, not fatal).
    The evaluate/stop seams may be sync or async (the real apply/start/observe
    seams are async).
    """

    selected = select_constellations_to_test(candidates, max_count=max_count)
    outcome = SearchOutcome(best=None, tested=selected)
    for candidate in selected:
        try:
            result = evaluate(candidate)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001 - a failed deploy never kills the loop
            outcome.errors.append(f"{candidate.plan_hash or candidate.rank_order}: {exc}")
            result = ConstellationResult(
                rank_order=candidate.rank_order,
                model_path="",
                context_tokens=candidate.context_tokens,
                plan_signature=candidate.plan_hash,
                feasible=False,
            )
        outcome.results.append(result)

    outcome.best = select_best(outcome.results, priority=priority)
    if stop is not None and outcome.best is not None:
        for result in outcome.results:
            if result is outcome.best or not result.feasible:
                continue
            try:
                stopped = stop(result)
                if inspect.isawaitable(stopped):
                    await stopped
                outcome.stopped.append(result)
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                outcome.errors.append(f"stop {result.plan_signature}: {exc}")
    return outcome