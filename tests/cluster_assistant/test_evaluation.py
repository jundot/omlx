# SPDX-License-Identifier: Apache-2.0
"""Constellation evaluation (fake-driven): pick the best MEASURED deployment.

Ports the LocalInferenz planner's Pareto/priority methodology. The rule under
test: measured constellations beat unmeasured ones, an unmeasured rate is never
fast, and the user's priority (quality/speed/context/ttft) decides among the
measured. No real cluster (AGENTS.md).
"""

from __future__ import annotations

from omlx.cluster.assistant.evaluation import (
    ConstellationResult,
    pareto_front,
    select_best,
)


def _row(
    name: str,
    *,
    nodes: int = 2,
    decode: float | None = 50.0,
    prefill: float | None = 500.0,
    ttft: float | None = 100.0,
    quality: float = 4.0,
    context: int = 32768,
    feasible: bool = True,
) -> ConstellationResult:
    return ConstellationResult(
        rank_order=tuple(f"{name}-{i}" for i in range(nodes)),
        model_path=f"/models/{name}",
        context_tokens=context,
        plan_signature=name * 16,
        decode_tok_per_s=decode,
        prefill_tok_per_s=prefill,
        ttft_ms=ttft,
        quality_score=quality,
        feasible=feasible,
    )


def test_speed_picks_the_fastest_measured() -> None:
    fast = _row("fast", decode=60.0)
    slow = _row("slow", decode=30.0)
    assert select_best([slow, fast], priority="speed") is fast


def test_measured_beats_unmeasured() -> None:
    measured = _row("m", decode=20.0)
    unmeasured = _row("u", decode=None)
    # Even a slow measured row beats an unmeasured one (unknown is not fast).
    assert select_best([unmeasured, measured], priority="speed") is measured


def test_unmeasured_never_dominates_measured() -> None:
    measured = _row("m", decode=20.0, quality=4.0, context=32768, nodes=2)
    unmeasured = _row("u", decode=None, quality=5.0, context=131072, nodes=2)
    front = pareto_front([measured, unmeasured])
    assert measured in front
    # The unmeasured row cannot dominate the measured one on rate.
    assert not (front == [unmeasured])


def test_fewer_nodes_breaks_a_speed_tie() -> None:
    two = _row("two", nodes=2, decode=50.0)
    three = _row("three", nodes=3, decode=50.0)
    assert select_best([three, two], priority="speed") is two


def test_ttft_picks_the_lowest() -> None:
    quick = _row("quick", ttft=40.0, decode=40.0)
    slow = _row("slow", ttft=200.0, decode=60.0)
    assert select_best([slow, quick], priority="ttft") is quick


def test_quality_picks_the_highest_quality() -> None:
    big = _row("big", quality=5.0, decode=30.0)
    small = _row("small", quality=3.0, decode=60.0)
    assert select_best([small, big], priority="quality") is big


def test_context_picks_the_largest_context() -> None:
    long = _row("long", context=131072, decode=30.0)
    short = _row("short", context=8192, decode=60.0)
    assert select_best([short, long], priority="context") is long


def test_no_feasible_returns_none() -> None:
    assert select_best([_row("x", feasible=False)], priority="speed") is None
    assert select_best([], priority="balanced") is None


def test_balanced_returns_a_pareto_member() -> None:
    a = _row("a", decode=60.0, quality=4.0, context=32768, nodes=2)
    b = _row("b", decode=30.0, quality=5.0, context=131072, nodes=3)
    best = select_best([a, b], priority="balanced")
    assert best in pareto_front([a, b])


def test_pareto_excludes_dominated() -> None:
    better = _row("better", decode=60.0, quality=5.0, context=131072, nodes=2)
    worse = _row("worse", decode=30.0, quality=4.0, context=32768, nodes=3)
    front = pareto_front([better, worse])
    assert front == [better]

# --- Candidate selection: rank beforehand, test diverse constellations --------

from omlx.cluster.assistant.election import PlanCandidate
from omlx.cluster.assistant.evaluation import select_constellations_to_test


def _candidate(order: tuple[str, ...], *, feasible: bool = True) -> PlanCandidate:
    return PlanCandidate(
        rank_order=order,
        stages=(),
        feasible=feasible,
        reason="estimate",
        critical_stage_seconds=1.0,
    )


def test_best_estimate_is_always_tested_first() -> None:
    candidates = [
        _candidate(("a", "b")),
        _candidate(("a", "b", "c")),
        _candidate(("b", "a")),
    ]
    selected = select_constellations_to_test(candidates, max_count=3)
    assert selected[0] is candidates[0]


def test_diversity_prefers_a_different_node_count() -> None:
    candidates = [
        _candidate(("a", "b")),        # best estimate, 2 nodes
        _candidate(("b", "a")),        # 2 nodes, different order
        _candidate(("a", "b", "c")),   # 3 nodes
    ]
    selected = select_constellations_to_test(candidates, max_count=2)
    # The second slot goes to the 3-node constellation, not the 2-node reorder.
    assert len(selected[1].rank_order) == 3


def test_max_count_limits_the_budget() -> None:
    candidates = [_candidate(("a", "b")), _candidate(("a", "b", "c")),
                  _candidate(("a", "b", "c", "d")), _candidate(("b", "a"))]
    assert len(select_constellations_to_test(candidates, max_count=3)) == 3
    assert len(select_constellations_to_test(candidates, max_count=1)) == 1


def test_infeasible_candidates_are_excluded() -> None:
    candidates = [
        _candidate(("a", "b"), feasible=False),
        _candidate(("a", "b", "c")),
    ]
    selected = select_constellations_to_test(candidates, max_count=3)
    assert all(c.feasible for c in selected)
    assert selected[0].rank_order == ("a", "b", "c")


def test_empty_or_zero_budget_returns_nothing() -> None:
    assert select_constellations_to_test([], max_count=3) == []
    assert select_constellations_to_test([_candidate(("a", "b"))], max_count=0) == []


def test_fewer_candidates_than_budget_returns_all() -> None:
    candidates = [_candidate(("a", "b")), _candidate(("a", "b", "c"))]
    assert len(select_constellations_to_test(candidates, max_count=3)) == 2
