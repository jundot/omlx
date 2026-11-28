# SPDX-License-Identifier: Apache-2.0
"""The automatic search loop (fake-driven): evaluate, pick best, stop losers.

The hardware driving is injected (evaluate/stop seams), so the loop logic is
tested off-cluster (AGENTS.md). oMLX's planner/apply/start are not exercised
here; the seams stand in for them.
"""

from __future__ import annotations

import asyncio

from omlx.cluster.assistant.election import PlanCandidate
from omlx.cluster.assistant.evaluation import ConstellationResult
from omlx.cluster.assistant.search import run_search


def _run_search(*args, **kwargs):
    return asyncio.run(run_search(*args, **kwargs))


def _candidate(order: tuple[str, ...]) -> PlanCandidate:
    return PlanCandidate(
        rank_order=order, stages=(), feasible=True, reason="estimate",
        plan_hash=order[0] * 16, context_tokens=32768,
    )


def _result(candidate: PlanCandidate, decode: float | None) -> ConstellationResult:
    return ConstellationResult(
        rank_order=candidate.rank_order,
        model_path="/models/m",
        context_tokens=candidate.context_tokens,
        plan_signature=candidate.plan_hash,
        decode_tok_per_s=decode,
        feasible=True,
    )


def test_evaluates_up_to_the_budget_and_picks_the_best_measured() -> None:
    candidates = [_candidate(("a", "b")), _candidate(("a", "b", "c")),
                  _candidate(("b", "a")), _candidate(("c", "a"))]
    rates = {("a", "b"): 40.0, ("a", "b", "c"): 70.0, ("b", "a"): 55.0}
    evaluated = []

    def evaluate(candidate):
        evaluated.append(candidate.rank_order)
        return _result(candidate, rates.get(candidate.rank_order, 30.0))

    outcome = _run_search(candidates, evaluate=evaluate, max_count=3, priority="speed")
    assert len(evaluated) == 3
    assert outcome.best.decode_tok_per_s == 70.0
    assert outcome.best.rank_order == ("a", "b", "c")


def test_stops_the_losers_but_not_the_winner() -> None:
    candidates = [_candidate(("a", "b")), _candidate(("a", "b", "c"))]
    rates = {("a", "b"): 40.0, ("a", "b", "c"): 70.0}
    stopped = []

    outcome = _run_search(
        candidates,
        evaluate=lambda c: _result(c, rates[c.rank_order]),
        stop=lambda r: stopped.append(r.rank_order),
        max_count=2,
        priority="speed",
    )
    assert outcome.best.rank_order == ("a", "b", "c")
    assert stopped == [("a", "b")]  # the loser stopped, the winner kept
    assert len(outcome.stopped) == 1


def test_a_failed_deployment_is_recorded_and_the_loop_continues() -> None:
    candidates = [_candidate(("a", "b")), _candidate(("a", "b", "c"))]

    def evaluate(candidate):
        if candidate.rank_order == ("a", "b"):
            raise RuntimeError("staging failed")
        return _result(candidate, 70.0)

    outcome = _run_search(candidates, evaluate=evaluate, max_count=2)
    assert len(outcome.results) == 2
    assert outcome.results[0].feasible is False
    assert outcome.best.rank_order == ("a", "b", "c")
    assert any("staging failed" in e for e in outcome.errors)


def test_no_feasible_candidate_yields_no_best() -> None:
    candidates = [_candidate(("a", "b"))]

    def evaluate(candidate):
        raise RuntimeError("cannot deploy")

    outcome = _run_search(candidates, evaluate=evaluate, max_count=3)
    assert outcome.best is None
    assert outcome.results[0].feasible is False


def test_stop_failure_is_recorded_not_fatal() -> None:
    candidates = [_candidate(("a", "b")), _candidate(("a", "b", "c"))]
    rates = {("a", "b"): 40.0, ("a", "b", "c"): 70.0}

    def stop(result):
        raise RuntimeError("coordinator unreachable")

    outcome = _run_search(
        candidates,
        evaluate=lambda c: _result(c, rates[c.rank_order]),
        stop=stop,
        max_count=2,
        priority="speed",
    )
    assert outcome.best.rank_order == ("a", "b", "c")
    assert any("stop" in e for e in outcome.errors)


def test_unmeasured_loser_is_not_stopped_as_a_deployment() -> None:
    # An infeasible result has no running deployment; stop is skipped for it.
    candidates = [_candidate(("a", "b")), _candidate(("a", "b", "c"))]
    stopped = []

    def evaluate(candidate):
        if candidate.rank_order == ("a", "b"):
            return ConstellationResult(
                rank_order=candidate.rank_order, model_path="/models/m",
                context_tokens=32768, plan_signature=candidate.plan_hash,
                feasible=False,
            )
        return _result(candidate, 70.0)

    outcome = _run_search(
        candidates, evaluate=evaluate,
        stop=lambda r: stopped.append(r.rank_order), max_count=2,
    )
    assert stopped == []  # the infeasible one is not a running deployment
    assert outcome.best.rank_order == ("a", "b", "c")

def test_async_evaluate_seam_is_awaited() -> None:
    candidates = [_candidate(("a", "b")), _candidate(("a", "b", "c"))]
    rates = {("a", "b"): 40.0, ("a", "b", "c"): 70.0}

    async def evaluate(candidate):
        await asyncio.sleep(0)
        return _result(candidate, rates[candidate.rank_order])

    async def stop(result):
        await asyncio.sleep(0)

    outcome = _run_search(
        candidates, evaluate=evaluate, stop=stop, max_count=2, priority="speed"
    )
    assert outcome.best.rank_order == ("a", "b", "c")
    assert len(outcome.stopped) == 1
