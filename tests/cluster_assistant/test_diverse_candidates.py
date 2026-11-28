# SPDX-License-Identifier: Apache-2.0
"""The diverse-candidate generator for the measurement search loop.

``diverse_candidates`` exposes the best feasible plan per cluster size (2..4
Macs) so the search can measure the fewer-nodes vs more-context tradeoff. These
tests are fake-driven (AGENTS.md): no Mac, model, GPU or SSH — only measured
numbers, because the generator is a pure function over the same election and
solver machinery. The helpers mirror ``test_policy.py``'s ``facts``/``layout``
(local copies, per the conftest convention of not importing test modules).

The regression proof that ``solve_cluster_plan`` and ``elect`` are unchanged is
the existing ``test_policy.py`` + ``test_election.py`` suites, run unchanged.
"""

from __future__ import annotations

import pytest

from omlx.cluster.assistant.election import (
    ComputeProfile,
    NodeFacts,
    NoFeasiblePlanError,
    elect,
    rank_candidates,
)
from omlx.cluster.assistant.policy import diverse_candidates, solve_cluster_plan
from omlx.cluster.planner import ModelLayout

GIB = 1024**3


def facts(
    node_id: str,
    *,
    capacity_gib: float = 48.0,
    admission_gib: float = 20.5,
    p50_gib: float | None = None,
    p95_gib: float | None = None,
    tier: str = "balanced",
    host: str = "",
    decode_gib_per_s: float | None = None,
) -> NodeFacts:
    # The compute mapping mirrors test_election.py: a relative speed signal is
    # scaled into the plausible memory-bandwidth window (E4=B) so the election
    # ranks by critical stage rather than treating it as unknown compute.
    compute = None
    if decode_gib_per_s is not None:
        compute = ComputeProfile(
            decode_weight_bytes_per_second=decode_gib_per_s * 100 * GIB,
            prefill_weight_bytes_per_second=decode_gib_per_s * 2 * 100 * GIB,
        )
    return NodeFacts(
        node_id=node_id,
        capacity_gib=capacity_gib,
        admission_p05_gib=admission_gib,
        admission_p50_gib=p50_gib,
        admission_p95_gib=p95_gib,
        memory_guard_tier=tier,
        endpoint_host=host,
        compute=compute,
    )


def layout(
    *,
    layers: int = 20,
    layer_gib: float = 2.0,
    fixed_gib: float = 1.0,
    table_gib: float = 0.0,
    table_kind: str = "",
    kv_bytes_per_token_per_layer: int = 0,
    max_context_tokens: int = 0,
    path: str = "/models/fake",
    supports_pipeline: bool = True,
) -> ModelLayout:
    return ModelLayout(
        source=path,
        fixed_weight_bytes=int(fixed_gib * GIB),
        layer_weight_bytes=(int(layer_gib * GIB),) * layers,
        supports_pipeline=supports_pipeline,
        tensor_parallel_ssd_layer_bytes=(
            (int(table_gib * GIB),) * layers if table_gib else ()
        ),
        table_offload_kind=table_kind,
        kv_bytes_per_token_per_layer=kv_bytes_per_token_per_layer,
        max_context_tokens=max_context_tokens,
    )


# A small model every subset of the four-Mac cluster can hold, so all three
# sizes (2, 3, 4) are feasible and the generator must expose one of each.
def four_macs() -> list[NodeFacts]:
    return [
        facts("m5", capacity_gib=128.0, admission_gib=96.0),
        facts("m2", capacity_gib=64.0, admission_gib=54.0),
        facts("m1", capacity_gib=32.0, admission_gib=23.0),
        facts("m4", capacity_gib=32.0, admission_gib=21.0),
    ]


def small_model() -> ModelLayout:
    return layout(layers=20, layer_gib=1.0, fixed_gib=0.0)  # 20 GiB


# ---------------------------------------------------------------------------
# One deployable candidate per fitting size
# ---------------------------------------------------------------------------


def test_returns_one_deployable_candidate_per_fitting_size() -> None:
    candidates = diverse_candidates(
        four_macs(), small_model(), target_context_tokens=8192
    )

    assert sorted(len(c.rank_order) for c in candidates) == [2, 3, 4]
    assert len({c.plan_hash for c in candidates}) == 3
    for candidate in candidates:
        assert candidate.feasible is True
        assert candidate.shard_plan is not None
        assert candidate.plan_hash
        # The carried plan is the one the hash names, so a caller can deploy it.
        assert candidate.shard_plan.plan_hash == candidate.plan_hash
        assert len(candidate.rank_order) == len(candidate.shard_plan.assignments)
        assert len(candidate.rank_order) >= 2


def test_the_two_node_candidate_matches_the_solver_on_a_pair() -> None:
    # With exactly two Macs the generator's single candidate must be the very
    # plan solve_cluster_plan elects: same machinery, same signature.
    nodes = [
        facts("a", capacity_gib=64.0, admission_gib=40.0),
        facts("b", capacity_gib=64.0, admission_gib=40.0),
    ]
    model = layout()

    selection = solve_cluster_plan(nodes, model, target_context_tokens=8192)
    candidates = diverse_candidates(nodes, model, target_context_tokens=8192)

    assert len(candidates) == 1
    assert candidates[0].plan_hash == selection.election.selected.plan_hash
    assert candidates[0].rank_order == selection.election.rank_order


def test_candidate_carries_the_policy_that_reproduces_its_signature() -> None:
    # Regression: the search loop's plan_apply must re-elect the identical
    # signature. diverse_candidates therefore carries the resolved resource
    # policy (the solver's), not None — otherwise plan_apply falls back to the
    # policy-less election and refuses with plan_mismatch.
    nodes = [
        facts("a", capacity_gib=64.0, admission_gib=40.0),
        facts("b", capacity_gib=64.0, admission_gib=40.0),
    ]
    model = layout()
    selection = solve_cluster_plan(nodes, model, target_context_tokens=8192)
    candidates = diverse_candidates(nodes, model, target_context_tokens=8192)

    assert candidates[0].resource_policy is not None
    assert candidates[0].resource_policy.to_dict() == selection.policy.to_dict()


# ---------------------------------------------------------------------------
# Best-first ranking and the budget cap
# ---------------------------------------------------------------------------


def test_candidates_are_ranked_best_first_by_the_election_scoring() -> None:
    nodes = four_macs()
    full = diverse_candidates(nodes, small_model(), target_context_tokens=8192)

    # The list is already in the election's own best-first order.
    assert list(rank_candidates(full, nodes)) == full


def test_measured_compute_ranks_by_critical_stage() -> None:
    # With trustworthy probes the frozen scoring ranks by the predicted critical
    # stage (ascending), then headroom. The generator must apply that same order,
    # not a node-count order: the best-first list is the election's own ranking.
    nodes = [
        facts("m5", capacity_gib=128.0, admission_gib=96.0, decode_gib_per_s=4.0),
        facts("m2", capacity_gib=64.0, admission_gib=54.0, decode_gib_per_s=2.0),
        facts("m1", capacity_gib=32.0, admission_gib=23.0, decode_gib_per_s=0.8),
        facts("m4", capacity_gib=32.0, admission_gib=21.0, decode_gib_per_s=0.6),
    ]
    full = diverse_candidates(nodes, small_model(), target_context_tokens=8192)

    assert len(full) == 3
    seconds = [c.critical_stage_seconds for c in full]
    assert all(value is not None for value in seconds)
    assert seconds == sorted(seconds)
    assert list(rank_candidates(full, nodes)) == full


def test_max_count_caps_the_list_to_the_best_prefix() -> None:
    nodes = four_macs()
    model = small_model()
    full = diverse_candidates(nodes, model, target_context_tokens=8192, max_count=3)

    assert len(full) == 3
    assert diverse_candidates(nodes, model, target_context_tokens=8192, max_count=2) == full[:2]
    assert diverse_candidates(nodes, model, target_context_tokens=8192, max_count=1) == full[:1]
    assert diverse_candidates(nodes, model, target_context_tokens=8192, max_count=0) == []
    assert diverse_candidates(nodes, model, target_context_tokens=8192, max_count=-1) == []


# ---------------------------------------------------------------------------
# Sizes that cannot fit are skipped; a single Mac is never a cluster
# ---------------------------------------------------------------------------


def test_only_the_fitting_size_is_returned() -> None:
    # The two tiny Macs cannot hold a stage under ANY lever: a stage needs the
    # replicated fixed weights (1 GiB) plus at least one layer (2 GiB) = 3 GiB,
    # and their whole capacity is 2 GiB — no tier/admission lever can raise a
    # budget above installed RAM. So every constellation that includes a tiny
    # Mac is infeasible; only the big pair fits, and only at size 2.
    nodes = [
        facts("a", capacity_gib=64.0, admission_gib=40.0),
        facts("b", capacity_gib=64.0, admission_gib=40.0),
        facts("tiny1", capacity_gib=2.0, admission_gib=1.5),
        facts("tiny2", capacity_gib=2.0, admission_gib=1.5),
    ]
    model = layout(layers=20, layer_gib=2.0, fixed_gib=1.0)  # 41 GiB

    candidates = diverse_candidates(nodes, model, target_context_tokens=8192)

    assert len(candidates) == 1
    assert set(candidates[0].rank_order) == {"a", "b"}
    assert candidates[0].shard_plan is not None


def test_an_infeasible_model_returns_no_candidates() -> None:
    nodes = [facts("a"), facts("b"), facts("c")]
    model = layout(layers=120, layer_gib=4.0)  # 480 GiB: hopeless everywhere

    assert diverse_candidates(nodes, model, target_context_tokens=8192) == []


def test_a_non_pipeline_model_returns_no_candidates() -> None:
    nodes = [facts("a"), facts("b")]

    assert (
        diverse_candidates(
            nodes, layout(supports_pipeline=False), target_context_tokens=8192
        )
        == []
    )


def test_a_single_mac_never_yields_a_cluster_candidate() -> None:
    # The model fits one Mac, but a one-Mac "plan" is not a deployable cluster.
    nodes = [facts("a", capacity_gib=64.0, admission_gib=50.0)]

    assert diverse_candidates(nodes, layout(), target_context_tokens=8192) == []


# ---------------------------------------------------------------------------
# The measured fabric is respected exactly as in the election
# ---------------------------------------------------------------------------


def test_incomplete_fabric_excludes_the_affected_orders() -> None:
    nodes = [
        facts("a", capacity_gib=64.0, admission_gib=40.0),
        facts("b", capacity_gib=64.0, admission_gib=40.0),
        facts("c", capacity_gib=64.0, admission_gib=40.0),
    ]
    model = layout(layers=20, layer_gib=1.0, fixed_gib=0.0)
    links = [["a", "b"]]  # only one measured link: no 3-Mac line is complete

    candidates = diverse_candidates(
        nodes, model, target_context_tokens=8192, fabric_links=links
    )

    # Only the linked pair survives; the unlinked pairs and the triple are out.
    assert len(candidates) == 1
    assert set(candidates[0].rank_order) == {"a", "b"}
    # Consistent with the election: the same fabric refuses the whole triple.
    with pytest.raises(NoFeasiblePlanError):
        elect(nodes, model, context_tokens=8192, fabric_links=links)


def test_a_complete_fabric_line_keeps_the_three_node_candidate() -> None:
    nodes = [
        facts("a", capacity_gib=64.0, admission_gib=40.0),
        facts("b", capacity_gib=64.0, admission_gib=40.0),
        facts("c", capacity_gib=64.0, admission_gib=40.0),
    ]
    model = layout(layers=20, layer_gib=1.0, fixed_gib=0.0)
    links = [["a", "b"], ["b", "c"]]  # a complete line a-b-c

    candidates = diverse_candidates(
        nodes, model, target_context_tokens=8192, fabric_links=links
    )

    assert sorted(len(c.rank_order) for c in candidates) == [2, 3]
    triple = next(c for c in candidates if len(c.rank_order) == 3)
    # The elected order walks the measured line (a-b-c or its reverse).
    assert triple.rank_order in {("a", "b", "c"), ("c", "b", "a")}


# ---------------------------------------------------------------------------
# Input contract mirrors the solver
# ---------------------------------------------------------------------------


def test_invalid_inputs_raise_like_the_solver() -> None:
    model = layout()
    with pytest.raises(ValueError):
        diverse_candidates([], model, target_context_tokens=8192)
    with pytest.raises(ValueError):
        diverse_candidates(
            [facts("a"), facts("a")], model, target_context_tokens=8192
        )
    with pytest.raises(ValueError):
        diverse_candidates([facts("a"), facts("b")], model, target_context_tokens=0)
    with pytest.raises(ValueError):
        diverse_candidates(
            [facts("a"), facts("b")], model, target_context_tokens=8192, microbatch_size=0
        )
