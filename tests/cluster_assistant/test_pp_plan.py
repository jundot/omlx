# SPDX-License-Identifier: Apache-2.0
"""Phase D: 3/4-node pipeline plans.

The solver is pure, so these pin the plan shape before a cluster is available:
contiguous stages, rank 0 on the tail, every stage inside its admission, and the
fewest Macs that fit.
"""

from __future__ import annotations

from omlx.cluster.assistant.election import NodeFacts
from omlx.cluster.assistant.policy import solve_cluster_plan
from omlx.cluster.planner import ModelLayout

GIB = 1024**3


def facts(node_id: str, admission_gib: float, *, capacity_gib: float | None = None) -> NodeFacts:
    return NodeFacts(
        node_id=node_id,
        capacity_gib=capacity_gib or admission_gib + 10,
        admission_p05_gib=admission_gib,
    )


def layout(*, layers: int = 60, layer_gib: float = 2.0, fixed_gib: float = 0.0, mtp_gib: float = 0.0) -> ModelLayout:
    return ModelLayout(
        source="/models/fake",
        fixed_weight_bytes=int(fixed_gib * GIB),
        layer_weight_bytes=(int(layer_gib * GIB),) * layers,
        supports_pipeline=True,
        mtp_weight_bytes=int(mtp_gib * GIB),
    )


def solve(nodes, model, **overrides):
    return solve_cluster_plan(
        nodes,
        model,
        target_context_tokens=8192,
        **overrides,
    )


def _assert_contiguous(selection, layer_count: int) -> None:
    stages = sorted(selection.election.selected.stages, key=lambda item: item.rank)
    assert [stage.rank for stage in stages] == list(range(len(stages)))
    # Rank 0 holds the tail; the highest rank holds layers from zero.
    assert stages[0].last_layer == layer_count
    by_layer = sorted(stages, key=lambda item: item.first_layer)
    assert by_layer[0].first_layer == 0
    assert by_layer[-1].last_layer == layer_count
    for before, after in zip(by_layer, by_layer[1:]):
        assert before.last_layer == after.first_layer
    assert all(stage.headroom_bytes >= 0 for stage in stages)


def test_three_nodes_produce_contiguous_stages_and_a_tail_on_rank_zero() -> None:
    nodes = [
        facts("m5", 60.0),
        facts("m2", 40.0),
        facts("m1", 35.0),
    ]
    selection = solve(nodes, layout(layers=120, layer_gib=1.0))

    assert len(selection.election.rank_order) == 3
    _assert_contiguous(selection, 120)
    selected = selection.election.selected
    assert selected is not None
    assert selected.stages[0].node_id == selection.election.coordinator_node
    assert selected.stages[0].last_layer == 120


def test_four_nodes_all_keep_every_stage_inside_its_admission() -> None:
    nodes = [
        facts("m5", 96.0),
        facts("m2", 54.0),
        facts("m1", 23.0),
        facts("m4", 21.0),
    ]
    selection = solve(nodes, layout(layers=80, layer_gib=1.0), node_selection="all")

    assert len(selection.election.rank_order) == 4
    _assert_contiguous(selection, 80)
    assert selection.excluded_nodes == ()
    for stage in selection.election.selected.stages:
        assert stage.resident_bytes <= stage.admission_bytes


def test_the_fewest_macs_win_and_the_other_two_are_named() -> None:
    nodes = [
        facts("m5", 96.0),
        facts("m2", 54.0),
        facts("m1", 23.0),
        facts("m4", 21.0),
    ]
    selection = solve(nodes, layout(layers=60, layer_gib=1.0))

    assert len(selection.election.rank_order) == 2
    assert {item.node_id for item in selection.excluded_nodes} == {"m1", "m4"}
    assert "macs_not_needed" in {item.code for item in selection.warnings}


def test_only_when_two_cannot_fit_does_the_solver_grow_to_three() -> None:
    nodes = [
        facts("m5", 60.0),
        facts("m2", 40.0),
        facts("m1", 35.0),
    ]
    selection = solve(nodes, layout(layers=120, layer_gib=1.0))

    assert len(selection.election.rank_order) == 3
    assert selection.size_refusals
    assert selection.size_refusals[0].node_count == 2


def test_the_mtp_head_is_charged_to_rank_zero() -> None:
    nodes = [facts("m5", 96.0), facts("m2", 54.0)]
    model = layout(layers=40, layer_gib=1.0, mtp_gib=4.0)

    without = solve(nodes, model)
    with_mtp = solve(nodes, model, mtp=True)

    assert with_mtp.election.plan_signature != without.election.plan_signature
    assert with_mtp.election.selected.rank0_extra_bytes == 4 * GIB
    assert with_mtp.election.selected.stages[0].headroom_bytes < (
        without.election.selected.stages[0].headroom_bytes
    )


def test_a_three_node_permutation_is_always_a_valid_plan() -> None:
    nodes = [
        facts("a", 50.0),
        facts("b", 45.0),
        facts("c", 40.0),
    ]
    selection = solve(nodes, layout(layers=90, layer_gib=1.0), node_selection="all")

    # Every considered rank order is a permutation of the selected set and
    # every stage is contiguous; the election never invents a rank.
    for candidate in selection.election.considered:
        assert sorted(candidate.rank_order) == ["a", "b", "c"]
        stages = sorted(candidate.stages, key=lambda item: item.rank)
        assert stages[0].last_layer == 90
        by_layer = sorted(stages, key=lambda item: item.first_layer)
        assert by_layer[0].first_layer == 0
        assert by_layer[-1].last_layer == 90
        for before, after in zip(by_layer, by_layer[1:]):
            assert before.last_layer == after.first_layer
