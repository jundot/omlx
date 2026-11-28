# SPDX-License-Identifier: Apache-2.0
"""Phase D: any Mac can originate; the origin never decides rank 0.

These are pure election/solver tests. The live any-to-any start is a later
hardware session; the decision that must hold before then is that the app origin
is not an input to the plan.
"""

from __future__ import annotations

from omlx.cluster.assistant.election import NodeFacts
from omlx.cluster.assistant.policy import solve_cluster_plan
from omlx.cluster.planner import ModelLayout

GIB = 1024**3


def facts(node_id: str, admission_gib: float) -> NodeFacts:
    return NodeFacts(
        node_id=node_id,
        capacity_gib=admission_gib + 10,
        admission_p05_gib=admission_gib,
    )


def layout(*, layers: int = 60, layer_gib: float = 1.0, fixed_gib: float = 0.0) -> ModelLayout:
    return ModelLayout(
        source="/models/fake",
        fixed_weight_bytes=int(fixed_gib * GIB),
        layer_weight_bytes=(int(layer_gib * GIB),) * layers,
        supports_pipeline=True,
    )


def solve(nodes, model, **overrides):
    return solve_cluster_plan(
        nodes,
        model,
        target_context_tokens=8192,
        **overrides,
    )


NODES = [
    facts("m5", 96.0),
    facts("m2", 54.0),
    facts("m1", 23.0),
    facts("m4", 21.0),
]


def test_every_origin_elects_the_same_plan() -> None:
    """The Mac the user opened the app on is recorded, never used to choose."""

    model = layout(layers=60)
    plans = {
        origin: solve(NODES, model, origin_node=origin) for origin in ("m5", "m2", "m1", "m4")
    }

    signatures = {selection.election.plan_signature for selection in plans.values()}
    coordinators = {selection.election.coordinator_node for selection in plans.values()}
    orders = {selection.election.rank_order for selection in plans.values()}

    assert len(signatures) == 1
    assert len(coordinators) == 1
    assert len(orders) == 1
    assert plans["m1"].election.origin_node == "m1"


def test_an_infeasible_origin_is_not_offered_as_a_fallback() -> None:
    """A small Mac whose tail cannot hold the replicated weights is not rank 0."""

    nodes = [facts("m5", 96.0), facts("m2", 54.0), facts("m1", 23.0)]
    model = layout(layers=30, fixed_gib=30.0)

    selection = solve(nodes, model, origin_node="m1")

    assert selection.election.coordinator_node != "m1"
    assert selection.feasible is True
    assert "m1" not in selection.election.rank_order


def test_every_selected_mac_is_a_real_measured_node() -> None:
    model = layout(layers=60)
    selection = solve(NODES, model, origin_node="m4")

    measured = {item.node_id for item in NODES}
    assert set(selection.election.rank_order) <= measured
    assert all(stage.node_id in measured for stage in selection.election.selected.stages)


def test_the_origin_is_never_added_only_because_it_is_the_app() -> None:
    """A 150 GiB-class model on four Macs uses the pair that covers it."""

    nodes = [
        facts("m5", 96.0),
        facts("m2", 54.0),
        facts("m1", 23.0),
        facts("m4", 21.0),
    ]
    # 125 GiB: only the M5 + M2 pair covers it (96 + 54 = 150), so the app on
    # m1 must still elect m5/m2 and name m1/m4 as excluded.
    model = layout(layers=125)

    selection = solve(nodes, model, origin_node="m1")

    assert set(selection.election.rank_order) == {"m5", "m2"}
    assert {item.node_id for item in selection.excluded_nodes} == {"m1", "m4"}
    assert selection.election.coordinator_node != "m1"


def test_a_three_mac_plan_can_originate_anywhere() -> None:
    nodes = [facts("m5", 60.0), facts("m2", 40.0), facts("m1", 35.0)]
    model = layout(layers=120)

    for origin in ("m5", "m2", "m1"):
        selection = solve(nodes, model, node_selection="all", origin_node=origin)
        assert len(selection.election.rank_order) == 3
        assert selection.election.origin_node == origin
        assert set(selection.election.rank_order) == {"m5", "m2", "m1"}
