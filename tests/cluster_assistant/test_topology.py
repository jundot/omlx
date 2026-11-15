# SPDX-License-Identifier: Apache-2.0
"""The measured fabric as a pipeline line.

A pipeline is a chain, and the validation cluster is cabled as a line. These
tests pin the two rules the resolve iteration added: a longer plan is only
feasible when every consecutive rank pair has a measured link, and the explicit
2-Mac fallback consent changes the plan signature.
"""

from __future__ import annotations

import pytest

from omlx.cluster.assistant.election import NodeFacts, NoFeasiblePlanError, elect
from omlx.cluster.assistant.topology import missing_line_links
from omlx.cluster.planner import ModelLayout

GIB = 1024**3


def facts(node_id: str, admission_gib: float = 30.0) -> NodeFacts:
    return NodeFacts(
        node_id=node_id,
        capacity_gib=admission_gib + 10,
        admission_p05_gib=admission_gib,
    )


def layout(*, layers: int = 40) -> ModelLayout:
    return ModelLayout(
        source="/models/fake",
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=(1 * GIB,) * layers,
        supports_pipeline=True,
    )




def test_a_missing_neighbour_link_names_the_exact_pair() -> None:
    order = ("a", "b", "c")

    assert missing_line_links(order, [("a", "b")]) == (("b", "c"),)
    assert missing_line_links(order, [("a", "b"), ("b", "c")]) == ()
    assert missing_line_links(("only",), []) == ()
    assert missing_line_links(order, [("c", "b"), ("b", "a")]) == ()


def test_a_three_mac_plan_must_put_the_connected_mac_in_the_middle() -> None:
    nodes = [facts("a"), facts("b"), facts("c")]

    result = elect(
        nodes,
        layout(layers=60),
        context_tokens=8192,
        fabric_links=[("a", "b"), ("b", "c")],
    )

    assert result.rank_order in {("a", "b", "c"), ("c", "b", "a")}
    assert missing_line_links(
        result.rank_order, [("a", "b"), ("b", "c")]
    ) == ()


def test_a_three_mac_plan_with_only_one_link_is_refused() -> None:
    nodes = [facts("a"), facts("b"), facts("c")]

    with pytest.raises(NoFeasiblePlanError) as raised:
        elect(
            nodes,
            layout(layers=60),
            context_tokens=8192,
            fabric_links=[("a", "b")],
        )

    assert "no measured link" in str(raised.value)


def test_a_four_mac_line_selects_a_hamiltonian_order() -> None:
    nodes = [facts("a"), facts("b"), facts("c"), facts("d")]
    links = [("a", "b"), ("b", "c"), ("c", "d")]

    result = elect(nodes, layout(layers=80), context_tokens=8192, fabric_links=links)

    assert missing_line_links(result.rank_order, links) == ()
    assert set(result.rank_order) == {"a", "b", "c", "d"}


def test_a_two_mac_plan_needs_its_pair_link() -> None:
    nodes = [facts("a"), facts("b")]

    with pytest.raises(NoFeasiblePlanError):
        elect(nodes, layout(layers=40), context_tokens=8192, fabric_links=[])

    result = elect(
        nodes,
        layout(layers=40),
        context_tokens=8192,
        fabric_links=[("a", "b")],
    )
    assert result.rank_order == ("a", "b") or result.rank_order == ("b", "a")


def test_an_unmeasured_fabric_keeps_the_nominal_ring() -> None:
    nodes = [facts("a"), facts("b"), facts("c")]

    result = elect(nodes, layout(layers=60), context_tokens=8192, fabric_links=None)

    assert len(result.rank_order) == 3


def test_the_two_mac_consent_is_part_of_the_signature() -> None:
    nodes = [facts("a"), facts("b")]
    links = [("a", "b")]

    plain = elect(nodes, layout(layers=40), context_tokens=8192, fabric_links=links)
    fallback = elect(
        nodes,
        layout(layers=40),
        context_tokens=8192,
        fabric_links=links,
        two_mac_fallback=True,
    )

    assert plain.rank_order == fallback.rank_order
    assert plain.plan_signature != fallback.plan_signature
