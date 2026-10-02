# SPDX-License-Identifier: Apache-2.0
"""What a link is worth, and how honestly we say we know it."""

from __future__ import annotations

from types import SimpleNamespace

from omlx.cluster.link_bandwidth import (
    GB,
    LinkBandwidth,
    bandwidth_between,
    bandwidth_graph,
    link_bandwidth,
    slowest_link_in,
)


def _link(source, peer, kind="thunderbolt", gbps=None):
    return SimpleNamespace(
        kind=kind, source_node_id=source, peer_node_id=peer, link_speed_gbps=gbps
    )


def _profile(node_id, gb_per_s):
    return SimpleNamespace(
        node_id=node_id, collective_bandwidth_bytes_per_second=gb_per_s * GB
    )


# --- Evidence ranking -------------------------------------------------------


def test_a_measurement_is_preferred_to_the_cables_label():
    measured = link_bandwidth(
        _link("A", "B", gbps=120), measured_by_node={"A": 6.6 * GB, "B": 6.6 * GB}
    )
    assert measured.source == "measured"
    assert measured.gigabytes_per_second == 6.6


def test_a_link_is_only_measured_when_both_ends_were():
    """One endpoint's rate says nothing about the pair."""

    half = link_bandwidth(_link("A", "B", gbps=120), measured_by_node={"A": 6.6 * GB})
    assert half.source == "nominal"


def test_the_negotiated_speed_is_used_when_nothing_was_measured():
    nominal = link_bandwidth(_link("A", "B", gbps=80))
    assert nominal.source == "nominal"
    # A line rate is a ceiling, not a delivery.
    assert nominal.bytes_per_second < 80 * 1000**3 / 8


def test_an_unknown_link_falls_back_to_a_conservative_constant():
    assumed = link_bandwidth(_link("A", "B", kind="ethernet"))
    assert assumed.source == "assumed"
    assert not assumed.fast


def test_a_pair_is_bounded_by_its_slower_end():
    link = link_bandwidth(
        _link("A", "B"), measured_by_node={"A": 6.6 * GB, "B": 1.2 * GB}
    )
    assert link.gigabytes_per_second == 1.2


# --- Fast enough for an all-reduce -----------------------------------------


def test_bandwidth_alone_does_not_make_a_link_fast():
    """Same cable, RDMA 28.6 tok/s vs TCP ring 6.6 — latency, not throughput."""

    ethernet = LinkBandwidth("A", "B", 5.0 * GB, "nominal", "ethernet")
    assert not ethernet.fast

    thunderbolt = LinkBandwidth("A", "B", 5.0 * GB, "nominal", "thunderbolt")
    assert thunderbolt.fast


def test_a_measurement_can_promote_a_kind_we_would_not_have_trusted():
    """Measuring an all-reduce answers the question the kind only proxies."""

    measured = LinkBandwidth("A", "B", 5.0 * GB, "measured", "ethernet")
    assert measured.fast


def test_a_slow_measurement_demotes_a_kind_we_would_have_trusted():
    slow_tb = LinkBandwidth("A", "B", 0.5 * GB, "measured", "thunderbolt")
    assert not slow_tb.fast


# --- The graph --------------------------------------------------------------


def test_the_graph_is_undirected():
    graph = bandwidth_graph([_link("A", "B", gbps=120)])
    assert bandwidth_between(graph, "A", "B") == bandwidth_between(graph, "B", "A")
    assert bandwidth_between(graph, "A", "B") > 0


def test_an_absent_link_is_zero_not_an_error():
    assert bandwidth_between(bandwidth_graph([]), "A", "B") == 0.0


def test_better_evidence_wins_when_a_pair_appears_twice():
    """The same link seen from both ends must not downgrade to the worse view."""

    graph = bandwidth_graph(
        [_link("A", "B", gbps=120), _link("B", "A")],
        [_profile("A", 6.6), _profile("B", 6.6)],
    )
    assert len(graph) == 1
    assert next(iter(graph.values())).source == "measured"


def test_a_self_link_is_ignored():
    assert bandwidth_graph([_link("A", "A", gbps=120)]) == {}


# --- Group speed ------------------------------------------------------------


def test_a_group_runs_at_its_slowest_link():
    graph = bandwidth_graph(
        [_link("A", "B"), _link("B", "C"), _link("A", "C")],
        [_profile("A", 6.6), _profile("B", 6.6), _profile("C", 1.1)],
    )
    slowest = slowest_link_in(graph, ["A", "B", "C"])
    assert slowest.gigabytes_per_second == 1.1


def test_a_missing_link_is_not_reported_as_a_slow_one():
    """"Not connected" and "connected slowly" are different answers."""

    graph = bandwidth_graph([_link("A", "B")])
    assert slowest_link_in(graph, ["A", "B", "C"]) is None


def test_a_link_describes_its_own_evidence():
    described = LinkBandwidth("A", "B", 6.6 * GB, "measured", "thunderbolt").describe()
    assert "6.60 GB/s" in described and "measured" in described
