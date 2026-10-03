# SPDX-License-Identifier: Apache-2.0
"""Collision-checked, VPN-aware fabric subnet selection."""

from __future__ import annotations

import ipaddress

import pytest

from omlx.cluster.transport import (
    LinkSetupError,
    choose_fabric_subnet,
)


def _net(cidr: str) -> ipaddress.IPv4Network:
    return ipaddress.ip_network(cidr)


def test_prefers_a_vpn_excludable_range_when_nothing_is_occupied():
    # 172.16.0.0/12 leads because full-tunnel VPNs commonly exclude it, so the
    # fabric survives a VPN that would swallow a 10.x link.
    assert str(choose_fabric_subnet([])) == "172.16.99.0/24"


def test_home_lan_does_not_push_it_off_the_preferred_range():
    assert (
        str(choose_fabric_subnet([_net("192.168.0.0/24")])) == "172.16.99.0/24"
    )


def test_skips_candidates_that_overlap_an_occupied_network():
    # A /23 at .100 covers both .100 and .101, so the next free /24 is .102.
    chosen = choose_fabric_subnet(
        [
            _net("10.0.0.0/8"),
            _net("172.16.99.0/24"),
            _net("172.16.100.0/23"),
        ]
    )
    assert str(chosen) == "172.16.102.0/24"


def test_never_reproduces_the_swallowed_10_0_1_incident():
    # 10.x is never a leading candidate; the choice stays in the VPN-excludable
    # range even when the LAN is busy.
    chosen = choose_fabric_subnet([_net("192.168.0.0/24")])
    assert chosen.overlaps(_net("172.16.0.0/12"))
    assert not chosen.overlaps(_net("10.0.1.0/24"))


def test_raises_when_every_candidate_collides():
    occupied = [_net("172.16.0.0/12"), _net("10.0.0.0/8")]
    with pytest.raises(LinkSetupError, match="free private subnet"):
        choose_fabric_subnet(occupied)


# --- C4: candidates inside a detected VPN exclusion rank first -------------


def test_an_exclusion_contained_candidate_wins_over_the_static_order():
    # The peer's VPN provably excludes 10.0.0.0/8, so a 10.x candidate beats
    # the static 172.16.x preference — the incident's trick, systematized.
    chosen = choose_fabric_subnet([], preferred=[_net("10.0.0.0/8")])

    assert str(chosen) == "10.90.99.0/24"


def test_a_preferred_candidate_still_passes_the_collision_check():
    # A wrong or partial exclusion read must not poison selection: the first
    # exclusion-contained candidate is occupied, so the next one is taken.
    chosen = choose_fabric_subnet(
        [_net("10.90.99.0/24")], preferred=[_net("10.0.0.0/8")]
    )

    assert str(chosen) == "10.91.99.0/24"


def test_an_exclusion_matching_no_candidate_preserves_the_static_order():
    chosen = choose_fabric_subnet([], preferred=[_net("203.0.113.0/24")])

    assert str(chosen) == "172.16.99.0/24"


def test_empty_preferred_is_byte_for_byte_todays_behavior():
    occupied = [
        _net("10.0.0.0/8"),
        _net("172.16.99.0/24"),
        _net("172.16.100.0/23"),
    ]

    assert choose_fabric_subnet(occupied, preferred=()) == choose_fabric_subnet(
        occupied
    )
    assert str(choose_fabric_subnet([], preferred=())) == "172.16.99.0/24"
