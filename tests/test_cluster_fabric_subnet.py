# SPDX-License-Identifier: Apache-2.0
"""Collision-checked, VPN-aware fabric subnet selection."""

from __future__ import annotations

import ipaddress

from omlx.cluster.transport import (
    choose_fabric_subnet,
)


def _net(cidr: str) -> ipaddress.IPv4Network:
    return ipaddress.ip_network(cidr)


def test_defaults_to_a_10x_range_when_nothing_is_occupied_or_preferred():
    # #2867 review: the static default must stay close to the previous
    # single hardcoded default (10.0.1.1/24), not silently move every fresh
    # cluster onto 172.16.x -- that's a deliberate, separate behavior
    # change deferred to its own PR with a migration note. 172.16.x is
    # still the SECOND static tier, and only a detected VPN's ``preferred``
    # promotes it ahead (see the tests below).
    assert str(choose_fabric_subnet([])) == "10.90.99.0/24"


def test_home_lan_does_not_push_it_off_the_default_range():
    assert (
        str(choose_fabric_subnet([_net("192.168.0.0/24")])) == "10.90.99.0/24"
    )


def test_skips_candidates_that_overlap_an_occupied_network():
    # The two leading 10.x candidates are both occupied, so the next free
    # one in static order is taken.
    chosen = choose_fabric_subnet(
        [
            _net("172.16.0.0/12"),
            _net("10.90.99.0/24"),
            _net("10.91.99.0/24"),
        ]
    )
    assert str(chosen) == "10.92.99.0/24"


def test_a_vpn_that_excludes_172_16_wins_the_incident_fix():
    # The real incident this feature exists for: a full-tunnel VPN tunnels
    # 10.x (swallowing the default choice) while provably excluding
    # 172.16.0.0/12 from the tunnel -- the exclusion must promote a 172.x
    # candidate ahead of the now-default-leading 10.x ones.
    chosen = choose_fabric_subnet([], preferred=[_net("172.16.0.0/12")])
    assert str(chosen) == "172.16.99.0/24"


def test_raising_10x_default_when_it_is_the_thing_actually_tunneled():
    # 10.x sitting first in the static order is fine right up until it's
    # occupied (tunneled) too -- collision-checking still routes around it,
    # same as any other candidate.
    chosen = choose_fabric_subnet([_net("10.0.0.0/8")])
    assert not chosen.overlaps(_net("10.0.0.0/8"))


def test_last_resort_returns_the_least_colliding_candidate_instead_of_raising():
    # #2867 review: an aggressive VPN policy vetoing every static candidate
    # must not hard-fail Start Cluster with a 503 -- the old hardcoded
    # default at least attempted a link, and this must too. Every candidate
    # collides with something here (10.0.0.0/8 hits the whole 10.x tier,
    # 172.16.0.0/12 hits the whole 172.x tier), but the static leader,
    # 10.90.99.0/24, is ALSO hit by a second, narrower occupied range --
    # making it collide twice while every other candidate only collides
    # once, so the least-bad pick must skip past it to the next candidate.
    occupied = [
        _net("10.0.0.0/8"),
        _net("172.16.0.0/12"),
        _net("10.90.99.0/24"),
    ]
    chosen = choose_fabric_subnet(occupied)
    assert str(chosen) == "10.91.99.0/24"


def test_last_resort_is_deterministic_on_a_full_tie():
    # Every candidate collides with the exact same catch-all range -- ties
    # break on static order (first candidate wins), matching this module's
    # existing "rank order makes retries deterministic" design.
    chosen = choose_fabric_subnet([_net("0.0.0.0/0")])
    assert str(chosen) == "10.90.99.0/24"


# --- C4: candidates inside a detected VPN exclusion rank first -------------


def test_an_exclusion_contained_candidate_wins_over_the_static_order():
    # A VPN exclusion promotes a matching candidate ahead of the rest of the
    # static list, even one that isn't already the static leader -- the
    # incident's 172.16.99.x trick, systematized.
    chosen = choose_fabric_subnet([], preferred=[_net("172.16.0.0/12")])

    assert str(chosen) == "172.16.99.0/24"


def test_a_preferred_candidate_still_passes_the_collision_check():
    # A wrong or partial exclusion read must not poison selection: the first
    # exclusion-contained candidate is occupied, so the next one is taken.
    chosen = choose_fabric_subnet(
        [_net("172.16.99.0/24")], preferred=[_net("172.16.0.0/12")]
    )

    assert str(chosen) == "172.16.100.0/24"


def test_an_exclusion_matching_no_candidate_preserves_the_static_order():
    chosen = choose_fabric_subnet([], preferred=[_net("203.0.113.0/24")])

    assert str(chosen) == "10.90.99.0/24"


def test_empty_preferred_is_byte_for_byte_todays_behavior():
    occupied = [
        _net("172.16.0.0/12"),
        _net("10.90.99.0/24"),
        _net("10.91.99.0/24"),
    ]

    assert choose_fabric_subnet(occupied, preferred=()) == choose_fabric_subnet(
        occupied
    )
    assert str(choose_fabric_subnet([], preferred=())) == "10.90.99.0/24"
