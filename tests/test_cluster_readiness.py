# SPDX-License-Identifier: Apache-2.0
"""The readiness ladder must be total, ordered, and never silent.

Failure #5 in one sentence: a node sat between "cable in" and "collective
proven" with no name for where it was and no words for how to move. Every
rung therefore has a place in the order and copy that ships with it.
"""

from __future__ import annotations

import pytest

from omlx.cluster.models import TransportState
from omlx.cluster.readiness import (
    LADDER_ORDER,
    TransportReadiness,
    is_fabric_verified,
    ladder_copy,
    ladder_rank,
    link_ladder_state,
    node_readiness,
)
from omlx.cluster.transport import LinkEndpoint, LinkStatus, SharedLink


def _link(state: str, ladder: str = "") -> LinkStatus:
    return LinkStatus(
        state=state,
        title="t",
        detail="d",
        backend="ring",
        ready=False,
        ladder=ladder,
    )


def _shared(kind: str = "rdma", ok: bool = True) -> SharedLink:
    endpoint = LinkEndpoint(host="a.local", interface="en6", address="172.16.99.1")
    peer = LinkEndpoint(host="b.local", interface="en5", address="172.16.99.2")
    return SharedLink(
        source=endpoint if ok else None,
        peer=peer if ok else None,
        kind=kind,
        reason="checked",
    )


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_ladder_order_is_total_over_the_enum():
    assert set(LADDER_ORDER) == set(TransportState)
    assert len(LADDER_ORDER) == len(TransportState)


def test_ladder_ranks_are_strictly_increasing():
    ranks = [ladder_rank(state) for state in LADDER_ORDER]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)


def test_the_existing_four_states_keep_their_wire_values():
    """Older peers serialize these exact strings; they must never move."""

    assert TransportState.UNAVAILABLE.value == "unavailable"
    assert TransportState.DISABLED.value == "disabled"
    assert TransportState.ENABLED_NO_PEER.value == "enabled_no_peer"
    assert (
        TransportState.PEER_LINKED_CONFIG_PENDING.value
        == "peer_linked_config_pending"
    )


def test_fabric_verified_is_derived_from_the_ladder():
    below = LADDER_ORDER[: ladder_rank(TransportState.FABRIC_VERIFIED)]
    assert all(not is_fabric_verified(state) for state in below)
    assert is_fabric_verified(TransportState.FABRIC_VERIFIED)
    assert is_fabric_verified(TransportState.COLLECTIVE_OK)


# ---------------------------------------------------------------------------
# Copy: states without copy don't ship
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", list(TransportState))
def test_every_state_has_nonempty_reason_and_remedy(state):
    reason, remedy = ladder_copy(state)
    assert reason.strip(), f"{state.value} shipped without a reason"
    assert remedy.strip(), f"{state.value} shipped without a remedy"


def test_stale_self_assigned_address_gets_its_own_copy():
    """The real incident: 169.254 renders like an address and carries nothing."""

    reason, remedy = ladder_copy(
        TransportState.PEER_LINKED_CONFIG_PENDING,
        stale_addresses=("169.254.42.1",),
    )
    assert "169.254.42.1" in reason
    assert "never finished configuring" in reason
    assert "Fabric Doctor" in remedy


def test_node_readiness_detects_stale_addresses_from_evidence():
    readiness = node_readiness(
        TransportState.PEER_LINKED_CONFIG_PENDING,
        (("rdma_en1", "169.254.42.1"),),
    )
    assert isinstance(readiness, TransportReadiness)
    assert "self-assigned" in readiness.reason
    payload = readiness.to_dict()
    assert payload["state"] == "peer_linked_config_pending"
    assert payload["reason"] and payload["remedy"]


def test_node_readiness_with_routable_addresses_uses_the_plain_copy():
    readiness = node_readiness(
        TransportState.ADDRESSED, (("rdma_en1", "172.16.99.1"),)
    )
    assert "self-assigned" not in readiness.reason
    assert readiness.reason and readiness.remedy


# ---------------------------------------------------------------------------
# Link-level derivation: each evidence combination maps to exactly one rung
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("link_status", "shared", "verify", "collective", "expected"),
    [
        # No evidence at all.
        (None, None, None, None, TransportState.UNAVAILABLE),
        # classify_link states map through their declared ladder.
        (_link("unknown", "unavailable"), None, None, None, TransportState.UNAVAILABLE),
        (
            _link("rdma_not_enabled", "disabled"),
            None,
            None,
            None,
            TransportState.DISABLED,
        ),
        (
            _link("ethernet", "enabled_no_peer"),
            None,
            None,
            None,
            TransportState.ENABLED_NO_PEER,
        ),
        (
            _link("rdma_needs_setup", "peer_linked_config_pending"),
            None,
            None,
            None,
            TransportState.PEER_LINKED_CONFIG_PENDING,
        ),
        # One-ended rdma_ready evidence tops out at ROUTED.
        (_link("rdma_ready", "routed"), None, None, None, TransportState.ROUTED),
        # A LinkStatus without a ladder falls back to the state mapping.
        (_link("rdma_ready"), None, None, None, TransportState.ROUTED),
        (_link("thunderbolt"), None, None, None, TransportState.PEER_LINKED_CONFIG_PENDING),
        # Two-ended proof earns REACHABLE — an ethernet shared link does not.
        (_link("rdma_ready"), _shared("rdma"), None, None, TransportState.REACHABLE),
        (_link("rdma_ready"), _shared("ethernet"), None, None, TransportState.ROUTED),
        (_link("rdma_ready"), _shared("rdma", ok=False), None, None, TransportState.ROUTED),
        # Verification and the collective handshake claim the top rungs.
        (
            _link("rdma_ready"),
            _shared("rdma"),
            (True, "120 Gb/s"),
            None,
            TransportState.FABRIC_VERIFIED,
        ),
        (
            _link("rdma_ready"),
            _shared("rdma"),
            (False, "bandwidth below floor"),
            None,
            TransportState.REACHABLE,
        ),
        (
            _link("rdma_ready"),
            _shared("rdma"),
            (True, "120 Gb/s"),
            True,
            TransportState.COLLECTIVE_OK,
        ),
        (
            _link("rdma_ready"),
            _shared("rdma"),
            None,
            (True, "2 ranks"),
            TransportState.COLLECTIVE_OK,
        ),
    ],
)
def test_evidence_maps_to_exactly_one_rung(
    link_status, shared, verify, collective, expected
):
    assert (
        link_ladder_state(link_status, shared, verify, collective) is expected
    )


def test_node_local_evidence_can_never_claim_reachable():
    """Single-host evidence proving 'reachable' was failure #5's lie."""

    for state in ("rdma_ready", "rdma_needs_setup", "thunderbolt", "ethernet"):
        rung = link_ladder_state(_link(state))
        assert ladder_rank(rung) < ladder_rank(TransportState.REACHABLE)
