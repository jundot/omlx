# SPDX-License-Identifier: Apache-2.0
"""The transport readiness ladder: every rung between "cable in" and
"collective proven" is a named state carrying copy a person can act on.

The real failure this closes: after a reboot the Thunderbolt link came up
without authorization, the node sat at ``peer_linked_config_pending`` with a
stale self-assigned 169.254 address, and the only visible symptom was an
unrelated 400 from autoconfigure. A ladder state without a reason and a remedy
is exactly that failure again, so states without copy don't ship — the copy
table below is total over the enum and tested to stay that way.

Everything here is pure over already-collected evidence: no SSH, no
subprocess, so each derivation is testable without two Macs.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from typing import Any

from .models import TransportState

# Rungs in climbing order. Node-local evidence can climb to ROUTED but never
# higher: REACHABLE and above require two-ended proof (a single host claiming
# reachability was failure #5's lie in miniature).
LADDER_ORDER: tuple[TransportState, ...] = (
    TransportState.UNAVAILABLE,
    TransportState.DISABLED,
    TransportState.ENABLED_NO_PEER,
    TransportState.PEER_LINKED_CONFIG_PENDING,
    TransportState.ADDRESSED,
    TransportState.ROUTED,
    TransportState.REACHABLE,
    TransportState.FABRIC_VERIFIED,
    TransportState.COLLECTIVE_OK,
)

_LADDER_RANK: dict[TransportState, int] = {
    state: rank for rank, state in enumerate(LADDER_ORDER)
}


def ladder_rank(state: TransportState) -> int:
    """Position of a state on the ladder; higher is closer to a working collective."""

    return _LADDER_RANK[state]


def is_fabric_verified(state: TransportState) -> bool:
    """``fabric_verified`` is derived: true iff the ladder reached FABRIC_VERIFIED."""

    return ladder_rank(state) >= _LADDER_RANK[TransportState.FABRIC_VERIFIED]


@dataclass(frozen=True)
class TransportReadiness:
    """One rung plus the copy rendered verbatim in the CLI and GUI."""

    state: TransportState
    reason: str
    remedy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "remedy": self.remedy,
        }


# reason: what the evidence says is true right now.
# remedy: the one next action that earns the next rung.
_COPY: dict[TransportState, tuple[str, str]] = {
    TransportState.UNAVAILABLE: (
        "This system does not report an RDMA-capable transport.",
        "Use Macs with Thunderbolt 5 on a current macOS, or run the cluster "
        "over the TCP ring.",
    ),
    TransportState.DISABLED: (
        "RDMA is switched off in the operating system.",
        "Shut down, hold the power button to enter macOS Recovery, run "
        "`rdma_ctl enable` in Utilities → Terminal, and restart.",
    ),
    TransportState.ENABLED_NO_PEER: (
        "RDMA is enabled, but no Thunderbolt peer is connected.",
        "Connect a Thunderbolt 5 cable between the Macs, then detect the "
        "link again.",
    ),
    TransportState.PEER_LINKED_CONFIG_PENDING: (
        "A Thunderbolt peer is connected, but the link has no usable IP "
        "address yet.",
        "Press Start Cluster — oMLX will configure and verify the link after "
        "administrator approval — or run Fabric Doctor.",
    ),
    TransportState.ADDRESSED: (
        "The RDMA interface carries a routable address, but no route to the "
        "peer uses it yet.",
        "Run Fabric Doctor to check the route, or detect the link again "
        "after both Macs are awake.",
    ),
    TransportState.ROUTED: (
        "The route to the peer uses the RDMA interface, but reachability has "
        "not been proven from both ends.",
        "Run Fabric Doctor to verify the link end to end, or press Start — "
        "activation verifies before launching.",
    ),
    TransportState.REACHABLE: (
        "Both Macs answered on their fabric addresses in both directions.",
        "Press Start — the fabric bandwidth check runs during activation.",
    ),
    TransportState.FABRIC_VERIFIED: (
        "The fabric passed verification at the expected link bandwidth.",
        "Press Start — the collective handshake is the only rung left.",
    ),
    TransportState.COLLECTIVE_OK: (
        "A collective handshake completed across the fabric.",
        "Nothing to fix — the cluster is ready to serve.",
    ),
}

# The stale self-assigned address deserves its own copy: it is the single most
# misdiagnosed rung (macOS shows a link and an IP, and neither is usable).
_STALE_ADDRESS_REMEDY = "Run Fabric Doctor → Re-address link."


def ladder_copy(state: TransportState, **evidence: Any) -> tuple[str, str]:
    """(reason, remedy) for a rung, specialised by evidence where it matters.

    Total over the enum: every state returns non-empty copy.
    """

    stale = tuple(evidence.get("stale_addresses") or ())
    if state is TransportState.PEER_LINKED_CONFIG_PENDING and stale:
        return (
            f"The link has a self-assigned address ({', '.join(stale)}) — "
            "macOS never finished configuring it.",
            _STALE_ADDRESS_REMEDY,
        )
    return _COPY[state]


def _stale_rdma_addresses(
    rdma_addresses: tuple[tuple[str, str], ...],
) -> tuple[str, ...]:
    """Self-assigned/loopback addresses on RDMA interfaces — present but unusable."""

    stale: list[str] = []
    for _device, address in rdma_addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if ip.is_link_local or ip.is_loopback:
            stale.append(address)
    return tuple(stale)


def node_readiness(
    state: TransportState,
    rdma_addresses: tuple[tuple[str, str], ...] = (),
) -> TransportReadiness:
    """Readiness for one node's local evidence, with copy specialised to it."""

    reason, remedy = ladder_copy(
        state, stale_addresses=_stale_rdma_addresses(rdma_addresses)
    )
    return TransportReadiness(state=state, reason=reason, remedy=remedy)


# LinkStatus.state (transport.py) → the highest rung that evidence supports.
_LINK_STATE_LADDER: dict[str, TransportState] = {
    "unknown": TransportState.UNAVAILABLE,
    "ethernet": TransportState.ENABLED_NO_PEER,
    "rdma_not_enabled": TransportState.DISABLED,
    "thunderbolt": TransportState.PEER_LINKED_CONFIG_PENDING,
    "rdma_needs_setup": TransportState.PEER_LINKED_CONFIG_PENDING,
    "rdma_ready": TransportState.ROUTED,
}


def _passed(result: Any) -> bool:
    """A verification result — bool, (ok, detail), or None — as a pass/fail."""

    if result is None:
        return False
    if isinstance(result, tuple):
        return bool(result and result[0])
    return bool(result)


def link_ladder_state(
    link_status: Any,
    shared_link: Any = None,
    verify_result: Any = None,
    collective_result: Any = None,
) -> TransportState:
    """The link's rung, from strongest evidence down.

    Pure over its inputs. ``link_status`` is a ``transport.LinkStatus`` (or
    None), ``shared_link`` a ``transport.SharedLink`` (or None) whose two-ended
    address proof earns REACHABLE, ``verify_result`` the fabric bandwidth
    verification, and ``collective_result`` a completed collective handshake.
    """

    if _passed(collective_result):
        return TransportState.COLLECTIVE_OK
    if _passed(verify_result):
        return TransportState.FABRIC_VERIFIED
    if (
        shared_link is not None
        and getattr(shared_link, "ok", False)
        and getattr(shared_link, "kind", "") == "rdma"
    ):
        return TransportState.REACHABLE
    if link_status is None:
        return TransportState.UNAVAILABLE
    ladder = getattr(link_status, "ladder", "")
    if ladder:
        return TransportState(ladder)
    return _LINK_STATE_LADDER.get(
        getattr(link_status, "state", "unknown"), TransportState.UNAVAILABLE
    )
