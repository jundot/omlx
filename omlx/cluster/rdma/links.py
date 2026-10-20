# SPDX-License-Identifier: Apache-2.0
"""Match the connect daemon's peer links to enrolled cluster nodes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from . import layout
from .daemon import DaemonStatus, PeerStatus


@dataclass(frozen=True)
class NodeAddress:
    """How a cluster node can be recognised and reached."""

    node_id: str
    ssh: str
    addresses: tuple[str, ...] = ()
    hostname: str | None = None
    python_executable: str | None = None

    def matches(self, host: str) -> bool:
        """Whether the daemon's peer host names this node."""
        target = self.ssh.rsplit("@", 1)[-1].strip("[]")
        names = {target, *self.addresses}
        if self.hostname:
            names.update({self.hostname, self.hostname.split(".")[0]})
        return host.strip("[]") in names


@dataclass(frozen=True)
class RdmaLink:
    """One daemon peer link, tied to the cluster node it reaches when known."""

    name: str
    up: bool
    reason: str
    peer_host: str | None = None
    peer_node_id: str | None = None
    device: str | None = None
    request_bytes: int | None = None
    reply_bytes: int | None = None
    since: float | None = None

    @property
    def usable(self) -> bool:
        return self.up and self.peer_node_id is not None and not self.reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "up": self.up,
            "usable": self.usable,
            "reason": self.reason,
            "peer_host": self.peer_host,
            "peer_node_id": self.peer_node_id,
            "device": self.device,
            "request_bytes": self.request_bytes,
            "reply_bytes": self.reply_bytes,
            "since": self.since,
        }


def _link(peer: PeerStatus, nodes: tuple[NodeAddress, ...]) -> RdmaLink:
    try:
        layout.valid_link_name(peer.name)
    except ValueError as exc:
        return RdmaLink(peer.name, peer.up, str(exc))
    if peer.host is None:
        return RdmaLink(
            peer.name, peer.up, "mcdma-rpcd is too old to report the peer host"
        )
    matches = [node for node in nodes if node.matches(peer.host)]
    base = {
        "name": peer.name,
        "up": peer.up,
        "peer_host": peer.host,
        "device": peer.device,
        "request_bytes": peer.request_bytes,
        "reply_bytes": peer.reply_bytes,
        "since": peer.since,
    }
    if len(matches) != 1:
        found = "no" if not matches else "more than one"
        return RdmaLink(reason=f"{found} enrolled node answers to {peer.host}", **base)
    if not peer.up:
        return RdmaLink(
            reason="mcdma-rpcd reports this link down",
            peer_node_id=matches[0].node_id,
            **base,
        )
    return RdmaLink(reason="", peer_node_id=matches[0].node_id, **base)


def discover_links(
    status: DaemonStatus, nodes: Iterable[NodeAddress]
) -> tuple[RdmaLink, ...]:
    """The coordinator's RDMA links, each resolved to a node or explained."""
    known = tuple(nodes)
    return tuple(_link(peer, known) for peer in status.peers)
