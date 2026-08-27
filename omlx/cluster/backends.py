# SPDX-License-Identifier: Apache-2.0
"""Per-cluster collective backend selection: JACCL everywhere, or TCP ring.

The rule is deliberately binary and fail-safe: ``jaccl`` is selected only when
**every** cluster member reports ``rdma_ctl`` enabled *and* at least one RDMA
device; anything else selects the TCP ``ring``. A single member without RDMA
silently downgrading only its own stage is exactly how half-configured JACCL
clusters hang at ``mx.distributed.init`` — the whole cluster shares one
backend, so the weakest member decides, out loud.

This module is pure: members arrive as data (from ``collect_cluster_status``
locally, from peer probes remotely, or from posted host records), which keeps
selection offline-testable. Live ``rdma_ctl`` state is re-verified at launch
preflight; a selection made from stale evidence can only fail closed there.

Ring is a first-class runtime path, not a degraded one: ring deployments run
through the same ``DistributedJobSupervisor`` verified teardown (process-group
SIGTERM→SIGKILL escalation with proof, plus the marker-pid rank sweep —
including remote ranks over SSH) as JACCL deployments. Nothing in the teardown
machinery is backend-conditional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class MemberFabric:
    """One member's RDMA capability, as observed by a probe or host record."""

    node_id: str
    rdma_ctl_enabled: bool = False
    rdma_devices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("member node_id is required")
        object.__setattr__(
            self, "rdma_devices", tuple(sorted(set(self.rdma_devices)))
        )

    @property
    def jaccl_ready(self) -> bool:
        return self.rdma_ctl_enabled and bool(self.rdma_devices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "rdma_ctl_enabled": self.rdma_ctl_enabled,
            "rdma_devices": list(self.rdma_devices),
            "jaccl_ready": self.jaccl_ready,
        }


@dataclass(frozen=True)
class BackendSelection:
    """The backend a whole cluster will use, and why."""

    backend: str  # "jaccl" | "ring"
    reason: str
    blockers: tuple[str, ...] = ()
    members: tuple[MemberFabric, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "reason": self.reason,
            "blockers": list(self.blockers),
            "members": [member.to_dict() for member in self.members],
        }


def select_cluster_backend(members: Iterable[MemberFabric]) -> BackendSelection:
    """Pick ``jaccl`` when rdma_ctl is enabled on ALL members, else ``ring``.

    ``jaccl-ring`` (Thunderbolt without RDMA) remains manually selectable on
    the deployment endpoints; automatic selection never picks it because its
    hostfile still needs a complete RDMA device matrix, which "no RDMA"
    members by definition cannot supply.
    """

    members = tuple(members)
    if len(members) < 2:
        raise ValueError("backend selection needs at least two cluster members")
    node_ids = [member.node_id for member in members]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("cluster member node IDs must be unique")

    blockers = tuple(
        member.node_id
        for member in members
        if not member.jaccl_ready
    )
    if not blockers:
        return BackendSelection(
            backend="jaccl",
            reason=(
                f"rdma_ctl is enabled with RDMA devices on all "
                f"{len(members)} members"
            ),
            members=members,
        )
    details = []
    for member in members:
        if member.jaccl_ready:
            continue
        if not member.rdma_ctl_enabled:
            details.append(f"{member.node_id} has rdma_ctl disabled")
        else:
            details.append(f"{member.node_id} reports no RDMA device")
    return BackendSelection(
        backend="ring",
        reason=(
            "JACCL needs rdma_ctl enabled on every member; falling back to "
            "the TCP ring for the whole cluster (" + "; ".join(details) + ")"
        ),
        blockers=blockers,
        members=members,
    )


def members_from_host_records(hosts: Iterable[Any]) -> tuple[MemberFabric, ...]:
    """Derive member fabric evidence from posted host records.

    A host that carries a complete RDMA connectivity matrix (one entry per
    peer, null diagonal, no null off-diagonal) is treated as rdma_ctl-enabled:
    that matrix can only be produced from live interface names. This is
    request evidence, not a probe — launch preflight re-verifies the live
    state, so stale input fails closed at activation rather than launching a
    collective that blocks.
    """

    hosts = list(hosts)
    size = len(hosts)
    members: list[MemberFabric] = []
    for host in hosts:
        rdma = list(getattr(host, "rdma", ()) or ())
        complete = len(rdma) == size and size >= 2
        devices: list[str] = []
        if complete:
            for index, path in enumerate(rdma):
                expected_none = index == len(members)
                if expected_none:
                    if path is not None:
                        complete = False
                        break
                    continue
                if path is None:
                    complete = False
                    break
                if isinstance(path, (list, tuple)):
                    devices.extend(str(item) for item in path)
                else:
                    devices.append(str(path))
        members.append(
            MemberFabric(
                node_id=str(getattr(host, "node_id", "")),
                rdma_ctl_enabled=complete,
                rdma_devices=tuple(devices) if complete else (),
            )
        )
    return tuple(members)
