# SPDX-License-Identifier: Apache-2.0
"""Per launch: re-verify the coordinator's RDMA link to rank 1 and add it to the contract."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from . import layout
from .daemon import DaemonStatus, read_status
from .link_probe import QUICK, verify_link
from .links import NodeAddress, RdmaLink, discover_links
from .stage_plan import StageLink
from .store import RdmaLinkStore, get_rdma_link_store
from .verification import LinkVerification, read_driver_identity
from .words import load_word_ops

logger = logging.getLogger(__name__)
DISABLE_ENV = "OMLX_RDMA_STAGE_LINKS"
# The owner recorded while the dashboard verifies a link.
VERIFYING = "a dashboard verification"
# One owner per link at a time: a second probe would take the worker's service end away.
_claims_lock = threading.Lock()
_claims: dict[str, str] = {}


def claimed_links() -> dict[str, str]:
    """Held links, as link name to owner: a deployment ID or VERIFYING."""
    with _claims_lock:
        return dict(_claims)


def claim_link(name: str, owner: str) -> str | None:
    """Hold `name` for `owner`; returns whoever else holds it, or None once held."""
    with _claims_lock:
        holder = _claims.get(name)
        if holder is not None and holder != owner:
            return holder
        _claims[name] = owner
        return None


def release_links(owner: str) -> None:
    """Free every link `owner` holds."""
    with _claims_lock:
        for name in [name for name, holder in _claims.items() if holder == owner]:
            del _claims[name]


def release_link(name: str, owner: str) -> None:
    """Free `name` if `owner` still holds it."""
    with _claims_lock:
        if _claims.get(name) == owner:
            del _claims[name]


def describe_owner(owner: str) -> str:
    """How a link's holder reads in a message."""
    return owner if owner == VERIFYING else f"deployment {owner}"


def enrolled_node_addresses() -> tuple[NodeAddress, ...]:
    """Every enrolled CUDA worker as a matchable, reachable address."""
    from ..enrollment import get_cluster_enrollment

    try:
        nodes = get_cluster_enrollment().list_nodes()
    except RuntimeError:
        return ()
    return tuple(
        NodeAddress(
            node_id=node.node_id,
            ssh=node.ssh,
            addresses=tuple(node.addresses),
            hostname=node.hostname,
            python_executable=node.python_executable,
        )
        for node in nodes
    )


def _store() -> RdmaLinkStore | None:
    try:
        return get_rdma_link_store()
    except RuntimeError:
        return None


def _report(reason: str, edge: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "active": edge is not None and edge.get("verified", False),
        "reason": reason,
        "edges": [edge] if edge else [],
    }


def effective_report(
    report: dict[str, Any] | None, ranks: tuple[dict[str, Any], ...]
) -> dict[str, Any] | None:
    """The pre-launch report, corrected by what the ranks decided when they voted."""
    edges = [
        edge
        for rank in ranks
        for edge in (rank.get("stage_links") or {}).get("edges", [])
    ]
    if not report or not edges or all(edge.get("active") for edge in edges):
        return report
    reason = next(
        (
            edge["reason"]
            for edge in edges
            if not edge.get("active") and edge.get("reason")
        ),
        "a rank could not use the RDMA link",
    )
    return {**report, "active": False, "reason": reason}


def _pick(links: tuple[RdmaLink, ...], node_id: str) -> RdmaLink | None:
    reaching = [link for link in links if link.peer_node_id == node_id]
    return next(
        (link for link in reaching if link.usable), reaching[0] if reaching else None
    )


def attach_stage_links(
    deployment: Any,
    *,
    status_reader: Callable[[], DaemonStatus] = read_status,
    nodes: Callable[[], tuple[NodeAddress, ...]] = enrolled_node_addresses,
    verify: Callable[..., LinkVerification] = verify_link,
    store: Callable[[], RdmaLinkStore | None] = _store,
) -> tuple[Any, dict[str, Any]]:
    """The deployment with its verified stage links, and a report of what was decided and why."""
    cleared = replace(deployment, stage_links=())
    release_links(deployment.deployment_id)
    if os.environ.get(DISABLE_ENV, "").strip().lower() in {"0", "false", "no", "off"}:
        return cleared, _report(f"disabled by {DISABLE_ENV}")
    if deployment.backend != "ring" or len(deployment.hosts) < 2:
        return cleared, _report(
            "only multi-node TCP-ring deployments have a stage edge to move"
        )
    if deployment.tensor_parallel_size != 1:
        return cleared, _report("tensor-parallel deployments keep MLX's collectives")
    status = status_reader()
    if not status.reachable:
        return cleared, _report(status.reason)
    # Rank 0 is always this coordinator; the deployment pins its SSH target to 127.0.0.1.
    receiver = deployment.hosts[0].node_id
    rank_one = deployment.hosts[1]
    sender = rank_one.node_id
    known = nodes()
    enrolled = next((item for item in known if item.node_id == sender), None)
    if enrolled is None:
        return cleared, _report(f"rank 1 ({sender}) is not an enrolled CUDA worker")
    # Probe exactly where mlx.launch starts rank 1, not wherever the enrollment record points.
    node = replace(
        enrolled,
        ssh=rank_one.ssh,
        addresses=tuple(dict.fromkeys((*rank_one.ips, *enrolled.addresses))),
        python_executable=rank_one.python_executable or enrolled.python_executable,
    )
    known = tuple(node if item.node_id == sender else item for item in known)
    link = _pick(discover_links(status, known), sender)
    if link is None:
        return cleared, _report(f"no mcdma-rpcd link from {receiver} reaches {sender}")
    owner = claim_link(link.name, deployment.deployment_id)
    if owner is not None:
        return cleared, _report(
            f"link {link.name} is in use by {describe_owner(owner)}"
        )
    ops, ops_reason = load_word_ops()
    verification = verify(
        link,
        node,
        status=status,
        driver=read_driver_identity(),
        ops=ops,
        ops_reason=ops_reason,
        settings=QUICK,
    )
    records = store()
    if records is not None:
        records.record(verification)
    edge = {"sender_rank": 1, "receiver_rank": 0, **verification.to_dict()}
    if not verification.verified:
        release_links(deployment.deployment_id)
        logger.warning(
            "RDMA stage link %s not used: %s", link.name, verification.reason
        )
        return cleared, _report(
            f"link {link.name} failed its pre-launch check: {verification.reason}", edge
        )
    stage = StageLink(
        sender_rank=1,
        receiver_rank=0,
        link=link.name,
        service_socket=layout.service_socket_path(link.name),
    )
    return replace(deployment, stage_links=(stage,)), _report(
        f"rank 1 sends to rank 0 over {link.name}", edge
    )
