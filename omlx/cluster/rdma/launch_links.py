# SPDX-License-Identifier: Apache-2.0
"""Per launch: re-verify an RDMA link for every stage edge and add the verified ones to the contract."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from . import layout
from .daemon import DaemonStatus, read_status
from .link_probe import read_remote_status, verify_link, verify_remote_link
from .links import NodeAddress, RdmaLink, discover_links
from .probe_run import QUICK
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


def _report(reason: str) -> dict[str, Any]:
    return {"active": False, "reason": reason, "edges": []}


def effective_report(
    report: dict[str, Any] | None, ranks: tuple[dict[str, Any], ...]
) -> dict[str, Any] | None:
    """The pre-launch report, corrected by what the ranks decided when they voted."""
    if not report:
        return report
    states = [rank.get("stage_links") or {} for rank in ranks]
    voted = [edge for state in states for edge in state.get("edges", [])]
    refused: dict[tuple[Any, Any], str] = {}
    for edge in voted:
        if not edge.get("active"):
            key = (edge.get("sender_rank"), edge.get("receiver_rank"))
            refused.setdefault(
                key, edge.get("reason") or "a rank could not use the RDMA link"
            )
    edges = []
    for edge in report.get("edges", []):
        # A hop is live once its probe passed and no rank voted it down.
        why = refused.get((edge.get("sender_rank"), edge.get("receiver_rank")))
        live = bool(edge.get("verified")) and why is None
        edges.append({**edge, "active": live, **({"reason": why} if why else {})})
    corrected = {**report, "edges": edges}
    if refused:
        corrected["active"] = (
            any(edge["active"] for edge in edges)
            if edges
            else any(edge.get("active") for edge in voted)
        )
        corrected["reason"] = next(iter(refused.values()))
    relays = [
        state["token_relay"]
        for state in states
        if isinstance(state.get("token_relay"), dict)
    ]
    if relays:
        # Every rank decides the relay alike; any rank reporting it off says why.
        corrected["token_relay"] = next(
            (relay for relay in relays if not relay.get("active")), relays[0]
        )
    return corrected


def _pick(links: tuple[RdmaLink, ...], node_id: str) -> RdmaLink | None:
    reaching = [link for link in links if link.peer_node_id == node_id]
    return next(
        (link for link in reaching if link.usable), reaching[0] if reaching else None
    )


def _worker_nodes(
    deployment: Any, known: tuple[NodeAddress, ...]
) -> dict[int, NodeAddress]:
    """Each worker rank's enrolled node, reached exactly where mlx.launch starts that rank."""
    workers = {}
    for rank, host in enumerate(deployment.hosts[1:], start=1):
        enrolled = next((item for item in known if item.node_id == host.node_id), None)
        if enrolled is not None:
            workers[rank] = replace(
                enrolled,
                ssh=host.ssh,
                addresses=tuple(dict.fromkeys((*host.ips, *enrolled.addresses))),
                python_executable=host.python_executable or enrolled.python_executable,
            )
    return workers


def _attach_edge(
    deployment: Any,
    receiver: int,
    workers: dict[int, NodeAddress],
    known: tuple[NodeAddress, ...],
    *,
    status_reader: Callable[[], DaemonStatus],
    remote_status: Callable[[NodeAddress], DaemonStatus],
    verify: Callable[..., LinkVerification],
    verify_remote: Callable[..., LinkVerification],
    store: Callable[[], RdmaLinkStore | None],
) -> tuple[StageLink | None, dict[str, Any], str]:
    """Verify edge receiver+1 -> receiver: its stage link, the edge's evidence, and what was decided."""
    sender = receiver + 1
    base = {"sender_rank": sender, "receiver_rank": receiver}

    def refused(reason: str) -> tuple[None, dict[str, Any], str]:
        return None, {**base, "verified": False, "reason": reason}, reason

    for rank in (sender, receiver):
        if rank and rank not in workers:
            node_id = deployment.hosts[rank].node_id
            return refused(f"rank {rank} ({node_id}) is not an enrolled CUDA worker")
    node = workers[sender]
    # Rank 0 is this coordinator; every other receiver is a worker reached over SSH.
    client = workers.get(receiver)
    status = status_reader() if client is None else remote_status(client)
    if not status.reachable:
        return refused(status.reason)
    link = _pick(discover_links(status, known), node.node_id)
    if link is None:
        return refused(
            f"no mcdma-rpcd link from {deployment.hosts[receiver].node_id} "
            f"reaches {node.node_id}"
        )
    key = link.name if client is None else f"{client.node_id}/{link.name}"
    owner = claim_link(key, deployment.deployment_id)
    if owner is not None:
        return refused(f"link {link.name} is in use by {describe_owner(owner)}")
    if client is None:
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
        # Only the coordinator's links have a dashboard row to keep evidence for.
        records = store()
        if records is not None:
            records.record(verification)
    else:
        verification = verify_remote(link, client, node, status=status, settings=QUICK)
    edge = {**base, **verification.to_dict()}
    if not verification.verified:
        release_link(key, deployment.deployment_id)
        logger.warning(
            "RDMA stage link %s not used: %s", link.name, verification.reason
        )
        return (
            None,
            edge,
            f"link {link.name} failed its pre-launch check: {verification.reason}",
        )
    stage = StageLink(
        sender_rank=sender,
        receiver_rank=receiver,
        link=link.name,
        service_socket=layout.service_socket_path(link.name),
    )
    return stage, edge, f"rank {sender} sends to rank {receiver} over {link.name}"


def attach_stage_links(
    deployment: Any,
    *,
    status_reader: Callable[[], DaemonStatus] = read_status,
    remote_status: Callable[[NodeAddress], DaemonStatus] = read_remote_status,
    nodes: Callable[[], tuple[NodeAddress, ...]] = enrolled_node_addresses,
    verify: Callable[..., LinkVerification] = verify_link,
    verify_remote: Callable[..., LinkVerification] = verify_remote_link,
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
    enrolled = nodes()
    workers = _worker_nodes(deployment, enrolled)
    reached = {node.node_id: node for node in workers.values()}
    known = tuple(reached.get(item.node_id, item) for item in enrolled)
    receivers = range(len(deployment.hosts) - 1)
    # Edges are probed at once so a long pipeline does not multiply the launch delay.
    with ThreadPoolExecutor(max_workers=len(receivers)) as pool:
        results = list(
            pool.map(
                lambda receiver: _attach_edge(
                    deployment,
                    receiver,
                    workers,
                    known,
                    status_reader=status_reader,
                    remote_status=remote_status,
                    verify=verify,
                    verify_remote=verify_remote,
                    store=store,
                ),
                receivers,
            )
        )
    stages: list[StageLink] = []
    edges: list[dict[str, Any]] = []
    messages: list[str] = []
    for stage, edge, message in results:
        if stage is not None and any(item.link == stage.link for item in stages):
            # The contract names links uniquely; each connect daemon must use its own names.
            release_link(
                f"{workers[stage.receiver_rank].node_id}/{stage.link}",
                deployment.deployment_id,
            )
            stage = None
            message = f"link name {edge['link']} is used by two edges; give each link its own name"
            edge = {**edge, "verified": False, "reason": message}
        if stage is not None:
            stages.append(stage)
        edges.append(edge)
        messages.append(message)
    report = {"active": bool(stages), "reason": "; ".join(messages), "edges": edges}
    return replace(deployment, stage_links=tuple(stages)), report
