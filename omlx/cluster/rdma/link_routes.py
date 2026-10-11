# SPDX-License-Identifier: Apache-2.0
"""Admin handlers for the coordinator's RDMA links, registered on the cluster router."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .daemon import read_status
from .launch_links import (
    VERIFYING,
    claim_link,
    claimed_links,
    describe_owner,
    enrolled_node_addresses,
    release_link,
)
from .link_probe import FULL, verify_link
from .links import discover_links
from .store import RdmaLinkStore, get_rdma_link_store
from .verification import cached_driver_identity, link_identity, read_driver_identity
from .words import load_word_ops

# claim_link lets an owner claim again, and every dashboard verification is VERIFYING.
_verify_claim_lock = threading.Lock()


class RdmaLinkVerifyRequest(BaseModel):
    """Which link to verify."""

    model_config = ConfigDict(extra="forbid")

    link: str = Field(min_length=1, max_length=20, pattern=r"^[A-Za-z0-9_-]+$")


def _store() -> RdmaLinkStore | None:
    try:
        return get_rdma_link_store()
    except RuntimeError:
        return None


def _inventory() -> dict[str, Any]:
    status = read_status()
    links = discover_links(status, enrolled_node_addresses())
    store = _store()
    records = store.all() if store is not None else {}
    driver = cached_driver_identity()
    ops, ops_reason = load_word_ops()
    claims = claimed_links()
    now = time.time()
    rows = []
    for link in links:
        record = records.get(link.name)
        stale = (
            record.stale_reason(link_identity(link, status, driver), now)
            if record
            else "never verified"
        )
        holder = claims.get(link.name)
        rows.append(
            {
                **link.to_dict(),
                "in_use_by": holder if holder != VERIFYING else None,
                "verifying": holder == VERIFYING,
                "verified": stale is None,
                "stale_reason": stale,
                "verification": record.to_dict() if record else None,
            }
        )
    return {
        "daemon": status.to_dict(),
        "helper": {"available": ops is not None, "reason": ops_reason},
        "driver": {"version": driver.version, "uuid": driver.uuid} if driver else None,
        "store_error": store.load_error
        if store is not None
        else "RDMA link store is not configured",
        "links": rows,
    }


def _verify(name: str) -> dict[str, Any]:
    status = read_status()
    if not status.reachable:
        raise HTTPException(status_code=409, detail=status.reason)
    nodes = enrolled_node_addresses()
    link = next(
        (item for item in discover_links(status, nodes) if item.name == name), None
    )
    if link is None:
        raise HTTPException(
            status_code=404, detail=f"mcdma-rpcd has no link named {name}"
        )
    node = next((item for item in nodes if item.node_id == link.peer_node_id), None)
    if node is None:
        raise HTTPException(
            status_code=409,
            detail=link.reason or f"link {name} does not reach an enrolled worker",
        )
    with _verify_claim_lock:
        owner = claimed_links().get(name) or claim_link(name, VERIFYING)
    if owner is not None:
        raise HTTPException(
            status_code=409, detail=f"link {name} is in use by {describe_owner(owner)}"
        )
    try:
        ops, ops_reason = load_word_ops()
        verification = verify_link(
            link,
            node,
            status=status,
            driver=read_driver_identity(),
            ops=ops,
            ops_reason=ops_reason,
            settings=FULL,
        )
    finally:
        release_link(name, VERIFYING)
    store = _store()
    if store is not None:
        store.record(verification)
    return verification.to_dict()


async def cluster_rdma_links() -> dict[str, Any]:
    """The coordinator's RDMA links, each with its latest evidence and whether it still holds."""
    return await asyncio.to_thread(_inventory)


async def cluster_rdma_link_verify(request: RdmaLinkVerifyRequest) -> dict[str, Any]:
    """Probe one link live in both directions and record the result."""
    return await asyncio.to_thread(_verify, request.link)
