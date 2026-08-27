# SPDX-License-Identifier: Apache-2.0
"""Cluster v2 pairing endpoints (Module B).

Two routers, both under ``/api/cluster`` (the v2 surface; the legacy 3-step
copy-paste endpoints stay untouched under ``/admin/api/cluster`` as
"Advanced (legacy)"):

* ``pair_router`` — unauthenticated joiner-facing endpoints.  Mounted like
  ``join_router`` (public but pinned): the pair request carries only
  ``blake2s(code + node_id)``, and the served cluster key is encrypted under
  a PBKDF2 key derived from the code, so the admin's approve step remains
  the sole trust decision.
* ``pair_admin_router`` — admin-facing approve/deny/unpair plus the joiner
  side of the flow (``/pair/join`` begin/poll/cancel), which mutates this
  node's own local join state.  Mounted with ``Depends(require_admin)``
  exactly like the existing cluster router.

Both are wired in ``omlx/server.py:_register_cluster_routes``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .pairing import (
    CODE_DIGITS,
    EnrollmentDriveError,
    PairingCodeError,
    PairingError,
    PairingExpiredError,
    PairingLockoutError,
    PairingManager,
    PairingRequestError,
    PairingStateError,
    get_pairing_manager,
)

pair_router = APIRouter(prefix="/api/cluster", tags=["cluster-v2-pairing"])
pair_admin_router = APIRouter(prefix="/api/cluster", tags=["cluster-v2-pairing-admin"])

_get_pairing_manager: Any = get_pairing_manager


def set_pairing_manager_getter(getter: Any) -> None:
    """Inject the server-owned manager without importing ``omlx.server``."""

    global _get_pairing_manager
    _get_pairing_manager = getter


def _manager() -> PairingManager:
    try:
        return _get_pairing_manager()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


class PairRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=255)
    friendly_name: str = Field(min_length=1, max_length=255)
    caps: dict[str, Any] = Field(default_factory=dict)
    code_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    http_port: int | None = Field(default=None, ge=1, le=65535)
    addrs: list[str] = Field(default_factory=list, max_length=8)
    ssh_public_key: str | None = Field(default=None, max_length=8192)


class PairApproveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=255)
    code: str = Field(pattern=rf"^[0-9]{{{CODE_DIGITS}}}$")


class PairDenyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=255)


class PairJoinBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coordinator_addr: str = Field(min_length=1, max_length=255)


def _pairing_http_error(exc: PairingError) -> HTTPException:
    if isinstance(exc, PairingLockoutError):
        return HTTPException(status_code=423, detail=str(exc))
    if isinstance(exc, PairingExpiredError):
        return HTTPException(status_code=410, detail=str(exc))
    if isinstance(exc, PairingCodeError):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, EnrollmentDriveError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, PairingStateError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


@pair_router.post("/pair/request", status_code=202)
async def cluster_pair_request(body: PairRequestBody):
    """Register a joiner's request as awaiting_approval (code never crosses)."""

    manager = _manager()
    try:
        snapshot = await asyncio.to_thread(
            manager.handle_join_request, body.model_dump()
        )
    except PairingError as exc:
        raise _pairing_http_error(exc) from exc
    return snapshot


@pair_router.get("/pair/status/{node_id}")
async def cluster_pair_status(node_id: str):
    """Joiner poll: pending/approved/denied plus the wrapped cluster key."""

    manager = _manager()
    if not node_id or len(node_id) > 255:
        raise HTTPException(status_code=400, detail="invalid node_id")
    return await asyncio.to_thread(manager.join_status, node_id)


@pair_admin_router.post("/pair/approve")
async def cluster_pair_approve(body: PairApproveBody):
    """Type the code shown on the joiner; verify, issue cluster key, enroll."""

    manager = _manager()
    try:
        result = await asyncio.to_thread(manager.approve, body.node_id, body.code)
    except PairingError as exc:
        raise _pairing_http_error(exc) from exc
    # Flip the in-memory discovery record so the freshly paired peer stops
    # showing as discovered before its next announcement refresh.
    try:
        from .discovery import get_discovery_service

        get_discovery_service().mark_paired(body.node_id)
    except Exception:  # discovery disabled or not configured — harmless
        pass
    return result


@pair_admin_router.post("/pair/deny")
async def cluster_pair_deny(body: PairDenyBody):
    """Refuse a pending join request (audit-logged)."""

    manager = _manager()
    try:
        denied = await asyncio.to_thread(manager.deny, body.node_id)
    except PairingError as exc:
        raise _pairing_http_error(exc) from exc
    if not denied:
        raise HTTPException(status_code=404, detail="no pending join request")
    return {"ok": True, "node_id": body.node_id, "state": "denied"}


@pair_admin_router.post("/pair/join")
async def cluster_pair_join(body: PairJoinBody):
    """Joiner side: show a 6-digit code here, the other Mac approves it.

    Mints the code, POSTs the pair/request to ``coordinator_addr``, and
    returns the local join snapshot; the UI then polls ``GET /pair/join``.
    """

    manager = _manager()
    try:
        return await asyncio.to_thread(manager.begin_join, body.coordinator_addr)
    except PairingError as exc:
        raise _pairing_http_error(exc) from exc


@pair_admin_router.get("/pair/join")
async def cluster_pair_join_state():
    """Local join snapshot for the wizard; polling drives approval completion."""

    manager = _manager()
    return await asyncio.to_thread(manager.poll_join_once)


@pair_admin_router.post("/pair/join/cancel")
async def cluster_pair_join_cancel():
    """Abandon this node's join in progress (idempotent)."""

    manager = _manager()
    return await asyncio.to_thread(manager.cancel_join)


@pair_admin_router.delete("/devices/{node_id}")
async def cluster_unpair_device(node_id: str):
    """Unpair: drop the registry entry, the cluster key, and SSH trust."""

    manager = _manager()
    if not node_id or len(node_id) > 255:
        raise HTTPException(status_code=400, detail="invalid node_id")
    try:
        return await asyncio.to_thread(manager.unpair, node_id)
    except PairingError as exc:
        raise _pairing_http_error(exc) from exc
