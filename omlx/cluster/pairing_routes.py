# SPDX-License-Identifier: Apache-2.0
"""Cluster v2 pairing endpoints (Module B).

Two routers, both under ``/api/cluster`` (the v2 surface; the legacy 3-step
copy-paste endpoints stay untouched under ``/admin/api/cluster`` as
"Advanced (legacy)"):

* ``pair_router`` — unauthenticated joiner-facing endpoints. The pair request
  carries a salted PBKDF2 verifier binding the node and both SSH identities,
  and the coordinator binds enrollment to the HTTP source address. The served
  cluster key is separately encrypted under a code-derived key, so the admin's
  approve step remains the sole trust decision.
* ``pair_admin_router`` — admin-facing approve/deny/unpair.  Mounted with
  ``Depends(require_admin)`` exactly like the existing cluster router.

Both are wired in ``omlx/server.py:_register_cluster_routes``.
"""

from __future__ import annotations

import asyncio
import ipaddress
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .discovery_routes import ProbeRateLimiter
from .pairing import (
    CODE_DIGITS,
    EnrollmentDriveError,
    PairingCodeError,
    PairingError,
    PairingExpiredError,
    PairingLockoutError,
    PairingManager,
    PairingStateError,
    get_pairing_manager,
)

pair_router = APIRouter(prefix="/api/cluster", tags=["cluster-v2-pairing"])
pair_admin_router = APIRouter(prefix="/api/cluster", tags=["cluster-v2-pairing-admin"])

_get_pairing_manager: Any = get_pairing_manager
pair_request_rate_limiter = ProbeRateLimiter(rate_per_second=0.5, burst=8)
pair_status_rate_limiter = ProbeRateLimiter(rate_per_second=5.0, burst=20)


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
    code_salt: str = Field(min_length=24, max_length=64)
    http_port: int | None = Field(default=None, ge=1, le=65535)
    addrs: list[str] = Field(default_factory=list, max_length=8)
    ssh_public_key: str = Field(min_length=32, max_length=8192)
    ssh_host_public_key: str = Field(min_length=32, max_length=8192)


class PairApproveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=255)
    code: str = Field(pattern=rf"^[0-9]{{{CODE_DIGITS}}}$")


class PairDenyBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=255)


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
async def cluster_pair_request(body: PairRequestBody, request: Request):
    """Register a joiner's request as awaiting_approval (code never crosses)."""

    manager = _manager()
    payload = body.model_dump()
    source = request.client.host if request.client is not None else ""
    if not pair_request_rate_limiter.allow(source or "unknown"):
        raise HTTPException(status_code=429, detail="pair request rate limit exceeded")
    try:
        ipaddress.ip_address(source.split("%", 1)[0])
    except ValueError:
        payload["addrs"] = []
    else:
        # Never enroll an arbitrary address supplied in the public JSON body.
        # The HTTP source is the only coordinator-observed endpoint.
        payload["addrs"] = [source]
    try:
        snapshot = await asyncio.to_thread(manager.handle_join_request, payload)
    except PairingError as exc:
        raise _pairing_http_error(exc) from exc
    return snapshot


@pair_router.get("/pair/status/{node_id}")
async def cluster_pair_status(node_id: str, request: Request):
    """Joiner poll: pending/approved/denied plus the wrapped cluster key."""

    manager = _manager()
    source = request.client.host if request.client is not None else ""
    if not pair_status_rate_limiter.allow(source or "unknown"):
        raise HTTPException(status_code=429, detail="pair status rate limit exceeded")
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
    try:
        from .discovery import get_discovery_service

        get_discovery_service().mark_paired(body.node_id)
    except Exception:
        # Pairing remains authoritative when discovery is disabled or stopped.
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
