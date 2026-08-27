# SPDX-License-Identifier: Apache-2.0
"""Cluster v2 discovery/identity HTTP endpoints.

``discovery_router`` is mounted WITHOUT ``require_admin`` in
``omlx/server.py`` so peers can probe ``/api/cluster/node_id`` before any
trust exists; it is still gated by the distributed-inference exposure flag
(``require_distributed_inference_enabled``) like the enrollment router. The
probe endpoint is rate-limited per source IP. ``/api/cluster/devices``
carries the trusted inventory and requires admin like the rest of the
cluster surface.
"""

from __future__ import annotations

import asyncio
import ipaddress
import threading
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from .._version import __version__
from ..admin.auth import require_admin
from .discovery import local_addr_dicts
from .identity import get_node_identity
from .registry import get_device_registry

discovery_router = APIRouter(prefix="/api/cluster", tags=["cluster-discovery"])


class ProbeRateLimiter:
    """Token-bucket rate limiter for the unauthenticated probe endpoint."""

    def __init__(self, rate_per_second: float = 5.0, burst: int = 10) -> None:
        self._rate = float(rate_per_second)
        self._burst = int(burst)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, updated = self._buckets.get(key, (float(self._burst), now))
            tokens = min(
                float(self._burst), tokens + (now - updated) * self._rate
            )
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            # Bound the map: drop idle buckets once it grows past 4096 keys.
            if len(self._buckets) > 4096:
                self._buckets = {
                    k: v
                    for k, v in self._buckets.items()
                    if now - v[1] < 600.0
                }
            return True


probe_rate_limiter = ProbeRateLimiter()


def discovery_service_or_none():
    from .discovery import get_discovery_service

    try:
        return get_discovery_service()
    except RuntimeError:
        return None


def _cluster_name() -> str:
    service = discovery_service_or_none()
    if service is not None:
        return service.config.cluster_name
    # Discovery disabled still reports the persisted settings-level name.
    from .discovery import load_cluster_name

    return load_cluster_name()


@discovery_router.get("/node_id")
async def cluster_node_id_probe(request: Request):
    """Public, rate-limited identity probe used by peer verification.

    Deliberately unauthenticated: a discovering node must be able to confirm
    an announced address belongs to the announced node_id before any pairing
    trust exists. It reveals only the stable node_id, the oMLX version, and
    the cluster name — no capabilities, no device inventory.
    """

    client = request.client.host if request.client else "unknown"
    if not probe_rate_limiter.allow(client):
        raise HTTPException(
            status_code=429,
            detail="probe rate limit exceeded",
            headers={"Retry-After": "1"},
        )
    try:
        identity = get_node_identity()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="cluster identity is not configured"
        ) from exc
    return {
        "node_id": identity.node_id,
        "version": __version__,
        "cluster_name": _cluster_name(),
    }


@discovery_router.get("/discovery/health")
async def cluster_discovery_health(is_admin: bool = Depends(require_admin)):
    """Local-network self-test for the wizard's checks row.

    ``multicast_rx_within_5s`` is False when no foreign HELLO arrived in the
    last five seconds — on macOS that is how a denied Local Network
    permission presents, so the UI pairs it with actionable guidance instead
    of a silently empty device list. Shape is pinned by the wizard fixture
    (tests/ui/fixtures/cluster_v2/discovery_health_ok.json).
    """

    service = discovery_service_or_none()
    if service is None:
        return {
            "multicast_rx_within_5s": False,
            "last_multicast_rx_at": None,
            "mdns_active": False,
            "transport": "disabled",
        }
    return {
        "multicast_rx_within_5s": bool(service.multicast_ok),
        "last_multicast_rx_at": service.last_multicast_rx_at,
        "mdns_active": bool(service.mdns_active),
        "transport": service.transport_summary(),
    }


@discovery_router.get("/discovery/health/detail")
async def cluster_discovery_health_detail(
    is_admin: bool = Depends(require_admin),
):
    """Extended discovery self-diagnostics (loop liveness, TX health).

    Separate route so the pinned wizard fixture shape on
    ``/discovery/health`` stays untouched. This is how "no peers nearby" is
    distinguished from "discovery loop dead / socket wedged / interface
    renumbered underneath us" in the field.
    """

    service = discovery_service_or_none()
    if service is None:
        raise HTTPException(status_code=503, detail="discovery is disabled")
    return service.health()


@discovery_router.post("/devices/manual")
async def cluster_add_manual_peer(
    request: Request, is_admin: bool = Depends(require_admin)
):
    """Manually add a peer by IP — the deterministic path over Thunderbolt.

    Multicast is best-effort on macOS (Local Network permission, interface
    renumbering, routers that filter v6 multicast). When two machines share
    a direct Thunderbolt link, pointing each node at the other's link
    address (typically the 10.0.0.x pair) skips all of that. The candidate
    is verified through the same HTTP probe path as multicast-discovered
    peers before it is returned as verified.
    """

    service = discovery_service_or_none()
    if service is None:
        raise HTTPException(status_code=503, detail="discovery is disabled")
    client = request.client.host if request.client else "unknown"
    if not probe_rate_limiter.allow(client):
        raise HTTPException(
            status_code=429,
            detail="probe rate limit exceeded",
            headers={"Retry-After": "1"},
        )
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="invalid JSON body")
    raw_ip = body.get("ip")
    try:
        ip = str(ipaddress.ip_address(str(raw_ip)))
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"invalid IP address: {raw_ip!r}"
        )
    port = body.get("port", 8000)
    if not isinstance(port, int) or isinstance(port, bool) or not (
        1 <= port <= 65535
    ):
        raise HTTPException(status_code=400, detail="invalid port")
    service.add_manual(ip, port)
    # Probe synchronously (bounded by the configured probe timeout) so the
    # caller learns immediately whether the address answers as an oMLX node.
    await asyncio.to_thread(service.probe_candidate_now, ip, port)
    peer = next(
        (
            p
            for p in service.peers()
            if any(a["ip"] == ip for a in p.addrs)
        ),
        None,
    )
    return {
        "ok": True,
        "ip": ip,
        "port": port,
        "verified": peer is not None,
        "peer": peer.to_dict() if peer is not None else None,
    }


def _enrich_paired_row(
    row: dict[str, Any], record: dict[str, Any] | None
) -> None:
    """Fill gaps on a paired row from the matching discovery record.

    Pairings completed before capabilities were exchanged persisted empty
    ``caps``; the matching registry/service record carries the announced
    values. Only fields actually known are merged — nothing is fabricated —
    and the persisted row is untouched (the route returns copies). The row
    also gains a normalized ``addrs`` list of ``{"ip", "if_type"}`` dicts,
    sourced from ``last_addrs`` plus the discovered record, so the UI's
    address pickers read paired rows the same way as discovered rows.
    """

    addrs: list[dict[str, str]] = []
    for ip in row.get("last_addrs") or []:
        if isinstance(ip, str) and ip and all(a["ip"] != ip for a in addrs):
            addrs.append({"ip": ip, "if_type": "paired"})
    if record is not None:
        for addr in record.get("addrs") or []:
            if not isinstance(addr, dict) or not addr.get("ip"):
                continue
            existing = next(
                (a for a in addrs if a["ip"] == addr["ip"]), None
            )
            if existing is not None:
                # The discovered entry knows the real interface type.
                existing["if_type"] = addr.get("if_type") or existing["if_type"]
            elif len(addrs) < 16:
                addrs.append(
                    {
                        "ip": addr["ip"],
                        "if_type": addr.get("if_type") or "discovered",
                    }
                )
    if addrs:
        row["addrs"] = addrs
    # Persisted membership is not liveness. Until this process has observed the
    # peer, render an amber/suspect shell instead of the old implicit green row.
    row.setdefault("http_port", 0)
    if record is None:
        row["state"] = "suspect"
        row["last_seen"] = None
        row.setdefault("link", "unknown")
        return
    caps = row.get("caps") or {}
    if not caps.get("ram_gb") or not caps.get("chip"):
        known_caps = record.get("caps") or {}
        if known_caps.get("ram_gb") or known_caps.get("chip"):
            row["caps"] = dict(known_caps)
    for key in ("version", "friendly_name"):
        if record.get(key):
            row[key] = record[key]
    state = record.get("state")
    row["state"] = (
        state if state in {"discovered", "suspect", "dead"} else "suspect"
    )
    last_seen = record.get("last_seen")
    row["last_seen"] = (
        float(last_seen)
        if isinstance(last_seen, (int, float)) and not isinstance(last_seen, bool)
        else None
    )
    row["link"] = str(record.get("link") or "unknown")
    http_port = record.get("http_port")
    if (
        isinstance(http_port, int)
        and not isinstance(http_port, bool)
        and 1 <= http_port <= 65535
    ):
        row["http_port"] = http_port


@discovery_router.get("/devices")
async def cluster_devices(is_admin: bool = Depends(require_admin)):
    """Cluster device inventory for the wizard UI.

    ``multicast_ok`` is the macOS Local Network permission signal: it is
    False when no foreign HELLO was received in the last 5 seconds, which is
    how a denied/blocked local-network permission presents. The UI should
    surface "check System Settings → Privacy & Security → Local Network"
    rather than showing a silently empty device list.
    """

    try:
        identity = get_node_identity()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503, detail="cluster identity is not configured"
        ) from exc

    try:
        registry = get_device_registry()
    except RuntimeError:
        registry = None
    service = discovery_service_or_none()

    paired: list[dict[str, Any]] = registry.paired() if registry else []
    paired_ids = {row["node_id"] for row in paired}
    # Membership in this list IS the paired state, but the registry rows do
    # not carry the flag themselves -- the self record below sets it
    # explicitly. Without it, wizard consumers keyed on device.paired (the
    # per-Mac role list, the card's Pair/Paired state) treat an already
    # paired peer as unpaired.
    for row in paired:
        row["paired"] = True

    # Seam with Module B's enrollment: a paired device that completed SSH
    # TOFU enrollment carries its enrolled ssh_target, so the UI probes and
    # plans against the enrolled target instead of guessing from probe IPs.
    try:
        from .enrollment import get_cluster_enrollment

        enrollment = get_cluster_enrollment()
    except RuntimeError:
        enrollment = None
    if enrollment is not None:
        enrolled = {node.node_id: node for node in enrollment.list_nodes()}
        for row in paired:
            node = enrolled.get(row.get("node_id"))
            if node is not None:
                row["ssh_target"] = node.ssh
    observed_records: dict[str, dict[str, Any]] = {}
    if registry:
        for entry in registry.discovered():
            observed_records[entry["node_id"]] = entry
    if service is not None:
        for peer in service.peers():
            # Paired peers still belong in the observation map: their live
            # state, timestamp, port and interface metadata enrich the durable
            # paired shell. They are filtered only from the unpaired list.
            observed_records[peer.node_id] = peer.to_dict()
    discovered_records = {
        node_id: record
        for node_id, record in observed_records.items()
        if node_id not in paired_ids and not record.get("paired")
    }

    # Devices paired before caps were exchanged carry empty caps on disk;
    # enrich them from what discovery has actually seen before suppressing
    # their node_ids from the discovered list below.
    for row in paired:
        _enrich_paired_row(row, observed_records.get(row.get("node_id")))

    # Nothing flips the discovery service's in-memory ``PeerRecord.paired``
    # the moment pairing completes, so a stale (possibly dead) record for a
    # now-paired node_id would otherwise render a zombie "Pair" card next
    # to the paired row. Paired node_ids never appear as discovered.
    for node_id in paired_ids:
        discovered_records.pop(node_id, None)

    # Seam with Module B: a posted pair/request must surface in the device
    # list as an awaiting_approval row so the wizard renders code entry +
    # approve/deny (fixture: tests/ui/.../devices_pending_approval.json).
    # Pending state wins over a plain discovered record for the same node.
    try:
        from .pairing import get_pairing_manager

        pairing_manager = get_pairing_manager()
    except RuntimeError:
        pairing_manager = None
    if pairing_manager is not None:
        for pending in pairing_manager.pending_requests():
            row = dict(discovered_records.get(pending["node_id"], {}))
            row.update(pending)
            row["state"] = "awaiting_approval"
            discovered_records[pending["node_id"]] = row

    self_record: dict[str, Any] = {
        "node_id": identity.node_id,
        "friendly_name": identity.friendly_name,
        "version": __version__,
        "cluster_name": _cluster_name(),
        "caps": service.config.caps.to_dict() if service is not None else {},
        "addrs": local_addr_dicts(),
        "http_port": service.config.http_port if service is not None else 0,
        "paired": True,
        "last_seen": time.time(),
        "link": "unknown",
        "state": "discovered",
    }
    return {
        "paired": paired,
        "discovered": list(discovered_records.values()),
        "self": self_record,
        "multicast_ok": bool(service.multicast_ok) if service else False,
        "mdns_available": bool(service.mdns_available) if service else False,
    }
