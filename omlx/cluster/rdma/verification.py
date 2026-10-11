# SPDX-License-Identifier: Apache-2.0
"""Evidence that a link moved verified bytes both ways, and when that evidence expires."""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .daemon import DaemonStatus
from .links import RdmaLink

# Evidence older than a day is re-earned rather than trusted.
MAX_AGE_S = 24 * 60 * 60
_KMUTIL = (
    "/usr/bin/kmutil",
    "showloaded",
    "--list-only",
    "--bundle-identifier",
    "org.mcdma.cx5.native",
)
_LOADED = re.compile(r"org\.mcdma\.cx5\.native \(([^)]+)\) ([0-9A-Fa-f-]{36})")
# Dashboard polling re-reads the loaded driver at most this often.
_DRIVER_TTL_S = 60.0
_driver_lock = threading.Lock()
_driver_cache: tuple[float, DriverIdentity | None] | None = None


@dataclass(frozen=True)
class ProbeMeasurements:
    """What one live probe measured on a link."""

    round_trips: int
    latency_p50_us: float
    latency_p99_us: float
    to_peer_bytes: int
    to_peer_gbit_s: float
    from_peer_bytes: int
    from_peer_gbit_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_trips": self.round_trips,
            "latency_p50_us": self.latency_p50_us,
            "latency_p99_us": self.latency_p99_us,
            "to_peer_bytes": self.to_peer_bytes,
            "to_peer_gbit_s": self.to_peer_gbit_s,
            "from_peer_bytes": self.from_peer_bytes,
            "from_peer_gbit_s": self.from_peer_gbit_s,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProbeMeasurements:
        return cls(
            round_trips=int(payload["round_trips"]),
            latency_p50_us=float(payload["latency_p50_us"]),
            latency_p99_us=float(payload["latency_p99_us"]),
            to_peer_bytes=int(payload["to_peer_bytes"]),
            to_peer_gbit_s=float(payload["to_peer_gbit_s"]),
            from_peer_bytes=int(payload["from_peer_bytes"]),
            from_peer_gbit_s=float(payload["from_peer_gbit_s"]),
        )


@dataclass(frozen=True)
class LinkVerification:
    """The outcome of verifying one link, with the identity it was earned under."""

    link: str
    peer_node_id: str
    verified: bool
    reason: str
    checked_at: float
    identity: dict[str, str] = field(default_factory=dict)
    measurements: ProbeMeasurements | None = None

    def stale_reason(
        self, identity: dict[str, str], now: float, max_age_s: float = MAX_AGE_S
    ) -> str | None:
        """Why this evidence no longer covers the link, or None if it still does."""
        if not self.verified:
            return self.reason or "the last verification failed"
        if now - self.checked_at > max_age_s:
            return "the last verification is more than a day old"
        changed = sorted(
            key
            for key in identity.keys() | self.identity.keys()
            if identity.get(key) != self.identity.get(key)
        )
        if changed:
            return "the link changed since it was verified: " + ", ".join(changed)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "link": self.link,
            "peer_node_id": self.peer_node_id,
            "verified": self.verified,
            "reason": self.reason,
            "checked_at": self.checked_at,
            "identity": dict(sorted(self.identity.items())),
            "measurements": self.measurements.to_dict() if self.measurements else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LinkVerification:
        measurements = payload.get("measurements")
        identity = payload.get("identity") or {}
        if not isinstance(identity, dict):
            raise ValueError("link verification identity must be an object")
        return cls(
            link=str(payload["link"]),
            peer_node_id=str(payload["peer_node_id"]),
            verified=bool(payload["verified"]),
            reason=str(payload.get("reason", "")),
            checked_at=float(payload["checked_at"]),
            identity={str(key): str(value) for key, value in identity.items()},
            measurements=ProbeMeasurements.from_dict(measurements)
            if measurements
            else None,
        )


@dataclass(frozen=True)
class DriverIdentity:
    """The MCDMA kernel driver loaded on this Mac."""

    version: str
    uuid: str


def read_driver_identity(
    run: Callable[..., Any] = subprocess.run, *, platform: str = sys.platform
) -> DriverIdentity | None:
    """The loaded MCDMA driver's version and UUID, or None when it is not loaded."""
    if platform != "darwin":
        return None
    try:
        result = run(
            list(_KMUTIL), capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    found = _LOADED.search(result.stdout or "")
    return DriverIdentity(found.group(1), found.group(2).upper()) if found else None


def cached_driver_identity(
    clock: Callable[[], float] = time.monotonic,
) -> DriverIdentity | None:
    """The loaded driver through a one-minute cache, so polling never runs kmutil."""
    global _driver_cache
    with _driver_lock:
        now = clock()
        if _driver_cache is None or now - _driver_cache[0] > _DRIVER_TTL_S:
            _driver_cache = (now, read_driver_identity())
        return _driver_cache[1]


def link_identity(
    link: RdmaLink, status: DaemonStatus, driver: DriverIdentity | None
) -> dict[str, str]:
    """Everything whose change invalidates a verification of `link`."""
    identity = {
        "daemon_version": status.version or "unknown",
        "peer_host": link.peer_host or "",
        "peer_node_id": link.peer_node_id or "",
        "device": link.device or "",
        "request_bytes": str(link.request_bytes or 0),
        "reply_bytes": str(link.reply_bytes or 0),
        "connected_since": f"{link.since:.0f}" if link.since is not None else "",
    }
    if driver is not None:
        identity["driver_version"] = driver.version
        identity["driver_uuid"] = driver.uuid
    return identity
