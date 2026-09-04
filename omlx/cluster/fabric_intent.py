# SPDX-License-Identifier: Apache-2.0
"""Recorded fabric-addressing intent: the subnet the cluster settled on, durably.

A good address assignment must survive reboots and oMLX's own re-runs.
``transport.configure_link`` records the subnet it settled on here — with
provenance: who chose it, why, and how it was actually applied — and treats
the record as authoritative on its next run, re-applying it instead of
re-addressing, for as long as it still passes the collision check.

``addressing`` matters downstream: a ``networksetup`` assignment is owned by a
network service and survives reboot on its own, while a raw ``ifconfig`` one
drifts on reboot and needs the watchdog (C5) to re-apply this very record.
"""

from __future__ import annotations

import ipaddress
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_CHOSEN_BY = ("auto", "doctor", "user")
_ADDRESSING = ("networksetup", "ifconfig")


@dataclass(frozen=True)
class FabricIntent:
    """One durable record of the fabric subnet the cluster settled on."""

    subnet: str  # "172.16.99.0/24"
    hosts: tuple[str, str]
    chosen_by: str  # "auto" | "doctor" | "user"
    reason: str  # e.g. "vpn_exclusion", "collision_free_default"
    recorded_at: float
    addressing: str  # "networksetup" | "ifconfig"  (what was actually applied)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hosts"] = list(self.hosts)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FabricIntent":
        if not isinstance(data, dict):
            raise ValueError("fabric intent record is malformed")
        try:
            subnet = str(data["subnet"])
            # A record whose subnet cannot parse must never steer addressing.
            ipaddress.ip_network(subnet)
            hosts = tuple(str(host) for host in data["hosts"])
            if len(hosts) != 2 or not all(host.strip() for host in hosts):
                raise ValueError("fabric intent must name exactly two hosts")
            chosen_by = str(data["chosen_by"])
            if chosen_by not in _CHOSEN_BY:
                raise ValueError(f"unknown chosen_by: {chosen_by!r}")
            addressing = str(data["addressing"])
            if addressing not in _ADDRESSING:
                raise ValueError(f"unknown addressing: {addressing!r}")
            return cls(
                subnet=subnet,
                hosts=hosts,  # type: ignore[arg-type]
                chosen_by=chosen_by,
                reason=str(data["reason"]),
                recorded_at=float(data["recorded_at"]),
                addressing=addressing,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"fabric intent record is malformed: {exc}") from exc


@dataclass(frozen=True)
class DriftFinding:
    """Live addressing no longer matches the recorded intent (C5 watchdog).

    ``auto_restore`` and ``incident`` are policy *as data* — this module never
    performs the restoration or records the incident itself. Silent re-assert
    is only ever appropriate when the collision check still passes and the
    intent was applied via ``networksetup`` (the service configuration
    persists, so re-asserting it needs no new consent); every other case must
    surface as a WARN incident inviting a consented Fabric Doctor re-address,
    never a silent privileged action.

    ``addressing="networksetup"`` (and therefore ``auto_restore=True``) is
    dormant today: ``configure_link`` split the networksetup write-half out
    into its own not-yet-landed change (#2875 review) rather than enable it
    unvalidated, so every intent this build records is ``"ifconfig"``. Kept
    here (not removed) since the store already accepts both values and this
    is the exact behavior that follow-up needs the moment it ships.
    """

    kind: str  # "address_lost" | "address_changed" | "intent_collides"
    live: str  # what the fabric interfaces carry right now
    expected: str  # the subnet the intent recorded
    auto_restore: bool = False
    incident: str = ""  # WARN copy for the caller; empty when auto_restore


# The design's exact collision copy: the recorded choice is now unusable and
# only a consented re-address can produce a new one.
_COLLISION_INCIDENT = (
    "The link's saved addresses now collide with a new VPN range — "
    "Fabric Doctor needs to pick new ones."
)


def _drift_finding(kind: str, live: str, intent: FabricIntent) -> DriftFinding:
    if kind == "intent_collides":
        return DriftFinding(
            kind=kind,
            live=live,
            expected=intent.subnet,
            auto_restore=False,
            incident=_COLLISION_INCIDENT,
        )
    if intent.addressing == "networksetup":
        # The service configuration persists across reboots; re-asserting the
        # recorded addresses grants nothing new, so the caller may restore
        # silently (configure_link's tier 2 is that re-assert).
        return DriftFinding(
            kind=kind, live=live, expected=intent.subnet, auto_restore=True
        )
    return DriftFinding(
        kind=kind,
        live=live,
        expected=intent.subnet,
        auto_restore=False,
        incident=(
            f"The link's saved fabric addresses ({intent.subnet}) are no "
            "longer applied — addresses set with ifconfig drift on reboot. "
            "Run Fabric Doctor to re-apply them."
        ),
    )


def detect_drift(
    intent: FabricIntent,
    live_interfaces: Iterable[Any],
    *,
    collides: Callable[[ipaddress.IPv4Network], bool] | None = None,
) -> DriftFinding | None:
    """Compare live fabric addressing against the recorded intent. Pure.

    ``live_interfaces`` is the fresh ``probe_host_interfaces`` reading the
    caller already holds (duck-typed ``HostInterfaces``: ``addresses`` entries
    with ``interface``/``address``, plus ``rdma_interfaces`` and
    ``thunderbolt_interfaces`` naming the fabric-capable interfaces).
    ``collides`` is the caller's collision check over everything *else* the
    hosts carry or route (``hostile_networks`` minus the intent's own subnet,
    exactly like ``configure_link``'s ``own_networks`` filter — passing the
    raw hostile set would make a healthy link "collide" with itself).

    A collision outranks the address comparison: a collided intent must never
    be re-applied, silently or otherwise, so ``intent_collides`` wins even
    when the addresses are also missing or changed.

    Explicit non-goal: no event-driven wake/network-change watcher — this is
    deliberately poll-driven and catches drift within one dashboard tick.
    """

    try:
        network = ipaddress.ip_network(intent.subnet)
    except ValueError:
        # A record whose subnet cannot parse must never steer addressing;
        # from_dict refuses these, so this is belt-and-braces only.
        return None

    hosts = tuple(live_interfaces)  # tolerate one-shot iterables
    if collides is not None and collides(network):  # type: ignore[arg-type]
        return _drift_finding("intent_collides", _live_summary(hosts), intent)

    fabric_addresses: list[tuple[str, str]] = []
    for host in hosts:
        fabric_interfaces = frozenset(
            getattr(host, "rdma_interfaces", ()) or ()
        ) | frozenset(getattr(host, "thunderbolt_interfaces", ()) or ())
        for entry in getattr(host, "addresses", ()) or ():
            if entry.interface not in fabric_interfaces:
                continue
            fabric_addresses.append((entry.interface, entry.address))

    for _, address in fabric_addresses:
        with suppress(ValueError):
            if ipaddress.ip_address(address) in network:
                return None  # live matches intent
    if fabric_addresses:
        live = ", ".join(
            f"{interface} {address}" for interface, address in fabric_addresses
        )
        return _drift_finding("address_changed", live, intent)
    return _drift_finding("address_lost", "", intent)


def _live_summary(live_interfaces: Iterable[Any]) -> str:
    parts = [
        f"{entry.interface} {entry.address}"
        for host in live_interfaces
        for entry in (getattr(host, "addresses", ()) or ())
    ]
    return ", ".join(parts)


class FabricIntentStore:
    """Thread-safe, persisted single-record store: the latest intent wins.

    Unlike the incident ring this is not a history — a re-address supersedes
    the previous choice entirely, and only the current choice may steer
    addressing. A corrupt or unreadable file fails closed (no intent, the
    parse error kept in ``load_error``) so a damaged record can never poison
    link setup; the next successful ``record`` overwrites it atomically.
    """

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self.path = self.base_path / "cluster" / "fabric-intent.json"
        self._lock = threading.RLock()
        self._intent: FabricIntent | None = None
        self.load_error: str | None = None
        try:
            self._load()
        except ValueError as exc:
            self._intent = None
            self.load_error = str(exc)

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._intent = None
                return
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"could not read fabric intent record: {exc}"
                ) from exc
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError("unsupported fabric intent schema")
            raw_intent = payload.get("intent")
            self._intent = (
                FabricIntent.from_dict(raw_intent) if raw_intent is not None else None
            )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "intent": self._intent.to_dict() if self._intent else None,
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".fabric-intent.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)

    def record(
        self,
        *,
        subnet: str,
        hosts: tuple[str, str],
        chosen_by: str,
        reason: str,
        addressing: str,
    ) -> FabricIntent:
        """Replace the record; everything is validated before anything writes."""

        intent = FabricIntent.from_dict(
            {
                "subnet": subnet,
                "hosts": list(hosts),
                "chosen_by": chosen_by,
                "reason": reason,
                "recorded_at": time.time(),
                "addressing": addressing,
            }
        )
        with self._lock:
            previous = self._intent
            self._intent = intent
            try:
                self._save()
            except Exception:
                self._intent = previous
                raise
            self.load_error = None
            return intent

    def current(self) -> FabricIntent | None:
        with self._lock:
            return self._intent

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": 1,
                "intent": self._intent.to_dict() if self._intent else None,
                "load_error": self.load_error,
            }


_intent_lock = threading.Lock()
_configured_intent: FabricIntentStore | None = None


def configure_fabric_intent(base_path: Path) -> FabricIntentStore:
    global _configured_intent
    with _intent_lock:
        _configured_intent = FabricIntentStore(base_path)
        return _configured_intent


def get_fabric_intent() -> FabricIntentStore:
    with _intent_lock:
        if _configured_intent is None:
            raise RuntimeError("fabric intent store is not configured")
        return _configured_intent
