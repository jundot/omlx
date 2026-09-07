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
