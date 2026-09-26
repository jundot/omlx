# SPDX-License-Identifier: Apache-2.0
"""Atomic persistence for user-approved distributed model deployments."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from .deployment import ClusterDeployment

# Version 2 persists per-node model paths (``path_map`` on each deployment).
# Version 1 files are migrated in memory on load and rewritten atomically.
REGISTRY_SCHEMA_VERSION = 2
_SUPPORTED_REGISTRY_SCHEMAS = (1, REGISTRY_SCHEMA_VERSION)


def _model_key(model: str) -> str:
    path = Path(model).expanduser()
    if path.exists():
        return str(path.resolve())
    return model.strip()


class ClusterRegistry:
    """Thread-safe deployment registry with no credentials or private keys."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self.path = self.base_path / "cluster" / "deployments.json"
        self._lock = threading.RLock()
        self._deployments: dict[str, ClusterDeployment] = {}
        self.load_error: str | None = None
        #: Set when an on-disk legacy schema was upgraded on load.
        self.migrated_from: int | None = None
        try:
            self._load()
        except ValueError as exc:
            # A corrupt optional cluster file must never prevent the local
            # oMLX server and GUI from starting. Fail closed: activate none.
            self._deployments = {}
            self.load_error = str(exc)

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._deployments = {}
                return
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"could not read cluster deployment registry: {exc}"
                ) from exc
            if not isinstance(payload, dict) or payload.get("schema_version") not in (
                _SUPPORTED_REGISTRY_SCHEMAS
            ):
                raise ValueError("unsupported cluster deployment registry schema")
            raw_deployments = payload.get("deployments")
            if not isinstance(raw_deployments, list):
                raise ValueError("cluster deployment registry is malformed")
            deployments = [
                ClusterDeployment.from_dict(item) for item in raw_deployments
            ]
            self._deployments = {
                _model_key(deployment.model): deployment for deployment in deployments
            }
            if len(self._deployments) != len(deployments):
                raise ValueError("cluster deployment registry has duplicate models")
            if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
                # Legacy files hold no per-node paths; every deployment decoded
                # above with an empty path_map, which is the exact behavior the
                # file described. Persist the upgrade so the migration happens
                # once, but never let an unwritable file block server startup.
                self.migrated_from = int(payload.get("schema_version", 1))
                with suppress(OSError):
                    self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "deployments": [
                deployment.to_dict()
                for _, deployment in sorted(self._deployments.items())
            ],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".deployments.",
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

    def list(self) -> tuple[ClusterDeployment, ...]:
        with self._lock:
            return tuple(
                deployment for _, deployment in sorted(self._deployments.items())
            )

    def get_for_model(self, model: str) -> ClusterDeployment | None:
        with self._lock:
            return self._deployments.get(_model_key(model))

    def get(self, deployment_id: str) -> ClusterDeployment | None:
        with self._lock:
            return next(
                (
                    deployment
                    for deployment in self._deployments.values()
                    if deployment.deployment_id == deployment_id
                ),
                None,
            )

    def upsert(self, deployment: ClusterDeployment) -> None:
        with self._lock:
            duplicate = self.get(deployment.deployment_id)
            if duplicate is not None and _model_key(duplicate.model) != _model_key(
                deployment.model
            ):
                raise ValueError(
                    f"deployment ID {deployment.deployment_id!r} is already in use"
                )
            previous = dict(self._deployments)
            self._deployments[_model_key(deployment.model)] = deployment
            try:
                self._save()
            except Exception:
                self._deployments = previous
                raise
            self.load_error = None

    def remove(self, deployment_id: str) -> bool:
        with self._lock:
            key = next(
                (
                    key
                    for key, deployment in self._deployments.items()
                    if deployment.deployment_id == deployment_id
                ),
                None,
            )
            if key is None:
                return False
            previous = dict(self._deployments)
            del self._deployments[key]
            try:
                self._save()
            except Exception:
                self._deployments = previous
                raise
            return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "deployments": [deployment.to_dict() for deployment in self.list()],
            "load_error": self.load_error,
            "migrated_from": self.migrated_from,
        }


_registry_lock = threading.Lock()
_configured_registry: ClusterRegistry | None = None


def configure_cluster_registry(base_path: Path) -> ClusterRegistry:
    global _configured_registry
    with _registry_lock:
        _configured_registry = ClusterRegistry(base_path)
        return _configured_registry


def get_cluster_registry() -> ClusterRegistry:
    with _registry_lock:
        if _configured_registry is None:
            raise RuntimeError("cluster registry is not configured")
        return _configured_registry


# ---------------------------------------------------------------------------
# Cluster v2: trusted device inventory (paired nodes) + discovered device cache
# ---------------------------------------------------------------------------

import time  # noqa: E402


def default_devices_path() -> Path:
    """Default on-disk location, honoring the OMLX_BASE_PATH override."""

    env_value = os.environ.get("OMLX_BASE_PATH")
    base = Path(env_value).expanduser() if env_value else Path.home() / ".omlx"
    return base / "cluster" / "devices.json"


class DeviceRegistry:
    """Inventory of cluster devices.

    *Paired* devices are trusted and persisted atomically (mode 0600) to
    ``devices.json`` (schema_version 1). *Discovered-but-unpaired* devices
    are kept in memory only — an untrusted announcer must never gain a
    persisted foothold merely by showing up on the LAN.

    The merge API is consumed by both discovery (continuous upserts of
    announced metadata) and pairing (marking a node trusted). Merge never
    downgrades a paired device back to discovered, and dedupes on
    ``node_id``.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else default_devices_path()
        self._lock = threading.RLock()
        self._paired: dict[str, dict[str, Any]] = {}
        self._discovered: dict[str, dict[str, Any]] = {}
        self.load_error: str | None = None
        try:
            self._load()
        except ValueError as exc:
            # Mirror ClusterRegistry: a corrupt optional file must never
            # prevent the server from starting. Fail closed, trust nobody.
            self._paired = {}
            self.load_error = str(exc)

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._paired = {}
                return
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"could not read cluster device registry: {exc}"
                ) from exc
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError("unsupported cluster device registry schema")
            raw_devices = payload.get("devices")
            if not isinstance(raw_devices, list):
                raise ValueError("cluster device registry is malformed")
            devices: dict[str, dict[str, Any]] = {}
            for item in raw_devices:
                device = self._validate_paired(item)
                devices[device["node_id"]] = device
            if len(devices) != len(raw_devices):
                raise ValueError("cluster device registry has duplicate node_ids")
            self._paired = devices

    @staticmethod
    def _validate_paired(item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError("cluster device entry is malformed")
        node_id = item.get("node_id")
        friendly_name = item.get("friendly_name")
        paired_at = item.get("paired_at")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("cluster device entry is missing a node_id")
        if not isinstance(friendly_name, str) or not friendly_name:
            raise ValueError("cluster device entry is missing a friendly_name")
        if not isinstance(paired_at, (int, float)):
            raise ValueError("cluster device entry is missing paired_at")
        caps = item.get("caps")
        last_addrs = item.get("last_addrs")
        device = {
            "node_id": node_id,
            "friendly_name": friendly_name,
            "caps": dict(caps) if isinstance(caps, dict) else {},
            "paired_at": float(paired_at),
            "last_addrs": [str(a) for a in last_addrs if isinstance(a, str)][:16]
            if isinstance(last_addrs, list)
            else [],
        }
        http_port = item.get("http_port")
        if http_port is not None:
            if (
                not isinstance(http_port, int)
                or isinstance(http_port, bool)
                or not 1 <= http_port <= 65535
            ):
                raise ValueError("cluster device entry has an invalid HTTP port")
            device["http_port"] = http_port
        return device

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "devices": [self._paired[key] for key in sorted(self._paired)],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".devices.", suffix=".tmp", dir=self.path.parent
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

    # -- merge API (discovery + pairing) ------------------------------------

    @staticmethod
    def _merge_caps(
        existing: dict[str, Any], incoming: dict[str, Any]
    ) -> dict[str, Any]:
        """Field-wise non-downgrading caps merge.

        A discovery HELLO carries a structurally complete but value-empty
        caps dict (``{"chip": "", "ram_gb": 0.0, ...}``); a truthiness check
        would let it clobber the real capabilities exchanged at pairing.
        Only real values win: non-empty chip, non-zero ram_gb, the union of
        backends, and sticky-true fabric flags.
        """

        merged = dict(existing)
        if incoming.get("chip"):
            merged["chip"] = incoming["chip"]
        if incoming.get("ram_gb"):
            merged["ram_gb"] = incoming["ram_gb"]
        backends = incoming.get("backends")
        if backends:
            merged["backends"] = sorted(
                set(existing.get("backends") or []) | set(backends)
            )
        for flag in ("thunderbolt", "jaccl"):
            # Sticky-true, but never introduce the key out of nowhere —
            # records that never had fabric flags keep their shape.
            if existing.get(flag) or incoming.get(flag) or flag in existing:
                merged[flag] = bool(existing.get(flag)) or bool(incoming.get(flag))
        for key, value in incoming.items():
            if key not in merged and value:
                merged[key] = value
        return merged

    @staticmethod
    def _merge_addrs(existing: list[str], incoming: list[str]) -> list[str]:
        """Union, preserving order — a later sparse announcement must not
        drop routable addresses the pairing recorded."""

        merged = list(existing)
        for ip in incoming:
            if ip not in merged:
                merged.append(ip)
        return merged[:16]

    def merge(
        self,
        device: Any,
        *,
        paired_at: float | None = None,
    ) -> dict[str, Any]:
        """Upsert announced device metadata, keyed on ``node_id``.

        Accepts any object/dict with ``node_id`` and optional
        ``friendly_name``/``caps``/``addrs`` or ``last_addrs`` (a discovery
        ``PeerRecord`` works as-is). Updates to an already-paired device are
        persisted; updates to an unpaired device stay memory-only. Merges
        never downgrade: empty announced fields cannot erase known ones.
        """

        record = self._coerce(device)
        node_id = record["node_id"]
        with self._lock:
            if node_id in self._paired:
                merged = self._paired[node_id]
                if record.get("friendly_name"):
                    merged["friendly_name"] = record["friendly_name"]
                if record.get("caps"):
                    merged["caps"] = self._merge_caps(
                        dict(merged.get("caps") or {}), record["caps"]
                    )
                if record.get("last_addrs"):
                    merged["last_addrs"] = self._merge_addrs(
                        list(merged.get("last_addrs") or []), record["last_addrs"]
                    )
                if record.get("http_port"):
                    merged["http_port"] = record["http_port"]
                if paired_at is not None:
                    merged["paired_at"] = float(paired_at)
                previous = dict(self._paired)
                try:
                    self._save()
                except Exception:
                    self._paired = previous
                    raise
                self.load_error = None
                return dict(merged)
            merged = self._discovered.get(node_id, {"node_id": node_id})
            if record.get("friendly_name"):
                merged["friendly_name"] = record["friendly_name"]
            if record.get("caps"):
                merged["caps"] = self._merge_caps(
                    dict(merged.get("caps") or {}), record["caps"]
                )
            if record.get("last_addrs"):
                merged["last_addrs"] = self._merge_addrs(
                    list(merged.get("last_addrs") or []), record["last_addrs"]
                )
            if record.get("http_port"):
                merged["http_port"] = record["http_port"]
            self._discovered[node_id] = merged
            return dict(merged)

    @staticmethod
    def _coerce(device: Any) -> dict[str, Any]:
        if isinstance(device, dict):
            record = dict(device)
        else:
            record = {
                "node_id": getattr(device, "node_id", None),
                "friendly_name": getattr(device, "friendly_name", None),
                "caps": getattr(device, "caps", None),
                "http_port": getattr(device, "http_port", None),
            }
            addrs = getattr(device, "addrs", None)
            if addrs is not None:
                record["last_addrs"] = [
                    a.get("ip") if isinstance(a, dict) else getattr(a, "ip", None)
                    for a in addrs
                ]
        node_id = record.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("device record is missing a node_id")
        if "last_addrs" not in record and isinstance(record.get("addrs"), list):
            record["last_addrs"] = [
                a.get("ip") if isinstance(a, dict) else a for a in record["addrs"]
            ]
        caps = record.get("caps")
        if caps is not None and not isinstance(caps, dict):
            caps = caps.to_dict() if hasattr(caps, "to_dict") else {}
        normalized: dict[str, Any] = {"node_id": node_id}
        if isinstance(record.get("friendly_name"), str):
            normalized["friendly_name"] = record["friendly_name"]
        if isinstance(caps, dict):
            normalized["caps"] = dict(caps)
        addrs = record.get("last_addrs")
        if isinstance(addrs, list):
            normalized["last_addrs"] = [str(a) for a in addrs if isinstance(a, str)][
                :16
            ]
        http_port = record.get("http_port")
        if http_port is not None and http_port != 0 and http_port != "":
            if (
                not isinstance(http_port, int)
                or isinstance(http_port, bool)
                or not 1 <= http_port <= 65535
            ):
                raise ValueError("device record has an invalid HTTP port")
            normalized["http_port"] = http_port
        return normalized

    def mark_paired(
        self,
        node_id: str,
        *,
        friendly_name: str | None = None,
        caps: dict[str, Any] | None = None,
        addrs: list[str] | None = None,
        http_port: int | None = None,
        paired_at: float | None = None,
    ) -> dict[str, Any]:
        """Promote a device to trusted/paired and persist it."""

        with self._lock:
            existing = self._paired.get(node_id) or self._discovered.get(node_id, {})
            if http_port is not None and (
                not isinstance(http_port, int)
                or isinstance(http_port, bool)
                or not 1 <= http_port <= 65535
            ):
                raise ValueError("paired device HTTP port is invalid")
            device = {
                "node_id": node_id,
                "friendly_name": friendly_name
                or existing.get("friendly_name")
                or node_id,
                "caps": dict(caps)
                if caps is not None
                else dict(existing.get("caps") or {}),
                "paired_at": float(paired_at if paired_at is not None else time.time()),
                "last_addrs": list(addrs)
                if addrs is not None
                else list(existing.get("last_addrs") or []),
            }
            effective_port = http_port or existing.get("http_port")
            if effective_port:
                device["http_port"] = int(effective_port)
            previous = dict(self._paired)
            self._paired[node_id] = device
            self._discovered.pop(node_id, None)
            try:
                self._save()
            except Exception:
                self._paired = previous
                raise
            self.load_error = None
            return dict(device)

    def unpair(self, node_id: str) -> bool:
        """Revoke trust. The device drops back to memory-only if re-seen."""

        with self._lock:
            if node_id not in self._paired:
                return False
            previous = dict(self._paired)
            device = self._paired.pop(node_id)
            try:
                self._save()
            except Exception:
                self._paired = previous
                raise
            # Keep a memory-only discovered shell so the UI can still show
            # the now-untrusted device if it is still announcing.
            shell = {
                "node_id": node_id,
                "friendly_name": device["friendly_name"],
                "caps": device["caps"],
                "last_addrs": device["last_addrs"],
            }
            if device.get("http_port"):
                shell["http_port"] = device["http_port"]
            self._discovered.setdefault(
                node_id,
                shell,
            )
            return True

    # -- reads ---------------------------------------------------------------

    def paired(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(self._paired[key]) for key in sorted(self._paired)]

    def discovered(self) -> list[dict[str, Any]]:
        """Unpaired devices (memory-only), excluding paired node_ids."""

        with self._lock:
            return [
                dict(self._discovered[key])
                for key in sorted(self._discovered)
                if key not in self._paired
            ]

    def get(self, node_id: str) -> dict[str, Any] | None:
        with self._lock:
            if node_id in self._paired:
                return dict(self._paired[node_id])
            found = self._discovered.get(node_id)
            return dict(found) if found is not None else None

    def is_paired(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self._paired

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "paired": self.paired(),
            "discovered": self.discovered(),
            "load_error": self.load_error,
        }


_device_registry_lock = threading.Lock()
_configured_device_registry: DeviceRegistry | None = None


def configure_device_registry(
    path: Path | str | None = None,
) -> DeviceRegistry:
    global _configured_device_registry
    with _device_registry_lock:
        _configured_device_registry = DeviceRegistry(path)
        return _configured_device_registry


def get_device_registry() -> DeviceRegistry:
    with _device_registry_lock:
        if _configured_device_registry is None:
            raise RuntimeError("cluster device registry is not configured")
        return _configured_device_registry


def reset_configured_device_registry() -> None:
    """Test hook: drop the process-wide device registry."""

    global _configured_device_registry
    with _device_registry_lock:
        _configured_device_registry = None
