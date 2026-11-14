# SPDX-License-Identifier: Apache-2.0
"""Persisted compute profiles for the Cluster Assistant's next plan.

The synthetic distributed probe measures each rank's effective model-weight
bytes per second and the collective link. It is model-independent: it says what
the Macs and the fabric can do, not what a checkpoint will decode. The probe runs
after ``apply`` on the elected coordinator; its profiles are stored here on the
origin and read by the **next** measurement, so the next election is planned with
measured compute.

Rules:

* one profile per Mac, newest wins; a profile is never invented;
* the store is a scratch record under ``<base_path>/cluster/``, mode 0600,
  atomic replacement, fail-closed on load;
* no model path, key, SSH target or host is stored — only node IDs and numbers;
* a stale profile is still a measurement (its ``measured_at`` travels with it);
  the caller decides, not this module.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PERFORMANCE_SCHEMA_VERSION = 1
PERFORMANCE_FILE_NAME = "assistant-performance.json"
MAX_PERFORMANCE_NODES = 64


@dataclass(frozen=True)
class StoredPerformance:
    """One Mac's measured synthetic compute, with where it came from."""

    node_id: str
    profile: Mapping[str, Any]
    deployment_id: str = ""
    world_size: int = 0
    backend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "profile": dict(self.profile),
            "deployment_id": self.deployment_id,
            "world_size": self.world_size,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StoredPerformance:
        node_id = str(payload.get("node_id") or "").strip()
        profile = payload.get("profile")
        if not node_id or not isinstance(profile, Mapping):
            raise ValueError("stored performance is malformed")
        return cls(
            node_id=node_id,
            profile=dict(profile),
            deployment_id=str(payload.get("deployment_id") or ""),
            world_size=_optional_int(payload.get("world_size")) or 0,
            backend=str(payload.get("backend") or ""),
        )


class AssistantPerformanceStore:
    """Thread-safe, atomic, bounded store of per-Mac compute profiles."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self.path = self.base_path / "cluster" / PERFORMANCE_FILE_NAME
        self._lock = threading.RLock()
        self._records: dict[str, StoredPerformance] = {}
        self.load_error: str | None = None
        try:
            self._load()
        except (KeyError, TypeError, ValueError) as exc:
            self._records = {}
            self.load_error = str(exc)

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._records = {}
                return
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != PERFORMANCE_SCHEMA_VERSION
            ):
                raise ValueError("unsupported assistant performance schema")
            raw = payload.get("profiles") or []
            if not isinstance(raw, list):
                raise ValueError("assistant performance file is malformed")
            records = [StoredPerformance.from_dict(item) for item in raw]
            if len({item.node_id for item in records}) != len(records):
                raise ValueError("assistant performance file has duplicate nodes")
            self._records = {item.node_id: item for item in records}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": PERFORMANCE_SCHEMA_VERSION,
            "profiles": [
                self._records[node_id].to_dict() for node_id in sorted(self._records)
            ],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".assistant-performance.",
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

    def save(
        self,
        profiles: Sequence[Mapping[str, Any]],
        *,
        deployment_id: str = "",
        world_size: int = 0,
        backend: str = "",
    ) -> tuple[StoredPerformance, ...]:
        """Store the probe's profiles, one per node. Returns what was stored."""

        stored: list[StoredPerformance] = []
        for item in profiles:
            node_id = str(item.get("node_id") or "").strip()
            if not node_id:
                continue
            profile = dict(item)
            stored.append(
                StoredPerformance(
                    node_id=node_id,
                    profile=profile,
                    deployment_id=deployment_id,
                    world_size=int(world_size),
                    backend=str(backend),
                )
            )
        if not stored:
            return ()
        with self._lock:
            previous = dict(self._records)
            for item in stored:
                self._records[item.node_id] = item
            # Keep the newest N nodes; the MVP is at most four, so this is
            # only a guard against an old, much larger measurement.
            surplus = len(self._records) - MAX_PERFORMANCE_NODES
            if surplus > 0:
                for node_id in sorted(self._records)[:surplus]:
                    del self._records[node_id]
            try:
                self._save()
            except Exception:
                self._records = previous
                raise
        return tuple(stored)

    def for_nodes(self, node_ids: Sequence[str]) -> dict[str, StoredPerformance]:
        """The stored profile for each requested node, if one exists."""

        with self._lock:
            return {
                node_id: self._records[node_id]
                for node_id in node_ids
                if node_id in self._records
            }

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": PERFORMANCE_SCHEMA_VERSION,
                "profiles": [
                    self._records[node_id].to_dict()
                    for node_id in sorted(self._records)
                ],
            }


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
