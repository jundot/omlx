# SPDX-License-Identifier: Apache-2.0
"""The helper's state folder: its identity and the hosts it is paired with."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(path: Path, data: dict[str, Any]) -> None:
    """Replace *path* atomically; the file is readable by this user only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class State:
    """Identity and pairing keys under one folder.

    The folder must persist across restarts (a volume in Docker): a new
    ``node_id`` is a new device to every host, and the pairing keys are lost.
    """

    def __init__(self, root: str | Path, name: str | None = None):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        identity = _read(self.root / "identity.json")
        if not identity.get("node_id"):
            identity = {
                "schema_version": 1,
                "node_id": str(uuid.uuid4()),
                "created_at": time.time(),
            }
        wanted = name or identity.get("friendly_name") or socket.gethostname()
        if identity.get("friendly_name") != wanted:
            identity["friendly_name"] = wanted
            _write(self.root / "identity.json", identity)
        self.node_id: str = identity["node_id"]
        self.name: str = identity["friendly_name"]

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    def hosts(self) -> dict[str, dict[str, Any]]:
        """Paired hosts by node id."""
        with self._lock:
            return _read(self.root / "pairing-keys.json").get("hosts") or {}

    def cluster_key(self, node_id: str) -> bytes | None:
        record = self.hosts().get(node_id)
        try:
            return bytes.fromhex(record["cluster_key"]) if record else None
        except (KeyError, ValueError):
            return None

    def save_host(self, node_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            path = self.root / "pairing-keys.json"
            hosts = _read(path).get("hosts") or {}
            hosts[node_id] = record
            _write(path, {"schema_version": 1, "hosts": hosts})

    def remove_host(self, node_id: str) -> bool:
        with self._lock:
            path = self.root / "pairing-keys.json"
            hosts = _read(path).get("hosts") or {}
            if hosts.pop(node_id, None) is None:
                return False
            _write(path, {"schema_version": 1, "hosts": hosts})
            return True
