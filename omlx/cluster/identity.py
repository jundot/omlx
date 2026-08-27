# SPDX-License-Identifier: Apache-2.0
"""Stable per-node identity for oMLX cluster v2.

Every node owns a ``NodeIdentity`` persisted at
``~/.omlx/cluster/identity.json`` (mode 0600, atomic write). The ``node_id``
is an immutable uuid4 string; the ``friendly_name`` is display-only and may
be renamed or collision-repaired without affecting pairing, discovery, or
deployments, which all key on ``node_id``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

IDENTITY_SCHEMA_VERSION = 1
_MAX_NAME_LENGTH = 63
_NAME_SANITIZE = re.compile(r"\s+")


def default_identity_path() -> Path:
    """Default on-disk location, honoring the OMLX_BASE_PATH override."""

    env_value = os.environ.get("OMLX_BASE_PATH")
    base = (
        Path(env_value).expanduser() if env_value else Path.home() / ".omlx"
    )
    return base / "cluster" / "identity.json"


def _scutil_local_hostname() -> str | None:
    """Best-effort macOS LocalHostName; ``None`` off-macOS or on failure."""

    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(  # noqa: S603 - fixed system executable
            ["/usr/sbin/scutil", "--get", "LocalHostName"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    name = result.stdout.strip()
    return name or None


def _default_friendly_name() -> str:
    for candidate in (_scutil_local_hostname(), socket.gethostname()):
        if candidate:
            name = _NAME_SANITIZE.sub("-", candidate.strip())[
                :_MAX_NAME_LENGTH
            ]
            if name:
                return name
    return "omlx-node"


def _resolve_collision(name: str, taken: set[str]) -> str:
    """Suffix ``-2``, ``-3`` ... until the friendly name is unique."""

    if name not in taken:
        return name
    for suffix in range(2, 1000):
        candidate = f"{name}-{suffix}"
        if candidate not in taken:
            return candidate
    # Absurdly unlikely; fall back to a random tail rather than loop forever.
    return f"{name}-{uuid.uuid4().hex[:6]}"


@dataclass
class NodeIdentity:
    """A node's stable identity. ``node_id`` never changes once created."""

    node_id: str
    friendly_name: str
    created_at: float
    schema_version: int = IDENTITY_SCHEMA_VERSION
    _path: Path | None = field(default=None, repr=False, compare=False)
    load_error: str | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "node_id": self.node_id,
            "friendly_name": self.friendly_name,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NodeIdentity:
        if not isinstance(payload, dict):
            raise ValueError("cluster identity is malformed")
        if payload.get("schema_version") != IDENTITY_SCHEMA_VERSION:
            raise ValueError("unsupported cluster identity schema version")
        node_id = payload.get("node_id")
        friendly_name = payload.get("friendly_name")
        created_at = payload.get("created_at")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("cluster identity is missing a node_id")
        if not isinstance(friendly_name, str) or not friendly_name:
            raise ValueError("cluster identity is missing a friendly_name")
        if not isinstance(created_at, (int, float)):
            raise ValueError("cluster identity is missing created_at")
        return cls(
            node_id=node_id,
            friendly_name=friendly_name,
            created_at=float(created_at),
        )

    def rename(self, new_name: str) -> None:
        """Update the display name and persist it. ``node_id`` is immutable."""

        cleaned = _NAME_SANITIZE.sub("-", str(new_name).strip())[
            :_MAX_NAME_LENGTH
        ]
        if not cleaned:
            raise ValueError("friendly name must not be empty")
        self.friendly_name = cleaned
        if self._path is not None:
            _atomic_save(self._path, self.to_dict())


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".identity.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary)


def load_or_create(
    path: Path | str | None = None,
    *,
    taken_names: Iterable[str] = (),
    clock: Callable[[], float] = time.time,
) -> NodeIdentity:
    """Load the persisted identity, creating and persisting one if absent.

    ``taken_names`` lists friendly names already claimed by other cluster
    members; on collision the loaded/created name gains a ``-2`` suffix
    (escalating as needed) while ``node_id`` stays untouched.

    A corrupt identity file never rotates ``node_id`` silently: the corrupt
    file is left on disk for inspection, a fresh in-memory identity is
    returned with ``load_error`` set, and nothing is written.
    """

    identity_path = Path(path) if path is not None else default_identity_path()
    taken = {name for name in taken_names}

    identity: NodeIdentity | None = None
    load_error: str | None = None
    if identity_path.exists():
        try:
            payload = json.loads(identity_path.read_text(encoding="utf-8"))
            identity = NodeIdentity.from_dict(payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            load_error = f"could not read cluster identity: {exc}"
            logger.warning(
                "%s at %s; using a fresh in-memory identity (the corrupt "
                "file was left untouched)",
                load_error,
                identity_path,
            )

    created = identity is None
    if identity is None:
        identity = NodeIdentity(
            node_id=str(uuid.uuid4()),
            friendly_name=_default_friendly_name(),
            created_at=clock(),
        )
    identity._path = identity_path
    identity.load_error = load_error

    repaired = _resolve_collision(identity.friendly_name, taken)
    collision_repaired = repaired != identity.friendly_name
    identity.friendly_name = repaired

    # Persist on first creation or when collision repair changed the name —
    # but never overwrite a file we failed to parse.
    if load_error is None and (created or collision_repaired):
        try:
            _atomic_save(identity_path, identity.to_dict())
        except OSError as exc:
            identity.load_error = f"could not persist cluster identity: {exc}"
            logger.warning("%s", identity.load_error)
    return identity


_identity_lock = threading.Lock()
_configured_identity: NodeIdentity | None = None


def configure_node_identity(
    base_path: Path | str | None = None,
    *,
    taken_names: Iterable[str] = (),
) -> NodeIdentity:
    """Configure the process-wide identity used by the discovery endpoints."""

    global _configured_identity
    path = (
        Path(base_path) / "cluster" / "identity.json"
        if base_path is not None
        else None
    )
    identity = load_or_create(path, taken_names=taken_names)
    with _identity_lock:
        _configured_identity = identity
    return identity


def get_node_identity() -> NodeIdentity:
    with _identity_lock:
        if _configured_identity is None:
            raise RuntimeError("cluster node identity is not configured")
        return _configured_identity


def reset_configured_identity() -> None:
    """Test hook: drop the process-wide identity."""

    global _configured_identity
    with _identity_lock:
        _configured_identity = None
