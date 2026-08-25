# SPDX-License-Identifier: Apache-2.0
"""One-time enrollment state for headless cluster workers.

Join credentials are deliberately memory-only.  A server restart invalidates
every unclaimed command, while completed node records persist without tokens or
public-key material.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# A fresh worker may need the command's bounded prerequisite installation
# before Python can claim this key. Keep it single-use, but long enough for
# that first-run path and a human paste delay.
JOIN_KEY_TTL_SECONDS = 30 * 60
# The worker claims before provisioning, then may spend up to 30 minutes in a
# single dependency install and repeat that repair once after ``pip check``.
# Keep the one-time session alive for the complete bounded bootstrap.
JOIN_SESSION_TTL_SECONDS = 2 * 60 * 60
MAX_PENDING_JOIN_KEYS = 32
MAX_ENROLLED_NODES = 64


class EnrollmentError(ValueError):
    """A join credential or enrollment payload was refused."""


def _secret_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class _JoinKey:
    join_id: str
    digest: str
    controller_url: str
    source_digest: str
    created_at: float
    expires_at: float
    used_at: float | None = None


@dataclass(frozen=True)
class _JoinSession:
    digest: str
    join_id: str
    controller_url: str
    source_digest: str
    node_id: str
    hostname: str
    ssh_user: str
    ssh_port: int
    addresses: tuple[str, ...]
    expires_at: float


@dataclass(frozen=True)
class EnrolledNode:
    node_id: str
    hostname: str
    ssh: str
    ssh_user: str
    ssh_port: int
    addresses: tuple[str, ...]
    accelerator: str
    platform: str
    python_executable: str
    source_digest: str
    ssh_host_fingerprint: str
    joined_at: float
    last_seen_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "ssh": self.ssh,
            "ssh_user": self.ssh_user,
            "ssh_port": self.ssh_port,
            "addresses": list(self.addresses),
            "accelerator": self.accelerator,
            "platform": self.platform,
            "python_executable": self.python_executable,
            "source_digest": self.source_digest,
            "ssh_host_fingerprint": self.ssh_host_fingerprint,
            "joined_at": self.joined_at,
            "last_seen_at": self.last_seen_at,
            "status": "joined",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EnrolledNode:
        if not isinstance(value, dict):
            raise EnrollmentError("enrolled node record must be an object")
        addresses = value.get("addresses")
        if not isinstance(addresses, list) or not all(
            isinstance(item, str) for item in addresses
        ):
            raise EnrollmentError("enrolled node addresses are malformed")
        return cls(
            node_id=str(value["node_id"]),
            hostname=str(value["hostname"]),
            ssh=str(value["ssh"]),
            ssh_user=str(value["ssh_user"]),
            ssh_port=int(value["ssh_port"]),
            addresses=tuple(addresses),
            accelerator=str(value["accelerator"]),
            platform=str(value["platform"]),
            python_executable=str(value["python_executable"]),
            source_digest=str(value["source_digest"]),
            ssh_host_fingerprint=str(value["ssh_host_fingerprint"]),
            joined_at=float(value["joined_at"]),
            last_seen_at=float(value["last_seen_at"]),
        )


class ClusterEnrollmentStore:
    """Thread-safe one-time join keys plus credential-free node persistence."""

    def __init__(self, base_path: Path, *, clock=time.time) -> None:
        self.base_path = Path(base_path)
        self.path = self.base_path / "cluster" / "enrolled-nodes.json"
        self._clock = clock
        self._lock = threading.RLock()
        self._join_keys: dict[str, _JoinKey] = {}
        self._sessions: dict[str, _JoinSession] = {}
        self._nodes: dict[str, EnrolledNode] = {}
        self.load_error: str | None = None
        try:
            self._load()
        except EnrollmentError as exc:
            self._nodes = {}
            self.load_error = str(exc)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EnrollmentError(f"could not read enrolled nodes: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise EnrollmentError("unsupported enrolled-node registry schema")
        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list):
            raise EnrollmentError("enrolled-node registry is malformed")
        nodes = [EnrolledNode.from_dict(item) for item in raw_nodes]
        self._nodes = {node.node_id: node for node in nodes}
        if len(self._nodes) != len(nodes):
            raise EnrollmentError("enrolled-node registry has duplicate node IDs")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "nodes": [node.to_dict() for node in self.list_nodes()],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".enrolled-nodes.", suffix=".tmp", dir=self.path.parent
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

    def _prune(self) -> None:
        now = self._clock()
        self._sessions = {
            digest: session
            for digest, session in self._sessions.items()
            if session.expires_at >= now
        }
        live_session_join_ids = {
            session.join_id for session in self._sessions.values()
        }
        self._join_keys = {
            digest: record
            for digest, record in self._join_keys.items()
            if record.expires_at >= now or record.join_id in live_session_join_ids
        }

    def issue_join_key(
        self,
        *,
        controller_url: str,
        source_digest: str,
        ttl: int = JOIN_KEY_TTL_SECONDS,
    ) -> tuple[str, dict[str, Any]]:
        if not 30 <= int(ttl) <= JOIN_KEY_TTL_SECONDS:
            raise EnrollmentError(
                f"join-key TTL must be between 30 and {JOIN_KEY_TTL_SECONDS} seconds"
            )
        with self._lock:
            self._prune()
            active = [record for record in self._join_keys.values() if record.used_at is None]
            if len(active) >= MAX_PENDING_JOIN_KEYS:
                raise EnrollmentError("too many pending join keys; revoke one and retry")
            raw_key = secrets.token_urlsafe(32)
            now = self._clock()
            record = _JoinKey(
                join_id=secrets.token_hex(8),
                digest=_secret_digest(raw_key),
                controller_url=controller_url,
                source_digest=source_digest,
                created_at=now,
                expires_at=now + int(ttl),
            )
            self._join_keys[record.digest] = record
            return raw_key, self._join_key_dict(record)

    @staticmethod
    def _join_key_dict(record: _JoinKey) -> dict[str, Any]:
        return {
            "join_id": record.join_id,
            "controller_url": record.controller_url,
            "source_digest": record.source_digest,
            "created_at": record.created_at,
            "expires_at": record.expires_at,
            "status": "used" if record.used_at is not None else "pending",
            "used_at": record.used_at,
        }

    def claim(
        self,
        raw_key: str,
        *,
        node_id: str,
        hostname: str,
        ssh_user: str,
        ssh_port: int,
        addresses: tuple[str, ...],
    ) -> tuple[str, _JoinSession]:
        digest = _secret_digest(raw_key)
        with self._lock:
            self._prune()
            record = self._join_keys.get(digest)
            now = self._clock()
            if record is None or record.expires_at < now:
                raise EnrollmentError("invalid or expired join key")
            if record.used_at is not None:
                raise EnrollmentError("join key has already been used")
            record.used_at = now
            raw_session = secrets.token_urlsafe(32)
            session = _JoinSession(
                digest=_secret_digest(raw_session),
                join_id=record.join_id,
                controller_url=record.controller_url,
                source_digest=record.source_digest,
                node_id=node_id,
                hostname=hostname,
                ssh_user=ssh_user,
                ssh_port=ssh_port,
                addresses=addresses,
                expires_at=now + JOIN_SESSION_TTL_SECONDS,
            )
            self._sessions[session.digest] = session
            return raw_session, session

    def authorize_session(self, raw_session: str) -> _JoinSession:
        digest = _secret_digest(raw_session)
        with self._lock:
            self._prune()
            session = self._sessions.get(digest)
            if session is None:
                raise EnrollmentError("invalid or expired join session")
            return session

    def complete(self, raw_session: str, node: EnrolledNode) -> EnrolledNode:
        digest = _secret_digest(raw_session)
        with self._lock:
            session = self.authorize_session(raw_session)
            if node.source_digest != session.source_digest:
                raise EnrollmentError("worker source digest does not match the join command")
            if (
                node.node_id != session.node_id
                or node.hostname != session.hostname
                or node.ssh_user != session.ssh_user
                or node.ssh_port != session.ssh_port
                or node.addresses != session.addresses
            ):
                raise EnrollmentError("worker identity changed after claiming the join key")
            if node.node_id not in self._nodes and len(self._nodes) >= MAX_ENROLLED_NODES:
                raise EnrollmentError("the enrolled-node limit has been reached")
            previous = dict(self._nodes)
            self._nodes[node.node_id] = node
            try:
                self._save()
            except Exception:
                self._nodes = previous
                raise
            self._sessions.pop(digest, None)
            self.load_error = None
            return node

    def list_nodes(self) -> tuple[EnrolledNode, ...]:
        with self._lock:
            return tuple(self._nodes[key] for key in sorted(self._nodes))

    def remove_node(self, node_id: str) -> bool:
        with self._lock:
            if node_id not in self._nodes:
                return False
            previous = dict(self._nodes)
            del self._nodes[node_id]
            try:
                self._save()
            except Exception:
                self._nodes = previous
                raise
            return True

    def revoke_join_key(self, join_id: str) -> bool:
        with self._lock:
            record_digest = next(
                (
                    digest
                    for digest, record in self._join_keys.items()
                    if record.join_id == join_id
                ),
                None,
            )
            if record_digest is None:
                return False
            del self._join_keys[record_digest]
            self._sessions = {
                digest: session
                for digest, session in self._sessions.items()
                if session.join_id != join_id
            }
            return True

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            self._prune()
            return {
                "schema_version": 1,
                "join_keys": [
                    self._join_key_dict(record)
                    for record in sorted(
                        self._join_keys.values(), key=lambda item: item.created_at
                    )
                ],
                "nodes": [node.to_dict() for node in self.list_nodes()],
                "load_error": self.load_error,
            }


_store_lock = threading.Lock()
_configured_store: ClusterEnrollmentStore | None = None


def configure_cluster_enrollment(base_path: Path) -> ClusterEnrollmentStore:
    global _configured_store
    with _store_lock:
        _configured_store = ClusterEnrollmentStore(base_path)
        return _configured_store


def get_cluster_enrollment() -> ClusterEnrollmentStore:
    with _store_lock:
        if _configured_store is None:
            raise RuntimeError("cluster enrollment is not configured")
        return _configured_store
