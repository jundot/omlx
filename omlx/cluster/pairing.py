# SPDX-License-Identifier: Apache-2.0
"""Code-based cluster pairing and enrollment bridge (cluster v2, Module B).

Replaces the 3-step copy-paste relay (shared secret + key-exchange tokens in
both directions) with an exo-style flow:

1. The JOINER shows a 6-digit code (valid 10 minutes).
2. The joiner POSTs ``/api/cluster/pair/request`` to the coordinator carrying
   its identity and a salted, slow code verifier bound to the node and SSH
   identities — never the code itself.
3. The coordinator dashboard lists the pending request
   (``state: "awaiting_approval"``) and the operator types the code shown on
   the joiner into ``/api/cluster/pair/approve``.
4. Approval verifies the code-bound identities (3 attempts, then a 10-minute lockout),
   generates the 32-byte ``cluster_key``, wraps it under a key derived from
   the code with PBKDF2-HMAC-SHA256 (210k iterations), persists the peer as
   paired, and drives symmetric, fail-closed SSH enrollment with the
   authenticated user key and daemon host key.
5. The joiner polls ``/api/cluster/pair/status/{node_id}`` and unwraps the
   cluster key locally with the code only it possesses.

``DELETE /api/cluster/devices/{node_id}`` unpairs: the device-store entry and
the stored cluster key are removed, any legacy ``enrolled-nodes.json`` entry
is dropped, and the SSH trust installed at approve time is revoked
best-effort.

Module A (identity + discovery) is built concurrently on another branch.
Every interaction with it is feature-detected: a real ``DeviceRegistry`` /
``NodeIdentity`` is used when importable, otherwise schema-compatible local
fallbacks (same files, same ``schema_version: 1``) keep this module fully
functional and offline-testable.  See ``DeviceRegistryBridge`` for the exact
method names looked up on Module A's registry.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import platform
import secrets
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# --- Pairing policy (from ops/notes/omlx_cluster_v2_spec.md, Module B) ------

CODE_DIGITS = 6
CODE_TTL_SECONDS = 10 * 60
MAX_CODE_ATTEMPTS = 3
LOCKOUT_SECONDS = 10 * 60
PBKDF2_ITERATIONS = 210_000
CLUSTER_KEY_BYTES = 32
WRAP_KDF = "PBKDF2-HMAC-SHA256"
_WRAP_TAG_DOMAIN = b"omlx-cluster-key-v1"
MAX_PENDING_REQUESTS = 32

DEFAULT_BASE_PATH = Path.home() / ".omlx"
IDENTITY_SCHEMA_VERSION = 1
DEVICE_SCHEMA_VERSION = 1
KEY_SCHEMA_VERSION = 1


# --- Errors -----------------------------------------------------------------


class PairingError(ValueError):
    """Base class for pairing refusals; ``reason`` is machine-readable."""


class PairingRequestError(PairingError):
    """A join request payload was malformed or could not be accepted."""


class PairingStateError(PairingError):
    """No such pending request / device, or the state transition is invalid."""


class PairingCodeError(PairingError):
    """The presented code does not match the pending request's code hash."""


class PairingLockoutError(PairingError):
    """Too many wrong codes; approval is locked out until ``retry_after``."""


class PairingExpiredError(PairingError):
    """The pending request outlived the 10-minute code validity window."""


class EnrollmentDriveError(PairingError):
    """The fail-closed SSH TOFU enrollment step failed after code verify."""


# --- Code + key-wrapping crypto (stdlib only) --------------------------------


def generate_pairing_code() -> str:
    """Return a zero-padded 6-digit pairing code shown on the joiner."""

    return f"{secrets.randbelow(10**CODE_DIGITS):0{CODE_DIGITS}d}"


def validate_pairing_code(code: str) -> str:
    if not isinstance(code, str) or not code.isdigit() or len(code) != CODE_DIGITS:
        raise PairingRequestError(f"pairing code must be {CODE_DIGITS} digits")
    return code


def pairing_code_hash(
    code: str,
    node_id: str,
    ssh_public_key: str = "",
    ssh_host_public_key: str = "",
    salt: bytes = b"",
) -> str:
    """Slowly derive a verifier bound to the node and both SSH identities."""

    payload = json.dumps(
        {
            "node_id": node_id,
            "ssh_public_key": ssh_public_key,
            "ssh_host_public_key": ssh_host_public_key,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    verifier_salt = bytes(salt) + hashlib.sha256(payload).digest()
    return hashlib.pbkdf2_hmac(
        "sha256",
        code.encode("utf-8"),
        verifier_salt,
        PBKDF2_ITERATIONS,
        dklen=32,
    ).hex()


def _derive_wrap_key(code: str, salt: bytes, iterations: int) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", code.encode("utf-8"), salt, iterations, dklen=32
    )


def wrap_cluster_key(
    cluster_key: bytes, code: str, *, iterations: int = PBKDF2_ITERATIONS
) -> dict[str, Any]:
    """Encrypt ``cluster_key`` under a key derived from the pairing code.

    The derived key is used exactly once, so XOR is a one-time pad here; an
    HMAC over the ciphertext detects both tampering and wrong-code unwraps.
    """

    if len(cluster_key) != CLUSTER_KEY_BYTES:
        raise PairingError(f"cluster key must be {CLUSTER_KEY_BYTES} bytes")
    salt = secrets.token_bytes(16)
    wrap_key = _derive_wrap_key(code, salt, iterations)
    ciphertext = bytes(a ^ b for a, b in zip(cluster_key, wrap_key))
    tag = hmac.new(wrap_key, _WRAP_TAG_DOMAIN + ciphertext, hashlib.sha256).digest()
    return {
        "kdf": WRAP_KDF,
        "iterations": iterations,
        "salt": base64.b64encode(salt).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "tag": base64.b64encode(tag).decode(),
    }


def unwrap_cluster_key(package: dict[str, Any], code: str) -> bytes:
    """Recover the cluster key; a wrong code fails closed with PairingCodeError."""

    try:
        iterations = int(package["iterations"])
        salt = base64.b64decode(package["salt"], validate=True)
        ciphertext = base64.b64decode(package["ciphertext"], validate=True)
        tag = base64.b64decode(package["tag"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise PairingRequestError("malformed wrapped cluster key") from exc
    if package.get("kdf") != WRAP_KDF or not 1 <= iterations <= 10_000_000:
        raise PairingRequestError("unsupported cluster-key wrap parameters")
    wrap_key = _derive_wrap_key(code, salt, iterations)
    expected = hmac.new(
        wrap_key, _WRAP_TAG_DOMAIN + ciphertext, hashlib.sha256
    ).digest()
    if not hmac.compare_digest(tag, expected):
        raise PairingCodeError("cluster key unwrap failed (wrong code or tampering)")
    if len(ciphertext) != CLUSTER_KEY_BYTES:
        raise PairingRequestError("malformed wrapped cluster key")
    return bytes(a ^ b for a, b in zip(ciphertext, wrap_key))


def coordinator_identity_tag(
    cluster_key: bytes,
    coordinator: dict[str, Any],
) -> str:
    """Bind the approved coordinator identity to the unwrapped cluster key."""

    payload = json.dumps(
        coordinator,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(
        cluster_key, b"omlx-coordinator-v1\0" + payload, hashlib.sha256
    ).hexdigest()


# --- Small persisted stores (same conventions as registry.py/enrollment.py) -


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
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


def _read_json_object(path: Path, schema_version: int, what: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PairingError(f"could not read {what}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != schema_version:
        raise PairingError(f"unsupported {what} schema")
    return payload


class PairingAuditLog:
    """Append-only JSON-lines audit trail for pairing decisions (0600)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(
        self, event: str, *, node_id: str = "", detail: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        entry = {
            "ts": time.time(),
            "event": event,
            "node_id": node_id,
            "detail": detail or {},
        }
        line = json.dumps(entry, sort_keys=True)
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fresh = not self.path.exists()
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
                if fresh:
                    os.chmod(self.path, 0o600)
            except OSError:
                # Auditing must never break a pairing state transition.
                logger.exception("could not append to the pairing audit log")
        return entry


class PairingKeyStore:
    """Per-peer cluster keys at ``<base>/cluster/pairing-keys.json`` (0600).

    Written on both sides: the coordinator records the key it issued, the
    joiner records the key it unwrapped.  Also retains the peer's SSH public
    key and addresses so ``unpair`` can revoke exactly what ``approve``
    installed.
    """

    def __init__(self, base_path: Path, *, clock: Callable[[], float] = time.time):
        self.base_path = Path(base_path)
        self.path = self.base_path / "cluster" / "pairing-keys.json"
        self._clock = clock
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self.load_error: str | None = None
        try:
            self._load()
        except PairingError as exc:
            # Fail closed, like ClusterRegistry: trust nothing from a corrupt file.
            self._records = {}
            self.load_error = str(exc)

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = _read_json_object(self.path, KEY_SCHEMA_VERSION, "pairing-key store")
        records = payload.get("keys")
        if not isinstance(records, dict):
            raise PairingError("pairing-key store is malformed")
        self._records = {str(k): dict(v) for k, v in records.items()}

    def _save(self) -> None:
        _atomic_write_json(
            self.path,
            {"schema_version": KEY_SCHEMA_VERSION, "keys": self._records},
        )

    def set(self, node_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            previous = dict(self._records)
            self._records[node_id] = dict(record)
            try:
                self._save()
            except Exception:
                self._records = previous
                raise
            self.load_error = None

    def get(self, node_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(node_id)
            return dict(record) if record is not None else None

    def remove(self, node_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.pop(node_id, None)
            if record is None:
                return None
            try:
                self._save()
            except Exception:
                self._records[node_id] = record
                raise
            return dict(record)

    def list(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {key: dict(value) for key, value in sorted(self._records.items())}


class JsonDeviceStore:
    """Fallback paired-device store, schema-compatible with Module A.

    Persists to ``<base>/cluster/devices.json`` — the exact file Module A's
    ``DeviceRegistry`` owns — using the documented shape: paired devices as
    ``{node_id, friendly_name, caps, paired_at, last_addrs, state}`` under
    ``schema_version: 1``.  Used only when no Module A registry is injected;
    when Module A lands, the integrator passes its ``DeviceRegistry`` and this
    class is unused (``DeviceRegistryBridge`` adapts it instead).

    Pending (``awaiting_approval``) requests are deliberately NOT written
    here: like enrollment.py join keys, they are memory-only so a restart
    invalidates half-finished pairings.
    """

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path)
        self.path = self.base_path / "cluster" / "devices.json"
        self._lock = threading.RLock()
        self._devices: dict[str, dict[str, Any]] = {}
        self.load_error: str | None = None
        try:
            self._load()
        except PairingError as exc:
            self._devices = {}
            self.load_error = str(exc)

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = _read_json_object(self.path, DEVICE_SCHEMA_VERSION, "device registry")
        devices = payload.get("devices")
        if not isinstance(devices, list):
            raise PairingError("device registry is malformed")
        self._devices = {
            str(item["node_id"]): dict(item)
            for item in devices
            if isinstance(item, dict) and "node_id" in item
        }

    def _save(self) -> None:
        _atomic_write_json(
            self.path,
            {
                "schema_version": DEVICE_SCHEMA_VERSION,
                "devices": [self._devices[key] for key in sorted(self._devices)],
            },
        )

    # PairingManager device-store protocol ---------------------------------

    def put_paired(self, record: dict[str, Any]) -> None:
        with self._lock:
            previous = dict(self._devices)
            merged = dict(previous.get(record["node_id"], {}))
            merged.update(record)
            merged["state"] = "paired"
            self._devices[record["node_id"]] = merged
            try:
                self._save()
            except Exception:
                self._devices = previous
                raise
            self.load_error = None

    def get(self, node_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._devices.get(node_id)
            return dict(record) if record is not None else None

    def list_paired(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(self._devices[key]) for key in sorted(self._devices)]

    def remove(self, node_id: str) -> bool:
        with self._lock:
            if node_id not in self._devices:
                return False
            previous = dict(self._devices)
            del self._devices[node_id]
            try:
                self._save()
            except Exception:
                self._devices = previous
                raise
            return True


class DeviceRegistryBridge:
    """Adapt Module A's ``DeviceRegistry`` to the pairing device-store protocol.

    Module A ships concurrently; its exact merge API is its own deliverable.
    Each operation feature-detects the documented candidates in order and
    raises a clear error if none exist, so an API drift fails loudly at
    integration time rather than silently dropping peers.

    Integration note: the real ``DeviceRegistry.mark_paired`` takes keyword
    fields, not a record dict, and its ``merge`` only persists *already
    paired* devices — routing a pairing approval through ``merge`` would
    leave the new peer memory-only. ``mark_paired`` is therefore detected
    first and called with the record's fields.
    """

    _PUT = ("upsert", "merge", "put", "add", "set_paired")
    _REMOVE = ("remove", "delete", "unpair", "drop")
    _GET = ("get", "device", "find")
    _LIST = ("list_paired", "paired", "list", "devices")

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def _call(self, names: Iterable[str], *args: Any) -> Any:
        for name in names:
            method = getattr(self._registry, name, None)
            if callable(method):
                return method(*args)
        raise PairingStateError(
            "DeviceRegistry exposes none of "
            + ", ".join(names)
            + "; update DeviceRegistryBridge for Module A's API"
        )

    def put_paired(self, record: dict[str, Any]) -> None:
        mark_paired = getattr(self._registry, "mark_paired", None)
        if callable(mark_paired):
            mark_paired(
                record["node_id"],
                friendly_name=record.get("friendly_name") or None,
                caps=record.get("caps") or None,
                addrs=record.get("last_addrs") or None,
                paired_at=record.get("paired_at"),
            )
            return
        self._call(self._PUT, record)

    def get(self, node_id: str) -> dict[str, Any] | None:
        result = self._call(self._GET, node_id)
        return dict(result) if isinstance(result, dict) else None

    def list_paired(self) -> list[dict[str, Any]]:
        result = self._call(self._LIST)
        if result is None:
            return []
        return [dict(item) for item in result]

    def remove(self, node_id: str) -> bool:
        return bool(self._call(self._REMOVE, node_id))


def _coerce_device_store(registry: Any, base_path: Path) -> Any:
    """Return a device store: injected Module A registry, protocol object, or fallback."""

    if registry is None:
        return JsonDeviceStore(base_path)
    if all(
        callable(getattr(registry, name, None))
        for name in ("put_paired", "get", "list_paired", "remove")
    ):
        return registry
    return DeviceRegistryBridge(registry)


# --- Local identity (Module A bridge) ---------------------------------------


def _default_friendly_name() -> str:
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["scutil", "--get", "LocalHostName"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            name = result.stdout.strip()
            if result.returncode == 0 and name:
                return name
        except (OSError, subprocess.TimeoutExpired):
            pass
    return socket.gethostname()


def _identity_from_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": str(payload["node_id"]),
        "friendly_name": str(payload["friendly_name"]),
        "created_at": float(payload["created_at"]),
        "schema_version": IDENTITY_SCHEMA_VERSION,
    }


def _fallback_identity(base_path: Path) -> dict[str, Any]:
    """``load_or_create`` against the same file Module A's identity.py owns.

    Same path (``~/.omlx/cluster/identity.json``), same schema_version 1,
    same 0600 atomic write — when Module A merges it adopts this identity
    instead of minting a second node_id.
    """

    path = Path(base_path) / "cluster" / "identity.json"
    if path.exists():
        payload = _read_json_object(path, IDENTITY_SCHEMA_VERSION, "cluster identity")
        return _identity_from_dict(payload)
    identity = {
        "node_id": str(uuid.uuid4()),
        "friendly_name": _default_friendly_name(),
        "created_at": time.time(),
        "schema_version": IDENTITY_SCHEMA_VERSION,
    }
    _atomic_write_json(path, identity)
    return identity


def load_node_identity(base_path: Path) -> dict[str, Any]:
    """Prefer Module A's ``identity.load_or_create``; fall back locally."""

    try:
        from .identity import (
            load_or_create as _load_or_create,  # type: ignore[import-not-found]
        )
    except ImportError:
        return _fallback_identity(base_path)
    identity = _load_or_create(base_path / "cluster" / "identity.json")
    return {
        "node_id": identity.node_id,
        "friendly_name": identity.friendly_name,
        "created_at": float(getattr(identity, "created_at", time.time())),
        "schema_version": getattr(identity, "schema_version", IDENTITY_SCHEMA_VERSION),
    }


# --- Pending requests (memory-only, like enrollment.py join keys) ------------


@dataclass
class _PendingRequest:
    node_id: str
    friendly_name: str
    caps: dict[str, Any]
    code_hash: str
    code_salt: bytes
    addrs: list[str]
    http_port: int | None
    ssh_public_key: str | None
    ssh_host_public_key: str
    created_at: float
    expires_at: float
    attempts: int = 0
    locked_until: float | None = None
    approving: bool = False

    def to_dict(self, now: float) -> dict[str, Any]:
        locked = self.locked_until is not None and self.locked_until > now
        return {
            "node_id": self.node_id,
            "friendly_name": self.friendly_name,
            "caps": self.caps,
            "addrs": list(self.addrs),
            "http_port": self.http_port,
            "state": "awaiting_approval",
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "attempts": self.attempts,
            "locked": locked,
            "locked_until": self.locked_until if locked else None,
            "approving": self.approving,
        }


# --- SSH TOFU enrollment drivers (defaults drive omlx/cluster/ssh_keys.py) ---


def default_enrollment_driver(peer: dict[str, Any]) -> dict[str, Any]:
    """Drive the existing fail-closed SSH TOFU enrollment for an approved peer.

    Reuses the exact primitives of the legacy 3-step flow instead of
    reinventing trust stores:

    * ``ssh_keys.get_or_create_ssh_key`` — the coordinator's dedicated
      ``~/.ssh/omlx_cluster`` identity must exist;
    * ``ssh_keys.install_authorized_key`` — the joiner's user key (sent in
      the join request) is authorized locally so the joiner can be a client;
    * ``ssh_keys.pin_enrolled_host_key`` — pin the code-authenticated daemon
      host key for each coordinator-observed address, refusing changed hosts.
    """

    from .ssh_keys import (
        get_or_create_ssh_key,
        install_authorized_key,
        pin_enrolled_host_key,
    )

    key_pair = get_or_create_ssh_key()
    result: dict[str, Any] = {
        "coordinator_fingerprint": key_pair.fingerprint,
        "authorized_key_installed": False,
        "host_keys_pinned": [],
    }
    peer_public_key = normalize_ssh_public_key(peer.get("ssh_public_key") or "")
    peer_host_key = normalize_ssh_public_key(peer.get("ssh_host_public_key") or "")
    addresses = [str(address) for address in (peer.get("addrs") or []) if address]
    if not addresses:
        raise EnrollmentDriveError("SSH enrollment requires a verified peer address")
    result["authorized_key_installed"] = install_authorized_key(
        public_key=peer_public_key
    )
    for address in addresses:
        target = ssh_host_target(address)
        if pin_enrolled_host_key(hostname=target, public_key=peer_host_key):
            result["host_keys_pinned"].append(address)
    return result


def default_revocation_driver(revocation: dict[str, Any]) -> dict[str, Any]:
    """Best-effort undo of what ``default_enrollment_driver`` installed.

    The device-store removal (the trust anchor) already happened before this
    runs; here we only retract SSH material: the peer's authorized_keys line
    and its known_hosts pins.  Errors are collected, never raised — a
    partially cleaned SSH file must not resurrect a removed device.
    """

    result: dict[str, Any] = {"authorized_key_removed": False, "errors": []}
    ssh_dir = Path.home() / ".ssh"
    peer_public_key = (revocation.get("peer_public_key") or "").strip()
    if peer_public_key:
        authorized_keys = ssh_dir / "authorized_keys"
        try:
            if authorized_keys.exists():
                lines = authorized_keys.read_text(encoding="utf-8").splitlines()
                kept = [line for line in lines if line.strip() != peer_public_key]
                if len(kept) != len(lines):
                    authorized_keys.write_text(
                        "".join(line + "\n" for line in kept), encoding="utf-8"
                    )
                    os.chmod(authorized_keys, 0o600)
                    result["authorized_key_removed"] = True
        except OSError as exc:
            result["errors"].append(f"authorized_keys: {exc}")
    keygen = "/usr/bin/ssh-keygen"
    for address in revocation.get("addrs") or []:
        try:
            subprocess.run(
                [keygen, "-R", address],
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["errors"].append(f"known_hosts {address}: {exc}")
    return result


def _default_http_post(url: str, payload: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _default_http_get(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _local_ssh_public_key() -> str | None:
    """Best-effort local cluster pubkey for the join request (never fatal)."""

    try:
        from .ssh_keys import get_or_create_ssh_key
    except ImportError:
        return None
    try:
        return get_or_create_ssh_key().public_key
    except Exception:
        return None


def _local_ssh_host_public_key() -> str | None:
    """Read the public half of the local SSH daemon host identity."""

    override = os.environ.get("OMLX_CLUSTER_SSH_HOST_PUBLIC_KEY")
    candidates = (
        [Path(override).expanduser()]
        if override
        else [
            Path("/etc/ssh/ssh_host_ed25519_key.pub"),
            Path("/etc/ssh/ssh_host_rsa_key.pub"),
        ]
    )
    for path in candidates:
        try:
            return normalize_ssh_public_key(path.read_text(encoding="utf-8").strip())
        except (OSError, PairingRequestError):
            continue
    return None


def normalize_ssh_public_key(public_key: str) -> str:
    """Validate and strip an OpenSSH user key to its type and key data."""

    if not isinstance(public_key, str) or "\n" in public_key or "\r" in public_key:
        raise PairingRequestError("join request carries an invalid SSH public key")
    parts = public_key.strip().split()
    if len(parts) < 2:
        raise PairingRequestError("join request carries an invalid SSH public key")
    normalized = " ".join(parts[:2])
    try:
        from .ssh_keys import ssh_public_key_fingerprint

        ssh_public_key_fingerprint(normalized)
    except (ImportError, RuntimeError, ValueError) as exc:
        raise PairingRequestError(
            "join request carries an invalid SSH public key"
        ) from exc
    return normalized


def ssh_host_target(address: str) -> str:
    """Format one validated IP for the existing known_hosts API."""

    raw = str(address).strip()
    if "%" in raw:
        raise EnrollmentDriveError(
            "SSH enrollment requires an address without an IPv6 scope suffix"
        )
    try:
        parsed = ipaddress.ip_address(raw)
    except ValueError as exc:
        raise EnrollmentDriveError(
            "SSH enrollment received an invalid address"
        ) from exc
    return f"[{parsed}]" if parsed.version == 6 else str(parsed)


# --- PairingManager -----------------------------------------------------------


class PairingManager:
    """Joiner + coordinator sides of code-based pairing.

    Parameters are dependency-injection seams; every default is production
    behavior and every test substitutes fakes:

    * ``registry`` — Module A ``DeviceRegistry`` (feature-detected via
      :class:`DeviceRegistryBridge`), an object already speaking the
      device-store protocol, or ``None`` for the schema-compatible
      :class:`JsonDeviceStore` fallback.
    * ``enrollment_store`` — the legacy ``ClusterEnrollmentStore``; consulted
      on ``unpair`` so CUDA-enrolled entries with the same node_id are
      dropped too.  Optional.
    * ``identity`` — dict with ``node_id``/``friendly_name``; default bridges
      to Module A's identity module (or the compatible local file).
    * ``enrollment_driver`` / ``revocation_driver`` — SSH TOFU trust
      install/retract steps; defaults drive ``omlx/cluster/ssh_keys.py``.
    * ``http_post`` / ``http_get`` — joiner-side transport; loopback fakes
      in tests, urllib in production.
    """

    def __init__(
        self,
        registry: Any = None,
        enrollment_store: Any = None,
        *,
        base_path: Path | None = None,
        identity: dict[str, Any] | None = None,
        key_store: PairingKeyStore | None = None,
        caps_provider: Callable[[], dict[str, Any]] | None = None,
        address_provider: Callable[[], list[str]] | None = None,
        ssh_key_provider: Callable[[], str | None] | None = None,
        ssh_host_key_provider: Callable[[], str | None] | None = None,
        enrollment_driver: Callable[[dict[str, Any]], Any] | None = None,
        revocation_driver: Callable[[dict[str, Any]], Any] | None = None,
        http_post: Callable[[str, dict[str, Any], float], Any] | None = None,
        http_get: Callable[[str, float], Any] | None = None,
        clock: Callable[[], float] = time.time,
        audit: PairingAuditLog | Callable[..., Any] | None = None,
    ) -> None:
        self.base_path = Path(base_path) if base_path is not None else DEFAULT_BASE_PATH
        self._identity = (
            dict(identity)
            if identity is not None
            else load_node_identity(self.base_path)
        )
        self._devices = _coerce_device_store(registry, self.base_path)
        self._enrollment_store = enrollment_store
        self._key_store = (
            key_store
            if key_store is not None
            else PairingKeyStore(self.base_path, clock=clock)
        )
        self._caps_provider = caps_provider
        self._address_provider = address_provider
        self._ssh_key_provider = ssh_key_provider or _local_ssh_public_key
        self._ssh_host_key_provider = (
            ssh_host_key_provider or _local_ssh_host_public_key
        )
        self._enrollment_driver = enrollment_driver or default_enrollment_driver
        self._revocation_driver = revocation_driver or default_revocation_driver
        self._http_post = http_post or _default_http_post
        self._http_get = http_get or _default_http_get
        self._clock = clock
        if audit is None:
            audit = PairingAuditLog(self.base_path / "cluster" / "pairing-audit.jsonl")
        self._audit = audit
        self._lock = threading.RLock()
        self._pending: dict[str, _PendingRequest] = {}
        self._denied: dict[str, float] = {}
        self._local_code: dict[str, Any] | None = None

    # -- helpers ------------------------------------------------------------

    @property
    def node_id(self) -> str:
        return self._identity["node_id"]

    @property
    def friendly_name(self) -> str:
        return self._identity["friendly_name"]

    def local_addrs(self) -> list[str]:
        """Validated local IPs included in the peer's approved trust record."""

        if self._address_provider is None:
            return []
        try:
            candidates = self._address_provider()
        except Exception:
            return []
        addresses: list[str] = []
        for candidate in candidates or []:
            raw = str(candidate).strip()
            try:
                ipaddress.ip_address(raw.split("%", 1)[0])
            except ValueError:
                continue
            normalized = raw
            if normalized not in addresses:
                addresses.append(normalized)
            if len(addresses) >= 8:
                break
        return addresses

    def local_ssh_material(self) -> dict[str, Any]:
        """Complete local trust material the opposite peer must install."""

        addresses = self.local_addrs()
        if not addresses:
            raise EnrollmentDriveError(
                "SSH enrollment requires a verified local address"
            )
        return {
            "node_id": self.node_id,
            "friendly_name": self.friendly_name,
            "caps": self._caps_provider() if self._caps_provider else {},
            "addrs": addresses,
            "ssh_public_key": normalize_ssh_public_key(self._ssh_key_provider() or ""),
            "ssh_host_public_key": normalize_ssh_public_key(
                self._ssh_host_key_provider() or ""
            ),
        }

    def _record_audit(
        self, event: str, *, node_id: str = "", detail: dict[str, Any] | None = None
    ) -> None:
        record = getattr(self._audit, "record", None)
        if callable(record):
            record(event, node_id=node_id, detail=detail)
        elif callable(self._audit):
            self._audit(event, node_id=node_id, detail=detail)

    def _prune_pending(self, now: float, keep: str | None = None) -> None:
        expired = [
            node_id
            for node_id, pending in self._pending.items()
            if pending.expires_at < now and node_id != keep
        ]
        for node_id in expired:
            del self._pending[node_id]
            self._record_audit("join_request_expired", node_id=node_id)
        self._denied = {
            node_id: ts for node_id, ts in self._denied.items() if ts + 3600 > now
        }

    # -- joiner side ----------------------------------------------------------

    def start_join(self) -> dict[str, Any]:
        """Generate the 6-digit code the joiner displays; valid 10 minutes."""

        code = generate_pairing_code()
        now = self._clock()
        self._local_code = {
            "code": code,
            "created_at": now,
            "expires_at": now + CODE_TTL_SECONDS,
            "completing": False,
        }
        return {"code": code, "expires_at": self._local_code["expires_at"]}

    def build_join_request(self, code: str | None = None) -> dict[str, Any]:
        """Assemble the pair/request payload — identity + code hash, no code."""

        if code is None:
            if self._local_code is None:
                raise PairingStateError("no join in progress; call start_join first")
            if self._local_code["expires_at"] < self._clock():
                raise PairingExpiredError("the displayed pairing code has expired")
            code = self._local_code["code"]
        validate_pairing_code(code)
        caps = self._caps_provider() if self._caps_provider else {}
        ssh_public_key = normalize_ssh_public_key(self._ssh_key_provider() or "")
        ssh_host_public_key = normalize_ssh_public_key(
            self._ssh_host_key_provider() or ""
        )
        code_salt = secrets.token_bytes(16)
        return {
            "node_id": self.node_id,
            "friendly_name": self.friendly_name,
            "caps": dict(caps),
            "code_hash": pairing_code_hash(
                code,
                self.node_id,
                ssh_public_key,
                ssh_host_public_key,
                code_salt,
            ),
            "code_salt": base64.b64encode(code_salt).decode("ascii"),
            "http_port": None,
            "addrs": self.local_addrs(),
            "ssh_public_key": ssh_public_key,
            "ssh_host_public_key": ssh_host_public_key,
        }

    def request_join(
        self,
        coordinator_addr: str,
        code: str | None = None,
        *,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        """POST ``/api/cluster/pair/request`` to the coordinator."""

        payload = self.build_join_request(code)
        url = f"http://{coordinator_addr}/api/cluster/pair/request"
        response = self._http_post(url, payload, timeout)
        self._record_audit(
            "join_requested",
            node_id=self.node_id,
            detail={"coordinator": coordinator_addr},
        )
        return {"state": "awaiting_approval", "response": response}

    def poll_join(
        self, coordinator_addr: str, *, timeout: float = 5.0
    ) -> dict[str, Any]:
        """GET the coordinator's view of this node's pairing status."""

        url = f"http://{coordinator_addr}/api/cluster/pair/status/{self.node_id}"
        return self._http_get(url, timeout)

    def complete_join(
        self, status: dict[str, Any], code: str | None = None
    ) -> dict[str, Any]:
        """Unwrap the approved cluster key and persist the coordinator as paired.

        ``status`` is the payload from :meth:`poll_join` (or the loopback
        equivalent).  Fail-closed: a wrong code or tampered package raises
        and nothing is stored.
        """

        if status.get("state") != "approved":
            raise PairingStateError(
                f"coordinator has not approved this node (state={status.get('state')!r})"
            )
        package = status.get("cluster_key_package")
        if not isinstance(package, dict):
            raise PairingRequestError("approval status is missing the wrapped key")
        with self._lock:
            local_code = self._local_code
            if local_code is None:
                raise PairingStateError("no join in progress; call start_join first")
            if local_code["expires_at"] < self._clock():
                self._local_code = None
                raise PairingExpiredError("the displayed pairing code has expired")
            if local_code.get("completing"):
                raise PairingStateError("join completion is already in progress")
            if code is None:
                code = local_code["code"]
            local_code["completing"] = True
        try:
            cluster_key = unwrap_cluster_key(package, code)
        except Exception:
            with self._lock:
                if self._local_code is local_code:
                    local_code["completing"] = False
            raise
        coordinator = status.get("coordinator") or {}
        coordinator_id = str(coordinator.get("node_id") or "")
        if not coordinator_id:
            with self._lock:
                if self._local_code is local_code:
                    local_code["completing"] = False
            raise PairingRequestError("approval status is missing coordinator identity")
        coordinator_peer = {
            "node_id": coordinator_id,
            "friendly_name": coordinator.get("friendly_name", ""),
            "caps": coordinator.get("caps") or {},
            "addrs": list(coordinator.get("addrs") or []),
            "ssh_public_key": coordinator.get("ssh_public_key"),
            "ssh_host_public_key": coordinator.get("ssh_host_public_key"),
        }
        observed_tag = str(status.get("coordinator_identity_tag") or "")
        try:
            expected_tag = coordinator_identity_tag(cluster_key, coordinator_peer)
        except Exception:
            with self._lock:
                if self._local_code is local_code:
                    local_code["completing"] = False
            raise
        if not hmac.compare_digest(observed_tag, expected_tag):
            with self._lock:
                if self._local_code is local_code:
                    local_code["completing"] = False
            raise PairingCodeError("approval status coordinator identity was altered")
        enrolled = False
        try:
            self._enrollment_driver(coordinator_peer)
            enrolled = True
        except Exception as exc:
            self._record_audit(
                "join_enrollment_failed",
                node_id=coordinator_id,
                detail={"error": str(exc)},
            )
            with self._lock:
                if self._local_code is local_code:
                    local_code["completing"] = False
            raise EnrollmentDriveError(
                f"SSH enrollment failed; coordinator was not paired: {exc}"
            ) from exc
        paired_at = self._clock()
        record = {
            "node_id": coordinator_id,
            "friendly_name": coordinator.get("friendly_name", ""),
            "caps": coordinator.get("caps") or {},
            "paired_at": paired_at,
            "last_addrs": coordinator_peer["addrs"],
            "state": "paired",
            "role": "coordinator",
        }
        key_record = {
            "cluster_key": cluster_key.hex(),
            "peer_public_key": coordinator.get("ssh_public_key"),
            "addrs": coordinator_peer["addrs"],
            "paired_at": paired_at,
            "role": "coordinator",
        }
        key_written = False
        try:
            with self._lock:
                if self._local_code is not local_code or not local_code.get(
                    "completing"
                ):
                    raise PairingStateError(
                        "join state changed while completion was in progress"
                    )
                self._key_store.set(coordinator_id, key_record)
                key_written = True
                self._devices.put_paired(record)
                self._local_code = None
        except Exception as exc:
            rollback_errors: list[str] = []
            if key_written:
                try:
                    self._key_store.remove(coordinator_id)
                except Exception as rollback_exc:
                    rollback_errors.append(f"pairing key: {rollback_exc}")
            if enrolled:
                try:
                    self._revocation_driver(
                        {
                            "peer_public_key": coordinator.get("ssh_public_key"),
                            "addrs": coordinator_peer["addrs"],
                        }
                    )
                except Exception as rollback_exc:
                    rollback_errors.append(f"SSH enrollment: {rollback_exc}")
            with self._lock:
                if self._local_code is local_code:
                    local_code["completing"] = False
            self._record_audit(
                "join_persistence_failed",
                node_id=coordinator_id,
                detail={"error": str(exc), "rollback_errors": rollback_errors},
            )
            suffix = (
                "; rollback issues: " + "; ".join(rollback_errors)
                if rollback_errors
                else ""
            )
            raise PairingError(
                "join state could not be persisted; coordinator was not paired: "
                f"{exc}{suffix}"
            ) from exc
        self._record_audit("join_completed", node_id=coordinator_id)
        return record

    # -- coordinator side -----------------------------------------------------

    def handle_join_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Register a joiner's pair/request as ``awaiting_approval``.

        The code itself never arrives — only a salted PBKDF2 verifier bound to
        the node and both SSH identities — so this endpoint can stay
        unauthenticated; the admin's approve step (typing the code shown on the
        joiner) is the trust decision.
        """

        if not isinstance(payload, dict):
            raise PairingRequestError("join request must be an object")
        node_id = str(payload.get("node_id") or "").strip()
        friendly_name = str(payload.get("friendly_name") or "").strip()
        code_hash = str(payload.get("code_hash") or "").strip()
        code_salt_encoded = str(payload.get("code_salt") or "").strip()
        caps = payload.get("caps")
        addrs = payload.get("addrs") or []
        http_port = payload.get("http_port")
        ssh_public_key = payload.get("ssh_public_key")
        ssh_host_public_key = payload.get("ssh_host_public_key")
        if not node_id or len(node_id) > 255:
            raise PairingRequestError("join request is missing a node_id")
        if node_id == self.node_id:
            raise PairingRequestError("a node cannot join its own cluster")
        if not friendly_name or len(friendly_name) > 255:
            raise PairingRequestError("join request is missing a friendly_name")
        if len(code_hash) != 64 or any(c not in "0123456789abcdef" for c in code_hash):
            raise PairingRequestError("join request carries a malformed code_hash")
        try:
            code_salt = base64.b64decode(code_salt_encoded, validate=True)
        except (binascii.Error, ValueError, TypeError) as exc:
            raise PairingRequestError(
                "join request carries a malformed code_salt"
            ) from exc
        if len(code_salt) != 16:
            raise PairingRequestError("join request carries a malformed code_salt")
        if not isinstance(caps, dict):
            raise PairingRequestError("join request caps must be an object")
        if not isinstance(addrs, list) or not all(isinstance(a, str) for a in addrs):
            raise PairingRequestError("join request addrs must be a list of strings")
        normalized_addrs: list[str] = []
        for address in addrs:
            raw = address.strip()
            try:
                ipaddress.ip_address(raw.split("%", 1)[0])
            except ValueError as exc:
                raise PairingRequestError(
                    "join request carries an invalid peer address"
                ) from exc
            if raw not in normalized_addrs:
                normalized_addrs.append(raw)
        if http_port is not None and not (1 <= int(http_port) <= 65535):
            raise PairingRequestError("join request http_port is out of range")
        ssh_public_key = normalize_ssh_public_key(ssh_public_key or "")
        ssh_host_public_key = normalize_ssh_public_key(ssh_host_public_key or "")

        with self._lock:
            now = self._clock()
            self._prune_pending(now)
            existing = self._pending.get(node_id)
            if existing is not None:
                same_request = (
                    hmac.compare_digest(existing.code_hash, code_hash)
                    and hmac.compare_digest(existing.code_salt, code_salt)
                    and existing.ssh_public_key == ssh_public_key
                    and existing.ssh_host_public_key == ssh_host_public_key
                    and existing.addrs == normalized_addrs[:8]
                )
                if same_request:
                    return existing.to_dict(now)
                raise PairingStateError(
                    "a different join request is already pending for this node_id"
                )
            live = [p for p in self._pending.values() if p.node_id != node_id]
            if len(live) >= MAX_PENDING_REQUESTS:
                raise PairingRequestError(
                    "too many pending join requests; approve or deny one first"
                )
            request = _PendingRequest(
                node_id=node_id,
                friendly_name=friendly_name,
                caps=dict(caps),
                code_hash=code_hash,
                code_salt=code_salt,
                addrs=normalized_addrs[:8],
                http_port=int(http_port) if http_port is not None else None,
                ssh_public_key=ssh_public_key,
                ssh_host_public_key=ssh_host_public_key,
                created_at=now,
                expires_at=now + CODE_TTL_SECONDS,
            )
            self._pending[node_id] = request
            self._denied.pop(node_id, None)
        self._record_audit(
            "join_request_received",
            node_id=node_id,
            detail={"friendly_name": friendly_name, "addrs": list(request.addrs)},
        )
        return request.to_dict(self._clock())

    def pending_requests(self) -> list[dict[str, Any]]:
        """Pending requests for the devices list (``state: awaiting_approval``)."""

        with self._lock:
            now = self._clock()
            self._prune_pending(now)
            return [self._pending[key].to_dict(now) for key in sorted(self._pending)]

    def approve(self, node_id: str, code: str) -> dict[str, Any]:
        """Verify the displayed code, issue the cluster key, enroll the peer.

        Order is fail-closed: code verify → key wrap → SSH TOFU enrollment →
        only then persist the peer as paired.  If enrollment driving raises,
        the peer is NOT stored; the pending request survives for a retry.
        """

        validate_pairing_code(code)
        with self._lock:
            now = self._clock()
            # Expire *other* pending requests, but keep this one long enough
            # to report a precise "expired" instead of a generic "no pending".
            self._prune_pending(now, keep=node_id)
            pending = self._pending.get(node_id)
            if pending is None:
                if node_id in self._denied:
                    raise PairingStateError("this join request was denied")
                raise PairingStateError("no pending join request for this node_id")
            if pending.expires_at < now:
                del self._pending[node_id]
                self._record_audit("join_request_expired", node_id=node_id)
                raise PairingExpiredError("the pairing code has expired")
            if pending.approving:
                raise PairingStateError("approval is already in progress")
            if pending.locked_until is not None and pending.locked_until > now:
                self._record_audit(
                    "approve_locked_out",
                    node_id=node_id,
                    detail={"retry_after": pending.locked_until},
                )
                raise PairingLockoutError(
                    f"too many wrong codes; locked until {pending.locked_until:.0f}"
                )
            if pending.locked_until is not None and pending.locked_until <= now:
                # Lockout served: fresh set of attempts, code still valid by TTL.
                pending.locked_until = None
                pending.attempts = 0
            expected = pairing_code_hash(
                code,
                node_id,
                pending.ssh_public_key or "",
                pending.ssh_host_public_key,
                pending.code_salt,
            )
            if not hmac.compare_digest(expected, pending.code_hash):
                pending.attempts += 1
                detail = {"attempts": pending.attempts}
                if pending.attempts >= MAX_CODE_ATTEMPTS:
                    pending.locked_until = now + LOCKOUT_SECONDS
                    detail["locked_until"] = pending.locked_until
                    self._record_audit(
                        "approve_lockout", node_id=node_id, detail=detail
                    )
                    raise PairingLockoutError(
                        f"too many wrong codes; locked until {pending.locked_until:.0f}"
                    )
                self._record_audit("approve_wrong_code", node_id=node_id, detail=detail)
                raise PairingCodeError(
                    f"code does not match ({MAX_CODE_ATTEMPTS - pending.attempts} "
                    "attempts left)"
                )
            # Reserve this exact pending request before leaving the manager
            # lock for SSH I/O. Deny, unpair, and duplicate approval must not
            # race an enrollment that has already passed code verification.
            pending.approving = True

        try:
            coordinator_material = self.local_ssh_material()
        except Exception as exc:
            self._record_audit(
                "approve_enrollment_failed",
                node_id=node_id,
                detail={"error": str(exc)},
            )
            with self._lock:
                if self._pending.get(node_id) is pending:
                    pending.approving = False
            raise EnrollmentDriveError(
                f"SSH enrollment failed; peer was not paired: {exc}"
            ) from exc

        cluster_key = secrets.token_bytes(CLUSTER_KEY_BYTES)
        package = wrap_cluster_key(cluster_key, code)
        coordinator_tag = coordinator_identity_tag(
            cluster_key,
            coordinator_material,
        )

        peer = {
            "node_id": pending.node_id,
            "friendly_name": pending.friendly_name,
            "caps": pending.caps,
            "addrs": list(pending.addrs),
            "http_port": pending.http_port,
            "ssh_public_key": pending.ssh_public_key,
            "ssh_host_public_key": pending.ssh_host_public_key,
        }
        try:
            enrollment = self._enrollment_driver(peer)
        except Exception as exc:
            self._record_audit(
                "approve_enrollment_failed", node_id=node_id, detail={"error": str(exc)}
            )
            with self._lock:
                if self._pending.get(node_id) is pending:
                    pending.approving = False
            raise EnrollmentDriveError(
                f"SSH enrollment failed; peer was not paired: {exc}"
            ) from exc

        paired_at = self._clock()
        record = {
            "node_id": pending.node_id,
            "friendly_name": pending.friendly_name,
            "caps": pending.caps,
            "paired_at": paired_at,
            "last_addrs": list(pending.addrs),
            "state": "paired",
            "role": "peer",
        }
        key_record = {
            "cluster_key": cluster_key.hex(),
            # The joiner retrieves this package via pair/status and unwraps it
            # with the code; safe to persist (0600) and serve.
            "cluster_key_package": package,
            "coordinator": coordinator_material,
            "coordinator_identity_tag": coordinator_tag,
            "peer_public_key": pending.ssh_public_key,
            "addrs": list(pending.addrs),
            "paired_at": paired_at,
            "role": "peer",
        }
        key_written = False
        try:
            with self._lock:
                if self._pending.get(node_id) is not pending or not pending.approving:
                    raise PairingStateError(
                        "join request changed while approval was in progress"
                    )
                self._key_store.set(pending.node_id, key_record)
                key_written = True
                self._devices.put_paired(record)
                self._pending.pop(node_id, None)
        except Exception as exc:
            rollback_errors: list[str] = []
            if key_written:
                try:
                    self._key_store.remove(pending.node_id)
                except Exception as rollback_exc:
                    rollback_errors.append(f"pairing key: {rollback_exc}")
            try:
                self._revocation_driver(
                    {
                        "peer_public_key": pending.ssh_public_key,
                        "addrs": list(pending.addrs),
                    }
                )
            except Exception as rollback_exc:
                rollback_errors.append(f"SSH enrollment: {rollback_exc}")
            with self._lock:
                if self._pending.get(node_id) is pending:
                    pending.approving = False
            detail = {"error": str(exc), "rollback_errors": rollback_errors}
            self._record_audit(
                "approve_persistence_failed",
                node_id=node_id,
                detail=detail,
            )
            suffix = (
                "; rollback issues: " + "; ".join(rollback_errors)
                if rollback_errors
                else ""
            )
            raise PairingError(
                f"pairing state could not be persisted; peer was not paired: "
                f"{exc}{suffix}"
            ) from exc
        self._record_audit(
            "approve_success",
            node_id=node_id,
            detail={"friendly_name": pending.friendly_name},
        )
        return {
            "node_id": node_id,
            "state": "paired",
            "paired_at": paired_at,
            "cluster_key_package": package,
            "coordinator": coordinator_material,
            "coordinator_identity_tag": coordinator_tag,
            "enrollment": enrollment,
        }

    def deny(self, node_id: str) -> bool:
        """Refuse a pending request; the joiner sees ``denied`` on its next poll."""

        with self._lock:
            now = self._clock()
            self._prune_pending(now)
            pending = self._pending.get(node_id)
            if pending is None:
                return False
            if pending.approving:
                raise PairingStateError("approval is already in progress")
            self._pending.pop(node_id, None)
            self._denied[node_id] = now
        self._record_audit("join_request_denied", node_id=node_id)
        return True

    def join_status(self, node_id: str) -> dict[str, Any]:
        """Coordinator-side status consumed by the joiner's poll loop.

        The wrapped cluster key is safe to serve here: it is encrypted under
        a PBKDF2 key derived from the code only the joiner operator can see.
        """

        with self._lock:
            now = self._clock()
            self._prune_pending(now)
            pending = self._pending.get(node_id)
            denied = node_id in self._denied
        if pending is not None:
            return pending.to_dict(now)
        key_record = self._key_store.get(node_id)
        if key_record is not None:
            device = self._devices.get(node_id) or {}
            status: dict[str, Any] = {
                "node_id": node_id,
                "state": "approved",
                "paired_at": key_record.get("paired_at"),
                # Same coordinator block approve() returns: the joiner
                # persists it via complete_join, so caps/ssh key must ride
                # here too — the poll path is the only one the UI drives.
                "coordinator": key_record.get("coordinator") or {},
                "coordinator_identity_tag": key_record.get("coordinator_identity_tag"),
            }
            package = key_record.get("cluster_key_package")
            if package is not None:
                status["cluster_key_package"] = package
            if device:
                status["friendly_name"] = device.get("friendly_name")
            return status
        if denied:
            return {"node_id": node_id, "state": "denied"}
        return {"node_id": node_id, "state": "unknown"}

    def unpair(self, node_id: str) -> dict[str, Any]:
        """Remove a paired device and revoke the trust installed at pairing.

        Registry removal is the trust anchor and happens first (fail-closed);
        SSH material retraction is best-effort and reported, never raised.
        Also drops any legacy ``enrolled-nodes.json`` entry with the same
        node_id so both trusted-node inventories stay in step.
        """

        with self._lock:
            now = self._clock()
            self._prune_pending(now)
            pending = self._pending.get(node_id)
            if pending is not None and pending.approving:
                raise PairingStateError("approval is already in progress")
            was_pending = self._pending.pop(node_id, None) is not None
        device = self._devices.get(node_id)
        removed_device = self._devices.remove(node_id)
        key_record = self._key_store.get(node_id)
        if key_record is not None:
            self._key_store.remove(node_id)
        if not (removed_device or key_record or was_pending):
            raise PairingStateError("unknown device")

        removed_enrollment = False
        if self._enrollment_store is not None:
            remove_node = getattr(self._enrollment_store, "remove_node", None)
            if callable(remove_node):
                with suppress(Exception):
                    removed_enrollment = bool(remove_node(node_id))

        revocation: dict[str, Any] = {"authorized_key_removed": False, "errors": []}
        material = {
            "peer_public_key": (key_record or {}).get("peer_public_key"),
            "addrs": (key_record or {}).get("addrs")
            or (device or {}).get("last_addrs")
            or [],
        }
        if material["peer_public_key"] or material["addrs"]:
            try:
                revocation = self._revocation_driver(material) or revocation
            except Exception as exc:
                revocation["errors"].append(str(exc))

        self._record_audit(
            "device_unpaired",
            node_id=node_id,
            detail={
                "removed_device": removed_device,
                "removed_key": key_record is not None,
                "removed_enrollment": removed_enrollment,
                "was_pending": was_pending,
            },
        )
        return {
            "node_id": node_id,
            "unpaired": True,
            "removed_device": removed_device,
            "removed_key": key_record is not None,
            "removed_enrollment": removed_enrollment,
            "was_pending": was_pending,
            "revocation": revocation,
        }

    def paired_devices(self) -> list[dict[str, Any]]:
        """Persisted paired devices (discovered-but-unpaired never land here)."""

        return self._devices.list_paired()

    def devices_view(self) -> dict[str, Any]:
        """Coordinator devices list: paired + awaiting_approval + self."""

        return {
            "paired": self.paired_devices(),
            "pending": self.pending_requests(),
            "self": {
                "node_id": self.node_id,
                "friendly_name": self.friendly_name,
                "caps": self._caps_provider() if self._caps_provider else {},
            },
        }


# --- Configured singleton (mirrors enrollment.py's configure/get pattern) ----

_manager_lock = threading.Lock()
_configured_manager: PairingManager | None = None


def configure_pairing_manager(
    base_path: Path,
    *,
    registry: Any = None,
    enrollment_store: Any = None,
    **kwargs: Any,
) -> PairingManager:
    """Create the process-wide manager at server startup.

    ``registry`` should be Module A's ``DeviceRegistry`` once that branch is
    integrated; until then ``None`` selects the schema-compatible fallback.
    """

    global _configured_manager
    with _manager_lock:
        _configured_manager = PairingManager(
            registry,
            enrollment_store,
            base_path=base_path,
            **kwargs,
        )
        return _configured_manager


def get_pairing_manager() -> PairingManager:
    with _manager_lock:
        if _configured_manager is None:
            raise RuntimeError("cluster pairing is not configured")
        return _configured_manager


def reset_pairing_manager() -> None:
    """Test hook: drop the configured manager so tests stay isolated."""

    global _configured_manager
    with _manager_lock:
        _configured_manager = None
