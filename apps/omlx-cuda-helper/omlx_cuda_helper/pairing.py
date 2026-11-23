# SPDX-License-Identifier: Apache-2.0
"""Join a host from a helper that has no screen.

The helper makes a 6-digit code and prints it in its log; the operator types
it in the host's Cluster tab. The code never crosses the network: the host
gets a slow verifier, and returns the cluster key wrapped under the code.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from .state import State

logger = logging.getLogger("omlx_cuda_helper.pairing")

PBKDF2_ITERATIONS = 210_000
CODE_TTL_SECONDS = 600.0
_POLL_SECONDS = 2.0
_WRAP_TAG_DOMAIN = b"omlx-cluster-key-v1"
_WRAP_KDF = "PBKDF2-HMAC-SHA256"


class PairingError(RuntimeError):
    pass


def code_hash(code: str, node_id: str, salt: bytes) -> str:
    """Verifier of *code* bound to this node and to "no SSH identities"."""
    payload = json.dumps(
        {"node_id": node_id, "ssh_public_key": "", "ssh_host_public_key": ""},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.pbkdf2_hmac(
        "sha256",
        code.encode(),
        salt + hashlib.sha256(payload).digest(),
        PBKDF2_ITERATIONS,
        dklen=32,
    ).hex()


def unwrap_cluster_key(package: dict[str, Any], code: str) -> bytes:
    try:
        iterations = int(package["iterations"])
        salt = base64.b64decode(package["salt"], validate=True)
        ciphertext = base64.b64decode(package["ciphertext"], validate=True)
        tag = base64.b64decode(package["tag"], validate=True)
    except (KeyError, TypeError, ValueError) as error:
        raise PairingError("malformed wrapped cluster key") from error
    if package.get("kdf") != _WRAP_KDF or not 1 <= iterations <= 10_000_000:
        raise PairingError("unsupported cluster-key wrap parameters")
    wrap_key = hashlib.pbkdf2_hmac("sha256", code.encode(), salt, iterations, dklen=32)
    expected = hmac.new(
        wrap_key, _WRAP_TAG_DOMAIN + ciphertext, hashlib.sha256
    ).digest()
    if not hmac.compare_digest(tag, expected) or len(ciphertext) != 32:
        raise PairingError("the host answered with another pairing code")
    return bytes(a ^ b for a, b in zip(ciphertext, wrap_key))


def identity_tag(cluster_key: bytes, coordinator: dict[str, Any]) -> str:
    """Binds the host identity it reported to the key only it could wrap."""
    payload = json.dumps(coordinator, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(
        cluster_key, b"omlx-coordinator-v1\0" + payload, hashlib.sha256
    ).hexdigest()


def _http(url: str, body: dict[str, Any] | None = None, timeout: float = 5.0) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read(1 << 20))


class Joiner:
    """At most one join attempt at a time; a new one withdraws the old."""

    def __init__(
        self,
        state: State,
        caps: Callable[[], dict[str, Any]],
        http_port: int,
        *,
        http: Callable[..., dict] = _http,
        poll_seconds: float = _POLL_SECONDS,
    ):
        self.state, self.caps, self.http_port = state, caps, http_port
        self._http, self._poll_seconds = http, poll_seconds
        self._lock = threading.Lock()
        self._attempt: dict[str, Any] | None = None

    def join(self, address: str, port: int, node_id: str | None = None) -> None:
        """Ask the host at *address* to pair, then wait for its approval."""
        host = f"[{address}]" if ":" in address else address
        base = f"http://{host}:{port}/api/cluster"
        try:
            found = self._http(f"{base}/node_id", None, 5.0)
        except (OSError, ValueError) as error:
            raise PairingError(
                f"no oMLX answers at {address}:{port}: {error}"
            ) from error
        if node_id is not None and found.get("node_id") != node_id:
            raise PairingError(f"{address}:{port} is not the oMLX that sent the invite")
        code = f"{secrets.randbelow(10**6):06d}"
        salt = secrets.token_bytes(16)
        cancel = secrets.token_hex(32)
        attempt = {
            "base": base,
            "host_node_id": str(found.get("node_id") or ""),
            "address": address,
            "port": port,
            "code": code,
            "cancel": cancel,
            "expires": time.monotonic() + CODE_TTL_SECONDS,
            "stop": threading.Event(),
        }
        self.cancel()
        try:
            self._http(
                f"{base}/pair/request",
                {
                    "node_id": self.state.node_id,
                    "friendly_name": self.state.name,
                    "caps": self.caps(),
                    "code_hash": code_hash(code, self.state.node_id, salt),
                    "code_salt": base64.b64encode(salt).decode(),
                    "cancel_token_hash": hashlib.sha256(cancel.encode()).hexdigest(),
                    "http_port": self.http_port,
                },
                10.0,
            )
        except (OSError, ValueError) as error:
            raise PairingError(
                f"{address}:{port} refused the request: {error}"
            ) from error
        with self._lock:
            self._attempt = attempt
        logger.info(
            "PAIRING CODE %s  enter it on the Cluster tab of the oMLX at %s "
            "(expires in 10 min)",
            code,
            address,
        )
        threading.Thread(
            target=self._wait, args=(attempt,), name="omlx-helper-join", daemon=True
        ).start()

    def cancel(self) -> None:
        with self._lock:
            attempt, self._attempt = self._attempt, None
        if attempt is None:
            return
        attempt["stop"].set()
        with suppress(OSError, ValueError):
            self._http(
                f"{attempt['base']}/pair/request/cancel",
                {"node_id": self.state.node_id, "token": attempt["cancel"]},
                5.0,
            )

    def _wait(self, attempt: dict[str, Any]) -> None:
        url = f"{attempt['base']}/pair/status/{self.state.node_id}"
        stop: threading.Event = attempt["stop"]
        while not stop.wait(self._poll_seconds):
            if time.monotonic() > attempt["expires"]:
                logger.info("PAIRING expired: the code was not entered in time")
                break
            try:
                status = self._http(url, None, 5.0)
            except (OSError, ValueError):
                continue
            state = status.get("state")
            if state == "awaiting_approval":
                continue
            if state == "approved":
                try:
                    self._finish(attempt, status)
                except PairingError as error:
                    logger.error("PAIRING failed: %s", error)
            elif state == "denied":
                logger.info("PAIRING denied on the host")
            else:
                logger.info("PAIRING failed: the host no longer has the request")
            break
        with self._lock:
            if self._attempt is attempt:
                self._attempt = None

    def _finish(self, attempt: dict[str, Any], status: dict[str, Any]) -> None:
        coordinator = status.get("coordinator")
        if not isinstance(coordinator, dict):
            raise PairingError("the host sent no identity")
        key = unwrap_cluster_key(
            status.get("cluster_key_package") or {}, attempt["code"]
        )
        tag = str(status.get("coordinator_identity_tag") or "")
        if not hmac.compare_digest(tag, identity_tag(key, coordinator)):
            raise PairingError("the host identity does not match its key")
        node_id = str(coordinator.get("node_id") or "")
        if not node_id or node_id != attempt["host_node_id"]:
            raise PairingError("another oMLX answered the pairing")
        name = str(coordinator.get("friendly_name") or node_id)
        self.state.save_host(
            node_id,
            {
                "cluster_key": key.hex(),
                "name": name,
                "addrs": [attempt["address"]],
                "http_port": status.get("coordinator_http_port") or attempt["port"],
                "paired_at": time.time(),
            },
        )
        logger.info("PAIRING approved: paired with %s (%s)", name, node_id)
