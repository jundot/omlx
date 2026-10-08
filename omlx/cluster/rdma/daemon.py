# SPDX-License-Identifier: Apache-2.0
"""Read-only status of the local mcdma-rpcd connect daemon; oMLX never starts or stops it."""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SOCKET = "/tmp/mcdma-rpcd.sock"
SOCKET_ENV = "OMLX_MCDMA_RPCD_SOCKET"
_MAX_REPLY_BYTES = 64 * 1024
# The mailbox protocol oMLX speaks; a daemon that announces another one is not used.
PROTOCOL = 1


def daemon_socket_path() -> str:
    """The connect daemon's control socket, honouring the operator's override."""
    return os.environ.get(SOCKET_ENV) or DEFAULT_SOCKET


@dataclass(frozen=True)
class PeerStatus:
    """One peer link as the connect daemon reports it."""

    name: str
    up: bool
    calls: int = 0
    failures: int = 0
    mib: int = 0
    host: str | None = None
    port: int | None = None
    device: str | None = None
    request_bytes: int | None = None
    reply_bytes: int | None = None
    since: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "up": self.up,
            "calls": self.calls,
            "failures": self.failures,
            "mib": self.mib,
            "host": self.host,
            "port": self.port,
            "device": self.device,
            "request_bytes": self.request_bytes,
            "reply_bytes": self.reply_bytes,
            "since": self.since,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PeerStatus:
        return cls(
            name=str(payload["name"]),
            up=bool(payload["up"]),
            calls=int(payload.get("calls") or 0),
            failures=int(payload.get("failures") or 0),
            mib=int(payload.get("mib") or 0),
            host=_text(payload.get("host")),
            port=_number(payload.get("port"), int),
            device=_text(payload.get("device")),
            request_bytes=_number(payload.get("request_bytes"), int),
            reply_bytes=_number(payload.get("reply_bytes"), int),
            since=_number(payload.get("since"), float),
        )


@dataclass(frozen=True)
class DaemonStatus:
    """The connect daemon's answer, or why there was none."""

    socket_path: str
    reachable: bool
    reason: str
    version: str | None = None
    peers: tuple[PeerStatus, ...] = field(default_factory=tuple)

    def peer(self, name: str) -> PeerStatus | None:
        return next((peer for peer in self.peers if peer.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "socket_path": self.socket_path,
            "reachable": self.reachable,
            "reason": self.reason,
            "version": self.version,
            "peers": [peer.to_dict() for peer in self.peers],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DaemonStatus:
        return cls(
            socket_path=str(payload["socket_path"]),
            reachable=bool(payload["reachable"]),
            reason=str(payload.get("reason", "")),
            version=_text(payload.get("version")),
            peers=tuple(
                PeerStatus.from_dict(peer) for peer in payload.get("peers", [])
            ),
        )


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def _number(value: Any, kind: Callable[[Any], Any]) -> Any:
    return None if value is None else kind(value)


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _peer(tokens: list[str]) -> PeerStatus:
    # PEER name up|down calls N failures N MiB N, then protocol-1 key=value extras.
    counters = dict(zip(tokens[2:8:2], tokens[3:9:2]))
    extras = dict(token.split("=", 1) for token in tokens[8:] if "=" in token)
    megabytes = _int(extras.get("req_mib")), _int(extras.get("rep_mib"))
    return PeerStatus(
        name=tokens[0],
        up=tokens[1] == "up",
        calls=_int(counters.get("calls")) or 0,
        failures=_int(counters.get("failures")) or 0,
        mib=_int(counters.get("MiB")) or 0,
        host=extras.get("host"),
        port=_int(extras.get("port")),
        device=extras.get("device"),
        request_bytes=megabytes[0] << 20 if megabytes[0] else None,
        reply_bytes=megabytes[1] << 20 if megabytes[1] else None,
        since=_float(extras.get("since")),
    )


def parse_status(lines: Iterable[str]) -> tuple[str | None, tuple[PeerStatus, ...]]:
    """Parse STATUS reply lines into the daemon version and its peers."""
    version = None
    peers = []
    for line in lines:
        tokens = line.split()
        if len(tokens) >= 4 and tokens[0] == "VERSION" and tokens[1] == "mcdma-rpcd":
            version = tokens[3]
        elif len(tokens) >= 3 and tokens[0] == "PEER" and tokens[2] in {"up", "down"}:
            peers.append(_peer(tokens[1:]))
    return version, tuple(peers)


def _ask(path: str, command: bytes, timeout_s: float) -> list[str]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as control:
        control.settimeout(timeout_s)
        control.connect(path)
        control.sendall(command)
        received = b""
        while not received.endswith(b"END\n"):
            chunk = control.recv(4096)
            if not chunk:
                break
            received += chunk
            if len(received) > _MAX_REPLY_BYTES:
                raise OSError("mcdma-rpcd status reply is too large")
    return received.decode(errors="replace").splitlines()


def read_status(
    socket_path: str | None = None,
    *,
    timeout_s: float = 2.0,
    ask: Callable[[str, bytes, float], list[str]] = _ask,
) -> DaemonStatus:
    """Ask the connect daemon for its peers; never raises for an absent daemon."""
    path = socket_path or daemon_socket_path()
    try:
        lines = ask(path, b"STATUS\n", timeout_s)
    except FileNotFoundError:
        return DaemonStatus(
            path, False, f"mcdma-rpcd is not running: no socket at {path}"
        )
    except OSError as exc:
        return DaemonStatus(path, False, f"mcdma-rpcd at {path} did not answer: {exc}")
    version, peers = parse_status(lines)
    announced = next(
        (
            line.split()[2]
            for line in lines
            if line.startswith("VERSION mcdma-rpcd ") and len(line.split()) >= 4
        ),
        None,
    )
    if announced is not None and announced != str(PROTOCOL):
        reason = f"mcdma-rpcd at {path} speaks protocol {announced}; oMLX needs protocol {PROTOCOL}"
        return DaemonStatus(path, False, reason, version)
    if not peers:
        return DaemonStatus(
            path, True, "mcdma-rpcd answered but serves no peers", version
        )
    up = sum(peer.up for peer in peers)
    return DaemonStatus(
        path, True, f"{up} of {len(peers)} peer links up", version, peers
    )
