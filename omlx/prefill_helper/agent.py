# SPDX-License-Identifier: Apache-2.0
"""Drives a helper daemon over its control channel.

A helper daemon (``caps.control == "agent"``) has no SSH: it listens on a
control port and proves it holds the cluster key that pairing gave both ends.
This module offers the same functions as :mod:`launcher`, so the manager and
the routes use either through :func:`launcher.for_node`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import socket
import time
import urllib.request
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..cluster.identity import get_node_identity
from ..cluster.pairing import get_pairing_manager
from ..cluster.registry import get_device_registry
from ..cluster.transport import probe_host_interfaces
from . import wire
from .adapter import StageAdapter
from .adapters import adapter_for
from .launcher import LaunchError, required_files

__all__ = ["LaunchError", "required_files"]

# The row name of the first accelerator check, and what it shows.
VIA = "control"
CONTROL_VERSION = 1
_READY_TIMEOUT_S = 900.0
# The helper ends its workers within 10 s, by force if it must.
_STOP_TIMEOUT_S = 15.0
_CHUNK_BYTES = 16 << 20
# A probe of a link the helper may not be on ends this fast.
_LINK_PROBE_TIMEOUT_S = 2.0
# A helper takes its pairing key by polling this Mac every 2 s after approval.
_KEY_PICKUP_S = 10.0


@dataclass(frozen=True)
class AgentTarget:
    node_id: str
    name: str
    # This Mac connects and copies here.
    address: str
    link: str
    control_port: int
    key: bytes
    warnings: tuple[str, ...] = ()
    # Negotiated speed of this Mac's interface toward the helper, in Mbit/s.
    speed_mbps: int | None = None
    # When this Mac paired with it, as wall-clock seconds.
    paired_at: float = 0.0

    @property
    def endpoint(self) -> str:
        if ":" in self.address:
            return f"[{self.address}]:{self.control_port}"
        return f"{self.address}:{self.control_port}"


def _control_key(cluster_key: bytes) -> bytes:
    return hmac.new(cluster_key, b"omlx-prefill-control-v1", hashlib.sha256).digest()


def _data_token(cluster_key: bytes, session: bytes) -> bytes:
    digest = hmac.new(
        cluster_key, b"omlx-prefill-data-v1\0" + session, hashlib.sha256
    ).hexdigest()
    return digest.encode()


def _identity(address: str, port: int) -> dict[str, Any] | None:
    url = f"http://{address}:{port}/api/cluster/node_id"
    try:
        with urllib.request.urlopen(
            url, timeout=3
        ) as response:  # noqa: S310 - a paired LAN address
            reply = json.loads(response.read(65536))
    except (OSError, ValueError):
        return None
    return reply if isinstance(reply, dict) else None


def _local_link(address: str) -> tuple[str, int | None]:
    """("wifi" or "ethernet", negotiated Mbit/s) of the route to *address*."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            # No packet is sent; this only asks for the route's source address.
            probe.connect((address, 9))
            source = probe.getsockname()[0]
        local = probe_host_interfaces("localhost")
    except (OSError, RuntimeError):
        return "ethernet", None
    for item in local.addresses:
        if item.address == source:
            if item.interface in local.wifi_interfaces:
                return "wifi", None
            return "ethernet", local.media_speeds.get(item.interface)
    return "ethernet", None


def resolve_target(node_id: str) -> AgentTarget:
    """The paired helper's control endpoint, checked by its node id."""
    registry = get_device_registry()
    device = registry.get(node_id) if registry.is_paired(node_id) else None
    key = get_pairing_manager().control_key(node_id)
    if device is None or key is None:
        raise LaunchError("This helper is not paired. Pair it in the Cluster tab.")
    port = int(device.get("http_port") or 0)
    for address in device.get("last_addrs", []):
        if ":" in address or not port:
            continue
        found = _identity(address, port)
        if not found or found.get("node_id") != node_id:
            continue
        try:
            control_port = int(found["control_port"])
        except (KeyError, TypeError, ValueError):
            continue
        link, speed = _local_link(address)
        return _fastest_link(
            AgentTarget(
                node_id=node_id,
                name=device["friendly_name"],
                address=address,
                link=link,
                control_port=control_port,
                key=key,
                speed_mbps=speed,
                paired_at=float(device.get("paired_at") or 0.0),
            )
        )
    raise LaunchError(f"{device['friendly_name']} does not answer.")


def _slower(*speeds: int | None) -> int | None:
    known = [speed for speed in speeds if speed]
    return min(known) if len(known) == len(speeds) else None


def _fastest_link(target: AgentTarget) -> AgentTarget:
    """*target* moved to a faster wired link the helper reports, if one answers.

    Discovery finds the helper on the network both machines share. A direct
    cable gets no DHCP lease, so without static addresses the helper is
    reached there at its IPv6 link-local address, scoped to this Mac's port.
    """
    try:
        links = remote_info(target)["links"]
    except LaunchError:
        return target
    cables = [link for link in links if link.get("link_local") and not link.get("ipv4")]
    try:
        ports = probe_host_interfaces("localhost").media_speeds if cables else {}
    except (OSError, RuntimeError):
        ports = {}
    current = target.speed_mbps
    candidates = []
    for link in links:
        helper_speed = link.get("mbps")
        addresses = [cidr.split("/", 1)[0] for cidr in link.get("ipv4") or []]
        if target.address in addresses:
            current = _slower(target.speed_mbps, helper_speed)
        for address in addresses:
            kind, speed = _local_link(address)
            if kind != "wifi" and address != target.address:
                candidates.append((_slower(speed, helper_speed), address))
        if link in cables:
            for interface, speed in ports.items():
                address = f"{link['link_local']}%{interface}"
                candidates.append((_slower(speed, helper_speed), address))
    for speed, address in sorted(candidates, key=lambda c: -(c[0] or 0)):
        if not speed or speed <= (current or 0):
            break
        moved = replace(target, address=address, link="ethernet", speed_mbps=speed)
        try:
            _connect(moved, timeout=_LINK_PROBE_TIMEOUT_S).close()
        except LaunchError:
            continue
        return moved
    return target


class _RefusedError(Exception):
    """The helper did not take this Mac's key."""


def _connect(target: AgentTarget, timeout: float = 10.0) -> socket.socket:
    # Right after the pairing the helper may not have taken its key yet.
    retry_until = target.paired_at + _KEY_PICKUP_S
    while True:
        try:
            return _connect_once(target, timeout)
        except _RefusedError as error:
            if time.time() >= retry_until:
                raise LaunchError(
                    f"{target.name} did not accept this Mac: {error}. Pair it "
                    "again in the Cluster tab."
                ) from error
        time.sleep(0.5)


def _connect_once(target: AgentTarget, timeout: float) -> socket.socket:
    try:
        sock = socket.create_connection(
            (target.address, target.control_port), timeout=timeout
        )
    except OSError as error:
        raise LaunchError(f"{target.name} does not answer: {error}") from error
    try:
        wire.tune(sock)
        wire.send(
            sock,
            {
                "op": "control",
                "v": CONTROL_VERSION,
                "node_id": get_node_identity().node_id,
            },
        )
        wire.authenticate(sock, _control_key(target.key), server=False)
        hello, _ = wire.recv(sock)
        if hello.get("op") != "hello" or hello.get("node_id") != target.node_id:
            raise wire.ProtocolError("unexpected helper greeting")
        if hello.get("v") != CONTROL_VERSION:
            raise LaunchError(
                f"{target.name} speaks another control version; update the "
                "older of oMLX and the helper."
            )
    except (OSError, wire.ProtocolError) as error:
        sock.close()
        raise _RefusedError(str(error)) from error
    except BaseException:
        sock.close()
        raise
    sock.settimeout(None)
    return sock


def _call(
    sock: socket.socket,
    meta: dict[str, Any],
    arrays: dict[str, tuple[np.ndarray, str]] | None = None,
) -> dict[str, Any]:
    try:
        wire.send(sock, meta, arrays)
        reply, _ = wire.recv(sock)
    except (OSError, wire.ProtocolError) as error:
        raise LaunchError(f"The helper connection was lost: {error}") from error
    if reply.get("op") == "error":
        raise LaunchError(str(reply.get("error") or "the helper refused"))
    return reply


@contextmanager
def _session(target: AgentTarget) -> Iterator[socket.socket]:
    sock = _connect(target)
    try:
        yield sock
    finally:
        sock.close()


def remote_info(target: AgentTarget, *, host: dict | None = None) -> dict:
    """The helper's version, contracts, GPU memory per device and free disk."""
    with _session(target) as sock:
        info = _call(sock, {"op": "info", "host": host})
    session = info.get("session") or {}
    held = list(session.get("gpu_bytes") or [])
    gpus = info.get("gpus") or []
    # A session running there makes way for this one, so its memory counts.
    # Its size is the plan's estimate, so a GPU never gets more than it has.
    budgets = [
        min(
            int(gpu["total"]), int(gpu["free"]) + (int(held[i]) if i < len(held) else 0)
        )
        for i, gpu in enumerate(gpus)
    ]
    names = [str(gpu.get("name") or "GPU") for gpu in gpus]
    device = names[0] if len(set(names)) == 1 and names else " + ".join(names)
    if len(names) > 1 and len(set(names)) == 1:
        device = f"{len(names)}x {device}"
    speed = info.get("link_mbps")
    return {
        "version": info.get("version"),
        "data_protocol": info.get("data_protocol"),
        "contracts": info.get("contracts") or {},
        "budgets": budgets,
        "budget": sum(budgets),
        "memory": sum(int(gpu["total"]) for gpu in gpus),
        "device": device,
        "disk_free": info.get("disk_free"),
        "link_mbps": int(speed) if speed else None,
        "links": list(info.get("links") or []),
        "lending": (
            {"host": session.get("host"), "model": session.get("model")}
            if session
            else None
        ),
    }


def incompatible(info: dict, adapter: StageAdapter | None, name: str) -> str | None:
    """Why this helper cannot serve the family, or None when it can."""
    if info.get("data_protocol") != wire.PROTOCOL:
        return (
            f"{name} speaks another prefill protocol; update the older of "
            "oMLX and the helper."
        )
    contracts = info.get("contracts") or {}
    if adapter is None:
        return None if contracts else f"{name} supports no model family."
    theirs = next((contracts[t] for t in adapter.model_types if t in contracts), None)
    if theirs is None:
        return f"{name} does not support this model family yet."
    if theirs != adapter.contract:
        older = "the helper" if theirs < adapter.contract else "oMLX on this Mac"
        return f"{name} and this Mac disagree on this model family; update {older}."
    return None


def locate_files(
    target: AgentTarget, model_name: str, files: dict[str, int]
) -> dict[str, Any]:
    """Where the helper has the model, or what its copy is missing."""
    with _session(target) as sock:
        return _call(sock, {"op": "probe", "name": model_name, "files": files})


def list_copies(target: AgentTarget) -> dict:
    where = locate_files(target, "", {})
    return {"copies": where["copies"], "free": where["free"]}


def delete_copies(target: AgentTarget, names: list[str]) -> int:
    with _session(target) as sock:
        return int(_call(sock, {"op": "delete", "names": names})["free"])


def copy_files(
    target: AgentTarget,
    model_path: str | Path,
    destination: str,
    names: list[str],
    progress: Callable[[str], None] | None = None,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Push *names* over the control channel, resuming a stopped copy.

    After ``cancelled`` turns true the file in flight stops at its next
    chunk and stays on the helper as a partial file.
    """
    source = Path(model_path).expanduser()
    folder = Path(destination).name
    with _session(target) as sock:
        sizes = {name: (source / name).stat().st_size for name in names}
        partial = _call(sock, {"op": "probe", "name": folder, "files": sizes}).get(
            "partial", {}
        )
        for name in names:
            size = sizes[name]
            # A whole partial file still gets an empty closing chunk.
            offset = min(int(partial.get(name, 0)), size)
            with open(source / name, "rb") as f:
                f.seek(offset)
                while True:
                    if cancelled is not None and cancelled():
                        return
                    data = f.read(_CHUNK_BYTES)
                    last = offset + len(data) >= size
                    _call(
                        sock,
                        {
                            "op": "put",
                            "name": folder,
                            "file": name,
                            "size": size,
                            "offset": offset,
                            "crc": zlib.crc32(data),
                            "last": last,
                        },
                        {"data": (np.frombuffer(data, dtype=np.uint8), "uint8")},
                    )
                    offset += len(data)
                    if last:
                        break
            if progress is not None:
                progress(name)


@dataclass
class AgentProcess:
    """A started helper session: the control connection that owns it."""

    sock: socket.socket
    port: int
    hello: dict[str, Any]
    token: bytes

    def stop(self) -> None:
        # The helper answers once its workers are gone, so a start right
        # after this finds their GPU memory free. Without an answer, closing
        # the connection still ends the session there.
        with suppress(Exception):
            self.sock.settimeout(_STOP_TIMEOUT_S)
            wire.send(self.sock, {"op": "stop"})
            wire.recv(self.sock)
        self.sock.close()


def launch(
    target: AgentTarget,
    model_dir: str,
    model_type: str,
    layers: int,
    *,
    host_name: str,
    warm_tokens: int,
    a8_min_tokens: int | None = None,
) -> AgentProcess:
    """Start a session and wait until the helper loaded and warmed its layers.

    ``a8_min_tokens`` has no meaning on a helper daemon and is not sent.
    """
    adapter = adapter_for(model_type)
    session = secrets.token_bytes(16)
    sock = _connect(target)
    try:
        sock.settimeout(_READY_TIMEOUT_S)
        ready = _call(
            sock,
            {
                "op": "start",
                "model": os.path.basename(model_dir),
                "model_type": model_type,
                "layers": layers,
                "warm_tokens": warm_tokens,
                "host_name": host_name,
                "session": session.hex(),
                "contract": adapter.contract if adapter is not None else None,
            },
        )
        sock.settimeout(None)
        return AgentProcess(
            sock,
            int(ready["port"]),
            dict(ready.get("hello") or {}),
            _data_token(target.key, session),
        )
    except BaseException:
        sock.close()
        raise
