# SPDX-License-Identifier: Apache-2.0
"""Bonjour publication and discovery for nearby Macs running oMLX."""

from __future__ import annotations

import base64
import binascii
import json
import re
import secrets
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from .token_auth import sign_pairing_payload, verify_pairing_signature

_DNS_SD = "/usr/bin/dns-sd"
_MAX_OUTPUT_BYTES = 64 * 1024
_MAX_PEERS = 16
_REACHED_AT = re.compile(
    r"\bcan be reached at\s+([A-Za-z0-9._-]+)\.?:([0-9]{1,5})\b",
    re.IGNORECASE,
)

# oMLX-specific Bonjour service type for richer peer discovery
_OMLX_SERVICE = "_omlx._tcp."
_OMLX_SERVICE_NAME = "oMLX Distributed"

# Pairing token validity window
_PAIRING_TOKEN_TTL = 300  # 5 minutes
_PUBLISH_RESTART_DELAY = 30.0


class _BonjourProcess(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


BonjourSpawner = Callable[[Sequence[str]], _BonjourProcess]


def _spawn_bonjour(args: Sequence[str]) -> _BonjourProcess:
    return subprocess.Popen(  # noqa: S603 - fixed dns-sd executable and arguments
        tuple(args),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


class BonjourPublisher:
    """Keep this oMLX server visible to peers for its whole process lifetime."""

    def __init__(
        self,
        *,
        port: int,
        version: str,
        hostname: str | None = None,
        executable: str | None = None,
        spawner: BonjourSpawner = _spawn_bonjour,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= int(port) <= 65535:
            raise ValueError("Bonjour service port must be between 1 and 65535")
        self._port = int(port)
        self._version = str(version)[:128]
        self._hostname = (hostname or socket.gethostname()).removesuffix(".local")
        self._executable = executable or shutil.which("dns-sd") or _DNS_SD
        self._spawner = spawner
        self._clock = clock
        self._process: _BonjourProcess | None = None
        self._restart_after = 0.0

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def command(self) -> tuple[str, ...]:
        instance = f"oMLX on {self._hostname}"[:63]
        return (
            self._executable,
            "-R",
            instance,
            _OMLX_SERVICE,
            "local.",
            str(self._port),
            f"hostname={self._hostname}",
            f"version={self._version}",
            "ssh_port=22",
        )

    def start(self) -> bool:
        """Publish once; callers may use ``ensure_running`` for supervision."""

        if self.running:
            return True
        now = self._clock()
        if now < self._restart_after:
            return False
        try:
            self._process = self._spawner(self.command)
        except OSError:
            self._process = None
            self._restart_after = now + _PUBLISH_RESTART_DELAY
            return False
        if self._process.poll() is not None:
            self._process = None
            self._restart_after = now + _PUBLISH_RESTART_DELAY
            return False
        return True

    def ensure_running(self) -> bool:
        """Restart a publisher that exited, with a bounded retry rate."""

        if self.running:
            return True
        if self._process is not None:
            self._process = None
            self._restart_after = self._clock() + _PUBLISH_RESTART_DELAY
        return self.start()

    def stop(self) -> None:
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


@dataclass(frozen=True)
class DiscoveryOutput:
    stdout: str
    error: str | None = None


DiscoveryRunner = Callable[[Sequence[str], float], DiscoveryOutput]


def capture_dns_sd(args: Sequence[str], timeout: float) -> DiscoveryOutput:
    """Capture bounded initial dns-sd output; discovery commands stay open."""

    try:
        completed = subprocess.run(
            tuple(args),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        raw = completed.stdout
        error = (
            completed.stderr.decode(errors="replace").strip()
            if completed.returncode not in (0, -15)
            else None
        )
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or b""
        error = None
    except OSError as exc:
        return DiscoveryOutput("", str(exc))
    if isinstance(raw, str):
        raw = raw.encode()
    if len(raw) > _MAX_OUTPUT_BYTES:
        return DiscoveryOutput("", "Bonjour discovery output was too large")
    return DiscoveryOutput(raw.decode(errors="replace"), error)


def parse_browse_instances(output: str, service_type: str = "_ssh._tcp.") -> tuple[str, ...]:
    """Parse service instance names without trusting their display text."""

    instances: list[str] = []
    marker = service_type
    for line in output.splitlines():
        if marker not in line:
            continue
        instance = line.split(marker, 1)[1].strip()
        if (
            not instance
            or instance.lower().startswith("instance name")
            or len(instance.encode()) > 255
            or any(ord(char) < 32 for char in instance)
        ):
            continue
        if instance not in instances:
            instances.append(instance)
    return tuple(instances[:_MAX_PEERS])


def parse_lookup_target(output: str) -> tuple[str, int] | None:
    match = _REACHED_AT.search(output)
    if match is None:
        return None
    port = int(match.group(2))
    if not 1 <= port <= 65535:
        return None
    return match.group(1).removesuffix("."), port


def _bonjour_host_label(hostname: str) -> str:
    """Return the Mac name shared by internal DNS and Bonjour aliases."""

    return hostname.rstrip(".").lower().split(".", 1)[0]


def discover_ssh_peers(
    *,
    timeout: float = 1.5,
    runner: DiscoveryRunner = capture_dns_sd,
) -> dict[str, Any]:
    """Browse and resolve SSH services; returned peers are untrusted hints."""

    if not 0.1 <= timeout <= 10:
        raise ValueError("Bonjour discovery timeout must be between 0.1 and 10s")
    executable = shutil.which("dns-sd") or _DNS_SD
    local_hostname = _bonjour_host_label(socket.gethostname())
    browse = runner(
        [executable, "-B", "_ssh._tcp", "local."],
        timeout,
    )
    if browse.error:
        return {
            "peers": [],
            "warning": f"Bonjour SSH discovery unavailable: {browse.error}",
            "trusted": False,
        }

    def resolve(instance: str) -> dict[str, Any] | None:
        lookup = runner(
            [executable, "-L", instance, "_ssh._tcp", "local."],
            min(timeout, 1.0),
        )
        target = parse_lookup_target(lookup.stdout)
        if target is None:
            return None
        hostname, port = target
        if (
            port != 22
            or _bonjour_host_label(hostname) == local_hostname
        ):
            return None
        return {
            "name": instance,
            "ssh": hostname,
            "service": "_ssh._tcp.local.",
        }

    instances = parse_browse_instances(browse.stdout)
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(instances)))) as executor:
        resolved = executor.map(resolve, instances)
        peers = [peer for peer in resolved if peer is not None]
    return {
        "peers": peers,
        "warning": None,
        "trusted": False,
    }


def generate_pairing_token(*, shared_secret: str) -> str:
    """Generate a short-lived pairing token for copy/paste pairing."""

    token = secrets.token_urlsafe(32)
    # One clock read: the verifier reconstructs created_at as
    # expires_at - TTL, so two separate time.time() calls would sign a
    # created_at that can never be recomputed and every token would fail.
    now = time.time()
    token_data = {
        "token": token,
        "created_at": now,
        "expires_at": now + _PAIRING_TOKEN_TTL,
    }
    token_json = json.dumps(token_data, sort_keys=True)
    signature = sign_pairing_payload(token_json, shared_secret=shared_secret)
    payload = {
        "token": token,
        "signature": signature,
        "expires_at": token_data["expires_at"],
    }
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()


def verify_pairing_token(encoded_token: str, *, shared_secret: str) -> bool:
    """Verify a pairing token's signature and TTL."""

    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded_token))
        token = payload["token"]
        signature = payload["signature"]
        expires_at = payload["expires_at"]
    except (binascii.Error, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False

    if (
        not isinstance(expires_at, (int, float))
        or isinstance(expires_at, bool)
    ):
        return False

    if time.time() > expires_at:
        return False

    token_data = {
        "token": token,
        "created_at": expires_at - _PAIRING_TOKEN_TTL,
        "expires_at": expires_at,
    }
    return verify_pairing_signature(
        json.dumps(token_data, sort_keys=True),
        signature,
        shared_secret=shared_secret,
    )


def discover_omlx_peers(
    *,
    timeout: float = 2.0,
    runner: DiscoveryRunner = capture_dns_sd,
) -> dict[str, Any]:
    """Browse for oMLX-specific Bonjour services with richer metadata."""

    if not 0.1 <= timeout <= 10:
        raise ValueError("Bonjour discovery timeout must be between 0.1 and 10s")
    executable = shutil.which("dns-sd") or _DNS_SD
    local_hostname = _bonjour_host_label(socket.gethostname())

    browse = runner(
        [executable, "-B", _OMLX_SERVICE, "local."],
        timeout,
    )

    if browse.error:
        return {
            "peers": [],
            "warning": f"oMLX Bonjour discovery unavailable: {browse.error}",
            "trusted": False,
        }

    def resolve_omlx_peer(instance: str) -> dict[str, Any] | None:
        lookup = runner(
            [executable, "-L", instance, _OMLX_SERVICE, "local."],
            min(timeout, 1.0),
        )
        target = parse_lookup_target(lookup.stdout)
        if target is None:
            return None
        hostname, port = target
        if _bonjour_host_label(hostname) == local_hostname:
            return None
        return {
            "name": instance,
            "ssh": hostname,
            "service": _OMLX_SERVICE_NAME,
            "port": port,
            "transport": "detecting",
        }

    instances = parse_browse_instances(browse.stdout, service_type=_OMLX_SERVICE)
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(instances)))) as executor:
        resolved = executor.map(resolve_omlx_peer, instances)
        peers = [peer for peer in resolved if peer is not None]

    return {
        "peers": peers,
        "warning": None,
        "trusted": False,
        # Discovery is deliberately unauthenticated. A pairing token is only
        # created by the explicit endpoint after the user supplies the same
        # out-of-band secret on both Macs.
        "pairing_token": None,
    }


def discover_all_peers(
    *,
    timeout: float = 2.0,
    runner: DiscoveryRunner = capture_dns_sd,
    transport_probe: TransportProbe | None = None,
) -> dict[str, Any]:
    """Discover both SSH and oMLX-specific peers, merging results.

    Transport metadata comes from the cache filled by ``/transports``; pass
    ``transport_probe`` to supply it synchronously instead. Discovery itself
    never opens an SSH connection.
    """

    ssh_result = discover_ssh_peers(timeout=min(timeout, 1.5), runner=runner)
    omlx_result = discover_omlx_peers(timeout=timeout, runner=runner)

    ssh_peers = {peer["ssh"]: peer for peer in ssh_result.get("peers", [])}
    omlx_peers = {peer["ssh"]: peer for peer in omlx_result.get("peers", [])}

    merged_peers = []
    for ssh, peer in ssh_peers.items():
        if ssh in omlx_peers:
            merged_peers.append(omlx_peers[ssh])
        else:
            peer["service"] = "_ssh._tcp.local."
            peer["transport"] = "detecting"
            merged_peers.append(peer)

    for ssh, peer in omlx_peers.items():
        if ssh not in ssh_peers:
            merged_peers.append(peer)

    warnings = []
    if ssh_result.get("warning"):
        warnings.append(ssh_result["warning"])
    if omlx_result.get("warning"):
        warnings.append(omlx_result["warning"])

    # Enrich peers with transport metadata (non-blocking)
    _enrich_peer_transports(merged_peers, probe=transport_probe)

    return {
        "peers": merged_peers,
        "warning": "; ".join(warnings) if warnings else None,
        "trusted": False,
        "pairing_token": omlx_result.get("pairing_token"),
    }


# Transport facts learned by an explicit probe (the /transports endpoint), keyed
# by SSH host. Discovery only ever *reads* this: probing costs an SSH round trip
# per host, which must never sit on the request path of a peer listing.
_TRANSPORT_CACHE: dict[str, dict[str, Any]] = {}

TransportProbe = Callable[[Sequence[str]], dict[str, dict[str, Any]]]


def record_peer_transports(transports: Sequence[Any]) -> None:
    """Cache transport facts so later discoveries can report them for free.

    Called by the ``/transports`` endpoint after a real probe. Accepts anything
    with ``peer_node_id``/``kind``/``link_speed_gbps`` attributes.
    """

    for transport in transports:
        peer = getattr(transport, "peer_node_id", None)
        if not peer:
            continue
        kind = getattr(transport, "kind", "unknown")
        _TRANSPORT_CACHE[peer] = {
            "transport": kind,
            "link_speed_gbps": getattr(transport, "link_speed_gbps", None),
            "rdma_available": kind == "rdma",
        }


def clear_peer_transport_cache() -> None:
    """Drop cached transport facts (topology may have changed)."""

    _TRANSPORT_CACHE.clear()


def _enrich_peer_transports(
    peers: list[dict[str, Any]],
    *,
    probe: TransportProbe | None = None,
) -> None:
    """Attach transport metadata to peer dicts without touching the network.

    Each peer gets ``transport`` ("thunderbolt", "ethernet", "rdma",
    "detecting"), ``link_speed_gbps`` and ``rdma_available``.

    By default this reads only ``_TRANSPORT_CACHE``, so discovery stays fast and
    offline; peers with nothing cached stay ``"detecting"`` and the dashboard can
    call ``/transports`` to fill them in. An earlier version called
    ``detect_transports`` inline, which hung the test suite on SSH to hosts that
    do not exist, and a later one moved that into a daemon thread that kept
    mutating ``peers`` after this function returned. Neither belongs on a request
    path — pass ``probe`` to supply facts synchronously instead.
    """

    if not peers:
        return

    facts: dict[str, dict[str, Any]] = dict(_TRANSPORT_CACHE)
    if probe is not None:
        facts.update(probe([peer["ssh"] for peer in peers]))

    for peer in peers:
        known = facts.get(peer["ssh"])
        peer["transport"] = known["transport"] if known else "detecting"
        peer["link_speed_gbps"] = known["link_speed_gbps"] if known else None
        peer["rdma_available"] = known["rdma_available"] if known else False
