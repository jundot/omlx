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


# ---------------------------------------------------------------------------
# Cluster v2: PeerRecord + DiscoveryService
#
# Layered peer discovery (see ops/notes/cluster_discovery_onboarding_comparison.md):
#   1. mDNS (_omlx._tcp.local.) via python-zeroconf — optional; absence or any
#      mDNS failure disables only mDNS, never the process.
#   2. IPv6 link-local multicast fallback (exo-style): HELLO/WASSUP on
#      ff12::6f6d:6c78 udp/53413 with per-interface joins and nonce self-drop.
#   3. Manual peer add (add_manual) for multicast-filtered networks.
#   4. Tailscale opportunistic candidates when the tailscale CLI exists.
# Every announced address is verified against GET /api/cluster/node_id before
# it is trusted; peers never graduate from "discovered" without that probe.
# ---------------------------------------------------------------------------

import errno  # noqa: E402
import hashlib  # noqa: E402
import http.client  # noqa: E402
import ipaddress  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import urllib.request  # noqa: E402
from collections import deque  # noqa: E402
from contextlib import suppress  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

from .._version import __version__ as _OMLX_VERSION  # noqa: E402

logger = logging.getLogger(__name__)  # noqa: E402
_probe_diagnostics = threading.local()

MDNS_SERVICE_TYPE = "_omlx._tcp.local."
MULTICAST_GROUP = "ff12::6f6d:6c78"
MULTICAST_PORT = 53413
_HELLO_MAGIC = b"OMLX"
_WASSUP_MAGIC = b"OMLXW"
_HELLO_STRUCT = struct.Struct(">4sQQ")  # magic, nonce, cluster_hash
_MAX_DATAGRAM = 2048
_RECENT_NONCES = 64
_MAX_CANDIDATES = 256
_MACOS_TAILSCALE_CLI = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"

# Multicast-join failures we tolerate per-interface (macOS gotcha #3: skip
# interfaces that do not support multicast instead of dying).
_JOIN_IGNORED_ERRNOS = {
    errno.EAFNOSUPPORT,
    errno.EADDRNOTAVAIL,
    errno.ENODEV,
    errno.ENXIO,
    errno.EINVAL,
}

# HELLO-send and WASSUP-reply failure logs are rate-limited per interface /
# peer (a down Thunderbolt bridge otherwise spams one line per second per
# interface forever — 7k lines in nine minutes observed in the field).
_TX_FAIL_LOG_INTERVAL = 60.0
# After this many consecutive HELLO rounds in which every send failed, the
# multicast socket is discarded and rebuilt. macOS gotcha #6: Thunderbolt
# hotplug renumbers interfaces, which wedges a bound socket's per-interface
# multicast state; rebuilding is the only reliable escape.
_TX_FAIL_RESET_ROUNDS = 3


def cluster_hash_u64(cluster_name: str) -> int:
    """blake2s(cluster_name)[:8] as a big-endian u64."""

    digest = hashlib.blake2s(str(cluster_name).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def encode_hello(nonce: int, cluster_hash: int) -> bytes:
    return _HELLO_STRUCT.pack(_HELLO_MAGIC, nonce & 0xFFFFFFFFFFFFFFFF, cluster_hash)


def decode_hello(data: bytes) -> tuple[int, int] | None:
    """Decode a HELLO datagram to (nonce, cluster_hash); ``None`` if invalid."""

    if len(data) != _HELLO_STRUCT.size:
        return None
    magic, nonce, cluster_hash = _HELLO_STRUCT.unpack(data)
    if magic != _HELLO_MAGIC:
        return None
    return nonce, cluster_hash


def encode_wassup(nonce: int, node_id: str, http_port: int) -> bytes:
    payload = json.dumps(
        {"nonce": nonce, "node_id": node_id, "http_port": int(http_port)},
        separators=(",", ":"),
    ).encode("utf-8")
    return _WASSUP_MAGIC + payload


def decode_wassup(data: bytes) -> dict[str, Any] | None:
    """Decode a WASSUP datagram; ``None`` if invalid."""

    if not data.startswith(_WASSUP_MAGIC):
        return None
    try:
        payload = json.loads(data[len(_WASSUP_MAGIC):].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    nonce = payload.get("nonce")
    node_id = payload.get("node_id")
    http_port = payload.get("http_port")
    if not isinstance(nonce, int) or not 0 <= nonce < 1 << 64:
        return None
    if not isinstance(node_id, str) or not node_id or len(node_id) > 255:
        return None
    if not isinstance(http_port, int) or not 1 <= http_port <= 65535:
        return None
    return {"nonce": nonce, "node_id": node_id, "http_port": http_port}


@dataclass
class PeerCaps:
    chip: str = ""
    ram_gb: float = 0.0
    backends: list[str] = field(default_factory=list)
    thunderbolt: bool = False
    jaccl: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "chip": self.chip,
            "ram_gb": self.ram_gb,
            "backends": list(self.backends),
            "thunderbolt": self.thunderbolt,
            "jaccl": self.jaccl,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> PeerCaps:
        if not isinstance(payload, dict):
            return cls()
        backends = payload.get("backends")
        return cls(
            chip=str(payload.get("chip") or "")[:128],
            ram_gb=float(payload.get("ram_gb") or 0.0),
            backends=[str(b) for b in backends][:8]
            if isinstance(backends, list)
            else [],
            thunderbolt=bool(payload.get("thunderbolt")),
            jaccl=bool(payload.get("jaccl")),
        )


def _rdma_fabric_caps(caps: "PeerCaps", runner: Callable[..., Any] = subprocess.run) -> None:
    """Detect the Thunderbolt RDMA fabric and set ``thunderbolt``/``jaccl``.

    macOS-only, best-effort, never raises: ``rdma_ctl status`` reports
    whether RDMA is enabled and ``ibv_devices`` lists the ``rdma_*`` devices,
    which exist only on the Thunderbolt fabric. The wizard's RDMA check row
    and the JACCL-vs-ring backend choice both key off these flags.
    """

    if sys.platform != "darwin":
        return
    rdma_ctl = "/usr/bin/rdma_ctl"
    ibv_devices = "/usr/bin/ibv_devices"
    # A custom runner is the injectable capability surface used by probes and
    # tests on Macs whose base image does not ship RDMA tools. Only the real
    # subprocess path requires those absolute executables to exist locally.
    if runner is subprocess.run and not (
        os.path.exists(rdma_ctl) and os.path.exists(ibv_devices)
    ):
        return
    try:
        status = runner(  # noqa: S603 - fixed system executable
            [rdma_ctl, "status"], capture_output=True, text=True, timeout=5.0, check=False
        )
        enabled = (
            status.returncode == 0
            and status.stdout.strip().splitlines()
            and status.stdout.strip().splitlines()[0].strip().lower() == "enabled"
        )
        devices = runner(  # noqa: S603 - fixed system executable
            [ibv_devices], capture_output=True, text=True, timeout=5.0, check=False
        )
        # Same shape as probe.py's _RDMA_DEVICE_RE: device lines are
        # whitespace-indented under a two-row header.
        rdma_devs = (
            [
                stripped.split()[0]
                for line in devices.stdout.splitlines()
                if (stripped := line.strip()).startswith("rdma_")
            ]
            if devices.returncode == 0
            else []
        )
        caps.thunderbolt = bool(rdma_devs)
        caps.jaccl = bool(enabled and rdma_devs)
    except (OSError, subprocess.SubprocessError):
        pass


def local_caps() -> PeerCaps:
    """Best-effort local capability snapshot; never raises."""

    caps = PeerCaps()
    try:
        if sys.platform == "darwin":
            chip = subprocess.run(  # noqa: S603 - fixed system executable
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if chip.returncode == 0:
                caps.chip = chip.stdout.strip()[:128]
            mem = subprocess.run(  # noqa: S603 - fixed system executable
                ["/usr/sbin/sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if mem.returncode == 0:
                caps.ram_gb = round(int(mem.stdout.strip()) / (1 << 30), 1)
        else:
            caps.chip = platform.machine()
            with suppress(OSError, ValueError):
                caps.ram_gb = round(
                    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                    / (1 << 30),
                    1,
                )
    except (OSError, subprocess.SubprocessError):
        pass
    _rdma_fabric_caps(caps)
    return caps


PeerLink = str  # "tb" | "ethernet" | "wifi" | "tailscale" | "unknown"
PeerState = str  # "discovered" | "suspect" | "dead"


@dataclass
class PeerRecord:
    node_id: str
    friendly_name: str = ""
    version: str = ""
    cluster_name: str = ""
    caps: PeerCaps = field(default_factory=PeerCaps)
    addrs: list[dict[str, str]] = field(default_factory=list)
    http_port: int = 0
    paired: bool = False
    last_seen: float = 0.0
    link: PeerLink = "unknown"
    state: PeerState = "discovered"

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "friendly_name": self.friendly_name,
            "version": self.version,
            "cluster_name": self.cluster_name,
            "caps": self.caps.to_dict(),
            "addrs": [dict(a) for a in self.addrs],
            "http_port": self.http_port,
            "paired": self.paired,
            "last_seen": self.last_seen,
            "link": self.link,
            "state": self.state,
        }


@dataclass
class DiscoveryConfig:
    cluster_name: str = "omlx"
    http_port: int = 8000
    version: str = _OMLX_VERSION
    caps: PeerCaps = field(default_factory=local_caps)
    hello_interval: float = 1.0
    heartbeat_interval: float = 2.0
    probe_interval: float = 10.0
    probe_timeout: float = 3.0
    suspect_after: float = 6.0
    dead_after: float = 30.0
    multicast_window: float = 5.0
    iface_poll_interval: float = 5.0
    tailscale_interval: float = 30.0
    enable_mdns: bool = True
    enable_multicast: bool = True
    enable_tailscale: bool = True


def default_cluster_config_path(base_path: Path | str | None = None) -> Path:
    """On-disk location of the persisted cluster config (cluster.json)."""

    if base_path is not None:
        base = Path(base_path)
    else:
        env_value = os.environ.get("OMLX_BASE_PATH")
        base = Path(env_value).expanduser() if env_value else Path.home() / ".omlx"
    return base / "cluster" / "cluster.json"


def load_cluster_name(base_path: Path | str | None = None) -> str:
    """Persisted settings-level cluster name, defaulting to ``"omlx"``.

    Read from ``<base>/cluster/cluster.json`` (``{"cluster_name": ...}``).
    A missing or malformed file never fails startup — the default keeps the
    pre-config behavior, and nodes only discover each other when the names
    (and therefore the cluster hashes) match.
    """

    try:
        payload = json.loads(
            default_cluster_config_path(base_path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return DiscoveryConfig.cluster_name
    if not isinstance(payload, dict):
        return DiscoveryConfig.cluster_name
    name = payload.get("cluster_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return DiscoveryConfig.cluster_name


def save_cluster_name(
    cluster_name: str, base_path: Path | str | None = None
) -> Path:
    """Persist the cluster name atomically with owner-only permissions."""

    name = str(cluster_name).strip()
    if not name:
        raise ValueError("cluster_name cannot be empty")
    path = default_cluster_config_path(base_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "cluster_name": name}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


def _default_interface_lister() -> list[str]:
    """Names of multicast-capable candidate interfaces (never raises)."""

    names: list[str] = []
    if sys.platform == "darwin":
        try:
            result = subprocess.run(  # noqa: S603 - fixed system executable
                ["/sbin/ifconfig", "-l"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if result.returncode == 0:
                names = result.stdout.split()
        except (OSError, subprocess.SubprocessError):
            names = []
    if not names:
        with suppress(OSError):
            names = [name for _, name in socket.if_nameindex()]
    return [n for n in names if not n.startswith("lo")]


def local_addr_dicts() -> list[dict[str, str]]:
    """Addresses this node considers its own, for the devices ``self`` row.

    Best-effort and never raises. Thunderbolt bridge addresses (typically
    the self-assigned 10.0.0.0/24 pair on a direct link) are included so the
    UI can show the direct-link path next to LAN and Tailscale ones.
    """

    addrs: list[dict[str, str]] = []
    if sys.platform == "darwin":
        try:
            result = subprocess.run(  # noqa: S603 - fixed system executable
                ["/sbin/ifconfig", "-a"],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0:
            current: str | None = None
            for line in result.stdout.splitlines():
                header = re.match(r"^([a-z0-9]+): ", line)
                if header:
                    current = header.group(1)
                    continue
                if current is None or current.startswith(
                    ("lo", "awdl", "llw", "anpi", "ap", "gif", "stf")
                ):
                    continue
                match = re.match(r"^\s+inet6?\s+(\S+)", line)
                if match is None:
                    continue
                ip = match.group(1)
                if_type = "vpn" if current.startswith("utun") else "lan"
                if ip.startswith("100."):
                    if_type = "tailscale"
                addrs.append({"ip": ip, "if_type": if_type})
            return addrs[:16]
    # Portable fallback: primary IPv4 via the routing table (no traffic is
    # actually sent to TEST-NET-1).
    with suppress(OSError):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("192.0.2.1", 80))
            addrs.append({"ip": sock.getsockname()[0], "if_type": "lan"})
        finally:
            sock.close()
    return addrs


def _http_probe_node_id(
    ip: str, port: int, timeout: float
) -> dict[str, Any] | None:
    """GET http://ip:port/api/cluster/node_id; ``None`` on any failure."""

    host = f"[{ip}]" if ":" in ip else ip
    url = f"http://{host}:{port}/api/cluster/node_id"
    _probe_diagnostics.value = {"transport": "direct", "error": ""}
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - cluster LAN probe of an announced address
            payload = json.loads(response.read(65536).decode("utf-8"))
    except ValueError as exc:
        _probe_diagnostics.value = {
            "transport": "direct",
            "error": f"invalid response: {exc}",
        }
        return None
    except OSError as direct_exc:
        # Some macOS installations deny Homebrew Python outbound access to a
        # direct Thunderbolt subnet while Apple's system Python can reach the
        # same address. Rank control and JACCL bootstrap already use this
        # bounded carrier. Reuse it for persisted-pair reboot recovery so a
        # verified 10.0.0.x address does not require another manual seed.
        try:
            is_tailscale = ipaddress.ip_address(ip) in ipaddress.ip_network(
                "100.64.0.0/10"
            )
        except ValueError:
            is_tailscale = False
        if is_tailscale:
            _probe_diagnostics.value = {
                "transport": "direct",
                "error": str(direct_exc),
            }
            return None
        try:
            payload = _system_proxy_probe_node_id(ip, port, timeout)
        except (OSError, RuntimeError, TimeoutError, ValueError) as proxy_exc:
            _probe_diagnostics.value = {
                "transport": "system-proxy",
                "error": f"direct={direct_exc}; proxy={proxy_exc}",
            }
            return None
        if payload is None:
            _probe_diagnostics.value = {
                "transport": "system-proxy",
                "error": f"direct={direct_exc}; proxy returned no identity",
            }
            return None
        _probe_diagnostics.value = {"transport": "system-proxy", "error": ""}
    if not isinstance(payload, dict) or not isinstance(
        payload.get("node_id"), str
    ):
        return None
    return payload


def _system_proxy_probe_node_id(
    ip: str,
    port: int,
    timeout: float,
) -> dict[str, Any] | None:
    """Probe one peer through the bounded macOS system-Python carrier."""

    from .system_socket_proxy import (
        open_system_tcp_proxy,
        should_proxy_control_socket,
    )

    if not should_proxy_control_socket(ip):
        return None
    proxy = open_system_tcp_proxy(ip, port, timeout=timeout)
    try:
        host = f"[{ip}]" if ":" in ip else ip
        request = (
            "GET /api/cluster/node_id HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Accept: application/json\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        proxy.stream.sendall(request)
        response = http.client.HTTPResponse(proxy.stream)
        response.begin()
        if response.status != 200:
            return None
        payload = json.loads(response.read(65536).decode("utf-8"))
        if response.read(1):
            return None
        return payload if isinstance(payload, dict) else None
    finally:
        proxy.close()


def _classify_link(ip: str, if_type: str, rtt: float | None) -> PeerLink:
    """Best-effort link classification.

    Tailscale is certain (100.64.0.0/10); Thunderbolt vs ethernet vs wifi is
    an RTT heuristic over the verified address until interface metadata says
    otherwise.
    """

    if if_type == "tailscale":
        return "tailscale"
    with suppress(ValueError):
        if ipaddress.ip_address(ip) in ipaddress.ip_network("100.64.0.0/10"):
            return "tailscale"
    if rtt is None:
        return "unknown"
    if rtt < 0.001:
        return "tb"
    if rtt < 0.010:
        return "ethernet"
    return "wifi"


def _load_zeroconf() -> Any | None:
    """Import python-zeroconf if installed; ``None`` disables only mDNS."""

    try:
        import zeroconf
    except ImportError:
        return None
    return zeroconf


def _tailscale_executable() -> str | None:
    """Return the CLI from PATH or the normal signed macOS app bundle."""

    discovered = shutil.which("tailscale")
    if discovered:
        return discovered
    candidate = Path(
        os.environ.get("OMLX_TAILSCALE_CLI", _MACOS_TAILSCALE_CLI)
    ).expanduser()
    if sys.platform == "darwin" and candidate.is_file() and os.access(
        candidate, os.X_OK
    ):
        return str(candidate)
    return None


class DiscoveryService:
    """Always-on peer discovery for cluster v2.

    All network-touching seams are injectable (``socket_factory``,
    ``prober``, ``interface_lister``, ``tailscale_status``,
    ``zeroconf_module``) so unit tests run fully offline.
    """

    def __init__(
        self,
        identity: Any,
        registry: Any,
        config: DiscoveryConfig | None = None,
        *,
        socket_factory: Callable[[], Any] | None = None,
        prober: Callable[[str, int, float], dict[str, Any] | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        interface_lister: Callable[[], list[str]] = _default_interface_lister,
        tailscale_status: Callable[[], dict[str, Any] | None] | None = None,
        zeroconf_module: Any = "auto",
    ) -> None:
        self.identity = identity
        self.registry = registry
        self.config = config or DiscoveryConfig()
        self._clock = clock
        self._prober = prober or _http_probe_node_id
        self._socket_factory = socket_factory or self._open_multicast_socket
        self._interface_lister = interface_lister
        self._tailscale_status = tailscale_status or self._read_tailscale_status
        if zeroconf_module == "auto":
            self._zc = _load_zeroconf() if self.config.enable_mdns else None
        else:
            self._zc = zeroconf_module or None

        self._cluster_hash = cluster_hash_u64(self.config.cluster_name)
        self._peers: dict[str, PeerRecord] = {}
        # (ip, port) -> {node_id hint, if_type, last_probe, rtt, verified}
        self._candidates: dict[tuple[str, int], dict[str, Any]] = {}
        self._nonces: deque[int] = deque(maxlen=_RECENT_NONCES)
        self._callbacks: list[Callable[[PeerRecord], None]] = []
        self._hash_mismatch_logged = False
        self._last_hello_at: float | None = None
        self._last_hello_wall: float | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._socket: Any | None = None
        self._joined: dict[str, int] = {}  # ifname -> ifindex
        self._tx_fail_logged_at: dict[int, float] = {}  # ifindex -> monotonic
        self._reply_fail_logged_at: dict[str, float] = {}  # peer_ip -> monotonic
        self._consecutive_tx_fail_rounds = 0
        self._last_tx_ok_wall: float | None = None
        self._last_tx_error: str | None = None
        self._needs_socket_reset = False
        self._consecutive_socket_resets = 0
        self._last_socket_reset_at = 0.0
        self._local_network_blocked_suspected = False
        self._zc_instance: Any | None = None
        self._zc_browser: Any | None = None
        self._zc_info: Any | None = None
        self._rehydrate_paired_candidates()

    # -- public API ----------------------------------------------------------

    @property
    def mdns_available(self) -> bool:
        return self._zc is not None

    @property
    def multicast_ok(self) -> bool:
        """True when a foreign HELLO arrived within the multicast window.

        The UI uses this to warn about macOS Local Network permission
        denial: discovery silently finds nothing when the OS blocks us.
        """

        last = self._last_hello_at
        return last is not None and (
            self._clock() - last
        ) < self.config.multicast_window

    @property
    def last_multicast_rx_at(self) -> float | None:
        """Wall-clock time of the last foreign HELLO, or None if never."""

        return self._last_hello_wall

    @property
    def mdns_active(self) -> bool:
        """True when our mDNS service announcement is actually registered."""

        return self._zc_info is not None

    def transport_summary(self) -> str:
        """Active discovery transports, e.g. ``"mdns+multicast"``."""

        parts: list[str] = []
        if self.mdns_active:
            parts.append("mdns")
        if self.config.enable_multicast:
            parts.append("multicast")
        return "+".join(parts) if parts else "none"

    def peers(self) -> list[PeerRecord]:
        with self._lock:
            return [
                self._peers[key] for key in sorted(self._peers)
            ]

    def mark_paired(self, node_id: str) -> None:
        """Flip the in-memory paired flag when pairing completes out-of-band.

        ``_merge_registry`` refreshes ``peer.paired`` on every observed
        announcement; this covers the window — and dead peers — between the
        pairing approval and the peer's next announcement, so the device
        inventory stops listing a freshly paired node as discovered.
        """

        with self._lock:
            peer = self._peers.get(node_id)
            if peer is not None:
                peer.paired = True

    def health(self) -> dict[str, Any]:
        """Extended self-diagnostics for the discovery health detail route.

        The base health endpoint only reports inbound multicast state; when
        discovery silently dies (a crashed loop thread, a wedged socket,
        interfaces renumbered by Thunderbolt hotplug) these fields are how
        the UI and field debugging tell those apart from "no peers nearby".
        """

        with self._lock:
            threads = {t.name: t.is_alive() for t in self._threads}
            joined = sorted(self._joined)
            candidate_states = [
                {
                    "ip": ip,
                    "port": port,
                    "node_id": value.get("node_id"),
                    "if_type": value.get("if_type"),
                    "verified": bool(value.get("verified", False)),
                    "last_transport": value.get("last_transport"),
                    "last_error": str(value.get("last_error") or "")[:1000],
                }
                for (ip, port), value in sorted(self._candidates.items())
            ]
        return {
            "multicast_loop_alive": threads.get("omlx-discovery-mcast", False),
            "maintenance_loop_alive": threads.get(
                "omlx-discovery-maint", False
            ),
            "socket_open": self._socket is not None,
            "joined_interfaces": joined,
            "last_hello_tx_ok_at": self._last_tx_ok_wall,
            "last_hello_tx_error": self._last_tx_error,
            "consecutive_tx_fail_rounds": self._consecutive_tx_fail_rounds,
            "socket_resets": self._consecutive_socket_resets,
            "local_network_blocked_suspected": (
                self._local_network_blocked_suspected
            ),
            "candidates": len(self._candidates),
            "candidate_states": candidate_states,
            "peers": len(self._peers),
        }

    def on_change(self, callback: Callable[[PeerRecord], None]) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def add_manual(self, ip: str, port: int) -> None:
        """Manually added peer candidate; verified via the same probe path."""

        self._add_candidate(str(ip), int(port), node_id=None, if_type="manual")
        # Probe on the next maintenance tick; tests may call probe_now().

    def _rehydrate_paired_candidates(self) -> None:
        """Seed probes from trusted addresses that survived a server restart.

        Discovery announcements are deliberately best-effort on macOS. A peer
        that was manually added over a direct Thunderbolt address must remain
        reachable when multicast is blocked, so paired registry rows retain the
        verified address and HTTP port. Unpaired discoveries are never read here
        because the registry keeps them memory-only.
        """

        paired = getattr(self.registry, "paired", None)
        if not callable(paired):
            return
        try:
            records = paired()
        except Exception:
            logger.debug("paired discovery seeds could not be read", exc_info=True)
            return
        for record in records:
            if not isinstance(record, dict):
                continue
            node_id = record.get("node_id")
            # Schema-v1 rows written before endpoint persistence have no port.
            # oMLX nodes conventionally share the configured server port (the
            # same documented heuristic as Tailscale discovery); a successful
            # probe writes the exact value back for subsequent restarts.
            port = record.get("http_port") or self.config.http_port
            if (
                not isinstance(node_id, str)
                or not node_id
                or not isinstance(port, int)
                or isinstance(port, bool)
                or not 1 <= port <= 65535
            ):
                continue
            for raw_ip in record.get("last_addrs") or ():
                if not isinstance(raw_ip, str):
                    continue
                try:
                    # Scope identifiers are process/interface-local and cannot
                    # safely survive a reboot. Retain the address itself; a
                    # routable/manual IPv4 remains the deterministic path.
                    parsed = ipaddress.ip_address(raw_ip.split("%", 1)[0])
                except ValueError:
                    continue
                # A persisted IPv6 scope identifier cannot be trusted after
                # interface renumbering, and link-local IPv6 without a scope is
                # not dialable. Multicast can rediscover it with a fresh scope;
                # deterministic reboot recovery uses the routable/manual path.
                if parsed.version == 6 and parsed.is_link_local:
                    continue
                ip = str(parsed)
                self._add_candidate(
                    ip,
                    port,
                    node_id=node_id,
                    if_type="paired",
                )

    def start(self) -> None:
        with self._lock:
            if self._threads:
                return
            self._stop.clear()
            if self.config.enable_multicast:
                self._spawn(self._multicast_loop, "omlx-discovery-mcast")
            self._spawn(self._maintenance_loop, "omlx-discovery-maint")
            if self._zc is not None:
                try:
                    self._start_mdns()
                except Exception as exc:  # mDNS must never kill the process
                    logger.warning("mDNS announce/browse disabled: %s", exc)
                    self._zc_instance = None

    def stop(self) -> None:
        self._stop.set()
        for thread in list(self._threads):
            thread.join(timeout=2.0)
        self._threads = []
        with suppress(Exception):
            if self._socket is not None:
                self._socket.close()
        self._socket = None
        with suppress(Exception):
            if self._zc_instance is not None:
                if self._zc_info is not None:
                    self._zc_instance.unregister_service(self._zc_info)
                if self._zc_browser is not None:
                    self._zc_browser.cancel()
                self._zc_instance.close()
        self._zc_instance = self._zc_browser = self._zc_info = None

    def probe_now(self) -> None:
        """Run one probe sweep synchronously (maintenance loop does this)."""

        self._probe_sweep()

    def probe_candidate_now(self, ip: str, port: int) -> None:
        """Probe one explicit candidate now, bypassing periodic dedupe."""

        self._probe_candidate(str(ip), int(port))

    def tick_liveness(self) -> None:
        """Evaluate suspect/dead transitions synchronously (loop does this)."""

        self._liveness_sweep()

    # -- internals -----------------------------------------------------------

    def _spawn(self, target: Callable[[], None], name: str) -> None:
        def _supervised() -> None:
            while not self._stop.is_set():
                try:
                    target()
                except BaseException:
                    if self._stop.is_set():
                        return
                    # A daemon thread dying silently is how discovery goes
                    # dark without a trace; log loudly and restart instead.
                    logger.critical(
                        "discovery thread %s crashed; restarting in 2s",
                        name,
                        exc_info=True,
                    )
                    self._stop.wait(2.0)
                    continue
                if self._stop.is_set():
                    return
                # Clean return without a stop request is also a bug —
                # discovery is always-on. Restart rather than going dark.
                logger.warning(
                    "discovery thread %s exited unexpectedly; "
                    "restarting in 2s",
                    name,
                )
                self._stop.wait(2.0)

        thread = threading.Thread(target=_supervised, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    def _fire_change(self, peer: PeerRecord) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(peer)
            except Exception:  # a UI callback must never kill discovery
                logger.exception("discovery on_change callback failed")

    def _open_multicast_socket(self) -> Any:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("::", MULTICAST_PORT))
        sock.settimeout(0.25)
        return sock

    def _sync_interfaces(self, sock: Any) -> None:
        """Join the multicast group on every multicast-capable interface.

        Thunderbolt bridges appear/disappear with cable state (macOS gotcha
        #4), so this is re-run periodically. macOS also *renumbers*
        interfaces on Thunderbolt hotplug (gotcha #5): a cached ifindex then
        silently points at the wrong interface and every send fails with
        EHOSTUNREACH, so stale or renumbered entries are dropped here and
        re-joined below.
        """

        try:
            names = self._interface_lister()
        except Exception:
            return
        current = set(names)
        for name, ifindex in list(self._joined.items()):
            if name not in current:
                # Interface vanished; its group membership dies with it.
                self._joined.pop(name, None)
                continue
            try:
                now_index = socket.if_nametoindex(name)
            except OSError:
                self._joined.pop(name, None)
                continue
            if now_index != ifindex:
                logger.info(
                    "interface %s renumbered (%d -> %d); "
                    "re-joining multicast group",
                    name,
                    ifindex,
                    now_index,
                )
                self._joined.pop(name, None)
        for name in names:
            if name in self._joined:
                continue
            try:
                ifindex = socket.if_nametoindex(name)
            except OSError:
                continue
            membership = socket.inet_pton(
                socket.AF_INET6, MULTICAST_GROUP
            ) + struct.pack("@I", ifindex)
            try:
                sock.setsockopt(
                    socket.IPPROTO_IPV6, socket.IPV6_JOIN_GROUP, membership
                )
            except OSError as exc:
                if exc.errno not in _JOIN_IGNORED_ERRNOS:
                    logger.debug("multicast join on %s failed: %s", name, exc)
                continue
            self._joined[name] = ifindex

    def _multicast_loop(self) -> None:
        try:
            sock = self._socket_factory()
        except OSError as exc:
            logger.warning("cluster multicast discovery unavailable: %s", exc)
            return
        self._socket = sock
        last_hello = 0.0
        last_ifaces = 0.0
        last_reset_warn = 0.0
        try:
            while not self._stop.is_set():
                if self._needs_socket_reset:
                    # Exponential backoff: a process-level wedge (e.g. a
                    # denied macOS Local Network permission) survives socket
                    # rebuilds, so rebuilding every second just spams the
                    # log and burns CPU. Back off to at most one rebuild
                    # per 60s; the first success resets the schedule.
                    now_mono = time.monotonic()
                    backoff = min(
                        2.0 ** self._consecutive_socket_resets, 60.0
                    )
                    if now_mono - self._last_socket_reset_at < backoff:
                        self._stop.wait(1.0)
                        continue
                    self._needs_socket_reset = False
                    self._last_socket_reset_at = now_mono
                    self._consecutive_socket_resets += 1
                    with suppress(Exception):
                        sock.close()
                    with self._lock:
                        self._joined.clear()
                    try:
                        sock = self._socket_factory()
                    except OSError as exc:
                        now_wall = time.monotonic()
                        if now_wall - last_reset_warn >= 60.0:
                            last_reset_warn = now_wall
                            logger.warning(
                                "multicast socket rebuild failed: %s; "
                                "retrying",
                                exc,
                            )
                        self._needs_socket_reset = True
                        self._stop.wait(1.0)
                        continue
                    self._socket = sock
                    last_ifaces = 0.0  # force an immediate re-join pass
                    if self._consecutive_socket_resets == 6:
                        self._local_network_blocked_suspected = True
                        logger.critical(
                            "cluster multicast still dead after %d socket "
                            "rebuilds — on macOS this is usually a denied "
                            "Local Network permission (System Settings → "
                            "Privacy & Security → Local Network) or a wedged "
                            "process (a server restart clears it); peers can "
                            "always be added manually via POST "
                            "/api/cluster/devices/manual",
                            self._consecutive_socket_resets,
                        )
                now = self._clock()
                if now - last_ifaces >= self.config.iface_poll_interval:
                    self._sync_interfaces(sock)
                    last_ifaces = now
                if now - last_hello >= self.config.hello_interval:
                    self._send_hello(sock)
                    last_hello = now
                try:
                    data, addr = sock.recvfrom(_MAX_DATAGRAM)
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    continue
                try:
                    self._handle_datagram(data, addr, sock)
                except Exception:
                    # A malformed datagram or a handler bug must never kill
                    # the loop (and with it, all discovery).
                    logger.exception("discovery datagram handler failed")
        finally:
            with suppress(Exception):
                sock.close()
            if self._socket is sock:
                self._socket = None
            with self._lock:
                self._joined.clear()

    def _send_hello(self, sock: Any) -> None:
        nonce = secrets.randbits(64)
        with self._lock:
            self._nonces.append(nonce)
        payload = encode_hello(nonce, self._cluster_hash)
        with self._lock:
            joined = list(self._joined.values()) or [0]
        any_ok = False
        last_error: str | None = None
        now = self._clock()
        for ifindex in joined:
            # Link-scope multicast needs the egress interface in the
            # destination's scope id. (The previous design set
            # IPV6_MULTICAST_IF on the shared socket per round instead; that
            # state goes stale when macOS renumbers interfaces on
            # Thunderbolt hotplug and every send then fails EHOSTUNREACH.)
            target = (
                (MULTICAST_GROUP, MULTICAST_PORT, 0, ifindex)
                if ifindex
                else (MULTICAST_GROUP, MULTICAST_PORT)
            )
            try:
                sock.sendto(payload, target)
                any_ok = True
            except OSError as exc:
                last_error = f"if {ifindex}: {exc}"
                if exc.errno not in _JOIN_IGNORED_ERRNOS and (
                    now - self._tx_fail_logged_at.get(ifindex, 0.0)
                    >= _TX_FAIL_LOG_INTERVAL
                ):
                    self._tx_fail_logged_at[ifindex] = now
                    logger.debug(
                        "HELLO send on if %d failed: %s", ifindex, exc
                    )
        if any_ok:
            self._consecutive_tx_fail_rounds = 0
            self._consecutive_socket_resets = 0
            self._local_network_blocked_suspected = False
            self._last_tx_ok_wall = time.time()
            self._last_tx_error = None
        else:
            self._consecutive_tx_fail_rounds += 1
            self._last_tx_error = last_error
            if (
                self._consecutive_tx_fail_rounds >= _TX_FAIL_RESET_ROUNDS
                and any(joined)
            ):
                if self._consecutive_tx_fail_rounds == _TX_FAIL_RESET_ROUNDS:
                    # Log once per failure streak; the loop's backoff
                    # schedule paces the actual rebuilds from here.
                    logger.warning(
                        "HELLO sends failed on every joined interface for "
                        "%d rounds; scheduling multicast socket rebuild",
                        self._consecutive_tx_fail_rounds,
                    )
                self._needs_socket_reset = True

    def _handle_datagram(self, data: bytes, addr: Any, sock: Any = None) -> None:
        hello = decode_hello(data)
        if hello is not None:
            self._handle_hello(hello[0], hello[1], addr, sock)
            return
        wassup = decode_wassup(data)
        if wassup is not None:
            self._handle_wassup(wassup, addr)

    def _handle_hello(
        self, nonce: int, cluster_hash: int, addr: Any, sock: Any = None
    ) -> None:
        if cluster_hash != self._cluster_hash:
            # A foreign oMLX cluster shares the LAN; ignore silently for
            # discovery but make the situation diagnosable (once).
            if not self._hash_mismatch_logged:
                self._hash_mismatch_logged = True
                logger.info(
                    "Ignoring cluster HELLO with a non-matching cluster hash "
                    "(another oMLX cluster on this LAN?)"
                )
            return
        with self._lock:
            if nonce in self._nonces:
                return  # nonce-echo self-drop: our own HELLO came back
        self._last_hello_at = self._clock()
        self._last_hello_wall = time.time()
        # Inbound multicast works, so any Local Network permission wedge is
        # definitively over.
        self._local_network_blocked_suspected = False
        if isinstance(addr, tuple):
            peer_ip = addr[0]
            peer_port = addr[1]
            scope_id = addr[3] if len(addr) > 3 else 0
        else:
            peer_ip, peer_port, scope_id = str(addr), MULTICAST_PORT, 0
        # Answer unicast so the sender learns our node_id + http_port. A
        # link-local HELLO source requires the ingress interface as scope id
        # in the reply destination; without it the kernel has no route and
        # the handshake silently never completes.
        reply = encode_wassup(
            nonce, self.identity.node_id, self.config.http_port
        )
        target_sock = sock if sock is not None else self._socket
        if target_sock is not None:
            target = (
                (peer_ip, peer_port, 0, scope_id)
                if scope_id
                else (peer_ip, peer_port)
            )
            try:
                target_sock.sendto(reply, target)
            except OSError as exc:
                now = self._clock()
                if (
                    now - self._reply_fail_logged_at.get(peer_ip, 0.0)
                    >= _TX_FAIL_LOG_INTERVAL
                ):
                    self._reply_fail_logged_at[peer_ip] = now
                    logger.debug(
                        "WASSUP reply to %s failed: %s", peer_ip, exc
                    )

    def _handle_wassup(self, payload: dict[str, Any], addr: Any) -> None:
        with self._lock:
            if payload["nonce"] not in self._nonces:
                return  # not an echo of a nonce we issued
        node_id = payload["node_id"]
        if node_id == self.identity.node_id:
            return
        peer_ip = addr[0] if isinstance(addr, tuple) else str(addr)
        if "%" in peer_ip:
            peer_ip = peer_ip.split("%", 1)[0]
        now = self._clock()
        with self._lock:
            peer = self._peers.get(node_id)
            is_new = peer is None
            if peer is None:
                peer = PeerRecord(node_id=node_id)
                self._peers[node_id] = peer
            peer.http_port = payload["http_port"]
            peer.last_seen = now
            if peer.state == "dead":
                peer.state = "discovered"
            if not any(a["ip"] == peer_ip for a in peer.addrs):
                peer.addrs.append({"ip": peer_ip, "if_type": "unknown"})
                peer.addrs = peer.addrs[-8:]
        self._merge_registry(peer)
        self._add_candidate(
            peer_ip, payload["http_port"], node_id=node_id, if_type="unknown"
        )
        if is_new:
            self._fire_change(peer)
        # Higher node_id (string compare) initiates contact: the higher node
        # probes immediately; the lower node's next periodic probe sweep
        # verifies its own candidate entry.
        if self.identity.node_id > node_id:
            self._probe_candidate(peer_ip, payload["http_port"])

    def _add_candidate(
        self, ip: str, port: int, *, node_id: str | None, if_type: str
    ) -> None:
        with self._lock:
            if len(self._candidates) >= _MAX_CANDIDATES:
                return
            key = (ip, port)
            candidate = self._candidates.get(key)
            if candidate is None:
                candidate = {
                    "node_id": node_id,
                    "if_type": if_type,
                    # Due immediately even when oMLX starts within the first
                    # few seconds after boot and the monotonic clock is < the
                    # normal probe interval.
                    "last_probe": float("-inf"),
                    "rtt": None,
                    "verified": False,
                }
                self._candidates[key] = candidate
            else:
                if node_id:
                    candidate["node_id"] = node_id
                if if_type != "unknown":
                    candidate["if_type"] = if_type

    def _probe_sweep(self) -> None:
        now = self._clock()
        with self._lock:
            due = [
                (ip, port)
                for (ip, port), candidate in self._candidates.items()
                if now - candidate["last_probe"]
                >= (
                    self.config.heartbeat_interval
                    if candidate.get("verified")
                    else self.config.probe_interval
                )
            ]
            due.sort(
                key=lambda key: (
                    not bool(self._candidates[key].get("verified")),
                    self._candidates[key].get("node_id") is None,
                    self._candidates[key].get("if_type") != "paired",
                )
            )
        for ip, port in due:
            self._probe_candidate(ip, port)

    def _probe_candidate(self, ip: str, port: int) -> None:
        with self._lock:
            candidate = self._candidates.get((ip, port))
            if candidate is None:
                return
            candidate["last_probe"] = self._clock()
            hint = candidate["node_id"]
            if_type = candidate["if_type"]
        started = self._clock()
        _probe_diagnostics.value = None
        try:
            result = self._prober(ip, port, self.config.probe_timeout)
        except Exception as exc:
            result = None
            diagnostic = {
                "transport": "custom",
                "error": str(exc),
            }
        else:
            diagnostic = getattr(_probe_diagnostics, "value", None) or {
                "transport": "custom",
                "error": "" if result is not None else "probe returned no identity",
            }
        rtt = self._clock() - started
        if result is None:
            with self._lock:
                candidate = self._candidates.get((ip, port))
                if candidate is not None:
                    candidate["last_transport"] = diagnostic.get("transport")
                    candidate["last_error"] = diagnostic.get("error")
            return
        node_id = result.get("node_id")
        if hint is not None and node_id != hint:
            # Announced address does not answer with the announced node_id —
            # drop it (stale DHCP lease, spoofed announcement, etc.).
            with self._lock:
                self._candidates.pop((ip, port), None)
                peer = self._peers.get(hint)
                if peer is not None:
                    peer.addrs = [a for a in peer.addrs if a["ip"] != ip]
            return
        now = self._clock()
        with self._lock:
            candidate = self._candidates.get((ip, port))
            if candidate is not None:
                candidate["verified"] = True
                candidate["rtt"] = rtt
                candidate["node_id"] = node_id
                candidate["last_transport"] = diagnostic.get("transport")
                candidate["last_error"] = ""
            peer = self._peers.get(node_id)
            is_new = peer is None
            if peer is None:
                peer = PeerRecord(node_id=node_id)
                self._peers[node_id] = peer
            peer.version = str(result.get("version") or peer.version)
            peer.cluster_name = str(
                result.get("cluster_name") or peer.cluster_name
            )
            if result.get("friendly_name"):
                peer.friendly_name = str(result["friendly_name"])
            peer.http_port = port
            peer.last_seen = now
            if not any(a["ip"] == ip for a in peer.addrs):
                peer.addrs.append({"ip": ip, "if_type": if_type})
                peer.addrs = peer.addrs[-8:]
            peer.link = _classify_link(ip, if_type, rtt)
        self._merge_registry(peer)
        if is_new:
            self._fire_change(peer)

    def _merge_registry(self, peer: PeerRecord) -> None:
        if self.registry is None:
            return
        try:
            self.registry.merge(
                {
                    "node_id": peer.node_id,
                    "friendly_name": peer.friendly_name,
                    "caps": peer.caps.to_dict(),
                    "addrs": peer.addrs,
                    "http_port": peer.http_port,
                }
            )
            with self._lock:
                peer.paired = bool(self.registry.is_paired(peer.node_id))
        except Exception:
            logger.debug("device registry merge failed", exc_info=True)

    def _liveness_sweep(self) -> None:
        now = self._clock()
        transitions: list[PeerRecord] = []
        with self._lock:
            for peer in self._peers.values():
                silence = now - peer.last_seen
                if silence >= self.config.dead_after:
                    target = "dead"
                elif silence >= self.config.suspect_after:
                    target = "suspect"
                else:
                    target = "discovered"
                if peer.state != target:
                    peer.state = target
                    transitions.append(peer)
        for peer in transitions:
            self._fire_change(peer)

    def _maintenance_loop(self) -> None:
        last_probe = 0.0
        last_tailscale = 0.0
        sweep_interval = min(
            self.config.probe_interval,
            self.config.heartbeat_interval,
        )
        while not self._stop.is_set():
            now = self._clock()
            if now - last_probe >= sweep_interval:
                self._probe_sweep()
                last_probe = now
            self._liveness_sweep()
            if (
                self.config.enable_tailscale
                and now - last_tailscale >= self.config.tailscale_interval
            ):
                self._tailscale_sweep()
                last_tailscale = now
            self._stop.wait(0.5)

    # -- Tailscale (opportunistic, never required) ---------------------------

    @staticmethod
    def _read_tailscale_status() -> dict[str, Any] | None:
        executable = _tailscale_executable()
        if executable is None:
            return None
        try:
            result = subprocess.run(  # noqa: S603 - tailscale CLI lookup
                [executable, "status", "--json"],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _tailscale_sweep(self) -> None:
        try:
            status = self._tailscale_status()
        except Exception:
            return
        if not status:
            return
        peers = status.get("Peer")
        if not isinstance(peers, dict):
            return
        for entry in peers.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("Online") is False:
                continue
            ips = entry.get("TailscaleIPs")
            if not isinstance(ips, list):
                continue
            for ip in ips:
                if isinstance(ip, str) and ip.startswith("100."):
                    # Peer's oMLX port is unknown; our own configured port is
                    # the best heuristic (documented — Tailscale is
                    # opportunistic, never required).
                    self._add_candidate(
                        ip,
                        self.config.http_port,
                        node_id=None,
                        if_type="tailscale",
                    )

    # -- mDNS (python-zeroconf, optional) ------------------------------------

    def _start_mdns(self) -> None:
        zc = self._zc
        addresses = self._local_addresses()
        caps_json = json.dumps(
            self.config.caps.to_dict(), separators=(",", ":")
        )
        properties = {
            "id": self.identity.node_id,
            "name": self.identity.friendly_name,
            "ver": self.config.version,
            "cl": self.config.cluster_name,
            "caps": caps_json,
            "port": str(self.config.http_port),
        }
        instance = f"{self.identity.friendly_name}.{MDNS_SERVICE_TYPE}"[:255]
        info = zc.ServiceInfo(
            MDNS_SERVICE_TYPE,
            instance,
            addresses=addresses,
            port=self.config.http_port,
            properties=properties,
        )
        self._zc_instance = zc.Zeroconf()
        self._zc_info = info
        listener = _MdnsListener(self)
        self._zc_browser = zc.ServiceBrowser(
            self._zc_instance, MDNS_SERVICE_TYPE, listener
        )
        self._zc_instance.register_service(info)

    @staticmethod
    def _local_addresses() -> list[bytes]:
        packed: list[bytes] = []
        with suppress(OSError):
            hostname = socket.gethostname()
            for family, _, _, _, sockaddr in socket.getaddrinfo(
                hostname, None
            ):
                if family == socket.AF_INET:
                    packed.append(socket.inet_pton(family, sockaddr[0]))
                elif family == socket.AF_INET6 and not sockaddr[0].startswith(
                    "fe80::1"  # skip loopback-ish
                ):
                    packed.append(
                        socket.inet_pton(family, sockaddr[0].split("%", 1)[0])
                    )
        return packed[:8]

    def _handle_mdns_service(self, info: Any) -> None:
        """Process one resolved _omlx._tcp service (zeroconf callback path)."""

        try:
            props = {
                (k.decode("utf-8", "replace") if isinstance(k, bytes) else k):
                (v.decode("utf-8", "replace") if isinstance(v, bytes) else v)
                for k, v in (info.properties or {}).items()
            }
            node_id = str(props.get("id") or "")
            if not node_id or node_id == self.identity.node_id:
                return
            cluster = str(props.get("cl") or "")
            if cluster != self.config.cluster_name:
                if not self._hash_mismatch_logged:
                    self._hash_mismatch_logged = True
                    logger.info(
                        "Ignoring mDNS service for a different oMLX cluster "
                        "(%r != %r)",
                        cluster,
                        self.config.cluster_name,
                    )
                return
            try:
                caps = PeerCaps.from_dict(json.loads(props.get("caps") or "{}"))
            except json.JSONDecodeError:
                caps = PeerCaps()
            port = int(getattr(info, "port", 0) or 0)
            addresses: list[str] = []
            parsed = getattr(info, "parsed_addresses", None)
            if callable(parsed):
                addresses = [a for a in parsed() if isinstance(a, str)]
            now = self._clock()
            with self._lock:
                peer = self._peers.get(node_id)
                is_new = peer is None
                if peer is None:
                    peer = PeerRecord(node_id=node_id)
                    self._peers[node_id] = peer
                peer.friendly_name = str(props.get("name") or peer.friendly_name)
                peer.version = str(props.get("ver") or peer.version)
                peer.cluster_name = cluster
                peer.caps = caps
                if port:
                    peer.http_port = port
                peer.last_seen = now
                for ip in addresses:
                    if not any(a["ip"] == ip for a in peer.addrs):
                        peer.addrs.append({"ip": ip, "if_type": "mdns"})
                peer.addrs = peer.addrs[-8:]
            self._merge_registry(peer)
            if port:
                for ip in addresses:
                    self._add_candidate(
                        ip, port, node_id=node_id, if_type="mdns"
                    )
            if is_new:
                self._fire_change(peer)
        except Exception:
            # mDNS callbacks run on zeroconf threads; nothing here may kill
            # the process (macOS gotcha #3).
            logger.exception("failed to process mDNS service")


class _MdnsListener:
    """zeroconf ServiceListener adapter; every callback is exception-safe."""

    def __init__(self, service: DiscoveryService) -> None:
        self._service = service

    def _resolve(self, zc: Any, type_: str, name: str) -> None:
        try:
            info = zc.get_service_info(type_, name, timeout=3000)
        except Exception:
            return
        if info is not None:
            self._service._handle_mdns_service(info)

    # zeroconf >= 0.28 style
    def add_service(self, zc: Any, type_: str, name: str) -> None:
        self._resolve(zc, type_, name)

    def update_service(self, zc: Any, type_: str, name: str) -> None:
        self._resolve(zc, type_, name)

    def remove_service(self, zc: Any, type_: str, name: str) -> None:
        pass

    # older zeroconf style
    def update_service_callback(self, *args: Any, **kwargs: Any) -> None:
        pass


_discovery_service_lock = threading.Lock()
_configured_discovery_service: DiscoveryService | None = None


def configure_discovery_service(
    service: DiscoveryService | None,
) -> DiscoveryService | None:
    """Set (or clear) the process-wide discovery service for the endpoints."""

    global _configured_discovery_service
    with _discovery_service_lock:
        _configured_discovery_service = service
        return _configured_discovery_service


def get_discovery_service() -> DiscoveryService:
    with _discovery_service_lock:
        if _configured_discovery_service is None:
            raise RuntimeError("cluster discovery service is not configured")
        return _configured_discovery_service


def announced_caps() -> dict[str, Any]:
    """Capability dict for pairing payloads; never raises.

    The pairing manager is configured before the discovery service starts,
    so the service is resolved at call time: when it is up this returns
    exactly the caps the service announces (``DiscoveryConfig.caps``); when
    discovery is disabled (or not configured yet) a fresh local snapshot is
    taken, which is what the service would announce anyway. Any failure
    degrades to ``{}`` rather than breaking a pairing.
    """

    try:
        service = get_discovery_service()
    except RuntimeError:
        service = None
    if service is not None:
        try:
            caps = service.config.caps.to_dict()
        except Exception:  # pragma: no cover - defensive
            caps = {}
        if caps:
            return caps
    try:
        return local_caps().to_dict()
    except Exception:  # pragma: no cover - defensive
        return {}
