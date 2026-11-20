# SPDX-License-Identifier: Apache-2.0
"""Transport detection for distributed cluster peers.

Wraps ``mlx._distributed_utils.config`` to discover available transports
(TB4, TB5, Ethernet, RDMA) without reimplementing the underlying probing.
"""

from __future__ import annotations

import ipaddress
import json
import re
import shlex
import subprocess
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

from .ssh_policy import apply_cluster_ssh_policy, cluster_ssh_options

# Link speed thresholds for distinguishing TB4 from TB5
_TB4_MAX_SPEED_GTBS = 40  # TB4 is up to 40 Gb/s
_TB5_MIN_SPEED_GTBS = 80  # TB5 starts at 80 Gb/s
_MLX_CONFIG_RUN_LOCK = threading.Lock()


@dataclass(frozen=True)
class TransportInfo:
    """Information about a transport link to a peer."""

    kind: str  # "thunderbolt", "ethernet", "rdma"
    interface: str
    peer_node_id: str
    # Both endpoints, so rank placement can reason about the fabric as a graph.
    # Without a source, "there is a Thunderbolt link to node B" says nothing
    # about which node it starts from.
    source_node_id: str = ""
    link_speed_gbps: int | None = None
    rdma_device: str | None = None
    tb_version: str | None = None  # "TB4" or "TB5"


@dataclass(frozen=True)
class TransportMatrix:
    """Full transport matrix for a cluster."""

    transports: tuple[TransportInfo, ...]
    backend: str  # "ring", "jaccl", "jaccl-ring"


def _import_mlx_config() -> Any:
    """Import the MLX distributed config module, failing with an actionable error."""

    try:
        import mlx._distributed_utils.config as config

        return config
    except ImportError as exc:
        raise RuntimeError(
            "mlx._distributed_utils.config is unavailable. "
            "Ensure MLX is installed with distributed support."
        ) from exc


@contextmanager
def _mlx_config_ssh_policy(config: Any):
    """Make MLX's hard-coded ``ssh`` subprocess use the cluster policy."""

    original_run = getattr(config, "run", None)
    if original_run is None:
        # Small test doubles and future MLX versions may inject transport by a
        # different seam. Their own implementation remains authoritative.
        yield
        return

    with _MLX_CONFIG_RUN_LOCK:

        def policy_run(argv: list[str], *args: Any, **kwargs: Any):
            if argv and str(argv[0]).rsplit("/", 1)[-1] in {"ssh", "scp"}:
                argv = apply_cluster_ssh_policy(argv, connect_timeout=10)
            return original_run(argv, *args, **kwargs)

        config.run = policy_run
        try:
            yield
        finally:
            config.run = original_run


def _run_ssh(ssh_hostname: str, command: str) -> str:
    """Run a command over SSH and return stdout."""

    result = subprocess.run(
        [
            "ssh",
            *cluster_ssh_options(connect_timeout=10),
            ssh_hostname,
            command,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"SSH command failed on {ssh_hostname}: {command}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return result.stdout


_SPEED = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*(gb/s|gbps|mb/s|mbps)", re.IGNORECASE)


def _parse_thunderbolt_speed(speed_str: str) -> int | None:
    """Parse a Thunderbolt link speed string to Gb/s.

    system_profiler reports values like ``"Up to 120 Gb/s"``, so the number
    must be extracted rather than assumed to be the whole string.
    """

    if not speed_str:
        return None
    match = _SPEED.search(speed_str)
    if match is None:
        return None
    value = float(match.group(1))
    if match.group(2).lower().startswith("mb"):
        value /= 1000
    return int(value)


def _detect_tb_version(link_speed_gbps: int | None) -> str | None:
    """Determine TB4 vs TB5 from link speed."""

    if link_speed_gbps is None:
        return None
    if link_speed_gbps >= _TB5_MIN_SPEED_GTBS:
        return "TB5"
    if link_speed_gbps <= _TB4_MAX_SPEED_GTBS:
        return "TB4"
    return None


def _receptacle_speeds(node: Any) -> list[int]:
    """Collect link speeds from a system_profiler bus entry and its children.

    Verified against ``system_profiler SPThunderboltDataType -json`` on macOS:
    the speed lives in ``receptacle_<n>_tag.current_speed_key`` as a string
    like ``"Up to 120 Gb/s"``, not at the top level of the bus entry.
    """

    speeds: list[int] = []
    if not isinstance(node, dict):
        return speeds
    for key, value in node.items():
        if key.endswith("_tag") and isinstance(value, dict):
            parsed = _parse_thunderbolt_speed(str(value.get("current_speed_key", "")))
            if parsed:
                speeds.append(parsed)
    for item in node.get("_items", []):
        speeds.extend(_receptacle_speeds(item))
    return speeds


def _extract_tb_link_speed(ssh_hostname: str) -> int | None:
    """Extract the fastest Thunderbolt link speed reported by a host.

    Callers run inside ``detect_transports``' degradation handler, so failures
    are allowed to propagate rather than being swallowed a second time here.
    """

    output = _run_ssh(ssh_hostname, "system_profiler SPThunderboltDataType -json")
    data = json.loads(output)
    speeds: list[int] = []
    for bus in data.get("SPThunderboltDataType", []):
        speeds.extend(_receptacle_speeds(bus))
    return max(speeds) if speeds else None


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _rdma_devices(ssh_hostname: str) -> list[str]:
    """RDMA devices on one host, queried locally when the host is this machine.

    MLX's own ``check_rdma`` shells out to ``ssh <host> ibv_devices`` for every
    host including ``127.0.0.1``. On a Mac without SSH-to-self configured that
    fails, and RDMA is reported as unavailable on hardware that has it — a
    false negative observed on a machine with rdma_en1/en2/en6 present.
    """

    if ssh_hostname in _LOCAL_HOSTS:
        command = ["ibv_devices"]
    else:
        command = [
            "ssh",
            *cluster_ssh_options(connect_timeout=10),
            ssh_hostname,
            "ibv_devices",
        ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"RDMA device probe failed for {ssh_hostname}: {exc}"
        ) from exc
    if getattr(result, "returncode", 0) != 0:
        detail = str(result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"RDMA device probe failed for {ssh_hostname}"
            + (f": {detail[-400:]}" if detail else "")
        )
    return [
        token
        for token in result.stdout.split()
        if token.startswith("rdma_")
    ]


def _rdma_available(hosts: list[str] | tuple[str, ...], *, ssh_prefix: str = "") -> bool:
    """True when every host reports at least one RDMA device."""

    return bool(hosts) and all(
        _rdma_devices(f"{ssh_prefix}{host}") for host in hosts
    )


def detect_transports(
    hosts: list[str] | tuple[str, ...],
    *,
    ssh_prefix: str = "",
) -> tuple[TransportInfo, ...]:
    """Detect available transports for a set of cluster hosts.

    Uses MLX's ``extract_connectivity`` and ``check_rdma`` to discover
    Thunderbolt topology and RDMA availability. Reports Ethernet only when
    both probes ran and found no fast link. A failed probe returns an empty
    tuple, which callers read as an unknown link.

    Args:
        hosts: List of SSH hostnames for the cluster nodes.
        ssh_prefix: Optional prefix for SSH hostnames (e.g., "user@").

    Returns:
        Tuple of ``TransportInfo`` objects, one per peer link.
    """

    config = _import_mlx_config()

    # Build MLX Host objects
    mlx_hosts = [
        config.Host(
            rank=i,
            ssh_hostname=f"{ssh_prefix}{h}",
            ips=[],
            rdma=[],
        )
        for i, h in enumerate(hosts)
    ]

    transports: list[TransportInfo] = []
    physical_edges: set[tuple[int, int]] = set()
    probe_failed = False

    # Try Thunderbolt connectivity
    try:
        with _mlx_config_ssh_policy(config):
            tb_hosts, uuid_reverse_index = config.extract_connectivity(
                mlx_hosts, verbose=False
            )
        connectivity = config.make_connectivity_matrix(tb_hosts, uuid_reverse_index)

        # Extract transport info from the connectivity matrix
        for i, _host in enumerate(hosts):
            for j, peer_host in enumerate(hosts):
                if i == j:
                    continue
                # Check if there's a TB connection between host i and host j
                if connectivity and i < len(connectivity):
                    row = connectivity[i]
                    if j < len(row) and row[j]:
                        # TB connection exists
                        physical_edges.add((i, j))
                        link_speed = _extract_tb_link_speed(f"{ssh_prefix}{hosts[i]}")
                        tb_version = _detect_tb_version(link_speed)
                        transports.append(
                            TransportInfo(
                                kind="thunderbolt",
                                interface=f"Thunderbolt {j}",
                                peer_node_id=peer_host,
                                source_node_id=hosts[i],
                                link_speed_gbps=link_speed,
                                tb_version=tb_version,
                            )
                        )
    except Exception:
        probe_failed = True

    # Check RDMA
    try:
        rdma_available = _rdma_available(hosts, ssh_prefix=ssh_prefix)
        if rdma_available:
            rdma_edges = set(physical_edges)
            # MLX may be unable to report the Thunderbolt UUID topology even
            # after the link has routable RDMA addresses. Recover real edges
            # from shared RDMA subnets. Never turn "each Mac has an RDMA
            # device" into a fictional full mesh: on three Macs that routes
            # ranks over ports which are not physically connected.
            if not rdma_edges:
                interfaces = [
                    probe_host_interfaces(f"{ssh_prefix}{host}") for host in hosts
                ]
                for i, source in enumerate(interfaces):
                    for j, peer in enumerate(interfaces):
                        if i == j:
                            continue
                        if shared_link_addresses(source, peer).kind == "rdma":
                            rdma_edges.add((i, j))
            for i, j in sorted(rdma_edges):
                peer_host = hosts[j]
                transports.append(
                    TransportInfo(
                        kind="rdma",
                        interface="rdma",
                        peer_node_id=peer_host,
                        source_node_id=hosts[i],
                        rdma_device="rdma_en",
                    )
                )
    except Exception:
        probe_failed = True

    if not transports and not probe_failed:
        for i, _host in enumerate(hosts):
            for j, peer_host in enumerate(hosts):
                if i == j:
                    continue
                transports.append(
                    TransportInfo(
                        kind="ethernet",
                        interface="ethernet",
                        peer_node_id=peer_host,
                        source_node_id=hosts[i],
                    )
                )

    return tuple(transports)


def select_backend(transports: tuple[TransportInfo, ...]) -> str:
    """Select the optimal backend based on detected transports.

    - RDMA-capable Thunderbolt mesh → jaccl
    - Thunderbolt ring → jaccl-ring
    - Ethernet only → ring
    """

    has_rdma = any(t.kind == "rdma" for t in transports)
    has_tb = any(t.kind == "thunderbolt" for t in transports)

    if has_rdma:
        return "jaccl"
    if has_tb:
        return "jaccl-ring"
    return "ring"


def detect_cluster_transports(
    hosts: list[str] | tuple[str, ...],
    *,
    ssh_prefix: str = "",
) -> TransportMatrix:
    """Detect transports and return a full transport matrix.

    Args:
        hosts: List of SSH hostnames for the cluster nodes.
        ssh_prefix: Optional prefix for SSH hostnames.

    Returns:
        ``TransportMatrix`` with transports and the backend they allow.
    """

    transports = detect_transports(hosts, ssh_prefix=ssh_prefix)
    backend = select_backend(transports)

    return TransportMatrix(
        transports=transports,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Link addressing
#
# A hostfile naming an address a host no longer has fails as
# "[ring] Couldn't bind socket (error: 49)" — EADDRNOTAVAIL — well after the
# launch looked healthy. macOS renumbers Thunderbolt interfaces across reboots
# and cable changes, and an address set with ``ifconfig`` goes with the old name
# (en6 becoming en4 took 10.0.1.1 with it), so an address is only true at the
# moment it is read off the host. Nothing below caches; that is the point.
# ---------------------------------------------------------------------------

# Two hosts can carry these at once without being able to reach each other:
# loopback is per-host, and a 169.254 address means DHCP failed, so unrelated
# Macs agree on the subnet and on nothing else.
_SELF_ASSIGNED_NETWORK = ipaddress.ip_network("169.254.0.0/16")
_UNROUTABLE_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    _SELF_ASSIGNED_NETWORK,
)

# Which shared link to prefer when hosts have several. RDMA over Thunderbolt
# beats plain Thunderbolt beats whatever else routes.
_LINK_KIND_RANK = {"rdma": 3, "thunderbolt": 2, "ethernet": 1}

# ifconfig prints the flag word in hex, so bridge100's "flags=8a63" is a header
# and vmenet0's "flags=8963" only looks decimal by accident.
_IFCONFIG_HEADER = re.compile(
    r"^(?P<name>[^\s:]+):\s+flags=[0-9a-fA-F]+<(?P<flags>[^>]*)>"
)
# "1000baseT", "2500Base-T", "10Gbase-T", "100baseTX" in an ifconfig media line.
_MEDIA_SPEED = re.compile(r"(\d+)(G?)base", re.IGNORECASE)
_HARDWARE_PORT = re.compile(r"^Hardware Port:\s*(?P<port>.+)$")
_HARDWARE_DEVICE = re.compile(r"^Device:\s*(?P<device>\S+)$")


@dataclass(frozen=True)
class InterfaceAddress:
    """One IPv4 address an interface carries right now."""

    interface: str
    address: str
    prefix_length: int

    @property
    def network(self) -> ipaddress.IPv4Network:
        return ipaddress.IPv4Network(
            f"{self.address}/{self.prefix_length}", strict=False
        )

    def __str__(self) -> str:
        return f"{self.interface} {self.address}/{self.prefix_length}"


@dataclass(frozen=True)
class HostInterfaces:
    """One host's addressing as it stood at the moment it was read."""

    host: str
    addresses: tuple[InterfaceAddress, ...] = ()
    rdma_interfaces: frozenset[str] = frozenset()
    thunderbolt_interfaces: frozenset[str] = frozenset()
    wifi_interfaces: frozenset[str] = frozenset()
    # Up interfaces holding only a self-assigned 169.254 address, which never
    # appear in ``addresses``.
    self_assigned_interfaces: frozenset[str] = frozenset()
    # IPv6 link-local address per up interface. A Thunderbolt Bridge left on
    # DHCP has one, and it routes when scoped to the interface.
    link_local_addresses: Mapping[str, str] = field(default_factory=dict)
    # Negotiated Ethernet speed per up interface, in Mbit/s.
    media_speeds: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LinkEndpoint:
    """Where one end of a link answers."""

    host: str
    interface: str
    address: str


@dataclass(frozen=True)
class SharedLink:
    """The addresses a pair of hosts can reach each other on, or why they cannot."""

    source: LinkEndpoint | None = None
    peer: LinkEndpoint | None = None
    kind: str = "none"
    reason: str = ""
    link_speed_gbps: int | None = None

    @property
    def ok(self) -> bool:
        return self.source is not None and self.peer is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "source": vars(self.source) if self.source else None,
            "peer": vars(self.peer) if self.peer else None,
            "kind": self.kind,
            "reason": self.reason,
            "link_speed_gbps": self.link_speed_gbps,
        }


def _prefix_length(token: str) -> int | None:
    """Prefix length from an ifconfig netmask, which macOS prints in hex."""

    try:
        value = (
            int(token, 16)
            if token.lower().startswith("0x")
            else int(ipaddress.IPv4Address(token))
        )
        # A non-contiguous mask raises here rather than yielding a bit count
        # that would silently widen the subnet.
        return ipaddress.ip_network(f"0.0.0.0/{ipaddress.IPv4Address(value)}").prefixlen
    except ValueError:
        return None


def _interface_address(interface: str, fields: list[str]) -> InterfaceAddress | None:
    """One ``inet`` line of ifconfig output, or None if it cannot carry a link."""

    try:
        address = ipaddress.IPv4Address(fields[1])
    except ValueError:
        return None
    if any(address in network for network in _UNROUTABLE_NETWORKS):
        return None
    prefix_length = 32
    if "netmask" in fields:
        index = fields.index("netmask") + 1
        parsed = _prefix_length(fields[index]) if index < len(fields) else None
        if parsed is None:
            return None
        prefix_length = parsed
    return InterfaceAddress(interface, str(address), prefix_length)


def _up_inet_lines(
    output: str, keyword: str = "inet"
) -> Iterator[tuple[str, list[str]]]:
    """``(interface, fields)`` for each ``keyword`` line of an up interface.

    Addresses on a down interface are dropped: ifconfig keeps listing them, and
    a rank told to bind one gets the same EADDRNOTAVAIL as a stale address.
    """

    interface = ""
    is_up = False
    for line in output.splitlines():
        if line[:1].strip():
            # Every interface block starts at column 0. A header we cannot read
            # must still end the previous block, or its addresses get attributed
            # to the interface listed above it.
            header = _IFCONFIG_HEADER.match(line)
            interface = header.group("name") if header else ""
            is_up = header is not None and "UP" in header.group("flags").split(",")
            continue
        fields = line.split()
        if interface and is_up and len(fields) >= 2 and fields[0] == keyword:
            yield interface, fields


def parse_interface_addresses(output: str) -> tuple[InterfaceAddress, ...]:
    """Routable IPv4 addresses per interface, from ``ifconfig -a`` output."""

    addresses: list[InterfaceAddress] = []
    for interface, fields in _up_inet_lines(output):
        entry = _interface_address(interface, fields)
        if entry is not None:
            addresses.append(entry)
    return tuple(addresses)


def parse_self_assigned_interfaces(output: str) -> frozenset[str]:
    """Up interfaces holding a self-assigned 169.254 address, from ``ifconfig -a``.

    A Thunderbolt Bridge left on its default DHCP setting gets only this, and
    nothing can route it: MLX's ring connects without a source address, so the
    kernel sends 169.254 traffic out of the primary interface.
    """

    interfaces: set[str] = set()
    for interface, fields in _up_inet_lines(output):
        with suppress(ValueError):
            if ipaddress.IPv4Address(fields[1]) in _SELF_ASSIGNED_NETWORK:
                interfaces.add(interface)
    return frozenset(interfaces)


def parse_link_local_addresses(output: str) -> dict[str, str]:
    """IPv6 link-local address per up interface, from ``ifconfig -a``."""

    addresses: dict[str, str] = {}
    for interface, fields in _up_inet_lines(output, "inet6"):
        address = fields[1].partition("%")[0]
        with suppress(ValueError):
            if ipaddress.IPv6Address(address).is_link_local:
                addresses.setdefault(interface, address)
    return addresses


def parse_media_speeds(output: str) -> dict[str, int]:
    """Negotiated Ethernet speed in Mbit/s per up interface, from ``ifconfig -a``."""

    speeds: dict[str, int] = {}
    for interface, fields in _up_inet_lines(output, "media:"):
        match = _MEDIA_SPEED.search(" ".join(fields))
        if match is not None:
            speeds[interface] = int(match[1]) * (1000 if match[2] else 1)
    return speeds


def _hardware_port_devices(output: str, port_prefix: str) -> frozenset[str]:
    """Devices of the hardware ports named ``port_prefix``*, from ``networksetup``."""

    interfaces: set[str] = set()
    port = ""
    for raw in output.splitlines():
        line = raw.strip()
        match = _HARDWARE_PORT.match(line)
        if match is not None:
            port = match.group("port").strip()
            continue
        match = _HARDWARE_DEVICE.match(line)
        if match is not None:
            if port.startswith(port_prefix):
                interfaces.add(match.group("device"))
            port = ""
    return frozenset(interfaces)


def parse_thunderbolt_interfaces(output: str) -> frozenset[str]:
    """Interface names macOS calls Thunderbolt, from ``-listallhardwareports``.

    A shared subnet says nothing about which cable carries it. This is what
    separates the Thunderbolt link from the office LAN when both are routable.
    """

    return _hardware_port_devices(output, "Thunderbolt")


def parse_wifi_interfaces(output: str) -> frozenset[str]:
    """Interface names macOS calls Wi-Fi, from ``-listallhardwareports``."""

    return _hardware_port_devices(output, "Wi-Fi")


def _read(ssh_hostname: str, command: list[str]) -> str:
    """Run one read-only command on a host, returning "" when it cannot run."""

    if ssh_hostname not in _LOCAL_HOSTS:
        command = [
            "ssh",
            *cluster_ssh_options(connect_timeout=10),
            ssh_hostname,
            *command,
        ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout


def probe_host_interfaces(ssh_hostname: str) -> HostInterfaces:
    """Read one host's live addressing, in full, in one pass.

    ``ifconfig -a`` costs the same round trip as a single interface, and asking
    for everything at once is what lets the caller find a link it did not
    already know the name of.
    """

    ifconfig = _read(ssh_hostname, ["ifconfig", "-a"])
    addresses = parse_interface_addresses(ifconfig)
    ports = _read(ssh_hostname, ["networksetup", "-listallhardwareports"])
    return HostInterfaces(
        host=ssh_hostname,
        addresses=addresses,
        rdma_interfaces=frozenset(
            device.removeprefix("rdma_") for device in _rdma_devices(ssh_hostname)
        ),
        thunderbolt_interfaces=parse_thunderbolt_interfaces(ports),
        wifi_interfaces=parse_wifi_interfaces(ports),
        self_assigned_interfaces=parse_self_assigned_interfaces(ifconfig),
        link_local_addresses=parse_link_local_addresses(ifconfig),
        media_speeds=parse_media_speeds(ifconfig),
    )


def _pair_kind(
    source: HostInterfaces,
    peer: HostInterfaces,
    source_address: InterfaceAddress,
    peer_address: InterfaceAddress,
) -> str:
    """What kind of link a candidate address pair actually rides on.

    Both ends must agree: one Thunderbolt port talking to an Ethernet port is
    two different cables that happen to share a subnet, not a fast link.
    """

    if (
        source_address.interface in source.rdma_interfaces
        and peer_address.interface in peer.rdma_interfaces
    ):
        return "rdma"
    if (
        source_address.interface in source.thunderbolt_interfaces
        and peer_address.interface in peer.thunderbolt_interfaces
    ):
        return "thunderbolt"
    return "ethernet"


_Candidate = tuple[str, InterfaceAddress, InterfaceAddress]
LinkVerifier = Callable[[SharedLink], tuple[bool, str]]


def _preference(candidate: _Candidate) -> tuple[int, int, str, str]:
    """Sort key putting the best address pair first.

    What the link is beats how specific the subnet is, which beats the interface
    names — the last only so the same fabric always yields the same answer.
    """

    kind, source_address, peer_address = candidate
    return (
        -_LINK_KIND_RANK[kind],
        -source_address.prefix_length,
        source_address.interface,
        peer_address.interface,
    )


def _candidate_links(
    source: HostInterfaces,
    peer: HostInterfaces,
    *,
    link_speed_gbps: int | None = None,
) -> tuple[SharedLink, ...]:
    """Every shared address pair, fastest first."""

    candidates: list[_Candidate] = []
    for source_address in source.addresses:
        for peer_address in peer.addresses:
            if source_address.network != peer_address.network:
                continue
            if source_address.address == peer_address.address:
                # Both ends holding one address is a coincidence of local
                # bridges — VM networks all pick 192.168.x.1 — not a link.
                continue
            kind = _pair_kind(source, peer, source_address, peer_address)
            candidates.append((kind, source_address, peer_address))

    links = []
    for kind, source_address, peer_address in sorted(candidates, key=_preference):
        label = (
            kind
            if link_speed_gbps is None
            else f"{kind} at {link_speed_gbps} Gb/s"
        )
        links.append(
            SharedLink(
                source=LinkEndpoint(
                    source.host, source_address.interface, source_address.address
                ),
                peer=LinkEndpoint(
                    peer.host, peer_address.interface, peer_address.address
                ),
                kind=kind,
                reason=(
                    f"{source.host} {source_address} and {peer.host} {peer_address} "
                    f"share {source_address.network} over {label}"
                ),
                link_speed_gbps=link_speed_gbps,
            )
        )
    return tuple(links)


def shared_link_addresses(
    source: HostInterfaces,
    peer: HostInterfaces,
    *,
    link_speed_gbps: int | None = None,
) -> SharedLink:
    """The best address each of two hosts should bind to reach the other.

    Pure over probed data, so every branch is testable without two Macs.
    Candidates are ranked by what the link is — RDMA over Thunderbolt, then
    Thunderbolt, then anything routable — and then by how specific the subnet
    is: a /30 between two Macs is a cable, a /16 is the building.
    """

    if not source.addresses:
        return SharedLink(reason=f"{source.host} has no routable IPv4 address")
    if not peer.addresses:
        return SharedLink(reason=f"{peer.host} has no routable IPv4 address")

    links = _candidate_links(source, peer, link_speed_gbps=link_speed_gbps)
    if not links:
        return SharedLink(
            reason=(
                f"{source.host} and {peer.host} share no subnet. "
                f"{source.host} has {', '.join(str(a) for a in source.addresses)}; "
                f"{peer.host} has {', '.join(str(a) for a in peer.addresses)}."
            )
        )
    return links[0]


def link_warnings(
    source: HostInterfaces, peer: HostInterfaces, link: SharedLink
) -> tuple[str, ...]:
    """What to fix about the path two hosts were given, when it is not Thunderbolt.

    A Thunderbolt cable left on its self-assigned address is skipped, and the
    pair then talks over whatever else they share, often Wi-Fi.
    """

    if link.kind in ("thunderbolt", "rdma"):
        return ()
    on_wifi = link.ok and (
        link.source.interface in source.wifi_interfaces
        or link.peer.interface in peer.wifi_interfaces
    )
    stranded = [
        sorted(host.self_assigned_interfaces & host.thunderbolt_interfaces)
        for host in (source, peer)
    ]
    if all(stranded):
        if on_wifi:
            instead = " The cluster uses Wi-Fi instead."
        elif link.ok:
            instead = f" The cluster uses {link.source.interface} instead."
        else:
            instead = ""
        return (
            f"{source.host} ({', '.join(stranded[0])}) and {peer.host} "
            f"({', '.join(stranded[1])}) are connected by Thunderbolt, but it has "
            "only a self-assigned 169.254 address, which the cluster cannot use."
            f"{instead} Give the Thunderbolt link a manual IPv4 address on both "
            "Macs, for example 10.10.0.1 and 10.10.0.2 with subnet mask "
            "255.255.255.0.",
        )
    if on_wifi:
        return (
            f"{source.host} and {peer.host} reach each other over Wi-Fi. Model "
            "copies and every decode step cross it, so both are slow.",
        )
    return ()


def _pair_link_speed(
    transports: Sequence[TransportInfo],
    source_host: str,
    peer_host: str,
) -> int | None:
    """The fastest speed detection reported for the link between two hosts."""

    speeds = [
        transport.link_speed_gbps
        for transport in transports
        if transport.link_speed_gbps
        and {transport.source_node_id, transport.peer_node_id}
        == {source_host, peer_host}
    ]
    return max(speeds) if speeds else None


def resolve_link_addresses(
    source_host: str,
    peer_host: str,
    *,
    transports: Sequence[TransportInfo] = (),
    probe: Callable[[str], HostInterfaces] = probe_host_interfaces,
    verify: LinkVerifier | None = None,
) -> SharedLink:
    """Ask both hosts where they answer, rather than trusting a written address.

    ``probe`` is injectable so a caller holding fresh readings need not pay for
    a second round of SSH, but nothing here remembers an address between calls —
    remembering is what renumbering invalidates.
    """

    source = probe(source_host)
    peer = probe(peer_host)
    speed = _pair_link_speed(transports, source_host, peer_host)
    link = shared_link_addresses(source, peer, link_speed_gbps=speed)
    if verify is None or not link.ok:
        return link
    rejected = []
    for candidate in _candidate_links(source, peer, link_speed_gbps=speed):
        reachable, reason = verify(candidate)
        if reachable:
            return candidate
        rejected.append(reason)
    return SharedLink(
        kind=link.kind,
        reason="No verified shared path. " + " ".join(dict.fromkeys(rejected)),
        link_speed_gbps=link.link_speed_gbps,
    )


LinkCommandRunner = Callable[
    [str, Sequence[str]], subprocess.CompletedProcess[str]
]


def _run_link_command(
    ssh_hostname: str,
    command: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    """Run one bounded link check locally or through the paired SSH identity."""

    argv = list(command)
    if ssh_hostname not in _LOCAL_HOSTS:
        # ssh joins remote arguments into one string for the peer's shell.
        argv = [
            "ssh",
            *cluster_ssh_options(connect_timeout=5),
            ssh_hostname,
            shlex.join(argv),
        ]
    try:
        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(argv, 255, "", str(exc))


def _route_interface(output: str) -> str:
    """Interface selected by macOS ``route -n get`` output."""

    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() == "interface":
            return value.strip()
    return ""


def verify_link_reachability(
    link: SharedLink,
    *,
    runner: LinkCommandRunner | None = None,
) -> tuple[bool, str]:
    """Prove both endpoints route and answer over the selected interfaces.

    Sharing a subnet is only a candidate. Macs can retain stale addresses on
    old Thunderbolt interfaces, and two unreachable interfaces can therefore
    look like a perfect point-to-point link. The route must name the selected
    interface in both directions and one bounded ICMP probe must succeed from
    each endpoint before the address enters a hostfile.
    """

    source, peer = link.source, link.peer
    if source is None or peer is None:
        return False, link.reason or "the selected link has no endpoints"
    run = runner or _run_link_command
    directions = ((source, peer), (peer, source))
    for local, remote in directions:
        route = run(local.host, ("/sbin/route", "-n", "get", remote.address))
        selected = _route_interface(route.stdout) if route.returncode == 0 else ""
        if selected != local.interface:
            # When ``route -n get`` itself fails, binding a TCP connection to
            # the candidate source address proves both the route and that the
            # peer's SSH service answers on that exact path.
            script = (
                "import socket,sys\n"
                "s=socket.create_connection((sys.argv[2],22),timeout=3,"
                "source_address=(sys.argv[1],0))\n"
                "s.close()"
            )
            if route.returncode != 0:
                bound = run(
                    local.host,
                    # Absolute like the other remote stdlib probes: a bare
                    # python3 resolves through the login shell's PATH.
                    ("/usr/bin/python3", "-c", script, local.address, remote.address),
                )
                if bound.returncode == 0:
                    continue
            detail = (
                f"route uses {selected}"
                if selected
                else "the host reported no usable route"
            )
            return False, (
                f"{local.host} cannot use {local.interface} to reach "
                f"{remote.address}: {detail}."
            )
        ping = run(
            local.host,
            ("/sbin/ping", "-n", "-c", "1", "-W", "1000", remote.address),
        )
        if ping.returncode != 0:
            return False, (
                f"{local.host} routes {remote.address} over {local.interface}, "
                "but the peer did not answer on that address."
            )
    return True, link.reason
