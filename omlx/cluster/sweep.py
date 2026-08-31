# SPDX-License-Identifier: Apache-2.0
"""Active subnet sweep for cluster nodes that do not advertise Bonjour."""

from __future__ import annotations

import ipaddress
import re
import socket
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from omlx.settings import ServerSettings
from .ssh_identity import _SSH_PORT

_MAX_WORKERS = 32
_DEFAULT_PROBE_TIMEOUT = 0.5  # seconds per TCP connect attempt

# RFC 5737 TEST-NET-1 — never routed, never assigned; UDP "connect" against it
# never leaves the host but makes the kernel pick the outbound route, which is
# how the local interface address is discovered without a DNS lookup.
_LOCAL_IP_PROBE_ADDRESS = "192.0.2.1"
_LOCAL_IP_PROBE_PORT = 80
_DEFAULT_PREFIX_LENGTH = 24


def detect_local_subnet() -> str:
    """The local IPv4 network the host's default route uses, as a CIDR."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect((_LOCAL_IP_PROBE_ADDRESS, _LOCAL_IP_PROBE_PORT))
            local_ip = sock.getsockname()[0]
    except OSError:
        return ""

    prefix = _interface_prefix_length(local_ip)
    if prefix is None:
        prefix = _DEFAULT_PREFIX_LENGTH
    return str(ipaddress.ip_network(f"{local_ip}/{prefix}", strict=False))


def _interface_prefix_length(local_ip: str) -> int | None:
    """Read the prefix length of the interface owning ``local_ip``.

    Linux exposes it as ``192.168.1.10/24`` in ``ip -o -4 addr``; macOS and BSD
    expose a hexadecimal netmask on the matching ``ifconfig`` ``inet`` line.
    Returns ``None`` when it cannot be determined (no ``ip``/``ifconfig``, or
    the address is not on an interface) so the caller can fall back.
    """

    for command in (["ip", "-o", "-4", "addr", "show"], ["ifconfig"]):
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            if not re.search(
                rf"\b{re.escape(local_ip)}(?:/|\s)", line
            ):
                continue
            match = re.search(r"/(\d{1,2})\b", line)
            if match:
                return int(match.group(1))
            match = re.search(r"netmask\s+0x([0-9a-fA-F]+)", line)
            if match:
                prefix = bin(int(match.group(1), 16)).count("1")
                return prefix
            match = re.search(r"netmask\s+([0-9.]+)", line)
            if match:
                return ipaddress.IPv4Network(
                    f"0.0.0.0/{match.group(1)}"
                ).prefixlen
    return None


def default_sweep_ports() -> tuple[int, ...]:
    """Ports that identify a cluster node: SSH plus the oMLX admin port."""

    return (_SSH_PORT, ServerSettings.port)


def expand_cidr(cidr: str) -> list[str]:
    """Expand a CIDR string to its list of IPv4 addresses."""

    try:
        network = ipaddress.ip_network(cidr.strip(), strict=False)
    except ValueError as exc:
        raise ValueError(f"invalid CIDR: {cidr!r}") from exc
    if network.version != 4:
        raise ValueError(f"subnet sweep requires an IPv4 CIDR: {cidr!r}")
    return [str(address) for address in network.hosts()]


def is_port_open(address: str, port: int, timeout: float = _DEFAULT_PROBE_TIMEOUT) -> bool:
    """Return True when a TCP connection to the port succeeds."""

    try:
        ipaddress.ip_address(address)
    except ValueError as exc:
        raise ValueError(f"invalid sweep address: {address!r}") from exc
    if not (1 <= int(port) <= 65535):
        raise ValueError("sweep port must be between 1 and 65535")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(max(0.1, float(timeout)))
        try:
            sock.connect((address, int(port)))
        except OSError:
            return False
    return True


def sweep_subnet(
    cidr: str,
    ports: Sequence[int] | None = None,
    timeout: float = _DEFAULT_PROBE_TIMEOUT,
    max_workers: int = _MAX_WORKERS,
) -> list[dict]:
    """Probe every address in a CIDR for open ports.

    Returns entries like ``{"address": "192.168.1.5", "open_ports": [22, 8000]}``
    only for addresses that answer at least one probe.  ``ports`` and
    ``timeout`` are validated by ``is_port_open`` per host.
    """

    resolved_ports = tuple(ports) if ports is not None else default_sweep_ports()
    addresses = expand_cidr(cidr)
    workers = max(1, min(max_workers, max(1, len(addresses))))
    found: list[dict] = []

    def probe(address: str) -> tuple[str, list[int]]:
        open_ports = [
            port
            for port in resolved_ports
            if is_port_open(address, port, timeout=timeout)
        ]
        return address, open_ports

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(probe, address): address for address in addresses}
        for future in as_completed(futures):
            address, open_ports = future.result()
            if open_ports:
                found.append({"address": address, "open_ports": open_ports})
    return sorted(found, key=lambda entry: ipaddress.ip_address(entry["address"]))
