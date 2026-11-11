# SPDX-License-Identifier: Apache-2.0
"""What this machine has: GPUs, memory, disk and network addresses."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

_NO_MEMORY = ("[N/A]", "[Not Supported]")


def _run(*command: str) -> str:
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def gpus(only: list[int] | None = None) -> list[dict[str, Any]]:
    """NVIDIA GPUs as {index, name, total, free, integrated}, sizes in bytes.

    An integrated GPU (DGX Spark) has no memory of its own, so nvidia-smi
    reports none: it holds its layers in the system RAM.
    """
    out = _run(
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    )
    found = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        integrated = parts[2] in _NO_MEMORY
        try:
            index = int(parts[0])
            if integrated:
                total, free = memory()
            else:
                total, free = int(parts[2]) << 20, int(parts[3]) << 20
        except ValueError:
            continue
        if only is None or index in only:
            found.append(
                {
                    "index": index,
                    "name": parts[1],
                    "total": total,
                    "free": free,
                    "integrated": integrated,
                }
            )
    return found


def wsl() -> bool:
    """Whether this runs on WSL, where CUDA has no concurrent managed memory."""
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def chip_label(found: list[dict[str, Any]]) -> str:
    """ "2x NVIDIA RTX A6000" for the device card on the host."""
    names = [g["name"] for g in found]
    if not names:
        return "No NVIDIA GPU"
    if len(set(names)) == 1:
        return names[0] if len(names) == 1 else f"{len(names)}x {names[0]}"
    return " + ".join(names)


def memory() -> tuple[int, int]:
    """(total, available) system RAM in bytes."""
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            values[key] = int(rest.split()[0]) << 10
    except (OSError, ValueError, IndexError):
        return 0, 0
    return values.get("MemTotal", 0), values.get("MemAvailable", 0)


def disk_free(path: Path) -> int:
    while not path.is_dir():
        path = path.parent
    return shutil.disk_usage(path).free


# Interfaces of containers and bridges on this machine, never a link to a Mac.
_VIRTUAL = ("docker", "br-", "veth", "virbr")


def _interfaces() -> list[dict[str, Any]]:
    try:
        return json.loads(
            _run("ip", "-j", "-4", "addr", "show", "scope", "global") or "[]"
        )
    except ValueError:
        return []


def local_ipv4() -> list[str]:
    """Routable IPv4 addresses of this machine, the default route's first."""
    found = [
        info["local"]
        for item in _interfaces()
        for info in item.get("addr_info", [])
        if info.get("local") and not item.get("ifname", "").startswith(_VIRTUAL)
    ]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            # No packet is sent; this only asks the kernel for the route.
            probe.connect(("192.0.2.1", 9))
            primary = probe.getsockname()[0]
    except OSError:
        primary = ""
    if primary and primary in found:
        found.remove(primary)
    return ([primary] if primary and not primary.startswith("127.") else []) + found


def _speed(interface: str) -> int | None:
    try:
        speed = int(Path(f"/sys/class/net/{interface}/speed").read_text())
    except (OSError, ValueError):
        return None
    return speed if speed > 0 else None


def links() -> list[dict[str, Any]]:
    """Wired interfaces with a carrier: IPv4 networks, IPv6 link-local, Mbit/s.

    A direct cable to the Mac gets no DHCP lease; its IPv6 link-local address
    is the one the Mac can reach.
    """
    try:
        items = json.loads(_run("ip", "-j", "addr", "show") or "[]")
    except ValueError:
        return []
    found = []
    for item in items:
        name = item.get("ifname", "")
        if (
            "LOWER_UP" not in item.get("flags", [])
            or name == "lo"
            or name.startswith(_VIRTUAL)
            or Path(f"/sys/class/net/{name}/wireless").exists()
        ):
            continue
        addresses = item.get("addr_info", [])
        found.append(
            {
                "interface": name,
                "ipv4": [
                    f"{a['local']}/{a['prefixlen']}"
                    for a in addresses
                    if a.get("family") == "inet" and a.get("scope") == "global"
                ],
                "link_local": next(
                    (
                        a["local"]
                        for a in addresses
                        if a.get("family") == "inet6" and a.get("scope") == "link"
                    ),
                    None,
                ),
                "mbps": _speed(name),
            }
        )
    return found


def link_mbps(address: str) -> int | None:
    """Negotiated speed of the wired interface that holds *address*."""
    address = address.split("%", 1)[0]
    for link in links():
        held = [cidr.split("/", 1)[0] for cidr in link["ipv4"]]
        if address in held or address == link["link_local"]:
            return link["mbps"]
    return None
