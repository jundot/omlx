# SPDX-License-Identifier: Apache-2.0
"""Read-only local capability and Thunderbolt/RDMA probes."""

from __future__ import annotations

import ipaddress
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from omlx._version import __version__
from omlx.utils import hardware

from .models import (
    ClusterStatus,
    RDMACapability,
    RouteCapability,
    RuntimeCapability,
    ThunderboltPort,
    TransportState,
)

_RDMA_CTL = "/usr/bin/rdma_ctl"
_IBV_DEVICES = "/usr/bin/ibv_devices"
_SYSTEM_PROFILER = "/usr/sbin/system_profiler"
_ROUTE = "/sbin/route"
_IPCONFIG = "/usr/sbin/ipconfig"
_IP = "/usr/sbin/ip"
_RDMA = "/usr/sbin/rdma"
_RDMA_DEVICE_RE = re.compile(
    r"^\s*((?:rdma_|mlx5_|roce)[A-Za-z0-9_.:-]*)(?:\s+|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None


CommandRunner = Callable[..., CommandResult]


@dataclass(frozen=True)
class AcceleratorHardware:
    kind: str
    vendor: str
    memory_kind: str
    name: str
    physical_memory_bytes: int
    recommended_working_set_bytes: int
    distributed_backends: tuple[str, ...]


def detect_accelerator_hardware() -> AcceleratorHardware:
    """Read MLX's active device without assuming macOS or Metal."""

    try:
        import mlx.core as mx

        def available(namespace: str) -> bool:
            try:
                return bool(getattr(mx, namespace).is_available())
            except Exception:
                return False

        kind = "cuda" if available("cuda") else "metal" if available("metal") else "cpu"
        vendor = (
            "nvidia" if kind == "cuda" else "apple" if kind == "metal" else "unknown"
        )
        try:
            raw_info = mx.device_info()
            info = raw_info if isinstance(raw_info, dict) else {}
        except Exception:
            info = {}
        name = str(
            info.get("device_name")
            or info.get("name")
            or (
                "NVIDIA CUDA"
                if kind == "cuda"
                else "Apple Silicon"
                if kind == "metal"
                else "CPU"
            )
        )
        physical = int(info.get("memory_size") or 0)
        recommended = int(info.get("max_recommended_working_set_size") or physical)
        backends = []
        for backend in ("ring", "jaccl", "nccl"):
            try:
                if mx.distributed.is_available(backend):
                    backends.append(backend)
            except Exception:
                continue
        return AcceleratorHardware(
            kind=kind,
            vendor=vendor,
            memory_kind=(
                "unified"
                if kind == "metal"
                or (
                    kind == "cuda"
                    and platform.machine().lower() in {"arm64", "aarch64"}
                )
                else "vram"
                if kind == "cuda"
                else "system"
            ),
            name=name,
            physical_memory_bytes=physical,
            recommended_working_set_bytes=recommended,
            distributed_backends=tuple(backends),
        )
    except Exception:
        return AcceleratorHardware(
            kind="cpu",
            vendor="unknown",
            memory_kind="system",
            name="CPU",
            physical_memory_bytes=0,
            recommended_working_set_bytes=0,
            distributed_backends=(),
        )


def _system_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


def run_command(args: Sequence[str], *, timeout: float = 10.0) -> CommandResult:
    """Run one fixed argv command without a shell."""

    argv = tuple(str(arg) for arg in args)
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(args=argv, returncode=None, error=str(exc))
    return CommandResult(
        args=argv,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _tool(name: str, fallback: str) -> str:
    return shutil.which(name) or fallback


def parse_rdma_control(result: CommandResult) -> str:
    if not result.ok:
        return "unavailable"
    value = result.stdout.strip().splitlines()
    if not value:
        return "unknown"
    normalized = value[0].strip().lower()
    if normalized in {"enabled", "disabled"}:
        return normalized
    return "unknown"


def parse_ibv_devices(result: CommandResult) -> tuple[str, ...]:
    if not result.ok:
        return ()
    devices: list[str] = []
    for line in result.stdout.splitlines():
        match = _RDMA_DEVICE_RE.match(line)
        if match and match.group(1) not in devices:
            devices.append(match.group(1))
    return tuple(devices)


def parse_linux_rdma_links(result: CommandResult) -> dict[str, str]:
    """Map Linux ibverbs device names to their network interfaces."""

    if not result.ok:
        return {}
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, list):
        return {}
    links: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_device = str(item.get("ifname") or item.get("dev") or "")
        device = raw_device.split("/", 1)[0].strip()
        interface = str(item.get("netdev") or item.get("netdev_name") or "").strip()
        if device and interface:
            links.setdefault(device, interface)
    return links


def parse_linux_interface_address(result: CommandResult) -> str | None:
    """Return the first globally routable address reported by ``ip -j``."""

    if not result.ok:
        return None
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list):
        return None
    candidates: list[str] = []
    for interface in payload:
        if not isinstance(interface, dict):
            continue
        for address in interface.get("addr_info", []):
            if not isinstance(address, dict):
                continue
            value = str(address.get("local") or "").strip()
            if not value or address.get("scope") not in {None, "global", "link"}:
                continue
            try:
                parsed = ipaddress.ip_address(value)
            except ValueError:
                continue
            if parsed.is_loopback:
                continue
            candidates.append(str(parsed))
    return candidates[0] if candidates else None


def parse_linux_route(
    result: CommandResult,
    *,
    destination: str,
    rdma_interfaces: Sequence[str],
) -> RouteCapability | None:
    """Parse the JSON form of Linux ``ip route get``."""

    if not result.ok:
        return None
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return None
    row = payload[0]
    interface = _optional_string(row.get("dev"))
    gateway = _optional_string(row.get("gateway") or row.get("via"))
    return RouteCapability(
        destination=destination,
        interface=interface,
        gateway=gateway,
        uses_rdma_interface=bool(interface and interface in set(rdma_interfaces)),
    )


def _nested_device_names(value: Any) -> tuple[str, ...]:
    names: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key == "device_name_key" and isinstance(child, str):
                    if child not in names:
                        names.append(child)
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return tuple(names)


def parse_thunderbolt_ports(result: CommandResult) -> tuple[ThunderboltPort, ...]:
    if not result.ok:
        return ()
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError):
        return ()

    buses = payload.get("SPThunderboltDataType", [])
    if not isinstance(buses, list):
        return ()

    ports: list[ThunderboltPort] = []
    for bus in buses:
        if not isinstance(bus, dict):
            continue
        bus_name = str(bus.get("_name") or "unknown")
        for key, value in bus.items():
            if not key.startswith("receptacle_") or not isinstance(value, dict):
                continue
            status = str(value.get("receptacle_status_key") or "unknown")
            peer_names = _nested_device_names(value)
            peer_connected = (
                status not in {"", "unknown"}
                and "no_devices_connected" not in status.lower()
            )
            # Some macOS builds expose the nested peer without a useful status.
            peer_connected = peer_connected or bool(peer_names)
            ports.append(
                ThunderboltPort(
                    bus_name=bus_name,
                    receptacle_id=_optional_string(value.get("receptacle_id_key")),
                    speed=_optional_string(value.get("current_speed_key")),
                    status=status,
                    peer_connected=peer_connected,
                    peer_names=peer_names,
                )
            )
    return tuple(ports)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def parse_route(
    result: CommandResult,
    *,
    destination: str,
    rdma_devices: Sequence[str],
) -> RouteCapability | None:
    if not result.ok:
        return None

    fields: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip().lower()] = value.strip()

    interface = fields.get("interface")
    rdma_interfaces = {
        device.removeprefix("rdma_")
        for device in rdma_devices
        if device.startswith("rdma_")
    }
    return RouteCapability(
        destination=destination,
        interface=interface,
        gateway=fields.get("gateway"),
        uses_rdma_interface=bool(interface and interface in rdma_interfaces),
    )


def _warning_for(result: CommandResult, label: str) -> str | None:
    if result.ok:
        return None
    detail = result.error or result.stderr.strip() or f"exit {result.returncode}"
    return f"{label} probe unavailable: {detail}"


def _advertised_python_executable() -> str:
    """The interpreter another Mac should run over SSH to reach this node.

    ``sys.executable`` loses the environment a launcher (the packaged app or a
    run-from-source script) assembled, so running it bare over SSH cannot import
    oMLX. The worker shim published at server start re-creates that environment.
    Advertise the shim only when it wraps THIS interpreter — a shim rewritten by
    a different install must not be advertised.
    """

    from pathlib import Path

    shim = Path.home() / ".omlx" / "bin" / "omlx-cluster-python"
    try:
        if (
            shim.is_file()
            and os.access(shim, os.X_OK)
            and sys.executable in shim.read_text(encoding="utf-8")
        ):
            return str(shim)
    except OSError:
        pass
    return sys.executable


def collect_cluster_status(
    *,
    route_to: str | None = None,
    runner: CommandRunner = run_command,
    now: Callable[[], datetime] | None = None,
) -> ClusterStatus:
    """Collect a read-only snapshot of this node's cluster capability."""

    if route_to is not None:
        try:
            route_to = str(ipaddress.ip_address(route_to))
        except ValueError as exc:
            raise ValueError("route_to must be an IPv4 or IPv6 address") from exc

    system = platform.system().strip().lower()
    linux = system == "linux"
    rdma_result = (
        runner([_tool("rdma", _RDMA), "-j", "link", "show"], timeout=5.0)
        if linux
        else runner([_tool("rdma_ctl", _RDMA_CTL), "status"], timeout=5.0)
    )
    devices_result = runner(
        [_tool("ibv_devices", _IBV_DEVICES)],
        timeout=5.0,
    )
    thunderbolt_result = (
        CommandResult(args=(), returncode=0, stdout="{}")
        if linux
        else runner(
            [_tool("system_profiler", _SYSTEM_PROFILER), "SPThunderboltDataType", "-json"],
            timeout=15.0,
        )
    )

    rdma_devices = parse_ibv_devices(devices_result)
    linux_rdma_links = parse_linux_rdma_links(rdma_result) if linux else {}
    rdma_status = (
        "enabled" if linux and rdma_devices else
        "unavailable" if linux else
        parse_rdma_control(rdma_result)
    )
    rdma_addresses: list[tuple[str, str]] = []
    for device in rdma_devices:
        interface = (
            linux_rdma_links.get(device, "")
            if linux
            else device.removeprefix("rdma_")
        )
        if not interface:
            continue
        address_result = (
            runner(
                [_tool("ip", _IP), "-j", "address", "show", "dev", interface],
                timeout=5.0,
            )
            if linux
            else runner(
                [_tool("ipconfig", _IPCONFIG), "getifaddr", interface],
                timeout=5.0,
            )
        )
        address = (
            parse_linux_interface_address(address_result)
            if linux
            else address_result.stdout.strip() if address_result.ok else None
        )
        if address:
            try:
                address = str(ipaddress.ip_address(address))
            except ValueError:
                continue
            rdma_addresses.append((device, address))
    ports = parse_thunderbolt_ports(thunderbolt_result)
    peer_connected = any(port.peer_connected for port in ports)

    route_result: CommandResult | None = None
    route: RouteCapability | None = None
    if route_to is not None:
        if linux:
            route_result = runner(
                [_tool("ip", _IP), "-j", "route", "get", route_to],
                timeout=5.0,
            )
            route = parse_linux_route(
                route_result,
                destination=route_to,
                rdma_interfaces=tuple(linux_rdma_links.values()),
            )
        else:
            route_result = runner(
                [_tool("route", _ROUTE), "-n", "get", route_to],
                timeout=5.0,
            )
            route = parse_route(
                route_result,
                destination=route_to,
                rdma_devices=rdma_devices,
            )

    if rdma_status == "disabled":
        transport_state = TransportState.DISABLED
    elif rdma_status == "enabled" and peer_connected:
        transport_state = TransportState.PEER_LINKED_CONFIG_PENDING
    elif rdma_status == "enabled":
        transport_state = TransportState.ENABLED_NO_PEER
    else:
        transport_state = TransportState.UNAVAILABLE

    warnings = [
        warning
        for warning in (
            _warning_for(rdma_result, "rdma") if linux else _warning_for(rdma_result, "rdma_ctl"),
            _warning_for(devices_result, "ibv_devices"),
            None if linux else _warning_for(thunderbolt_result, "Thunderbolt"),
            _warning_for(route_result, "route") if route_result else None,
        )
        if warning
    ]
    if rdma_status == "enabled" and not rdma_devices:
        warnings.append("RDMA is enabled, but ibv_devices reported no RDMA interfaces.")
    if not linux and rdma_status == "enabled" and not peer_connected:
        warnings.append("RDMA is enabled, but no Thunderbolt peer is connected.")
    if route is not None and route.interface and not route.uses_rdma_interface:
        warnings.append(
            f"Route to {route.destination} uses {route.interface}, "
            "not an RDMA-capable interface."
        )

    accelerator = detect_accelerator_hardware()
    if accelerator.kind == "cuda":
        physical_memory_bytes = (
            accelerator.physical_memory_bytes or _system_memory_bytes()
        )
        recommended_working_set_bytes = (
            accelerator.recommended_working_set_bytes or physical_memory_bytes
        )
        chip_name = accelerator.name
    elif system == "darwin" and accelerator.kind == "metal":
        detected = hardware.detect_hardware()
        physical_memory_bytes = int(detected.total_memory_gb * 1024**3)
        recommended_working_set_bytes = detected.max_working_set_bytes
        chip_name = detected.chip_name
    else:
        physical_memory_bytes = accelerator.physical_memory_bytes or _system_memory_bytes()
        recommended_working_set_bytes = (
            accelerator.recommended_working_set_bytes or physical_memory_bytes
        )
        chip_name = accelerator.name
    ceiling_measured = True
    try:
        # This is deliberately the same computation a distributed rank runs
        # immediately before loading. ``recommended_working_set_bytes`` is a
        # useful hardware fact, but it does not include the active oMLX memory
        # tier or current unified-memory pressure.
        from .memory_guard import ceiling_breakdown

        admission_ceiling_bytes = int(ceiling_breakdown().get("hard_limit", 0))
    except Exception:
        # Capability probing must remain available on a worker-only or partly
        # upgraded install. The guard machinery is genuinely absent here, which
        # is different from a measured zero and must fall back rather than
        # advertise installed RAM.
        admission_ceiling_bytes = 0
        ceiling_measured = False
    if not ceiling_measured and accelerator.kind == "cuda":
        # CUDA does not expose Metal's recommended working-set guard. With no
        # live free-memory probe at all, retain ten percent for the OS, CUDA
        # context, communication buffers, and load-time transients rather than
        # advertising all installed memory as model capacity. A *measured* zero
        # (another process such as vLLM already owns the VRAM) is a real "no
        # room right now" and must not be inflated back to installed size, or
        # the planner would place a shard that OOMs on load.
        admission_ceiling_bytes = int(recommended_working_set_bytes * 0.90)
    timestamp = (now or (lambda: datetime.now(UTC)))()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    return ClusterStatus(
        collected_at=timestamp.isoformat(),
        hostname=socket.gethostname(),
        platform=platform.platform(),
        chip_name=chip_name,
        physical_memory_bytes=physical_memory_bytes,
        recommended_working_set_bytes=recommended_working_set_bytes,
        runtime=RuntimeCapability(
            omlx_version=__version__,
            mlx_version=hardware.get_mlx_version(),
            mlx_lm_version=hardware.get_mlx_lm_version(),
            python_version=platform.python_version(),
            macos_version=platform.mac_ver()[0] or "unknown",
            os_name=platform.system().lower() or "unknown",
            os_version=platform.release() or "unknown",
            python_executable=_advertised_python_executable(),
        ),
        transport_state=transport_state,
        rdma=RDMACapability(
            control_status=rdma_status,
            devices=rdma_devices,
            addresses=tuple(rdma_addresses),
            network_interfaces=tuple(
                (device, linux_rdma_links[device])
                for device in rdma_devices
                if device in linux_rdma_links
            ) if linux else tuple(
                (device, device.removeprefix("rdma_"))
                for device in rdma_devices
                if device.startswith("rdma_")
            ),
        ),
        thunderbolt_ports=ports,
        route=route,
        admission_ceiling_bytes=admission_ceiling_bytes,
        warnings=tuple(warnings),
        accelerator=accelerator.kind,
        accelerator_vendor=accelerator.vendor,
        memory_kind=accelerator.memory_kind,
        distributed_backends=accelerator.distributed_backends,
        fabric_kind=(
            "connectx-7"
            if accelerator.kind == "cuda"
            and any(
                device.lower().startswith(("mlx5_", "roce")) for device in rdma_devices
            )
            else None
        ),
        # Fabric grouping is an operator decision made and verified through the
        # dashboard. Environment variables made a normal setup step terminal-
        # only and let stale process state masquerade as a verified topology.
        fabric_group_id=None,
    )


def format_cluster_status(status: ClusterStatus) -> str:
    """Render a compact human-readable node status."""

    gib = 1024**3
    lines = [
        "oMLX cluster node status",
        f"Node:        {status.hostname}",
        f"Chip:        {status.chip_name}",
        f"Accelerator: {status.accelerator} ({status.accelerator_vendor})",
        (
            "Memory:      "
            f"{status.physical_memory_bytes / gib:.2f} GiB physical / "
            f"{status.recommended_working_set_bytes / gib:.2f} GiB recommended"
        ),
        (
            "Runtime:     "
            f"oMLX {status.runtime.omlx_version} / "
            f"MLX {status.runtime.mlx_version} / "
            f"MLX-LM {status.runtime.mlx_lm_version}"
        ),
        f"Transport:   {status.transport_state.value}",
        (
            "RDMA:        "
            f"{status.rdma.control_status}"
            + (f" ({', '.join(status.rdma.devices)})" if status.rdma.devices else "")
        ),
        (
            "Thunderbolt: "
            + (
                "peer connected"
                if status.thunderbolt_peer_connected
                else "no peer connected"
            )
            + f" ({len(status.thunderbolt_ports)} ports detected)"
        ),
    ]
    if status.route:
        route_kind = "RDMA" if status.route.uses_rdma_interface else "non-RDMA"
        lines.append(
            f"Route:       {status.route.destination} via "
            f"{status.route.interface or 'unknown'} ({route_kind})"
        )
    if status.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in status.warnings)
    return "\n".join(lines)
