# SPDX-License-Identifier: Apache-2.0
"""Serializable models for the oMLX cluster control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

CLUSTER_PROTOCOL_VERSION = "1.0"
WORKER_PROTOCOL_VERSION = 1


class TransportState(StrEnum):
    """Observed transport readiness.

    The states deliberately distinguish operating-system RDMA enablement from a
    physically connected peer and from a configured/benchmarked JACCL session.
    This slice can report only through ``peer_linked_config_pending``.
    """

    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"
    ENABLED_NO_PEER = "enabled_no_peer"
    PEER_LINKED_CONFIG_PENDING = "peer_linked_config_pending"


@dataclass(frozen=True)
class RuntimeCapability:
    omlx_version: str
    mlx_version: str
    mlx_lm_version: str
    python_version: str
    macos_version: str
    os_name: str = "darwin"
    os_version: str = "unknown"
    python_executable: str = ""
    # Content identifiers, not mutable paths.  TP layout qualifications use
    # them to reject a vector after either the collective library or a custom
    # kernel bundle changes without a package-version bump.
    jaccl_identifier: str = ""
    kernel_identifier: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "omlx_version": self.omlx_version,
            "mlx_version": self.mlx_version,
            "mlx_lm_version": self.mlx_lm_version,
            "python_version": self.python_version,
            "macos_version": self.macos_version,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "python_executable": self.python_executable,
            "jaccl_identifier": self.jaccl_identifier,
            "kernel_identifier": self.kernel_identifier,
        }


@dataclass(frozen=True)
class RDMACapability:
    control_status: str
    devices: tuple[str, ...]
    addresses: tuple[tuple[str, str], ...] = ()
    network_interfaces: tuple[tuple[str, str], ...] = ()

    @property
    def enabled(self) -> bool:
        return self.control_status == "enabled"

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_status": self.control_status,
            "enabled": self.enabled,
            "devices": list(self.devices),
            "addresses": dict(self.addresses),
            "network_interfaces": dict(self.network_interfaces),
        }


@dataclass(frozen=True)
class ThunderboltPort:
    bus_name: str
    receptacle_id: str | None
    speed: str | None
    status: str
    peer_connected: bool
    peer_names: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "bus_name": self.bus_name,
            "receptacle_id": self.receptacle_id,
            "speed": self.speed,
            "status": self.status,
            "peer_connected": self.peer_connected,
            "peer_names": list(self.peer_names),
        }


@dataclass(frozen=True)
class RouteCapability:
    destination: str
    interface: str | None
    gateway: str | None
    uses_rdma_interface: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "destination": self.destination,
            "interface": self.interface,
            "gateway": self.gateway,
            "uses_rdma_interface": self.uses_rdma_interface,
        }


@dataclass(frozen=True)
class PowerModeCapability:
    """Read-only macOS performance-state facts relevant to calibration."""

    ac_low_power_mode: bool | None = None
    battery_low_power_mode: bool | None = None
    source: str = "unavailable"

    @property
    def low_power_mode_enabled(self) -> bool:
        return self.ac_low_power_mode is True or self.battery_low_power_mode is True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ac_low_power_mode": self.ac_low_power_mode,
            "battery_low_power_mode": self.battery_low_power_mode,
            "low_power_mode_enabled": self.low_power_mode_enabled,
        }


@dataclass(frozen=True)
class ClusterStatus:
    collected_at: str
    hostname: str
    platform: str
    chip_name: str
    physical_memory_bytes: int
    recommended_working_set_bytes: int
    runtime: RuntimeCapability
    transport_state: TransportState
    rdma: RDMACapability
    thunderbolt_ports: tuple[ThunderboltPort, ...]
    power: PowerModeCapability = field(default_factory=PowerModeCapability)
    route: RouteCapability | None = None
    # The live ProcessMemoryEnforcer ceiling can be lower than both physical
    # RAM and MLX's nominal recommended working set when another application
    # is using unified memory. Cluster planning must use this same number or a
    # plan can pass in the GUI and be refused by the rank moments later.
    admission_ceiling_bytes: int = 0
    warnings: tuple[str, ...] = ()
    protocol_version: str = CLUSTER_PROTOCOL_VERSION
    # Accelerator identity is deliberately explicit. Inferring CUDA from a
    # chip-name substring made the dashboard presentation drive scheduling
    # policy and could turn an unrecognised NVIDIA device into a Mac.
    accelerator: str = "metal"
    accelerator_vendor: str = "apple"
    memory_kind: str = "unified"
    distributed_backends: tuple[str, ...] = ()
    # A worker may advertise a verified high-speed fabric group. Members with
    # the same ID are presented as one logical supernode while retaining their
    # individual rank, memory, and health records.
    fabric_kind: str | None = None
    fabric_group_id: str | None = None
    fabric_verified: bool = False

    @property
    def thunderbolt_peer_connected(self) -> bool:
        return any(port.peer_connected for port in self.thunderbolt_ports)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "collected_at": self.collected_at,
            "node": {
                "hostname": self.hostname,
                "platform": self.platform,
                "chip_name": self.chip_name,
                "physical_memory_bytes": self.physical_memory_bytes,
                "recommended_working_set_bytes": self.recommended_working_set_bytes,
                "admission_ceiling_bytes": self.admission_ceiling_bytes,
                "accelerator": self.accelerator,
                "accelerator_vendor": self.accelerator_vendor,
                "memory_kind": self.memory_kind,
                "distributed_backends": list(self.distributed_backends),
                "fabric_kind": self.fabric_kind,
                "fabric_group_id": self.fabric_group_id,
                "fabric_verified": self.fabric_verified,
                "power": self.power.to_dict(),
            },
            "runtime": self.runtime.to_dict(),
            "transport": {
                "state": self.transport_state.value,
                "rdma": self.rdma.to_dict(),
                "thunderbolt": {
                    "peer_connected": self.thunderbolt_peer_connected,
                    "ports": [port.to_dict() for port in self.thunderbolt_ports],
                },
                "route": self.route.to_dict() if self.route else None,
            },
            "warnings": list(self.warnings),
        }
