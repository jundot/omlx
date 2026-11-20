# SPDX-License-Identifier: Apache-2.0
"""Tests for read-only cluster capability probing."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from omlx.cluster import probe
from omlx.cluster.models import TransportState
from omlx.cluster.probe import CommandResult
from omlx.utils.hardware import HardwareInfo


class FakeRunner:
    def __init__(self, outputs: dict[str, tuple[int, str, str]]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def __call__(self, args, *, timeout: float) -> CommandResult:
        argv = tuple(str(arg) for arg in args)
        self.calls.append((argv, timeout))
        name = Path(argv[0]).name
        returncode, stdout, stderr = self.outputs.get(
            name,
            (127, "", f"{name}: unavailable"),
        )
        return CommandResult(
            args=argv,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )


NO_PEER_THUNDERBOLT = json.dumps(
    {
        "SPThunderboltDataType": [
            {
                "_name": "thunderboltusb4_bus_0",
                "device_name_key": "MacBook Pro",
                "receptacle_1_tag": {
                    "current_speed_key": "Up to 120 Gb/s",
                    "receptacle_id_key": "1",
                    "receptacle_status_key": "receptacle_no_devices_connected",
                },
            }
        ]
    }
)

CONNECTED_THUNDERBOLT = json.dumps(
    {
        "SPThunderboltDataType": [
            {
                "_name": "thunderboltusb4_bus_0",
                "device_name_key": "MacBook Pro",
                "receptacle_1_tag": {
                    "current_speed_key": "Up to 120 Gb/s",
                    "receptacle_id_key": "1",
                    "receptacle_status_key": "receptacle_device_connected",
                    "device_name_key": "Mac Studio",
                },
            }
        ]
    }
)

IBV_OUTPUT = """\
    device            node GUID
    ------          ----------------
    rdma_en1         a0910a0a8bd8ac05
    rdma_en2         a2910a0a8bd8ac05
"""


def _patch_hardware(monkeypatch) -> None:
    monkeypatch.setattr(
        probe.hardware,
        "detect_hardware",
        lambda: HardwareInfo(
            chip_name="Apple M5 Max",
            total_memory_gb=128.0,
            max_working_set_bytes=115_448_725_504,
            mlx_device_name="Apple M5 Max",
        ),
    )
    monkeypatch.setattr(probe.hardware, "get_mlx_version", lambda: "0.32.0")
    monkeypatch.setattr(probe.hardware, "get_mlx_lm_version", lambda: "0.31.3")
    monkeypatch.setattr(probe.socket, "gethostname", lambda: "MacBook-Pro")
    monkeypatch.setattr(probe.platform, "platform", lambda: "macOS-26.5.2-arm64")
    monkeypatch.setattr(probe.platform, "python_version", lambda: "3.11.14")
    monkeypatch.setattr(probe.platform, "mac_ver", lambda: ("26.5.2", (), "arm64"))
    monkeypatch.setattr(
        "omlx.cluster.memory_guard.ceiling_breakdown",
        lambda *_a, **_k: {"hard_limit": 100 * 1024**3},
    )


def test_collect_status_distinguishes_enabled_from_linked(monkeypatch):
    _patch_hardware(monkeypatch)
    runner = FakeRunner(
        {
            "rdma_ctl": (0, "enabled\n", ""),
            "ibv_devices": (0, IBV_OUTPUT, ""),
            "ipconfig": (0, "169.254.42.1\n", ""),
            "system_profiler": (0, NO_PEER_THUNDERBOLT, ""),
            "route": (
                0,
                "   route to: 198.51.100.197\n"
                "destination: 198.51.100.197\n"
                "  interface: en7\n",
                "",
            ),
        }
    )

    status = probe.collect_cluster_status(
        route_to="198.51.100.197",
        runner=runner,
        now=lambda: datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
    )

    assert status.transport_state is TransportState.ENABLED_NO_PEER
    assert status.rdma.devices == ("rdma_en1", "rdma_en2")
    assert status.rdma.addresses[0] == ("rdma_en1", "169.254.42.1")
    assert status.thunderbolt_peer_connected is False
    assert status.route is not None
    assert status.route.interface == "en7"
    assert status.route.uses_rdma_interface is False
    assert status.physical_memory_bytes == 128 * 1024**3
    assert status.admission_ceiling_bytes == 100 * 1024**3
    assert any("no Thunderbolt peer" in item for item in status.warnings)
    assert any("not an RDMA-capable interface" in item for item in status.warnings)

    serialized = status.to_dict()
    assert serialized["protocol_version"] == "1.0"
    assert serialized["node"]["admission_ceiling_bytes"] == 100 * 1024**3
    assert serialized["transport"]["state"] == "enabled_no_peer"
    assert serialized["transport"]["rdma"]["enabled"] is True
    assert serialized["transport"]["rdma"]["addresses"]["rdma_en1"] == "169.254.42.1"


def test_collect_status_reports_connected_rdma_route(monkeypatch):
    _patch_hardware(monkeypatch)
    runner = FakeRunner(
        {
            "rdma_ctl": (0, "enabled\n", ""),
            "ibv_devices": (0, IBV_OUTPUT, ""),
            "ipconfig": (0, "169.254.42.1\n", ""),
            "system_profiler": (0, CONNECTED_THUNDERBOLT, ""),
            "route": (
                0,
                "destination: 169.254.42.2\n  interface: en1\n",
                "",
            ),
        }
    )

    status = probe.collect_cluster_status(
        route_to="169.254.42.2",
        runner=runner,
    )

    assert status.transport_state is TransportState.PEER_LINKED_CONFIG_PENDING
    assert status.thunderbolt_peer_connected is True
    assert status.thunderbolt_ports[0].peer_names == ("Mac Studio",)
    assert status.route is not None
    assert status.route.uses_rdma_interface is True
    assert not any("no Thunderbolt peer" in item for item in status.warnings)


def test_collect_status_rejects_non_ip_route_target():
    with pytest.raises(ValueError, match="IPv4 or IPv6"):
        probe.collect_cluster_status(route_to="studio.local")


def test_parse_invalid_thunderbolt_payload_returns_no_ports():
    result = CommandResult(
        args=("/usr/sbin/system_profiler",),
        returncode=0,
        stdout="{not-json",
    )
    assert probe.parse_thunderbolt_ports(result) == ()


def test_collect_status_does_not_advertise_an_ssh_user():
    # Dropped until a consumer lands: an unvalidated login-name string on the
    # wire is exactly the shape the validate_ssh_target fix exists to keep out
    # of ssh argv construction.
    status = probe.collect_cluster_status()
    assert "ssh_user" not in status.to_dict()["node"]


def test_mlx_version_uses_core_module_version(monkeypatch):
    monkeypatch.setattr(probe.hardware, "HAS_MLX", True)
    monkeypatch.setattr(
        probe.hardware,
        "mx",
        SimpleNamespace(__version__="0.32.0"),
    )
    assert probe.hardware.get_mlx_version() == "0.32.0"


