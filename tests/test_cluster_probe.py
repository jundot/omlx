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
    monkeypatch.setattr(
        probe,
        "detect_accelerator_hardware",
        lambda: probe.AcceleratorHardware(
            kind="metal",
            vendor="apple",
            memory_kind="unified",
            name="Apple M5 Max",
            physical_memory_bytes=128 * 1024**3,
            recommended_working_set_bytes=115_448_725_504,
            distributed_backends=("ring", "jaccl"),
        ),
    )
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
    assert serialized["node"]["accelerator"] == "metal"
    assert serialized["node"]["accelerator_vendor"] == "apple"
    assert serialized["node"]["distributed_backends"] == ["ring", "jaccl"]
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


def _cuda_gb10(**overrides):
    fields = dict(
        kind="cuda",
        vendor="nvidia",
        memory_kind="unified",
        name="NVIDIA GB10",
        physical_memory_bytes=128 * 1024**3,
        recommended_working_set_bytes=128 * 1024**3,
        distributed_backends=("ring", "nccl"),
    )
    fields.update(overrides)
    return probe.AcceleratorHardware(**fields)


def test_cuda_status_falls_back_to_safe_budget_when_the_guard_is_absent(monkeypatch):
    """With no ceiling measurement at all, reserve ten percent of installed."""

    _patch_hardware(monkeypatch)
    monkeypatch.setattr(probe, "detect_accelerator_hardware", _cuda_gb10)

    def _guard_unavailable(*_a, **_k):
        raise RuntimeError("memory guard machinery is not installed")

    monkeypatch.setattr(
        "omlx.cluster.memory_guard.ceiling_breakdown", _guard_unavailable
    )
    runner = FakeRunner(
        {
            "ibv_devices": (
                0,
                "    device            node GUID\n"
                "    ------          ----------------\n"
                "    mlx5_0          a0910a0a8bd8ac05\n",
                "",
            ),
        }
    )

    status = probe.collect_cluster_status(runner=runner)

    assert status.accelerator == "cuda"
    assert status.accelerator_vendor == "nvidia"
    assert status.chip_name == "NVIDIA GB10"
    assert status.fabric_kind == "connectx-7"
    assert status.fabric_group_id is None
    assert status.admission_ceiling_bytes == int(128 * 1024**3 * 0.90)
    assert status.to_dict()["node"]["memory_kind"] == "unified"


def test_cuda_measured_zero_free_is_not_inflated_to_installed_size(monkeypatch):
    """A measured empty VRAM must stay zero, not be advertised as capacity.

    ``_cuda_ceiling_breakdown`` returns ``hard_limit == 0`` when the live free
    memory is zero, e.g. another service such as vLLM already owns the whole
    GB10. Re-inflating that to ninety percent of installed memory made the
    dashboard advertise room that is not there, and the planner would place a
    shard that OOMs on load. The sibling ``probe_remote_admission_ceiling``
    already fails closed on the same zero; this keeps the capability probe
    consistent with it.
    """

    _patch_hardware(monkeypatch)
    monkeypatch.setattr(probe, "detect_accelerator_hardware", _cuda_gb10)
    monkeypatch.setattr(
        "omlx.cluster.memory_guard.ceiling_breakdown",
        lambda *_a, **_k: {"hard_limit": 0},
    )

    status = probe.collect_cluster_status(runner=FakeRunner({}))

    assert status.accelerator == "cuda"
    assert status.admission_ceiling_bytes == 0


def test_cuda_measured_ceiling_is_used_verbatim(monkeypatch):
    """A real measured ceiling is neither inflated nor floored."""

    _patch_hardware(monkeypatch)
    monkeypatch.setattr(probe, "detect_accelerator_hardware", _cuda_gb10)
    monkeypatch.setattr(
        "omlx.cluster.memory_guard.ceiling_breakdown",
        lambda *_a, **_k: {"hard_limit": 40 * 1024**3},
    )

    status = probe.collect_cluster_status(runner=FakeRunner({}))

    assert status.admission_ceiling_bytes == 40 * 1024**3


def test_linux_cuda_probe_maps_connectx_device_to_network_interface(monkeypatch):
    _patch_hardware(monkeypatch)
    monkeypatch.setattr(probe.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        probe,
        "detect_accelerator_hardware",
        lambda: probe.AcceleratorHardware(
            kind="cuda",
            vendor="nvidia",
            memory_kind="unified",
            name="NVIDIA GB10",
            physical_memory_bytes=128 * 1024**3,
            recommended_working_set_bytes=128 * 1024**3,
            distributed_backends=("ring", "nccl"),
        ),
    )
    monkeypatch.setattr(
        probe.hardware,
        "detect_hardware",
        lambda: (_ for _ in ()).throw(AssertionError("macOS probe ran on Linux")),
    )
    runner = FakeRunner(
        {
            "rdma": (
                0,
                json.dumps([{"ifname": "mlx5_0/1", "netdev": "enp1s0f0np0"}]),
                "",
            ),
            "ibv_devices": (
                0,
                "    device            node GUID\n"
                "    ------          ----------------\n"
                "    mlx5_0          a0910a0a8bd8ac05\n",
                "",
            ),
            "ip": (
                0,
                json.dumps(
                    [
                        {
                            "ifname": "enp1s0f0np0",
                            "addr_info": [
                                {"local": "192.168.100.1", "scope": "global"}
                            ],
                        }
                    ]
                ),
                "",
            ),
        }
    )

    status = probe.collect_cluster_status(runner=runner)
    serialized = status.to_dict()

    assert status.fabric_kind == "connectx-7"
    assert serialized["transport"]["rdma"]["addresses"] == {
        "mlx5_0": "192.168.100.1"
    }
    assert serialized["transport"]["rdma"]["network_interfaces"] == {
        "mlx5_0": "enp1s0f0np0"
    }


def test_linux_probe_accepts_dgx_spark_roce_device_names():
    devices = probe.parse_ibv_devices(
        probe.CommandResult(
            args=("ibv_devices",),
            returncode=0,
            stdout=(
                "    device             node GUID\n"
                "    ------          ----------------\n"
                "    rocep1s0f1       10b6760300f01ade\n"
                "    roceP2p1s0f1     10b6760300f01ae2\n"
            ),
        )
    )
    links = probe.parse_linux_rdma_links(
        probe.CommandResult(
            args=("rdma",),
            returncode=0,
            stdout=json.dumps(
                [
                    {
                        "ifname": "rocep1s0f1",
                        "state": "ACTIVE",
                        "netdev": "enp1s0f1np1",
                    },
                    {
                        "ifname": "roceP2p1s0f1",
                        "state": "ACTIVE",
                        "netdev": "enP2p1s0f1np1",
                    },
                ]
            ),
        )
    )

    assert devices == ("rocep1s0f1", "roceP2p1s0f1")
    assert links == {
        "rocep1s0f1": "enp1s0f1np1",
        "roceP2p1s0f1": "enP2p1s0f1np1",
    }
