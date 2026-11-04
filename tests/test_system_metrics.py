# SPDX-License-Identifier: Apache-2.0
"""Tests for GET /api/system-metrics and omlx.system_metrics."""

import subprocess
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from omlx import system_metrics
from omlx.server import ServerState, app

IOREG_SAMPLE = """\
+-o AGXAcceleratorG17G  <class AGXAcceleratorG17G, id 0x100000478, registered>
    {
      "IOMatchedAtBoot" = Yes
      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=5993463808,"Tiler Utilization %"=0,"Renderer Utilization %"=0,"Device Utilization %"=37,"In use system memory"=698859520}
      "model" = "Apple M5"
    }
"""

VM_STAT_SAMPLE = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    58056.
Pages occupied by compressor:                 176347.
"""

TOP_LEVEL_KEYS = {
    "timestamp",
    "uptime_seconds",
    "cpu",
    "gpu",
    "memory",
    "thermal_state",
}
CPU_KEYS = {
    "total_percent",
    "efficiency_percent",
    "performance_percent",
    "efficiency_cores",
    "performance_cores",
    "load_averages",
}
GPU_KEYS = {"utilization_percent", "memory_in_use_bytes"}
MEMORY_KEYS = {
    "total_bytes",
    "wired_bytes",
    "active_bytes",
    "inactive_bytes",
    "compressed_bytes",
    "free_bytes",
}


@pytest.fixture(autouse=True)
def fresh_probe_state():
    """Reset module-level caches and the CPU sampler between tests."""
    system_metrics._cache.clear()
    system_metrics._cpu_primed = False
    system_metrics._cpu_sample_at = 0.0
    system_metrics._cpu_last = None
    yield
    system_metrics._cache.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def server_state():
    state = ServerState()
    with patch("omlx.server._server_state", state):
        yield state


def _assert_shape(data):
    assert set(data) == TOP_LEVEL_KEYS
    assert set(data["cpu"]) == CPU_KEYS
    assert set(data["gpu"]) == GPU_KEYS
    assert set(data["memory"]) == MEMORY_KEYS


class TestRoute:
    def test_returns_full_key_set(self, client, server_state):
        resp = client.get("/api/system-metrics")
        assert resp.status_code == 200
        data = resp.json()
        _assert_shape(data)
        assert isinstance(data["timestamp"], float)
        assert data["uptime_seconds"] is None or data["uptime_seconds"] >= 0
        assert data["memory"]["total_bytes"] > 0
        assert len(data["cpu"]["load_averages"]) == 3

    def test_cpu_percentages_null_on_first_call_then_filled(
        self, client, server_state, monkeypatch
    ):
        monkeypatch.setattr(system_metrics, "_CPU_MIN_INTERVAL", 0.0)
        monkeypatch.setattr(system_metrics, "efficiency_core_count", lambda: 2)
        import psutil

        monkeypatch.setattr(
            psutil,
            "cpu_percent",
            lambda percpu, interval: [10.0, 20.0, 30.0, 50.0],
        )
        monkeypatch.setattr(psutil, "cpu_count", lambda logical: 4)
        first = client.get("/api/system-metrics").json()["cpu"]
        assert first["total_percent"] is None
        assert first["efficiency_cores"] == 2
        assert first["performance_cores"] == 2
        second = client.get("/api/system-metrics").json()["cpu"]
        assert second["total_percent"] == 27.5
        assert second["efficiency_percent"] == 15.0
        assert second["performance_percent"] == 40.0
        assert second["efficiency_cores"] == 2
        assert second["performance_cores"] == 2

    def test_all_probes_failing_still_returns_200_with_nulls(
        self, client, server_state, monkeypatch
    ):
        import psutil

        def boom(*_a, **_k):
            raise RuntimeError("probe failed")

        monkeypatch.setattr(psutil, "cpu_percent", boom)
        monkeypatch.setattr(psutil, "virtual_memory", boom)
        monkeypatch.setattr(psutil, "boot_time", boom)
        monkeypatch.setattr(psutil, "cpu_count", boom)
        monkeypatch.setattr(subprocess, "run", boom)
        monkeypatch.setattr(system_metrics.os, "getloadavg", boom)
        monkeypatch.setattr(system_metrics, "_IS_DARWIN", True)
        monkeypatch.setattr(system_metrics.ctypes, "CDLL", boom)

        resp = client.get("/api/system-metrics")
        assert resp.status_code == 200
        data = resp.json()
        _assert_shape(data)
        assert data["uptime_seconds"] is None
        assert data["thermal_state"] is None
        assert all(v is None for v in data["cpu"].values())
        assert all(v is None for v in data["gpu"].values())
        assert all(v is None for v in data["memory"].values())

    def test_requires_key_when_configured(self, client, server_state):
        server_state.api_key = "main-key"
        server_state.bind_host = "0.0.0.0"
        assert client.get("/api/system-metrics").status_code == 401
        bad = client.get(
            "/api/system-metrics", headers={"Authorization": "Bearer nope"}
        )
        assert bad.status_code == 401
        ok = client.get(
            "/api/system-metrics", headers={"Authorization": "Bearer main-key"}
        )
        assert ok.status_code == 200
        _assert_shape(ok.json())

    def test_accepts_sub_key(self, client, server_state):
        from omlx.settings import GlobalSettings, SubKeyEntry

        settings = GlobalSettings()
        settings.auth.sub_keys = [SubKeyEntry(key="sub-key-1234567", name="sub")]
        server_state.global_settings = settings
        server_state.api_key = "main-key"
        server_state.bind_host = "0.0.0.0"
        resp = client.get(
            "/api/system-metrics",
            headers={"Authorization": "Bearer sub-key-1234567"},
        )
        assert resp.status_code == 200


class TestParsers:
    def test_ioreg_sample(self):
        assert system_metrics.parse_ioreg_gpu(IOREG_SAMPLE) == {
            "utilization_percent": 37.0,
            "memory_in_use_bytes": 698859520,
        }

    def test_ioreg_driver_key_is_not_confused_with_in_use_memory(self):
        text = '"PerformanceStatistics" = {"In use system memory (driver)"=123}'
        assert system_metrics.parse_ioreg_gpu(text) == {
            "utilization_percent": None,
            "memory_in_use_bytes": None,
        }

    def test_ioreg_garbage(self):
        assert system_metrics.parse_ioreg_gpu("") == {
            "utilization_percent": None,
            "memory_in_use_bytes": None,
        }

    def test_vm_stat_compressed(self):
        assert system_metrics.parse_vm_stat_compressed(VM_STAT_SAMPLE) == (
            176347 * 16384
        )
        assert system_metrics.parse_vm_stat_compressed("nothing") is None

    def test_ioreg_result_is_cached(self, monkeypatch):
        monkeypatch.setattr(system_metrics, "_IS_DARWIN", True)
        calls = []

        def fake_run(*cmd):
            calls.append(cmd)
            return IOREG_SAMPLE

        monkeypatch.setattr(system_metrics, "_run", fake_run)
        first = system_metrics._gpu_metrics()
        second = system_metrics._gpu_metrics()
        assert first == second
        assert first["utilization_percent"] == 37.0
        assert len(calls) == 1


class TestClusterSplit:
    def test_efficiency_cores_first(self):
        per_cpu = [10.0, 20.0, 80.0, 100.0, 60.0, 40.0]
        out = system_metrics.split_cpu_clusters(per_cpu, 2)
        assert out["efficiency_cores"] == 2
        assert out["performance_cores"] == 4
        assert out["efficiency_percent"] == 15.0
        assert out["performance_percent"] == 70.0
        assert out["total_percent"] == 51.7

    def test_no_efficiency_cluster_counts_every_core_as_performance(self):
        out = system_metrics.split_cpu_clusters([20.0, 40.0], 0)
        assert out["efficiency_cores"] == 0
        assert out["efficiency_percent"] is None
        assert out["performance_cores"] == 2
        assert out["performance_percent"] == 30.0
        assert out["total_percent"] == 30.0

    def test_percentages_clamped(self):
        out = system_metrics.split_cpu_clusters([120.0, -5.0], 1)
        assert out["efficiency_percent"] == 100.0
        assert out["performance_percent"] == 0.0

    def test_efficiency_core_count_needs_two_levels(self, monkeypatch):
        monkeypatch.setattr(system_metrics, "_IS_DARWIN", True)
        values = {"hw.nperflevels": 1, "hw.perflevel1.logicalcpu": 4}
        monkeypatch.setattr(system_metrics, "_sysctl_int", values.get)
        assert system_metrics.efficiency_core_count() == 0
        system_metrics._cache.clear()
        values["hw.nperflevels"] = 2
        assert system_metrics.efficiency_core_count() == 4


class TestMemory:
    def test_free_is_total_minus_wired_active_compressed(self, monkeypatch):
        from collections import namedtuple

        import psutil

        VM = namedtuple("VM", "total wired active inactive")
        monkeypatch.setattr(psutil, "virtual_memory", lambda: VM(1000, 100, 200, 300))
        monkeypatch.setattr(system_metrics, "_compressed_bytes", lambda: 50)
        mem = system_metrics._memory_metrics()
        assert mem == {
            "total_bytes": 1000,
            "wired_bytes": 100,
            "active_bytes": 200,
            "inactive_bytes": 300,
            "compressed_bytes": 50,
            "free_bytes": 650,
        }

    def test_free_null_when_compressed_unknown(self, monkeypatch):
        monkeypatch.setattr(system_metrics, "_compressed_bytes", lambda: None)
        assert system_metrics._memory_metrics()["free_bytes"] is None
