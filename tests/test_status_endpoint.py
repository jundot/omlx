# SPDX-License-Identifier: Apache-2.0
"""Tests for GET /api/status endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from omlx.server import ServerState, app


@pytest.fixture
def client():
    return TestClient(app)


class TestStatusEndpoint:
    """Tests for /api/status lightweight status endpoint."""

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        """Set up a clean server state for each test."""
        state = ServerState()
        with patch("omlx.server._server_state", state):
            self._state = state
            yield

    def test_returns_ok_when_pool_is_none(self, client):
        """When engine pool is not initialized, return basic status."""
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["models_discovered"] == 0
        assert data["models_loaded"] == 0
        assert data["models_loading"] == 0
        assert data["loaded_models"] == []
        assert data["active_requests"] == 0
        assert data["waiting_requests"] == 0
        assert "version" in data
        assert "uptime_seconds" in data

    def test_returns_pool_info(self, client):
        """When engine pool exists, return model and memory stats."""
        pool = MagicMock(spec=[
            "model_count", "loaded_model_count", "get_loaded_model_ids",
            "current_model_memory", "_entries",
        ])
        pool.model_count = 5
        pool.loaded_model_count = 2
        pool.get_loaded_model_ids.return_value = ["model-a", "model-b"]
        pool.current_model_memory = 16 * 1024**3
        enforcer = MagicMock(spec=["get_final_ceiling"])
        enforcer.get_final_ceiling.return_value = 32 * 1024**3
        self._state.process_memory_enforcer = enforcer

        entry_a = MagicMock(spec=["is_loading", "engine"])
        entry_a.is_loading = False
        entry_a.engine = None
        entry_b = MagicMock(spec=["is_loading", "engine"])
        entry_b.is_loading = True
        entry_b.engine = None
        pool._entries = {"model-a": entry_a, "model-b": entry_b}

        self._state.engine_pool = pool

        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["models_discovered"] == 5
        assert data["models_loaded"] == 2
        assert data["models_loading"] == 1
        assert data["loaded_models"] == ["model-a", "model-b"]
        assert data["model_memory_used"] == 16 * 1024**3
        assert data["model_memory_max"] == 32 * 1024**3
        assert "GB" in data["model_memory_used_formatted"]
        assert "GB" in data["model_memory_max_formatted"]

    def test_status_ignores_memory_ceiling_error(self, client):
        """Memory telemetry failures should not break status polling."""
        pool = MagicMock(spec=[
            "model_count", "loaded_model_count", "get_loaded_model_ids",
            "current_model_memory", "_entries",
        ])
        pool.model_count = 1
        pool.loaded_model_count = 1
        pool.get_loaded_model_ids.return_value = ["model-a"]
        pool.current_model_memory = 16 * 1024**3
        pool._entries = {}
        enforcer = MagicMock(spec=["get_final_ceiling"])
        enforcer.get_final_ceiling.side_effect = RuntimeError(
            "host_statistics64 failed"
        )
        self._state.engine_pool = pool
        self._state.process_memory_enforcer = enforcer

        resp = client.get("/api/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["model_memory_max"] is None
        assert data["model_memory_max_formatted"] == "unlimited"

    def test_health_ignores_memory_ceiling_error(self, client):
        """Health should stay healthy when optional memory telemetry fails."""
        pool = MagicMock(spec=[
            "model_count", "loaded_model_count", "current_model_memory",
        ])
        pool.model_count = 1
        pool.loaded_model_count = 1
        pool.current_model_memory = 16 * 1024**3
        enforcer = MagicMock(spec=["get_final_ceiling"])
        enforcer.get_final_ceiling.side_effect = RuntimeError(
            "host_statistics64 failed"
        )
        self._state.engine_pool = pool
        self._state.process_memory_enforcer = enforcer

        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["engine_pool"]["final_ceiling"] == 0

    def test_aggregates_active_waiting_requests(self, client):
        """Active/waiting request counts are summed across loaded engines."""
        # Build a mock engine with scheduler
        scheduler = MagicMock(spec=["waiting"])
        scheduler.waiting = [1, 2]  # 2 waiting

        core = MagicMock(spec=["_output_collectors", "scheduler"])
        core._output_collectors = {"req-1": None, "req-2": None, "req-3": None}
        core.scheduler = scheduler

        async_core = MagicMock(spec=["engine"])
        async_core.engine = core

        engine = MagicMock(spec=["_engine"])
        engine._engine = async_core

        entry = MagicMock(spec=["is_loading", "engine"])
        entry.is_loading = False
        entry.engine = engine

        pool = MagicMock(spec=[
            "model_count", "loaded_model_count", "get_loaded_model_ids",
            "current_model_memory", "_entries",
        ])
        pool.model_count = 1
        pool.loaded_model_count = 1
        pool.get_loaded_model_ids.return_value = ["model-a"]
        pool.current_model_memory = 0
        pool._entries = {"model-a": entry}

        self._state.engine_pool = pool

        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_requests"] == 3
        assert data["waiting_requests"] == 2

    def test_requires_auth_when_api_key_set(self, client):
        """The endpoint should require an API key when one is configured."""
        self._state.api_key = "test-secret-key"
        resp = client.get("/api/status")
        assert resp.status_code == 401

        resp = client.get(
            "/api/status",
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert resp.status_code == 200

    def test_serving_metrics_included(self, client):
        """Check that serving metrics from ServerMetrics are present."""
        resp = client.get("/api/status")
        data = resp.json()
        expected_keys = [
            "total_requests", "total_prompt_tokens", "total_completion_tokens",
            "total_cached_tokens", "cache_efficiency",
            "avg_prefill_tps", "avg_generation_tps",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_unlimited_memory_max(self, client):
        """When no enforcer is present, formatted shows 'unlimited'."""
        pool = MagicMock(spec=[
            "model_count", "loaded_model_count", "get_loaded_model_ids",
            "current_model_memory", "_entries",
        ])
        pool.model_count = 0
        pool.loaded_model_count = 0
        pool.get_loaded_model_ids.return_value = []
        pool.current_model_memory = 0
        pool._entries = {}
        self._state.process_memory_enforcer = None

        self._state.engine_pool = pool

        resp = client.get("/api/status")
        data = resp.json()
        assert data["model_memory_max"] is None
        assert data["model_memory_max_formatted"] == "unlimited"


class TestStatusCustomKernels:
    """/api/status reports native custom kernel availability.

    Diagnosability for silently-degraded source installs: without the
    native extensions the affected model families fall back to much slower
    generic paths (issue #2137), and this block makes that detectable by
    external polling instead of only a log line.
    """

    @pytest.fixture(autouse=True)
    def setup_server_state(self):
        state = ServerState()
        with patch("omlx.server._server_state", state):
            yield

    def test_custom_kernels_block_lists_every_package(self, client):
        from omlx.custom_kernels import NATIVE_KERNEL_PACKAGES

        resp = client.get("/api/status")
        assert resp.status_code == 200
        kernels = resp.json()["custom_kernels"]
        assert set(kernels) == set(NATIVE_KERNEL_PACKAGES)
        for report in kernels.values():
            assert set(report) == {"available", "import_error"}
            assert isinstance(report["available"], bool)
            assert report["import_error"] is None or isinstance(
                report["import_error"], str
            )

    def test_available_packages_report_no_import_error(self, client):
        resp = client.get("/api/status")
        for name, report in resp.json()["custom_kernels"].items():
            if report["available"]:
                assert report["import_error"] is None, name
            else:
                assert report["import_error"], name

    def test_native_kernel_status_never_raises_on_broken_package(self):
        from omlx import custom_kernels

        real_import = custom_kernels.importlib.import_module

        def broken_import(name, *args, **kwargs):
            if name.endswith(".fast"):
                raise RuntimeError("simulated native import explosion")
            return real_import(name, *args, **kwargs)

        with patch.object(
            custom_kernels.importlib, "import_module", side_effect=broken_import
        ):
            status = custom_kernels.native_kernel_status()
        for name, report in status.items():
            assert report["available"] is False, name
            assert "simulated native import explosion" in report["import_error"]
