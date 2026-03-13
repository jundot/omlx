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
            "current_model_memory", "max_model_memory", "_entries",
        ])
        pool.model_count = 5
        pool.loaded_model_count = 2
        pool.get_loaded_model_ids.return_value = ["model-a", "model-b"]
        pool.current_model_memory = 16 * 1024**3
        pool.max_model_memory = 32 * 1024**3

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
            "current_model_memory", "max_model_memory", "_entries",
        ])
        pool.model_count = 1
        pool.loaded_model_count = 1
        pool.get_loaded_model_ids.return_value = ["model-a"]
        pool.current_model_memory = 0
        pool.max_model_memory = None
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
        """When max_model_memory is None, formatted shows 'unlimited'."""
        pool = MagicMock(spec=[
            "model_count", "loaded_model_count", "get_loaded_model_ids",
            "current_model_memory", "max_model_memory", "_entries",
        ])
        pool.model_count = 0
        pool.loaded_model_count = 0
        pool.get_loaded_model_ids.return_value = []
        pool.current_model_memory = 0
        pool.max_model_memory = None
        pool._entries = {}

        self._state.engine_pool = pool

        resp = client.get("/api/status")
        data = resp.json()
        assert data["model_memory_max"] is None
        assert data["model_memory_max_formatted"] == "unlimited"
