# SPDX-License-Identifier: Apache-2.0
"""Route-level tests for external endpoint benchmark requests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import routes as admin_routes


class _FakeEntry:
    def __init__(self, model_type="llm"):
        self.model_type = model_type
        self.model_path = "/fake"


class _FakePool:
    def __init__(self):
        self._entries = {"local-model": _FakeEntry()}

    def get_entry(self, model_id):
        return self._entries.get(model_id)


EXTERNAL = {
    "base_url": "http://localhost:8001/v1",
    "api_key": "sk-test",
    "model": "remote-model",
}


@pytest.fixture
def client(monkeypatch):
    pool = _FakePool()
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)

    async def _fake_require_admin():
        return True

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clean_bench_state():
    """Keep module-level run/queue state isolated between tests.

    Cleared on both sides of the test: other test modules leak runs into
    _benchmark_runs (a leftover status=="running" run makes /start 409).
    """
    from omlx.admin import accuracy_benchmark, benchmark

    benchmark._benchmark_runs.clear()
    accuracy_benchmark._queue.clear()
    yield
    benchmark._benchmark_runs.clear()
    accuracy_benchmark._queue.clear()


class TestExternalThroughputRoutes:
    def test_start_external_skips_local_model_validation(self, client):
        with patch("omlx.admin.benchmark.run_benchmark", AsyncMock()):
            r = client.post(
                "/admin/api/bench/start",
                json={
                    "model_id": "remote-model",
                    "prompt_lengths": [1024],
                    "batch_sizes": [],
                    "external": EXTERNAL,
                },
            )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "started"

    def test_start_local_unknown_model_still_404(self, client):
        r = client.post(
            "/admin/api/bench/start",
            json={
                "model_id": "remote-model",
                "prompt_lengths": [1024],
                "batch_sizes": [],
            },
        )
        assert r.status_code == 404

    def test_active_reports_external_flag(self, client):
        with patch("omlx.admin.benchmark.run_benchmark", AsyncMock()):
            r = client.post(
                "/admin/api/bench/start",
                json={
                    "model_id": "remote-model",
                    "prompt_lengths": [1024],
                    "external": EXTERNAL,
                },
            )
        assert r.status_code == 200, r.text

        active = client.get("/admin/api/bench/active").json()
        assert active["running"] is True
        assert active["external"] is True
        assert active["model_id"] == "remote-model"
        # The endpoint must never leak connection details
        assert "base_url" not in active
        assert "api_key" not in active

    def test_active_reports_external_false_for_local(self, client):
        with patch("omlx.admin.benchmark.run_benchmark", AsyncMock()):
            r = client.post(
                "/admin/api/bench/start",
                json={"model_id": "local-model", "prompt_lengths": [1024]},
            )
        assert r.status_code == 200, r.text

        active = client.get("/admin/api/bench/active").json()
        assert active["external"] is False


class TestExternalAccuracyRoutes:
    def test_queue_add_external_skips_local_model_validation(self, client):
        with patch(
            "omlx.admin.accuracy_benchmark.start_next_from_queue",
            MagicMock(return_value=None),
        ):
            r = client.post(
                "/admin/api/bench/accuracy/queue/add",
                json={
                    "model_id": "remote-model",
                    "benchmarks": {"mmlu": 30},
                    "batch_size": 4,
                    "external": EXTERNAL,
                },
            )
        assert r.status_code == 200, r.text
        queue = r.json()["queue"]
        assert queue[-1]["model_id"] == "remote-model"
        assert queue[-1]["external"] is True

    def test_queue_add_local_unknown_model_still_404(self, client):
        r = client.post(
            "/admin/api/bench/accuracy/queue/add",
            json={
                "model_id": "remote-model",
                "benchmarks": {"mmlu": 30},
            },
        )
        assert r.status_code == 404
