# SPDX-License-Identifier: Apache-2.0
"""Tests for the token usage ledger.

Covers per-session ledger persistence (append on close, JSONL cap),
lifetime aggregation via the existing ServerMetrics singleton, cloud API
cost estimation, and the /admin/api/usage/* routes.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import routes as admin_routes
from omlx.server_metrics import ServerMetrics, get_server_metrics, reset_server_metrics
from omlx.usage_ledger import (
    MODEL_PRICING,
    UsageLedger,
    close_current_session,
    estimate_api_cost,
    estimate_per_model_cost,
    get_usage_ledger,
    reset_usage_ledger,
)


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Isolate the ServerMetrics / UsageLedger singletons between tests.

    Both are process-wide globals shared with every other test module, so
    each test must start (and leave) them pointing at no persistence path.
    """
    reset_server_metrics()
    reset_usage_ledger()
    yield
    reset_server_metrics()
    reset_usage_ledger()


@pytest.fixture
def client():
    """Build a TestClient with auth bypassed, mirroring test_admin_restart.py."""

    async def _fake_require_admin():
        return True

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    return TestClient(app)


class TestCostEstimation:
    """Tests for estimate_api_cost / estimate_per_model_cost."""

    def test_known_model_cost_math(self):
        """1M prompt + 1M completion tokens on gpt-4o = $2.50 + $10.00."""
        cost = estimate_api_cost(
            "gpt-4o", prompt_tokens=1_000_000, completion_tokens=1_000_000
        )
        assert cost == pytest.approx(2.50 + 10.00)

    def test_known_model_partial_tokens(self):
        """A real served model_id string (with a date suffix) still matches."""
        cost = estimate_api_cost(
            "claude-3-5-sonnet-20241022",
            prompt_tokens=500_000,
            completion_tokens=100_000,
        )
        expected = 500_000 / 1_000_000 * 3.00 + 100_000 / 1_000_000 * 15.00
        assert cost == pytest.approx(expected)

    def test_unknown_model_returns_none_not_zero(self):
        """A local MLX model isn't in the pricing table -> None, never $0."""
        assert estimate_api_cost("mlx-community/my-local-model-4bit", 1000, 1000) is None

    def test_empty_model_id_returns_none(self):
        assert estimate_api_cost("", 1000, 1000) is None

    def test_longest_key_match_wins(self):
        """'gpt-4o-mini' must not be shadowed by the shorter 'gpt-4o' key."""
        cost = estimate_api_cost("gpt-4o-mini-2024-07-18", 1_000_000, 0)
        assert cost == pytest.approx(MODEL_PRICING["gpt-4o-mini"]["input"])
        assert cost != pytest.approx(MODEL_PRICING["gpt-4o"]["input"])

    def test_per_model_cost_sums_known_and_skips_unknown(self):
        per_model = {
            "gpt-4o": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
            "local-mlx-model": {"prompt_tokens": 1_000_000, "completion_tokens": 0},
        }
        cost = estimate_per_model_cost(per_model)
        assert cost == pytest.approx(2.50)

    def test_per_model_cost_all_unknown_is_none(self):
        per_model = {"local-model-a": {"prompt_tokens": 100, "completion_tokens": 50}}
        assert estimate_per_model_cost(per_model) is None

    def test_per_model_cost_empty_is_none(self):
        assert estimate_per_model_cost({}) is None


class TestSessionClose:
    """Tests for closing a ServerMetrics session into the ledger."""

    def test_close_current_session_appends_record(self, tmp_path):
        ledger_path = tmp_path / "usage_sessions.jsonl"
        reset_usage_ledger(ledger_path)

        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100, completion_tokens=50, model_id="gpt-4o"
        )
        close_current_session(metrics)

        assert ledger_path.exists()
        lines = ledger_path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["session_id"] == metrics.session_id
        assert record["requests"] == 1
        assert record["prompt_tokens"] == 100
        assert record["completion_tokens"] == 50
        assert record["per_model"]["gpt-4o"]["prompt_tokens"] == 100
        assert record["ended_at"] >= record["started_at"]

    def test_close_current_session_skips_empty_session(self, tmp_path):
        """A restart before any traffic must not pollute the ledger."""
        ledger_path = tmp_path / "usage_sessions.jsonl"
        reset_usage_ledger(ledger_path)

        metrics = ServerMetrics()
        close_current_session(metrics)

        assert not ledger_path.exists()

    def test_reset_server_metrics_closes_previous_session(self, tmp_path):
        stats_path = tmp_path / "stats.json"
        ledger_path = tmp_path / "usage_sessions.jsonl"

        reset_server_metrics(stats_path=stats_path, ledger_path=ledger_path)
        m1 = get_server_metrics()
        m1.record_request_complete(
            prompt_tokens=200, completion_tokens=80, model_id="claude-3-5-sonnet"
        )

        # Simulated restart: closes m1's session into the ledger and opens a
        # fresh session.
        reset_server_metrics(stats_path=stats_path, ledger_path=ledger_path)
        m2 = get_server_metrics()
        assert m2 is not m1
        assert m2.total_requests == 0

        sessions = get_usage_ledger().load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == m1.session_id
        assert sessions[0]["prompt_tokens"] == 200
        assert sessions[0]["ended_at"] is not None


class TestLifetimeAggregation:
    """Tests for per-model breakdown and cross-session accumulation."""

    def test_get_per_model_breakdown_session_and_alltime(self):
        metrics = ServerMetrics()
        metrics.record_request_complete(
            prompt_tokens=100, completion_tokens=50, model_id="gpt-4o"
        )
        metrics.record_request_complete(
            prompt_tokens=200, completion_tokens=80, model_id="gpt-4o"
        )

        session_bd = metrics.get_per_model_breakdown(scope="session")
        alltime_bd = metrics.get_per_model_breakdown(scope="alltime")
        assert session_bd["gpt-4o"]["prompt_tokens"] == 300
        assert alltime_bd["gpt-4o"]["prompt_tokens"] == 300

    def test_lifetime_accumulates_across_sessions(self, tmp_path):
        stats_path = tmp_path / "stats.json"
        ledger_path = tmp_path / "usage_sessions.jsonl"

        reset_server_metrics(stats_path=stats_path, ledger_path=ledger_path)
        get_server_metrics().record_request_complete(
            prompt_tokens=100, completion_tokens=50, model_id="gpt-4o"
        )

        reset_server_metrics(stats_path=stats_path, ledger_path=ledger_path)
        get_server_metrics().record_request_complete(
            prompt_tokens=200, completion_tokens=80, model_id="gpt-4o"
        )

        alltime = get_server_metrics().get_snapshot(scope="alltime")
        assert alltime["total_prompt_tokens"] == 300
        assert alltime["total_completion_tokens"] == 130

        breakdown = get_server_metrics().get_per_model_breakdown(scope="alltime")
        cost = estimate_per_model_cost(breakdown)
        assert cost == pytest.approx(estimate_api_cost("gpt-4o", 300, 130))

        # The two closed-out sessions plus... only one restart happened
        # here, so only the first session (100/50) is in the ledger; the
        # second is still the live session.
        sessions = get_usage_ledger().load_sessions()
        assert len(sessions) == 1
        assert sessions[0]["prompt_tokens"] == 100


class TestLedgerCap:
    """Tests for the JSONL size cap."""

    def test_cap_keeps_last_n_sessions(self, tmp_path):
        ledger_path = tmp_path / "usage_sessions.jsonl"
        ledger = UsageLedger(ledger_path=ledger_path, max_sessions=5)

        for i in range(8):
            ledger.record_session_close(
                {
                    "session_id": f"s{i}",
                    "started_at": float(i),
                    "ended_at": float(i + 1),
                    "requests": 1,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "per_model": {},
                }
            )

        lines = ledger_path.read_text().splitlines()
        assert len(lines) == 5

        sessions = ledger.load_sessions()
        assert [s["session_id"] for s in sessions] == ["s7", "s6", "s5", "s4", "s3"]


class TestUsageEndpoints:
    """Tests for GET /admin/api/usage/sessions and /admin/api/usage/lifetime."""

    def test_sessions_endpoint_includes_live_and_closed(self, client, tmp_path):
        stats_path = tmp_path / "stats.json"
        ledger_path = tmp_path / "usage_sessions.jsonl"

        reset_server_metrics(stats_path=stats_path, ledger_path=ledger_path)
        get_server_metrics().record_request_complete(
            prompt_tokens=100, completion_tokens=50, model_id="gpt-4o"
        )
        # Restart: closes the first session into the ledger.
        reset_server_metrics(stats_path=stats_path, ledger_path=ledger_path)
        get_server_metrics().record_request_complete(
            prompt_tokens=10, completion_tokens=5, model_id="gpt-4o"
        )

        resp = client.get("/admin/api/usage/sessions")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) == 2

        live, closed = sessions[0], sessions[1]
        for entry in (live, closed):
            for field in (
                "session_id",
                "started_at",
                "ended_at",
                "requests",
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "per_model",
                "estimated_api_cost_usd",
            ):
                assert field in entry

        assert live["ended_at"] is None
        assert live["prompt_tokens"] == 10
        assert closed["ended_at"] is not None
        assert closed["prompt_tokens"] == 100
        assert closed["estimated_api_cost_usd"] == pytest.approx(
            estimate_api_cost("gpt-4o", 100, 50)
        )

    def test_sessions_endpoint_scope_session_returns_only_live(self, client, tmp_path):
        stats_path = tmp_path / "stats.json"
        ledger_path = tmp_path / "usage_sessions.jsonl"

        reset_server_metrics(stats_path=stats_path, ledger_path=ledger_path)
        get_server_metrics().record_request_complete(
            prompt_tokens=100, completion_tokens=50
        )
        reset_server_metrics(stats_path=stats_path, ledger_path=ledger_path)

        resp = client.get("/admin/api/usage/sessions?scope=session")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["ended_at"] is None

    def test_lifetime_endpoint_unknown_model_cost_is_none(self, client):
        get_server_metrics().record_request_complete(
            prompt_tokens=100, completion_tokens=50, model_id="local-mlx-model"
        )

        resp = client.get("/admin/api/usage/lifetime?scope=session")
        assert resp.status_code == 200
        body = resp.json()
        assert body["estimated_api_cost_usd"] is None
        assert len(body["per_model"]) == 1
        assert body["per_model"][0]["model_id"] == "local-mlx-model"
        assert body["per_model"][0]["estimated_api_cost_usd"] is None

    def test_lifetime_endpoint_known_model_cost(self, client):
        get_server_metrics().record_request_complete(
            prompt_tokens=1_000_000, completion_tokens=1_000_000, model_id="gpt-4o"
        )

        resp = client.get("/admin/api/usage/lifetime?scope=session")
        assert resp.status_code == 200
        body = resp.json()
        assert body["estimated_api_cost_usd"] == pytest.approx(12.50)
        assert body["total_prompt_tokens"] == 1_000_000
        assert body["scope"] == "session"

    def test_lifetime_endpoint_alltime_scope(self, client, tmp_path):
        stats_path = tmp_path / "stats.json"
        reset_server_metrics(stats_path=stats_path)
        get_server_metrics().record_request_complete(
            prompt_tokens=500, completion_tokens=200, model_id="gpt-4o"
        )

        resp = client.get("/admin/api/usage/lifetime?scope=alltime")
        assert resp.status_code == 200
        body = resp.json()
        assert body["scope"] == "alltime"
        assert body["total_prompt_tokens"] == 500
        assert body["total_completion_tokens"] == 200
