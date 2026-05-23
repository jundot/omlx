# SPDX-License-Identifier: Apache-2.0
"""Verify SchedulerQueueFullError maps to HTTP 503 + Retry-After in server.py."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.exceptions import SchedulerQueueFullError


def _build_test_app():
    """Build a minimal FastAPI app that re-uses the same exception handler.

    Importing omlx.server would pull in heavy server-state init. We pluck the
    handler function out of the module and register it against a fresh app
    so the test stays fast and free of state.
    """
    import omlx.server as srv

    app = FastAPI()
    app.add_exception_handler(
        SchedulerQueueFullError, srv.scheduler_queue_full_handler
    )

    @app.get("/v1/raise")
    def raise_queue_full():
        raise SchedulerQueueFullError(current_depth=32, max_depth=32)

    @app.get("/health/raise")
    def raise_queue_full_health():
        raise SchedulerQueueFullError(current_depth=33, max_depth=32)

    return app


class TestQueueFullHandler:
    def test_returns_503(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        assert resp.status_code == 503

    def test_has_retry_after_header(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        assert resp.headers.get("Retry-After") == "1"

    def test_api_route_uses_openai_error_body(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/v1/raise")
        body = resp.json()
        # _openai_error_body wraps in {"error": {...}}
        assert "error" in body
        assert "queue full" in body["error"]["message"].lower()
        # Depth numbers surface to the client
        assert "32/32" in body["error"]["message"]

    def test_non_api_route_uses_plain_detail(self):
        with TestClient(_build_test_app()) as client:
            resp = client.get("/health/raise")
        body = resp.json()
        assert "detail" in body
        assert "queue full" in body["detail"].lower()
