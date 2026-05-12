# SPDX-License-Identifier: Apache-2.0
"""Tests for the admin server-restart route.

Covers the supervisor-gating contract: the endpoint refuses with 503 when
``OMLX_SUPERVISED`` is not set in the environment (plain ``omlx serve``)
and accepts with 202 + schedules a SIGTERM when running under the menu
bar supervisor.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.admin import routes as admin_routes


@pytest.fixture
def client(monkeypatch):
    """Build a TestClient with auth bypassed for the restart route."""
    async def _fake_require_admin():
        return True

    app = FastAPI()
    app.include_router(admin_routes.router)
    app.dependency_overrides[admin_routes.require_admin] = _fake_require_admin
    return TestClient(app)


class TestRestartServerRoute:
    def test_returns_503_when_unsupervised(self, client, monkeypatch):
        """No OMLX_SUPERVISED env var = no supervisor = no respawn path."""
        monkeypatch.delenv("OMLX_SUPERVISED", raising=False)

        r = client.post("/admin/api/server/restart")
        assert r.status_code == 503
        body = r.json()
        assert "detail" in body
        assert "supervisor" in body["detail"].lower()

    def test_returns_202_when_supervised(self, client, monkeypatch):
        """With OMLX_SUPERVISED set, the handler returns 202 immediately.

        ``_schedule_self_terminate`` is replaced with a spy so the test
        process never actually receives SIGTERM. Patching the seam (not
        ``asyncio.get_running_loop``) keeps FastAPI's TestClient portal
        intact.
        """
        monkeypatch.setenv("OMLX_SUPERVISED", "menubar")

        with patch("omlx.admin.routes._schedule_self_terminate") as spy:
            r = client.post("/admin/api/server/restart")

        assert r.status_code == 202, r.text
        body = r.json()
        assert body["status"] == "restarting"
        assert body["supervisor"] == "menubar"
        assert body["expected_downtime_seconds"] > 0
        # The handler must schedule the SIGTERM (not invoke it synchronously)
        # and pass a positive delay so FastAPI can flush the 202 first.
        spy.assert_called_once()
        ((delay,), _kwargs) = spy.call_args
        assert delay > 0

    def test_supervisor_label_round_trips(self, client, monkeypatch):
        """Whatever supervisor identifier is set in env comes back in
        the response — useful for the dashboard and for diagnosing
        which supervisor is responsible for the respawn."""
        monkeypatch.setenv("OMLX_SUPERVISED", "launchd")

        with patch("omlx.admin.routes._schedule_self_terminate"):
            r = client.post("/admin/api/server/restart")

        assert r.status_code == 202
        assert r.json()["supervisor"] == "launchd"

    def test_unsupervised_does_not_schedule_termination(self, client, monkeypatch):
        """503 path must not schedule a SIGTERM — otherwise plain
        ``omlx serve`` instances would die with no respawn after a
        single accidental click against an unsupervised server."""
        monkeypatch.delenv("OMLX_SUPERVISED", raising=False)

        with patch("omlx.admin.routes._schedule_self_terminate") as spy:
            r = client.post("/admin/api/server/restart")

        assert r.status_code == 503
        spy.assert_not_called()
