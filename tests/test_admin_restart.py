# SPDX-License-Identifier: Apache-2.0
"""Tests for the admin server-restart route.

Covers the supervisor contract: under the menu bar supervisor
(``OMLX_SUPERVISED`` set) the endpoint returns 202 and schedules a plain
SIGTERM for the supervisor to respawn. Without a supervisor it falls back
to spawning its own replacement process before the SIGTERM (#1814
recovery half), unless ``OMLX_NO_SELF_RESPAWN`` opts back into the old
503 refusal.
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
    def test_returns_202_and_respawns_when_unsupervised(self, client, monkeypatch):
        """No OMLX_SUPERVISED env var: self-respawn fallback (#1814) kicks
        in instead of refusing outright, so a lone ``omlx serve`` process
        can still recover from a web-panel restart."""
        monkeypatch.delenv("OMLX_SUPERVISED", raising=False)
        monkeypatch.delenv("OMLX_NO_SELF_RESPAWN", raising=False)

        with patch("omlx.admin.routes._schedule_self_terminate") as spy:
            r = client.post("/admin/api/server/restart")

        assert r.status_code == 202, r.text
        body = r.json()
        assert body["status"] == "restarting"
        assert body["supervisor"] == "self"
        assert body["expected_downtime_seconds"] > 0
        spy.assert_called_once()
        (args, kwargs) = spy.call_args
        assert args[0] > 0
        assert kwargs.get("respawn") is True

    def test_no_self_respawn_kill_switch_returns_503(self, client, monkeypatch):
        """OMLX_NO_SELF_RESPAWN=1 restores the old refuse-outright behavior
        for operators who don't want an unsupervised process replacing
        itself, and must not schedule any termination."""
        monkeypatch.delenv("OMLX_SUPERVISED", raising=False)
        monkeypatch.setenv("OMLX_NO_SELF_RESPAWN", "1")

        with patch("omlx.admin.routes._schedule_self_terminate") as spy:
            r = client.post("/admin/api/server/restart")

        assert r.status_code == 503
        body = r.json()
        assert "detail" in body
        assert "supervisor" in body["detail"].lower()
        spy.assert_not_called()

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

    def test_no_self_respawn_kill_switch_does_not_schedule_termination(
        self, client, monkeypatch
    ):
        """The 503 refusal path (OMLX_NO_SELF_RESPAWN=1) must not schedule
        a SIGTERM — otherwise plain ``omlx serve`` instances would die
        with no respawn after a single accidental click."""
        monkeypatch.delenv("OMLX_SUPERVISED", raising=False)
        monkeypatch.setenv("OMLX_NO_SELF_RESPAWN", "1")

        with patch("omlx.admin.routes._schedule_self_terminate") as spy:
            r = client.post("/admin/api/server/restart")

        assert r.status_code == 503
        spy.assert_not_called()

    def test_supervised_path_does_not_pass_respawn(self, client, monkeypatch):
        """The supervised branch must stay byte-for-byte unchanged: it
        calls ``_schedule_self_terminate`` with only the delay, never
        ``respawn=True`` — the supervisor owns respawn there."""
        monkeypatch.setenv("OMLX_SUPERVISED", "menubar")

        with patch("omlx.admin.routes._schedule_self_terminate") as spy:
            r = client.post("/admin/api/server/restart")

        assert r.status_code == 202
        spy.assert_called_once_with(0.5)


class TestRespawnSelf:
    """Unit tests for the unsupervised-restart replacement-process spawn."""

    def test_spawns_detached_process_with_same_argv(self, monkeypatch):
        import sys as _sys

        monkeypatch.setattr(
            _sys, "argv", ["/opt/venv/bin/omlx", "serve", "--port", "8000"]
        )

        class FakeProc:
            pid = 4242

        captured = {}

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return FakeProc()

        with patch("subprocess.Popen", side_effect=fake_popen):
            pid = admin_routes._respawn_self()

        assert pid == 4242
        assert captured["argv"] == [
            _sys.executable,
            "/opt/venv/bin/omlx",
            "serve",
            "--port",
            "8000",
        ]
        assert captured["kwargs"]["start_new_session"] is True
        assert captured["kwargs"]["stdout"] is not None
        assert captured["kwargs"]["stderr"] is not None

    def test_returns_none_when_spawn_fails(self):
        with patch("subprocess.Popen", side_effect=OSError("no more processes")):
            assert admin_routes._respawn_self() is None


class TestScheduleSelfTerminateRespawn:
    """Confirms `respawn=True` spawns the replacement before signaling self."""

    def test_respawn_true_spawns_before_sigterm(self, monkeypatch):
        import asyncio

        calls: list = []
        monkeypatch.setattr(
            admin_routes, "_respawn_self", lambda: calls.append("respawn") or 1
        )
        monkeypatch.setattr(
            admin_routes.os,
            "kill",
            lambda pid, sig: calls.append(("kill", pid, sig)),
        )

        async def _run():
            admin_routes._schedule_self_terminate(0.01, respawn=True)
            await asyncio.sleep(0.1)

        asyncio.run(_run())

        assert calls == [
            "respawn",
            ("kill", admin_routes.os.getpid(), admin_routes.signal.SIGTERM),
        ]

    def test_respawn_false_skips_replacement(self, monkeypatch):
        import asyncio

        calls: list = []
        monkeypatch.setattr(
            admin_routes, "_respawn_self", lambda: calls.append("respawn")
        )
        monkeypatch.setattr(
            admin_routes.os,
            "kill",
            lambda pid, sig: calls.append(("kill", pid, sig)),
        )

        async def _run():
            admin_routes._schedule_self_terminate(0.01)
            await asyncio.sleep(0.1)

        asyncio.run(_run())

        assert calls == [("kill", admin_routes.os.getpid(), admin_routes.signal.SIGTERM)]
