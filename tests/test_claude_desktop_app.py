# SPDX-License-Identifier: Apache-2.0
"""Tests for `omlx launch claude_desktop`.

Covers the launch path: no model picker (tier aliases resolve server-side),
the on-demand desktop-mode switch via the admin API, and the gateway launch
message.
"""

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omlx.cli import launch_command
from omlx.integrations.claude import ClaudeCodeIntegration
from omlx.integrations.claude_desktop_app import ClaudeDesktopAppIntegration
from omlx.integrations.opencode import OpenCodeIntegration


def _make_settings(desktop_enabled=True, api_key="test-key"):
    """Build stub settings with a save mock; returns (settings, save_mock)."""
    save = MagicMock()
    settings = SimpleNamespace(
        server=SimpleNamespace(host="127.0.0.1", port=8000),
        auth=SimpleNamespace(api_key=api_key),
        integrations=SimpleNamespace(claude_desktop_model=None),
        claude_code=SimpleNamespace(
            desktop_enabled=desktop_enabled,
            opus_model=None,
            sonnet_model=None,
            haiku_model=None,
        ),
        save=save,
    )
    return settings, save


def _make_args(tool="claude_desktop"):
    return argparse.Namespace(
        tool=tool,
        host=None,
        port=None,
        api_key=None,
        model=None,
        tools_profile="coding",
        opus_model=None,
        sonnet_model=None,
        haiku_model=None,
    )


def _health_response():
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    return resp


def _status_response():
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = {"models": []}
    return resp


def _models_response(*model_ids):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "data": [{"id": m, "model_type": "llm"} for m in model_ids]
    }
    return resp


class _FakeApiResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload or {}

    def json(self):
        return self._payload


class _FakeAdminSession:
    """Fake requests.Session recording POST URL/body pairs."""

    def __init__(self, calls, login_resp=None, settings_resp=None):
        self._calls = calls
        self._login_resp = login_resp or _FakeApiResponse(200, {"success": True})
        self._settings_resp = settings_resp or _FakeApiResponse(
            200, {"success": True}
        )

    def post(self, url, json=None, timeout=None):
        self._calls.append((url, json))
        if url.endswith("/admin/api/login"):
            return self._login_resp
        if url.endswith("/admin/api/global-settings"):
            return self._settings_resp
        return _FakeApiResponse(404, {})


def _run_launch(args, settings, integration, requests_mock):
    with (
        patch("requests.get", side_effect=requests_mock),
        patch("omlx.integrations.get_integration", return_value=integration),
        patch("omlx.settings.GlobalSettings.load", return_value=settings),
    ):
        launch_command(args)


def _mock_integration(display_name):
    integration = MagicMock()
    integration.display_name = display_name
    integration.is_installed.return_value = True
    return integration


class TestRequiresModelFlag:
    def test_claude_desktop_skips_model(self):
        assert ClaudeDesktopAppIntegration().requires_model_selection is False

    def test_other_integrations_require_model(self):
        assert ClaudeCodeIntegration().requires_model_selection is True
        assert OpenCodeIntegration().requires_model_selection is True


class TestClaudeDesktopLaunch:
    def test_no_picker_with_multiple_models(self, capsys):
        """Multiple models available: select_model must not be called."""
        integration = _mock_integration("Claude Desktop")
        integration.requires_model_selection = False
        settings, _ = _make_settings(desktop_enabled=True)

        _run_launch(
            _make_args(),
            settings,
            integration,
            [
                _health_response(),
                _status_response(),
                _models_response("model-a", "model-b"),
            ],
        )

        integration.select_model.assert_not_called()
        integration.launch.assert_called_once()
        ctx = integration.launch.call_args.args[0]
        assert ctx.model == ""
        out = capsys.readouterr().out
        assert "Launching Claude Desktop..." in out
        assert "with model" not in out

    @pytest.mark.parametrize("is_tty", [True, False])
    def test_disabled_exits_with_dashboard_guidance(self, capsys, is_tty):
        """Disabled switch exits 1 with dashboard guidance, zero admin HTTP.

        Replaces the removed interactive login flow (Y/n prompt + /admin/api
        login + /admin/api/global-settings): the gate now fires before the
        health check, prints dashboard guidance, and attempts no HTTP.
        """
        integration = _mock_integration("Claude Desktop")
        integration.requires_model_selection = False
        settings, save = _make_settings(desktop_enabled=False)
        session_factory = MagicMock()
        requests_get = MagicMock()

        def _fail_prompt(_prompt=""):
            raise AssertionError("must not prompt: interactive flow removed")

        with (
            patch("sys.stdin.isatty", return_value=is_tty),
            patch("builtins.input", side_effect=_fail_prompt),
            patch("requests.get", requests_get),
            patch("requests.Session", session_factory),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
            pytest.raises(SystemExit) as exc,
        ):
            launch_command(_make_args())

        assert exc.value.code == 1
        save.assert_not_called()
        integration.launch.assert_not_called()
        # No admin HTTP attempted: neither the old Session POSTs nor even
        # the health-check GET (the gate fires first).
        session_factory.assert_not_called()
        requests_get.assert_not_called()
        out = capsys.readouterr().out
        assert "not enabled" in out
        assert "dashboard" in out
        assert "Integrations → Claude Desktop" in out

    def test_server_unreachable_exits_1_without_save(self, capsys):
        """With desktop enabled, an unreachable server aborts at health check."""
        integration = _mock_integration("Claude Desktop")
        integration.requires_model_selection = False
        settings, save = _make_settings(desktop_enabled=True)

        session_factory = MagicMock()

        with (
            patch(
                "requests.get",
                side_effect=ConnectionError("refused"),
            ),
            patch("requests.Session", session_factory),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
            pytest.raises(SystemExit) as exc,
        ):
            launch_command(_make_args())

        assert exc.value.code == 1
        save.assert_not_called()
        integration.launch.assert_not_called()
        session_factory.assert_not_called()
        out = capsys.readouterr().out
        assert "oMLX server is not running" in out
        assert "Start the server first" in out

    def test_enabled_flag_never_prompts(self):
        """With the switch on, input must not be consulted."""
        integration = _mock_integration("Claude Desktop")
        integration.requires_model_selection = False
        settings, save = _make_settings(desktop_enabled=True)

        def _fail_prompt(_prompt=""):
            raise AssertionError("must not prompt when desktop mode is enabled")

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", side_effect=_fail_prompt),
            patch(
                "requests.get", side_effect=[_health_response(), _status_response()]
            ),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(_make_args())

        integration.launch.assert_called_once()
        save.assert_not_called()

    def test_non_interactive_exits_1_with_instructions(self, capsys):
        """Without a TTY there is no prompt: exit 1 and explain how to enable."""
        integration = _mock_integration("Claude Desktop")
        integration.requires_model_selection = False
        settings, save = _make_settings(desktop_enabled=False)

        with (
            patch("sys.stdin.isatty", return_value=False),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
            pytest.raises(SystemExit) as exc,
        ):
            launch_command(_make_args())

        assert exc.value.code == 1
        save.assert_not_called()
        integration.launch.assert_not_called()
        out = capsys.readouterr().out
        assert "Integrations → Claude Desktop" in out


class TestOtherIntegrationsKeepPicker:
    def test_picker_still_shown(self, capsys):
        """Regression guard: model-based integrations still use select_model."""
        integration = _mock_integration("OpenCode")
        integration.requires_model_selection = True
        integration.select_model.return_value = "picked-model"
        settings = SimpleNamespace(
            server=SimpleNamespace(host="127.0.0.1", port=8000),
            auth=SimpleNamespace(api_key="test-key"),
        )

        _run_launch(
            _make_args(tool="opencode"),
            settings,
            integration,
            [
                _health_response(),
                _status_response(),
                _models_response("picked-model", "other-model"),
            ],
        )

        integration.select_model.assert_called_once()
        ctx = integration.launch.call_args.args[0]
        assert ctx.model == "picked-model"
        assert "with model picked-model" in capsys.readouterr().out
