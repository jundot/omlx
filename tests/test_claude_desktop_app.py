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
        assert ClaudeDesktopAppIntegration().requires_model is False

    def test_other_integrations_require_model(self):
        assert ClaudeCodeIntegration().requires_model is True
        assert OpenCodeIntegration().requires_model is True


class TestClaudeDesktopLaunch:
    def test_no_picker_with_multiple_models(self, capsys):
        """Multiple models available: select_model must not be called."""
        integration = _mock_integration("Claude Desktop")
        integration.requires_model = False
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
        assert "via oMLX gateway" in out
        assert "with model" not in out

    def test_disabled_flag_yes_enables_via_admin_api(self, capsys):
        """Answering Y enables the flag through the admin API, not a file save."""
        integration = ClaudeDesktopAppIntegration()
        settings, save = _make_settings(desktop_enabled=False)
        calls: list = []
        fake_session = _FakeAdminSession(calls)

        with (
            patch.object(integration, "is_installed", return_value=True),
            patch(
                "omlx.integrations.claude_desktop_app.configure_omlx_gateway",
                return_value=True,
            ) as gateway,
            patch(
                "omlx.integrations.claude_desktop_app.is_claude_desktop_running",
                return_value=True,
            ),
            patch("sys.platform", "darwin"),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
            patch("requests.Session", return_value=fake_session),
            patch("requests.get", side_effect=[_health_response(), _status_response()]),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(_make_args())

        assert settings.claude_code.desktop_enabled is True
        save.assert_not_called()
        assert calls[0][0].endswith("/admin/api/login")
        assert calls[0][1] == {"api_key": "test-key"}
        assert calls[1][0].endswith("/admin/api/global-settings")
        assert calls[1][1] == {"claude_code_desktop_enabled": True}
        assert "restart_desktop" not in calls[1][1]
        gateway.assert_called_once()
        out = capsys.readouterr().out
        assert "Tier alias models are now available" in out
        assert "Restart the oMLX server" not in out
        assert "saved to settings.json" not in out

    def test_disabled_flag_yes_never_saves_file(self, capsys):
        """GlobalSettings.save must not be called on the Y path."""
        integration = _mock_integration("Claude Desktop")
        integration.requires_model = False
        settings, save = _make_settings(desktop_enabled=False)
        calls: list = []

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
            patch(
                "requests.Session",
                return_value=_FakeAdminSession(calls),
            ),
            patch(
                "requests.get", side_effect=[_health_response(), _status_response()]
            ),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
            patch(
                "omlx.settings.GlobalSettings.save",
                side_effect=AssertionError("must not save to file"),
            ),
        ):
            launch_command(_make_args())

        save.assert_not_called()
        assert len(calls) == 2
        capsys.readouterr()

    def test_disabled_flag_no_exits_cleanly_without_launch(self, capsys):
        """Answering N exits 0 without configuring or launching anything."""
        integration = _mock_integration("Claude Desktop")
        integration.requires_model = False
        settings, save = _make_settings(desktop_enabled=False)
        requests_mock = MagicMock()
        session_factory = MagicMock()

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="n"),
            patch("requests.get", requests_mock),
            patch("requests.Session", session_factory),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(_make_args())

        save.assert_not_called()
        integration.launch.assert_not_called()
        requests_mock.assert_not_called()
        session_factory.assert_not_called()
        assert "nothing to launch" in capsys.readouterr().out

    def test_login_401_exits_1_without_configure(self, capsys):
        """A rejected API key aborts before configuring or launching."""
        integration = ClaudeDesktopAppIntegration()
        settings, save = _make_settings(desktop_enabled=False)
        calls: list = []
        fake_session = _FakeAdminSession(
            calls,
            login_resp=_FakeApiResponse(401, {"detail": "Invalid API key"}),
        )

        with (
            patch.object(integration, "is_installed", return_value=True),
            patch(
                "omlx.integrations.claude_desktop_app.configure_omlx_gateway",
                return_value=True,
            ) as gateway,
            patch("sys.platform", "darwin"),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
            patch("requests.Session", return_value=fake_session),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
            pytest.raises(SystemExit) as exc,
        ):
            launch_command(_make_args())

        assert exc.value.code == 1
        save.assert_not_called()
        gateway.assert_not_called()
        assert any(url.endswith("/admin/api/login") for url, _ in calls)
        assert not any(
            url.endswith("/admin/api/global-settings") for url, _ in calls
        )
        assert "rejected (401)" in capsys.readouterr().out

    def test_empty_api_key_unauthenticated_401_exits_1(self, capsys):
        """Without a local key the settings POST is tried cookieless; 401 aborts."""
        integration = ClaudeDesktopAppIntegration()
        settings, save = _make_settings(desktop_enabled=False, api_key="")
        calls: list = []
        fake_session = _FakeAdminSession(
            calls,
            settings_resp=_FakeApiResponse(401, {"detail": "Unauthorized"}),
        )

        with (
            patch.object(integration, "is_installed", return_value=True),
            patch(
                "omlx.integrations.claude_desktop_app.configure_omlx_gateway",
                return_value=True,
            ) as gateway,
            patch("sys.platform", "darwin"),
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
            patch("requests.Session", return_value=fake_session),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
            pytest.raises(SystemExit) as exc,
        ):
            launch_command(_make_args())

        assert exc.value.code == 1
        save.assert_not_called()
        gateway.assert_not_called()
        assert not any(url.endswith("/admin/api/login") for url, _ in calls)
        assert any(
            url.endswith("/admin/api/global-settings") for url, _ in calls
        )
        assert "not authorized (401)" in capsys.readouterr().out

    def test_server_unreachable_exits_1_without_save(self, capsys):
        """A network error aborts without a silent file fallback."""
        integration = _mock_integration("Claude Desktop")
        integration.requires_model = False
        settings, save = _make_settings(desktop_enabled=False)

        session = MagicMock()
        session.post.side_effect = ConnectionError("refused")

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
            patch("requests.Session", return_value=session),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
            pytest.raises(SystemExit) as exc,
        ):
            launch_command(_make_args())

        assert exc.value.code == 1
        save.assert_not_called()
        integration.launch.assert_not_called()
        assert "Could not reach oMLX server" in capsys.readouterr().out

    def test_enabled_flag_never_prompts(self):
        """With the switch on, input must not be consulted."""
        integration = _mock_integration("Claude Desktop")
        integration.requires_model = False
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
        integration.requires_model = False
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
        integration.requires_model = True
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
