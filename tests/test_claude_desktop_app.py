# SPDX-License-Identifier: Apache-2.0
"""Tests for `omlx launch claude_desktop`.

Covers the launch path: no model picker (tier aliases resolve server-side),
the on-demand desktop-mode switch, and the gateway launch message.
"""

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omlx.cli import launch_command
from omlx.integrations.claude import ClaudeCodeIntegration
from omlx.integrations.claude_desktop_app import ClaudeDesktopAppIntegration
from omlx.integrations.opencode import OpenCodeIntegration


def _make_settings(desktop_enabled=True):
    """Build stub settings with a save mock; returns (settings, save_mock)."""
    save = MagicMock()
    settings = SimpleNamespace(
        server=SimpleNamespace(host="127.0.0.1", port=8000),
        auth=SimpleNamespace(api_key="test-key"),
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

    def test_disabled_flag_yes_enables_saves_and_configures(self, capsys):
        """Answering Y persists the flag and proceeds to configure the gateway."""
        integration = ClaudeDesktopAppIntegration()
        settings, save = _make_settings(desktop_enabled=False)

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
            patch("requests.get", side_effect=[_health_response(), _status_response()]),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(_make_args())

        assert settings.claude_code.desktop_enabled is True
        save.assert_called_once()
        gateway.assert_called_once()
        out = capsys.readouterr().out
        assert "saved to settings.json" in out
        assert "restart the app" in out

    def test_disabled_flag_no_exits_cleanly_without_launch(self, capsys):
        """Answering N exits 0 without configuring or launching anything."""
        integration = _mock_integration("Claude Desktop")
        integration.requires_model = False
        settings, save = _make_settings(desktop_enabled=False)
        requests_mock = MagicMock()

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="n"),
            patch("requests.get", requests_mock),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(_make_args())

        save.assert_not_called()
        integration.launch.assert_not_called()
        requests_mock.assert_not_called()
        assert "nothing to launch" in capsys.readouterr().out

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
