# SPDX-License-Identifier: Apache-2.0
"""Tests for the Claude Desktop launch integration."""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

from omlx.integrations import INTEGRATIONS, claude_desktop_app, get_integration
from omlx.integrations.base import IntegrationContext
from omlx.integrations.claude_desktop_app import (
    CLAUDE_DESKTOP_BUNDLE_ID,
    ClaudeDesktopAppIntegration,
    find_claude_desktop_bundle,
)


def ctx(**overrides) -> IntegrationContext:
    defaults = {
        "host": "127.0.0.1",
        "port": 8000,
        "api_key": "",
        "model": "",
    }
    defaults.update(overrides)
    return IntegrationContext(**defaults)


def make_bundle(
    root: Path, name: str = "Claude.app", bundle_id: str = CLAUDE_DESKTOP_BUNDLE_ID
) -> Path:
    contents = root / name / "Contents"
    contents.mkdir(parents=True)
    with (contents / "Info.plist").open("wb") as f:
        plistlib.dump({"CFBundleIdentifier": bundle_id}, f)
    return root / name


def _force_darwin(monkeypatch) -> None:
    monkeypatch.setattr(claude_desktop_app.sys, "platform", "darwin")


class TestRegistry:
    def test_registered(self):
        assert "claude_desktop" in INTEGRATIONS
        assert (
            "ClaudeDesktopAppIntegration"
            in __import__("omlx.integrations", fromlist=["__all__"]).__all__
        )

    def test_get_integration(self):
        integ = get_integration("claude_desktop")
        assert isinstance(integ, ClaudeDesktopAppIntegration)
        assert integ.name == "claude_desktop"
        assert integ.display_name == "Claude Desktop"
        assert integ.type == "config_file"


class TestFindBundle:
    def test_finds_bundle(self, tmp_path, monkeypatch):
        bundle = make_bundle(tmp_path)
        monkeypatch.setattr(claude_desktop_app, "_APP_BUNDLE_ROOTS", (tmp_path,))
        assert find_claude_desktop_bundle() == bundle

    def test_wrong_bundle_id_not_matched(self, tmp_path, monkeypatch):
        make_bundle(tmp_path, bundle_id="com.example.other")
        monkeypatch.setattr(claude_desktop_app, "_APP_BUNDLE_ROOTS", (tmp_path,))
        assert find_claude_desktop_bundle() is None

    def test_missing_bundle(self, tmp_path, monkeypatch):
        monkeypatch.setattr(claude_desktop_app, "_APP_BUNDLE_ROOTS", (tmp_path,))
        assert find_claude_desktop_bundle() is None


class TestIsInstalled:
    def test_false_when_bundle_missing(self, tmp_path, monkeypatch):
        _force_darwin(monkeypatch)
        monkeypatch.setattr(claude_desktop_app, "_APP_BUNDLE_ROOTS", (tmp_path,))
        assert ClaudeDesktopAppIntegration().is_installed() is False

    def test_true_when_bundle_present(self, tmp_path, monkeypatch):
        _force_darwin(monkeypatch)
        make_bundle(tmp_path)
        monkeypatch.setattr(claude_desktop_app, "_APP_BUNDLE_ROOTS", (tmp_path,))
        assert ClaudeDesktopAppIntegration().is_installed() is True

    def test_false_off_macos(self, tmp_path, monkeypatch):
        monkeypatch.setattr(claude_desktop_app.sys, "platform", "linux")
        make_bundle(tmp_path)
        monkeypatch.setattr(claude_desktop_app, "_APP_BUNDLE_ROOTS", (tmp_path,))
        assert ClaudeDesktopAppIntegration().is_installed() is False


class TestCommand:
    def test_get_command(self):
        cmd = ClaudeDesktopAppIntegration().get_command(ctx())
        assert cmd == "omlx launch claude_desktop" or cmd.endswith(
            "launch claude_desktop"
        )
        assert "claude_desktop" in cmd

    def test_launch_help_lists_tool(self):
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "launch", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "claude_desktop" in result.stdout


class TestConfigureLaunch:
    def test_configure_delegates_to_gateway(self, monkeypatch):
        captured = {}

        def fake_configure(port, api_key=None, home=None):
            captured["port"] = port
            captured["api_key"] = api_key
            return True

        monkeypatch.setattr(
            claude_desktop_app, "configure_omlx_gateway", fake_configure
        )
        ClaudeDesktopAppIntegration().configure(ctx(port=8123, api_key="secret"))
        assert captured == {"port": 8123, "api_key": "secret"}

    def test_launch_opens_app_when_not_running(self, monkeypatch):
        monkeypatch.setattr(claude_desktop_app.sys, "platform", "darwin")
        monkeypatch.setattr(
            claude_desktop_app, "is_claude_desktop_running", lambda: False
        )
        monkeypatch.setattr(
            claude_desktop_app, "configure_omlx_gateway", lambda **kwargs: True
        )
        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr(claude_desktop_app.subprocess, "run", fake_run)
        ClaudeDesktopAppIntegration().launch(ctx(port=8000, api_key="k"))
        assert captured["args"] == ["open", "-a", "Claude"]

    def test_launch_warns_when_already_running(self, monkeypatch, capsys):
        monkeypatch.setattr(claude_desktop_app.sys, "platform", "darwin")
        monkeypatch.setattr(
            claude_desktop_app, "is_claude_desktop_running", lambda: True
        )
        monkeypatch.setattr(
            claude_desktop_app, "configure_omlx_gateway", lambda **kwargs: True
        )

        def fail_run(*args, **kwargs):
            raise AssertionError("must not open the app when already running")

        monkeypatch.setattr(claude_desktop_app.subprocess, "run", fail_run)
        ClaudeDesktopAppIntegration().launch(ctx(port=8000, api_key="k"))
        out = capsys.readouterr().out
        assert "restart" in out.lower()


class TestDashboard:
    def test_js_getter_present(self):
        js_path = (
            Path(__file__).resolve().parent.parent
            / "omlx"
            / "admin"
            / "static"
            / "js"
            / "dashboard.js"
        )
        content = js_path.read_text(encoding="utf-8")
        assert "claudeDesktopCommand" in content
        assert "_launchCmd('claude_desktop')" in content

    def test_status_template_has_launcher_block(self):
        html_path = (
            Path(__file__).resolve().parent.parent
            / "omlx"
            / "admin"
            / "templates"
            / "dashboard"
            / "_status.html"
        )
        content = html_path.read_text(encoding="utf-8")
        assert 'x-text="claudeDesktopCommand"' in content
        assert "copyToClipboard(claudeDesktopCommand)" in content

    def test_launch_list_includes_tool(self):
        from omlx.integrations import list_integrations

        assert "claude_desktop" in {i.name for i in list_integrations()}
