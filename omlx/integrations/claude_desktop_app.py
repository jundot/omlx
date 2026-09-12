# SPDX-License-Identifier: Apache-2.0
"""Claude Desktop app integration.

Shares the oMLX gateway configuration used by the admin routes
(:mod:`omlx.integrations.claude_desktop`) and launches the macOS
Claude Desktop GUI.

Usage:
    omlx launch claude_desktop

Which configures the gateway profile and then opens the app with
``open -a Claude``. The gateway profile only takes effect after the
app restarts, so when Claude Desktop is already running launch prints
a reminder to restart it.
"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext
from omlx.integrations.claude_desktop import configure_omlx_gateway
from omlx.utils.install import get_cli_command_prefix

CLAUDE_DESKTOP_BUNDLE_ID = "com.anthropic.claudefordesktop"

_APP_BUNDLE_NAMES = ("Claude.app",)
_APP_BUNDLE_ROOTS = (Path("/Applications"), Path.home() / "Applications")


def find_claude_desktop_bundle() -> Path | None:
    """Return the Claude Desktop app bundle path, or None.

    Matches on CFBundleIdentifier rather than the folder name.
    """
    for root in _APP_BUNDLE_ROOTS:
        for name in _APP_BUNDLE_NAMES:
            bundle = root / name
            plist_path = bundle / "Contents" / "Info.plist"
            if not plist_path.is_file():
                continue
            try:
                with plist_path.open("rb") as f:
                    info = plistlib.load(f)
            except (OSError, plistlib.InvalidFileException):
                continue
            if info.get("CFBundleIdentifier") == CLAUDE_DESKTOP_BUNDLE_ID:
                return bundle
    return None


def is_claude_desktop_running() -> bool:
    """Return True when the Claude Desktop app process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-x", "Claude"],
            check=False,
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class ClaudeDesktopAppIntegration(Integration):
    """Claude Desktop integration that configures the oMLX gateway profile."""

    def __init__(self):
        super().__init__(
            name="claude_desktop",
            display_name="Claude Desktop",
            type="config_file",
            install_check="Claude",
            install_hint=(
                "Install Claude Desktop for macOS from https://claude.ai/download"
            ),
        )

    def is_installed(self) -> bool:
        if sys.platform != "darwin":
            return False
        return find_claude_desktop_bundle() is not None

    def get_command(self, ctx: IntegrationContext) -> str:
        return f"{get_cli_command_prefix()} launch claude_desktop"

    def configure(self, ctx: IntegrationContext) -> None:
        configure_omlx_gateway(port=ctx.port, api_key=ctx.api_key or "omlx")

    def launch(self, ctx: IntegrationContext) -> None:
        if sys.platform != "darwin":
            print("Claude Desktop is only available on macOS.")
            return
        self.configure(ctx)
        if is_claude_desktop_running():
            print(
                "Claude Desktop is already running: restart the app "
                "for the oMLX gateway configuration to take effect."
            )
            return
        subprocess.run(
            ["open", "-a", "Claude"],
            check=False,
            timeout=15,
        )


__all__ = [
    "CLAUDE_DESKTOP_BUNDLE_ID",
    "ClaudeDesktopAppIntegration",
    "find_claude_desktop_bundle",
    "is_claude_desktop_running",
]
