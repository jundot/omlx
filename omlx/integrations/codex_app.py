# SPDX-License-Identifier: Apache-2.0
"""Codex App (OpenAI Codex App Desktop) integration.

This integration launches the Codex App Desktop GUI/TUI via the
``codex app`` subcommand, while sharing the same config file as
the CLI variant (``~/.codex/config.toml``).

OpenAI renamed the desktop app from Codex to ChatGPT. In-place
upgrades keep the old ``/Applications/Codex.app`` folder name while
fresh installs use ``ChatGPT.app``; the bundle identifier stays
``com.openai.codex`` either way, so detection matches on the
identifier rather than the folder name. The bundle also ships its
own codex CLI binary, which is used as a fallback when no ``codex``
is on PATH (DMG-only installs).

Usage:
    omlx launch codex_app --model qwen3.5

Which launches:
    codex app

Both CLI and App use the same config file:
    ~/.codex/config.toml
"""

from __future__ import annotations

import os
import plistlib
import shutil
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext
from omlx.integrations.codex import CODEX_CONFIG_PATH, write_codex_config
from omlx.utils.install import get_cli_command_prefix

CODEX_APP_BUNDLE_ID = "com.openai.codex"

# Both pre-rename (Codex.app) and post-rename (ChatGPT.app) folder names.
_APP_BUNDLE_NAMES = ("Codex.app", "ChatGPT.app")
_APP_BUNDLE_ROOTS = (Path("/Applications"), Path.home() / "Applications")


def find_codex_app_bundle() -> Path | None:
    """Return the Codex/ChatGPT desktop app bundle path, or None.

    Matches on CFBundleIdentifier so the legacy ChatGPT chat app
    (com.openai.chat) is not mistaken for the renamed Codex app.
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
            if info.get("CFBundleIdentifier") == CODEX_APP_BUNDLE_ID:
                return bundle
    return None


def resolve_codex_binary() -> str | None:
    """Resolve a runnable codex CLI: PATH first, then the app-bundled one."""
    path_bin = shutil.which("codex")
    if path_bin:
        return path_bin
    bundle = find_codex_app_bundle()
    if bundle is not None:
        bundled = bundle / "Contents" / "Resources" / "codex"
        if bundled.is_file() and os.access(bundled, os.X_OK):
            return str(bundled)
    return None


class CodexAppIntegration(Integration):
    """Codex App Desktop integration that configures ~/.codex/config.toml for oMLX."""

    CONFIG_PATH = CODEX_CONFIG_PATH

    def __init__(self):
        super().__init__(
            name="codex_app",
            display_name="Codex App",
            type="config_file",
            install_check="codex",
            install_hint=(
                "Install the ChatGPT (formerly Codex) desktop app, "
                "or: npm install -g @openai/codex"
            ),
        )

    def is_installed(self) -> bool:
        # The desktop app alone counts as installed: its bundle ships a
        # codex CLI that launch() falls back to when none is on PATH.
        return resolve_codex_binary() is not None

    def get_command(self, ctx: IntegrationContext) -> str:
        return (
            f"{get_cli_command_prefix()} "
            f"launch codex_app --model {ctx.model or 'select-a-model'}"
        )

    def configure(self, ctx: IntegrationContext) -> None:
        write_codex_config(self.CONFIG_PATH, ctx)

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()
        env["OMLX_API_KEY"] = ctx.auth_token

        # Launch codex app (desktop GUI/TUI) instead of codex CLI
        # Note: codex app doesn't accept -m flag, model is set in config
        codex_bin = resolve_codex_binary() or "codex"
        args = [codex_bin, "app"]
        args.extend(ctx.extra_args)

        os.execvpe(codex_bin, args, env)
