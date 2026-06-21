# SPDX-License-Identifier: Apache-2.0
"""Goose Desktop integration.

This integration configures and launches the Goose Desktop app (https://github.com/block/goose)
pointing at oMLX as its OpenAI-compatible API backend.

Goose Desktop uses a configuration file at:
    ~/.config/goose/config.yaml

Key settings:
    api.key   - API key for authentication
    api.url   - Base URL of the OpenAI-compatible API

Usage:
    omlx launch goose_desktop --model qwen3.5

Which launches:
    open -a "Goose"
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import yaml

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_command_prefix


GOOSE_DESKTOP_CONFIG_PATH = Path.home() / ".config" / "goose" / "config.yaml"


def _write_goose_config(config_path: Path, ctx: IntegrationContext) -> None:
    """Write Goose Desktop config pointing to oMLX."""
    existing: dict = {}
    if config_path.exists():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, yaml.YAMLError) as e:
            print(f"Warning: could not parse {config_path}: {e}")
            print("Creating new config file.")

    # Create timestamped backup
    if config_path.exists():
        timestamp = int(time.time())
        backup = config_path.with_suffix(f".{timestamp}.bak")
        try:
            shutil.copy2(config_path, backup)
            print(f"Backup: {backup}")
        except OSError as e:
            print(f"Warning: could not create backup: {e}")

    # Ensure api section exists
    if "api" not in existing or not isinstance(existing.get("api"), dict):
        existing["api"] = {}

    api_config: dict = existing["api"]  # type: ignore[assignment]

    # Point at oMLX's OpenAI-compatible API
    api_config["url"] = ctx.openai_base_url
    api_config["key"] = ctx.auth_token

    # Set default model if provided
    if ctx.model:
        existing["model"] = ctx.model

    config_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_content = yaml.safe_dump(
        existing, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    config_path.write_text(
        yaml_content.rstrip() + "\n",
        encoding="utf-8",
    )
    print(f"Config written: {config_path}")


class GooseDesktopIntegration(Integration):
    """Goose Desktop integration that writes ~/.config/goose/config.yaml."""

    CONFIG_PATH = GOOSE_DESKTOP_CONFIG_PATH

    def __init__(self):
        super().__init__(
            name="goose_desktop",
            display_name="Goose Desktop",
            type="config_file",
            install_check="goose-desktop",  # overridden by is_installed()
            install_hint=(
                "brew install --cask block-goose  "
                "# or download from https://block.github.io/goose"
            ),
        )

    def is_installed(self) -> bool:
        """Check if Goose.app is installed via brew cask."""
        return Path("/Applications/Goose.app").exists()

    def get_command(self, ctx: IntegrationContext) -> str:
        return (
            f"{get_cli_command_prefix()} "
            f"launch goose_desktop --model {ctx.model or 'select-a-model'}"
        )

    def configure(self, ctx: IntegrationContext) -> None:
        _write_goose_config(self.CONFIG_PATH, ctx)

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()

        # Also set env vars in case the desktop app reads them
        env["OPENAI_API_BASE"] = ctx.openai_base_url
        env["OPENAI_API_KEY"] = ctx.auth_token

        if ctx.model:
            env["GOOSE_MODEL"] = ctx.model

        # Disable telemetry for local usage
        env["GOOSE_TELEMETRY"] = "0"

        print(f"Launching Goose Desktop with model {ctx.model}...")
        os.execvpe("open", ["open", "-a", "Goose"], env)
