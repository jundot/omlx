# SPDX-License-Identifier: Apache-2.0
"""Codex App (OpenAI Codex App Desktop) integration.

This integration launches the Codex App Desktop GUI/TUI via the
``codex app`` subcommand, while sharing the same config file as
the CLI variant (``~/.codex/config.toml``).

Usage:
    omlx launch codex_app --model qwen3.5

Which launches:
    codex app

Both CLI and App use the same config file:
    ~/.codex/config.toml
"""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_command_prefix


class CodexAppIntegration(Integration):
    """Codex App Desktop integration that configures ~/.codex/config.toml for oMLX."""

    CONFIG_PATH = Path.home() / ".codex" / "config.toml"

    def __init__(self):
        super().__init__(
            name="codex_app",
            display_name="Codex App",
            type="config_file",
            install_check="codex",
            install_hint="npm install -g @openai/codex",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        return (
            f"{get_cli_command_prefix()} "
            f"launch codex_app --model {ctx.model or 'select-a-model'}"
        )

    def configure(self, ctx: IntegrationContext) -> None:
        config_path = self.CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)

        existing_content = ""
        if config_path.exists():
            # Create backup
            timestamp = int(time.time())
            backup = config_path.with_suffix(f".{timestamp}.bak")
            try:
                shutil.copy2(config_path, backup)
                existing_content = config_path.read_text(encoding="utf-8")
                print(f"Backup: {backup}")
            except OSError as e:
                print(f"Warning: could not create backup or read config: {e}")

        # Parse existing config lines to preserve other settings
        lines = existing_content.splitlines()
        new_lines = []
        in_any_section = False
        in_omlx_section = False

        # Keys to override at the top level
        top_level_overrides = {
            "model": f'"{ctx.model or "select-a-model"}"',
            "model_provider": '"omlx"',
        }

        # If it is a reasoning model, add reasoning effort
        is_reasoning = (
            bool(ctx.reasoning)
            if ctx.reasoning is not None
            else bool(re.search(r"\b(thinking|o1|o3|r1)\b", ctx.model.lower()))
        )
        if is_reasoning:
            top_level_overrides["model_reasoning_effort"] = '"high"'

        # Keys managed by oMLX that should be removed when not applicable
        managed_keys = {"model_reasoning_effort"} - set(top_level_overrides.keys())

        seen_keys = set()

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                in_any_section = True
                in_omlx_section = stripped == "[model_providers.omlx]"

            # Handle top-level keys
            if not in_any_section and "=" in stripped:
                key = stripped.split("=")[0].strip()
                if key in top_level_overrides:
                    new_lines.append(f"{key} = {top_level_overrides[key]}")
                    seen_keys.add(key)
                    continue
                if key in managed_keys:
                    continue

            # Skip old oMLX section
            if in_omlx_section:
                continue

            new_lines.append(line)

        # Add missing top-level keys
        for key, val in top_level_overrides.items():
            if key not in seen_keys:
                new_lines.insert(0, f"{key} = {val}")

        # Append new oMLX provider section
        new_lines.append("\n[model_providers.omlx]")
        new_lines.append('name = "oMLX"')
        new_lines.append(f'base_url = "{ctx.openai_base_url}"')
        new_lines.append('env_key = "OMLX_API_KEY"')

        config_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        print(f"Config updated: {config_path}")

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()
        env["OMLX_API_KEY"] = ctx.auth_token

        # Launch codex app (desktop GUI/TUI) instead of codex CLI
        # Note: codex app doesn't accept -m flag, model is set in config
        args = ["codex", "app"]
        args.extend(ctx.extra_args)

        os.execvpe("codex", args, env)
