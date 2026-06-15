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

from omlx.integrations.base import Integration, IntegrationContext
from omlx.integrations.codex import CODEX_CONFIG_PATH, write_codex_config
from omlx.utils.install import get_cli_command_prefix


class CodexAppIntegration(Integration):
    """Codex App Desktop integration that configures ~/.codex/config.toml for oMLX."""

    CONFIG_PATH = CODEX_CONFIG_PATH

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
        write_codex_config(self.CONFIG_PATH, ctx)

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()
        env["OMLX_API_KEY"] = ctx.auth_token

        # Launch codex app (desktop GUI/TUI) instead of codex CLI
        # Note: codex app doesn't accept -m flag, model is set in config
        args = ["codex", "app"]
        args.extend(ctx.extra_args)

        os.execvpe("codex", args, env)
