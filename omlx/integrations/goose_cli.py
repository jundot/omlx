# SPDX-License-Identifier: Apache-2.0
"""Goose CLI integration.

This integration configures and launches the Goose CLI agent (https://github.com/block/goose)
pointing at oMLX as its OpenAI-compatible API backend.

Goose uses the following environment variables for its OpenAI provider:
    OPENAI_API_BASE  - The API base URL (set to oMLX's /v1 endpoint)
    OPENAI_API_KEY   - The API key for authentication

Usage:
    omlx launch goose_cli --model qwen3.5

Which launches:
    goose
"""

from __future__ import annotations

import os

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_command_prefix


class GooseCliIntegration(Integration):
    """Goose CLI integration using OpenAI-compatible env vars."""

    def __init__(self):
        super().__init__(
            name="goose_cli",
            display_name="Goose CLI",
            type="env_var",
            install_check="goose",
            install_hint=(
                "brew install goose"
            ),
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        return (
            f"{get_cli_command_prefix()} "
            f"launch goose_cli --model {ctx.model or 'select-a-model'}"
        )

    def launch(self, ctx: IntegrationContext) -> None:
        env = self._scrubbed_env()

        # Point goose at oMLX's OpenAI-compatible API
        env["OPENAI_API_BASE"] = ctx.openai_base_url
        # Use the actual omlx API key so goose authenticates correctly.
        # Fallback to "omlx" only when no API key is configured (open server).
        env["OPENAI_API_KEY"] = ctx.auth_token

        # Set the default model if provided
        if ctx.model:
            env["GOOSE_MODEL"] = ctx.model

        # Set context window if provided (goose respects this for token limits)
        if ctx.context_window:
            env["OPENAI_MAX_TOKENS"] = str(ctx.context_window)

        # Set max output tokens if provided
        if ctx.max_tokens:
            env["OPENAI_MAX_COMPLETION_TOKENS"] = str(ctx.max_tokens)

        # Disable goose telemetry for local usage
        env["GOOSE_TELEMETRY"] = "0"

        print(f"Launching Goose CLI with model {ctx.model}...")
        if ctx.context_window:
            print(f"Context window: {ctx.context_window:,} tokens")
        if ctx.max_tokens:
            print(f"Max output tokens: {ctx.max_tokens:,}")

        os.execvpe("goose", ["goose", *ctx.extra_args], env)
