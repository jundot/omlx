"""Claude Code integration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_prefix


class ClaudeCodeIntegration(Integration):
    """Claude Code integration using ANTHROPIC_BASE_URL env vars."""

    def __init__(self):
        super().__init__(
            name="claude",
            display_name="Claude Code",
            type="env_var",
            install_check="claude",
            install_hint="npm install -g @anthropic-ai/claude-code",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        return f"{get_cli_prefix()} launch claude"

    def _find_claude_binary(self) -> str:
        """Find the claude binary in PATH or ~/.claude/local/."""
        if shutil.which("claude"):
            return "claude"
        local = Path.home() / ".claude" / "local" / "claude"
        if local.exists():
            return str(local)
        return "claude"

    def launch(self, ctx: IntegrationContext) -> None:
        env = self._scrubbed_env()
        env["ANTHROPIC_BASE_URL"] = ctx.base_url
        # Use the actual omlx API key so Claude Code authenticates correctly.
        # Fallback to "omlx" only when no API key is configured (open server).
        env["ANTHROPIC_AUTH_TOKEN"] = ctx.auth_token
        env["ANTHROPIC_API_KEY"] = ""
        env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
        # Large timeout for local model inference (model loading + generation).
        env["API_TIMEOUT_MS"] = "3000000"
        # Disable telemetry and non-essential background traffic.
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

        if ctx.model:
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = ctx.model
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = ctx.model
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = ctx.model
            env["CLAUDE_CODE_SUBAGENT_MODEL"] = ctx.model

        if ctx.context_window:
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(ctx.context_window)

        binary = self._find_claude_binary()
        argv = [binary, *ctx.extra_args]
        print(f"Launching Claude Code with model {ctx.model}...")
        if ctx.context_window:
            print(f"Auto-compact window: {ctx.context_window:,} tokens")
        os.execvpe(binary, argv, env)
