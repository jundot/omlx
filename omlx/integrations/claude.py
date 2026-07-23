"""Claude Code integration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_command_prefix


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
        return f"{get_cli_command_prefix()} launch claude"

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

        opus_model = ctx.opus_model or ctx.model
        sonnet_model = ctx.sonnet_model or ctx.model
        haiku_model = ctx.haiku_model or ctx.model

        if opus_model:
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = opus_model
        if sonnet_model:
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = sonnet_model
        if haiku_model:
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = haiku_model

        subagent_model = haiku_model or sonnet_model or opus_model
        if subagent_model:
            env["CLAUDE_CODE_SUBAGENT_MODEL"] = subagent_model

        if ctx.context_window:
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(ctx.context_window)

        binary = self._find_claude_binary()
        # argv[0] is cosmetic to the OS/claude itself (verified: claude does not
        # inspect its own argv[0]) but is what tools like `ps` and zellij's
        # session-resurrection command discovery see. Marking it lets external
        # tooling tell an omlx-launched session apart from a bare `claude`
        # invocation or one launched through a different wrapper (e.g. ollama's
        # `--model` flag), which is otherwise impossible once execvpe replaces
        # this process.
        argv = ["claude-omlx", *ctx.extra_args]
        print(f"Launching Claude Code with model {ctx.model}...")
        if ctx.context_window:
            print(f"Auto-compact window: {ctx.context_window:,} tokens")
        os.execvpe(binary, argv, env)
