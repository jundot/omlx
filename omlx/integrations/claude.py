"""Claude Code integration."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from omlx.integrations.base import Integration


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

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        from omlx.utils.install import get_cli_prefix
        return f"{get_cli_prefix()} launch claude"

    def _find_claude_binary(self) -> str:
        """Find the claude binary in PATH or ~/.claude/local/."""
        if shutil.which("claude"):
            return "claude"
        local = Path.home() / ".claude" / "local" / "claude"
        if local.exists():
            return str(local)
        return "claude"

    def launch(
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        context_window: int | None = None,
        **kwargs,
    ) -> None:
        env = os.environ.copy()
        # Remove omlx-cli Python env vars so Claude Code's subprocess hooks
        # don't inherit our bundled Python 3.11 environment.
        for _key in ("PYTHONHOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE"):
            env.pop(_key, None)

        env["ANTHROPIC_BASE_URL"] = f"http://{host}:{port}"
        # Use the actual omlx API key so Claude Code authenticates correctly.
        # Fallback to "omlx" only when no API key is configured (open server).
        env["ANTHROPIC_AUTH_TOKEN"] = api_key or "omlx"
        env["ANTHROPIC_API_KEY"] = ""
        env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
        # Large timeout for local model inference (model loading + generation).
        env["API_TIMEOUT_MS"] = "3000000"
        # Disable telemetry and non-essential background traffic.
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

        if model:
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
            env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
            env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
            env["CLAUDE_CODE_SUBAGENT_MODEL"] = model

        if context_window:
            env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(context_window)

        binary = self._find_claude_binary()
        print(f"Launching Claude Code with model {model}...")
        if context_window:
            print(f"Auto-compact window: {context_window:,} tokens")
        os.execvpe(binary, [binary], env)
