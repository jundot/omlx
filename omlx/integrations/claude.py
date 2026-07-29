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
        if ctx.autocompact_threshold_pct is not None:
            # Claude Code's CLI parses this as a plain 0-100 percentage
            # (`n>0 && n<=100`, then divides by 100 internally) — not a
            # 0-1 fraction. Sending "0.8" for 80% is silently accepted
            # (0.8 <= 100) but interpreted as 0.8%, firing compaction
            # almost immediately. Confirmed via decompiled CLI strings
            # (2.1.220) after live testing showed early auto-compact
            # around 11-13% of the real window instead of 80%.
            env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(
                ctx.autocompact_threshold_pct
            )

        binary = self._find_claude_binary()
        # Deny the LSP tool. Claude Code attaches its full schema to the tools
        # array the moment a language server connects mid-session; tool schemas
        # render into the system region of the prompt, so that one insertion
        # re-prefills the whole conversation on a prefix-caching server (#2349).
        # LSP code intelligence is marginal against a local model, so the
        # cache stability is the better default here. Skip it when the caller
        # already passes their own --disallowedTools so we never fight their
        # choice or duplicate the flag.
        extra_args = list(ctx.extra_args)
        if not any(
            a in ("--disallowedTools", "--disallowed-tools") for a in extra_args
        ):
            extra_args = ["--disallowedTools", "LSP", *extra_args]
        argv = [binary, *extra_args]
        print(f"Launching Claude Code with model {ctx.model}...")
        if ctx.context_window:
            window_msg = f"Auto-compact window: {ctx.context_window:,} tokens"
            if ctx.autocompact_threshold_pct is not None:
                trigger_at = int(ctx.context_window * ctx.autocompact_threshold_pct / 100)
                window_msg += (
                    f" (compacts at {ctx.autocompact_threshold_pct}% "
                    f"≈ {trigger_at:,} tokens)"
                )
            print(window_msg)
        os.execvpe(binary, argv, env)
