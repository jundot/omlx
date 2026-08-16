"""Claude Code integration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_command_prefix

CLAUDE_CODE_MIN_CONTEXT_WINDOW = 48 * 1024

# Claude Code treats these flags as enabled whenever their value is non-empty,
# including values such as "0" and "false".
_CROSS_SESSION_NONEMPTY_BLOCKING_VARS = (
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "DISABLE_TELEMETRY",
)

# These flags use normal boolean semantics, so explicit false values do not
# disable feature-flag fetching.
_CROSS_SESSION_BOOLEAN_BLOCKING_VARS = (
    "DO_NOT_TRACK",
    "DISABLE_GROWTHBOOK",
)


def _env_flag_set(env: dict[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() not in ("", "0", "false", "no")


def _cross_session_blockers(env: dict[str, str]) -> list[str]:
    blocking = [
        name
        for name in _CROSS_SESSION_NONEMPTY_BLOCKING_VARS
        if env.get(name, "") != ""
    ]
    blocking.extend(
        name
        for name in _CROSS_SESSION_BOOLEAN_BLOCKING_VARS
        if _env_flag_set(env, name)
    )
    return blocking


def claude_code_model_disabled_reason(model_info: dict) -> str | None:
    context_window = model_info.get("max_context_window")
    if not isinstance(context_window, int):
        return None
    if context_window >= CLAUDE_CODE_MIN_CONTEXT_WINDOW:
        return None
    return (
        "Claude Code requires at least 48K context "
        f"(configured: {context_window:,} tokens)"
    )


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

    def model_disabled_reason(self, model_info: dict) -> str | None:
        return claude_code_model_disabled_reason(model_info)

    def _find_claude_binary(self) -> str:
        """Find the claude binary in PATH or ~/.claude/local/."""
        if shutil.which("claude"):
            return "claude"
        local = Path.home() / ".claude" / "local" / "claude"
        if local.exists():
            return str(local)
        return "claude"

    def launch(self, ctx: IntegrationContext) -> None:
        disabled_reason = self.model_disabled_reason(
            {"id": ctx.model, "max_context_window": ctx.context_window}
        )
        if disabled_reason:
            print(f"Cannot launch Claude Code with model '{ctx.model}'.")
            print(disabled_reason)
            raise SystemExit(1)

        env = self._scrubbed_env()
        env["ANTHROPIC_BASE_URL"] = ctx.base_url
        # Use the actual omlx API key so Claude Code authenticates correctly.
        # Fallback to "omlx" only when no API key is configured (open server).
        env["ANTHROPIC_AUTH_TOKEN"] = ctx.auth_token
        env["ANTHROPIC_API_KEY"] = ""
        env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
        # Large timeout for local model inference (model loading + generation).
        env["API_TIMEOUT_MS"] = "3000000"

        if ctx.cross_session:
            # Cross-session messaging (ListAgents/SendMessage) needs telemetry
            # and feature-flag traffic enabled, so
            # CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC is deliberately left
            # unset here. Unrelated Anthropic-bound services stay off via
            # their own granular opt-outs instead.
            env["DISABLE_AUTOUPDATER"] = "1"
            env["DISABLE_ERROR_REPORTING"] = "1"
            env["DISABLE_FEEDBACK_COMMAND"] = "1"
            env["CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY"] = "1"

            blocking = _cross_session_blockers(env)
            if blocking:
                print(
                    f"Warning: {', '.join(blocking)} is set in your environment; "
                    "cross-session messaging will remain unavailable despite "
                    "--cross-session. Unset it to enable messaging."
                )
        else:
            # Disable telemetry and non-essential background traffic. This
            # also disables the feature-flag evaluation that cross-session
            # messaging (ListAgents/SendMessage) depends on, so the launched
            # session won't be reachable from other Claude Code sessions on
            # this machine — pass --cross-session to allow that instead.
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
            # Claude Code (2.1.220+) reads CLAUDE_CODE_MAX_CONTEXT_TOKENS to
            # set the *detected* context window for custom model IDs that
            # don't canonicalize to "claude-*"; auto-compact then fires at
            # min(detected_window, CLAUDE_CODE_AUTO_COMPACT_WINDOW). Setting
            # both to the same real, operator-configured max_context_window
            # keeps reported usage and the auto-compact denominator on the
            # same scale with no scaling math and no reliance on the
            # internal/undocumented CLAUDE_AUTOCOMPACT_PCT_OVERRIDE test hook.
            # Caveat (confirmed live): this is ignored for model IDs that
            # canonicalize to "claude-*" — Claude Code trusts its own
            # built-in context window for those regardless of this variable.
            env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(ctx.context_window)

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
            print(f"Auto-compact window: {ctx.context_window:,} tokens")
        os.execvpe(binary, argv, env)
