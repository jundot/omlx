"""GitHub Copilot CLI integration."""

from __future__ import annotations

import os

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_command_prefix


class CopilotIntegration(Integration):
    """Copilot integration using custom provider BYOK environment variables."""

    def __init__(self):
        super().__init__(
            name="copilot",
            display_name="Copilot CLI",
            type="env_var",
            install_check="copilot",
            install_hint="npm install -g @github/copilot",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        return (
            f"{get_cli_command_prefix()} launch copilot "
            f"--model {ctx.model or 'select-a-model'}"
        )

    def launch(self, ctx: IntegrationContext) -> None:
        env = self._scrubbed_env()
        env["COPILOT_PROVIDER_BASE_URL"] = ctx.openai_base_url
        env["COPILOT_PROVIDER_TYPE"] = "openai"

        # Copilot CLI appears to have issues with the completions endpoint, responses appears to work as expected.
        env["COPILOT_PROVIDER_WIRE_API"] = "responses"
        env["COPILOT_PROVIDER_BEARER_TOKEN"] = ctx.auth_token

        if ctx.model:
            env["COPILOT_MODEL"] = ctx.model
            env["COPILOT_PROVIDER_MODEL_ID"] = ctx.model
            env["COPILOT_PROVIDER_WIRE_MODEL"] = ctx.model

        if ctx.context_window:
            env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] = str(ctx.context_window)
        if ctx.max_tokens:
            env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] = str(ctx.max_tokens)

        print(f"Launching Copilot CLI with model {ctx.model}...")
        if ctx.context_window:
            print(f"Max prompt tokens: {ctx.context_window:,}")
        if ctx.max_tokens:
            print(f"Max output tokens: {ctx.max_tokens:,}")
        os.execvpe("copilot", ["copilot", *ctx.extra_args], env)
