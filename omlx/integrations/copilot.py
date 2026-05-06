"""GitHub Copilot CLI integration."""

from __future__ import annotations

import os

from omlx.integrations.base import Integration
from omlx.utils.install import get_cli_prefix


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

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        return f"{get_cli_prefix()} launch copilot --model {model or 'select-a-model'}"

    def launch(
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        context_window: int | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> None:
        env = os.environ.copy()
        # Remove omlx-cli Python env vars so Copilot subprocesses don't inherit
        # our bundled Python runtime.
        for _key in ("PYTHONHOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE"):
            env.pop(_key, None)

        env["COPILOT_PROVIDER_BASE_URL"] = f"http://{host}:{port}/v1"
        env["COPILOT_PROVIDER_TYPE"] = "openai"

        # Copilot CLI appears to have issues with the completions endpoint, responses appears to work as expected.
        env["COPILOT_PROVIDER_WIRE_API"] = "responses"
        env["COPILOT_PROVIDER_BEARER_TOKEN"] = api_key or "omlx"

        if model:
            env["COPILOT_MODEL"] = model
            env["COPILOT_PROVIDER_MODEL_ID"] = model
            env["COPILOT_PROVIDER_WIRE_MODEL"] = model

        if context_window:
            env["COPILOT_PROVIDER_MAX_PROMPT_TOKENS"] = str(context_window)
        if max_tokens:
            env["COPILOT_PROVIDER_MAX_OUTPUT_TOKENS"] = str(max_tokens)

        print(f"Launching Copilot CLI with model {model}...")
        if context_window:
            print(f"Max prompt tokens: {context_window:,}")
        if max_tokens:
            print(f"Max output tokens: {max_tokens:,}")
        os.execvpe("copilot", ["copilot"], env)
