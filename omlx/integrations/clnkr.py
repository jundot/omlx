"""clnkr integration."""

from __future__ import annotations

import os

from omlx.integrations.base import Integration
from omlx.utils.install import get_cli_prefix


class ClnkrIntegration(Integration):
    """clnkr integration using OpenAI-compatible environment variables."""

    def __init__(self):
        super().__init__(
            name="clnkr",
            display_name="clnkr",
            type="env_var",
            install_check="clnkr",
            install_hint="See https://clnkr.ai/",
        )

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        return f"{get_cli_prefix()} launch clnkr --model {model or 'select-a-model'}"

    def launch(
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        extra_args: list[str] | None = None,
        **kwargs,
    ) -> None:
        env = self._scrubbed_env()
        env["CLNKR_API_KEY"] = api_key or "omlx"
        env["CLNKR_BASE_URL"] = f"http://{host}:{port}/v1"
        env["CLNKR_PROVIDER"] = "openai"
        env["CLNKR_PROVIDER_API"] = "openai-chat-completions"

        if model:
            env["CLNKR_MODEL"] = model

        args = ["clnkr"]
        args.extend(extra_args or [])

        print(f"Launching clnkr with model {model}...")
        os.execvpe("clnkr", args, env)
