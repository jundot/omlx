"""OpenCode integration."""

from __future__ import annotations

import os
from pathlib import Path

from omlx.integrations.base import Integration


class OpenCodeIntegration(Integration):
    """OpenCode integration that writes ~/.config/opencode/opencode.json."""

    CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"

    def __init__(self):
        super().__init__(
            name="opencode",
            display_name="OpenCode",
            type="config_file",
            install_check="opencode",
            install_hint="curl -fsSL https://opencode.ai/install | bash",
        )

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        return (
            f"/Applications/oMLX.app/Contents/MacOS/omlx-cli "
            f"launch opencode --model {model or 'select-a-model'}"
        )

    def configure(self, port: int, api_key: str, model: str) -> None:
        def updater(config: dict) -> None:
            config.setdefault("provider", {})
            provider_config = {
                "npm": "@ai-sdk/openai-compatible",
                "name": "oMLX",
                "options": {
                    "baseURL": f"http://127.0.0.1:{port}/v1",
                },
            }
            if api_key:
                provider_config["options"]["apiKey"] = api_key
            if model:
                provider_config["models"] = {
                    model: {
                        "name": model,
                    },
                }
            config["provider"]["omlx"] = provider_config

            # Set as default model
            if model:
                config["model"] = f"omlx/{model}"

        self._write_json_config(self.CONFIG_PATH, updater)

    def launch(self, port: int, api_key: str, model: str, **kwargs) -> None:
        self.configure(port, api_key, model)

        env = os.environ.copy()
        args = ["opencode"]

        os.execvpe("opencode", args, env)
