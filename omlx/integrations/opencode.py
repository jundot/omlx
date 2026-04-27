"""OpenCode integration."""

from __future__ import annotations

import os
from pathlib import Path

from omlx.integrations.base import Integration
from omlx.utils.install import get_cli_prefix


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
            f"{get_cli_prefix()} "
            f"launch opencode --model {model or 'select-a-model'}"
        )

    @staticmethod
    def _modalities_for_model(model_type: str | None) -> dict[str, list[str]]:
        """Build OpenCode modality metadata for the selected oMLX model."""
        input_modalities = ["text"]
        if model_type == "vlm":
            input_modalities.append("image")
        return {
            "input": input_modalities,
            "output": ["text"],
        }

    def configure(
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        context_window: int | None = None,
        max_tokens: int | None = None,
        model_type: str | None = None,
    ) -> None:
        def updater(config: dict) -> None:
            config.setdefault("provider", {})
            provider_config = {
                "npm": "@ai-sdk/openai-compatible",
                "name": "oMLX",
                "options": {
                    "baseURL": f"http://{host}:{port}/v1",
                },
            }
            if api_key:
                provider_config["options"]["apiKey"] = api_key
            if model:
                model_entry: dict = {
                    "name": model,
                    "modalities": self._modalities_for_model(model_type),
                }
                if model_type == "vlm":
                    model_entry["attachment"] = True
                if context_window:
                    model_entry["limit"] = {
                        "context": context_window,
                        "output": max_tokens or context_window,
                    }
                provider_config["models"] = {model: model_entry}
            config["provider"]["omlx"] = provider_config

            # Set as default model
            if model:
                config["model"] = f"omlx/{model}"

        self._write_json_config(self.CONFIG_PATH, updater)

    def launch(self, port: int, api_key: str, model: str, host: str = "127.0.0.1", **kwargs) -> None:
        context_window = kwargs.pop("context_window", None)
        max_tokens = kwargs.pop("max_tokens", None)
        model_type = kwargs.pop("model_type", None)
        self.configure(
            port, api_key, model, host=host,
            context_window=context_window, max_tokens=max_tokens,
            model_type=model_type,
        )

        env = os.environ.copy()
        args = ["opencode"]

        os.execvpe("opencode", args, env)
