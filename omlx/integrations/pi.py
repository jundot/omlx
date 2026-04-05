"""Pi integration."""

from __future__ import annotations

import os
import re
from pathlib import Path

from omlx.integrations.base import Integration
from omlx.utils.install import get_cli_prefix


class PiIntegration(Integration):
    """Pi integration that configures ~/.pi/agent/models.json and settings.json."""

    MODELS_PATH = Path.home() / ".pi" / "agent" / "models.json"
    SETTINGS_PATH = Path.home() / ".pi" / "agent" / "settings.json"

    def __init__(self):
        super().__init__(
            name="pi",
            display_name="Pi",
            type="config_file",
            install_check="pi",
            install_hint="npm install -g @mariozechner/pi-coding-agent",
        )

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        return (
            f"{get_cli_prefix()} "
            f"launch pi --model {model or 'select-a-model'}"
        )

    @staticmethod
    def _is_reasoning_model(model: str | None) -> bool:
        return bool(re.search(r"\b(thinking|o1|o3|r1)\b", (model or "").lower()))

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
        def update_models(config: dict) -> None:
            config.setdefault("providers", {})
            provider_config: dict = {
                "baseUrl": f"http://{host}:{port}/v1",
                "api": "openai-completions",
                "apiKey": api_key or "omlx",
                "authHeader": True,
            }
            if model:
                model_entry: dict = {
                    "id": model,
                    "name": model,
                    "reasoning": self._is_reasoning_model(model),
                    "input": ["text", "image"] if model_type == "vlm" else ["text"],
                    "cost": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                    },
                }
                if context_window:
                    model_entry["contextWindow"] = context_window
                if max_tokens:
                    model_entry["maxTokens"] = max_tokens
                provider_config["models"] = [model_entry]
            config["providers"]["omlx"] = provider_config

        def update_settings(config: dict) -> None:
            config["defaultProvider"] = "omlx"
            if model:
                config["defaultModel"] = model

        self._write_json_config(self.MODELS_PATH, update_models)
        self._write_json_config(self.SETTINGS_PATH, update_settings)

    def launch(self, port: int, api_key: str, model: str, host: str = "127.0.0.1", **kwargs) -> None:
        self.configure(
            port,
            api_key,
            model,
            host=host,
            context_window=kwargs.get("context_window"),
            max_tokens=kwargs.get("max_tokens"),
            model_type=kwargs.get("model_type"),
        )

        env = os.environ.copy()
        args = ["pi"]
        if model:
            args.extend(["--model", f"omlx/{model}"])

        os.execvpe("pi", args, env)
