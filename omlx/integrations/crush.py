# SPDX-License-Identifier: Apache-2.0
"""Crush integration."""

from __future__ import annotations

import os
from pathlib import Path

from omlx.integrations.base import Integration
from omlx.utils.install import get_cli_prefix


class CrushIntegration(Integration):
    """Crush integration that writes ~/.config/crush/crush.json."""

    CONFIG_PATH = Path.home() / ".config" / "crush" / "crush.json"

    def __init__(self):
        super().__init__(
            name="crush",
            display_name="Crush",
            type="config_file",
            install_check="crush",
            install_hint="brew install charmbracelet/tap/crush",
        )

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        return (
            f"{get_cli_prefix()} "
            f"launch crush --model {model or 'select-a-model'}"
        )

    def configure(
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        context_window: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        def updater(config: dict) -> None:
            config.setdefault("providers", {})
            provider_config: dict = {
                "name": "oMLX",
                "type": "openai-compat",
                "base_url": f"http://{host}:{port}/v1",
                "api_key": api_key or "omlx",
            }
            if model:
                model_entry: dict = {"id": model, "name": model}
                if context_window:
                    model_entry["context_window"] = context_window
                if max_tokens:
                    model_entry["default_max_tokens"] = max_tokens
                provider_config["models"] = [model_entry]
            config["providers"]["omlx"] = provider_config

            if model:
                config.setdefault("models", {})
                selection = {"provider": "omlx", "model": model}
                config["models"]["large"] = selection
                config["models"]["small"] = dict(selection)

        self._write_json_config(self.CONFIG_PATH, updater)

    def launch(self, port: int, api_key: str, model: str, host: str = "127.0.0.1", **kwargs) -> None:
        context_window = kwargs.pop("context_window", None)
        max_tokens = kwargs.pop("max_tokens", None)
        self.configure(
            port, api_key, model, host=host,
            context_window=context_window, max_tokens=max_tokens,
        )

        env = os.environ.copy()
        args = ["crush"]

        os.execvpe("crush", args, env)
