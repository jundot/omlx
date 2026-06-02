"""OpenCode integration."""

from __future__ import annotations

import os
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext
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

    def get_command(self, ctx: IntegrationContext) -> str:
        return (
            f"{get_cli_prefix()} "
            f"launch opencode --model {ctx.model or 'select-a-model'}"
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

    def configure(self, ctx: IntegrationContext) -> None:
        def updater(config: dict) -> None:
            config.setdefault("provider", {})
            provider_config = {
                "npm": "@ai-sdk/openai-compatible",
                "name": "oMLX",
                "options": {
                    "baseURL": ctx.openai_base_url,
                },
            }
            if ctx.api_key:
                provider_config["options"]["apiKey"] = ctx.api_key
            if ctx.model:
                model_entry: dict = {
                    "name": ctx.model,
                    "modalities": self._modalities_for_model(ctx.model_type),
                }
                if ctx.supports_images:
                    model_entry["attachment"] = True
                if ctx.context_window:
                    model_entry["limit"] = {
                        "context": ctx.context_window,
                        "output": ctx.max_tokens or ctx.context_window,
                    }
                provider_config["models"] = {ctx.model: model_entry}
            config["provider"]["omlx"] = provider_config

            # Set as default model
            if ctx.model:
                config["model"] = f"omlx/{ctx.model}"

        self._write_json_config(self.CONFIG_PATH, updater)

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()
        args = ["opencode"]

        os.execvpe("opencode", args, env)
