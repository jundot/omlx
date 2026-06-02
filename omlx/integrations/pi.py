"""Pi integration."""

from __future__ import annotations

import os
import re
from pathlib import Path

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_prefix


def _get_agent_dir() -> Path:
    """Get the pi agent config directory, respecting PI_CODING_AGENT_DIR."""
    env_dir = os.environ.get("PI_CODING_AGENT_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / ".pi" / "agent"


class PiIntegration(Integration):
    """Pi integration that configures the pi agent config directory."""

    AGENT_DIR = _get_agent_dir()
    MODELS_PATH = AGENT_DIR / "models.json"
    SETTINGS_PATH = AGENT_DIR / "settings.json"

    def __init__(self):
        super().__init__(
            name="pi",
            display_name="Pi",
            type="config_file",
            install_check="pi",
            install_hint="npm install -g @mariozechner/pi-coding-agent",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        return (
            f"{get_cli_prefix()} "
            f"launch pi --model {ctx.model or 'select-a-model'}"
        )

    @staticmethod
    def _is_reasoning_model(model: str | None) -> bool:
        return bool(re.search(r"\b(thinking|o1|o3|r1)\b", (model or "").lower()))

    def configure(self, ctx: IntegrationContext) -> None:
        def update_models(config: dict) -> None:
            config.setdefault("providers", {})
            provider_config: dict = {
                "baseUrl": ctx.openai_base_url,
                "api": "openai-completions",
                "apiKey": ctx.auth_token,
                "authHeader": True,
            }
            if ctx.model:
                reasoning = (
                    bool(ctx.reasoning)
                    if ctx.reasoning is not None
                    else self._is_reasoning_model(ctx.model)
                )
                model_entry: dict = {
                    "id": ctx.model,
                    "name": ctx.model,
                    "reasoning": reasoning,
                    "input": ["text", "image"] if ctx.supports_images else ["text"],
                    "cost": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                    },
                }
                if ctx.context_window:
                    model_entry["contextWindow"] = ctx.context_window
                if ctx.max_tokens:
                    model_entry["maxTokens"] = ctx.max_tokens
                provider_config["models"] = [model_entry]
            config["providers"]["omlx"] = provider_config

        def update_settings(config: dict) -> None:
            config["defaultProvider"] = "omlx"
            if ctx.model:
                config["defaultModel"] = ctx.model

        self._write_json_config(self.MODELS_PATH, update_models)
        self._write_json_config(self.SETTINGS_PATH, update_settings)

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()
        args = ["pi"]
        if ctx.model:
            args.extend(["--model", f"omlx/{ctx.model}"])

        os.execvpe("pi", args, env)
