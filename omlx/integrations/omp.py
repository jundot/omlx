"""Oh My Pi (omp) integration."""

from __future__ import annotations

import os
import re
import shutil
import time
from pathlib import Path

import yaml

from omlx.integrations.base import Integration, IntegrationContext
from omlx.utils.install import get_cli_prefix


def _get_agent_dir() -> Path:
    """Get the omp agent config directory, respecting PI_CODING_AGENT_DIR.

    oh-my-pi honors the same PI_CODING_AGENT_DIR override as upstream pi, but
    defaults to ~/.omp/agent (not ~/.pi/agent).
    """
    env_dir = os.environ.get("PI_CODING_AGENT_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / ".omp" / "agent"


class OhMyPiIntegration(Integration):
    """Oh My Pi (omp) integration that configures the omp agent config directory.

    omp is a fork of pi-coding-agent. It reads custom OpenAI-compatible
    providers from ~/.omp/agent/models.yml using the same providers.<name>
    schema as pi, but in YAML rather than JSON. omp only auto-migrates an
    existing models.json to models.yml when models.yml is absent, so we
    write/merge models.yml directly to stay robust across relaunches.
    """

    AGENT_DIR = _get_agent_dir()
    MODELS_PATH = AGENT_DIR / "models.yml"

    def __init__(self):
        super().__init__(
            name="omp",
            display_name="Oh My Pi",
            type="config_file",
            install_check="omp",
            install_hint=(
                "bun install -g @oh-my-pi/pi-coding-agent "
                "(or: curl -fsSL https://omp.sh/install | sh)"
            ),
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        return (
            f"{get_cli_prefix()} "
            f"launch omp --model {ctx.model or 'select-a-model'}"
        )

    @staticmethod
    def _is_reasoning_model(model: str | None) -> bool:
        return bool(re.search(r"\b(thinking|o1|o3|r1)\b", (model or "").lower()))

    def _read_config(self, config_path: Path) -> dict:
        existing: dict = {}
        if not config_path.exists():
            return existing

        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as e:
            print(f"Warning: could not parse {config_path}: {e}")
            print("Creating new config file.")
            return existing

        if loaded is None:
            return existing
        if not isinstance(loaded, dict):
            print(f"Warning: {config_path} does not contain a YAML object.")
            print("Creating new config file.")
            return existing
        return loaded

    @staticmethod
    def _create_backup(config_path: Path) -> None:
        if not config_path.exists():
            return

        timestamp = int(time.time())
        backup = config_path.with_suffix(f".{timestamp}.bak")
        try:
            shutil.copy2(config_path, backup)
            print(f"Backup: {backup}")
        except OSError as e:
            print(f"Warning: could not create backup: {e}")

    def configure(self, ctx: IntegrationContext) -> None:
        config_path = self.MODELS_PATH
        config = self._read_config(config_path)
        self._create_backup(config_path)

        providers = config.setdefault("providers", {})
        if not isinstance(providers, dict):
            providers = {}
            config["providers"] = providers

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
        providers["omlx"] = provider_config

        config_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_content = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        config_path.write_text(
            yaml_content.rstrip() + "\n",
            encoding="utf-8",
        )
        print(f"Config written: {config_path}")

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()
        args = ["omp"]
        if ctx.model:
            args.extend(["--model", f"omlx/{ctx.model}"])

        os.execvpe("omp", args, env)
