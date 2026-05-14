"""Hermes Agent integration."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import yaml

from omlx.integrations.base import Integration
from omlx.utils.install import get_cli_prefix

HERMES_MIN_CONTEXT_LENGTH = 64_000


class HermesIntegration(Integration):
    """Hermes Agent integration that writes ~/.hermes/config.yaml."""

    CONFIG_PATH = Path.home() / ".hermes" / "config.yaml"

    def __init__(self):
        super().__init__(
            name="hermes",
            display_name="Hermes Agent",
            type="config_file",
            install_check="hermes",
            install_hint=(
                "curl -fsSL "
                "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh "
                "| bash"
            ),
        )

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        return (
            f"{get_cli_prefix()} "
            f"launch hermes --model {model or 'select-a-model'}"
        )

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

    def configure(
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        context_window: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        config_path = self.CONFIG_PATH
        config = self._read_config(config_path)
        self._create_backup(config_path)

        providers = config.setdefault("providers", {})
        if not isinstance(providers, dict):
            providers = {}
            config["providers"] = providers

        provider_config = providers.get("omlx", {})
        if not isinstance(provider_config, dict):
            provider_config = {}
        provider_config.update(
            {
                "name": "oMLX",
                "base_url": f"http://{host}:{port}/v1",
                "api_key": api_key or "omlx",
                "api_mode": "chat_completions",
            }
        )
        if model:
            provider_config["default_model"] = model
        providers["omlx"] = provider_config

        model_config = config.get("model", {})
        if not isinstance(model_config, dict):
            model_config = {}
        for stale_key in ("base_url", "api_key", "api", "api_mode", "transport"):
            model_config.pop(stale_key, None)
        model_config["provider"] = "omlx"
        if model:
            model_config["default"] = model
        if context_window is not None:
            if context_window < HERMES_MIN_CONTEXT_LENGTH:
                print(
                    "Warning: Hermes Agent requires at least "
                    f"{HERMES_MIN_CONTEXT_LENGTH:,} context tokens; "
                    f"oMLX reports {context_window:,}. Writing the Hermes "
                    "minimum so the agent can start. Increase oMLX Sampling "
                    "max_context_window for long sessions."
                )
            model_config["context_length"] = max(
                context_window,
                HERMES_MIN_CONTEXT_LENGTH,
            )
        else:
            model_config.pop("context_length", None)
        if max_tokens is not None:
            model_config["max_tokens"] = max_tokens
        else:
            model_config.pop("max_tokens", None)
        config["model"] = model_config

        config_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_content = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        config_path.write_text(
            yaml_content.rstrip() + "\n",
            encoding="utf-8",
        )
        print(f"Config written: {config_path}")

    def launch(
        self,
        port: int,
        api_key: str,
        model: str,
        host: str = "127.0.0.1",
        **kwargs,
    ) -> None:
        context_window = kwargs.pop("context_window", None)
        max_tokens = kwargs.pop("max_tokens", None)
        self.configure(
            port,
            api_key,
            model,
            host=host,
            context_window=context_window,
            max_tokens=max_tokens,
        )

        env = os.environ.copy()
        for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE"):
            env.pop(key, None)

        # Hermes Agent v0.12.0's classic prompt_toolkit REPL registers an
        # invalid Ctrl+Shift+C keybinding ("c-S-c") on startup. The modern TUI
        # path avoids that startup crash and is the supported interactive UX.
        args = ["hermes", "--provider", "omlx", "--tui"]
        if model:
            args.extend(["--model", model])

        os.execvpe("hermes", args, env)
