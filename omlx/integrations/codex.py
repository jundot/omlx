"""Codex (OpenAI Codex CLI) integration."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from omlx.integrations.base import Integration

# Codex config.toml template with oMLX provider
_CONFIG_TEMPLATE = """\
model = "{model}"
model_provider = "omlx"

[model_providers.omlx]
name = "oMLX"
base_url = "http://127.0.0.1:{port}/v1"
env_key = "OMLX_API_KEY"
"""


class CodexIntegration(Integration):
    """Codex integration that configures ~/.codex/config.toml for oMLX."""

    CONFIG_PATH = Path.home() / ".codex" / "config.toml"

    def __init__(self):
        super().__init__(
            name="codex",
            display_name="Codex",
            type="config_file",
            install_check="codex",
            install_hint="npm install -g @openai/codex",
        )

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        return (
            f"/Applications/oMLX.app/Contents/MacOS/omlx-cli "
            f"launch codex --model {model or 'select-a-model'}"
        )

    def configure(self, port: int, api_key: str, model: str) -> None:
        config_path = self.CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Create backup if exists
        if config_path.exists():
            timestamp = int(time.time())
            backup = config_path.with_suffix(f".{timestamp}.bak")
            try:
                shutil.copy2(config_path, backup)
                print(f"Backup: {backup}")
            except OSError as e:
                print(f"Warning: could not create backup: {e}")

        content = _CONFIG_TEMPLATE.format(
            model=model or "select-a-model",
            port=port,
        )
        config_path.write_text(content, encoding="utf-8")
        print(f"Config written: {config_path}")

    def launch(self, port: int, api_key: str, model: str, **kwargs) -> None:
        self.configure(port, api_key, model)

        env = os.environ.copy()
        env["OMLX_API_KEY"] = api_key or "omlx"

        args = ["codex"]
        if model:
            args.extend(["-m", model])

        os.execvpe("codex", args, env)
