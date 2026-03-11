"""Base class for external tool integrations."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Integration:
    """Base integration definition."""

    name: str  # "codex", "opencode", "openclaw"
    display_name: str  # "Codex", "OpenCode", "OpenClaw"
    type: str  # "env_var" or "config_file"
    install_check: str  # binary name to check with `which`
    install_hint: str  # installation instructions

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        """Generate the command string for clipboard/display."""
        raise NotImplementedError

    def configure(self, port: int, api_key: str, model: str) -> None:
        """Configure the tool (write config files, etc.)."""
        pass

    def launch(self, port: int, api_key: str, model: str, **kwargs) -> None:
        """Configure and launch the tool."""
        raise NotImplementedError

    def is_installed(self) -> bool:
        """Check if the tool binary is available."""
        return shutil.which(self.install_check) is not None

    def _write_json_config(
        self,
        config_path: Path,
        updater: callable,
    ) -> None:
        """Read, update, and write a JSON config file with backup.

        Args:
            config_path: Path to the config file.
            updater: Function that takes existing config dict and modifies it in-place.
        """
        existing: dict = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: could not parse {config_path}: {e}")
                print("Creating new config file.")
                existing = {}

            # Create timestamped backup
            timestamp = int(time.time())
            backup = config_path.with_suffix(f".{timestamp}.bak")
            try:
                shutil.copy2(config_path, backup)
                print(f"Backup: {backup}")
            except OSError as e:
                print(f"Warning: could not create backup: {e}")

        updater(existing)

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Config written: {config_path}")
