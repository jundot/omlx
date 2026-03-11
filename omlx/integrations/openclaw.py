"""OpenClaw integration."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from omlx.integrations.base import Integration

DEFAULT_GATEWAY_PORT = 18789


class OpenClawIntegration(Integration):
    """OpenClaw integration that writes ~/.openclaw/openclaw.json."""

    CONFIG_PATH = Path.home() / ".openclaw" / "openclaw.json"

    def __init__(self):
        super().__init__(
            name="openclaw",
            display_name="OpenClaw",
            type="config_file",
            install_check="openclaw",
            install_hint="npm install -g openclaw",
        )

    def get_command(
        self, port: int, api_key: str, model: str, host: str = "127.0.0.1"
    ) -> str:
        return (
            f"/Applications/oMLX.app/Contents/MacOS/omlx-cli "
            f"launch openclaw --model {model or 'select-a-model'}"
        )

    def configure(
        self,
        port: int,
        api_key: str,
        model: str,
        tools_profile: str = "coding",
    ) -> None:
        def updater(config: dict) -> None:
            config.setdefault("models", {}).setdefault("providers", {})
            provider_config = {
                "baseUrl": f"http://127.0.0.1:{port}/v1",
                "apiKey": api_key or "omlx",
                "api": "openai-completions",
            }
            if model:
                provider_config["models"] = [
                    {
                        "id": model,
                        "name": model,
                        "api": "openai-completions",
                        "reasoning": False,
                        "input": ["text"],
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                        "contextWindow": 131072,
                        "maxTokens": 8192,
                    }
                ]
            config["models"]["providers"]["omlx"] = provider_config

            # Set as default model
            if model:
                config.setdefault("agents", {}).setdefault(
                    "defaults", {}
                ).setdefault("model", {})
                config["agents"]["defaults"]["model"]["primary"] = f"omlx/{model}"

            # Set tools profile
            config.setdefault("tools", {})
            config["tools"]["profile"] = tools_profile

        self._write_json_config(self.CONFIG_PATH, updater)

    def _gateway_info(self) -> tuple[str, int]:
        """Read gateway token and port from OpenClaw config."""
        token = ""
        port = DEFAULT_GATEWAY_PORT
        try:
            data = json.loads(self.CONFIG_PATH.read_text())
            gw = data.get("gateway", {})
            if p := gw.get("port"):
                port = int(p)
            auth = gw.get("auth", {})
            if t := auth.get("token"):
                token = t
        except Exception:
            pass
        return token, port

    @staticmethod
    def _port_open(host: str, port: int) -> bool:
        """Check if a TCP port is accepting connections."""
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            return False

    @staticmethod
    def _wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
        """Wait until a TCP port is accepting connections."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    return True
            except OSError:
                time.sleep(0.25)
        return False

    def _is_onboarded(self) -> bool:
        """Check if OpenClaw onboarding was completed."""
        try:
            data = json.loads(self.CONFIG_PATH.read_text())
            return bool(data.get("wizard", {}).get("lastRunAt"))
        except Exception:
            return False

    EXEC_APPROVALS_PATH = Path.home() / ".openclaw" / "exec-approvals.json"

    def configure_exec_approvals(self, tools_profile: str = "coding") -> None:
        """Configure exec-approvals.json based on tools profile.

        Profiles "coding" and "full" get unrestricted exec.
        Others get allowlist-based restrictions.
        """
        allow_exec = tools_profile in ("coding", "full")

        def updater(config: dict) -> None:
            config.setdefault("defaults", {})
            if allow_exec:
                config["defaults"]["security"] = "full"
                config["defaults"]["ask"] = "off"
            else:
                config["defaults"]["security"] = "allowlist"
                config["defaults"]["ask"] = "on-miss"

        self._write_json_config(self.EXEC_APPROVALS_PATH, updater)

    def launch(
        self,
        port: int,
        api_key: str,
        model: str,
        tools_profile: str = "coding",
        **kwargs,
    ) -> None:
        self.configure(port, api_key, model, tools_profile=tools_profile)
        self.configure_exec_approvals(tools_profile)

        env = os.environ.copy()
        bin_name = "openclaw"

        # Run onboarding if not yet completed (like ollama does)
        if not self._is_onboarded():
            print("Setting up OpenClaw with oMLX...")
            print(f"  Model: {model}")
            subprocess.run(
                [
                    bin_name,
                    "onboard",
                    "--non-interactive",
                    "--accept-risk",
                    "--auth-choice",
                    "skip",
                    "--gateway-token",
                    "omlx",
                    "--install-daemon",
                    "--skip-channels",
                    "--skip-skills",
                ],
                env=env,
            )
            # Onboarding overwrites config, re-apply
            self.configure(port, api_key, model, tools_profile=tools_profile)
            self.configure_exec_approvals(tools_profile)

        _, gw_port = self._gateway_info()
        addr = ("localhost", gw_port)

        # Restart gateway if already running so it picks up config changes
        if self._port_open(*addr):
            subprocess.run(
                [bin_name, "daemon", "restart"],
                env=env,
                capture_output=True,
            )
            if not self._wait_for_port(*addr, timeout=10.0):
                print("Warning: gateway did not come back after restart")

        # Start gateway if not running
        if not self._port_open(*addr):
            gw = subprocess.Popen(
                [bin_name, "gateway", "run", "--force"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not self._wait_for_port(*addr, timeout=30.0):
                gw.kill()
                print(f"Gateway did not start on port {gw_port}")
                sys.exit(1)

        print("OpenClaw is running")

        # Launch TUI (replaces this process)
        os.execvpe(bin_name, [bin_name, "tui"], env)
