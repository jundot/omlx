"""Configuration management for oMLX menubar app."""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def get_app_support_dir() -> Path:
    """Get the Application Support directory for oMLX."""
    app_support = Path.home() / "Library" / "Application Support" / "oMLX"
    app_support.mkdir(parents=True, exist_ok=True)
    return app_support


def get_config_path() -> Path:
    """Get the config file path."""
    return get_app_support_dir() / "config.json"


def get_log_path() -> Path:
    """Get the log file path."""
    log_dir = get_app_support_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "server.log"


@dataclass
class ServerConfig:
    """Server configuration settings.

    The server reads its own detailed settings from {base_path}/settings.json.
    The app only needs base_path, port, model_dir, and behavioral flags.
    """

    base_path: str = str(Path.home() / ".omlx")
    port: int = 8000
    model_dir: str = ""  # Empty string means use default: {base_path}/models
    launch_at_login: bool = False
    start_server_on_launch: bool = False

    def get_effective_model_dir(self) -> str:
        """Get the model directory, using base_path/models if not specified."""
        if self.model_dir:
            return self.model_dir
        return str(Path(self.base_path).expanduser() / "models")

    @property
    def is_first_run(self) -> bool:
        """Check if this is the first run (no config file exists)."""
        return not get_config_path().exists()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ServerConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save(self) -> None:
        with open(get_config_path(), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls) -> "ServerConfig":
        config_path = get_config_path()
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return cls.from_dict(json.load(f))
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
        return cls()

    def get_server_api_key(self) -> Optional[str]:
        """Read the API key from the server's settings.json."""
        settings_file = Path(self.base_path).expanduser() / "settings.json"
        if settings_file.exists():
            try:
                with open(settings_file) as f:
                    data = json.load(f)
                return data.get("auth", {}).get("api_key")
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def set_server_api_key(self, api_key: str) -> None:
        """Write API key to server's settings.json."""
        settings_file = Path(self.base_path).expanduser() / "settings.json"
        data = {}
        if settings_file.exists():
            try:
                with open(settings_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        if "auth" not in data:
            data["auth"] = {}
        data["auth"]["api_key"] = api_key
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_file, "w") as f:
            json.dump(data, f, indent=2)

    def update_server_api_key_runtime(self, new_api_key: str) -> bool:
        """Update API key on a running server via admin API.

        Authenticates with the current API key, then updates to the new one.
        This changes both the runtime state and settings.json on the server side.

        Returns True if successful, False if server is not reachable or auth fails.
        """
        base_url = f"http://127.0.0.1:{self.port}"
        current_key = self.get_server_api_key()

        try:
            session = requests.Session()
            session.trust_env = False

            if current_key:
                # Login with current API key
                login_resp = session.post(
                    f"{base_url}/admin/api/login",
                    json={"api_key": current_key},
                    timeout=2,
                )
                if login_resp.status_code != 200:
                    return False
            else:
                # No current key = use setup endpoint
                resp = session.post(
                    f"{base_url}/admin/api/setup-api-key",
                    json={
                        "api_key": new_api_key,
                        "api_key_confirm": new_api_key,
                    },
                    timeout=2,
                )
                return resp.status_code == 200

            # Update via admin global-settings API
            resp = session.post(
                f"{base_url}/admin/api/global-settings",
                json={"api_key": new_api_key},
                timeout=2,
            )
            return resp.status_code == 200

        except requests.RequestException as e:
            logger.debug(f"Failed to update API key on running server: {e}")
            return False

    def load_server_settings(self) -> dict:
        """Load model_dir and port from server's settings.json.

        Returns:
            {"model_dir": str, "port": int} or empty dict if not found
        """
        settings_file = Path(self.base_path).expanduser() / "settings.json"
        if not settings_file.exists():
            return {}

        try:
            with open(settings_file) as f:
                data = json.load(f)
            return {
                "model_dir": data.get("model", {}).get("model_dir"),
                "port": data.get("server", {}).get("port", 8000),
            }
        except (json.JSONDecodeError, OSError) as e:
            return {}

    def sync_from_server_settings(self):
        """Sync port and model_dir from server's settings.json to app config.

        Call this before showing Preferences/Welcome to display server's current settings.
        """
        server_settings = self.load_server_settings()

        if "port" in server_settings and server_settings["port"]:
            self.port = server_settings["port"]

        if "model_dir" in server_settings and server_settings["model_dir"]:
            self.model_dir = server_settings["model_dir"]

    def sync_model_dir_to_server_settings(self):
        """Write app's model_dir to server's settings.json if model_dirs not already set.

        Called after the welcome screen sets a model directory, so the server
        picks it up from settings.json instead of a CLI flag.
        """
        if not self.model_dir:
            return
        settings_file = Path(self.base_path).expanduser() / "settings.json"
        data = {}
        if settings_file.exists():
            try:
                with open(settings_file) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        if "model" not in data:
            data["model"] = {}
        existing_dirs = data["model"].get("model_dirs", [])
        if not existing_dirs:
            data["model"]["model_dirs"] = [self.model_dir]
            data["model"]["model_dir"] = self.model_dir
            settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(settings_file, "w") as f:
                json.dump(data, f, indent=2)

    def build_serve_args(self) -> list:
        """Build command line arguments for omlx serve.

        Note: --model-dir is intentionally NOT passed here. The server reads
        model_dirs from settings.json directly. Passing --model-dir would
        overwrite multi-directory settings saved via the web admin.
        """
        base = str(Path(self.base_path).expanduser())
        args = [
            "serve",
            "--base-path", base,
            "--port", str(self.port),
        ]
        return args
