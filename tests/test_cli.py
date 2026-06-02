# SPDX-License-Identifier: Apache-2.0
"""
CLI tests for oMLX.

Tests CLI argument parsing, command setup, and help text.
Note: Configuration validation tests are in test_config.py.
"""

import argparse
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omlx._version import __version__


class TestCLIModule:
    """Tests for CLI module existence and basic functionality."""

    def test_cli_module_importable(self):
        """Test that CLI module can be imported."""
        from omlx import cli
        assert hasattr(cli, "main")

    def test_cli_has_serve_command(self):
        """Test that CLI has serve command setup."""
        from omlx import cli
        # The module should have the main entry point
        assert callable(cli.main)


class TestCLIHelp:
    """Tests for CLI help functionality."""

    def test_main_help(self):
        """Test main CLI help output."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should succeed with help
        assert result.returncode == 0
        # Should show available commands
        assert "serve" in result.stdout.lower()

    def test_main_version(self):
        """Test main CLI version output."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == __version__
        assert result.stderr == ""

    def test_serve_help(self):
        """Test serve command help output."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should succeed with help
        assert result.returncode == 0
        # Should show serve options
        stdout_lower = result.stdout.lower()
        assert "host" in stdout_lower
        assert "port" in stdout_lower
        assert "model-dir" in stdout_lower


class TestCLIEntryPoint:
    """Tests for CLI entry point functionality."""

    def test_module_runnable(self):
        """Test that CLI module is runnable."""
        # Should not crash when running with --help
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_invalid_command_error(self):
        """Test error handling for invalid command."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "invalid_command"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should fail with non-zero exit code
        assert result.returncode != 0

    def test_no_command_shows_help(self):
        """Test that no command shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should exit with non-zero (no command provided)
        assert result.returncode != 0


class TestServeCommandOptions:
    """Tests for serve command options via help output."""

    def test_serve_has_model_dir_option(self):
        """Test that serve command has --model-dir option."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--model-dir" in result.stdout

    def test_serve_has_no_max_memory_options(self):
        """The --max-model-memory and --max-process-memory CLI flags are removed."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--max-model-memory" not in result.stdout
        assert "--max-process-memory" not in result.stdout

    def test_serve_no_model_specific_options(self):
        """Test that serve command does not have model-specific options (managed via admin page)."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # These options are now managed via admin page, not CLI
        assert "--pin" not in result.stdout
        assert "--default-model" not in result.stdout
        assert "--max-tokens" not in result.stdout
        assert "--temperature" not in result.stdout
        assert "--top-p" not in result.stdout
        assert "--top-k" not in result.stdout
        assert "--force-sampling" not in result.stdout

    def test_serve_has_host_port_options(self):
        """Test that serve command has --host and --port options."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_serve_has_scheduler_options(self):
        """Test that serve command has scheduler options."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--max-concurrent-requests" in result.stdout
        assert "--embedding-batch-size" in result.stdout

    def test_serve_has_cache_options(self):
        """Test that serve command has cache options."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--paged-ssd-cache-dir" in result.stdout
        assert "--paged-ssd-cache-max-size" in result.stdout
        assert "--no-cache" in result.stdout

    def test_serve_has_mcp_option(self):
        """Test that serve command has --mcp-config option."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--mcp-config" in result.stdout

    def test_serve_has_base_path_option(self):
        """Test that serve command has --base-path option."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--base-path" in result.stdout

    def test_serve_has_api_key_option(self):
        """Test that serve command has --api-key option."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--api-key" in result.stdout



class TestLaunchCommandOptions:
    """Tests for launch command options via help output."""

    def test_launch_has_host_port_options(self):
        """Test that launch command has --host and --port options."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "launch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_launch_has_model_option(self):
        """Test that launch command has --model option."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "launch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--model" in result.stdout

    def test_launch_lists_hermes(self):
        """Test that launch help lists Hermes as an available integration."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "launch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "hermes" in result.stdout
        assert "Hermes Agent" in result.stdout


class TestLaunchCommandFunction:
    """Tests for launch command runtime behavior."""

    def test_launch_command_passes_model_type_to_integration(self):
        """VLM model metadata should be forwarded to integrations."""
        from omlx.cli import launch_command

        integration = MagicMock()
        integration.display_name = "OpenCode"
        integration.is_installed.return_value = True

        health_response = MagicMock()
        health_response.raise_for_status.return_value = None

        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {
            "models": [
                {
                    "id": "qwen2.5-vl",
                    "model_type": "vlm",
                    "max_context_window": 32768,
                    "max_tokens": 8192,
                }
            ]
        }

        settings = MagicMock()
        settings.server.host = "127.0.0.1"
        settings.server.port = 8000

        args = argparse.Namespace(
            tool="opencode",
            host=None,
            port=None,
            api_key="test-key",
            model="qwen2.5-vl",
            tools_profile="coding",
        )

        with (
            patch("requests.get", side_effect=[health_response, status_response]),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(args)

        integration.launch.assert_called_once()
        ctx = integration.launch.call_args.args[0]
        assert ctx.host == "127.0.0.1"
        assert ctx.port == 8000
        assert ctx.api_key == "test-key"
        assert ctx.model == "qwen2.5-vl"
        assert ctx.tools_profile == "coding"
        assert ctx.context_window == 32768
        assert ctx.max_tokens == 8192
        assert ctx.model_type == "vlm"
        assert ctx.extra_args == ()

    def test_launch_command_resolves_alias_status_metadata(self):
        """Alias model IDs should keep status metadata from the real model."""
        from omlx.cli import launch_command

        integration = MagicMock()
        integration.display_name = "OpenCode"
        integration.is_installed.return_value = True

        health_response = MagicMock()
        health_response.raise_for_status.return_value = None

        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {
            "models": [
                {
                    "id": "qwen2.5-vl-raw",
                    "model_alias": "gpt-4o",
                    "model_type": "vlm",
                    "max_context_window": 32768,
                    "max_tokens": 8192,
                    "enable_thinking": False,
                }
            ]
        }

        settings = MagicMock()
        settings.server.host = "127.0.0.1"
        settings.server.port = 8000

        args = argparse.Namespace(
            tool="opencode",
            host=None,
            port=None,
            api_key="test-key",
            model="gpt-4o",
            tools_profile="coding",
        )

        with (
            patch("requests.get", side_effect=[health_response, status_response]),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(args)

        ctx = integration.launch.call_args.args[0]
        assert ctx.model == "gpt-4o"
        assert ctx.context_window == 32768
        assert ctx.max_tokens == 8192
        assert ctx.model_type == "vlm"
        assert ctx.reasoning is False

    def test_launch_command_forwards_extra_args(self):
        """Unknown CLI tokens (e.g. --resume <id>) should reach integration.launch."""
        from omlx.cli import launch_command

        integration = MagicMock()
        integration.display_name = "Claude Code"
        integration.is_installed.return_value = True

        health_response = MagicMock()
        health_response.raise_for_status.return_value = None

        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {
            "models": [
                {
                    "id": "qwen2.5-vl",
                    "model_type": "llm",
                    "max_context_window": 32768,
                    "max_tokens": 8192,
                }
            ]
        }

        settings = MagicMock()
        settings.server.host = "127.0.0.1"
        settings.server.port = 8000

        args = argparse.Namespace(
            tool="claude",
            host=None,
            port=None,
            api_key="test-key",
            model="qwen2.5-vl",
            tools_profile="coding",
        )

        with (
            patch("requests.get", side_effect=[health_response, status_response]),
            patch("omlx.integrations.get_integration", return_value=integration),
            patch("omlx.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(args, extra_args=["--resume", "abc123"])

        ctx = integration.launch.call_args.args[0]
        assert ctx.extra_args == ("--resume", "abc123")


class TestLaunchArgvParsing:
    """Tests for top-level argv parsing of `omlx launch ...`."""

    def test_serve_still_rejects_unknown_args(self):
        """Non-launch commands must keep strict argparse rejection."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--bogus-flag"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert "unrecognized arguments" in result.stderr or "--bogus-flag" in result.stderr


class TestServeCommandFunctions:
    """Tests for serve command function."""

    @staticmethod
    def _make_serve_args(tmp_path, host="127.0.0.1", port=8000, **overrides):
        defaults = {
            "model_dir": None,
            "host": host,
            "port": port,
            "log_level": None,
            "sse_keepalive_mode": None,
            "max_concurrent_requests": None,
            "embedding_batch_size": None,
            "paged_ssd_cache_dir": None,
            "paged_ssd_cache_max_size": None,
            "hot_cache_max_size": None,
            "no_cache": True,
            "initial_cache_blocks": None,
            "mcp_config": None,
            "hf_endpoint": None,
            "ms_endpoint": None,
            "http_proxy": None,
            "https_proxy": None,
            "no_proxy": None,
            "ca_bundle": None,
            "base_path": str(tmp_path),
            "api_key": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @staticmethod
    def _make_settings(tmp_path, host="127.0.0.1", port=8000):
        log_dir = tmp_path / "logs"
        settings = SimpleNamespace()
        settings.base_path = tmp_path
        settings.server = SimpleNamespace(host=host, port=port, log_level="info")
        settings.huggingface = SimpleNamespace(endpoint=None)
        settings.modelscope = SimpleNamespace(endpoint=None)
        settings.network = SimpleNamespace(
            http_proxy=None,
            https_proxy=None,
            no_proxy=None,
            ca_bundle=None,
        )
        settings.logging = SimpleNamespace(
            retention_days=7,
            get_log_dir=lambda base_path: log_dir,
        )
        settings.model = SimpleNamespace(
            get_model_dirs=lambda base_path: [tmp_path / "models"],
        )
        settings.memory = SimpleNamespace(memory_guard_tier="balanced")
        settings.mcp = SimpleNamespace(config_path=None)
        settings.cache = SimpleNamespace(
            enabled=False,
            get_ssd_cache_dir=lambda base_path: tmp_path / "cache",
            get_ssd_cache_max_size_bytes=lambda base_path: 0,
            get_hot_cache_max_size_bytes=lambda: 0,
        )
        settings.auth = SimpleNamespace(api_key=None)
        settings.ensure_directories = lambda: log_dir.mkdir(parents=True, exist_ok=True)
        settings.validate = lambda: []
        settings.save = MagicMock()
        settings.to_scheduler_config = lambda: SimpleNamespace(
            paged_ssd_cache_dir=None,
            paged_ssd_cache_max_size=0,
            hot_cache_max_size=0,
        )
        return settings

    @staticmethod
    def _reserve_port(host="127.0.0.1"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        sock.listen(1)
        return sock

    def test_serve_command_exists(self):
        """Test that serve_command function exists."""
        from omlx.cli import serve_command
        assert callable(serve_command)

    def test_serve_model_dir_optional_with_default(self):
        """Test that serve --model-dir is optional with default ~/.omlx/models."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should show that model-dir has a default
        assert "default" in result.stdout.lower()
        # Help text should mention ~/.omlx/models or similar
        assert ".omlx" in result.stdout or "model" in result.stdout.lower()

    def test_invalid_embedding_batch_size_is_not_persisted(self, tmp_path):
        """Invalid CLI scheduler values should fail before saving settings.json."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "omlx.cli",
                "serve",
                "--base-path",
                str(tmp_path),
                "--embedding-batch-size",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert "embedding_batch_size" in result.stdout
        assert not (tmp_path / "settings.json").exists()

    def test_serve_exits_on_port_conflict_before_importing_server(
        self, tmp_path, monkeypatch
    ):
        """Port conflicts should fail before server import can preload pinned models."""
        import uvicorn

        from omlx.cli import serve_command

        listener = self._reserve_port()
        host, port = listener.getsockname()
        settings = self._make_settings(tmp_path, host=host, port=port)
        args = self._make_serve_args(tmp_path, host=host, port=port)
        previous_server = sys.modules.pop("omlx.server", None)
        events = []

        original_bind_socket = uvicorn.Config.bind_socket

        def tracking_bind_socket(config):
            events.append("bind")
            return original_bind_socket(config)

        monkeypatch.setattr("omlx.settings.init_settings", lambda **kwargs: settings)
        monkeypatch.setattr(
            "omlx.logging_config.configure_file_logging",
            lambda **kwargs: None,
        )
        monkeypatch.setattr("faulthandler.enable", lambda *args, **kwargs: None)
        monkeypatch.setattr("uvicorn.Config.bind_socket", tracking_bind_socket)
        try:
            with pytest.raises(SystemExit) as exc:
                serve_command(args)

            assert exc.value.code != 0
            assert events == ["bind"]
            assert "omlx.server" not in sys.modules
        finally:
            listener.close()
            if previous_server is not None:
                sys.modules["omlx.server"] = previous_server

    def test_serve_hands_prebound_socket_to_uvicorn(self, tmp_path, monkeypatch):
        """Successful serve startup should pass the pre-bound socket into uvicorn."""
        import omlx
        import uvicorn

        from omlx.cli import serve_command

        host, port = "127.0.0.1", 0
        settings = self._make_settings(tmp_path, host=host, port=port)
        args = self._make_serve_args(tmp_path, host=host, port=port)
        events = []

        fake_server = ModuleType("omlx.server")

        async def app(scope, receive, send):
            return None

        def fake_init_server(**kwargs):
            events.append("init")

        fake_server.app = app
        fake_server.init_server = MagicMock(side_effect=fake_init_server)
        monkeypatch.setitem(sys.modules, "omlx.server", fake_server)
        monkeypatch.setattr(omlx, "server", fake_server, raising=False)

        fake_mlx = ModuleType("mlx")
        fake_mlx_core = ModuleType("mlx.core")
        fake_mlx_core.device_info = lambda: {"memory_size": 0}
        fake_mlx_core.set_cache_limit = MagicMock()
        fake_mlx.core = fake_mlx_core
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)

        monkeypatch.setattr("omlx.settings.init_settings", lambda **kwargs: settings)
        monkeypatch.setattr(
            "omlx.logging_config.configure_file_logging",
            lambda **kwargs: None,
        )
        monkeypatch.setattr("faulthandler.enable", lambda *args, **kwargs: None)
        captured = {}
        original_bind_socket = uvicorn.Config.bind_socket

        def tracking_bind_socket(config):
            sock = original_bind_socket(config)
            events.append("bind")
            return sock

        def fake_run(self, sockets=None):
            self.config.load()
            events.append("run")
            captured["socket_name"] = sockets[0].getsockname()
            captured["socket_count"] = len(sockets)

        monkeypatch.setattr("uvicorn.Config.bind_socket", tracking_bind_socket)
        monkeypatch.setattr("uvicorn.Server.run", fake_run)

        serve_command(args)

        fake_server.init_server.assert_called_once()
        assert events == ["bind", "init", "run"]
        assert captured["socket_count"] == 1
        assert captured["socket_name"][0] == host
        assert captured["socket_name"][1] > 0


class TestHasCliOverrides:
    """Tests for _has_cli_overrides() — detects explicitly passed CLI args."""

    @staticmethod
    def _make_args(**kwargs):
        """Namespace with all serve defaults (None), then apply overrides."""
        defaults = {
            "model_dir": None,
            "port": None,
            "host": None,
            "log_level": None,
            "embedding_batch_size": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_no_overrides_returns_false(self):
        from omlx.cli import _has_cli_overrides
        assert _has_cli_overrides(self._make_args()) is False

    def test_host_explicit(self):
        from omlx.cli import _has_cli_overrides
        assert _has_cli_overrides(self._make_args(host="0.0.0.0")) is True
        # Even the default value, when explicitly passed, counts as override
        assert _has_cli_overrides(self._make_args(host="127.0.0.1")) is True

    def test_port_explicit(self):
        from omlx.cli import _has_cli_overrides
        assert _has_cli_overrides(self._make_args(port=9000)) is True
        assert _has_cli_overrides(self._make_args(port=8000)) is True

    def test_model_dir_explicit(self):
        from omlx.cli import _has_cli_overrides
        assert _has_cli_overrides(self._make_args(model_dir="/tmp/models")) is True

    def test_log_level_explicit(self):
        from omlx.cli import _has_cli_overrides
        assert _has_cli_overrides(self._make_args(log_level="info")) is True
        assert _has_cli_overrides(self._make_args(log_level="debug")) is True

    def test_embedding_batch_size_explicit(self):
        from omlx.cli import _has_cli_overrides
        assert _has_cli_overrides(self._make_args(embedding_batch_size=4)) is True

    def test_multiple_overrides(self):
        from omlx.cli import _has_cli_overrides
        assert _has_cli_overrides(self._make_args(host="0.0.0.0", port=9000)) is True

    def test_empty_namespace(self):
        from omlx.cli import _has_cli_overrides
        assert _has_cli_overrides(argparse.Namespace()) is False


class TestCLIDocstrings:
    """Tests for CLI module docstrings and descriptions."""

    def test_main_has_description(self):
        """Test that main help has description."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should have some description
        assert "omlx" in result.stdout.lower() or "llm" in result.stdout.lower()

    def test_serve_has_description(self):
        """Test that serve command has description."""
        result = subprocess.run(
            [sys.executable, "-m", "omlx.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should describe multi-model serving
        assert "multi-model" in result.stdout.lower() or "server" in result.stdout.lower()
