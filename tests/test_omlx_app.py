# SPDX-License-Identifier: Apache-2.0
"""
Tests for oMLX menubar app components (packaging/omlx_app/).

These tests cover ServerConfig and ServerManager without requiring
the full PyObjC GUI framework.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import the modules under test
sys.path.insert(0, str(Path(__file__).parent.parent / "packaging"))
from omlx_app.config import ServerConfig, get_app_support_dir, get_config_path, get_log_path
from omlx_app.server_manager import PortConflict, ServerManager, ServerStatus


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    def test_default_values(self):
        """Test ServerConfig default values."""
        config = ServerConfig()
        assert config.base_path == str(Path.home() / ".omlx")
        assert config.port == 8000
        assert config.model_dir == ""
        assert config.launch_at_login is False
        assert config.start_server_on_launch is False

    def test_custom_values(self):
        """Test ServerConfig with custom values."""
        config = ServerConfig(
            base_path="/custom/base",
            port=9000,
            model_dir="/path/to/models",
            launch_at_login=True,
            start_server_on_launch=True,
        )
        assert config.base_path == "/custom/base"
        assert config.port == 9000
        assert config.model_dir == "/path/to/models"
        assert config.launch_at_login is True
        assert config.start_server_on_launch is True

    def test_to_dict(self):
        """Test ServerConfig.to_dict() method."""
        config = ServerConfig(
            base_path="/custom/base",
            port=8080,
            model_dir="/models",
        )
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["base_path"] == "/custom/base"
        assert result["port"] == 8080
        assert result["model_dir"] == "/models"
        expected_keys = {
            "base_path", "port", "model_dir",
            "launch_at_login", "start_server_on_launch"
        }
        assert set(result.keys()) == expected_keys

    def test_from_dict(self):
        """Test ServerConfig.from_dict() method."""
        data = {
            "base_path": "/custom/base",
            "port": 3000,
            "model_dir": "/custom/models",
            "launch_at_login": True,
            "start_server_on_launch": False,
        }
        config = ServerConfig.from_dict(data)

        assert config.base_path == "/custom/base"
        assert config.port == 3000
        assert config.model_dir == "/custom/models"
        assert config.launch_at_login is True
        assert config.start_server_on_launch is False

    def test_from_dict_ignores_unknown_keys(self):
        """Test that from_dict ignores unknown keys."""
        data = {
            "base_path": "/test",
            "port": 8000,
            "unknown_key": "should_be_ignored",
            "another_unknown": 123,
        }
        config = ServerConfig.from_dict(data)

        assert config.base_path == "/test"
        assert config.port == 8000
        assert not hasattr(config, "unknown_key")
        assert not hasattr(config, "another_unknown")

    def test_from_dict_partial_data(self):
        """Test from_dict with partial data uses defaults."""
        data = {"port": 9999}
        config = ServerConfig.from_dict(data)

        assert config.port == 9999
        assert config.base_path == str(Path.home() / ".omlx")  # default
        assert config.model_dir == ""  # default

    def test_save_and_load(self, tmp_path: Path):
        """Test save and load round-trip."""
        config_file = tmp_path / "config.json"

        with patch("omlx_app.config.get_config_path", return_value=config_file):
            original = ServerConfig(
                base_path="/test/base",
                port=8888,
                model_dir="/test/models",
            )
            original.save()

            assert config_file.exists()

            loaded = ServerConfig.load()
            assert loaded.base_path == original.base_path
            assert loaded.port == original.port
            assert loaded.model_dir == original.model_dir

    def test_load_returns_default_on_missing_file(self, tmp_path: Path):
        """Test load returns default config when file doesn't exist."""
        config_file = tmp_path / "nonexistent.json"

        with patch("omlx_app.config.get_config_path", return_value=config_file):
            config = ServerConfig.load()
            assert config.base_path == str(Path.home() / ".omlx")
            assert config.port == 8000

    def test_load_returns_default_on_invalid_json(self, tmp_path: Path):
        """Test load returns default config on invalid JSON."""
        config_file = tmp_path / "config.json"
        config_file.write_text("{ invalid json }")

        with patch("omlx_app.config.get_config_path", return_value=config_file):
            config = ServerConfig.load()
            assert config.base_path == str(Path.home() / ".omlx")
            assert config.port == 8000

    def test_get_effective_model_dir_custom(self):
        """Test get_effective_model_dir with custom model_dir."""
        config = ServerConfig(model_dir="/custom/models")
        assert config.get_effective_model_dir() == "/custom/models"

    def test_get_effective_model_dir_default(self):
        """Test get_effective_model_dir uses base_path/models when empty."""
        config = ServerConfig(base_path="/test/base", model_dir="")
        assert config.get_effective_model_dir() == str(
            Path("/test/base").expanduser() / "models"
        )

    def test_build_serve_args_default(self):
        """Test build_serve_args with default config."""
        config = ServerConfig(base_path="/test/base", port=8000, model_dir="")
        args = config.build_serve_args()

        base = str(Path("/test/base").expanduser())
        assert args == ["serve", "--base-path", base, "--port", "8000"]

    def test_build_serve_args_with_model_dir(self):
        """Test build_serve_args includes model_dir when set."""
        config = ServerConfig(
            base_path="/test/base",
            port=9000,
            model_dir="/path/to/models",
        )
        args = config.build_serve_args()

        base = str(Path("/test/base").expanduser())
        model_dir = str(Path("/path/to/models").expanduser())
        assert args == [
            "serve",
            "--base-path", base,
            "--port", "9000",
            "--model-dir", model_dir,
        ]

    def test_build_serve_args_preserves_order(self):
        """Test that build_serve_args has consistent argument order."""
        config = ServerConfig(base_path="/base", model_dir="/models")
        args = config.build_serve_args()

        assert args[0] == "serve"
        assert args[1] == "--base-path"
        assert args[3] == "--port"

    def test_is_first_run(self, tmp_path: Path):
        """Test is_first_run property."""
        config_file = tmp_path / "config.json"

        with patch("omlx_app.config.get_config_path", return_value=config_file):
            config = ServerConfig()
            assert config.is_first_run is True

            config_file.write_text("{}")
            assert config.is_first_run is False

    def test_get_server_api_key(self, tmp_path: Path):
        """Test reading API key from server settings.json."""
        config = ServerConfig(base_path=str(tmp_path))
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "auth": {"api_key": "test-key-123"}
        }))

        assert config.get_server_api_key() == "test-key-123"

    def test_get_server_api_key_missing(self, tmp_path: Path):
        """Test get_server_api_key returns None when no settings file."""
        config = ServerConfig(base_path=str(tmp_path))
        assert config.get_server_api_key() is None

    def test_load_server_settings(self, tmp_path: Path):
        """Test loading settings from server's settings.json."""
        config = ServerConfig(base_path=str(tmp_path))
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "model": {"model_dir": "/server/models"},
            "server": {"port": 9000},
        }))

        result = config.load_server_settings()
        assert result["model_dir"] == "/server/models"
        assert result["port"] == 9000

    def test_load_server_settings_missing_file(self, tmp_path: Path):
        """Test load_server_settings returns empty dict when no file."""
        config = ServerConfig(base_path=str(tmp_path))
        assert config.load_server_settings() == {}

    def test_sync_from_server_settings(self, tmp_path: Path):
        """Test syncing port and model_dir from server settings.json."""
        config = ServerConfig(base_path=str(tmp_path), port=8000, model_dir="")
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "model": {"model_dir": "/synced/models"},
            "server": {"port": 9999},
        }))

        config.sync_from_server_settings()
        assert config.port == 9999
        assert config.model_dir == "/synced/models"

    def test_sync_from_server_settings_no_file(self, tmp_path: Path):
        """Test sync does nothing when no settings.json exists."""
        config = ServerConfig(base_path=str(tmp_path), port=8000, model_dir="")
        config.sync_from_server_settings()
        assert config.port == 8000
        assert config.model_dir == ""

    def test_set_server_api_key_new_file(self, tmp_path: Path):
        """Test setting API key creates settings.json if not exists."""
        config = ServerConfig(base_path=str(tmp_path))
        config.set_server_api_key("my-secret-key")

        settings_file = tmp_path / "settings.json"
        assert settings_file.exists()
        with open(settings_file) as f:
            data = json.load(f)
        assert data["auth"]["api_key"] == "my-secret-key"

    def test_set_server_api_key_existing_file(self, tmp_path: Path):
        """Test setting API key preserves other settings."""
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({
            "server": {"port": 9000},
            "auth": {"api_key": "old-key"},
        }))

        config = ServerConfig(base_path=str(tmp_path))
        config.set_server_api_key("new-key")

        with open(settings_file) as f:
            data = json.load(f)
        assert data["auth"]["api_key"] == "new-key"
        assert data["server"]["port"] == 9000  # preserved

    def test_set_and_get_api_key_roundtrip(self, tmp_path: Path):
        """Test set then get API key roundtrip."""
        config = ServerConfig(base_path=str(tmp_path))
        config.set_server_api_key("roundtrip-key")
        assert config.get_server_api_key() == "roundtrip-key"

    @patch("omlx_app.config.requests.Session")
    def test_update_server_api_key_runtime_success(self, mock_session_cls, tmp_path):
        """Test runtime API key update on running server."""
        config = ServerConfig(base_path=str(tmp_path))
        # Set current key so login works
        config.set_server_api_key("current-key")

        mock_session = Mock()
        mock_session_cls.return_value = mock_session
        mock_session.post.side_effect = [
            Mock(status_code=200),  # login
            Mock(status_code=200),  # settings update
        ]

        result = config.update_server_api_key_runtime("new-key")
        assert result is True
        assert mock_session.post.call_count == 2

    @patch("omlx_app.config.requests.Session")
    def test_update_server_api_key_runtime_server_down(self, mock_session_cls, tmp_path):
        """Test runtime update returns False when server unreachable."""
        import requests as req
        config = ServerConfig(base_path=str(tmp_path))
        config.set_server_api_key("current-key")

        mock_session = Mock()
        mock_session_cls.return_value = mock_session
        mock_session.post.side_effect = req.ConnectionError("refused")

        result = config.update_server_api_key_runtime("new-key")
        assert result is False


class TestServerStatus:
    """Tests for ServerStatus enum."""

    def test_status_values(self):
        """Test all ServerStatus values exist."""
        assert ServerStatus.STOPPED.value == "stopped"
        assert ServerStatus.STARTING.value == "starting"
        assert ServerStatus.RUNNING.value == "running"
        assert ServerStatus.STOPPING.value == "stopping"
        assert ServerStatus.ERROR.value == "error"

    def test_status_comparison(self):
        """Test status enum comparison."""
        assert ServerStatus.STOPPED != ServerStatus.RUNNING
        assert ServerStatus.RUNNING == ServerStatus.RUNNING


class TestServerManager:
    """Tests for ServerManager class."""

    @pytest.fixture
    def config(self, tmp_path) -> ServerConfig:
        """Create a test config."""
        return ServerConfig(
            base_path=str(tmp_path / "base"),
            port=8765,
            model_dir=str(tmp_path / "models"),
        )

    @pytest.fixture
    def manager(self, config: ServerConfig) -> ServerManager:
        """Create a ServerManager instance."""
        return ServerManager(config)

    def test_initial_status(self, manager: ServerManager):
        """Test initial server status is STOPPED."""
        assert manager.status == ServerStatus.STOPPED
        assert manager.error_message is None

    def test_status_property(self, manager: ServerManager):
        """Test status property."""
        assert manager.status == ServerStatus.STOPPED
        manager._status = ServerStatus.RUNNING
        assert manager.status == ServerStatus.RUNNING

    def test_error_message_property(self, manager: ServerManager):
        """Test error_message property."""
        assert manager.error_message is None
        manager._error_message = "Test error"
        assert manager.error_message == "Test error"

    def test_set_status_callback(self, manager: ServerManager):
        """Test status callback registration."""
        callback = Mock()
        manager.set_status_callback(callback)
        assert manager._status_callback == callback

    def test_update_status_calls_callback(self, manager: ServerManager):
        """Test that _update_status calls the registered callback."""
        callback = Mock()
        manager.set_status_callback(callback)

        manager._update_status(ServerStatus.RUNNING)

        callback.assert_called_once_with(ServerStatus.RUNNING)
        assert manager.status == ServerStatus.RUNNING

    def test_update_status_with_error(self, manager: ServerManager):
        """Test _update_status with error message."""
        manager._update_status(ServerStatus.ERROR, "Connection failed")

        assert manager.status == ServerStatus.ERROR
        assert manager.error_message == "Connection failed"

    def test_update_status_callback_exception_handled(self, manager: ServerManager):
        """Test that callback exceptions are handled gracefully."""
        def bad_callback(status):
            raise RuntimeError("Callback error")

        manager.set_status_callback(bad_callback)
        # Should not raise
        manager._update_status(ServerStatus.RUNNING)
        assert manager.status == ServerStatus.RUNNING

    def test_get_health_url(self, manager: ServerManager):
        """Test health URL generation."""
        url = manager._get_health_url()
        assert url == "http://127.0.0.1:8765/health"

    def test_get_api_url(self, manager: ServerManager):
        """Test API URL generation."""
        url = manager.get_api_url()
        assert url == "http://127.0.0.1:8765"

    def test_update_config(self, manager: ServerManager):
        """Test config update."""
        new_config = ServerConfig(base_path="/new/base", port=9999, model_dir="/new/path")
        manager.update_config(new_config)

        assert manager.config.base_path == "/new/base"
        assert manager.config.port == 9999
        assert manager.config.model_dir == "/new/path"

    def test_is_running_no_process(self, manager: ServerManager):
        """Test is_running when no process exists."""
        assert manager.is_running() is False

    def test_is_running_with_process(self, manager: ServerManager):
        """Test is_running with mock process."""
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process still running
        manager._process = mock_process

        assert manager.is_running() is True

    def test_is_running_process_terminated(self, manager: ServerManager):
        """Test is_running when process has terminated."""
        mock_process = Mock()
        mock_process.poll.return_value = 0  # Process exited
        manager._process = mock_process

        assert manager.is_running() is False

    @patch("omlx_app.server_manager.requests.get")
    def test_check_health_success(self, mock_get, manager: ServerManager):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert manager.check_health() is True
        mock_get.assert_called_once_with("http://127.0.0.1:8765/health", timeout=2)

    @patch("omlx_app.server_manager.requests.get")
    def test_check_health_failure(self, mock_get, manager: ServerManager):
        """Test failed health check (non-200 status)."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        assert manager.check_health() is False

    @patch("omlx_app.server_manager.requests.get")
    def test_check_health_connection_error(self, mock_get, manager: ServerManager):
        """Test health check with connection error."""
        import requests
        mock_get.side_effect = requests.RequestException("Connection refused")

        assert manager.check_health() is False

    def test_start_already_running(self, manager: ServerManager):
        """Test start returns False when already running."""
        manager._status = ServerStatus.RUNNING
        result = manager.start()
        assert result is False

    def test_start_already_starting(self, manager: ServerManager):
        """Test start returns False when already starting."""
        manager._status = ServerStatus.STARTING
        result = manager.start()
        assert result is False

    @patch.object(ServerManager, "_is_port_in_use", return_value=False)
    @patch("omlx_app.server_manager.subprocess.Popen")
    @patch("omlx_app.server_manager.get_log_path")
    @patch("builtins.open", new_callable=MagicMock)
    def test_start_success(
        self, mock_open, mock_log_path, mock_popen, mock_port_check, manager: ServerManager, tmp_path
    ):
        """Test successful server start."""
        mock_log_path.return_value = tmp_path / "server.log"
        mock_process = Mock()
        mock_process.pid = 12345
        mock_popen.return_value = mock_process

        result = manager.start()

        assert result is True
        assert manager.status == ServerStatus.STARTING
        mock_popen.assert_called_once()

        # Verify command includes omlx.cli
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert "-m" in cmd
        assert "omlx.cli" in cmd
        assert "serve" in cmd

        # Clean up the health check thread
        manager._stop_health_check.set()

    @patch.object(ServerManager, "_is_port_in_use", return_value=False)
    @patch("omlx_app.server_manager.subprocess.Popen")
    @patch("omlx_app.server_manager.get_log_path")
    def test_start_popen_exception(
        self, mock_log_path, mock_popen, mock_port_check, manager: ServerManager, tmp_path
    ):
        """Test start handles Popen exception."""
        mock_log_path.return_value = tmp_path / "server.log"
        mock_popen.side_effect = OSError("Failed to start process")

        result = manager.start()

        assert result is False
        assert manager.status == ServerStatus.ERROR
        assert "Failed to start process" in manager.error_message

    def test_stop_when_stopped(self, manager: ServerManager):
        """Test stop returns False when already stopped."""
        result = manager.stop()
        assert result is False

    def test_stop_when_error(self, manager: ServerManager):
        """Test stop returns False when in error state."""
        manager._status = ServerStatus.ERROR
        result = manager.stop()
        assert result is False

    def test_stop_no_process(self, manager: ServerManager):
        """Test stop when status is running but no process."""
        manager._status = ServerStatus.RUNNING
        manager._process = None

        result = manager.stop()

        assert result is True
        assert manager.status == ServerStatus.STOPPED

    @patch("omlx_app.server_manager.os.killpg")
    @patch("omlx_app.server_manager.os.getpgid")
    def test_stop_graceful(
        self, mock_getpgid, mock_killpg, manager: ServerManager
    ):
        """Test graceful stop with SIGTERM."""
        mock_getpgid.return_value = 12345
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.wait.return_value = 0

        manager._status = ServerStatus.RUNNING
        manager._process = mock_process

        result = manager.stop()

        assert result is True
        assert manager.status == ServerStatus.STOPPED
        mock_killpg.assert_called_with(12345, signal.SIGTERM)

    @patch("omlx_app.server_manager.os.killpg")
    @patch("omlx_app.server_manager.os.getpgid")
    def test_stop_force_kill_on_timeout(
        self, mock_getpgid, mock_killpg, manager: ServerManager
    ):
        """Test force kill when graceful stop times out."""
        mock_getpgid.return_value = 12345
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="test", timeout=10),
            0,  # Second call succeeds after SIGKILL
        ]

        manager._status = ServerStatus.RUNNING
        manager._process = mock_process

        result = manager.stop(timeout=0.1)

        assert result is True
        assert manager.status == ServerStatus.STOPPED
        # Should have called SIGTERM first, then SIGKILL
        assert mock_killpg.call_count == 2
        calls = mock_killpg.call_args_list
        assert calls[0] == ((12345, signal.SIGTERM),)
        assert calls[1] == ((12345, signal.SIGKILL),)

    @patch("omlx_app.server_manager.os.killpg")
    @patch("omlx_app.server_manager.os.getpgid")
    def test_stop_closes_log_file(
        self, mock_getpgid, mock_killpg, manager: ServerManager
    ):
        """Test that stop closes the log file handle."""
        mock_getpgid.return_value = 12345
        mock_process = Mock()
        mock_process.pid = 12345
        mock_process.wait.return_value = 0

        mock_log_handle = Mock()
        manager._status = ServerStatus.RUNNING
        manager._process = mock_process
        manager._log_file_handle = mock_log_handle

        manager.stop()

        mock_log_handle.close.assert_called_once()
        assert manager._log_file_handle is None

    @patch.object(ServerManager, "stop")
    @patch.object(ServerManager, "start")
    def test_restart(self, mock_start, mock_stop, manager: ServerManager):
        """Test restart calls stop then start."""
        mock_start.return_value = True

        with patch("time.sleep"):  # Don't actually sleep in tests
            result = manager.restart()

        mock_stop.assert_called_once()
        mock_start.assert_called_once()
        assert result is True


class TestPortConflict:
    """Tests for port conflict detection and handling."""

    @pytest.fixture
    def config(self, tmp_path) -> ServerConfig:
        """Create a test config."""
        return ServerConfig(
            base_path=str(tmp_path / "base"),
            port=8765,
            model_dir=str(tmp_path / "models"),
        )

    @pytest.fixture
    def manager(self, config: ServerConfig) -> ServerManager:
        """Create a ServerManager instance."""
        return ServerManager(config)

    def test_port_conflict_dataclass(self):
        """Test PortConflict dataclass fields."""
        conflict = PortConflict(pid=12345, is_omlx=True)
        assert conflict.pid == 12345
        assert conflict.is_omlx is True

        conflict_no_pid = PortConflict(pid=None, is_omlx=False)
        assert conflict_no_pid.pid is None
        assert conflict_no_pid.is_omlx is False

    @patch.object(ServerManager, "_find_port_owner_pid", return_value=12345)
    @patch.object(ServerManager, "_is_omlx_server", return_value=True)
    @patch.object(ServerManager, "_is_port_in_use", return_value=True)
    def test_start_returns_port_conflict_omlx(
        self, mock_port, mock_omlx, mock_pid, manager: ServerManager
    ):
        """Test start returns PortConflict when port is used by oMLX."""
        result = manager.start()
        assert isinstance(result, PortConflict)
        assert result.pid == 12345
        assert result.is_omlx is True

    @patch.object(ServerManager, "_find_port_owner_pid", return_value=99999)
    @patch.object(ServerManager, "_is_omlx_server", return_value=False)
    @patch.object(ServerManager, "_is_port_in_use", return_value=True)
    def test_start_returns_port_conflict_not_omlx(
        self, mock_port, mock_omlx, mock_pid, manager: ServerManager
    ):
        """Test start returns PortConflict when port is used by another app."""
        result = manager.start()
        assert isinstance(result, PortConflict)
        assert result.pid == 99999
        assert result.is_omlx is False

    @patch.object(ServerManager, "_is_omlx_server", return_value=True)
    def test_adopt_success(self, mock_omlx, manager: ServerManager):
        """Test adopting an existing oMLX server."""
        result = manager.adopt()
        assert result is True
        assert manager.status == ServerStatus.RUNNING
        assert manager._adopted is True
        assert manager._process is None

        # Clean up the health check thread
        manager._stop_health_check.set()
        if manager._health_check_thread:
            manager._health_check_thread.join(timeout=2)

    @patch.object(ServerManager, "_is_omlx_server", return_value=False)
    def test_adopt_failure(self, mock_omlx, manager: ServerManager):
        """Test adopt fails when target is not oMLX."""
        result = manager.adopt()
        assert result is False
        assert manager._adopted is False

    def test_stop_adopted_server(self, manager: ServerManager):
        """Test stopping an adopted server doesn't kill processes."""
        manager._adopted = True
        manager._status = ServerStatus.RUNNING
        manager._process = None

        result = manager.stop()

        assert result is True
        assert manager.status == ServerStatus.STOPPED
        assert manager._adopted is False

    def test_is_running_adopted(self, manager: ServerManager):
        """Test is_running for adopted servers."""
        manager._adopted = True
        manager._status = ServerStatus.RUNNING
        assert manager.is_running() is True

        manager._status = ServerStatus.ERROR
        assert manager.is_running() is False

    @patch("omlx_app.server_manager.subprocess.run")
    def test_find_port_owner_pid(self, mock_run, manager: ServerManager):
        """Test finding PID of the process on the port."""
        mock_run.return_value = Mock(returncode=0, stdout="12345\n")
        assert manager._find_port_owner_pid() == 12345

    @patch("omlx_app.server_manager.subprocess.run")
    def test_find_port_owner_pid_no_process(self, mock_run, manager: ServerManager):
        """Test finding PID when no process is listening."""
        mock_run.return_value = Mock(returncode=1, stdout="")
        assert manager._find_port_owner_pid() is None

    @patch("omlx_app.server_manager.os.kill")
    def test_kill_external_server(self, mock_kill, manager: ServerManager):
        """Test killing an external server process."""
        # Process exits after SIGTERM
        mock_kill.side_effect = [None, OSError("No such process")]
        result = manager._kill_external_server(12345)
        assert result is True
        mock_kill.assert_any_call(12345, signal.SIGTERM)


class TestPathHelpers:
    """Tests for path helper functions."""

    @patch("omlx_app.config.Path.home")
    def test_get_app_support_dir(self, mock_home, tmp_path):
        """Test get_app_support_dir creates directory."""
        mock_home.return_value = tmp_path

        result = get_app_support_dir()

        expected = tmp_path / "Library" / "Application Support" / "oMLX"
        assert result == expected
        assert expected.exists()

    @patch("omlx_app.config.get_app_support_dir")
    def test_get_config_path(self, mock_get_app_support):
        """Test get_config_path returns correct path."""
        mock_get_app_support.return_value = Path("/app/support")

        result = get_config_path()

        assert result == Path("/app/support/config.json")

    @patch("omlx_app.config.get_app_support_dir")
    def test_get_log_path_creates_log_dir(self, mock_get_app_support, tmp_path):
        """Test get_log_path creates logs directory."""
        mock_get_app_support.return_value = tmp_path

        result = get_log_path()

        assert result == tmp_path / "logs" / "server.log"
        assert (tmp_path / "logs").exists()
