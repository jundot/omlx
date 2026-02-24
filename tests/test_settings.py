# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.settings module."""

import json
import os
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from omlx.settings import (
    AuthSettings,
    CacheSettings,
    ClaudeCodeSettings,
    GlobalSettings,
    LoggingSettings,
    MCPSettings,
    ModelSettings,
    SamplingSettings,
    SchedulerSettings,
    ServerSettings,
    get_settings,
    get_ssd_capacity,
    get_system_memory,
    init_settings,
    reset_settings,
)


class TestServerSettings:
    """Tests for ServerSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = ServerSettings()
        assert settings.host == "127.0.0.1"
        assert settings.port == 8000
        assert settings.log_level == "info"
        assert settings.cors_origins == ["*"]

    def test_custom_values(self):
        """Test custom values."""
        settings = ServerSettings(
            host="0.0.0.0",
            port=9000,
            log_level="debug",
            cors_origins=["https://example.com"],
        )
        assert settings.host == "0.0.0.0"
        assert settings.port == 9000
        assert settings.log_level == "debug"
        assert settings.cors_origins == ["https://example.com"]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = ServerSettings(host="127.0.0.1", port=8000, log_level="info")
        result = settings.to_dict()
        assert result == {
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "info",
            "cors_origins": ["*"],
        }

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"host": "0.0.0.0", "port": 9000, "log_level": "debug"}
        settings = ServerSettings.from_dict(data)
        assert settings.host == "0.0.0.0"
        assert settings.port == 9000
        assert settings.log_level == "debug"
        assert settings.cors_origins == ["*"]  # default

    def test_from_dict_with_cors_origins(self):
        """Test creation from dictionary with cors_origins."""
        data = {
            "host": "0.0.0.0",
            "port": 9000,
            "cors_origins": ["https://chat.example.com", "http://localhost:3000"],
        }
        settings = ServerSettings.from_dict(data)
        assert settings.cors_origins == [
            "https://chat.example.com",
            "http://localhost:3000",
        ]

    def test_from_dict_with_defaults(self):
        """Test creation from partial dictionary uses defaults."""
        data = {"port": 9000}
        settings = ServerSettings.from_dict(data)
        assert settings.host == "127.0.0.1"  # default
        assert settings.port == 9000
        assert settings.log_level == "info"  # default
        assert settings.cors_origins == ["*"]  # default


class TestModelSettings:
    """Tests for ModelSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = ModelSettings()
        assert settings.model_dirs == []
        assert settings.model_dir is None
        assert settings.max_model_memory == "auto"

    def test_get_max_model_memory_bytes_auto(self):
        """Test auto memory calculation."""
        settings = ModelSettings(max_model_memory="auto")
        # Should return 80% of system memory
        expected = int(get_system_memory() * 0.8)
        assert settings.get_max_model_memory_bytes() == expected

    def test_get_max_model_memory_bytes_auto_uppercase(self):
        """Test auto memory calculation with uppercase AUTO."""
        settings = ModelSettings(max_model_memory="AUTO")
        expected = int(get_system_memory() * 0.8)
        assert settings.get_max_model_memory_bytes() == expected

    def test_get_max_model_memory_bytes_explicit(self):
        """Test explicit memory value."""
        settings = ModelSettings(max_model_memory="16GB")
        assert settings.get_max_model_memory_bytes() == 16 * 1024**3

    def test_get_max_model_memory_bytes_various_units(self):
        """Test explicit memory with various units."""
        settings = ModelSettings(max_model_memory="512MB")
        assert settings.get_max_model_memory_bytes() == 512 * 1024**2

        settings = ModelSettings(max_model_memory="1TB")
        assert settings.get_max_model_memory_bytes() == 1024**4

    def test_get_model_dirs_default(self):
        """Test default model directories."""
        settings = ModelSettings()
        base_path = Path("/tmp/omlx")
        assert settings.get_model_dirs(base_path) == [Path("/tmp/omlx/models")]
        assert settings.get_model_dir(base_path) == Path("/tmp/omlx/models")

    def test_get_model_dirs_custom(self):
        """Test custom model directories."""
        settings = ModelSettings(model_dirs=["/custom/models"])
        base_path = Path("/tmp/omlx")
        assert settings.get_model_dirs(base_path) == [Path("/custom/models")]

    def test_get_model_dirs_multiple(self):
        """Test multiple model directories."""
        settings = ModelSettings(model_dirs=["/path/a", "/path/b"])
        base_path = Path("/tmp/omlx")
        result = settings.get_model_dirs(base_path)
        assert len(result) == 2
        assert result[0] == Path("/path/a")
        assert result[1] == Path("/path/b")
        # get_model_dir returns the first (primary) directory
        assert settings.get_model_dir(base_path) == Path("/path/a")

    def test_get_model_dirs_with_tilde(self):
        """Test model directory with tilde expansion."""
        settings = ModelSettings(model_dirs=["~/models"])
        base_path = Path("/tmp/omlx")
        result = settings.get_model_dirs(base_path)
        assert "~" not in str(result[0])  # Should be expanded

    def test_get_model_dirs_backward_compat(self):
        """Test backward compatibility: model_dir fallback when model_dirs is empty."""
        settings = ModelSettings(model_dir="/legacy/models")
        base_path = Path("/tmp/omlx")
        assert settings.get_model_dirs(base_path) == [Path("/legacy/models")]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = ModelSettings(model_dirs=["/models"], max_model_memory="32GB")
        result = settings.to_dict()
        assert result == {
            "model_dirs": ["/models"],
            "model_dir": "/models",
            "max_model_memory": "32GB",
        }

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"model_dirs": ["/models"], "max_model_memory": "64GB"}
        settings = ModelSettings.from_dict(data)
        assert settings.model_dirs == ["/models"]
        assert settings.max_model_memory == "64GB"

    def test_from_dict_backward_compat(self):
        """Test from_dict migrates old model_dir to model_dirs."""
        data = {"model_dir": "/legacy/models", "max_model_memory": "64GB"}
        settings = ModelSettings.from_dict(data)
        assert settings.model_dirs == ["/legacy/models"]
        assert settings.model_dir == "/legacy/models"


class TestSchedulerSettings:
    """Tests for SchedulerSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = SchedulerSettings()
        assert settings.max_num_seqs == 8
        assert settings.prefill_batch_size == 8
        assert settings.completion_batch_size == 8

    def test_custom_values(self):
        """Test custom values."""
        settings = SchedulerSettings(
            max_num_seqs=128, prefill_batch_size=4, completion_batch_size=16
        )
        assert settings.max_num_seqs == 128
        assert settings.prefill_batch_size == 4
        assert settings.completion_batch_size == 16

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = SchedulerSettings()
        result = settings.to_dict()
        assert result == {
            "max_num_seqs": 8,
            "prefill_batch_size": 8,
            "completion_batch_size": 8,
            "max_kv_cache_memory": "auto",
        }

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "max_num_seqs": 512,
            "prefill_batch_size": 16,
            "completion_batch_size": 64,
        }
        settings = SchedulerSettings.from_dict(data)
        assert settings.max_num_seqs == 512
        assert settings.prefill_batch_size == 16
        assert settings.completion_batch_size == 64


class TestCacheSettings:
    """Tests for CacheSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = CacheSettings()
        assert settings.enabled is True
        assert settings.ssd_cache_dir is None
        assert settings.ssd_cache_max_size == "auto"

    def test_get_ssd_cache_dir_default(self):
        """Test default SSD cache directory."""
        settings = CacheSettings(ssd_cache_dir=None)
        base_path = Path("/tmp/omlx")
        assert settings.get_ssd_cache_dir(base_path) == Path("/tmp/omlx/cache")

    def test_get_ssd_cache_dir_custom(self):
        """Test custom SSD cache directory."""
        settings = CacheSettings(ssd_cache_dir="/custom/cache")
        base_path = Path("/tmp/omlx")
        assert settings.get_ssd_cache_dir(base_path) == Path("/custom/cache")

    def test_get_ssd_cache_max_size_bytes_auto(self):
        """Test auto SSD cache size calculation."""
        settings = CacheSettings(ssd_cache_max_size="auto")
        base_path = Path("/tmp/omlx")
        cache_dir = settings.get_ssd_cache_dir(base_path)
        expected = int(get_ssd_capacity(cache_dir) * 0.1)
        assert settings.get_ssd_cache_max_size_bytes(base_path) == expected

    def test_get_ssd_cache_max_size_bytes_explicit(self):
        """Test explicit SSD cache size."""
        settings = CacheSettings(ssd_cache_max_size="100GB")
        base_path = Path("/tmp/omlx")
        assert settings.get_ssd_cache_max_size_bytes(base_path) == 100 * 1024**3

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = CacheSettings(
            enabled=False, ssd_cache_dir="/cache", ssd_cache_max_size="50GB"
        )
        result = settings.to_dict()
        assert result == {
            "enabled": False,
            "ssd_cache_dir": "/cache",
            "ssd_cache_max_size": "50GB",
        }

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "enabled": False,
            "ssd_cache_dir": "/cache",
            "ssd_cache_max_size": "200GB",
        }
        settings = CacheSettings.from_dict(data)
        assert settings.enabled is False
        assert settings.ssd_cache_dir == "/cache"
        assert settings.ssd_cache_max_size == "200GB"


class TestAuthSettings:
    """Tests for AuthSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = AuthSettings()
        assert settings.api_key is None

    def test_custom_values(self):
        """Test custom values."""
        settings = AuthSettings(api_key="test-secret-key")
        assert settings.api_key == "test-secret-key"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = AuthSettings(api_key="my-key")
        result = settings.to_dict()
        assert result == {"api_key": "my-key"}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"api_key": "loaded-key"}
        settings = AuthSettings.from_dict(data)
        assert settings.api_key == "loaded-key"


class TestMCPSettings:
    """Tests for MCPSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = MCPSettings()
        assert settings.config_path is None

    def test_custom_values(self):
        """Test custom values."""
        settings = MCPSettings(config_path="/path/to/mcp.json")
        assert settings.config_path == "/path/to/mcp.json"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = MCPSettings(config_path="/mcp/config.json")
        result = settings.to_dict()
        assert result == {"config_path": "/mcp/config.json"}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"config_path": "/some/path.json"}
        settings = MCPSettings.from_dict(data)
        assert settings.config_path == "/some/path.json"


class TestLoggingSettings:
    """Tests for LoggingSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = LoggingSettings()
        assert settings.log_dir is None
        assert settings.retention_days == 7

    def test_custom_values(self):
        """Test custom values."""
        settings = LoggingSettings(log_dir="/custom/logs", retention_days=14)
        assert settings.log_dir == "/custom/logs"
        assert settings.retention_days == 14

    def test_get_log_dir_default(self):
        """Test default log directory."""
        settings = LoggingSettings(log_dir=None)
        base_path = Path("/tmp/omlx")
        assert settings.get_log_dir(base_path) == Path("/tmp/omlx/logs")

    def test_get_log_dir_custom(self):
        """Test custom log directory."""
        settings = LoggingSettings(log_dir="/custom/logs")
        base_path = Path("/tmp/omlx")
        assert settings.get_log_dir(base_path) == Path("/custom/logs")

    def test_get_log_dir_with_tilde(self):
        """Test log directory with tilde expansion."""
        settings = LoggingSettings(log_dir="~/logs")
        base_path = Path("/tmp/omlx")
        result = settings.get_log_dir(base_path)
        assert "~" not in str(result)  # Should be expanded

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = LoggingSettings(log_dir="/logs", retention_days=30)
        result = settings.to_dict()
        assert result == {"log_dir": "/logs", "retention_days": 30}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"log_dir": "/my/logs", "retention_days": 14}
        settings = LoggingSettings.from_dict(data)
        assert settings.log_dir == "/my/logs"
        assert settings.retention_days == 14

    def test_from_dict_with_defaults(self):
        """Test creation from partial dictionary uses defaults."""
        data = {"log_dir": "/custom"}
        settings = LoggingSettings.from_dict(data)
        assert settings.log_dir == "/custom"
        assert settings.retention_days == 7  # default


class TestGlobalSettings:
    """Tests for GlobalSettings class."""

    def test_defaults(self):
        """Test default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            assert settings.server.host == "127.0.0.1"
            assert settings.server.port == 8000
            assert settings.model.max_model_memory == "auto"
            assert settings.scheduler.max_num_seqs == 8
            assert settings.cache.enabled is True
            assert settings.auth.api_key is None
            assert settings.mcp.config_path is None

    def test_load_from_file(self):
        """Test loading settings from JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create settings file
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "server": {"port": 9000},
                        "auth": {"api_key": "test-key"},
                    }
                )
            )

            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.port == 9000
            assert settings.auth.api_key == "test-key"

    def test_load_from_file_all_sections(self):
        """Test loading all settings sections from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "server": {"host": "0.0.0.0", "port": 9000, "log_level": "debug"},
                        "model": {"model_dir": "/models", "max_model_memory": "64GB"},
                        "scheduler": {
                            "max_num_seqs": 128,
                            "prefill_batch_size": 4,
                            "completion_batch_size": 16,
                        },
                        "cache": {
                            "enabled": False,
                            "ssd_cache_dir": "/cache",
                            "ssd_cache_max_size": "50GB",
                        },
                        "auth": {"api_key": "secret"},
                        "mcp": {"config_path": "/mcp.json"},
                    }
                )
            )

            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.host == "0.0.0.0"
            assert settings.server.port == 9000
            assert settings.server.log_level == "debug"
            assert settings.model.model_dirs == ["/models"]  # Migrated from model_dir
            assert settings.model.model_dir == "/models"  # Backward compat field
            assert settings.model.max_model_memory == "64GB"
            assert settings.scheduler.max_num_seqs == 128
            assert settings.cache.enabled is False
            assert settings.cache.ssd_cache_dir == "/cache"
            assert settings.auth.api_key == "secret"
            assert settings.mcp.config_path == "/mcp.json"

    def test_load_nonexistent_file_uses_defaults(self):
        """Test loading with no settings file uses defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.host == "127.0.0.1"
            assert settings.server.port == 8000

    def test_load_invalid_json_uses_defaults(self):
        """Test loading invalid JSON file logs warning and uses defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text("{ invalid json }")

            settings = GlobalSettings.load(base_path=tmpdir)
            # Should use defaults due to parse error
            assert settings.server.port == 8000

    def test_save(self):
        """Test saving settings to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            settings.server.port = 9001
            settings.auth.api_key = "saved-key"
            settings.save()

            # Verify file was created
            settings_file = Path(tmpdir) / "settings.json"
            assert settings_file.exists()

            # Verify content
            data = json.loads(settings_file.read_text())
            assert data["version"] == "1.0"
            assert data["server"]["port"] == 9001
            assert data["auth"]["api_key"] == "saved-key"

    def test_save_and_load_cors_origins(self):
        """Test saving and loading cors_origins through settings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            settings.server.cors_origins = ["https://chat.example.com"]
            settings.save()

            loaded = GlobalSettings.load(base_path=tmpdir)
            assert loaded.server.cors_origins == ["https://chat.example.com"]

    def test_load_cors_origins_from_file(self):
        """Test loading cors_origins from settings.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "server": {
                            "port": 8000,
                            "cors_origins": [
                                "https://chat.example.com",
                                "http://localhost:3000",
                            ],
                        },
                    }
                )
            )

            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.cors_origins == [
                "https://chat.example.com",
                "http://localhost:3000",
            ]

    def test_load_without_cors_origins_uses_default(self):
        """Test that missing cors_origins in settings.json uses default ['*']."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps({"version": "1.0", "server": {"port": 9000}})
            )

            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.cors_origins == ["*"]

    def test_save_creates_directory(self):
        """Test save creates base directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "nested" / "omlx"
            settings = GlobalSettings(base_path=base)
            settings.save()

            assert base.exists()
            assert (base / "settings.json").exists()

    def test_ensure_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "omlx"
            settings = GlobalSettings(base_path=base)
            settings.ensure_directories()

            assert base.exists()
            assert (base / "models").exists()
            assert (base / "cache").exists()
            assert (base / "logs").exists()

    def test_ensure_directories_custom_paths(self):
        """Test directory creation with custom paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "omlx"
            custom_models = Path(tmpdir) / "custom_models"
            custom_cache = Path(tmpdir) / "custom_cache"
            custom_logs = Path(tmpdir) / "custom_logs"

            settings = GlobalSettings(base_path=base)
            settings.model.model_dir = str(custom_models)
            settings.cache.ssd_cache_dir = str(custom_cache)
            settings.logging.log_dir = str(custom_logs)
            settings.ensure_directories()

            assert base.exists()
            assert custom_models.exists()
            assert custom_cache.exists()
            assert custom_logs.exists()

    def test_validate_valid_settings(self):
        """Test validation with valid settings."""
        settings = GlobalSettings()
        errors = settings.validate()
        assert errors == []

    def test_validate_invalid_port_low(self):
        """Test validation catches port below 1."""
        settings = GlobalSettings()
        settings.server.port = 0
        errors = settings.validate()
        assert any("port" in e.lower() for e in errors)

    def test_validate_invalid_port_high(self):
        """Test validation catches port above 65535."""
        settings = GlobalSettings()
        settings.server.port = 70000
        errors = settings.validate()
        assert any("port" in e.lower() for e in errors)

    def test_validate_invalid_log_level(self):
        """Test validation catches invalid log level."""
        settings = GlobalSettings()
        settings.server.log_level = "invalid"
        errors = settings.validate()
        assert any("log_level" in e.lower() for e in errors)

    def test_validate_valid_log_levels(self):
        """Test validation accepts all valid log levels."""
        for level in ["debug", "info", "warning", "error", "critical"]:
            settings = GlobalSettings()
            settings.server.log_level = level
            errors = settings.validate()
            assert errors == []

    def test_validate_invalid_max_model_memory(self):
        """Test validation catches invalid memory size."""
        settings = GlobalSettings()
        settings.model.max_model_memory = "invalid"
        errors = settings.validate()
        assert any("max_model_memory" in e.lower() for e in errors)

    def test_validate_invalid_scheduler_values(self):
        """Test validation catches invalid scheduler values."""
        settings = GlobalSettings()
        settings.scheduler.max_num_seqs = 0
        errors = settings.validate()
        assert any("max_num_seqs" in e.lower() for e in errors)

        settings = GlobalSettings()
        settings.scheduler.prefill_batch_size = -1
        errors = settings.validate()
        assert any("prefill_batch_size" in e.lower() for e in errors)

        settings = GlobalSettings()
        settings.scheduler.completion_batch_size = 0
        errors = settings.validate()
        assert any("completion_batch_size" in e.lower() for e in errors)

    def test_validate_invalid_cache_size(self):
        """Test validation catches invalid cache size."""
        settings = GlobalSettings()
        settings.cache.ssd_cache_max_size = "not-a-size"
        errors = settings.validate()
        assert any("ssd_cache_max_size" in e.lower() for e in errors)

    def test_validate_multiple_errors(self):
        """Test validation returns multiple errors."""
        settings = GlobalSettings()
        settings.server.port = 0
        settings.scheduler.max_num_seqs = -1
        settings.model.max_model_memory = "invalid"
        errors = settings.validate()
        assert len(errors) >= 3

    def test_env_override_server(self):
        """Test environment variable override for server settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "OMLX_HOST": "0.0.0.0",
                    "OMLX_PORT": "9999",
                    "OMLX_LOG_LEVEL": "debug",
                },
                clear=False,
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.server.host == "0.0.0.0"
                assert settings.server.port == 9999
                assert settings.server.log_level == "debug"

    def test_env_override_model(self):
        """Test environment variable override for model settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "OMLX_MODEL_DIR": "/env/models",
                    "OMLX_MAX_MODEL_MEMORY": "128GB",
                },
                clear=False,
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.model.model_dir == "/env/models"
                assert settings.model.max_model_memory == "128GB"

    def test_env_override_scheduler(self):
        """Test environment variable override for scheduler settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "OMLX_MAX_NUM_SEQS": "512",
                    "OMLX_PREFILL_BATCH_SIZE": "16",
                    "OMLX_COMPLETION_BATCH_SIZE": "64",
                },
                clear=False,
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.scheduler.max_num_seqs == 512
                assert settings.scheduler.prefill_batch_size == 16
                assert settings.scheduler.completion_batch_size == 64

    def test_env_override_cache(self):
        """Test environment variable override for cache settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "OMLX_CACHE_ENABLED": "false",
                    "OMLX_SSD_CACHE_DIR": "/env/cache",
                    "OMLX_SSD_CACHE_MAX_SIZE": "200GB",
                },
                clear=False,
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.cache.enabled is False
                assert settings.cache.ssd_cache_dir == "/env/cache"
                assert settings.cache.ssd_cache_max_size == "200GB"

    def test_env_override_cache_enabled_values(self):
        """Test various values for OMLX_CACHE_ENABLED."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for value in ["true", "1", "yes"]:
                with patch.dict(
                    os.environ, {"OMLX_CACHE_ENABLED": value}, clear=False
                ):
                    settings = GlobalSettings.load(base_path=tmpdir)
                    assert settings.cache.enabled is True

            for value in ["false", "0", "no"]:
                with patch.dict(
                    os.environ, {"OMLX_CACHE_ENABLED": value}, clear=False
                ):
                    settings = GlobalSettings.load(base_path=tmpdir)
                    assert settings.cache.enabled is False

    def test_env_override_auth(self):
        """Test environment variable override for auth settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"OMLX_API_KEY": "env-key"}, clear=False):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.auth.api_key == "env-key"

    def test_env_override_mcp(self):
        """Test environment variable override for MCP settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ, {"OMLX_MCP_CONFIG": "/env/mcp.json"}, clear=False
            ):
                settings = GlobalSettings.load(base_path=tmpdir)
                assert settings.mcp.config_path == "/env/mcp.json"

    def test_env_override_invalid_port_logs_warning(self):
        """Test invalid OMLX_PORT logs warning and keeps default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"OMLX_PORT": "not-a-number"}, clear=False):
                settings = GlobalSettings.load(base_path=tmpdir)
                # Should keep default due to parse error
                assert settings.server.port == 8000

    def test_env_override_after_file(self):
        """Test env vars override file settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create settings file
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps({"version": "1.0", "server": {"port": 9000}})
            )

            with patch.dict(os.environ, {"OMLX_PORT": "8888"}, clear=False):
                settings = GlobalSettings.load(base_path=tmpdir)
                # Env should override file
                assert settings.server.port == 8888

    def test_cli_override(self):
        """Test CLI argument override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                port=8888,
                host="0.0.0.0",
                log_level="warning",
                model_dir="/cli/models",
                max_model_memory="32GB",
                api_key="cli-key",
            )
            settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
            assert settings.server.port == 8888
            assert settings.server.host == "0.0.0.0"
            assert settings.server.log_level == "warning"
            assert settings.model.model_dir == "/cli/models"
            assert settings.model.max_model_memory == "32GB"
            assert settings.auth.api_key == "cli-key"

    def test_cli_override_partial(self):
        """Test CLI override with some None values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(port=7777, host=None, log_level=None)
            settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
            assert settings.server.port == 7777
            assert settings.server.host == "127.0.0.1"  # default
            assert settings.server.log_level == "info"  # default

    def test_cli_override_scheduler(self):
        """Test CLI override for scheduler settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                max_num_seqs=64,
                prefill_batch_size=2,
                completion_batch_size=8,
            )
            settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
            assert settings.scheduler.max_num_seqs == 64
            assert settings.scheduler.prefill_batch_size == 2
            assert settings.scheduler.completion_batch_size == 8

    def test_cli_override_cache(self):
        """Test CLI override for cache settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                cache_enabled=False,
                ssd_cache_dir="/cli/cache",
                ssd_cache_max_size="500GB",
            )
            settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
            assert settings.cache.enabled is False
            assert settings.cache.ssd_cache_dir == "/cli/cache"
            assert settings.cache.ssd_cache_max_size == "500GB"

    def test_cli_override_mcp(self):
        """Test CLI override for MCP settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(mcp_config="/cli/mcp.json")
            settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
            assert settings.mcp.config_path == "/cli/mcp.json"

    def test_priority_cli_over_env_over_file(self):
        """Test that CLI > env > file > defaults priority is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create settings file with port 9000
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps({"version": "1.0", "server": {"port": 9000}})
            )

            # Set env to port 8888
            with patch.dict(os.environ, {"OMLX_PORT": "8888"}, clear=False):
                # CLI sets port to 7777
                args = Namespace(port=7777)
                settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
                # CLI should win
                assert settings.server.port == 7777

            # Without CLI, env should win over file
            settings = GlobalSettings.load(base_path=tmpdir)
            # Env is no longer set, so file should be used
            assert settings.server.port == 9000

    def test_to_dict(self):
        """Test conversion to dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            settings.server.port = 9000
            result = settings.to_dict()

            assert result["version"] == "1.0"
            assert result["base_path"] == tmpdir
            assert result["server"]["port"] == 9000
            assert "model" in result
            assert "scheduler" in result
            assert "cache" in result
            assert "auth" in result
            assert "mcp" in result

    def test_to_scheduler_config(self):
        """Test conversion to SchedulerConfig."""
        settings = GlobalSettings()
        settings.scheduler.max_num_seqs = 128
        settings.scheduler.prefill_batch_size = 4
        settings.scheduler.completion_batch_size = 16

        scheduler_config = settings.to_scheduler_config()
        assert scheduler_config.max_num_seqs == 128
        assert scheduler_config.prefill_batch_size == 4
        assert scheduler_config.completion_batch_size == 16


class TestInitSettings:
    """Tests for init_settings and get_settings."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def teardown_method(self):
        """Reset settings after each test."""
        reset_settings()

    def test_init_settings(self):
        """Test initializing global settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = init_settings(base_path=tmpdir)
            assert settings is not None
            assert get_settings() is settings

    def test_init_settings_with_cli_args(self):
        """Test initializing with CLI arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(port=9999)
            settings = init_settings(base_path=tmpdir, cli_args=args)
            assert settings.server.port == 9999
            assert get_settings().server.port == 9999

    def test_get_settings_uninitialized_raises(self):
        """Test get_settings raises RuntimeError when not initialized."""
        with pytest.raises(RuntimeError, match="Settings not initialized"):
            get_settings()

    def test_reset_settings(self):
        """Test resetting settings allows re-initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize first
            init_settings(base_path=tmpdir)
            assert get_settings() is not None

            # Reset
            reset_settings()

            # Should raise now
            with pytest.raises(RuntimeError):
                get_settings()

            # Can initialize again
            settings = init_settings(base_path=tmpdir)
            assert settings is not None

    def test_multiple_init_overwrites(self):
        """Test calling init_settings multiple times overwrites."""
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            settings1 = init_settings(base_path=tmpdir1)
            settings2 = init_settings(base_path=tmpdir2)

            assert get_settings() is settings2
            # Use resolve() to handle macOS /var -> /private/var symlink
            assert get_settings().base_path.resolve() == Path(tmpdir2).resolve()


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_system_memory(self):
        """Test system memory detection."""
        memory = get_system_memory()
        assert memory > 0
        # Should be at least 1GB on any modern system
        assert memory >= 1024**3

    def test_get_system_memory_returns_int(self):
        """Test that get_system_memory returns an integer."""
        memory = get_system_memory()
        assert isinstance(memory, int)

    def test_get_ssd_capacity(self):
        """Test SSD capacity detection."""
        capacity = get_ssd_capacity(Path("/"))
        assert capacity > 0

    def test_get_ssd_capacity_returns_int(self):
        """Test that get_ssd_capacity returns an integer."""
        capacity = get_ssd_capacity(Path("/"))
        assert isinstance(capacity, int)

    def test_get_ssd_capacity_nonexistent_path(self):
        """Test SSD capacity for non-existent path uses parent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "does" / "not" / "exist"
            capacity = get_ssd_capacity(nonexistent)
            assert capacity > 0

    def test_get_ssd_capacity_with_string_path(self):
        """Test SSD capacity with string path."""
        capacity = get_ssd_capacity("/")
        assert capacity > 0

    def test_get_ssd_capacity_with_tilde(self):
        """Test SSD capacity with tilde path expansion."""
        capacity = get_ssd_capacity("~/")
        assert capacity > 0


class TestSettingsVersionMigration:
    """Tests for settings version handling."""

    def test_load_different_version_logs_migration(self):
        """Test loading settings with different version logs migration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps({"version": "0.9", "server": {"port": 9000}})
            )

            settings = GlobalSettings.load(base_path=tmpdir)
            # Should still load the settings
            assert settings.server.port == 9000

    def test_load_without_version(self):
        """Test loading settings file without version field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(json.dumps({"server": {"port": 9000}}))

            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.port == 9000


class TestSettingsEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_settings_file(self):
        """Test loading empty settings file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text("{}")

            settings = GlobalSettings.load(base_path=tmpdir)
            # Should use all defaults
            assert settings.server.port == 8000
            assert settings.server.host == "127.0.0.1"

    def test_partial_section_in_file(self):
        """Test loading file with partial section data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_file = Path(tmpdir) / "settings.json"
            settings_file.write_text(
                json.dumps(
                    {
                        "version": "1.0",
                        "server": {"port": 9000},
                        # host and log_level should use defaults
                    }
                )
            )

            settings = GlobalSettings.load(base_path=tmpdir)
            assert settings.server.port == 9000
            assert settings.server.host == "127.0.0.1"  # default
            assert settings.server.log_level == "info"  # default

    def test_cli_args_without_expected_attrs(self):
        """Test CLI args object without all expected attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Minimal args object
            args = Namespace(port=9000)
            settings = GlobalSettings.load(base_path=tmpdir, cli_args=args)
            assert settings.server.port == 9000
            # Other settings should be defaults
            assert settings.server.host == "127.0.0.1"

    def test_save_with_unicode_api_key(self):
        """Test saving settings with unicode in values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            settings.auth.api_key = "key-with-unicode-\u4e2d\u6587"
            settings.save()

            # Reload and verify
            loaded = GlobalSettings.load(base_path=tmpdir)
            assert loaded.auth.api_key == "key-with-unicode-\u4e2d\u6587"


class TestSamplingSettings:
    """Tests for SamplingSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = SamplingSettings()
        assert settings.max_context_window == 32768
        assert settings.max_tokens == 32768
        assert settings.temperature == 1.0
        assert settings.top_p == 0.95
        assert settings.top_k == 40
        assert settings.repetition_penalty == 1.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = SamplingSettings(max_context_window=4096)
        d = settings.to_dict()
        assert d["max_context_window"] == 4096
        assert "max_tokens" in d
        assert "repetition_penalty" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"max_context_window": 8192, "max_tokens": 1024, "repetition_penalty": 1.2}
        settings = SamplingSettings.from_dict(data)
        assert settings.max_context_window == 8192
        assert settings.max_tokens == 1024
        assert settings.repetition_penalty == 1.2

    def test_from_dict_defaults(self):
        """Test from_dict uses defaults for missing fields."""
        settings = SamplingSettings.from_dict({})
        assert settings.max_context_window == 32768
        assert settings.repetition_penalty == 1.0


class TestClaudeCodeSettings:
    """Tests for ClaudeCodeSettings dataclass."""

    def test_defaults(self):
        """Test default values."""
        settings = ClaudeCodeSettings()
        assert settings.context_scaling_enabled is False
        assert settings.target_context_size == 200000

    def test_custom_values(self):
        """Test custom values."""
        settings = ClaudeCodeSettings(
            context_scaling_enabled=True, target_context_size=131072
        )
        assert settings.context_scaling_enabled is True
        assert settings.target_context_size == 131072

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settings = ClaudeCodeSettings(
            context_scaling_enabled=True, target_context_size=100000
        )
        result = settings.to_dict()
        assert result == {
            "context_scaling_enabled": True,
            "target_context_size": 100000,
        }

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"context_scaling_enabled": True, "target_context_size": 131072}
        settings = ClaudeCodeSettings.from_dict(data)
        assert settings.context_scaling_enabled is True
        assert settings.target_context_size == 131072

    def test_from_dict_defaults(self):
        """Test from_dict uses defaults for missing fields."""
        settings = ClaudeCodeSettings.from_dict({})
        assert settings.context_scaling_enabled is False
        assert settings.target_context_size == 200000


class TestCORSMiddleware:
    """Test that CORS middleware is correctly applied to the server."""

    def test_cors_preflight(self):
        """Test that CORS preflight requests get proper response headers."""
        from fastapi.testclient import TestClient

        from omlx.server import app, init_server

        with tempfile.TemporaryDirectory() as tmpdir:
            settings = GlobalSettings(base_path=Path(tmpdir))
            init_server(
                model_dirs=[tmpdir],
                max_model_memory=0,
                global_settings=settings,
            )

            client = TestClient(app)
            resp = client.options(
                "/v1/models",
                headers={
                    "Origin": "https://example.com",
                    "Access-Control-Request-Method": "GET",
                },
            )
            assert resp.status_code == 200
            assert "access-control-allow-origin" in resp.headers
            assert resp.headers["access-control-allow-origin"] == "*"
