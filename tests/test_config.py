# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.config module."""

import os
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from omlx.config import (
    parse_size,
    ServerConfig,
    ModelConfig,
    GenerationConfig,
    SchedulerConfig,
    CacheConfig,
    PagedSSDCacheConfig,
    MCPConfig,
    OMLXConfig,
)


class TestParseSize:
    """Test cases for parse_size function."""

    def test_parse_bytes(self):
        """Test parsing byte values."""
        assert parse_size("100B") == 100
        assert parse_size("0B") == 0
        assert parse_size("1024B") == 1024

    def test_parse_kilobytes(self):
        """Test parsing KB values."""
        assert parse_size("1KB") == 1024
        assert parse_size("100KB") == 100 * 1024
        assert parse_size("1.5KB") == int(1.5 * 1024)

    def test_parse_megabytes(self):
        """Test parsing MB values."""
        assert parse_size("1MB") == 1024**2
        assert parse_size("512MB") == 512 * 1024**2
        assert parse_size("2.5MB") == int(2.5 * 1024**2)

    def test_parse_gigabytes(self):
        """Test parsing GB values."""
        assert parse_size("1GB") == 1024**3
        assert parse_size("16GB") == 16 * 1024**3
        assert parse_size("32.5GB") == int(32.5 * 1024**3)

    def test_parse_terabytes(self):
        """Test parsing TB values."""
        assert parse_size("1TB") == 1024**4
        assert parse_size("2TB") == 2 * 1024**4

    def test_parse_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        assert parse_size("1gb") == 1024**3
        assert parse_size("1Gb") == 1024**3
        assert parse_size("1gB") == 1024**3
        assert parse_size("1GB") == 1024**3

    def test_parse_with_whitespace(self):
        """Test parsing with leading/trailing whitespace."""
        assert parse_size("  1GB  ") == 1024**3
        assert parse_size("\t16GB\n") == 16 * 1024**3

    def test_parse_plain_number(self):
        """Test parsing plain number as bytes."""
        assert parse_size("1024") == 1024
        assert parse_size("0") == 0

    def test_parse_invalid_raises_error(self):
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError):
            parse_size("invalid")
        with pytest.raises(ValueError):
            parse_size("abc123")
        with pytest.raises(ValueError):
            parse_size("1XB")  # Invalid unit


class TestServerConfig:
    """Test cases for ServerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.log_level == "info"
        assert config.cors_origins == ["*"]

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            log_level="debug",
            cors_origins=["http://localhost:3000"],
        )
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.log_level == "debug"
        assert config.cors_origins == ["http://localhost:3000"]


class TestModelConfig:
    """Test cases for ModelConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.model_name == ""
        assert config.trust_remote_code is True
        assert config.model_path is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            model_name="llama-3.1-8b",
            trust_remote_code=False,
            model_path="/path/to/model",
        )
        assert config.model_name == "llama-3.1-8b"
        assert config.trust_remote_code is False
        assert config.model_path == "/path/to/model"


class TestGenerationConfig:
    """Test cases for GenerationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GenerationConfig()
        assert config.max_tokens == 32768
        assert config.temperature == 1.0
        assert config.top_p == 0.95
        assert config.top_k == 0
        assert config.force_sampling is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = GenerationConfig(
            max_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            force_sampling=True,
        )
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.force_sampling is True


class TestSchedulerConfig:
    """Test cases for SchedulerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SchedulerConfig()
        assert config.max_num_seqs == 8
        assert config.prefill_batch_size == 8
        assert config.completion_batch_size == 8
        assert config.stream_interval == 1
        assert config.enable_thinking is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SchedulerConfig(
            max_num_seqs=128,
            prefill_batch_size=4,
            completion_batch_size=16,
            stream_interval=2,
            enable_thinking=True,
        )
        assert config.max_num_seqs == 128
        assert config.prefill_batch_size == 4
        assert config.completion_batch_size == 16
        assert config.stream_interval == 2
        assert config.enable_thinking is True


class TestPagedSSDCacheConfig:
    """Test cases for PagedSSDCacheConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PagedSSDCacheConfig()
        assert config.enabled is False
        assert config.cache_dir is None
        assert config.max_size == "100GB"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PagedSSDCacheConfig(
            enabled=True,
            cache_dir=Path("/tmp/cache"),
            max_size="50GB",
        )
        assert config.enabled is True
        assert config.cache_dir == Path("/tmp/cache")
        assert config.max_size == "50GB"

    def test_max_size_bytes_property(self):
        """Test max_size_bytes property calculation."""
        config = PagedSSDCacheConfig(max_size="100GB")
        assert config.max_size_bytes == 100 * 1024**3

        config = PagedSSDCacheConfig(max_size="50MB")
        assert config.max_size_bytes == 50 * 1024**2

        config = PagedSSDCacheConfig(max_size="1TB")
        assert config.max_size_bytes == 1024**4


class TestMCPConfig:
    """Test cases for MCPConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MCPConfig()
        assert config.config_path is None
        assert config.enabled is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MCPConfig(
            config_path="/path/to/mcp.json",
            enabled=True,
        )
        assert config.config_path == "/path/to/mcp.json"
        assert config.enabled is True


class TestOMLXConfig:
    """Test cases for OMLXConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OMLXConfig()
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.generation, GenerationConfig)
        assert isinstance(config.scheduler, SchedulerConfig)
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.paged_ssd_cache, PagedSSDCacheConfig)
        assert isinstance(config.mcp, MCPConfig)
        assert config.continuous_batching is False

    def test_from_env_default(self):
        """Test from_env with no environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            config = OMLXConfig.from_env()
            assert config.server.host == "0.0.0.0"
            assert config.server.port == 8000

    def test_from_env_with_variables(self):
        """Test from_env with environment variables set."""
        env_vars = {
            "OMLX_HOST": "127.0.0.1",
            "OMLX_PORT": "9000",
            "OMLX_LOG_LEVEL": "debug",
            "OMLX_MODEL": "test-model",
            "OMLX_TRUST_REMOTE_CODE": "false",
            "OMLX_MAX_TOKENS": "4096",
            "OMLX_TEMPERATURE": "0.5",
            "OMLX_CONTINUOUS_BATCHING": "true",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = OMLXConfig.from_env()
            assert config.server.host == "127.0.0.1"
            assert config.server.port == 9000
            assert config.server.log_level == "debug"
            assert config.model.model_name == "test-model"
            assert config.model.trust_remote_code is False
            assert config.generation.max_tokens == 4096
            assert config.generation.temperature == 0.5
            assert config.continuous_batching is True

    def test_from_env_paged_ssd_cache(self):
        """Test from_env with paged SSD cache environment variables."""
        env_vars = {
            "OMLX_PAGED_SSD_CACHE_DIR": "/tmp/ssd_cache",
            "OMLX_PAGED_SSD_CACHE_MAX_SIZE": "50GB",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = OMLXConfig.from_env()
            assert config.paged_ssd_cache.enabled is True
            assert config.paged_ssd_cache.cache_dir == Path("/tmp/ssd_cache")
            assert config.paged_ssd_cache.max_size == "50GB"

    def test_from_env_mcp(self):
        """Test from_env with MCP environment variables."""
        env_vars = {
            "OMLX_MCP_CONFIG": "/path/to/mcp.json",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = OMLXConfig.from_env()
            assert config.mcp.enabled is True
            assert config.mcp.config_path == "/path/to/mcp.json"

    def test_from_cli_args(self):
        """Test from_cli_args with argparse namespace."""
        args = Namespace(
            host="127.0.0.1",
            port=9000,
            log_level="debug",
            model="test-model",
            trust_remote_code=False,
            max_tokens=4096,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            continuous_batching=True,
            paged_ssd_cache_dir=None,
            paged_ssd_cache_max_size=None,
            mcp_config=None,
        )
        with patch.dict(os.environ, {}, clear=True):
            config = OMLXConfig.from_cli_args(args)
            assert config.server.host == "127.0.0.1"
            assert config.server.port == 9000
            assert config.model.model_name == "test-model"
            assert config.generation.max_tokens == 4096
            assert config.continuous_batching is True

    def test_from_cli_args_paged_ssd_cache(self):
        """Test from_cli_args with paged SSD cache arguments."""
        args = Namespace(
            paged_ssd_cache_dir="/tmp/ssd_cache",
            paged_ssd_cache_max_size="50GB",
        )
        with patch.dict(os.environ, {}, clear=True):
            config = OMLXConfig.from_cli_args(args)
            assert config.paged_ssd_cache.enabled is True
            assert config.paged_ssd_cache.cache_dir == Path("/tmp/ssd_cache")
            assert config.paged_ssd_cache.max_size == "50GB"

    def test_from_cli_args_mcp(self):
        """Test from_cli_args with MCP arguments."""
        args = Namespace(
            mcp_config="/path/to/mcp.json",
        )
        with patch.dict(os.environ, {}, clear=True):
            config = OMLXConfig.from_cli_args(args)
            assert config.mcp.enabled is True
            assert config.mcp.config_path == "/path/to/mcp.json"

    def test_to_dict(self):
        """Test to_dict method."""
        config = OMLXConfig()
        result = config.to_dict()

        assert "server" in result
        assert "model" in result
        assert "generation" in result
        assert "scheduler" in result
        assert "cache" in result
        assert "paged_ssd_cache" in result
        assert "mcp" in result
        assert "continuous_batching" in result

        assert result["server"]["host"] == "0.0.0.0"
        assert result["server"]["port"] == 8000

    def test_to_dict_with_paged_ssd_cache_dir(self):
        """Test to_dict with paged SSD cache directory."""
        config = OMLXConfig()
        config.paged_ssd_cache.cache_dir = Path("/tmp/cache")
        result = config.to_dict()

        assert result["paged_ssd_cache"]["cache_dir"] == "/tmp/cache"

    def test_validate_valid_config(self):
        """Test validate with valid configuration."""
        config = OMLXConfig()
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_port(self):
        """Test validate with invalid port."""
        config = OMLXConfig()
        config.server.port = 0
        errors = config.validate()
        assert any("port" in error.lower() for error in errors)

        config.server.port = 70000
        errors = config.validate()
        assert any("port" in error.lower() for error in errors)

    def test_validate_invalid_max_tokens(self):
        """Test validate with invalid max_tokens."""
        config = OMLXConfig()
        config.generation.max_tokens = 0
        errors = config.validate()
        assert any("max_tokens" in error.lower() for error in errors)

        config.generation.max_tokens = -1
        errors = config.validate()
        assert any("max_tokens" in error.lower() for error in errors)

    def test_validate_invalid_temperature(self):
        """Test validate with invalid temperature."""
        config = OMLXConfig()
        config.generation.temperature = -0.1
        errors = config.validate()
        assert any("temperature" in error.lower() for error in errors)

        config.generation.temperature = 2.5
        errors = config.validate()
        assert any("temperature" in error.lower() for error in errors)

    def test_validate_invalid_top_p(self):
        """Test validate with invalid top_p."""
        config = OMLXConfig()
        config.generation.top_p = -0.1
        errors = config.validate()
        assert any("top_p" in error.lower() for error in errors)

        config.generation.top_p = 1.5
        errors = config.validate()
        assert any("top_p" in error.lower() for error in errors)

    def test_validate_paged_ssd_cache_no_dir(self):
        """Test validate with paged SSD cache enabled but no directory."""
        config = OMLXConfig()
        config.paged_ssd_cache.enabled = True
        config.paged_ssd_cache.cache_dir = None
        errors = config.validate()
        assert any("cache_dir" in error.lower() for error in errors)

    def test_validate_multiple_errors(self):
        """Test validate with multiple errors."""
        config = OMLXConfig()
        config.server.port = 0
        config.generation.max_tokens = -1
        config.generation.temperature = -1.0
        errors = config.validate()
        assert len(errors) >= 3
