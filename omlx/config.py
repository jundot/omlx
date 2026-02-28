# SPDX-License-Identifier: Apache-2.0
"""
Centralized configuration for oMLX.

This module provides unified configuration management with:
- Pydantic validation
- Environment variable support
- CLI argument mapping
- Default values with sensible defaults
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def parse_size(size_str: str) -> int:
    """
    Parse a human-readable size string to bytes.

    Args:
        size_str: Size string like "100GB", "50MB", "1TB".

    Returns:
        Size in bytes.
    """
    size_str = size_str.strip().upper()

    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }

    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                value = float(size_str[: -len(unit)])
                return int(value * multiplier)
            except ValueError:
                pass

    # Try parsing as plain number (bytes)
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size string: {size_str}")


@dataclass
class ServerConfig:
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str = ""
    trust_remote_code: bool = True
    model_path: Optional[str] = None


@dataclass
class GenerationConfig:
    """Generation parameters configuration."""

    max_tokens: int = 32768
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    force_sampling: bool = False


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    max_num_seqs: int = 8
    prefill_batch_size: int = 8
    completion_batch_size: int = 8
    stream_interval: int = 1
    enable_thinking: Optional[bool] = None


@dataclass
class CacheConfig:
    """Cache configuration (deprecated, kept for compatibility)."""

    # All cache options moved to PagedSSDCacheConfig
    pass


@dataclass
class PagedSSDCacheConfig:
    """Paged SSD cache configuration. oMLX only supports paged SSD-based caching."""

    enabled: bool = False
    cache_dir: Optional[Path] = None
    max_size: str = "100GB"
    hot_cache_max_size: str = "0"  # "0" = disabled, e.g. "8GB"

    @property
    def max_size_bytes(self) -> int:
        """Get max size in bytes."""
        return parse_size(self.max_size)

    @property
    def hot_cache_max_size_bytes(self) -> int:
        """Get hot cache max size in bytes. 0 means disabled."""
        return parse_size(self.hot_cache_max_size)


@dataclass
class MCPConfig:
    """MCP (Model Context Protocol) configuration."""

    config_path: Optional[str] = None
    enabled: bool = False


@dataclass
class OMLXConfig:
    """
    Centralized configuration for oMLX.

    This class combines all configuration sections and provides
    environment variable overrides and CLI argument mapping.
    """

    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    paged_ssd_cache: PagedSSDCacheConfig = field(default_factory=PagedSSDCacheConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)

    # Feature flags
    continuous_batching: bool = False

    @classmethod
    def from_env(cls) -> "OMLXConfig":
        """
        Create config from environment variables.

        Environment variables are prefixed with OMLX_.
        """
        config = cls()

        # Server settings
        config.server.host = os.getenv("OMLX_HOST", config.server.host)
        config.server.port = int(os.getenv("OMLX_PORT", str(config.server.port)))
        config.server.log_level = os.getenv("OMLX_LOG_LEVEL", config.server.log_level)

        # Model settings
        config.model.model_name = os.getenv("OMLX_MODEL", config.model.model_name)
        config.model.trust_remote_code = os.getenv(
            "OMLX_TRUST_REMOTE_CODE", "true"
        ).lower() == "true"

        # Generation settings
        config.generation.max_tokens = int(
            os.getenv("OMLX_MAX_TOKENS", str(config.generation.max_tokens))
        )
        config.generation.temperature = float(
            os.getenv("OMLX_TEMPERATURE", str(config.generation.temperature))
        )

        # Paged SSD cache settings
        paged_ssd_dir = os.getenv("OMLX_PAGED_SSD_CACHE_DIR")
        if paged_ssd_dir:
            config.paged_ssd_cache.enabled = True
            config.paged_ssd_cache.cache_dir = Path(paged_ssd_dir)
            config.paged_ssd_cache.max_size = os.getenv(
                "OMLX_PAGED_SSD_CACHE_MAX_SIZE", config.paged_ssd_cache.max_size
            )

        # MCP settings
        mcp_config = os.getenv("OMLX_MCP_CONFIG")
        if mcp_config:
            config.mcp.enabled = True
            config.mcp.config_path = mcp_config

        # Feature flags
        config.continuous_batching = os.getenv(
            "OMLX_CONTINUOUS_BATCHING", "false"
        ).lower() == "true"

        return config

    @classmethod
    def from_cli_args(cls, args: Any) -> "OMLXConfig":
        """
        Create config from argparse namespace.

        Args:
            args: Argparse namespace with CLI arguments.

        Returns:
            OMLXConfig instance.
        """
        config = cls.from_env()  # Start with env vars

        # Override with CLI args if provided
        if hasattr(args, "host") and args.host:
            config.server.host = args.host
        if hasattr(args, "port") and args.port:
            config.server.port = args.port
        if hasattr(args, "log_level") and args.log_level:
            config.server.log_level = args.log_level

        if hasattr(args, "model") and args.model:
            config.model.model_name = args.model
        if hasattr(args, "trust_remote_code"):
            config.model.trust_remote_code = args.trust_remote_code

        if hasattr(args, "max_tokens") and args.max_tokens:
            config.generation.max_tokens = args.max_tokens
        if hasattr(args, "temperature") and args.temperature is not None:
            config.generation.temperature = args.temperature
        if hasattr(args, "top_p") and args.top_p is not None:
            config.generation.top_p = args.top_p
        if hasattr(args, "top_k") and args.top_k is not None:
            config.generation.top_k = args.top_k

        if hasattr(args, "continuous_batching"):
            config.continuous_batching = args.continuous_batching

        # Paged SSD cache settings
        if hasattr(args, "paged_ssd_cache_dir") and args.paged_ssd_cache_dir:
            config.paged_ssd_cache.enabled = True
            config.paged_ssd_cache.cache_dir = Path(args.paged_ssd_cache_dir)
        if hasattr(args, "paged_ssd_cache_max_size") and args.paged_ssd_cache_max_size:
            config.paged_ssd_cache.max_size = args.paged_ssd_cache_max_size

        if hasattr(args, "mcp_config") and args.mcp_config:
            config.mcp.enabled = True
            config.mcp.config_path = args.mcp_config

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict

        return {
            "server": asdict(self.server),
            "model": asdict(self.model),
            "generation": asdict(self.generation),
            "scheduler": asdict(self.scheduler),
            "cache": asdict(self.cache),
            "paged_ssd_cache": {
                **asdict(self.paged_ssd_cache),
                "cache_dir": str(self.paged_ssd_cache.cache_dir) if self.paged_ssd_cache.cache_dir else None,
            },
            "mcp": asdict(self.mcp),
            "continuous_batching": self.continuous_batching,
        }

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []

        # Server validation
        if not 0 < self.server.port < 65536:
            errors.append(f"Invalid port: {self.server.port}")

        # Generation validation
        if self.generation.max_tokens <= 0:
            errors.append(f"max_tokens must be positive: {self.generation.max_tokens}")
        if not 0.0 <= self.generation.temperature <= 2.0:
            errors.append(f"temperature must be 0.0-2.0: {self.generation.temperature}")
        if not 0.0 <= self.generation.top_p <= 1.0:
            errors.append(f"top_p must be 0.0-1.0: {self.generation.top_p}")

        # Paged SSD cache validation
        if self.paged_ssd_cache.enabled:
            if not self.paged_ssd_cache.cache_dir:
                errors.append("Paged SSD cache enabled but no cache_dir specified")

        return errors
