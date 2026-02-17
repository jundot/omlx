# SPDX-License-Identifier: Apache-2.0
"""
Global settings management for oMLX.

This module provides a centralized settings system with:
- Hierarchical configuration (CLI > env > file > defaults)
- Automatic directory creation
- System resource detection (RAM, SSD capacity)
- Settings persistence to JSON file

Usage:
    from omlx.settings import init_settings, get_settings

    # At startup
    init_settings(cli_args=args)

    # Anywhere else
    settings = get_settings()
    print(settings.server.port)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import parse_size

if TYPE_CHECKING:
    from .scheduler import SchedulerConfig

logger = logging.getLogger(__name__)

# Settings file version for future migrations
SETTINGS_VERSION = "1.0"

# Default base path
DEFAULT_BASE_PATH = Path.home() / ".omlx"


def get_system_memory() -> int:
    """
    Return total system RAM in bytes.

    Uses psutil if available, falls back to os.sysconf on Unix.

    Returns:
        Total RAM in bytes.
    """
    try:
        import psutil

        return psutil.virtual_memory().total
    except ImportError:
        pass

    # Fallback for Unix systems
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size
    except (AttributeError, ValueError):
        pass

    # Default to 16GB if detection fails
    logger.warning("Could not detect system memory, defaulting to 16GB")
    return 16 * 1024**3


def get_ssd_capacity(path: str | Path) -> int:
    """
    Return disk capacity in bytes for the given path.

    Args:
        path: Path to check disk capacity for.

    Returns:
        Total disk capacity in bytes.
    """
    path = Path(path).expanduser().resolve()

    # Ensure parent directory exists for capacity check
    check_path = path
    while not check_path.exists() and check_path.parent != check_path:
        check_path = check_path.parent

    try:
        usage = shutil.disk_usage(check_path)
        return usage.total
    except OSError as e:
        logger.warning(f"Could not get disk capacity for {path}: {e}")
        # Default to 500GB if detection fails
        return 500 * 1024**3


@dataclass
class ServerSettings:
    """Server configuration settings."""

    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServerSettings:
        """Create from dictionary."""
        return cls(
            host=data.get("host", "127.0.0.1"),
            port=data.get("port", 8000),
            log_level=data.get("log_level", "info"),
            cors_origins=data.get("cors_origins", ["*"]),
        )


@dataclass
class ModelSettings:
    """Model configuration settings."""

    model_dir: str | None = None  # None means ~/.omlx/models
    max_model_memory: str = "auto"  # "auto" means 80% of RAM

    def get_model_dir(self, base_path: Path) -> Path:
        """
        Get the resolved model directory path.

        Args:
            base_path: Base oMLX directory.

        Returns:
            Resolved model directory path.
        """
        if self.model_dir:
            return Path(self.model_dir).expanduser().resolve()
        return base_path / "models"

    def get_max_model_memory_bytes(self) -> int:
        """
        Get max model memory in bytes.

        Returns:
            Max model memory in bytes (80% of RAM if "auto").
        """
        if self.max_model_memory.lower() == "auto":
            return int(get_system_memory() * 0.8)
        return parse_size(self.max_model_memory)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_dir": self.model_dir,
            "max_model_memory": self.max_model_memory,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSettings:
        """Create from dictionary."""
        return cls(
            model_dir=data.get("model_dir"),
            max_model_memory=data.get("max_model_memory", "auto"),
        )


@dataclass
class SchedulerSettings:
    """Scheduler configuration settings."""

    max_num_seqs: int = 8
    prefill_batch_size: int = 8
    completion_batch_size: int = 8

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchedulerSettings:
        """Create from dictionary."""
        return cls(
            max_num_seqs=data.get("max_num_seqs", 8),
            prefill_batch_size=data.get("prefill_batch_size", 8),
            completion_batch_size=data.get("completion_batch_size", 8),
        )


@dataclass
class CacheSettings:
    """Cache configuration settings."""

    enabled: bool = True
    ssd_cache_dir: str | None = None  # None means ~/.omlx/cache
    ssd_cache_max_size: str = "auto"  # "auto" means 10% of SSD capacity

    def get_ssd_cache_dir(self, base_path: Path) -> Path:
        """
        Get the resolved SSD cache directory path.

        Args:
            base_path: Base oMLX directory.

        Returns:
            Resolved SSD cache directory path.
        """
        if self.ssd_cache_dir:
            return Path(self.ssd_cache_dir).expanduser().resolve()
        return base_path / "cache"

    def get_ssd_cache_max_size_bytes(self, base_path: Path) -> int:
        """
        Get max SSD cache size in bytes.

        Args:
            base_path: Base oMLX directory.

        Returns:
            Max SSD cache size in bytes (10% of SSD if "auto").
        """
        if self.ssd_cache_max_size.lower() == "auto":
            cache_dir = self.get_ssd_cache_dir(base_path)
            return int(get_ssd_capacity(cache_dir) * 0.1)
        return parse_size(self.ssd_cache_max_size)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "ssd_cache_dir": self.ssd_cache_dir,
            "ssd_cache_max_size": self.ssd_cache_max_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheSettings:
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            ssd_cache_dir=data.get("ssd_cache_dir"),
            ssd_cache_max_size=data.get("ssd_cache_max_size", "auto"),
        )


@dataclass
class AuthSettings:
    """Authentication configuration settings."""

    api_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"api_key": self.api_key}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthSettings:
        """Create from dictionary."""
        return cls(api_key=data.get("api_key"))


@dataclass
class MCPSettings:
    """MCP (Model Context Protocol) configuration settings."""

    config_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"config_path": self.config_path}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPSettings:
        """Create from dictionary."""
        return cls(config_path=data.get("config_path"))


@dataclass
class SamplingSettings:
    """Default sampling parameters for generation."""

    max_context_window: int = 32768
    max_tokens: int = 32768
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    repetition_penalty: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_context_window": self.max_context_window,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplingSettings:
        """Create from dictionary."""
        return cls(
            max_context_window=data.get("max_context_window", 32768),
            max_tokens=data.get("max_tokens", 32768),
            temperature=data.get("temperature", 1.0),
            top_p=data.get("top_p", 0.95),
            top_k=data.get("top_k", 40),
            repetition_penalty=data.get("repetition_penalty", 1.0),
        )


@dataclass
class LoggingSettings:
    """Logging configuration settings."""

    log_dir: str | None = None  # None means {base_path}/logs
    retention_days: int = 7  # Number of days to keep rotated log files

    def get_log_dir(self, base_path: Path) -> Path:
        """
        Get the resolved log directory path.

        Args:
            base_path: Base oMLX directory.

        Returns:
            Resolved log directory path.
        """
        if self.log_dir:
            return Path(self.log_dir).expanduser().resolve()
        return base_path / "logs"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "log_dir": self.log_dir,
            "retention_days": self.retention_days,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoggingSettings:
        """Create from dictionary."""
        return cls(
            log_dir=data.get("log_dir"),
            retention_days=data.get("retention_days", 7),
        )


@dataclass
class ClaudeCodeSettings:
    """Claude Code integration settings."""

    context_scaling_enabled: bool = False
    target_context_size: int = 200000  # Claude Code default (200k)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "context_scaling_enabled": self.context_scaling_enabled,
            "target_context_size": self.target_context_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClaudeCodeSettings:
        """Create from dictionary."""
        return cls(
            context_scaling_enabled=data.get("context_scaling_enabled", False),
            target_context_size=data.get("target_context_size", 200000),
        )


@dataclass
class GlobalSettings:
    """
    Global settings for oMLX.

    Combines all settings sections and provides methods for:
    - Loading from file with CLI/env overrides
    - Saving to file
    - Directory management
    - Validation
    """

    base_path: Path = field(default_factory=lambda: DEFAULT_BASE_PATH)
    server: ServerSettings = field(default_factory=ServerSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    scheduler: SchedulerSettings = field(default_factory=SchedulerSettings)
    cache: CacheSettings = field(default_factory=CacheSettings)
    auth: AuthSettings = field(default_factory=AuthSettings)
    mcp: MCPSettings = field(default_factory=MCPSettings)
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    claude_code: ClaudeCodeSettings = field(default_factory=ClaudeCodeSettings)

    @classmethod
    def load(
        cls,
        base_path: str | Path | None = None,
        cli_args: Any | None = None,
    ) -> GlobalSettings:
        """
        Load settings with priority hierarchy: CLI > env > file > defaults.

        Args:
            base_path: Base directory for oMLX (default: ~/.omlx).
            cli_args: Argparse namespace with CLI arguments.

        Returns:
            Loaded GlobalSettings instance.
        """
        # Resolve base path
        if base_path:
            resolved_base = Path(base_path).expanduser().resolve()
        else:
            resolved_base = DEFAULT_BASE_PATH

        # Start with defaults
        settings = cls(base_path=resolved_base)

        # Load from file if exists
        settings_file = resolved_base / "settings.json"
        if settings_file.exists():
            settings._load_from_file(settings_file)
            logger.debug(f"Loaded settings from {settings_file}")

        # Apply environment variable overrides
        settings._apply_env_overrides()

        # Apply CLI argument overrides
        if cli_args:
            settings._apply_cli_overrides(cli_args)

        return settings

    def _load_from_file(self, path: Path) -> None:
        """
        Load settings from a JSON file.

        Args:
            path: Path to the settings JSON file.
        """
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            # Check version for future migrations
            version = data.get("version", "1.0")
            if version != SETTINGS_VERSION:
                logger.info(
                    f"Settings file version {version} differs from "
                    f"current {SETTINGS_VERSION}, migrating..."
                )

            # Load each section
            if "server" in data:
                self.server = ServerSettings.from_dict(data["server"])
            if "model" in data:
                self.model = ModelSettings.from_dict(data["model"])
            if "scheduler" in data:
                self.scheduler = SchedulerSettings.from_dict(data["scheduler"])
            if "cache" in data:
                self.cache = CacheSettings.from_dict(data["cache"])
            if "auth" in data:
                self.auth = AuthSettings.from_dict(data["auth"])
            if "mcp" in data:
                self.mcp = MCPSettings.from_dict(data["mcp"])
            if "sampling" in data:
                self.sampling = SamplingSettings.from_dict(data["sampling"])
            if "logging" in data:
                self.logging = LoggingSettings.from_dict(data["logging"])
            if "claude_code" in data:
                self.claude_code = ClaudeCodeSettings.from_dict(data["claude_code"])

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse settings file {path}: {e}")
        except OSError as e:
            logger.warning(f"Failed to read settings file {path}: {e}")

    def _apply_env_overrides(self) -> None:
        """Apply OMLX_* environment variable overrides."""
        # Server settings
        if host := os.getenv("OMLX_HOST"):
            self.server.host = host
        if port := os.getenv("OMLX_PORT"):
            try:
                self.server.port = int(port)
            except ValueError:
                logger.warning(f"Invalid OMLX_PORT value: {port}")
        if log_level := os.getenv("OMLX_LOG_LEVEL"):
            self.server.log_level = log_level

        # Model settings
        if model_dir := os.getenv("OMLX_MODEL_DIR"):
            self.model.model_dir = model_dir
        if max_model_memory := os.getenv("OMLX_MAX_MODEL_MEMORY"):
            self.model.max_model_memory = max_model_memory

        # Scheduler settings
        if max_num_seqs := os.getenv("OMLX_MAX_NUM_SEQS"):
            try:
                self.scheduler.max_num_seqs = int(max_num_seqs)
            except ValueError:
                logger.warning(f"Invalid OMLX_MAX_NUM_SEQS value: {max_num_seqs}")
        if prefill_batch := os.getenv("OMLX_PREFILL_BATCH_SIZE"):
            try:
                self.scheduler.prefill_batch_size = int(prefill_batch)
            except ValueError:
                logger.warning(f"Invalid OMLX_PREFILL_BATCH_SIZE: {prefill_batch}")
        if completion_batch := os.getenv("OMLX_COMPLETION_BATCH_SIZE"):
            try:
                self.scheduler.completion_batch_size = int(completion_batch)
            except ValueError:
                logger.warning(
                    f"Invalid OMLX_COMPLETION_BATCH_SIZE: {completion_batch}"
                )

        # Cache settings
        if cache_enabled := os.getenv("OMLX_CACHE_ENABLED"):
            self.cache.enabled = cache_enabled.lower() in ("true", "1", "yes")
        if ssd_cache_dir := os.getenv("OMLX_SSD_CACHE_DIR"):
            self.cache.ssd_cache_dir = ssd_cache_dir
        if ssd_cache_max := os.getenv("OMLX_SSD_CACHE_MAX_SIZE"):
            self.cache.ssd_cache_max_size = ssd_cache_max

        # Auth settings
        if api_key := os.getenv("OMLX_API_KEY"):
            self.auth.api_key = api_key

        # MCP settings
        if mcp_config := os.getenv("OMLX_MCP_CONFIG"):
            self.mcp.config_path = mcp_config

        # Logging settings
        if log_dir := os.getenv("OMLX_LOG_DIR"):
            self.logging.log_dir = log_dir
        if retention_days := os.getenv("OMLX_LOG_RETENTION_DAYS"):
            try:
                self.logging.retention_days = int(retention_days)
            except ValueError:
                logger.warning(f"Invalid OMLX_LOG_RETENTION_DAYS: {retention_days}")

    def _apply_cli_overrides(self, args: Any) -> None:
        """
        Apply CLI argument overrides.

        Args:
            args: Argparse namespace with CLI arguments.
        """
        # Server settings
        if hasattr(args, "host") and args.host is not None:
            self.server.host = args.host
        if hasattr(args, "port") and args.port is not None:
            self.server.port = args.port
        if hasattr(args, "log_level") and args.log_level is not None:
            self.server.log_level = args.log_level

        # Model settings
        if hasattr(args, "model_dir") and args.model_dir is not None:
            self.model.model_dir = args.model_dir
        if hasattr(args, "max_model_memory") and args.max_model_memory is not None:
            self.model.max_model_memory = args.max_model_memory

        # Scheduler settings
        if hasattr(args, "max_num_seqs") and args.max_num_seqs is not None:
            self.scheduler.max_num_seqs = args.max_num_seqs
        if hasattr(args, "prefill_batch_size") and args.prefill_batch_size is not None:
            self.scheduler.prefill_batch_size = args.prefill_batch_size
        if (
            hasattr(args, "completion_batch_size")
            and args.completion_batch_size is not None
        ):
            self.scheduler.completion_batch_size = args.completion_batch_size

        # Cache settings
        if hasattr(args, "cache_enabled") and args.cache_enabled is not None:
            self.cache.enabled = args.cache_enabled
        if hasattr(args, "ssd_cache_dir") and args.ssd_cache_dir is not None:
            self.cache.ssd_cache_dir = args.ssd_cache_dir
        if hasattr(args, "ssd_cache_max_size") and args.ssd_cache_max_size is not None:
            self.cache.ssd_cache_max_size = args.ssd_cache_max_size

        # Auth settings
        if hasattr(args, "api_key") and args.api_key is not None:
            self.auth.api_key = args.api_key

        # MCP settings
        if hasattr(args, "mcp_config") and args.mcp_config is not None:
            self.mcp.config_path = args.mcp_config

    def save(self) -> None:
        """Save current settings to the settings file."""
        self.ensure_directories()

        settings_file = self.base_path / "settings.json"
        data = {
            "version": SETTINGS_VERSION,
            "server": self.server.to_dict(),
            "model": self.model.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "cache": self.cache.to_dict(),
            "auth": self.auth.to_dict(),
            "mcp": self.mcp.to_dict(),
            "sampling": self.sampling.to_dict(),
            "logging": self.logging.to_dict(),
            "claude_code": self.claude_code.to_dict(),
        }

        try:
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved settings to {settings_file}")
        except OSError as e:
            logger.error(f"Failed to save settings to {settings_file}: {e}")
            raise

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.base_path,
            self.model.get_model_dir(self.base_path),
            self.cache.get_ssd_cache_dir(self.base_path),
            self.logging.get_log_dir(self.base_path),
        ]

        for directory in directories:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    logger.debug(f"Created directory: {directory}")
                except OSError as e:
                    logger.error(f"Failed to create directory {directory}: {e}")
                    raise

    def validate(self) -> list[str]:
        """
        Validate all settings.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        # Server validation
        if not 1 <= self.server.port <= 65535:
            errors.append(
                f"Invalid port: {self.server.port} (must be 1-65535)"
            )

        valid_log_levels = {"debug", "info", "warning", "error", "critical"}
        if self.server.log_level.lower() not in valid_log_levels:
            errors.append(
                f"Invalid log_level: {self.server.log_level} "
                f"(must be one of {valid_log_levels})"
            )

        # Model validation
        if self.model.max_model_memory.lower() != "auto":
            try:
                size = parse_size(self.model.max_model_memory)
                if size <= 0:
                    errors.append("max_model_memory must be positive")
            except ValueError as e:
                errors.append(f"Invalid max_model_memory: {e}")

        # Scheduler validation
        if self.scheduler.max_num_seqs <= 0:
            errors.append(
                f"Invalid max_num_seqs: {self.scheduler.max_num_seqs} (must be > 0)"
            )
        if self.scheduler.prefill_batch_size <= 0:
            errors.append(
                f"Invalid prefill_batch_size: {self.scheduler.prefill_batch_size} "
                "(must be > 0)"
            )
        if self.scheduler.completion_batch_size <= 0:
            errors.append(
                f"Invalid completion_batch_size: "
                f"{self.scheduler.completion_batch_size} (must be > 0)"
            )

        # Cache validation
        if self.cache.ssd_cache_max_size.lower() != "auto":
            try:
                size = parse_size(self.cache.ssd_cache_max_size)
                if size <= 0:
                    errors.append("ssd_cache_max_size must be positive")
            except ValueError as e:
                errors.append(f"Invalid ssd_cache_max_size: {e}")

        # Sampling validation
        if self.sampling.max_tokens <= 0:
            errors.append(
                f"Invalid sampling max_tokens: {self.sampling.max_tokens} (must be > 0)"
            )
        if not 0.0 <= self.sampling.temperature <= 2.0:
            errors.append(
                f"Invalid sampling temperature: {self.sampling.temperature} "
                "(must be 0.0-2.0)"
            )
        if not 0.0 <= self.sampling.top_p <= 1.0:
            errors.append(
                f"Invalid sampling top_p: {self.sampling.top_p} (must be 0.0-1.0)"
            )
        if self.sampling.top_k < 0:
            errors.append(
                f"Invalid sampling top_k: {self.sampling.top_k} (must be >= 0)"
            )

        # Claude Code validation
        if self.claude_code.target_context_size <= 0:
            errors.append(
                f"Invalid target_context_size: "
                f"{self.claude_code.target_context_size} (must be > 0)"
            )

        return errors

    def to_scheduler_config(self) -> SchedulerConfig:
        """
        Convert settings to SchedulerConfig for engine initialization.

        Returns:
            SchedulerConfig instance with values from settings.
        """
        from .scheduler import SchedulerConfig

        return SchedulerConfig(
            max_num_seqs=self.scheduler.max_num_seqs,
            prefill_batch_size=self.scheduler.prefill_batch_size,
            completion_batch_size=self.scheduler.completion_batch_size,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert all settings to a dictionary."""
        return {
            "version": SETTINGS_VERSION,
            "base_path": str(self.base_path),
            "server": self.server.to_dict(),
            "model": self.model.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "cache": self.cache.to_dict(),
            "auth": self.auth.to_dict(),
            "mcp": self.mcp.to_dict(),
            "sampling": self.sampling.to_dict(),
            "logging": self.logging.to_dict(),
            "claude_code": self.claude_code.to_dict(),
        }


# Global singleton instance
_global_settings: GlobalSettings | None = None


def get_settings() -> GlobalSettings:
    """
    Get the global settings instance.

    Returns:
        The global GlobalSettings instance.

    Raises:
        RuntimeError: If settings have not been initialized.
    """
    global _global_settings
    if _global_settings is None:
        raise RuntimeError(
            "Settings not initialized. Call init_settings() first."
        )
    return _global_settings


def init_settings(
    base_path: str | Path | None = None,
    cli_args: Any | None = None,
) -> GlobalSettings:
    """
    Initialize global settings (call once at startup).

    Args:
        base_path: Base directory for oMLX (default: ~/.omlx).
        cli_args: Argparse namespace with CLI arguments.

    Returns:
        The initialized GlobalSettings instance.
    """
    global _global_settings
    _global_settings = GlobalSettings.load(base_path=base_path, cli_args=cli_args)
    logger.info(f"Initialized settings with base_path: {_global_settings.base_path}")
    return _global_settings


def reset_settings() -> None:
    """
    Reset global settings (primarily for testing).

    This clears the global singleton, allowing init_settings to be called again.
    """
    global _global_settings
    _global_settings = None
