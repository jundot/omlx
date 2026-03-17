# SPDX-License-Identifier: Apache-2.0
"""
Logging configuration for oMLX.

This module provides centralized logging configuration with support for:
- Standard logging with configurable levels
- Structured JSON logging (optional)
- Request context tracking
- File logging with daily rotation
- Consistent formatting across all modules
"""

import logging
import sys
from contextvars import ContextVar
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

# Context variable for request ID tracking
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class RequestContextFilter(logging.Filter):
    """
    Add request_id to log records.

    This filter adds the current request ID (if set) to all log records,
    enabling request-level log correlation.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id attribute to log record."""
        record.request_id = _request_id.get() or "-"
        return True


class AdminStatsAccessFilter(logging.Filter):
    """Suppress repetitive uvicorn access logs for admin polling endpoints."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "/admin/api/stats" in msg:
            return False
        if "/admin/api/login" in msg:
            return False
        return True


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for terminal output.

    Uses ANSI color codes to highlight different log levels.
    """

    COLORS = {
        5: "\033[90m",                 # Gray (TRACE)
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        color = self.COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return _request_id.get()


def set_request_id(request_id: Optional[str]) -> None:
    """Set the current request ID in context."""
    _request_id.set(request_id)


def configure_logging(
    level: str = "INFO",
    format_style: str = "standard",
    include_request_id: bool = True,
    colored: bool = True,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_style: "standard" for plain text, "json" for structured JSON.
        include_request_id: Whether to include request_id in log format.
        colored: Whether to use colored output (only for standard format).
    """
    # Determine log level
    level_name = level.upper()
    log_level = 5 if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)

    # Build format string
    if include_request_id:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"
    else:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(log_level)

    # Create formatter
    if format_style == "json":
        formatter = JsonFormatter(format_str)
    elif colored and sys.stderr.isatty():
        formatter = ColoredFormatter(format_str)
    else:
        formatter = logging.Formatter(format_str)

    handler.setFormatter(formatter)

    # Add request context filter if needed
    if include_request_id:
        handler.addFilter(RequestContextFilter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Set specific loggers
    logging.getLogger("omlx").setLevel(log_level)
    logging.getLogger("uvicorn").setLevel(log_level)
    logging.getLogger("uvicorn.access").addFilter(AdminStatsAccessFilter())

    # Suppress noisy third-party loggers unless trace level
    third_party_level = log_level if log_level <= 5 else logging.INFO
    logging.getLogger("httpx").setLevel(third_party_level)
    logging.getLogger("httpcore").setLevel(third_party_level)


class JsonFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Outputs logs as JSON objects for easy parsing by log aggregators.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        import json
        import time

        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request_id if available
        request_id = getattr(record, "request_id", None)
        if request_id and request_id != "-":
            log_data["request_id"] = request_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key in ["request_id", "model", "tokens", "latency_ms"]:
            if hasattr(record, key) and key not in log_data:
                log_data[key] = getattr(record, key)

        return json.dumps(log_data)


def get_logger(
    name: str,
    request_id: Optional[str] = None,
) -> logging.Logger:
    """
    Get a logger with optional request context.

    Args:
        name: Logger name (usually __name__).
        request_id: Optional request ID to set in context.

    Returns:
        Logger instance.
    """
    if request_id:
        set_request_id(request_id)

    return logging.getLogger(name)


class RequestLogContext:
    """
    Context manager for request-scoped logging.

    Usage:
        with RequestLogContext(request_id="abc123"):
            logger.info("Processing request")
    """

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.previous_id: Optional[str] = None

    def __enter__(self) -> "RequestLogContext":
        self.previous_id = _request_id.get()
        _request_id.set(self.request_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _request_id.set(self.previous_id)


def configure_file_logging(
    log_dir: Path,
    level: str = "INFO",
    include_request_id: bool = True,
    retention_days: int = 7,
) -> None:
    """
    Configure file logging with daily rotation.

    Adds a file handler to the root logger that writes to {log_dir}/server.log
    with automatic daily rotation. Old log files are automatically deleted
    after retention_days.

    Args:
        log_dir: Directory to store log files.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        include_request_id: Whether to include request_id in log format.
        retention_days: Number of days to retain old logs.
    """
    # Ensure log directory exists
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    level_name = level.upper()
    log_level = 5 if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)

    # Build format string (no colors for file)
    if include_request_id:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"
    else:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create file handler with daily rotation
    # File: server.log, rotated files: server.log.YYYY-MM-DD
    log_file = log_dir / "server.log"

    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=retention_days,
        encoding="utf-8",
    )
    file_handler.suffix = "%Y-%m-%d"  # Results in server.log.2024-01-15
    file_handler.setLevel(log_level)

    formatter = logging.Formatter(format_str)
    file_handler.setFormatter(formatter)

    # Add request context filter
    if include_request_id:
        file_handler.addFilter(RequestContextFilter())

    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
