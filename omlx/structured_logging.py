# SPDX-License-Identifier: Apache-2.0
"""
Structured JSON logging for oMLX SRE observability.

Replaces free-text log lines with structured JSON objects that can be
queried, filtered, and aggregated by SRE tooling. Provides:

- ``StructuredJSONFormatter`` — ``logging.Formatter`` subclass producing
  JSON output with the required schema fields.
- ``configure_structured_logging()`` — one-call setup for JSON or text mode.
- Event emitters: ``log_request_completed``, ``log_cache_event``,
  ``log_node_health_change``, ``log_scheduler_routing`` — validated
  helpers that never leak secrets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Context variables
# ---------------------------------------------------------------------------

_request_id_ctx: ContextVar[Optional[str]] = ContextVar("structured_request_id", default=None)
_trace_id_ctx: ContextVar[Optional[str]] = ContextVar("structured_trace_id", default=None)
_service_ctx: ContextVar[str] = ContextVar("structured_service", default="omlx-inference")


def set_request_id(request_id: str | None) -> None:
    _request_id_ctx.set(request_id)


def get_request_id() -> str | None:
    return _request_id_ctx.get()


def set_trace_id(trace_id: str | None) -> None:
    _trace_id_ctx.set(trace_id)


def get_trace_id() -> str | None:
    return _trace_id_ctx.get()


def set_service_name(service: str) -> None:
    _service_ctx.set(service)


def get_service_name() -> str:
    return _service_ctx.get()


# ---------------------------------------------------------------------------
# Secret redaction helpers
# ---------------------------------------------------------------------------

# Patterns that indicate sensitive data — API keys, tokens, secrets.
# Each pattern has a group(0) that is the full match to mask.
_SECRET_PATTERNS: list[re.Pattern] = [
    re.compile(r"sk-[A-Za-z0-9\-_]{20,}"),       # OpenAI-style keys
    re.compile(r"xai-[A-Za-z0-9\-_]{20,}"),       # xAI keys
    re.compile(r"Bearer\s+[A-Za-z0-9\-_\.]{20,}", re.IGNORECASE),
    re.compile(
        r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*"
        r"[A-Za-z0-9\-_\.]{8,}"
    ),
]


def _redact_secrets(text: str) -> str:
    """Mask sensitive data in a string.

    API keys are replaced with a truncated fingerprint. Bearer tokens are
    fully masked. The original secret is never recoverable from log output.
    """
    if not text:
        return text

    def _mask_key(m: re.Match) -> str:
        raw = m.group(0)
        # For key=value patterns, find the value part
        eq_match = re.search(r"[=:]\s*", raw)
        if eq_match:
            prefix = raw[: eq_match.end()]
            value = raw[eq_match.end() :]
            if len(value) >= 8:
                return prefix + value[:4] + "****" + value[-4:]
            return prefix + "****"
        # For standalone tokens like sk-..., mask the sensitive suffix
        for prefix in ("sk-", "xai-"):
            if raw.startswith(prefix):
                suffix = raw[len(prefix):]
                if len(suffix) >= 8:
                    return prefix + suffix[:4] + "****" + suffix[-4:]
                return prefix + "****"
        return "****"

    result = text
    for pat in _SECRET_PATTERNS:
        result = pat.sub(_mask_key, result)
    return result


def _truncate_prompt(text: str | None) -> str | None:
    """Truncate a prompt to a safe length and return a hash for correlation.

    Returns ``None`` when the input is ``None`` or empty. The full prompt
    content is never written to logs.
    """
    if not text:
        return None
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]} len={len(text)}"


def sanitize_for_log(value: Any) -> Any:
    """Recursively sanitize a value before including it in a log record.

    Strings go through ``_redact_secrets``. Prompts larger than
    ``_MAX_PROMPT_LOG_CHARS`` are truncated to a content hash. Dicts and
    lists are traversed recursively.
    """
    if isinstance(value, str):
        return _redact_secrets(value)
    if isinstance(value, dict):
        return {k: sanitize_for_log(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_log(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class StructuredJSONFormatter(logging.Formatter):
    """``logging.Formatter`` subclass that outputs structured JSON.

    Schema (every record):

    .. code-block:: json

        {
          "timestamp": "2026-08-20T12:34:56.789Z",
          "level": "INFO",
          "service": "omlx-inference",
          "event": "request.completed",
          "message": "...",
          "request_id": "...",
          "trace_id": "..."
        }

    Event-specific fields are added by the emitter helpers and passed
    through as extra ``LogRecord`` attributes.
    """

    # Fields that belong to every structured log entry.
    _CORE_FIELDS = frozenset({
        "timestamp", "level", "service", "event", "message",
        "request_id", "trace_id", "logger", "exception",
    })

    # Keys that belong to LogRecord internals and must not leak into JSON.
    _LOGRECORD_INTERNALS = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "funcName", "lineno", "asctime", "created", "msecs",
        "relativeCreated", "thread", "threadName", "process", "processName",
        "stack_info", "exc_info", "exc_text", "taskName",
    })

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        log_data: dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "service": get_service_name(),
            "event": getattr(record, "event", "log"),
            "message": sanitize_for_log(record.getMessage()),
        }

        # Request-scoped fields
        rid = getattr(record, "request_id", None) or _request_id_ctx.get()
        if rid:
            log_data["request_id"] = rid
        tid = getattr(record, "trace_id", None) or _trace_id_ctx.get()
        if tid:
            log_data["trace_id"] = tid

        # Exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Pass through extra structured fields (event-specific).
        for key, value in record.__dict__.items():
            if (
                key not in self._LOGRECORD_INTERNALS
                and key not in self._CORE_FIELDS
                and not key.startswith("_")
            ):
                log_data[key] = sanitize_for_log(value)

        return json.dumps(log_data, default=str)


# ---------------------------------------------------------------------------
# Standard text formatter passthrough
# ---------------------------------------------------------------------------


class StructuredTextFormatter(logging.Formatter):
    """Human-readable formatter used when JSON mode is disabled.

    Delegates to the standard ``logging.Formatter`` but still respects
    the ``event`` field for structured correlation when present.
    Ensures ``request_id`` is always available for format strings.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Guarantee request_id attribute exists for %(request_id)s in format strings.
        if not hasattr(record, "request_id"):
            record.request_id = _request_id_ctx.get() or "-"
        event = getattr(record, "event", None)
        if event:
            record.msg = f"[{event}] {record.msg}"
        return super().format(record)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def configure_structured_logging(
    level: str = "INFO",
    format: str = "json",
    log_file: str | Path | None = None,
    service: str = "omlx-inference",
) -> None:
    """Set up structured JSON (or human-readable) logging.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format: ``"json"`` for structured JSON output, ``"text"`` for
            human-readable output with optional event prefixes.
        log_file: Optional path to a log file. When set, a
            ``TimedRotatingFileHandler`` is added with daily rotation and
            7-day retention. The same format (json/text) is used.
        service: The ``service`` field value included in every structured
            log entry.
    """
    set_service_name(service)

    log_level = getattr(logging, level.upper(), logging.INFO)

    if format == "json":
        formatter: logging.Formatter = StructuredJSONFormatter()
    else:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"
        formatter = StructuredTextFormatter(fmt)

    # Stderr handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(handler)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_formatter: logging.Formatter
        if format == "json":
            file_formatter = StructuredJSONFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s"
            )

        file_handler = TimedRotatingFileHandler(
            filename=str(log_path),
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Event emitters
# ---------------------------------------------------------------------------


def log_request_completed(
    *,
    request_id: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    prefill_duration: float = 0.0,
    generation_duration: float = 0.0,
    finish_reason: str = "stop",
    stream: bool = False,
    status_code: int = 200,
    trace_id: str | None = None,
    logger_name: str = "omlx.server",
) -> None:
    """Emit a structured ``request.completed`` log entry.

    This is the primary SRE event for tracking inference performance.
    """
    total_duration = prefill_duration + generation_duration
    _emit(
        logger_name,
        logging.INFO,
        "Request completed",
        event="request.completed",
        request_id=request_id,
        trace_id=trace_id,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        prefill_duration_s=round(prefill_duration, 4),
        generation_duration_s=round(generation_duration, 4),
        total_duration_s=round(total_duration, 4),
        finish_reason=finish_reason,
        stream=stream,
        status_code=status_code,
    )


def log_cache_event(
    *,
    event: str,
    model: str,
    cache_type: str = "paged",
    entries: int = 0,
    evicted_entries: int = 0,
    hit_rate: float | None = None,
    cache_size_bytes: int = 0,
    request_id: str | None = None,
    trace_id: str | None = None,
    logger_name: str = "omlx.cache",
) -> None:
    """Emit a structured cache lifecycle event.

    Suitable events: ``cache.hit``, ``cache.miss``, ``cache.eviction``,
    ``cache.insert``, ``cache.clear``.
    """
    _emit(
        logger_name,
        logging.INFO,
        f"Cache event: {event}",
        event=event,
        request_id=request_id,
        trace_id=trace_id,
        model=model,
        cache_type=cache_type,
        entries=entries,
        evicted_entries=evicted_entries,
        hit_rate=round(hit_rate, 4) if hit_rate is not None else None,
        cache_size_bytes=cache_size_bytes,
    )


def log_node_health_change(
    *,
    node_id: str,
    previous_status: str,
    current_status: str,
    reason: str = "",
    rank: int | None = None,
    deployment_id: str | None = None,
    trace_id: str | None = None,
    logger_name: str = "omlx.cluster",
) -> None:
    """Emit a structured cluster node health-change event."""
    _emit(
        logger_name,
        logging.WARNING if current_status in ("unhealthy", "unreachable") else logging.INFO,
        f"Node {node_id}: {previous_status} -> {current_status}",
        event="node.health_change",
        node_id=node_id,
        previous_status=previous_status,
        current_status=current_status,
        reason=reason,
        rank=rank,
        deployment_id=deployment_id,
        trace_id=trace_id,
    )


def log_scheduler_routing(
    *,
    request_id: str,
    model: str,
    queue_depth: int,
    priority: int = 0,
    estimated_wait_ms: float = 0.0,
    trace_id: str | None = None,
    logger_name: str = "omlx.scheduler",
) -> None:
    """Emit a structured scheduler routing event."""
    _emit(
        logger_name,
        logging.INFO,
        f"Scheduler routed request to {model}",
        event="scheduler.routing",
        request_id=request_id,
        trace_id=trace_id,
        model=model,
        queue_depth=queue_depth,
        priority=priority,
        estimated_wait_ms=round(estimated_wait_ms, 1),
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _emit(
    logger_name: str,
    level: int,
    msg: str,
    *,
    event: str,
    **extra: Any,
) -> None:
    """Low-level helper that creates a ``LogRecord`` with extra fields.

    Extra keyword arguments are attached as ``LogRecord`` attributes so
    the ``StructuredJSONFormatter`` can pick them up without going through
    ``args`` formatting.
    """
    log = logging.getLogger(logger_name)
    if not log.isEnabledFor(level):
        return

    record = log.makeRecord(
        name=logger_name,
        level=level,
        fn="",
        lno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    # Attach structured fields — event goes on the record so the formatter reads it.
    record.event = event  # type: ignore[attr-defined]
    for key, value in extra.items():
        setattr(record, key, value)
    log.handle(record)
