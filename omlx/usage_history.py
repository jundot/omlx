# SPDX-License-Identifier: Apache-2.0
"""Local, bounded serving history. No request objects or content enter this module."""

import logging
import math
import sqlite3
import threading
import time
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)
RETENTION_DAYS = 400
FLUSH_SECONDS = 5
_MAX_PENDING_BUCKETS = 4096
_FIELDS = (
    "requests",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "prefill_seconds",
    "generation_seconds",
    "request_seconds",
    "timed_requests",
)


def _hour(timestamp: float) -> int:
    # Preserve fold on the repeated DST hour; also handles half-hour offsets.
    return int(
        datetime.fromtimestamp(timestamp)
        .replace(minute=0, second=0, microsecond=0)
        .timestamp()
    )


def _summary(values) -> dict:
    result = dict(zip(_FIELDS, values, strict=True))
    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    result["cache_efficiency"] = (
        result["cached_tokens"] / result["prompt_tokens"]
        if result["prompt_tokens"]
        else 0.0
    )
    result["generation_tps"] = (
        result["completion_tokens"] / result["generation_seconds"]
        if result["generation_seconds"]
        else None
    )
    result["prefill_tps"] = (
        (result["prompt_tokens"] - result["cached_tokens"]) / result["prefill_seconds"]
        if result["prefill_seconds"]
        else None
    )
    result["average_request_seconds"] = (
        result["request_seconds"] / result["timed_requests"]
        if result["timed_requests"]
        else None
    )
    return result


def _bounds(period: str, now: float) -> tuple[datetime, datetime]:
    today = datetime.fromtimestamp(now).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end = today + timedelta(days=1)
    if period == "today":
        start = today
    elif period == "yesterday":
        start, end = today - timedelta(days=1), today
    elif period == "month":
        start = today.replace(day=1)
    elif period in ("7d", "30d", "90d"):
        start = today - timedelta(days=int(period[:-1]) - 1)
    else:
        raise ValueError("Unsupported usage range")
    return start, end


class UsageHistory:
    """One writer, bounded pending aggregates, short locks, no inference-path I/O.

    Queries read committed snapshots (up to FLUSH_SECONDS behind). SQLite work
    runs only at startup, on the writer, or on admin worker threads.
    """

    def __init__(self, path: Path):
        self.path = path.resolve()
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._pending: dict[tuple[int, str], list] = {}
        self._stop = threading.Event()
        self._closed = False
        self.available = False
        self.dropped_requests = 0
        self._last_prune = 0.0
        self._bucket_minute: int | None = None
        self._bucket_hour = 0
        try:
            self._initialize()
            self.available = True
        except (OSError, sqlite3.Error, ValueError):
            logger.warning("Usage history unavailable; serving continues")
        self._thread = threading.Thread(
            target=self._run, name="omlx-usage-history", daemon=True
        )
        self._thread.start()

    def _initialize(self) -> None:
        connection = None
        try:
            connection = self._connect()
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise sqlite3.DatabaseError("Usage integrity check failed")
        except sqlite3.DatabaseError as exc:
            code = getattr(exc, "sqlite_errorcode", None)
            if (
                code not in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB)
                and str(exc) != "Usage integrity check failed"
            ):
                raise
            if connection is not None:
                connection.close()
                connection = None
            # Keep one bounded backup for manual recovery. Never replace a newer
            # schema or mistake permissions/locking failures for corruption.
            self.path.replace(self.path.with_suffix(".sqlite3.corrupt"))
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(self.path) + suffix)
                if sidecar.exists():
                    sidecar.replace(Path(str(self.path) + ".corrupt" + suffix))
            logger.warning(
                "Corrupt usage history preserved as usage.sqlite3.corrupt; starting fresh"
            )
            connection = self._connect()
        finally:
            if connection is not None:
                connection.close()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=1)
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("Unsupported usage schema version")
            if version == 0:
                connection.execute("PRAGMA auto_vacuum=INCREMENTAL")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            if version == 0:
                with connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute("""
                        CREATE TABLE model_usage_hourly (
                            timestamp_hour INTEGER NOT NULL,
                            model_id TEXT NOT NULL,
                            requests INTEGER NOT NULL,
                            prompt_tokens INTEGER NOT NULL,
                            completion_tokens INTEGER NOT NULL,
                            cached_tokens INTEGER NOT NULL,
                            prefill_seconds REAL NOT NULL,
                            generation_seconds REAL NOT NULL,
                            request_seconds REAL NOT NULL,
                            timed_requests INTEGER NOT NULL,
                            PRIMARY KEY (timestamp_hour, model_id)
                        ) WITHOUT ROWID
                    """)
                    connection.execute("PRAGMA user_version=1")
            return connection
        except Exception:
            connection.close()
            raise

    def record(
        self,
        *,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        prefill_duration: float,
        generation_duration: float,
        request_duration: float | None = None,
        timestamp: float | None = None,
    ) -> None:
        # Validate only scalar counters. Never accept a request or arbitrary metadata.
        counts = (prompt_tokens, completion_tokens, cached_tokens)
        durations = (prefill_duration, generation_duration, request_duration or 0.0)
        if (
            not isinstance(model_id, str)
            or len(model_id) > 1024
            or any(not isinstance(n, int) or n < 0 for n in counts)
            or cached_tokens > prompt_tokens
            or any(not math.isfinite(n) or n < 0 for n in durations)
        ):
            return
        timestamp = time.time() if timestamp is None else timestamp
        minute = int(timestamp) // 60
        values = [1, *counts, *durations, int(request_duration is not None)]
        with self._lock:
            if self._closed:
                return
            # macOS local-time conversion is relatively expensive. Reuse one
            # minute's hour conversion; minute boundaries also cover fractional
            # offsets and DST transitions without per-request calendar work.
            if minute != self._bucket_minute:
                self._bucket_hour = _hour(timestamp)
                self._bucket_minute = minute
            key = (self._bucket_hour, model_id)
            if key not in self._pending:
                if len(self._pending) >= _MAX_PENDING_BUCKETS:
                    self.dropped_requests += 1
                    return
                self._pending[key] = values
            else:
                self._pending[key] = [
                    a + b for a, b in zip(self._pending[key], values, strict=True)
                ]

    def _run(self) -> None:
        while not self._stop.wait(FLUSH_SECONDS):
            self.flush()

    def flush(self) -> bool:
        """Persist a batch; never called by the inference path."""
        with self._flush_lock:
            with self._lock:
                batch, self._pending = self._pending, {}
            connection = None
            try:
                connection = self._connect()
                with connection:
                    connection.executemany(
                        "INSERT INTO model_usage_hourly VALUES (?,?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(timestamp_hour, model_id) DO UPDATE SET "
                        + ",".join(f"{f}={f}+excluded.{f}" for f in _FIELDS),
                        [
                            (hour, model, *values)
                            for (hour, model), values in batch.items()
                        ],
                    )
                    now = time.time()
                    if now - self._last_prune >= 86400:
                        connection.execute(
                            "DELETE FROM model_usage_hourly WHERE timestamp_hour < ?",
                            (_hour(now - RETENTION_DAYS * 86400),),
                        )
                if now - self._last_prune >= 86400:
                    self._last_prune = now
                    # Maintenance failure must not replay a committed batch.
                    with suppress(sqlite3.Error):
                        connection.execute("PRAGMA incremental_vacuum(100)")
                self.available = True
                return True
            except (OSError, sqlite3.Error, ValueError):
                if self.available:
                    logger.warning("Usage history write failed; serving continues")
                self.available = False
                # Keep a bounded aggregate for retry, never a growing request queue.
                with self._lock:
                    for key, values in batch.items():
                        if key in self._pending:
                            self._pending[key] = [
                                a + b
                                for a, b in zip(self._pending[key], values, strict=True)
                            ]
                        elif len(self._pending) < _MAX_PENDING_BUCKETS:
                            self._pending[key] = values
                        else:
                            self.dropped_requests += values[0]
                return False
            finally:
                if connection is not None:
                    connection.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._stop.set()
        self._thread.join(timeout=3)
        self.flush()

    def query(
        self, period: str = "today", model: str = "", *, now: float | None = None
    ) -> dict:
        now = time.time() if now is None else now
        start, end = _bounds(period, now)
        # Read-only connection: polling cannot silently recreate a deleted database.
        connection = None
        try:
            connection = sqlite3.connect(
                self.path.as_uri() + "?mode=ro", uri=True, timeout=1
            )
            rows = connection.execute(
                "SELECT * FROM model_usage_hourly WHERE timestamp_hour >= ? "
                "AND timestamp_hour < ?" + (" AND model_id = ?" if model else ""),
                (
                    (start.timestamp(), end.timestamp(), model)
                    if model
                    else (start.timestamp(), end.timestamp())
                ),
            ).fetchall()
        finally:
            if connection is not None:
                connection.close()
        totals = [0] * len(_FIELDS)
        models: dict[str, list] = {}
        days = {}
        day_cursor = start
        while day_cursor < end:
            days[day_cursor.date().isoformat()] = [0] * len(_FIELDS)
            day_cursor += timedelta(days=1)
        # 24 cells per calendar day, repeated DST hours combine; missing hours are zero.
        heatmap = {day: [0] * 24 for day in days}
        hourly: dict[int, list] = {}
        for hour, model_id, *values in rows:
            local = datetime.fromtimestamp(hour)
            day = local.date().isoformat()
            totals = [a + b for a, b in zip(totals, values, strict=True)]
            models.setdefault(model_id, [0] * len(_FIELDS))
            models[model_id] = [
                a + b for a, b in zip(models[model_id], values, strict=True)
            ]
            days[day] = [a + b for a, b in zip(days[day], values, strict=True)]
            heatmap[day][local.hour] += values[1] + values[2]
            hourly.setdefault(hour, [0] * len(_FIELDS))
            hourly[hour] = [a + b for a, b in zip(hourly[hour], values, strict=True)]
        return {
            "range": period,
            "start": start.astimezone().isoformat(),
            "end": end.astimezone().isoformat(),
            "timezone": "server local time",
            "retention_days": RETENTION_DAYS,
            "flush_seconds": FLUSH_SECONDS,
            "available": self.available,
            "dropped_requests": self.dropped_requests,
            "totals": _summary(totals),
            "models": sorted(
                [{"model_id": key, **_summary(value)} for key, value in models.items()],
                key=lambda item: item["total_tokens"],
                reverse=True,
            ),
            "daily": [{"date": key, **_summary(value)} for key, value in days.items()],
            "hourly": [
                {"timestamp_hour": key, **_summary(value)}
                for key, value in sorted(hourly.items())
            ],
            "heatmap": [
                {"date": key, "tokens": value} for key, value in heatmap.items()
            ],
        }
