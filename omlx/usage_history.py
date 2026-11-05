# SPDX-License-Identifier: Apache-2.0
"""Local, bounded serving history. No request objects or content enter this module.

Per-client history (opt-in) receives only short labels from
``omlx.client_identity``: an API key's name and the peer IP, never the key itself.
"""

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
_KEY_KINDS = ("main_key", "sub_key", "none")
_MAX_CLIENT_LABEL = 256
# Added without bumping user_version: older builds ignore the extra table, so
# downgrading keeps model history readable instead of refusing the file.
_CLIENT_TABLE = """
    CREATE TABLE IF NOT EXISTS client_usage_hourly (
        timestamp_hour INTEGER NOT NULL,
        model_id TEXT NOT NULL,
        key_kind TEXT NOT NULL,
        key_id TEXT NOT NULL,
        client_ip TEXT NOT NULL,
        requests INTEGER NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        completion_tokens INTEGER NOT NULL,
        cached_tokens INTEGER NOT NULL,
        prefill_seconds REAL NOT NULL,
        generation_seconds REAL NOT NULL,
        request_seconds REAL NOT NULL,
        timed_requests INTEGER NOT NULL,
        PRIMARY KEY (timestamp_hour, model_id, key_kind, key_id, client_ip)
    ) WITHOUT ROWID
"""


def _add(target: dict, key: tuple, values: list) -> bool:
    """Accumulate into a bounded pending map; False when a new key won't fit."""
    if key in target:
        target[key] = [a + b for a, b in zip(target[key], values, strict=True)]
    elif len(target) < _MAX_PENDING_BUCKETS:
        target[key] = values
    else:
        return False
    return True


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


def _ranked(rows) -> list[dict]:
    return sorted(rows, key=lambda item: item["total_tokens"], reverse=True)


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

    def __init__(self, path: Path, *, enabled: bool = True, by_client: bool = False):
        self.path = path.resolve()
        self.enabled = enabled
        self.by_client = by_client
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._pending: dict[tuple[int, str], list] = {}
        # Separate bound: many distinct clients must not crowd out model totals.
        self._pending_clients: dict[tuple[int, str, str, str, str], list] = {}
        self._stop = threading.Event()
        self._closed = False
        self.available = False
        self._initialized = False
        self.dropped_requests = 0
        self.dropped_client_requests = 0
        self._last_prune = 0.0
        self._bucket_minute: int | None = None
        self._bucket_hour = 0
        if enabled:
            try:
                self._initialize()
                self._initialized = True
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
            connection.execute(_CLIENT_TABLE)
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
        client: tuple[str, str, str] | None = None,
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
            if self._closed or not self.enabled:
                return
            # macOS local-time conversion is relatively expensive. Reuse one
            # minute's hour conversion; minute boundaries also cover fractional
            # offsets and DST transitions without per-request calendar work.
            if minute != self._bucket_minute:
                self._bucket_hour = _hour(timestamp)
                self._bucket_minute = minute
            if not _add(self._pending, (self._bucket_hour, model_id), values):
                self.dropped_requests += 1
                return
            if not self.by_client or client is None:
                return
            kind, label, ip = client
            if kind not in _KEY_KINDS or any(
                not isinstance(text, str) or len(text) > _MAX_CLIENT_LABEL
                for text in (label, ip)
            ):
                return
            key = (self._bucket_hour, model_id, kind, label, ip)
            if not _add(self._pending_clients, key, list(values)):
                self.dropped_client_requests += 1

    def set_enabled(self, enabled: bool) -> None:
        """Runtime toggle. Disabling flushes pending aggregates; the file stays."""
        with self._lock:
            changed = self.enabled != enabled
            self.enabled = enabled
        if changed:
            self.flush()

    def set_by_client(self, enabled: bool) -> None:
        """Runtime toggle for per-client attribution. Existing rows are kept."""
        with self._lock:
            self.by_client = enabled

    def _run(self) -> None:
        while not self._stop.wait(FLUSH_SECONDS):
            # While disabled, only retry aggregates left over from a failed
            # flush; otherwise leave usage.sqlite3 alone.
            if self.enabled or self._pending or self._pending_clients:
                self.flush()

    def flush(self) -> bool:
        """Persist a batch; never called by the inference path."""
        with self._flush_lock:
            with self._lock:
                batch, self._pending = self._pending, {}
                client_batch, self._pending_clients = self._pending_clients, {}
            now = time.time()
            prune_due = now - self._last_prune >= 86400
            if (
                not batch
                and not client_batch
                and (not self.enabled or (self.available and not prune_due))
            ):
                return True
            connection = None
            try:
                # Initialization is deferred when recording starts disabled.
                if not self._initialized:
                    self._initialize()
                    self._initialized = True
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
                    connection.executemany(
                        "INSERT INTO client_usage_hourly "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(timestamp_hour, model_id, key_kind, key_id, "
                        "client_ip) "
                        "DO UPDATE SET "
                        + ",".join(f"{f}={f}+excluded.{f}" for f in _FIELDS),
                        [(*key, *values) for key, values in client_batch.items()],
                    )
                    if prune_due:
                        cutoff = _hour(now - RETENTION_DAYS * 86400)
                        for table in ("model_usage_hourly", "client_usage_hourly"):
                            connection.execute(
                                f"DELETE FROM {table} WHERE timestamp_hour < ?",
                                (cutoff,),
                            )
                if prune_due:
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
                        if not _add(self._pending, key, values):
                            self.dropped_requests += values[0]
                    for key, values in client_batch.items():
                        if not _add(self._pending_clients, key, values):
                            self.dropped_client_requests += values[0]
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

    @staticmethod
    def _query_clients(connection, start: datetime, end: datetime, model: str) -> list:
        sums = ",".join(f"SUM({f})" for f in _FIELDS)
        try:
            return connection.execute(
                f"SELECT key_kind, key_id, client_ip, {sums} "
                "FROM client_usage_hourly "
                "WHERE timestamp_hour >= ? AND timestamp_hour < ?"
                + (" AND model_id = ?" if model else "")
                + " GROUP BY key_kind, key_id, client_ip",
                (start.timestamp(), end.timestamp(), *((model,) if model else ())),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            # A file last written by a build without per-client history.
            if "no such table" in str(exc):
                return []
            raise

    def query(
        self,
        period: str = "today",
        model: str = "",
        *,
        include_details: bool = False,
        now: float | None = None,
    ) -> dict:
        now = time.time() if now is None else now
        start, end = _bounds(period, now)
        rows: list = []
        client_rows: list = []
        # Read-only connection: polling cannot silently recreate a deleted database.
        # Disabled history answers with an empty, explicitly flagged payload
        # without opening the database at all.
        connection = None
        try:
            if self.enabled:
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
                client_rows = self._query_clients(connection, start, end, model)
        finally:
            if connection is not None:
                connection.close()
        totals = [0] * len(_FIELDS)
        models: dict[str, list] = {}
        # 24 cells per calendar day, repeated DST hours combine; missing hours are zero.
        heatmap = {}
        day_cursor = start
        while day_cursor < end:
            heatmap[day_cursor.date().isoformat()] = [0] * 24
            day_cursor += timedelta(days=1)
        days = {day: [0] * len(_FIELDS) for day in heatmap} if include_details else {}
        hourly: dict[int, list] = {}
        # One query, three views: each key+IP pair, per key, and per IP.
        pairs: dict[tuple[str, str, str], list] = {}
        by_key: dict[tuple[str, str], list] = {}
        by_ip: dict[str, list] = {}
        for kind, key_id, ip, *values in client_rows:
            pairs[(kind, key_id, ip)] = values
            for target, group in ((by_key, (kind, key_id)), (by_ip, ip)):
                target[group] = [
                    a + b
                    for a, b in zip(
                        target.get(group, [0] * len(_FIELDS)), values, strict=True
                    )
                ]
        for hour, model_id, *values in rows:
            local = datetime.fromtimestamp(hour)
            day = local.date().isoformat()
            totals = [a + b for a, b in zip(totals, values, strict=True)]
            models.setdefault(model_id, [0] * len(_FIELDS))
            models[model_id] = [
                a + b for a, b in zip(models[model_id], values, strict=True)
            ]
            _requests, prompt_tokens, completion_tokens, *_ = values
            heatmap[day][local.hour] += prompt_tokens + completion_tokens
            if include_details:
                days[day] = [a + b for a, b in zip(days[day], values, strict=True)]
                hourly.setdefault(hour, [0] * len(_FIELDS))
                hourly[hour] = [
                    a + b for a, b in zip(hourly[hour], values, strict=True)
                ]
        result = {
            "range": period,
            "start": start.astimezone().isoformat(),
            "end": end.astimezone().isoformat(),
            "timezone": "server local time",
            "retention_days": RETENTION_DAYS,
            "flush_seconds": FLUSH_SECONDS,
            "enabled": self.enabled,
            "available": self.available,
            "dropped_requests": self.dropped_requests,
            "by_client": self.by_client,
            "dropped_client_requests": self.dropped_client_requests,
            "totals": _summary(totals),
            "models": sorted(
                [{"model_id": key, **_summary(value)} for key, value in models.items()],
                key=lambda item: item["total_tokens"],
                reverse=True,
            ),
            "heatmap": [
                {"date": key, "tokens": value} for key, value in heatmap.items()
            ],
            # Rows recorded while per-client tracking was on stay visible after
            # it is turned off; ``by_client`` says whether new ones accrue.
            "clients": _ranked(
                {"key_kind": kind, "key_id": key_id, "client_ip": ip, **_summary(v)}
                for (kind, key_id, ip), v in pairs.items()
            ),
            "clients_by_key": _ranked(
                {"key_kind": kind, "key_id": key_id, **_summary(v)}
                for (kind, key_id), v in by_key.items()
            ),
            "clients_by_ip": _ranked(
                {"client_ip": ip, **_summary(v)} for ip, v in by_ip.items()
            ),
        }
        if include_details:
            result["daily"] = [
                {"date": key, **_summary(value)} for key, value in days.items()
            ]
            result["hourly"] = [
                {"timestamp_hour": key, **_summary(value)}
                for key, value in sorted(hourly.items())
            ]
        return result
