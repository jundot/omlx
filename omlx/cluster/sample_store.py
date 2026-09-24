# SPDX-License-Identifier: Apache-2.0
"""Bounded rolling-window sample store for time-series metrics."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1

_DEFAULT_RETENTION_DAYS = 30
_DEFAULT_SAMPLE_INTERVAL = 60


class SampleStore:
    """Append-only rolling-window store for timestamped metric samples.

    Thread-safe via ``threading.RLock``.  Samples are buffered in
    memory and persisted to disk on explicit ``flush()`` or
    automatically when the buffer exceeds 256 entries.  Old samples
    are evicted on every ``record()`` call so memory stays bounded.
    """

    _FLUSH_THRESHOLD = 256

    def __init__(
        self,
        path: Path,
        *,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
        sample_interval: int = _DEFAULT_SAMPLE_INTERVAL,
    ) -> None:
        self._path = path
        self._retention_days = retention_days
        self._sample_interval = sample_interval
        self._lock = threading.RLock()
        self._samples: list[dict[str, Any]] = []
        self._dirty = False
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            self._samples = []
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if data.get("schema_version") != _SCHEMA_VERSION:
                self._samples = []
                return
            raw = data.get("samples")
            if not isinstance(raw, list):
                self._samples = []
                return
            self._samples = raw
        except (OSError, json.JSONDecodeError, TypeError):
            self._samples = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "samples": self._samples,
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".sample-store.",
            suffix=".tmp",
            dir=self._path.parent,
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict(self, now: float) -> None:
        cutoff = now - self._retention_days * 86400
        if self._samples and self._samples[0]["ts"] < cutoff:
            lo, hi = 0, len(self._samples)
            while lo < hi:
                mid = (lo + hi) // 2
                if self._samples[mid]["ts"] < cutoff:
                    lo = mid + 1
                else:
                    hi = mid
            self._samples = self._samples[lo:]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        ts: float | None = None,
        request_count: int = 0,
        error_count: int = 0,
        ttft_sum: float = 0.0,
        tpot_sum: float = 0.0,
    ) -> None:
        """Append a single sample.  Buffers in memory; auto-flushes to disk."""
        now = ts if ts is not None else time.time()
        sample = {
            "ts": now,
            "request_count": request_count,
            "error_count": error_count,
            "ttft_sum": ttft_sum,
            "tpot_sum": tpot_sum,
        }
        with self._lock:
            self._samples.append(sample)
            self._evict(now)
            self._dirty = True
            if len(self._samples) >= self._FLUSH_THRESHOLD:
                self._save()
                self._dirty = False

    def flush(self) -> None:
        """Force-persist buffered samples to disk."""
        with self._lock:
            if self._dirty:
                self._save()
                self._dirty = False

    def query(
        self,
        start: float,
        end: float,
    ) -> list[dict[str, Any]]:
        """Return samples where ``start <= ts <= end``."""
        with self._lock:
            return [s for s in self._samples if start <= s["ts"] <= end]

    @staticmethod
    def downsample(
        samples: list[dict[str, Any]],
        target_count: int,
    ) -> list[dict[str, Any]]:
        """Downsample *samples* to at most *target_count* points.

        Splits the time range into ``target_count`` equal buckets and
        sums each bucket's values.
        """
        if not samples or target_count <= 0:
            return []
        if len(samples) <= target_count:
            return samples

        ts_min = samples[0]["ts"]
        ts_max = samples[-1]["ts"]
        span = ts_max - ts_min or 1.0
        bucket_size = span / target_count

        result: list[dict[str, Any]] = []
        bucket: list[dict[str, Any]] = []
        current_bucket_idx: int = -1

        def _flush() -> None:
            if not bucket:
                return
            result.append(
                {
                    "ts": bucket[-1]["ts"],
                    "request_count": sum(b["request_count"] for b in bucket),
                    "error_count": sum(b["error_count"] for b in bucket),
                    "ttft_sum": round(sum(b["ttft_sum"] for b in bucket), 6),
                    "tpot_sum": round(sum(b["tpot_sum"] for b in bucket), 6),
                }
            )

        for sample in samples:
            pos = (sample["ts"] - ts_min) / bucket_size
            idx = min(int(pos), target_count - 1)
            if idx != current_bucket_idx and bucket:
                _flush()
                bucket = []
                current_bucket_idx = idx
            elif current_bucket_idx == -1:
                current_bucket_idx = idx
            bucket.append(sample)
        _flush()

        return result

    def sample_count(self) -> int:
        """Return the current in-memory sample count (for testing)."""
        with self._lock:
            return len(self._samples)
