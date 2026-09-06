# SPDX-License-Identifier: Apache-2.0
"""Opt-in per-layer / per-projection memory tracing for MoE expert streaming.

Motivation (Fase J prefill-memory investigation)
------------------------------------------------
The dominant prefill memory consumer on oQ4e models is the *transient* Metal
working set, not the mmap'd N-gram table and not the KV cache. Measuring it
required shelling out to ``vmmap`` and eyeballing Activity Monitor, which is
far too coarse to attribute cost to a layer or a projection.

This module records, at each interesting point inside the streaming switch
layers, the four numbers that actually matter on Apple Silicon UMA:

``active``
    ``mx.get_active_memory()`` — bytes held by live MLX arrays. This is the
    lazy-graph retention term: an assembled expert bank stays *active* for as
    long as the graph that consumes it is unevaluated.
``cache``
    ``mx.get_cache_memory()`` — the reusable Metal buffer pool. Grows when
    temporaries are released back to MLX rather than to the driver.
``peak``
    ``mx.get_peak_memory()`` — allocator high-water mark.
``footprint``
    ``get_phys_footprint()`` — the mach per-process ledger, which *includes*
    IOAccelerator-backed (Metal) allocations. This is the number that
    previously only ``vmmap -summary`` could produce, and it is what jetsam
    compares against.

Usage
-----
Enabled by environment only, so production pays nothing::

    OMLX_EXPERT_STREAMING_MEMTRACE=/path/to/trace.jsonl   # append JSONL rows
    OMLX_EXPERT_STREAMING_MEMTRACE=1                      # in-memory + stderr summary
    OMLX_EXPERT_STREAMING_MEMTRACE_EVERY=1                # sample every row (default)

When the variable is unset, ``memtrace`` is a null object whose methods are
no-ops, so call sites in the hot path cost one attribute lookup plus a call.

The tracer is sampled at import time and cannot be re-armed at runtime; this
keeps the hot path branch-free (``memtrace.enabled`` is a class attribute).

Consumers
---------
``bench/bench_expert_streaming.py`` reads ``memtrace.summary()`` to report the
prefill peak footprint per phase, which is acceptance criterion #3 of the
Fase J prefill-memory plan.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_TRACE_PATH = os.environ.get("OMLX_EXPERT_STREAMING_MEMTRACE", "") or None
_SAMPLE_EVERY = max(1, int(os.environ.get("OMLX_EXPERT_STREAMING_MEMTRACE_EVERY", "1")))

# Fase L1: numeric fields whose per-event aggregates land in summary() so a
# bench run can report, e.g., the mean/max positions and bank bytes per
# ctx.ensure event without post-processing the JSONL. Keep this set small and
# fixed: aggregation is per-record overhead on the traced path only.
_TRACKED_NUMERIC = frozenset(
    {
        "positions",
        "uniq",
        "miss",
        "hits",
        "misses",
        "bank_bytes",
        "ctx_bank_bytes",
        "ctx_inflight_bytes",
        "ctx_prefetch_count",
        "inflight",
        "inflight_bytes",
        "hot_positions",
        "cold_positions",
        "hot_bank_bytes",
        "cold_bank_bytes",
        "experts",
        "bytes",
        "n_proj",
        "n_loaded",
        "segments",
        "call_s",
    }
)


def _phys_footprint() -> int:
    """Best-effort mach phys_footprint (includes IOAccelerator/Metal)."""
    try:
        from omlx.utils.proc_memory import get_phys_footprint

        return int(get_phys_footprint())
    except Exception:
        return 0


def _rss_bytes() -> int:
    """Current-process resident size (bytes on Darwin, KB on Linux)."""
    try:
        import resource

        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kilobytes.
        return int(v) if os.uname().sysname == "Darwin" else int(v) * 1024
    except Exception:
        return 0


def _mlx_snapshot() -> tuple[int, int, int]:
    """Return (active, cache, peak) MLX memory counters, tolerating stubs."""
    try:
        import mlx.core as mx

        active = getattr(mx, "get_active_memory", None)
        cache = getattr(mx, "get_cache_memory", None)
        peak = getattr(mx, "get_peak_memory", None)
        return (
            int(active()) if active is not None else 0,
            int(cache()) if cache is not None else 0,
            int(peak()) if peak is not None else 0,
        )
    except Exception:
        return (0, 0, 0)


class _NullTracer:
    """No-op tracer used when tracing is disabled (the default)."""

    enabled = False
    path = None

    def record(self, event: str, **fields: Any) -> None:
        return None

    def set_context(self, **ctx: Any) -> None:
        return None

    def clear_context(self) -> None:
        return None

    @contextmanager
    def scope(self, event: str, **fields: Any) -> Iterator[None]:
        yield None

    def summary(self) -> dict[str, Any]:
        return {"enabled": False}

    def reset(self) -> None:
        return None

    def flush(self) -> None:
        return None


class MemTracer:
    """Records memory samples as JSONL rows and tracks running peaks.

    Thread-safe: the streaming IO pool drives loads from worker threads, so
    ``record`` serializes writes. Sampling itself is cheap (three MLX counter
    reads plus one libproc call), but it is *not* free — that is why the whole
    module is gated behind an environment variable.
    """

    enabled = True

    def __init__(self, path: str | None = None, sample_every: int = 1) -> None:
        self.path = path
        self.sample_every = max(1, int(sample_every))
        self._lock = threading.Lock()
        # Fase M6: ambient context (phase, request_id, engine_id) attached
        # to every row, and a per-(layer, proj) event sequence counter so
        # ordering can be reconstructed without timestamp resolution.
        self._context: dict[str, Any] = {}
        self._event_seq: dict[tuple, int] = {}
        self._file = None
        self._seq = 0
        self._rows: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()
        self._peaks: dict[str, int] = {}
        # Fase L1: {event: {field: {sum, n, max}}} over _TRACKED_NUMERIC fields.
        self._agg: dict[str, dict[str, dict[str, float]]] = {}
        # Keep at most this many rows in memory when no path is given, so an
        # unattended long run cannot grow unbounded.
        self._max_rows = 200_000

    # -- sampling ---------------------------------------------------------

    def _sample(self) -> dict[str, int]:
        active, cache, peak = _mlx_snapshot()
        return {
            "active": active,
            "cache": cache,
            "peak": peak,
            "footprint": _phys_footprint(),
            "rss": _rss_bytes(),
        }

    def _open(self) -> None:
        if self.path and self._file is None:
            self._file = open(self.path, "a", buffering=1)  # noqa: SIM115

    # -- public API -------------------------------------------------------

    def record(self, event: str, **fields: Any) -> None:
        """Append one sample row for ``event`` with arbitrary extra fields."""
        with self._lock:
            self._seq += 1
            if self._seq % self.sample_every != 0:
                return
            row: dict[str, Any] = {
                "seq": self._seq,
                "t": round(time.perf_counter() - self._t0, 6),
                "event": event,
            }
            if self._context:
                row.update(self._context)
            row.update(fields)
            # Fase M6: monotone per-(layer, proj) event sequence.
            if "layer" in fields and "proj" in fields:
                k = (fields.get("layer"), fields.get("proj"))
                self._event_seq[k] = self._event_seq.get(k, 0) + 1
                row["event_seq"] = self._event_seq[k]
            row.update(self._sample())
            for key, val in fields.items():
                if key in _TRACKED_NUMERIC and isinstance(val, (int, float)) and not isinstance(val, bool):
                    acc = self._agg.setdefault(event, {}).setdefault(
                        key, {"sum": 0.0, "n": 0, "max": float("-inf")}
                    )
                    acc["sum"] += float(val)
                    acc["n"] += 1
                    acc["max"] = max(acc["max"], float(val))
            for key in ("active", "cache", "peak", "footprint", "rss", "bank_bytes"):
                val = row.get(key)
                if isinstance(val, int) and val > self._peaks.get(key, 0):
                    self._peaks[key] = val
            if self._file is None:
                self._open()
            if self._file is not None:
                self._file.write(json.dumps(row, default=str) + "\n")
            elif len(self._rows) < self._max_rows:
                self._rows.append(row)

    def set_context(self, **ctx: Any) -> None:
        """Fase M6: ambient fields (phase, request_id...) appended to every
        subsequent row until clear_context()."""
        with self._lock:
            self._context.update(ctx)

    def clear_context(self) -> None:
        with self._lock:
            self._context.clear()

    @contextmanager
    def scope(self, event: str, **fields: Any) -> Iterator[None]:
        """Record ``<event>.enter`` and ``<event>.exit`` around a block."""
        self.record(f"{event}.enter", **fields)
        try:
            yield None
        finally:
            self.record(f"{event}.exit", **fields)

    def peaks(self) -> dict[str, int]:
        """Return the running per-metric peak observed so far."""
        with self._lock:
            return dict(self._peaks)

    def rows(self) -> list[dict[str, Any]]:
        """Return in-memory rows (empty when writing to a path)."""
        with self._lock:
            return list(self._rows)

    def summary(self) -> dict[str, Any]:
        """Return a compact peak report, GiB-normalized for readability."""
        peaks = self.peaks()
        out: dict[str, Any] = {
            "enabled": True,
            "path": self.path,
            "samples": self._seq,
            "peaks_bytes": dict(peaks),
        }
        for key, val in peaks.items():
            out[f"peak_{key}_gib"] = round(val / 1024**3, 3)
        if self._agg:
            agg_out: dict[str, dict[str, dict[str, float]]] = {}
            for event, fields_ in self._agg.items():
                agg_out[event] = {}
                for field, acc in fields_.items():
                    if acc["n"] <= 0:
                        continue
                    agg_out[event][field] = {
                        "mean": round(acc["sum"] / acc["n"], 1),
                        "max": round(acc["max"], 1),
                    }
            out["event_aggregates"] = agg_out
        return out

    def reset(self) -> None:
        """Drop accumulated peaks and in-memory rows (keeps the file)."""
        with self._lock:
            self._peaks.clear()
            self._rows.clear()
            self._agg.clear()
            self._seq = 0
            self._t0 = time.perf_counter()

    def flush(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.flush()


# ---------------------------------------------------------------------------
# Module-level singleton — armed at import, immutable thereafter.
# ---------------------------------------------------------------------------


def _build_tracer() -> MemTracer | _NullTracer:
    if not _TRACE_PATH:
        return _NullTracer()
    if _TRACE_PATH == "1":
        tracer = MemTracer(path=None, sample_every=_SAMPLE_EVERY)
        logger.warning(
            "Expert streaming memtrace armed (in-memory). "
            "Set OMLX_EXPERT_STREAMING_MEMTRACE=<path.jsonl> to persist rows."
        )
        return tracer
    try:
        return MemTracer(path=_TRACE_PATH, sample_every=_SAMPLE_EVERY)
    except OSError as e:
        logger.warning("Expert streaming memtrace disabled, cannot open %s: %s", _TRACE_PATH, e)
        return _NullTracer()


memtrace: MemTracer | _NullTracer = _build_tracer()


__all__ = ["memtrace", "MemTracer"]
