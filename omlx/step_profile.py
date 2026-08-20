# SPDX-License-Identifier: Apache-2.0
"""Opt-in wall-time profile of the serving hot path (``OMLX_STEP_PROFILE=1``).

Answers one question: of the wall time a decode step spends outside the
model forward, where does it go — scheduling, cache bookkeeping, response
post-processing, or SSE emission? Serving-layer overhead is invisible to
model-level profilers, and past regressions shipped because nobody could
cheaply see this split.

Buckets are accumulated process-wide and logged as one summary line every
``_LOG_EVERY`` scheduler steps, then reset, so each logged window stands on
its own. When the env var is unset every helper is a no-op returning a
constant — the hot path pays one attribute load and one falsy check.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("OMLX_STEP_PROFILE", "").strip().lower() in {
    "1",
    "true",
    "on",
    "yes",
}


def _log_every_from_env() -> int:
    # Parse defensively: a bad value must not break server startup —
    # least of all when the profiler is disabled anyway.
    try:
        return max(1, int(os.environ.get("OMLX_STEP_PROFILE_EVERY", "500")))
    except (TypeError, ValueError):
        return 500


_LOG_EVERY = _log_every_from_env()

_lock = threading.Lock()
# name -> [total_seconds, samples, max_seconds]
_buckets: dict[str, list[float]] = {}
# Buckets are process-wide, so the window is counted in maybe_log CALLS,
# not in any caller's step counter — with several engines each passing its
# own counter, comparing against a shared last-logged value misfires.
_calls_since_log = 0


def enabled() -> bool:
    return _ENABLED


def tick() -> float:
    """Start a measurement; pair with :func:`add_since`."""
    if not _ENABLED:
        return 0.0
    return time.perf_counter()


def add(name: str, seconds: float) -> None:
    if not _ENABLED:
        return
    with _lock:
        bucket = _buckets.get(name)
        if bucket is None:
            _buckets[name] = [seconds, 1, seconds]
        else:
            bucket[0] += seconds
            bucket[1] += 1
            if seconds > bucket[2]:
                bucket[2] = seconds


def add_since(name: str, t0: float) -> None:
    if not _ENABLED:
        return
    add(name, time.perf_counter() - t0)


def snapshot(reset: bool = False) -> dict[str, dict[str, float]]:
    if not _ENABLED:
        return {}
    with _lock:
        out = {
            name: {
                "seconds": round(total, 4),
                "samples": count,
                "max": round(peak, 4),
            }
            for name, (total, count, peak) in _buckets.items()
        }
        if reset:
            _buckets.clear()
    return out


def maybe_log(step_counter: int) -> None:
    """Log one summary line every ``_LOG_EVERY`` calls, then reset.

    ``step_counter`` only labels the line (the calling engine's step).
    """
    global _calls_since_log
    if not _ENABLED:
        return
    with _lock:
        _calls_since_log += 1
        if _calls_since_log < _LOG_EVERY:
            return
        _calls_since_log = 0
    snap = snapshot(reset=True)
    if not snap:
        return
    # No percentages: buckets overlap (sched.* nests inside step.schedule,
    # admission prefill inside both), so shares of the bucket sum would
    # misrepresent the wall-clock split.
    parts = ", ".join(
        f"{name} {entry['seconds']:.3f}s "
        f"(n={entry['samples']}, max={entry['max']:.3f}s)"
        for name, entry in sorted(
            snap.items(), key=lambda item: -item[1]["seconds"]
        )
    )
    logger.info("step-profile @%d: %s", step_counter, parts)
