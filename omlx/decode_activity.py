# SPDX-License-Identifier: Apache-2.0
"""
Process-global decode-activity registry for cross-engine prefill fairness.

Each Scheduler publishes its running-decode count once per step(); a
prefilling engine asks ``others_decoding()`` to decide whether to cap its
chunk size so the other engine's decode kernels can interleave on the
shared GPU. Entries carry a monotonic timestamp and expire after a short
TTL so a wedged or torn-down engine never throttles the rest of the pool.

Mirrors the shape of ``prefill_progress.PrefillProgressTracker``:
~50ns lock acquire/release per call, CPU counters only.
"""

from __future__ import annotations

import threading
import time


class DecodeActivityRegistry:
    """Thread-safe registry of engines currently running decode steps.

    Also carries the process-global prefill hold deadline: when several
    engines prefill against the same victim decode, per-engine holds with
    independent phases never leave the GPU quiet at the same time, so the
    hold window must be shared. Deadlines are ``time.perf_counter()``
    values (the scheduler's hold clock) and self-expire, so there is
    nothing to clean up on engine teardown.
    """

    def __init__(self) -> None:
        self._active: dict[str, tuple[float, int]] = {}
        self._hold_until = 0.0
        self._lock = threading.Lock()

    def extend_hold(self, until_ts: float) -> None:
        """Extend the shared prefill hold deadline (monotone max)."""
        with self._lock:
            if until_ts > self._hold_until:
                self._hold_until = until_ts

    def hold_until(self) -> float:
        """The shared prefill hold deadline (0.0 when never set)."""
        with self._lock:
            return self._hold_until

    def publish(self, key: str, running_count: int) -> None:
        """Record *key*'s running-decode count (0 removes the entry)."""
        with self._lock:
            if running_count > 0:
                self._active[key] = (time.monotonic(), running_count)
            else:
                self._active.pop(key, None)

    def remove(self, key: str) -> None:
        """Drop *key* (engine shutdown/reset)."""
        with self._lock:
            self._active.pop(key, None)

    def others_decoding(self, key: str, ttl_s: float = 2.5) -> bool:
        """True when another engine published a live decode within *ttl_s*."""
        now = time.monotonic()
        with self._lock:
            return any(
                other != key and (now - ts) < ttl_s
                for other, (ts, _count) in self._active.items()
            )

    def clear(self) -> None:
        with self._lock:
            self._active.clear()
            self._hold_until = 0.0


_registry: DecodeActivityRegistry | None = None
_registry_lock = threading.Lock()


def get_decode_activity() -> DecodeActivityRegistry:
    """Get or create the global DecodeActivityRegistry singleton."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = DecodeActivityRegistry()
    return _registry
