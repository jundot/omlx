# SPDX-License-Identifier: Apache-2.0
"""Process-global registry of foreground requests that have arrived.

The decode registry and prefill tracker only see work once it is running.
Recovery starts work it cannot interrupt, so it also needs to know about a
request between the transport accepting it and its first forward.

The entry is kept past admission because other engines in the pool cannot see
this engine's queues. Entries expire so a request that vanishes without a
departure cannot hold recovery off forever.
"""

from __future__ import annotations

import threading
import time

DEFAULT_TTL_S = 30.0


class ForegroundArrivalRegistry:
    """Thread-safe registry of arrived foreground requests, keyed by id.

    Written from the event loop and read from every engine's executor, which
    is why it owns a lock rather than relying on the GIL: expiry iterates the
    mapping, and a dict that grows mid-iteration raises.
    """

    def __init__(self) -> None:
        self._arrived: dict[str, float] = {}
        self._lock = threading.Lock()

    def note_arrival(self, request_id: str) -> None:
        with self._lock:
            self._arrived[request_id] = time.monotonic()

    def note_admission(self, request_id: str) -> None:
        """Restart the expiry clock: the engine owns this request now."""
        with self._lock:
            if request_id in self._arrived:
                self._arrived[request_id] = time.monotonic()

    def note_departure(self, request_id: str) -> None:
        with self._lock:
            self._arrived.pop(request_id, None)

    def count(self, ttl_s: float = DEFAULT_TTL_S) -> int:
        """How many arrivals are live, expiring anything older than *ttl_s*."""
        return len(self.expire(ttl_s)[0])

    def expire(self, ttl_s: float = DEFAULT_TTL_S) -> tuple[list[str], list[str]]:
        """Drop entries older than *ttl_s*; return (live ids, expired ids)."""
        deadline = time.monotonic() - ttl_s
        with self._lock:
            if not self._arrived:
                return [], []
            stale = [rid for rid, at in self._arrived.items() if at < deadline]
            for rid in stale:
                self._arrived.pop(rid, None)
            return list(self._arrived), stale

    def clear(self) -> None:
        with self._lock:
            self._arrived.clear()


_registry: ForegroundArrivalRegistry | None = None
_registry_lock = threading.Lock()


def get_foreground_arrivals() -> ForegroundArrivalRegistry:
    """Get or create the global ForegroundArrivalRegistry singleton."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ForegroundArrivalRegistry()
    return _registry
