# SPDX-License-Identifier: Apache-2.0
"""Process-global registry of foreground requests that have arrived.

``EnginePool`` runs several engines in one process against one accelerator, so
"is this engine idle" and "is this process idle" are different questions.
``DecodeActivityRegistry`` answers the first for decode and
``PrefillProgressTracker`` for prefill, but both are progress signals: they
exist only once work is already running. Between the transport accepting a
request and that request's first forward there is a span in which every
progress signal in the process reads idle.

Background canonical-state recovery cares about exactly that span, because it
starts work it cannot interrupt. So this registry records the *arrival*
directly rather than inferring it afterwards:

* ``note_arrival`` runs on the asyncio event loop, before the request is handed
  to the engine's executor;
* ``note_admission`` re-stamps it once the engine owns it, which restarts the
  expiry clock against the request's real lifetime rather than against its
  queue wait;
* ``note_departure`` drops it when the request finishes, aborts or fails to be
  admitted at all.

The entry deliberately outlives admission. On the admitting engine that is
redundant — its own ``waiting``/``running`` lists already say so — but on every
*other* engine in the pool it is the only thing that does, from admission until
that engine's first prefill chunk reaches the progress tracker.

Entries carry a monotonic timestamp and expire, so a request that arrives and
then vanishes — cancelled in flight, rejected before ``add_request``, lost to an
exception — cannot hold recovery off for the life of the process.

Mirrors the shape of ``decode_activity.DecodeActivityRegistry``: one lock, CPU
counters only, never held across a model call.
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
