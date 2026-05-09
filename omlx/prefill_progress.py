# SPDX-License-Identifier: Apache-2.0
"""
Lightweight prefill progress tracker for dashboard display.

Updated by BatchGenerator's prompt_progress_callback (CPU counters only,
zero GPU overhead). Read by admin stats API to show per-request PP progress
in the Active Models card.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


class PrefillProgressTracker:
    """Thread-safe tracker for per-request prefill progress.

    Each entry stores (processed_tokens, total_tokens, model_id, timing) for a
    request that is currently in its prefill phase.  Entries are auto-removed
    when processed >= total (prefill complete).

    Performance: ~50ns lock acquire/release + O(1) dict write per update.
    Called once per prefill chunk (default 2048 tokens).
    """

    def __init__(self) -> None:
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def update(
        self,
        request_id: str,
        processed: int,
        total: int,
        model_id: str,
        phase: str = "prefill",
        detail: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update prefill/spec-prefill progress for a request.

        Auto-removes the entry when processed >= total (prefill complete).
        Tracks phase timing for speed/ETA calculation.
        """
        now = time.monotonic()
        with self._lock:
            if processed >= total:
                self._progress.pop(request_id, None)
            else:
                prev = self._progress.get(request_id)
                phase_changed = prev is not None and prev.get("phase") != phase
                start_time = now if prev is None or phase_changed else prev["start_time"]

                if prev is not None and not phase_changed:
                    dt = now - prev["last_time"]
                    dtok = processed - prev["processed"]
                    if dt > 0 and dtok > 0:
                        speed = dtok / dt
                    else:
                        speed = prev.get("speed", 0.0)
                else:
                    speed = 0.0

                entry = dict(extra or {})
                entry.update(
                    {
                        "processed": processed,
                        "total": total,
                        "model_id": model_id,
                        "start_time": start_time,
                        "last_time": now,
                        "speed": speed,
                        "phase": phase,
                        "detail": detail,
                    }
                )
                self._progress[request_id] = entry

    def remove(self, request_id: str) -> None:
        """Explicitly remove a request (e.g. on abort or finish)."""
        with self._lock:
            self._progress.pop(request_id, None)

    def get_model_progress(self, model_id: str) -> List[Dict[str, Any]]:
        """Return list of prefilling requests for a given model."""
        with self._lock:
            results = []
            for rid, entry in self._progress.items():
                if entry["model_id"] != model_id:
                    continue
                elapsed = time.monotonic() - entry["start_time"]
                speed = entry.get("speed", 0.0)
                remaining = entry["total"] - entry["processed"]
                eta = remaining / speed if speed > 0 else None
                result = {
                    "request_id": rid,
                    "processed": entry["processed"],
                    "total": entry["total"],
                    "speed": round(speed, 1),
                    "eta": round(eta, 1) if eta is not None else None,
                    "elapsed": round(elapsed, 1),
                    "phase": entry.get("phase", "prefill"),
                    "detail": entry.get("detail"),
                }
                for key in (
                    "scored_tokens",
                    "selected_tokens",
                    "keep_percent",
                    "prompt_tokens",
                    "system_tokens",
                    "conversation_tokens",
                    "cached_tokens",
                ):
                    if key in entry:
                        result[key] = entry[key]
                results.append(result)
            return results

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._progress.clear()


# Module-level singleton, lazily created.
_tracker: Optional[PrefillProgressTracker] = None
_tracker_lock = threading.Lock()


def get_prefill_tracker() -> PrefillProgressTracker:
    """Get or create the global PrefillProgressTracker singleton."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = PrefillProgressTracker()
    return _tracker
