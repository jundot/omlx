# SPDX-License-Identifier: Apache-2.0
"""
Lightweight prefill progress tracker for dashboard display.

Updated by BatchGenerator's prompt_progress_callback (CPU counters only,
zero GPU overhead). Read by admin stats API to show per-request PP progress
in the Active Models card.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional


class PrefillProgressTracker:
    """Thread-safe tracker for per-request prefill progress.

    Each entry stores (processed_tokens, total_tokens, model_id) for a
    request that is currently in its prefill phase.  Entries are auto-removed
    when processed >= total (prefill complete).

    Performance: ~50ns lock acquire/release + O(1) dict write per update.
    Called once per prefill chunk (default 2048 tokens).
    """

    def __init__(self) -> None:
        # request_id -> {"processed": int, "total": int, "model_id": str}
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def update(self, request_id: str, processed: int, total: int, model_id: str) -> None:
        """Update prefill progress for a request.

        Auto-removes the entry when processed >= total (prefill complete).
        """
        with self._lock:
            if processed >= total:
                self._progress.pop(request_id, None)
            else:
                self._progress[request_id] = {
                    "processed": processed,
                    "total": total,
                    "model_id": model_id,
                }

    def remove(self, request_id: str) -> None:
        """Explicitly remove a request (e.g. on abort or finish)."""
        with self._lock:
            self._progress.pop(request_id, None)

    def get_model_progress(self, model_id: str) -> List[Dict[str, Any]]:
        """Return list of prefilling requests for a given model."""
        with self._lock:
            return [
                {
                    "request_id": rid,
                    "processed": entry["processed"],
                    "total": entry["total"],
                }
                for rid, entry in self._progress.items()
                if entry["model_id"] == model_id
            ]

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
