# SPDX-License-Identifier: Apache-2.0
"""
Per-scheduler EWMA of bytes-per-prefill-token.

Used by the adaptive prefill throttle in Scheduler: when current memory
enters the caution zone (>= hard_cap * safe_zone_ratio), the next chunk
is sized so its predicted transient stays under the remaining headroom.

Owned by each Scheduler instance (one EWMA per loaded model), distinct
from the global PrefillProgressTracker which feeds the admin dashboard.
"""

from __future__ import annotations


class PrefillTransientTracker:
    """EWMA estimator of MLX prefill chunk transient bytes per token.

    Updated post-chunk from `phys_footprint()` deltas. The first chunk
    has no measurement yet — callers fall back to a static estimate
    (MemoryMonitor.estimate_prefill_peak_bytes) until samples > 0.
    """

    _EWMA_ALPHA = 0.3  # weight on the most recent chunk

    def __init__(self, model_id: str = "") -> None:
        self._model_id = model_id
        self._ewma_per_token: float = 0.0
        self._samples: int = 0
        # Last observed delta for debug log inspection.
        self._last_delta_bytes: int = 0
        self._last_n_tokens: int = 0

    def update(self, n_tokens: int, transient_bytes: int) -> None:
        """Record one chunk observation.

        Negative deltas (MLX cache pool reclaim larger than this chunk's
        allocation) are skipped — they would bias the EWMA toward zero
        and underestimate the next chunk's footprint.
        """
        if n_tokens <= 0:
            return
        if transient_bytes <= 0:
            return

        per_token = transient_bytes / n_tokens
        if self._samples == 0:
            self._ewma_per_token = per_token
        else:
            self._ewma_per_token = (
                self._EWMA_ALPHA * per_token
                + (1.0 - self._EWMA_ALPHA) * self._ewma_per_token
            )
        self._samples += 1
        self._last_delta_bytes = transient_bytes
        self._last_n_tokens = n_tokens

    def predict(self, n_tokens: int, *, safety_factor: float = 1.2) -> int:
        """Predicted transient bytes for a chunk of `n_tokens`.

        Returns 0 when no samples have been observed yet — caller must
        fall back to a static estimator in that case.
        """
        if self._samples == 0 or n_tokens <= 0:
            return 0
        return int(self._ewma_per_token * n_tokens * safety_factor)

    @property
    def bytes_per_token(self) -> float:
        """Current EWMA value (bytes per prefill token). 0.0 if no samples."""
        return self._ewma_per_token

    @property
    def samples(self) -> int:
        """Number of chunks recorded since last reset."""
        return self._samples

    @property
    def last_delta_bytes(self) -> int:
        """Bytes added by the most recently measured chunk."""
        return self._last_delta_bytes

    @property
    def last_n_tokens(self) -> int:
        """Token count of the most recently measured chunk."""
        return self._last_n_tokens

    def reset(self) -> None:
        """Drop all observations (e.g. on model reload or after a long idle)."""
        self._ewma_per_token = 0.0
        self._samples = 0
        self._last_delta_bytes = 0
        self._last_n_tokens = 0
