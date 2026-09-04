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

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class _TransientHistory:
    ewma_per_token: float = 0.0
    samples: int = 0
    last_delta_bytes: int = 0
    last_n_tokens: int = 0
    observed_max_bytes: int = 0


class PrefillTransientTracker:
    """EWMA estimator of MLX prefill chunk transient bytes per token.

    Updated post-chunk from `phys_footprint()` deltas. The first chunk
    has no measurement yet — callers fall back to a static estimate
    (MemoryMonitor.estimate_prefill_peak_bytes) until samples > 0.
    """

    _EWMA_ALPHA = 0.3  # weight on the most recent chunk
    # Candidates above this are rejected from the running max: a one-off
    # Metal/pool spike this large is not a repeatable chunk transient, and
    # charging it at admission would refuse prompts that always fit. A
    # genuinely recurring giant transient still reaches the guard through
    # the last-delta/EWMA terms of _predicted_chunk_transient.
    _OBSERVED_MAX_CLAMP_BYTES = 4 * 1024**3
    # observed_max_bytes only ever INCREASED before idle-reset existed: one
    # pathological-but-legitimate (under the clamp) floor-chunk spike early
    # in a long-running (hours/days, continuously-loaded model) session
    # permanently raised the admission floor for the rest of that session,
    # even if the spike never recurred (§B4).
    #
    # The first fix tried here decayed the max on every subsequent
    # floor-size sample. Review caught the flaw: floor samples only occur
    # once the adaptive throttle is already down at its minimum chunk size
    # -- the pressure regime -- so that "clock" only ran exactly when the
    # floor was load-bearing, eroding the protection fastest right when it
    # mattered most. Forgetting now hangs off wall-clock idle time instead
    # (``_IDLE_RESET_SECONDS``, checked in ``update``): unrelated to
    # pressure, so it cannot erode a floor that's actively earning its
    # keep, and it still clears a stale spike once the session goes quiet
    # long enough that the spike is no longer representative.
    _IDLE_RESET_SECONDS = 30 * 60
    # A sample whose per_token exceeds the current EWMA by more than this
    # ratio is treated as measurement noise (a tail/residual prefill chunk,
    # not a genuine cost-per-token regime change) and excluded from the EWMA
    # blend. Chosen from a real incident (2026-07-29, Qwen3.6-35B-A3B):
    # baseline samples ranged ~525-1867 KB/token (largest legitimate
    # fluctuation ~1.7x the running EWMA) before a single n=185 tail chunk
    # measured 10497.1 KB/token — a ~13.6x jump off an EWMA of 773.3 KB/token
    # — and pushed the EWMA to 3690.5 KB/token in one update, poisoning every
    # later admission check for the rest of the process lifetime. 8x sits
    # above the largest observed legitimate fluctuation and below the
    # observed outlier.
    _EWMA_OUTLIER_RATIO = 8.0

    def __init__(self, model_id: str = "") -> None:
        self._model_id = model_id
        self._dense_history = _TransientHistory()
        self._gathered_history = _TransientHistory()
        # Net process footprint released by negative post-chunk deltas. MLX may
        # need to allocate that pool again on the next chunk, so the scheduler
        # prices it once until a positive measurement confirms reallocation.
        self._recent_reclaim_bytes: int = 0
        # Monotonic timestamp of the last recorded sample; None before the
        # first one. ``update`` resets the tracker when it's been longer
        # than ``_IDLE_RESET_SECONDS`` since this, so a stale observed-max
        # from before a long quiet stretch doesn't keep gating admission
        # once the session resumes (§B4/3.4).
        self._last_update_monotonic: float | None = None

    def _history(self, gathered_core: bool) -> _TransientHistory:
        return self._gathered_history if gathered_core is True else self._dense_history

    def record_reclaim(self, reclaimed_bytes: int) -> None:
        """Accumulate footprint released since the last positive sample."""
        if reclaimed_bytes > 0:
            self._recent_reclaim_bytes += int(reclaimed_bytes)

    def clear_reclaim(self) -> None:
        """Drop the charge once any positive measurement confirms realloc.

        Callers invoke this for every positive delta, including samples the
        EWMA gates skip (sub-floor tails, speed-priority partials) — the
        footprint has grown back, so keeping the charge would double count
        against the guard's gates.
        """
        self._recent_reclaim_bytes = 0

    def update(
        self,
        n_tokens: int,
        transient_bytes: int,
        *,
        floor_sample: bool = False,
        gathered_core: bool = False,
    ) -> None:
        """Record one chunk observation.

        Negative deltas (MLX cache pool reclaim larger than this chunk's
        allocation) are skipped — they would bias the EWMA toward zero
        and underestimate the next chunk's footprint.

        ``floor_sample`` marks a chunk at the throttle's floor size. Only
        those feed the running max: admission charges the floor chunk, and
        chunk-transient maxima are NOT size-invariant across models
        (Qwen3.6 measured ~3.0GB at 2048-token chunks vs far less at the
        floor; charging the big-chunk max at admission rejected every
        prompt at a 21GB ceiling). Big-chunk transients stay the throttle's
        domain via the EWMA/last-delta terms. A floor sample only ever
        raises the max within one idle window (see ``_IDLE_RESET_SECONDS``
        and this method's idle check below) — it does not decay it, since
        floor samples occur exactly under the pressure the max exists to
        protect against.

        A sample whose per-token rate exceeds the current EWMA by more than
        ``_EWMA_OUTLIER_RATIO`` is excluded from the EWMA blend (see that
        constant's docstring) — it still counts toward ``samples`` and
        still updates ``last_delta_bytes``/``last_n_tokens`` raw, so a
        genuine regime change remains visible via those fields even while
        the accumulated EWMA is protected from a single noisy reading.
        """
        if n_tokens <= 0:
            return
        if transient_bytes <= 0:
            return

        now = time.monotonic()
        if (
            self._last_update_monotonic is not None
            and now - self._last_update_monotonic > self._IDLE_RESET_SECONDS
        ):
            # A stale observed-max earned under a past pressure spike is no
            # longer representative once the session has gone quiet this
            # long; forget it here (wall-clock, not sample-count driven) so
            # it can't keep gating admission for a session that has since
            # moved on. Unrelated to pressure -- see _IDLE_RESET_SECONDS.
            logger.debug(
                "PrefillTransientTracker(%s): resetting after %.0fs idle "
                "(observed_max dense=%.2fMB gathered=%.2fMB)",
                self._model_id,
                now - self._last_update_monotonic,
                self._dense_history.observed_max_bytes / 1024**2,
                self._gathered_history.observed_max_bytes / 1024**2,
            )
            self.reset()
        self._last_update_monotonic = now

        self._recent_reclaim_bytes = 0

        history = self._history(gathered_core)

        # The very first sample in each execution regime carries weight
        # page-fault and load-residue noise, so it seeds that regime's EWMA
        # but is excluded from its running max.
        if floor_sample and history.samples > 0:
            if transient_bytes <= self._OBSERVED_MAX_CLAMP_BYTES:
                history.observed_max_bytes = max(
                    history.observed_max_bytes, transient_bytes
                )
            else:
                logger.debug(
                    "PrefillTransientTracker(%s): rejected %d-byte outlier "
                    "from observed max (clamp %d)",
                    self._model_id,
                    transient_bytes,
                    self._OBSERVED_MAX_CLAMP_BYTES,
                )

        per_token = transient_bytes / n_tokens
        if history.samples == 0:
            history.ewma_per_token = per_token
        elif per_token > history.ewma_per_token * self._EWMA_OUTLIER_RATIO:
            # Reject from the EWMA blend: a single sample this far above
            # the running rate is more likely a noisy phys_footprint()
            # delta (see _record_chunk_transient's docstring on
            # buffer-pool-driven noise) than a real per-token cost jump.
            # last_delta_bytes/last_n_tokens below still record it raw for
            # diagnostics — only the accumulated EWMA is protected.
            logger.debug(
                "PrefillTransientTracker(%s): rejected %.1f-byte/token "
                "outlier from EWMA (current %.1f, ratio limit %.1fx)",
                self._model_id,
                per_token,
                history.ewma_per_token,
                self._EWMA_OUTLIER_RATIO,
            )
        else:
            history.ewma_per_token = (
                self._EWMA_ALPHA * per_token
                + (1.0 - self._EWMA_ALPHA) * history.ewma_per_token
            )
        history.samples += 1
        history.last_delta_bytes = transient_bytes
        history.last_n_tokens = n_tokens

    def predict(
        self,
        n_tokens: int,
        *,
        safety_factor: float = 1.2,
        gathered_core: bool = False,
    ) -> int:
        """Predicted transient bytes for a chunk of `n_tokens`.

        Returns 0 when no samples have been observed yet — caller must
        fall back to a static estimator in that case.
        """
        samples = self.samples_for(gathered_core)
        if samples == 0 or n_tokens <= 0:
            return 0
        return int(self.bytes_per_token_for(gathered_core) * n_tokens * safety_factor)

    def bytes_per_token_for(self, gathered_core: bool) -> float:
        """Return the EWMA for the selected execution regime."""
        return self._history(gathered_core).ewma_per_token

    def samples_for(self, gathered_core: bool) -> int:
        """Return the sample count for the selected execution regime."""
        return self._history(gathered_core).samples

    def last_delta_bytes_for(self, gathered_core: bool) -> int:
        """Return the latest measured delta for the selected regime."""
        return self._history(gathered_core).last_delta_bytes

    def last_n_tokens_for(self, gathered_core: bool) -> int:
        """Return the latest measured width for the selected regime."""
        return self._history(gathered_core).last_n_tokens

    def observed_max_bytes_for(self, gathered_core: bool) -> int:
        """Return the floor-size maximum for the selected regime."""
        return self._history(gathered_core).observed_max_bytes

    @property
    def bytes_per_token(self) -> float:
        """Current EWMA value (bytes per prefill token). 0.0 if no samples."""
        return self._dense_history.ewma_per_token

    @property
    def samples(self) -> int:
        """Number of chunks recorded since last reset."""
        return self._dense_history.samples

    @property
    def last_delta_bytes(self) -> int:
        """Bytes added by the most recently measured chunk."""
        return self._dense_history.last_delta_bytes

    @property
    def last_n_tokens(self) -> int:
        """Token count of the most recently measured chunk."""
        return self._dense_history.last_n_tokens

    @property
    def observed_max_bytes(self) -> int:
        """Bound on the largest accepted floor-chunk transient (dense
        regime) this idle window (0 if none yet). Rises instantly on a new
        peak; forgotten (not decayed) once ``_IDLE_RESET_SECONDS`` passes
        with no new sample, so it doesn't stay pinned at its all-time high
        for the rest of a long-running session (see ``update``'s idle
        check)."""
        return self._dense_history.observed_max_bytes

    @property
    def recent_reclaim_bytes(self) -> int:
        """Footprint released since the last positive chunk measurement."""
        return self._recent_reclaim_bytes

    def reset(self) -> None:
        """Drop all observations (model reload gets a fresh tracker by
        construction instead of calling this; ``update`` calls it directly
        after ``_IDLE_RESET_SECONDS`` of no samples)."""
        self._dense_history = _TransientHistory()
        self._gathered_history = _TransientHistory()
        self._recent_reclaim_bytes = 0
        self._last_update_monotonic = None
