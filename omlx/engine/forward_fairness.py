# SPDX-License-Identifier: Apache-2.0
"""Let embedding forwards share GPU time with active chat decodes.

Embedding forwards bypass Scheduler.step(). This gate applies the same
shared hold deadline and decode-time debt between embedding batches.
"""

from __future__ import annotations

import time

import mlx.core as mx

from ..decode_activity import get_decode_activity
from ..scheduler import (
    _DECODE_ACTIVITY_TTL_S,
    _DECODE_FAIR_SHARE,
    _DECODE_STALL_TARGET_MS,
    SchedulerConfig,
)
from ..utils.proc_memory import get_phys_footprint

_FALLBACK_CONTENDED_ITEMS = 8
_EMA_ALPHA = 0.2


class ForwardFairnessGate:
    """Run holds and accounting on the global MLX executor thread.

    The event loop reads the per-item timing estimate for batch sizing and
    updates the shared config. The enforcer supplies the soft watermark.
    """

    def __init__(
        self, key: str, scheduler_config: SchedulerConfig | None = None
    ) -> None:
        self._key = key
        self._config = scheduler_config
        self._per_item_s_ema: float | None = None
        self._memory_soft_limit_bytes: int = 0

    def set_memory_soft_limit(self, soft_limit_bytes: int) -> None:
        self._memory_soft_limit_bytes = max(0, soft_limit_bytes)

    @property
    def enabled(self) -> bool:
        """Live view of the ``decode_fairness`` toggle (default on)."""
        return self._config is None or self._config.decode_fairness

    def contended(self) -> bool:
        return self.enabled and get_decode_activity().others_decoding(
            self._key, _DECODE_ACTIVITY_TTL_S
        )

    def chunk_cap(self) -> int | None:
        """Estimate an item cap from measured time per item under contention."""
        if not self.contended():
            return None
        ema = self._per_item_s_ema
        if not ema or ema <= 0.0:
            return _FALLBACK_CONTENDED_ITEMS
        cap = int((_DECODE_STALL_TARGET_MS / 1000.0) / ema)
        return max(1, cap)

    def wait_turn(self) -> bool:
        """Wait before the forward; exit early if the other decode finishes.

        Checking on the executor honors holds set by earlier queued forwards.
        """
        contended = False
        while self.contended():
            contended = True
            delay = get_decode_activity().hold_until() - time.perf_counter()
            if delay <= 0.0:
                break
            time.sleep(min(delay, 0.1))
        return contended

    def settle(self, forward_seconds: float, items: int, contended: bool) -> None:
        """Update batch timing and reserve decode time after a contended forward."""
        if items > 0 and forward_seconds > 0.0:
            per_item = forward_seconds / items
            ema = self._per_item_s_ema
            self._per_item_s_ema = (
                per_item
                if ema is None
                else (1.0 - _EMA_ALPHA) * ema + _EMA_ALPHA * per_item
            )
        if contended or self.contended():
            get_decode_activity().extend_hold(
                time.perf_counter() + max(0.0, forward_seconds) * _DECODE_FAIR_SHARE
            )

    def should_clear_cache(self) -> bool:
        """Keep reusable buffers during decode only below the soft watermark."""
        if not self.contended() or self._memory_soft_limit_bytes <= 0:
            return True
        try:
            budget = self._config.hot_cache_budget if self._config else None
            hot_cache_bytes = max(0, int(budget.total_bytes)) if budget else 0
            # Match Scheduler._current_usage_bytes: serialized hot-cache bytes
            # already have a separate reservation in the memory ceiling.
            phys = max(0, get_phys_footprint() - hot_cache_bytes)
            usage = max(mx.get_active_memory(), phys)
        except Exception:
            return True  # Clear if memory telemetry cannot be read.
        return usage >= self._memory_soft_limit_bytes
