# SPDX-License-Identifier: Apache-2.0
"""Decode-fairness participation for engines that bypass the Scheduler.

EmbeddingEngine and RerankerEngine submit their forward passes straight
onto the global MLX executor and never pass through ``Scheduler.step()``,
so the ``decode_fairness`` protocol (chunk caps, shared hold windows and
decode-time debt -- see scheduler.py) cannot see them: a sustained
``/v1/embeddings`` load keeps the GPU command queue full with back-to-back
prefill-sized forwards and starves concurrent chat decode. Metal cannot
preempt a running kernel, so bounding forward duration and leaving the GPU
quiet during hold windows IS the interleave mechanism; this gate applies
those two rules to non-scheduler forwards, reusing the process-global
``DecodeActivityRegistry`` hold deadline that scheduler prefills already
honor -- concurrent embedders and prefillers pause together.

The gate is inert while no other engine is decoding (uncontended forwards
run back-to-back exactly as before), and it honors the same live
``decode_fairness`` toggle as the scheduler: the admin API mutates the
shared SchedulerConfig object this gate holds a reference to.
"""

from __future__ import annotations

import time
from typing import Any

from ..decode_activity import get_decode_activity
from ..scheduler import (
    _DECODE_ACTIVITY_TTL_S,
    _DECODE_FAIR_SHARE,
    _DECODE_STALL_TARGET_MS,
)

# Cold-start cap: items per contended forward before the first measurement
# has seeded the per-item EMA.
_FALLBACK_CONTENDED_ITEMS = 8
# EMA smoothing for the measured per-item forward duration.
_EMA_ALPHA = 0.2


class ForwardFairnessGate:
    """Fairness gate for one non-scheduler engine.

    ``wait_turn`` and ``settle`` run on the single global MLX executor
    thread, which serializes them across all non-scheduler engines;
    ``chunk_cap`` reads from the event loop. The only cross-thread state
    is a float EMA, whose torn reads are impossible under CPython.
    """

    def __init__(self, key: str, scheduler_config: Any | None = None) -> None:
        self._key = key
        self._config = scheduler_config
        self._per_item_s_ema: float | None = None

    @property
    def enabled(self) -> bool:
        """Live view of the ``decode_fairness`` toggle (default on)."""
        if self._config is None:
            return True
        return bool(getattr(self._config, "decode_fairness", True))

    def contended(self) -> bool:
        """True when another engine published a live decode recently."""
        if not self.enabled:
            return False
        try:
            return get_decode_activity().others_decoding(
                self._key, _DECODE_ACTIVITY_TTL_S
            )
        except Exception:
            return False

    def chunk_cap(self) -> int | None:
        """Max items per forward while contended (None = uncapped).

        Sized in TIME like the scheduler's contended prefill cap: a
        forward is the victim's decode stall, so the cap derives from the
        stall target and the measured per-item forward duration.
        """
        if not self.contended():
            return None
        ema = self._per_item_s_ema
        if not ema or ema <= 0.0:
            return _FALLBACK_CONTENDED_ITEMS
        cap = int((_DECODE_STALL_TARGET_MS / 1000.0) / ema)
        return max(1, cap)

    def wait_turn(self) -> bool:
        """Wait out the shared hold window; returns True if contended.

        Runs on the MLX executor thread immediately before the forward,
        which closes the check-then-run gap between concurrent requests:
        the task that lost the executor race re-reads the deadline the
        winner extended in ``settle``. Exits early when the victim decode
        finishes mid-hold.
        """
        contended = False
        while self.contended():
            contended = True
            try:
                delay = get_decode_activity().hold_until() - time.perf_counter()
            except Exception:
                break
            if delay <= 0.0:
                break
            time.sleep(min(delay, 0.1))
        return contended

    def settle(self, forward_seconds: float, items: int, contended: bool) -> None:
        """Account one finished forward.

        Updates the per-item duration EMA and, when the forward ran (or
        now runs) against a live decode, extends the shared hold by the
        same ``chunk_time * share`` debt scheduler prefills pay.
        """
        if items > 0 and forward_seconds > 0.0:
            per_item = forward_seconds / items
            ema = self._per_item_s_ema
            self._per_item_s_ema = (
                per_item
                if ema is None
                else (1.0 - _EMA_ALPHA) * ema + _EMA_ALPHA * per_item
            )
        if not (contended or self.contended()):
            return
        try:
            get_decode_activity().extend_hold(
                time.perf_counter()
                + max(0.0, forward_seconds) * _DECODE_FAIR_SHARE
            )
        except Exception:
            pass

    def should_clear_cache(self) -> bool:
        """Whether the end-of-forward ``mx.clear_cache`` should run.

        The flush is process-global: under decode contention it dumps the
        decoding engine's warm buffer pool on every forward, forcing that
        engine back to allocator round-trips mid-decode (mirrors
        ``Scheduler._should_clear_after_chunk``).
        """
        return not self.contended()
