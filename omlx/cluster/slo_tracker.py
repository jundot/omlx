# SPDX-License-Identifier: Apache-2.0
"""SLO tracker — evaluates compliance and burn rate for defined SLOs."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..slo_definitions import DEFAULT_SLOS, SLO, SLOMetricKind, SLOStatus


@dataclass(slots=True)
class SLOEvaluation:
    """Result of evaluating one SLO over its rolling window."""

    slo: SLO
    status: SLOStatus
    compliance_pct: float
    burn_rate: float
    window_seconds: int
    sample_count: int
    current_value: float | None = None
    time_to_breach_seconds: float | None = None


class _RollingWindow:
    """Thread-safe fixed-duration rolling window for numeric samples."""

    __slots__ = ("_max_seconds", "_lock", "_samples")

    def __init__(self, max_seconds: int) -> None:
        self._max_seconds = max_seconds
        self._lock = threading.Lock()
        self._samples: deque[tuple[float, float]] = deque()

    def record(self, value: float, *, now: float | None = None) -> None:
        ts = now if now is not None else time.time()
        with self._lock:
            self._samples.append((ts, value))
            self._evict(ts)

    def values(self, *, now: float | None = None) -> list[float]:
        ts = now if now is not None else time.time()
        with self._lock:
            self._evict(ts)
            return [v for _, v in self._samples]

    def _evict(self, now: float) -> None:
        cutoff = now - self._max_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()


class SLOTracker:
    """Evaluates SLO compliance and burn rate from rolling metric windows.

    This tracker maintains its own rolling buffers keyed by ``metric_key``.
    When ``SampleStore`` from #2773 is available, callers can bridge data
    into these windows via :meth:`record`.
    """

    def __init__(self, slos: list[SLO] | None = None) -> None:
        self._slos = list(slos or DEFAULT_SLOS)
        self._lock = threading.Lock()
        # One rolling window per unique metric_key, sized to the longest
        # window that references it.
        max_windows: dict[str, int] = {}
        for slo in self._slos:
            key = slo.metric_key
            max_windows[key] = max(max_windows.get(key, 0), slo.window_seconds)
        self._windows: dict[str, _RollingWindow] = {
            key: _RollingWindow(span) for key, span in max_windows.items()
        }

    @property
    def slos(self) -> list[SLO]:
        return list(self._slos)

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def record(self, metric_key: str, value: float, *, now: float | None = None) -> None:
        """Append a sample into the rolling window for *metric_key*."""
        window = self._windows.get(metric_key)
        if window is not None:
            window.record(value, now=now)

    def record_many(self, metric_key: str, values: list[float], *, now: float | None = None) -> None:
        """Bulk-append samples into the rolling window for *metric_key*."""
        window = self._windows.get(metric_key)
        if window is not None:
            for v in values:
                window.record(v, now=now)

    def clear(self, metric_key: str | None = None) -> None:
        """Clear one or all rolling windows."""
        if metric_key is not None:
            window = self._windows.get(metric_key)
            if window is not None:
                window.clear()
        else:
            for window in self._windows.values():
                window.clear()

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _percentile(sorted_vals: list[float], pct: float) -> float | None:
        if not sorted_vals:
            return None
        idx = min(int(len(sorted_vals) * pct / 100.0), len(sorted_vals) - 1)
        return sorted_vals[idx]

    def _latest_sample(self, metric_key: str) -> float | None:
        window = self._windows.get(metric_key)
        if window is None:
            return None
        vals = window.values()
        return vals[-1] if vals else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, slo: SLO, *, now: float | None = None) -> SLOEvaluation:
        """Evaluate a single SLO and return its current status."""
        window = self._windows.get(slo.metric_key)
        if window is None:
            return SLOEvaluation(
                slo=slo,
                status=SLOStatus.HEALTHY,
                compliance_pct=100.0,
                burn_rate=0.0,
                window_seconds=slo.window_seconds,
                sample_count=0,
            )

        samples = window.values(now=now)
        sample_count = len(samples)
        ts = now if now is not None else time.time()

        if sample_count == 0:
            return SLOEvaluation(
                slo=slo,
                status=SLOStatus.HEALTHY,
                compliance_pct=100.0,
                burn_rate=0.0,
                window_seconds=slo.window_seconds,
                sample_count=0,
            )

        # Compute the aggregate metric depending on kind
        sorted_vals = sorted(samples)
        if slo.metric_kind == SLOMetricKind.LATENCY:
            # Use the percentile that matches the name; fall back to median
            pct_map = {"p50": 50, "p95": 95, "p99": 99}
            pct = 50
            for token, p in pct_map.items():
                if token in slo.name:
                    pct = p
                    break
            current_value = self._percentile(sorted_vals, pct)
            # Lower is better for latency
            compliant = current_value is not None and current_value <= slo.target
            degraded = (
                slo.degraded_target is not None
                and current_value is not None
                and current_value <= slo.degraded_target
            )
        elif slo.metric_kind == SLOMetricKind.THROUGHPUT:
            current_value = sum(samples) / len(samples)
            # Higher is better for throughput
            compliant = current_value >= slo.target
            degraded = slo.degraded_target is not None and current_value >= slo.degraded_target
        elif slo.metric_kind == SLOMetricKind.CACHE_HIT_RATE:
            current_value = sum(samples) / len(samples)
            # Higher is better
            compliant = current_value >= slo.target
            degraded = slo.degraded_target is not None and current_value >= slo.degraded_target
        else:
            # AVAILABILITY — treat samples as boolean (1=up, 0=down)
            current_value = (sum(samples) / len(samples)) * 100.0
            compliant = current_value >= slo.target
            degraded = slo.degraded_target is not None and current_value >= slo.degraded_target

        # Determine status
        if compliant:
            status = SLOStatus.HEALTHY
        elif degraded:
            status = SLOStatus.DEGRADED
        else:
            status = SLOStatus.BREACHED

        # Compliance %: fraction of samples that meet the target
        if slo.metric_kind in (SLOMetricKind.LATENCY,):
            compliant_count = sum(1 for v in samples if v <= slo.target)
        else:
            compliant_count = sum(1 for v in samples if v >= slo.target)
        compliance_pct = (compliant_count / sample_count) * 100.0

        # Burn rate: how fast we are consuming error budget
        error_budget_pct = 100.0 - slo.target if slo.metric_kind != SLOMetricKind.LATENCY else slo.target
        # For latency, the budget is measured in time fraction
        # Simplified: burn_rate = (100 - compliance_pct) / (100 - target)
        if slo.metric_kind == SLOMetricKind.LATENCY:
            # Error budget = allowed fraction above target
            # Simplified: 1 - compliance_pct / 100 gives the actual error rate
            actual_error_rate = 1.0 - (compliance_pct / 100.0)
            budget_fraction = 1.0 - (slo.target / 100.0) if slo.target < 100 else 0.01
            burn_rate = actual_error_rate / budget_fraction if budget_fraction > 0 else 0.0
        else:
            actual_error_rate = 1.0 - (compliance_pct / 100.0)
            budget_fraction = 1.0 - (slo.target / 100.0) if slo.target < 100 else 0.01
            burn_rate = actual_error_rate / budget_fraction if budget_fraction > 0 else 0.0

        # Time to breach: if burn_rate > 1, estimate how long until budget exhausted
        time_to_breach: float | None = None
        if burn_rate > 1.0:
            # budget_remaining = budget_fraction * window_size (in samples)
            # At current burn rate, budget exhausted in window / burn_rate seconds
            time_to_breach = slo.window_seconds / burn_rate

        return SLOEvaluation(
            slo=slo,
            status=status,
            compliance_pct=round(compliance_pct, 2),
            burn_rate=round(burn_rate, 4),
            window_seconds=slo.window_seconds,
            sample_count=sample_count,
            current_value=round(current_value, 4) if current_value is not None else None,
            time_to_breach_seconds=round(time_to_breach, 1) if time_to_breach is not None else None,
        )

    def status(self, *, now: float | None = None) -> list[SLOEvaluation]:
        """Evaluate all SLOs and return their current status."""
        return [self.evaluate(slo, now=now) for slo in self._slos]

    def status_dict(self, *, now: float | None = None) -> dict[str, Any]:
        """Return a serialisable dict of all SLO statuses."""
        evaluations = self.status(now=now)
        return {
            "slos": [
                {
                    "name": ev.slo.name,
                    "metric_kind": ev.slo.metric_kind.value,
                    "description": ev.slo.description,
                    "target": ev.slo.target,
                    "window_seconds": ev.window_seconds,
                    "status": ev.status.value,
                    "compliance_pct": ev.compliance_pct,
                    "burn_rate": ev.burn_rate,
                    "sample_count": ev.sample_count,
                    "current_value": ev.current_value,
                    "time_to_breach_seconds": ev.time_to_breach_seconds,
                }
                for ev in evaluations
            ],
            "overall_status": _overall_status(evaluations),
        }


def _overall_status(evaluations: list[SLOEvaluation]) -> str:
    """Derive the aggregate status across all SLO evaluations."""
    if any(ev.status == SLOStatus.BREACHED for ev in evaluations):
        return SLOStatus.BREACHED.value
    if any(ev.status == SLOStatus.DEGRADED for ev in evaluations):
        return SLOStatus.DEGRADED.value
    return SLOStatus.HEALTHY.value
