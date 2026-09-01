# SPDX-License-Identifier: Apache-2.0
"""Error budget tracker — computes remaining budget and consumption rate per SLO."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..slo_definitions import SLO
from .slo_tracker import SLOTracker


@dataclass(slots=True)
class ErrorBudget:
    """Remaining error budget for a single SLO."""

    slo_name: str
    budget_remaining_pct: float
    consumption_rate: float
    time_to_exhaustion_hours: float | None
    window_seconds: int
    sample_count: int


class ErrorBudgetTracker:
    """Tracks error budget consumption for all SLOs using an SLOTracker.

    The error budget is the allowed error rate: ``100 - target`` for
    availability-style SLOs, or the fraction of samples allowed to be
    non-compliant.  Consumption is derived from the current burn rate.
    """

    ALERT_THRESHOLDS = (10.0, 25.0, 50.0)

    def __init__(self, slo_tracker: SLOTracker | None = None) -> None:
        self._tracker = slo_tracker or SLOTracker()
        self._lock = threading.Lock()

    @property
    def tracker(self) -> SLOTracker:
        return self._tracker

    def _compute_budget(
        self, slo: SLO, *, now: float | None = None
    ) -> ErrorBudget:
        evaluation = self._tracker.evaluate(slo, now=now)

        if evaluation.sample_count == 0:
            budget_remaining = 100.0
            rate = 0.0
        else:
            # Allowed error budget as a percentage
            error_budget_allowed_pct = 100.0 - slo.target
            # Actual errors observed
            actual_error_pct = 100.0 - evaluation.compliance_pct

            if error_budget_allowed_pct > 0:
                budget_remaining = max(
                    0.0,
                    min(
                        100.0,
                        (error_budget_allowed_pct - actual_error_pct)
                        / error_budget_allowed_pct
                        * 100.0,
                    ),
                )
            else:
                budget_remaining = 100.0 if actual_error_pct == 0 else 0.0

            rate = max(0.0, evaluation.burn_rate - 1.0)

        time_to_exhaustion: float | None = None
        if rate > 0.0 and budget_remaining > 0.0:
            time_to_exhaustion = budget_remaining / rate

        return ErrorBudget(
            slo_name=slo.name,
            budget_remaining_pct=round(budget_remaining, 2),
            consumption_rate=round(rate, 4),
            time_to_exhaustion_hours=round(time_to_exhaustion, 1)
            if time_to_exhaustion is not None
            else None,
            window_seconds=slo.window_seconds,
            sample_count=evaluation.sample_count,
        )

    def check_budget(
        self, slo_name: str, *, now: float | None = None
    ) -> ErrorBudget:
        """Return the error budget for a named SLO."""
        slo = next(
            (s for s in self._tracker.slos if s.name == slo_name), None
        )
        if slo is None:
            raise ValueError(f"unknown SLO: {slo_name}")
        with self._lock:
            return self._compute_budget(slo, now=now)

    def all_budgets(self, *, now: float | None = None) -> list[ErrorBudget]:
        """Return error budgets for all defined SLOs."""
        with self._lock:
            return [self._compute_budget(slo, now=now) for slo in self._tracker.slos]

    def can_deploy(
        self, *, now: float | None = None
    ) -> tuple[bool, list[str]]:
        """Check whether deployment is allowed based on error budgets.

        Deployment is blocked when any SLO has less than 10% budget remaining.

        Returns:
            Tuple of (can_deploy, list of blocking SLO names).
        """
        budgets = self.all_budgets(now=now)
        blocking = [
            b.slo_name
            for b in budgets
            if b.budget_remaining_pct < 10.0
        ]
        return (len(blocking) == 0, blocking)

    def alert_thresholds(self, *, now: float | None = None) -> list[dict[str, object]]:
        """Return budget entries that have crossed a threshold.

        Thresholds are 10%, 25% and 50% remaining.
        """
        budgets = self.all_budgets(now=now)
        alerts: list[dict[str, object]] = []
        for budget in budgets:
            for threshold in self.ALERT_THRESHOLDS:
                if budget.budget_remaining_pct < threshold:
                    alerts.append(
                        {
                            "slo_name": budget.slo_name,
                            "budget_remaining_pct": budget.budget_remaining_pct,
                            "threshold_pct": threshold,
                            "consumption_rate": budget.consumption_rate,
                        }
                    )
                    break  # report only the tightest threshold crossed
        return alerts

    def budget_status(self, *, now: float | None = None) -> dict[str, object]:
        """Serializable budget status suitable for the error-budget API."""
        can, blocking = self.can_deploy(now=now)
        budgets = self.all_budgets(now=now)
        return {
            "slos": [
                {
                    "slo_name": b.slo_name,
                    "budget_remaining_pct": b.budget_remaining_pct,
                    "consumption_rate": b.consumption_rate,
                    "time_to_exhaustion_hours": b.time_to_exhaustion_hours,
                    "window_seconds": b.window_seconds,
                    "sample_count": b.sample_count,
                }
                for b in budgets
            ],
            "can_deploy": can,
            "blocking_slos": blocking,
        }
