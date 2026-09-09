# SPDX-License-Identifier: Apache-2.0
"""Deployment gate — blocks deployment activation when error budget is exhausted."""

from __future__ import annotations

from dataclasses import dataclass

from .error_budget import ErrorBudgetTracker


@dataclass(frozen=True, slots=True)
class GateResult:
    """Result of a deployment gate check."""

    allowed: bool
    blocking_slos: list[str]
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "blocking_slos": self.blocking_slos,
            "message": self.message,
        }


BUDGET_THRESHOLD_PCT = 10.0


class DeploymentGate:
    """Prevents deployment activation when error budget is nearly exhausted.

    A deployment is blocked when any SLO has less than ``BUDGET_THRESHOLD_PCT``
    percent of its error budget remaining.
    """

    def __init__(
        self,
        budget_tracker: ErrorBudgetTracker | None = None,
        *,
        threshold_pct: float = BUDGET_THRESHOLD_PCT,
    ) -> None:
        self._tracker = budget_tracker or ErrorBudgetTracker()
        self._threshold_pct = threshold_pct

    @property
    def budget_tracker(self) -> ErrorBudgetTracker:
        return self._tracker

    def check(self, *, now: float | None = None) -> GateResult:
        """Evaluate error budget and decide whether deployment may proceed."""
        budgets = self._tracker.all_budgets(now=now)
        blocking = [
            b.slo_name
            for b in budgets
            if b.budget_remaining_pct < self._threshold_pct
        ]

        if not blocking:
            return GateResult(
                allowed=True,
                blocking_slos=[],
                message="Deployment allowed — all SLO error budgets are sufficient.",
            )

        slo_list = ", ".join(blocking)
        return GateResult(
            allowed=False,
            blocking_slos=blocking,
            message=(
                f"Deployment blocked — {len(blocking)} SLO(s) have less than "
                f"{self._threshold_pct:.0f}% error budget remaining: {slo_list}"
            ),
        )
