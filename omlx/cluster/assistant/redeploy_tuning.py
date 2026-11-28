# SPDX-License-Identifier: Apache-2.0
"""Real per-config redeploy tuning for MTP on/off and the MTP adaptive depth.

The sampler is a per-request knob, so it can be swept on a running cluster. The
MTP switch and the MTP adaptive depth are **per-model settings**: changing them
requires a redeploy, so the only honest way to tune them is to deploy each config
and measure its real rate (mtplx does the same; it is a one-off cost).

This module is pure. It plans the configs and ranks the measured rows; the
redeploy+measure seam is injected, so the sweep is fully testable without a Mac.
``routes`` wires the seam to ``evaluate_constellation`` (the proven search body)
and persists the winner in the ``TuningPresetStore`` so a later Start replays it.

Honesty rules, unchanged from the rest of the assistant:

* a model without an MTP head is never offered an MTP config — only ``mtp_off``
  is planned and ``mtp_applicable`` is false;
* a config that fails to deploy or reports no rate is not ranked (it is kept as
  an infeasible/unmeasured row for the report, never invented as a rate);
* the winner is the highest *measured* decode rate, then prefill, then the
  lower MTP depth (a shallower draft is cheaper to run when it ties).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

#: The MTP adaptive depth candidates. ``None`` (adaptive) is measured first so a
#: tie against a fixed depth stays adaptive, which oMLX prefers.
DEFAULT_DEPTHS: tuple[int, ...] = (1, 2, 3, 4)


@dataclass(frozen=True)
class RedeployConfig:
    """One config to deploy and measure: MTP off, or on at a fixed/adaptive depth."""

    mtp: bool
    mtp_depth: int | None = None

    @property
    def label(self) -> str:
        if not self.mtp:
            return "mtp_off"
        if self.mtp_depth is None:
            return "mtp_on_adaptive"
        return f"mtp_on_depth_{self.mtp_depth}"

    def settings_patch(self) -> dict[str, Any]:
        """The model settings this config needs before the launch.

        ``mtp_enabled`` is left to the plan's ``mtp`` switch (it is a launch
        input, not something this patch should fight), so only a fixed adaptive
        depth is written here. An empty patch means "measure the plan as is".
        """

        if self.mtp and self.mtp_depth is not None:
            return {"mtp_adaptive_max_depth": self.mtp_depth}
        return {}


@dataclass
class RedeployRow:
    """One measured config. A failure is a row, not an exception."""

    config: RedeployConfig
    feasible: bool
    decode_tok_per_s: float | None = None
    prefill_tok_per_s: float | None = None
    ttft_ms: float | None = None
    operation_id: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.label,
            "mtp": self.config.mtp,
            "mtp_depth": self.config.mtp_depth,
            "feasible": self.feasible,
            "decode_tok_per_s": self.decode_tok_per_s,
            "prefill_tok_per_s": self.prefill_tok_per_s,
            "ttft_ms": self.ttft_ms,
            "operation_id": self.operation_id,
            "error": self.error,
        }


@dataclass
class RedeployResult:
    rows: list[RedeployRow] = field(default_factory=list)
    best: RedeployConfig | None = None
    mtp_applicable: bool = True

    def to_dict(self) -> dict[str, Any]:
        best = self.best
        return {
            "mtp_applicable": self.mtp_applicable,
            "rows": [row.to_dict() for row in self.rows],
            "best": None
            if best is None
            else {
                "config": best.label,
                "mtp": best.mtp,
                "mtp_depth": best.mtp_depth,
            },
        }


def plan_configs(
    *,
    mtp_capable: bool,
    depths: Sequence[int] = DEFAULT_DEPTHS,
) -> list[RedeployConfig]:
    """The MTP off vs on comparison, plus the depth candidates when available.

    A model without an MTP head gets only ``mtp_off``: reserving a draft head it
    does not have would burn memory for numbers it cannot produce.
    """

    configs = [RedeployConfig(mtp=False)]
    if mtp_capable:
        configs.append(RedeployConfig(mtp=True, mtp_depth=None))
        configs.extend(
            RedeployConfig(mtp=True, mtp_depth=depth) for depth in depths
        )
    return configs


def choose_best(rows: Sequence[RedeployRow]) -> RedeployConfig | None:
    """Highest measured decode, then prefill, then the lower MTP depth."""

    measured = [
        row for row in rows if row.feasible and row.decode_tok_per_s is not None
    ]
    if not measured:
        return None

    def key(row: RedeployRow) -> tuple[float, float, int]:
        depth = row.config.mtp_depth
        # A fixed depth loses to adaptive (None) on a tie, so its sort value is
        # the depth itself and adaptive is 0.
        depth_rank = 0 if depth is None else depth
        return (
            float(row.decode_tok_per_s or 0.0),
            float(row.prefill_tok_per_s or 0.0),
            -depth_rank,
        )

    return max(measured, key=key).config


async def sweep(
    measure: Callable[[RedeployConfig], Awaitable[RedeployRow]],
    configs: Sequence[RedeployConfig],
    *,
    mtp_capable: bool = True,
) -> RedeployResult:
    """Deploy and measure every config in order; the caller owns the deploys.

    ``measure`` is the seam that redeploys one config and returns its measured
    row (a failed deploy is a row with ``feasible=False`` — it must never raise
    for a deployment refusal, so one bad config cannot abort the sweep).
    """

    rows: list[RedeployRow] = []
    for config in configs:
        rows.append(await measure(config))
    return RedeployResult(
        rows=rows,
        best=choose_best(rows),
        mtp_applicable=mtp_capable,
    )


def choose_candidate(
    candidates: Sequence[Any],
    *,
    wanted_order: Sequence[str],
    wanted_context: int,
) -> Any | None:
    """The candidate to measure for one config: the approved split first.

    MTP on/off changes rank-0's weight, so a config can legitimately prefer a
    different rank order (turning MTP off moved rank 0 from the M2 Ultra to the
    M5 Max here). The A/B is then the *best* constellation at the target context
    for that config — fewest Macs first, as everywhere else — instead of a
    fabricated "infeasible" row. The exact approved order is still preferred
    whenever it is feasible at this config.
    """

    order = tuple(wanted_order)
    exact = [
        candidate
        for candidate in candidates
        if tuple(candidate.rank_order) == order
    ]
    chosen = next(
        (
            candidate
            for candidate in exact
            if candidate.context_tokens == wanted_context
        ),
        None,
    )
    if chosen is not None:
        return chosen
    if exact:
        return min(
            exact,
            key=lambda candidate: abs(candidate.context_tokens - wanted_context),
        )
    same_context = [
        candidate
        for candidate in candidates
        if getattr(candidate, "feasible", True)
        and candidate.context_tokens == wanted_context
    ]
    if not same_context:
        return candidates[0] if candidates else None
    fewest = min(len(candidate.rank_order) for candidate in same_context)
    return next(
        candidate
        for candidate in same_context
        if len(candidate.rank_order) == fewest
    )
