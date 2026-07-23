# SPDX-License-Identifier: Apache-2.0
"""Admission policy for SpecPrefill draft scoring."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SpecPrefillScoringPlan:
    """Scoring inputs accepted for one request."""

    tokens_to_score: Sequence[int]
    effective_system: int
    keep_pct: float
    n_to_score: int


def plan_specprefill_scoring(
    *,
    remaining_tokens: Sequence[int],
    system_prompt_end: int,
    cached_tokens: int,
    requested_threshold: int | None,
    requested_keep_pct: float | None,
    default_threshold: int,
    default_keep_pct: float,
) -> SpecPrefillScoringPlan | None:
    """Return scoring inputs when both admission checks accept the request."""
    threshold = requested_threshold or default_threshold
    keep_pct = requested_keep_pct or default_keep_pct

    # First admission applies to all remaining prompt tokens.
    if len(remaining_tokens) <= threshold:
        return None

    # Exclude only uncached system-prefix tokens before the second admission.
    effective_system = max(0, system_prompt_end - cached_tokens)
    tokens_to_score = (
        remaining_tokens[effective_system:]
        if effective_system > 0
        else remaining_tokens
    )
    n_to_score = len(tokens_to_score)
    if n_to_score <= threshold:
        return None

    return SpecPrefillScoringPlan(
        tokens_to_score=tokens_to_score,
        effective_system=effective_system,
        keep_pct=keep_pct,
        n_to_score=n_to_score,
    )
