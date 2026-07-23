# SPDX-License-Identifier: Apache-2.0
"""Tests for SpecPrefill scoring admission."""

from __future__ import annotations

import pytest

from omlx.specprefill.policy import plan_specprefill_scoring

DEFAULT_THRESHOLD = 8
DEFAULT_KEEP_PCT = 0.20


def _conversation_tokens(count: int) -> list[int]:
    return list(range(1_000, 1_000 + count))


def _plan(
    remaining_tokens: list[int],
    *,
    system_prompt_end: int = 0,
    cached_tokens: int = 0,
    requested_threshold: int | None = None,
    requested_keep_pct: float | None = None,
):
    return plan_specprefill_scoring(
        remaining_tokens=remaining_tokens,
        system_prompt_end=system_prompt_end,
        cached_tokens=cached_tokens,
        requested_threshold=requested_threshold,
        requested_keep_pct=requested_keep_pct,
        default_threshold=DEFAULT_THRESHOLD,
        default_keep_pct=DEFAULT_KEEP_PCT,
    )


@pytest.mark.parametrize(
    ("system_token_count", "conversation_token_count"),
    [(0, 8), (0, 7), (3, 8), (3, 7)],
)
def test_two_stage_admission_rejects_threshold_boundaries(
    system_token_count: int, conversation_token_count: int
):
    remaining_tokens = list(range(system_token_count)) + _conversation_tokens(
        conversation_token_count
    )

    assert (
        _plan(remaining_tokens, system_prompt_end=system_token_count) is None
    )


@pytest.mark.parametrize(
    ("system_prompt_end", "cached_tokens", "expected_effective_system"),
    [(5, 0, 5), (5, 3, 2), (5, 5, 0), (5, 8, 0)],
)
def test_system_prefix_exclusion_preserves_the_scoring_slice(
    system_prompt_end: int,
    cached_tokens: int,
    expected_effective_system: int,
):
    remaining_tokens = list(range(system_prompt_end)) + _conversation_tokens(10)

    plan = _plan(
        remaining_tokens,
        system_prompt_end=system_prompt_end,
        cached_tokens=cached_tokens,
    )

    assert plan is not None
    assert plan.effective_system == expected_effective_system
    assert list(plan.tokens_to_score) == remaining_tokens[expected_effective_system:]
    assert plan.n_to_score == len(remaining_tokens) - expected_effective_system


@pytest.mark.parametrize(
    ("requested_threshold", "token_count", "should_admit"),
    [(None, 8, False), (0, 8, False), (4, 5, True), (12, 10, False)],
)
def test_default_and_override_thresholds_control_admission(
    requested_threshold: int | None, token_count: int, should_admit: bool
):
    plan = _plan(
        _conversation_tokens(token_count),
        requested_threshold=requested_threshold,
    )

    assert (plan is not None) is should_admit


@pytest.mark.parametrize(
    ("requested_keep_pct", "expected_keep_pct"),
    [(None, DEFAULT_KEEP_PCT), (0, DEFAULT_KEEP_PCT), (0.35, 0.35)],
)
def test_keep_percentage_uses_default_or_override(
    requested_keep_pct: float | None, expected_keep_pct: float
):
    plan = _plan(_conversation_tokens(10), requested_keep_pct=requested_keep_pct)

    assert plan is not None
    assert plan.keep_pct == expected_keep_pct
