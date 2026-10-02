# SPDX-License-Identifier: Apache-2.0
"""Tests for SpecPrefill target planning."""

from __future__ import annotations

import pytest

from omlx.specprefill.planning import plan_specprefill_target


def _all_tokens(system_token_count: int, conversation_token_count: int) -> list[int]:
    return list(range(system_token_count)) + list(
        range(1_000, 1_000 + conversation_token_count)
    )


@pytest.mark.parametrize(
    ("selected_indices", "expected_remove", "expected_sparse_count"),
    [([0, 2, 7], True, 2), ([0, 2], False, 2), ([7, 1, 7, 4], True, 3)],
)
def test_kickoff_selection_changes_tracker_count_once(
    selected_indices: list[int], expected_remove: bool, expected_sparse_count: int
):
    plan = plan_specprefill_target(
        all_tokens=_all_tokens(system_token_count=3, conversation_token_count=8),
        system_token_count=3,
        selected_indices=selected_indices,
        position_offset=3,
    )

    assert plan.remove_kickoff_index is expected_remove
    assert plan.sparse_selected_token_count == expected_sparse_count
    assert plan.total_tracker_prefill_count == 3 + expected_sparse_count


@pytest.mark.parametrize("system_token_count", [0, 3])
def test_target_plan_preserves_phase_inputs_and_position_offset(
    system_token_count: int,
):
    all_tokens = _all_tokens(system_token_count, conversation_token_count=8)
    plan = plan_specprefill_target(
        all_tokens=all_tokens,
        system_token_count=system_token_count,
        selected_indices=[0, 4],
        position_offset=17,
    )

    assert plan.system_token_count == system_token_count
    assert list(plan.conversation_tokens) == all_tokens[system_token_count:]
    assert plan.conversation_token_count == 8
    assert plan.generation_kickoff_index == 7
    assert plan.position_offset == 17
