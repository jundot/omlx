# SPDX-License-Identifier: Apache-2.0
"""Target-phase planning for SpecPrefill sparse prefill."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SpecPrefillTargetPlan:
    """Target-prefill inputs derived from selected conversation indices."""

    system_token_count: int
    conversation_tokens: Sequence[int]
    conversation_token_count: int
    generation_kickoff_index: int
    remove_kickoff_index: bool
    sparse_selected_token_count: int
    total_tracker_prefill_count: int
    position_offset: int


def plan_specprefill_target(
    *,
    all_tokens: Sequence[int],
    system_token_count: int,
    selected_indices: Sequence[int],
    position_offset: int,
) -> SpecPrefillTargetPlan:
    """Derive target-prefill inputs without touching MLX state."""
    conversation_tokens = all_tokens[system_token_count:]
    conversation_token_count = len(conversation_tokens)
    generation_kickoff_index = conversation_token_count - 1

    # BatchGenerator processes the generation-kickoff token separately.
    remove_kickoff_index = generation_kickoff_index in selected_indices
    sparse_selected_token_count = len(selected_indices) - int(remove_kickoff_index)

    return SpecPrefillTargetPlan(
        system_token_count=system_token_count,
        conversation_tokens=conversation_tokens,
        conversation_token_count=conversation_token_count,
        generation_kickoff_index=generation_kickoff_index,
        remove_kickoff_index=remove_kickoff_index,
        sparse_selected_token_count=sparse_selected_token_count,
        total_tracker_prefill_count=(
            system_token_count + sparse_selected_token_count
        ),
        position_offset=position_offset,
    )
