"""Greedy DFly2 verification contract with rollback at the target-cache boundary.

The target forward owns its real cache rollback.  This module deliberately owns
only the DFly2 bridge mirror: after the first rejected proposal, invoke the
same absolute-position truncation on the bridge that the target cache receives.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class GreedyVerification:
    target_tokens: mx.array  # [1, proposals]
    accepted: int
    rejected_position: int | None


def greedy_prefix(
    *,
    proposals: mx.array,
    target_logits: Sequence[mx.array],
    anchor_position: int,
    rollback: Callable[[int], None],
) -> GreedyVerification:
    """Compare DFly proposals with target greedy IDs and truncate on mismatch.

    Candidate `proposals[:, 0]` corresponds to absolute target position
    `anchor_position`. A mismatch at relative slot ``i`` means target cache and
    DFly bridge must discard candidate state at positions >= ``anchor + i``;
    target-side sampling/continuation is intentionally outside this pure helper.
    """
    if proposals.ndim != 2 or proposals.shape[0] != 1:
        raise ValueError("proposals must be [1, proposal_count]")
    if anchor_position < 0:
        raise ValueError("anchor_position must be non-negative")
    count = int(proposals.shape[1])
    if len(target_logits) != count:
        raise ValueError("one target-logit row is required for every proposal")
    if count == 0:
        return GreedyVerification(target_tokens=proposals, accepted=0, rejected_position=None)

    target_tokens = mx.stack([mx.argmax(row, axis=-1).astype(mx.int32).reshape(1) for row in target_logits], axis=1)
    # Materialize just IDs for control flow; model logits/cache values remain MLX.
    proposed_ids = [int(value) for value in proposals.reshape(-1).tolist()]
    target_ids = [int(value) for value in target_tokens.reshape(-1).tolist()]
    accepted = next((i for i, (draft, target) in enumerate(zip(proposed_ids, target_ids)) if draft != target), count)
    rejected_position = None if accepted == count else anchor_position + accepted
    if rejected_position is not None:
        rollback(rejected_position)
    return GreedyVerification(target_tokens=target_tokens, accepted=accepted, rejected_position=rejected_position)
