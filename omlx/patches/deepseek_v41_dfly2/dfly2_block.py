"""DFly2 block-forward orchestration independent of a serving scheduler.

This is the exact state-machine boundary around the DSV41-specific layer stack:
slot 0 fuses the committed target hidden with the bonus embedding; the draft
layers receive per-layer target history; each following slot is corrected with
the immediately previous token embedding and greedily decoded left-to-right.
The target-specific MLA/MoE layer implementation is supplied by the caller.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import mlx.core as mx

from .dsv41_bridge import DSV41History


Layer = Callable[[mx.array, int, DSV41History], mx.array]


def draft_block(
    *,
    embed: Callable[[mx.array], mx.array],
    fusion: Callable[[mx.array, mx.array], mx.array],
    layers: Sequence[Layer],
    final_norm: Callable[[mx.array], mx.array],
    correct: Callable[[mx.array, mx.array], mx.array],
    lm_head: Callable[[mx.array], mx.array],
    histories: Sequence[DSV41History],
    bonus_token: mx.array,
    target_hidden: mx.array,
    mask_token_id: int,
    block_size: int,
    start_position: int,
) -> tuple[mx.array, list[mx.array]]:
    """Draft `block_size - 1` greedy proposals from a DFly2 B-block.

    `histories` contains only target-selected rows (`K + W`); every supplied
    DSV41 draft attention layer must append its own causal B-block keys. This
    separation is essential: injecting block placeholders into target history
    double-counts B positions and breaks the official `K + W + B` layout.
    """
    if block_size < 2:
        raise ValueError("block_size must contain a bonus token and one proposal")
    if len(layers) != len(histories):
        raise ValueError("one DSV41 target history is required for every draft layer")
    if bonus_token.shape != (1,):
        raise ValueError("bonus_token must be a single [1] token id")
    if target_hidden.ndim != 3 or target_hidden.shape[:2] != (1, 1):
        raise ValueError("target_hidden must be [1,1,D] for one request row")

    ids = mx.concatenate(
        [bonus_token.reshape(1, 1).astype(mx.int32), mx.full((1, block_size - 1), mask_token_id, mx.int32)],
        axis=1,
    )
    x = embed(ids)
    x = mx.concatenate([fusion(x[:, :1], target_hidden.astype(x.dtype)), x[:, 1:]], axis=1)
    for layer, history in zip(layers, histories):
        x = layer(x, start_position, history)
    x = final_norm(x)

    proposals: list[mx.array] = []
    logits: list[mx.array] = []
    previous = bonus_token
    for slot in range(1, block_size):
        corrected = correct(x[:, slot], embed(previous.reshape(1, 1))[:, 0].astype(x.dtype))
        slot_logits = lm_head(corrected)
        token = mx.argmax(slot_logits, axis=-1).astype(mx.int32)
        proposals.append(token.reshape(1))
        logits.append(slot_logits)
        previous = token.reshape(1)
    return mx.stack(proposals, axis=1), logits
