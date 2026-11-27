"""Engine-facing runtime for the DFly2 draft.

The package ships the pure math (:func:`draft_block`); the serving path expects
a runtime object that can produce a masked block of hidden states and apply the
hidden correction for one step. This module is that adapter, kept free of any
model wiring so it can be exercised with synthetic layers.
"""

from __future__ import annotations

from typing import Callable, Sequence

import mlx.core as mx

from .dfly2_block import draft_block
from .dfly2_context import DFly2LayerContext


class DFly2Runtime:
    """Draft runtime bound to already-loaded weights.

    ``layers`` are the draft attention layers in stage order; ``fusion``,
    ``final_norm``, ``correct`` and ``lm_head`` come from the same checkpoint's
    heads and are shared across stages.
    """

    def __init__(
        self,
        *,
        layers: Sequence[Callable[[mx.array, int, DFly2LayerContext], mx.array]],
        fusion: Callable[[mx.array, mx.array], mx.array],
        final_norm: Callable[[mx.array], mx.array],
        correct: Callable[[mx.array, mx.array], mx.array],
        lm_head: Callable[[mx.array], mx.array],
        mask_token_id: int,
    ) -> None:
        if not layers:
            raise ValueError("DFly2Runtime needs at least one draft layer")
        self.layers = list(layers)
        self.fusion = fusion
        self.final_norm = final_norm
        self.correct_hidden = correct
        self.lm_head = lm_head
        self.mask_token_id = int(mask_token_id)

    def block_hidden(
        self,
        embed: Callable[[mx.array], mx.array],
        target_hidden: mx.array,
        bonus_token: mx.array,
        start_position: int,
        histories: Sequence[DFly2LayerContext],
        width: int,
    ) -> mx.array:
        """Hidden states for ``width`` slots: the bonus token plus proposals.

        Mirrors the first half of :func:`draft_block` exactly — same fusion, same
        layer order, same final norm — but stops before the head, because the
        engine takes the logits itself in ``dspark_step_logits``.
        """
        if width < 1:
            raise ValueError("width must cover at least the bonus token")
        if len(self.layers) != len(histories):
            raise ValueError("one target history is required for every draft layer")
        if bonus_token.shape != (1,):
            raise ValueError("bonus_token must be a single [1] token id")
        if target_hidden.ndim != 3 or target_hidden.shape[:2] != (1, 1):
            raise ValueError("target_hidden must be [1,1,D] for one request row")
        ids = mx.concatenate(
            [
                bonus_token.reshape(1, 1).astype(mx.int32),
                mx.full((1, width - 1), self.mask_token_id, mx.int32),
            ],
            axis=1,
        )
        x = embed(ids)
        x = mx.concatenate([self.fusion(x[:, :1], target_hidden.astype(x.dtype)), x[:, 1:]], axis=1)
        for layer, history in zip(self.layers, histories):
            x = layer(x, int(start_position), history)
        return self.final_norm(x)

    def correct(
        self,
        embed: Callable[[mx.array], mx.array],
        hidden: mx.array,
        previous_token: mx.array,
    ) -> mx.array:
        """Correct one slot's hidden state with the previous token's embedding."""
        token = previous_token.reshape(-1)[:1].astype(mx.int32)
        return self.correct_hidden(hidden, embed(token.reshape(1, 1))[:, 0].astype(hidden.dtype))

    def draft(
        self,
        *,
        embed: Callable[[mx.array], mx.array],
        target_hidden: mx.array,
        bonus_token: mx.array,
        start_position: int,
        histories: Sequence[DFly2LayerContext],
        block_size: int,
    ) -> tuple[mx.array, list[mx.array]]:
        """Full proposal block, delegating to the shipped reference math."""
        return draft_block(
            embed=embed,
            fusion=self.fusion,
            layers=self.layers,
            final_norm=self.final_norm,
            correct=self.correct_hidden,
            lm_head=self.lm_head,
            histories=histories,
            bonus_token=bonus_token,
            target_hidden=target_hidden,
            mask_token_id=self.mask_token_id,
            block_size=block_size,
            start_position=int(start_position),
        )
