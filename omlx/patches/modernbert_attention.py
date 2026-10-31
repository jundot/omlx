# SPDX-License-Identifier: Apache-2.0
"""Finite ModernBERT attention masks for mlx-vlm (issue #3507).

mlx-vlm builds the additive mask with -1e9, which becomes -inf in fp16, so a
fully masked padded query produces NaN.
"""

import logging

import mlx.core as mx
from mlx_vlm.models.modernbert.modernbert import Model

logger = logging.getLogger(__name__)

_MASKED = -60000.0


def _masks(self, attention_mask, dtype):
    """Preserve padding/window masks without overflowing fp16 SDPA scores."""
    batch_size, seq_len = attention_mask.shape
    additive = mx.where(attention_mask == 1, 0.0, _MASKED).astype(dtype)
    global_mask = mx.broadcast_to(
        additive[:, None, None, :], (batch_size, 1, seq_len, seq_len)
    )
    positions = mx.arange(seq_len)
    in_window = (
        mx.abs(positions[:, None] - positions[None, :]) <= self.local_attention // 2
    )
    sliding_mask = mx.where(in_window[None, None, :, :], global_mask, _MASKED)
    return global_mask, sliding_mask.astype(dtype)


def patch_modernbert_attention(model) -> None:
    """Patch the loaded ModernBERT class before compiling its forward pass."""
    if not isinstance(model, Model) or Model._masks is _masks:
        return
    Model._masks = _masks
    logger.info("Applied ModernBERT finite attention mask patch")
