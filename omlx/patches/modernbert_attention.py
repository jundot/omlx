# SPDX-License-Identifier: Apache-2.0
"""Finite ModernBERT attention masks for mlx-embeddings (issue #3507).

Remove after the pinned dependency includes Blaizzy/mlx-embeddings#80.
"""

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def _update_attention_mask(self, attention_mask):
    """Preserve padding/window masks without overflowing fp16 SDPA scores."""
    batch_size, seq_len = attention_mask.shape
    # Keep the additive mask floating point, including for int32 token masks.
    # -1e9 becomes -inf in fp16; fully masked padded queries then produce NaN.
    # Norm weights retain the compute dtype even when embedding weights are uint32.
    dtype = self.embeddings.norm.weight.dtype
    additive_mask = mx.where(attention_mask == 1, 0.0, -60000.0).astype(dtype)
    global_mask = mx.broadcast_to(
        additive_mask[:, None, None, :], (batch_size, 1, seq_len, seq_len)
    )
    positions = mx.arange(seq_len)
    in_window = (
        mx.abs(positions[:, None] - positions[None, :])
        <= self.config.local_attention // 2
    )
    local_mask = mx.where(in_window[None, None, :, :], global_mask, -60000.0)
    return global_mask, local_mask


def patch_modernbert_attention(model):
    """Patch the loaded ModernBERT class before compiling its forward pass."""
    if type(model).__module__ != "mlx_embeddings.models.modernbert":
        return
    model_cls = type(model.model)
    if model_cls._update_attention_mask is _update_attention_mask:
        return
    model_cls._update_attention_mask = _update_attention_mask
    logger.info("Applied ModernBERT finite attention mask patch")
