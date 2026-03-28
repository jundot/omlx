# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-vlm Qwen3_5Attention to remove mask truncation.

mlx-vlm's Qwen3_5Attention.__call__ truncates the attention mask via
``mask[..., :kv_seq_len]`` using a scalar ``kv_seq_len`` derived from
``cache.offset``.  With batched caches containing different per-element
offsets (BatchKVCache), any scalar reduction is lossy -- the mask gets
truncated too short for the longer prompt, causing
``ValueError: [broadcast_shapes]``.

The mask from ``create_attention_mask()`` / ``BatchKVCache.make_mask()``
is already correctly sized.  The mlx-lm version of the same model
(Qwen3NextAttention) does NOT truncate -- it passes the mask directly
to SDPA.  This patch makes the mlx-vlm code path match.

Supported model types: qwen3_5
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Optional

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)

_SUPPORTED_MODEL_TYPES = {"qwen3_5"}

# Track whether the class-level patch has been applied
_class_patch_applied = False


def _get_model_type(model: Any) -> str | None:
    """Extract model_type string from a loaded model."""
    for attr in ("model_type", "args"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        if isinstance(obj, str):
            return obj
        mt = getattr(obj, "model_type", None)
        if isinstance(mt, str):
            return mt
    return None


def _make_patched_attention_call():
    """Create a patched __call__ for Qwen3_5Attention.

    Removes the mask truncation (``mask[..., :kv_seq_len]``) that uses a
    scalar ``kv_seq_len`` derived from ``cache.offset``.  The mask from
    ``create_attention_mask()`` is already correctly sized for batched
    inference.
    """
    from mlx_vlm.models.base import scaled_dot_product_attention
    from mlx_vlm.models.qwen3_5.language import apply_multimodal_rotary_pos_emb

    def patched_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, D = x.shape

        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            B, L, self.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)

        if position_ids is None:
            position_ids = mx.arange(cache.offset, cache.offset + L)
            position_ids = mx.expand_dims(position_ids, axis=0)
            position_ids = mx.tile(position_ids, (3, 1, 1))

        cos, sin = self.rotary_emb(values, position_ids)

        # Mask truncation removed: the mask from create_attention_mask()
        # is already correctly sized.  The original code truncated via
        # mask[..., :kv_seq_len] using a scalar kv_seq_len derived from
        # cache.offset, which is wrong for batched caches with different
        # per-element offsets.

        queries, keys = apply_multimodal_rotary_pos_emb(queries, keys, cos, sin)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(output * mx.sigmoid(gate))

    return patched_call


def apply_vlm_attention_mask_patch(model: Any) -> bool:
    """Apply mask-truncation removal to mlx-vlm Qwen3_5Attention.

    Args:
        model: A loaded model instance (VLMModelAdapter or raw VLM).

    Returns:
        True if the patch was applied, False if not needed or not supported.
    """
    global _class_patch_applied

    model_type = _get_model_type(model)
    if model_type not in _SUPPORTED_MODEL_TYPES:
        return False

    if _class_patch_applied:
        return True

    try:
        from mlx_vlm.models.qwen3_5.language import Qwen3_5Attention
    except ImportError:
        logger.debug("VLM attention mask patch: qwen3_5 module not found")
        return False

    # Forward compatibility: skip if upstream already removed the truncation
    try:
        source = inspect.getsource(Qwen3_5Attention.__call__)
        if "kv_seq_len" not in source:
            logger.debug(
                "VLM attention mask patch: upstream already removed truncation, skipping"
            )
            _class_patch_applied = True
            return False
    except (OSError, TypeError):
        pass

    Qwen3_5Attention.__call__ = _make_patched_attention_call()

    _class_patch_applied = True
    logger.info("VLM attention mask truncation patch applied for qwen3_5")
    return True
