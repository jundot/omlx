# SPDX-License-Identifier: Apache-2.0
"""Patch mlx-lm Llama 4 attention scaling for batched cache offsets."""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_omlx_llama4_attention_offset_patch"


def _llama4_attn_scales(
    offset: Any,
    length: int,
    floor_scale: int | float,
    attn_scale: float,
) -> mx.array:
    """Return Llama 4 query scale shaped for (B, H, L, D) broadcasting."""
    positions = mx.arange(1, length + 1)
    if isinstance(offset, mx.array):
        if offset.ndim == 0:
            positions = offset + positions
        else:
            positions = offset[..., None] + positions
    else:
        positions = offset + positions

    scales = mx.log(mx.floor(positions / floor_scale) + 1.0) * attn_scale + 1.0
    if scales.ndim == 1:
        return scales[None, None, :, None]
    if scales.ndim == 2:
        return scales[:, None, :, None]
    raise ValueError(f"Unexpected Llama 4 attention scale shape: {scales.shape}")


def _make_patched_attention_call(llama4_module):
    def patched_attention_call(self, x, mask=None, cache=None):
        batch_size, seq_len, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.reshape(batch_size, seq_len, self.n_heads, -1).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(batch_size, seq_len, self.n_kv_heads, -1).transpose(
            0, 2, 1, 3
        )

        offset = cache.offset if cache is not None else 0

        if self.use_rope:
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if self.use_qk_norm:
            queries = mx.fast.rms_norm(queries, weight=None, eps=1e-6)
            keys = mx.fast.rms_norm(keys, weight=None, eps=1e-6)

        if self.attn_temperature_tuning and not self.use_rope:
            attn_scales = _llama4_attn_scales(
                offset,
                seq_len,
                self.floor_scale,
                self.attn_scale,
            )
            queries = (queries * attn_scales).astype(queries.dtype)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = llama4_module.scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)

    patched_attention_call._omlx_llama4_attention_offset_patch = True
    return patched_attention_call


def apply_llama4_attention_patch() -> bool:
    """Patch mlx-lm's Llama 4 Attention.__call__ once.

    mlx-lm's current Llama 4 implementation passes ``BatchKVCache.offset``
    directly to ``mx.arange(start, stop)`` in no-RoPE global attention layers.
    MLX only accepts Python scalar arange bounds, so a batched cache offset
    raises ``TypeError`` before generation can start.
    """
    try:
        from mlx_lm.models import llama4
    except ImportError:
        logger.debug("llama4_attention: mlx_lm.models.llama4 not available")
        return False

    attention_cls = getattr(llama4, "Attention", None)
    if attention_cls is None:
        logger.debug("llama4_attention: Attention class not found")
        return False

    current_call = attention_cls.__dict__.get("__call__")
    if getattr(current_call, _PATCH_MARKER, False):
        return False

    attention_cls.__call__ = _make_patched_attention_call(llama4)
    logger.info("llama4_attention: patched mlx_lm.models.llama4.Attention.__call__")
    return True


__all__ = ["apply_llama4_attention_patch", "_llama4_attn_scales"]
