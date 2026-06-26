# SPDX-License-Identifier: Apache-2.0
"""mlx-vlm diffusion runtime compatibility patches."""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_APPLIED = False


def apply_mlx_vlm_diffusion_patch() -> bool:
    """Patch diffusion soft embeddings to honor non-affine quantization modes.

    mlx-vlm dequantizes the diffusion decoder embedding table once so soft
    token probabilities can use a regular matmul. The upstream helper omitted
    ``mode=embed_tokens.mode`` when calling ``mx.dequantize``; MXFP4 embeddings
    have no affine biases, so MLX interpreted them as affine and raised:

        [dequantize] Biases must be provided for affine quantization.
    """
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx.core as mx
        import mlx.nn as nn

        diffusion_mod = importlib.import_module("mlx_vlm.generate.diffusion")
    except Exception as e:  # noqa: BLE001
        logger.debug("mlx-vlm diffusion patch import failed: %s", e)
        return False

    original = getattr(diffusion_mod, "_diffusion_soft_embedding_weight", None)
    if original is None:
        return False
    if getattr(original, "_omlx_mxfp4_embedding_patch", False):
        _APPLIED = True
        return False

    def _patched_diffusion_soft_embedding_weight(embed_tokens):
        if isinstance(embed_tokens, nn.QuantizedEmbedding):
            return mx.dequantize(
                embed_tokens.weight,
                embed_tokens.scales,
                embed_tokens.biases,
                group_size=embed_tokens.group_size,
                bits=embed_tokens.bits,
                mode=getattr(embed_tokens, "mode", "affine"),
            )
        return embed_tokens.weight

    _patched_diffusion_soft_embedding_weight._omlx_mxfp4_embedding_patch = True
    diffusion_mod._diffusion_soft_embedding_weight = (
        _patched_diffusion_soft_embedding_weight
    )
    _APPLIED = True
    return True


def is_applied() -> bool:
    return _APPLIED
