# SPDX-License-Identifier: Apache-2.0
"""MLX 0.32.2 compatibility for the pinned Qwen3-VL vision tower.

The pinned mlx-vlm revision passes ``grid_thw[i, 0]`` (an MLX scalar array)
as ``mx.repeat``'s repeat count.  MLX 0.32.2 requires a Python integer.  This
is the same scalarization shipped upstream for the shared Qwen vision tower;
the tensor graph and image resolution remain otherwise unchanged.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


def _call_with_scalar_repeat(
    self: Any,
    hidden_states: mx.array,
    grid_thw: mx.array,
    **kwargs: Any,
) -> mx.array:
    del kwargs

    hidden_states = self.patch_embed(hidden_states)
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

    cu_seqlens = []
    for i in range(grid_thw.shape[0]):
        spatial_seq_len = grid_thw[i, 1] * grid_thw[i, 2]
        temporal_patches = int(grid_thw[i, 0])
        cu_seqlens.append(mx.repeat(spatial_seq_len, temporal_patches))

    cu_seqlens = mx.concatenate(cu_seqlens)
    cu_seqlens = mx.cumsum(cu_seqlens.astype(mx.int32), axis=0)
    cu_seqlens = mx.pad(
        cu_seqlens,
        (1, 0),
        mode="constant",
        constant_values=0,
    )

    deepstack_feature_lists = []
    for layer_num, block in enumerate(self.blocks):
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[
                self.deepstack_visual_indexes.index(layer_num)
            ](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)
    return hidden_states, deepstack_feature_lists


_call_with_scalar_repeat._omlx_qwen35_scalar_repeat = True


def apply_qwen35_vision_repeat_patch() -> bool:
    """Patch only the older mlx-vlm implementation that needs scalarization."""
    try:
        from mlx_vlm.models.qwen3_vl.vision import VisionModel
    except Exception:
        logger.debug("Qwen3-VL vision repeat patch import failed", exc_info=True)
        return False

    current = VisionModel.__call__
    if getattr(current, "_omlx_qwen35_scalar_repeat", False):
        return True
    try:
        source = inspect.getsource(current)
    except (OSError, TypeError):
        source = ""
    if "int(grid_thw[i, 0])" in source:
        return True
    if "mx.repeat(seq_len, grid_thw[i, 0])" not in source:
        logger.warning(
            "Qwen3-VL vision implementation is unfamiliar; refusing compatibility patch"
        )
        return False

    VisionModel.__call__ = _call_with_scalar_repeat
    logger.info("Qwen3-VL MLX-safe temporal repeat patch applied")
    return True


__all__ = ["apply_qwen35_vision_repeat_patch"]
