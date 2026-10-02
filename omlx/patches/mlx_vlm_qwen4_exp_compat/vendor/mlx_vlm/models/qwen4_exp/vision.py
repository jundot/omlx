"""Qwen4 vision compatibility bridge for the pinned mlx-vlm revision.

The ``VisionModel.__call__`` implementation below follows mlx-vlm commit
``1249c7d`` (PR #1982) and is used under mlx-vlm's MIT license:

Copyright © 2025 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import mlx.core as mx

from ..qwen3_vl import VisionModel as Qwen3VLVisionModel


class VisionModel(Qwen3VLVisionModel):
    def __init__(self, config):
        # The pinned qwen3_vl tower predates the qwen4_exp type aliases; its
        # graph is otherwise the same published ViT used by Flash Next.
        original_type = config.model_type
        config.model_type = "qwen3_5_moe_vision"
        try:
            super().__init__(config)
        finally:
            config.model_type = original_type
        self.model_type = original_type

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        **kwargs,
    ) -> mx.array:
        """Run the shared Qwen ViT with MLX-safe temporal repeat counts.

        oMLX currently pins mlx-vlm before upstream #1982.  That revision
        passes a scalar ``mx.array`` as ``mx.repeat(..., repeats=...)``, which
        newer MLX releases reject.  Qwen4-Exp inherits that tower for both
        images and videos, so keep the upstream computation verbatim while
        scalarizing only the temporal repeat count.  This is the same fix
        shipped by mlx-vlm commit 1249c7d.
        """
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
