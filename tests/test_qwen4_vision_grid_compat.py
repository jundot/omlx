# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp image/video grid compatibility with newer MLX releases."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


class _CaptureBlock:
    def __init__(self):
        self.cu_seqlens = None

    def __call__(self, hidden_states, *, cu_seqlens, rotary_pos_emb):
        self.cu_seqlens = cu_seqlens
        assert rotary_pos_emb.shape == hidden_states.shape
        return hidden_states


def _vision_shell():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models.qwen4_exp import VisionModel

    # Exercise the real Qwen4 override while replacing expensive ViT modules
    # with shape-preserving test doubles.  This reaches the exact mlx.repeat
    # boundary that regressed on MLX 0.32.2.
    vision = VisionModel.__new__(VisionModel)
    vision.patch_embed = lambda hidden_states: hidden_states
    vision.fast_pos_embed_interpolate = lambda grid_thw: mx.zeros(
        (int(mx.sum(mx.prod(grid_thw, axis=1)).item()), 4), dtype=mx.float32
    )
    vision.rot_pos_emb = lambda grid_thw: mx.zeros(
        (int(mx.sum(mx.prod(grid_thw, axis=1)).item()), 4), dtype=mx.float32
    )
    block = _CaptureBlock()
    vision.blocks = [block]
    vision.deepstack_visual_indexes = []
    vision.deepstack_merger_list = []
    vision.merger = lambda hidden_states: hidden_states
    return SimpleNamespace(vision=vision, block=block)


def test_qwen4_image_grid_scalarizes_mlx_repeat_count():
    shell = _vision_shell()
    grid_thw = mx.array([[1, 2, 3]], dtype=mx.int64)
    hidden_states = mx.zeros((6, 4), dtype=mx.float32)

    output, deepstack = shell.vision(hidden_states, grid_thw)
    mx.eval(output, shell.block.cu_seqlens)

    assert output.shape == (6, 4)
    assert deepstack == []
    assert shell.block.cu_seqlens.tolist() == [0, 6]


def test_qwen4_video_and_mixed_grids_preserve_frame_boundaries():
    shell = _vision_shell()
    # One two-frame video followed by a one-frame image.  The shared tower is
    # used for both modalities, and each temporal patch needs its own spatial
    # attention boundary.
    grid_thw = mx.array([[2, 2, 3], [1, 2, 2]], dtype=mx.int64)
    hidden_states = mx.zeros((16, 4), dtype=mx.float32)

    output, deepstack = shell.vision(hidden_states, grid_thw)
    mx.eval(output, shell.block.cu_seqlens)

    assert output.shape == (16, 4)
    assert deepstack == []
    assert shell.block.cu_seqlens.tolist() == [0, 6, 12, 16]
