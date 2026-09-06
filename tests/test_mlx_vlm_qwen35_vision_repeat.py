from types import SimpleNamespace

import mlx.core as mx

from omlx.patches import mlx_vlm_qwen35_vision_repeat as patch


def test_qwen35_vision_repeat_scalarizes_temporal_count():
    fake = SimpleNamespace(
        patch_embed=lambda value: value,
        fast_pos_embed_interpolate=lambda grid: mx.zeros((2, 4)),
        rot_pos_emb=lambda grid: mx.zeros((2, 4)),
        blocks=[],
        deepstack_visual_indexes=[],
        deepstack_merger_list=[],
        merger=lambda value: value,
    )

    output, deepstack = patch._call_with_scalar_repeat(
        fake,
        mx.zeros((2, 4)),
        mx.array([[1, 1, 2]], dtype=mx.int32),
    )

    mx.eval(output)
    assert output.shape == (2, 4)
    assert deepstack == []


def test_qwen35_vision_repeat_patch_is_idempotent():
    assert patch.apply_qwen35_vision_repeat_patch()
    assert patch.apply_qwen35_vision_repeat_patch()
