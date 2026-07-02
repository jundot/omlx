"""Unit tests for torch-free Qwen3-VL video preprocessing."""

from types import SimpleNamespace

import numpy as np
import pytest

from omlx.utils.video import TorchFreeQwen3VLVideoProcessor, smart_resize


def _make_processor(**overrides):
    image_processor = SimpleNamespace(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        do_rescale=True,
        rescale_factor=1 / 255.0,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_convert_rgb=True,
        **overrides,
    )
    return TorchFreeQwen3VLVideoProcessor(image_processor)


def _make_video(num_frames: int, height: int, width: int, value: int = 0) -> np.ndarray:
    return np.full((num_frames, height, width, 3), value, dtype=np.uint8)


def test_shape_eight_frames_240x320():
    processor = _make_processor()
    video = _make_video(8, 240, 320)
    out = processor([video])

    resized_h, resized_w = smart_resize(
        8,
        240,
        320,
        temporal_factor=2,
        factor=32,
        min_pixels=processor.min_pixels,
        max_pixels=processor.max_pixels,
    )
    grid_t = 4
    grid_h = resized_h // 16
    grid_w = resized_w // 16

    assert resized_h == 256
    assert resized_w == 320
    assert out["pixel_values_videos"].shape == (
        grid_t * grid_h * grid_w,
        3 * 2 * 16 * 16,
    )
    assert out["video_grid_thw"].tolist() == [[grid_t, grid_h, grid_w]]


def test_odd_frame_count_pads_last_frame():
    processor = _make_processor()
    video = _make_video(7, 240, 320)
    out = processor([video])

    resized_h, resized_w = smart_resize(
        7,
        240,
        320,
        temporal_factor=2,
        factor=32,
        min_pixels=processor.min_pixels,
        max_pixels=processor.max_pixels,
    )
    grid_t = 4
    grid_h = resized_h // 16
    grid_w = resized_w // 16

    assert out["video_grid_thw"].tolist() == [[grid_t, grid_h, grid_w]]
    assert out["pixel_values_videos"].shape[0] == grid_t * grid_h * grid_w


def test_normalization_constant_color():
    processor = _make_processor()
    video = _make_video(2, 32, 32, value=128)
    out = processor([video])
    expected = (128 / 255.0 - 0.5) / 0.5

    values = out["pixel_values_videos"]
    assert values.dtype == np.float32
    np.testing.assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


def test_patch_ordering_black_then_white():
    processor = _make_processor()
    frames = [
        np.zeros((32, 32, 3), dtype=np.uint8),
        np.full((32, 32, 3), 255, dtype=np.uint8),
    ]
    out = processor([frames])

    black = (0 / 255.0 - 0.5) / 0.5
    white = (1.0 - 0.5) / 0.5
    patch = out["pixel_values_videos"][0]
    spatial = 16 * 16
    assert np.all(patch[0:spatial] == pytest.approx(black))
    assert np.all(patch[spatial : 2 * spatial] == pytest.approx(white))


def test_two_videos_concatenated_grids():
    processor = _make_processor()
    video_a = _make_video(4, 64, 64)
    video_b = _make_video(4, 96, 128)
    out = processor([video_a, video_b])

    out_a = processor([video_a])
    out_b = processor([video_b])
    expected_rows = out_a["pixel_values_videos"].shape[0] + out_b["pixel_values_videos"].shape[0]

    assert out["pixel_values_videos"].shape[0] == expected_rows
    assert out["video_grid_thw"].shape == (2, 3)
    assert out["video_grid_thw"].tolist() == [
        out_a["video_grid_thw"][0].tolist(),
        out_b["video_grid_thw"][0].tolist(),
    ]


def test_video_metadata_fields_and_fps():
    processor = _make_processor()
    video = _make_video(4, 64, 64)
    out = processor([video], fps=3.5, return_metadata=True)

    metadata = out["video_metadata"][0]
    assert metadata.total_num_frames == 4
    assert metadata.fps == 3.5
    assert metadata.frames_indices == [0, 1, 2, 3]
    assert metadata.width == 64
    assert metadata.height == 64


def test_output_dtype_float32():
    processor = _make_processor()
    video = _make_video(2, 32, 32)
    out = processor([video])
    assert out["pixel_values_videos"].dtype == np.float32


def test_default_call_output_is_mlx_convertible():
    import mlx.core as mx

    processor = _make_processor()
    video = _make_video(4, 64, 64)
    out = processor([video])

    assert set(out.keys()) == {"pixel_values_videos", "video_grid_thw"}
    for value in out.values():
        mx.array(value)


def test_return_metadata_opt_in():
    processor = _make_processor()
    video = _make_video(4, 64, 64)
    out = processor([video], fps=3.5, return_metadata=True)

    assert "video_metadata" in out
    assert out["video_metadata"][0].fps == 3.5


def test_shim_attributes_from_image_processor():
    image_processor = SimpleNamespace(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        do_rescale=True,
        rescale_factor=1 / 255.0,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        do_convert_rgb=True,
    )
    processor = TorchFreeQwen3VLVideoProcessor(image_processor)
    assert processor.merge_size == 2
    assert processor.temporal_patch_size == 2
