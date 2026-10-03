import base64
import hashlib
import json
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from omlx.api.openai_models import ContentPart
from omlx.exceptions import InvalidRequestError
from omlx.utils import video as video_module
from omlx.utils.image import extract_images_from_messages
from omlx.utils.video import (
    NATIVE_VIDEO_MAX_FRAMES,
    VideoInfo,
    _sample_indices,
    attach_native_video_processor,
    expand_video_parts,
    get_max_video_bytes,
    native_video_token_count,
    probe_video,
    write_video_data_uri,
)


def test_sample_indices_are_bounded_and_chronological():
    assert _sample_indices(100, 4) == [0, 33, 66, 99]
    assert _sample_indices(3, 16) == [0, 1, 2]
    assert _sample_indices(0, 16) == []


def test_content_part_preserves_video_url():
    part = ContentPart(
        type="video_url",
        video_url={"url": "data:video/mp4;base64,AAAA"},
    )

    assert part.video_url is not None
    assert part.video_url.url.startswith("data:video/mp4")


def test_video_parts_expand_to_images_in_original_order(monkeypatch):
    first = Image.new("RGB", (2, 2), "red")
    second = Image.new("RGB", (2, 2), "blue")
    monkeypatch.setattr(
        "omlx.utils.video._decode_video_frames",
        lambda _url, _max_frames: [first, second],
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {
                    "type": "video_url",
                    "video_url": {"url": "data:video/mp4;base64,AAAA"},
                },
                {"type": "text", "text": "after"},
            ],
        }
    ]

    expanded = expand_video_parts(messages)
    text_messages, images, audio = extract_images_from_messages(expanded)

    assert [part["type"] for part in expanded[0]["content"]] == [
        "text",
        "image",
        "image",
        "text",
    ]
    assert text_messages == [{"role": "user", "content": "before\nafter"}]
    assert [image.getpixel((0, 0)) for image in images] == [
        (255, 0, 0),
        (0, 0, 255),
    ]
    assert audio == []


# ---------------------------------------------------------------------------
# Native video (Qwen3.5 / Qwen3.6 / Qwen3.8)
# ---------------------------------------------------------------------------


def _qwen_processor():
    """An mlx-vlm Qwen3-VL processor with only the image processor populated."""
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    processor = Qwen3VLProcessor.__new__(Qwen3VLProcessor)
    processor.image_processor = SimpleNamespace(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        # Image budget; video must not inherit it.
        min_pixels=65536,
        max_pixels=16777216,
    )
    return processor


def test_write_video_data_uri_returns_file_and_payload_digest():
    payload = b"\x00\x00\x00\x18ftypmp42"
    uri = "data:video/mp4;base64," + base64.b64encode(payload).decode()

    path, digest = write_video_data_uri(uri)
    try:
        assert path.suffix == ".mp4"
        assert path.read_bytes() == payload
        assert digest == hashlib.sha256(payload).hexdigest()
    finally:
        path.unlink()


def test_write_video_data_uri_rejects_remote_urls():
    with pytest.raises(InvalidRequestError, match="base64 data URIs"):
        write_video_data_uri("https://example.com/clip.mp4")


@pytest.mark.parametrize("size", [1, 2, 3, 5, 6, 7, 47, 48, 100])
def test_write_video_data_uri_decodes_across_chunks(monkeypatch, size):
    # Eight base64 characters per chunk, so every size above crosses chunk
    # boundaries and ends with each possible padding length.
    monkeypatch.setattr(video_module, "_BASE64_CHUNK_CHARS", 8)
    payload = bytes((i * 37) % 256 for i in range(size))
    uri = " data:video/mp4;base64," + base64.b64encode(payload).decode() + "\n"

    path, digest = write_video_data_uri(uri)
    try:
        assert path.read_bytes() == payload
        assert digest == hashlib.sha256(payload).hexdigest()
    finally:
        path.unlink()


@pytest.mark.parametrize(
    "encoded",
    [
        "AA==AAAA",  # padding before the end of the payload
        "AAAAAAAAA!AA",  # invalid character in a later chunk
        "AAAAAAAAA",  # truncated final group
    ],
)
def test_write_video_data_uri_rejects_invalid_base64_and_removes_file(
    monkeypatch, tmp_path, encoded
):
    monkeypatch.setattr(video_module, "_BASE64_CHUNK_CHARS", 4)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    with pytest.raises(InvalidRequestError, match="invalid base64"):
        write_video_data_uri("data:video/mp4;base64," + encoded)
    assert list(tmp_path.iterdir()) == []


def test_write_video_data_uri_rejects_empty_payload():
    with pytest.raises(InvalidRequestError, match="empty"):
        write_video_data_uri("data:video/mp4;base64,")


@pytest.mark.parametrize("cli_override", [False, True])
def test_video_size_limit_follows_settings(monkeypatch, tmp_path, cli_override):
    from argparse import Namespace

    from omlx import settings as settings_module

    monkeypatch.setattr(settings_module, "_global_settings", None)
    monkeypatch.setenv("OMLX_MAX_VIDEO_UPLOAD_SIZE", "1KB")
    settings_module.init_settings(
        base_path=tmp_path,
        cli_args=Namespace(max_video_upload_size="2KB") if cli_override else None,
    )
    limit = (2 if cli_override else 1) * 1024
    assert get_max_video_bytes() == limit

    fits = b"\0" * (limit - limit % 3)
    path, _ = write_video_data_uri(
        "data:video/mp4;base64," + base64.b64encode(fits).decode()
    )
    path.unlink()
    too_big = base64.b64encode(b"\0" * (limit + 3)).decode()
    with pytest.raises(InvalidRequestError, match=f"exceeds {limit} bytes"):
        write_video_data_uri("data:video/mp4;base64," + too_big)


def test_video_size_limit_defaults_without_settings(monkeypatch):
    from omlx import settings as settings_module

    monkeypatch.setattr(settings_module, "_global_settings", None)
    assert get_max_video_bytes() == 200 * 1024 * 1024


class _FakeCapture:
    def __init__(self, opened, frames=0, fps=0.0, width=0, height=0):
        self._opened = opened
        self._props = {7: frames, 5: fps, 3: width, 4: height}
        self.released = False

    def isOpened(self):  # noqa: N802
        return self._opened

    def get(self, prop):
        return self._props[prop]

    def release(self):
        self.released = True


def _fake_cv2(capture):
    return SimpleNamespace(
        VideoCapture=lambda _path: capture,
        CAP_PROP_FRAME_COUNT=7,
        CAP_PROP_FPS=5,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
    )


def test_probe_video_reads_container_metadata(monkeypatch, tmp_path):
    capture = _FakeCapture(True, frames=49, fps=24.0, width=768, height=576)
    monkeypatch.setattr("omlx.utils.video.require_opencv", lambda: _fake_cv2(capture))

    assert probe_video(tmp_path / "clip.mp4") == VideoInfo(49, 24.0, 768, 576)
    assert capture.released


@pytest.mark.parametrize(
    "capture",
    [
        _FakeCapture(False),
        _FakeCapture(True, frames=1, fps=24.0, width=768, height=576),
        _FakeCapture(True, frames=0, fps=0.0, width=0, height=0),
    ],
)
def test_probe_video_rejects_clips_that_cannot_be_sampled(
    monkeypatch, tmp_path, capture
):
    monkeypatch.setattr("omlx.utils.video.require_opencv", lambda: _fake_cv2(capture))

    with pytest.raises(InvalidRequestError):
        probe_video(tmp_path / "clip.mp4")
    assert capture.released


def test_attach_uses_official_video_budget_and_caps_frames():
    processor = _qwen_processor()

    assert attach_native_video_processor(processor)

    video = processor.video_processor
    assert (video.min_pixels, video.max_pixels) == (4096, 25165824)
    assert video.max_frames == NATIVE_VIDEO_MAX_FRAMES
    assert (video.patch_size, video.temporal_patch_size, video.merge_size) == (
        16,
        2,
        2,
    )


def test_attach_prefers_checkpoint_video_config(tmp_path):
    (tmp_path / "video_preprocessor_config.json").write_text(
        json.dumps(
            {
                "size": {"shortest_edge": 8192, "longest_edge": 1000000},
                "fps": 1,
                "max_frames": 4096,
            }
        )
    )
    processor = _qwen_processor()

    assert attach_native_video_processor(processor, tmp_path)

    video = processor.video_processor
    assert (video.min_pixels, video.max_pixels, video.fps) == (8192, 1000000, 1)
    assert video.max_frames == NATIVE_VIDEO_MAX_FRAMES


def test_attach_ignores_processors_without_qwen_video_placeholders():
    processor = SimpleNamespace(image_processor=SimpleNamespace(patch_size=16))

    assert not attach_native_video_processor(processor)
    assert not hasattr(processor, "video_processor")


@pytest.mark.parametrize(
    "info,frames",
    [
        # 2 s at 24 fps, sampled at 2 fps: 4 frames, under the pixel budget.
        (VideoInfo(49, 24.0, 768, 576), 4),
        # 20 s at 10 fps: 40 frames, resized to fit the pixel budget.
        (VideoInfo(200, 10.0, 320, 240), 40),
    ],
)
def test_native_video_token_count_matches_the_processor(tmp_path, info, frames):
    (tmp_path / "video_preprocessor_config.json").write_text(
        json.dumps({"size": {"shortest_edge": 4096, "longest_edge": 1000000}})
    )
    processor = _qwen_processor()
    attach_native_video_processor(processor, tmp_path)
    video = processor.video_processor

    clip = np.zeros((frames, 3, info.height, info.width), dtype=np.uint8)
    grid_t, grid_h, grid_w = video(videos=[clip])["video_grid_thw"][0].tolist()
    vision_tokens = grid_t * grid_h * grid_w // video.merge_size**2

    count = native_video_token_count(info, video)

    # Vision tokens exactly, plus the marker allowance of each temporal patch.
    assert count == vision_tokens + 12 * grid_t


def test_native_video_token_count_rejects_degenerate_frames():
    processor = _qwen_processor()
    attach_native_video_processor(processor)

    with pytest.raises(InvalidRequestError, match="Unsupported video dimensions"):
        native_video_token_count(
            VideoInfo(48, 24.0, 8000, 16), processor.video_processor
        )
