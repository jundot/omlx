# SPDX-License-Identifier: Apache-2.0
"""Video inputs: bounded frame sampling for MiMo, native clips for Qwen3-VL-family models."""

import base64
import binascii
import hashlib
import json
import logging
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

from ..exceptions import InvalidRequestError
from ..settings import get_settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_VIDEO_BYTES = 200 * 1024 * 1024
DEFAULT_MAX_VIDEO_FRAMES = 16
# Base64 characters decoded per step (a multiple of 4). An inline clip is
# written to disk chunk by chunk, so no decoded copy of it is held in memory.
_BASE64_CHUNK_CHARS = 4 * 1024 * 1024

# Qwen3.5/3.6/3.8 checkpoints (dense and MoE) carry the Qwen3-VL video tower:
# temporal patches of two frames and 3D mRoPE with a time axis. mlx-vlm's
# torch-free Qwen3-VL video processor turns a clip into pixel_values_videos +
# video_grid_thw and renders the per-patch timestamps into the prompt.
NATIVE_VIDEO_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})
# Upper bound on sampled frames per clip (default sampling is 2 fps). Frames are
# decoded at source resolution before resizing, so this bounds host memory
# (~400 MB at 1080p); the prompt-token budget is set by max_pixels, not by it.
NATIVE_VIDEO_MAX_FRAMES = 64
# size.shortest_edge / size.longest_edge from the video_preprocessor_config.json
# that ships with Qwen3-VL, Qwen3.5, Qwen3.6 and Qwen3.8. The budget covers all
# sampled frames of a clip, so it caps a clip at 12,288 vision tokens.
_QWEN3_VL_VIDEO_PIXELS = {"min_pixels": 4096, "max_pixels": 25165824}
_VIDEO_PROCESSOR_KEYS = (
    "patch_size",
    "temporal_patch_size",
    "merge_size",
    "image_mean",
    "image_std",
    "fps",
    "min_frames",
    "max_frames",
    "min_pixels",
    "max_pixels",
)
_VIDEO_INPUT_ERROR = (
    "Video inputs must be base64 data URIs (data:video/...;base64,...). "
    "Remote URLs and local file paths are not supported."
)


def _video_url(part: Any) -> str | None:
    value = (
        part.get("video_url", part.get("input_video"))
        if isinstance(part, dict)
        else getattr(part, "video_url", None)
    )
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("url")
    if value is not None:
        return getattr(value, "url", None)
    return None


def get_max_video_bytes() -> int:
    """Return the resolved inline video limit in bytes."""
    try:
        settings = get_settings()
    except RuntimeError:
        return DEFAULT_MAX_VIDEO_BYTES
    return settings.server.max_video_upload_bytes()


def _locate_video_payload(value: str) -> tuple[int, int, str]:
    """Find the base64 payload of a video data URI without copying it.

    Returns the payload's start and end offsets in ``value`` and the file
    suffix for its media type.
    """
    if not isinstance(value, str):
        raise InvalidRequestError(_VIDEO_INPUT_ERROR, field="messages")
    start, end = 0, len(value)
    while start < end and value[start].isspace():
        start += 1
    while end > start and value[end - 1].isspace():
        end -= 1
    if not value.startswith("data:", start, end):
        raise InvalidRequestError(_VIDEO_INPUT_ERROR, field="messages")

    comma = value.find(",", start, end)
    prefix_lower = value[start:comma].lower() if comma >= 0 else ""
    if (
        comma < 0
        or not prefix_lower.startswith("data:video/")
        or ";base64" not in prefix_lower
    ):
        raise InvalidRequestError(
            "video_url must use a base64 video data URI.", field="messages"
        )

    media_type = prefix_lower.split(";", 1)[0].split("/", 1)[-1]
    suffix = {"quicktime": ".mov", "x-matroska": ".mkv"}.get(
        media_type, f".{media_type}"
    )
    return comma + 1, end, suffix


def _sample_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0 or max_frames <= 0:
        return []
    count = min(frame_count, max_frames)
    if count == 1:
        return [0]
    return [round(i * (frame_count - 1) / (count - 1)) for i in range(count)]


def _decode_video_frames(value: str, max_frames: int) -> list[Image.Image]:
    try:
        import cv2
    except ImportError as exc:
        raise InvalidRequestError(
            "Video input requires OpenCV (opencv-python-headless).",
            field="messages",
        ) from exc

    path, _ = write_video_data_uri(value)
    capture = None
    try:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise InvalidRequestError("Could not decode video input.", field="messages")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = _sample_indices(frame_count, max_frames)
        if not indices:
            raise InvalidRequestError(
                "Video contains no decodable frames.", field="messages"
            )

        frames: list[Image.Image] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok:
                frames.append(
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
        if not frames:
            raise InvalidRequestError(
                "Video contains no decodable frames.", field="messages"
            )
        return frames
    finally:
        if capture is not None:
            capture.release()
        path.unlink(missing_ok=True)


def expand_video_parts(
    messages: list[dict[str, Any]],
    *,
    max_frames: int = DEFAULT_MAX_VIDEO_FRAMES,
) -> list[dict[str, Any]]:
    """Replace each video content part with sampled in-memory image frames."""
    expanded: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            expanded.append(message)
            continue

        new_content: list[Any] = []
        changed = False
        for part in content:
            part_type = (
                part.get("type")
                if isinstance(part, dict)
                else getattr(part, "type", None)
            )
            if part_type not in ("video", "video_url", "input_video"):
                new_content.append(part)
                continue

            url = _video_url(part)
            if not url:
                raise InvalidRequestError(
                    "Video content part is missing video_url.", field="messages"
                )
            new_content.extend(
                {"type": "image", "image": frame}
                for frame in _decode_video_frames(url, max_frames)
            )
            changed = True

        if changed:
            updated = dict(message)
            updated["content"] = new_content
            expanded.append(updated)
        else:
            expanded.append(message)
    return expanded


def require_opencv():
    """Import OpenCV for frame decoding, or reject the request clearly."""
    try:
        import cv2
    except ImportError as exc:
        raise InvalidRequestError(
            "Video input requires OpenCV (opencv-python-headless).",
            field="messages",
        ) from exc
    return cv2


def write_video_data_uri(value: str) -> tuple[Path, str]:
    """Validate an inline video and write it to a temporary file.

    The payload is checked against ``max_video_upload_size`` before anything is
    decoded, then decoded in chunks straight to the file, so memory use does not
    grow with the clip. Returns the file path and the SHA-256 of the payload,
    which identifies the clip in the prefix cache. The caller deletes the file.
    """
    start, end, suffix = _locate_video_payload(value)
    limit = get_max_video_bytes()
    if (end - start) * 3 // 4 > limit:
        raise InvalidRequestError(
            f"Video payload exceeds {limit} bytes.", field="messages"
        )
    if start == end:
        raise InvalidRequestError("Video payload is empty.", field="messages")

    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        path = Path(handle.name)
        try:
            offset = start
            while offset < end:
                stop = offset + _BASE64_CHUNK_CHARS
                if end - stop < 4:
                    # The tail, with any padding, goes into the last chunk.
                    stop = end
                elif value[stop - 1] == "=":
                    raise binascii.Error("padding before the end of the payload")
                chunk = base64.b64decode(value[offset:stop], validate=True)
                handle.write(chunk)
                digest.update(chunk)
                offset = stop
        except (binascii.Error, ValueError) as exc:
            path.unlink(missing_ok=True)
            raise InvalidRequestError(
                "video_url contains invalid base64 data.", field="messages"
            ) from exc
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    return path, digest.hexdigest()


@dataclass(frozen=True)
class VideoInfo:
    """Container metadata of a clip, read without decoding any frame."""

    frame_count: int
    fps: float
    width: int
    height: int


def probe_video(path: Path) -> VideoInfo:
    """Read a clip's metadata, rejecting files mlx-vlm could not sample."""
    cv2 = require_opencv()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise InvalidRequestError("Could not decode video input.", field="messages")
        info = VideoInfo(
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()
    # Frames are sampled in pairs (one temporal patch), so a clip needs two.
    if info.frame_count < 2 or info.width <= 0 or info.height <= 0:
        raise InvalidRequestError(
            "Video must contain at least two decodable frames.", field="messages"
        )
    return info


def _read_video_preprocessor_config(model_path: str | Path | None) -> dict[str, Any]:
    if model_path is None:
        return {}
    path = Path(model_path) / "video_preprocessor_config.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Ignoring unreadable %s", path)
        return {}
    if not isinstance(raw, dict):
        return {}
    options = {key: raw[key] for key in _VIDEO_PROCESSOR_KEYS if key in raw}
    size = raw.get("size")
    if isinstance(size, dict):
        if size.get("shortest_edge") is not None:
            options.setdefault("min_pixels", size["shortest_edge"])
        if size.get("longest_edge") is not None:
            options.setdefault("max_pixels", size["longest_edge"])
    return options


def attach_native_video_processor(
    processor: Any, model_path: str | Path | None = None
) -> bool:
    """Give a Qwen3-VL-family processor a torch-free video processor.

    oMLX drops ``video_processor`` while loading processors because the
    transformers implementations require torchvision, and quantized
    checkpoints often ship without ``video_preprocessor_config.json``. The
    processor is rebuilt from mlx-vlm's numpy port: patch, merge and
    normalization come from the checkpoint's image processor, the pixel budget
    from the checkpoint's video config or else the official Qwen3-VL one.
    Returns True when the processor can take video input afterwards.
    """
    try:
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
            Qwen3VLProcessor,
            Qwen3VLVideoProcessor,
        )
    except ImportError:
        logger.warning("mlx-vlm has no Qwen3-VL video processor; video input disabled")
        return False
    # Only mlx-vlm's processor renders timestamped video placeholders without
    # transformers' torch-backed video metadata.
    if not isinstance(processor, Qwen3VLProcessor):
        return False
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return False

    options: dict[str, Any] = {
        name: getattr(image_processor, name)
        for name in (
            "patch_size",
            "temporal_patch_size",
            "merge_size",
            "image_mean",
            "image_std",
        )
        if getattr(image_processor, name, None) is not None
    }
    options.update(_QWEN3_VL_VIDEO_PIXELS)
    options.update(_read_video_preprocessor_config(model_path))
    options["max_frames"] = min(
        int(options.get("max_frames", NATIVE_VIDEO_MAX_FRAMES)),
        NATIVE_VIDEO_MAX_FRAMES,
    )
    processor.video_processor = Qwen3VLVideoProcessor(**options)
    return True


def native_video_token_count(info: VideoInfo, video_processor: Any) -> int:
    """Count the prompt tokens a clip expands to under mlx-vlm's sampling.

    Mirrors ``mlx_vlm.utils.load_video`` (frame count from fps, bounded by
    min/max frames, in pairs) and the processor's resize, so the result equals
    what preprocessing produces. Each temporal patch also carries a
    ``<t.t seconds>`` marker and vision start/end tokens, bounded here by 12.
    """
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import _smart_resize_video
    from mlx_vlm.utils import resolve_video_sampling

    sampling = resolve_video_sampling(
        SimpleNamespace(video_processor=video_processor), {}
    )
    step = sampling.frame_factor
    low = math.ceil(sampling.min_frames / step) * step
    high = math.floor(min(sampling.max_frames, info.frame_count) / step) * step
    fps = info.fps if info.fps > 0 else 1.0
    frames = min(
        max(info.frame_count / fps * sampling.fps, low), high, info.frame_count
    )
    frames = math.floor(frames / step) * step

    patch = video_processor.patch_size
    merge = video_processor.merge_size
    temporal = video_processor.temporal_patch_size
    try:
        height, width = _smart_resize_video(
            num_frames=frames,
            height=info.height,
            width=info.width,
            temporal_factor=temporal,
            factor=patch * merge,
            min_pixels=video_processor.min_pixels,
            max_pixels=video_processor.max_pixels,
        )
    except ValueError as exc:
        raise InvalidRequestError(
            f"Unsupported video dimensions: {exc}", field="messages"
        ) from exc
    groups = math.ceil(frames / temporal)
    per_group = (height // patch) * (width // patch) // (merge * merge)
    return groups * (per_group + 12)


def estimate_native_video_tokens(value: str, video_processor: Any) -> int:
    """Prompt tokens of an inline clip, for the prefill preflight.

    Reads only container metadata; no frame is decoded.
    """
    path, _ = write_video_data_uri(value)
    try:
        info = probe_video(path)
    finally:
        path.unlink(missing_ok=True)
    return native_video_token_count(info, video_processor)
