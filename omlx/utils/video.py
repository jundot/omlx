# SPDX-License-Identifier: Apache-2.0
"""Bounded video-frame sampling for MiMo's vision encoder."""

import base64
import binascii
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from ..exceptions import InvalidRequestError

DEFAULT_MAX_VIDEO_BYTES = 200 * 1024 * 1024
DEFAULT_MAX_VIDEO_FRAMES = 16
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


def _decode_video_data_uri(value: str) -> tuple[bytes, str]:
    if not isinstance(value, str) or not value.strip().startswith("data:"):
        raise InvalidRequestError(_VIDEO_INPUT_ERROR, field="messages")

    prefix, separator, encoded = value.strip().partition(",")
    prefix_lower = prefix.lower()
    if (
        separator != ","
        or not prefix_lower.startswith("data:video/")
        or ";base64" not in prefix_lower
    ):
        raise InvalidRequestError(
            "video_url must use a base64 video data URI.", field="messages"
        )

    estimated_size = len(encoded) * 3 // 4
    if estimated_size > DEFAULT_MAX_VIDEO_BYTES:
        raise InvalidRequestError(
            f"Video payload exceeds {DEFAULT_MAX_VIDEO_BYTES} bytes.",
            field="messages",
        )
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError(
            "video_url contains invalid base64 data.", field="messages"
        ) from exc
    if not payload:
        raise InvalidRequestError("Video payload is empty.", field="messages")

    media_type = prefix.split(";", 1)[0].split("/", 1)[-1].lower()
    suffix = {"quicktime": ".mov", "x-matroska": ".mkv"}.get(
        media_type, f".{media_type}"
    )
    return payload, suffix


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

    payload, suffix = _decode_video_data_uri(value)
    path: Path | None = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(payload)
            path = Path(handle.name)

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
        if path is not None:
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
