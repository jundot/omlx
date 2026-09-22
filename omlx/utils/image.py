# SPDX-License-Identifier: Apache-2.0
"""
Image processing utilities for VLM (Vision-Language Model) support.

This module provides functions for loading images from base64 data URIs,
extracting images from OpenAI-format messages, and computing image
hashes for prefix cache deduplication.
"""

import base64
import binascii
import hashlib
import io
import math
import struct
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

from ..exceptions import InvalidRequestError
from ..settings import get_settings

DEFAULT_MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50 MiB
DEFAULT_MAX_IMAGE_SIDE_LENGTH = 2048  # 2048 px


def get_max_image_bytes() -> int:
    """Return the resolved image payload limit in bytes."""
    try:
        settings = get_settings()
    except RuntimeError:
        return DEFAULT_MAX_IMAGE_BYTES
    return settings.server.max_image_upload_bytes()


def get_max_image_side_length() -> int:
    """Return the resolved image side limit (0 disables resizing)."""
    try:
        settings = get_settings()
    except RuntimeError:
        return DEFAULT_MAX_IMAGE_SIDE_LENGTH
    return settings.server.max_image_side_length


_IMAGE_INPUT_ERROR = (
    "Image inputs must be base64 data URIs "
    "(data:image/...;base64,...). Remote URLs and local file paths are not supported."
)
_AUDIO_INPUT_ERROR = (
    "input_audio.data must be a base64 string or base64 data URI. "
    "Local file paths are not supported."
)


def _decode_base64_data_uri(value: str, *, field: str) -> bytes:
    """Decode a base64 data URI, mapping malformed input to a request error."""
    if not isinstance(value, str):
        raise InvalidRequestError(_IMAGE_INPUT_ERROR, field=field)

    stripped = value.strip()
    if not stripped.startswith("data:"):
        raise InvalidRequestError(_IMAGE_INPUT_ERROR, field=field)

    prefix, separator, encoded = stripped.partition(",")
    prefix_lower = prefix.lower()
    if (
        separator != ","
        or not prefix_lower.startswith("data:image/")
        or ";base64" not in prefix_lower
    ):
        raise InvalidRequestError(
            f"{field} must use a base64 image data URI.",
            field=field,
        )

    # Pre-check base64 encoded length to reject massive inputs before decoding
    max_bytes = get_max_image_bytes()
    if max_bytes > 0:
        max_encoded_len = int(math.ceil(max_bytes * 4 / 3)) + 1024
        if len(encoded) > max_encoded_len:
            raise InvalidRequestError(
                f"{field} image payload exceeds the maximum allowed limit of {max_bytes} bytes.",
                field=field,
            )

    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError(
            f"{field} contains invalid base64 data.",
            field=field,
        ) from exc

    if max_bytes > 0 and len(decoded) > max_bytes:
        raise InvalidRequestError(
            f"{field} image payload ({len(decoded)} bytes) exceeds the maximum allowed limit of {max_bytes} bytes.",
            field=field,
        )

    return decoded


def _decode_input_audio_data(data: str, *, field: str = "input_audio.data") -> bytes:
    """Decode input_audio.data without falling back to filesystem paths."""
    stripped = data.strip()
    if stripped.startswith("data:"):
        prefix, separator, encoded = stripped.partition(",")
        if separator != "," or ";base64" not in prefix.lower():
            raise InvalidRequestError(
                f"{field} must use a base64 data URI.",
                field=field,
            )
    else:
        encoded = stripped

    try:
        return base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidRequestError(_AUDIO_INPUT_ERROR, field=field) from exc


def validate_image_data_uri(value: str, *, field: str = "image") -> str:
    """Validate that a request-facing image reference is an inline data URI."""
    _decode_base64_data_uri(value, field=field)
    return value


# Decoded-image cache.
#
# Multi-turn agent loops resend the same historical screenshots on every turn
# (byte-identical base64 data URIs). Decoding a PNG/JPEG is CPU-bound and,
# once the conversation accumulates many screenshots, re-decoding the whole
# history every turn dominates time-to-first-token. Keying the decoded RGB
# image by the hash of its source bytes lets repeated turns reuse prior
# decodes and only decode genuinely new images. Bounded by total decoded
# pixel bytes so the cache cannot grow without limit.
_IMAGE_DECODE_CACHE_MAX_BYTES = 512 * 1024 * 1024  # 512 MiB

_image_decode_cache: "OrderedDict[str, Image.Image]" = OrderedDict()
_image_decode_cache_bytes = 0
_image_decode_cache_generation = 0
_image_decode_cache_lock = threading.Lock()


def _decoded_pixel_bytes(img: Image.Image) -> int:
    width, height = img.size
    # Pillow stores RGB pixels in four-byte slots, plus one pointer per row.
    return width * height * 4 + height * struct.calcsize("P")


def clear_image_decode_cache() -> int:
    """Drop cache references and return their accounted bytes.

    Active requests retain their images. Decodes started before this clear
    may finish for their callers, but must not repopulate the cache.
    """
    global _image_decode_cache_bytes, _image_decode_cache_generation
    with _image_decode_cache_lock:
        released = _image_decode_cache_bytes
        _image_decode_cache.clear()
        _image_decode_cache_bytes = 0
        _image_decode_cache_generation += 1
        return released


def _cached_image(key: str) -> Image.Image | None:
    with _image_decode_cache_lock:
        hit = _image_decode_cache.get(key)
        if hit is not None:
            _image_decode_cache.move_to_end(key)
        return hit


def load_image(url_or_base64: str, *, field: str = "image_url") -> Image.Image:
    """
    Load an image from a base64 data URI.

    Supports:
    - Data URIs: "data:image/jpeg;base64,..." format

    Decoded images are cached by source-byte hash so that re-sending the
    same image across turns does not re-run the CPU-bound decode.

    Args:
        url_or_base64: Image base64 data URI string

    Returns:
        PIL Image object

    Raises:
        InvalidRequestError: If the input is not a valid image data URI
    """
    img_bytes = _decode_base64_data_uri(url_or_base64, field=field)
    return _load_image_bytes(img_bytes, field=field)


def _load_image_bytes(
    img_bytes: bytes, *, field: str, generation: int | None = None
) -> Image.Image:
    max_bytes = get_max_image_bytes()
    if max_bytes > 0 and len(img_bytes) > max_bytes:
        raise InvalidRequestError(
            f"{field} image payload ({len(img_bytes)} bytes) exceeds the maximum allowed limit of {max_bytes} bytes.",
            field=field,
        )

    key = hashlib.sha256(img_bytes).hexdigest()
    with _image_decode_cache_lock:
        if generation is None:
            generation = _image_decode_cache_generation
    hit = _cached_image(key)
    if hit is not None:
        return hit

    try:
        loaded = Image.open(io.BytesIO(img_bytes))
        # Apply EXIF orientation (phone photos etc.) before processing.
        # Matches mlx-vlm's load_image which calls ImageOps.exif_transpose().
        oriented = ImageOps.exif_transpose(loaded)
        # Ensure RGB format (RGBA/P/L etc. cause broadcast errors in vision processors)
        rgb = oriented.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise InvalidRequestError(
            f"{field} exceeds maximum allowed image resolution (decompression bomb detected).",
            field=field,
        ) from exc
    except Exception as exc:
        raise InvalidRequestError(
            f"{field} does not contain a decodable image.",
            field=field,
        ) from exc

    # Downscale oversized images preserving aspect ratio to prevent memory spikes in VLMs
    max_side = get_max_image_side_length()
    if max_side > 0 and (rgb.width > max_side or rgb.height > max_side):
        resample = getattr(Image, "Resampling", Image).LANCZOS
        rgb.thumbnail((max_side, max_side), resample=resample)

    nbytes = _decoded_pixel_bytes(rgb)
    if nbytes <= _IMAGE_DECODE_CACHE_MAX_BYTES:
        global _image_decode_cache_bytes
        with _image_decode_cache_lock:
            if generation != _image_decode_cache_generation:
                return rgb
            # Another thread may have decoded the same key while we were
            # decoding; reconcile accounting before inserting.
            stale = _image_decode_cache.pop(key, None)
            if stale is not None:
                _image_decode_cache_bytes -= _decoded_pixel_bytes(stale)
            _image_decode_cache[key] = rgb
            _image_decode_cache_bytes += nbytes
            while (
                _image_decode_cache_bytes > _IMAGE_DECODE_CACHE_MAX_BYTES
                and _image_decode_cache
            ):
                _, evicted = _image_decode_cache.popitem(last=False)
                _image_decode_cache_bytes -= _decoded_pixel_bytes(evicted)

    return rgb


def extract_images_from_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Image.Image], List]:
    """
    Extract images and audio from OpenAI-format messages.

    Processes messages containing content arrays with image_url or input_audio
    parts, loads the media, and returns cleaned text-only messages alongside
    the loaded images and audio files.

    Args:
        messages: List of OpenAI-format chat messages. Each message may have
            content as a string or a list of content parts
            (text/image_url/input_audio).

    Returns:
        Tuple of (text_messages, images, audio):
        - text_messages: Messages with media parts removed, text parts joined
        - images: List of loaded PIL Image objects in order of appearance
        - audio: List of BytesIO audio buffers
    """
    text_messages = []
    images = []
    audio = []
    pending_images = []
    with _image_decode_cache_lock:
        generation = _image_decode_cache_generation

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if not isinstance(content, list):
            # Simple string content — pass through
            text_messages.append({"role": role, "content": content or ""})
            # Preserve extra fields (tool_calls, tool_call_id, etc.)
            for key in msg:
                if key not in ("role", "content"):
                    text_messages[-1][key] = msg[key]
            continue

        # Content array with text, image_url, and/or input_audio parts
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")
            else:
                # Pydantic model (ContentPart)
                part_type = getattr(part, "type", None)

            if part_type == "text":
                text = (
                    part.get("text")
                    if isinstance(part, dict)
                    else getattr(part, "text", None)
                )
                if text:
                    text_parts.append(text)

            elif part_type == "image":
                # Internal video expansion passes sampled frames in memory.
                image = (
                    part.get("image")
                    if isinstance(part, dict)
                    else getattr(part, "image", None)
                )
                if isinstance(image, Image.Image):
                    images.append(image.convert("RGB"))

            elif part_type in ("image_url", "input_image"):
                # OpenAI chat format: {"type":"image_url","image_url":{"url":"..."}}
                # Responses-style format: {"type":"input_image","image_url":"..."}
                image_url_obj = (
                    part.get("image_url")
                    if isinstance(part, dict)
                    else getattr(part, "image_url", None)
                )
                if image_url_obj is None and isinstance(part, dict):
                    image_url_obj = part.get("input_image")

                url = None
                if isinstance(image_url_obj, str):
                    url = image_url_obj
                elif isinstance(image_url_obj, dict):
                    url = image_url_obj.get("url")
                elif image_url_obj is not None:
                    url = getattr(image_url_obj, "url", None)

                if url:
                    img_bytes = _decode_base64_data_uri(url, field="image_url")
                    key = hashlib.sha256(img_bytes).hexdigest()
                    hit = _cached_image(key)
                    if hit is None:
                        pending_images.append((len(images), img_bytes))
                    images.append(hit)

            elif part_type == "input_audio":
                # OpenAI audio format: {"type":"input_audio","input_audio":{"data":"...","format":"wav"}}
                input_audio = (
                    part.get("input_audio")
                    if isinstance(part, dict)
                    else getattr(part, "input_audio", None)
                )
                if input_audio and isinstance(input_audio, dict):
                    data = input_audio.get("data", "")
                    if isinstance(data, str):
                        audio.append(io.BytesIO(_decode_input_audio_data(data)))
                    elif isinstance(data, bytes):
                        audio.append(io.BytesIO(data))
                    else:
                        audio.append(data)

            elif part_type in ("video", "video_url", "input_video"):
                raise InvalidRequestError(
                    "Video input is not supported by oMLX.",
                    field="messages",
                )

        new_msg = {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
        # Preserve extra fields
        for key in msg:
            if key not in ("role", "content"):
                new_msg[key] = msg[key]
        text_messages.append(new_msg)

    # Retain every existing hit before inserting misses. A long chronological
    # history can exceed the cache; decoding its oldest misses first would
    # otherwise evict later hits before this request reaches them.
    for index, img_bytes in pending_images:
        images[index] = _load_image_bytes(
            img_bytes, field="image_url", generation=generation
        )

    return text_messages, images, audio


def compute_image_hash(images: List[Image.Image]) -> Optional[str]:
    """
    Compute a SHA256 hash from a list of images for prefix cache deduplication.

    Uses image size and raw pixel data to produce a deterministic hash.
    Returns None if images list is empty.

    Args:
        images: List of PIL Image objects

    Returns:
        Hex-encoded SHA256 hash string, or None if no images
    """
    if not images:
        return None

    hasher = hashlib.sha256()
    for img in images:
        # Include image dimensions
        hasher.update(f"{img.size[0]}x{img.size[1]}".encode())
        # Include raw pixel data (convert to RGB for consistency)
        rgb_img = img.convert("RGB")
        hasher.update(rgb_img.tobytes())

    return hasher.hexdigest()


def compute_per_image_hashes(images: List[Image.Image]) -> List[str]:
    """Compute individual SHA256 hashes for each image.

    Returns a list of hex-encoded hash strings, one per image.
    """
    hashes = []
    for img in images:
        hasher = hashlib.sha256()
        hasher.update(f"{img.size[0]}x{img.size[1]}".encode())
        rgb_img = img.convert("RGB")
        hasher.update(rgb_img.tobytes())
        hashes.append(hasher.hexdigest())
    return hashes
