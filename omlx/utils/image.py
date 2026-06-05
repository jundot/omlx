# SPDX-License-Identifier: Apache-2.0
"""
Image processing utilities for VLM (Vision-Language Model) support.

This module provides functions for loading images from URLs/base64,
extracting images from OpenAI-format messages, and computing image
hashes for prefix cache deduplication.
"""

import base64
import hashlib
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def load_image(url_or_base64: str) -> Image.Image:
    """
    Load an image from a URL or base64 data URI.

    Supports:
    - HTTP/HTTPS URLs: Downloads the image
    - Data URIs: "data:image/jpeg;base64,..." format

    Args:
        url_or_base64: Image URL or base64 data URI string

    Returns:
        PIL Image object

    Raises:
        ValueError: If the URL format is unsupported
        IOError: If the image cannot be loaded
    """
    if url_or_base64.startswith("data:"):
        # base64 data URI: "data:image/jpeg;base64,<data>"
        try:
            _, data_part = url_or_base64.split(",", 1)
        except ValueError:
            raise ValueError(f"Invalid data URI format: {url_or_base64[:50]}...")
        img_bytes = base64.b64decode(data_part)
        img = Image.open(io.BytesIO(img_bytes))
    elif url_or_base64.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(url_or_base64, timeout=30) as response:
            img_bytes = response.read()
        img = Image.open(io.BytesIO(img_bytes))
    else:
        # Try as local file path
        img = Image.open(url_or_base64)

    # Apply EXIF orientation (phone photos etc.) before processing.
    # Matches mlx-vlm's load_image which calls ImageOps.exif_transpose().
    img = ImageOps.exif_transpose(img)
    # Ensure RGB format (RGBA/P/L etc. cause broadcast errors in vision processors)
    return img.convert("RGB")


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
        - audio: List of BytesIO/str audio references for load_audio()
    """
    import binascii

    text_messages = []
    images = []
    audio = []

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
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if text:
                    text_parts.append(text)

            elif part_type in ("image_url", "input_image"):
                # OpenAI chat format: {"type":"image_url","image_url":{"url":"..."}}
                # Responses-style format: {"type":"input_image","image_url":"..."}
                image_url_obj = (
                    part.get("image_url") if isinstance(part, dict)
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
                    try:
                        img = load_image(url)
                        images.append(img)
                    except Exception as e:
                        logger.warning(f"Failed to load image: {e}")

            elif part_type == "input_audio":
                # OpenAI audio format: {"type":"input_audio","input_audio":{"data":"...","format":"wav"}}
                input_audio = (
                    part.get("input_audio") if isinstance(part, dict)
                    else getattr(part, "input_audio", None)
                )
                if input_audio and isinstance(input_audio, dict):
                    data = input_audio.get("data", "")
                    if isinstance(data, str):
                        stripped = data.strip()
                        # Handle data URI (data:audio/wav;base64,...)
                        if stripped.startswith("data:"):
                            prefix, separator, encoded = stripped.partition(",")
                            if separator == "," and ";base64" in prefix:
                                try:
                                    audio.append(io.BytesIO(base64.b64decode(encoded, validate=True)))
                                except (binascii.Error, ValueError) as exc:
                                    logger.warning(f"Failed to decode input_audio base64: {exc}")
                                continue
                        # Try raw base64 decode
                        try:
                            audio.append(io.BytesIO(base64.b64decode(stripped, validate=True)))
                        except (binascii.Error, ValueError):
                            # Not base64 — treat as file path/reference
                            audio.append(stripped)
                    elif isinstance(data, bytes):
                        audio.append(io.BytesIO(data))
                    else:
                        audio.append(data)

        new_msg = {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
        # Preserve extra fields
        for key in msg:
            if key not in ("role", "content"):
                new_msg[key] = msg[key]
        text_messages.append(new_msg)

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
