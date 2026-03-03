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

from PIL import Image

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
        return Image.open(io.BytesIO(img_bytes))
    elif url_or_base64.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(url_or_base64, timeout=30) as response:
            img_bytes = response.read()
        return Image.open(io.BytesIO(img_bytes))
    else:
        # Try as local file path
        return Image.open(url_or_base64)


def extract_images_from_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """
    Extract images from OpenAI-format messages.

    Processes messages containing content arrays with image_url parts,
    loads the images, and returns cleaned text-only messages alongside
    the loaded images.

    Args:
        messages: List of OpenAI-format chat messages. Each message may have
            content as a string or a list of content parts (text/image_url).

    Returns:
        Tuple of (text_messages, images):
        - text_messages: Messages with image parts removed, text parts joined
        - images: List of loaded PIL Image objects in order of appearance
    """
    text_messages = []
    images = []

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

        # Content array with text and/or image_url parts
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

            elif part_type == "image_url":
                image_url_obj = (
                    part.get("image_url") if isinstance(part, dict)
                    else getattr(part, "image_url", None)
                )
                if image_url_obj:
                    url = (
                        image_url_obj.get("url") if isinstance(image_url_obj, dict)
                        else getattr(image_url_obj, "url", None)
                    )
                    if url:
                        try:
                            img = load_image(url)
                            images.append(img)
                        except Exception as e:
                            logger.warning(f"Failed to load image: {e}")

        new_msg = {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
        # Preserve extra fields
        for key in msg:
            if key not in ("role", "content"):
                new_msg[key] = msg[key]
        text_messages.append(new_msg)

    return text_messages, images


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
