# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 Vision image-token layout and rank-zero preprocessing.

The arithmetic is derived from the public checkpoint's reference
``inference/image_processor.py``.  It intentionally has no MLX dependency so
the layout contract can be tested on Linux CI.
"""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageOps

from omlx.deepseek_v4_vision import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEWLINE,
    IMAGE_PAD,
    IMAGE_START,
)

COMPRESS_PAD_TO = 4
_MAX_IMAGE_BYTES = 50 * 1024 * 1024


@dataclass(frozen=True)
class ImageInput:
    start: int
    patches: np.ndarray
    n_vit_h: int
    n_vit_w: int
    types: tuple[int, ...]
    permutation: tuple[int, ...]


def grid_tokens(
    height: int, width: int, patch_size: int, downsample_ratio: int
) -> tuple[int, int, int]:
    n_llm_h = math.ceil((height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((width // patch_size) / downsample_ratio)
    count = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2:
        count += n_llm_w + 1
    count += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, count


def _solve_resize_ratio(height, width, patch, ratio, budget):
    aspect = height / width
    max_w_float = math.sqrt((budget - 2) / aspect + 0.25) - 0.5
    max_h_float = max_w_float * aspect
    if max_w_float < 1.0:
        max_w = 1
        max_h = (budget - 2) // (max_w + 1)
        max_h -= max_h % 2
        best_w, best_h = max_w * patch * ratio, max_h * patch * ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = (budget - 2) // max_h - 1
        if max_w <= 1:
            raise ValueError("image aspect ratio cannot fit the token budget")
        best_w, best_h = max_w * patch * ratio, max_h * patch * ratio
    else:
        max_w, max_h = math.floor(max_w_float), math.floor(max_h_float)
        max_h -= max_h % 2
        scale = min(max_w * patch * ratio / width, max_h * patch * ratio / height)
        best_w = math.floor(width * scale / patch) * patch
        best_h = math.floor(height * scale / patch) * patch
    n_h, n_w, count = grid_tokens(best_h, best_w, patch, ratio)
    return n_h, n_w, best_h, best_w, count


def safe_resize(height, width, best_h, best_w, patch, ratio, max_tokens):
    limit = max_tokens - (COMPRESS_PAD_TO - 1)
    n_h, n_w, count = grid_tokens(best_h, best_w, patch, ratio)
    budget = limit
    while count > limit:
        n_h, n_w, best_h, best_w, count = _solve_resize_ratio(
            height, width, patch, ratio, budget
        )
        budget -= 1
    return n_h, n_w, best_h, best_w


def build_image_block(
    n_llm_h: int, n_llm_w: int, start_pos: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return checkpoint sentinel types and aligner-to-N-layout permutation."""

    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = (rows // 2 * row_len % 2) * 2

    types = ([IMAGE] * n_llm_w + [IMAGE_NEWLINE]) * n_llm_h
    types += [IMAGE_PAD] * (row_len * pad_h)
    order = np.arange(rows * row_len).reshape(rows // 2, 2, row_len)
    order = order.transpose(0, 2, 1).reshape(-1)
    image_index = np.full(rows * row_len, -1, dtype=np.int64)
    image_index.reshape(rows, row_len)[:n_llm_h, :n_llm_w] = np.arange(
        n_llm_h * n_llm_w
    ).reshape(n_llm_h, n_llm_w)
    permutation = image_index[order]
    permutation = tuple(int(v) for v in permutation if v >= 0)
    ordered_types = [types[int(index)] for index in order]
    block = (
        [IMAGE_PAD] * compress_pad
        + [IMAGE_START]
        + ordered_types
        + [IMAGE_PAD] * pad_last
        + [IMAGE_END]
    )
    return tuple(block), permutation


def _image_bytes(record: Any) -> bytes:
    if isinstance(record, str):
        record = {"url": record}
    if not isinstance(record, Mapping):
        raise ValueError("image input must be a URL/data record")
    raw = record.get("data")
    if isinstance(raw, bytes):
        payload = raw
    elif isinstance(raw, str):
        payload = base64.b64decode(raw, validate=True)
    else:
        source = record.get("source")
        if isinstance(source, Mapping):
            if source.get("data") is not None:
                payload = base64.b64decode(str(source["data"]), validate=True)
            else:
                record = {"url": source.get("url")}
                payload = b""
        else:
            payload = b""
        if not payload:
            url = record.get("url") or record.get("image_url")
            if isinstance(url, Mapping):
                url = url.get("url")
            if not isinstance(url, str) or not url:
                raise ValueError("image record has no data or URL")
            if url.startswith("data:"):
                header, separator, encoded = url.partition(",")
                if not separator or ";base64" not in header:
                    raise ValueError("only base64 image data URLs are supported")
                payload = base64.b64decode(encoded, validate=True)
            elif url.startswith(("https://", "http://")):
                with urlopen(url, timeout=30) as response:  # noqa: S310
                    payload = response.read(_MAX_IMAGE_BYTES + 1)
            else:
                raise ValueError("image URL must use http(s) or a base64 data URL")
    if not payload or len(payload) > _MAX_IMAGE_BYTES:
        raise ValueError("image payload is empty or exceeds 50 MiB")
    return payload


def load_image(record: Any, config: Mapping[str, Any]):
    patch = int(config["vision_patch_size"])
    with Image.open(io.BytesIO(_image_bytes(record))) as source:
        image = source.convert("RGB")
    width, height = image.size
    max_ratio = int(config.get("vision_max_wh_ratio", 8) or 8)
    if width > height * max_ratio:
        width = height * max_ratio
    min_pixels = int(config.get("vision_min_pixels", 147456) or 147456)
    if 0 < width * height < min_pixels:
        scale = math.sqrt(min_pixels / (width * height))
        width, height = int(width * scale), int(height * scale)
    best_w = math.ceil(width / patch) * patch
    best_h = math.ceil(height / patch) * patch
    downsample = int(config["vision_downsample_ratio"])
    n_llm_h, n_llm_w, best_h, best_w = safe_resize(
        height,
        width,
        best_h,
        best_w,
        patch,
        downsample,
        int(config["vision_max_n_token"]),
    )
    n_vit_h, n_vit_w = best_h // patch, best_w // patch
    if image.width >= max_ratio * image.height:
        image = image.resize((best_w, best_h))
    else:
        image = ImageOps.pad(image, (best_w, best_h), color=(127, 127, 127))
    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    pixels = (pixels - 0.5) / 0.5
    patches = pixels.reshape(3, n_vit_h, patch, n_vit_w, patch)
    patches = patches.transpose(1, 3, 0, 2, 4).reshape(
        n_vit_h * n_vit_w, 3, patch, patch
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def prepare_token_ids(
    prompt_tokens: Sequence[int],
    images: Sequence[Any],
    *,
    image_token_id: int,
    config: Mapping[str, Any],
) -> tuple[list[int], tuple[ImageInput, ...]]:
    """Expand placeholders and preprocess images in deterministic prompt order."""

    placeholders = sum(int(token) == image_token_id for token in prompt_tokens)
    if placeholders != len(images):
        raise ValueError(
            f"found {placeholders} DeepSeek image placeholders but received "
            f"{len(images)} images"
        )
    vocab_size = int(config["vocab_size"])
    tokens: list[int] = []
    prepared: list[ImageInput] = []
    image_iterator = iter(images)
    for token in prompt_tokens:
        if int(token) != image_token_id:
            tokens.append(int(token))
            continue
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image(
            next(image_iterator), config
        )
        types, permutation = build_image_block(n_llm_h, n_llm_w, len(tokens))
        prepared.append(
            ImageInput(
                start=len(tokens),
                patches=patches,
                n_vit_h=n_vit_h,
                n_vit_w=n_vit_w,
                types=types,
                permutation=permutation,
            )
        )
        tokens.extend(vocab_size + kind for kind in types)
    return tokens, tuple(prepared)
