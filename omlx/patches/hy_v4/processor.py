# SPDX-License-Identifier: Apache-2.0
"""Image processor and processor for Hy4 and Hy4-flash.

The installed transformers has no Hy4 processor class, so the processor is
registered with mlx-vlm's AutoProcessor hook (see ``apply_patch``). An image
becomes ``(h / 2) * (w / 2 + 1)`` placeholder tokens: one per merged 2x2 patch
block, plus one per merged row for the newline vector the projector appends.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.models.base import load_chat_template
from PIL import Image

_IMAGE_TOKEN = "<｜hy_image▁pad:opensource｜>"
_VIDEO_TOKEN = "<｜hy_video▁pad:opensource｜>"
# Largest aspect ratio smart_resize accepts.
_MAX_ASPECT = 200


def image_token_count(height: int, width: int, merge_size: int = 2) -> int:
    """Placeholder tokens of one image grid, newline slots included."""
    return (height // merge_size) * (width // merge_size + 1)


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Pixel size an image is resized to: multiples of ``factor`` inside the
    pixel budget, keeping the aspect ratio as close as the grid allows."""
    if min(height, width, factor, min_pixels) <= 0 or max_pixels < min_pixels:
        raise ValueError("Image dimensions and pixel limits must be positive")
    if max(height, width) / min(height, width) > _MAX_ASPECT:
        raise ValueError(f"Image aspect ratio must not exceed {_MAX_ASPECT}")
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    h, w = max(h, factor), max(w, factor)
    if h * w > max_pixels:
        raise ValueError("Image aspect ratio and patch rounding exceed max_pixels")
    return h, w


def _to_rgb(image) -> Image.Image:
    """RGB image; transparent pixels are composited onto white."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if not isinstance(image, Image.Image):
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            scale = 255.0 if array.max() <= 1.0 else 1.0
            array = np.clip(array * scale, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
    if "A" in image.getbands() or image.info.get("transparency") is not None:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


class HyV4ImageProcessor:
    """Resize, normalize and patchify images into ``[patches, C * P * P]``.

    Patches come in raster order of the resized image's patch grid, flattened
    as (channel, row, column), together with ``image_grid_thw`` rows ``[1, h, w]``.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 16,
        merge_size: int = 2,
        min_pixels: int = 4096,
        max_pixels: int = 4194304,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        rescale_factor: float = 1 / 255,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.rescale_factor = rescale_factor

    @property
    def max_image_tokens(self) -> int:
        """Most placeholder tokens an image can take at the largest allowed
        pixel count and aspect ratio."""
        cells = self.max_pixels // (self.patch_size * self.merge_size) ** 2
        return cells + math.isqrt(cells * _MAX_ASPECT) + 1

    def grid(
        self, height: int, width: int, min_pixels=None, max_pixels=None
    ) -> tuple[int, int]:
        """Patch grid ``(h, w)`` of an image of the given pixel size."""
        h, w = smart_resize(
            height,
            width,
            self.patch_size * self.merge_size,
            min_pixels or self.min_pixels,
            max_pixels or self.max_pixels,
        )
        return h // self.patch_size, w // self.patch_size

    def count_image_tokens(self, height: int, width: int) -> int:
        grid_h, grid_w = self.grid(height, width)
        return image_token_count(grid_h, grid_w, self.merge_size)

    def _process_one(
        self, image, min_pixels=None, max_pixels=None
    ) -> tuple[np.ndarray, list[int]]:
        image = _to_rgb(image)
        grid_h, grid_w = self.grid(image.height, image.width, min_pixels, max_pixels)
        size = (grid_w * self.patch_size, grid_h * self.patch_size)
        image = image.resize(size, resample=Image.Resampling.BICUBIC)
        pixels = np.asarray(image).transpose(2, 0, 1)
        pixels = (pixels.astype(np.float64) * self.rescale_factor).astype(np.float32)
        mean = np.asarray(self.image_mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.image_std, dtype=np.float32)[:, None, None]
        pixels = (pixels - mean) / std
        patch = self.patch_size
        patches = pixels.reshape(3, grid_h, patch, grid_w, patch)
        patches = patches.transpose(1, 3, 0, 2, 4).reshape(grid_h * grid_w, -1)
        return patches, [1, grid_h, grid_w]

    def __call__(self, images, min_pixels=None, max_pixels=None, **kwargs) -> dict:
        if not isinstance(images, (list, tuple)):
            images = [images]
        processed = [self._process_one(i, min_pixels, max_pixels) for i in images]
        return {
            "pixel_values": np.concatenate([p for p, _ in processed], axis=0),
            "image_grid_thw": np.asarray([g for _, g in processed], dtype=np.int64),
        }

    preprocess = __call__


class HyV4Processor:
    """Tokenizes prompts and expands image placeholders to their token count."""

    def __init__(self, image_processor, tokenizer, chat_template=None):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.image_token = getattr(tokenizer, "image_token", None) or _IMAGE_TOKEN
        self.video_token = getattr(tokenizer, "video_token", None) or _VIDEO_TOKEN
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

    @property
    def model_input_names(self):
        return [
            *self.tokenizer.model_input_names,
            *self.image_processor.model_input_names,
            "mm_token_type_ids",
        ]

    def apply_chat_template(self, *args, **kwargs):
        # The template itself renders the image markers; rendering stays the
        # tokenizer's, as for text-only requests.
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def _expand_images(self, texts: list[str], grids) -> list[str]:
        counts = [
            image_token_count(int(h), int(w), self.image_processor.merge_size)
            for _, h, w in grids
        ]
        expanded, used = [], 0
        for text in texts:
            pieces = text.split(self.image_token)
            found = len(pieces) - 1
            if used + found > len(counts):
                raise ValueError(
                    f"The prompt has more image placeholders than the {len(counts)} "
                    "images supplied"
                )
            out = pieces[0]
            for piece, count in zip(pieces[1:], counts[used : used + found]):
                out += self.image_token * count + piece
            used += found
            expanded.append(out)
        if used != len(counts):
            raise ValueError(
                f"{len(counts)} images supplied for {used} image placeholders"
            )
        return expanded

    def __call__(self, images=None, text=None, videos=None, audio=None, **kwargs):
        if videos is not None or audio is not None:
            raise ValueError("Hy4 supports image input only")
        texts = [text] if isinstance(text, str) else list(text)
        # mlx-vlm forwards per-request image sizing to every processor.
        sizing = {key: kwargs.pop(key, None) for key in ("min_pixels", "max_pixels")}
        for key in ("resized_height", "resized_width"):
            if kwargs.pop(key, None) is not None:
                raise ValueError(f"Hy4 sizes images by pixel budget, not {key}")
        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images, **sizing)
            texts = self._expand_images(texts, image_inputs["image_grid_thw"])
        elif any(self.image_token in t for t in texts):
            raise ValueError("The prompt has image placeholders but no images")
        return_tensors = kwargs.pop("return_tensors", None)
        kwargs.setdefault("add_special_tokens", False)
        encoded = dict(self.tokenizer(texts, **kwargs))
        data = {k: np.asarray(v, dtype=np.int32) for k, v in encoded.items()}
        data["mm_token_type_ids"] = (data["input_ids"] == self.image_token_id).astype(
            np.int32
        )
        data.update(image_inputs)
        if return_tensors == "np":
            return data
        return {k: mx.array(v) for k, v in data.items()}

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        from transformers import AutoTokenizer

        for key in ("use_fast", "trust_remote_code", "config"):
            kwargs.pop(key, None)
        tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
        load_chat_template(tokenizer, path)
        # A raw HF tokenizer skips mlx-lm's think marker inference.
        start, end = "<think:opensource>", "</think:opensource>"
        vocab = tokenizer.get_vocab()
        if start in vocab and end in vocab:
            tokenizer.think_start = start
            tokenizer.think_end = end
            tokenizer.think_start_id = vocab[start]
            tokenizer.think_end_id = vocab[end]
        config_path = Path(path) / "processor_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        image_config = config.get("image_processor", config)
        size = image_config.get("size") or {}
        options = {
            name: image_config[name]
            for name in ("patch_size", "merge_size", "image_mean", "image_std")
            if name in image_config
        }
        # ``size`` holds the pixel budget under edge-like names.
        for name, key in (
            ("min_pixels", "shortest_edge"),
            ("max_pixels", "longest_edge"),
        ):
            value = image_config.get(name, size.get(key))
            if value is not None:
                options[name] = value
        if "rescale_factor" in image_config:
            options["rescale_factor"] = image_config["rescale_factor"]
        return cls(
            HyV4ImageProcessor(**options),
            tokenizer,
            chat_template=getattr(tokenizer, "chat_template", None),
        )
