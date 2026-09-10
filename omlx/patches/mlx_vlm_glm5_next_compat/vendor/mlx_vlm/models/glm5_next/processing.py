# SPDX-License-Identifier: Apache-2.0
"""Torch-free processor support for GLM-5.3-Flash.

The official checkpoint targets a newer Transformers release than oMLX.  This
module mirrors the GLM-5 Next image layout with NumPy and Pillow so image
requests remain available without pulling in torch or torchvision.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch, load_chat_template, to_mlx

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_IMAGE_KWARGS = {
    "do_convert_rgb",
    "do_normalize",
    "do_rescale",
    "image_mean",
    "image_std",
    "max_image_tokens",
    "merge_size",
    "min_image_tokens",
    "patch_expand_factor",
    "patch_size",
    "rescale_factor",
    "temporal_patch_size",
}


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    *,
    temporal_factor: int = 2,
    factor: int = 28,
    min_image_tokens: int = 16,
    max_image_tokens: int = 8000,
) -> tuple[int, int]:
    """Return GLM's aligned canvas dimensions for an image or video."""
    if num_frames <= 0 or height <= 0 or width <= 0:
        raise ValueError("Image dimensions and frame count must be positive.")

    pixels_per_token = temporal_factor * factor**2
    min_pixels = min_image_tokens * pixels_per_token
    max_pixels = max_image_tokens * pixels_per_token

    def align(value: int) -> int:
        return math.ceil(value / factor) * factor

    aligned_frames = max(
        temporal_factor,
        round(num_frames / temporal_factor) * temporal_factor,
    )
    aligned_height = align(height)
    aligned_width = align(width)
    aligned_pixel_budget = aligned_frames * aligned_height * aligned_width

    if aligned_pixel_budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)))
        aligned_width = align(max(1, math.ceil(width * scale)))
        aligned_pixel_budget = aligned_frames * aligned_height * aligned_width

    if aligned_pixel_budget <= max_pixels:
        return aligned_height, aligned_width

    minimum_pixels = aligned_frames * factor**2
    if max_pixels < minimum_pixels:
        raise ValueError(
            f"max_image_tokens={max_image_tokens} is too small for one "
            "aligned patch."
        )

    low, high = 1, height
    best_height, best_width = factor, factor
    while low <= high:
        content_height = (low + high) // 2
        content_width = max(1, math.floor(width * content_height / height))
        candidate_height = align(content_height)
        candidate_width = align(content_width)
        if aligned_frames * candidate_height * candidate_width <= max_pixels:
            best_height, best_width = candidate_height, candidate_width
            low = content_height + 1
        else:
            high = content_height - 1
    return best_height, best_width


def _to_channel_first(image: Any, do_convert_rgb: bool) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if hasattr(image, "convert"):
        if do_convert_rgb:
            image = image.convert("RGB")
        array = np.asarray(image)
    else:
        array = np.asarray(image)

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {array.shape}.")
    if array.shape[-1] in (1, 3, 4):
        array = np.transpose(array, (2, 0, 1))
    if array.shape[0] == 4:
        array = array[:3]
    if array.shape[0] == 1 and do_convert_rgb:
        array = np.repeat(array, 3, axis=0)
    if array.shape[0] != 3:
        raise ValueError(f"Expected an RGB image, got shape {array.shape}.")
    return array


def _resize_channel_first(
    image: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    if image.shape[-2:] == (target_height, target_width):
        return image
    array = np.transpose(image, (1, 2, 0))
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    resized = Image.fromarray(array).resize(
        (target_width, target_height),
        resample=Image.Resampling.BICUBIC,
    )
    return np.transpose(np.asarray(resized), (2, 0, 1))


class Glm5NextImageProcessor(ImageProcessingMixin):
    """NumPy/Pillow implementation of the GLM-5 Next image processor."""

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        patch_expand_factor: int = 1,
        min_image_tokens: int = 16,
        max_image_tokens: int = 8000,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        do_convert_rgb: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.patch_expand_factor = patch_expand_factor
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = list(image_mean or OPENAI_CLIP_MEAN)
        self.image_std = list(image_std or OPENAI_CLIP_STD)
        self.do_convert_rgb = do_convert_rgb

    def fetch_images(self, images):
        if not isinstance(images, list):
            images = [images]
        return [_to_channel_first(image, self.do_convert_rgb) for image in images]

    def _process_one(self, image: Any, **kwargs) -> tuple[np.ndarray, list[int]]:
        do_convert_rgb = kwargs.get("do_convert_rgb", self.do_convert_rgb)
        do_rescale = kwargs.get("do_rescale", self.do_rescale)
        rescale_factor = kwargs.get("rescale_factor", self.rescale_factor)
        do_normalize = kwargs.get("do_normalize", self.do_normalize)
        image_mean = kwargs.get("image_mean", self.image_mean)
        image_std = kwargs.get("image_std", self.image_std)
        image = _to_channel_first(image, do_convert_rgb)
        _, height, width = image.shape
        patch_size = kwargs.get("patch_size", self.patch_size)
        temporal_patch_size = kwargs.get(
            "temporal_patch_size", self.temporal_patch_size
        )
        merge_size = kwargs.get("merge_size", self.merge_size)
        patch_expand_factor = kwargs.get(
            "patch_expand_factor", self.patch_expand_factor
        )
        min_image_tokens = kwargs.get("min_image_tokens", self.min_image_tokens)
        max_image_tokens = kwargs.get("max_image_tokens", self.max_image_tokens)
        factor = patch_size * merge_size * patch_expand_factor
        target_height, target_width = smart_resize(
            temporal_patch_size,
            height,
            width,
            temporal_factor=temporal_patch_size,
            factor=factor,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
        )

        pixels_per_token = temporal_patch_size * factor**2
        scale = min(target_height / height, target_width / width)
        if temporal_patch_size * height * width >= (
            pixels_per_token * min_image_tokens
        ):
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))
        image = _resize_channel_first(image, content_height, content_width)

        canvas = np.zeros(
            (image.shape[0], target_height, target_width), dtype=image.dtype
        )
        canvas[:, :content_height, :content_width] = image
        image = canvas.astype(np.float32)
        if do_rescale:
            image *= rescale_factor
        if do_normalize:
            mean = np.asarray(image_mean, dtype=np.float32)[:, None, None]
            std = np.asarray(image_std, dtype=np.float32)[:, None, None]
            image = (image - mean) / std

        channels = image.shape[0]
        grid_h = target_height // patch_size
        grid_w = target_width // patch_size
        patches = image.reshape(
            channels,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)
        patches = np.expand_dims(patches, axis=5)
        patches = np.repeat(patches, temporal_patch_size, axis=5)
        patches = patches.reshape(
            grid_h * grid_w,
            channels * temporal_patch_size * patch_size * patch_size,
        )
        return patches.astype(np.float32), [1, grid_h, grid_w]

    def __call__(self, images=None, **kwargs):
        return self.preprocess(images=images, **kwargs)

    def preprocess(self, images=None, return_tensors=None, **kwargs) -> BatchFeature:
        if images is None:
            raise ValueError("images must not be None")
        if not isinstance(images, list):
            images = [images]
        image_kwargs = {
            key: value for key, value in kwargs.items() if key in _IMAGE_KWARGS
        }
        processed = [self._process_one(image, **image_kwargs) for image in images]
        data = {
            "pixel_values": np.concatenate([item[0] for item in processed], axis=0),
            "image_grid_thw": np.asarray(
                [item[1] for item in processed], dtype=np.int64
            ),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict | None = None
    ) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        patch_expand_factor = images_kwargs.get(
            "patch_expand_factor", self.patch_expand_factor
        )
        target_height, target_width = smart_resize(
            images_kwargs.get("temporal_patch_size", self.temporal_patch_size),
            height,
            width,
            temporal_factor=images_kwargs.get(
                "temporal_patch_size", self.temporal_patch_size
            ),
            factor=patch_size * merge_size * patch_expand_factor,
            min_image_tokens=images_kwargs.get(
                "min_image_tokens", self.min_image_tokens
            ),
            max_image_tokens=images_kwargs.get(
                "max_image_tokens", self.max_image_tokens
            ),
        )
        return (target_height // patch_size) * (target_width // patch_size)


def _load_json(model_path: str | Path, filename: str) -> dict | None:
    local = Path(model_path) / filename
    if local.exists():
        return json.loads(local.read_text())
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(str(model_path), filename)
        return json.loads(Path(downloaded).read_text())
    except Exception:
        return None


def _image_processor_kwargs(model_path: str | Path) -> dict:
    config = _load_json(model_path, "config.json") or {}
    vision = config.get("vision_config", {}) or {}
    processor = _load_json(model_path, "processor_config.json") or {}
    image = processor.get("image_processor", {}) or {}

    result = {}
    for key in _IMAGE_KWARGS:
        if key in image:
            result[key] = image[key]
    for source, target in (
        ("patch_size", "patch_size"),
        ("temporal_patch_size", "temporal_patch_size"),
        ("spatial_merge_size", "merge_size"),
    ):
        if target not in result and source in vision:
            result[target] = vision[source]
    return result


class Glm5NextProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        image_processor = image_processor or Glm5NextImageProcessor()
        self.image_token = "<|image|>"
        self.video_token = "<|video|>"
        self.image_token_id = (
            tokenizer.convert_tokens_to_ids(self.image_token) if tokenizer else None
        )
        self.video_token_id = (
            tokenizer.convert_tokens_to_ids(self.video_token) if tokenizer else None
        )
        super().__init__(
            image_processor,
            tokenizer,
            chat_template=chat_template,
        )

    def __call__(
        self,
        images=None,
        text=None,
        padding=True,
        padding_side=None,
        add_special_tokens=False,
        return_tensors="mlx",
        return_mm_token_type_ids=True,
        **kwargs,
    ) -> BatchFeature:
        image_inputs = {}
        if images is not None:
            image_kwargs = {
                key: value for key, value in kwargs.items() if key in _IMAGE_KWARGS
            }
            image_inputs = dict(
                self.image_processor(
                    images=images,
                    return_tensors=None,
                    **image_kwargs,
                )
            )

        if text is None:
            text = [""]
        elif not isinstance(text, list):
            text = [text]
        text = ["" if item is None else str(item) for item in text]

        if image_inputs:
            placeholder = "<|glm5_next_image_placeholder|>"
            image_idx = 0
            grids = image_inputs["image_grid_thw"]
            for prompt_idx, prompt in enumerate(text):
                while self.image_token in prompt:
                    if image_idx >= len(grids):
                        raise ValueError("More image tokens were provided than images.")
                    count = int(
                        np.prod(grids[image_idx]) // self.image_processor.merge_size**2
                    )
                    prompt = prompt.replace(self.image_token, placeholder * count, 1)
                    image_idx += 1
                text[prompt_idx] = prompt.replace(placeholder, self.image_token)
            if image_idx != len(grids):
                raise ValueError("More images were provided than image tokens.")

        tokenizer_kwargs = dict(kwargs)
        for key in _IMAGE_KWARGS:
            tokenizer_kwargs.pop(key, None)
        if padding_side is not None:
            tokenizer_kwargs["padding_side"] = padding_side
        text_inputs = dict(
            self.tokenizer(
                text,
                padding=padding,
                add_special_tokens=add_special_tokens,
                return_tensors=None,
                **tokenizer_kwargs,
            )
        )

        if return_mm_token_type_ids and self.image_token_id is not None:
            input_ids = np.asarray(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(input_ids)
            mm_token_type_ids[input_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        data = {**text_inputs, **image_inputs}
        if return_tensors == "mlx":
            data = to_mlx(data)
        return BatchFeature(data=data)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_names = getattr(self.tokenizer, "model_input_names", [])
        return list(
            dict.fromkeys(
                tokenizer_names
                + self.image_processor.model_input_names
                + ["mm_token_type_ids"]
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer_kwargs = dict(kwargs)
        chat_template = tokenizer_kwargs.pop("chat_template", None)
        tokenizer_kwargs.pop("return_tensors", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            **tokenizer_kwargs,
        )
        if Path(pretrained_model_name_or_path).exists():
            load_chat_template(tokenizer, pretrained_model_name_or_path)
        processor_config = (
            _load_json(pretrained_model_name_or_path, "processor_config.json") or {}
        )
        return cls(
            image_processor=Glm5NextImageProcessor(
                **_image_processor_kwargs(pretrained_model_name_or_path)
            ),
            tokenizer=tokenizer,
            chat_template=(
                chat_template
                or processor_config.get("chat_template")
                or getattr(tokenizer, "chat_template", None)
            ),
        )


install_auto_processor_patch("glm5_next", Glm5NextProcessor)

__all__ = [
    "Glm5NextImageProcessor",
    "Glm5NextProcessor",
    "smart_resize",
]
