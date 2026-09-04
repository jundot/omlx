import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.image_utils import ImageInput, is_valid_image, load_image


def _is_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _is_single_image(value: Any) -> bool:
    return isinstance(value, (str, Image.Image, np.ndarray)) or is_valid_image(value)


def _flatten_images(images) -> list:
    if images is None:
        return []
    if _is_single_image(images):
        return [images]
    if isinstance(images, (list, tuple)):
        flattened = []
        for image in images:
            flattened.extend(_flatten_images(image))
        return flattened
    return [images]


def image_counts_per_sample(images) -> list[int]:
    """Return flattened image counts for each text sample."""
    if images is None:
        return []
    if _is_single_image(images):
        return [1]
    if not isinstance(images, (list, tuple)):
        return [1]
    if not images:
        return []
    if all(_is_single_image(image) for image in images):
        return [len(images)]
    return [len(_flatten_images(sample)) for sample in images]


def split_image_sizes_by_sample(image_sizes, images) -> list[list[tuple[int, int]]]:
    def _flatten_sizes(sizes):
        if hasattr(sizes, "tolist"):
            sizes = sizes.tolist()
        if (
            isinstance(sizes, (list, tuple))
            and len(sizes) == 2
            and all(np.isscalar(dim) for dim in sizes)
        ):
            return [(int(sizes[0]), int(sizes[1]))]
        flattened = []
        for size in sizes:
            flattened.extend(_flatten_sizes(size))
        return flattened

    flat_sizes = _flatten_sizes(image_sizes)

    grouped_sizes = []
    start = 0
    for count in image_counts_per_sample(images):
        grouped_sizes.append(flat_sizes[start : start + count])
        start += count
    return grouped_sizes


def get_resize_output_image_size(
    image_size: tuple[int, int],
    size: int | tuple[int, int],
    patch_size: int | tuple[int, int],
) -> tuple[int, int]:
    max_height, max_width = size if isinstance(size, (tuple, list)) else (size, size)
    patch_height, patch_width = (
        patch_size
        if isinstance(patch_size, (tuple, list))
        else (patch_size, patch_size)
    )
    height, width = image_size

    ratio = max(height / max_height, width / max_width)
    if ratio > 1:
        height = int(math.floor(height / ratio))
        width = int(math.floor(width / ratio))

    num_height_tokens = (height - 1) // patch_height + 1
    num_width_tokens = (width - 1) // patch_width + 1
    return num_height_tokens * patch_height, num_width_tokens * patch_width


def _size_to_pair(size) -> tuple[int, int]:
    if isinstance(size, dict):
        if size.get("longest_edge") is not None:
            edge = int(size["longest_edge"])
            return edge, edge
        if size.get("height") is not None and size.get("width") is not None:
            return int(size["height"]), int(size["width"])
    if isinstance(size, (list, tuple)):
        if len(size) == 1:
            return int(size[0]), int(size[0])
        return int(size[0]), int(size[1])
    edge = int(size)
    return edge, edge


def _patch_size_to_pair(patch_size) -> tuple[int, int]:
    if isinstance(patch_size, dict):
        return int(patch_size["height"]), int(patch_size["width"])
    if isinstance(patch_size, (list, tuple)):
        if len(patch_size) == 1:
            return int(patch_size[0]), int(patch_size[0])
        return int(patch_size[0]), int(patch_size[1])
    patch = int(patch_size)
    return patch, patch


def _load_json_config(pretrained_model_name_or_path, filename: str) -> dict:
    model_path = Path(pretrained_model_name_or_path)
    if model_path.exists():
        config_path = model_path / filename
        if not config_path.exists():
            return {}
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = Path(hf_hub_download(pretrained_model_name_or_path, filename))
        except Exception:
            return {}

    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _to_pil_image(image) -> Image.Image:
    if isinstance(image, str):
        image = load_image(image) if _is_url(image) else Image.open(image)
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
    if array.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dimensions, got {array.ndim}.")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


class PixtralImageProcessor(ImageProcessingMixin):
    model_input_names = ["pixel_values", "image_sizes"]

    def __init__(
        self,
        size: dict | int = None,
        patch_size: dict | int = None,
        image_mean: Iterable[float] | None = None,
        image_std: Iterable[float] | None = None,
        do_resize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
        resample=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size or {"longest_edge": 1024}
        self.patch_size = patch_size or {"height": 16, "width": 16}
        self.image_mean = list(image_mean or [0.48145466, 0.4578275, 0.40821073])
        self.image_std = list(image_std or [0.26862954, 0.26130258, 0.27577711])
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        if resample is None:
            resample = Image.Resampling.BICUBIC
        elif isinstance(resample, int):
            resample = Image.Resampling(resample)
        self.resample = resample

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor_config = _load_json_config(
            pretrained_model_name_or_path, "processor_config.json"
        )
        image_config = processor_config.get("image_processor", {})
        if not image_config:
            image_config = _load_json_config(
                pretrained_model_name_or_path, "preprocessor_config.json"
            )

        for key in ("size", "patch_size"):
            if key not in image_config and key in processor_config:
                image_config[key] = processor_config[key]

        image_config = dict(image_config)
        for key in (
            "image_processor_type",
            "processor_class",
            "return_tensors",
            "device",
            "disable_grouping",
            "input_data_format",
        ):
            image_config.pop(key, None)

        valid_overrides = {
            "size",
            "patch_size",
            "image_mean",
            "image_std",
            "do_resize",
            "do_rescale",
            "rescale_factor",
            "do_normalize",
            "do_convert_rgb",
            "resample",
        }
        image_config.update(
            {key: value for key, value in kwargs.items() if key in valid_overrides}
        )
        return cls(**image_config)

    def __call__(self, images: ImageInput, **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)

    def preprocess(self, images: ImageInput, **kwargs) -> BatchFeature:
        return_tensors = kwargs.pop("return_tensors", None)
        size = _size_to_pair(kwargs.pop("size", self.size))
        patch_size = _patch_size_to_pair(kwargs.pop("patch_size", self.patch_size))
        do_resize = kwargs.pop("do_resize", self.do_resize)
        do_rescale = kwargs.pop("do_rescale", self.do_rescale)
        rescale_factor = kwargs.pop("rescale_factor", self.rescale_factor)
        do_normalize = kwargs.pop("do_normalize", self.do_normalize)
        image_mean = kwargs.pop("image_mean", self.image_mean)
        image_std = kwargs.pop("image_std", self.image_std)
        resample = kwargs.pop("resample", self.resample)

        pixel_values = []
        image_sizes = []
        for image in _flatten_images(images):
            pil_image = _to_pil_image(image)
            target_size = (pil_image.height, pil_image.width)
            if do_resize:
                target_size = get_resize_output_image_size(
                    target_size, size=size, patch_size=patch_size
                )
                pil_image = pil_image.resize(
                    (target_size[1], target_size[0]), resample=resample
                )

            image_array = np.array(pil_image).astype(np.float32)
            if do_rescale:
                image_array *= rescale_factor
            if do_normalize:
                mean = np.array(image_mean, dtype=np.float32)
                std = np.array(image_std, dtype=np.float32)
                image_array = (image_array - mean) / std
            image_array = image_array.transpose(2, 0, 1)

            pixel_values.append(image_array)
            image_sizes.append(target_size)

        if not pixel_values:
            raise ValueError("You must provide at least one image.")

        max_height = max(image.shape[1] for image in pixel_values)
        max_width = max(image.shape[2] for image in pixel_values)
        padded_images = [
            np.pad(
                image,
                (
                    (0, 0),
                    (0, max_height - image.shape[1]),
                    (0, max_width - image.shape[2]),
                ),
            )
            for image in pixel_values
        ]

        return BatchFeature(
            data={
                "pixel_values": np.stack(padded_images).astype(np.float32),
                "image_sizes": image_sizes,
            },
            tensor_type=return_tensors,
        )


__all__ = [
    "PixtralImageProcessor",
    "get_resize_output_image_size",
    "image_counts_per_sample",
    "split_image_sizes_by_sample",
]
