import itertools
import json
import math
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin

from ..base import install_auto_processor_patch, load_chat_template, to_mlx


def smart_resize(height: int, width: int, patch_size: int, max_tokens: int):
    ideal_h = height / patch_size
    ideal_w = width / patch_size
    ratio = ideal_w / ideal_h if ideal_h > 0 else 1.0
    if ideal_h * ideal_w > max_tokens:
        ideal_h = math.sqrt(max_tokens / ratio)
        ideal_w = ideal_h * ratio
    candidates = set(
        itertools.product(
            (math.floor(ideal_h), math.ceil(ideal_h)),
            (math.floor(ideal_w), math.ceil(ideal_w)),
        )
    )
    candidates = [
        (h, w) for h, w in candidates if h >= 1 and w >= 1 and h * w <= max_tokens
    ]
    if not candidates:
        candidates = [(max(1, round(ideal_h)), max(1, round(ideal_w)))]
    patch_h, patch_w = min(
        candidates, key=lambda grid: abs(grid[0] / grid[1] - height / width)
    )
    return patch_h * patch_size, patch_w * patch_size


def _to_pil(image):
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


class MuseGlimmerImageProcessor(ImageProcessingMixin):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        max_image_tokens: int = 4096,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        rescale_factor: float = 1 / 255.0,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.max_image_tokens = max_image_tokens
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.rescale_factor = rescale_factor

    def _process_one(self, image):
        image = _to_pil(image)
        width, height = image.size
        resized_h, resized_w = smart_resize(
            height,
            width,
            self.patch_size * self.merge_size,
            self.max_image_tokens,
        )
        image = image.resize((resized_w, resized_h), resample=Image.Resampling.LANCZOS)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        array *= self.rescale_factor
        mean = np.asarray(self.image_mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.image_std, dtype=np.float32)[:, None, None]
        array = (array - mean) / std

        channels = array.shape[0]
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
        patches = array.reshape(
            channels,
            grid_h,
            self.patch_size,
            grid_w,
            self.patch_size,
        ).transpose(1, 3, 0, 2, 4)
        patches = np.repeat(patches[:, :, None, ...], self.temporal_patch_size, axis=2)
        patches = patches.reshape(
            grid_h * grid_w,
            self.temporal_patch_size * channels * self.patch_size * self.patch_size,
        )
        return patches, [1, grid_h, grid_w]

    def __call__(self, images, **kwargs):
        if not isinstance(images, list):
            images = [images]
        processed = [self._process_one(image) for image in images]
        return {
            "pixel_values": np.concatenate([item[0] for item in processed], axis=0),
            "image_grid_thw": np.asarray(
                [item[1] for item in processed], dtype=np.int64
            ),
        }

    def preprocess(self, images, **kwargs):
        return self(images, **kwargs)


class MuseGlimmerProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, **kwargs
    ):
        self.image_token = "<|patch|>"
        self.image_start_token = "<|image_start|>"
        self.image_end_token = "<|image_end|>"
        self.video_token = "<|video|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        super().__init__(
            image_processor, tokenizer, chat_template=chat_template, **kwargs
        )

    def __call__(
        self,
        images=None,
        text: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images)

        if not isinstance(text, list):
            text = [text]
        text = list(text)
        if images is not None:
            grids = image_inputs["image_grid_thw"]
            merge_length = self.image_processor.merge_size**2
            image_idx = 0
            for prompt_idx, prompt in enumerate(text):
                while self.image_token in prompt:
                    count = int(np.prod(grids[image_idx]) // merge_length)
                    replacement = (
                        self.image_start_token
                        + "<|image_placeholder|>" * count
                        + self.image_end_token
                    )
                    text[prompt_idx] = text[prompt_idx].replace(
                        self.image_token, replacement, 1
                    )
                    prompt = text[prompt_idx]
                    image_idx += 1
                text[prompt_idx] = text[prompt_idx].replace(
                    "<|image_placeholder|>", self.image_token
                )

        return_tensors = kwargs.pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **kwargs)
        data = to_mlx({**text_inputs, **image_inputs})
        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return list(
            dict.fromkeys(
                self.tokenizer.model_input_names
                + self.image_processor.model_input_names
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        from transformers import AutoTokenizer

        kwargs.pop("use_fast", None)
        kwargs.pop("trust_remote_code", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        load_chat_template(tokenizer, pretrained_model_name_or_path)

        config_path = Path(pretrained_model_name_or_path) / "processor_config.json"
        processor_config = (
            json.loads(config_path.read_text()) if config_path.exists() else {}
        )
        image_config = processor_config.get("image_processor", processor_config)
        image_processor = MuseGlimmerImageProcessor(
            patch_size=image_config.get("patch_size", 14),
            temporal_patch_size=image_config.get("temporal_patch_size", 2),
            merge_size=image_config.get("merge_size", 2),
            max_image_tokens=image_config.get("max_image_tokens", 4096),
            image_mean=image_config.get("image_mean", [0.5, 0.5, 0.5]),
            image_std=image_config.get("image_std", [0.5, 0.5, 0.5]),
            rescale_factor=image_config.get("rescale_factor", 1 / 255.0),
        )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=getattr(tokenizer, "chat_template", None),
        )


install_auto_processor_patch("muse_glimmer", MuseGlimmerProcessor)
