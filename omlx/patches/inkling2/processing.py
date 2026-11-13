# SPDX-License-Identifier: Apache-2.0
"""Torch-free Inkling 2 image processor and processor.

Port of transformers' ``Inkling2ImageProcessor``/``Inkling2Processor``: no
resizing; each image is padded on the right and bottom to a multiple of the
40 px image-token block (fill -1 on the 0-255 scale, before normalization),
normalized to [-1, 1] and cut into one block per image token in raster order.
Transparent images are composited over an 8 px checkerboard first. The chat
template emits one ``<|unused_200054|>`` per image, which the processor
expands to ``rows * columns`` image tokens.
"""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import PreTrainedTokenizerFast


class Inkling2ImageProcessor:
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        size=None,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        rescale_factor=1 / 255,
        **kwargs,
    ):
        size = size or {"height": 40, "width": 40}
        self.block_h = int(size["height"])
        self.block_w = int(size["width"])
        self.mean = np.asarray(image_mean, dtype=np.float32)
        self.std = np.asarray(image_std, dtype=np.float32)
        self.rescale_factor = float(rescale_factor)

    @staticmethod
    def _rgb(image) -> np.ndarray:
        """HWC float32 RGB on the 0-255 scale."""
        if isinstance(image, np.ndarray):
            array = image
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=-1)
            return array[..., :3].astype(np.float32)
        if image.has_transparency_data:
            rgba = np.asarray(image.convert("RGBA"), dtype=np.int32)
            rows = np.arange(rgba.shape[0])[:, None] // 8
            cols = np.arange(rgba.shape[1])[None, :] // 8
            background = np.where((rows + cols) % 2 == 0, 204, 153)[..., None]
            alpha = rgba[..., 3:4]
            rgb = (rgba[..., :3] * alpha + background * (255 - alpha) + 127) // 255
            return rgb.astype(np.float32)
        return np.asarray(image.convert("RGB"), dtype=np.float32)

    def preprocess(self, images):
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]
        blocks, grids = [], []
        for image in images:
            x = self._rgb(image)
            height, width, channels = x.shape
            rows = math.ceil(height / self.block_h)
            cols = math.ceil(width / self.block_w)
            padded = np.full(
                (rows * self.block_h, cols * self.block_w, channels), -1.0, np.float32
            )
            padded[:height, :width] = x
            x = (padded * self.rescale_factor - self.mean) / self.std
            x = x.reshape(rows, self.block_h, cols, self.block_w, channels)
            blocks.append(
                x.transpose(0, 2, 1, 3, 4).reshape(
                    -1, self.block_h, self.block_w, channels
                )
            )
            grids.append([1, rows, cols])
        return {
            "pixel_values": np.concatenate(blocks, axis=0),
            "image_grid_thw": np.asarray(grids, dtype=np.int64),
        }

    def __call__(self, images, **kwargs):
        return self.preprocess(images)


class Inkling2Processor:
    """Tokenizer plus image processor; expands each image placeholder."""

    def __init__(
        self, tokenizer, image_processor=None, image_token="<|unused_200054|>"
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor or Inkling2ImageProcessor()
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        path = Path(pretrained_model_name_or_path)
        config = {}
        if (path / "processor_config.json").exists():
            config = json.loads((path / "processor_config.json").read_text())
        tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
        _ensure_pad_token(tokenizer, path)
        image_processor = Inkling2ImageProcessor(**config.get("image_processor", {}))
        image_token = config.get("image_token", "<|unused_200054|>")
        return cls(tokenizer, image_processor, image_token=image_token)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        names = self.image_processor.model_input_names + list(
            self.tokenizer.model_input_names
        )
        return list(dict.fromkeys(names))

    def __call__(
        self,
        text=None,
        images=None,
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        **kwargs,
    ):
        if kwargs.get("videos") is not None or kwargs.get("audio") is not None:
            raise ValueError("Inkling 2 takes text and images only.")
        if text is None:
            raise ValueError("text is required")
        texts = [text] if isinstance(text, str) else list(text)
        outputs = {}
        if images is not None and (not hasattr(images, "__len__") or len(images)):
            processed = self.image_processor.preprocess(images)
            outputs.update(processed)
            counts = [int(r * c) for _, r, c in processed["image_grid_thw"].tolist()]
            cursor = 0
            expanded = []
            for prompt in texts:
                pieces = prompt.split(self.image_token)
                if cursor + len(pieces) - 1 > len(counts):
                    raise ValueError(
                        f"The prompts hold more image placeholders than the "
                        f"{len(counts)} images given."
                    )
                rebuilt = pieces[0]
                for piece in pieces[1:]:
                    rebuilt += self.image_token * counts[cursor] + piece
                    cursor += 1
                expanded.append(rebuilt)
            if cursor != len(counts):
                raise ValueError(
                    f"{len(counts)} images were given for {cursor} image placeholders."
                )
            texts = expanded
        encoded = self.tokenizer(
            texts,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
            return_tensors="np",
        )
        outputs["input_ids"] = encoded["input_ids"]
        outputs["attention_mask"] = encoded["attention_mask"]
        return outputs


def _ensure_pad_token(tokenizer, path: Path) -> None:
    """The checkpoint names no pad or eos token string; use the config's eos."""
    if tokenizer.pad_token is not None:
        return
    eos = None
    with contextlib.suppress(OSError, ValueError):
        eos = json.loads((path / "config.json").read_text()).get("eos_token_id")
    if eos is None:
        return
    eos = eos[0] if isinstance(eos, (list, tuple)) else eos
    token = tokenizer.convert_ids_to_tokens(int(eos))
    if tokenizer.eos_token is None:
        tokenizer.eos_token = token
    tokenizer.pad_token = token
