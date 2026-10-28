# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3/LFM3-VL checkpoint reference implementations.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

"""NumPy preprocessing for LFM3 images and pre-sampled video frames."""

import json
import math
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizerFast
from transformers.feature_extraction_utils import BatchFeature

from ..base import install_auto_processor_patch, load_chat_template, to_mlx
from ..qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from ..qwen3_vl.processing_qwen3_vl import (
    Qwen3VLImageProcessor,
    Qwen3VLVideoProcessor,
    _load_qwen_vl_json,
    _resize_video_frames,
)


def _pixel_kwargs(config):
    config = dict(config)
    size = config.pop("size", {})
    if size:
        config["min_pixels"] = size["shortest_edge"]
        config["max_pixels"] = size["longest_edge"]
    return config


class Lfm3VlVideoProcessor(Qwen3VLVideoProcessor):
    sample_frames_in_loader = True

    def sample_frames(self, metadata, fps=None, max_frames=None):
        """The reference rule: odd counts are kept and padded later by repeating
        the last frame, rather than floored to the temporal patch size."""
        total = metadata.total_num_frames
        count = int(total / metadata.fps * (fps or self.fps))
        count = min(max(count, self.min_frames), max_frames or self.max_frames, total)
        return np.linspace(0, total - 1, count).round().astype(int)

    def _process_one(self, video):
        t, channels, height, width = video.shape
        p, m, temporal = self.patch_size, self.merge_size, self.temporal_patch_size
        if t < temporal:
            raise ValueError(f"LFM3-VL video requires at least {temporal} frames")
        if channels == 1:
            video = np.repeat(video, 3, axis=1)
            channels = 3
        factor = p * m
        if min(height, width) < factor:
            scale = factor / min(height, width)
            height, width = int(height * scale), int(width * scale)
        if max(height, width) / min(height, width) > 200:
            raise ValueError("Video aspect ratio exceeds 200")
        target_h, target_w = (
            round(height / factor) * factor,
            round(width / factor) * factor,
        )
        rounded_t = round(t / temporal) * temporal
        if rounded_t * target_h * target_w > self.max_pixels:
            beta = math.sqrt(t * height * width / self.max_pixels)
            target_h = max(factor, math.floor(height / beta / factor) * factor)
            target_w = max(factor, math.floor(width / beta / factor) * factor)
        elif rounded_t * target_h * target_w < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (t * height * width))
            target_h = math.ceil(height * beta / factor) * factor
            target_w = math.ceil(width * beta / factor) * factor
        video = _resize_video_frames(video, target_h, target_w)
        pixels = video.astype(np.float32)
        if self.do_rescale:
            pixels *= self.rescale_factor
        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32)[None, :, None, None]
            std = np.array(self.image_std, dtype=np.float32)[None, :, None, None]
            pixels = (pixels - mean) / std
        if pad := -t % temporal:
            pixels = np.concatenate([pixels, np.repeat(pixels[-1:], pad, axis=0)])
        grid_t, grid_h, grid_w = len(pixels) // temporal, target_h // p, target_w // p
        pixels = pixels.reshape(
            grid_t, temporal, channels, grid_h // m, m, p, grid_w // m, m, p
        )
        pixels = pixels.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return pixels.reshape(grid_t * grid_h * grid_w, channels * temporal * p * p), [
            grid_t,
            grid_h,
            grid_w,
        ]


class Lfm3VlProcessor(Qwen2VLProcessor):
    def __init__(
        self,
        image_processor,
        tokenizer,
        video_processor,
        processor_config=None,
        **kwargs,
    ):
        super().__init__(image_processor, tokenizer, video_processor, **kwargs)
        self.image_token = "<|image|>"
        self.video_token = "<|video|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.processor_config = processor_config or {}

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        config = _load_qwen_vl_json(model_path, "processor_config.json") or {}
        image_config = _pixel_kwargs(config.get("image_processor", {}))
        video_config = _pixel_kwargs(config.get("video_processor", {}))
        for key in ("min_pixels", "max_pixels"):
            if key in kwargs:
                image_config[key] = kwargs.pop(key)
        tokenizer = load_chat_template(
            PreTrainedTokenizerFast.from_pretrained(model_path, **kwargs), model_path
        )
        return cls(
            Qwen3VLImageProcessor(**image_config),
            tokenizer,
            Lfm3VlVideoProcessor(**video_config),
            processor_config=config,
            chat_template=tokenizer.chat_template,
        )

    @staticmethod
    def _expand(texts, token, replacements):
        index = 0
        expanded = []
        for text in texts:
            parts = text.split(token)
            output = parts[0]
            for part in parts[1:]:
                if index == len(replacements):
                    raise ValueError(f"More {token} placeholders than supplied media")
                output += replacements[index] + part
                index += 1
            expanded.append(output)
        if index != len(replacements):
            raise ValueError(
                f"Expected {len(replacements)} {token} placeholders, got {index}"
            )
        return expanded

    @staticmethod
    def group_seconds(groups, fps):
        # as the reference: per-frame seconds first, so x.x5 rounds the same way
        return (groups[:, 0] / fps + groups[:, -1] / fps) / 2

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        fps=None,
        video_metadata=None,
        add_special_tokens=False,
        padding_side="left",
        **kwargs,
    ):
        text = [text] if isinstance(text, str) else list(text or [])
        image_kwargs = dict(kwargs.pop("images_kwargs", {}))
        for key in ("min_pixels", "max_pixels", "resized_height", "resized_width"):
            if key in kwargs:
                image_kwargs[key] = kwargs.pop(key)
        if "size" in image_kwargs:
            image_kwargs.update(_pixel_kwargs({"size": image_kwargs.pop("size")}))
        data = {}
        if images is not None:
            data.update(self.image_processor(images=images, **image_kwargs))
            counts = (
                data["image_grid_thw"].prod(axis=-1)
                // self.image_processor.merge_size**2
            )
            text = self._expand(
                text, self.image_token, [self.image_token * int(n) for n in counts]
            )
        video_kwargs = dict(kwargs.pop("videos_kwargs", {}))
        metadata = video_metadata or video_kwargs.pop("video_metadata", None)
        if kwargs.pop("do_sample_frames", video_kwargs.get("do_sample_frames", False)):
            raise ValueError("Pass pre-sampled video frames to LFM3-VL")
        if videos is not None:
            # Frames are sampled by the caller (the CLI video loader supplies frames).
            if not isinstance(videos, list) or (
                videos and self.video_processor._is_video_frame(videos[0])
            ):
                videos = [videos]
            data.update(self.video_processor(videos=videos))
            if metadata is None:
                rates = (
                    fps
                    if isinstance(fps, (list, tuple))
                    else [fps or 24.0] * len(videos)
                )
                metadata = [
                    dict(
                        fps=rate,
                        frames_indices=list(
                            range(len(self.video_processor._prepare_video(video)))
                        ),
                    )
                    for rate, video in zip(rates, videos)
                ]
            if len(metadata) != len(videos):
                raise ValueError("Expected one video metadata record per video")
            replacements = []
            for (t, h, w), meta in zip(data["video_grid_thw"], metadata):
                fps = (
                    (meta.get("fps") or 24.0)
                    if isinstance(meta, dict)
                    else (meta.fps or 24.0)
                )
                # as the reference: drops float noise (torchcodec's 30.000000000000004)
                fps = round(fps, 6)
                indices = (
                    meta.get("frames_indices")
                    if isinstance(meta, dict)
                    else meta.frames_indices
                )
                indices = (
                    list(range(int(t) * self.video_processor.temporal_patch_size))
                    if indices is None
                    else list(indices)
                )
                if not indices or fps <= 0:
                    raise ValueError(
                        "Video metadata requires frame indices and a positive fps"
                    )
                group = self.video_processor.temporal_patch_size
                indices.extend([indices[-1]] * (-len(indices) % group))
                timestamps = self.group_seconds(
                    np.asarray(indices).reshape(-1, group), fps
                )
                if len(timestamps) != t:
                    raise ValueError(
                        "Video timestamps do not match the processed frame groups"
                    )
                replacements.append(
                    "".join(
                        f"<{timestamp:.1f} seconds><|vision_start|>"
                        + self.video_token
                        * int(h * w // self.video_processor.merge_size**2)
                        + "<|vision_end|>"
                        for timestamp in timestamps
                    )
                )
            text = self._expand(text, self.video_token, replacements)
        return_tensors = kwargs.pop("return_tensors", "mlx")
        kwargs.pop("return_mm_token_type_ids", None)
        kwargs.setdefault("return_token_type_ids", False)
        data.update(
            self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                padding_side=padding_side,
                **kwargs,
            )
        )
        if return_tensors == "mlx":
            return BatchFeature(data=to_mlx(data))
        return BatchFeature(data=data, tensor_type=return_tensors)

    def save_pretrained(self, save_directory, **kwargs):
        self.tokenizer.save_pretrained(save_directory, **kwargs)
        config = dict(self.processor_config)
        config.pop("auto_map", None)
        config["processor_class"] = type(self).__name__
        for key in ("image_processor", "video_processor"):
            component = getattr(self, key)
            settings = dict(config.get(key, {}))
            settings.pop("auto_map", None)
            for name in (
                "patch_size",
                "temporal_patch_size",
                "merge_size",
                "image_mean",
                "image_std",
                "do_rescale",
                "rescale_factor",
                "do_normalize",
                "do_convert_rgb",
            ):
                settings[name] = getattr(component, name)
            settings["size"] = {
                "shortest_edge": component.min_pixels,
                "longest_edge": component.max_pixels,
            }
            config[key] = settings
        Path(save_directory, "processor_config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )


install_auto_processor_patch("lfm3_vl", Lfm3VlProcessor)
