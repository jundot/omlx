# SPDX-License-Identifier: Apache-2.0
"""Torch-free Qwen3-VL video preprocessing for oMLX."""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.video_utils import VideoMetadata

# Video pixel budgets from transformers Qwen3VLVideoProcessor defaults.
_DEFAULT_VIDEO_MIN_PIXELS = 128 * 32 * 32
_DEFAULT_VIDEO_MAX_PIXELS = 32 * 32 * 768


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    *,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = _DEFAULT_VIDEO_MIN_PIXELS,
    max_pixels: int = _DEFAULT_VIDEO_MAX_PIXELS,
) -> Tuple[int, int]:
    """Mirror HF ``smart_resize`` for Qwen3-VL videos."""
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def _resize_video_frames(video: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bicubic resize each frame of a ``(T, C, H, W)`` video."""
    t, c, h, w = video.shape
    if target_h == h and target_w == w:
        return video
    out = np.empty((t, c, target_h, target_w), dtype=video.dtype)
    for i, frame in enumerate(video):
        arr = np.transpose(frame, (1, 2, 0))
        if arr.dtype in (np.float32, np.float64):
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        pil = pil.resize((target_w, target_h), resample=Image.BICUBIC)
        out[i] = np.transpose(np.array(pil), (2, 0, 1))
    return out


def _to_video_tchw(video: Union[np.ndarray, Sequence[Any]]) -> np.ndarray:
    """Normalize video input to ``(T, C, H, W)`` uint8."""
    if isinstance(video, np.ndarray):
        arr = video
    elif isinstance(video, list):
        frames = []
        for frame in video:
            if isinstance(frame, np.ndarray):
                frames.append(frame)
            else:
                frames.append(np.asarray(frame))
        if not frames:
            raise ValueError("Empty video frame list")
        if frames[0].ndim == 3 and frames[0].shape[-1] in (1, 3, 4):
            arr = np.stack(frames, axis=0)
        elif frames[0].ndim == 3 and frames[0].shape[0] in (1, 3, 4):
            arr = np.stack(frames, axis=0)
        else:
            raise ValueError(f"Unsupported frame shape: {frames[0].shape}")
    else:
        arr = np.asarray(video)

    if arr.ndim != 4:
        raise ValueError(f"Expected video with 4 dimensions, got shape {arr.shape}")

    if arr.shape[1] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        tchw = arr
    elif arr.shape[-1] in (1, 3, 4):
        tchw = np.transpose(arr, (0, 3, 1, 2))
    else:
        raise ValueError(f"Cannot infer channel dimension for shape {arr.shape}")

    if tchw.dtype != np.uint8:
        if np.issubdtype(tchw.dtype, np.floating) and tchw.max() <= 1.0:
            tchw = (tchw * 255).clip(0, 255).astype(np.uint8)
        else:
            tchw = tchw.clip(0, 255).astype(np.uint8)
    return tchw


class TorchFreeQwen3VLVideoProcessor:
    """Numpy/PIL shim for ``Qwen3VLVideoProcessor`` in torch-free environments."""

    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        image_processor: Any,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        fps: float = 2.0,
        min_frames: int = 4,
        max_frames: int = 768,
    ) -> None:
        self.patch_size = int(getattr(image_processor, "patch_size", 16))
        self.temporal_patch_size = int(
            getattr(image_processor, "temporal_patch_size", 2)
        )
        self.merge_size = int(getattr(image_processor, "merge_size", 2))
        self.do_rescale = bool(getattr(image_processor, "do_rescale", True))
        self.rescale_factor = float(
            getattr(image_processor, "rescale_factor", 1 / 255.0)
        )
        self.do_normalize = bool(getattr(image_processor, "do_normalize", True))
        self.image_mean = list(
            getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        )
        self.image_std = list(getattr(image_processor, "image_std", [0.5, 0.5, 0.5]))
        self.do_convert_rgb = bool(getattr(image_processor, "do_convert_rgb", True))
        self.min_pixels = (
            int(min_pixels)
            if min_pixels is not None
            else _DEFAULT_VIDEO_MIN_PIXELS
        )
        self.max_pixels = (
            int(max_pixels)
            if max_pixels is not None
            else _DEFAULT_VIDEO_MAX_PIXELS
        )
        self.fps = float(fps)
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)

    def _process_one(self, video: np.ndarray) -> Tuple[np.ndarray, List[int], VideoMetadata]:
        video = _to_video_tchw(video)
        t, c, h, w = video.shape
        if c == 1 and self.do_convert_rgb:
            video = np.repeat(video, 3, axis=1)
            c = 3
        elif c == 4 and self.do_convert_rgb:
            video = video[:, :3]
            c = 3

        resized_h, resized_w = smart_resize(
            num_frames=t,
            height=h,
            width=w,
            temporal_factor=self.temporal_patch_size,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        video = _resize_video_frames(video, resized_h, resized_w)

        video_f = video.astype(np.float32)
        if self.do_rescale and video.dtype == np.uint8:
            video_f = video_f * self.rescale_factor
        if self.do_normalize:
            mean = np.array(self.image_mean, dtype=np.float32)[None, :, None, None]
            std = np.array(self.image_std, dtype=np.float32)[None, :, None, None]
            video_f = (video_f - mean) / std

        pad = (-video_f.shape[0]) % self.temporal_patch_size
        if pad:
            video_f = np.concatenate(
                [video_f, np.repeat(video_f[-1:], pad, axis=0)], axis=0
            )

        t_padded = video_f.shape[0]
        grid_t = t_padded // self.temporal_patch_size
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
        ps = self.patch_size
        tps = self.temporal_patch_size
        ms = self.merge_size

        patches = video_f[None, ...]
        patches = patches.reshape(
            1,
            grid_t,
            tps,
            c,
            grid_h // ms,
            ms,
            ps,
            grid_w // ms,
            ms,
            ps,
        )
        patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flatten = patches.reshape(1, grid_t * grid_h * grid_w, c * tps * ps * ps)

        metadata = VideoMetadata(
            total_num_frames=t,
            fps=None,
            width=w,
            height=h,
            frames_indices=list(range(t)),
        )
        return flatten[0], [grid_t, grid_h, grid_w], metadata

    def __call__(self, videos: Any, **kwargs: Any) -> BatchFeature:
        return_metadata = kwargs.pop("return_metadata", False)
        fps = kwargs.pop("fps", None)
        if not isinstance(videos, list):
            videos = [videos]

        all_patches: List[np.ndarray] = []
        all_thw: List[List[int]] = []
        all_metadata: List[VideoMetadata] = []

        for idx, video in enumerate(videos):
            patches, thw, metadata = self._process_one(video)
            if fps is not None:
                video_fps = fps[idx] if isinstance(fps, (list, tuple)) else fps
                metadata.fps = float(video_fps)
            elif metadata.fps is None:
                metadata.fps = self.fps
            all_patches.append(patches)
            all_thw.append(thw)
            all_metadata.append(metadata)

        data = {
            "pixel_values_videos": np.concatenate(all_patches, axis=0).astype(
                np.float32
            ),
            "video_grid_thw": np.array(all_thw, dtype=np.int64),
        }
        if return_metadata:
            data["video_metadata"] = all_metadata
        return BatchFeature(data=data)

    def preprocess(self, videos: Any, **kwargs: Any) -> BatchFeature:
        return self(videos, **kwargs)
