"""
Torch-free processor for MiniMax M3 VL.

Based on the Hugging Face Transformers MiniMax M3 VL processors:
https://github.com/huggingface/transformers/tree/main/src/transformers/models/minimax_m3_vl
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_processing_utils import BaseVideoProcessor

from ..base import install_auto_processor_patch, load_chat_template, to_mlx

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
MAX_RATIO = 200
DEFAULT_MIN_PIXELS = 4 * 28 * 28
DEFAULT_IMAGE_MAX_PIXELS = 451584
DEFAULT_VIDEO_MAX_PIXELS = 768 * 28 * 28
DEFAULT_IMAGE_MAX_TOTAL_PIXELS = 3584 * 3584
DEFAULT_VIDEO_MAX_TOTAL_PIXELS = 301_056_000
DEFAULT_MIN_SHORT_SIDE_PIXEL = 112

_IMAGE_PROCESSOR_KWARGS = {
    "do_resize",
    "resample",
    "do_rescale",
    "rescale_factor",
    "do_normalize",
    "image_mean",
    "image_std",
    "patch_size",
    "temporal_patch_size",
    "merge_size",
    "min_pixels",
    "max_pixels",
    "max_long_side_pixel",
    "max_total_pixels",
}

_VIDEO_PROCESSOR_KWARGS = _IMAGE_PROCESSOR_KWARGS | {
    "fps",
    "min_frames",
    "max_frames",
    "total_pixels",
    "video_metadata",
    "return_metadata",
}


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS,
    max_long_side_pixel: Optional[int] = None,
    max_total_pixels: Optional[int] = None,
    min_short_side_pixel: int = DEFAULT_MIN_SHORT_SIDE_PIXEL,
) -> Tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            "absolute aspect ratio must be smaller than "
            f"{MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )

    if max_long_side_pixel is None:
        h_bar = max(factor, round(height / factor) * factor)
        w_bar = max(factor, round(width / factor) * factor)
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
    else:
        long_side = max(height, width)
        short_side = min(height, width)
        scaled_height = float(height)
        scaled_width = float(width)
        if long_side > max_long_side_pixel:
            beta = max_long_side_pixel / long_side
            scaled_height = height * beta
            scaled_width = width * beta
        elif short_side < min_short_side_pixel:
            beta = min_short_side_pixel / short_side
            scaled_height = height * beta
            scaled_width = width * beta

        h_bar = max(factor, round(scaled_height / factor) * factor)
        w_bar = max(factor, round(scaled_width / factor) * factor)

    if max_total_pixels is not None and h_bar * w_bar > max_total_pixels:
        raise ValueError(
            f"image area {h_bar * w_bar} exceeds max_total_pixels "
            f"{max_total_pixels} after resizing"
        )
    return h_bar, w_bar


def _to_numpy_image(image: Any, do_convert_rgb: bool = True) -> np.ndarray:
    if isinstance(image, (str, Path)):
        image = Image.open(image)

    if hasattr(image, "convert"):
        if do_convert_rgb:
            image = image.convert("RGB")
        arr = np.asarray(image)
    else:
        arr = np.asarray(image)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {arr.shape}.")

    if arr.shape[0] in (1, 3, 4) and (
        arr.shape[-1] not in (1, 3, 4) or arr.shape[1] == arr.shape[2]
    ):
        pass
    elif arr.shape[-1] in (1, 3, 4):
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected image as CHW or HWC, got shape {arr.shape}.")

    if arr.shape[0] == 4:
        arr = arr[:3]
    if arr.shape[0] == 1 and do_convert_rgb:
        arr = np.repeat(arr, 3, axis=0)
    if arr.shape[0] != 3:
        raise ValueError(f"Expected image with 3 channels, got shape {arr.shape}.")

    return arr


def _to_pil_frame(frame_chw: np.ndarray) -> Image.Image:
    frame_hwc = np.transpose(frame_chw, (1, 2, 0))
    if frame_hwc.dtype != np.uint8:
        frame = frame_hwc.astype(np.float32)
        if frame.size and np.nanmax(frame) <= 1.0:
            frame = frame * 255.0
        frame_hwc = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame_hwc)


def _resize_frame(
    frame_chw: np.ndarray, height: int, width: int, resample
) -> np.ndarray:
    if frame_chw.shape[-2:] == (height, width):
        return frame_chw
    image = _to_pil_frame(frame_chw)
    image = image.resize((width, height), resample=resample)
    return np.transpose(np.asarray(image), (2, 0, 1))


def _normalize_pixels(
    pixels: np.ndarray,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: List[float],
    image_std: List[float],
) -> np.ndarray:
    pixels = pixels.astype(np.float32, copy=False)
    if do_rescale:
        pixels = pixels * rescale_factor
    if do_normalize:
        mean = np.asarray(image_mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(image_std, dtype=np.float32).reshape(3, 1, 1)
        pixels = (pixels - mean) / std
    return pixels


def _patchify(
    pixels: np.ndarray, patch_size: int, temporal_patch_size: int, merge_size: int
):
    if pixels.ndim != 4:
        raise ValueError(f"Expected pixels as (T, C, H, W), got shape {pixels.shape}.")

    pad = (-pixels.shape[0]) % temporal_patch_size
    if pad:
        pixels = np.concatenate(
            [pixels, np.repeat(pixels[-1:], pad, axis=0)], axis=0
        )

    frames, channels, height, width = pixels.shape
    factor = patch_size * merge_size
    if height % factor != 0 or width % factor != 0:
        raise ValueError(
            f"Image/video size {(height, width)} must be divisible by "
            f"patch_size * merge_size ({factor})."
        )

    grid_t = frames // temporal_patch_size
    grid_h = height // patch_size
    grid_w = width // patch_size

    patches = pixels.reshape(
        1,
        grid_t,
        temporal_patch_size,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    flat = patches.reshape(
        grid_t * grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size,
    )
    return flat, [grid_t, grid_h, grid_w]


def _to_numpy_video(video: Any, do_convert_rgb: bool = True) -> np.ndarray:
    if isinstance(video, (list, tuple)) and video and hasattr(video[0], "convert"):
        frames = [
            _to_numpy_image(frame, do_convert_rgb=do_convert_rgb) for frame in video
        ]
        return np.stack(frames, axis=0)

    arr = np.asarray(video)
    if arr.ndim != 4:
        raise ValueError(f"Expected video with 4 dimensions, got shape {arr.shape}.")

    if arr.shape[1] in (1, 3, 4) and (
        arr.shape[-1] not in (1, 3, 4) or arr.shape[2] == arr.shape[3]
    ):
        video_chw = arr
    elif arr.shape[-1] in (1, 3, 4):
        video_chw = np.transpose(arr, (0, 3, 1, 2))
    else:
        raise ValueError(f"Expected video as TCHW or THWC, got shape {arr.shape}.")

    if video_chw.shape[1] == 4:
        video_chw = video_chw[:, :3]
    if video_chw.shape[1] == 1 and do_convert_rgb:
        video_chw = np.repeat(video_chw, 3, axis=1)
    if video_chw.shape[1] != 3:
        raise ValueError(
            f"Expected video with 3 channels, got shape {video_chw.shape}."
        )
    return video_chw


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_video_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [value]
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if hasattr(first, "convert"):
            return [value]
        return list(value)
    return [value]


class MiniMaxM3VLImageProcessor(ImageProcessingMixin):
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        resample=Image.Resampling.BICUBIC,
        size: Optional[Dict[str, int]] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_IMAGE_MAX_PIXELS,
        max_long_side_pixel: Optional[int] = None,
        min_short_side_pixel: int = DEFAULT_MIN_SHORT_SIDE_PIXEL,
        max_total_pixels: int = DEFAULT_IMAGE_MAX_TOTAL_PIXELS,
        **kwargs,
    ):
        self.do_resize = do_resize
        self.resample = resample
        self.size = size or {"height": 672, "width": 672}
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or OPENAI_CLIP_MEAN
        self.image_std = image_std or OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_long_side_pixel = max_long_side_pixel
        self.min_short_side_pixel = min_short_side_pixel
        self.max_total_pixels = max_total_pixels

    def _process_one(self, image, **kwargs) -> Tuple[np.ndarray, List[int]]:
        do_resize = kwargs.get("do_resize", self.do_resize)
        resample = kwargs.get("resample", self.resample)
        do_rescale = kwargs.get("do_rescale", self.do_rescale)
        rescale_factor = kwargs.get("rescale_factor", self.rescale_factor)
        do_normalize = kwargs.get("do_normalize", self.do_normalize)
        image_mean = kwargs.get("image_mean", self.image_mean)
        image_std = kwargs.get("image_std", self.image_std)
        patch_size = kwargs.get("patch_size", self.patch_size)
        temporal_patch_size = kwargs.get(
            "temporal_patch_size", self.temporal_patch_size
        )
        merge_size = kwargs.get("merge_size", self.merge_size)
        min_pixels = kwargs.get("min_pixels", self.min_pixels)
        max_pixels = kwargs.get("max_pixels", self.max_pixels)
        max_long_side_pixel = kwargs.get(
            "max_long_side_pixel", self.max_long_side_pixel
        )
        max_total_pixels = kwargs.get("max_total_pixels", self.max_total_pixels)

        image = _to_numpy_image(image, do_convert_rgb=self.do_convert_rgb)
        _, height, width = image.shape
        if do_resize:
            height, width = smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_long_side_pixel=max_long_side_pixel,
                max_total_pixels=max_total_pixels,
                min_short_side_pixel=self.min_short_side_pixel,
            )
            image = _resize_frame(image, height, width, resample)

        image = _normalize_pixels(
            image,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
        )
        pixels = np.repeat(image[None, ...], temporal_patch_size, axis=0)
        return _patchify(pixels, patch_size, temporal_patch_size, merge_size)

    def __call__(self, images=None, **kwargs):
        return self.preprocess(images=images, **kwargs)

    def preprocess(self, images=None, return_tensors=None, **kwargs) -> BatchFeature:
        images = _as_list(images)
        if not images:
            return BatchFeature(
                data={
                    "pixel_values": np.zeros((0, 0), dtype=np.float32),
                    "image_grid_thw": np.zeros((0, 3), dtype=np.int64),
                },
                tensor_type=None,
            )

        pixel_values = []
        grid_thw = []
        image_kwargs = {
            k: v for k, v in kwargs.items() if k in _IMAGE_PROCESSOR_KWARGS
        }
        for image in images:
            patches, thw = self._process_one(image, **image_kwargs)
            pixel_values.append(patches)
            grid_thw.append(thw)

        data = {
            "pixel_values": np.concatenate(pixel_values, axis=0).astype(np.float32),
            "image_grid_thw": np.asarray(grid_thw, dtype=np.int64),
        }
        return BatchFeature(data=to_mlx(data) if return_tensors == "mlx" else data)

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs=None
    ) -> int:
        images_kwargs = images_kwargs or {}
        patch_size = images_kwargs.get("patch_size", self.patch_size)
        merge_size = images_kwargs.get("merge_size", self.merge_size)
        min_pixels = images_kwargs.get("min_pixels", self.min_pixels)
        max_pixels = images_kwargs.get("max_pixels", self.max_pixels)
        max_long_side_pixel = images_kwargs.get(
            "max_long_side_pixel", self.max_long_side_pixel
        )
        max_total_pixels = images_kwargs.get(
            "max_total_pixels", self.max_total_pixels
        )
        height, width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_long_side_pixel=max_long_side_pixel,
            max_total_pixels=max_total_pixels,
            min_short_side_pixel=self.min_short_side_pixel,
        )
        return (height // patch_size) * (width // patch_size)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_processor_type": self.__class__.__name__,
            "do_resize": self.do_resize,
            "size": self.size,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }


class MiniMaxM3VLVideoProcessor(BaseVideoProcessor):
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        resample=Image.Resampling.BICUBIC,
        size: Optional[Dict[str, int]] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        do_sample_frames: bool = False,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_VIDEO_MAX_PIXELS,
        total_pixels: int = int(64000 * 28 * 28 * 0.9),
        fps: float = 1.0,
        min_frames: int = 4,
        max_frames: int = 768,
        max_long_side_pixel: Optional[int] = None,
        min_short_side_pixel: int = DEFAULT_MIN_SHORT_SIDE_PIXEL,
        max_total_pixels: int = DEFAULT_VIDEO_MAX_TOTAL_PIXELS,
        **kwargs,
    ):
        self.do_resize = do_resize
        self.resample = resample
        self.size = size or {"height": 672, "width": 672}
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean or OPENAI_CLIP_MEAN
        self.image_std = image_std or OPENAI_CLIP_STD
        self.do_convert_rgb = do_convert_rgb
        self.do_sample_frames = do_sample_frames
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.max_long_side_pixel = max_long_side_pixel
        self.min_short_side_pixel = min_short_side_pixel
        self.max_total_pixels = max_total_pixels

    def _process_one(self, video, **kwargs) -> Tuple[np.ndarray, List[int]]:
        do_resize = kwargs.get("do_resize", self.do_resize)
        resample = kwargs.get("resample", self.resample)
        do_rescale = kwargs.get("do_rescale", self.do_rescale)
        rescale_factor = kwargs.get("rescale_factor", self.rescale_factor)
        do_normalize = kwargs.get("do_normalize", self.do_normalize)
        image_mean = kwargs.get("image_mean", self.image_mean)
        image_std = kwargs.get("image_std", self.image_std)
        patch_size = kwargs.get("patch_size", self.patch_size)
        temporal_patch_size = kwargs.get(
            "temporal_patch_size", self.temporal_patch_size
        )
        merge_size = kwargs.get("merge_size", self.merge_size)
        min_pixels = kwargs.get("min_pixels", self.min_pixels)
        max_pixels = kwargs.get("max_pixels", self.max_pixels)
        max_long_side_pixel = kwargs.get(
            "max_long_side_pixel", self.max_long_side_pixel
        )
        max_total_pixels = kwargs.get("max_total_pixels", self.max_total_pixels)

        video = _to_numpy_video(video, do_convert_rgb=self.do_convert_rgb)
        frames, _, height, width = video.shape
        if do_resize:
            height, width = smart_resize(
                height,
                width,
                factor=patch_size * merge_size,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_long_side_pixel=max_long_side_pixel,
                max_total_pixels=None,
                min_short_side_pixel=self.min_short_side_pixel,
            )
            if height * width * frames > max_total_pixels:
                raise ValueError(
                    f"video volume {height * width * frames} exceeds "
                    f"max_total_pixels {max_total_pixels} after resizing"
                )
            video = np.stack(
                [_resize_frame(frame, height, width, resample) for frame in video],
                axis=0,
            )

        video = np.stack(
            [
                _normalize_pixels(
                    frame,
                    do_rescale,
                    rescale_factor,
                    do_normalize,
                    image_mean,
                    image_std,
                )
                for frame in video
            ],
            axis=0,
        )
        return _patchify(video, patch_size, temporal_patch_size, merge_size)

    def __call__(self, videos=None, **kwargs):
        return self.preprocess(videos=videos, **kwargs)

    def preprocess(self, videos=None, return_tensors=None, **kwargs) -> BatchFeature:
        videos = _as_video_list(videos)
        if not videos:
            return BatchFeature(
                data={
                    "pixel_values_videos": np.zeros((0, 0), dtype=np.float32),
                    "video_grid_thw": np.zeros((0, 3), dtype=np.int64),
                },
                tensor_type=None,
            )

        video_kwargs = {
            k: v for k, v in kwargs.items() if k in _VIDEO_PROCESSOR_KWARGS
        }
        pixel_values = []
        grid_thw = []
        for video in videos:
            patches, thw = self._process_one(video, **video_kwargs)
            pixel_values.append(patches)
            grid_thw.append(thw)

        data = {
            "pixel_values_videos": np.concatenate(pixel_values, axis=0).astype(
                np.float32
            ),
            "video_grid_thw": np.asarray(grid_thw, dtype=np.int64),
        }
        if "video_metadata" in kwargs:
            data["video_metadata"] = kwargs["video_metadata"]
        return BatchFeature(data=to_mlx(data) if return_tensors == "mlx" else data)

    def get_number_of_video_patches(
        self, num_frames: int, height: int, width: int, videos_kwargs=None
    ) -> int:
        videos_kwargs = videos_kwargs or {}
        patch_size = videos_kwargs.get("patch_size", self.patch_size)
        temporal_patch_size = videos_kwargs.get(
            "temporal_patch_size", self.temporal_patch_size
        )
        merge_size = videos_kwargs.get("merge_size", self.merge_size)
        min_pixels = videos_kwargs.get("min_pixels", self.min_pixels)
        max_pixels = videos_kwargs.get("max_pixels", self.max_pixels)
        max_long_side_pixel = videos_kwargs.get(
            "max_long_side_pixel", self.max_long_side_pixel
        )
        height, width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_long_side_pixel=max_long_side_pixel,
            min_short_side_pixel=self.min_short_side_pixel,
        )
        padded_frames = (
            math.ceil(num_frames / temporal_patch_size) * temporal_patch_size
        )
        grid_t = padded_frames // temporal_patch_size
        return grid_t * (height // patch_size) * (width // patch_size)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_processor_type": self.__class__.__name__,
            "do_resize": self.do_resize,
            "size": self.size,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "do_convert_rgb": self.do_convert_rgb,
            "do_sample_frames": self.do_sample_frames,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "total_pixels": self.total_pixels,
            "fps": self.fps,
            "min_frames": self.min_frames,
            "max_frames": self.max_frames,
        }


def _load_json(
    pretrained_model_name_or_path, relative_name: str
) -> Optional[Dict[str, Any]]:
    local_path = Path(pretrained_model_name_or_path) / relative_name
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))
    try:
        from huggingface_hub import hf_hub_download

        fetched = Path(hf_hub_download(pretrained_model_name_or_path, relative_name))
        return json.loads(fetched.read_text(encoding="utf-8"))
    except Exception:
        return None


def _component_kwargs(raw: Dict[str, Any], allowed: set, defaults: Dict[str, Any]):
    out = dict(defaults)
    size = raw.get("size") or {}
    if "height" in size and "width" in size and "max_pixels" not in raw:
        out["max_pixels"] = int(size["height"]) * int(size["width"])
    if "shortest_edge" in size and "min_pixels" not in raw:
        out["min_pixels"] = size["shortest_edge"]
    if "longest_edge" in size and "max_pixels" not in raw:
        out["max_pixels"] = size["longest_edge"]

    for key in allowed:
        if key in raw and raw[key] is not None:
            out[key] = raw[key]
    return out


def _vision_defaults(pretrained_model_name_or_path) -> Dict[str, Any]:
    config = _load_json(pretrained_model_name_or_path, "config.json") or {}
    vision_config = config.get("vision_config") or {}
    compression = (
        config.get("img_token_compression_config")
        or vision_config.get("img_token_compression_config")
        or {}
    )
    defaults = {}
    if vision_config.get("patch_size") is not None:
        defaults["patch_size"] = vision_config["patch_size"]
    if compression.get("temporal_patch_size") is not None:
        defaults["temporal_patch_size"] = compression["temporal_patch_size"]
    elif vision_config.get("temporal_patch_size") is not None:
        defaults["temporal_patch_size"] = vision_config["temporal_patch_size"]
    if compression.get("spatial_merge_size") is not None:
        defaults["merge_size"] = compression["spatial_merge_size"]
    elif vision_config.get("spatial_merge_size") is not None:
        defaults["merge_size"] = vision_config["spatial_merge_size"]
    return defaults


def _image_kwargs(pretrained_model_name_or_path) -> Dict[str, Any]:
    proc_cfg = _load_json(pretrained_model_name_or_path, "processor_config.json") or {}
    raw = _load_json(pretrained_model_name_or_path, "preprocessor_config.json") or {}
    raw.update(proc_cfg.get("image_processor", {}) or {})
    return _component_kwargs(
        raw,
        _IMAGE_PROCESSOR_KWARGS | {"do_convert_rgb", "size"},
        _vision_defaults(pretrained_model_name_or_path),
    )


def _video_kwargs(pretrained_model_name_or_path) -> Dict[str, Any]:
    proc_cfg = _load_json(pretrained_model_name_or_path, "processor_config.json") or {}
    raw = _load_json(pretrained_model_name_or_path, "video_preprocessor_config.json")
    if raw is None:
        raw = proc_cfg.get("video_processor", {}) or {}
    return _component_kwargs(
        raw,
        _VIDEO_PROCESSOR_KWARGS | {"do_convert_rgb", "do_sample_frames", "size"},
        _vision_defaults(pretrained_model_name_or_path),
    )


class MiniMaxM3VLProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "video_processor"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    video_processor_class = "AutoVideoProcessor"

    IMAGE_TOKEN = "]<]image[>["
    VIDEO_TOKEN = "]<]video[>["
    VISION_START_TOKEN = "]<]start of image[>["
    VISION_END_TOKEN = "]<]end of image[>["

    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        image_processor = image_processor or MiniMaxM3VLImageProcessor()
        video_processor = video_processor or MiniMaxM3VLVideoProcessor()
        self.image_token = self.IMAGE_TOKEN
        self.video_token = self.VIDEO_TOKEN
        self.image_token_id = (
            tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN) if tokenizer else None
        )
        self.video_token_id = (
            tokenizer.convert_tokens_to_ids(self.VIDEO_TOKEN) if tokenizer else None
        )
        super().__init__(
            image_processor, tokenizer, video_processor, chat_template=chat_template
        )
        self.vision_start_token_id = (
            tokenizer.convert_tokens_to_ids(self.VISION_START_TOKEN)
            if tokenizer
            else None
        )
        self.vision_end_token_id = (
            tokenizer.convert_tokens_to_ids(self.VISION_END_TOKEN)
            if tokenizer
            else None
        )

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        merge_length = self.image_processor.merge_size**2
        grid = image_inputs["image_grid_thw"][image_idx]
        num_image_tokens = int(np.asarray(grid).prod() // merge_length)
        return (
            self.VISION_START_TOKEN
            + self.IMAGE_TOKEN * num_image_tokens
            + self.VISION_END_TOKEN
        )

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        merge_length = self.video_processor.merge_size**2
        grid_thw = np.asarray(video_inputs["video_grid_thw"][video_idx])
        grid_t = int(grid_thw[0])
        frame_seqlen = int(np.asarray(grid_thw[1:]).prod() // merge_length)
        metadata = video_inputs.get("video_metadata", [None] * (video_idx + 1))[
            video_idx
        ]
        temporal_patch_size = self.video_processor.temporal_patch_size
        chunk = ""
        for frame in range(grid_t):
            if (
                metadata is not None
                and getattr(metadata, "fps", None) is not None
                and getattr(metadata, "frames_indices", None) is not None
            ):
                frames_indices = metadata.frames_indices
                ts = (
                    frames_indices[
                        min(frame * temporal_patch_size, len(frames_indices) - 1)
                    ]
                    / metadata.fps
                )
                chunk += f"]<]{ts:.1f} seconds[>["
            elif (
                isinstance(metadata, dict)
                and metadata.get("fps") is not None
                and metadata.get("frames_indices") is not None
            ):
                frames_indices = metadata["frames_indices"]
                ts = (
                    frames_indices[
                        min(frame * temporal_patch_size, len(frames_indices) - 1)
                    ]
                    / metadata["fps"]
                )
                chunk += f"]<]{ts:.1f} seconds[>["
            chunk += (
                self.VISION_START_TOKEN
                + self.VIDEO_TOKEN * frame_seqlen
                + self.VISION_END_TOKEN
            )
        return chunk

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        data = {}
        if image_sizes is not None:
            merge_size = kwargs.get("merge_size", self.image_processor.merge_size)
            patches = [
                self.image_processor.get_number_of_image_patches(*size, kwargs)
                for size in image_sizes
            ]
            data["num_image_patches"] = patches
            data["num_image_tokens"] = [num // merge_size**2 for num in patches]
        if video_sizes is not None:
            merge_size = kwargs.get("merge_size", self.video_processor.merge_size)
            patches = [
                self.video_processor.get_number_of_video_patches(*size, kwargs)
                for size in video_sizes
            ]
            data["num_video_patches"] = patches
            data["num_video_tokens"] = [num // merge_size**2 for num in patches]
        return data

    def __call__(
        self,
        images: Optional[Any] = None,
        text: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                List[TextInput],
                List[PreTokenizedInput],
            ]
        ] = None,
        videos: Optional[Any] = None,
        padding: bool = True,
        padding_side: Optional[str] = None,
        add_special_tokens: bool = False,
        return_tensors: Optional[str] = "mlx",
        return_mm_token_type_ids: Optional[bool] = None,
        **kwargs,
    ) -> BatchFeature:
        image_inputs = {}
        video_inputs = {}

        if images is not None:
            image_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in _IMAGE_PROCESSOR_KWARGS
            }
            image_inputs = self.image_processor(
                images=images, return_tensors=None, **image_kwargs
            )

        if videos is not None:
            video_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in _VIDEO_PROCESSOR_KWARGS
            }
            video_inputs = self.video_processor(
                videos=videos, return_tensors=None, **video_kwargs
            )

        if text is None:
            text = [""]
        elif not isinstance(text, list):
            text = [text]
        text = ["" if item is None else str(item) for item in text]

        if image_inputs:
            image_placeholder = "<|minimax_m3_image_placeholder|>"
            image_idx = 0
            num_images = len(image_inputs["image_grid_thw"])
            for i, prompt in enumerate(text):
                while self.IMAGE_TOKEN in prompt:
                    if image_idx >= num_images:
                        raise ValueError(
                            "More image tokens were provided than images."
                        )
                    prompt = prompt.replace(
                        self.IMAGE_TOKEN,
                        self.replace_image_token(image_inputs, image_idx).replace(
                            self.IMAGE_TOKEN, image_placeholder
                        ),
                        1,
                    )
                    image_idx += 1
                text[i] = prompt.replace(image_placeholder, self.IMAGE_TOKEN)

        if video_inputs:
            video_placeholder = "<|minimax_m3_video_placeholder|>"
            video_idx = 0
            num_videos = len(video_inputs["video_grid_thw"])
            for i, prompt in enumerate(text):
                while self.VIDEO_TOKEN in prompt:
                    if video_idx >= num_videos:
                        raise ValueError(
                            "More video tokens were provided than videos."
                        )
                    prompt = prompt.replace(
                        self.VIDEO_TOKEN,
                        self.replace_video_token(video_inputs, video_idx).replace(
                            self.VIDEO_TOKEN, video_placeholder
                        ),
                        1,
                    )
                    video_idx += 1
                text[i] = prompt.replace(video_placeholder, self.VIDEO_TOKEN)

        tokenizer_kwargs = dict(kwargs)
        for key in _IMAGE_PROCESSOR_KWARGS | _VIDEO_PROCESSOR_KWARGS:
            tokenizer_kwargs.pop(key, None)
        tokenizer_kwargs.pop("video_metadata", None)
        tokenizer_kwargs.pop("return_metadata", None)
        tokenizer_kwargs.pop("fps", None)
        if padding_side is not None:
            tokenizer_kwargs["padding_side"] = padding_side

        text_inputs = self.tokenizer(
            text,
            padding=padding,
            add_special_tokens=add_special_tokens,
            return_tensors=None,
            **tokenizer_kwargs,
        )
        text_inputs = dict(text_inputs)

        if return_mm_token_type_ids:
            input_ids = np.asarray(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(input_ids)
            if self.image_token_id is not None:
                mm_token_type_ids[input_ids == self.image_token_id] = 1
            if self.video_token_id is not None:
                mm_token_type_ids[input_ids == self.video_token_id] = 2
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        data = {**text_inputs, **dict(image_inputs), **dict(video_inputs)}
        data.pop("video_metadata", None)
        if return_tensors == "mlx":
            data = to_mlx(data)
        return BatchFeature(data=data)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self,
        generated_outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = (
            self.tokenizer.model_input_names if self.tokenizer is not None else []
        )
        image_processor_input_names = self.image_processor.model_input_names
        video_processor_input_names = self.video_processor.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
                + image_processor_input_names
                + video_processor_input_names
                + ["mm_token_type_ids"]
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer_kwargs = dict(kwargs)
        chat_template = tokenizer_kwargs.pop("chat_template", None)
        tokenizer_kwargs.pop("return_tensors", None)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, **tokenizer_kwargs
        )
        if Path(pretrained_model_name_or_path).exists():
            load_chat_template(tokenizer, pretrained_model_name_or_path)

        proc_cfg = (
            _load_json(pretrained_model_name_or_path, "processor_config.json") or {}
        )
        chat_template = (
            chat_template
            or proc_cfg.get("chat_template")
            or getattr(tokenizer, "chat_template", None)
        )

        image_processor = MiniMaxM3VLImageProcessor(
            **_image_kwargs(pretrained_model_name_or_path)
        )
        video_processor = MiniMaxM3VLVideoProcessor(
            **_video_kwargs(pretrained_model_name_or_path)
        )
        return cls(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
        )

    def save_pretrained(self, save_directory, **kwargs):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            tokenizer_files = self.tokenizer.save_pretrained(str(save_dir), **kwargs)
            saved_files.extend(str(path) for path in tokenizer_files)

        processor_config = {
            "processor_class": self.__class__.__name__,
            "image_processor": self.image_processor.to_dict(),
            "video_processor": self.video_processor.to_dict(),
        }
        if self.chat_template is not None:
            processor_config["chat_template"] = self.chat_template
        processor_path = save_dir / "processor_config.json"
        processor_path.write_text(
            json.dumps(processor_config, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        saved_files.append(str(processor_path))

        preprocessor_path = save_dir / "preprocessor_config.json"
        preprocessor_path.write_text(
            json.dumps(self.image_processor.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        saved_files.append(str(preprocessor_path))

        video_preprocessor_path = save_dir / "video_preprocessor_config.json"
        video_preprocessor_path.write_text(
            json.dumps(self.video_processor.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        saved_files.append(str(video_preprocessor_path))

        if self.chat_template is not None:
            chat_template_path = save_dir / "chat_template.json"
            chat_template_path.write_text(
                json.dumps({"chat_template": self.chat_template}, indent=2),
                encoding="utf-8",
            )
            saved_files.append(str(chat_template_path))

        return saved_files


MiniMaxVLProcessor = MiniMaxM3VLProcessor

__all__ = [
    "MiniMaxM3VLImageProcessor",
    "MiniMaxM3VLProcessor",
    "MiniMaxM3VLVideoProcessor",
    "MiniMaxVLProcessor",
    "smart_resize",
]

install_auto_processor_patch("minimax_m3_vl", MiniMaxM3VLProcessor)
