"""Torch-free MLX Processor for WeMM-Video compatible with both standalone CLI and oMLX/mlx-vlm."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer, BatchFeature, WhisperFeatureExtractor
from transformers.processing_utils import ProcessorMixin

from .media import (
    AUDIO_MARKER,
    IMAGE_MARKER,
    VIDEO_MARKER,
    audio_features,
    build_audio_token_ids,
    build_video_token_ids,
    decode_audio,
    decode_video,
    load_image,
    preprocess_frames_to_patches,
    resolve_media,
    tokens_per_audio_segment,
)


class _SimpleImageProcessorMeta:
    """Minimal metadata shim expected by mlx-vlm and oMLX boundary helpers."""

    def __init__(self, patch_size=16, temporal_patch_size=2, merge_size=2, min_pixels=65536, max_pixels=16777216, fps=2.0, min_frames=4, max_frames=128):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.fps = fps
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        self.image_mean = (0.5, 0.5, 0.5)
        self.image_std = (0.5, 0.5, 0.5)
        self.model_input_names = ["pixel_values", "image_grid_thw"]

    def preprocess(self, images, **kwargs):
        if isinstance(images, (list, tuple)):
            return [load_image(im) for im in images]
        return load_image(images)


class WeMMProcessor(ProcessorMixin):
    """Unified omni-modal processor for WeMM-Video returning MLX arrays without PyTorch."""

    attributes = ["tokenizer", "feature_extractor"]
    tokenizer_class = "AutoTokenizer"
    feature_extractor_class = "WhisperFeatureExtractor"
    supports_multiple_audio = True

    def __init__(
        self,
        tokenizer,
        feature_extractor,
        audio_config: Optional[Dict[str, Any]] = None,
        *,
        fps: float = 2.0,
        min_frames: int = 4,
        max_frames: int = 128,
        total_pixels: int = 25165824,
        min_pixels: int = 4096,
        image_min_pixels: int = 65536,
        image_max_pixels: int = 16777216,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        system_prompt: str = "You are a helpful assistant.",
        timestamp_interval_sec: float = 1.0,
        chat_template: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.chat_template = chat_template or getattr(tokenizer, "chat_template", None)
        self.audio_config = copy.deepcopy(
            audio_config
            or {
                "num_mel_bins": 128,
                "second_per_window": 0.1,
                "second_stride": 0.1,
                "qformer_config": {"query_length": 1},
            }
        )
        self.fps = float(fps)
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.total_pixels = int(total_pixels)
        self.min_pixels = int(min_pixels)
        self.image_min_pixels = int(image_min_pixels)
        self.image_max_pixels = int(image_max_pixels)
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.merge_size = int(merge_size)
        self.system_prompt = system_prompt
        self.timestamp_interval_sec = float(timestamp_interval_sec)
        self._recent_video_paths: List[str] = []

        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.audio_token = "<|audio_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids(self.vision_end_token)
        self.image_processor = _SimpleImageProcessorMeta(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            min_pixels=image_min_pixels,
            max_pixels=image_max_pixels,
        )
        self.video_processor = _SimpleImageProcessorMeta(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            min_pixels=min_pixels,
            max_pixels=total_pixels,
            fps=fps,
            min_frames=min_frames,
            max_frames=max_frames,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs) -> "WeMMProcessor":
        model_dir = Path(pretrained_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=False,
        )
        jinja_path = model_dir / "chat_template.jinja"
        if jinja_path.is_file():
            tokenizer.chat_template = jinja_path.read_text(encoding="utf-8")
        if getattr(tokenizer, "chat_template", None) and "<|audio_start|><|audio_pad|><|audio_end|>" not in tokenizer.chat_template:
            tokenizer.chat_template = tokenizer.chat_template.replace(
                "{{- '<|vision_start|><|video_pad|><|vision_end|>' }}\n            {%- elif 'text' in item %}",
                "{{- '<|vision_start|><|video_pad|><|vision_end|>' }}\n"
                "            {%- elif 'audio' in item or 'audio_url' in item or 'input_audio' in item or item.type == 'audio' %}\n"
                "                {{- '<|audio_start|><|audio_pad|><|audio_end|>' }}\n"
                "            {%- elif 'text' in item %}",
                1,
            )

        cfg = {}
        cfg_path = model_dir / "config.json"
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        proc_cfg = {}
        proc_path = model_dir / "processor_config.json"
        if proc_path.is_file():
            proc_cfg = json.loads(proc_path.read_text(encoding="utf-8"))
        preproc_cfg = {}
        preproc_path = model_dir / "preprocessor_config.json"
        if preproc_path.is_file():
            preproc_cfg = json.loads(preproc_path.read_text(encoding="utf-8"))
        vid_cfg = {}
        vid_path = model_dir / "video_preprocessor_config.json"
        if vid_path.is_file():
            vid_cfg = json.loads(vid_path.read_text(encoding="utf-8"))

        audio_cfg = cfg.get("audio_config") or proc_cfg.get("audio_config") or {}
        num_mel_bins = int(audio_cfg.get("num_mel_bins", 128))
        extractor = WhisperFeatureExtractor(
            feature_size=num_mel_bins,
            sampling_rate=16000,
            hop_length=160,
            chunk_length=30,
        )
        defaults = proc_cfg.get("wemm_defaults", {})
        img_size = preproc_cfg.get("size", {})
        vid_size = vid_cfg.get("size", {})

        return cls(
            tokenizer=tokenizer,
            feature_extractor=extractor,
            audio_config=audio_cfg,
            fps=kwargs.get("fps", defaults.get("fps", 2.0)),
            min_frames=kwargs.get("min_frames", defaults.get("min_frames", 4)),
            max_frames=kwargs.get("max_frames", defaults.get("max_frames", 128)),
            total_pixels=kwargs.get("total_pixels", defaults.get("total_pixels", vid_size.get("longest_edge", 25165824))),
            min_pixels=kwargs.get("min_pixels", defaults.get("min_pixels", vid_size.get("shortest_edge", 4096))),
            image_min_pixels=img_size.get("shortest_edge", 65536),
            image_max_pixels=img_size.get("longest_edge", 16777216),
            patch_size=preproc_cfg.get("patch_size", 16),
            temporal_patch_size=preproc_cfg.get("temporal_patch_size", 2),
            merge_size=preproc_cfg.get("merge_size", 2),
            system_prompt=kwargs.get("system_prompt", defaults.get("system_prompt", "You are a helpful assistant.")),
            timestamp_interval_sec=kwargs.get("timestamp_interval_sec", defaults.get("timestamp_interval_sec", 1.0)),
            chat_template=getattr(tokenizer, "chat_template", None),
        )

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
            "audio_feature",
            "input_features",
        ]

    def _normalize_messages(
        self, messages: List[Dict[str, Any]], use_audio_in_video: bool = False
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        normalized: List[Dict[str, Any]] = []
        routes: List[Dict[str, Any]] = []
        for message in messages:
            row = {k: v for k, v in message.items() if k != "content"}
            content = message.get("content")
            if isinstance(content, str):
                row["content"] = content
            else:
                parts: List[str] = []
                for item in content or []:
                    kind = item["type"]
                    if kind == "image_url":
                        payload = item["image_url"]
                        src = payload["url"] if isinstance(payload, dict) else payload
                        item = {"type": "image", "image": src}
                        kind = "image"
                    elif kind in ("video_url", "video_audio_url"):
                        payload = item.get("video_url")
                        nested = payload if isinstance(payload, dict) else {}
                        video_src = nested.get("url") if nested else payload
                        audio = item.get("audio_url") or nested.get("audio")
                        audio_src = audio.get("url") if isinstance(audio, dict) else audio
                        if video_src:
                            kind = "video_audio" if (audio_src or nested.get("use_audio") or item.get("use_audio")) else "video"
                            item = {**item, **nested, "type": kind, "video": video_src}
                            if audio_src:
                                item["audio"] = audio_src
                        elif kind == "video_audio_url" and audio_src:
                            kind, item = "audio", {"type": "audio", "audio": audio_src}
                        else:
                            raise ValueError("Missing video URL in media item")
                    elif kind in ("audio_url", "input_audio"):
                        payload = item[kind]
                        if kind == "audio_url":
                            src = payload["url"] if isinstance(payload, dict) else payload
                        else:
                            src = f"data:audio/{payload['format']};base64,{payload['data']}"
                        item = {"type": "audio", "audio": src}
                        kind = "audio"

                    if kind == "video" and item.get("use_audio_in_video", use_audio_in_video):
                        kind = "video_audio"

                    if kind == "text":
                        parts.append(item["text"])
                    elif kind in ("image", "video", "audio", "video_audio"):
                        key = "video" if kind == "video_audio" else kind
                        src = item.get(key, item.get("path"))
                        if src is not None:
                            routes.append({**item, "type": kind, "source": src})
                        marker = {
                            "image": IMAGE_MARKER,
                            "video": VIDEO_MARKER,
                            "video_audio": VIDEO_MARKER,
                            "audio": AUDIO_MARKER,
                        }[kind]
                        parts.append(marker)
                    else:
                        raise ValueError(f"Unsupported content type: {kind}")
                row["content"] = "".join(parts)
            normalized.append(row)
        return normalized, routes

    def apply_chat_template(
        self,
        conversation: List[Dict[str, Any]],
        *,
        tokenize: bool = False,
        return_dict: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        use_audio_in_video: bool = False,
        **kwargs,
    ):
        if tokenize:
            result = self.prepare_messages(
                conversation,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
                use_audio_in_video=use_audio_in_video,
                **kwargs,
            )
            return result if return_dict else result["input_ids"]
        normalized, _ = self._normalize_messages(conversation, use_audio_in_video=use_audio_in_video)
        return self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            **kwargs,
        )

    def prepare_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        use_audio_in_video: bool = False,
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        total_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        timestamp_interval_sec: Optional[float] = None,
    ) -> BatchFeature:
        normalized, routes = self._normalize_messages(messages, use_audio_in_video=use_audio_in_video)
        prompt = self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        return self._assemble_inputs(
            prompt,
            routes,
            fps=fps,
            num_frames=num_frames,
            min_frames=min_frames,
            max_frames=max_frames,
            total_pixels=total_pixels,
            min_pixels=min_pixels,
            timestamp_interval_sec=timestamp_interval_sec,
        )

    def _assemble_inputs(
        self,
        prompt: str,
        routes: List[Dict[str, Any]],
        *,
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        total_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        timestamp_interval_sec: Optional[float] = None,
    ) -> BatchFeature:
        image_pixels: List[np.ndarray] = []
        image_grids: List[np.ndarray] = []
        video_pixels: List[np.ndarray] = []
        video_grids: List[np.ndarray] = []
        audio_feats_ordered: List[np.ndarray] = []

        blocks: Dict[str, List[List[int]]] = {"image": [], "video": [], "audio": []}
        block_features: Dict[str, List[Optional[np.ndarray]]] = {"image": [], "video": [], "audio": []}
        metadata: List[Dict[str, Any]] = []

        interval_default = float(
            self.timestamp_interval_sec if timestamp_interval_sec is None else timestamp_interval_sec
        )
        window = float(self.audio_config.get("second_per_window", 0.1))
        seg_tokens = tokens_per_audio_segment(self.audio_config)

        for route in routes:
            kind = route["type"]
            source = resolve_media(route["source"])
            feature: Optional[np.ndarray] = None
            info: Dict[str, Any] = {"type": kind}

            if kind == "image":
                img = load_image(source)
                arr = np.asarray(img, dtype=np.uint8)
                pix, grid = preprocess_frames_to_patches(
                    arr,
                    patch_size=self.patch_size,
                    temporal_patch_size=self.temporal_patch_size,
                    merge_size=self.merge_size,
                    min_pixels=self.image_min_pixels,
                    max_pixels=self.image_max_pixels,
                    is_video=False,
                )
                image_pixels.append(pix)
                image_grids.append(grid)
                n_img_tokens = int(np.prod(grid[0])) // (self.merge_size**2)
                blocks["image"].append(
                    [self.vision_start_token_id]
                    + [self.image_token_id] * n_img_tokens
                    + [self.vision_end_token_id]
                )
                info["grid"] = grid[0].tolist()
            elif kind == "audio":
                samples = decode_audio(source)
                duration = len(samples) / 16000.0
                feature = audio_features(samples, self.feature_extractor, pcm16=True)
                count = int(feature.shape[0]) * seg_tokens
                interval = float(route.get("timestamp_interval_sec", interval_default))
                blocks["audio"].append(
                    build_audio_token_ids(
                        self.tokenizer,
                        count=count,
                        window=window,
                        duration=duration,
                        timestamp_interval_sec=interval,
                    )
                )
                info.update(duration=duration, segments=int(feature.shape[0]), audio_tokens=count)
            else:
                eff_fps = float(route.get("fps", self.fps if fps is None else fps))
                eff_num = route.get("num_frames", num_frames)
                eff_min = int(route.get("min_frames", self.min_frames if min_frames is None else min_frames))
                eff_max = int(route.get("max_frames", self.max_frames if max_frames is None else max_frames))
                eff_total_px = int(route.get("total_pixels", self.total_pixels if total_pixels is None else total_pixels))
                eff_min_px = int(route.get("min_pixels", self.min_pixels if min_pixels is None else min_pixels))

                frames, vid_info = decode_video(
                    source,
                    fps=eff_fps,
                    num_frames=eff_num,
                    min_frames=eff_min,
                    max_frames=eff_max,
                )
                ext_meta = route.get("video_metadata")
                if ext_meta is not None:
                    if isinstance(ext_meta, dict):
                        for mk in ("total_num_frames", "fps", "frames_indices", "duration"):
                            if ext_meta.get(mk) is not None:
                                vid_info[mk] = ext_meta[mk]
                    else:
                        for mk in ("total_num_frames", "fps", "frames_indices", "duration"):
                            if getattr(ext_meta, mk, None) is not None:
                                vid_info[mk] = getattr(ext_meta, mk)
                bundled_audio = vid_info.pop("audio_bytes", None)
                pix, grid = preprocess_frames_to_patches(
                    frames,
                    patch_size=self.patch_size,
                    temporal_patch_size=self.temporal_patch_size,
                    merge_size=self.merge_size,
                    min_pixels=eff_min_px,
                    max_pixels=eff_total_px,
                    is_video=True,
                )
                video_pixels.append(pix)
                video_grids.append(grid)

                audio_count, audio_duration = 0, None
                if kind == "video_audio" or bundled_audio is not None:
                    audio_src = route.get("audio", bundled_audio if bundled_audio is not None else source)
                    if isinstance(audio_src, np.ndarray) and route.get("audio_path"):
                        audio_src = route["audio_path"]
                    try:
                        samples = decode_audio(audio_src)
                    except (RuntimeError, ValueError):
                        info["audio_fallback"] = "no audio stream; video only"
                    else:
                        audio_duration = len(samples) / 16000.0
                        feature = audio_features(samples, self.feature_extractor, pcm16=False)
                        audio_count = int(feature.shape[0]) * seg_tokens
                        info["segments"] = int(feature.shape[0])

                vid_ids, groups, times = build_video_token_ids(
                    self.tokenizer,
                    tuple(grid[0].tolist()),
                    vid_info,
                    temporal_patch_size=self.temporal_patch_size,
                    merge_size=self.merge_size,
                    audio_count=audio_count,
                    audio_duration=audio_duration,
                    audio_window=window,
                )
                blocks["video"].append(vid_ids)
                info.update(
                    vid_info,
                    grid=grid[0].tolist(),
                    audio_tokens=audio_count,
                    audio_duration=audio_duration,
                    audio_groups=groups,
                    timestamps=times,
                )

            block_kind = "video" if kind == "video_audio" else kind
            block_features[block_kind].append(feature)
            metadata.append(info)

        source_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        patterns = {
            "image": self.tokenizer.encode(IMAGE_MARKER, add_special_tokens=False),
            "video": self.tokenizer.encode(VIDEO_MARKER, add_special_tokens=False),
            "audio": self.tokenizer.encode(AUDIO_MARKER, add_special_tokens=False),
        }
        offsets = {"image": 0, "video": 0, "audio": 0}
        out_ids: List[int] = []
        idx = 0
        while idx < len(source_ids):
            matched = False
            for kind, pat in patterns.items():
                if pat and source_ids[idx : idx + len(pat)] == pat:
                    if offsets[kind] >= len(blocks[kind]):
                        raise ValueError(f"Extra {kind} placeholder in prompt")
                    out_ids.extend(blocks[kind][offsets[kind]])
                    feat = block_features[kind][offsets[kind]]
                    if feat is not None:
                        audio_feats_ordered.append(feat)
                    offsets[kind] += 1
                    idx += len(pat)
                    matched = True
                    break
            if not matched:
                out_ids.append(source_ids[idx])
                idx += 1

        data: Dict[str, Any] = {
            "input_ids": mx.array([out_ids], dtype=mx.int32),
            "attention_mask": mx.ones((1, len(out_ids)), dtype=mx.int32),
        }
        if image_pixels:
            data["pixel_values"] = mx.array(np.concatenate(image_pixels, axis=0))
            data["image_grid_thw"] = mx.array(np.concatenate(image_grids, axis=0), dtype=mx.int32)
        if video_pixels:
            data["pixel_values_videos"] = mx.array(np.concatenate(video_pixels, axis=0))
            data["video_grid_thw"] = mx.array(np.concatenate(video_grids, axis=0), dtype=mx.int32)
        if audio_feats_ordered:
            stacked_audio = mx.array(np.concatenate(audio_feats_ordered, axis=0))
            data["audio_feature"] = stacked_audio
            data["input_features"] = stacked_audio

        result = BatchFeature(data=data)
        result.media_metadata = metadata
        result.prompt_text = prompt
        return result

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        audio=None,
        audios=None,
        messages=None,
        use_audio_in_video: bool = True,
        fps: Optional[float] = None,
        video_metadata=None,
        **kwargs,
    ) -> BatchFeature:
        """Dual-mode entry point: WeMM messages API + mlx-vlm/oMLX prepare_inputs API."""
        if messages is not None:
            return self.prepare_messages(
                messages,
                use_audio_in_video=use_audio_in_video,
                fps=fps,
                **{k: v for k, v in kwargs.items() if k in (
                    "add_generation_prompt", "enable_thinking", "num_frames",
                    "min_frames", "max_frames", "total_pixels", "min_pixels",
                    "timestamp_interval_sec",
                )},
            )

        prompt = text[0] if isinstance(text, list) else text
        if not isinstance(prompt, str):
            raise ValueError("Expected prompt string or messages list")

        routes: List[Dict[str, Any]] = []
        if images is not None:
            img_list = images if isinstance(images, list) else [images]
            for img in img_list:
                routes.append({"type": "image", "source": img})

        audio_list = audios if audios is not None else audio
        if videos is not None:
            vid_list = videos if isinstance(videos, list) else [videos]
            meta_list = video_metadata if isinstance(video_metadata, (list, tuple)) else ([video_metadata] * len(vid_list) if video_metadata is not None else [None] * len(vid_list))
            recent_paths = list(self._recent_video_paths)
            self._recent_video_paths.clear()
            for idx, vid in enumerate(vid_list):
                entry: Dict[str, Any] = {
                    "type": "video_audio" if use_audio_in_video else "video",
                    "source": vid,
                }
                if isinstance(vid, np.ndarray) and idx < len(recent_paths):
                    entry["audio_path"] = recent_paths[idx]
                if fps is not None:
                    entry["fps"] = fps[idx] if isinstance(fps, (list, tuple)) else fps
                if idx < len(meta_list) and meta_list[idx] is not None:
                    entry["video_metadata"] = meta_list[idx]
                routes.append(entry)

        if audio_list is not None:
            aud_list = audio_list if isinstance(audio_list, list) else [audio_list]
            for aud in aud_list:
                routes.append({"type": "audio", "source": aud})

        if audio_list and AUDIO_MARKER not in prompt and "<|audio_pad|>" not in prompt:
            prompt = prompt.replace("<|im_start|>user\n", f"<|im_start|>user\n{AUDIO_MARKER}\n", 1)

        return self._assemble_inputs(prompt, routes, fps=fps if isinstance(fps, (int, float)) else None)
