# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3-Omni checkpoint reference implementation.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

"""LFM3-VL image/video preprocessing plus LFM3-Audio speech, with media spans."""

from transformers.feature_extraction_utils import BatchFeature

from ..base import install_auto_processor_patch, to_mlx
from ..lfm3_audio import audio
from ..lfm3_audio.processing_lfm3_audio import (
    Lfm3AudioFeatureExtractor,
    Lfm3AudioProcessor,
)
from ..lfm3_vl.processing_lfm3_vl import Lfm3VlProcessor

SPAN = "<|vision_start|>", "<|vision_end|>"

# processor_config.json["feature_extractor"] values the log-mel front end implements
_FEATURES = {
    "sampling_rate": audio.SAMPLE_RATE,
    "n_fft": audio.N_FFT,
    "win_length": audio.N_FFT,
    "hop_length": audio.HOP,
    "feature_size": audio.NUM_MEL_BINS,
    "log_floor": audio.LOG_FLOOR,
    "normalization_offset": 4.0,
    "normalization_divisor": 4.0,
    "encoder_stride": 8,
    "padding_side": "right",
}


class Lfm3OmniProcessor(Lfm3VlProcessor):
    supports_multiple_audio = True

    # transformers reads the processor's components from this signature
    def __init__(
        self,
        image_processor,
        tokenizer,
        video_processor,
        processor_config=None,
        **kwargs,
    ):
        super().__init__(
            image_processor, tokenizer, video_processor, processor_config, **kwargs
        )
        settings = self.processor_config.get("feature_extractor", {})
        changed = {k for k, v in _FEATURES.items() if settings.get(k, v) != v}
        if changed:
            raise ValueError(f"Unsupported LFM3-Omni feature settings: {changed}")
        self.feature_extractor = Lfm3AudioFeatureExtractor()
        # plain text generation stops where the model would start to speak
        self.additional_eos_token_ids = self.tokenizer.convert_tokens_to_ids(
            ["<|audio_start|>", "<|mixed_start|>"]
        )

    @staticmethod
    def wrap_media(text):
        """Every image, video and audio placeholder inside exactly one span pair,
        whether or not the chat template already wrapped it."""
        start, end = SPAN
        for token in ("<|image|>", "<|video|>", "<|audio|>"):
            text = text.replace(start + token + end, token)
            text = text.replace(token, start + token + end)
        return text

    @staticmethod
    def group_seconds(groups, fps):
        # the Omni reference divides the index sum, unlike LFM3-VL
        return (groups[:, 0] + groups[:, -1]) / (2 * fps)

    def __call__(
        self,
        text=None,
        images=None,
        videos=None,
        audio=None,
        audio_out_codes=None,
        fps=None,
        video_metadata=None,
        add_special_tokens=False,
        padding_side="left",
        return_tensors="mlx",
        **kwargs,
    ):
        """LFM3-VL's arguments plus LFM3-Audio's ``audio`` and ``audio_out_codes``
        (the loader passes only the arguments named here)."""
        texts = [text] if isinstance(text, str) else list(text or [])
        texts, speech = Lfm3AudioProcessor.expand_speech(
            [self.wrap_media(t) for t in texts], audio, audio_out_codes
        )
        data = {
            **super().__call__(
                texts,
                images,
                videos,
                fps=fps,
                video_metadata=video_metadata,
                add_special_tokens=add_special_tokens,
                padding_side=padding_side,
                return_tensors=None,
                **kwargs,
            )
        }
        data.update(speech)
        if return_tensors == "mlx":
            return BatchFeature(data=to_mlx(data))
        return BatchFeature(data=data, tensor_type=return_tensors)


install_auto_processor_patch("lfm3_omni", Lfm3OmniProcessor)
