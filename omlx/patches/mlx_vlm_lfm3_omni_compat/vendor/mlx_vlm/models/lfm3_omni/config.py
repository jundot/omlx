# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3-Omni checkpoint reference implementation.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional

from ..lfm3.config import ModelConfig as Lfm3Config
from ..lfm3_audio.audio import AudioEncoderConfig
from ..lfm3_vl.config import ModelConfig as Lfm3VlConfig
from ..lfm3_vl.config import TextConfig, VisionConfig


@dataclass
class ModelConfig(Lfm3VlConfig):
    """The canonical layout: LFM3-VL's text and vision fields plus LFM3-Audio's
    speech fields. ``lm_config`` is the ``text_config`` the audio model reads."""

    audio_encoder_config: Optional[dict] = None
    depth_config: Optional[dict] = None
    model_type: str = "lfm3_omni"
    num_codebooks: int = 12
    codebook_size: int = 2017
    audio_eos_code: int = 2016
    maskgit_schedule: list = field(default_factory=lambda: [4, 4, 4])
    interleaved_n_text: int = 6
    interleaved_n_audio: int = 12
    audio_token_id: int = 125031
    audio_out_token_id: int = 125032
    audio_start_token_id: int = 125012
    text_end_token_id: int = 125014
    mixed_start_token_id: int = 125015
    eos_token_id: list = field(default_factory=lambda: [124900, 125014])

    def __post_init__(self):
        super().__post_init__()
        if self.audio_encoder_config is None or self.depth_config is None:
            raise ValueError("LFM3-Omni requires audio_encoder_config and depth_config")
        if isinstance(self.audio_encoder_config, dict):
            self.audio_encoder_config = AudioEncoderConfig.from_dict(
                self.audio_encoder_config
            )
        if isinstance(self.depth_config, dict):
            self.depth_config = Lfm3Config.from_dict(self.depth_config)
        if sum(self.maskgit_schedule) != self.num_codebooks:
            raise ValueError("maskgit_schedule must reveal every codebook once")

    @property
    def lm_config(self):
        # the loader replaces text_config after __post_init__
        return self.text_config


__all__ = ["ModelConfig", "TextConfig", "VisionConfig"]
