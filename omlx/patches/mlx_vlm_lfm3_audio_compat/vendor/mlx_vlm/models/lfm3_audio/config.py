# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3-Audio checkpoint reference implementation.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from ..base import BaseModelConfig
from ..lfm3.config import ModelConfig as Lfm3Config
from .audio import AudioEncoderConfig


@dataclass
class ModelConfig(BaseModelConfig):
    """The checkpoint's training-export layout (``lfm``, ``encoder``, ``depthformer``).

    The loader replaces any ``text_config`` / ``audio_config`` attribute with a
    config built from an empty dict, so the backbone, encoder and depth decoder
    configs live in ``lm_config``, ``audio_encoder_config`` and ``depth_config``,
    which is what ``Model`` reads.
    """

    lfm: dict
    encoder: dict
    depthformer: dict
    model_type: str = "lfm3_audio"
    codebooks: int = 12
    audio_vocab_size: int = 2017
    maskgit_steps: int = 3
    interleaved_n_text: int = 6
    interleaved_n_audio: int = 12
    audio_token_id: int = 125031
    audio_out_token_id: int = 125032
    audio_start_token_id: int = 125012
    text_end_token_id: int = 125014
    mixed_start_token_id: int = 125015
    eos_token_id: list = field(default_factory=lambda: [124900, 125014])

    def __post_init__(self):
        self.lm_config = Lfm3Config.from_dict(self.lfm)
        self.audio_encoder_config = AudioEncoderConfig(
            text_config=self.encoder["lfm"],
            conv_channels=self.encoder["dim_conv"],
            hidden_size=self.encoder["dim_out"],
            num_mel_bins=self.encoder["dim_in"],
            sliding_window=self.encoder["sliding_window"],
        )
        self.depth_config = Lfm3Config.from_dict(self.depthformer)
        self.num_codebooks = self.codebooks
        self.codebook_size = self.audio_vocab_size
        self.audio_eos_code = self.audio_vocab_size - 1
        steps = self.maskgit_steps
        self.maskgit_schedule = [self.codebooks // steps] * steps
        self.maskgit_schedule[-1] += self.codebooks % steps
