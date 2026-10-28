# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3/LFM3-VL checkpoint reference implementations.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

from ..base import BaseModelConfig
from ..lfm3.config import ModelConfig as TextConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "lfm3_vl_vision"
    hidden_size: int = 1024
    intermediate_size: int = 2560
    num_hidden_layers: int = 14
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    conv_kernel_size: int = 3
    norm_eps: float = 1e-5
    layer_types: Optional[list[str]] = None
    rope_parameters: Optional[dict] = None

    def __post_init__(self):
        if self.layer_types is None or len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                "Vision layer_types must contain num_hidden_layers entries"
            )
        if not set(self.layer_types) <= {"conv", "full_attention"}:
            raise ValueError("Unsupported LFM3 vision layer type")
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_type": "axial", "rope_theta": 1e6}
        if self.rope_parameters.get("rope_type", "axial") != "axial":
            raise ValueError("LFM3 vision requires axial rotary embeddings")


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "lfm3_vl"
    downsample_factor: int = 2
    projector_hidden_size: int = 4096
    projector_hidden_act: str = "gelu"
    projector_bias: bool = True
    projector_layer_norm_eps: float = 1e-6
    image_token_id: int = 125030
    video_token_id: int = 125029
    eos_token_id: int = 124900

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig.from_dict(self.vision_config)
        if self.downsample_factor != self.vision_config.spatial_merge_size:
            raise ValueError(
                "Projector downsampling must match the vision spatial merge size"
            )
        if self.projector_hidden_act != "gelu":
            raise ValueError("LFM3-VL supports a GELU projector")
