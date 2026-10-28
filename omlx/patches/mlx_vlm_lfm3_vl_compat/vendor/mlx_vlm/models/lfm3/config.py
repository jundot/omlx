# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3/LFM3-VL checkpoint reference implementations.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

from ..base import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "lfm3"
    hidden_size: int = 2048
    intermediate_size: int = 7680
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: Optional[int] = None
    vocab_size: int = 128000
    norm_eps: float = 1e-5
    conv_L_cache: int = 7
    conv_bias: bool = False
    layer_types: Optional[list[str]] = None
    rope_parameters: Optional[dict] = None
    sliding_window: Optional[int] = 256
    use_sandwich_norm: bool = True
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 262144
    eos_token_id: int = 124900

    def __post_init__(self):
        self.head_dim = self.head_dim or self.hidden_size // self.num_attention_heads
        if self.layer_types is None:
            raise ValueError("LFM3 requires an explicit layer_types list")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain num_hidden_layers entries")
        if not set(self.layer_types) <= {"conv", "full_attention", "sliding_attention"}:
            raise ValueError("Unsupported LFM3 layer type")
        if "sliding_attention" in self.layer_types and not self.sliding_window:
            raise ValueError("Sliding attention requires a positive sliding_window")
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {"rope_theta": 1e6, "partial_rotary_factor": 0.5},
                "sliding_attention": {"rope_theta": 1e4},
            }


TextConfig = ModelConfig
