# SPDX-License-Identifier: Apache-2.0
"""Configuration objects for the Qwen4-Exp mlx-vlm port."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..qwen3_5_moe.config import ModelConfig as Qwen3_5MoeModelConfig
from ..qwen3_5_moe.config import TextConfig as Qwen3_5MoeTextConfig
from ..qwen3_5_moe.config import VisionConfig as Qwen3_5MoeVisionConfig
from ..qwen3_vl.config import _config_kwargs, _maybe_deserialize_config


@dataclass
class VisionConfig(Qwen3_5MoeVisionConfig):
    model_type: str = "qwen4_exp"

    def __post_init__(self):
        super().__post_init__()
        # The Qwen4 preview reuses the Qwen3.5-MoE ViT byte-for-byte; the
        # bundled mlx-vlm tower only gates construction on this type string.
        self.model_type = "qwen3_5_moe_vision"


@dataclass
class TextConfig(Qwen3_5MoeTextConfig):
    """Qwen3.5-MoE plus the Qwen4 experimental runtime parameters."""

    layer_types: Optional[List[str]] = None
    hc_count: int = 4
    hc_lowrank: Optional[int] = None
    ple_layer_ids: List[int] = field(default_factory=list)
    ple_embed_dim: Optional[int] = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 128
    indexer_n_heads: Optional[int] = None
    indexer_kv_heads: Optional[int] = None
    indexer_head_dim: Optional[int] = None
    indexer_budget: Optional[int] = None
    indexer_compress_ratio: Optional[int] = None
    output_gate_type: str = "sigmoid"

    def __post_init__(self):
        super().__post_init__()
        if self.layer_types is None:
            self.layer_types = [
                (
                    "full_attention"
                    if (layer_idx + 1) % self.full_attention_interval == 0
                    else "linear_attention"
                )
                for layer_idx in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per decoder layer")

        self.ple_layer_ids = sorted(set(self.ple_layer_ids))
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size
        if self.hc_lowrank is None:
            self.hc_lowrank = max(1, self.hidden_size // 8)

        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ple_layer_ids and self.ple_embed_dim % ngram_heads != 0:
            raise ValueError("ple_embed_dim must be divisible by all n-gram heads")
        if any(
            layer < 1 or layer > self.num_hidden_layers for layer in self.ple_layer_ids
        ):
            raise ValueError("ple_layer_ids are one-indexed decoder layer ids")

        indexer_values = (
            self.indexer_n_heads,
            self.indexer_kv_heads,
            self.indexer_head_dim,
            self.indexer_budget,
            self.indexer_compress_ratio,
        )
        if any(value is not None for value in indexer_values):
            if any(value is None for value in indexer_values):
                raise ValueError(
                    "all QSA indexer parameters must be configured together"
                )
            if self.indexer_kv_heads != 1:
                raise ValueError("Qwen4-Exp QSA requires indexer_kv_heads=1")
            if self.indexer_budget % self.indexer_compress_ratio:
                raise ValueError(
                    "indexer_budget must be divisible by indexer_compress_ratio"
                )


@dataclass
class ModelConfig(Qwen3_5MoeModelConfig):
    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params["vision_config"] = _maybe_deserialize_config(
            VisionConfig, params.get("vision_config")
        )
        params["text_config"] = _maybe_deserialize_config(
            TextConfig, params.get("text_config"), require_all_fields=True
        )
        return cls(**_config_kwargs(cls, params))
