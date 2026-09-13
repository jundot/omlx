from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..qwen3_5.config import resolve_qwen_eos_token_id, sanitize_quantization_config
from ..qwen3_vl.config import VisionConfig as Qwen3VLVisionConfig
from ..qwen3_vl.config import _config_kwargs, _maybe_deserialize_config


@dataclass
class VisionConfig(Qwen3VLVisionConfig):
    model_type: str = "qwen4_exp"

    def __post_init__(self):
        if self.deepstack_visual_indexes:
            raise ValueError(
                "Qwen4-Exp does not use deepstack visual features, but "
                f"deepstack_visual_indexes={self.deepstack_visual_indexes} was set."
            )
        self.deepstack_visual_indexes = []


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    linear_num_value_heads: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    shared_expert_intermediate_size: int
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    hc_count: int = 4
    hc_lowrank: int = 320
    head_dim: Optional[int] = None
    layer_types: Optional[List[str]] = None
    full_attention_interval: int = 4
    ple_layer_ids: List[int] = field(default_factory=list)
    ple_embed_dim: Optional[int] = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    seed: int = 1234
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    output_gate_type: str = "sigmoid"
    hidden_act: str = "silu"
    norm_topk_prob: bool = True
    mtp_num_hidden_layers: int = 1
    mtp_use_dedicated_embeddings: bool = False
    mtp: Optional[Dict] = None
    eos_token_id: Optional[Union[int, List[int]]] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    rope_parameters: Optional[Dict[str, Union[float, str, bool, List[int]]]] = field(
        default_factory=lambda: {
            "type": "default",
            "mrope_section": [11, 11, 10],
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        }
    )

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size

        self.ple_layer_ids = sorted(set(self.ple_layer_ids or []))
        if self.layer_types is None:
            self.layer_types = [
                (
                    "linear_attention"
                    if (i + 1) % self.full_attention_interval
                    else "qwen_sparse_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = [
                "qwen_sparse_attention" if kind == "full_attention" else kind
                for kind in self.layer_types
            ]

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                "layer_types must contain one entry per decoder layer: "
                f"{len(self.layer_types)} != {self.num_hidden_layers}"
            )
        unsupported = set(self.layer_types) - {
            "linear_attention",
            "qwen_sparse_attention",
        }
        if unsupported:
            raise ValueError(f"Unsupported Qwen4-Exp layer types: {unsupported}")
        if self.hc_count <= 1:
            raise ValueError("Qwen4-Exp requires hc_count > 1")
        if self.output_gate_type not in {"sigmoid", "silu"}:
            raise ValueError("Qwen4-Exp output_gate_type must be 'sigmoid' or 'silu'")
        if self.indexer_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA requires indexer_kv_heads=1")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError(
                "indexer_budget must be divisible by indexer_compress_ratio"
            )
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if ngram_heads <= 0 or self.ple_embed_dim % ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by the total number of n-gram heads"
            )
        invalid_ple_layers = [
            layer_id
            for layer_id in self.ple_layer_ids
            if layer_id < 1 or layer_id > self.num_hidden_layers
        ]
        if invalid_ple_layers:
            raise ValueError(f"Invalid one-indexed PLE layers: {invalid_ple_layers}")
        non_linear_ple_layers = [
            layer_id
            for layer_id in self.ple_layer_ids
            if self.layer_types[layer_id - 1] != "linear_attention"
        ]
        if non_linear_ple_layers:
            raise ValueError(
                "PLE is only supported on linear-attention layers, got "
                f"{non_linear_ple_layers}"
            )
        if self.ple_layer_ids and self.eos_token_id is None:
            raise ValueError("eos_token_id is required when PLE is enabled")

        if self.rope_parameters:
            if (
                "type" not in self.rope_parameters
                and "rope_type" in self.rope_parameters
            ):
                self.rope_parameters["type"] = self.rope_parameters.pop("rope_type")
            required_keys = {
                "mrope_section",
                "type",
                "rope_theta",
                "partial_rotary_factor",
            }
            if not required_keys.issubset(self.rope_parameters):
                raise ValueError(f"rope_parameters must contain keys {required_keys}")


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_id: int = 248056
    video_token_id: int = 248057
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    vocab_size: int = 248320
    eos_token_id: Optional[Union[int, List[int]]] = None
    quantization: Optional[Dict] = None
    quantization_config: Optional[Dict] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id
        self.eos_token_id = resolve_qwen_eos_token_id(
            self.eos_token_id, self.text_config
        )
        quantization = self.quantization
        self.quantization = sanitize_quantization_config(quantization)
        if self.quantization_config == quantization:
            self.quantization_config = self.quantization
        else:
            self.quantization_config = sanitize_quantization_config(
                self.quantization_config
            )

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
