import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..base import BaseModelConfig


def _config_kwargs(config_cls, params):
    return {
        k: v for k, v in params.items() if k in inspect.signature(config_cls).parameters
    }


def _maybe_deserialize_config(config_cls, params):
    if not isinstance(params, dict):
        return params
    return config_cls(**_config_kwargs(config_cls, params))


def _sanitize_quantization_key(key: str) -> str:
    replacements = (
        ("model.language_model.", "language_model."),
        ("model.vision_tower.", "vision_tower."),
        ("model.multi_modal_projector.", "multi_modal_projector."),
        ("model.patch_merge_mlp.", "patch_merge_mlp."),
    )
    for old, new in replacements:
        if key == old[:-1]:
            return new[:-1]
        if key.startswith(old):
            return key.replace(old, new, 1)
    if key == "lm_head" or key.startswith("lm_head."):
        return f"language_model.{key}"
    if key.startswith("model."):
        return f"language_model.{key}"
    return key


def _sanitize_quantization_config(quantization):
    if not isinstance(quantization, dict):
        return quantization

    sanitized = {}
    for key, value in quantization.items():
        if key == "ignored_layers" and isinstance(value, list):
            value = [
                _sanitize_quantization_key(item) if isinstance(item, str) else item
                for item in value
            ]
        sanitized[_sanitize_quantization_key(key)] = value
    return sanitized


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "minimax_m3"
    hidden_size: int = 6144
    intermediate_size: int = 3072
    dense_intermediate_size: int = 12288
    shared_intermediate_size: int = 3072
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_hidden_layers: int = 60
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000
    rotary_dim: Optional[int] = None
    partial_rotary_factor: float = 0.5
    rope_scaling: Optional[Dict[str, Any]] = None
    max_position_embeddings: int = 1048576
    vocab_size: int = 200064
    tie_word_embeddings: bool = False
    hidden_act: str = "swigluoai"
    swiglu_alpha: float = 1.702
    swiglu_beta: float = 1.0
    swiglu_limit: float = 7.0
    use_qk_norm: bool = True
    qk_norm_type: str = "per_head"
    use_gemma_norm: bool = True
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    n_shared_experts: int = 1
    scoring_func: str = "sigmoid"
    use_routing_bias: bool = True
    routed_scaling_factor: float = 2.0
    moe_layer_freq: List[int] = field(default_factory=list)
    mlp_layer_types: Optional[List[str]] = None
    sparse_attention_config: Optional[Dict[str, Any]] = None
    layer_types: Optional[List[str]] = None
    index_n_heads: Optional[int] = None
    index_head_dim: Optional[int] = None
    index_block_size: Optional[int] = None
    index_topk_blocks: Optional[int] = None
    index_local_blocks: Optional[int] = None
    attention_output_gate: bool = False
    architectures: Optional[List[str]] = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rotary_dim is None:
            self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if isinstance(self.rope_scaling, dict) and "type" not in self.rope_scaling:
            self.rope_scaling = dict(self.rope_scaling)
            if "rope_type" in self.rope_scaling:
                self.rope_scaling["type"] = self.rope_scaling["rope_type"]
        if not self.moe_layer_freq:
            if self.mlp_layer_types is not None:
                self.moe_layer_freq = [
                    1 if layer_type == "sparse" else 0
                    for layer_type in self.mlp_layer_types
                ]
            else:
                self.moe_layer_freq = self._default_layer_frequency()
        sparse_freq = self._sparse_frequency_from_layer_types()
        if self.sparse_attention_config is None:
            if sparse_freq is None:
                sparse_freq = self._default_layer_frequency()
            self.sparse_attention_config = {
                "use_sparse_attention": True,
                "sparse_index_dim": (
                    self.index_head_dim if self.index_head_dim is not None else 128
                ),
                "sparse_num_index_heads": (
                    self.index_n_heads if self.index_n_heads is not None else 4
                ),
                "sparse_topk_blocks": (
                    self.index_topk_blocks if self.index_topk_blocks is not None else 16
                ),
                "sparse_block_size": (
                    self.index_block_size if self.index_block_size is not None else 128
                ),
                "sparse_disable_index_value": sparse_freq.copy(),
                "sparse_score_type": "max",
                "sparse_init_block": 0,
                "sparse_local_block": (
                    self.index_local_blocks
                    if self.index_local_blocks is not None
                    else 1
                ),
                "sparse_attention_freq": sparse_freq,
            }
        else:
            self.sparse_attention_config = dict(self.sparse_attention_config)
            if (
                sparse_freq is not None
                and "sparse_attention_freq" not in self.sparse_attention_config
            ):
                self.sparse_attention_config["sparse_attention_freq"] = sparse_freq
            if sparse_freq is not None:
                self.sparse_attention_config.setdefault("use_sparse_attention", True)
            self._apply_sparse_index_aliases()
            sparse_freq = self.sparse_attention_config.get("sparse_attention_freq")
            if sparse_freq is None and isinstance(
                self.sparse_attention_config.get("sparse_disable_index_value"), list
            ):
                sparse_freq = self.sparse_attention_config["sparse_disable_index_value"]
                self.sparse_attention_config["sparse_attention_freq"] = (
                    sparse_freq.copy()
                )
                self.sparse_attention_config.setdefault("use_sparse_attention", True)
            if (
                isinstance(sparse_freq, list)
                and "sparse_disable_index_value" not in self.sparse_attention_config
            ):
                self.sparse_attention_config["sparse_disable_index_value"] = (
                    sparse_freq.copy()
                )

    def _default_layer_frequency(self) -> List[int]:
        dense_layers = min(3, self.num_hidden_layers)
        return [0] * dense_layers + [1] * (self.num_hidden_layers - dense_layers)

    def _sparse_frequency_from_layer_types(self) -> Optional[List[int]]:
        if self.layer_types is None:
            return None
        return [
            1 if layer_type == "minimax_m3_sparse" else 0
            for layer_type in self.layer_types
        ]

    def _apply_sparse_index_aliases(self):
        aliases = {
            "sparse_index_dim": self.index_head_dim,
            "sparse_num_index_heads": self.index_n_heads,
            "sparse_topk_blocks": self.index_topk_blocks,
            "sparse_block_size": self.index_block_size,
            "sparse_local_block": self.index_local_blocks,
        }
        for key, value in aliases.items():
            if value is not None and key not in self.sparse_attention_config:
                self.sparse_attention_config[key] = value

    def is_moe_layer(self, layer_idx: int) -> bool:
        if layer_idx >= len(self.moe_layer_freq):
            return True
        return bool(self.moe_layer_freq[layer_idx])

    def has_sparse_index(self, layer_idx: int) -> bool:
        if not self.sparse_attention_config.get("use_sparse_attention", False):
            return False
        freq = self.sparse_attention_config.get("sparse_attention_freq")
        if isinstance(freq, list) and layer_idx < len(freq):
            return bool(freq[layer_idx])
        return False


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "clip_vision_model"
    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_attention_heads: int = 16
    num_hidden_layers: int = 32
    image_size: int = 2016
    patch_size: int = 14
    num_channels: int = 3
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    attention_dropout: float = 0.0
    projection_dim: int = 6144
    vocab_size: int = 32000
    position_embedding_type: str = "rope"
    rope_mode: str = "3d"
    rope_theta: float = 10000.0
    vision_segment_max_frames: int = 4
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    img_token_compression_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        compression = self.img_token_compression_config or {}
        self.spatial_merge_size = compression.get("spatial_merge_size", 2)
        self.temporal_patch_size = compression.get("temporal_patch_size", 2)


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "minimax_m3_vl"
    architectures: Optional[List[str]] = None
    auto_map: Optional[Dict[str, Any]] = None
    image_token_index: int = 200025
    video_token_index: int = 200026
    image_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    vision_start_token_id: Optional[int] = None
    vision_end_token_id: Optional[int] = None
    vision_token_id: Optional[int] = None
    image_seq_length: int = 576
    process_image_mode: str = "dynamic_res"
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int = 6144
    img_token_compression_config: Dict[str, Any] = field(default_factory=dict)
    multimodal_projector_bias: bool = True
    patch_merge_bias: bool = True
    image_grid_pinpoints: Optional[Union[str, List[Any]]] = None
    vision_feature_layer: Union[int, List[int]] = -1
    vision_feature_select_strategy: str = "full"
    ignore_index: int = -100
    vocab_size: int = 200064
    eos_token_id: Optional[Union[int, List[int]]] = None
    num_reward_heads: Optional[int] = None
    transformers_version: Optional[str] = None
    quantization: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        self.text_config = _maybe_deserialize_config(TextConfig, self.text_config)
        self.vision_config = _maybe_deserialize_config(VisionConfig, self.vision_config)
        if (
            self.img_token_compression_config
            and not self.vision_config.img_token_compression_config
        ):
            self.vision_config.img_token_compression_config = (
                self.img_token_compression_config
            )
            self.vision_config.__post_init__()
        if self.image_token_id is None:
            self.image_token_id = self.image_token_index
        if self.video_token_id is None:
            self.video_token_id = self.video_token_index
        if self.vision_start_token_id is None:
            self.vision_start_token_id = 200029
        if self.vision_end_token_id is None:
            self.vision_end_token_id = 200030
        quantization = self.quantization
        self.quantization = _sanitize_quantization_config(quantization)
        if self.quantization_config == quantization:
            self.quantization_config = self.quantization
        else:
            self.quantization_config = _sanitize_quantization_config(
                self.quantization_config
            )

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params.setdefault("text_config", {})
        params.setdefault("vision_config", {})
        compression = params.get("img_token_compression_config")
        if compression and isinstance(params["vision_config"], dict):
            params["vision_config"] = dict(params["vision_config"])
            params["vision_config"].setdefault(
                "img_token_compression_config", compression
            )
        params["text_config"] = _maybe_deserialize_config(
            TextConfig, params.get("text_config")
        )
        params["vision_config"] = _maybe_deserialize_config(
            VisionConfig, params.get("vision_config")
        )
        return cls(**_config_kwargs(cls, params))
