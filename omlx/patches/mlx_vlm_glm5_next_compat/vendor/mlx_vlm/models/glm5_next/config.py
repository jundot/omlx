from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base import BaseModelConfig


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: Optional[int]
    n_routed_experts: Optional[int]
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    num_experts_per_tok: int
    first_k_dense_replace: int
    max_position_embeddings: int
    rms_norm_eps: float
    index_topk: int
    index_head_dim: int
    index_n_heads: int
    layer_types: List[str]
    mlp_layer_types: List[str]
    linear_attn_config: Dict[str, Any]
    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: Optional[float] = -5.0
    qk_head_dim: int = 256
    head_dim: int = 0
    n_group: int = 1
    topk_group: int = 1
    moe_layer_freq: int = 1
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    mla_use_nope: bool = True
    indexer_rope_interleave: bool = True
    indexer_types: Optional[List[str]] = None
    index_kpool: int = 4
    index_kpool_compress: bool = True
    index_kpool_always_select_tail: bool = True
    swiglu_limit: float = 10.0
    hc_mult: int = 4
    hc_eps: float = 1e-06
    hc_sinkhorn_iters: int = 20
    mhc: bool = True
    moe_router_dtype: str = "float32"
    num_nextn_predict_layers: int = 0
    router_aux_loss_coef: float = 0.001
    output_router_logits: bool = False
    rope_theta: float = 10000.0
    rope_parameters: Optional[Dict] = None
    rope_scaling: Optional[Dict] = None
    pad_token_id: int = 154820
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.rope_parameters is not None:
            self.rope_scaling = self.rope_parameters
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)

        cfg = self.linear_attn_config or {}
        self.linear_num_heads = cfg.get("num_heads", self.linear_num_heads)
        self.linear_head_dim = cfg.get("head_dim", self.linear_head_dim)
        self.linear_conv_kernel_dim = cfg.get(
            "short_conv_kernel_size", self.linear_conv_kernel_dim
        )
        self.linear_lower_bound = cfg.get("gate_lower_bound", self.linear_lower_bound)
        if cfg.get("safe_gate", True) and self.linear_lower_bound is None:
            self.linear_lower_bound = -5.0


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    patch_size: int
    out_hidden_size: int
    projection_intermediate_size: int = 10240
    image_size: int = 448
    in_channels: int = 3
    rms_norm_eps: float = 1e-05
    attention_bias: bool = True
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    swiglu_limit: float = 10.0

    @classmethod
    def from_dict(cls, params):
        # mlx-vlm inserts an empty vision_config for text-only checkpoints.
        # Preserve that distinction instead of trying to construct a vision
        # tower from missing required dimensions.
        if not params:
            return None
        return super().from_dict(params)


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    model_type: str
    vision_config: Optional[VisionConfig] = None
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
    tie_word_embeddings: bool = False
    pad_token_id: int = 154820
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.eos_token_id is None:
            self.eos_token_id = [154820, 154827, 154829]
