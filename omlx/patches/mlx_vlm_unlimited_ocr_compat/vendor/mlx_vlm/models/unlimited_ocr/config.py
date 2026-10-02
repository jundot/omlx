import inspect
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..deepseekocr.config import (
    BatchCollateOutput,
    Conversation,
    MLPConfig,
    ProjectorConfig,
    SAMViTConfig,
)
from ..deepseekocr.config import VisionConfig as DeepseekOCRVisionConfig
from ..deepseekocr.config import VLChatProcessorOutput


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "deepseek_v2"
    vocab_size: int = 129280
    hidden_size: int = 1280
    intermediate_size: int = 6848
    moe_intermediate_size: int = 896
    num_hidden_layers: int = 12
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    n_shared_experts: Optional[int] = 2
    n_routed_experts: Optional[int] = 64
    routed_scaling_factor: float = 1.0
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None
    qk_rope_head_dim: int = 0
    v_head_dim: int = 128
    qk_nope_head_dim: int = 0
    topk_method: str = "greedy"
    n_group: Optional[int] = 1
    topk_group: Optional[int] = 1
    num_experts_per_tok: Optional[int] = 6
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_traditional: bool = False
    rope_scaling: Dict = None
    attention_bias: bool = False
    scoring_func: str = "softmax"
    attn_type: str = "DeepseekV2Attention"
    use_mla: bool = False
    sliding_window_size: Optional[int] = 128
    sliding_window: Optional[int] = 128

    def __post_init__(self):
        if self.qk_nope_head_dim == 0:
            self.attn_type = "LlamaAttention"

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass
class VisionConfig(DeepseekOCRVisionConfig):
    width: Union[int, Dict] = 1152


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    projector_config: ProjectorConfig
    model_type: str
    ignore_index: int = -100
    image_token_index: int = 128815
    vision_feature_select_strategy: str = "default"
    select_layer: int = -1
    pad_id: int = 100001
    num_image_tokens: int = 576
    vocab_size: int = 129280
    tile_tag: str = "2D"
    global_view_pos: str = "head"
    eos_token_id: Optional[Union[int, List[int]]] = 1
    quantization: Optional[Dict] = None

    @classmethod
    def from_dict(cls, params):
        if "language_config" in params:
            params["text_config"] = params["language_config"]
            del params["language_config"]

        return cls(
            text_config=TextConfig.from_dict(params["text_config"]),
            vision_config=VisionConfig.from_dict(params["vision_config"]),
            projector_config=ProjectorConfig.from_dict(params["projector_config"]),
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
                and k not in ["text_config", "vision_config", "projector_config"]
            },
        )


__all__ = [
    "BatchCollateOutput",
    "Conversation",
    "MLPConfig",
    "ModelConfig",
    "ProjectorConfig",
    "SAMViTConfig",
    "TextConfig",
    "VLChatProcessorOutput",
    "VisionConfig",
]
