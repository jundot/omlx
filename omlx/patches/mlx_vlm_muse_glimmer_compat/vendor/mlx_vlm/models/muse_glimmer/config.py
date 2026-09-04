import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


def _config_kwargs(config_cls, params):
    return {
        key: value
        for key, value in params.items()
        if key in inspect.signature(config_cls).parameters
    }


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "muse_glimmer_vision"
    patch_size: int = 14
    patch_temporal: int = 2
    merge_size: int = 2
    pos_emb_height: int = 32
    pos_emb_width: int = 32
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_attention_heads: int = 16
    num_hidden_layers: int = 50
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    max_position_embeddings: int = 1024
    rope_parameters: Optional[Dict] = None
    layer_types: Optional[List[str]] = None

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 10000.0, "rope_type": "default"}
        if self.layer_types is None:
            self.layer_types = [
                (
                    "full_attention"
                    if (idx + 1) % 4 == 0 or idx == self.num_hidden_layers - 1
                    else "window_attention"
                )
                for idx in range(self.num_hidden_layers)
            ]


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = "muse_glimmer_text"
    vocab_size: int = 202048
    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    hidden_activation: str = "silu"
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 2048
    rope_parameters: Optional[Dict] = None
    layer_types: Optional[List[str]] = None
    layer_rope_theta: Optional[List[Union[int, float]]] = None
    qk_scale_factor: float = 3.87
    output_multiplier: float = 0.19611613513818404
    final_logit_softcapping: float = 20.0
    tie_word_embeddings: bool = False
    bos_token_id: Optional[int] = 200000
    eos_token_id: Optional[Union[int, List[int]]] = 200001
    pad_token_id: Optional[int] = None

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 500000.0, "rope_type": "default"}
        if self.layer_types is None:
            self.layer_types = [
                (
                    "full_attention"
                    if (self.num_hidden_layers - 1 - idx) % 4 == 0
                    else "sliding_attention"
                )
                for idx in range(self.num_hidden_layers)
            ]
        if self.layer_rope_theta is None:
            theta = self.rope_parameters.get("rope_theta", 500000.0)
            self.layer_rope_theta = [
                0 if layer_type == "full_attention" else theta
                for layer_type in self.layer_types
            ]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "muse_glimmer"
    image_token_id: int = 200092
    video_token_id: int = 200091
    out_hidden_size: int = 6144
    projector_hidden_size: int = 4096
    projector_hidden_act: str = "gelu"
    eos_token_id: Optional[Union[int, List[int]]] = None
    vocab_size: int = 202048
    thinking_start_token: str = "to=self<|message|>"
    thinking_end_token: str = "<|eom|>"

    def __post_init__(self):
        if self.eos_token_id is None:
            self.eos_token_id = self.text_config.eos_token_id
        self.vocab_size = self.text_config.vocab_size

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        text_config = params.get("text_config", {})
        vision_config = params.get("vision_config", {})
        if isinstance(text_config, dict):
            params["text_config"] = TextConfig(
                **_config_kwargs(TextConfig, text_config)
            )
        if isinstance(vision_config, dict):
            params["vision_config"] = VisionConfig(
                **_config_kwargs(VisionConfig, vision_config)
            )
        return cls(**_config_kwargs(cls, params))
