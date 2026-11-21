from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..base import BaseModelConfig
from ..qwen3_5.config import TextConfig as Qwen3_5DenseTextConfig
from ..qwen3_5_moe.config import (
    TextConfig as Qwen3_5MoeTextConfig,
    VisionConfig as Qwen3_5MoeVisionConfig,
)


def _filter_dataclass_kwargs(cls, params: Dict[str, Any]) -> Dict[str, Any]:
    valid = set(inspect.signature(cls).parameters.keys())
    return {k: v for k, v in params.items() if k in valid}


@dataclass
class QFormerConfig(BaseModelConfig):
    model_type: str = "bert"
    hidden_size: int = 4104
    num_hidden_layers: int = 2
    num_attention_heads: int = 12
    intermediate_size: int = 12288
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    encoder_width: int = 1280
    query_length: int = 1
    out_hidden_size: int = 2048
    add_cross_attention: bool = True
    cross_attention_freq: int = 1
    vocab_size: int = 30522
    max_position_embeddings: int = 512

    @classmethod
    def from_dict(cls, params: Optional[Dict[str, Any]]) -> "QFormerConfig":
        if isinstance(params, cls):
            return params
        params = dict(params or {})
        return cls(**_filter_dataclass_kwargs(cls, params))


@dataclass
class AudioConfig(BaseModelConfig):
    model_type: str = "whisper"
    d_model: int = 1280
    encoder_layers: int = 32
    encoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    num_mel_bins: int = 128
    max_source_positions: int = 1500
    activation_function: str = "gelu"
    scale_embedding: bool = False
    second_per_window: float = 0.1
    second_stride: float = 0.1
    qformer_config: QFormerConfig = field(default_factory=QFormerConfig)

    @classmethod
    def from_dict(cls, params: Optional[Dict[str, Any]]) -> "AudioConfig":
        if isinstance(params, cls):
            return params
        params = dict(params or {})
        qfc = params.get("qformer_config")
        params["qformer_config"] = QFormerConfig.from_dict(qfc)
        return cls(**_filter_dataclass_kwargs(cls, params))


@dataclass
class VisionConfig(Qwen3_5MoeVisionConfig):
    model_type: str = "qwen3_5_moe"


@dataclass
class TextConfig(Qwen3_5MoeTextConfig):
    layer_types: Optional[List[str]] = None
    attn_output_gate: bool = True
    mamba_ssm_dtype: str = "float32"
    partial_rotary_factor: float = 0.25

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "TextConfig":
        if isinstance(params, cls):
            return params
        params = dict(params)
        if params.get("rope_parameters"):
            rp = dict(params["rope_parameters"])
            if "type" not in rp and "rope_type" in rp:
                rp["type"] = rp.pop("rope_type")
            params["rope_parameters"] = rp
        return cls(**_filter_dataclass_kwargs(cls, params))


@dataclass
class DenseTextConfig(Qwen3_5DenseTextConfig):
    layer_types: Optional[List[str]] = None
    attn_output_gate: bool = True
    mamba_ssm_dtype: str = "float32"
    partial_rotary_factor: float = 0.25

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "DenseTextConfig":
        if isinstance(params, cls):
            return params
        params = dict(params)
        if params.get("rope_parameters"):
            rp = dict(params["rope_parameters"])
            if "type" not in rp and "rope_type" in rp:
                rp["type"] = rp.pop("rope_type")
            params["rope_parameters"] = rp
        return cls(**_filter_dataclass_kwargs(cls, params))


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: Union[TextConfig, DenseTextConfig]
    vision_config: VisionConfig
    audio_config: Optional[AudioConfig] = None
    model_type: str = "wemm_qwen3_5_moe"
    ignore_index: int = -100
    image_token_id: int = 248056
    video_token_id: int = 248057
    audio_token_id: int = 248076
    image_token_index: Optional[int] = None
    video_token_index: Optional[int] = None
    audio_token_index: Optional[int] = None
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    vocab_size: int = 248320
    eos_token_id: Optional[Union[int, List[int]]] = None
    tie_word_embeddings: bool = False
    quantization: Optional[Dict[str, Any]] = None
    quantization_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.image_token_index is None:
            self.image_token_index = self.image_token_id
        if self.video_token_index is None:
            self.video_token_index = self.video_token_id
        if self.audio_token_index is None:
            self.audio_token_index = self.audio_token_id
        if self.eos_token_id is None and self.text_config is not None:
            self.eos_token_id = getattr(self.text_config, "eos_token_id", None)
        if self.quantization is None and self.quantization_config is not None:
            self.quantization = self.quantization_config
        elif self.quantization_config is None and self.quantization is not None:
            self.quantization_config = self.quantization

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelConfig":
        if isinstance(params, cls):
            return params
        params = dict(params)
        if isinstance(params.get("vision_config"), dict):
            params["vision_config"] = VisionConfig.from_dict(params["vision_config"])
        if isinstance(params.get("text_config"), dict):
            tc = params["text_config"]
            if int(tc.get("num_experts", 0) or 0) > 0:
                params["text_config"] = TextConfig.from_dict(tc)
            else:
                params["text_config"] = DenseTextConfig.from_dict(tc)
        if isinstance(params.get("audio_config"), dict):
            params["audio_config"] = AudioConfig.from_dict(params["audio_config"])
        return cls(**_filter_dataclass_kwargs(cls, params))
