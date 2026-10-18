import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from omlx.patches.muse_spark.muse_spark_model import ModelArgs

from ..base import BaseModelConfig
from ..muse_glimmer.config import VisionConfig

# The text tower is the oMLX mlx-lm module, so the LLM and VLM paths share it.
TextConfig = ModelArgs


def _config_kwargs(config_cls, params):
    return {
        key: value
        for key, value in params.items()
        if key in inspect.signature(config_cls).parameters
    }


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = field(default_factory=TextConfig)
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    model_type: str = "muse_spark"
    image_token_id: int = 200092
    video_token_id: int = 200091
    audio_token_id: int = 200038
    out_hidden_size: int = 6144
    projector_hidden_size: int = 4096
    projector_hidden_act: str = "gelu"
    eos_token_id: Optional[Union[int, List[int]]] = None
    vocab_size: int = 202048
    quantization_config: Optional[Dict[str, Any]] = None
    thinking_start_token: str = "to=self<|message|>"
    thinking_end_token: str = "<|eom|>"

    def __post_init__(self):
        self.vocab_size = self.text_config.vocab_size

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        text_config = params.get("text_config", {})
        vision_config = params.get("vision_config", {})
        if isinstance(text_config, dict):
            if params.get("eos_token_id") is None:
                params["eos_token_id"] = text_config.get("eos_token_id")
            params["text_config"] = TextConfig.from_dict(text_config)
        if isinstance(vision_config, dict):
            params["vision_config"] = VisionConfig(
                **_config_kwargs(VisionConfig, vision_config)
            )
        return cls(**_config_kwargs(cls, params))
