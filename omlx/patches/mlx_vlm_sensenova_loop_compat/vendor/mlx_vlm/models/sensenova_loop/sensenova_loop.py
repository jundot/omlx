import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5.qwen3_5 import Model as Qwen3_5Model
from ..qwen3_5.vision import VisionModel
from .config import ModelConfig
from .language import LanguageModel


class Model(Qwen3_5Model):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)

    def sanitize(self, weights):
        # The release includes a training-only gate. Inference uses eval_count
        # from loop_settings, exactly as the reference execution plan does.
        weights = {
            key: value
            for key, value in weights.items()
            if key
            not in {
                "model.language_model.loop_gate.logits",
                "language_model.model.loop_gate.logits",
            }
        }
        weights = super().sanitize(weights)
        return {
            key: (
                value.astype(mx.float32)
                if ".mhc_attn." in key or ".mhc_ffn." in key
                else value
            )
            for key, value in weights.items()
        }
