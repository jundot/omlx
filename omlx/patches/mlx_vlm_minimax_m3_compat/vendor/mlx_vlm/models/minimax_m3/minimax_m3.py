from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from ..minimax_m3_vl.config import (
    TextConfig,
    _config_kwargs,
    _sanitize_quantization_config,
)
from ..minimax_m3_vl.language import LanguageModel
from ..minimax_m3_vl.minimax_m3_vl import (
    _pack_uint8_weight,
    _sanitize_moe_weights,
)


@dataclass
class ModelConfig(TextConfig):
    quantization: Optional[dict] = None
    quantization_config: Optional[dict] = None

    def __post_init__(self):
        super().__post_init__()
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
        params = dict(params or {})
        text_config = params.get("text_config")
        if isinstance(text_config, dict) and text_config:
            params = {**params, **text_config}
        return cls(**_config_kwargs(cls, params))


class Model(nn.Module):
    _is_text_model = True

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        if pixel_values is not None:
            raise ValueError("MiniMax M3 text-only models do not accept image inputs.")
        if input_ids is None:
            raise ValueError("input_ids are required for MiniMax M3 text-only models.")
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        if pixel_values is not None:
            raise ValueError("MiniMax M3 text-only models do not accept image inputs.")
        return self.language_model(input_ids, mask=mask, cache=cache, **kwargs)

    def sanitize(self, weights):
        sanitized_weights = {}
        for key, value in weights.items():
            if key.startswith("language_model."):
                pass
            elif key.startswith("model.") or key.startswith("lm_head."):
                key = f"language_model.{key}"
            sanitized_weights[key] = value
        weights.clear()

        scale_keys = {
            key.replace(".weight_scale_inv", ".weight")
            for key in sanitized_weights
            if key.endswith(".weight_scale_inv")
        }
        for weight_key in scale_keys:
            weight = sanitized_weights.get(weight_key)
            if weight is not None:
                sanitized_weights[weight_key] = _pack_uint8_weight(weight)

        for key in list(sanitized_weights):
            if key.endswith(".weight_scale_inv"):
                sanitized_weights[key.replace(".weight_scale_inv", ".scales")] = (
                    sanitized_weights.pop(key)
                )

        args = self.language_model.args
        _sanitize_moe_weights(sanitized_weights, args)
        return sanitized_weights

    def make_cache(self):
        return self.language_model.make_cache()

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self.language_model.speculative_logits_from_hidden(hidden)

    def rollback_speculative_cache(
        self, caches, gdn_states, accepted, block_size
    ) -> int:
        return self.language_model.rollback_speculative_cache(
            caches, gdn_states, accepted, block_size
        )
