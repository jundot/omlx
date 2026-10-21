from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.muse_spark.muse_spark_model import (
    cast_predicate,
    install_global_scales,
    is_modelopt_export,
    sanitize_experts,
    use_upstream_values,
)
from omlx.patches.muse_spark.upstream_layout import (
    is_upstream_layout,
    read_upstream_text,
    read_upstream_vision,
    upstream_key,
)

from ..base import InputEmbeddingsFeatures
from ..muse_glimmer.muse_glimmer import VisionAdapter, masked_scatter
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel

_UNSUPPORTED_INPUTS = ("pixel_values_videos", "input_values")


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = VisionModel(config.vision_config)
        self.vision_adapter = VisionAdapter(config)
        self.vision_projection = nn.Linear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=False,
        )
        self.perception_eps = config.text_config.rms_norm_eps

    def encode_image(self, pixel_values, image_grid_thw=None, **kwargs):
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required to encode Muse Spark images.")
        dtype = self.vision_tower.patch_embedder.patch_embedding.weight.dtype
        features = self.vision_tower(pixel_values.astype(dtype), image_grid_thw)
        features = self.vision_projection(self.vision_adapter(features))
        return mx.fast.rms_norm(features, None, self.perception_eps)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        for name in _UNSUPPORTED_INPUTS:
            if kwargs.get(name) is not None:
                raise ValueError(f"Muse Spark does not support {name} in oMLX.")

        inputs_embeds = self.language_model.model.embed(input_ids)
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        cached = kwargs.get("cached_image_features")
        if cached is None:
            cached = self.encode_image(
                pixel_values, image_grid_thw=kwargs.get("image_grid_thw")
            )
        image_features = cached.astype(inputs_embeds.dtype)

        token_mask = input_ids == self.config.image_token_id
        count = int(token_mask.sum().item())
        if count != image_features.shape[0]:
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens={count}, features={image_features.shape[0]}"
            )
        expanded_mask = mx.broadcast_to(token_mask[..., None], inputs_embeds.shape)
        inputs_embeds = masked_scatter(inputs_embeds, expanded_mask, image_features)
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        embeddings = self.get_input_embeddings(
            input_ids=input_ids, pixel_values=pixel_values, **kwargs
        )
        return self.language_model(
            input_ids, cache=cache, inputs_embeds=embeddings.inputs_embeds
        )

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        upstream = is_upstream_layout(weights)
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("model.audio_tower."):
                continue
            if key.startswith("model.language_model."):
                key = "language_model.model." + key[len("model.language_model.") :]
            elif key.startswith("lm_head."):
                key = "language_model." + key
            elif key.startswith("model."):
                key = key[len("model.") :]
            sanitized[key] = value
        if upstream:
            text, vision = self.config.text_config, self.config.vision_config
            sanitized = read_upstream_text(
                sanitized,
                "language_model.model.layers.",
                n_heads=text.num_attention_heads,
                n_kv_heads=text.num_key_value_heads,
                head_dim=text.head_dim,
            )
            sanitized = read_upstream_vision(
                sanitized,
                "vision_tower.layers.",
                vision.hidden_size // vision.num_attention_heads,
            )
            use_upstream_values(self)
        else:
            sanitized = sanitize_experts(sanitized)
        install_global_scales(
            self, sanitized, is_modelopt_export(self.config.quantization_config)
        )
        return sanitized

    def quantization_path_aliases(self, path):
        # Upstream-layout checkpoints key their per-module quantization by
        # their own module paths.
        return (upstream_key(path),)

    @property
    def cast_predicate(self):
        return cast_predicate
