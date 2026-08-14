from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .config import ModelConfig
from .language import LanguageModel, RMSNormNoScale
from .vision import VisionModel, gelu


def masked_scatter(input_tensor: mx.array, mask: mx.array, source: mx.array):
    shape = input_tensor.shape
    flat_input = input_tensor.flatten()
    flat_mask = mask.flatten()
    flat_input[flat_mask] = source.flatten()
    return flat_input.reshape(shape)


class VisionAdapter(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(
            config.out_hidden_size, config.projector_hidden_size, bias=False
        )
        self.fc2 = nn.Linear(
            config.projector_hidden_size, config.projector_hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return gelu(self.fc2(gelu(self.fc1(x))))


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
        self.perception_emb_norm = RMSNormNoScale(config.text_config.rms_norm_eps)

    def _encode_image(self, pixel_values, image_grid_thw=None):
        if image_grid_thw is None:
            raise ValueError(
                "image_grid_thw is required to encode Muse Glimmer images."
            )
        dtype = self.vision_tower.patch_embedder.patch_embedding.weight.dtype
        features = self.vision_tower(pixel_values.astype(dtype), image_grid_thw)
        features = self.vision_adapter(features)
        features = self.vision_projection(features)
        return self.perception_emb_norm(features)

    # oMLX: public alias so oMLX's vision feature cache can precompute and
    # persist image features (engine/vlm.py::_compute_vision_features probes
    # for an `encode_image` attribute and replays them at generation time via
    # the `cached_image_features` kwarg of get_input_embeddings).
    def encode_image(self, pixel_values, image_grid_thw=None, **kwargs):
        return self._encode_image(pixel_values, image_grid_thw=image_grid_thw)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ):
        language_model = self.language_model.model
        inputs_embeds = language_model.embed_norm(
            language_model.embed_tokens(input_ids)
        )
        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos")
        if pixel_values is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        grid_thw = kwargs.get("image_grid_thw")
        if grid_thw is None:
            grid_thw = kwargs.get("video_grid_thw")
        cached = kwargs.get("cached_image_features")
        image_features = (
            cached
            if cached is not None
            else self._encode_image(pixel_values, image_grid_thw=grid_thw)
        ).astype(inputs_embeds.dtype)

        token_mask = (input_ids == self.config.image_token_id) | (
            input_ids == self.config.video_token_id
        )
        expanded_mask = mx.broadcast_to(token_mask[..., None], inputs_embeds.shape)
        if int(token_mask.sum().item()) != image_features.shape[0]:
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens={int(token_mask.sum().item())}, "
                f"features={image_features.shape[0]}"
            )
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
            input_ids,
            cache=cache,
            inputs_embeds=embeddings.inputs_embeds,
        )

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if "rotary_emb.inv_freq" in key:
                continue
            if key.startswith("model.language_model."):
                key = key.replace("model.language_model.", "language_model.model.", 1)
            elif key.startswith("model."):
                key = key[len("model.") :]
            elif key.startswith("lm_head."):
                key = "language_model." + key
            sanitized[key] = value
        return sanitized
