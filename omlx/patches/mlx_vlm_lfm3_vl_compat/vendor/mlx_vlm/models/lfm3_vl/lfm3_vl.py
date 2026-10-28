# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3/LFM3-VL checkpoint reference implementations.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import InputEmbeddingsFeatures
from ..lfm3.language import LanguageModel
from .vision import VisionModel


class MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config.vision_config.hidden_size * config.downsample_factor**2
        self.layer_norm = nn.LayerNorm(
            config.vision_config.hidden_size, eps=config.projector_layer_norm_eps
        )
        self.linear_1 = nn.Linear(
            self.input_dim, config.projector_hidden_size, bias=config.projector_bias
        )
        self.linear_2 = nn.Linear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )

    def __call__(self, x):
        x = self.layer_norm(x).reshape(-1, self.input_dim)
        return self.linear_2(nn.gelu(self.linear_1(x)))


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.multi_modal_projector = MultiModalProjector(config)
        self.language_model = LanguageModel(config.text_config)

    def encode_images(self, pixel_values, **kwargs):
        grid = kwargs.get("image_grid_thw")
        if grid is None:
            raise ValueError("image_grid_thw is required with pixel_values")
        return self.multi_modal_projector(self.vision_tower(pixel_values, grid))

    @staticmethod
    def _merge(embeddings, input_ids, features, token_id):
        indices = np.flatnonzero(np.asarray(input_ids).reshape(-1) == token_id)
        if len(indices) != features.shape[0]:
            raise ValueError(
                f"Visual tokens ({len(indices)}) and features ({features.shape[0]}) do not match"
            )
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        flat[mx.array(indices)] = features.astype(embeddings.dtype)
        return flat.reshape(embeddings.shape)

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        embeddings = self.language_model.model.embed_tokens(input_ids)
        if pixel_values is not None:
            features = kwargs.get("cached_image_features")
            if features is None:
                features = self.encode_images(pixel_values, **kwargs)
            if isinstance(features, (list, tuple)):
                features = mx.concatenate(features, axis=0)
            embeddings = self._merge(
                embeddings, input_ids, features, self.config.image_token_id
            )
        if kwargs.get("pixel_values_videos") is not None:
            features = self.encode_images(
                kwargs["pixel_values_videos"],
                image_grid_thw=kwargs.get("video_grid_thw"),
            )
            embeddings = self._merge(
                embeddings, input_ids, features, self.config.video_token_id
            )
        return InputEmbeddingsFeatures(inputs_embeds=embeddings)

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        features = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids,
            cache=cache,
            inputs_embeds=features.inputs_embeds,
            position_ids=kwargs.get("position_ids"),
        )

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        result = {}
        for name, value in weights.items():
            if name.startswith("model.language_model."):
                name = name.replace("model.language_model.", "language_model.model.", 1)
            elif name.startswith("model."):
                name = name[len("model.") :]
            elif name.startswith("lm_head."):
                name = "language_model." + name
            if (
                self.config.text_config.tie_word_embeddings
                and name == "language_model.lm_head.weight"
            ):
                continue
            if name.endswith("conv.conv.weight") and value.shape[1] == 1:
                value = value.transpose(0, 2, 1)
            elif name.endswith("spatial_conv.weight") and value.shape[1] == 1:
                value = value.transpose(0, 2, 3, 1)
            result[name] = value
        return result
