# Copyright 2026 The HuggingFace Team. All rights reserved.
# Adapted for MLX from the LFM3/LFM3-VL checkpoint reference implementations.
# Licensed under the Apache License, Version 2.0; see LICENSE in ../lfm3.
# SPDX-License-Identifier: Apache-2.0

import mlx.nn as nn

from ..base import InputEmbeddingsFeatures
from .language import LanguageModel


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(self, input_ids, pixel_values=None, mask=None, cache=None, **kwargs):
        return self.language_model(input_ids, cache=cache, **kwargs)

    def sanitize(self, weights):
        result = {}
        for name, value in weights.items():
            if name.startswith(("model.", "lm_head.")):
                name = "language_model." + name
            if (
                self.config.tie_word_embeddings
                and name == "language_model.lm_head.weight"
            ):
                continue
            if name.endswith("conv.conv.weight") and value.shape[1] == 1:
                value = value.transpose(0, 2, 1)
            result[name] = value
        return result

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()
