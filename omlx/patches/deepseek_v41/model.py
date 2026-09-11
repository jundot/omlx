# SPDX-License-Identifier: MIT
"""V4.1 text/vision model exposing the mlx-vlm embedding adapter contract."""

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import InputEmbeddingsFeatures

from .config import ModelConfig
from .language import LanguageModel
from .vision import Aligner, ViT


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)
        if config.vision_enabled:
            self.vision = ViT(config)
            self.aligner = Aligner(config)
            self.image_start = mx.zeros((config.dim,))
            self.image_newline = mx.zeros((config.dim,))
            self.image_end = mx.zeros((config.dim,))

    def get_input_embeddings(self, input_ids, pixel_values=None, **kwargs):
        h = self.language_model.embed(input_ids)
        if pixel_values is None:
            if bool(mx.any(input_ids == self.config.image_token_id).item()):
                raise ValueError("Missing pixels for image tokens")
            return InputEmbeddingsFeatures(inputs_embeds=h)
        if not self.config.vision_enabled:
            raise ValueError("No vision encoder in this configuration")
        if input_ids.shape[0] != 1:
            raise ValueError("Prepare image embeddings per request")
        grids, spans, types = (
            kwargs[k] for k in ("image_grids", "image_spans", "image_types")
        )
        if not (len(grids) == len(spans) == len(types)):
            raise ValueError("Image metadata counts differ")
        offset = 0
        for (height, width), (start, length), kinds in zip(grids, spans, types):
            count = height * width
            features = self.aligner(
                self.vision(pixel_values[offset : offset + count], height, width),
                height,
                width,
            )
            offset += count
            if kinds.count(1) != len(features) or length != len(kinds):
                raise ValueError("Image layout does not match aligned features")
            values, index = [], 0
            delimiters = {0: self.image_start, 2: self.image_newline, 3: self.image_end}
            for kind in kinds:
                if kind == 1:
                    values.append(features[index])
                    index += 1
                else:
                    values.append(delimiters[kind])
            if not bool(
                mx.all(
                    input_ids[0, start : start + length] == self.config.image_token_id
                ).item()
            ):
                raise ValueError("Image span does not cover image token ids")
            h[:, start : start + length] = mx.stack(values).astype(h.dtype)
        if offset != pixel_values.shape[0]:
            raise ValueError("Unused image patches")
        return InputEmbeddingsFeatures(inputs_embeds=h)

    def __call__(self, input_ids, pixel_values=None, cache=None, **kwargs):
        embeddings = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        return self.language_model(
            input_ids,
            cache=cache,
            inputs_embeds=(
                embeddings.inputs_embeds if pixel_values is not None else None
            ),
            **{
                k: kwargs[k]
                for k in ("return_hidden", "return_dspark_hidden", "n_confirmed")
                if k in kwargs
            },
        )

    def close(self):
        prefetch = getattr(self.language_model, "_engram_prefetch", None)
        try:
            if prefetch is not None:
                prefetch.close()
        finally:
            for layer in self.language_model.layers:
                if "engram" in layer:
                    close = getattr(layer.engram.embed, "close", None)
                    if close is not None:
                        close()
