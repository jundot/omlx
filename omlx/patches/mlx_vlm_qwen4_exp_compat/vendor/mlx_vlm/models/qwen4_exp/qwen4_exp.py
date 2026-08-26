# SPDX-License-Identifier: Apache-2.0
"""Multimodal shell for the Qwen4-Exp language runtime."""

import mlx.core as mx
import mlx.nn as nn

from ..qwen3_5 import Model as Qwen3_5Model
from .config import ModelConfig
from .language import (
    DiskBackedQuantizedEmbedding,
    LanguageModel,
    compile_hyper_connections,
    fuse_hyper_connection_projections,
)

_TEXT_FIRST_IGNORED_PREFIXES = (
    "mtp.",
    "language_model.mtp.",
    "model.mtp.",
    "vision_tower.",
)
_PLE_SHARD_FRAGMENT = ".ple.ple_embedding.ngram_embedding.shard_"


def filter_text_first_weights(weights, *, drop_ple: bool = False):
    """Drop deferred towers and normalize the two observed PLE conv layouts."""
    filtered = []
    for name, value in weights:
        if name.startswith(_TEXT_FIRST_IGNORED_PREFIXES):
            continue
        if drop_ple and _PLE_SHARD_FRAGMENT in name:
            continue
        if (
            name.endswith(".ple.conv1d.weight")
            and getattr(value, "ndim", None) == 3
            and value.shape[1] != 1
            and value.shape[2] == 1
        ):
            value = mx.moveaxis(value, 1, 2)
        filtered.append((name, value))
    return filtered


class TextOnlyVisionTower(nn.Module):
    """Parameter-free placeholder used by the first text-only runtime."""

    def __call__(self, *_args, **_kwargs):
        raise ValueError(
            "Qwen4-Exp vision inputs are disabled in the text-first oMLX runtime"
        )


class Model(Qwen3_5Model):
    """Reuse Qwen3.5's input-merging API with the Qwen4 text decoder."""

    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config
        self.vision_tower = TextOnlyVisionTower()
        self.language_model = LanguageModel(config.text_config, config)
        self._disk_backed_ple = any(
            layer.ple is not None
            and isinstance(
                layer.ple.ple_embedding.ngram_embedding,
                DiskBackedQuantizedEmbedding,
            )
            for layer in self.language_model.model.layers
        )

    def load_weights(self, weights, strict=True):
        result = super().load_weights(
            filter_text_first_weights(weights, drop_ple=self._disk_backed_ple),
            strict=strict,
        )
        fuse_hyper_connection_projections(self)
        compile_hyper_connections(self)
        return result


__all__ = ["Model", "TextOnlyVisionTower", "filter_text_first_weights"]
