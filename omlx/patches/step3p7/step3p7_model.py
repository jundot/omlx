# SPDX-License-Identifier: Apache-2.0
"""Step 3.7 Flash text-only wrapper for mlx-lm.

Vendored from ml-explore/mlx-lm#1325 so oMLX can load Step 3.7 text-only
MLX checkpoints while the pinned mlx-lm does not yet ship this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.step3p5 import Model as Step3p5Model
from mlx_lm.models.step3p5 import ModelArgs as TextConfig


@dataclass
class ModelArgs(BaseModelArgs):
    text_config: TextConfig | dict
    model_type: str = "step3p7"

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.language_model = Step3p5Model(config.text_config)

    def __call__(
        self,
        inputs: mx.array,
        cache: Any | None = None,
    ):
        return self.language_model(inputs, cache)

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith("vision_model")
            and not k.startswith("vit_large_projector")
        }
        weights = self.language_model.sanitize(weights)
        return {
            k if k.startswith("language_model.") else f"language_model.{k}": v
            for k, v in weights.items()
        }

    def shard(self, group: mx.distributed.Group | None = None):
        self.language_model.shard(group)

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate
