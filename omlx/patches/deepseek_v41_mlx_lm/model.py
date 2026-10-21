# SPDX-License-Identifier: Apache-2.0
"""``mlx_lm.models.deepseek_v41``: oMLX's DeepSeek V4.1 for a cluster rank.

The worker's progressive loader calls ``load_pipeline_stage`` or
``load_tensor_shard`` instead of mlx-lm's loader: the checkpoint format
(packed projections, Engram tables) is oMLX's own. The planner reads
``pipeline_stage_starts`` because layers share compressed KV and index state
with earlier source layers.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.language import LanguageModel
from omlx.patches.deepseek_v41.loading import load
from omlx.patches.deepseek_v41.model import Model as _LoadedModel


class ModelArgs(ModelConfig):
    """The config of a checkpoint a cluster rank can load."""

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        # Ranks read their part of converted shards. A source checkpoint is
        # converted while one Mac loads it whole.
        if "omlx_deepseek_v41" not in params:
            raise ValueError(
                "DeepSeek V4.1 cluster ranks load converted checkpoints only"
            )
        return super().from_dict(params)


# Read by the planner. Ranks load neither the vision tower nor the draft heads,
# and the vision config would otherwise make the planner refuse a split.
# LanguageModel inherits PipelineMixin.pipeline, which the worker replaces with
# the planner's split.
CLUSTER_UNLOADED_WEIGHT_PREFIXES = (
    "vision.",
    "aligner.",
    "image_",
    "language_model.mtp.",
)
SUPPORTS_PIPELINE = True

__all__ = [
    "LanguageModel",
    "Model",
    "ModelArgs",
    "load_pipeline_stage",
    "load_tensor_shard",
    "pipeline_stage_starts",
]


def pipeline_stage_starts(config: dict[str, Any]) -> tuple[int, ...]:
    return ModelConfig.from_dict(config).pipeline_stage_starts()


class Model(_LoadedModel):
    """A loaded stage, presented the way mlx-lm serves a model."""

    # Worker ranks may advance their stage without the vocabulary projection.
    _omlx_supports_rank_zero_logits = True

    def __call__(
        self,
        inputs: mx.array,
        cache: Any = None,
        skip_logits: bool = False,
    ) -> mx.array:
        return self.language_model(inputs, cache=cache, skip_logits=skip_logits)

    @property
    def model(self) -> LanguageModel:
        """mlx-lm and the worker read the pipeline stage from ``model.model``."""

        return self.language_model

    @property
    def layers(self) -> list:
        return self.language_model.pipeline_layers

    @property
    def _omlx_output_vocab_size(self) -> int:
        return int(self.config.vocab_size)

    def make_cache(self) -> list:
        return self.language_model.make_cache()


def load_pipeline_stage(model_path: Any, group: Any) -> Model:
    """This rank's stage of a converted checkpoint, text model only."""

    model, _ = load(
        model_path, stage=lambda language_model: language_model.pipeline(group)
    )
    model.__class__ = Model
    return model


def load_tensor_shard(model_path: Any, group: Any) -> Model:
    """This rank's tensor-parallel slice of a converted checkpoint, text model only."""

    # Every rank reads every Engram row, so the tables stay on SSD.
    model, _ = load(model_path, tensor=group, engram_ssd_offload=True)
    model.__class__ = Model
    return model
