from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.muse_spark.muse_spark_model import (
    MuseSparkModel,
    make_cache,
    softcap_logits,
)

from ..base import LanguageModelOutput
from .config import TextConfig


class LanguageModel(nn.Module):
    """mlx-vlm wrapper around the oMLX mlx-lm text model.

    It has no ``sanitize``: mlx-vlm would build a second language model only
    to call it. ``Model.sanitize`` maps the weights instead.
    """

    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = MuseSparkModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        hidden = self.model(inputs, cache, inputs_embeds)
        return LanguageModelOutput(
            logits=softcap_logits(self.lm_head(hidden), self.args)
        )

    @property
    def layers(self):
        return self.model.pipeline_layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        return make_cache(self.args, self.layers)
