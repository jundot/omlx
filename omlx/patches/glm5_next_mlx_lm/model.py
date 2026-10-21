# SPDX-License-Identifier: Apache-2.0
"""``mlx_lm.models.glm5_next``: GLM-5.3 in the shape an mlx-lm rank loads.

The layers and kernels are the vendored mlx-vlm tree's. This adds a text-only
root whose parameter paths match the checkpoint index, a forward over one
pipeline stage, and the rank-zero logits contract. Single-node serving does not
use this module.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.pipeline import PipelineMixin

from omlx.cluster.pipeline_compat import _cache_dependency
from omlx.model_discovery import _VISION_WEIGHT_PREFIXES
from omlx.patches.deepseek_v4.switch_layers import install_nvfp4_global_scales
from omlx.patches.mlx_vlm_glm5_next_compat import (
    apply_mlx_vlm_glm5_next_compat_patch,
)
from omlx.utils.layer_pipeline import LayerPipeline

# ``mlx_vlm.models.glm5_next`` resolves to oMLX's vendored tree only after this.
apply_mlx_vlm_glm5_next_compat_patch()

from mlx_vlm.models.glm5_next import language as _language  # noqa: E402
from mlx_vlm.models.glm5_next.config import ModelConfig, TextConfig  # noqa: E402

# Read by the planner, which would otherwise refuse a checkpoint with a vision
# config. The inherited PipelineMixin.pipeline is the one the worker replaces
# with the planner's split, so the assignment contract is found there.
SUPPORTS_PIPELINE = True


class ModelArgs(ModelConfig):
    """``ModelConfig`` with the nested configs built, as mlx-vlm's loader does."""

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> ModelArgs:
        args = super().from_dict(params)
        args.text_config = TextConfig.from_dict(params["text_config"])
        args.vision_config = None
        return args


def _keep_send_in_graph(cache_entry: Any, value: mx.array) -> None:
    # A sparse-attention layer caches CacheList(KVCache, PoolingCache).
    caches = getattr(cache_entry, "caches", None)
    if caches:
        cache_entry = caches[0]
    _cache_dependency(cache_entry, value, mx)


class PipelinedTextModel(_language.Glm5NextModel, PipelineMixin):
    """``Glm5NextModel`` that may hold one pipeline stage.

    Ranks are numbered in reverse: the highest rank holds layer 0 and rank 0
    holds the last layers and produces the output.
    """

    # The vendored __init__ does not chain to PipelineMixin.__init__.
    pipeline_rank = 0
    pipeline_size = 1
    start_idx = 0
    end_idx = None

    def __call__(
        self,
        inputs: mx.array,
        cache: Any = None,
        inputs_embeds: mx.array | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size
        layers = self.pipeline_layers
        if cache is None:
            cache = [None] * len(layers)

        # Cache slots are local to this stage, which may lack either kind.
        fa_idx = next(
            (i for i, layer in enumerate(layers) if not layer.is_linear), None
        )
        ssm_idx = next((i for i, layer in enumerate(layers) if layer.is_linear), None)
        fa_cache = cache[fa_idx] if fa_idx is not None else None
        fa_mask = _language.create_attention_mask(
            h, fa_cache[0] if fa_cache else None, return_array=True
        )
        ssm_mask = _language.create_ssm_mask(
            h, cache[ssm_idx] if ssm_idx is not None else None
        )

        h = mx.broadcast_to(
            h[:, :, None, :], (h.shape[0], h.shape[1], self.hc_mult, h.shape[2])
        )
        h = mx.contiguous(h)
        # Stage handoffs travel in this dtype: recv_like takes it from the
        # embedding. GLM-5.3 checkpoints already keep layer outputs in it.
        boundary_dtype = h.dtype
        # Later stages use the embedding only as the shape to receive into.
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, pipeline_rank + 1)

        # Same prefill memory and decode scheduling policy as the vendored
        # forward.
        prefill = h.shape[1] >= 256
        layer_pipeline = (
            LayerPipeline(on_evaluated=mx.clear_cache, lazy_last=True)
            if prefill
            else None
        )
        eval_every = (
            _language._DECODE_EVAL_EVERY
            if h.shape[0] == 1 and h.shape[1] <= _language._DECODE_BLOCK
            else 0
        )
        n_layers = len(layers)
        defer = h.shape[:2] == (1, 1)
        for i, (layer, c) in enumerate(zip(layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            if defer:
                # The last layer of a stage returns an array, not a deferral.
                h = layer(h, mask=mask, cache=c, defer=i + 1 < n_layers)
            else:
                h = layer(h, mask=mask, cache=c)
            if layer_pipeline is not None:
                layer_pipeline.push(h)
            elif eval_every and (i + 1) % eval_every == 0 and i + 1 < n_layers:
                mx.async_eval(h.arrays() if isinstance(h, _language._HCDeferred) else h)
        if layer_pipeline is not None:
            layer_pipeline.drain()

        if pipeline_rank != 0:
            h = mx.distributed.send(
                h.astype(boundary_dtype), (pipeline_rank - 1) % pipeline_size
            )
            if cache:
                _keep_send_in_graph(cache[-1], h)

        # Collapse the streams first so the gather moves (B, S, D), not HC
        # times that. Only rank 0's rows are kept.
        h = h.mean(axis=2)
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h.astype(boundary_dtype))[: h.shape[0]]
        return self.norm(h)


class Model(nn.Module):
    """GLM-5.3 text model, presented the way mlx-lm loads and serves one."""

    # Worker ranks may advance their stage without the vocabulary projection.
    _omlx_supports_rank_zero_logits = True

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = _language.LanguageModel(args.text_config, args)
        # Same instance and parameters, with the pipeline contract added.
        self.language_model.model.__class__ = PipelinedTextModel

    def __call__(
        self,
        inputs: mx.array,
        cache: Any = None,
        skip_logits: bool = False,
    ) -> mx.array:
        if skip_logits:
            return self.language_model.model(inputs, cache=cache)
        return self.language_model(inputs, cache=cache).logits

    @property
    def model(self) -> PipelinedTextModel:
        """mlx-lm pipelines ``model.model`` when it has ``pipeline()``."""

        return self.language_model.model

    @property
    def layers(self) -> list:
        return self.language_model.model.pipeline_layers

    @property
    def _omlx_output_vocab_size(self) -> int:
        return int(self.args.text_config.vocab_size)

    def make_cache(self) -> list:
        # The vendored make_cache only reads ``layers``; give it this stage's.
        return _language.LanguageModel.make_cache(SimpleNamespace(layers=self.layers))

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        text = {}
        for key, value in weights.items():
            if key.startswith(_VISION_WEIGHT_PREFIXES):
                continue
            if key.startswith("model.language_model."):
                key = "model." + key[len("model.language_model.") :]
            elif key.startswith("language_model."):
                key = key[len("language_model.") :]
            text[key] = value
        sanitized = self.language_model.sanitize(text)
        return install_nvfp4_global_scales(
            self, {f"language_model.{key}": value for key, value in sanitized.items()}
        )
