# SPDX-License-Identifier: Apache-2.0
"""Streaming FP8 PLE support for Qwen4-Exp oQ conversion.

The FP8 Qwen3.8-Flash-Next checkpoint variant stores each N-gram PLE shard
as raw F8_E4M3 rows and keeps one shared ``weight_scale`` for the whole
sharded embedding.  That layout is intentionally different from the
ordinary ``weight`` + per-module scale pairs handled by
:class:`_LazyTensorIndex`.  The official BF16 checkpoint uses the same shard
names but remains on the normal floating-point path.

The runtime decodes a selected FP8 row first and applies the shared scale
afterwards.  oQ must preserve the same ordering: decode the raw FP8 shard,
quantize those unscaled values to affine storage, and keep ``weight_scale``
for the runtime to apply exactly once.  Exposing the shard as a virtual BF16
tensor makes sanitize-plan discovery and the streaming quantizer see the
correct logical dtype without materializing the 51.2B PLE table eagerly.
"""

from __future__ import annotations

import logging
import weakref

import mlx.core as mx

logger = logging.getLogger(__name__)

_FP8_DTYPE = "F8_E4M3"


def _is_qwen4_exp(config: dict) -> bool:
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        text_config = {}
    return any(
        str(model_type or "").lower().startswith("qwen4_exp")
        for model_type in (config.get("model_type"), text_config.get("model_type"))
    )


def _candidate_keys(config: dict):
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        return
    num_shards = int(text_config.get("split_ngram_parts", 0) or 0)
    if num_shards <= 0:
        return
    for layer_id in text_config.get("ple_layer_ids", ()) or ():
        layer_index = int(layer_id) - 1
        if layer_index < 0:
            continue
        prefixes = (
            f"model.language_model.layers.{layer_index}.ple.ple_embedding."
            "ngram_embedding",
            f"language_model.model.layers.{layer_index}.ple.ple_embedding."
            "ngram_embedding",
        )
        for shard_index in range(num_shards):
            for prefix in prefixes:
                yield f"{prefix}.shard_{shard_index}.weight"
                yield f"{prefix}.shards.{shard_index}.weight"


def _materializer(index_ref, key: str):
    def materialize():
        index = index_ref()
        if index is None:
            raise RuntimeError(
                "Qwen4 PLE materializer outlived its tensor index; the index "
                "must stay alive while virtual tensors are being read"
            )
        raw = index.load_source(key)
        decoded = mx.from_fp8(raw, dtype=mx.bfloat16)
        mx.eval(decoded)
        del raw
        return decoded

    return materialize


def register(index, config: dict) -> int:
    """Expose raw Qwen4 PLE FP8 shards as unscaled BF16 virtual tensors."""
    if not _is_qwen4_exp(config):
        return 0

    index_ref = weakref.ref(index)
    registered = 0
    for key in _candidate_keys(config):
        shape = index.source_shape(key)
        if shape is None or index.source_dtype(key) != _FP8_DTYPE:
            continue
        if len(shape) != 2:
            raise ValueError(f"Qwen4 PLE shard must be rank 2, got {key}: {shape}")
        index.register_virtual(
            key,
            shape,
            "BF16",
            _materializer(index_ref, key),
            hides=(key,),
        )
        registered += 1

    if registered:
        logger.info(
            "Qwen4 PLE: %d raw FP8 shards exposed as streaming BF16 tensors",
            registered,
        )
    return registered
