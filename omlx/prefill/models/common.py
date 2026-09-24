"""Shared runtime checks for explicitly supported local MLX models."""

from __future__ import annotations

from typing import Any


def local_weight_shape(
    module: Any, *, embedding: bool = False, experts: int | None = None
) -> tuple[int, int] | None:
    """Recognize local dense/affine weights without evaluating their arrays."""
    import mlx.core as mx
    import mlx.nn as nn

    supported_types = (
        (nn.Embedding, nn.QuantizedEmbedding)
        if embedding
        else (nn.Linear, nn.QuantizedLinear)
    )
    if experts is not None:
        from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear

        supported_types = (SwitchLinear, QuantizedSwitchLinear)
    if type(module) not in supported_types:
        return None
    weight = getattr(module, "weight", None)
    if not isinstance(weight, mx.array) or weight.ndim != (3 if experts else 2):
        return None
    if experts is not None and weight.shape[0] != experts:
        return None
    output_size, packed_input_size = weight.shape[-2:]
    if type(module) is supported_types[0]:
        if weight.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            return None
        return output_size, packed_input_size
    bits = getattr(module, "bits", None)
    if (
        getattr(module, "mode", None) != "affine"
        or type(bits) is not int
        or bits not in (2, 3, 4, 5, 6, 8)
        or weight.dtype != mx.uint32
        or packed_input_size * 32 % bits
    ):
        return None
    return output_size, packed_input_size * 32 // bits


def execution_fallback(model_parts: tuple[Any, ...]) -> str | None:
    if any(
        getattr(part, "pipeline_size", 1) != 1
        or getattr(part, "pipeline_rank", 0) != 0
        or getattr(part, "sharding_group", None) is not None
        for part in model_parts
    ):
        return "distributed_execution"
    if any(hasattr(part, "_omlx_prefill") for part in model_parts):
        return "custom_execution"

    custom_execution_flags = (
        "_omlx_mtp_decode_enabled",
        "_omlx_ane_mlp_prefill_count",
        "_omlx_ane_gdn_prefill_count",
        "_omlx_ane_down_prefill_count",
        "_omlx_ane_dual_prefill_count",
    )
    if any(
        getattr(part, flag, False)
        for part in model_parts
        for flag in custom_execution_flags
    ):
        return "custom_execution"
    return None
