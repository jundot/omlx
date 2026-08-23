# SPDX-License-Identifier: Apache-2.0
"""Opt-in ANE/CPU/GPU prefill for DeepSeek-V4-Flash dense projections.

Splits the mxfp8 shared-expert gate/up and down projections, the attention-input
projections, grouped wo_a, and the wq_b attention projection (with the
sparse-layer indexer wq_b stacked into the same procedure) between an INT8 ANE
channel prefix and an affine-q8 GPU suffix.
The query projections may also place a benchmark-selected FP16 middle slice
on the CPU using Qwen's shared-resource scheduler. Routed experts, shared MLP
CPU work, wo_b, decode, and DSpark verify calls keep their existing paths.
Enabled through the per-model
``deepseek_ane_prefill_*`` settings on the batched engine.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx

from omlx.custom_kernels.nax import is_nax_available
from omlx.patches.deepseek_v4.decode_consistency import is_armed
from omlx.patches.qwen35_ane_prefill import (
    _bank_split_ladder,
    _compile_dual_banks,
    _eligible_input,
)

logger = logging.getLogger(__name__)

# The suffix is requantized from mxfp8 to affine q8 so the existing hybrid
# merge primitive can dispatch it; measured GPU time is format-neutral.
_SUFFIX_BITS = 8
_SUFFIX_GROUP_SIZE = 64
_PROFILE_MLP = "deepseek_mlp"
_PROFILE_DOWN = "deepseek_down"
_PROFILE_ATTENTION_INPUT = "deepseek_attention_input"
_PROFILE_QUERY = "deepseek_query"
_PROFILE_WO_A = "deepseek_wo_a"
# Fractions follow the Phase A shape microbench optima. The indexer wq_b
# rides inside the attention wq_b op (same q_residual input), so it adds no
# operations of its own. wo_b is deliberately not accelerated: its per-op
# host synchronization exceeded the offload savings in-model at K=8192.
_FRACTIONS = {
    "mlp": 0.5,
    # Requested 65% becomes 62.5% after the 128-row per-instance alignment.
    "mlp_down": 0.65,
    "wo_a": 0.5,
    "wq_b": 0.5,
    "indexer_wq_b": 0.5,
    # Exact 4K shape probes. Local and ratio-128 stacks balance at 50%; the
    # wider ratio-4 stack remains GPU-bound until the requested 60% point
    # (56.25% after the 128-row per-instance alignment).
    "attention_input": 0.5,
    "attention_input_sparse": 0.6,
}


@dataclass(frozen=True)
class _AneConfig:
    sequence_length: int
    cpu_threads: int = 12
    cpu_shared_resource: bool = True
    tail_padding_min_tokens: int = 0


@dataclass(frozen=True)
class _LinearState:
    model: Any
    model1: Any
    weight: mx.array
    scales: mx.array
    biases: mx.array
    ane_outputs: int
    gpu_outputs: int
    cpu_weight: mx.array | None = None
    cpu_outputs: int = 0


@dataclass(frozen=True)
class _MlpState:
    model: Any
    model1: Any
    weight: mx.array
    scales: mx.array
    biases: mx.array
    ane_outputs: int
    gpu_outputs: int
    cpu_weight: mx.array | None = None
    cpu_outputs: int = 0


@dataclass(frozen=True)
class _StackedState:
    model: Any
    model1: Any
    weight: mx.array
    scales: mx.array
    biases: mx.array
    ane_outputs: int
    gpu_outputs: int
    split: int
    cpu_weight: mx.array | None = None
    cpu_outputs: int = 0


@dataclass(frozen=True)
class _AttentionInputState:
    model: Any
    model1: Any
    weight: mx.array
    scales: mx.array
    biases: mx.array
    ane_outputs: int
    gpu_outputs: int
    segments: tuple[tuple[str, int], ...]
    cpu_weight: mx.array | None = None
    cpu_outputs: int = 0


@dataclass(frozen=True)
class _GroupedLinearState:
    model: Any
    model1: Any
    weight: mx.array
    scales: mx.array
    biases: mx.array
    ane_outputs: int
    gpu_outputs: int
    groups: int


def _mxfp8_linear_dims(linear: Any) -> tuple[int, int] | None:
    """Return (out_features, in_features) for an eligible mxfp8 linear."""
    if getattr(linear, "mode", None) != "mxfp8":
        return None
    if getattr(linear, "bits", None) != 8 or getattr(linear, "group_size", None) != 32:
        return None
    weight = getattr(linear, "weight", None)
    scales = getattr(linear, "scales", None)
    if weight is None or scales is None:
        return None
    if weight.dtype != mx.uint32 or scales.dtype != mx.uint8 or weight.ndim != 2:
        return None
    if "bias" in linear:
        return None
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1]) * 4
    if in_features % _SUFFIX_GROUP_SIZE:
        return None
    return out_features, in_features


def _dequant_rows(linear: Any, start: int, stop: int) -> mx.array:
    return mx.dequantize(
        linear.weight[start:stop],
        linear.scales[start:stop],
        group_size=32,
        bits=8,
        mode="mxfp8",
    )


def _mxfp8_grouped_dims(linear: Any) -> tuple[int, int, int] | None:
    """Return (groups, out_per_group, in_per_group) for MultiLinear."""
    if getattr(linear, "mode", None) != "mxfp8":
        return None
    if getattr(linear, "bits", None) != 8 or getattr(linear, "group_size", None) != 32:
        return None
    weight = getattr(linear, "weight", None)
    scales = getattr(linear, "scales", None)
    if (
        weight is None
        or scales is None
        or weight.dtype != mx.uint32
        or scales.dtype != mx.uint8
        or weight.ndim != 3
        or scales.ndim != 3
        or getattr(linear, "bias", None) is not None
    ):
        return None
    groups = int(weight.shape[0])
    out_features = int(weight.shape[1])
    in_features = int(weight.shape[2]) * 4
    if groups < 2 or in_features % _SUFFIX_GROUP_SIZE:
        return None
    return groups, out_features, in_features


def _dequant_grouped_rows(linear: Any, start: int, stop: int) -> mx.array:
    return mx.concatenate(
        [
            mx.dequantize(
                linear.weight[group, start:stop],
                linear.scales[group, start:stop],
                group_size=32,
                bits=8,
                mode="mxfp8",
            )
            for group in range(int(linear.weight.shape[0]))
        ]
    )


def _quantized_linear_dims(linear: Any) -> tuple[int, int] | None:
    """Return dimensions for a 2-D MLX quantized linear.

    DeepSeek attention projections are normally mxfp8, while the indexer
    weight projection inherits the model's affine-q8 default. The combined
    attention-input procedure requantizes one common suffix, so it can safely
    accept either source format after dequantization.
    """
    weight = getattr(linear, "weight", None)
    scales = getattr(linear, "scales", None)
    group_size = int(getattr(linear, "group_size", 0) or 0)
    bits = int(getattr(linear, "bits", 0) or 0)
    mode = str(getattr(linear, "mode", "affine") or "affine")
    if (
        weight is None
        or scales is None
        or weight.dtype != mx.uint32
        or weight.ndim != 2
        or scales.ndim != 2
        or group_size <= 0
        or bits <= 0
        or mode not in ("affine", "mxfp8")
        or "bias" in linear
    ):
        return None
    out_features = int(weight.shape[0])
    in_features = int(scales.shape[1]) * group_size
    if in_features <= 0 or in_features % _SUFFIX_GROUP_SIZE:
        return None
    return out_features, in_features


def _dequant_quantized_rows(linear: Any, start: int, stop: int) -> mx.array:
    biases = getattr(linear, "biases", None)
    return mx.dequantize(
        linear.weight[start:stop],
        linear.scales[start:stop],
        biases[start:stop] if biases is not None else None,
        group_size=int(linear.group_size),
        bits=int(linear.bits),
        mode=str(getattr(linear, "mode", "affine") or "affine"),
    )


def _requant_suffix(rows: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    weight, scales, biases = mx.quantize(
        rows, group_size=_SUFFIX_GROUP_SIZE, bits=_SUFFIX_BITS
    )
    return mx.contiguous(weight), mx.contiguous(scales), mx.contiguous(biases)


def _split_rows(
    out_features: int,
    ane_fraction: float,
    cpu_fraction: float = 0.0,
) -> tuple[int, int, int] | None:
    per_instance = int(out_features * ane_fraction / 2 // 128) * 128
    cpu_outputs = int(out_features * cpu_fraction // 64) * 64
    gpu_outputs = out_features - 2 * per_instance - cpu_outputs
    if per_instance < 128 or gpu_outputs < 64 or gpu_outputs % 64:
        return None
    return per_instance, cpu_outputs, gpu_outputs


def _prepare_linear(
    linear: Any, ane_fraction: float, cpu_fraction: float = 0.0
) -> tuple[_LinearState, mx.array, mx.array] | None:
    dims = _mxfp8_linear_dims(linear)
    if dims is None:
        return None
    out_features, _ = dims
    split = _split_rows(out_features, ane_fraction, cpu_fraction)
    if split is None:
        return None
    per_instance, cpu_outputs, gpu_outputs = split
    ane_outputs = 2 * per_instance
    gpu_start = ane_outputs + cpu_outputs
    dense0 = mx.contiguous(_dequant_rows(linear, 0, per_instance).astype(mx.float32))
    dense1 = mx.contiguous(
        _dequant_rows(linear, per_instance, ane_outputs).astype(mx.float32)
    )
    cpu_weight = (
        mx.contiguous(_dequant_rows(linear, ane_outputs, gpu_start).astype(mx.float16))
        if cpu_outputs
        else None
    )
    suffix = _requant_suffix(
        mx.contiguous(_dequant_rows(linear, gpu_start, out_features))
    )
    state = _LinearState(
        None,
        None,
        *suffix,
        ane_outputs,
        gpu_outputs,
        cpu_weight,
        cpu_outputs,
    )
    return state, dense0, dense1


def _prepare_mlp(mlp: Any) -> tuple[_MlpState, mx.array, mx.array] | None:
    gate_dims = _mxfp8_linear_dims(getattr(mlp, "gate_proj", None))
    up_dims = _mxfp8_linear_dims(getattr(mlp, "up_proj", None))
    if gate_dims is None or gate_dims != up_dims:
        return None
    out_features, _ = gate_dims
    split = _split_rows(out_features, _FRACTIONS["mlp"])
    if split is None:
        return None
    per_instance, _, gpu_outputs = split
    ane_outputs = 2 * per_instance

    def block(start: int, stop: int) -> mx.array:
        return mx.concatenate(
            (
                _dequant_rows(mlp.gate_proj, start, stop),
                _dequant_rows(mlp.up_proj, start, stop),
            )
        )

    dense0 = mx.contiguous(block(0, per_instance).astype(mx.float32))
    dense1 = mx.contiguous(block(per_instance, ane_outputs).astype(mx.float32))
    suffix = _requant_suffix(mx.contiguous(block(ane_outputs, out_features)))
    state = _MlpState(None, None, *suffix, ane_outputs, gpu_outputs)
    return state, dense0, dense1


def _prepare_grouped_linear(
    linear: Any, ane_fraction: float
) -> tuple[_GroupedLinearState, mx.array, mx.array] | None:
    dims = _mxfp8_grouped_dims(linear)
    if dims is None:
        return None
    groups, out_features, _ = dims
    split = _split_rows(out_features, ane_fraction)
    if split is None:
        return None
    per_instance, _, gpu_outputs = split
    ane_outputs = 2 * per_instance
    dense0 = mx.contiguous(
        _dequant_grouped_rows(linear, 0, per_instance).astype(mx.float32)
    )
    dense1 = mx.contiguous(
        _dequant_grouped_rows(linear, per_instance, ane_outputs).astype(mx.float32)
    )
    suffix = _requant_suffix(
        mx.contiguous(_dequant_grouped_rows(linear, ane_outputs, out_features))
    )
    state = _GroupedLinearState(
        None,
        None,
        *suffix,
        ane_outputs,
        gpu_outputs,
        groups,
    )
    return state, dense0, dense1


def _prepare_stacked(
    attn_linear: Any,
    indexer_linear: Any,
    ane_fraction: float,
    cpu_fraction: float = 0.0,
) -> tuple[_StackedState, mx.array, mx.array] | None:
    attn_dims = _mxfp8_linear_dims(attn_linear)
    indexer_dims = _mxfp8_linear_dims(indexer_linear)
    if attn_dims is None or indexer_dims is None or attn_dims[1] != indexer_dims[1]:
        return None
    attn_out = attn_dims[0]
    out_features = attn_out + indexer_dims[0]
    split = _split_rows(out_features, ane_fraction, cpu_fraction)
    if split is None:
        return None
    per_instance, cpu_outputs, gpu_outputs = split
    ane_outputs = 2 * per_instance
    gpu_start = ane_outputs + cpu_outputs

    def stacked_rows(start: int, stop: int) -> mx.array:
        parts = []
        if start < attn_out:
            parts.append(_dequant_rows(attn_linear, start, min(stop, attn_out)))
        if stop > attn_out:
            parts.append(
                _dequant_rows(indexer_linear, max(start - attn_out, 0), stop - attn_out)
            )
        return parts[0] if len(parts) == 1 else mx.concatenate(parts)

    dense0 = mx.contiguous(stacked_rows(0, per_instance).astype(mx.float32))
    dense1 = mx.contiguous(stacked_rows(per_instance, ane_outputs).astype(mx.float32))
    cpu_weight = (
        mx.contiguous(stacked_rows(ane_outputs, gpu_start).astype(mx.float16))
        if cpu_outputs
        else None
    )
    suffix = _requant_suffix(mx.contiguous(stacked_rows(gpu_start, out_features)))
    state = _StackedState(
        None,
        None,
        *suffix,
        ane_outputs,
        gpu_outputs,
        attn_out,
        cpu_weight,
        cpu_outputs,
    )
    return state, dense0, dense1


def _attention_input_linears(attn: Any) -> tuple[tuple[str, Any], ...]:
    """The fixed set of projections that consume the same attention input."""
    candidates: list[tuple[str, Any]] = [
        ("wq_a", getattr(attn, "wq_a", None)),
        ("wkv", getattr(attn, "wkv", None)),
    ]
    compressor = getattr(attn, "compressor", None)
    if compressor is not None:
        candidates.extend(
            (
                ("compressor_wkv", getattr(compressor, "wkv", None)),
                ("compressor_wgate", getattr(compressor, "wgate", None)),
            )
        )
    indexer = getattr(attn, "indexer", None)
    indexer_compressor = getattr(indexer, "compressor", None)
    if indexer_compressor is not None:
        candidates.extend(
            (
                ("indexer_compressor_wkv", getattr(indexer_compressor, "wkv", None)),
                (
                    "indexer_compressor_wgate",
                    getattr(indexer_compressor, "wgate", None),
                ),
                ("indexer_weights", getattr(indexer, "weights_proj", None)),
            )
        )
    return tuple((name, linear) for name, linear in candidates if linear is not None)


def _prepare_attention_input(
    attn: Any,
) -> tuple[_AttentionInputState, mx.array, mx.array] | None:
    linears = _attention_input_linears(attn)
    if len(linears) < 2:
        return None
    dimensions = [_quantized_linear_dims(linear) for _, linear in linears]
    if any(dims is None for dims in dimensions):
        return None
    input_dims = {dims[1] for dims in dimensions if dims is not None}
    if len(input_dims) != 1:
        return None

    segments = tuple(
        (name, dims[0])
        for (name, _), dims in zip(linears, dimensions)
        if dims is not None
    )
    out_features = sum(size for _, size in segments)
    fraction = (
        _FRACTIONS["attention_input_sparse"]
        if getattr(attn, "indexer", None) is not None
        else _FRACTIONS["attention_input"]
    )
    split = _split_rows(out_features, fraction)
    if split is None:
        return None
    per_instance, _, gpu_outputs = split
    ane_outputs = 2 * per_instance

    boundaries: list[tuple[Any, int, int]] = []
    cursor = 0
    for (_, linear), (_, size) in zip(linears, segments):
        boundaries.append((linear, cursor, cursor + size))
        cursor += size

    def stacked_rows(start: int, stop: int) -> mx.array:
        parts = []
        for linear, lower, upper in boundaries:
            if start >= upper or stop <= lower:
                continue
            parts.append(
                _dequant_quantized_rows(
                    linear,
                    max(start - lower, 0),
                    min(stop, upper) - lower,
                )
            )
        return parts[0] if len(parts) == 1 else mx.concatenate(parts)

    dense0 = mx.contiguous(stacked_rows(0, per_instance).astype(mx.float32))
    dense1 = mx.contiguous(
        stacked_rows(per_instance, ane_outputs).astype(mx.float32)
    )
    suffix = _requant_suffix(
        mx.contiguous(stacked_rows(ane_outputs, out_features))
    )
    state = _AttentionInputState(
        None,
        None,
        *suffix,
        ane_outputs,
        gpu_outputs,
        segments,
    )
    return state, dense0, dense1


def _compile_grouped_dual_banks(
    weights0: list[mx.array],
    weights1: list[mx.array],
    sequence_length: int,
    groups: int,
) -> tuple[list[Any], list[Any], int] | None:
    from omlx.custom_kernels.qwen35_prefill import fast

    return _bank_split_ladder(
        [int(weight.nbytes) for weight in weights0],
        lambda start, stop: (
            fast.qwen35_ane_compile_linear_grouped_bank(
                weights0[start:stop], sequence_length, 1, groups
            ),
            fast.qwen35_ane_compile_linear_grouped_bank(
                weights1[start:stop], sequence_length, 2, groups
            ),
        ),
    )


def _padded_short_input(
    x: mx.array,
    config: _AneConfig,
    *,
    grouped: bool = False,
) -> mx.array | None:
    """Pad one profitable token tail without catching decode or DSpark verify."""
    if is_armed() or x.dtype not in (mx.float16, mx.bfloat16):
        return None
    if grouped:
        if x.ndim not in (3, 4) or (x.ndim == 4 and int(x.shape[0]) != 1):
            return None
    elif x.ndim != 3 or int(x.shape[0]) != 1:
        return None
    rows = int(x.shape[-2])
    threshold = int(config.tail_padding_min_tokens or 0)
    if not 0 < threshold <= rows < config.sequence_length:
        return None
    padding = [(0, 0)] * x.ndim
    padding[-2] = (0, config.sequence_length - rows)
    return mx.pad(x, padding)


def _profile_category(owner: Any) -> str:
    return str(getattr(owner, "_omlx_ane_profile_category", "gdn"))


def _profile_record(owner: Any, **values: int | float) -> None:
    try:
        from omlx.custom_kernels.qwen35_prefill import fast

        fast.qwen35_ane_profile_record(_profile_category(owner), **values)
    except Exception:
        logger.debug("Unable to record DeepSeek ANE dispatch metadata", exc_info=True)


def _profile_fallback(owner: Any, x: mx.array, config: _AneConfig | None) -> None:
    values = {"fallback_operations": 1}
    if is_armed():
        values["dspark_bypasses"] = 1
    elif getattr(owner, "_omlx_ane_failed", False):
        values["runtime_failures"] = 1
    elif config is None or getattr(owner, "_omlx_ane_state", None) is None:
        values["state_rejections"] = 1
    elif x.dtype not in (mx.float16, mx.bfloat16):
        values["dtype_rejections"] = 1
    else:
        values["shape_rejections"] = 1
    _profile_record(owner, **values)


def _hybrid_combined_exact(owner: Any, x: mx.array) -> mx.array | None:
    config = getattr(owner, "_omlx_ane_config", None)
    if config is None or getattr(owner, "_omlx_ane_failed", False):
        return None
    if is_armed() or not _eligible_input(x, config):
        return None
    state = owner._omlx_ane_state
    if state.scales.dtype != x.dtype:
        return None
    try:
        from omlx.custom_kernels.qwen35_prefill import fast

        profile_category = fast.qwen35_ane_profile_category_id(
            _profile_category(owner)
        )

        if state.cpu_weight is not None:
            return fast.qwen35_ane_dual_cpu_fp16_affine_qmm_t(
                x,
                state.cpu_weight,
                state.weight,
                state.scales,
                state.biases,
                state.model,
                state.model1,
                _SUFFIX_BITS,
                8,
                _SUFFIX_GROUP_SIZE,
                profile_category,
                config.cpu_threads,
                config.cpu_shared_resource,
            )
        return fast.qwen35_ane_dual_affine_qmm_t(
            x,
            state.weight,
            state.scales,
            state.biases,
            state.model,
            state.model1,
            _SUFFIX_BITS,
            8,
            _SUFFIX_GROUP_SIZE,
            profile_category,
        )
    except Exception:
        owner._omlx_ane_failed = True
        logger.warning(
            "Disabling ANE prefill for one DeepSeek projection after a runtime failure",
            exc_info=True,
        )
        return None


def _hybrid_combined(owner: Any, x: mx.array) -> mx.array | None:
    """Run an exact tile or zero-pad one configured profitable short tail."""
    rows = int(x.shape[-2]) if x.ndim >= 2 else 0
    combined = _hybrid_combined_exact(owner, x)
    if combined is not None:
        _profile_record(owner, exact_operations=1, logical_tokens=rows)
        return combined
    config = getattr(owner, "_omlx_ane_config", None)
    if config is None or getattr(owner, "_omlx_ane_failed", False):
        _profile_fallback(owner, x, config)
        return None
    padded_input = _padded_short_input(x, config)
    if padded_input is None:
        _profile_fallback(owner, x, config)
        return None
    padded = _hybrid_combined_exact(owner, padded_input)
    if padded is None:
        _profile_fallback(owner, x, config)
        return None
    _profile_record(
        owner,
        padded_operations=1,
        logical_tokens=rows,
        padded_tokens=config.sequence_length - rows,
    )
    return padded[..., :rows, :]


def _grouped_backend_exact(linear: Any, x: mx.array) -> mx.array | None:
    state = getattr(linear, "_omlx_ane_state", None)
    config = getattr(linear, "_omlx_ane_config", None)
    if (
        not isinstance(state, _GroupedLinearState)
        or config is None
        or getattr(linear, "_omlx_ane_failed", False)
        or is_armed()
        or x.dtype not in (mx.float16, mx.bfloat16)
        or x.ndim not in (3, 4)
        or (x.ndim == 4 and int(x.shape[0]) != 1)
        or int(x.shape[-3]) != state.groups
        or int(x.size // (int(x.shape[-1]) * state.groups)) != config.sequence_length
        or state.scales.dtype != x.dtype
    ):
        return None
    try:
        from omlx.custom_kernels.qwen35_prefill import fast

        profile_category = fast.qwen35_ane_profile_category_id(
            _profile_category(linear)
        )

        return fast.qwen35_ane_dual_grouped_affine_qmm_t(
            x,
            state.weight,
            state.scales,
            state.biases,
            state.model,
            state.model1,
            state.groups,
            _SUFFIX_BITS,
            8,
            _SUFFIX_GROUP_SIZE,
            profile_category,
        )
    except Exception:
        linear._omlx_ane_failed = True
        logger.warning(
            "Disabling grouped DeepSeek ANE prefill after a runtime failure",
            exc_info=True,
        )
        return None


def _grouped_backend(linear: Any, x: mx.array) -> mx.array | None:
    rows = int(x.shape[-2]) if x.ndim >= 2 else 0
    combined = _grouped_backend_exact(linear, x)
    if combined is not None:
        _profile_record(linear, exact_operations=1, logical_tokens=rows)
        return combined
    state = getattr(linear, "_omlx_ane_state", None)
    config = getattr(linear, "_omlx_ane_config", None)
    if (
        not isinstance(state, _GroupedLinearState)
        or config is None
        or getattr(linear, "_omlx_ane_failed", False)
    ):
        _profile_fallback(linear, x, config)
        return None
    padded_input = _padded_short_input(x, config, grouped=True)
    if padded_input is None:
        _profile_fallback(linear, x, config)
        return None
    padded = _grouped_backend_exact(linear, padded_input)
    if padded is None:
        _profile_fallback(linear, x, config)
        return None
    _profile_record(
        linear,
        padded_operations=1,
        logical_tokens=rows,
        padded_tokens=config.sequence_length - rows,
    )
    return padded[..., :rows, :]


def _linear_backend(linear: Any, x: mx.array) -> mx.array | None:
    # The raw merge emits [instance0, instance1, CPU, GPU], which is the
    # original channel order for this contiguous row split.
    if not isinstance(getattr(linear, "_omlx_ane_state", None), _LinearState):
        return None
    return _hybrid_combined(linear, x)


def _stacked_backend(attn_linear: Any, x: mx.array):
    state = getattr(attn_linear, "_omlx_ane_state", None)
    if not isinstance(state, _StackedState):
        return None
    combined = _hybrid_combined(attn_linear, x)
    if combined is None:
        return None
    return combined[..., : state.split], combined[..., state.split :]


def _attention_input_backend(attn: Any, x: mx.array):
    state = getattr(attn, "_omlx_ane_state", None)
    if not isinstance(state, _AttentionInputState):
        return None
    combined = _hybrid_combined(attn, x)
    if combined is None:
        return None
    outputs = {}
    start = 0
    for name, size in state.segments:
        outputs[name] = combined[..., start : start + size]
        start += size
    return outputs


def _mlp_backend(mlp: Any, x: mx.array) -> mx.array | None:
    if not isinstance(getattr(mlp, "_omlx_ane_state", None), _MlpState):
        return None
    combined = _hybrid_combined(mlp, x)
    if combined is None:
        return None
    state = mlp._omlx_ane_state
    half = state.ane_outputs // 2
    gate = mx.concatenate(
        (
            combined[..., 0:half],
            combined[..., 2 * half : 3 * half],
            combined[..., 4 * half : 4 * half + state.gpu_outputs],
        ),
        axis=-1,
    )
    up = mx.concatenate(
        (
            combined[..., half : 2 * half],
            combined[..., 3 * half : 4 * half],
            combined[..., 4 * half + state.gpu_outputs :],
        ),
        axis=-1,
    )
    module = importlib.import_module("mlx_lm.models.deepseek_v4")
    if getattr(mlp, "fp32_swiglu", False):
        hidden = module._limited_swiglu(
            gate.astype(mx.float32),
            up.astype(mx.float32),
            mlp.swiglu_limit,
        ).astype(x.dtype)
    else:
        hidden = module._limited_swiglu(gate, up, mlp.swiglu_limit)
    down = _linear_backend(mlp.down_proj, hidden)
    return mlp.down_proj(hidden) if down is None else down


def enable_deepseek_v4_ane_prefill(
    model: Any,
    *,
    sequence_length: int = 2048,
    down_enabled: bool = True,
    down_fraction: float = _FRACTIONS["mlp_down"],
    wo_a_enabled: bool = True,
    wo_a_fraction: float = _FRACTIONS["wo_a"],
    cpu_fraction: float = 0.125,
    cpu_threads: int = 12,
    cpu_shared_resource: bool = True,
    tail_padding_min_tokens: int = 0,
) -> int:
    """Enable the hybrid ANE backend on eligible DeepSeek-V4 projections.

    Returns the number of accelerated projections; zero is a safe no-op.
    """
    if sequence_length < 1024 or sequence_length % 64:
        raise ValueError("ANE prefill sequence_length must be a multiple of 64 >= 1024")
    if not 0.0 <= cpu_fraction < 0.5:
        raise ValueError("DeepSeek CPU fraction must be between 0.0 and 0.5")
    if not 0.0 < down_fraction < 1.0:
        raise ValueError(
            "DeepSeek shared-down ANE fraction must be between 0.0 and 1.0"
        )
    if not 0.0 < wo_a_fraction < 1.0:
        raise ValueError("DeepSeek wo_a ANE fraction must be between 0.0 and 1.0")
    if not 0 <= cpu_threads <= 64:
        raise ValueError("DeepSeek CPU worker count must be between 0 and 64")
    if not (
        tail_padding_min_tokens == 0
        or 2 <= tail_padding_min_tokens < sequence_length
    ):
        raise ValueError(
            "DeepSeek ANE tail padding threshold must be zero or between 2 "
            "and sequence_length - 1"
        )

    # The shared CPU middle is only competitive at the measured 4K tile.
    # At 2K, fresh shape probes put plain wq_b at 5.41 ms with the default
    # 12.5% CPU share versus 4.77 ms with ANE/GPU alone; stacked wq_b was
    # effectively flat (5.90 versus 5.81 ms). Keep the ANE/GPU experiment
    # available for smaller fixed shapes, but do not silently add a known
    # CPU regression when a user changes the prompt block size.
    if cpu_fraction and sequence_length < 4096:
        logger.warning(
            "DeepSeek query CPU offload disabled for sequence_length=%d; "
            "shape benchmarks require at least 4096 rows",
            sequence_length,
        )
        cpu_fraction = 0.0

    env = os.environ.get("OMLX_QWEN35_ANE_PREFILL", "").strip().lower()
    if env in ("0", "false", "off"):
        return 0
    if env not in ("1", "true", "on") and is_nax_available():
        logger.info(
            "DeepSeek ANE prefill skipped: NAX GPU, tensor-unit prefill is faster"
        )
        return 0

    try:
        from omlx.custom_kernels.qwen35_prefill import fast

        if not fast.qwen35_ane_available():
            logger.warning(
                "Private ANE runtime unavailable; DeepSeek ANE prefill skipped"
            )
            return 0
        if not fast.has_symbol("qwen35_ane_compile_linear_bank") or not fast.has_symbol(
            "qwen35_ane_dual_affine_qmm_t"
        ):
            logger.warning(
                "ANE extension predates procedure banks; DeepSeek ANE prefill skipped"
            )
            return 0
        grouped_available = bool(
            fast.has_symbol("qwen35_ane_compile_linear_grouped_bank")
            and fast.has_symbol("qwen35_ane_dual_grouped_affine_qmm_t")
        )
        if not grouped_available:
            logger.warning(
                "ANE extension predates grouped hybrid qmm; DeepSeek wo_a "
                "offload disabled"
            )
        if cpu_fraction and not fast.has_symbol(
            "qwen35_ane_dual_cpu_fp16_affine_qmm_t"
        ):
            logger.warning(
                "ANE extension predates CPU sharing; DeepSeek query CPU "
                "offload disabled"
            )
            cpu_fraction = 0.0
        if (
            cpu_fraction
            and cpu_shared_resource
            and not fast.qwen35_cpu_shared_resource_available()
        ):
            logger.warning(
                "Shared-resource CPU scheduling is unavailable; DeepSeek "
                "query CPU offload disabled"
            )
            cpu_fraction = 0.0
    except Exception:
        logger.warning("ANE native extension unavailable; DeepSeek ANE prefill skipped")
        return 0

    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        return 0

    config = _AneConfig(
        sequence_length,
        cpu_threads,
        cpu_shared_resource,
        tail_padding_min_tokens,
    )
    prepared: list[tuple[Any, Any, mx.array, mx.array]] = []
    grouped_prepared: list[tuple[Any, _GroupedLinearState, mx.array, mx.array]] = []
    mlp_count = 0
    down_count = 0
    wo_a_count = 0
    stacked_count = 0
    attention_input_count = 0
    for layer in layers:
        attn = getattr(layer, "attn", None)
        shared = getattr(getattr(layer, "ffn", None), "shared_experts", None)
        if shared is not None:
            try:
                prep = _prepare_mlp(shared)
            except Exception:
                logger.warning(
                    "Skipping one DeepSeek shared expert while preparing its "
                    "ANE procedure",
                    exc_info=True,
                )
                prep = None
            if prep is not None:
                shared._omlx_ane_profile_category = _PROFILE_MLP
                prepared.append((shared, *prep))
                mlp_count += 1
                down_prep = None
                if down_enabled:
                    try:
                        down_prep = _prepare_linear(
                            getattr(shared, "down_proj", None),
                            down_fraction,
                        )
                    except Exception:
                        logger.warning(
                            "Skipping one DeepSeek shared down projection while "
                            "preparing its ANE procedure",
                            exc_info=True,
                        )
                if down_prep is not None:
                    shared.down_proj._omlx_ane_profile_category = _PROFILE_DOWN
                    prepared.append((shared.down_proj, *down_prep))
                    down_count += 1
        if attn is not None:
            try:
                input_prep = _prepare_attention_input(attn)
            except Exception:
                logger.warning(
                    "Skipping one DeepSeek attention-input ANE procedure",
                    exc_info=True,
                )
                input_prep = None
            if input_prep is not None:
                attn._omlx_ane_profile_category = _PROFILE_ATTENTION_INPUT
                prepared.append((attn, *input_prep))
                attention_input_count += 1
            if grouped_available and wo_a_enabled:
                try:
                    wo_a_prep = _prepare_grouped_linear(
                        getattr(attn, "wo_a", None), wo_a_fraction
                    )
                except Exception:
                    logger.warning(
                        "Skipping one DeepSeek grouped wo_a ANE procedure",
                        exc_info=True,
                    )
                    wo_a_prep = None
                if wo_a_prep is not None:
                    attn.wo_a._omlx_ane_profile_category = _PROFILE_WO_A
                    grouped_prepared.append((attn.wo_a, *wo_a_prep))
                    wo_a_count += 1
        linear = getattr(attn, "wq_b", None)
        if linear is None:
            continue
        prep = None
        indexer_linear = getattr(getattr(attn, "indexer", None), "wq_b", None)
        if indexer_linear is not None:
            try:
                prep = _prepare_stacked(
                    linear,
                    indexer_linear,
                    _FRACTIONS["wq_b"],
                    cpu_fraction,
                )
            except Exception:
                logger.warning(
                    "Skipping one DeepSeek stacked wq_b while preparing its "
                    "ANE procedure",
                    exc_info=True,
                )
            if prep is not None:
                stacked_count += 1
        if prep is None:
            try:
                prep = _prepare_linear(
                    linear,
                    _FRACTIONS["wq_b"],
                    cpu_fraction,
                )
            except Exception:
                logger.warning(
                    "Skipping one DeepSeek wq_b while preparing its ANE procedure",
                    exc_info=True,
                )
                prep = None
        if prep is not None:
            linear._omlx_ane_profile_category = _PROFILE_QUERY
            prepared.append((linear, *prep))
    if not prepared and not grouped_prepared:
        return 0

    weights0 = [entry[2] for entry in prepared]
    weights1 = [entry[3] for entry in prepared]
    # Evaluate the requantized suffixes here as well: leaving them lazy would
    # replay their dequant/quantize chains on whichever thread first evaluates
    # a prefill chunk, with stream state from this loader thread.
    suffix_arrays = []
    for _, state, _, _ in prepared:
        suffix_arrays.extend((state.weight, state.scales, state.biases))
        if state.cpu_weight is not None:
            suffix_arrays.append(state.cpu_weight)
    if prepared:
        mx.eval(*weights0, *weights1, *suffix_arrays)
        banked = _compile_dual_banks(weights0, weights1, sequence_length)
        if banked is None:
            logger.warning("DeepSeek ANE prefill disabled: bank compilation failed")
            return 0
        models0, models1, resident_programs = banked
    else:
        models0, models1, resident_programs = [], [], 0
    for index, (owner, state, _, _) in enumerate(prepared):
        owner._omlx_ane_config = config
        owner._omlx_ane_state = replace(
            state, model=models0[index], model1=models1[index]
        )

    grouped_resident_programs = 0
    if grouped_prepared:
        grouped_weights0 = [entry[2] for entry in grouped_prepared]
        grouped_weights1 = [entry[3] for entry in grouped_prepared]
        grouped_suffixes = [
            value
            for _, state, _, _ in grouped_prepared
            for value in (state.weight, state.scales, state.biases)
        ]
        mx.eval(*grouped_weights0, *grouped_weights1, *grouped_suffixes)
        groups = grouped_prepared[0][1].groups
        grouped_banks = _compile_grouped_dual_banks(
            grouped_weights0, grouped_weights1, sequence_length, groups
        )
        if grouped_banks is None:
            logger.warning(
                "DeepSeek wo_a ANE offload disabled: grouped bank compilation failed"
            )
            grouped_prepared = []
            wo_a_count = 0
        else:
            grouped_models0, grouped_models1, grouped_resident_programs = grouped_banks
            for index, (owner, state, _, _) in enumerate(grouped_prepared):
                owner._omlx_ane_config = config
                owner._omlx_ane_state = replace(
                    state,
                    model=grouped_models0[index],
                    model1=grouped_models1[index],
                )

    module = importlib.import_module("mlx_lm.models.deepseek_v4")
    module.register_ane_linear_backend(_linear_backend)
    module.register_ane_mlp_backend(_mlp_backend)
    register_stacked = getattr(module, "register_ane_stacked_q_backend", None)
    if register_stacked is not None:
        register_stacked(_stacked_backend)
    register_attention_input = getattr(
        module, "register_ane_attention_input_backend", None
    )
    if register_attention_input is not None:
        register_attention_input(_attention_input_backend)
    register_grouped = getattr(module, "register_ane_grouped_linear_backend", None)
    if register_grouped is not None and grouped_prepared:
        register_grouped(_grouped_backend)

    count = len(prepared) + len(grouped_prepared)
    model._omlx_ane_mlp_prefill_count = mlp_count
    model._omlx_ane_gdn_prefill_count = 0
    model._omlx_ane_dual_prefill_count = count
    model._omlx_ane_resident_program_count = (
        resident_programs + grouped_resident_programs
    )
    model._omlx_ane_procedure_count = count
    model._omlx_ane_attention_input_prefill_count = attention_input_count
    model._omlx_ane_down_prefill_count = down_count
    model._omlx_ane_wo_a_prefill_count = wo_a_count
    model._omlx_ane_query_prefill_count = (
        len(prepared) - mlp_count - down_count - attention_input_count
    )
    model._omlx_ane_tail_padding_min_tokens = tail_padding_min_tokens
    model._omlx_ane_dspark_native_compatible = True
    model._omlx_ane_cpu_prefill_count = sum(
        state.cpu_weight is not None for _, state, _, _ in prepared
    )
    logger.info(
        "Eagerly compiled %d DeepSeek ANE procedures (%d shared experts, "
        "%d shared down projections, %d grouped wo_a projections, "
        "%d attention-input stacks, %d query projections, %d with the "
        "indexer wq_b stacked in) "
        "into %d instance-pinned ANE programs (sequence_length=%d, "
        "down=%s/%.3f, wo_a=%s/%.3f, query_cpu_fraction=%.3f, "
        "cpu_threads=%d, shared_resource=%s, "
        "tail_padding_min_tokens=%d, dspark_native=true)",
        count,
        mlp_count,
        down_count,
        wo_a_count,
        attention_input_count,
        len(prepared) - mlp_count - down_count - attention_input_count,
        stacked_count,
        resident_programs + grouped_resident_programs,
        sequence_length,
        down_enabled,
        down_fraction,
        wo_a_enabled,
        wo_a_fraction,
        cpu_fraction,
        cpu_threads,
        cpu_shared_resource,
        tail_padding_min_tokens,
    )
    return count
