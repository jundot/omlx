"""Opt-in ANE/GPU hybrid prefill for dense Qwen3.5/3.6/3.8 MLPs.

The private ANE runtime only accepts fixed shapes, so this backend is attached
to a specific loaded model and sequence length. Unsupported layers, flattened
token counts, dtypes, and decode/verify calls fall through unchanged.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

logger = logging.getLogger(__name__)

_COMPILE_LOCK = threading.RLock()
_PATCHED_CLASSES: set[type] = set()
_VLM_HOOK_INSTALLED = False
_VLM_GDN_HOOK_INSTALLED = False
_GDN_MODULES: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()
# Legacy extensions compile one program per slice, and the private runtime on
# the reference M3 Ultra accepts 120 resident programs. Current extensions pack
# all slices into one multi-procedure program per ANE instance and bypass this
# fallback-only budget.
_ANE_RESIDENT_PROGRAM_LIMIT = 120
# First retry cap for split procedure banks after a monolithic bank fails to
# load. Program-create maps a bank's whole weight blob into the owning ANE's
# ~4 GiB device address window, so single-die chips reject two monolithic
# dual banks; 1 GiB spans keep every create well under the window.
_ANE_BANK_RETRY_MAX_BYTES = 1 << 30
@dataclass(frozen=True)
class _AnePrefillConfig:
    sequence_length: int
    fraction: float
    variant: int
    dual_ane: bool = False
    cpu_fraction: float = 0.0
    cpu_down_fraction: float = 0.0
    cpu_threads: int = 8
    cpu_shared_resource: bool = True


@dataclass(frozen=True)
class _AneGDNConfig:
    sequence_length: int
    fraction: float
    variant: int
    dual_ane: bool = False
    cpu_fraction: float = 0.0
    cpu_threads: int = 8
    cpu_shared_resource: bool = True


@dataclass(frozen=True)
class _CpuLinearState:
    weight: mx.array
    gpu_weight: mx.array
    gpu_scales: mx.array
    gpu_biases: mx.array
    bits: int
    group_size: int


@dataclass(frozen=True)
class _CombinedMLPState:
    model: Any
    weight: mx.array
    scales: mx.array
    biases: mx.array
    ane_outputs: int
    gpu_outputs: int
    model1: Any | None = None
    group_size: int = 128
    bits: int = 4
    cpu_weight: mx.array | None = None
    cpu_outputs: int = 0
    down_cpu: _CpuLinearState | None = None


@dataclass(frozen=True)
class _CombinedGDNState:
    model: Any
    weight: mx.array
    scales: mx.array
    biases: mx.array
    qkv_outputs: int
    z_outputs: int
    bits: int
    group_size: int
    model1: Any | None = None
    b_outputs: int = 0
    a_outputs: int = 0
    cpu_weight: mx.array | None = None
    cpu_outputs: int = 0


def _target_verify(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if bool(kwargs.get("target_verify", False)):
        return True
    return bool(args and isinstance(args[0], bool) and args[0])


def _eligible_affine_linear(
    linear: Any, dtype: mx.Dtype, *, bits: int, group_size: int
) -> bool:
    if not isinstance(linear, nn.QuantizedLinear):
        return False
    weight = getattr(linear, "weight", None)
    scales = getattr(linear, "scales", None)
    biases = getattr(linear, "biases", None)
    if weight is None or scales is None or biases is None or weight.ndim != 2:
        return False
    input_dim = int(weight.shape[1]) * 32 // bits
    return bool(
        dtype in (mx.float16, mx.bfloat16)
        and getattr(linear, "bits", None) == bits
        and getattr(linear, "group_size", None) == group_size
        and getattr(linear, "mode", None) == "affine"
        and "bias" not in linear
        and weight.dtype == mx.uint32
        and scales.dtype == dtype
        and biases.dtype == dtype
        and scales.shape == biases.shape
        and int(weight.shape[1]) * 32 == input_dim * bits
        and input_dim % group_size == 0
        and scales.shape == (weight.shape[0], input_dim // group_size)
    )


def _affine_spec(
    linear: Any,
    dtype: mx.Dtype,
    *,
    allowed_bits: tuple[int, ...] = (4, 5, 6, 8),
) -> tuple[int, int] | None:
    """Return a supported affine ``(bits, group_size)`` pair for ``linear``."""
    bits = getattr(linear, "bits", None)
    group_size = getattr(linear, "group_size", None)
    if bits not in allowed_bits or group_size not in (64, 128):
        return None
    if not _eligible_affine_linear(
        linear,
        dtype,
        bits=int(bits),
        group_size=int(group_size),
    ):
        return None
    return int(bits), int(group_size)


def _fused_swiglu_symbol(bits: int, *, dual: bool) -> str:
    if bits == 4:
        return "qwen35_ane_dual_q4_swiglu_t" if dual else "qwen35_ane_q4_swiglu_t"
    if bits in (5, 6, 8):
        return (
            "qwen35_ane_dual_affine_swiglu_t"
            if dual
            else "qwen35_ane_affine_swiglu_t"
        )
    raise ValueError(f"Unsupported ANE SwiGLU bit width: {bits}")


def _eligible_input(x: mx.array, config: _AnePrefillConfig) -> bool:
    if x.dtype not in (mx.float16, mx.bfloat16) or x.ndim < 3:
        return False
    input_dim = int(x.shape[-1])
    return int(x.size // input_dim) == config.sequence_length


def _tiled_input_plan(
    x: mx.array,
    sequence_length: int,
) -> tuple[int, int] | None:
    """Return ``(full_blocks, tail_rows)`` for a tileable wide prefill.

    Only a single prompt is tiled.  Flattening a real batch would lose the
    sequence boundaries required by the GDN recurrence.  Exact fixed shapes
    continue through the original fast path and do not use this planner.
    """
    if (
        x.dtype not in (mx.float16, mx.bfloat16)
        or x.ndim != 3
        or int(x.shape[0]) != 1
        or sequence_length <= 0
    ):
        return None
    rows = int(x.shape[-2])
    full_blocks, tail_rows = divmod(rows, sequence_length)
    if full_blocks < 1:
        return None
    return full_blocks, tail_rows


def _tail_qmm_or_linear(linear: Any, x: mx.array, variant: int) -> mx.array:
    # Wide-tile tails follow the same routing thresholds as the non-ANE
    # prefill fallback: the native qmm only pays off from the patch's
    # min-tokens boundary (2048 default, 16384 for q8); shorter tails use
    # stock MLX.
    from omlx.patches.qwen35_q4_mlp import (
        _Q8_MIN_TOKENS,
        _linear_qmm,
        _route_min_tokens_for_bits,
    )

    bits = getattr(linear, "bits", None)
    min_tokens = int(os.environ.get("OMLX_QWEN35_Q4_LINEAR_MIN_TOKENS", "2048"))
    q8_min_tokens = int(
        os.environ.get("OMLX_QWEN35_Q8_LINEAR_MIN_TOKENS", str(_Q8_MIN_TOKENS))
    )
    if x.shape[-2] < _route_min_tokens_for_bits(bits, min_tokens, q8_min_tokens):
        return linear(x)
    return _linear_qmm(linear, x, variant)


def configure_qwen35_ane_prefill_scheduler(
    scheduler: Any,
    sequence_length: int,
) -> bool:
    """Keep normal wide prompt chunks; projection backends tile internally."""
    if sequence_length < 1024 or sequence_length % 64:
        raise ValueError(
            "ANE prefill sequence_length must be a multiple of 64 >= 1024"
        )
    config = getattr(scheduler, "config", None)
    if config is None:
        return False
    step = int(getattr(config, "prefill_step_size", 0) or 0)
    floor = int(getattr(scheduler, "_qwen35_prefill_floor", 0) or 0)
    delivered_cap = max(step, floor)
    block_size = int(getattr(config, "paged_cache_block_size", 0) or 0)
    if getattr(scheduler, "block_aware_cache", None) is not None and block_size:
        # Boundary snapshots cut every prefill chunk at the next cache block
        # edge, so the block size caps the delivered width regardless of the
        # configured step or the qwen35 floor.
        delivered_cap = min(delivered_cap, block_size) if delivered_cap else block_size
    if delivered_cap and sequence_length > delivered_cap:
        logger.warning(
            "Qwen ANE prefill sequence_length=%d exceeds the delivered prefill "
            "chunk width (~%d tokens). Chunks narrower than the compiled shape "
            "cannot tile onto it, so the ANE will compile but never execute. "
            "Set sequence_length=%d or smaller.",
            sequence_length,
            delivered_cap,
            delivered_cap,
        )
    logger.info(
        "Qwen ANE prefill preserving scheduler chunks; projection tile=%d "
        "(step=%d, floor=%d)",
        sequence_length,
        step,
        floor,
    )
    return True


def _eligible_pair(mlp: Any) -> bool:
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    gate_dtype = getattr(getattr(gate, "scales", None), "dtype", None)
    gate_spec = _affine_spec(gate, gate_dtype, allowed_bits=(4, 5, 6, 8))
    up_spec = _affine_spec(up, gate_dtype, allowed_bits=(4, 5, 6, 8))
    down_spec = _affine_spec(
        down,
        getattr(getattr(down, "scales", None), "dtype", None),
        allowed_bits=(2, 4, 5, 6, 8),
    )
    return bool(
        gate_spec is not None
        and gate_spec == up_spec
        and down_spec is not None
        and gate.weight.shape == up.weight.shape
        and gate.scales.shape == up.scales.shape
        and int(down.weight.shape[1]) * 32
        == int(gate.weight.shape[0]) * int(getattr(down, "bits", 0))
        and int(down.weight.shape[0])
        == int(gate.weight.shape[1]) * 32 // int(getattr(gate, "bits", 0))
    )


def _cpu_gate_kernel_symbol(bits: int, *, dual: bool = True) -> str | None:
    if bits == 4:
        return (
            "qwen35_ane_dual_cpu_fp16_q4_swiglu_t"
            if dual
            else "qwen35_ane_cpu_fp16_q4_swiglu_t"
        )
    if bits in (5, 6, 8):
        return (
            "qwen35_ane_dual_cpu_fp16_swiglu_t"
            if dual
            else "qwen35_ane_cpu_fp16_swiglu_t"
        )
    return None


def _cpu_gdn_kernel_symbol(*, dual: bool) -> str:
    return (
        "qwen35_ane_dual_cpu_fp16_affine_qmm_t"
        if dual
        else "qwen35_ane_cpu_fp16_affine_qmm_t"
    )


def _prepare_cpu_linear(
    linear: Any, fraction: float
) -> _CpuLinearState | None:
    """Eagerly split one affine projection into FP16 CPU and quantized GPU rows."""
    if fraction <= 0 or getattr(linear, "scales", None) is None:
        return None
    spec = _affine_spec(linear, mx.float16)
    if spec is None:
        return None
    bits, group_size = spec
    output_dim = int(linear.weight.shape[0])
    cpu_outputs = (int(output_dim * fraction) // 64) * 64
    gpu_outputs = output_dim - cpu_outputs
    if cpu_outputs <= 0 or gpu_outputs <= 0 or gpu_outputs % 64:
        return None
    weight = mx.contiguous(
        mx.dequantize(
            linear.weight[:cpu_outputs],
            linear.scales[:cpu_outputs],
            linear.biases[:cpu_outputs],
            group_size=group_size,
            bits=bits,
        ).astype(mx.float16)
    )
    gpu_weight = mx.contiguous(linear.weight[cpu_outputs:])
    gpu_scales = mx.contiguous(linear.scales[cpu_outputs:])
    gpu_biases = mx.contiguous(linear.biases[cpu_outputs:])
    mx.eval(weight, gpu_weight, gpu_scales, gpu_biases)
    return _CpuLinearState(
        weight=weight,
        gpu_weight=gpu_weight,
        gpu_scales=gpu_scales,
        gpu_biases=gpu_biases,
        bits=bits,
        group_size=group_size,
    )


def _compile_pair(mlp: Any, config: _AnePrefillConfig) -> _CombinedMLPState | None:
    from omlx.custom_kernels.qwen35_prefill import fast

    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    if not _eligible_pair(mlp):
        return None

    cache = getattr(mlp, "_omlx_ane_prefill_cache", None)
    if cache is None:
        cache = {}
        mlp._omlx_ane_prefill_cache = cache

    output_dim = int(gate.weight.shape[0])
    bits = int(gate.bits)
    group_size = int(gate.group_size)
    dual_ane = bool(
        config.dual_ane
        and fast.has_symbol("qwen35_ane_dual_affine_qmm_t")
        and fast.has_symbol(_fused_swiglu_symbol(bits, dual=True))
    )
    if bits != 4 and not fast.has_symbol(_fused_swiglu_symbol(bits, dual=dual_ane)):
        return None
    alignment = 128 if dual_ane else 64
    ane_outputs = (int(output_dim * config.fraction) // alignment) * alignment
    cpu_enabled = bool(
        config.cpu_fraction > 0
        and gate.scales.dtype == mx.float16
        and up.scales.dtype == mx.float16
        and fast.has_symbol(_cpu_gate_kernel_symbol(bits, dual=dual_ane))
    )
    cpu_outputs = (
        (int(output_dim * config.cpu_fraction) // 64) * 64 if cpu_enabled else 0
    )
    gpu_start = ane_outputs + cpu_outputs
    gpu_outputs = output_dim - gpu_start
    if ane_outputs <= 0 or gpu_outputs <= 0 or gpu_outputs % 64:
        return None

    key = (
        config.sequence_length,
        ane_outputs,
        bits,
        cpu_outputs,
        config.cpu_down_fraction,
        group_size,
        "dual" if dual_ane else "linear",
    )
    if key in cache:
        return cache[key]

    with _COMPILE_LOCK:
        if key in cache:
            return cache[key]

        def dense_slice(start: int, end: int) -> mx.array:
            return mx.contiguous(
                mx.concatenate(
                    [
                        mx.dequantize(
                            linear.weight[start:end],
                            linear.scales[start:end],
                            linear.biases[start:end],
                            group_size=group_size,
                            bits=bits,
                        ).astype(mx.float32)
                        for linear in (gate, up)
                    ],
                    axis=0,
                )
            )

        if dual_ane:
            split = ane_outputs // 2
            dense0 = dense_slice(0, split)
            dense1 = dense_slice(split, ane_outputs)
        else:
            dense0 = dense_slice(0, ane_outputs)
            dense1 = None
        cpu_weight = None
        if cpu_outputs:
            cpu_weight = mx.contiguous(
                mx.concatenate(
                    [
                        mx.dequantize(
                            linear.weight[ane_outputs:gpu_start],
                            linear.scales[ane_outputs:gpu_start],
                            linear.biases[ane_outputs:gpu_start],
                            group_size=group_size,
                            bits=bits,
                        ).astype(mx.float16)
                        for linear in (gate, up)
                    ],
                    axis=0,
                )
            )
        weight = mx.contiguous(
            mx.concatenate((gate.weight[gpu_start:], up.weight[gpu_start:]), axis=0)
        )
        scales = mx.contiguous(
            mx.concatenate((gate.scales[gpu_start:], up.scales[gpu_start:]), axis=0)
        )
        biases = mx.contiguous(
            mx.concatenate((gate.biases[gpu_start:], up.biases[gpu_start:]), axis=0)
        )
        values = [dense0, weight, scales, biases]
        if cpu_weight is not None:
            values.append(cpu_weight)
        if dense1 is not None:
            values.append(dense1)
        mx.eval(*values)
        model = (
            fast.qwen35_ane_compile_linear(dense0, config.sequence_length, 1)
            if dual_ane
            else fast.qwen35_ane_compile_linear(dense0, config.sequence_length)
        )
        model1 = (
            fast.qwen35_ane_compile_linear(dense1, config.sequence_length, 2)
            if dense1 is not None
            else None
        )
        state = _CombinedMLPState(
            model=model,
            weight=weight,
            scales=scales,
            biases=biases,
            ane_outputs=ane_outputs,
            gpu_outputs=gpu_outputs,
            bits=bits,
            model1=model1,
            group_size=group_size,
            cpu_weight=cpu_weight,
            cpu_outputs=cpu_outputs,
            down_cpu=_prepare_cpu_linear(
                mlp.down_proj, config.cpu_down_fraction
            ),
        )
        cache[key] = state
        logger.debug(
            "Compiled combined private ANE Qwen gate+up slice %d->%d at "
            "sequence length %d",
            dense0.shape[1],
            dense0.shape[0] + (dense1.shape[0] if dense1 is not None else 0),
            config.sequence_length,
        )
        return state


def _prepare_pair_for_bank(
    mlp: Any, config: _AnePrefillConfig
) -> tuple[_CombinedMLPState, mx.array, mx.array | None] | None:
    from omlx.custom_kernels.qwen35_prefill import fast

    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    if not _eligible_pair(mlp):
        return None
    output_dim = int(gate.weight.shape[0])
    bits = int(gate.bits)
    group_size = int(gate.group_size)
    dual_ane = bool(config.dual_ane)
    if bits != 4 and not fast.has_symbol(
        _fused_swiglu_symbol(bits, dual=dual_ane)
    ):
        return None
    alignment = 128 if dual_ane else 64
    ane_outputs = (int(output_dim * config.fraction) // alignment) * alignment
    cpu_enabled = bool(
        config.cpu_fraction > 0
        and gate.scales.dtype == mx.float16
        and up.scales.dtype == mx.float16
        and fast.has_symbol(_cpu_gate_kernel_symbol(bits, dual=dual_ane))
    )
    cpu_outputs = (
        (int(output_dim * config.cpu_fraction) // 64) * 64 if cpu_enabled else 0
    )
    gpu_start = ane_outputs + cpu_outputs
    gpu_outputs = output_dim - gpu_start
    if ane_outputs <= 0 or gpu_outputs <= 0 or gpu_outputs % 64:
        return None

    def dense_slice(start: int, end: int) -> mx.array:
        return mx.contiguous(
            mx.concatenate(
                [
                    mx.dequantize(
                        linear.weight[start:end],
                        linear.scales[start:end],
                        linear.biases[start:end],
                        group_size=group_size,
                        bits=bits,
                    ).astype(mx.float32)
                    for linear in (gate, up)
                ],
                axis=0,
            )
        )

    if dual_ane:
        split = ane_outputs // 2
        dense0 = dense_slice(0, split)
        dense1 = dense_slice(split, ane_outputs)
    else:
        dense0 = dense_slice(0, ane_outputs)
        dense1 = None
    cpu_weight = None
    if cpu_outputs:
        cpu_weight = mx.contiguous(
            mx.concatenate(
                [
                    mx.dequantize(
                        linear.weight[ane_outputs:gpu_start],
                        linear.scales[ane_outputs:gpu_start],
                        linear.biases[ane_outputs:gpu_start],
                        group_size=group_size,
                        bits=bits,
                    ).astype(mx.float16)
                    for linear in (gate, up)
                ],
                axis=0,
            )
        )
    weight = mx.contiguous(
        mx.concatenate((gate.weight[gpu_start:], up.weight[gpu_start:]), axis=0)
    )
    scales = mx.contiguous(
        mx.concatenate((gate.scales[gpu_start:], up.scales[gpu_start:]), axis=0)
    )
    biases = mx.contiguous(
        mx.concatenate((gate.biases[gpu_start:], up.biases[gpu_start:]), axis=0)
    )
    values = [dense0, weight, scales, biases]
    if dense1 is not None:
        values.append(dense1)
    if cpu_weight is not None:
        values.append(cpu_weight)
    mx.eval(*values)
    return (
        _CombinedMLPState(
            model=None,
            weight=weight,
            scales=scales,
            biases=biases,
            ane_outputs=ane_outputs,
            gpu_outputs=gpu_outputs,
            bits=bits,
            model1=None,
            group_size=group_size,
            cpu_weight=cpu_weight,
            cpu_outputs=cpu_outputs,
            down_cpu=_prepare_cpu_linear(
                mlp.down_proj, config.cpu_down_fraction
            ),
        ),
        dense0,
        dense1,
    )


def _prepare_pair_runtime_state(
    mlp: Any,
    config: _AnePrefillConfig,
    model: Any,
    model1: Any,
) -> _CombinedMLPState | None:
    """Prepare only the mutable CPU/GPU slices for a compiled ANE width.

    Hardware tuning compiles one representative procedure for each ANE width
    into a small calibration bank.  CPU and GPU boundaries can then move
    without dequantizing or recompiling the ANE prefix again.  Keeping this
    helper beside the production preparation code also guarantees that the
    tuner exercises the same row alignment, q4 eligibility, and down-split
    implementation as normal inference.
    """
    from omlx.custom_kernels.qwen35_prefill import fast

    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    if not _eligible_pair(mlp):
        return None
    output_dim = int(gate.weight.shape[0])
    bits = int(gate.bits)
    group_size = int(gate.group_size)
    alignment = 128 if config.dual_ane else 64
    ane_outputs = (int(output_dim * config.fraction) // alignment) * alignment
    cpu_enabled = bool(
        config.cpu_fraction > 0
        and gate.scales.dtype == mx.float16
        and up.scales.dtype == mx.float16
        and fast.has_symbol(
            _cpu_gate_kernel_symbol(bits, dual=bool(config.dual_ane))
        )
    )
    cpu_outputs = (
        (int(output_dim * config.cpu_fraction) // 64) * 64 if cpu_enabled else 0
    )
    gpu_start = ane_outputs + cpu_outputs
    gpu_outputs = output_dim - gpu_start
    if ane_outputs <= 0 or gpu_outputs <= 0 or gpu_outputs % 64:
        return None

    cpu_weight = None
    if cpu_outputs:
        cpu_weight = mx.contiguous(
            mx.concatenate(
                [
                    mx.dequantize(
                        linear.weight[ane_outputs:gpu_start],
                        linear.scales[ane_outputs:gpu_start],
                        linear.biases[ane_outputs:gpu_start],
                        group_size=group_size,
                        bits=bits,
                    ).astype(mx.float16)
                    for linear in (gate, up)
                ],
                axis=0,
            )
        )
    weight = mx.contiguous(
        mx.concatenate((gate.weight[gpu_start:], up.weight[gpu_start:]), axis=0)
    )
    scales = mx.contiguous(
        mx.concatenate((gate.scales[gpu_start:], up.scales[gpu_start:]), axis=0)
    )
    biases = mx.contiguous(
        mx.concatenate((gate.biases[gpu_start:], up.biases[gpu_start:]), axis=0)
    )
    values = [weight, scales, biases]
    if cpu_weight is not None:
        values.append(cpu_weight)
    mx.eval(*values)
    return _CombinedMLPState(
        model=model,
        weight=weight,
        scales=scales,
        biases=biases,
        ane_outputs=ane_outputs,
        gpu_outputs=gpu_outputs,
        model1=model1,
        group_size=group_size,
        bits=bits,
        cpu_weight=cpu_weight,
        cpu_outputs=cpu_outputs,
        down_cpu=_prepare_cpu_linear(mlp.down_proj, config.cpu_down_fraction),
    )


def _gdn_linears(gdn: Any) -> tuple[Any, Any, Any, Any]:
    return (
        getattr(gdn, "in_proj_qkv", None),
        getattr(gdn, "in_proj_z", None),
        getattr(gdn, "in_proj_b", None),
        getattr(gdn, "in_proj_a", None),
    )


def _post_ane_linear(
    linear: Any,
    x: mx.array,
    variant: int,
    *,
    q8_threshold_env: str,
    cpu_state: _CpuLinearState | None = None,
    cpu_threads: int = 8,
    cpu_shared_resource: bool = True,
) -> mx.array:
    """Use the measured short-q8 winner for projections outside the split.

    The custom q8 tile only overtakes MLX's stock affine matmul at long token
    counts. ANE operates on a fixed 2K shape, so forcing that tile for the MLP
    down or the small GDN b/a projections would give back part of the offload
    gain. Other quantizations retain the existing exact native route.
    """
    if cpu_state is not None:
        from omlx.custom_kernels.qwen35_prefill import fast

        return fast.qwen35_cpu_fp16_affine_qmm_t(
            x,
            cpu_state.weight,
            cpu_state.gpu_weight,
            cpu_state.gpu_scales,
            cpu_state.gpu_biases,
            cpu_state.bits,
            variant,
            cpu_state.group_size,
            cpu_threads,
            cpu_shared_resource,
        )

    from omlx.patches.qwen35_q4_mlp import _linear_qmm

    if getattr(linear, "bits", None) == 8:
        q8_min_tokens = int(os.environ.get(q8_threshold_env, "16384"))
        if x.ndim >= 3 and int(x.shape[-2]) < q8_min_tokens:
            return linear(x)
    return _linear_qmm(linear, x, variant)


def _register_gdn_module(gdn: Any) -> None:
    qkv, _, _, _ = _gdn_linears(gdn)
    if qkv is not None:
        _GDN_MODULES[id(qkv)] = gdn


def _eligible_gdn(gdn: Any) -> bool:
    qkv, z, b, a = _gdn_linears(gdn)
    dtype = getattr(getattr(qkv, "scales", None), "dtype", None)
    specs = [_affine_spec(linear, dtype) for linear in (qkv, z, b, a)]
    return bool(
        all(spec is not None for spec in specs)
        and all(
            int(linear.weight.shape[1]) * 32 // int(linear.bits)
            == int(qkv.weight.shape[1]) * 32 // int(qkv.bits)
            for linear in (qkv, z, b, a)
        )
        and int(qkv.weight.shape[0]) % 64 == 0
        and int(z.weight.shape[0]) % 64 == 0
    )


def _pack_affine_gdn_suffix(
    qkv: Any,
    b: Any,
    a: Any,
    qkv_offset: int,
    qkv_spec: tuple[int, int],
) -> tuple[mx.array, mx.array, mx.array, int, int] | None:
    if qkv_spec[0] != 6:
        return None
    dtype = qkv.scales.dtype
    if _affine_spec(b, dtype) != qkv_spec or _affine_spec(a, dtype) != qkv_spec:
        return None

    b_outputs = int(b.weight.shape[0])
    a_outputs = int(a.weight.shape[0])
    suffix_outputs = int(qkv.weight.shape[0]) - qkv_offset + b_outputs + a_outputs
    padding = (-suffix_outputs) % 128
    weights = [qkv.weight[qkv_offset:], b.weight, a.weight]
    scales = [qkv.scales[qkv_offset:], b.scales, a.scales]
    biases = [qkv.biases[qkv_offset:], b.biases, a.biases]
    if padding:
        weights.append(mx.zeros((padding, qkv.weight.shape[1]), dtype=mx.uint32))
        scales.append(mx.zeros((padding, qkv.scales.shape[1]), dtype=dtype))
        biases.append(mx.zeros((padding, qkv.biases.shape[1]), dtype=dtype))
    return (
        mx.contiguous(mx.concatenate(weights, axis=0)),
        mx.contiguous(mx.concatenate(scales, axis=0)),
        mx.contiguous(mx.concatenate(biases, axis=0)),
        b_outputs,
        a_outputs,
    )


def _compile_gdn(gdn: Any, config: _AneGDNConfig) -> _CombinedGDNState | None:
    from omlx.custom_kernels.qwen35_prefill import fast

    if not _eligible_gdn(gdn) or not fast.has_symbol("qwen35_ane_affine_qmm_t"):
        return None
    qkv, z, b, a = _gdn_linears(gdn)
    cache = getattr(gdn, "_omlx_ane_gdn_cache", None)
    if cache is None:
        cache = {}
        gdn._omlx_ane_gdn_cache = cache

    # Put z first in the logical concatenation. At the 40% split this keeps
    # almost all recurrent q/k/v channels on the exact q5 GPU path while ANE
    # handles the output gate plus a small qkv prefix.
    logical = (z, qkv)
    z_outputs = int(z.weight.shape[0])
    qkv_outputs = int(qkv.weight.shape[0])
    total_outputs = z_outputs + qkv_outputs
    qkv_spec = _affine_spec(qkv, qkv.scales.dtype)
    z_spec = _affine_spec(z, qkv.scales.dtype)
    if qkv_spec is None or z_spec is None:
        return None
    qkv_bits, qkv_group_size = qkv_spec
    dual_ane = bool(config.dual_ane and fast.has_symbol("qwen35_ane_dual_affine_qmm_t"))
    alignment = 128 if dual_ane else 64
    ane_outputs = (int(total_outputs * config.fraction) // alignment) * alignment
    cpu_enabled = bool(
        config.cpu_fraction > 0
        and qkv.scales.dtype == mx.float16
        and fast.has_symbol(_cpu_gdn_kernel_symbol(dual=dual_ane))
    )
    cpu_outputs = (
        (int(total_outputs * config.cpu_fraction) // 64) * 64
        if cpu_enabled
        else 0
    )
    gpu_outputs = total_outputs - ane_outputs - cpu_outputs
    # The native GPU suffix accepts one quantization format. Put all of z on
    # ANE so an oQ4e-style q5-z/q4-qkv mix leaves a homogeneous qkv suffix.
    if ane_outputs < z_outputs:
        return None
    qkv_offset = ane_outputs - z_outputs
    packed_suffix = (
        None
        if cpu_outputs
        else _pack_affine_gdn_suffix(qkv, b, a, qkv_offset, qkv_spec)
    )
    b_outputs = a_outputs = 0
    if packed_suffix is None:
        if gpu_outputs <= 0 or gpu_outputs % 64:
            return None
        gpu_offset = qkv_offset + cpu_outputs
        weight = mx.contiguous(qkv.weight[gpu_offset:])
        scales = mx.contiguous(qkv.scales[gpu_offset:])
        biases = mx.contiguous(qkv.biases[gpu_offset:])
    else:
        weight, scales, biases, b_outputs, a_outputs = packed_suffix
    key = (
        config.sequence_length,
        ane_outputs,
        cpu_outputs,
        qkv_spec,
        z_spec,
        (
            "z_qkv_b_a_pad_dual_affine"
            if dual_ane
            else "z_qkv_b_a_pad_affine"
        )
        if packed_suffix is not None
        else ("z_qkv_dual_row_int8" if dual_ane else "z_qkv_row_int8"),
    )
    if key in cache:
        return cache[key]

    with _COMPILE_LOCK:
        if key in cache:
            return cache[key]

        def dense_logical_slice(start: int, end: int) -> mx.array:
            parts: list[mx.array] = []
            offset = 0
            for linear in logical:
                outputs = int(linear.weight.shape[0])
                lo = max(start - offset, 0)
                hi = min(end - offset, outputs)
                if lo < hi:
                    spec = _affine_spec(linear, qkv.scales.dtype)
                    if spec is None:
                        raise RuntimeError("Unsupported mixed GDN quantization")
                    bits, group_size = spec
                    parts.append(
                        mx.dequantize(
                            linear.weight[lo:hi],
                            linear.scales[lo:hi],
                            linear.biases[lo:hi],
                            group_size=group_size,
                            bits=bits,
                        ).astype(mx.float32)
                    )
                offset += outputs
            return mx.contiguous(mx.concatenate(parts, axis=0))

        if dual_ane:
            split = ane_outputs // 2
            dense0 = dense_logical_slice(0, split)
            dense1 = dense_logical_slice(split, ane_outputs)
        else:
            dense0 = dense_logical_slice(0, ane_outputs)
            dense1 = None
        qkv_offset = ane_outputs - z_outputs
        gpu_offset = qkv_offset + cpu_outputs
        cpu_weight = None
        if cpu_outputs:
            cpu_weight = mx.contiguous(
                mx.dequantize(
                    qkv.weight[qkv_offset:gpu_offset],
                    qkv.scales[qkv_offset:gpu_offset],
                    qkv.biases[qkv_offset:gpu_offset],
                    group_size=qkv_group_size,
                    bits=qkv_bits,
                ).astype(mx.float16)
            )
        values = [dense0, weight, scales, biases]
        if cpu_weight is not None:
            values.append(cpu_weight)
        if dense1 is not None:
            values.append(dense1)
        mx.eval(*values)
        model = (
            fast.qwen35_ane_compile_linear(dense0, config.sequence_length, 1)
            if dual_ane
            else fast.qwen35_ane_compile_linear(dense0, config.sequence_length)
        )
        model1 = (
            fast.qwen35_ane_compile_linear(dense1, config.sequence_length, 2)
            if dense1 is not None
            else None
        )
        state = _CombinedGDNState(
            model=model,
            weight=weight,
            scales=scales,
            biases=biases,
            qkv_outputs=qkv_outputs,
            z_outputs=z_outputs,
            bits=qkv_bits,
            group_size=qkv_group_size,
            model1=model1,
            b_outputs=b_outputs,
            a_outputs=a_outputs,
            cpu_weight=cpu_weight,
            cpu_outputs=cpu_outputs,
        )
        cache[key] = state
        return state


def _prepare_gdn_for_bank(
    gdn: Any, config: _AneGDNConfig
) -> tuple[_CombinedGDNState, mx.array, mx.array | None] | None:
    if not _eligible_gdn(gdn):
        return None
    qkv, z, b, a = _gdn_linears(gdn)
    logical = (z, qkv)
    z_outputs = int(z.weight.shape[0])
    qkv_outputs = int(qkv.weight.shape[0])
    total_outputs = z_outputs + qkv_outputs
    qkv_spec = _affine_spec(qkv, qkv.scales.dtype)
    z_spec = _affine_spec(z, qkv.scales.dtype)
    if qkv_spec is None or z_spec is None:
        return None
    qkv_bits, qkv_group_size = qkv_spec
    dual_ane = bool(config.dual_ane)
    alignment = 128 if dual_ane else 64
    ane_outputs = (int(total_outputs * config.fraction) // alignment) * alignment
    from omlx.custom_kernels.qwen35_prefill import fast

    cpu_enabled = bool(
        config.cpu_fraction > 0
        and qkv.scales.dtype == mx.float16
        and fast.has_symbol(_cpu_gdn_kernel_symbol(dual=dual_ane))
    )
    cpu_outputs = (
        (int(total_outputs * config.cpu_fraction) // 64) * 64
        if cpu_enabled
        else 0
    )
    gpu_outputs = total_outputs - ane_outputs - cpu_outputs
    if ane_outputs < z_outputs or gpu_outputs <= 0 or gpu_outputs % 64:
        return None
    qkv_offset = ane_outputs - z_outputs
    packed_suffix = (
        None
        if cpu_outputs
        else _pack_affine_gdn_suffix(qkv, b, a, qkv_offset, qkv_spec)
    )
    b_outputs = a_outputs = 0
    if packed_suffix is None:
        if gpu_outputs <= 0 or gpu_outputs % 64:
            return None
        gpu_offset = qkv_offset + cpu_outputs
        weight = mx.contiguous(qkv.weight[gpu_offset:])
        scales = mx.contiguous(qkv.scales[gpu_offset:])
        biases = mx.contiguous(qkv.biases[gpu_offset:])
    else:
        weight, scales, biases, b_outputs, a_outputs = packed_suffix

    def dense_logical_slice(start: int, end: int) -> mx.array:
        parts: list[mx.array] = []
        offset = 0
        for linear in logical:
            outputs = int(linear.weight.shape[0])
            lo = max(start - offset, 0)
            hi = min(end - offset, outputs)
            if lo < hi:
                spec = _affine_spec(linear, qkv.scales.dtype)
                if spec is None:
                    raise RuntimeError("Unsupported mixed GDN quantization")
                bits, group_size = spec
                parts.append(
                    mx.dequantize(
                        linear.weight[lo:hi],
                        linear.scales[lo:hi],
                        linear.biases[lo:hi],
                        group_size=group_size,
                        bits=bits,
                    ).astype(mx.float32)
                )
            offset += outputs
        return mx.contiguous(mx.concatenate(parts, axis=0))

    if dual_ane:
        split = ane_outputs // 2
        dense0 = dense_logical_slice(0, split)
        dense1 = dense_logical_slice(split, ane_outputs)
    else:
        dense0 = dense_logical_slice(0, ane_outputs)
        dense1 = None
    qkv_offset = ane_outputs - z_outputs
    gpu_offset = qkv_offset + cpu_outputs
    cpu_weight = None
    if cpu_outputs:
        cpu_weight = mx.contiguous(
            mx.dequantize(
                qkv.weight[qkv_offset:gpu_offset],
                qkv.scales[qkv_offset:gpu_offset],
                qkv.biases[qkv_offset:gpu_offset],
                group_size=qkv_group_size,
                bits=qkv_bits,
            ).astype(mx.float16)
        )
    weight = mx.contiguous(qkv.weight[gpu_offset:])
    scales = mx.contiguous(qkv.scales[gpu_offset:])
    biases = mx.contiguous(qkv.biases[gpu_offset:])
    values = [dense0, weight, scales, biases]
    if dense1 is not None:
        values.append(dense1)
    if cpu_weight is not None:
        values.append(cpu_weight)
    mx.eval(*values)
    return (
        _CombinedGDNState(
            model=None,
            weight=weight,
            scales=scales,
            biases=biases,
            qkv_outputs=qkv_outputs,
            z_outputs=z_outputs,
            bits=qkv_bits,
            group_size=qkv_group_size,
            model1=None,
            b_outputs=b_outputs,
            a_outputs=a_outputs,
            cpu_weight=cpu_weight,
            cpu_outputs=cpu_outputs,
        ),
        dense0,
        dense1,
    )


def _prepare_gdn_runtime_state(
    gdn: Any,
    config: _AneGDNConfig,
    model: Any,
    model1: Any,
) -> _CombinedGDNState | None:
    """Move the CPU/GPU QKV boundary without recompiling the ANE prefix."""
    from omlx.custom_kernels.qwen35_prefill import fast

    if not _eligible_gdn(gdn):
        return None
    qkv, z, _, _ = _gdn_linears(gdn)
    qkv_spec = _affine_spec(qkv, qkv.scales.dtype)
    if qkv_spec is None:
        return None
    bits, group_size = qkv_spec
    z_outputs = int(z.weight.shape[0])
    qkv_outputs = int(qkv.weight.shape[0])
    total_outputs = z_outputs + qkv_outputs
    alignment = 128 if config.dual_ane else 64
    ane_outputs = (int(total_outputs * config.fraction) // alignment) * alignment
    cpu_enabled = bool(
        config.cpu_fraction > 0
        and qkv.scales.dtype == mx.float16
        and fast.has_symbol(
            _cpu_gdn_kernel_symbol(dual=bool(config.dual_ane))
        )
    )
    cpu_outputs = (
        (int(total_outputs * config.cpu_fraction) // 64) * 64
        if cpu_enabled
        else 0
    )
    gpu_outputs = total_outputs - ane_outputs - cpu_outputs
    if ane_outputs < z_outputs or gpu_outputs <= 0 or gpu_outputs % 64:
        return None
    qkv_offset = ane_outputs - z_outputs
    gpu_offset = qkv_offset + cpu_outputs
    cpu_weight = None
    if cpu_outputs:
        cpu_weight = mx.contiguous(
            mx.dequantize(
                qkv.weight[qkv_offset:gpu_offset],
                qkv.scales[qkv_offset:gpu_offset],
                qkv.biases[qkv_offset:gpu_offset],
                group_size=group_size,
                bits=bits,
            ).astype(mx.float16)
        )
    weight = mx.contiguous(qkv.weight[gpu_offset:])
    scales = mx.contiguous(qkv.scales[gpu_offset:])
    biases = mx.contiguous(qkv.biases[gpu_offset:])
    values = [weight, scales, biases]
    if cpu_weight is not None:
        values.append(cpu_weight)
    mx.eval(*values)
    return _CombinedGDNState(
        model=model,
        weight=weight,
        scales=scales,
        biases=biases,
        qkv_outputs=qkv_outputs,
        z_outputs=z_outputs,
        bits=bits,
        group_size=group_size,
        model1=model1,
        cpu_weight=cpu_weight,
        cpu_outputs=cpu_outputs,
    )


def _gdn_backend_exact(
    gdn: Any, x: mx.array, target_verify: bool = False
) -> tuple[mx.array, mx.array, mx.array, mx.array] | None:
    config = getattr(gdn, "_omlx_ane_gdn_config", None)
    if config is None or target_verify or not _eligible_input(x, config):
        return None
    if getattr(gdn, "_omlx_ane_gdn_failed", False):
        return None
    state = getattr(gdn, "_omlx_ane_gdn_state", None)
    if state is None:
        try:
            state = _compile_gdn(gdn, config)
            gdn._omlx_ane_gdn_state = state
        except Exception:
            gdn._omlx_ane_gdn_failed = True
            logger.warning(
                "Disabling ANE GDN prefill after a runtime failure", exc_info=True
            )
            return None
    if state is None or state.scales.dtype != x.dtype:
        return None
    try:
        from omlx.custom_kernels.qwen35_prefill import fast
        from omlx.patches.qwen35_q4_mlp import _post_ane_qmm_or_linear

        if state.cpu_weight is not None:
            if state.model1 is not None:
                combined = fast.qwen35_ane_dual_cpu_fp16_affine_qmm_t(
                    x,
                    state.cpu_weight,
                    state.weight,
                    state.scales,
                    state.biases,
                    state.model,
                    state.model1,
                    state.bits,
                    config.variant,
                    state.group_size,
                    1,
                    config.cpu_threads,
                    config.cpu_shared_resource,
                )
            else:
                combined = fast.qwen35_ane_cpu_fp16_affine_qmm_t(
                    x,
                    state.cpu_weight,
                    state.weight,
                    state.scales,
                    state.biases,
                    state.model,
                    state.bits,
                    config.variant,
                    state.group_size,
                    1,
                    config.cpu_threads,
                    config.cpu_shared_resource,
                )
        elif state.model1 is not None:
            combined = fast.qwen35_ane_dual_affine_qmm_t(
                x,
                state.weight,
                state.scales,
                state.biases,
                state.model,
                state.model1,
                state.bits,
                config.variant,
                state.group_size,
            )
        else:
            combined = fast.qwen35_ane_affine_qmm_t(
                x,
                state.weight,
                state.scales,
                state.biases,
                state.model,
                state.bits,
                config.variant,
                state.group_size,
            )
        z = combined[..., : state.z_outputs]
        mixed_qkv = combined[..., state.z_outputs : state.z_outputs + state.qkv_outputs]
        if state.b_outputs:
            suffix_start = state.z_outputs + state.qkv_outputs
            b = combined[..., suffix_start : suffix_start + state.b_outputs]
            a_start = suffix_start + state.b_outputs
            a = combined[..., a_start : a_start + state.a_outputs]
        else:
            _, _, b_proj, a_proj = _gdn_linears(gdn)
            b = _post_ane_qmm_or_linear(b_proj, x, config.variant)
            a = _post_ane_qmm_or_linear(a_proj, x, config.variant)
        return mixed_qkv, z, b, a
    except Exception:
        gdn._omlx_ane_gdn_failed = True
        logger.warning(
            "Disabling ANE GDN prefill after a runtime failure", exc_info=True
        )
        return None


def _gdn_backend(
    gdn: Any, x: mx.array, target_verify: bool = False
) -> tuple[mx.array, mx.array, mx.array, mx.array] | None:
    """Route exact or internally tiled tokenwise GDN input projections.

    The recurrent GDN update remains outside this backend, so concatenating
    independently projected row blocks is algebraically identical to one wide
    projection. Inputs without a complete fixed-shape tile fall through to the
    original GPU operation.
    """
    config = getattr(gdn, "_omlx_ane_gdn_config", None)
    if config is None or target_verify:
        return None
    input_dim = int(x.shape[-1]) if x.ndim else 0
    rows = int(x.size // input_dim) if input_dim else 0
    if rows == config.sequence_length:
        return _gdn_backend_exact(gdn, x, target_verify)
    if rows < config.sequence_length:
        # Decode and short chunks exit before the tiling planner; this wrapper
        # runs on every GDN call of every layer of every decode step.
        return None

    plan = _tiled_input_plan(
        x,
        config.sequence_length,
    )
    if plan is None:
        return None
    full_blocks, tail_rows = plan
    projected: list[tuple[mx.array, mx.array, mx.array, mx.array]] = []
    for block in range(full_blocks):
        start = block * config.sequence_length
        stop = start + config.sequence_length
        block_x = mx.contiguous(x[:, start:stop, :])
        output = _gdn_backend_exact(gdn, block_x, target_verify)
        if output is None:
            return None
        projected.append(output)

    if tail_rows:
        tail_x = x[:, full_blocks * config.sequence_length :, :]
        projected.append(
            tuple(
                _tail_qmm_or_linear(linear, tail_x, config.variant)
                for linear in _gdn_linears(gdn)
            )
        )

    return tuple(
        mx.concatenate([part[index] for part in projected], axis=-2)
        for index in range(4)
    )


def _backend_exact(
    mlp: Any,
    x: mx.array,
    target_verify: bool = False,
) -> mx.array | None:
    config = getattr(mlp, "_omlx_ane_prefill_config", None)
    if config is None or target_verify or not _eligible_input(x, config):
        return None
    if getattr(mlp, "_omlx_ane_prefill_failed", False):
        return None

    state = getattr(mlp, "_omlx_ane_prefill_state", None)
    if state is None:
        try:
            state = _compile_pair(mlp, config)
            mlp._omlx_ane_prefill_state = state
        except Exception:
            mlp._omlx_ane_prefill_failed = True
            logger.warning(
                "Disabling ANE prefill for one Qwen MLP after a runtime failure",
                exc_info=True,
            )
            return None
    if state is None:
        return None
    if state.scales.dtype != x.dtype or int(state.weight.shape[1]) * 32 != int(
        x.shape[-1]
    ) * state.bits:
        return None

    try:
        from omlx.custom_kernels.qwen35_prefill import fast

        if state.cpu_weight is not None:
            if state.model1 is not None and state.bits == 4:
                activation = fast.qwen35_ane_dual_cpu_fp16_q4_swiglu_t(
                    x,
                    state.cpu_weight,
                    state.weight,
                    state.scales,
                    state.biases,
                    state.model,
                    state.model1,
                    config.variant,
                    state.group_size,
                    config.cpu_threads,
                    config.cpu_shared_resource,
                )
            elif state.model1 is not None:
                activation = fast.qwen35_ane_dual_cpu_fp16_swiglu_t(
                    x,
                    state.cpu_weight,
                    state.weight,
                    state.scales,
                    state.biases,
                    state.model,
                    state.model1,
                    state.bits,
                    config.variant,
                    state.group_size,
                    config.cpu_threads,
                    config.cpu_shared_resource,
                )
            elif state.bits == 4:
                activation = fast.qwen35_ane_cpu_fp16_q4_swiglu_t(
                    x,
                    state.cpu_weight,
                    state.weight,
                    state.scales,
                    state.biases,
                    state.model,
                    config.variant,
                    state.group_size,
                    config.cpu_threads,
                    config.cpu_shared_resource,
                )
            else:
                activation = fast.qwen35_ane_cpu_fp16_swiglu_t(
                    x,
                    state.cpu_weight,
                    state.weight,
                    state.scales,
                    state.biases,
                    state.model,
                    state.bits,
                    config.variant,
                    state.group_size,
                    config.cpu_threads,
                    config.cpu_shared_resource,
                )
            return _post_ane_linear(
                mlp.down_proj,
                activation,
                config.variant,
                q8_threshold_env="OMLX_QWEN35_Q8_MLP_MIN_TOKENS",
                cpu_state=state.down_cpu,
                cpu_threads=config.cpu_threads,
                cpu_shared_resource=config.cpu_shared_resource,
            )

        if state.model1 is not None:
            if state.bits == 4:
                activation = fast.qwen35_ane_dual_q4_swiglu_t(
                    x,
                    state.weight,
                    state.scales,
                    state.biases,
                    state.model,
                    state.model1,
                    config.variant,
                    state.group_size,
                )
            else:
                if not fast.has_symbol(_fused_swiglu_symbol(state.bits, dual=True)):
                    return None
                activation = fast.qwen35_ane_dual_affine_swiglu_t(
                    x,
                    state.weight,
                    state.scales,
                    state.biases,
                    state.model,
                    state.model1,
                    state.bits,
                    config.variant,
                    state.group_size,
                )
            return _post_ane_linear(
                mlp.down_proj,
                activation,
                config.variant,
                q8_threshold_env="OMLX_QWEN35_Q8_MLP_MIN_TOKENS",
                cpu_state=state.down_cpu,
                cpu_threads=config.cpu_threads,
                cpu_shared_resource=config.cpu_shared_resource,
            )

        if state.bits != 4:
            if not fast.has_symbol(_fused_swiglu_symbol(state.bits, dual=False)):
                return None
            activation = fast.qwen35_ane_affine_swiglu_t(
                x,
                state.weight,
                state.scales,
                state.biases,
                state.model,
                state.bits,
                config.variant,
                state.group_size,
            )
            return _post_ane_linear(
                mlp.down_proj,
                activation,
                config.variant,
                q8_threshold_env="OMLX_QWEN35_Q8_MLP_MIN_TOKENS",
                cpu_state=state.down_cpu,
                cpu_threads=config.cpu_threads,
                cpu_shared_resource=config.cpu_shared_resource,
            )

        if fast.has_symbol("qwen35_ane_q4_swiglu_t"):
            activation = fast.qwen35_ane_q4_swiglu_t(
                x,
                state.weight,
                state.scales,
                state.biases,
                state.model,
                config.variant,
                state.group_size,
            )
            return _post_ane_linear(
                mlp.down_proj,
                activation,
                config.variant,
                q8_threshold_env="OMLX_QWEN35_Q8_MLP_MIN_TOKENS",
                cpu_state=state.down_cpu,
                cpu_threads=config.cpu_threads,
                cpu_shared_resource=config.cpu_shared_resource,
            )

        combined = fast.qwen35_ane_q4_affine_qmm_t(
            x,
            state.weight,
            state.scales,
            state.biases,
            state.model,
            config.variant,
            state.group_size,
        )
        ane_end = 2 * state.ane_outputs
        gpu_gate_end = ane_end + state.gpu_outputs

        gate = mx.concatenate(
            (
                combined[..., : state.ane_outputs],
                combined[..., ane_end:gpu_gate_end],
            ),
            axis=-1,
        )
        up = mx.concatenate(
            (
                combined[..., state.ane_outputs : ane_end],
                combined[..., gpu_gate_end:],
            ),
            axis=-1,
        )

        return _post_ane_linear(
            mlp.down_proj,
            swiglu(gate, up),
            config.variant,
            q8_threshold_env="OMLX_QWEN35_Q8_MLP_MIN_TOKENS",
            cpu_state=state.down_cpu,
            cpu_threads=config.cpu_threads,
            cpu_shared_resource=config.cpu_shared_resource,
        )
    except Exception:
        mlp._omlx_ane_prefill_failed = True
        logger.warning(
            "Disabling ANE prefill for one Qwen MLP after a runtime failure",
            exc_info=True,
        )
        return None


def _backend(
    mlp: Any,
    x: mx.array,
    target_verify: bool = False,
) -> mx.array | None:
    """Route exact or internally tiled MLP rows without shrinking attention.

    Full fixed-shape blocks use the existing ANE/GPU/CPU implementation.  A
    residual tail stays on the ordinary quantized linears. Inputs without a
    complete fixed-shape tile fall through to the original wide GPU operation.
    """
    config = getattr(mlp, "_omlx_ane_prefill_config", None)
    if config is None or target_verify:
        return None
    input_dim = int(x.shape[-1]) if x.ndim else 0
    rows = int(x.size // input_dim) if input_dim else 0
    if rows == config.sequence_length:
        return _backend_exact(mlp, x, target_verify)
    if rows < config.sequence_length:
        # Decode and short chunks exit before the tiling planner; this wrapper
        # runs on every MLP call of every layer of every decode step.
        return None

    plan = _tiled_input_plan(
        x,
        config.sequence_length,
    )
    if plan is None:
        return None
    full_blocks, tail_rows = plan
    outputs: list[mx.array] = []
    for block in range(full_blocks):
        start = block * config.sequence_length
        stop = start + config.sequence_length
        block_x = mx.contiguous(x[:, start:stop, :])
        output = _backend_exact(mlp, block_x, target_verify)
        if output is None:
            return None
        outputs.append(output)

    if tail_rows:
        tail_x = x[:, full_blocks * config.sequence_length :, :]
        gate = _tail_qmm_or_linear(mlp.gate_proj, tail_x, config.variant)
        up = _tail_qmm_or_linear(mlp.up_proj, tail_x, config.variant)
        outputs.append(
            _tail_qmm_or_linear(mlp.down_proj, swiglu(gate, up), config.variant)
        )
    return mx.concatenate(outputs, axis=-2)


def _wrap_class(cls: type) -> None:
    if cls in _PATCHED_CLASSES:
        return
    original: Callable[..., mx.array] = cls.__call__

    def patched(self, x, *args, **kwargs):
        output = _backend(self, x, _target_verify(args, kwargs))
        if output is not None:
            return output
        return original(self, x, *args, **kwargs)

    cls.__call__ = patched
    cls._omlx_ane_prefill_original_call = original
    _PATCHED_CLASSES.add(cls)


def _install_dispatch() -> bool:
    global _VLM_GDN_HOOK_INSTALLED, _VLM_HOOK_INSTALLED
    installed = False
    try:
        vlm = importlib.import_module("mlx_vlm.models.qwen3_5.language")
        register = getattr(vlm, "register_qwen3_5_mlp_prefill_backend", None)
        register_gdn = getattr(vlm, "register_qwen3_5_gdn_prefill_backend", None)
        cls = getattr(vlm, "Qwen3_5MLP", None)
        if cls is not None and getattr(cls, "_omlx_q4_mlp_patched", False):
            # The exact q4 MLP patch replaces __call__ and therefore bypasses
            # mlx-vlm's inner registration hook. Wrap that dispatcher so ANE
            # gets first refusal and the q4 implementation remains fallback.
            _wrap_class(cls)
            installed = True
        elif callable(register):
            if not _VLM_HOOK_INSTALLED:
                register(_backend)
                _VLM_HOOK_INSTALLED = True
            installed = True
        else:
            if cls is not None:
                _wrap_class(cls)
                installed = True
        if callable(register_gdn) and not _VLM_GDN_HOOK_INSTALLED:
            register_gdn(_gdn_backend)
            _VLM_GDN_HOOK_INSTALLED = True
        elif not _VLM_GDN_HOOK_INSTALLED:
            target_linears = getattr(vlm, "_target_verify_linears", None)
            if callable(target_linears):

                def ane_target_linears(linears, x, target_verify=False):
                    gdn = _GDN_MODULES.get(id(linears[0])) if linears else None
                    if gdn is not None:
                        output = _gdn_backend(gdn, x, target_verify)
                        if output is not None:
                            return output
                    return target_linears(linears, x, target_verify)

                vlm._target_verify_linears = ane_target_linears
                _VLM_GDN_HOOK_INSTALLED = True
    except Exception:
        logger.debug("mlx-vlm Qwen ANE dispatch hook unavailable", exc_info=True)

    try:
        lm = importlib.import_module("mlx_lm.models.qwen3_5")
        cls = getattr(lm, "MLP", None)
        if cls is not None:
            _wrap_class(cls)
            installed = True
        from omlx.patches.qwen35_q4_mlp import (
            register_qwen35_lm_gdn_prefill_backend,
        )

        # The mlx-lm GDN implementation does not use mlx-vlm's
        # _target_verify_linears helper.  Its q4 compatibility wrapper owns
        # the projection call site, so register there on every install.  The
        # assignment is deliberately idempotent and avoids stale process-wide
        # hook state across VLM -> LLM fallback and model reloads.
        register_qwen35_lm_gdn_prefill_backend(_gdn_backend)
        installed = True
    except Exception:
        logger.debug("mlx-lm Qwen ANE dispatch hook unavailable", exc_info=True)
    return installed


def _bank_chunk_spans(
    weights: list[mx.array], max_bytes: int
) -> list[tuple[int, int]]:
    """Split procedure weights into contiguous spans of at most ``max_bytes``.

    A span always holds at least one procedure, so an oversized single
    procedure still gets its own bank.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    span_bytes = 0
    for index, weight in enumerate(weights):
        nbytes = weight.nbytes
        if index > start and span_bytes + nbytes > max_bytes:
            spans.append((start, index))
            start = index
            span_bytes = 0
        span_bytes += nbytes
    spans.append((start, len(weights)))
    return spans


def _compile_dual_banks(
    weights0: list[mx.array],
    weights1: list[mx.array],
    sequence_length: int,
) -> tuple[list[Any], list[Any], int] | None:
    """Compile the two instance-pinned procedure banks with a split ladder.

    A monolithic bank maps its entire weight blob into the owning ANE's
    ~4 GiB device address window at program-create, so a single-die chip that
    must host both dual banks rejects the layout that fits one bank per die
    on M3 Ultra (issue #2781, load failure 0x20004). Smaller banks keep every
    program-create under the window while per-eval mapping pages between
    them, matching the behaviour that lets the per-layer path work there.

    Returns ``(models0, models1, resident_program_count)``, or ``None`` when
    every attempt failed and the caller should use the per-layer fallback.
    ``OMLX_QWEN35_ANE_BANK_MAX_BYTES`` forces an initial per-bank byte cap.
    The cap counts the packed source weights handed to the bank compiler,
    which run about four times the compiled INT8 program size.
    """
    from omlx.custom_kernels.qwen35_prefill import fast

    cap = 0
    raw = os.environ.get("OMLX_QWEN35_ANE_BANK_MAX_BYTES", "").strip()
    if raw:
        try:
            cap = max(int(raw), 0)
        except ValueError:
            logger.warning(
                "Ignoring non-integer OMLX_QWEN35_ANE_BANK_MAX_BYTES=%r", raw
            )
    total_bytes = sum(weight.nbytes for weight in weights0)
    largest_bytes = max((weight.nbytes for weight in weights0), default=0)

    for _ in range(4):
        spans = (
            [(0, len(weights0))] if cap <= 0 else _bank_chunk_spans(weights0, cap)
        )
        try:
            models0: list[Any] = []
            models1: list[Any] = []
            for start, stop in spans:
                models0.extend(
                    fast.qwen35_ane_compile_linear_bank(
                        weights0[start:stop], sequence_length, 1
                    )
                )
                models1.extend(
                    fast.qwen35_ane_compile_linear_bank(
                        weights1[start:stop], sequence_length, 2
                    )
                )
        except Exception:
            # Drop banks that loaded before the failure so their device
            # mappings are released before the smaller retry.
            models0 = models1 = []
            logger.warning(
                "ANE procedure bank compilation failed (%d banks per "
                "instance, %d procedures, %.2f GiB per instance)",
                len(spans),
                len(weights0),
                total_bytes / (1 << 30),
                exc_info=True,
            )
            if all(stop - start == 1 for start, stop in spans):
                break
            if cap <= 0:
                # First retry aims at two near-halves per instance: the fewest
                # banks that can load where the monolithic bank cannot, and the
                # fewest resident programs. A measured M3 Ultra A/B stayed
                # bit-stable across greedy reruns at two banks while many
                # small banks occasionally diverged at a greedy tie.
                cap = max(total_bytes // 2 + largest_bytes, 1)
            else:
                cap = min(cap // 2, _ANE_BANK_RETRY_MAX_BYTES)
            if cap < 1:
                break
            logger.info(
                "Retrying ANE procedure banks split at %d MB per bank",
                cap // (1 << 20),
            )
            continue
        if len(spans) > 1:
            logger.info(
                "Compiled %d ANE procedures into %d split banks per instance",
                len(weights0),
                len(spans),
            )
        return models0, models1, 2 * len(spans)

    logger.warning(
        "Packed dual-ANE compilation failed; falling back to per-layer programs"
    )
    return None


def _compile_single_banks(
    weights: list[mx.array],
    sequence_length: int,
) -> tuple[list[Any], int] | None:
    """Compile instance-0 procedure banks with the same split retry ladder."""
    from omlx.custom_kernels.qwen35_prefill import fast

    cap = 0
    raw = os.environ.get("OMLX_QWEN35_ANE_BANK_MAX_BYTES", "").strip()
    if raw:
        try:
            cap = max(int(raw), 0)
        except ValueError:
            logger.warning(
                "Ignoring non-integer OMLX_QWEN35_ANE_BANK_MAX_BYTES=%r", raw
            )
    total_bytes = sum(weight.nbytes for weight in weights)
    largest_bytes = max((weight.nbytes for weight in weights), default=0)

    for _ in range(4):
        spans = (
            [(0, len(weights))] if cap <= 0 else _bank_chunk_spans(weights, cap)
        )
        try:
            models: list[Any] = []
            for start, stop in spans:
                models.extend(
                    fast.qwen35_ane_compile_linear_bank(
                        weights[start:stop], sequence_length, 0
                    )
                )
        except Exception:
            models = []
            logger.warning(
                "Single-ANE procedure bank compilation failed (%d banks, "
                "%d procedures, %.2f GiB)",
                len(spans),
                len(weights),
                total_bytes / (1 << 30),
                exc_info=True,
            )
            if all(stop - start == 1 for start, stop in spans):
                break
            if cap <= 0:
                cap = max(total_bytes // 2 + largest_bytes, 1)
            else:
                cap = min(cap // 2, _ANE_BANK_RETRY_MAX_BYTES)
            if cap < 1:
                break
            logger.info(
                "Retrying single-ANE procedure banks split at %d MB per bank",
                cap // (1 << 20),
            )
            continue
        if len(spans) > 1:
            logger.info(
                "Compiled %d single-ANE procedures into %d split banks",
                len(weights),
                len(spans),
            )
        return models, len(spans)

    logger.warning("Packed single-ANE calibration bank compilation failed")
    return None


def _enable_dual_procedure_banks(
    model: Any,
    mlp_candidates: list[Any],
    config: _AnePrefillConfig,
    *,
    gdn: bool,
    gdn_fraction: float,
    gdn_max_layers: int,
    cpu_gdn_fraction: float = 0.0,
) -> tuple[int, int, int, int] | None:
    from omlx.custom_kernels.qwen35_prefill import fast

    if not (
        config.dual_ane
        and fast.has_symbol("qwen35_ane_compile_linear_bank")
        and fast.has_symbol("qwen35_ane_dual_affine_qmm_t")
    ):
        return None
    candidate_bits = {
        int(getattr(getattr(module, "gate_proj", None), "bits", 0))
        for module in mlp_candidates
    }
    if 4 in candidate_bits and not fast.has_symbol("qwen35_ane_dual_q4_swiglu_t"):
        return None
    if any(
        bits != 4
        and not fast.has_symbol(_fused_swiglu_symbol(bits, dual=True))
        for bits in candidate_bits
    ):
        return None

    prepared_mlps: list[tuple[Any, _CombinedMLPState, mx.array, mx.array]] = []
    prepared_gdns: list[tuple[Any, _CombinedGDNState, mx.array, mx.array]] = []
    gdn_config = _AneGDNConfig(
        config.sequence_length,
        gdn_fraction,
        config.variant,
        True,
        cpu_fraction=cpu_gdn_fraction,
        cpu_threads=config.cpu_threads,
        cpu_shared_resource=config.cpu_shared_resource,
    )
    with _COMPILE_LOCK:
        for module in mlp_candidates:
            try:
                prepared = _prepare_pair_for_bank(module, config)
            except Exception:
                logger.warning(
                    "Skipping one Qwen MLP while preparing its ANE procedure",
                    exc_info=True,
                )
                continue
            if prepared is not None:
                state, dense0, dense1 = prepared
                prepared_mlps.append((module, state, dense0, dense1))

        if gdn and gdn_max_layers:
            for module in model.modules() if hasattr(model, "modules") else ():
                if len(prepared_gdns) >= gdn_max_layers:
                    break
                if not _eligible_gdn(module):
                    continue
                try:
                    prepared = _prepare_gdn_for_bank(module, gdn_config)
                except Exception:
                    logger.warning(
                        "Skipping one Qwen GDN while preparing its ANE procedure",
                        exc_info=True,
                    )
                    continue
                if prepared is not None:
                    state, dense0, dense1 = prepared
                    prepared_gdns.append((module, state, dense0, dense1))

        all_prepared = [*prepared_mlps, *prepared_gdns]
        if not all_prepared:
            return (0, 0, 0, 0)
        if len(all_prepared) > 256:
            logger.warning(
                "ANE procedure bank exceeds the private 256-procedure limit; "
                "falling back to per-layer programs"
            )
            return None
        weights0 = [entry[2] for entry in all_prepared]
        weights1 = [entry[3] for entry in all_prepared]
        mx.eval(*weights0, *weights1)
        banked_models = _compile_dual_banks(
            weights0, weights1, config.sequence_length
        )
        if banked_models is None:
            return None
        models0, models1, resident_program_count = banked_models

        if len(models0) != len(all_prepared) or len(models1) != len(all_prepared):
            raise RuntimeError("ANE procedure bank returned an incomplete model list")

        procedure = 0
        for module, state, _, _ in prepared_mlps:
            state = replace(state, model=models0[procedure], model1=models1[procedure])
            module._omlx_ane_prefill_config = config
            module._omlx_ane_prefill_state = state
            procedure += 1
        for module, state, _, _ in prepared_gdns:
            state = replace(state, model=models0[procedure], model1=models1[procedure])
            module._omlx_ane_gdn_config = gdn_config
            module._omlx_ane_gdn_state = state
            _register_gdn_module(module)
            procedure += 1

        # Pay every procedure's first-evaluation cost now, while the model is
        # still loading, so the first user request measures inference rather
        # than ANE warmup. Guarded per-model so an older compiled extension
        # without warmup() degrades to the previous behavior instead of
        # failing the load.
        warm_start = time.perf_counter()
        warmed = 0
        for warm_model in (*models0, *models1):
            warmup = getattr(warm_model, "warmup", None)
            if warmup is None:
                continue
            warmup()
            warmed += 1
        if warmed:
            logger.info(
                "Warmed %d ANE procedures in %.1fs at load",
                warmed,
                time.perf_counter() - warm_start,
            )

    return (
        len(prepared_mlps),
        len(prepared_mlps),
        len(prepared_gdns),
        resident_program_count,
    )


def enable_qwen35_ane_prefill(
    model: Any,
    *,
    sequence_length: int = 2048,
    fraction: float = 0.53,
    variant: int = 8,
    max_layers: int = 64,
    gdn: bool = False,
    gdn_fraction: float = 0.50,
    gdn_max_layers: int = 48,
    dual_ane: bool = True,
    cpu_fraction: float = 0.0,
    cpu_down_fraction: float = 0.0,
    cpu_gdn_fraction: float = 0.0,
    cpu_threads: int = 8,
    cpu_shared_resource: bool = True,
) -> int:
    """Enable the private ANE backend on eligible MLPs in ``model``.

    Returns the number of marked dense Qwen MLP modules. A return value of zero
    is a safe no-op for other model families and unsupported runtimes.
    """
    if sequence_length < 1024 or sequence_length % 64:
        raise ValueError("ANE prefill sequence_length must be a multiple of 64 >= 1024")
    if not 0.05 <= fraction <= 0.90:
        raise ValueError("ANE prefill fraction must be between 0.05 and 0.90")
    if max_layers < 1:
        raise ValueError("ANE prefill max_layers must be positive")
    if not 0.05 <= gdn_fraction <= 0.90:
        raise ValueError("ANE GDN prefill fraction must be between 0.05 and 0.90")
    if gdn_max_layers < 0:
        raise ValueError("ANE GDN prefill max_layers must be non-negative")
    if not 0.0 <= cpu_fraction <= 0.25:
        raise ValueError("ANE CPU fp16 fraction must be between 0.0 and 0.25")
    if not 0.0 <= cpu_down_fraction <= 0.50:
        raise ValueError(
            "ANE CPU down-projection fraction must be between 0.0 and 0.50"
        )
    if not 0.0 <= cpu_gdn_fraction <= 0.50:
        raise ValueError("ANE CPU GDN fraction must be between 0.0 and 0.50")
    if not 0 <= cpu_threads <= 64:
        raise ValueError("ANE CPU worker count must be between 0 and 64")

    env = os.environ.get("OMLX_QWEN35_ANE_PREFILL", "").strip().lower()
    if env in ("0", "false", "off"):
        logger.info("Qwen ANE prefill disabled by OMLX_QWEN35_ANE_PREFILL")
        return 0
    try:
        from omlx.custom_kernels.qwen35_prefill import fast

        if not fast.qwen35_ane_available():
            logger.warning("Private ANE runtime unavailable; Qwen ANE prefill skipped")
            return 0
    except Exception:
        logger.warning("ANE native extension unavailable; Qwen ANE prefill skipped")
        return 0
    if not _install_dispatch():
        logger.warning(
            "Qwen ANE prefill: dispatch hook could not be installed "
            "(mlx-vlm/mlx-lm Qwen backend not registered); ANE prefill inactive, "
            "running prefill on GPU"
        )
        return 0

    config = _AnePrefillConfig(
        sequence_length=sequence_length,
        fraction=fraction,
        variant=variant,
        dual_ane=dual_ane,
        cpu_fraction=cpu_fraction,
        cpu_down_fraction=cpu_down_fraction,
        cpu_threads=cpu_threads,
        cpu_shared_resource=cpu_shared_resource,
    )
    candidates = []
    scanned_mlp = 0
    for module in model.modules() if hasattr(model, "modules") else ():
        if not all(
            hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")
        ):
            continue
        scanned_mlp += 1
        if not _eligible_pair(module):
            continue
        candidates.append(module)
        if len(candidates) >= max_layers:
            break

    if not candidates:
        # The most common silent no-op: ANE was requested but every dense MLP
        # failed the eligibility gate (needs affine int4/5/6/8 quant with
        # group_size 64 or 128). GDN may still be eligible below; the final
        # summary warns if nothing at all is compiled.
        logger.warning(
            "Qwen ANE prefill requested but no eligible MLP layers found "
            "(%d dense MLP module(s) scanned; ANE requires affine int4/5/6/8 "
            "quantization with group_size 64 or 128)",
            scanned_mlp,
        )

    if (
        cpu_fraction > 0 or cpu_down_fraction > 0 or cpu_gdn_fraction > 0
    ) and candidates:
        gate = getattr(candidates[0], "gate_proj", None)
        if getattr(getattr(gate, "scales", None), "dtype", None) != mx.float16:
            logger.warning(
                "Qwen ANE CPU sharing requires an FP16 checkpoint clone; "
                "continuing with ANE/GPU only"
            )
            config = replace(
                config, cpu_fraction=0.0, cpu_down_fraction=0.0
            )
            cpu_gdn_fraction = 0.0
        elif cpu_shared_resource:
            if fast.qwen35_cpu_shared_resource_available():
                logger.info(
                    "Qwen ANE CPU sharing using performance-aware scheduling "
                    "with %d workers",
                    cpu_threads or 8,
                )
            else:
                logger.warning(
                    "Performance-aware CPU scheduling is unavailable; "
                    "falling back to ordinary Accelerate scheduling"
                )
                config = replace(config, cpu_shared_resource=False)

    banked = _enable_dual_procedure_banks(
        model,
        candidates,
        config,
        gdn=gdn,
        gdn_fraction=gdn_fraction,
        gdn_max_layers=gdn_max_layers,
        cpu_gdn_fraction=cpu_gdn_fraction,
    )
    if banked is not None:
        count, dual_count, gdn_count, resident_programs = banked
        model._omlx_ane_mlp_prefill_count = count
        model._omlx_ane_gdn_prefill_count = gdn_count
        model._omlx_ane_dual_prefill_count = dual_count
        model._omlx_ane_resident_program_count = resident_programs
        model._omlx_ane_procedure_count = count + gdn_count
        if count or gdn_count:
            logger.info(
                "Eagerly compiled %d MLP and %d GDN procedures into %d "
                "instance-pinned ANE programs (sequence_length=%d)",
                count,
                gdn_count,
                resident_programs,
                sequence_length,
            )
        else:
            logger.warning(
                "Qwen ANE prefill enabled but 0 procedures were compiled; "
                "the whole model runs prefill on GPU"
            )
        return count

    count = 0
    dual_count = 0
    resident_programs = 0
    mlp_budget_exhausted = False
    for module in candidates:
        requested_programs = 2 if dual_ane else 1
        if resident_programs + requested_programs > _ANE_RESIDENT_PROGRAM_LIMIT:
            mlp_budget_exhausted = True
            break
        try:
            state = _compile_pair(module, config)
        except Exception:
            module._omlx_ane_prefill_failed = True
            logger.warning(
                "Skipping one Qwen MLP after eager ANE compilation failed",
                exc_info=True,
            )
            continue
        if state is None:
            continue
        module._omlx_ane_prefill_config = config
        module._omlx_ane_prefill_state = state
        actual_programs = 2 if getattr(state, "model1", None) is not None else 1
        resident_programs += actual_programs
        dual_count += int(actual_programs == 2)
        count += 1

    gdn_count = 0
    gdn_budget_exhausted = False
    if gdn and gdn_max_layers:
        gdn_config = _AneGDNConfig(
            sequence_length,
            gdn_fraction,
            variant,
            dual_ane,
            cpu_fraction=cpu_gdn_fraction,
            cpu_threads=cpu_threads,
            cpu_shared_resource=config.cpu_shared_resource,
        )
        for module in model.modules() if hasattr(model, "modules") else ():
            if gdn_count >= gdn_max_layers:
                break
            requested_programs = 2 if dual_ane else 1
            if resident_programs + requested_programs > _ANE_RESIDENT_PROGRAM_LIMIT:
                gdn_budget_exhausted = True
                break
            if not _eligible_gdn(module):
                continue
            try:
                state = _compile_gdn(module, gdn_config)
            except Exception:
                module._omlx_ane_gdn_failed = True
                logger.warning(
                    "Skipping one Qwen GDN after eager ANE compilation failed",
                    exc_info=True,
                )
                continue
            if state is None:
                continue
            module._omlx_ane_gdn_config = gdn_config
            module._omlx_ane_gdn_state = state
            _register_gdn_module(module)
            resident_programs += 2 if getattr(state, "model1", None) is not None else 1
            gdn_count += 1
    model._omlx_ane_mlp_prefill_count = count
    model._omlx_ane_gdn_prefill_count = gdn_count
    model._omlx_ane_dual_prefill_count = dual_count
    model._omlx_ane_resident_program_count = resident_programs
    model._omlx_ane_procedure_count = count + gdn_count

    if count:
        logger.info(
            "Eagerly compiled and enabled ANE/GPU Qwen prefill on %d MLPs "
            "(sequence_length=%d, ANE fraction=%.3f, dual_ane=%s)",
            count,
            sequence_length,
            fraction,
            dual_ane,
        )
    if mlp_budget_exhausted or gdn_budget_exhausted:
        logger.info(
            "Stopped eager ANE preparation at the %d-program private-runtime "
            "budget (%d MLPs, %d dual)",
            _ANE_RESIDENT_PROGRAM_LIMIT,
            count,
            dual_count,
        )
    if gdn and gdn_max_layers and gdn_budget_exhausted:
        logger.warning(
            "ANE program budget exhausted before GDN completed; %d of %d "
            "requested GDN layers compiled and the rest stay on GPU",
            gdn_count,
            gdn_max_layers,
        )
    if gdn_count:
        logger.info(
            "Eagerly compiled and enabled ANE/GPU Qwen GDN input "
            "projections on %d layers (fraction=%.3f, dual_ane=%s)",
            gdn_count,
            gdn_fraction,
            dual_ane,
        )
    if not count and not gdn_count:
        logger.warning(
            "Qwen ANE prefill enabled but 0 procedures were compiled; "
            "the whole model runs prefill on GPU"
        )
    return count


def qwen35_ane_prefill_status(model: Any) -> dict:
    """Report the ANE prefill state that :func:`enable_qwen35_ane_prefill`
    attached to ``model``.

    Returns a JSON-serialisable dict, safe to call on any model (an untouched
    model reports ``attempted=False``). ``configured`` is True only when at
    least one MLP or GDN procedure was compiled onto the ANE, so callers can
    tell an inactive ANE apart from a silent no-op.
    """
    attempted = hasattr(model, "_omlx_ane_mlp_prefill_count")
    mlp = int(getattr(model, "_omlx_ane_mlp_prefill_count", 0) or 0)
    gdn = int(getattr(model, "_omlx_ane_gdn_prefill_count", 0) or 0)
    return {
        "attempted": attempted,
        "configured": bool(mlp or gdn),
        "mlp_layers": mlp,
        "gdn_layers": gdn,
        "dual_ane_layers": int(getattr(model, "_omlx_ane_dual_prefill_count", 0) or 0),
        "resident_programs": int(
            getattr(model, "_omlx_ane_resident_program_count", 0) or 0
        ),
    }
