"""Opt-in ANE/GPU hybrid prefill for dense Qwen3.5/3.6/3.8 MLPs.

The private ANE runtime only accepts fixed shapes, so this backend is attached
to a specific loaded model and sequence length. Unsupported layers, flattened
token counts, dtypes, and decode/verify calls fall through unchanged.
"""

from __future__ import annotations

import gc
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

from omlx.utils import hardware, proc_memory

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
# A failed compile attempt's already-loaded partial banks are dropped
# (models cleared to []), but the ANE driver's own device-mapping release
# is asynchronous and sometimes lags the retry (observed directly: kernel
# log lines "ReleaseProgramResource: WARN: waitForPendingUpdate failed").
# Retrying immediately races that cleanup, so a machine with tight headroom
# can climb across attempts even though each individual attempt drops its
# references correctly. Gate every attempt (including the first) on real
# phys_footprint headroom against total system memory -- the same ledger
# the kernel's jetsam killer uses -- and give the driver a moment to catch
# up between attempts, instead of shrinking-and-retrying blind. Confirmed
# necessary live: a 4-attempt retry ladder (each individually bounded)
# still grew process RSS to 48.3GB on a 48GB machine and was jetsam-killed
# mid-retry.
_ANE_BANK_RETRY_MAX_MEMORY_FRACTION = 0.70
_ANE_BANK_RETRY_SETTLE_SECONDS = 0.5


def _ane_bank_memory_footprint_snapshot() -> tuple[int, int]:
    """``(current phys_footprint, total system memory)`` in bytes for the
    headroom gate and its skip-warning log, so "why did ANE bank compile
    stop retrying on this box" is answerable from one log line instead of
    just the fixed 70% threshold. Best-effort: returns ``(0, 0)`` if either
    measurement is unavailable."""
    try:
        return proc_memory.get_phys_footprint(), hardware.get_total_memory_bytes()
    except Exception:
        return 0, 0


def _ane_bank_memory_headroom_ok() -> bool:
    """True if phys_footprint has enough room left for another ANE procedure
    bank compile attempt. Defaults to allowing the attempt (returns True) if
    the measurement is unavailable, so a query failure never blocks the fast
    path -- this is a circuit breaker for a known failure mode, not a new
    hard dependency."""
    current, total = _ane_bank_memory_footprint_snapshot()
    if total <= 0:
        return True
    return current < total * _ANE_BANK_RETRY_MAX_MEMORY_FRACTION


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
    ane_down_fraction: float = 0.0
    fused_down: bool = False
    tail_padding_min_tokens: int = 0


@dataclass(frozen=True)
class _AneGDNConfig:
    sequence_length: int
    fraction: float
    variant: int
    dual_ane: bool = False
    cpu_fraction: float = 0.0
    cpu_threads: int = 8
    cpu_shared_resource: bool = True
    tail_padding_min_tokens: int = 0


@dataclass(frozen=True)
class _CpuLinearState:
    weight: mx.array
    gpu_weight: mx.array
    gpu_scales: mx.array
    gpu_biases: mx.array
    bits: int
    group_size: int


@dataclass(frozen=True)
class _AneDownState:
    model: Any
    model1: Any
    weight: mx.array
    scales: mx.array
    biases: mx.array
    ane_outputs: int
    gpu_outputs: int
    bits: int
    group_size: int
    cpu_weight: mx.array | None = None
    cpu_outputs: int = 0
    compile_weight0: mx.array | None = None
    compile_weight1: mx.array | None = None


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
    down_ane: _AneDownState | None = None


@dataclass(frozen=True)
class _FusedDownMLPState:
    model: Any
    model1: Any
    gate_up_weight: mx.array
    gate_up_scales: mx.array
    gate_up_biases: mx.array
    down_weight: mx.array
    down_scales: mx.array
    down_biases: mx.array
    cpu_gate_up_weight: mx.array | None = None
    cpu_down_weight: mx.array | None = None


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


def _pad_fixed_shape_tail(x: mx.array, sequence_length: int) -> mx.array:
    """Zero-pad one tokenwise projection input to an ANE program's shape."""
    rows = int(x.shape[-2])
    if rows <= 0 or rows >= sequence_length:
        raise ValueError("ANE tail padding requires 1..sequence_length-1 rows")
    return mx.pad(x, [(0, 0), (0, sequence_length - rows), (0, 0)])


def _tail_padding_profitable(rows: int, config: Any) -> bool:
    threshold = int(getattr(config, "tail_padding_min_tokens", 0) or 0)
    return 0 < threshold <= rows < int(config.sequence_length)


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
    from omlx.custom_kernels.qwen35_prefill import fast

    if fraction <= 0 or getattr(linear, "scales", None) is None:
        return None
    # Without the native symbol the dispatch wrapper would raise at first use
    # and latch the whole layer off; stay a clean no-op like the other sites.
    if not fast.has_symbol("qwen35_cpu_fp16_affine_qmm_t"):
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


def _prepare_down_for_bank(
    linear: Any,
    config: _AnePrefillConfig,
) -> _AneDownState | None:
    """Split down-projection output rows across dual ANE, CPU, and GPU.

    Unlike the abandoned hidden-channel split, every backend owns complete
    output rows. The results can therefore be concatenated without summing
    independently approximated partial dot products.
    """
    if config.ane_down_fraction <= 0 or not config.dual_ane:
        return None
    dtype = getattr(getattr(linear, "scales", None), "dtype", None)
    spec = _affine_spec(linear, dtype, allowed_bits=(4, 5, 6, 8))
    if spec is None:
        return None
    bits, group_size = spec
    output_dim = int(linear.weight.shape[0])
    ane_outputs = (
        int(output_dim * config.ane_down_fraction) // 128
    ) * 128
    split = ane_outputs // 2
    cpu_enabled = bool(
        config.cpu_down_fraction > 0
        and dtype == mx.float16
    )
    cpu_outputs = (
        (int(output_dim * config.cpu_down_fraction) // 64) * 64
        if cpu_enabled
        else 0
    )
    gpu_start = ane_outputs + cpu_outputs
    gpu_outputs = output_dim - gpu_start
    if (
        ane_outputs <= 0
        or split % 64
        or gpu_outputs <= 0
        or gpu_outputs % 64
    ):
        return None

    def dense_rows(start: int, stop: int) -> mx.array:
        return mx.contiguous(
            mx.dequantize(
                linear.weight[start:stop],
                linear.scales[start:stop],
                linear.biases[start:stop],
                group_size=group_size,
                bits=bits,
            ).astype(mx.float32)
        )

    dense0 = dense_rows(0, split)
    dense1 = dense_rows(split, ane_outputs)
    cpu_weight = None
    if cpu_outputs:
        cpu_weight = mx.contiguous(
            mx.dequantize(
                linear.weight[ane_outputs:gpu_start],
                linear.scales[ane_outputs:gpu_start],
                linear.biases[ane_outputs:gpu_start],
                group_size=group_size,
                bits=bits,
            ).astype(mx.float16)
        )
    weight = mx.contiguous(linear.weight[gpu_start:])
    scales = mx.contiguous(linear.scales[gpu_start:])
    biases = mx.contiguous(linear.biases[gpu_start:])
    values = [dense0, dense1, weight, scales, biases]
    if cpu_weight is not None:
        values.append(cpu_weight)
    mx.eval(*values)
    return _AneDownState(
        model=None,
        model1=None,
        weight=weight,
        scales=scales,
        biases=biases,
        ane_outputs=ane_outputs,
        gpu_outputs=gpu_outputs,
        bits=bits,
        group_size=group_size,
        cpu_weight=cpu_weight,
        cpu_outputs=cpu_outputs,
        compile_weight0=dense0,
        compile_weight1=dense1,
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
    down_ane = _prepare_down_for_bank(mlp.down_proj, config)
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
            down_cpu=(
                None
                if down_ane is not None
                else _prepare_cpu_linear(
                    mlp.down_proj, config.cpu_down_fraction
                )
            ),
            down_ane=down_ane,
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


def _post_ane_down(
    linear: Any,
    x: mx.array,
    state: _AneDownState | None,
    config: _AnePrefillConfig,
    fallback: _CpuLinearState | None,
) -> mx.array:
    if state is None:
        return _post_ane_linear(
            linear,
            x,
            config.variant,
            q8_threshold_env="OMLX_QWEN35_Q8_MLP_MIN_TOKENS",
            cpu_state=fallback,
            cpu_threads=config.cpu_threads,
            cpu_shared_resource=config.cpu_shared_resource,
        )
    if (
        state.model is None
        or state.model1 is None
        or state.scales.dtype != x.dtype
    ):
        raise RuntimeError("ANE down-projection state is incomplete")

    from omlx.custom_kernels.qwen35_prefill import fast

    if state.cpu_weight is not None:
        return fast.qwen35_ane_dual_cpu_fp16_affine_qmm_t(
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
            0,
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
        state.bits,
        config.variant,
        state.group_size,
        0,
    )


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


def _recurrent_safe_gdn_ane_outputs(
    z_outputs: int,
    qkv_outputs: int,
    fraction: float,
    alignment: int,
) -> int:
    """Return an aligned ANE slice that never enters recurrent QKV rows.

    The ANE compiler requantizes its source slice to per-output-channel INT8.
    Applying that approximation to QKV changes the recurrent state at every
    token, so the error can accumulate over a long prompt. Z is a token-local
    output gate: offloading all of it preserves useful ANE/GPU overlap without
    feeding approximate values back into the next recurrent step.
    """
    if z_outputs <= 0 or qkv_outputs <= 0 or z_outputs % alignment:
        return 0
    total_outputs = z_outputs + qkv_outputs
    requested = (int(total_outputs * fraction) // alignment) * alignment
    return z_outputs if requested >= z_outputs else 0


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

    # Put z first so the approximate ANE slice can stop at the token-local
    # gate boundary. Recurrent q/k/v channels remain on the source-precision
    # GPU path (or the explicitly configured FP16 CPU path).
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
    ane_outputs = _recurrent_safe_gdn_ane_outputs(
        z_outputs,
        qkv_outputs,
        config.fraction,
        alignment,
    )
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
    # The native GPU suffix accepts one quantization format. The quality-safe
    # split puts exactly all of z on ANE, leaving homogeneous qkv on the GPU.
    if ane_outputs != z_outputs:
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


def _min_viable_gdn_fraction(gdn: Any, alignment: int) -> float | None:
    """Smallest fraction whose aligned slice covers exactly z on ``gdn``.

    Wider requests are capped at z so approximate ANE rows never enter the
    recurrent qkv projection. ``None`` means z cannot be represented exactly
    with this ANE alignment.
    """
    qkv, z, _, _ = _gdn_linears(gdn)
    if qkv is None or z is None:
        return None
    z_outputs = int(z.weight.shape[0])
    total_outputs = z_outputs + int(qkv.weight.shape[0])
    if total_outputs <= 0 or z_outputs <= 0 or z_outputs % alignment:
        return None
    ane_min = z_outputs
    fraction = ane_min / total_outputs
    if (int(total_outputs * fraction) // alignment) * alignment < ane_min:
        fraction = (ane_min + 1) / total_outputs
    return fraction


def _log_gdn_recurrent_safe_cap(
    model: Any,
    requested_fraction: float,
    gdn_count: int,
    dual_ane: bool,
) -> None:
    """Report when a wider requested GDN slice is precision-capped at z."""
    if not gdn_count:
        return
    gdn = next(
        (
            module
            for module in (model.modules() if hasattr(model, "modules") else ())
            if getattr(module, "_omlx_ane_gdn_state", None) is not None
        ),
        None,
    )
    if gdn is None:
        return
    qkv, z, _, _ = _gdn_linears(gdn)
    z_outputs = int(z.weight.shape[0])
    qkv_outputs = int(qkv.weight.shape[0])
    total_outputs = z_outputs + qkv_outputs
    alignment = 128 if dual_ane else 64
    requested_outputs = (
        int(total_outputs * requested_fraction) // alignment
    ) * alignment
    if requested_outputs <= z_outputs:
        return
    logger.info(
        "Capped ANE GDN output rows from requested %.3f to %.3f: ANE handles "
        "only token-local z while recurrent qkv stays off the approximate "
        "ANE INT8 path",
        requested_fraction,
        z_outputs / total_outputs,
    )


def _warn_gdn_below_floor(
    model: Any, requested: bool, gdn_count: int, gdn_fraction: float, dual_ane: bool
) -> None:
    """Name the floor when a too-small fraction compiled no GDN procedure.

    Without this the only symptom of a below-floor fraction is gdn_layers=0
    in the status payload, which reads the same as a compile failure.
    """
    if not requested or gdn_count:
        return
    gdn = next(
        (
            module
            for module in (model.modules() if hasattr(model, "modules") else ())
            if _eligible_gdn(module)
        ),
        None,
    )
    if gdn is None:
        return
    floor = _min_viable_gdn_fraction(gdn, 128 if dual_ane else 64)
    if floor is None or gdn_fraction >= floor:
        return
    logger.warning(
        "ANE GDN prefill fraction %.3f is below this model's %.3f floor, so "
        "no GDN procedure could be compiled and GDN prefill stays on GPU",
        gdn_fraction,
        floor,
    )


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
    ane_outputs = _recurrent_safe_gdn_ane_outputs(
        z_outputs,
        qkv_outputs,
        config.fraction,
        alignment,
    )
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
    if ane_outputs != z_outputs or gpu_outputs <= 0 or gpu_outputs % 64:
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
    cpu_weight = None
    if cpu_outputs:
        cpu_weight = mx.contiguous(
            mx.dequantize(
                qkv.weight[qkv_offset : qkv_offset + cpu_outputs],
                qkv.scales[qkv_offset : qkv_offset + cpu_outputs],
                qkv.biases[qkv_offset : qkv_offset + cpu_outputs],
                group_size=qkv_group_size,
                bits=qkv_bits,
            ).astype(mx.float16)
        )
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
    ane_outputs = _recurrent_safe_gdn_ane_outputs(
        z_outputs,
        qkv_outputs,
        config.fraction,
        alignment,
    )
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
    if ane_outputs != z_outputs or gpu_outputs <= 0 or gpu_outputs % 64:
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


def _raise_if_latched(*models: Any) -> None:
    """Detect a wedged ANE program at graph-construction time.

    A failed or timed-out evaluation latches the program in native code
    (has_error); every later begin() would throw at evaluation time, where
    the per-module fallback cannot catch it. Raising here instead keeps the
    failure inside the module's existing try/except latch.
    """
    for model in models:
        has_error = getattr(model, "has_error", None)
        if has_error is not None and has_error():
            raise RuntimeError(
                "ANE program latched after a failed or timed-out evaluation"
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

        _raise_if_latched(state.model, state.model1)
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
        if _tail_padding_profitable(rows, config):
            padded = _gdn_backend_exact(
                gdn,
                _pad_fixed_shape_tail(x, config.sequence_length),
                target_verify,
            )
            if padded is not None:
                return tuple(value[..., :rows, :] for value in padded)
        # Decode and unprofitable short chunks exit before the tiling planner;
        # this wrapper runs on every GDN call of every layer of every decode.
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
        padded = None
        if _tail_padding_profitable(tail_rows, config):
            padded = _gdn_backend_exact(
                gdn,
                _pad_fixed_shape_tail(tail_x, config.sequence_length),
                target_verify,
            )
        if padded is not None:
            projected.append(
                tuple(value[..., :tail_rows, :] for value in padded)
            )
        else:
            projected.append(
                tuple(
                    _tail_qmm_or_linear(linear, tail_x, config.variant)
                    for linear in _gdn_linears(gdn)
                )
            )

    result = tuple(
        mx.concatenate([part[index] for part in projected], axis=-2)
        for index in range(4)
    )
    # The four tiled projections are sibling consumers of the same native
    # hybrid outputs. Schedule them together before the recurrent GDN graph
    # consumes individual branches; otherwise MLX can interleave their Metal
    # merge/lifetime boundaries after a prefix-cache restore (#3117). This is
    # asynchronous, so ANE/GPU overlap is preserved without a host fence.
    mx.async_eval(*result)
    return result


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

    fused_down_state = getattr(mlp, "_omlx_ane_fused_down_state", None)
    if fused_down_state is not None:
        try:
            from omlx.custom_kernels.qwen35_prefill import fast

            _raise_if_latched(fused_down_state.model, fused_down_state.model1)
            if fused_down_state.cpu_gate_up_weight is not None:
                if fused_down_state.cpu_down_weight is None:
                    raise RuntimeError("Incomplete fused CPU MLP state")
                return fast.qwen35_ane_dual_cpu_fp16_q4_swiglu_down_t(
                    x,
                    fused_down_state.cpu_gate_up_weight,
                    fused_down_state.cpu_down_weight,
                    fused_down_state.gate_up_weight,
                    fused_down_state.gate_up_scales,
                    fused_down_state.gate_up_biases,
                    fused_down_state.down_weight,
                    fused_down_state.down_scales,
                    fused_down_state.down_biases,
                    fused_down_state.model,
                    fused_down_state.model1,
                    config.variant,
                    128,
                    config.cpu_threads,
                    config.cpu_shared_resource,
                )
            return fast.qwen35_ane_dual_q4_swiglu_down_t(
                x,
                fused_down_state.gate_up_weight,
                fused_down_state.gate_up_scales,
                fused_down_state.gate_up_biases,
                fused_down_state.down_weight,
                fused_down_state.down_scales,
                fused_down_state.down_biases,
                fused_down_state.model,
                fused_down_state.model1,
                config.variant,
                128,
            )
        except Exception:
            mlp._omlx_ane_prefill_failed = True
            logger.warning(
                "Disabling fused ANE down prefill for one Qwen MLP after a "
                "runtime failure",
                exc_info=True,
            )
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

        _raise_if_latched(state.model, state.model1, state.down_ane)
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
            return _post_ane_down(
                mlp.down_proj,
                activation,
                state.down_ane,
                config,
                state.down_cpu,
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
            return _post_ane_down(
                mlp.down_proj,
                activation,
                state.down_ane,
                config,
                state.down_cpu,
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
            return _post_ane_down(
                mlp.down_proj,
                activation,
                state.down_ane,
                config,
                state.down_cpu,
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
            return _post_ane_down(
                mlp.down_proj,
                activation,
                state.down_ane,
                config,
                state.down_cpu,
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

        return _post_ane_down(
            mlp.down_proj,
            swiglu(gate, up),
            state.down_ane,
            config,
            state.down_cpu,
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
        if _tail_padding_profitable(rows, config):
            padded = _backend_exact(
                mlp,
                _pad_fixed_shape_tail(x, config.sequence_length),
                target_verify,
            )
            if padded is not None:
                return padded[..., :rows, :]
        # Decode and unprofitable short chunks exit before the tiling planner;
        # this wrapper runs on every MLP call of every layer of every decode.
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
        padded = None
        if _tail_padding_profitable(tail_rows, config):
            padded = _backend_exact(
                mlp,
                _pad_fixed_shape_tail(tail_x, config.sequence_length),
                target_verify,
            )
        if padded is not None:
            outputs.append(padded[..., :tail_rows, :])
        else:
            gate = _tail_qmm_or_linear(mlp.gate_proj, tail_x, config.variant)
            up = _tail_qmm_or_linear(mlp.up_proj, tail_x, config.variant)
            outputs.append(
                _tail_qmm_or_linear(
                    mlp.down_proj, swiglu(gate, up), config.variant
                )
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
        nbytes = getattr(weight, "nbytes", weight)
        if index > start and span_bytes + nbytes > max_bytes:
            spans.append((start, index))
            start = index
            span_bytes = 0
        span_bytes += nbytes
    spans.append((start, len(weights)))
    return spans


def _bank_split_ladder(
    source_bytes: list[int],
    compile_span: Any,
) -> tuple[list[Any], list[Any], int] | None:
    """Compile the two instance-pinned procedure banks with a split ladder.

    A monolithic bank maps its entire weight blob into the owning ANE's
    ~4 GiB device address window at program-create, so a single-die chip that
    must host both dual banks rejects the layout that fits one bank per die
    on M3 Ultra (issue #2781, load failure 0x20004). Smaller banks keep every
    program-create under the window while per-eval mapping pages between
    them, matching the behaviour that lets the per-layer path work there.

    ``compile_span(start, stop)`` compiles that span for both instances and
    returns ``(models0, models1)``. Returns ``(models0, models1,
    resident_program_count)``, or ``None`` when every attempt failed and the
    caller should use the per-layer fallback.
    ``OMLX_QWEN35_ANE_BANK_MAX_BYTES`` forces an initial per-bank byte cap.
    The cap counts the packed source weights handed to the bank compiler,
    which run about four times the compiled INT8 program size.
    """
    cap = 0
    raw = os.environ.get("OMLX_QWEN35_ANE_BANK_MAX_BYTES", "").strip()
    if raw:
        try:
            cap = max(int(raw), 0)
        except ValueError:
            logger.warning(
                "Ignoring non-integer OMLX_QWEN35_ANE_BANK_MAX_BYTES=%r", raw
            )
    total_bytes = sum(source_bytes)
    largest_bytes = max(source_bytes, default=0)

    for attempt in range(4):
        if not _ane_bank_memory_headroom_ok():
            current, total = _ane_bank_memory_footprint_snapshot()
            logger.warning(
                "Skipping ANE procedure bank compile attempt %d/4: phys "
                "footprint %.2f GiB / %.2f GiB total already past the "
                "%.0f%% safety fraction of system memory. Falling back to "
                "per-layer programs instead of risking a jetsam kill.",
                attempt + 1,
                current / (1 << 30),
                total / (1 << 30),
                _ANE_BANK_RETRY_MAX_MEMORY_FRACTION * 100,
            )
            return None
        spans = (
            [(0, len(source_bytes))]
            if cap <= 0
            else _bank_chunk_spans(source_bytes, cap)
        )
        try:
            models0: list[Any] = []
            models1: list[Any] = []
            for start, stop in spans:
                span0, span1 = compile_span(start, stop)
                models0.extend(span0)
                models1.extend(span1)
        except Exception:
            # Drop banks that loaded before the failure so their device
            # mappings are released before the smaller retry. The ANE
            # driver's own release is asynchronous, so force a GC pass and
            # give it a moment to actually land before the next attempt's
            # headroom check measures phys_footprint -- otherwise the check
            # can see this attempt's not-yet-reclaimed memory and either
            # falsely block the next attempt or (worse) not yet reflect it
            # and let the next attempt pile on top.
            models0 = models1 = []
            gc.collect()
            time.sleep(_ANE_BANK_RETRY_SETTLE_SECONDS)
            logger.warning(
                "ANE procedure bank compilation failed (%d banks per "
                "instance, %d procedures, %.2f GiB per instance)",
                len(spans),
                len(source_bytes),
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
                len(source_bytes),
                len(spans),
            )
        return models0, models1, 2 * len(spans)

    logger.warning(
        "Packed dual-ANE compilation failed; falling back to per-layer programs"
    )
    return None


def _compile_dual_banks(
    weights0: list[mx.array],
    weights1: list[mx.array],
    sequence_length: int,
) -> tuple[list[Any], list[Any], int] | None:
    """Compile the dual banks from fully staged fp32 slices.

    Kept for the calibration path and stale extensions; the production
    enable path streams slices through the incremental builder instead
    (issue #2781). See :func:`_bank_split_ladder` for the retry contract.
    """
    from omlx.custom_kernels.qwen35_prefill import fast

    return _bank_split_ladder(
        [int(weight.nbytes) for weight in weights0],
        lambda start, stop: (
            fast.qwen35_ane_compile_linear_bank(
                weights0[start:stop], sequence_length, 1
            ),
            fast.qwen35_ane_compile_linear_bank(
                weights1[start:stop], sequence_length, 2
            ),
        ),
    )


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

    for attempt in range(4):
        if not _ane_bank_memory_headroom_ok():
            current, total = _ane_bank_memory_footprint_snapshot()
            logger.warning(
                "Skipping single-ANE procedure bank compile attempt %d/4: "
                "phys footprint %.2f GiB / %.2f GiB total already past the "
                "%.0f%% safety fraction of system memory. Falling back to "
                "per-layer programs instead of risking a jetsam kill.",
                attempt + 1,
                current / (1 << 30),
                total / (1 << 30),
                _ANE_BANK_RETRY_MAX_MEMORY_FRACTION * 100,
            )
            return None
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
            gc.collect()
            time.sleep(_ANE_BANK_RETRY_SETTLE_SECONDS)
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


def _warm_ane_models(models: tuple[Any, ...] | list[Any]) -> None:
    """Pay private-runtime first-evaluation cost without making load fatal."""
    warm_start = time.perf_counter()
    warmed = 0
    try:
        for model in models:
            warmup = getattr(model, "warmup", None)
            if warmup is None:
                continue
            warmup()
            warmed += 1
    except Exception:
        logger.warning(
            "ANE warmup failed after %d procedures; continuing, the runtime "
            "failure latch handles broken procedures at first use",
            warmed,
            exc_info=True,
        )
    if warmed:
        logger.info(
            "Warmed %d ANE procedures in %.1fs at load",
            warmed,
            time.perf_counter() - warm_start,
        )


def _warm_cpu_sharing_path(
    sequence_length: int,
    cpu_mlps: list[Any],
    cpu_gdns: list[Any],
) -> None:
    """Dispatch one discarded dummy chunk through each CPU-shared module.

    When CPU sharing is on, the first dispatch additionally pays BNNS setup
    and the first touch of the eagerly dequantized FP16 CPU rows, which
    measured as a collapsed first request (prefill and decode). One dummy
    chunk per shared module moves that cost to load time; the merged output
    is discarded. Same soft-failure contract as the ANE warm loop.
    """
    if not cpu_mlps and not cpu_gdns:
        return
    warm_start = time.perf_counter()
    warmed = 0
    inputs: dict[int, mx.array] = {}

    def _warm_input(linear: Any) -> mx.array:
        dim = int(linear.weight.shape[1]) * 32 // int(linear.bits)
        x = inputs.get(dim)
        if x is None:
            x = mx.zeros((1, sequence_length, dim), dtype=linear.scales.dtype)
            mx.eval(x)
            inputs[dim] = x
        return x

    try:
        for module in cpu_mlps:
            out = _backend(module, _warm_input(module.gate_proj))
            if out is not None:
                mx.eval(out)
                warmed += 1
        for module in cpu_gdns:
            out = _gdn_backend(module, _warm_input(module.in_proj_qkv))
            if out is not None:
                mx.eval(*out)
                warmed += 1
    except Exception:
        logger.warning(
            "CPU sharing warmup failed after %d modules; continuing, "
            "the runtime failure latch handles broken modules at "
            "first use",
            warmed,
            exc_info=True,
        )
    if warmed:
        logger.info(
            "Warmed the CPU sharing path on %d modules in %.1fs at load",
            warmed,
            time.perf_counter() - warm_start,
        )


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

    prepared_mlps: list[tuple[Any, _CombinedMLPState]] = []
    prepared_gdns: list[tuple[Any, _CombinedGDNState]] = []
    gdn_config = _AneGDNConfig(
        config.sequence_length,
        gdn_fraction,
        config.variant,
        True,
        cpu_fraction=cpu_gdn_fraction,
        cpu_threads=config.cpu_threads,
        cpu_shared_resource=config.cpu_shared_resource,
        tail_padding_min_tokens=config.tail_padding_min_tokens,
    )
    with _COMPILE_LOCK:
        # Incremental staging (issue #2781): with the native builder each fp32
        # slice is handed over as soon as its layer is prepared and released
        # right away, so the peak fp32 staging is one layer instead of every
        # layer at once (~16 GiB on a 27B). The builder retains quarter-size
        # INT8 chunks and the split ladder recompiles spans from those chunks
        # without touching the fp32 sources. A stale extension without the
        # builder falls back to the previous hold-everything path.
        builder0 = builder1 = None
        if fast.has_symbol("AneLinearBankBuilder"):
            try:
                builder0 = fast.qwen35_ane_linear_bank_builder(
                    config.sequence_length
                )
                builder1 = fast.qwen35_ane_linear_bank_builder(
                    config.sequence_length
                )
            except Exception:
                logger.debug(
                    "ANE bank builder unavailable; staging all slices at once"
                )
                builder0 = builder1 = None
        weights0: list[mx.array] = []
        weights1: list[mx.array] = []
        source_bytes: list[int] = []

        def _stage(dense0: mx.array, dense1: mx.array) -> None:
            source_bytes.append(int(dense0.nbytes))
            if builder0 is not None:
                # The builder reads the raw fp32 buffers from C++, outside
                # MLX's own accessors, so make the GPU writes fully visible
                # before handing the pointers over.
                mx.eval(dense0, dense1)
                mx.synchronize()
                builder0.add(dense0)
                builder1.add(dense1)
            else:
                weights0.append(dense0)
                weights1.append(dense1)

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
                _stage(dense0, dense1)
                prepared_mlps.append((module, state))

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
                    _stage(dense0, dense1)
                    prepared_gdns.append((module, state))

        try:
            down_layer_stride = max(
                1,
                int(os.environ.get("OMLX_QWEN35_ANE_DOWN_LAYER_STRIDE", "1")),
            )
        except ValueError:
            down_layer_stride = 1
        down_entries = [
            (index, state.down_ane)
            for index, (_, state) in enumerate(prepared_mlps)
            if state.down_ane is not None and index % down_layer_stride == 0
        ]
        combine_down = bool(
            down_entries
            and os.environ.get(
                "OMLX_QWEN35_ANE_DOWN_COMBINED_BANK", ""
            ).strip().lower()
            in ("1", "true", "on")
        )
        procedure_entries: list[tuple[str, int]] = [
            *(("mlp", index) for index in range(len(prepared_mlps))),
            *(("gdn", index) for index in range(len(prepared_gdns))),
        ]
        if combine_down:
            for index, down_state in down_entries:
                _stage(down_state.compile_weight0, down_state.compile_weight1)
                procedure_entries.append(("down", index))
        separate_down_entries = [] if combine_down else down_entries
        down_weights0 = [
            state.compile_weight0 for _, state in separate_down_entries
        ]
        down_weights1 = [
            state.compile_weight1 for _, state in separate_down_entries
        ]
        procedure_count = len(procedure_entries) + len(separate_down_entries)
        if not procedure_count:
            return (0, 0, 0, 0)
        if procedure_count > 256:
            logger.warning(
                "ANE procedure bank exceeds the private 256-procedure limit; "
                "falling back to per-layer programs"
            )
            return None
        if builder0 is not None:
            banked_models = _bank_split_ladder(
                source_bytes,
                lambda start, stop: (
                    builder0.compile(1, start, stop),
                    builder1.compile(2, start, stop),
                ),
            )
        else:
            mx.eval(*weights0, *weights1)
            banked_models = _compile_dual_banks(
                weights0, weights1, config.sequence_length
            )
        weights0 = []
        weights1 = []
        builder0 = builder1 = None
        if banked_models is None:
            return None
        models0, models1, resident_program_count = banked_models
        down_models0: list[Any] = []
        down_models1: list[Any] = []
        if separate_down_entries:
            down_banks = _compile_dual_banks(
                down_weights0,
                down_weights1,
                config.sequence_length,
            )
            if down_banks is None:
                logger.warning(
                    "ANE down-projection bank compilation failed; "
                    "continuing with the established gate/GDN bank"
                )
                down_entries = []
                separate_down_entries = []
            else:
                down_models0, down_models1, down_programs = down_banks
                resident_program_count += down_programs
        down_weights0 = []
        down_weights1 = []

        if len(models0) != len(procedure_entries) or len(models1) != len(
            procedure_entries
        ):
            raise RuntimeError("ANE procedure bank returned an incomplete model list")

        selected_down_layers = {index for index, _ in down_entries}
        assigned_mlp_states = [
            state
            if index in selected_down_layers
            else replace(state, down_ane=None)
            for index, (_, state) in enumerate(prepared_mlps)
        ]
        assigned_gdn_states = [entry[1] for entry in prepared_gdns]
        if config.ane_down_fraction > 0 and not down_entries:
            assigned_mlp_states = [
                replace(state, down_ane=None) for state in assigned_mlp_states
            ]
        for procedure, (kind, index) in enumerate(procedure_entries):
            if kind == "mlp":
                assigned_mlp_states[index] = replace(
                    assigned_mlp_states[index],
                    model=models0[procedure],
                    model1=models1[procedure],
                )
            elif kind == "down":
                state = assigned_mlp_states[index]
                down_state = state.down_ane
                if down_state is None:
                    raise RuntimeError("Missing prepared ANE down state")
                assigned_mlp_states[index] = replace(
                    state,
                    down_ane=replace(
                        down_state,
                        model=models0[procedure],
                        model1=models1[procedure],
                        compile_weight0=None,
                        compile_weight1=None,
                    ),
                )
            else:
                assigned_gdn_states[index] = replace(
                    assigned_gdn_states[index],
                    model=models0[procedure],
                    model1=models1[procedure],
                )

        for procedure, (index, down_state) in enumerate(separate_down_entries):
            state = assigned_mlp_states[index]
            down_state = replace(
                down_state,
                model=down_models0[procedure],
                model1=down_models1[procedure],
                compile_weight0=None,
                compile_weight1=None,
            )
            assigned_mlp_states[index] = replace(
                state,
                down_ane=down_state,
            )

        for (module, _), state in zip(
            prepared_mlps, assigned_mlp_states, strict=True
        ):
            module._omlx_ane_prefill_config = config
            module._omlx_ane_prefill_state = state
        for (module, _), state in zip(
            prepared_gdns, assigned_gdn_states, strict=True
        ):
            module._omlx_ane_gdn_config = gdn_config
            module._omlx_ane_gdn_state = state
            _register_gdn_module(module)

        # Pay every procedure's first-evaluation cost now, while the model is
        # still loading, so the first user request measures inference rather
        # than ANE warmup. Guarded per-model so an older compiled extension
        # without warmup() degrades to the previous behavior instead of
        # failing the load.
        # A warmup failure latches ANE off for the owning module right here.
        # The per-module flag is checked at graph construction, which is the
        # only place the failure can still be intercepted: by evaluation time
        # the sticky per-procedure error re-raises inside the scheduler and
        # fails every request instead of falling back (#2940). Remaining
        # procedures keep warming, so one broken procedure costs one module
        # its ANE path rather than taking down the request path.
        warm_start = time.perf_counter()
        warmed = 0
        disabled = 0
        for procedure, (kind, index) in enumerate(procedure_entries):
            module = (
                prepared_gdns[index][0]
                if kind == "gdn"
                else prepared_mlps[index][0]
            )
            try:
                for warm_model in (models0[procedure], models1[procedure]):
                    warmup = getattr(warm_model, "warmup", None)
                    if warmup is None:
                        continue
                    warmup()
                    warmed += 1
            except Exception:
                if kind == "gdn":
                    module._omlx_ane_gdn_failed = True
                else:
                    module._omlx_ane_prefill_failed = True
                disabled += 1
                logger.warning(
                    "ANE warmup failed for procedure %d; disabling ANE for "
                    "its %s module and continuing",
                    procedure,
                    "GDN" if kind == "gdn" else "MLP",
                    exc_info=True,
                )
        for procedure, (index, _) in enumerate(separate_down_entries):
            module = prepared_mlps[index][0]
            try:
                for warm_model in (
                    down_models0[procedure],
                    down_models1[procedure],
                ):
                    warmup = getattr(warm_model, "warmup", None)
                    if warmup is None:
                        continue
                    warmup()
                    warmed += 1
            except Exception:
                module._omlx_ane_prefill_failed = True
                disabled += 1
                logger.warning(
                    "ANE down warmup failed for procedure %d; disabling ANE "
                    "for its MLP module and continuing",
                    procedure,
                    exc_info=True,
                )
        if disabled:
            logger.warning(
                "Disabled ANE on %d of %d modules after warmup failures; "
                "they fall back to GPU",
                disabled,
                procedure_count,
            )
        if warmed:
            logger.info(
                "Warmed %d ANE procedures in %.1fs at load",
                warmed,
                time.perf_counter() - warm_start,
            )

        # When CPU sharing is on, the first dispatch additionally pays BNNS
        # setup and the first touch of the eagerly dequantized FP16 CPU rows,
        # which measured as a collapsed first request (prefill and decode).
        # One dummy chunk per shared module moves that cost to load time; the
        # merged output is discarded. Same soft-failure contract as above.
        cpu_mlps = [
            module
            for module, state in prepared_mlps
            if getattr(state, "cpu_outputs", 0)
            or getattr(state, "down_cpu", None) is not None
            or getattr(getattr(state, "down_ane", None), "cpu_outputs", 0)
        ]
        cpu_gdns = [
            module
            for module, state in prepared_gdns
            if getattr(state, "cpu_outputs", 0)
        ]
        _warm_cpu_sharing_path(config.sequence_length, cpu_mlps, cpu_gdns)

    model._omlx_ane_down_prefill_count = sum(
        state.down_ane is not None for state in assigned_mlp_states
    )

    # Return the staging buffers to the OS now that compilation and warmup
    # are done. Synchronize first so no in-flight command buffer still
    # references a cached allocation (the issue #300 recipe); this runs on
    # the MLX executor thread like the engine pool's own calls (issue #2781).
    mx.synchronize()
    mx.clear_cache()

    return (
        len(prepared_mlps),
        len(prepared_mlps),
        len(prepared_gdns),
        resident_program_count,
    )


def _enable_fused_down_banks(
    model: Any,
    candidates: list[Any],
    config: _AnePrefillConfig,
) -> tuple[int, int] | None:
    """Compile contiguous 32-layer dual-ANE fused SwiGLU/down banks."""
    if not config.fused_down or not config.dual_ane or not candidates:
        return None
    from omlx.custom_kernels.qwen35_prefill import fast

    count = 0
    resident_programs = 0
    for chunk_start in range(0, len(candidates), 32):
        modules = candidates[chunk_start : chunk_start + 32]
        # Stream one gate/up/down triple at a time into the incremental
        # builders so each layer's fp32 staging arrays are released before
        # the next layer dequantizes (the issue #2781 recipe; the one-shot
        # path below kept a whole 32-layer chunk of fp32 alive at once).
        builder0 = builder1 = None
        if hasattr(fast, "qwen35_ane_fused_bank_builder"):
            try:
                builder0 = fast.qwen35_ane_fused_bank_builder(
                    config.sequence_length
                )
                builder1 = fast.qwen35_ane_fused_bank_builder(
                    config.sequence_length
                )
            except Exception:
                builder0 = builder1 = None
        staged: list[tuple[Any, Any]] = []
        legacy_weights: list[tuple[mx.array, ...]] = []
        for module in modules:
            value = _prepare_fused_down_for_bank(module, config)
            if value is None:
                continue
            state, weights = value
            if builder0 is not None:
                mx.eval(*weights)
                builder0.add(weights[0], weights[1], weights[2])
                builder1.add(weights[3], weights[4], weights[5])
            else:
                legacy_weights.append(weights)
            staged.append((module, state))
        if not staged:
            continue
        if builder0 is not None:
            models0 = builder0.compile(1, 0, builder0.size)
            models1 = builder1.compile(2, 0, builder1.size)
        else:
            mx.eval(*[w for weights in legacy_weights for w in weights])
            models0 = fast.qwen35_ane_compile_swiglu_down_bank(
                [weights[0] for weights in legacy_weights],
                [weights[1] for weights in legacy_weights],
                [weights[2] for weights in legacy_weights],
                config.sequence_length,
                1,
            )
            models1 = fast.qwen35_ane_compile_swiglu_down_bank(
                [weights[3] for weights in legacy_weights],
                [weights[4] for weights in legacy_weights],
                [weights[5] for weights in legacy_weights],
                config.sequence_length,
                2,
            )
        _warm_ane_models([*models0, *models1])
        for index, (module, state) in enumerate(staged):
            module._omlx_ane_prefill_config = config
            module._omlx_ane_fused_down_state = replace(
                state, model=models0[index], model1=models1[index]
            )
        count += len(staged)
        resident_programs += 2
        del staged, legacy_weights, builder0, builder1
        mx.clear_cache()
    if not count:
        return None
    model._omlx_ane_down_prefill_count = count
    return count, resident_programs


def _prepare_fused_down_for_bank(
    mlp: Any,
    config: _AnePrefillConfig,
) -> tuple[_FusedDownMLPState, tuple[mx.array, ...]] | None:
    """Prepare one fused MLP using the same slices in serving and tuning."""
    if not config.fused_down or not config.dual_ane:
        return None
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if any(
        int(getattr(linear, "bits", 0)) != 4
        or int(getattr(linear, "group_size", 0)) != 128
        for linear in (gate, up, down)
    ):
        return None
    hidden = int(gate.weight.shape[0])
    per_ane = (int(hidden * config.ane_down_fraction) // 128) * 128
    total_ane = 2 * per_ane
    cpu_hidden = (int(hidden * config.cpu_fraction) // 128) * 128
    gpu_start = total_ane + cpu_hidden
    if per_ane <= 0 or gpu_start >= hidden or (hidden - gpu_start) % 128:
        return None

    def dense_rows(linear: Any, start: int, stop: int) -> mx.array:
        return mx.contiguous(
            mx.dequantize(
                linear.weight[start:stop],
                linear.scales[start:stop],
                linear.biases[start:stop],
                group_size=128,
                bits=4,
            ).astype(mx.float32)
        )

    # Only columns [0:gpu_start] of the down matrix are ever consumed below
    # (down0/down1/cpu_down_weight); the GPU portion reads down.weight
    # directly, still quantized, at gate_up_weight/down_weight below.
    # Dequantizing the full matrix wasted a ~(hidden - gpu_start)-column
    # fp32 transient (~0.5GB/layer) that was computed and immediately
    # discarded. gpu_start is a multiple of 128 (per_ane/cpu_hidden are
    # both rounded down to 128 above), so slicing the packed axis at
    # gpu_start // 8 (4-bit: 8 values/int32) and gpu_start // 128 (the
    # quantization group_size) yields exactly the first gpu_start
    # unpacked columns -- same values as before, just without the wasted
    # suffix. See docs/qwen35-hardening-and-optimization.md C3.
    dense_down = mx.dequantize(
        down.weight[:, : gpu_start // 8],
        down.scales[:, : gpu_start // 128],
        down.biases[:, : gpu_start // 128],
        group_size=128,
        bits=4,
    ).astype(mx.float32)
    gate0 = dense_rows(gate, 0, per_ane)
    up0 = dense_rows(up, 0, per_ane)
    down0 = mx.contiguous(dense_down[:, :per_ane])
    gate1 = dense_rows(gate, per_ane, total_ane)
    up1 = dense_rows(up, per_ane, total_ane)
    down1 = mx.contiguous(dense_down[:, per_ane:total_ane])
    cpu_gate_up_weight = None
    cpu_down_weight = None
    if cpu_hidden:
        cpu_gate_up_weight = mx.contiguous(
            mx.concatenate(
                (
                    dense_rows(gate, total_ane, gpu_start),
                    dense_rows(up, total_ane, gpu_start),
                ),
                axis=0,
            ).astype(mx.float16)
        )
        cpu_down_weight = mx.contiguous(
            dense_down[:, total_ane:gpu_start].astype(mx.float16)
        )
    state = _FusedDownMLPState(
        model=None,
        model1=None,
        gate_up_weight=mx.contiguous(
            mx.concatenate((gate.weight[gpu_start:], up.weight[gpu_start:]), axis=0)
        ),
        gate_up_scales=mx.contiguous(
            mx.concatenate((gate.scales[gpu_start:], up.scales[gpu_start:]), axis=0)
        ),
        gate_up_biases=mx.contiguous(
            mx.concatenate((gate.biases[gpu_start:], up.biases[gpu_start:]), axis=0)
        ),
        down_weight=mx.contiguous(down.weight[:, gpu_start // 8 :]),
        down_scales=mx.contiguous(down.scales[:, gpu_start // 128 :]),
        down_biases=mx.contiguous(down.biases[:, gpu_start // 128 :]),
        cpu_gate_up_weight=cpu_gate_up_weight,
        cpu_down_weight=cpu_down_weight,
    )
    values = [
        state.gate_up_weight,
        state.gate_up_scales,
        state.gate_up_biases,
        state.down_weight,
        state.down_scales,
        state.down_biases,
        gate0,
        up0,
        down0,
        gate1,
        up1,
        down1,
    ]
    if cpu_gate_up_weight is not None and cpu_down_weight is not None:
        values.extend((cpu_gate_up_weight, cpu_down_weight))
    mx.eval(*values)
    return state, (gate0, up0, down0, gate1, up1, down1)


def _enable_fused_gdn_banks(
    model: Any,
    config: _AnePrefillConfig,
    *,
    fraction: float,
    max_layers: int,
    cpu_fraction: float,
) -> tuple[int, int] | None:
    """Compile the established dual-ANE GDN path alongside fused MLP/down."""
    if not config.dual_ane or max_layers <= 0:
        return None

    gdn_config = _AneGDNConfig(
        config.sequence_length,
        fraction,
        config.variant,
        True,
        cpu_fraction=cpu_fraction,
        cpu_threads=config.cpu_threads,
        cpu_shared_resource=config.cpu_shared_resource,
        tail_padding_min_tokens=config.tail_padding_min_tokens,
    )
    prepared: list[tuple[Any, _CombinedGDNState, mx.array, mx.array]] = []
    with _COMPILE_LOCK:
        for module in model.modules() if hasattr(model, "modules") else ():
            if len(prepared) >= max_layers:
                break
            if not _eligible_gdn(module):
                continue
            try:
                entry = _prepare_gdn_for_bank(module, gdn_config)
            except Exception:
                logger.warning(
                    "Skipping one Qwen GDN while preparing its fused-mode "
                    "ANE procedure",
                    exc_info=True,
                )
                continue
            if entry is not None:
                state, dense0, dense1 = entry
                if dense1 is None:
                    continue
                prepared.append((module, state, dense0, dense1))

        if not prepared:
            return None
        weights0 = [entry[2] for entry in prepared]
        weights1 = [entry[3] for entry in prepared]
        mx.eval(*weights0, *weights1)
        banked = _compile_dual_banks(
            weights0,
            weights1,
            config.sequence_length,
        )
        if banked is None:
            return None
        models0, models1, resident_programs = banked
        if len(models0) != len(prepared) or len(models1) != len(prepared):
            raise RuntimeError("ANE GDN bank returned an incomplete model list")
        _warm_ane_models([*models0, *models1])
        for index, (module, state, _, _) in enumerate(prepared):
            module._omlx_ane_gdn_config = gdn_config
            module._omlx_ane_gdn_state = replace(
                state,
                model=models0[index],
                model1=models1[index],
            )
            _register_gdn_module(module)

    return len(prepared), resident_programs


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
    ane_down_fraction: float = 0.0,
    fused_down: bool = False,
    cpu_gdn_fraction: float = 0.0,
    cpu_threads: int = 8,
    cpu_shared_resource: bool = True,
    tail_padding_min_tokens: int = 0,
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
    if getattr(model, "_omlx_ane_prefill_shed", False):
        # A fresh enable supersedes a runtime shed on this object.
        model._omlx_ane_prefill_shed = False
    if gdn_max_layers < 0:
        raise ValueError("ANE GDN prefill max_layers must be non-negative")
    if not 0.0 <= cpu_fraction <= 0.25:
        raise ValueError("ANE CPU fp16 fraction must be between 0.0 and 0.25")
    if not 0.0 <= cpu_down_fraction <= 0.50:
        raise ValueError(
            "ANE CPU down-projection fraction must be between 0.0 and 0.50"
        )
    if not 0.0 <= ane_down_fraction <= 0.50:
        raise ValueError(
            "ANE down-projection fraction must be between 0.0 and 0.50"
        )
    if not 0.0 <= cpu_gdn_fraction <= 0.50:
        raise ValueError("ANE CPU GDN fraction must be between 0.0 and 0.50")
    if not 0 <= cpu_threads <= 64:
        raise ValueError("ANE CPU worker count must be between 0 and 64")
    if not 0 <= tail_padding_min_tokens < sequence_length:
        raise ValueError(
            "ANE tail padding threshold must be zero or less than sequence_length"
        )

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
        ane_down_fraction=ane_down_fraction if dual_ane else 0.0,
        fused_down=fused_down and dual_ane,
        tail_padding_min_tokens=tail_padding_min_tokens,
    )
    model._omlx_ane_tail_padding_min_tokens = tail_padding_min_tokens
    if ane_down_fraction > 0 and not dual_ane:
        logger.warning(
            "Experimental ANE down projection currently requires dual ANE; "
            "continuing without ANE down offload"
        )
    candidates = []
    scanned_mlp = 0
    modules = model.modules() if hasattr(model, "modules") else ()
    for module in modules:
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

    if config.fused_down:
        if config.cpu_down_fraction > 0:
            logger.warning(
                "Fused ANE MLP/down uses one CPU hidden-channel share for "
                "gate/up and partial down; ignoring the output-row CPU down "
                "fraction"
            )
            config = replace(config, cpu_down_fraction=0.0)
        fused = _enable_fused_down_banks(model, candidates, config)
        if fused is not None:
            count, resident_programs = fused
            gdn_count = 0
            if gdn:
                fused_gdn = _enable_fused_gdn_banks(
                    model,
                    config,
                    fraction=gdn_fraction,
                    max_layers=gdn_max_layers,
                    cpu_fraction=cpu_gdn_fraction,
                )
                if fused_gdn is None:
                    logger.warning(
                        "Fused ANE down mode could not compile the requested "
                        "GDN procedures; leaving GDN on GPU"
                    )
                else:
                    gdn_count, gdn_programs = fused_gdn
                    resident_programs += gdn_programs
            model._omlx_ane_mlp_prefill_count = count
            model._omlx_ane_gdn_prefill_count = gdn_count
            model._omlx_ane_dual_prefill_count = count
            model._omlx_ane_resident_program_count = resident_programs
            model._omlx_ane_procedure_count = count + gdn_count
            _warn_gdn_below_floor(
                model,
                bool(gdn and gdn_max_layers),
                gdn_count,
                gdn_fraction,
                dual_ane,
            )
            _log_gdn_recurrent_safe_cap(
                model, gdn_fraction, gdn_count, dual_ane
            )
            logger.info(
                "Eagerly compiled %d fused MLP/down and %d GDN procedures "
                "into %d "
                "instance-pinned ANE programs (sequence_length=%d, "
                "gpu_suffix=%s)",
                count,
                gdn_count,
                resident_programs,
                sequence_length,
                (
                    "NAX"
                    if fast.qwen35_ane_hybrid_nax_enabled()
                    else "classic Metal"
                ),
            )
            cpu_mlps = []
            for module in candidates:
                fused_state = getattr(
                    module, "_omlx_ane_fused_down_state", None
                )
                if (
                    fused_state is not None
                    and fused_state.cpu_gate_up_weight is not None
                ):
                    cpu_mlps.append(module)
            cpu_gdns = [
                module
                for module in (
                    model.modules() if hasattr(model, "modules") else ()
                )
                if getattr(
                    getattr(module, "_omlx_ane_gdn_state", None),
                    "cpu_outputs",
                    0,
                )
            ]
            _warm_cpu_sharing_path(sequence_length, cpu_mlps, cpu_gdns)
            # Return the staging buffers to the OS; this branch skips the
            # dual-bank epilogue that normally does this (issue #2781).
            mx.synchronize()
            mx.clear_cache()
            return count

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
        down_count = int(getattr(model, "_omlx_ane_down_prefill_count", 0))
        model._omlx_ane_procedure_count = count + gdn_count + down_count
        _warn_gdn_below_floor(
            model, bool(gdn and gdn_max_layers), gdn_count, gdn_fraction, dual_ane
        )
        _log_gdn_recurrent_safe_cap(model, gdn_fraction, gdn_count, dual_ane)
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
            tail_padding_min_tokens=config.tail_padding_min_tokens,
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
    model._omlx_ane_down_prefill_count = 0
    model._omlx_ane_resident_program_count = resident_programs
    model._omlx_ane_procedure_count = count + gdn_count
    _warn_gdn_below_floor(
        model, bool(gdn and gdn_max_layers), gdn_count, gdn_fraction, dual_ane
    )
    _log_gdn_recurrent_safe_cap(model, gdn_fraction, gdn_count, dual_ane)

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


def ane_prefill_transient_bytes(model: Any) -> int:
    """Bytes of fp16 ANE I/O surfaces held by ``model``'s compiled slices.

    Every compiled procedure owns a fixed-shape input and output IOSurface of
    ``dim * sequence_length * 2`` bytes, allocated at compile time and dirtied
    at first use, which is exactly the first-request spike of issue #2841.
    Reads the dims off the live native models, so packing, dual splits, and
    partial banks are all accounted exactly. 0 when no ANE slice is attached.
    """
    total = 0
    for module in model.modules() if hasattr(model, "modules") else ():
        states = []
        for state_attr in (
            "_omlx_ane_prefill_state",
            "_omlx_ane_gdn_state",
            "_omlx_ane_fused_down_state",
        ):
            state = getattr(module, state_attr, None)
            if state is None:
                continue
            states.append(state)
            down_ane = getattr(state, "down_ane", None)
            if down_ane is not None:
                states.append(down_ane)
        for state in states:
            for ane_model in (state.model, getattr(state, "model1", None)):
                input_dim = getattr(ane_model, "input_dim", 0)
                output_dim = getattr(ane_model, "output_dim", 0)
                seq = getattr(ane_model, "sequence_length", 0)
                try:
                    total += (int(input_dim) + int(output_dim)) * int(seq) * 2
                except (TypeError, ValueError):
                    continue
    return total


def qwen35_ane_prefill_status(model: Any) -> dict:
    """Return JSON-serialisable ANE prefill configuration counters."""
    attempted = hasattr(model, "_omlx_ane_mlp_prefill_count")
    mlp = int(getattr(model, "_omlx_ane_mlp_prefill_count", 0) or 0)
    gdn = int(getattr(model, "_omlx_ane_gdn_prefill_count", 0) or 0)
    return {
        "attempted": attempted,
        "configured": bool(mlp or gdn),
        "shed": bool(getattr(model, "_omlx_ane_prefill_shed", False)),
        "mlp_layers": mlp,
        "gdn_layers": gdn,
        "dual_ane_layers": int(
            getattr(model, "_omlx_ane_dual_prefill_count", 0) or 0
        ),
        "resident_programs": int(
            getattr(model, "_omlx_ane_resident_program_count", 0) or 0
        ),
        "tail_padding_min_tokens": int(
            getattr(model, "_omlx_ane_tail_padding_min_tokens", 0) or 0
        ),
    }


def release_qwen35_ane_prefill(model: Any) -> tuple[int, int]:
    """Drop every compiled ANE prefill slice on ``model``; GPU-only from here.

    The compiled procedure banks keep the packed weight blobs they were built
    from mapped for the lifetime of the native programs (roughly 13 GB for a
    27B at mlp 0.35 / gdn 0.45) — exactly the resident memory a long-context
    prefill needs back once the KV cache has grown into the guard's sizing
    target. Latches every sliced module through the existing per-module
    failure flags first (so the dispatch sites fall back to stock GPU compute
    and never lazily recompile), then drops the state references; the native
    models free their programs and mapped blobs when the last reference dies.
    Idempotent, and scoped to this engine instance: the next load of the
    model rebuilds the banks from its settings.

    Returns ``(modules_released, resident_programs_before)``.
    """
    modules_released = 0
    programs_before = int(
        getattr(model, "_omlx_ane_resident_program_count", 0) or 0
    )
    for module in model.modules() if hasattr(model, "modules") else ():
        for state_attr, failed_attr in (
            ("_omlx_ane_prefill_state", "_omlx_ane_prefill_failed"),
            ("_omlx_ane_fused_down_state", "_omlx_ane_prefill_failed"),
            ("_omlx_ane_gdn_state", "_omlx_ane_gdn_failed"),
        ):
            if getattr(module, state_attr, None) is None:
                continue
            # Latch BEFORE dropping the state: the fetch sites lazily
            # recompile a missing state, and the failure flag is the one
            # switch they all check first.
            setattr(module, failed_attr, True)
            setattr(module, state_attr, None)
            modules_released += 1
    if modules_released:
        # Zero (not delete) the status counters and set the shed marker:
        # attempted=True/configured=False alone is indistinguishable from a
        # load-time compile failure, and a shed changes serving behavior
        # until the next load.
        model._omlx_ane_prefill_shed = True
        for counter in (
            "_omlx_ane_mlp_prefill_count",
            "_omlx_ane_gdn_prefill_count",
            "_omlx_ane_dual_prefill_count",
            "_omlx_ane_resident_program_count",
        ):
            if hasattr(model, counter):
                setattr(model, counter, 0)
    return modules_released, programs_before
