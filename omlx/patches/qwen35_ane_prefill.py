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
import weakref
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu

from omlx.custom_kernels.nax import is_nax_available

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


@dataclass(frozen=True)
class _AneGDNConfig:
    sequence_length: int
    fraction: float
    variant: int
    dual_ane: bool = False


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
    allowed_bits: tuple[int, ...] = (4, 5),
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


def _eligible_input(x: mx.array, config: _AnePrefillConfig) -> bool:
    if x.dtype not in (mx.float16, mx.bfloat16) or x.ndim < 3:
        return False
    input_dim = int(x.shape[-1])
    return int(x.size // input_dim) == config.sequence_length


def configure_qwen35_ane_prefill_scheduler(
    scheduler: Any,
    sequence_length: int,
) -> bool:
    """Align scheduler prompt chunks with the compiled fixed ANE shape."""
    if sequence_length < 1024 or sequence_length % 64:
        raise ValueError(
            "ANE prefill sequence_length must be a multiple of 64 >= 1024"
        )
    config = getattr(scheduler, "config", None)
    if config is None:
        return False
    config.prefill_step_size = int(sequence_length)
    if hasattr(scheduler, "_qwen35_prefill_floor"):
        scheduler._qwen35_prefill_floor = 0
    logger.info(
        "Qwen ANE prefill scheduler aligned to fixed shape %d",
        sequence_length,
    )
    return True


def _eligible_pair(mlp: Any) -> bool:
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    gate_dtype = getattr(getattr(gate, "scales", None), "dtype", None)
    gate_spec = _affine_spec(gate, gate_dtype, allowed_bits=(4,))
    up_spec = _affine_spec(up, gate_dtype, allowed_bits=(4,))
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
    group_size = int(gate.group_size)
    dual_ane = bool(
        config.dual_ane
        and fast.has_symbol("qwen35_ane_dual_q4_swiglu_t")
        and fast.has_symbol("qwen35_ane_dual_affine_qmm_t")
    )
    alignment = 128 if dual_ane else 64
    ane_outputs = (int(output_dim * config.fraction) // alignment) * alignment
    gpu_outputs = output_dim - ane_outputs
    if ane_outputs <= 0 or gpu_outputs <= 0 or gpu_outputs % 64:
        return None

    key = (
        config.sequence_length,
        ane_outputs,
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
                            bits=4,
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
        weight = mx.contiguous(
            mx.concatenate((gate.weight[ane_outputs:], up.weight[ane_outputs:]), axis=0)
        )
        scales = mx.contiguous(
            mx.concatenate((gate.scales[ane_outputs:], up.scales[ane_outputs:]), axis=0)
        )
        biases = mx.contiguous(
            mx.concatenate((gate.biases[ane_outputs:], up.biases[ane_outputs:]), axis=0)
        )
        values = [dense0, weight, scales, biases]
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
            model1=model1,
            group_size=group_size,
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
) -> tuple[_CombinedMLPState, mx.array, mx.array] | None:
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    if not _eligible_pair(mlp):
        return None
    output_dim = int(gate.weight.shape[0])
    group_size = int(gate.group_size)
    ane_outputs = (int(output_dim * config.fraction) // 128) * 128
    gpu_outputs = output_dim - ane_outputs
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
                        bits=4,
                    ).astype(mx.float32)
                    for linear in (gate, up)
                ],
                axis=0,
            )
        )

    split = ane_outputs // 2
    dense0 = dense_slice(0, split)
    dense1 = dense_slice(split, ane_outputs)
    weight = mx.contiguous(
        mx.concatenate((gate.weight[ane_outputs:], up.weight[ane_outputs:]), axis=0)
    )
    scales = mx.contiguous(
        mx.concatenate((gate.scales[ane_outputs:], up.scales[ane_outputs:]), axis=0)
    )
    biases = mx.contiguous(
        mx.concatenate((gate.biases[ane_outputs:], up.biases[ane_outputs:]), axis=0)
    )
    mx.eval(dense0, dense1, weight, scales, biases)
    return (
        _CombinedMLPState(
            model=None,
            weight=weight,
            scales=scales,
            biases=biases,
            ane_outputs=ane_outputs,
            gpu_outputs=gpu_outputs,
            model1=None,
            group_size=group_size,
        ),
        dense0,
        dense1,
    )


def _gdn_linears(gdn: Any) -> tuple[Any, Any, Any, Any]:
    return (
        getattr(gdn, "in_proj_qkv", None),
        getattr(gdn, "in_proj_z", None),
        getattr(gdn, "in_proj_b", None),
        getattr(gdn, "in_proj_a", None),
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


def _compile_gdn(gdn: Any, config: _AneGDNConfig) -> _CombinedGDNState | None:
    from omlx.custom_kernels.qwen35_prefill import fast

    if not _eligible_gdn(gdn) or not fast.has_symbol("qwen35_ane_affine_qmm_t"):
        return None
    qkv, z, _, _ = _gdn_linears(gdn)
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
    gpu_outputs = total_outputs - ane_outputs
    # The native GPU suffix accepts one quantization format. Put all of z on
    # ANE so an oQ4e-style q5-z/q4-qkv mix leaves a homogeneous qkv suffix.
    if ane_outputs < z_outputs or gpu_outputs <= 0 or gpu_outputs % 64:
        return None
    key = (
        config.sequence_length,
        ane_outputs,
        qkv_spec,
        z_spec,
        "z_qkv_dual_row_int8" if dual_ane else "z_qkv_row_int8",
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
        weight = mx.contiguous(qkv.weight[qkv_offset:])
        scales = mx.contiguous(qkv.scales[qkv_offset:])
        biases = mx.contiguous(qkv.biases[qkv_offset:])
        values = [dense0, weight, scales, biases]
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
        )
        cache[key] = state
        return state


def _prepare_gdn_for_bank(
    gdn: Any, config: _AneGDNConfig
) -> tuple[_CombinedGDNState, mx.array, mx.array] | None:
    if not _eligible_gdn(gdn):
        return None
    qkv, z, _, _ = _gdn_linears(gdn)
    logical = (z, qkv)
    z_outputs = int(z.weight.shape[0])
    qkv_outputs = int(qkv.weight.shape[0])
    total_outputs = z_outputs + qkv_outputs
    qkv_spec = _affine_spec(qkv, qkv.scales.dtype)
    z_spec = _affine_spec(z, qkv.scales.dtype)
    if qkv_spec is None or z_spec is None:
        return None
    qkv_bits, qkv_group_size = qkv_spec
    ane_outputs = (int(total_outputs * config.fraction) // 128) * 128
    gpu_outputs = total_outputs - ane_outputs
    if ane_outputs < z_outputs or gpu_outputs <= 0 or gpu_outputs % 64:
        return None

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

    split = ane_outputs // 2
    dense0 = dense_logical_slice(0, split)
    dense1 = dense_logical_slice(split, ane_outputs)
    qkv_offset = ane_outputs - z_outputs
    weight = mx.contiguous(qkv.weight[qkv_offset:])
    scales = mx.contiguous(qkv.scales[qkv_offset:])
    biases = mx.contiguous(qkv.biases[qkv_offset:])
    mx.eval(dense0, dense1, weight, scales, biases)
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
        ),
        dense0,
        dense1,
    )


def _gdn_backend(
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
        from omlx.patches.qwen35_q4_mlp import _linear_qmm

        if state.model1 is not None:
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
        _, _, b_proj, a_proj = _gdn_linears(gdn)
        b = _linear_qmm(b_proj, x, config.variant)
        a = _linear_qmm(a_proj, x, config.variant)
        return mixed_qkv, z, b, a
    except Exception:
        gdn._omlx_ane_gdn_failed = True
        logger.warning(
            "Disabling ANE GDN prefill after a runtime failure", exc_info=True
        )
        return None


def _backend(
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
    if state.scales.dtype != x.dtype or int(state.weight.shape[1]) * 8 != int(
        x.shape[-1]
    ):
        return None

    try:
        from omlx.custom_kernels.qwen35_prefill import fast
        from omlx.patches.qwen35_q4_mlp import _linear_qmm

        if state.model1 is not None:
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
            return _linear_qmm(mlp.down_proj, activation, config.variant)
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
            return _linear_qmm(mlp.down_proj, activation, config.variant)

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

        return _linear_qmm(
            mlp.down_proj,
            swiglu(gate, up),
            config.variant,
        )
    except Exception:
        mlp._omlx_ane_prefill_failed = True
        logger.warning(
            "Disabling ANE prefill for one Qwen MLP after a runtime failure",
            exc_info=True,
        )
        return None


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


def _enable_dual_procedure_banks(
    model: Any,
    mlp_candidates: list[Any],
    config: _AnePrefillConfig,
    *,
    gdn: bool,
    gdn_fraction: float,
    gdn_max_layers: int,
) -> tuple[int, int, int, int] | None:
    from omlx.custom_kernels.qwen35_prefill import fast

    if not (
        config.dual_ane
        and fast.has_symbol("qwen35_ane_compile_linear_bank")
        and fast.has_symbol("qwen35_ane_dual_q4_swiglu_t")
        and fast.has_symbol("qwen35_ane_dual_affine_qmm_t")
    ):
        return None

    prepared_mlps: list[tuple[Any, _CombinedMLPState, mx.array, mx.array]] = []
    prepared_gdns: list[tuple[Any, _CombinedGDNState, mx.array, mx.array]] = []
    gdn_config = _AneGDNConfig(
        config.sequence_length, gdn_fraction, config.variant, True
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

    env = os.environ.get("OMLX_QWEN35_ANE_PREFILL", "").strip().lower()
    if env in ("0", "false", "off"):
        logger.info("Qwen ANE prefill disabled by OMLX_QWEN35_ANE_PREFILL")
        return 0
    if env not in ("1", "true", "on") and is_nax_available():
        # Auto: on NAX GPUs (M5 family) the tensor units run the quantized
        # prefill matmuls faster than the ANE INT8 offload, so the hybrid
        # split makes the ANE the bottleneck while the fixed-shape scheduler
        # alignment also drops the wider Qwen prefill floor (M5 Pro report:
        # 17.6k-token TTFT 54.7s -> 65-88s on 0.6.1, issue #2779).
        # OMLX_QWEN35_ANE_PREFILL=1 forces the path on for benchmarking.
        logger.info(
            "Qwen ANE prefill skipped: NAX GPU, tensor-unit prefill is faster"
        )
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
        return 0

    config = _AnePrefillConfig(sequence_length, fraction, variant, dual_ane)
    candidates = []
    modules = model.modules() if hasattr(model, "modules") else ()
    for module in modules:
        if not all(
            hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")
        ):
            continue
        if not _eligible_pair(module):
            continue
        candidates.append(module)
        if len(candidates) >= max_layers:
            break

    banked = _enable_dual_procedure_banks(
        model,
        candidates,
        config,
        gdn=gdn,
        gdn_fraction=gdn_fraction,
        gdn_max_layers=gdn_max_layers,
    )
    if banked is not None:
        count, dual_count, gdn_count, resident_programs = banked
        model._omlx_ane_mlp_prefill_count = count
        model._omlx_ane_gdn_prefill_count = gdn_count
        model._omlx_ane_dual_prefill_count = dual_count
        model._omlx_ane_resident_program_count = resident_programs
        model._omlx_ane_procedure_count = count + gdn_count
        if count:
            logger.info(
                "Eagerly compiled %d MLP and %d GDN procedures into %d "
                "instance-pinned ANE programs (sequence_length=%d)",
                count,
                gdn_count,
                resident_programs,
                sequence_length,
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
        gdn_config = _AneGDNConfig(sequence_length, gdn_fraction, variant, dual_ane)
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
    return count
