"""ANE and CPU offload for the affine qmm and its profiling."""

from __future__ import annotations

import mlx.core as mx

from .fast import _IMPORT_ERROR, _ext


def is_native_available() -> bool:
    return _ext is not None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def has_symbol(name: str) -> bool:
    return hasattr(_ext, name)


def _up_suffix_kwargs(
    up_suffix: tuple[mx.array, mx.array, mx.array] | None,
) -> dict[str, mx.array]:
    if up_suffix is None:
        return {}
    weight, scales, biases = up_suffix
    return {"up_weight": weight, "up_scales": scales, "up_biases": biases}


def cpu_fp16_affine_qmm_t(
    x: mx.array,
    cpu_weight: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    bits: int,
    variant: int = 8,
    group_size: int = 128,
    cpu_threads: int = 0,
    cpu_shared_resource: bool = False,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "cpu_fp16_affine_qmm_t"):
        raise RuntimeError("CPU/GPU fp16 hybrid qmm native kernel is unavailable")
    args = (
        x,
        cpu_weight,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        bits,
        variant,
        group_size,
        cpu_threads,
    )
    if hasattr(_ext, "cpu_shared_resource_available"):
        return _ext.cpu_fp16_affine_qmm_t(
            *args, cpu_shared_resource
        )
    return _ext.cpu_fp16_affine_qmm_t(*args)


def ane_dual_cpu_fp16_q4_swiglu_t(
    x: mx.array,
    cpu_weight: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    ane_model0,
    ane_model1,
    variant: int = 8,
    group_size: int = 128,
    cpu_threads: int = 0,
    cpu_shared_resource: bool = False,
    up_suffix: tuple[mx.array, mx.array, mx.array] | None = None,
) -> mx.array:
    if _ext is None or not hasattr(
        _ext, "ane_dual_cpu_fp16_q4_swiglu_t"
    ):
        raise RuntimeError("ANE/CPU/GPU fp16 hybrid SwiGLU is unavailable")
    args = (
        x,
        cpu_weight,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        ane_model0,
        ane_model1,
        variant,
        group_size,
        cpu_threads,
    )
    if hasattr(_ext, "cpu_shared_resource_available"):
        return _ext.ane_dual_cpu_fp16_q4_swiglu_t(
            *args, cpu_shared_resource, **_up_suffix_kwargs(up_suffix)
        )
    return _ext.ane_dual_cpu_fp16_q4_swiglu_t(*args, **_up_suffix_kwargs(up_suffix))


def ane_cpu_fp16_q4_swiglu_t(
    x: mx.array,
    cpu_weight: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    ane_model,
    variant: int = 8,
    group_size: int = 128,
    cpu_threads: int = 0,
    cpu_shared_resource: bool = False,
    up_suffix: tuple[mx.array, mx.array, mx.array] | None = None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "ane_cpu_fp16_q4_swiglu_t"):
        raise RuntimeError("Single-ANE/CPU/GPU fp16 q4 SwiGLU is unavailable")
    return _ext.ane_cpu_fp16_q4_swiglu_t(
        x,
        cpu_weight,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        ane_model,
        variant,
        group_size,
        cpu_threads,
        cpu_shared_resource,
        **_up_suffix_kwargs(up_suffix),
    )


def ane_cpu_fp16_swiglu_t(
    x: mx.array,
    cpu_weight: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    ane_model,
    bits: int,
    variant: int = 8,
    group_size: int = 128,
    cpu_threads: int = 0,
    cpu_shared_resource: bool = False,
    up_suffix: tuple[mx.array, mx.array, mx.array] | None = None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "ane_cpu_fp16_swiglu_t"):
        raise RuntimeError("Single-ANE/CPU/GPU fp16 SwiGLU is unavailable")
    return _ext.ane_cpu_fp16_swiglu_t(
        x,
        cpu_weight,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        ane_model,
        bits,
        variant,
        group_size,
        cpu_threads,
        cpu_shared_resource,
        **_up_suffix_kwargs(up_suffix),
    )


def ane_cpu_fp16_affine_qmm_t(
    x: mx.array,
    cpu_weight: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    ane_model,
    bits: int,
    variant: int = 8,
    group_size: int = 128,
    profile_category: int = 1,
    cpu_threads: int = 0,
    cpu_shared_resource: bool = False,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "ane_cpu_fp16_affine_qmm_t"):
        raise RuntimeError("Single-ANE/CPU/GPU fp16 affine qmm is unavailable")
    return _ext.ane_cpu_fp16_affine_qmm_t(
        x,
        cpu_weight,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        ane_model,
        bits,
        variant,
        group_size,
        profile_category,
        cpu_threads,
        cpu_shared_resource,
    )


def ane_dual_cpu_fp16_swiglu_t(
    x: mx.array,
    cpu_weight: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    ane_model0,
    ane_model1,
    bits: int,
    variant: int = 8,
    group_size: int = 128,
    cpu_threads: int = 0,
    cpu_shared_resource: bool = False,
    up_suffix: tuple[mx.array, mx.array, mx.array] | None = None,
) -> mx.array:
    if _ext is None or not hasattr(
        _ext, "ane_dual_cpu_fp16_swiglu_t"
    ):
        raise RuntimeError("ANE/CPU/GPU fp16 hybrid SwiGLU is unavailable")
    return _ext.ane_dual_cpu_fp16_swiglu_t(
        x,
        cpu_weight,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        ane_model0,
        ane_model1,
        bits,
        variant,
        group_size,
        cpu_threads,
        cpu_shared_resource,
        **_up_suffix_kwargs(up_suffix),
    )


def ane_dual_cpu_fp16_affine_qmm_t(
    x: mx.array,
    cpu_weight: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    ane_model0,
    ane_model1,
    bits: int,
    variant: int = 8,
    group_size: int = 128,
    profile_category: int = 1,
    cpu_threads: int = 0,
    cpu_shared_resource: bool = False,
) -> mx.array:
    if _ext is None or not hasattr(
        _ext, "ane_dual_cpu_fp16_affine_qmm_t"
    ):
        raise RuntimeError("ANE/CPU/GPU fp16 hybrid affine qmm is unavailable")
    return _ext.ane_dual_cpu_fp16_affine_qmm_t(
        x,
        cpu_weight,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        ane_model0,
        ane_model1,
        bits,
        variant,
        group_size,
        profile_category,
        cpu_threads,
        cpu_shared_resource,
    )


def cpu_shared_resource_available() -> bool:
    """Whether dispatch_apply supports shared-resource scheduling attributes."""
    return bool(
        _ext is not None
        and hasattr(_ext, "cpu_shared_resource_available")
        and _ext.cpu_shared_resource_available()
    )


def ane_available() -> bool:
    """Whether the opt-in private ANE runtime is usable on this host."""
    return bool(
        _ext is not None
        and hasattr(_ext, "ane_available")
        and _ext.ane_available()
    )


def ane_hybrid_nax_enabled() -> bool:
    """Whether ANE hybrid GPU suffixes currently select bundled NAX QMM."""
    return bool(
        _ext is not None
        and hasattr(_ext, "ane_hybrid_nax_enabled")
        and _ext.ane_hybrid_nax_enabled()
    )


_ANE_PROFILE_KEYS = (
    "operations",
    "pack_ns",
    "ane_region_ns",
    "ane0_eval_ns",
    "ane1_eval_ns",
    "ane0_launch_ns",
    "ane1_launch_ns",
    "gpu_qmm_ns",
    "gpu_completion_ns",
    "cpu_matmul_ns",
    "cpu_completion_ns",
    "ane_last",
    "gpu_last",
    "gap_before_ns",
)


def ane_profile_reset() -> None:
    if _ext is not None and hasattr(_ext, "ane_profile_reset"):
        _ext.ane_profile_reset()


def ane_profile_set_enabled(enabled: bool) -> bool:
    """Enable native ANE phase counters for a bounded diagnostic window."""
    if _ext is None or not hasattr(_ext, "ane_profile_set_enabled"):
        return False
    _ext.ane_profile_set_enabled(bool(enabled))
    return True


def ane_profile_snapshot() -> dict[str, dict[str, float]]:
    if _ext is None or not hasattr(_ext, "ane_profile_snapshot"):
        return {}
    values = list(_ext.ane_profile_snapshot())
    width = len(_ANE_PROFILE_KEYS)
    if len(values) != 2 * width:
        return {}
    return {
        name: dict(zip(_ANE_PROFILE_KEYS, values[index * width : (index + 1) * width]))
        for index, name in enumerate(("mlp", "gdn"))
    }


def ane_compile_linear(
    weight: mx.array, sequence_length: int, ane_instance: int = 0
):
    if not ane_available():
        raise RuntimeError("Private ANE runtime is unavailable")
    try:
        return _ext.ane_compile_linear(weight, sequence_length, ane_instance)
    except TypeError:
        if ane_instance:
            raise RuntimeError(
                "The native extension does not support ANE instance pinning"
            ) from None
        return _ext.ane_compile_linear(weight, sequence_length)


def ane_bank_compiler_available() -> bool:
    """True when both the private ANE runtime and the procedure-bank compiler
    entry point are present. Callers that would otherwise discover
    unavailability via the RuntimeError below (the ANE tuner in particular)
    can probe this up front instead of failing deep inside a compile ladder
    (#3044)."""
    return (
        ane_available()
        and _ext is not None
        and hasattr(_ext, "ane_compile_linear_bank")
    )


def ane_compile_linear_bank(
    weights: list[mx.array], sequence_length: int, ane_instance: int
):
    if not ane_bank_compiler_available():
        raise RuntimeError("Private ANE procedure-bank compiler is unavailable")
    return _ext.ane_compile_linear_bank(
        weights, sequence_length, ane_instance
    )


def ane_linear_bank_builder(sequence_length: int):
    """Incremental bank builder: add() converts one fp32 slice at a time so
    the caller can release each staging array immediately (issue #2781)."""
    if not ane_available() or _ext is None or not hasattr(
        _ext, "AneLinearBankBuilder"
    ):
        raise RuntimeError("Private ANE procedure-bank builder is unavailable")
    return _ext.AneLinearBankBuilder(sequence_length)


def ane_fused_bank_builder(sequence_length: int):
    """Incremental fused SwiGLU/down bank builder: add() converts one fp32
    gate/up/down triple at a time so the caller can release each staging
    array immediately (the issue #2781 recipe applied to fused banks)."""
    if not ane_available() or _ext is None or not hasattr(
        _ext, "AneFusedBankBuilder"
    ):
        raise RuntimeError("Private ANE procedure-bank builder is unavailable")
    return _ext.AneFusedBankBuilder(sequence_length)


def ane_affine_qmm_t(
    x: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    ane_model,
    bits: int,
    variant: int = 8,
    group_size: int = 128,
    profile_category: int = 1,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "ane_affine_qmm_t"):
        raise RuntimeError("ANE hybrid affine qmm native kernel is unavailable")
    return _ext.ane_affine_qmm_t(
        x,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        ane_model,
        bits,
        variant,
        group_size,
        profile_category,
    )


def ane_compile_fp16_linear(weight: mx.array, sequence_length: int):
    if not ane_available() or not hasattr(
        _ext, "ane_compile_fp16_linear"
    ):
        raise RuntimeError("Private ANE fp16 runtime is unavailable")
    return _ext.ane_compile_fp16_linear(weight, sequence_length)


def ane_swiglu_down_available() -> bool:
    return bool(
        ane_available()
        and hasattr(_ext, "ane_compile_swiglu_down")
        and hasattr(_ext, "ane_q4_swiglu_down_t")
    )


def ane_compile_swiglu_down(
    gate_weight: mx.array,
    up_weight: mx.array,
    down_weight: mx.array,
    sequence_length: int,
    ane_instance: int = 0,
):
    if not ane_swiglu_down_available():
        raise RuntimeError("Private ANE SwiGLU/down runtime is unavailable")
    return _ext.ane_compile_swiglu_down(
        gate_weight, up_weight, down_weight, sequence_length, ane_instance
    )


def ane_compile_swiglu_down_bank(
    gate_weights: list[mx.array],
    up_weights: list[mx.array],
    down_weights: list[mx.array],
    sequence_length: int,
    ane_instance: int,
):
    if _ext is None or not hasattr(_ext, "ane_compile_swiglu_down_bank"):
        raise RuntimeError("Private ANE SwiGLU/down bank compiler is unavailable")
    return _ext.ane_compile_swiglu_down_bank(
        gate_weights,
        up_weights,
        down_weights,
        sequence_length,
        ane_instance,
    )


def ane_q4_affine_qmm_t(
    x: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    ane_model,
    variant: int = 8,
    group_size: int = 128,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "ane_q4_affine_qmm_t"):
        raise RuntimeError("ANE hybrid q4 qmm native kernel is unavailable")
    return _ext.ane_q4_affine_qmm_t(
        x,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        ane_model,
        variant,
        group_size,
    )


def ane_q4_swiglu_t(
    x: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    ane_model,
    variant: int = 8,
    group_size: int = 128,
    up_suffix: tuple[mx.array, mx.array, mx.array] | None = None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "ane_q4_swiglu_t"):
        raise RuntimeError("ANE hybrid q4 SwiGLU native kernel is unavailable")
    return _ext.ane_q4_swiglu_t(
        x,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        ane_model,
        variant,
        group_size,
        **_up_suffix_kwargs(up_suffix),
    )


def ane_affine_swiglu_t(
    x: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    ane_model,
    bits: int,
    variant: int = 8,
    group_size: int = 128,
    up_suffix: tuple[mx.array, mx.array, mx.array] | None = None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "ane_affine_swiglu_t"):
        raise RuntimeError("ANE hybrid affine SwiGLU native kernel is unavailable")
    return _ext.ane_affine_swiglu_t(
        x,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        ane_model,
        bits,
        variant,
        group_size,
        **_up_suffix_kwargs(up_suffix),
    )


def ane_dual_affine_qmm_t(
    x: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    ane_model0,
    ane_model1,
    bits: int,
    variant: int = 8,
    group_size: int = 128,
    profile_category: int = 1,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "ane_dual_affine_qmm_t"):
        raise RuntimeError("Dual ANE hybrid affine qmm native kernel is unavailable")
    return _ext.ane_dual_affine_qmm_t(
        x,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        ane_model0,
        ane_model1,
        bits,
        variant,
        group_size,
        profile_category,
    )


def ane_dual_q4_swiglu_t(
    x: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    ane_model0,
    ane_model1,
    variant: int = 8,
    group_size: int = 128,
    up_suffix: tuple[mx.array, mx.array, mx.array] | None = None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "ane_dual_q4_swiglu_t"):
        raise RuntimeError("Dual ANE hybrid q4 SwiGLU native kernel is unavailable")
    return _ext.ane_dual_q4_swiglu_t(
        x,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        ane_model0,
        ane_model1,
        variant,
        group_size,
        **_up_suffix_kwargs(up_suffix),
    )


def ane_dual_affine_swiglu_t(
    x: mx.array,
    gpu_weight: mx.array,
    gpu_scales: mx.array,
    gpu_biases: mx.array,
    ane_model0,
    ane_model1,
    bits: int,
    variant: int = 8,
    group_size: int = 128,
    up_suffix: tuple[mx.array, mx.array, mx.array] | None = None,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "ane_dual_affine_swiglu_t"):
        raise RuntimeError(
            "Dual ANE hybrid affine SwiGLU native kernel is unavailable"
        )
    return _ext.ane_dual_affine_swiglu_t(
        x,
        gpu_weight,
        gpu_scales,
        gpu_biases,
        ane_model0,
        ane_model1,
        bits,
        variant,
        group_size,
        **_up_suffix_kwargs(up_suffix),
    )


def ane_q4_swiglu_down_t(
    x: mx.array,
    gpu_gate_up_weight: mx.array,
    gpu_gate_up_scales: mx.array,
    gpu_gate_up_biases: mx.array,
    gpu_down_weight: mx.array,
    gpu_down_scales: mx.array,
    gpu_down_biases: mx.array,
    ane_model,
    variant: int = 8,
    group_size: int = 128,
) -> mx.array:
    if not ane_swiglu_down_available():
        raise RuntimeError("ANE hybrid SwiGLU/down native kernel is unavailable")
    return _ext.ane_q4_swiglu_down_t(
        x,
        gpu_gate_up_weight,
        gpu_gate_up_scales,
        gpu_gate_up_biases,
        gpu_down_weight,
        gpu_down_scales,
        gpu_down_biases,
        ane_model,
        variant,
        group_size,
    )


def ane_dual_q4_swiglu_down_t(
    x: mx.array,
    gpu_gate_up_weight: mx.array,
    gpu_gate_up_scales: mx.array,
    gpu_gate_up_biases: mx.array,
    gpu_down_weight: mx.array,
    gpu_down_scales: mx.array,
    gpu_down_biases: mx.array,
    ane_model0,
    ane_model1,
    variant: int = 8,
    group_size: int = 128,
) -> mx.array:
    if _ext is None or not hasattr(_ext, "ane_dual_q4_swiglu_down_t"):
        raise RuntimeError("Dual-ANE hybrid SwiGLU/down native kernel is unavailable")
    return _ext.ane_dual_q4_swiglu_down_t(
        x,
        gpu_gate_up_weight,
        gpu_gate_up_scales,
        gpu_gate_up_biases,
        gpu_down_weight,
        gpu_down_scales,
        gpu_down_biases,
        ane_model0,
        ane_model1,
        variant,
        group_size,
    )


def ane_dual_cpu_fp16_q4_swiglu_down_t(
    x: mx.array,
    cpu_gate_up_weight: mx.array,
    cpu_down_weight: mx.array,
    gpu_gate_up_weight: mx.array,
    gpu_gate_up_scales: mx.array,
    gpu_gate_up_biases: mx.array,
    gpu_down_weight: mx.array,
    gpu_down_scales: mx.array,
    gpu_down_biases: mx.array,
    ane_model0,
    ane_model1,
    variant: int = 8,
    group_size: int = 128,
    cpu_threads: int = 0,
    cpu_shared_resource: bool = False,
) -> mx.array:
    if _ext is None or not hasattr(
        _ext, "ane_dual_cpu_fp16_q4_swiglu_down_t"
    ):
        raise RuntimeError(
            "Dual-ANE/CPU fused SwiGLU/down native kernel is unavailable"
        )
    return _ext.ane_dual_cpu_fp16_q4_swiglu_down_t(
        x,
        cpu_gate_up_weight,
        cpu_down_weight,
        gpu_gate_up_weight,
        gpu_gate_up_scales,
        gpu_gate_up_biases,
        gpu_down_weight,
        gpu_down_scales,
        gpu_down_biases,
        ane_model0,
        ane_model1,
        variant,
        group_size,
        cpu_threads,
        cpu_shared_resource,
    )
