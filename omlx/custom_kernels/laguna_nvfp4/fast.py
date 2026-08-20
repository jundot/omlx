"""NVFP4 (E4M3) decode kernels for oMLX, ported from the
Layr-Labs/mlxfast-challenge Laguna runtime.

Public API
----------
has_native()            -> bool     native C++ extension available

shared_nvfp4_swiglu_qmv(x, w, scales, stream=None) -> mx.array
    Shared-expert fused gate/up NVFP4 QMV with in-kernel SwiGLU: one kernel
    computes silu(gate) * up over the fused [gate; up] NVFP4 plane. Falls
    back to stock mx.quantized_matmul(mode="nvfp4") + swiglu when the native
    extension is unavailable.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

import mlx.core as mx

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _detach_import_error(exc: Exception) -> Exception:
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None
    return exc


try:
    _ext = importlib.import_module("omlx.custom_kernels.laguna_nvfp4._ext")
except Exception as exc:  # pragma: no cover - depends on local native build
    _ext = None
    _IMPORT_ERROR: Exception | None = _detach_import_error(exc)
else:
    _IMPORT_ERROR = None


def _verify_abi(ext, import_error):
    """Disable native symbols when the nanobind ABI tag does not match mlx."""
    if ext is None:
        return ext, import_error
    probe = getattr(ext, "abi_probe", None)
    if probe is None:
        return ext, import_error
    try:
        probe(mx.zeros((1,)))
    except TypeError as exc:
        logger.warning(
            "%s: native kernels disabled — nanobind ABI mismatch "
            "(rebuild against the installed mlx).",
            __name__,
        )
        return None, _detach_import_error(exc)
    return ext, import_error


_ext, _IMPORT_ERROR = _verify_abi(_ext, _IMPORT_ERROR)


def has_native() -> bool:
    """True when the compiled C++ extension is loaded and ABI-verified."""
    return _ext is not None


def is_native_available() -> bool:
    """Alias for has_native() — matches omlx kernel package convention."""
    return has_native()


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def has_symbol(name: str) -> bool:
    return _ext is not None and hasattr(_ext, name)


def _swiglu(gate: mx.array, up: mx.array) -> mx.array:
    gate32 = gate.astype(mx.float32)
    return (gate32 * mx.sigmoid(gate32) * up.astype(mx.float32)).astype(
        gate.dtype
    )


def shared_nvfp4_swiglu_qmv(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    stream=None,
) -> mx.array:
    """Shared-expert fused gate/up NVFP4 QMV with in-kernel SwiGLU.

    Parameters
    ----------
    x      : [K] bf16                — shared-expert input (K = 2048)
    w      : [2N, K*4/32] uint8      — fused [gate; up] NVFP4 codes (N = 512)
    scales : [2N, K/16] uint8        — E4M3 group-16 scales

    Returns [N] bf16 silu(gate) * up.
    """
    if _ext is not None and has_symbol("shared_nvfp4_swiglu_qmv"):
        return _ext.shared_nvfp4_swiglu_qmv(x, w, scales, stream=stream)
    # Fallback: stock mlx nvfp4 matmul over the fused gate/up plane + swiglu.
    gate_up = mx.quantized_matmul(
        x[None, :],
        w,
        scales=scales,
        transpose=True,
        group_size=16,
        bits=4,
        mode="nvfp4",
        stream=stream,
    )  # [1, 2N]
    split = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :split], gate_up[..., split:]
    return _swiglu(gate, up).squeeze(0)


def shared_nvfp4_down_residual(
    activated: mx.array,
    down_weight: mx.array,
    down_scales: mx.array,
    routed: mx.array,
    residual: mx.array,
    stream=None,
) -> mx.array:
    """Shared-expert down_proj with routed + residual adds fused in one
    kernel (challenge lagunaSharedDownResidualKernel).

    Parameters
    ----------
    activated   : [K2] bf16              — swiglu output (K2 = 512)
    down_weight : [N, K2/2] uint8        — NVFP4 down_proj codes (N = 2048)
    down_scales : [N, K2/16] uint8       — E4M3 group-16 scales
    routed      : [N] bf16               — routed-expert output
    residual    : [N] bf16               — decoder residual

    Returns [N] bf16 residual + (routed + shared).
    """
    if _ext is not None and has_symbol("shared_nvfp4_down_residual"):
        return _ext.shared_nvfp4_down_residual(
            activated, down_weight, down_scales, routed, residual,
            stream=stream,
        )
    # Fallback: stock nvfp4 matmul + the two adds.
    shared = mx.quantized_matmul(
        activated[None, :], down_weight, scales=down_scales,
        transpose=True, group_size=16, bits=4, mode="nvfp4",
        stream=stream,
    ).squeeze(0)  # [N]
    return (residual + (routed + shared)).astype(mx.bfloat16)
