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


def _pair_interleave_fused(gate, up, rows=512):
    """Build the routed-expert pair-interleaved fused plane from stacked
    gate/up weight tensors (each [E, rows, K/8] uint32 or [E, rows, K/16]
    uint8 scales): per expert, 32-row [gate; up] pairs —
    plane[p*64 + r] = gate[p*32 + r], plane[p*64 + 32 + r] = up[p*32 + r].
    Mirrors the challenge's fused routed plane layout.
    """
    E = gate.shape[0]
    pairs = rows // 32
    flat = mx.reshape(gate, (E, pairs, 32) + gate.shape[2:])
    flat_u = mx.reshape(up, (E, pairs, 32) + up.shape[2:])
    inter = mx.stack([flat, flat_u], axis=2)  # [E, pairs, 2, 32, ...]
    return mx.reshape(inter, (E, rows * 2) + gate.shape[2:])


def routed_nvfp4_swiglu_qmv(
    input_x: mx.array,
    fused_weight: mx.array,
    fused_scales: mx.array,
    indices: mx.array,
    stream=None,
) -> mx.array:
    """Routed-expert fused gate/up NVFP4 QMV with in-kernel SwiGLU
    (challenge lagunaRoutedSwiGLUQMVKernel).

    Parameters
    ----------
    input_x      : [K] bf16                — routed-expert input (K = 2048)
    fused_weight : [E, 2N, K/8] uint32     — per-expert pair-interleaved
                                             [gate; up] planes (N = 512)
    fused_scales : [E, 2N, K/16] uint8     — E4M3 group-16 scales
    indices      : [R] uint32              — top-R routed expert ids (R = 8)

    Returns [R*N] bf16 per-slot silu(gate) * up.
    """
    if _ext is not None and has_symbol("routed_nvfp4_swiglu_qmv"):
        return _ext.routed_nvfp4_swiglu_qmv(
            input_x, fused_weight, fused_scales, indices, stream=stream)
    # Fallback: un-interleave the pair plane into [gate; up] rows and use
    # the stock nvfp4 matmul + swiglu per routed expert.
    N = fused_weight.shape[1] // 2
    pairs = N // 32
    fw = fused_weight.reshape(indices.shape[0] * 0 + fused_weight.shape[0]
                              if False else fused_weight.shape[0],
                              pairs, 2, 32, fused_weight.shape[-1])
    fs = fused_scales.reshape(fused_scales.shape[0], pairs, 2, 32,
                              fused_scales.shape[-1])
    gate_w = mx.concatenate([fw[:, p, 0] for p in range(pairs)], axis=1)
    up_w = mx.concatenate([fw[:, p, 1] for p in range(pairs)], axis=1)
    gate_s = mx.concatenate([fs[:, p, 0] for p in range(pairs)], axis=1)
    up_s = mx.concatenate([fs[:, p, 1] for p in range(pairs)], axis=1)
    outs = []
    for slot in range(indices.shape[0]):
        e = int(indices[slot])
        g = mx.quantized_matmul(
            input_x[None, :], gate_w[e], scales=gate_s[e], transpose=True,
            group_size=16, bits=4, mode="nvfp4", stream=stream)
        u = mx.quantized_matmul(
            input_x[None, :], up_w[e], scales=up_s[e], transpose=True,
            group_size=16, bits=4, mode="nvfp4", stream=stream)
        outs.append(_swiglu(g, u)[0])
    return mx.concatenate(outs)


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
