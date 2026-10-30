"""Bonsai t5 (base-3 ternary) kernels.

Public API
----------
has_native()                      -> bool  native C++ extension available
t5_qmv(x, w, scales)              -> M = 1 decode
t5_qmv_wide(x, w, scales)         -> M = 2..16 decode
t5_qmm(x, w, scales)              -> prefill GEMM

Stock MLX serves the uint32 1-bit and 2-bit affine formats.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .._loader import load_extension

_ext, _IMPORT_ERROR = load_extension(__package__, Path(__file__).parent)


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# t5: base-3 ternary packing (Identity I-D, ~1.585 bpw)
# ---------------------------------------------------------------------------
# Weight tensor: uint8, shape (N, n_groups * bytes_per_group)
#   bytes_per_group = 26 for group_size=128, 13 for group_size=64
# No bias tensor (always symmetric: dequant = scale * (q - 1), q ∈ {0,1,2})
# ---------------------------------------------------------------------------


def t5_qmv(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    stream=None,
) -> mx.array:
    """t5 base-3 ternary decode (M=1, ~1.585 bpw, I-D).

    Parameters
    ----------
    x      : [..., K]                   activations (float16 or bfloat16)
    w      : [..., N, n_groups*bpg]     uint8 t5 weight bytes
    scales : [..., N, n_groups]         scale per group (float16 or bfloat16)
    """
    if _ext is not None and has_symbol("t5_qmv"):
        return _ext.t5_qmv(x, w, scales, stream=stream)
    raise RuntimeError(
        "t5_qmv: native extension unavailable. "
        "Rebuild the bonsai extension to use t5-format weights."
    )


def t5_qmv_wide(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    stream=None,
) -> mx.array:
    """t5 base-3 ternary wide decode (M=2..5, I-D + I-C)."""
    if _ext is not None and has_symbol("t5_qmv_wide"):
        return _ext.t5_qmv_wide(x, w, scales, stream=stream)
    return t5_qmv(x, w, scales, stream=stream)


def t5_qmm(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    stream=None,
) -> mx.array:
    """t5 MMA GEMM for prefill (Identity I-M).

    Fused dequant + simdgroup matmul. Reads each t5 weight byte exactly once
    per matmul, without materialising an intermediate float weight matrix.

    Parameters
    ----------
    x      : [M, K]                  activations (float16 or bfloat16)
    w      : [N, n_groups*bpg]       uint8 t5 weight bytes
    scales : [N, n_groups]           scale per group
    Returns [M, N].
    """
    if _ext is not None and has_symbol("t5_qmm"):
        return _ext.t5_qmm(x, w, scales, stream=stream)
    raise RuntimeError(
        "t5_qmm: native extension unavailable. "
        "Rebuild the bonsai extension."
    )
