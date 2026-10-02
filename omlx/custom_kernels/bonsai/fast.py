"""Bonsai 1-bit / 2-bit decode kernel dispatch.

Public API
----------
has_native()            -> bool     native C++ extension available
is_nax_available()      -> bool     M5+ tensor-unit available (gen >= 18)

bonsai_q1_affine_qmv(x, w, scales, biases, stream=None) -> mx.array
    1-bit affine decode (M = 1).  Falls back to mx.quantized_matmul when
    the native extension is unavailable.

bonsai_q2_affine_qmv(x, w, scales, biases, stream=None) -> mx.array
    2-bit affine decode (M = 1).  Falls back to mx.quantized_matmul when
    the native extension is unavailable.

bonsai_qmv_wide(x, w, scales, biases, bits, stream=None) -> mx.array
    Small-batch affine decode (M = 1..5, bits = 1 or 2).  Falls back to
    mx.quantized_matmul.  Routing: 1/2-bit M>=2 use qmv_wide on gen-15+
    for weight reuse; 1-bit M=1 uses qmv_fast; 2-bit M=1 uses qmv_fast.

spec_decode_verify(draft_tokens, target_logits, stream=None)
    -> (n_accepted [B], committed [B, K+1])
    Fused greedy speculative-decode verify.  Uses the native Metal kernel when
    available; otherwise falls back to a pure-mlx op composition.
"""

from __future__ import annotations

import importlib
import logging
import re
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
    # Use absolute import to avoid circular import when __init__.py imports fast
    # before _ext is registered in sys.modules.
    _ext = importlib.import_module("omlx.custom_kernels.bonsai._ext")
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

# ---------------------------------------------------------------------------
# NAX (M5 tensor-unit) detection
# ---------------------------------------------------------------------------

_NAX_ARCH_RE = re.compile(r"applegpu_g(\d+)([a-z])")
_nax_available_cache: bool | None = None


def is_nax_available() -> bool:
    """True when the GPU is M5-class or later (gen >= 18, NAX tensor unit).

    Hardware capability probe only (informational).  This is NOT the
    routing predicate for stock NAX dispatch — that lives in
    omlx.custom_kernels.nax and additionally checks the OMLX_NAX env
    override, the installed metallib, and the macOS version.  Bonsai
    kernel routing uses _arch_gen(), not this.
    """
    global _nax_available_cache
    if _nax_available_cache is not None:
        return _nax_available_cache

    # Prefer the native extension's mirror of metal::is_nax_available().
    if _ext is not None and hasattr(_ext, "is_nax_available"):
        _nax_available_cache = bool(_ext.is_nax_available())
        return _nax_available_cache

    # Fallback: parse device_info arch string.
    try:
        arch = mx.device_info().get("architecture", "")
        m = _NAX_ARCH_RE.search(arch.lower())
        if m:
            gen = int(m.group(1))
            # gen-17 (M5-class applegpu_g17s) computes wrong results with NAX
            # qmm/gemm kernels; require gen >= 18.
            _nax_available_cache = gen >= 18
            return _nax_available_cache
    except Exception:
        pass

    _nax_available_cache = False
    return False


# ---------------------------------------------------------------------------
# Architecture generation (for qmv_wide routing)
# ---------------------------------------------------------------------------

_arch_gen_cache: int | None = None


def _arch_gen() -> int:
    global _arch_gen_cache
    if _arch_gen_cache is not None:
        return _arch_gen_cache
    try:
        arch = mx.device_info().get("architecture", "")
        m = _NAX_ARCH_RE.search(arch.lower())
        _arch_gen_cache = int(m.group(1)) if m else 0
    except Exception:
        _arch_gen_cache = 0
    return _arch_gen_cache


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
# 1-bit single-row decode  (qmv_fast)
# ---------------------------------------------------------------------------


def bonsai_q2_affine_qmv(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    stream=None,
) -> mx.array:
    """2-bit affine quantized matrix-vector multiply (decode, M=1)."""
    if _ext is not None and has_symbol("bonsai_q2_affine_qmv"):
        return _ext.bonsai_q2_affine_qmv(x, w, scales, biases, stream=stream)
    return mx.quantized_matmul(
        x, w, scales=scales, biases=biases, transpose=True,
        group_size=_infer_group_size(w, scales, 2), bits=2, stream=stream
    )


def _dequant_1bit(w, scales, biases, dtype, group_size=None):
    """Dequantize 1-bit affine weights (N, K//32 uint32) to (N, K) dtype.

    Stock mlx ships no affine_dequantize for bits=1, so prefill and direct
    quantized_matmul callers materialize the weight explicitly.
    """
    N, K32 = w.shape
    K = K32 * 32
    n_groups = scales.shape[-1]
    gs = K // n_groups
    shifts = mx.arange(32, dtype=mx.uint32)
    w_flat = ((w[:, :, None] >> shifts) & 0x1).astype(dtype).reshape(N, K)
    w_fp = w_flat * mx.repeat(scales.astype(dtype), gs, axis=-1)
    if biases is not None:
        w_fp = w_fp + mx.repeat(biases.astype(dtype), gs, axis=-1)
    return w_fp


def bonsai_q1_affine_qmv(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    stream=None,
) -> mx.array:
    """1-bit affine quantized matrix-vector multiply (decode, M=1).

    Parameters
    ----------
    x      : [..., K] activations (float16 or bfloat16)
    w      : [..., N, K//8] packed 1-bit weights
    scales : [..., N, K//group_size] scale factors
    biases : [..., N, K//group_size] bias/zero offsets

    Returns [... N] output.
    """
    if _ext is not None and has_symbol("bonsai_q1_affine_qmv"):
        return _ext.bonsai_q1_affine_qmv(x, w, scales, biases, stream=stream)
    # Fallback: stock mlx quantized_matmul (correct, slower for 1-bit)
    return mx.quantized_matmul(
        x, w, scales=scales, biases=biases, transpose=True,
        group_size=_infer_group_size(w, scales, 1), bits=1, stream=stream
    )


# ---------------------------------------------------------------------------
# Small-batch decode (qmv_wide, M = 2..5)
# ---------------------------------------------------------------------------


def _infer_group_size(w: mx.array, scales: mx.array, bits: int) -> int:
    """Derive group_size from packed weight / scale shapes.

    Bonsai 1-bit weights use uint8 packing (8 values per byte); the MLX
    standard format uses uint32 packing (32//bits values per element).
    Detect the format from w.dtype to compute K correctly for both layouts.
    """
    if w.dtype == mx.uint8:
        pack = 8   # Bonsai 1-bit uint8 format
    else:
        pack = 32 // bits  # MLX standard uint32 format
    K = w.shape[-1] * pack
    n_groups = scales.shape[-1]
    if n_groups <= 0 or K <= 0:
        raise ValueError(
            f"_infer_group_size: invalid shapes — weight {w.shape} (dtype "
            f"{w.dtype}), scale {scales.shape}, bits={bits}"
        )
    if K % n_groups != 0:
        raise ValueError(
            f"_infer_group_size: K={K} not divisible by n_groups={n_groups} "
            f"(weight {w.shape} dtype {w.dtype}, scale {scales.shape}, bits={bits})"
        )
    return K // n_groups


def _use_qmv_wide(bits: int, M: int) -> bool:
    """True when qmv_wide beats per-row qmv for these batch/bit settings.

    2-bit at M >= 3 on gen-15+: qmv_wide amortises the weight stream across
    all M vectors (1 read vs M reads), yielding 1.3–1.5× on large projections
    (benchmarked: gate/up/down_proj M=5 → 71→104 GB/s on g16s).

    1-bit at M >= 3 on gen-15+: qmv_wide is also instantiated for 1-bit;
    weight reuse benefit depends on whether the L2 cache absorbs the weight
    tensor at M>1 for a given model size.
    """
    if bits not in (1, 2) or M < 3:
        return False
    return _arch_gen() >= 15


def bonsai_qmv_wide(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    bits: int,
    stream=None,
) -> mx.array:
    """Small-batch affine quantized matmul (decode, M = 1..5, bits = 1 or 2).

    Routing:
      - 1-bit M=1:  qmv_fast  (no weight reuse possible)
      - 1-bit M>=2: qmv_wide  (weight loaded once, multiplied with all M vecs)
      - 2-bit M=1:  stock mlx quantized_matmul
      - 2-bit M>=2: qmv_wide  (same weight-reuse benefit)
      - gen < 15:   fall back to qmv_fast (1-bit) or stock mlx (2-bit)
    """
    M = x.shape[-2] if x.ndim >= 2 else 1

    if bits == 1:
        if _use_qmv_wide(bits, M) and _ext is not None and has_symbol("bonsai_q1_affine_qmv_wide"):
            return _ext.bonsai_q1_affine_qmv_wide(x, w, scales, biases, stream=stream)
        if _ext is not None and has_symbol("bonsai_q1_affine_qmv"):
            return _ext.bonsai_q1_affine_qmv(x, w, scales, biases, stream=stream)
    else:  # bits == 2
        if _use_qmv_wide(bits, M) and _ext is not None and has_symbol("bonsai_q2_affine_qmv_wide"):
            return _ext.bonsai_q2_affine_qmv_wide(x, w, scales, biases, stream=stream)
        # M=1 (or no wide kernel): use 2-bit qmv_fast instead of falling back to stock mlx
        if _ext is not None and has_symbol("bonsai_q2_affine_qmv"):
            return _ext.bonsai_q2_affine_qmv(x, w, scales, biases, stream=stream)

    group_size = _infer_group_size(w, scales, bits)
    return mx.quantized_matmul(
        x, w, scales=scales, biases=biases, transpose=True,
        group_size=group_size, bits=bits, stream=stream
    )


def bonsai_q1_affine_qmv_wide(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    stream=None,
) -> mx.array:
    """1-bit affine wide qmv (M=2..5 weight-reuse path)."""
    if _ext is not None and has_symbol("bonsai_q1_affine_qmv_wide"):
        return _ext.bonsai_q1_affine_qmv_wide(x, w, scales, biases, stream=stream)
    return bonsai_q1_affine_qmv(x, w, scales, biases, stream=stream)


def bonsai_q2_affine_qmv_wide(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    stream=None,
) -> mx.array:
    """2-bit affine wide qmv (M=2..5 weight-reuse path)."""
    if _ext is not None and has_symbol("bonsai_q2_affine_qmv_wide"):
        return _ext.bonsai_q2_affine_qmv_wide(x, w, scales, biases, stream=stream)
    return bonsai_q2_affine_qmv(x, w, scales, biases, stream=stream)


# ---------------------------------------------------------------------------
# Symmetric decode variants (identity I-B: bias = -scale*ratio)
# ---------------------------------------------------------------------------


def bonsai_q1_affine_qmv_sym(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    stream=None,
) -> mx.array:
    """1-bit symmetric qmv: skips biases DRAM load (I-B: bias = -scale/2)."""
    if _ext is not None and has_symbol("bonsai_q1_affine_qmv_sym"):
        return _ext.bonsai_q1_affine_qmv_sym(x, w, scales, biases, stream=stream)
    return bonsai_q1_affine_qmv(x, w, scales, biases, stream=stream)


def bonsai_q2_affine_qmv_sym(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    stream=None,
) -> mx.array:
    """2-bit symmetric qmv: skips biases DRAM load (I-B: bias = -scale)."""
    if _ext is not None and has_symbol("bonsai_q2_affine_qmv_sym"):
        return _ext.bonsai_q2_affine_qmv_sym(x, w, scales, biases, stream=stream)
    return bonsai_q2_affine_qmv(x, w, scales, biases, stream=stream)


def bonsai_q1_affine_qmv_wide_sym(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    stream=None,
) -> mx.array:
    """1-bit symmetric wide qmv: skips biases DRAM load (I-B)."""
    if _ext is not None and has_symbol("bonsai_q1_affine_qmv_wide_sym"):
        return _ext.bonsai_q1_affine_qmv_wide_sym(x, w, scales, biases, stream=stream)
    return bonsai_q1_affine_qmv(x, w, scales, biases, stream=stream)


def bonsai_q2_affine_qmv_wide_sym(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    stream=None,
) -> mx.array:
    """2-bit symmetric wide qmv: skips biases DRAM load (I-B)."""
    if _ext is not None and has_symbol("bonsai_q2_affine_qmv_wide_sym"):
        return _ext.bonsai_q2_affine_qmv_wide_sym(x, w, scales, biases, stream=stream)
    return bonsai_q2_affine_qmv_wide(x, w, scales, biases, stream=stream)


# ---------------------------------------------------------------------------
# t5: base-3 ternary packing (Identity I-D, ~1.585 bpw)
# ---------------------------------------------------------------------------
# Weight tensor: uint8, shape (N, n_groups * bytes_per_group)
#   bytes_per_group = 26 for group_size=128, 13 for group_size=64
# No bias tensor (always symmetric: dequant = scale * (q - 1), q ∈ {0,1,2})
# ---------------------------------------------------------------------------


def bonsai_t5_qmv(
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
    if _ext is not None and has_symbol("bonsai_t5_qmv"):
        return _ext.bonsai_t5_qmv(x, w, scales, stream=stream)
    raise RuntimeError(
        "bonsai_t5_qmv: native extension unavailable. "
        "Rebuild the bonsai extension to use t5-format weights."
    )


def bonsai_t5_qmv_wide(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    stream=None,
) -> mx.array:
    """t5 base-3 ternary wide decode (M=2..5, I-D + I-C)."""
    if _ext is not None and has_symbol("bonsai_t5_qmv_wide"):
        return _ext.bonsai_t5_qmv_wide(x, w, scales, stream=stream)
    return bonsai_t5_qmv(x, w, scales, stream=stream)


def bonsai_t5_qmm(
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
    if _ext is not None and has_symbol("bonsai_t5_qmm"):
        return _ext.bonsai_t5_qmm(x, w, scales, stream=stream)
    raise RuntimeError(
        "bonsai_t5_qmm: native extension unavailable. "
        "Rebuild the bonsai extension."
    )


# ---------------------------------------------------------------------------
# spec_decode_verify
# ---------------------------------------------------------------------------


def spec_decode_verify(
    draft_tokens: mx.array,
    target_logits: mx.array,
    stream=None,
) -> tuple[mx.array, mx.array]:
    """Greedy speculative-decoding verify.

    Parameters
    ----------
    draft_tokens  : [B, K] int32   — K drafted token ids
    target_logits : [B, K+1, V] float — target-model logits over last+draft

    Returns
    -------
    n_accepted : [B] int32        — accepted prefix length (0..K)
    committed  : [B, K+1] int32   — accepted draft prefix + corrected token
    """
    s = stream

    dft = draft_tokens.astype(mx.int32, stream=s) if draft_tokens.dtype != mx.int32 else draft_tokens
    tgt = mx.argmax(target_logits, axis=-1, stream=s)  # [B, K+1]
    tgt = tgt.astype(mx.int32, stream=s) if tgt.dtype != mx.int32 else tgt

    if _ext is not None and has_symbol("bonsai_spec_decode_verify"):
        return _ext.bonsai_spec_decode_verify(dft, tgt, stream=s)

    # Pure-mlx fallback — exactly the oracle from the Bonsai MLX fork
    # (mlx/fast.cpp::spec_decode_verify fallback lambda).

    B, K = dft.shape[0], dft.shape[1]

    t_pref = tgt[:, :K]                                         # [B, K]
    mism = mx.not_equal(dft, t_pref, stream=s)                  # [B, K] bool
    j = mx.broadcast_to(
        mx.reshape(mx.arange(K, dtype=mx.int32, stream=s), (1, K), stream=s),
        (B, K), stream=s
    )
    n_acc = mx.min(
        mx.where(mism, j, mx.full((B, K), K, mx.int32, stream=s), stream=s),
        axis=1, keepdims=False, stream=s
    )  # [B]

    n_acc2 = mx.reshape(n_acc, (B, 1), stream=s)                # [B, 1]
    corrected = mx.take_along_axis(tgt, n_acc2, axis=1, stream=s)  # [B, 1]
    j1 = mx.broadcast_to(
        mx.reshape(mx.arange(K + 1, dtype=mx.int32, stream=s), (1, K + 1), stream=s),
        (B, K + 1), stream=s
    )
    nacc_b = mx.broadcast_to(n_acc2, (B, K + 1), stream=s)
    d_ext = mx.concatenate(
        [dft, mx.zeros((B, 1), dtype=mx.int32, stream=s)], axis=1, stream=s
    )
    corr_b = mx.broadcast_to(corrected, (B, K + 1), stream=s)
    committed = mx.where(
        mx.less(j1, nacc_b, stream=s),
        d_ext,
        mx.where(
            mx.equal(j1, nacc_b, stream=s),
            corr_b,
            mx.zeros((B, K + 1), dtype=mx.int32, stream=s),
            stream=s
        ),
        stream=s
    )
    return n_acc, committed
