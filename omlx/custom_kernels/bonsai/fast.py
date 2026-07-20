"""Bonsai 1-bit / 2-bit decode kernel dispatch.

Public API
----------
has_native()            -> bool     native C++ extension available
is_nax_available()      -> bool     M5+ tensor-unit available (gen >= 18)

bonsai_q1_affine_qmv(x, w, scales, biases, stream=None) -> mx.array
    1-bit affine decode (M = 1).  Falls back to mx.quantized_matmul when
    the native extension is unavailable.

bonsai_qmv_wide(x, w, scales, biases, bits, stream=None) -> mx.array
    Small-batch affine decode (M = 2..5, bits = 1 or 2).  Falls back to
    mx.quantized_matmul.  Routing: 1-bit always uses qmv_fast; 2-bit uses
    qmv_wide only at M >= 3 on gen-15+.

spec_decode_verify(draft_tokens, target_logits, stream=None)
    -> (n_accepted [B], committed [B, K+1])
    Fused greedy speculative-decode verify.  Uses the native Metal kernel when
    available; otherwise falls back to a pure-mlx op composition.
"""

from __future__ import annotations

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
    from . import _ext
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
    """True when the GPU is M5-class or later (gen >= 18, NAX tensor unit)."""
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
    """Derive group_size from packed weight / scale shapes."""
    try:
        K = w.shape[-1] * (32 // bits)
        n_groups = scales.shape[-1]
        return K // n_groups
    except Exception:
        return 64


def _use_qmv_wide(bits: int, M: int) -> bool:
    """True when qmv_wide beats per-row qmv for these batch/bit settings.

    Mirrors the dispatch logic in the Bonsai MLX fork
    (mlx/backend/metal/quantized.cpp::use_qmv_wide):
      - fp modes: always route to qmv_wide
      - affine 1-bit: per-row qmv is faster (weight traffic is tiny)
      - affine 2-bit: break-even at M=3; qmv_wide wins for M >= 3 on gen-15+
    """
    if bits == 1:
        return False
    if bits == 2 and M < 3:
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
    """Small-batch affine quantized matmul (decode, M = 2..5, bits = 1 or 2).

    Routes to qmv_wide for 2-bit at M >= 3 on gen-15+; otherwise falls back
    to bonsai_q1_affine_qmv (1-bit) or stock mlx (2-bit narrow).
    """
    M = x.shape[-2] if x.ndim >= 2 else 1

    if bits == 1:
        # 1-bit is always faster with the per-row qmv_fast kernel.
        if _ext is not None and has_symbol("bonsai_q1_affine_qmv"):
            return _ext.bonsai_q1_affine_qmv(x, w, scales, biases, stream=stream)
    elif _use_qmv_wide(bits, M):
        if _ext is not None and has_symbol("bonsai_q2_affine_qmv_wide"):
            return _ext.bonsai_q2_affine_qmv_wide(x, w, scales, biases, stream=stream)

    group_size = _infer_group_size(w, scales, bits)
    return mx.quantized_matmul(
        x, w, scales=scales, biases=biases, transpose=True,
        group_size=group_size, bits=bits, stream=stream
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

    if _ext is not None and has_symbol("bonsai_spec_decode_verify"):
        return _ext.bonsai_spec_decode_verify(draft_tokens, target_logits, stream=s)

    # Pure-mlx fallback — exactly the oracle from the Bonsai MLX fork
    # (mlx/fast.cpp::spec_decode_verify fallback lambda).
    dft = draft_tokens.astype(mx.int32, stream=s) if draft_tokens.dtype != mx.int32 else draft_tokens
    tgt = mx.argmax(target_logits, axis=-1, stream=s)  # [B, K+1]
    tgt = tgt.astype(mx.int32, stream=s) if tgt.dtype != mx.int32 else tgt

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
