"""Bonsai 1-bit / 2-bit QuantizedLinear decode patch.

Intercepts ``QuantizedLinear.__call__`` for layers whose weight tensor is
1-bit or 2-bit affine-quantized and routes them through the Bonsai fast
decode kernels (qmv_fast for 1-bit, qmv_wide for 2-bit small-batch).

Activation condition
--------------------
Only active when:
  * ``bits`` in {1, 2}  and  ``mode == "affine"``
  * The input batch dimension M is in the decode regime (M <= 5)
  * The native bonsai extension is available (falls back silently otherwise)

Usage
-----
Call ``apply_bonsai_qmv_patch()`` once after model load.  It monkey-patches
``mlx.nn.QuantizedLinear`` globally, so all matching layers in the loaded
model are accelerated automatically.

Call ``remove_bonsai_qmv_patch()`` to restore the original implementation.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from omlx.custom_kernels.bonsai.fast import (
    bonsai_q1_affine_qmv,
    bonsai_q2_affine_qmv,
    bonsai_q1_affine_qmv_sym,
    bonsai_q2_affine_qmv_sym,
    bonsai_q1_affine_qmv_wide_sym,
    bonsai_q2_affine_qmv_wide_sym,
    bonsai_qmv_wide,
    bonsai_t5_qmv,
    bonsai_t5_qmv_wide,
    has_native,
    _use_qmv_wide,
)

logger = logging.getLogger(__name__)

_original_quantized_linear_call: Any = None
_patch_active = False

# Maximum input batch size routed through fast decode kernels.
# Above this threshold the model is prefilling — use stock mlx qmm_t instead.
_MAX_DECODE_M = 5

# t5 prefill threshold: qmv_wide re-reads weights ceil(M/5) times, one per
# threadgroup tile in the M dimension.  Above this M, dequantize to float16
# once and use MLX's optimised matmul instead (reads weights exactly twice:
# once for dequant, once for matmul — independent of M).
_T5_PREFILL_THRESHOLD = 16


def _get_cached_scales_biases(
    self: nn.QuantizedLinear, dtype: Any
) -> tuple[mx.array, mx.array]:
    """Return scales/biases cast to `dtype`, caching the result on the layer."""
    cache_attr = "_bonsai_sb_cache"
    cache = getattr(self, cache_attr, None)
    if cache is None or cache[0] != dtype:
        sc = self.scales.astype(dtype)
        bi = self.biases.astype(dtype)
        mx.eval(sc, bi)
        object.__setattr__(self, cache_attr, (dtype, sc, bi))
    _, sc, bi = getattr(self, cache_attr)
    return sc, bi


def _is_symmetric(self: nn.QuantizedLinear, bits: int) -> bool:
    """Return True if biases == -scales * ratio (identity I-B), cached per layer.

    1-bit: ratio = 0.5  (bias = -scale/2)
    2-bit: ratio = 1.0  (bias = -scale)

    Evaluated once on the first call; result is cached as _bonsai_sym_cache.
    """
    cache_attr = "_bonsai_sym_cache"
    cached = getattr(self, cache_attr, None)
    if cached is not None:
        return cached
    if bits not in (1, 2):
        object.__setattr__(self, cache_attr, False)
        return False
    ratio = 0.5 if bits == 1 else 1.0
    try:
        result = bool(mx.allclose(self.biases, -self.scales * ratio, atol=1e-4).item())
    except Exception:
        result = False
    object.__setattr__(self, cache_attr, result)
    return result


def _is_t5_format(self: nn.QuantizedLinear) -> bool:
    """Return True if the weight tensor is in t5 (base-3 ternary) format.

    t5 weights are stored as uint8 with bytes_per_group ∈ {13, 26}:
      13 bytes/group → group_size=64  (13×5=65; 1 padding trit)
      26 bytes/group → group_size=128 (26×5=130; 2 padding trits)

    Evaluated once on the first call; result cached as _bonsai_t5_cache.
    """
    cache_attr = "_bonsai_t5_cache"
    cached = getattr(self, cache_attr, None)
    if cached is not None:
        return cached
    w = self.weight
    if w.dtype != mx.uint8:
        object.__setattr__(self, cache_attr, False)
        return False
    sc = getattr(self, "scales", None)
    if sc is None or sc.shape[-1] == 0:
        object.__setattr__(self, cache_attr, False)
        return False
    n_groups = sc.shape[-1]
    w_cols   = w.shape[-1]
    if n_groups <= 0 or w_cols % n_groups != 0:
        object.__setattr__(self, cache_attr, False)
        return False
    bpg = w_cols // n_groups  # bytes per group
    result = bpg in (13, 26)
    object.__setattr__(self, cache_attr, result)
    return result


def _t5_dequant_matmul(self: nn.QuantizedLinear, x: mx.array) -> mx.array:
    """Prefill path for t5 weights: decode trits → dequantize → matmul.

    mlx's quantized_matmul kernel requires uint32 weights; for t5 (uint8) we
    dequantize to float and fall back to a regular matmul.  Called only when
    M > _MAX_DECODE_M (batch prefill); latency is dominated by the matmul,
    not the decode.
    """
    w = self.weight      # uint8, (N, n_groups * bpg)
    scales = self.scales  # (N, n_groups)
    N = w.shape[0]
    n_groups = scales.shape[-1]
    bpg = w.shape[1] // n_groups
    group_size = 64 if bpg == 13 else 128
    K = n_groups * group_size

    # Decode base-3: extract 5 trits per byte via repeated mod-3
    v = w.reshape(N, n_groups, bpg).astype(mx.uint32)
    trit_parts = []
    for _ in range(5):
        trit_parts.append(v % 3)
        v = v // 3
    # (N, n_groups, bpg, 5) → (N, n_groups, bpg*5) → trim padding
    trits = mx.stack(trit_parts, axis=-1).reshape(N, n_groups, bpg * 5)
    trits = trits[:, :, :group_size]  # (N, n_groups, group_size)

    # Dequantize: (trit - 1) * scale → {-scale, 0, +scale}
    dq = (trits.astype(x.dtype) - 1.0) * scales[..., None].astype(x.dtype)
    weight_fp = dq.reshape(N, K)  # (N, K)

    out = x @ weight_fp.T
    linear_bias = getattr(self, "bias", None)
    if linear_bias is not None:
        out = out + linear_bias
    return out


def _bonsai_quantized_linear_call(self: nn.QuantizedLinear, x: mx.array) -> mx.array:
    """Replacement for QuantizedLinear.__call__ for 1-bit and 2-bit layers."""
    bits: int = getattr(self, "bits", 4)
    mode: str = getattr(self, "mode", "affine")

    M = x.shape[-2] if x.ndim >= 2 else 1

    # t5 format: uint8 base-3 ternary weights — route before bits check.
    # Decode (M ≤ _T5_PREFILL_THRESHOLD): qmv kernels stream weights once.
    # Prefill (M > threshold): qmv_wide re-reads weights ceil(M/5) times per
    # threadgroup — for M=512 that's 103× DRAM traffic.  Dequantize once to
    # float16 and hand off to MLX's optimised matmul instead.
    if mode == "affine" and bits == 2 and _is_t5_format(self):
        if M > _T5_PREFILL_THRESHOLD:
            return _t5_dequant_matmul(self, x)
        w = self.weight
        scales = self.scales.astype(x.dtype)
        if M >= 2:
            out = bonsai_t5_qmv_wide(x, w, scales)
        else:
            out = bonsai_t5_qmv(x, w, scales)
        linear_bias = getattr(self, "bias", None)
        if linear_bias is not None:
            out = out + linear_bias
        return out

    # Only intercept 1-bit / 2-bit affine layers in decode regime.
    if mode != "affine" or bits not in (1, 2):
        return _original_quantized_linear_call(self, x)

    if M > _MAX_DECODE_M:
        return _original_quantized_linear_call(self, x)

    w = self.weight
    # Cache scales/biases cast to x's dtype (Metal kernel reads them as T).
    scales, biases = _get_cached_scales_biases(self, x.dtype)

    sym = _is_symmetric(self, bits)

    if _use_qmv_wide(bits, M):
        # M>=3 on gen-15+: stream weights once across all M vectors (I-C)
        if bits == 1 and sym:
            out = bonsai_q1_affine_qmv_wide_sym(x, w, scales, biases)
        elif bits == 2 and sym:
            out = bonsai_q2_affine_qmv_wide_sym(x, w, scales, biases)
        else:
            out = bonsai_qmv_wide(x, w, scales, biases, bits=bits)
    elif bits == 1:
        out = (bonsai_q1_affine_qmv_sym if sym else bonsai_q1_affine_qmv)(
            x, w, scales, biases
        )
    else:
        # 2-bit M=1 or M=2: qmv_fast
        out = (bonsai_q2_affine_qmv_sym if sym else bonsai_q2_affine_qmv)(
            x, w, scales, biases
        )

    # QuantizedLinear may have a bias term (separate from quantization biases).
    linear_bias = getattr(self, "bias", None)
    if linear_bias is not None:
        out = out + linear_bias
    return out


def apply_bonsai_qmv_patch() -> bool:
    """Monkey-patch QuantizedLinear for fast 1-bit / 2-bit decode.

    Returns True if the patch was applied (native extension available),
    False if skipped.
    """
    global _original_quantized_linear_call, _patch_active

    if _patch_active:
        return True

    if not has_native():
        logger.debug(
            "bonsai_qmv: native extension not available, skipping patch."
        )
        return False

    _original_quantized_linear_call = nn.QuantizedLinear.__call__
    nn.QuantizedLinear.__call__ = _bonsai_quantized_linear_call
    _patch_active = True
    logger.info("bonsai_qmv: QuantizedLinear patched for 1-bit / 2-bit decode.")
    return True


def remove_bonsai_qmv_patch() -> None:
    """Restore the original QuantizedLinear.__call__."""
    global _original_quantized_linear_call, _patch_active
    if not _patch_active or _original_quantized_linear_call is None:
        return
    nn.QuantizedLinear.__call__ = _original_quantized_linear_call
    _original_quantized_linear_call = None
    _patch_active = False
    logger.info("bonsai_qmv: QuantizedLinear patch removed.")


def is_patch_active() -> bool:
    return _patch_active
