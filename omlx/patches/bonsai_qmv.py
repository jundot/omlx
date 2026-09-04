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
    _dequant_1bit,
    bonsai_q1_affine_qmv,
    bonsai_q2_affine_qmv,
    bonsai_q1_affine_qmv_sym,
    bonsai_q2_affine_qmv_sym,
    bonsai_q1_affine_qmv_wide_sym,
    bonsai_q2_affine_qmv_wide_sym,
    bonsai_qmv_wide,
    bonsai_t5_qmv,
    bonsai_t5_qmv_wide,
    bonsai_t5_qmm,
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


def _get_scales_biases(self: nn.QuantizedLinear, dtype: Any) -> tuple[mx.array, mx.array]:
    """Return scales/biases. Casts to dtype if needed (infrequent; Bonsai uses fp16)."""
    sc = self.scales
    bi = getattr(self, "biases", None)
    if sc.dtype != dtype:
        sc = sc.astype(dtype)
    if bi is not None and bi.dtype != dtype:
        bi = bi.astype(dtype)
    return sc, bi


def _get_t5_scales(self: nn.QuantizedLinear, dtype: Any) -> mx.array:
    """Return t5 scales. Casts to dtype if needed."""
    sc = self.scales
    if sc.dtype != dtype:
        sc = sc.astype(dtype)
    return sc


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
    """Prefill path for t5 weights: fused t5 MMA GEMM (Identity I-M).

    Routes through bonsai_t5_qmm which decodes each t5 weight byte exactly
    once without materialising a float weight matrix.
    """
    # The qmv patch is only installed when the native extension is present
    # (apply_bonsai_qmv_patch gates on has_native), so this path can assume
    # the fused kernel.  The no-native t5 dequant chain lives in
    # bonsai_t5_load._t5_quantized_matmul, the only reachable copy.
    w = self.weight
    scales = self.scales

    # x may have leading batch dims; flatten to (M, K) for the kernel.
    x_flat = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
    out_flat = bonsai_t5_qmm(x_flat, w, scales)
    out = out_flat.reshape(x.shape[:-1] + (w.shape[0],))
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
    if mode == "affine" and _is_t5_format(self):
        if M > _T5_PREFILL_THRESHOLD:
            return _t5_dequant_matmul(self, x)
        w = self.weight
        scales = _get_t5_scales(self, x.dtype)
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

    # Prefill: M > _MAX_DECODE_M uses stock quantized_matmul which calls
    # affine_dequantize.  Stock MLX doesn't have affine_dequantize for bits=1
    # in its metallib.  For bits=1 prefill, dequantize to float16 explicitly.
    if M > _MAX_DECODE_M:
        if bits == 1:
            w_fp = _dequant_1bit(
                self.weight, self.scales, getattr(self, "biases", None), x.dtype
            )
            out = x @ w_fp.T
            linear_bias = getattr(self, "bias", None)
            if linear_bias is not None:
                out = out + linear_bias
            return out
        return _original_quantized_linear_call(self, x)

    w = self.weight
    # Cache scales/biases cast to x's dtype (Metal kernel reads them as T).
    scales, biases = _get_scales_biases(self, x.dtype)

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


# ---------------------------------------------------------------------------
# 1-bit / 2-bit construction-time patch
# ---------------------------------------------------------------------------
#
# Stock mlx-lm calls nn.quantize(model, bits=1) at construction time.
# mx.quantize(bits=1) is rejected at the C++ level before any inference
# patches can take effect.  This patch monkey-patches mx.quantize itself
# to create the uint32-packed weight tensor directly for bits=1/2.
# We also patch QuantizedEmbedding.__call__ for bits=1 since stock
# mlx doesn't have affine_dequantize for bits=1 in its metallib.

_original_mx_quantize = None
_original_quantized_embedding_call = None
_construct_patch_active = False


def _patched_mx_quantize(
    w: mx.array, group_size: int = 64, bits: int = 4, mode: str = "affine",
) -> tuple:
    if bits == 1:
        N, K = w.shape
        weight = mx.zeros((N, K // 32), dtype=mx.uint32)
        n_groups = K // group_size
        scales = mx.zeros((N, n_groups), dtype=mx.float16)
        biases = mx.zeros((N, n_groups), dtype=mx.float16)
        return weight, scales, biases
    return _original_mx_quantize(w, group_size, bits, mode)


def _patched_quantized_embedding_call(self: nn.QuantizedEmbedding, x: mx.array) -> mx.array:
    bits = getattr(self, "bits", 4)
    if bits != 1:
        return _original_quantized_embedding_call(self, x)

    # bits=1: stock mlx has no affine_dequantize for bits=1.
    # Gather only the looked-up rows, then dequantize those — dequantizing
    # the full table would materialize (num_embeddings, dims) every call.
    ids = x.astype(mx.int32)
    w_rows = self.weight[ids]  # (..., dims//32) uint32
    scales_rows = self.scales[ids]  # (..., n_groups)
    biases = getattr(self, "biases", None)

    K32 = w_rows.shape[-1]
    dims = K32 * 32
    n_groups = scales_rows.shape[-1]
    gs = dims // n_groups
    # Unpack 1-bit → (..., dims) float16
    shifts = mx.arange(32, dtype=mx.uint32)
    w_flat = ((w_rows[..., None] >> shifts) & 0x1).astype(mx.float16)
    w_flat = w_flat.reshape(*w_rows.shape[:-1], dims)
    scales_exp = mx.repeat(scales_rows, gs, axis=-1)
    w_fp = w_flat * scales_exp
    if biases is not None:
        biases_exp = mx.repeat(biases[ids], gs, axis=-1)
        w_fp = w_fp + biases_exp
    return w_fp


def apply_bonsai_construct_patch() -> bool:
    """Install the 1-bit/2-bit construction and embedding shim.

    No-ops if the underlying mlx already supports bits=1 natively (e.g. the
    PrismML fork), so native kernels handle everything without the shim.
    """
    global _construct_patch_active, _original_mx_quantize
    global _original_quantized_embedding_call
    if _construct_patch_active:
        return False

    # If mx.quantize already accepts bits=1, the shim is not needed.
    try:
        _probe = mx.random.normal((32, 128)).astype(mx.float16)
        mx.eval(mx.quantize(_probe, group_size=64, bits=1)[0])
        logger.info(
            "bonsai_construct: native bits=1 present (PrismML fork?); patch not needed."
        )
        return False
    except Exception:
        pass

    _original_mx_quantize = mx.quantize
    mx.quantize = _patched_mx_quantize

    _original_quantized_embedding_call = nn.QuantizedEmbedding.__call__
    nn.QuantizedEmbedding.__call__ = _patched_quantized_embedding_call

    _construct_patch_active = True
    logger.info(
        "bonsai_construct: mx.quantize + QuantizedEmbedding patched for 1-bit."
    )
    return True


def remove_bonsai_construct_patch() -> None:
    global _construct_patch_active
    if not _construct_patch_active:
        return
    if _original_mx_quantize is not None:
        mx.quantize = _original_mx_quantize
    if _original_quantized_embedding_call is not None:
        nn.QuantizedEmbedding.__call__ = _original_quantized_embedding_call
    _construct_patch_active = False
    logger.info("bonsai_construct: patches removed.")
