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
    bonsai_t5_qmm,
    has_native,
    _use_qmv_wide,
)

logger = logging.getLogger(__name__)

_original_quantized_linear_call: Any = None
_original_attention_call: Any = None
_original_mlp_call: Any = None
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
    once without materialising a float weight matrix.  Falls back to the
    Python dequant chain only when the native extension is unavailable.
    """
    w = self.weight
    scales = self.scales

    # Native fused kernel (preferred): reads weights once, no float materialisation.
    if has_native():
        # x may have leading batch dims; flatten to (M, K) for the kernel.
        x_flat = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
        out_flat = bonsai_t5_qmm(x_flat, w, scales)
        out = out_flat.reshape(x.shape[:-1] + (w.shape[0],))
        linear_bias = getattr(self, "bias", None)
        if linear_bias is not None:
            out = out + linear_bias
        return out

    # Fallback: Python MLX dequant chain (no native ext).
    N = w.shape[0]
    n_groups = scales.shape[-1]
    bpg = w.shape[1] // n_groups
    group_size = 64 if bpg == 13 else 128
    K = n_groups * group_size
    v = w.reshape(N, n_groups, bpg).astype(mx.uint32)
    trit_parts = []
    for _ in range(5):
        trit_parts.append(v % 3)
        v = v // 3
    trits = mx.stack(trit_parts, axis=-1).reshape(N, n_groups, bpg * 5)
    trits = trits[:, :, :group_size]
    dq = (trits.astype(x.dtype) - 1.0) * scales[..., None].astype(x.dtype)
    weight_fp = dq.reshape(N, K)
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
            w = self.weight  # (N, K//32) uint32
            scales = self.scales.astype(x.dtype)
            biases = getattr(self, "biases", None)
            if biases is not None:
                biases = biases.astype(x.dtype)
            N, K32 = w.shape
            K_full = K32 * 32
            n_groups = scales.shape[-1]
            gs = K_full // n_groups
            # Unpack 1-bit to float16: (N, K_full)
            shifts = mx.arange(32, dtype=mx.uint32)
            w_flat = ((w[:, :, None] >> shifts) & 0x1).astype(x.dtype).reshape(N, K_full)
            # Expand scales from (N, n_groups) to (N, K_full)
            scales_exp = mx.repeat(scales, gs, axis=-1)
            w_fp = w_flat * scales_exp
            if biases is not None:
                biases_exp = mx.repeat(biases, gs, axis=-1)
                w_fp = w_fp + biases_exp
            w_fp = w_fp.reshape(N, K_full)
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
# Load-time specialization: replace each QuantizedLinear.__call__ with a
# pre-bound closure that has routing already resolved, eliminating per-call
# Python overhead (bits/mode check, sym check, getattr, dtype cast).
# ---------------------------------------------------------------------------

def specialize_quantized_linears(model: nn.Module) -> int:
    """Replace patched __call__ on every QuantizedLinear with a fast closure.

    Called from apply_post_load_transforms after all weights are loaded.
    Returns the number of layers specialized.
    """
    n = 0
    for _name, module in model.named_modules():
        if not isinstance(module, nn.QuantizedLinear):
            continue

        bits = getattr(module, "bits", 4)
        mode = getattr(module, "mode", "affine")
        if mode != "affine" or bits not in (1, 2):
            continue

        # Capture everything at specialization time
        w = module.weight
        scales = module.scales
        biases = getattr(module, "biases", None)
        sym = bool(_is_symmetric(module, bits))
        linear_bias = getattr(module, "bias", None)

        # Resolve kernel choice once
        if bits == 1:
            if sym:
                fast_fn = bonsai_q1_affine_qmv_sym
                wide_fn = bonsai_q1_affine_qmv_wide_sym
            else:
                fast_fn = bonsai_q1_affine_qmv
                wide_fn = None
        else:  # bits == 2
            if sym:
                fast_fn = bonsai_q2_affine_qmv_sym
                wide_fn = bonsai_q2_affine_qmv_wide_sym
            else:
                fast_fn = bonsai_q2_affine_qmv
                wide_fn = None

        use_wide = _use_qmv_wide(bits, 3)

        # Save original call for prefill fallback (M > 5)
        _orig_call = module.__call__

        # Symmetric kernels: bias = -scale * ratio — skip dead biases arg
        if sym:
            def _decode(
                x: mx.array,
                _w=w, _sc=scales, _lb=linear_bias, _fn=fast_fn,
            ) -> mx.array:
                out = _fn(x, _w, _sc, biases)
                if _lb is not None:
                    out = out + _lb
                return out
        else:
            def _decode(
                x: mx.array,
                _w=w, _sc=scales, _bi=biases, _lb=linear_bias, _fn=fast_fn,
            ) -> mx.array:
                out = _fn(x, _w, _sc, _bi)
                if _lb is not None:
                    out = out + _lb
                return out

        # Build outer router
        if wide_fn is not None and use_wide:
            def _wide(
                x: mx.array,
                _w=w, _sc=scales, _bi=biases, _lb=linear_bias, _fn=wide_fn,
            ) -> mx.array:
                out = _fn(x, _w, _sc, _bi)
                if _lb is not None:
                    out = out + _lb
                return out

            def _outer(x: mx.array, _orig=_orig_call) -> mx.array:
                M = x.shape[-2] if x.ndim >= 2 else 1
                if M == 1:
                    return _decode(x)
                if M <= 5:
                    return _wide(x)
                return _orig(x)
        else:
            def _outer(x: mx.array, _orig=_orig_call) -> mx.array:
                M = x.shape[-2] if x.ndim >= 2 else 1
                if M <= 5:
                    return _decode(x)
                return _orig(x)

        module.__call__ = _outer
        n += 1

    if n:
        logger.info(
            "bonsai_qmv: load-time specialization applied to %d QuantizedLinear layers "
            "(per-call Python dispatch eliminated)",
            n,
        )
    return n


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
    if bits == 2:
        N, K = w.shape
        weight = mx.zeros((N, K // 16), dtype=mx.uint32)
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
    # Dequantize the embedding weight on the fly.
    w = self.weight  # (num_embeddings, dims//32) uint32
    scales = self.scales  # (num_embeddings, n_groups)
    biases = getattr(self, "biases", None)

    num_emb, K32 = w.shape
    dims = K32 * 32
    # Unpack 1-bit → (num_emb, dims) float16
    shifts = mx.arange(32, dtype=mx.uint32)
    w_flat = ((w[:, :, None] >> shifts) & 0x1).astype(mx.float16).reshape(num_emb, dims)
    n_groups = scales.shape[-1]
    gs = dims // n_groups
    scales_exp = mx.repeat(scales, gs, axis=-1)
    w_fp = w_flat * scales_exp
    if biases is not None:
        biases_exp = mx.repeat(biases, gs, axis=-1)
        w_fp = w_fp + biases_exp
    return w_fp[x.astype(mx.int32)]


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


# ---------------------------------------------------------------------------
# QKV / gate-up weight fusion (review item #2)
# ---------------------------------------------------------------------------
#
# For each attention block: q_proj, k_proj, v_proj share the same input x.
# Concatenate their quantized weights row-wise into one (Nq+Nk+Nv)×K matrix.
# One kernel call instead of three — eliminates ~128 dispatches per token
# and streams a longer contiguous weight region for better bandwidth util.
#
# Similarly for MLP: gate_proj + up_proj fuse into one gate_up_proj.

def _cat_quantized_weights(
    layers: list[nn.QuantizedLinear],
) -> tuple[mx.array, mx.array, mx.array, list[int]]:
    """Concatenate QuantizedLinear weight tensors row-wise.

    Returns (fused_weight, fused_scales, fused_biases, splits).
    `splits` = [N1, N2, ...] for splitting the fused output.
    """
    weights = [l.weight for l in layers]
    scales_list = [l.scales for l in layers]
    biases_list = [getattr(l, "biases", None) for l in layers]

    fused_w = mx.concatenate(weights, axis=0)
    fused_sc = mx.concatenate(scales_list, axis=0)
    fused_bi = None
    if all(b is not None for b in biases_list):
        fused_bi = mx.concatenate([b for b in biases_list if b is not None], axis=0)

    splits = [w.shape[0] for w in weights]
    return fused_w, fused_sc, fused_bi, splits


def _make_fused_qkv_project(
    q_proj: nn.QuantizedLinear,
    k_proj: nn.QuantizedLinear,
    v_proj: nn.QuantizedLinear,
):
    """Build a fused QKV projector closure.

    Returns a callable `fused_qkv(x) -> (queries, keys, values)` that issues
    one Bonsai kernel call and splits the output.
    """
    fused_w, fused_sc, fused_bi, splits = _cat_quantized_weights(
        [q_proj, k_proj, v_proj]
    )
    bits = q_proj.bits

    # Use specialization style: pre-bind closures
    from omlx.patches.bonsai_qmv import _is_symmetric

    sym = _is_symmetric(q_proj, bits)
    if bits == 1:
        fn = bonsai_q1_affine_qmv_sym if sym else bonsai_q1_affine_qmv
    else:
        fn = bonsai_q2_affine_qmv_sym if sym else bonsai_q2_affine_qmv

    def _fused_qkv(
        x: mx.array,
        _w=fused_w, _sc=fused_sc, _bi=fused_bi, _fn=fn, _sp=splits,
    ) -> tuple[mx.array, mx.array, mx.array]:
        out = _fn(x, _w, _sc, _bi)
        a, b = _sp[0], _sp[0] + _sp[1]
        return out[..., :a], out[..., a:b], out[..., b:]

    return _fused_qkv


def _make_fused_gate_up_project(
    gate_proj: nn.QuantizedLinear,
    up_proj: nn.QuantizedLinear,
):
    """Build a fused gate-up projector closure."""
    fused_w, fused_sc, fused_bi, splits = _cat_quantized_weights(
        [gate_proj, up_proj]
    )
    bits = gate_proj.bits
    from omlx.patches.bonsai_qmv import _is_symmetric
    sym = _is_symmetric(gate_proj, bits)
    if bits == 1:
        fn = bonsai_q1_affine_qmv_sym if sym else bonsai_q1_affine_qmv
    else:
        fn = bonsai_q2_affine_qmv_sym if sym else bonsai_q2_affine_qmv

    def _fused_gate_up(
        x: mx.array,
        _w=fused_w, _sc=fused_sc, _bi=fused_bi, _fn=fn, _sp=splits,
    ) -> tuple[mx.array, mx.array]:
        out = _fn(x, _w, _sc, _bi)
        return out[..., :_sp[0]], out[..., _sp[0]:]

    return _fused_gate_up


def fuse_attention_and_mlp_projections(model) -> int:
    """Fuse Q/K/V and gate/up projections across the model."""
    import mlx.nn as _nn
    n_attn = 0
    n_mlp = 0

    for _name, module in model.named_modules():
        # Check for attention module with all three quantized projections
        q_proj = getattr(module, "q_proj", None)
        k_proj = getattr(module, "k_proj", None)
        v_proj = getattr(module, "v_proj", None)

        if (
            isinstance(q_proj, nn.QuantizedLinear)
            and isinstance(k_proj, nn.QuantizedLinear)
            and isinstance(v_proj, nn.QuantizedLinear)
            and q_proj.bits in (1, 2)
            and q_proj.bits == k_proj.bits == v_proj.bits
            and q_proj.mode == k_proj.mode == v_proj.mode == "affine"
        ):
            fused_qkv = _make_fused_qkv_project(q_proj, k_proj, v_proj)
            module._fused_qkv = fused_qkv

            # Patch attention forward with fused QKV
            _mod = module  # capture for closure

            def _patched_attn_call(
                x, mask=None, cache=None,
                _m=_mod, _fused=fused_qkv,
            ):
                B, L, D = x.shape
                queries, keys, values = _fused(x)

                queries = _m.q_norm(queries.reshape(B, L, _m.n_heads, -1)).transpose(0, 2, 1, 3)
                keys = _m.k_norm(keys.reshape(B, L, _m.n_kv_heads, -1)).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, _m.n_kv_heads, -1).transpose(0, 2, 1, 3)

                if cache is not None:
                    queries = _m.rope(queries, offset=cache.offset)
                    keys = _m.rope(keys, offset=cache.offset)
                    keys, values = cache.update_and_fetch(keys, values)
                else:
                    queries = _m.rope(queries)
                    keys = _m.rope(keys)

                output = _nn.fast.scaled_dot_product_attention(
                    queries, keys, values, cache=cache, scale=_m.scale, mask=mask
                )
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return _m.o_proj(output)

            module.__call__ = _patched_attn_call
            n_attn += 1

        # Check for MLP with gate/up
        gate_proj = getattr(module, "gate_proj", None)
        up_proj = getattr(module, "up_proj", None)

        if (
            isinstance(gate_proj, nn.QuantizedLinear)
            and isinstance(up_proj, nn.QuantizedLinear)
            and gate_proj.bits in (1, 2)
            and gate_proj.bits == up_proj.bits
            and gate_proj.mode == up_proj.mode == "affine"
        ):
            fused_gate_up = _make_fused_gate_up_project(gate_proj, up_proj)
            module._fused_gate_up = fused_gate_up
            _mod = module

            def _patched_mlp_call(x, _m=_mod, _fused=fused_gate_up):
                gate, up = _fused(x)
                return _m.down_proj(_nn.silu(gate) * up)

            module.__call__ = _patched_mlp_call
            n_mlp += 1

    if n_attn or n_mlp:
        logger.info(
            "bonsai_qmv: fused %d attention + %d MLP projection blocks "
            "(QKV/gate-up concatenated, kernel calls reduced)",
            n_attn, n_mlp,
        )
    return n_attn + n_mlp
