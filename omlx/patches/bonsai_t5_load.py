"""Bonsai t5 weight loading and inference patches.

mlx's Module.load_weights with strict=True rejects t5-format uint8 weights
because they have a different shape/dtype than the uint32 placeholder that
QuantizedLinear creates for 2-bit affine layers:

  QuantizedLinear expects  weight: (N, K//16)  dtype=uint32
  t5-repacked file has     weight: (N, n_groups*bpg)  dtype=uint8

Patch 1 – Module.load_weights
  Reproduces the full strict behaviour (extra-keys / missing-keys errors)
  while allowing t5-format uint8 tensors to replace uint32 weight parameters.
  Weights stay in t5 uint8 format in memory (~23% less RAM than 2-bit).

Patch 2 – mx.quantized_matmul
  mlx_vlm (Qwen3.5 and similar) calls mx.quantized_matmul DIRECTLY, bypassing
  QuantizedLinear.__call__ and our bonsai_qmv inference patch.  This wrapper
  intercepts all calls: for uint8 t5 weights it routes all M through the fast
  bonsai_t5_qmv (M=1) / bonsai_t5_qmv_wide (M≥2) Metal kernels.  The wide
  kernel tiles M into groups of ≤5 in one dispatch — no float weight
  materialisation at any batch size.  Non-t5 calls (uint32) pass through
  unchanged.  A dequant+matmul fallback fires only when the native extension
  is unavailable.

Apply once via apply_bonsai_t5_load_patch() before mlx_vlm / mlx_lm load().
"""
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from omlx.custom_kernels.bonsai.fast import (
    _dequant_1bit,
    bonsai_q1_affine_qmv,
    bonsai_qmv_wide,
    bonsai_t5_qmv,
    bonsai_t5_qmv_wide,
    bonsai_t5_qmm,
    has_native,
)

# M threshold below which qmv kernels are used; above this bonsai_t5_qmm is used.
# Must match _T5_PREFILL_THRESHOLD in bonsai_qmv.py.
_T5_PREFILL_THRESHOLD = 16

logger = logging.getLogger(__name__)

_original_load_weights = None
_original_quantized_matmul = None
_patch_active = False


def _is_t5_weight_replacement(key: str, curr: mx.array, new: mx.array) -> bool:
    """True when *new* is a valid t5 uint8 weight replacing a 2-bit uint32 weight.

    Checks:
      - key ends with '.weight'
      - current parameter dtype is uint32 (2-bit affine mlx format)
      - incoming tensor dtype is uint8 (t5 base-3 packed format)
      - same row count (N)
      - column count is consistent with t5 group encoding:
          group_size=64  → bpg=13  → new.shape[1] % 13 == 0
          group_size=128 → bpg=26  → new.shape[1] % 26 == 0
        and the implied K matches the uint32 packing (16 values/uint32)
    """
    if not key.endswith(".weight"):
        return False
    if curr.dtype != mx.uint32 or new.dtype != mx.uint8:
        return False
    if curr.ndim != 2 or new.ndim != 2:
        return False
    if curr.shape[0] != new.shape[0]:
        return False
    for bpg, group_size in ((13, 64), (26, 128)):
        if new.shape[1] % bpg != 0:
            continue
        n_groups = new.shape[1] // bpg
        K = n_groups * group_size
        if curr.shape[1] == K // 16:
            return True
    return False


def _patched_load_weights(
    self: nn.Module,
    file_or_weights,
    strict: bool = True,
) -> nn.Module:
    """load_weights replacement that allows t5 uint8 weights past strict shape checks.

    t5 weights stay in uint8 format in memory; the shape check is bypassed
    only for valid t5 replacements.  Full strict key-existence checking is
    preserved.
    """
    weights = file_or_weights
    if isinstance(weights, str):
        weights = list(mx.load(weights).items())

    weights_dict = dict(weights)
    curr_weights = dict(tree_flatten(self.parameters()))
    weights = list(weights_dict.items())

    if strict:
        new_weights = weights_dict
        if extras := (new_weights.keys() - curr_weights.keys()):
            num_extra = len(extras)
            extras_str = ",\n".join(sorted(extras))
            raise ValueError(
                f"Received {num_extra} parameters not in model: \n{extras_str}."
            )
        if missing := (curr_weights.keys() - new_weights.keys()):
            num_missing = len(missing)
            missing_str = ",\n".join(sorted(missing))
            raise ValueError(f"Missing {num_missing} parameters: \n{missing_str}.")

        for k, v in curr_weights.items():
            v_new = new_weights[k]
            if not isinstance(v_new, mx.array):
                raise ValueError(
                    f"Expected mx.array but received {type(v_new)} for parameter {k}"
                )
            if v_new.shape != v.shape and not _is_t5_weight_replacement(k, v, v_new):
                raise ValueError(
                    f"Expected shape {v.shape} but received "
                    f"shape {v_new.shape} for parameter {k}"
                )

    if len(weights) != 0:
        self.update(tree_unflatten(weights), strict=False)
    return self


def _t5_quantized_matmul(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    transpose: bool = True,
    bits: int = 4,
    group_size: int = 64,
    **kwargs,
):
    """mx.quantized_matmul replacement that handles t5 uint8 and 1-bit weights.

    When *w* is uint8 (t5 base-3 packed):
      - M=1: route to qmv_fast Metal kernel (single-vector decode).
      - M≥2: route to qmv_wide Metal kernel (arbitrary M via tiling; reads each
        weight byte once across all M vectors — no float weight materialisation).
      The dequant+matmul fallback below is only reached when the native extension
      is unavailable.
    When *bits* is 1: stock mlx accepts the op but its metallib ships no b_1
    kernels, so dispatch dies with "Unable to load kernel".  This path exists
    because mlx-vlm model code (fused projections, as_linear) calls
    mx.quantized_matmul directly, bypassing the patched QuantizedLinear.
    For all other weight dtypes the original C function is called unchanged.
    """
    # 1-bit affine: route around the missing stock kernels.
    if bits == 1 and w.dtype == mx.uint32 and transpose:
        M = x.shape[-2] if x.ndim >= 2 else 1
        if has_native() and M == 1:
            return bonsai_q1_affine_qmv(
                x, w, scales.astype(x.dtype), biases.astype(x.dtype)
            )
        if has_native() and 2 <= M <= 5:
            return bonsai_qmv_wide(x, w, scales, biases, bits=1)
        # Prefill / no native ext: dequantize to x.dtype and matmul.
        w_fp = _dequant_1bit(w, scales, biases, x.dtype, group_size)
        return x @ w_fp.T

    # Normal path: uint32 weights → native MLX kernel (the common case).
    if w.dtype != mx.uint8:
        return _original_quantized_matmul(
            x, w, scales, biases, transpose=transpose, bits=bits, group_size=group_size,
            **kwargs
        )

    # t5 uint8 weight routing:
    #   Decode (M ≤ _T5_PREFILL_THRESHOLD): fast Metal qmv kernels — weight bytes
    #     read once per vector.
    #   Prefill (M > threshold): fused MMA GEMM (I-M) — reads each weight byte
    #     exactly once regardless of M, no float weight materialisation.
    M = x.shape[-2] if x.ndim >= 2 else 1

    if has_native() and M <= _T5_PREFILL_THRESHOLD:
        # scales dtype coercion is handled by the C++ ensure_dtype layer.
        if M >= 2:
            return bonsai_t5_qmv_wide(x, w, scales)
        return bonsai_t5_qmv(x, w, scales)

    if has_native():
        x_flat = x.reshape(-1, x.shape[-1]) if x.ndim > 2 else x
        out_flat = bonsai_t5_qmm(x_flat, w, scales)
        out = out_flat.reshape(x.shape[:-1] + (w.shape[0],))
        if not transpose:
            return out.mT
        return out

    # Fallback (no native ext): dequantize → matmul.
    N = w.shape[0]
    n_groups = scales.shape[-1]
    bpg = w.shape[-1] // n_groups
    gs = 64 if bpg == 13 else 128
    K = n_groups * gs
    v = w.reshape(N, n_groups, bpg).astype(mx.uint32)
    trit_parts = []
    for _ in range(5):
        trit_parts.append(v % 3)
        v = v // 3
    trits = mx.stack(trit_parts, axis=-1).reshape(N, n_groups, bpg * 5)[:, :, :gs]
    sc2 = scales.astype(x.dtype).reshape(N, n_groups, 1)
    weight_fp = ((trits.astype(x.dtype) - 1.0) * sc2).reshape(N, K)
    if transpose:
        return x @ weight_fp.T
    return x @ weight_fp


def free_t5_biases(model: nn.Module) -> int:
    """Replace bias tensors in t5-format layers with tiny placeholders.

    t5 ternary symmetric (I-D) never uses biases at inference time — the
    dequant is ``scale * (q - 1)`` with no additive offset.  The repacked
    safetensors file carries the original 2-bit biases purely for format
    compatibility; after loading they just waste ~420 MB of GPU memory.

    This function walks every QuantizedLinear whose weight dtype is uint8
    (t5 packing) and replaces its ``biases`` parameter with a zero scalar
    so the large bias tensor can be garbage-collected.

    Call once immediately after model.load_weights() / mlx_vlm.load().

    Returns bytes freed (approximate — actual reclaim depends on Python GC
    collecting the replaced bias tensors after model.update()).
    """
    params = dict(tree_flatten(model.parameters()))
    _tiny = mx.zeros((1,), dtype=mx.float16)
    updates: list[tuple[str, mx.array]] = []
    freed = 0

    for k, v in params.items():
        if not k.endswith(".biases"):
            continue
        w = params.get(k.replace(".biases", ".weight"))
        if w is not None and isinstance(w, mx.array) and w.dtype == mx.uint8:
            freed += int(v.size) * v.itemsize
            updates.append((k, _tiny))

    if updates:
        model.update(tree_unflatten(updates))
        mx.eval(model.parameters())
        logger.info(
            "bonsai_t5_load: freed %d unused bias tensors (~%.0f MB) from t5 layers.",
            len(updates),
            freed / 1e6,
        )

    return freed


def apply_bonsai_t5_load_patch() -> bool:
    """Monkey-patch Module.load_weights and mx.quantized_matmul for t5 weights.

    Returns True if newly applied, False if already active.
    """
    global _original_load_weights, _original_quantized_matmul, _patch_active
    if _patch_active:
        return False

    _original_load_weights = nn.Module.load_weights
    nn.Module.load_weights = _patched_load_weights

    import mlx.core as _mx
    _original_quantized_matmul = _mx.quantized_matmul
    _mx.quantized_matmul = _t5_quantized_matmul

    _patch_active = True
    logger.info(
        "bonsai_t5_load: Module.load_weights and mx.quantized_matmul patched "
        "for t5 uint8 weights."
    )
    return True


def remove_bonsai_t5_load_patch() -> None:
    global _original_load_weights, _original_quantized_matmul, _patch_active
    if not _patch_active:
        return
    if _original_load_weights is not None:
        nn.Module.load_weights = _original_load_weights
        _original_load_weights = None
    if _original_quantized_matmul is not None:
        import mlx.core as _mx
        _mx.quantized_matmul = _original_quantized_matmul
        _original_quantized_matmul = None
    _patch_active = False
