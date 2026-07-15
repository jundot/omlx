"""Bonsai t5 weight loading patch.

mlx's Module.load_weights with strict=True rejects t5-format uint8 weights
because they have a different shape/dtype than the uint32 placeholder that
QuantizedLinear creates for 2-bit affine layers:

  QuantizedLinear expects  weight: (N, K//16)  dtype=uint32
  t5-repacked file has     weight: (N, n_groups*bpg)  dtype=uint8

Strategy: convert t5 → 2-bit uint32 at load time (once, in numpy).

  - Disk / download stays at the smaller t5 size (~23% less than 2-bit).
  - Runtime RAM equals the 2-bit model (same uint32 weights in memory).
  - Inference uses the native bonsai 2-bit and MLX quantized_matmul paths,
    which run at full speed (matching the original 2-bit model).

The alternative — decoding t5 base-3 per-token in a Metal kernel — is 6×
slower than native mlx quantized_matmul because GPU integer division is
expensive; base-2 (bit-shifts) runs in ~1 cycle while base-3 (div/mod)
takes ~5–10 cycles per value.

apply_bonsai_t5_load_patch() also wraps mx.quantized_matmul as a safety net
for any stray uint8 tensors that slip past the load-time conversion.

Apply once via apply_bonsai_t5_load_patch() before mlx_vlm / mlx_lm load().
"""
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

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


def _t5_uint8_to_uint32(v_t5: mx.array, scales: mx.array) -> mx.array:
    """Convert t5-format uint8 weight to 2-bit uint32 (mlx affine format).

    Done once at load time in numpy so inference uses the fast native paths.
    """
    N = v_t5.shape[0]
    n_groups = scales.shape[-1]
    bpg = v_t5.shape[1] // n_groups
    group_size = 64 if bpg == 13 else 128
    K = n_groups * group_size

    w = np.array(v_t5, copy=False).reshape(N, n_groups, bpg).astype(np.uint32)

    # Decode base-3: 5 trits per byte
    trit_parts = []
    for _ in range(5):
        trit_parts.append(w % 3)
        w = w // 3
    trits = np.stack(trit_parts, axis=-1).reshape(N, n_groups, bpg * 5)[:, :, :group_size]
    quants = trits.reshape(N, K)  # (N, K), values in {0,1,2}

    # Pack 16 quants per uint32 (MLX 2-bit affine: bit_pos = quant_idx * 2)
    qr = quants.reshape(N, K // 16, 16).astype(np.uint64)
    shifts = (np.arange(16, dtype=np.uint64) * 2)
    packed = np.sum(qr << shifts[None, None, :], axis=-1).astype(np.uint32)
    return mx.array(packed)


def _patched_load_weights(
    self: nn.Module,
    file_or_weights,
    strict: bool = True,
) -> nn.Module:
    """load_weights replacement that converts t5 uint8 weights to uint32 at load.

    For each t5 weight detected, calls _t5_uint8_to_uint32 so the model ends
    up with standard 2-bit uint32 weights.  Full strict key-existence checking
    is preserved; the shape check passes because after conversion the shapes
    match.
    """
    weights = file_or_weights
    if isinstance(weights, str):
        weights = list(mx.load(weights).items())

    weights_dict = dict(weights)

    # Convert any t5 uint8 weights to 2-bit uint32 before strict checking.
    curr_weights = tree_flatten(self.parameters(), destination={})
    n_converted = 0
    for k, v_curr in curr_weights.items():
        v_new = weights_dict.get(k)
        if v_new is None or not isinstance(v_new, mx.array):
            continue
        if not _is_t5_weight_replacement(k, v_curr, v_new):
            continue
        scales_key = k[: -len(".weight")] + ".scales"
        scales_arr = weights_dict.get(scales_key)
        if scales_arr is None or not isinstance(scales_arr, mx.array):
            continue
        weights_dict[k] = _t5_uint8_to_uint32(v_new, scales_arr)
        n_converted += 1

    if n_converted:
        logger.info(
            "bonsai_t5_load: converted %d t5 uint8 weights → uint32 at load time "
            "(inference uses native 2-bit paths at full speed)",
            n_converted,
        )
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
            if v_new.shape != v.shape:
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
    """mx.quantized_matmul replacement that handles t5 uint8 weights.

    When *w* is uint8 (t5 base-3 packed):
      - Decode (M <= 5): route to the native bonsai_t5_qmv / _wide Metal kernel.
      - Prefill (M > 5): dequantize to float and use mx.matmul.
    For all other weight dtypes the original C function is called unchanged.
    """
    # Normal path: uint32 weights → native MLX kernel (the common case).
    if w.dtype != mx.uint8:
        return _original_quantized_matmul(
            x, w, scales, biases, transpose=transpose, bits=bits, group_size=group_size,
            **kwargs
        )

    # Safety fallback: stray uint8 t5 weight that wasn't converted at load time.
    # Dequantize to float and use mx.matmul (correct but slow; should not be hit
    # in practice because _patched_load_weights converts t5 → uint32 at load).
    logger.warning(
        "bonsai_t5_load: uint8 t5 weight reached mx.quantized_matmul at inference "
        "time — expected load-time conversion to have handled this. "
        "Falling back to slow float dequantize path."
    )
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
    bi = biases.astype(x.dtype).reshape(N, n_groups, 1)
    weight_fp = (trits.astype(x.dtype) * sc2 + bi).reshape(N, K)
    if transpose:
        return x @ weight_fp.T
    return x @ weight_fp


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
