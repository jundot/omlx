"""Bonsai t5 weight loading and inference patches.

mlx's Module.load_weights with strict=True rejects t5-format uint8 weights
because they have a different shape/dtype than the uint32 placeholder that
QuantizedLinear creates for 2-bit affine layers:

  QuantizedLinear expects  weight: (N, K//16)  dtype=uint32
  t5-repacked file has     weight: (N, n_groups*bpg)  dtype=uint8

Patch 1 – Module.load_weights
  Reproduces the full strict behaviour (extra-keys / missing-keys errors)
  while allowing t5-format uint8 tensors to replace uint32 weight parameters.

Patch 2 – mx.quantized_matmul
  mlx_vlm (Qwen3.5 and similar) calls mx.quantized_matmul DIRECTLY, bypassing
  QuantizedLinear.__call__ and our bonsai_qmv inference patch.  This wrapper
  intercepts all calls: for uint8 t5 weights it routes decode (M<=5) through
  the fast bonsai_t5_qmv / _wide Metal kernels and prefill (M>5) through a
  dequantize + mx.matmul path.  Non-t5 calls (uint32) pass through unchanged.

Apply once via apply_bonsai_t5_load_patch() before mlx_vlm / mlx_lm load().
"""
from __future__ import annotations

import logging

import mlx.core as mx
import mlx.nn as nn
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


def _patched_load_weights(
    self: nn.Module,
    file_or_weights,
    strict: bool = True,
) -> nn.Module:
    """load_weights replacement that allows t5 uint8 weights past the shape check."""
    weights = file_or_weights
    if isinstance(weights, str):
        weights = list(mx.load(weights).items())

    if strict:
        new_weights = dict(weights)
        curr_weights = tree_flatten(self.parameters(), destination={})

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
    """mx.quantized_matmul replacement that handles t5 uint8 weights.

    When *w* is uint8 (t5 base-3 packed):
      - Decode (M <= 5): route to the native bonsai_t5_qmv / _wide Metal kernel.
      - Prefill (M > 5): dequantize to float and use mx.matmul.
    For all other weight dtypes the original C function is called unchanged.
    """
    if w.dtype != mx.uint8:
        return _original_quantized_matmul(
            x, w, scales, biases, transpose=transpose, bits=bits, group_size=group_size,
            **kwargs
        )

    from omlx.custom_kernels.bonsai.fast import (
        bonsai_t5_qmv,
        bonsai_t5_qmv_wide,
        has_native,
        _use_qmv_wide,
    )
    _MAX_DECODE_M = 5
    M = x.shape[-2] if x.ndim >= 2 else 1

    if M <= _MAX_DECODE_M and has_native():
        sc = scales.astype(x.dtype)
        if _use_qmv_wide(2, M):
            return bonsai_t5_qmv_wide(x, w, sc)
        return bonsai_t5_qmv(x, w, sc)

    # Prefill: dequantize to float then matmul
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
