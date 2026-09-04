# SPDX-License-Identifier: Apache-2.0
"""M=1-equivalent reductions for DeepSeek DSpark target verification."""

from __future__ import annotations

import threading

_STATE = threading.local()
_PATCHED = False


def set_armed(flag: bool) -> None:
    _STATE.armed = bool(flag)


def is_armed() -> bool:
    return bool(getattr(_STATE, "armed", False))


def _batched_singleton_qmv(x, call):
    """Batch independent M=1 QMV rows without selecting MLX's M>1 kernel."""
    import mlx.core as mx

    width = int(x.shape[-1])
    rows = int(x.size) // width
    if rows <= 1:
        return call(x)

    flat = x.reshape(rows, width)
    # The sliced middle dimension deliberately leaves a non-row-contiguous
    # batch stride. MLX consequently dispatches one batched QMV with matrix
    # height 1 instead of qmv_wide with matrix height ``rows``. Each row then
    # uses the exact same reduction order as ordinary single-token decode.
    singleton_batch = mx.stack([flat, flat], axis=1)[:, :1, :]
    result = call(singleton_batch)
    return result.reshape((*x.shape[:-1], result.shape[-1]))


def matmul(x, weight_t):
    if not is_armed():
        return x @ weight_t
    width = int(x.shape[-1])
    rows = int(x.size) // width
    if rows <= 1:
        return x @ weight_t
    flat = x.reshape(rows, width)
    result = (weight_t.T @ flat[..., None]).squeeze(-1)
    return result.reshape((*x.shape[:-1], result.shape[-1]))


def apply() -> bool:
    global _PATCHED
    if _PATCHED:
        return True

    import mlx.nn as nn

    cls = nn.Linear
    marker = "_omlx_deepseek_verify_rowwise"
    if not getattr(cls, marker, False):
        linear_call = cls.__call__

        def patched_linear_call(self, x):
            if not is_armed():
                return linear_call(self, x)
            from omlx.patches.deepseek_v4.verify_qmv import (
                dense_eligible,
                exact_verify_gemv,
            )

            if dense_eligible(self, x):
                return exact_verify_gemv(self, x)
            width = int(x.shape[-1])
            rows = int(x.size) // width
            if rows <= 1:
                return linear_call(self, x)
            flat = x.reshape(rows, width)
            result = (self.weight @ flat[..., None]).squeeze(-1)
            if "bias" in self:
                result = result + self.bias
            return result.reshape((*x.shape[:-1], result.shape[-1]))

        cls.__call__ = patched_linear_call
        setattr(cls, marker, True)
    cls = nn.QuantizedLinear
    marker = "_omlx_deepseek_verify_rowwise"
    if not getattr(cls, marker, False):
        quantized_call = cls.__call__

        def patched_quantized_call(self, x):
            if not is_armed():
                return quantized_call(self, x)
            from omlx.patches.deepseek_v4.verify_qmv import (
                eligible,
                exact_verify_qmv,
            )

            if eligible(self, x):
                return exact_verify_qmv(self, x)
            return _batched_singleton_qmv(
                x,
                lambda row: quantized_call(self, row),
            )

        cls.__call__ = patched_quantized_call
        setattr(cls, marker, True)
    _PATCHED = True
    return True


__all__ = [
    "apply",
    "is_armed",
    "matmul",
    "set_armed",
]
