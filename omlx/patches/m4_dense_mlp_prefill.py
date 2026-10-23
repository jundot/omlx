# SPDX-License-Identifier: Apache-2.0
"""Opt-in M4 Max QMM tiles for measured Gemma 4 and Llama MLP shapes.

Only loaded gate/up/down projections are rebound. The original MLP forward,
activation, weights, decode and short verify paths remain unchanged.
"""

from __future__ import annotations

import logging
import os

import mlx.core as mx
import mlx.nn as nn

from .qwen35_q4_mlp import (
    _is_supported_affine_linear_shape,
    _native_qmm_for_bits,
)

logger = logging.getLogger(__name__)

_MIN_TOKENS = 512
_VARIANT = 8  # BM64/BK32/BN64, the existing non-NAX four-SIMD-group tile.
_MLP_CLASSES = frozenset(
    (
        "mlx_lm.models.llama.MLP",
        "mlx_lm.models.gemma4_text.MLP",
        "mlx_vlm.models.gemma4.language.MLP",
    )
)
# (input, output) shapes measured on Gemma 4 12B and llm-jp 4.1 8B.
_PROJECTION_SHAPES = frozenset(
    ((3840, 15360), (15360, 3840), (4096, 14336), (14336, 4096))
)


def _on_m4_max() -> bool:
    try:
        return (
            mx.metal.is_available()
            and mx.device_info().get("device_name") == "Apple M4 Max"
        )
    except Exception:
        return False


def _eligible(linear, dtype, input_dim: int) -> bool:
    weight = getattr(linear, "weight", None)
    return (
        weight is not None
        and weight.ndim == 2
        and (input_dim, weight.shape[0]) in _PROJECTION_SHAPES
        and linear.bits in (4, 8)
        and linear.group_size == 64
        and _is_supported_affine_linear_shape(linear, dtype, 3, _MIN_TOKENS, input_dim)
    )


class _M4PrefillLinear(nn.QuantizedLinear):
    def __call__(self, x):
        # Every decode/verify layer reaches this: keep the common exit cheap.
        if x.ndim != 3 or x.shape[1] < _MIN_TOKENS or x.shape[0] != 1:
            return super().__call__(x)
        if not _eligible(self, x.dtype, x.shape[-1]):
            return super().__call__(x)
        qmm = _native_qmm_for_bits(self.bits)
        if qmm is None:
            return super().__call__(x)
        # No explicit stream: the native wrapper inherits the owning engine's
        # thread-local mx.stream, just like the stock QuantizedLinear operation.
        try:
            # Lazy slices may look contiguous before eval; native QMM reads
            # dense rows. Contiguous defers the layout check/copy until eval
            # and shares buffers when the inputs already have dense rows.
            return qmm(
                mx.contiguous(x),
                mx.contiguous(self.weight),
                mx.contiguous(self.scales),
                mx.contiguous(self.biases),
                _VARIANT,
                64,
            )
        except ValueError:
            # Native admission also rejects layouts (including strided input)
            # that stock QMM handles. Do not swallow asynchronous Metal errors.
            return super().__call__(x)


def apply_m4_dense_mlp_prefill(model) -> int:
    """Rebind eligible loaded projections; return the number changed.

    Disabled by default. Set OMLX_M4_DENSE_MLP_PREFILL=1 before model loading;
    unset it and reload to restore the stock projections. Other chips, missing
    native kernels, custom linear subclasses and unmeasured shapes are untouched.
    """
    if os.environ.get("OMLX_M4_DENSE_MLP_PREFILL") != "1" or not _on_m4_max():
        return 0
    count = 0
    for _, module in model.named_modules():
        cls = type(module)
        if f"{cls.__module__}.{cls.__name__}" not in _MLP_CLASSES:
            continue
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(module, name, None)
            if type(linear) is not nn.QuantizedLinear:
                continue
            input_dim = linear.scales.shape[-1] * linear.group_size
            if _eligible(linear, linear.scales.dtype, input_dim):
                linear.__class__ = _M4PrefillLinear
                count += 1
    if count:
        logger.info(
            "M4 Max dense MLP prefill QMM applied: %d projections "
            "(variant=%d, min_tokens=%d)",
            count,
            _VARIANT,
            _MIN_TOKENS,
        )
    return count
