# SPDX-License-Identifier: Apache-2.0
"""Preserve oMLX projection routing during Qwen Lightning MTP verification."""

import sys
from functools import wraps

import mlx.nn as nn
from mlx_vlm.speculative.ops import linear as verify_linear

from .. import qwen35_packed_linear, qwen35_verify_qmm


def apply():
    original = verify_linear._use_target_verify_dense
    if getattr(original, "_omlx_mtp_batch_linear", False):
        return

    @wraps(original)
    def use_verify_dense(linear, x):
        if x.ndim == 3 and (
            x.shape[0] > 1
            or (qwen35_verify_qmm._is_armed() and isinstance(linear, nn.QuantizedLinear))
        ):
            return False
        return original(linear, x)

    original_linears = verify_linear._target_verify_linears
    original_quantized = verify_linear._target_verify_quantized_linear

    @wraps(original_quantized)
    def target_verify_quantized(linear, x):
        if qwen35_verify_qmm._is_armed() and x.ndim == 3 and x.shape[1] > 1:
            return linear(x)
        return original_quantized(linear, x)

    @wraps(original_linears)
    def target_verify_linears(linears, x):
        if x.ndim == 3 and (
            x.shape[0] > 1 or (x.shape[1] > 1 and qwen35_verify_qmm._is_armed())
        ):
            return tuple(verify_linear._target_verify_linear(linear, x) for linear in linears)
        return original_linears(linears, x)

    for name in (
        "mlx_vlm.models.qwen3_5.speculative_verifier",
        "mlx_vlm.models.qwen4_exp.language",
    ):
        language = sys.modules.get(name)
        if language is not None:
            language._target_verify_linears = target_verify_linears
            if hasattr(language, "_target_verify_quantized_linear"):
                language._target_verify_quantized_linear = target_verify_quantized
    verify_linear._target_verify_linears = target_verify_linears
    verify_linear._target_verify_quantized_linear = target_verify_quantized
    use_verify_dense._omlx_mtp_batch_linear = True
    verify_linear._use_target_verify_dense = use_verify_dense

    from mlx_vlm.models.qwen3_5.speculative_verifier import (
        Qwen3_5BatchInvariantForward,
    )

    qwen35_packed_linear.apply(Qwen3_5BatchInvariantForward)
