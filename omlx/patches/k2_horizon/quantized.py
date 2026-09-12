# SPDX-License-Identifier: MIT
# Kernel adapted from MLX qmv_wide_impl, Copyright © 2023-2024 Apple Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Eight-row affine Q8 projections for Uno on the validated M4 Max GPU."""

from functools import cache

import mlx.core as mx
import mlx.nn as nn

_SOURCE = r"""
    const int lane = thread_index_in_simdgroup % 8;
    const int row = threadgroup_position_in_grid.x * 8
        + simdgroup_index_in_threadgroup * 4 + thread_index_in_simdgroup / 8;
    const device uchar* wr = (const device uchar*)w + size_t(row) * K;
    float result[8] = {0};
    // Retain MLX's group, sub-chunk, and reduction order. Only the number
    // of input vectors sharing a decoded weight changes, from four to eight.
    #pragma clang loop unroll(disable)
    for (int g = lane; g < K / 64; g += 8) {
        float s = float(scales[row * (K / 64) + g]);
        float b = float(biases[row * (K / 64) + g]);
        #pragma unroll
        for (int sc = 0; sc < 8; sc++) {
            int k0 = g * 64 + sc * 8;
            float wd[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) wd[i] = s * wr[k0 + i] + b;
            #pragma unroll
            for (int v = 0; v < 8; v++) {
                float acc = 0;
                #pragma unroll
                for (int i = 0; i < 8; i++) acc += float(x[v * K + k0 + i]) * wd[i];
                result[v] += acc;
            }
        }
    }
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        result[v] += simd_shuffle_down(result[v], 4);
        result[v] += simd_shuffle_down(result[v], 2);
        result[v] += simd_shuffle_down(result[v], 1);
        if (lane == 0) y[v * N + row] = T(result[v]);
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="k2_uno_q8_block8",
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        source=_SOURCE,
    )


def project_linear(linear, x):
    if (
        not getattr(linear, "_omlx_uno_q8_block", False)
        or x.ndim != 3
        or x.shape[:2] != (1, 8)
        or x.dtype != mx.bfloat16
        or x.shape[-1] != linear.weight.shape[1] * 4
    ):
        return linear(x)
    n = linear.weight.shape[0]
    y = _kernel()(
        inputs=[x, linear.weight, linear.scales, linear.biases],
        template=[("T", x.dtype), ("K", x.shape[-1]), ("N", n)],
        grid=(64 * (n // 8), 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(*x.shape[:-1], n)],
        output_dtypes=[x.dtype],
    )[0]
    return y + linear.bias if "bias" in linear else y


def enable_q8_blocks(model):
    """Mark eligible base projections; keep parameter objects and LoRA separate."""
    from .uno_adapter import TARGETS, ConditionalLoRALinear

    if (
        not getattr(model, "_uno_adapter_loaded", False)
        or mx.default_device().type != mx.gpu
        or mx.device_info().get("architecture") != "applegpu_g16s"
    ):
        return 0
    projections = [
        getattr(getattr(layer, scope), name)
        for layer in model.layers
        for scope, names in TARGETS.items()
        for name in names
    ]
    if not model.args.tie_word_embeddings:
        projections.append(model.lm_head)
    count = 0
    for projection in projections:
        linear = (
            projection.linear
            if isinstance(projection, ConditionalLoRALinear)
            else projection
        )
        if (
            isinstance(linear, nn.QuantizedLinear)
            and linear.mode == "affine"
            and linear.bits == 8
            and linear.group_size == 64
            and linear.scales.dtype == linear.biases.dtype == mx.bfloat16
            and linear.weight.shape[0] >= 1024
            and linear.weight.shape[0] % 8 == 0
            and linear.weight.shape[1] * 4 >= 4096
            and linear.weight.shape[1] * 4 % 512 == 0
        ):
            linear._omlx_uno_q8_block = True
            count += 1
    return count
