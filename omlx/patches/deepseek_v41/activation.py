# SPDX-License-Identifier: MIT
"""Single-pass FP8 activation round trips with one SIMD group per scale group."""

from functools import cache

import mlx.core as mx

_ROUND = r"""
    // Clamp to 448 * 2^-126 so the power-of-two scale stays normal.
    const float amax = max(simd_max(abs(v)), 0x1.cp-118f);
    const int scale_exponent = max(int(ceil(log2(amax / 448.0f))), -126);
    // Integer powers of two must be exact, including FP8 rounding ties.
    const float scale = as_type<float>(uint(scale_exponent + 127) << 23);
    const float scaled = clamp(v / scale, -448.0f, 448.0f);
    const float a = abs(scaled);
    const int step_exponent = max(int(floor(log2(max(a, 0x1p-9f)))) - 3, -9);
    const float step = as_type<float>(uint(step_exponent + 127) << 23);
    const float q = sign(scaled) * min(rint(a / step) * step, 448.0f);
    if (i < n) y[i] = T(q * scale);
"""

_SOURCE = r"""
    const uint i = thread_position_in_grid.x;
    const uint n = N;
    const float v = i < n ? float(x[i]) : 0.0f;
""" + _ROUND

_TAIL_SOURCE = r"""
    const uint i = thread_position_in_grid.x;
    const uint n = N;
    float v = 0.0f;
    if (i < n) {
        float g = gate[i], u = up[i];
        if (limit[0] != 0.0f) {
            g = min(g, limit[0]);
            u = clamp(u, -limit[0], limit[0]);
        }
        // Match MLX sigmoid arithmetic before the intermediate dtype cast.
        const float neg_sigmoid = 1.0f / (1.0f + exp(abs(g)));
        const float sigmoid = g < 0 ? neg_sigmoid : 1.0f - neg_sigmoid;
        float value = (g * sigmoid) * u;
        if (WEIGHTED) value *= weights[i / D];
        v = float(T(value));
    }
""" + _ROUND


@cache
def _kernel(tail=False, paired=False):
    if paired:
        return mx.fast.metal_kernel(
            name="v41_paired_swiglu_fp8_activation",
            input_names=["pair", "weights", "limit"],
            output_names=["y"],
            source=_TAIL_SOURCE.replace(
                "float g = gate[i], u = up[i];",
                "const uint j = (i / D) * (2 * D) + i % D; "
                "float g = pair[j], u = pair[j + D];",
            ),
        )
    return mx.fast.metal_kernel(
        name="v41_swiglu_fp8_activation" if tail else "v41_fp8_activation",
        input_names=["gate", "up", "weights", "limit"] if tail else ["x"],
        output_names=["y"],
        source=_TAIL_SOURCE if tail else _SOURCE,
    )


def quantize_fp8_activation(x):
    return _kernel()(
        inputs=[x],
        template=[("T", x.dtype), ("N", x.size)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def quantize_swiglu_activation(gate, up, weights, dtype, limit):
    return _kernel(tail=True)(
        inputs=[
            gate,
            up,
            weights if weights is not None else mx.ones((1,)),
            mx.array([float(limit or 0)], mx.float32),
        ],
        template=[
            ("T", dtype),
            ("N", gate.size),
            ("D", gate.shape[-1]),
            ("WEIGHTED", weights is not None),
        ],
        grid=(gate.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[gate.shape],
        output_dtypes=[dtype],
    )[0]


def quantize_paired_swiglu_activation(pair, weights, dtype, limit):
    """Read concatenated gate/up rows without materializing two contiguous copies."""
    shape = (*pair.shape[:-1], pair.shape[-1] // 2)
    size = pair.size // 2
    return _kernel(paired=True)(
        inputs=[
            pair,
            weights if weights is not None else mx.ones((1,)),
            mx.array([float(limit or 0)], mx.float32),
        ],
        template=[
            ("T", dtype),
            ("N", size),
            ("D", shape[-1]),
            ("WEIGHTED", weights is not None),
        ],
        grid=(size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[dtype],
    )[0]
