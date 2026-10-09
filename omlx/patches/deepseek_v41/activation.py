# SPDX-License-Identifier: MIT
"""Single-pass FP8 activation round trips with one SIMD group per scale group."""

from functools import cache

import mlx.core as mx

_SCALE = r"""
    // Clamp to 448 * 2^-126 so the power-of-two scale stays normal.
    const float amax = max(simd_max(abs(v)), 0x1.cp-118f);
    const int scale_exponent = max(int(ceil(log2(amax / 448.0f))), -126);
    // Integer powers of two must be exact, including FP8 rounding ties.
    const float scale = as_type<float>(uint(scale_exponent + 127) << 23);
    const float scaled = clamp(v / scale, -448.0f, 448.0f);
    const float a = abs(scaled);
    const int step_exponent = max(int(floor(log2(max(a, 0x1p-9f)))) - 3, -9);
    const float step = as_type<float>(uint(step_exponent + 127) << 23);
"""

_ROUND = _SCALE + r"""
    const float q = sign(scaled) * min(rint(a / step) * step, 448.0f);
    if (i < n) y[i] = T(q * scale);
"""

# MLX to_fp8 (E4M3FN) bit conversion; q is already on the FP8 grid.
_FP8_HEADER = r"""
inline uchar v41_fp8_bits(float f) {
    uint bits = as_type<uint>(f);
    const uint sign = bits & 0x80000000u;
    bits ^= sign;
    uchar out;
    if (bits >= (543u << 21)) {
        out = 0x7E;
    } else if (bits < (121u << 23)) {
        const uint denorm = 141u << 23;
        out = uchar(as_type<uint>(as_type<float>(bits) + as_type<float>(denorm)) - denorm);
    } else {
        const uint odd = (bits >> 20) & 1;
        bits += ((uint)(7 - 127) << 23) + 0x7FFFF + odd;
        out = uchar(bits >> 20);
    }
    return out | uchar(sign >> 24);
}
"""

_PACK_SOURCE = (
    r"""
    const uint i = thread_position_in_grid.x;
    const float v = float(x[i]);
"""
    + _SCALE
    + r"""
    // MLX sign() maps -0.0 to +0.0.
    const float s = float(int(scaled > 0.0f) - int(scaled < 0.0f));
    const float q = s * min(rint(a / step) * step, 448.0f);
    const uint row = i / D, d = i % D;
    device uchar* out = packed + size_t(row) * (D + D / 32);
    out[d] = v41_fp8_bits(q);
    if (d % 32 == 0) out[D + d / 32] = uchar(scale_exponent + 127);
"""
)

_SOURCE = r"""
    const uint i = thread_position_in_grid.x;
    uint n = 1;
    for (int dim = 0; dim < x_ndim; ++dim) n *= x_shape[dim];
    const float v = i < n ? float(x[i]) : 0.0f;
""" + _ROUND

_TAIL_SOURCE = r"""
    const uint i = thread_position_in_grid.x;
    uint n = 1;
    for (int dim = 0; dim < gate_ndim; ++dim) n *= gate_shape[dim];
    float v = 0.0f;
    if (i < n) {
        float g = gate[i], u = up[i];
        if (limit[0] != 0.0f) {
            g = min(g, limit[0]);
            u = clamp(u, -limit[0], limit[0]);
        }
        // Match MLX sigmoid arithmetic before the intermediate dtype cast.
        const float neg_sigmoid = 1.0f / (1.0f + metal::precise::exp(abs(g)));
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
            source=_TAIL_SOURCE.replace("gate_ndim", "pair_ndim")
            .replace("gate_shape", "pair_shape")
            .replace("float v = 0.0f;", "n /= 2;\n    float v = 0.0f;")
            .replace(
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
        template=[("T", x.dtype)],
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
            ("D", shape[-1]),
            ("WEIGHTED", weights is not None),
        ],
        grid=(size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[dtype],
    )[0]


@cache
def _pack_kernel():
    return mx.fast.metal_kernel(
        name="v41_pack_fp8_activation",
        input_names=["x"],
        output_names=["packed"],
        source=_PACK_SOURCE,
        header=_FP8_HEADER,
    )


def pack_fp8_activation(x):
    """Pack FP8 rows as value bytes followed by one power-of-two scale byte per
    32 values, like ``pack_activation(x)`` without the intermediate arrays."""
    width = x.shape[-1]
    return _pack_kernel()(
        inputs=[x],
        template=[("T", x.dtype), ("D", width)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*x.shape[:-1], width + width // 32)],
        output_dtypes=[mx.uint8],
    )[0]
