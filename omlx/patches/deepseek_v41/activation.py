# SPDX-License-Identifier: MIT
"""Single-pass FP8 activation round trips with one SIMD group per scale group."""

from functools import cache

import mlx.core as mx

_SCALE = r"""
    // DeepSeek-V4.1-Flash inference/kernel.py act_quant: amax = max(amax, 1e-4); the ue8m0
    // scale is fast_round_scale = 2^ceil(log2(amax * fp32(1/448))) taken from the product's
    // IEEE bits (exponent, +1 if any mantissa bit is set) rather than a transcendental log2.
    const float amax = max(simd_max(abs(v)), 1e-4f);
    const uint scale_bits = as_type<uint>(amax * (1.0f / 448.0f));
    const int scale_exponent = int((scale_bits >> 23) & 0xFFu) - 127 + int((scale_bits & 0x7FFFFFu) != 0u);
    // Integer powers of two must be exact, including FP8 rounding ties.
    const float scale = as_type<float>(uint(scale_exponent + 127) << 23);
    const float scaled = clamp(v / scale, -448.0f, 448.0f);
    const float a = abs(scaled);
    // E4M3 grid step from the binade of a, read from its exponent bits (exact; a log2 can
    // misplace values just below a power of two). Subnormal grid below 2^-6 is 2^-9.
    const int step_exponent = max(int((as_type<uint>(a) >> 23) & 0xFFu) - 127 - 3, -9);
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


# RoPE on the last RD dims of each D-wide row (adjacent pairs, fp32, result in the input dtype);
# QUANT then applies act_quant to the rotated row. Thread t owns 8 contiguous elements of the
# [B, L, ROW / D, D] output, so an act_quant group of 32 is 4 adjacent lanes. HEADS_MAJOR reads
# x as [B, ROW / D, L, D] (the native attention output) without a transposing copy.
_ROPE_SOURCE = r"""
    const uint i = thread_position_in_grid.x * 8;
    if (i >= uint(N)) return;
    uint src = i;
    if (HEADS_MAJOR) {
        const uint h = (i / D) % (ROW / D), l = (i / ROW) % L, b = i / (ROW * L);
        src = ((b * (ROW / D) + h) * L + l) * D + i % D;
    }
    float v[8];
    for (int j = 0; j < 8; ++j) v[j] = float(x[src + j]);
    const uint d = i % D;
    if (d >= uint(D - RD)) {
        const uint p = ((i / ROW) % L) * (RD / 2) + (d - (D - RD)) / 2;
        for (int j = 0; j < 8; j += 2) {
            const float cs = cosv[p + j / 2], sn = INVERSE ? -sinv[p + j / 2] : sinv[p + j / 2];
            const float a = v[j], b = v[j + 1];
            v[j] = float(T(a * cs - b * sn));
            v[j + 1] = float(T(a * sn + b * cs));
        }
    }
    if (!QUANT) {
        for (int j = 0; j < 8; ++j) y[i + j] = T(v[j]);
        return;
    }
    // act_quant (kernel.py): amax over the 32-group held by 4 adjacent lanes, ue8m0 scale,
    // E4M3 round to nearest even, as _SCALE / _ROUND above.
    float amax = 0.0f;
    for (int j = 0; j < 8; ++j) amax = max(amax, abs(v[j]));
    amax = max(amax, simd_shuffle_xor(amax, 1u));
    amax = max(max(amax, simd_shuffle_xor(amax, 2u)), 1e-4f);
    const uint scale_bits = as_type<uint>(amax * (1.0f / 448.0f));
    const int scale_exponent = int((scale_bits >> 23) & 0xFFu) - 127 + int((scale_bits & 0x7FFFFFu) != 0u);
    const float scale = as_type<float>(uint(scale_exponent + 127) << 23);
    for (int j = 0; j < 8; ++j) {
        const float scaled = clamp(v[j] / scale, -448.0f, 448.0f);
        const float a = abs(scaled);
        const int step_exponent = max(int((as_type<uint>(a) >> 23) & 0xFFu) - 127 - 3, -9);
        const float step = as_type<float>(uint(step_exponent + 127) << 23);
        y[i + j] = T(sign(scaled) * min(rint(a / step) * step, 448.0f) * scale);
    }
"""


@cache
def _rope_kernel():
    return mx.fast.metal_kernel(
        name="v41_rope_activation",
        input_names=["x", "cosv", "sinv"],
        output_names=["y"],
        source=_ROPE_SOURCE,
    )


def rope_activation(x, cos, sin, inverse=False, quantize=False, heads_major=False):
    """_ROPE_SOURCE for cos / sin [L, RD / 2] (fp32); see language.rope for the arguments."""
    rd = cos.shape[-1] * 2
    length = x.shape[2] if heads_major else x.shape[1]
    shape = (x.shape[0], length, x.shape[1], x.shape[3]) if heads_major else x.shape
    return _rope_kernel()(
        inputs=[x, cos, sin],
        template=[
            ("T", x.dtype),
            ("N", x.size),
            ("D", x.shape[-1]),
            ("RD", rd),
            ("L", length),
            ("ROW", x.size // (x.shape[0] * length)),
            ("INVERSE", bool(inverse)),
            ("QUANT", bool(quantize)),
            ("HEADS_MAJOR", bool(heads_major)),
        ],
        grid=(x.size // 8, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[x.dtype],
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
