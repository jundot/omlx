# SPDX-License-Identifier: MIT
"""Fused reference-format FP8/FP4 KV packing; preserve rounding boundaries."""

from functools import cache
import mlx.core as mx

_SOURCE = r"""
 const uint d = thread_position_in_grid.x;
 const uint row = thread_position_in_grid.y;
 if (d >= D) return;
 const uint lane = thread_index_in_simdgroup;
 const float v = float(x[row * D + d]);
 auto power2 = [](int e) { return as_type<float>(uint(e + 127) << 23); };
 auto round8 = [&](float a) {
   float b = min(abs(a), 448.0f);
   // E4M3 binade from the exponent bits (exact), subnormal grid 2^-9 below 2^-6.
   int e = max(int((as_type<uint>(b) >> 23) & 255u) - 127 - 3, -9);
   float step = power2(e);
   return sign(a) * min(rint(b / step) * step, 448.0f);
 };
 auto code8 = [](float q) {
   uint b = as_type<uint>(q);
   uchar s = uchar((b >> 31) << 7);
   uint e = (b >> 23) & 255u;
   if (e <= 120u) return uchar(s | uchar(int(rint(abs(q) * 512.0f))));
   return uchar(s | uchar(((e - 120u) << 3) | ((b >> 20) & 7u)));
 };
 float amax = abs(v);
 for (uint offset = G / 2; offset > 0; offset >>= 1)
   amax = max(amax, simd_shuffle_xor(amax, offset));
 constexpr float limit = BITS == 8 ? 448.0f : 6.0f;
 amax = max(amax, as_type<float>(uint(MIN_BITS)));
 float scale;
 uchar sb;
 if (E4S) {
   // kernel.py fp4_quant_kernel: s = Cast(FP8, amax / fp4_max) -- a division, not * (1/6).
   scale = round8(amax / limit);
   sb = code8(scale);
 } else {
   // kernel.py fast_round_scale: 2^ceil(log2(amax * fp32(1/limit))) from the product's bits.
   const uint pb = as_type<uint>(amax * (1.0f / limit));
   int e = int((pb >> 23) & 255u) - 127 + int((pb & 0x7FFFFFu) != 0u);
   scale = power2(e);
   sb = uchar(e + 127);
 }
 float scaled = clamp(v / scale, -limit, limit);
 constexpr uint BYTES = D * BITS / 8;
 constexpr uint WIDTH = BYTES + D / G;
 if (BITS == 8) {
   y[row * WIDTH + d] = code8(round8(scaled));
 } else {
   float a = abs(scaled);
   uchar code = uchar(a > 0.25f) + uchar(a >= 0.75f) + uchar(a > 1.25f)
              + uchar(a >= 1.75f) + uchar(a > 2.5f) + uchar(a >= 3.5f)
              + uchar(a > 5.0f);
   code |= uchar(scaled < 0.0f) << 3;
   uint other = simd_shuffle_xor(uint(code), 1);
   if ((d & 1u) == 0u) y[row * WIDTH + d / 2] = code | uchar(other << 4);
 }
 if (lane % G == 0) y[row * WIDTH + BYTES + d / G] = sb;
"""


@mx.compile
def _reference_minimum(x, bits, e4m3_scale):
    # DeepSeek-V4.1-Flash kernel.py: act_quant max(amax, 1e-4); fp4_act_quant
    # 6*2^-9 (E4M3 scales) or 6*2^-126 (E8M0 scales).
    if bits == 8:
        return mx.maximum(x, 1e-4)
    return mx.maximum(x, 6.0 * (2.0**-9 if e4m3_scale else 2.0**-126))


@cache
def _minimum_bits(bits, e4m3_scale):
    # MLX compiled scalar literals can round differently from Metal constants.
    # Capture the reference value once, retaining even zero-group scale bytes.
    return int(
        _reference_minimum(mx.zeros((1,), mx.float32), bits, e4m3_scale)
        .view(mx.uint32)
        .item()
    )


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="v41_pack_kv_exact", input_names=["x"], output_names=["y"], source=_SOURCE
    )


def fused_pack_activation(x, bits=8, group_size=32, e4m3_scale=False):
    dim = x.shape[-1]
    rows = x.size // dim
    width = dim * bits // 8 + dim // group_size
    return _kernel()(
        inputs=[x],
        template=[
            ("D", dim),
            ("BITS", bits),
            ("G", group_size),
            ("E4S", e4m3_scale),
            ("MIN_BITS", _minimum_bits(bits, e4m3_scale)),
        ],
        grid=(dim, rows, 1),
        threadgroup=(min(256, dim), 1, 1),
        output_shapes=[(*x.shape[:-1], width)],
        output_dtypes=[mx.uint8],
    )[0]
