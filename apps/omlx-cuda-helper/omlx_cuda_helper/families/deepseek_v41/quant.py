# SPDX-License-Identifier: Apache-2.0
"""FP8 / FP4 activation formats of DeepSeek V4.1.

Groups along the last axis hold E4M3 or E2M1 values and one E8M0 or E4M3
scale. Packed caches store the value bytes, then the scale bytes.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx
import numpy as np

_FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# (threshold, inclusive) between adjacent FP4 levels; ties go to the code with
# an even low bit.
_FP4_STEPS = (
    (0.25, False),
    (0.75, True),
    (1.25, False),
    (1.75, True),
    (2.5, False),
    (3.5, True),
    (5.0, False),
)


def _power_of_two(exponent: mx.array) -> mx.array:
    """Exact fp32 ``2**exponent`` for normal exponents."""
    return ((exponent + 127).astype(mx.uint32) << 23).view(mx.float32)


def _exponent(x: mx.array) -> mx.array:
    """Unbiased exponent field of non-negative fp32 values."""
    return ((x.view(mx.uint32) >> 23) & 0xFF).astype(mx.int32) - 127


def round_fp8(x: mx.array) -> mx.array:
    """Nearest E4M3 value, ties to even, saturated at 448."""
    a = mx.minimum(mx.abs(x), 448.0)
    # The grid step comes from the exponent bits; below 2**-6 it is 2**-9.
    step = _power_of_two(mx.maximum(_exponent(a) - 3, -9))
    return mx.sign(x) * mx.minimum(mx.round(a / step) * step, 448.0)


def fp8_bits(q: mx.array) -> mx.array:
    """E4M3 codes of fp32 values that are already on the E4M3 grid."""
    bits = q.view(mx.uint32)
    exponent = (bits >> 23) & 0xFF
    subnormal = mx.round(mx.abs(q) * 512.0).astype(mx.uint32)
    normal = ((exponent - 120) << 3) | ((bits >> 20) & 7)
    code = mx.where(exponent <= 120, subnormal, normal) | ((bits >> 24) & 0x80)
    return code.astype(mx.uint8)


def _fp4_codes(scaled: mx.array) -> mx.array:
    a = mx.abs(scaled)
    code = mx.zeros(a.shape, mx.uint8)
    for threshold, inclusive in _FP4_STEPS:
        code = code + (a >= threshold if inclusive else a > threshold).astype(mx.uint8)
    return code | ((scaled < 0).astype(mx.uint8) << 3)


def _scaled_groups(x: mx.array, bits: int, group_size: int, e4m3_scale: bool):
    """(values / scale clipped to the format, scale, scale byte) per group."""
    if bits not in (4, 8) or x.shape[-1] % group_size:
        raise ValueError("Invalid activation quantization geometry")
    grouped = x.astype(mx.float32).reshape(*x.shape[:-1], -1, group_size)
    limit = 448.0 if bits == 8 else 6.0
    # Floors of the reference: 1e-4 for FP8, the smallest scale for FP4.
    floor = 1e-4 if bits == 8 else limit * (2.0**-9 if e4m3_scale else 2.0**-126)
    amax = mx.maximum(mx.max(mx.abs(grouped), axis=-1, keepdims=True), floor)
    if e4m3_scale:
        scale = round_fp8(amax / limit)
        scale_byte = fp8_bits(scale)
    else:
        # 2**ceil(log2(amax / limit)) from the bits of the fp32 product.
        product = (amax * mx.array(1.0 / limit, mx.float32)).view(mx.uint32)
        exponent = (
            ((product >> 23) & 0xFF).astype(mx.int32)
            - 127
            + ((product & 0x7FFFFF) != 0).astype(mx.int32)
        )
        scale = _power_of_two(exponent)
        scale_byte = (exponent + 127).astype(mx.uint8)
    return mx.clip(grouped / scale, -limit, limit), scale, scale_byte


def quantize(
    x: mx.array, bits: int = 8, group_size: int = 32, e4m3_scale: bool = False
) -> mx.array:
    """Round trip through the activation format, in the dtype of ``x``."""
    if not x.size:
        return x
    scaled, scale, _ = _scaled_groups(x, bits, group_size, e4m3_scale)
    if bits == 8:
        values = round_fp8(scaled)
    else:
        codes = _fp4_codes(scaled)
        values = mx.array(_FP4_LEVELS, mx.float32)[codes & 7]
        values = mx.where(codes & 8, -values, values)
    return (values * scale).reshape(x.shape).astype(x.dtype)


def pack(
    x: mx.array, bits: int = 8, group_size: int = 32, e4m3_scale: bool = False
) -> mx.array:
    """Rows as value bytes followed by one scale byte per group (uint8)."""
    width = x.shape[-1]
    if not x.size:
        packed_width = width * bits // 8 + width // group_size
        return mx.zeros((*x.shape[:-1], packed_width), mx.uint8)
    scaled, _, scale_byte = _scaled_groups(x, bits, group_size, e4m3_scale)
    scaled = scaled.reshape(x.shape)
    if bits == 8:
        values = fp8_bits(round_fp8(scaled))
    else:
        codes = _fp4_codes(scaled)
        values = codes[..., ::2] | (codes[..., 1::2] << 4)
    return mx.concatenate([values, scale_byte.squeeze(-1)], axis=-1)


@cache
def _fp8_table() -> mx.array:
    code = np.arange(256)
    exponent, mantissa = (code & 127) >> 3, (code & 7).astype(np.float32)
    value = np.where(
        exponent == 0,
        mantissa * 2.0**-9,
        np.ldexp(1.0 + mantissa / 8.0, exponent - 7),
    )
    return mx.array(np.where(code & 128, -value, value).astype(np.float32))


def from_fp8(codes: mx.array) -> mx.array:
    """fp32 values of E4M3 codes."""
    return _fp8_table()[codes]


def unpack(
    packed: mx.array,
    dtype: mx.Dtype,
    bits: int = 8,
    group_size: int = 32,
    e4m3_scale: bool = False,
) -> mx.array:
    """Values of packed rows; exact in bfloat16 for every cache format."""
    groups = packed.shape[-1] // (group_size * bits // 8 + 1)
    width = groups * group_size
    count = width * bits // 8
    codes, scales = packed[..., :count], packed[..., count:]
    if bits == 8:
        values = from_fp8(codes)
    else:
        codes = mx.stack([codes & 15, codes >> 4], axis=-1).reshape(
            *packed.shape[:-1], width
        )
        values = mx.array(_FP4_LEVELS, mx.float32)[codes & 7]
        values = mx.where(codes & 8, -values, values)
    scales = (
        from_fp8(scales) if e4m3_scale else _power_of_two(scales.astype(mx.int32) - 127)
    )
    values = values.reshape(*packed.shape[:-1], groups, group_size) * scales[..., None]
    return values.reshape(*packed.shape[:-1], width).astype(dtype)
