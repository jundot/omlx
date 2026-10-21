# SPDX-License-Identifier: MIT
"""Reference arithmetic for the published FP8/FP4 activation formats.

Persistent KV uses packed FP8/FP4 bytes and one-byte scales. Activation-only
round trips remain available for projection arithmetic and numerical tests.
"""

import re
from functools import cache

import mlx.core as mx
import mlx.nn as nn

from omlx.custom_kernels.quant_matmul import fast as quant_fast

from ..deepseek_v4.switch_layers import _AFFINE_NATIVE_MIN_ROUTES, QuantizedSwitchLinear
from .activation import pack_fp8_activation, quantize_fp8_activation
from .gemv import mxfp8_gemv


@cache
def _affine_blocks_beat_stock():
    """Whether the native affine block-list kernels beat mx.gather_qmm for sorted prefill
    routes: from ~1024 routes on M1 Ultra (#3409), never from the M3 GPU (g15) on."""
    match = re.match(r"applegpu_g(\d+)", mx.device_info().get("architecture", ""))
    return match is None or int(match.group(1)) < 15


def _normal_power_of_two(exponent):
    """Construct normal FP32 scales exactly, including the minimum exponent."""
    return ((exponent + 127).astype(mx.uint32) << 23).view(mx.float32)


def round_fp8(x):
    """E4M3FN, round to nearest even with finite saturation."""
    a = mx.minimum(mx.abs(x.astype(mx.float32)), 448.0)
    # Binade from the fp32 exponent bits (exact); a transcendental log2 can misplace values
    # just below a power of two, whereas DeepSeek casts with a true E4M3 round-to-nearest-even.
    exponent = ((a.view(mx.uint32) >> 23) & 0xFF).astype(mx.int32) - 127
    step = _normal_power_of_two(mx.maximum(exponent - 3, -9))
    return mx.sign(x) * mx.minimum(mx.round(a / step) * step, 448.0)


def _quantize_activation(x, bits=8, group_size=32, e4m3_scale=False):
    dtype, shape = x.dtype, x.shape
    if shape[-1] % group_size:
        raise ValueError("Activation width must divide by quantization group size")
    grouped = x.astype(mx.float32).reshape(*shape[:-1], -1, group_size)
    limit = 448.0 if bits == 8 else 6.0
    # DeepSeek-V4.1-Flash kernel.py floors: act_quant max(amax, 1e-4); fp4_act_quant
    # 6*2^-126 (E8M0 scales) or 6*2^-9 (E4M3 scales).
    minimum = 1e-4 if bits == 8 else limit * (2.0**-9 if e4m3_scale else 2.0**-126)
    amax = mx.maximum(mx.max(mx.abs(grouped), axis=-1, keepdims=True), minimum)
    if e4m3_scale:
        scale = round_fp8(amax / limit)
    else:
        # fast_round_scale: 2^ceil(log2(amax * fp32(1/limit))) from the product's IEEE bits.
        bits32 = (amax * mx.array(1.0 / limit, mx.float32)).view(mx.uint32)
        exponent = ((bits32 >> 23) & 0xFF).astype(mx.int32) - 127 + ((bits32 & 0x7FFFFF) != 0).astype(mx.int32)
        scale = _normal_power_of_two(exponent)
    scaled = mx.clip(grouped / scale, -limit, limit)
    if bits == 8:
        quantized = round_fp8(scaled)
    elif bits == 4:
        # E2M1 midpoint ties go to the code with an even low bit.
        a = mx.abs(scaled)
        q = mx.zeros_like(a)
        for threshold, value, inclusive in [
            (0.25, 0.5, False),
            (0.75, 1.0, True),
            (1.25, 1.5, False),
            (1.75, 2.0, True),
            (2.5, 3.0, False),
            (3.5, 4.0, True),
            (5.0, 6.0, False),
        ]:
            q = mx.where(a >= threshold if inclusive else a > threshold, value, q)
        quantized = mx.sign(scaled) * q
    else:
        raise ValueError("Only FP8 and FP4 are supported")
    return (quantized * scale).reshape(shape).astype(dtype)


_compiled_quantize_activation = mx.compile(_quantize_activation)


def quantize_activation(x, bits=8, group_size=32, e4m3_scale=False):
    if (
        bits == 8
        and group_size == 32
        and not e4m3_scale
        and x.shape[-1] % 32 == 0
        and x.dtype in (mx.float32, mx.float16, mx.bfloat16)
        and mx.default_device() != mx.cpu
    ):
        return quantize_fp8_activation(x) if x.size else x
    return _compiled_quantize_activation(x, bits, group_size, e4m3_scale)


@mx.compile
def _pack_activation_reference(x, bits=8, group_size=32, e4m3_scale=False):
    """Pack each row as value bytes followed by one scale byte per group."""
    if bits not in (4, 8) or x.shape[-1] % group_size:
        raise ValueError("Invalid packed activation geometry")
    if (
        bits == 8
        and group_size == 32
        and not e4m3_scale
        and x.size
        and x.dtype in (mx.float32, mx.float16, mx.bfloat16)
        and mx.default_device() != mx.cpu
    ):
        return pack_fp8_activation(x)
    return _pack_activation(x, bits, group_size, e4m3_scale)


def _pack_activation(x, bits, group_size, e4m3_scale):
    grouped = x.astype(mx.float32).reshape(
        *x.shape[:-1], x.shape[-1] // group_size, group_size
    )
    limit = 448.0 if bits == 8 else 6.0
    # Same floors and scale rules as _quantize_activation (DeepSeek kernel.py).
    minimum = 1e-4 if bits == 8 else limit * (2.0**-9 if e4m3_scale else 2.0**-126)
    amax = mx.maximum(mx.max(mx.abs(grouped), -1), minimum)
    if e4m3_scale:
        scale = round_fp8(amax / limit)
        scale_bytes = mx.to_fp8(scale)
    else:
        bits32 = (amax * mx.array(1.0 / limit, mx.float32)).view(mx.uint32)
        exponent = ((bits32 >> 23) & 0xFF).astype(mx.int32) - 127 + ((bits32 & 0x7FFFFF) != 0).astype(mx.int32)
        scale = _normal_power_of_two(exponent)
        scale_bytes = (exponent + 127).astype(mx.uint8)
    scaled = mx.clip(grouped / scale[..., None], -limit, limit).reshape(x.shape)
    if bits == 8:
        values = mx.to_fp8(round_fp8(scaled))
    else:
        a = mx.abs(scaled)
        code = mx.zeros(a.shape, mx.uint8)
        for i, (threshold, inclusive) in enumerate(
            [
                (0.25, False),
                (0.75, True),
                (1.25, False),
                (1.75, True),
                (2.5, False),
                (3.5, True),
                (5.0, False),
            ],
            1,
        ):
            code = mx.where(a >= threshold if inclusive else a > threshold, i, code)
        code = code.astype(mx.uint8) | ((scaled < 0).astype(mx.uint8) << 3)
        values = code[..., ::2] | (code[..., 1::2] << 4)
    return mx.concatenate([values, scale_bytes], -1)


def pack_activation(x, bits=8, group_size=32, e4m3_scale=False):
    if (
        mx.default_device() == mx.gpu
        and x.size
        and x.shape[-1] % 32 == 0
        and (bits, group_size, e4m3_scale)
        in ((8, 32, False), (4, 32, False), (4, 16, True))
        and x.dtype in (mx.float32, mx.float16, mx.bfloat16)
    ):
        from .packed_kv import fused_pack_activation

        return fused_pack_activation(x, bits, group_size, e4m3_scale)
    return _pack_activation_reference(x, bits, group_size, e4m3_scale)


def unpack_activation(
    packed, bits=8, group_size=32, e4m3_scale=False, dtype=mx.bfloat16
):
    """Decode already-selected rows; never expand a persistent KV cache."""
    groups = packed.shape[-1] // (group_size * bits // 8 + 1)
    width = groups * group_size
    nbytes = width * bits // 8
    values, scales = packed[..., :nbytes], packed[..., nbytes:]
    if bits == 8:
        values = mx.from_fp8(values, dtype=mx.float32)
    else:
        codes = mx.stack([values & 15, values >> 4], -1).reshape(
            *packed.shape[:-1], width
        )
        levels = mx.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], mx.float32)
        values = levels[codes & 7] * mx.where(codes & 8, -1.0, 1.0)
    scales = (
        mx.from_fp8(scales, dtype=mx.float32)
        if e4m3_scale
        else _normal_power_of_two(scales.astype(mx.int32) - 127)
    )
    return (
        (values.reshape(*packed.shape[:-1], groups, group_size) * scales[..., None])
        .reshape(*packed.shape[:-1], width)
        .astype(dtype)
    )


class QuantizedProjection(QuantizedSwitchLinear):
    """Packed MXFP or affine weights with optional official FP8 activations."""

    def __init__(
        self,
        weight,
        scales,
        bits,
        mode,
        biases=None,
        group_size=32,
        quantize_input=True,
    ):
        nn.Module.__init__(self)
        self.weight, self.scales = weight, scales
        self.bits, self.mode = bits, mode
        self.group_size = group_size
        self.quantize_input = quantize_input
        if biases is not None:
            self.biases = biases

    @property
    def input_dims(self):
        return self.weight.shape[-1] * 32 // self.bits

    def _can_use_affine_blocks(self, x, sorted_indices, dtype=None):
        dtype = dtype or x.dtype
        return (
            sorted_indices
            and x.ndim == 3
            and x.shape[-2] == 1
            and x.shape[0] >= _AFFINE_NATIVE_MIN_ROUTES
            and _affine_blocks_beat_stock()
            and dtype in (mx.float16, mx.bfloat16)
            and self.group_size == 64
            and self.bits in (2, 3, 4, 6, 8)
            and self._has_affine_metadata_dtype(dtype)
            and quant_fast.has_symbol("affine_gather_qmm_blocks")
        )

    def _native_block_kind(self, x, sorted_indices, dtype=None):
        if (dtype or x.dtype) not in (mx.float16, mx.bfloat16):
            return None
        return super()._native_block_kind(x, sorted_indices, dtype=dtype)

    def __call__(self, x, indices=None, sorted_indices=False, block_plan=None):
        if self.quantize_input:
            x = quantize_activation(x)
        return self.project_quantized(x, indices, sorted_indices, block_plan)

    def project_quantized(self, x, indices=None, sorted_indices=False, block_plan=None):
        """Project an input whose activation quantization is already complete."""
        if indices is None:
            if self.mode == "mxfp8":
                y = mxfp8_gemv(x, self.weight, self.scales)
                if y is not None:
                    return y
            return mx.quantized_matmul(
                x,
                self.weight,
                self.scales,
                self.get("biases"),
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
        return super().__call__(
            x, indices, sorted_indices=sorted_indices, block_plan=block_plan
        )
