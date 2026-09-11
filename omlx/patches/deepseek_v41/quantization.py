# SPDX-License-Identifier: MIT
"""Reference arithmetic for the published FP8/FP4 activation formats.

Persistent KV uses packed FP8/FP4 bytes and one-byte scales. Activation-only
round trips remain available for projection arithmetic and numerical tests.
"""

import mlx.core as mx
import mlx.nn as nn

from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast

from ..deepseek_v4.switch_layers import _AFFINE_NATIVE_MIN_ROUTES, QuantizedSwitchLinear
from .activation import quantize_fp8_activation


def _normal_power_of_two(exponent):
    """Construct normal FP32 scales exactly, including the minimum exponent."""
    return ((exponent + 127).astype(mx.uint32) << 23).view(mx.float32)


def round_fp8(x):
    """E4M3FN, round to nearest even with finite saturation."""
    a = mx.minimum(mx.abs(x.astype(mx.float32)), 448.0)
    exponent = mx.floor(mx.log2(mx.maximum(a, 2.0**-9)))
    step = _normal_power_of_two(mx.maximum(exponent - 3, -9))
    return mx.sign(x) * mx.minimum(mx.round(a / step) * step, 448.0)


def _quantize_activation(x, bits=8, group_size=32, e4m3_scale=False):
    dtype, shape = x.dtype, x.shape
    if shape[-1] % group_size:
        raise ValueError("Activation width must divide by quantization group size")
    grouped = x.astype(mx.float32).reshape(*shape[:-1], -1, group_size)
    limit = 448.0 if bits == 8 else 6.0
    minimum = limit * (2.0**-9 if e4m3_scale else 2.0**-126)
    amax = mx.maximum(mx.max(mx.abs(grouped), axis=-1, keepdims=True), minimum)
    scale = (
        round_fp8(amax / limit)
        if e4m3_scale
        else _normal_power_of_two(mx.maximum(mx.ceil(mx.log2(amax / limit)), -126))
    )
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
def pack_activation(x, bits=8, group_size=32, e4m3_scale=False):
    """Pack each row as value bytes followed by one scale byte per group."""
    if bits not in (4, 8) or x.shape[-1] % group_size:
        raise ValueError("Invalid packed activation geometry")
    grouped = x.astype(mx.float32).reshape(
        *x.shape[:-1], x.shape[-1] // group_size, group_size
    )
    limit = 448.0 if bits == 8 else 6.0
    minimum = limit * (2.0**-9 if e4m3_scale else 2.0**-126)
    amax = mx.maximum(mx.max(mx.abs(grouped), -1), minimum)
    if e4m3_scale:
        scale = round_fp8(amax / limit)
        scale_bytes = mx.to_fp8(scale)
    else:
        exponent = mx.maximum(mx.ceil(mx.log2(amax / limit)), -126)
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
            and dtype in (mx.float16, mx.bfloat16)
            and self.group_size == 64
            and self.bits in (2, 3, 4, 6, 8)
            and self._has_affine_metadata_dtype(dtype)
            and glm_fast.has_symbol("deepseek_affine_gather_qmm_blocks")
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
