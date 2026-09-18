"""FP8 activation fusion preserves quantization boundaries and input layouts."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from omlx.patches.deepseek_v41.activation import quantize_swiglu_activation
from omlx.patches.deepseek_v41.quantization import (
    _compiled_quantize_activation,
    quantize_activation,
)


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
def test_fp8_activation_midpoints_and_scale_boundaries(dtype):
    codes = mx.from_fp8(mx.arange(127, dtype=mx.uint8), dtype=mx.float32)
    levels = np.asarray(codes)
    mid = (levels[:-1] + levels[1:]) / 2
    values = np.concatenate(
        [
            mid,
            np.nextafter(mid, np.float32(-np.inf)),
            np.nextafter(mid, np.float32(np.inf)),
        ]
    )
    rows = np.zeros((len(values) * 2, 32), np.float32)
    rows[:, 0] = np.concatenate([values, -values])
    rows[:, -1] = 448
    exponents = [-10, 0, 6] if dtype == mx.float16 else [-110, -10, 0, 10, 110]
    for exponent in exponents:
        x = mx.array(rows * np.float32(2.0**exponent)).astype(dtype)
        expected = _compiled_quantize_activation(x)
        actual = quantize_activation(x)
        np.testing.assert_array_equal(
            actual.astype(mx.float32), expected.astype(mx.float32)
        )
    # Move the group maximum across power-of-two scale transitions.
    anchors = np.array(
        [
            np.nextafter(np.float32(448), np.float32(0)),
            448,
            np.nextafter(np.float32(448), np.float32(np.inf)),
        ],
        np.float32,
    )
    x = mx.array(np.repeat(anchors[:, None], 32, axis=1)).astype(dtype)
    np.testing.assert_array_equal(
        quantize_activation(x).astype(mx.float32),
        _compiled_quantize_activation(x).astype(mx.float32),
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("length", [1, 3, 33])
def test_fp8_activation_noncontiguous_rows_and_repeat(dtype, length):
    mx.random.seed(711)
    x = mx.random.normal((2, 64, length)).astype(dtype).transpose(0, 2, 1)
    expected = _compiled_quantize_activation(x)
    actual = quantize_activation(x)
    np.testing.assert_array_equal(
        actual.astype(mx.float32), expected.astype(mx.float32)
    )
    np.testing.assert_array_equal(
        actual.astype(mx.float32), quantize_activation(x).astype(mx.float32)
    )


def test_fp8_activation_empty_and_invalid_width():
    x = mx.zeros((1, 0, 64), mx.bfloat16)
    assert quantize_activation(x).shape == x.shape
    with pytest.raises(ValueError, match="width"):
        quantize_activation(mx.zeros((1, 33)))


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("limit", [0.0, 3.5, 10.0])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("input_dtype", [mx.float32, mx.bfloat16])
def test_swiglu_fusion_preserves_weighted_intermediate_rounding(
    dtype, limit, weighted, input_dtype
):
    mx.random.seed(749)
    gate = (mx.random.normal((17, 1, 128)) * 9).astype(input_dtype)
    up = (mx.random.normal(gate.shape) * 11).astype(input_dtype)
    weights = mx.linspace(0, 2, 17) if weighted else None
    gate_fp32, up_fp32 = gate.astype(mx.float32), up.astype(mx.float32)
    clipped_gate = mx.minimum(gate_fp32, limit) if limit else gate_fp32
    clipped_up = mx.clip(up_fp32, -limit, limit) if limit else up_fp32
    value = nn.silu(clipped_gate) * clipped_up
    if weights is not None:
        value *= weights[:, None, None]
    expected = _compiled_quantize_activation(value.astype(dtype))
    actual = quantize_swiglu_activation(gate, up, weights, dtype, limit)
    np.testing.assert_array_equal(
        actual.astype(mx.float32), expected.astype(mx.float32)
    )
    np.testing.assert_array_equal(
        actual.astype(mx.float32),
        quantize_swiglu_activation(gate, up, weights, dtype, limit).astype(mx.float32),
    )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("limit", [0.0, 3.5, 10.0])
def test_paired_swiglu_preserves_row_and_quantization_boundaries(
    dtype, weighted, limit
):
    from omlx.patches.deepseek_v41.activation import quantize_paired_swiglu_activation

    mx.random.seed(882)
    pair = (mx.random.normal((17, 1, 256)) * 11).astype(dtype)
    weights = mx.linspace(0, 2, 17) if weighted else None
    expected = quantize_swiglu_activation(
        pair[..., :128], pair[..., 128:], weights, dtype, limit
    )
    actual = quantize_paired_swiglu_activation(pair, weights, dtype, limit)
    repeated = quantize_paired_swiglu_activation(pair, weights, dtype, limit)
    np.testing.assert_array_equal(
        actual.astype(mx.float32), expected.astype(mx.float32)
    )
    np.testing.assert_array_equal(
        actual.astype(mx.float32), repeated.astype(mx.float32)
    )


@pytest.mark.parametrize("device", [mx.cpu, mx.gpu])
def test_normal_scales_are_exact_and_nonzero(device):
    from omlx.patches.deepseek_v41.quantization import _normal_power_of_two

    with mx.stream(device):
        exponent = mx.arange(-126, 128, dtype=mx.float32)
        actual = _normal_power_of_two(exponent)
        compiled = mx.compile(_normal_power_of_two)(exponent)
        expected = np.exp2(np.arange(-126, 128, dtype=np.float32))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(compiled, expected)


@pytest.mark.parametrize("bits,group,e4m3", [(8, 32, False), (4, 32, False), (4, 16, True)])
def test_zero_activation_groups_remain_zero(bits, group, e4m3):
    from omlx.patches.deepseek_v41.quantization import (
        _quantize_activation,
        pack_activation,
        unpack_activation,
    )

    x = mx.zeros((3, 128), mx.float32)
    for result in (
        _quantize_activation(x, bits, group, e4m3),
        _compiled_quantize_activation(x, bits, group, e4m3),
        quantize_activation(x, bits, group, e4m3),
        unpack_activation(pack_activation(x, bits, group, e4m3), bits, group, e4m3, mx.float32),
    ):
        # This rejects NaNs explicitly, including matching NaNs in both paths.
        assert mx.all(mx.isfinite(result)).item()
        np.testing.assert_array_equal(result, np.zeros((3, 128), np.float32))


def test_fp8_kernel_preserves_exact_power_of_two_scaled_values():
    # Every scale whose maximum finite FP8 value fits in a normal FP32.
    # A transcendental exp2 approximation can shift these values and FP8 ties.
    exponents = np.arange(-126, 119, dtype=np.int32)
    rows = np.repeat(np.ldexp(np.float32(448), exponents)[:, None], 32, axis=1)
    actual = quantize_activation(mx.array(rows))
    np.testing.assert_array_equal(actual, rows)
