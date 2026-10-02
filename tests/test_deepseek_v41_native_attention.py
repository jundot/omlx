import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v41.kernels import packed_sparse_attention
from omlx.patches.deepseek_v41.quantization import pack_activation

pytestmark = pytest.mark.skipif(
    not fast.has_symbol("deepseek_v41_packed_attention"),
    reason="Packed attention extension is not built",
)


@pytest.mark.parametrize(
    "start,length,pooled_count", [(0, 7, 0), (0, 7, 3), (131, 3, 40), (4096, 7, 1024)]
)
def test_native_packed_attention_causal_window_and_fp32_sinks(
    start, length, pooled_count
):
    mx.random.seed(193)
    q = mx.random.normal((1, length, 64, 512)).astype(mx.bfloat16)
    old = min(start, 128)
    window = pack_activation(mx.random.normal((1, old + length, 512)))
    pooled = pack_activation(mx.random.normal((1, pooled_count, 512)), 4, 16, True)
    if pooled_count:
        ci = mx.broadcast_to(
            mx.array(
                [-1, 0, 1, pooled_count - 1, pooled_count, pooled_count + 2], mx.int32
            ),
            (1, length, 6),
        )
    else:
        ci = mx.zeros((1, length, 0), mx.int32)
    positions = mx.arange(start, start + length)
    local = positions[:, None] - 127 + mx.arange(128)
    wi = mx.where(local >= max(0, start - old), local - (start - old), -1)[None]
    valid = (
        (ci >= 0) & (ci < pooled_count) & (ci < ((positions + 1) // 4)[None, :, None])
    )
    masked = mx.where(valid, ci, -1)
    sinks = mx.linspace(-4.123456, 4.234567, 64)
    expected = packed_sparse_attention(q, window, pooled, wi, masked, sinks, 512**-0.5)
    args = (
        q.transpose(0, 2, 1, 3),
        window[:, None],
        pooled,
        ci[:, None].astype(mx.uint32),
        sinks,
        512**-0.5,
        start,
        4,
        128,
    )
    actual = fast.deepseek_v41_packed_attention(*args).transpose(0, 2, 1, 3)
    repeated = fast.deepseek_v41_packed_attention(*args).transpose(0, 2, 1, 3)
    np.testing.assert_allclose(
        actual.astype(mx.float32), expected.astype(mx.float32), atol=0.008, rtol=0.008
    )
    assert mx.array_equal(actual, repeated).item()
    assert window.dtype == pooled.dtype == mx.uint8
    assert mx.all(mx.isfinite(actual)).item()


def test_native_packed_attention_does_not_round_fp32_sinks():
    q = mx.zeros((1, 64, 2, 512), mx.bfloat16)
    window = pack_activation(mx.ones((1, 2, 512)))[:, None]
    pooled = mx.zeros((1, 0, 288), mx.uint8)
    topk = mx.zeros((1, 1, 2, 0), mx.uint32)
    sinks = mx.linspace(-4.123456, 4.234567, 64)
    actual = fast.deepseek_v41_packed_attention(
        q, window, pooled, topk, sinks, 512**-0.5, 0, 4, 128
    )
    count = mx.array([1, 2], mx.float32)[None, :]
    expected = (count / (count + mx.exp(sinks[:, None]))).astype(mx.bfloat16)
    rounded = (
        count / (count + mx.exp(sinks.astype(mx.bfloat16).astype(mx.float32)[:, None]))
    ).astype(mx.bfloat16)
    assert not mx.array_equal(expected, rounded).item()
    assert mx.array_equal(actual[0, :, :, 0], expected).item()


@pytest.mark.parametrize("scale_byte", [64, 126, 127, 128, 192])
def test_native_packed_attention_fp8_codes_and_power_of_two_scales(scale_byte):
    codes = np.resize(
        np.concatenate([np.arange(127), np.arange(128, 255)]).astype(np.uint8), 512
    )
    magnitude = (codes & 127).astype(np.int32)
    exponent, mantissa = magnitude >> 3, magnitude & 7
    values = np.where(
        exponent == 0,
        np.ldexp(mantissa.astype(np.float32), -9),
        np.ldexp(1 + mantissa.astype(np.float32) / 8, exponent - 7),
    )
    values *= np.where(codes & 128, -1, 1)
    values = np.ldexp(values, scale_byte - 127).astype(np.float32)
    row = np.concatenate([codes, np.full(16, scale_byte, np.uint8)])
    window = mx.array(np.tile(row, (1, 1, 2, 1)))
    actual = fast.deepseek_v41_packed_attention(
        mx.zeros((1, 64, 2, 512), mx.bfloat16),
        window,
        mx.zeros((1, 0, 288), mx.uint8),
        mx.zeros((1, 1, 2, 0), mx.uint32),
        mx.full((64,), -100.0),
        512**-0.5,
        0,
        4,
        128,
    )
    expected = mx.broadcast_to(mx.array(values).astype(mx.bfloat16), actual.shape)
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("length", [2, 4, 6, 8])
@pytest.mark.parametrize("pooled_count", [0, 40, 1024])
def test_short_packed_attention_matches_independent_decode(length, pooled_count):
    mx.random.seed(291)
    q = mx.random.normal((1, length, 64, 512)).astype(mx.bfloat16)
    window = pack_activation(mx.random.normal((1, 128 + length, 512)))
    pooled = pack_activation(mx.random.normal((1, pooled_count, 512)), 4, 16, True)
    wi = mx.arange(1, 129)[None, None, :] + mx.arange(length)[None, :, None]
    ci = mx.broadcast_to(mx.arange(pooled_count), (1, length, pooled_count))
    # Distinct per-query visibility, including invalid and entirely masked rows.
    ci = mx.where(ci < mx.arange(length)[None, :, None] * 17, ci, -1)
    sinks = mx.linspace(-4.123456, 4.234567, 64)
    args = (q, window, pooled, wi, ci, sinks, 512**-0.5)
    actual = packed_sparse_attention(*args)
    expected = mx.concatenate(
        [
            packed_sparse_attention(
                q[:, i : i + 1],
                window,
                pooled,
                wi[:, i : i + 1],
                ci[:, i : i + 1],
                sinks,
                512**-0.5,
            )
            for i in range(length)
        ],
        axis=1,
    )
    assert mx.array_equal(actual, expected).item()
    assert mx.array_equal(actual, packed_sparse_attention(*args)).item()
    assert mx.all(mx.isfinite(actual)).item()
