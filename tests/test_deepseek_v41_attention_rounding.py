"""Independent 64-key online-softmax oracle with the official BF16 PV boundary."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches.deepseek_v41.kernels import packed_sparse_attention
from omlx.patches.deepseek_v41.quantization import pack_activation, unpack_activation


def reference(q, window, pooled, wi, ci, sink, scale, round_probabilities=True):
    keys = []
    masks = []
    for packed, ids, bits, group, e4 in [
        (window, wi, 8, 32, False),
        (pooled, ci, 4, 16, True),
    ]:
        if not ids.shape[-1]:
            continue
        valid = (ids >= 0) & (ids < packed.shape[1])
        if packed.shape[1]:
            values = unpack_activation(packed, bits, group, e4, mx.bfloat16)[0]
            values = values[mx.clip(ids, 0, packed.shape[1] - 1)].astype(mx.float32)
        else:
            values = mx.zeros((*ids.shape, q.shape[-1]))
        keys.append(values)
        masks.append(valid)
    if not keys:
        return mx.zeros_like(q)
    keys, masks = mx.concatenate(keys, -2), mx.concatenate(masks, -1)
    maximum = mx.full(q.shape[:-1], -1e30)
    denominator = mx.zeros(q.shape[:-1])
    numerator = mx.zeros(q.shape)
    for start in range(0, keys.shape[-2], 64):
        kv = keys[:, :, start : start + 64]
        score = q.astype(mx.float32) @ kv.swapaxes(-1, -2) * scale
        score = mx.where(masks[:, :, None, start : start + 64], score, -float("inf"))
        new_maximum = mx.maximum(maximum, mx.max(score, -1))
        correction = mx.exp(maximum - new_maximum)
        p = mx.exp(score - new_maximum[..., None])
        denominator = denominator * correction + mx.sum(p, -1)
        if round_probabilities:
            p = p.astype(mx.bfloat16).astype(mx.float32)
        numerator = numerator * correction[..., None] + p @ kv
        maximum = new_maximum
    return (numerator / (denominator + mx.exp(sink - maximum))[..., None]).astype(
        q.dtype
    )


@pytest.mark.parametrize(
    "length,window_count,pooled_count",
    [(1, 128, 513), (7, 135, 129), (17, 17, 19), (65, 65, 77), (129, 129, 97)],
)
def test_packed_attention_preserves_online_bf16_probability_rounding(
    length, window_count, pooled_count
):
    mx.random.seed(103 + length)
    q = mx.random.normal((1, length, 64, 512)).astype(mx.bfloat16)
    window = pack_activation(mx.random.normal((1, window_count, 512)))
    pooled = pack_activation(mx.random.normal((1, pooled_count, 512)), 4, 16, True)
    start = window_count - length
    pos = mx.arange(start, start + length)
    width = min(length, 128) if start == 0 else 128
    first = mx.maximum(pos[:, None] - 127, 0) if start == 0 else pos[:, None] - 127
    wi = first + mx.arange(width)
    wi = mx.where((wi >= 0) & (wi <= pos[:, None]), wi, -1)[None]
    ci = mx.broadcast_to(mx.arange(pooled_count), (1, length, pooled_count))
    ci = mx.where(ci < (pos[None, :, None] + 1) // 4, ci, -1)
    sink = mx.linspace(-2.1, 3.7, 64)
    args = (q, window, pooled, wi, ci, sink, 512**-0.5)
    expected = reference(*args)
    unrounded = reference(*args, round_probabilities=False)
    actual = packed_sparse_attention(*args)
    mx.eval(expected, unrounded, actual)

    def rms(value):
        return mx.sqrt(
            mx.mean((value.astype(mx.float32) - expected.astype(mx.float32)) ** 2)
        ).item()

    assert rms(actual) < rms(unrounded) * 0.2
    assert mx.array_equal(actual, packed_sparse_attention(*args)).item()
    if length > 1 and fast.has_symbol("deepseek_v41_packed_attention"):
        native = fast.deepseek_v41_packed_attention(
            q.transpose(0, 2, 1, 3),
            window[:, None],
            pooled,
            ci[:, None].astype(mx.uint32),
            sink,
            512**-0.5,
            start,
            4,
            128,
        ).transpose(0, 2, 1, 3)
        mx.eval(native)
        assert rms(native) < rms(unrounded) * 0.2


def test_masked_tiles_and_sink_stay_finite():
    q = mx.ones((1, 2, 5, 32), mx.bfloat16)
    window = mx.zeros((1, 0, 33), mx.uint8)
    pooled = mx.zeros((1, 0, 18), mx.uint8)
    indices = mx.full((1, 2, 193), -1, mx.int32)
    empty = mx.zeros((1, 2, 0), mx.int32)
    result = packed_sparse_attention(
        q, window, pooled, indices, empty, mx.array([-10000, -3, 0, 4, 10000]), 1
    )
    mx.eval(result)
    np.testing.assert_array_equal(result.astype(mx.float32), 0)


def test_rounding_tracks_growing_and_strided_kv():
    mx.random.seed(91)
    q = mx.random.normal((1, 1, 5, 64))[..., ::2].astype(mx.bfloat16)
    window = pack_activation(mx.random.normal((1, 256, 32)))[:, ::2]
    wi = mx.arange(128)[None, None]
    ci = mx.arange(34, dtype=mx.int32)[::2][None, None]
    sink = mx.linspace(-2, 2, 5)
    for count in [0, 4, 7, 70, 129, 4]:
        pooled = pack_activation(mx.random.normal((1, count * 2, 32)), 4, 16, True)[
            :, ::2
        ]
        args = (q, window, pooled, wi, ci, sink, 32**-0.5)
        actual, expected = packed_sparse_attention(*args), reference(*args)
        mx.eval(actual, expected)
        np.testing.assert_allclose(
            actual.astype(mx.float32),
            expected.astype(mx.float32),
            atol=0.002,
            rtol=0.002,
        )


@pytest.mark.parametrize("count", [1025, 2051])
def test_rounding_across_threadgroup_capacity(count):
    mx.random.seed(123)
    q = mx.random.normal((1, 1, 5, 32)).astype(mx.bfloat16)
    window = pack_activation(mx.random.normal((1, 128, 32)))
    pooled = pack_activation(mx.random.normal((1, count, 32)), 4, 16, True)
    wi = mx.arange(128)[None, None]
    ci = mx.arange(count)[None, None]
    args = (q, window, pooled, wi, ci, mx.linspace(-3, 3, 5), 32**-0.5)
    actual, expected = packed_sparse_attention(*args), reference(*args)
    mx.eval(actual, expected)
    np.testing.assert_allclose(
        actual.astype(mx.float32), expected.astype(mx.float32), atol=0.002, rtol=0.002
    )


@pytest.mark.parametrize("heads,dim,length", [(9, 96, 1), (64, 512, 1), (8, 32, 3)])
def test_matrix_rounding_with_full_candidate_lists(heads, dim, length):
    mx.random.seed(71)
    q = mx.random.normal((1, length, heads, dim)).astype(mx.bfloat16)
    window = pack_activation(mx.random.normal((1, 128, dim)))
    pooled = pack_activation(mx.random.normal((1, 513, dim)), 4, 16, True)
    wi = mx.broadcast_to(mx.arange(128), (1, length, 128))
    ci = mx.broadcast_to(mx.arange(513), (1, length, 513))
    args = (q, window, pooled, wi, ci, mx.linspace(-2, 2, heads), dim**-0.5)
    actual = packed_sparse_attention(*args)
    expected = reference(*args)
    unrounded = reference(*args, round_probabilities=False)
    error = mx.mean((actual.astype(mx.float32) - expected.astype(mx.float32)) ** 2)
    baseline = mx.mean(
        (unrounded.astype(mx.float32) - expected.astype(mx.float32)) ** 2
    )
    mx.eval(error, baseline)
    assert error.item() < baseline.item() * 0.04
    assert mx.array_equal(actual, packed_sparse_attention(*args)).item()
