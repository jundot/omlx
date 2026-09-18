"""Four-stream mHC mixing preserves accumulation and batch/sequence indexing."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41.language import _hc_post_reference, hc_post


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("shape", [(1, 1, 64), (2, 17, 128), (1, 3, 513)])
def test_fused_hc_post_matches_reference_and_repeats(dtype, shape):
    mx.random.seed(941)
    x = mx.random.normal(shape).astype(dtype)
    residual = mx.random.normal((*shape[:-1], 4, shape[-1])).astype(dtype)
    post = mx.random.uniform(shape=(*shape[:-1], 4)) * 2
    comb = mx.softmax(mx.random.normal((*shape[:-1], 4, 4)), axis=-1)
    expected = _hc_post_reference(x, residual, post, comb)
    actual = hc_post(x, residual, post, comb)
    repeated = hc_post(x, residual, post, comb)
    np.testing.assert_allclose(
        actual.astype(mx.float32), expected.astype(mx.float32), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(
        actual.astype(mx.float32), repeated.astype(mx.float32)
    )


def test_hc_post_noncontiguous_inputs_and_other_stream_counts():
    mx.random.seed(943)
    for streams in (2, 4, 8):
        x = mx.random.normal((2, 64, 3)).transpose(0, 2, 1).astype(mx.bfloat16)
        residual = (
            mx.random.normal((2, streams, 3, 64))
            .transpose(0, 2, 1, 3)
            .astype(mx.bfloat16)
        )
        post = mx.random.normal((2, streams, 3)).transpose(0, 2, 1)
        comb = mx.random.normal((2, 3, streams, streams)).swapaxes(-1, -2)
        np.testing.assert_allclose(
            hc_post(x, residual, post, comb).astype(mx.float32),
            _hc_post_reference(x, residual, post, comb).astype(mx.float32),
            rtol=0.008,
            atol=0.008,
        )


@pytest.mark.parametrize(
    "dtype,tolerance", [(mx.float32, 2e-6), (mx.float16, 0.001), (mx.bfloat16, 0.008)]
)
@pytest.mark.parametrize("width", [64, 513, 5120])
def test_pre_norm_fusion_bounds_and_repeat(dtype, tolerance, width):
    from omlx.patches.deepseek_v41.language import hc_pre, hc_pre_norm, norm

    mx.random.seed(944)
    x = mx.random.normal((2, 7, 4, width)).astype(dtype)
    pre = mx.random.uniform(shape=(2, 7, 4))
    weight = mx.random.uniform(shape=(width,))
    expected = norm(hc_pre(x, pre), weight, 1e-6)
    actual = hc_pre_norm(x, pre, weight, 1e-6)
    repeated = hc_pre_norm(x, pre, weight, 1e-6)
    np.testing.assert_allclose(
        actual.astype(mx.float32),
        expected.astype(mx.float32),
        rtol=tolerance,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        actual.astype(mx.float32), repeated.astype(mx.float32)
    )


def test_pre_norm_fusion_zero_and_intermediate_cast():
    from omlx.patches.deepseek_v41.language import hc_pre, hc_pre_norm, norm

    x = mx.zeros((1, 2, 4, 5120), mx.bfloat16)
    pre = mx.ones((1, 2, 4))
    weight = mx.ones((5120,))
    np.testing.assert_array_equal(
        hc_pre_norm(x, pre, weight, 1e-6).astype(mx.float32), 0
    )
    mx.random.seed(948)
    x = mx.random.normal(x.shape).astype(mx.bfloat16)
    expected = norm(hc_pre(x, pre), weight, 1e-6)
    unrounded = norm(
        mx.sum(x.astype(mx.float32) * pre[..., None], -2), weight, 1e-6
    ).astype(x.dtype)
    actual = hc_pre_norm(x, pre, weight, 1e-6)
    assert mx.sum(actual != unrounded).item() > 100
    np.testing.assert_allclose(
        actual.astype(mx.float32), expected.astype(mx.float32), rtol=0.008, atol=1e-7
    )


def test_pre_norm_fusion_cancellation_preserves_pre_reduction_order():
    from omlx.patches.deepseek_v41.language import hc_pre, hc_pre_norm, norm

    small = mx.linspace(0.25, 2, 64)
    large = mx.full((64,), 2**24)
    x = mx.stack([small, large, -large, small[::-1]])[None, None].astype(mx.bfloat16)
    pre = mx.ones((1, 1, 4))
    weight = mx.ones((64,))
    expected = norm(hc_pre(x, pre), weight, 1e-6)
    actual = hc_pre_norm(x, pre, weight, 1e-6)
    np.testing.assert_allclose(
        actual.astype(mx.float32), expected.astype(mx.float32), rtol=0.008, atol=1e-7
    )


@pytest.mark.parametrize("width", [64, 513, 5120])
def test_hc_projection_fp32_weight_accuracy_and_repeat(width):
    from omlx.patches.deepseek_v41.hyper_connection import fused_hc_projection

    mx.random.seed(952)
    x = mx.random.normal((2, 3, 4, width)).astype(mx.bfloat16)
    fn = mx.random.normal((24, 4 * width)) * 0.01
    actual = fused_hc_projection(x, fn, 1e-6)
    repeated = fused_hc_projection(x, fn, 1e-6)
    flat = np.asarray(x.astype(mx.float32)).astype(np.float64).reshape(2, 3, -1)
    weights = np.asarray(fn).astype(np.float64)
    expected = (flat @ weights.T) / np.sqrt(
        np.mean(flat * flat, axis=-1, keepdims=True) + 1e-6
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-6)
    np.testing.assert_array_equal(actual, repeated)


def test_prefill_hc_mixes_dispatch_and_controls_match_reference():
    from types import SimpleNamespace

    from omlx.patches.deepseek_v41.language import _hc_mixes, hc_mixes

    mx.random.seed(953)
    c = SimpleNamespace(hc_mult=4, norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20)
    x = mx.random.normal((2, 256, 4, 64)).astype(mx.bfloat16)
    fn = mx.random.normal((24, 256)) * 0.05
    scale = mx.array([0.3, 0.7, 1.1])
    base = mx.random.normal((24,))
    expected = _hc_mixes(x, fn, scale, base, 4, 1e-6, 1e-6, 20)
    actual = hc_mixes(x, fn, scale, base, c)
    repeated = hc_mixes(x, fn, scale, base, c)
    for a, e, r in zip(actual, expected, repeated):
        np.testing.assert_allclose(a, e, rtol=2e-5, atol=1e-6)
        np.testing.assert_array_equal(a, r)


def test_hc_projection_large_row_dispatch_matches_fp64():
    from omlx.patches.deepseek_v41.hyper_connection import fused_hc_projection

    mx.random.seed(960)
    x = mx.random.normal((1, 1024, 4, 513)).astype(mx.bfloat16)
    fn = mx.random.normal((24, 4 * 513)) * 0.01
    actual = fused_hc_projection(x, fn, 1e-6)
    flat = np.asarray(x.astype(mx.float32)).astype(np.float64).reshape(1, 1024, -1)
    expected = (flat @ np.asarray(fn).astype(np.float64).T) / np.sqrt(
        np.mean(flat * flat, -1, keepdims=True) + 1e-6
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-6)
    np.testing.assert_array_equal(actual, fused_hc_projection(x, fn, 1e-6))
