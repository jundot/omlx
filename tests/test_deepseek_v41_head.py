import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41.head import project_logits


@pytest.mark.parametrize("length", [1, 3, 5])
@pytest.mark.parametrize("width", [256, 264, 5120])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_bf16_head_fp32_accumulation_and_repeatability(width, dtype, length):
    mx.random.seed(190)
    weight = (mx.random.normal((4099, width)) * 0.05).astype(mx.bfloat16)
    x = mx.random.normal((1, length, width)).astype(dtype)
    actual = project_logits(x, weight)
    repeated = project_logits(x, weight)
    mx.eval(actual, repeated)
    assert actual.dtype == mx.float32
    assert actual.shape == (1, length, 4099)
    reference = (
        np.asarray(x.astype(mx.float32)).astype(np.float64)
        @ np.asarray(weight.astype(mx.float32)).astype(np.float64).T
    )
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=5e-5)
    np.testing.assert_array_equal(actual, repeated)


@pytest.mark.parametrize("shape", [(1, 3, 256), (2, 1, 256)])
def test_multi_token_head_retains_full_logits(shape):
    mx.random.seed(192)
    weight = mx.random.normal((4099, 256)).astype(mx.bfloat16)
    x = mx.random.normal(shape)
    actual = project_logits(x, weight)
    reference = x @ weight.astype(mx.float32).T
    assert actual.shape == (*shape[:-1], 4099)
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=5e-5)
    independent = mx.concatenate(
        [project_logits(row[None], weight) for row in x.reshape(-1, 256)], axis=0
    ).reshape(actual.shape)
    np.testing.assert_array_equal(actual, independent)
