import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41.quantization import (
    _pack_activation_reference,
    pack_activation,
)

FORMATS = [(8, 32, False), (4, 32, False), (4, 16, True)]


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("fmt", FORMATS)
def test_packed_bytes_random_dynamic_range(dtype, fmt):
    rng = np.random.default_rng(712)
    low, high = (-20, 12) if dtype == mx.float16 else (-120, 120)
    x = (
        rng.normal(size=(128, 512)) * np.exp2(rng.uniform(low, high, size=(128, 1)))
    ).astype(np.float32)
    x = mx.array(x).astype(dtype)
    expected = _pack_activation_reference(x, *fmt)
    actual = pack_activation(x, *fmt)
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("fmt", FORMATS)
def test_packed_bytes_zero_and_rounding_ties(dtype, fmt):
    bits, group, _ = fmt
    limit = 448.0 if bits == 8 else 6.0
    ties = np.array(
        [
            0.0,
            -0.0,
            0.25,
            -0.25,
            0.75,
            -0.75,
            1.25,
            -1.25,
            1.75,
            -1.75,
            2.5,
            -2.5,
            3.5,
            -3.5,
            5.0,
            -5.0,
        ],
        np.float32,
    )
    data = np.zeros((len(ties) * 3 + 1, 32), np.float32)
    for i, value in enumerate(ties):
        for j, v in enumerate(
            (
                np.nextafter(value, -np.inf, dtype=np.float32),
                value,
                np.nextafter(value, np.inf, dtype=np.float32),
            )
        ):
            data[i * 3 + j] = v
            data[i * 3 + j, group - 1 :: group] = limit
    data[-1, ::2] = -0.0
    x = mx.array(data).astype(dtype)
    expected = _pack_activation_reference(x, *fmt)
    actual = pack_activation(x, *fmt)
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("shape", [(1, 0, 512), (2, 16)])
def test_unsupported_geometry_uses_reference(shape):
    x = mx.zeros(shape, mx.bfloat16)
    expected = _pack_activation_reference(x, 4, 16, True)
    actual = pack_activation(x, 4, 16, True)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


def test_single_row_extract_keeps_independent_container():
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache

    cache = DeepseekV41Cache(2)
    cache.cache = [mx.array([5]), mx.ones((1, 5, 8))] + [None] * 5
    row = cache.extract(0)
    assert row.cache is not cache.cache
    assert row.cache[1] is cache.cache[1]
    row.cache[0] = mx.array([6])
    assert cache.size() == 5
    assert row.size() == 6


def test_batched_extract_selects_requested_row():
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache

    cache = DeepseekV41Cache(2)
    cache.cache = [mx.array([5, 7]), mx.arange(16).reshape(2, 1, 8)] + [None] * 5
    row = cache.extract(1)
    assert row.size() == 7
    np.testing.assert_array_equal(
        np.array(row.cache[1]), np.arange(8, 16).reshape(1, 1, 8)
    )
