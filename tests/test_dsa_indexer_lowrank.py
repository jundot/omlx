"""Reduced-dimension score ABI used by exact hierarchical index experiments."""

import mlx.core as mx
import pytest

from omlx.custom_kernels.glm_moe_dsa import fast


pytestmark = pytest.mark.skipif(
    not fast.is_native_available() or not fast.has_symbol("dsa_indexer_scores"),
    reason="native DSA indexer score kernel is unavailable",
)


@pytest.mark.parametrize("heads", (32, 64))
@pytest.mark.parametrize("dims", (16, 32, 48, 64, 96, 128))
def test_reduced_dimension_scores_match_composed_reference(heads, dims):
    mx.random.seed(20260823 + heads + dims)
    rows, keys = 17, 257
    q = mx.random.uniform(-0.25, 0.25, (1, heads, rows, dims)).astype(
        mx.bfloat16
    )
    k = mx.random.uniform(-0.25, 0.25, (1, 1, keys, dims)).astype(
        mx.bfloat16
    )
    weights = mx.random.uniform(-0.25, 0.25, (1, rows, heads)).astype(
        mx.bfloat16
    )

    actual = fast.dsa_indexer_scores(
        q,
        k,
        weights,
        causal=False,
        mask_ratio=0,
        use_nax=False,
    )
    dots = q.astype(mx.float32) @ k.swapaxes(-1, -2).astype(mx.float32)
    expected = mx.sum(
        mx.maximum(dots, 0)
        * weights.transpose(0, 2, 1)[:, :, :, None].astype(mx.float32),
        axis=1,
        keepdims=True,
    ).astype(mx.bfloat16)
    mx.eval(actual, expected)

    assert actual.shape == (1, 1, rows, keys)
    assert actual.dtype == mx.bfloat16
    assert mx.allclose(actual, expected, atol=0.02, rtol=0.02).item()


@pytest.mark.parametrize("dims", (0, 15, 24, 136))
def test_reduced_dimension_scores_reject_invalid_width(dims):
    q = mx.zeros((1, 64, 16, dims), dtype=mx.bfloat16)
    k = mx.zeros((1, 1, 64, dims), dtype=mx.bfloat16)
    weights = mx.zeros((1, 16, 64), dtype=mx.bfloat16)
    with pytest.raises(Exception):
        fast.dsa_indexer_scores(q, k, weights, causal=False)


def test_rank48_mma_scores_are_bit_exact_to_generic_kernel():
    mx.random.seed(20260824)
    q = mx.random.uniform(-0.25, 0.25, (1, 64, 65, 48)).astype(mx.bfloat16)
    k = mx.random.uniform(-0.25, 0.25, (1, 1, 769, 48)).astype(mx.bfloat16)
    weights = mx.random.uniform(-0.25, 0.25, (1, 65, 64)).astype(mx.bfloat16)
    expected = fast.dsa_indexer_scores(
        q,
        k,
        weights,
        causal=False,
        mask_ratio=4,
        mask_q_offset=2048,
        use_nax=False,
    )
    actual = fast.dsa_indexer_scores_mma(
        q,
        k,
        weights,
        mask_ratio=4,
        mask_q_offset=2048,
    )
    mx.eval(expected, actual)
    assert mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)).item()
