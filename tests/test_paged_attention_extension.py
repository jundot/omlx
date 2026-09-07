# SPDX-License-Identifier: Apache-2.0
"""Optional wheel tests against contiguous SDPA with ragged, permuted pages."""

import mlx.core as mx
import pytest

extension = pytest.importorskip("eco_paged_attention")


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_ragged_permuted_pages(dtype, head_dim):
    mx.random.seed(42)
    keys = mx.random.normal((8, 2, 16, head_dim)).astype(dtype)
    values = mx.random.normal(keys.shape).astype(dtype)
    query = mx.random.normal((2, 12, 1, head_dim)).astype(dtype)
    tables = mx.array([[6, 1, 5], [3, 0, 7]], dtype=mx.uint32)
    lengths = mx.array([35, 19], dtype=mx.uint32)
    scale = head_dim**-0.5
    actual = extension.paged_attention(
        query, keys, values, tables, lengths, scale=scale
    )
    references = []
    for i, (pages, length) in enumerate(zip(tables.tolist(), lengths.tolist())):
        k = mx.concatenate([keys[p] for p in pages], axis=1)[:, :length][None]
        v = mx.concatenate([values[p] for p in pages], axis=1)[:, :length][None]
        references.append(
            mx.fast.scaled_dot_product_attention(
                query[i : i + 1].astype(mx.float32),
                k.astype(mx.float32),
                v.astype(mx.float32),
                scale=scale,
            )
        )
    expected = mx.concatenate(references)
    mx.eval(actual, expected)
    assert mx.max(mx.abs(actual.astype(mx.float32) - expected)).item() < (
        0.01 if dtype == mx.bfloat16 else 0.002
    )


def test_rejects_page_outside_pool():
    q = mx.zeros((1, 2, 1, 64))
    kv = mx.zeros((1, 1, 16, 64))
    with pytest.raises(ValueError, match="outside"):
        extension.paged_attention(
            q,
            kv,
            kv,
            mx.array([[2]], dtype=mx.uint32),
            mx.array([1], dtype=mx.uint32),
            scale=0.125,
        )
