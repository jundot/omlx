# SPDX-License-Identifier: Apache-2.0
"""Ragged per-row rollback of quantized batch caches (speculative verify keeping a different count per row)."""

import mlx.core as mx
import numpy as np
import pytest

from omlx import affine4
from omlx.affine4 import BatchAffine4KVCache, BatchAffine8KVCache
from omlx.patches.mlx_lm_mtp import fused_batch


def random(shape, seed=0):
    return mx.random.normal(shape, key=mx.random.key(seed)).astype(mx.bfloat16)


def require_tensor_ops():
    if not affine4._tensor_ops_available():
        pytest.skip("Metal TensorOps attention is unavailable")


def row_reference(cache, queries, ends, causal_rows=0):
    """Dense fp32 attention of each batch row over its own first ``ends[b]`` columns."""
    keys, values = cache.dequantize()
    outs = []
    for b, end in enumerate(ends):
        q = queries[b : b + 1].astype(mx.float32)
        k, v = keys[b : b + 1, :, :end], values[b : b + 1, :, :end]
        outs.append(
            mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=q.shape[-1] ** -0.5,
                mask="causal" if causal_rows > 1 else None,
            )
        )
    return mx.concatenate(outs, axis=0)


@pytest.mark.parametrize("cache_class", [BatchAffine4KVCache, BatchAffine8KVCache])
@pytest.mark.parametrize("dim,heads,repeats", [(64, 2, 4), (256, 4, 6)])
def test_trim_rows_then_unmasked_decode_stops_at_row_ends(
    cache_class, dim, heads, repeats
):
    """A decode step with mask=None (Qwen3.5 left-padded decode) after trim_rows must not read dropped columns."""
    require_tensor_ops()
    cache = cache_class([0, 0, 0])
    cache.update_and_fetch(
        random((3, heads, 700, dim)), random((3, heads, 700, dim), seed=1)
    )
    # A 5-token verify block, then rows keep 5, 1 and 3 tokens.
    cache.update_and_fetch(
        random((3, heads, 5, dim), seed=2), random((3, heads, 5, dim), seed=3)
    )
    assert cache.trim_rows([0, 4, 2]) == 4
    assert cache._row_ends == [705, 701, 703]
    cache.update_and_fetch(
        random((3, heads, 1, dim), seed=4), random((3, heads, 1, dim), seed=5)
    )
    ends = cache._row_ends
    assert ends == [706, 702, 704]
    q = random((3, heads * repeats, 1, dim), seed=6)
    out = affine4._native_attention(cache, q, *cache.state, dim**-0.5, None)
    assert out is not None
    ref = row_reference(cache, q, ends)
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)), np.array(ref), atol=4e-3, rtol=4e-3
    )


@pytest.mark.parametrize("cache_class", [BatchAffine8KVCache])
def test_trim_rows_then_causal_verify_block(cache_class):
    """A causal multi-row verify after a ragged trim: row b's queries end at its own end."""
    require_tensor_ops()
    dim, heads, repeats, rows = 256, 4, 6, 4
    cache = cache_class([0, 0])
    cache.update_and_fetch(
        random((2, heads, 600, dim)), random((2, heads, 600, dim), seed=1)
    )
    cache.update_and_fetch(
        random((2, heads, 3, dim), seed=2), random((2, heads, 3, dim), seed=3)
    )
    cache.trim_rows([2, 0])
    mask = cache.make_mask(rows)
    cache.update_and_fetch(
        random((2, heads, rows, dim), seed=4), random((2, heads, rows, dim), seed=5)
    )
    ends = cache._row_ends
    q = random((2, heads * repeats, rows, dim), seed=6)
    for m in (None, mask):
        out = affine4._native_attention(
            cache, q, *cache.state, dim**-0.5, "causal" if m is None else m
        )
        assert out is not None
        ref = row_reference(cache, q, ends, causal_rows=rows)
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)), np.array(ref), atol=4e-3, rtol=4e-3
        )


def test_ragged_trim_rows_context_trims_and_hides_affine_leaves():
    cache = BatchAffine4KVCache([0, 0])
    cache.update_and_fetch(random((2, 2, 40, 64)), random((2, 2, 40, 64), seed=1))
    cache.update_and_fetch(random((2, 2, 4, 64), seed=2), random((2, 2, 4, 64), seed=3))

    seen = []
    with fused_batch._ragged_trim_rows([None, cache], [3, 1], 3):
        seen.append(getattr(cache, "is_speculating", False))
    assert seen == [True]
    assert not getattr(cache, "is_speculating", False)
    # Verify block of 4 (depth 3): rows keep 4 and 2.
    assert cache._row_ends == [44, 42]
    assert fused_batch._supports_batch_rollback(cache)


_RAGGED_CYCLES = ([3, 0, 2], [1, 1, 1], [0, 3, 1], [2, 0, 0])


def _kv(rows, length, *, heads=2, dim=32, seed=0):
    mx.random.seed(seed)
    keys = mx.random.normal((rows, heads, length, dim)).astype(mx.float16)
    values = mx.random.normal((rows, heads, length, dim)).astype(mx.float16)
    return keys, values


def _assert_rows_attend_like(batch_out, references):
    for row, reference in enumerate(references):
        assert mx.allclose(
            batch_out[row : row + 1], reference, atol=2e-3, rtol=2e-3
        ).item(), row


def test_affine8_batch_cache_trims_rows_without_moving_state():
    """Ragged verify rollback keeps each row's own end: extracted rows, mask
    and attention match single-row caches fed only the kept tokens, and
    filter / extend keep working afterwards."""
    from omlx.affine4 import Affine8KVCache, BatchAffine8KVCache

    lengths = [5, 9, 7]
    singles, references = [], []
    for row, length in enumerate(lengths):
        keys, values = _kv(1, length, seed=row)
        for target in (singles, references):
            cache = Affine8KVCache()
            cache.update_and_fetch(keys, values)
            target.append(cache)
    batch = Affine8KVCache.merge(singles)
    assert isinstance(batch, BatchAffine8KVCache)
    block = 4
    for cycle, counts in enumerate(_RAGGED_CYCLES):
        keys, values = _kv(3, block, seed=10 + cycle)
        batch.update_and_fetch(keys, values)
        assert batch.trim_rows(counts) == max(counts)
        for row, count in enumerate(counts):
            keep = block - count
            references[row].update_and_fetch(
                keys[row : row + 1, :, :keep], values[row : row + 1, :, :keep]
            )
    assert batch._row_ends is not None
    for row, reference in enumerate(references):
        got = batch.extract(row).state
        want = reference.state
        for g, w in zip(got, want):
            assert mx.array_equal(g.norms, w.norms).item()
            assert mx.array_equal(g.indices, w.indices).item()

    # A verify-shaped forward attends to each row's own prefix.
    queries = mx.random.normal((3, 4, 2, 32)).astype(mx.float16)
    keys, values = _kv(3, 2, seed=99)
    mask = batch.make_mask(2)
    ks, vs = batch.update_and_fetch(keys, values)
    out = batch.attention(queries, ks, vs, scale=0.2, mask=mask)
    expected = []
    for row, reference in enumerate(references):
        rk, rv = reference.update_and_fetch(keys[row : row + 1], values[row : row + 1])
        expected.append(
            reference.attention(
                queries[row : row + 1], rk, rv, scale=0.2, mask="causal"
            )
        )
    _assert_rows_attend_like(out, expected)

    batch.filter([2, 0])
    for row, reference in zip((0, 1), (references[2], references[0])):
        got = batch.extract(row).state
        assert mx.array_equal(got[0].indices, reference.state[0].indices).item()
    joiner = Affine8KVCache()
    joiner.update_and_fetch(*_kv(1, 6, seed=7))
    batch.extend(Affine8KVCache.merge([joiner]))
    assert batch._row_ends is None
    got = batch.extract(1).state
    assert mx.array_equal(got[1].norms, references[0].state[1].norms).item()
