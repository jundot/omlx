"""Checkpoint-free Metal numerical and sparse-addressing gates."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41.kernels import (
    packed_index_scores,
    packed_sparse_attention,
)
from omlx.patches.deepseek_v41.language import candidate_block_ids, sparse_attention
from omlx.patches.deepseek_v41.quantization import pack_activation, unpack_activation


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "dim,length,count",
    [
        (32, 1, 0),
        (64, 1, 17),
        (128, 1, 17),
        (256, 1, 17),
        (512, 1, 513),
        (512, 7, 129),
        (96, 1, 17),
        (96, 3, 37),
        (1024, 1, 17),
    ],
)
def test_packed_attention_split_and_mask(dtype, dim, length, count):
    rng = np.random.default_rng(71)
    q = mx.array(rng.normal(size=(1, length, 5, dim)).astype(np.float32)).astype(dtype)
    window = pack_activation(mx.array(rng.normal(size=(1, 17, dim)).astype(np.float32)))
    pooled = pack_activation(
        mx.array(rng.normal(size=(1, 541, dim)).astype(np.float32)), 4, 16, True
    )
    wi = mx.array(rng.integers(-3, 17, size=(1, length, 19)), mx.int32)
    ci = mx.array(rng.integers(-3, 541, size=(1, length, count)), mx.int32)
    sinks = mx.array([-10000, -3, 0, 4, 10000], mx.float32)
    actual = packed_sparse_attention(q, window, pooled, wi, ci, sinks, dim**-0.5)
    selected = mx.concatenate(
        [
            unpack_activation(window, dtype=mx.float32)[0][mx.maximum(wi[0], 0)][None],
            unpack_activation(pooled, 4, 16, True, mx.float32)[0][mx.maximum(ci[0], 0)][
                None
            ],
        ],
        -2,
    )
    expected = sparse_attention(
        q, selected, mx.concatenate([wi, ci], -1), sinks, dim**-0.5
    )
    tolerance = (
        3e-6 if dtype == mx.float32 else (0.002 if dtype == mx.float16 else 0.016)
    )
    np.testing.assert_allclose(
        actual.astype(mx.float32),
        expected.astype(mx.float32),
        atol=tolerance,
        rtol=tolerance,
    )
    assert mx.all(mx.isfinite(actual)).item()


def test_attention_empty_and_all_masked():
    q = mx.ones((1, 2, 3, 32))
    w, p = mx.zeros((1, 0, 33), mx.uint8), mx.zeros((1, 0, 18), mx.uint8)
    empty = mx.zeros((1, 2, 0), mx.int32)
    sinks = mx.array([-10000, 0, 10000], mx.float32)
    np.testing.assert_array_equal(
        packed_sparse_attention(q, w, p, empty, empty, sinks, 1), 0
    )
    invalid = mx.full((1, 2, 257), -1, mx.int32)
    np.testing.assert_array_equal(
        packed_sparse_attention(q, w, p, invalid, empty, sinks, 1), 0
    )


@pytest.mark.parametrize(
    "dim,heads,length",
    [(32, 8, 1), (128, 128, 3), (96, 9, 2), (96, 33, 3), (128, 65, 1), (32, 32, 2)],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("restricted", [False, True])
def test_index_scores_packed_candidates_and_causal(
    dim, heads, length, restricted, dtype
):
    rng = np.random.default_rng(14)
    n = 137
    q = mx.array(rng.normal(size=(1, length, heads, dim)).astype(np.float32)).astype(
        dtype
    )
    keys = pack_activation(mx.array(rng.normal(size=(1, n, dim)).astype(np.float32)), 4)
    weights = mx.array(rng.normal(size=(1, length, heads)).astype(np.float32)) / heads
    candidates = None
    if restricted:
        candidates = mx.array(
            np.tile([-1, 0, 2, 70, 132, 132, 136, 137], (1, length, 1)), mx.int32
        )
    actual = packed_index_scores(q, keys, weights, 263, 2, candidates)
    expanded = unpack_activation(keys, 4, dtype=mx.float32)
    expected = mx.sum(
        mx.maximum(mx.einsum("bshd,btd->bsht", q.astype(mx.float32), expanded), 0)
        * weights[..., None],
        -2,
    )
    valid = mx.arange(n) < ((263 + mx.arange(length) + 1) // 2)[:, None]
    expected = mx.where(valid, expected, -float("inf"))
    if candidates is not None:
        expected = mx.take_along_axis(expected, mx.clip(candidates, 0, n - 1), -1)
        expected = mx.where(
            (candidates >= 0) & (candidates < n), expected, -float("inf")
        )
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)


def test_candidate_block_ids_preserve_latest_and_partial_blocks():
    scores = mx.array(
        [
            [
                [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, -float("inf")],
                [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ]
        ]
    )
    lengths = mx.array([[[8], [9]]])
    ids = candidate_block_ids(scores, lengths, 2, 4)
    np.testing.assert_array_equal(ids, [[[0, 1], [0, 2]]])
    # All unavailable blocks must remain invalid even if the budget covers all.
    ids = candidate_block_ids(
        mx.full((1, 1, 9), -float("inf")), mx.array([[[0]]]), 4, 4
    )
    np.testing.assert_array_equal(ids, -1)


def test_empty_index_cache_with_explicit_candidates():
    q = mx.ones((1, 2, 8, 32))
    keys = mx.zeros((1, 0, 17), mx.uint8)
    weights = mx.ones((1, 2, 8))
    candidates = mx.array([[[-1, 0], [1, -1]]])
    actual = packed_index_scores(q, keys, weights, 0, 2, candidates)
    np.testing.assert_array_equal(actual, -float("inf"))


def test_noncontiguous_index_inputs():
    mx.random.seed(25)
    q = mx.random.normal((1, 6, 8, 32))[:, ::2]
    keys = pack_activation(mx.random.normal((1, 40, 32)), 4)[:, ::2]
    weights = mx.random.normal((1, 6, 8))[:, ::2]
    candidates = mx.broadcast_to(mx.arange(12)[None, None], (1, 3, 12))
    actual = packed_index_scores(q, keys, weights, 40, 1, candidates)
    expanded = unpack_activation(keys, 4, dtype=mx.float32)
    expected = mx.sum(
        mx.maximum(mx.einsum("bshd,btd->bsht", q.astype(mx.float32), expanded), 0)
        * weights[..., None],
        -2,
    )[..., :12]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    "count,width",
    [(0, 33), (7, 9001), (512, 5003), (2048, 5003), (3000, 5003), (50, 17)],
)
def test_tile_topk_matches_score_order(count, width):
    from omlx.patches.deepseek_v41.kernels import _tile_topk

    rng = np.random.default_rng(122)
    data = rng.normal(size=(1, 3, width)).astype(np.float32)
    data[0, 0, :5] = -np.inf
    values, ids = _tile_topk(mx.array(data), count, offset=71)
    mx.eval(values, ids)
    k = min(count, width)
    expected = np.sort(np.argsort(-data, axis=-1)[..., :k] + 71, axis=-1)
    np.testing.assert_array_equal(np.sort(np.asarray(ids), axis=-1), expected)
    np.testing.assert_array_equal(
        values, np.take_along_axis(data, np.asarray(ids) - 71, -1)
    )


def test_tile_topk_ties_use_absolute_ids():
    from omlx.patches.deepseek_v41.kernels import _tile_topk

    ids = mx.arange(5003, dtype=mx.int32)[::-1][None, None]
    scores = mx.zeros(ids.shape)
    _, chosen = _tile_topk(scores, 512, ids=ids)
    np.testing.assert_array_equal(mx.sort(chosen, axis=-1), np.arange(512)[None, None])


@pytest.mark.parametrize(
    "dim,heads,start,ratio", [(32, 3, 0, 4), (128, 8, 399, 2), (96, 9, 91, 3)]
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_streamed_index_and_candidate_blocks_match_full_scores(
    dim, heads, start, ratio, dtype
):
    from omlx.patches.deepseek_v41.kernels import packed_index_topk

    mx.random.seed(105)
    length, width = 19, 203
    q = mx.random.normal((1, length, heads, dim)).astype(dtype)
    keys = pack_activation(mx.random.normal((1, width, dim)), 4)
    weights = mx.random.normal((1, length, heads))
    scores = packed_index_scores(q, keys, weights, start, ratio)
    expected_order = mx.argsort(-scores, axis=-1)[..., :17].astype(mx.int32)
    expected = mx.sort(
        mx.where(
            mx.take_along_axis(scores, expected_order, -1) > -float("inf"),
            expected_order,
            -1,
        ),
        axis=-1,
    )
    lengths = (mx.arange(start + 1, start + length + 1) // ratio)[None, :, None]
    expected_blocks = mx.sort(candidate_block_ids(scores, lengths, 7, 8), axis=-1)
    actual, blocks = packed_index_topk(
        q,
        keys,
        weights,
        start,
        ratio,
        17,
        block_count=7,
        block_size=8,
        chunk_size=61,
        query_chunk_size=5,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(blocks, expected_blocks)


def test_streamed_index_empty_and_forced_recent_block():
    from omlx.patches.deepseek_v41.kernels import packed_index_topk

    q = mx.zeros((1, 5, 8, 32))
    weights = mx.ones((1, 5, 8))
    empty = mx.zeros((1, 0, 17), mx.uint8)
    ids, blocks = packed_index_topk(
        q, empty, weights, 0, 2, 512, block_count=2048, block_size=8
    )
    assert ids.shape == blocks.shape == (1, 5, 0)
    keys = pack_activation(mx.ones((1, 53, 32)), 4)
    ids, blocks = packed_index_topk(
        q,
        keys,
        weights,
        37,
        1,
        3,
        block_count=1,
        block_size=8,
        chunk_size=17,
        query_chunk_size=2,
    )
    np.testing.assert_array_equal(ids, np.broadcast_to(np.arange(3), (1, 5, 3)))
    np.testing.assert_array_equal(blocks, ((np.arange(38, 43) - 1) // 8)[None, :, None])


def test_streaming_bounds_scoring_and_skips_future_chunks(monkeypatch):
    from omlx.patches.deepseek_v41 import kernels

    calls = []
    original = kernels.packed_index_scores

    def tracked(q, keys, weights, start, ratio, **kwargs):
        if keys.shape[1]:
            calls.append((q.shape[1], keys.shape[1], start, kwargs.get("key_start", 0)))
        return original(q, keys, weights, start, ratio, **kwargs)

    monkeypatch.setattr(kernels, "packed_index_scores", tracked)
    q = mx.ones((1, 19, 8, 32))
    keys = pack_activation(mx.ones((1, 1000, 32)), 4)
    ids, _ = kernels.packed_index_topk(
        q, keys, mx.ones((1, 19, 8)), 0, 2, 17, chunk_size=5, query_chunk_size=4
    )
    mx.eval(ids)
    assert calls
    assert all(queries <= 4 and width <= 5 for queries, width, _, _ in calls)
    assert all(
        first + width <= (start + queries) // 2
        for queries, width, start, first in calls
    )
    # The first query has no completed compressed row; output size stays fixed.
    np.testing.assert_array_equal(ids[0, 0], -1)
    assert ids.shape == (1, 19, 17)


@pytest.mark.parametrize(
    "left_size,right_size,count",
    [(512, 513, 512), (5, 2, 7), (0, 33, 17), (2048, 13, 2048)],
)
def test_sorted_merge_ties_padding_and_uneven_runs(left_size, right_size, count):
    from omlx.patches.deepseek_v41.kernels import _merge_topk

    rng = np.random.default_rng(577)
    values = rng.integers(-3, 4, (1, 3, left_size + right_size)).astype(np.float32)
    values[..., -2:] = -np.inf
    ids = rng.integers(0, 17, values.shape).astype(np.int32)
    order = np.lexsort((ids, -values), axis=-1)
    expected = (
        np.take_along_axis(values, order, -1)[..., :count],
        np.take_along_axis(ids, order, -1)[..., :count],
    )
    runs = []
    for part in (slice(None, left_size), slice(left_size, None)):
        v, i = values[..., part], ids[..., part]
        local_order = np.lexsort((i, -v), axis=-1)
        runs.append(
            (
                mx.array(np.take_along_axis(v, local_order, -1)),
                mx.array(np.take_along_axis(i, local_order, -1)),
            )
        )
    actual = _merge_topk(*runs, count)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("block,count", [(3, 17), (8, 2048), (2, 3000), (1, 512)])
def test_fused_candidate_selection_matches_block_maxima(block, count):
    from omlx.patches.deepseek_v41.kernels import _tile_topk

    width, length, start, ratio = 10003, 3, 100, 2
    rng = np.random.default_rng(302)
    data = rng.normal(size=(1, length, width)).astype(np.float32)
    data[..., -3:] = -np.inf
    scores = mx.array(data)
    expected = candidate_block_ids(
        scores,
        (mx.arange(start + 1, start + length + 1) // ratio)[None, :, None],
        count,
        block,
    )
    values, ids = _tile_topk(
        scores, count, block_size=block, force_latest=True, start=start, ratio=ratio
    )
    actual = mx.sort(mx.where(values > -float("inf"), ids, -1), axis=-1)
    np.testing.assert_array_equal(actual, mx.sort(expected, axis=-1))


@pytest.mark.parametrize("length", [1, 9])
def test_streaming_drains_async_work_before_return(monkeypatch, length):
    from omlx.patches.deepseek_v41.kernels import packed_index_topk

    pending, submissions = set(), []
    evaluate, submit = mx.eval, mx.async_eval

    def tracked_eval(*arrays):
        result = evaluate(*arrays)
        pending.difference_update(id(a) for a in arrays)
        return result

    def tracked_submit(*arrays):
        assert not pending, "Previous batch must be drained before the next submission"
        result = submit(*arrays)
        pending.update(id(a) for a in arrays)
        submissions.append(len(arrays))
        return result

    q = mx.ones((1, length, 8, 32))
    keys = pack_activation(mx.ones((1, 137, 32)), 4)
    weights = mx.ones((1, length, 8))
    mx.eval(q, keys, weights)
    monkeypatch.setattr(mx, "eval", tracked_eval)
    monkeypatch.setattr(mx, "async_eval", tracked_submit)
    packed_index_topk(
        q,
        keys,
        weights,
        128,
        1,
        17,
        block_count=3,
        block_size=8,
        chunk_size=32,
        query_chunk_size=4,
    )
    assert len(submissions) > 1
    assert not pending


@pytest.mark.parametrize("count,runs", [(7, 3), (17, 5), (512, 9), (513, 5), (2048, 3)])
def test_multiway_merge_exact_rank_with_duplicate_ids(count, runs):
    from omlx.patches.deepseek_v41.kernels import _merge_topk

    rng = np.random.default_rng(779)
    values = rng.integers(-2, 3, (1, 2, runs, count)).astype(np.float32)
    values[:, 0, -1] = -np.inf
    ids = rng.integers(0, 37, values.shape).astype(np.int32)
    order = np.lexsort((ids, -values), axis=-1)
    values = np.take_along_axis(values, order, -1)
    ids = np.take_along_axis(ids, order, -1)
    arrays = (mx.array(values.reshape(1, 2, -1)), mx.array(ids.reshape(1, 2, -1)))
    actual = _merge_topk(arrays, arrays, count, runs=runs)
    fanin = 4 if count <= 512 else 2
    expected_values, expected_ids = [], []
    for first in range(0, runs, fanin):
        v = values[:, :, first : first + fanin].reshape(1, 2, -1)
        i = ids[:, :, first : first + fanin].reshape(1, 2, -1)
        chosen = np.lexsort((i, -v), axis=-1)[..., :count]
        expected_values.append(np.take_along_axis(v, chosen, -1))
        expected_ids.append(np.take_along_axis(i, chosen, -1))
    np.testing.assert_array_equal(actual[0], np.concatenate(expected_values, axis=-1))
    np.testing.assert_array_equal(actual[1], np.concatenate(expected_ids, axis=-1))


@pytest.mark.parametrize("length", [1, 4, 65, 512])
def test_single_tile_index_stays_lazy_and_matches_streaming(monkeypatch, length):
    from omlx.patches.deepseek_v41.kernels import packed_index_topk

    mx.random.seed(106)
    q = mx.random.normal((1, length, 8, 32))
    keys = pack_activation(mx.random.normal((1, 137, 32)), 4)
    weights = mx.random.normal((1, length, 8))
    args = (q, keys, weights, 128, 1, 17)
    expected = packed_index_topk(*args, block_count=3, block_size=8, chunk_size=32)
    mx.eval(expected)

    def unexpected_submission(*arrays):
        raise AssertionError("A single bounded tile must remain lazy")

    with monkeypatch.context() as patch:
        patch.setattr(mx, "eval", unexpected_submission)
        patch.setattr(mx, "async_eval", unexpected_submission)
        actual = packed_index_topk(*args, block_count=3, block_size=8)
    for result, reference in zip(actual, expected):
        np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("length", [1, 16, 512])
@pytest.mark.parametrize("iters", [1, 2, 20])
@pytest.mark.parametrize("eps", [1e-6, 1e-3])
def test_sinkhorn_fused_exact(length, iters, eps):
    from omlx.patches.deepseek_v41.hyper_connection import (
        sinkhorn,
        sinkhorn_reference,
    )

    mx.random.seed(108)
    values = mx.random.normal((1, length, 4, 4)) * 20
    comb = mx.softmax(values, -1) + eps
    np.testing.assert_array_equal(
        sinkhorn(comb, eps, iters), sinkhorn_reference(comb, eps, iters)
    )
    zeros = mx.zeros_like(comb)
    np.testing.assert_array_equal(
        sinkhorn(zeros, eps, iters), sinkhorn_reference(zeros, eps, iters)
    )
