# SPDX-License-Identifier: Apache-2.0
"""Small-query selection contracts; CPU doubles never execute Metal scoring."""

from __future__ import annotations

import copy
import importlib
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import BatchKVCache, KVCache

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches import mlx_vlm_glm5_next_compat as compat
from omlx.patches.deepseek_v4.cache_extras import BatchPoolingCache, PoolingCache


@pytest.fixture
def language():
    compat.apply_mlx_vlm_glm5_next_compat_patch()
    return importlib.import_module("mlx_vlm.models.glm5_next.language")


@pytest.fixture
def small(language):
    return importlib.import_module("mlx_vlm.models.glm5_next.small_indexer")


def _config():
    return SimpleNamespace(
        hidden_size=32,
        num_attention_heads=16,
        q_lora_rank=16,
        qk_rope_head_dim=0,
        kv_lora_rank=512,
        v_head_dim=16,
        qk_nope_head_dim=16,
        mla_use_nope=True,
        attention_bias=False,
        rms_norm_eps=1e-5,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_kpool=4,
        index_kpool_always_select_tail=True,
    )


@pytest.fixture
def attention(language):
    mx.random.seed(108)
    attn = language.Glm5NextSparseAttention(_config())
    attn.set_dtype(mx.float16)
    return attn


def _forbidden(*args, **kwargs):
    raise AssertionError("금지된 커널 또는 대체 선택 경로를 호출했습니다")


def _fake_scores(q, keys, weights, before, ratio):
    return _fake_scores_batched(q, keys, weights, [before], [keys.shape[1]], ratio)


def _fake_scores_batched(q, keys, weights, before, plen, ratio, pool_cache=None):
    batch, width = q.shape[:2]
    candidates = mx.arange(keys.shape[1])[None, None]
    positions = mx.array(before)[:, None] + mx.arange(width)[None]
    valid = (candidates < mx.array(plen)[:, None, None]) & (
        (candidates + 1) * ratio - 1 <= positions[..., None]
    )
    # Unique finite values make selector ordering observable without kernel math.
    scores = mx.broadcast_to(
        candidates.astype(mx.float32), (batch, width, keys.shape[1])
    )
    return mx.where(valid, scores, -1e30).astype(q.dtype)


@pytest.fixture
def cpu_paths(monkeypatch, language, small):
    # Exercise indexer + gathered attention independently of split-K.
    monkeypatch.setattr(language, "_DSA_DECODE_FUSED", False)
    monkeypatch.setattr(language, "_SMALL_QUERY_INDEXER", True)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    monkeypatch.setattr(language, "nax_sparse_mla_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: name == "dsa_topk_indices")
    monkeypatch.setattr(
        fast,
        "dsa_topk_indices",
        lambda scores, count: mx.argsort(-scores, axis=-1)[..., :count].astype(
            mx.int32
        ),
    )
    monkeypatch.setattr(language, "small_scores", _fake_scores)
    monkeypatch.setattr(small, "small_scores", _fake_scores)
    monkeypatch.setattr(small, "small_scores_batched", _fake_scores_batched)
    monkeypatch.setattr(mx.fast, "metal_kernel", _forbidden)
    return language, small


def _advance_pool(indexer, pool, keys):
    gates = mx.zeros(keys.shape, dtype=keys.dtype)
    processed = indexer._processed(pool)
    offset = mx.array(processed, mx.int32) if isinstance(processed, list) else processed
    ready, gate, _ = pool.accumulate_windows(keys, gates, offset)
    pool.update_and_fetch(indexer._compress_windows(ready, gate))


def _caches(indexer, lengths, *, batched=False):
    pools, kvs = [], []
    for length in lengths:
        pool = PoolingCache(indexer.index_kpool)
        keys = mx.broadcast_to(
            ((mx.arange(length, dtype=mx.float32) % 13) * 0.01)[None, :, None],
            (1, length, indexer.head_dim),
        ).astype(mx.float16)
        cut = max(1, length - 3)
        _advance_pool(indexer, pool, keys[:, :cut])
        if cut < length:
            _advance_pool(indexer, pool, keys[:, cut:])
        kv = KVCache()
        kv.update_and_fetch(
            mx.zeros((1, 1, length, 512), mx.float16),
            mx.zeros((1, 1, length, 0), mx.float16),
        )
        pools.append(pool)
        kvs.append(kv)
    if batched:
        return [BatchKVCache.merge(kvs), PoolingCache.merge(pools)]
    assert len(lengths) == 1
    return [kvs[0], pools[0]]


def _inputs(batch, width):
    x = mx.full((batch, width, 32), 0.125, mx.float16)
    qr = mx.full((batch, width, 16), 0.25, mx.float16)
    return x, qr


def _query(attn, batch, width, dtype=mx.float16):
    # BF16 here is metadata only, never an MLX array or arithmetic operand.
    return SimpleNamespace(
        shape=(batch, attn.num_heads, width, attn.q_head_dim), dtype=dtype
    )


def _snapshot(cache):
    def freeze(value):
        if isinstance(value, mx.array):
            return ("array", id(value), value.shape, value.dtype)
        if isinstance(value, (list, tuple)):
            return tuple(freeze(v) for v in value)
        return value

    return [{key: freeze(value) for key, value in vars(part).items()} for part in cache]


def _count_updates(monkeypatch, pool):
    calls = []
    for name in ("accumulate_windows", "update_and_fetch"):
        real = getattr(pool, name)

        def counted(*args, _real=real, _name=name, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(pool, name, counted)
    return calls


def _pick(indexer, x, qr, cache, batched, projected=None):
    if batched:
        return indexer.small_windows_batched(
            x, qr, cache[1], cache[0], projected=projected
        )
    return indexer.small_windows(x, qr, cache[1], projected=projected)


def _assert_pool_state(indexer, actual, expected):
    assert indexer._processed(actual) == indexer._processed(expected)
    assert indexer._pool_lengths(actual) == indexer._pool_lengths(expected)
    assert actual.remainder == expected.remainder
    lengths = indexer._pool_lengths(actual)
    lengths = lengths if isinstance(lengths, list) else [lengths]
    remainders = actual.remainder
    remainders = remainders if isinstance(remainders, list) else [remainders]
    # Batch append may overshoot physical extent; trim discards that tail.
    # Compare every valid byte and remainder, retaining exact equality.
    for b, (length, remainder) in enumerate(zip(lengths, remainders)):
        assert mx.array_equal(
            actual.pooled[b, :length], expected.pooled[b, :length]
        ).item()
        assert mx.array_equal(
            actual.buf_kv[b, :remainder], expected.buf_kv[b, :remainder]
        ).item()
        assert mx.array_equal(
            actual.buf_gate[b, :remainder], expected.buf_gate[b, :remainder]
        ).item()


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "future_pool,topk,expected",
    [
        (511, 2048, False),
        (512, 2048, True),
        (513, 2048, True),
        (2049, 2048, True),
        (513, 2044, False),
        (513, 2052, False),
        (512, 2052, True),
    ],
)
def test_guard_future_selection_boundary_is_pure(
    attention, cpu_paths, batched, future_pool, topk, expected
):
    width = 3
    lengths = [future_pool * 4 - width]
    if batched:
        lengths.append(17)
    cache = _caches(attention.indexer, lengths, batched=batched)
    attention.indexer.index_topk = topk
    state = _snapshot(cache)
    guard = attention._batched_fused_ok if batched else attention._decode_fused_ok
    assert (
        bool(
            guard(len(lengths), width, 0, cache, _query(attention, len(lengths), width))
        )
        is expected
    )
    assert _snapshot(cache) == state


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "dtype,expected",
    [(mx.float16, True), (mx.bfloat16, True), (mx.float32, False), (mx.int32, False)],
)
def test_guard_dtype_metadata(attention, cpu_paths, batched, dtype, expected):
    cache = _caches(attention.indexer, [3001], batched=batched)
    state = _snapshot(cache)
    guard = attention._batched_fused_ok if batched else attention._decode_fused_ok
    assert bool(guard(1, 3, 0, cache, _query(attention, 1, 3, dtype))) is expected
    assert _snapshot(cache) == state


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "unsupported",
    [
        "native_off",
        "nax_off",
        "switch_off",
        "dense_rows",
        "zero_rows",
        "wide_rows",
        "query_batch",
        "query_heads",
        "query_width",
        "query_dim",
        "ratio",
        "tail_disabled",
        "index_dim",
        "attention_heads",
        "latent_dim",
        "remainder",
    ],
)
def test_guard_rejects_unsupported_without_mutation(
    monkeypatch, attention, cpu_paths, language, batched, unsupported
):
    cache = _caches(attention.indexer, [3001], batched=batched)
    width, dense_rows = 3, 0
    query = _query(attention, 1, width)
    if unsupported == "native_off":
        monkeypatch.setattr(fast, "has_symbol", lambda name: False)
    elif unsupported == "nax_off":
        monkeypatch.setattr(language, "nax_sparse_mla_available", lambda: False)
    elif unsupported == "switch_off":
        monkeypatch.setattr(language, "_SMALL_QUERY_INDEXER", False)
    elif unsupported == "dense_rows":
        dense_rows = 1
    elif unsupported in ("zero_rows", "wide_rows"):
        width = 0 if unsupported == "zero_rows" else 9
        query = _query(attention, 1, width)
    elif unsupported.startswith("query_"):
        axis = {"query_batch": 0, "query_heads": 1, "query_width": 2, "query_dim": 3}[
            unsupported
        ]
        shape = list(query.shape)
        shape[axis] += 1
        query.shape = tuple(shape)
    elif unsupported == "ratio":
        cache[1].ratio = 2
    elif unsupported == "tail_disabled":
        attention.indexer.index_kpool_always_select_tail = False
    elif unsupported == "index_dim":
        attention.indexer.head_dim = 129
    elif unsupported == "attention_heads":
        attention.num_heads = 17
        query = _query(attention, 1, width)
    elif unsupported == "latent_dim":
        attention.kv_lora_rank = 513
    else:
        cache[1].remainder = [4] if batched else 4
    state = _snapshot(cache)
    guard = attention._batched_fused_ok if batched else attention._decode_fused_ok
    assert not guard(1, width, dense_rows, cache, query)
    assert _snapshot(cache) == state


@pytest.mark.parametrize(
    "broken",
    [
        "processed",
        "pool_lengths",
        "remainder",
        "length_count",
        "right_padding",
        "leftpad_shape",
        "kv_shape",
    ],
)
def test_batch_guard_proves_complete_history(attention, cpu_paths, broken):
    cache = _caches(attention.indexer, [3001, 2103], batched=True)
    pool, kv = cache[1], cache[0]
    if broken == "processed":
        pool._processed[0] += 4
    elif broken == "pool_lengths":
        pool._pool_lengths[0] -= 1
    elif broken == "remainder":
        pool.remainder[1] = 0
    elif broken == "length_count":
        pool._processed.pop()
    elif broken == "right_padding":
        pool.prepare(lengths=[3, 2])
    elif broken == "leftpad_shape":
        kv.left_padding = mx.zeros((2, 1), mx.int32)
    else:
        kv.keys = mx.zeros((2, 2, 3100, 512), mx.float16)
    state = _snapshot(cache)
    assert not attention._batched_fused_ok(2, 3, 0, cache, _query(attention, 2, 3))
    assert _snapshot(cache) == state


def test_batch_one_row_and_scalar_cache_dispatch(attention, cpu_paths):
    query = _query(attention, 1, 3)
    scalar = _caches(attention.indexer, [3001])
    batched = _caches(attention.indexer, [3001], batched=True)
    assert isinstance(batched[1], BatchPoolingCache)
    assert attention._decode_fused_ok(1, 3, 0, scalar, query)
    assert not attention._decode_fused_ok(1, 3, 0, batched, query)
    assert attention._batched_fused_ok(1, 3, 0, batched, query)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("width", [1, 3, 8])
@pytest.mark.parametrize("length", [2047, 2051, 3001, 8201])
def test_small_windows_updates_once_and_preserves_selector_order(
    monkeypatch, attention, cpu_paths, small, batched, width, length
):
    lengths = [length, 17] if batched else [length]
    cache = _caches(attention.indexer, lengths, batched=batched)
    calls = _count_updates(monkeypatch, cache[1])
    x, qr = _inputs(len(lengths), width)
    picked = _pick(attention.indexer, x, qr, cache, batched)
    assert calls == ["accumulate_windows", "update_and_fetch"]
    after = [n + width for n in lengths]
    processed = attention.indexer._processed(cache[1])
    assert processed == (after if batched else after[0])
    plen = [n // 4 for n in after]
    assert attention.indexer._pool_lengths(cache[1]) == (plen if batched else plen[0])
    assert cache[1].remainder == ([n % 4 for n in after] if batched else after[0] % 4)
    if max(after) <= attention.indexer.index_topk:
        assert picked is None
        return
    if batched:
        selected, rows = picked
        left = cache[0].left_padding.tolist()
        assert rows.dtype == mx.int32
        assert rows.shape == (len(lengths), 3)
        assert rows.tolist() == [[p, n, lp] for p, n, lp in zip(plen, lengths, left)]
    else:
        selected, before, pool_length = picked
        assert (before, pool_length) == (length, plen[0])
        selected = selected[None]
    assert selected.shape == (len(lengths), width, 512)
    assert selected.dtype == mx.int32
    scores = _fake_scores_batched(
        mx.zeros((len(lengths), width, 32, 128), mx.float16),
        cache[1].pooled,
        None,
        lengths,
        plen,
        4,
    )
    native = fast.dsa_topk_indices(scores[:, None], 512)[:, 0]
    assert mx.array_equal(selected, native).item()
    before = lengths if batched else lengths[0]
    expanded = small.expand_windows(
        attention.indexer,
        selected,
        before,
        plen if batched else plen[0],
        cache[0].left_padding if batched else None,
    )
    assert expanded.shape == (len(lengths), 1, width, 2051)
    assert expanded.dtype == mx.int32


@pytest.mark.parametrize("batched", [False, True])
def test_short_dense_none_updates_once_without_scoring(
    monkeypatch, attention, cpu_paths, small, language, batched
):
    cache = _caches(attention.indexer, [17, 21] if batched else [17], batched=batched)
    monkeypatch.setattr(language, "small_scores", _forbidden)
    monkeypatch.setattr(small, "small_scores_batched", _forbidden)
    monkeypatch.setattr(fast, "dsa_topk_indices", _forbidden)
    calls = _count_updates(monkeypatch, cache[1])
    x, qr = _inputs(2 if batched else 1, 3)
    assert _pick(attention.indexer, x, qr, cache, batched) is None
    assert calls == ["accumulate_windows", "update_and_fetch"]
    assert attention.indexer._processed(cache[1]) == ([20, 24] if batched else 20)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "failure", [None, RuntimeError, TypeError, ValueError, AttributeError]
)
def test_selector_failure_stops_after_single_update(
    monkeypatch, attention, cpu_paths, batched, failure
):
    cache = _caches(
        attention.indexer, [3001, 2103] if batched else [3001], batched=batched
    )
    calls = _count_updates(monkeypatch, cache[1])
    selector_calls = []

    def failed(scores, count):
        selector_calls.append(count)
        if failure is not None:
            raise failure("선택 실패 시험")
        return None

    monkeypatch.setattr(fast, "dsa_topk_indices", failed)
    monkeypatch.setattr(mx, "argpartition", _forbidden)
    x, qr = _inputs(2 if batched else 1, 3)
    with pytest.raises(RuntimeError, match="native top-k"):
        _pick(attention.indexer, x, qr, cache, batched)
    assert selector_calls == [512]
    assert calls == ["accumulate_windows", "update_and_fetch"]
    assert attention.indexer._processed(cache[1]) == ([3004, 2106] if batched else 3004)


@pytest.mark.parametrize(
    "failure", [None, RuntimeError, TypeError, ValueError, AttributeError, "shape"]
)
def test_native_topk_required_never_argpartitions(monkeypatch, small, failure):
    monkeypatch.setattr(mx, "argpartition", _forbidden)
    calls = []

    def selector(scores, count):
        calls.append(count)
        if failure == "shape":
            return mx.zeros((1, 3, 511), mx.int32)
        if failure is not None:
            raise failure("선택 실패 시험")
        return None

    with pytest.raises(RuntimeError, match="native top-k"):
        small.native_topk_required(selector, mx.zeros((1, 3, 513), mx.float16), 512)
    assert calls == [512]


def _expanded_reference(indexer, selected, before, plen, left):
    batch, width, _ = selected.shape
    before = before if isinstance(before, list) else [before] * batch
    plen = plen if isinstance(plen, list) else [plen] * batch
    ratio = indexer.index_kpool
    tail_on = indexer.index_kpool_always_select_tail and ratio > 1
    output_width = indexer.index_topk + (ratio - 1 if tail_on else 0)
    expected = []
    for b, rows in enumerate(selected.tolist()):
        result = []
        for s, windows in enumerate(rows):
            position = before[b] + s
            tokens = []
            for window in windows:
                valid = 0 <= window < plen[b] and (window + 1) * ratio - 1 <= position
                tokens.extend(
                    [
                        window * ratio + i + left[b] if valid else -1
                        for i in range(ratio)
                    ]
                )
            if tail_on:
                count = (position + 1) % ratio
                start = position + 1 - count
                tokens.extend(
                    [start + i + left[b] if i < count else -1 for i in range(ratio - 1)]
                )
            tokens.extend([-1] * max(0, output_width - len(tokens)))
            result.append(tokens[:output_width])
        expected.append([result])
    return expected


@pytest.mark.parametrize("tail_mod", range(4))
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("scalar_metadata", [False, True])
def test_expansion_matches_ordered_token_reference(
    small, tail_mod, batched, scalar_metadata
):
    indexer = SimpleNamespace(
        index_kpool=4, index_topk=16, index_kpool_always_select_tail=True
    )
    batch, width = (2 if batched else 1), 4
    selected = mx.array(
        [[[2, -1, 0, 4], [3, 2, 1, 0], [4, 0, 2, 1], [5, 1, 3, 0]]] * batch, mx.int32
    )
    before = (
        12 + tail_mod
        if scalar_metadata
        else [12 + tail_mod + b * 4 for b in range(batch)]
    )
    plen = 5 if scalar_metadata else [4 + b for b in range(batch)]
    left = [7 * b for b in range(batch)]
    got = small.expand_windows(
        indexer, selected, before, plen, mx.array(left, mx.int32) if batched else None
    )
    assert got.shape == (batch, 1, width, 19)
    assert got.dtype == mx.int32
    assert got.tolist() == _expanded_reference(indexer, selected, before, plen, left)


@pytest.mark.parametrize("tail_on", [False, True])
def test_expansion_pads_short_selection(small, tail_on):
    indexer = SimpleNamespace(
        index_kpool=4, index_topk=16, index_kpool_always_select_tail=tail_on
    )
    selected = mx.array([[[1, 0]]], mx.uint32)
    got = small.expand_windows(indexer, selected, 10, 2)
    assert got.dtype == mx.int32
    assert got.tolist() == _expanded_reference(indexer, selected, 10, 2, [0])


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("provided", ["all", "partial"])
def test_projected_inputs_avoid_duplicate_projections(
    monkeypatch, attention, cpu_paths, language, batched, provided
):
    batch, width = (2 if batched else 1), 3
    cache = _caches(
        attention.indexer, [3001, 2103] if batched else [3001], batched=batched
    )
    x, qr = _inputs(batch, width)
    idx = attention.indexer
    projected = (idx.wq_b(qr), idx.wk(x), idx.weights_proj(x))
    if provided == "partial":
        projected = (projected[0], None, projected[2])
    calls = []
    real = language.linear_forward

    def counted(layer, value):
        calls.append(layer)
        return real(layer, value)

    monkeypatch.setattr(language, "linear_forward", counted)
    picked = _pick(idx, x, qr, cache, batched, projected)
    assert picked is not None
    assert calls == ([idx.wk] if provided == "partial" else [])


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("trim", [1, 3])
def test_verify_trim_then_decode_matches_kept_prefix(
    attention, cpu_paths, batched, trim
):
    lengths = [3001, 2103] if batched else [3001]
    cache = _caches(attention.indexer, lengths, batched=batched)
    reference = copy.deepcopy(cache)
    idx = attention.indexer
    x, qr = _inputs(len(lengths), 4)
    _pick(idx, x, qr, cache, batched)
    assert cache[1].trim(trim) == trim
    _pick(idx, x[:, : 4 - trim], qr[:, : 4 - trim], reference, batched)
    _assert_pool_state(idx, cache[1], reference[1])
    x, qr = _inputs(len(lengths), 1)
    got = _pick(idx, x, qr, cache, batched)
    ref = _pick(idx, x, qr, reference, batched)
    assert mx.array_equal(got[0], ref[0]).item()
    if batched:
        assert mx.array_equal(got[1], ref[1]).item()
    else:
        assert got[1:] == ref[1:]
    _assert_pool_state(idx, cache[1], reference[1])


@pytest.mark.parametrize("projection", [0, 1, 2])
@pytest.mark.parametrize("invalid", ["dtype", "shape"])
@pytest.mark.parametrize("batched", [False, True])
def test_forward_resolves_missing_projection_before_pool_and_reuses_rejected_output(
    monkeypatch, attention, cpu_paths, language, projection, invalid, batched
):
    lengths = [3001, 2103] if batched else [3001]
    batch, width = len(lengths), 3
    cache = _caches(attention.indexer, lengths, batched=batched)
    x, qr = _inputs(batch, width)
    q = mx.zeros((batch, width, attention.num_heads * attention.q_head_dim), mx.float16)
    kv = mx.zeros((batch, width, attention.kv_lora_rank), mx.float16)
    monkeypatch.setattr(
        language.Glm5NextSparseAttention,
        "_decode_projections",
        lambda self, x, cache: (qr, q, kv, None),
    )
    idx = attention.indexer
    layers = [idx.wq_b, idx.wk, idx.weights_proj]
    produced, calls = [None, None, None], []
    state = _snapshot([cache[1]])
    real = language.linear_forward

    def project(layer, value):
        result = real(layer, value)
        for i, candidate in enumerate(layers):
            if layer is candidate:
                assert _snapshot([cache[1]]) == state
                calls.append(i)
                if i == projection:
                    result = (
                        result.astype(mx.float32)
                        if invalid == "dtype"
                        else result[..., :-1]
                    )
                produced[i] = result
        return result

    class GeneralRouteReachedError(Exception):
        pass

    def general(self, *args, **kwargs):
        assert _snapshot([cache[1]]) == state
        assert all(value is produced[i] for i, value in enumerate(kwargs["projected"]))
        raise GeneralRouteReachedError

    monkeypatch.setattr(language, "linear_forward", project)
    monkeypatch.setattr(language.Glm5NextIndexer, "__call__", general)
    monkeypatch.setattr(language.Glm5NextIndexer, "small_windows", _forbidden)
    monkeypatch.setattr(language.Glm5NextIndexer, "small_windows_batched", _forbidden)
    with pytest.raises(GeneralRouteReachedError):
        attention._forward(x, None, cache)
    assert calls == [0, 1, 2]
    assert _snapshot([cache[1]]) == state


def test_forward_batch_route_preserves_scalar_before_with_left_padding(
    monkeypatch, attention, cpu_paths, language, small
):
    cache = _caches(attention.indexer, [3001])
    cache[0].left_padding = mx.array([0], mx.int32)
    assert not attention._decode_fused_ok(1, 3, 0, cache, _query(attention, 1, 3))
    assert attention._batched_fused_ok(1, 3, 0, cache, _query(attention, 1, 3))
    calls = _count_updates(monkeypatch, cache[1])
    before_values = []
    real_expand = small.expand_windows

    def expand(indexer, selected, before, plen, left_padding=None):
        before_values.append(before)
        assert before == 3001
        assert plen == 751
        return real_expand(indexer, selected, before, plen, left_padding)

    def gather(self, q, kv, indices):
        assert indices.shape == (1, 1, 3, 2051)
        return mx.zeros((1, 3, self.num_heads * self.v_head_dim), mx.float16)

    monkeypatch.setattr(language, "expand_windows", expand)
    monkeypatch.setattr(language.Glm5NextSparseAttention, "_gathered_attention", gather)
    monkeypatch.setattr(language.Glm5NextIndexer, "small_windows", _forbidden)
    monkeypatch.setattr(language.Glm5NextIndexer, "__call__", _forbidden)
    x, _ = _inputs(1, 3)
    mx.eval(attention._forward(x, None, cache))
    assert before_values == [3001]
    assert calls == ["accumulate_windows", "update_and_fetch"]


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("native_on", [False, True])
def test_forward_selects_once_and_uses_existing_gathered_attention(
    monkeypatch, attention, cpu_paths, language, small, batched, native_on
):
    lengths = [3001, 2103] if batched else [3001]
    cache = _caches(attention.indexer, lengths, batched=batched)
    monkeypatch.setattr(
        fast, "has_symbol", lambda name: native_on and name == "dsa_topk_indices"
    )
    calls = _count_updates(monkeypatch, cache[1])
    routes, expanded_calls, gathered = [], [], []
    for name in ("small_windows", "small_windows_batched", "__call__"):
        real = getattr(language.Glm5NextIndexer, name)

        def counted(self, *args, _real=real, _name=name, **kwargs):
            routes.append(_name)
            return _real(self, *args, **kwargs)

        monkeypatch.setattr(language.Glm5NextIndexer, name, counted)
    real_expand = small.expand_windows

    def expansion(indexer, selected, before, plen, left_padding=None):
        out = real_expand(indexer, selected, before, plen, left_padding)
        expanded_calls.append((selected, before, plen, left_padding, out))
        return out

    def gather(self, q, kv, indices):
        gathered.append(indices)
        return mx.zeros((len(lengths), 3, self.num_heads * self.v_head_dim), mx.float16)

    monkeypatch.setattr(language, "expand_windows", expansion)
    monkeypatch.setattr(language.Glm5NextSparseAttention, "_gathered_attention", gather)
    x, _ = _inputs(len(lengths), 3)
    out = attention._forward(x, None, cache)
    mx.eval(out)
    assert routes == (
        ["small_windows_batched" if batched else "small_windows"]
        if native_on
        else ["__call__"]
    )
    assert calls == ["accumulate_windows", "update_and_fetch"]
    assert len(gathered) == 1
    assert len(expanded_calls) == int(native_on)
    assert gathered[0].shape == (len(lengths), 1, 3, 2051)
    assert gathered[0].dtype == mx.int32
    if native_on:
        selected, before, plen, left, expanded = expanded_calls[0]
        assert selected.shape == (len(lengths), 3, 512)
        assert before == (lengths if batched else lengths[0])
        assert plen == (
            [(n + 3) // 4 for n in lengths] if batched else (lengths[0] + 3) // 4
        )
        assert gathered[0] is expanded
        assert expanded.tolist() == _expanded_reference(
            attention.indexer,
            selected,
            before,
            plen,
            left.tolist() if left is not None else [0],
        )


@pytest.mark.parametrize(
    "bad_input", ["x", "qr", "projected_q", "projected_k", "projected_weights"]
)
def test_forward_rejects_mixed_projection_dtype_before_pool_mutation(
    monkeypatch, attention, cpu_paths, language, bad_input
):
    cache = _caches(attention.indexer, [3001])
    x, qr = _inputs(1, 3)
    q = mx.zeros((1, 3, attention.num_heads * attention.q_head_dim), mx.float16)
    kv = mx.zeros((1, 3, attention.kv_lora_rank), mx.float16)
    projected = [
        mx.zeros((1, 3, 32 * 128), mx.float16),
        mx.zeros((1, 3, 128), mx.float16),
        mx.zeros((1, 3, 32), mx.float16),
    ]
    if bad_input == "x":
        x = x.astype(mx.float32)
    elif bad_input == "qr":
        qr = qr.astype(mx.float32)
    else:
        i = {"projected_q": 0, "projected_k": 1, "projected_weights": 2}[bad_input]
        projected[i] = projected[i].astype(mx.float32)
    monkeypatch.setattr(
        language.Glm5NextSparseAttention,
        "_decode_projections",
        lambda self, x, cache: (qr, q, kv, tuple(projected)),
    )
    monkeypatch.setattr(language.Glm5NextIndexer, "small_windows", _forbidden)
    monkeypatch.setattr(language.Glm5NextIndexer, "small_windows_batched", _forbidden)
    state = _snapshot([cache[1]])

    class GeneralRouteReachedError(Exception):
        pass

    def general(self, *args, **kwargs):
        assert _snapshot([cache[1]]) == state
        raise GeneralRouteReachedError

    monkeypatch.setattr(language.Glm5NextIndexer, "__call__", general)
    with pytest.raises(GeneralRouteReachedError):
        attention._forward(x, None, cache)


@pytest.mark.parametrize("batched", [False, True])
def test_forward_dense_none_does_not_retry_indexer(
    monkeypatch, attention, cpu_paths, language, batched
):
    lengths = [2045, 2044] if batched else [2045]
    cache = _caches(attention.indexer, lengths, batched=batched)
    calls = _count_updates(monkeypatch, cache[1])
    monkeypatch.setattr(language.Glm5NextIndexer, "__call__", _forbidden)
    x, _ = _inputs(len(lengths), 3)
    mx.eval(attention._forward(x, None, cache))
    assert calls == ["accumulate_windows", "update_and_fetch"]


@pytest.mark.parametrize("batched", [False, True])
def test_forward_native_failure_does_not_retry_indexer(
    monkeypatch, attention, cpu_paths, language, batched
):
    lengths = [3001, 2103] if batched else [3001]
    cache = _caches(attention.indexer, lengths, batched=batched)
    calls = _count_updates(monkeypatch, cache[1])
    monkeypatch.setattr(fast, "dsa_topk_indices", lambda scores, count: None)
    monkeypatch.setattr(language.Glm5NextIndexer, "__call__", _forbidden)
    monkeypatch.setattr(mx, "argpartition", _forbidden)
    x, _ = _inputs(len(lengths), 3)
    with pytest.raises(RuntimeError, match="native top-k"):
        attention._forward(x, None, cache)
    assert calls == ["accumulate_windows", "update_and_fetch"]


def test_pool_storage_uses_real_batch_backing_stride(attention, cpu_paths, small):
    cache = _caches(attention.indexer, [2103, 3001], batched=True)
    pool = cache[1]
    x, qr = _inputs(2, 4)
    attention.indexer.small_windows_batched(x, qr, pool, cache[0])
    keys = pool.pooled
    assert pool._pool_buf.shape[1] > keys.shape[1]
    storage, stride = small._pool_storage(keys, pool)
    assert storage is pool._pool_buf
    assert stride == pool._pool_buf.shape[1] * 128
    assert mx.array_equal(storage[:, : keys.shape[1]], keys).item()
    storage, stride = small._pool_storage(keys, None)
    assert storage is keys
    assert stride == keys.shape[1] * 128


@pytest.mark.parametrize("broken", ["extent", "batch", "dim", "dtype", "capacity"])
def test_pool_storage_declines_mismatched_backing(small, broken):
    keys = mx.zeros((2, 37, 128), mx.float16)
    pool = SimpleNamespace(
        _pool_extent=37, _pool_buf=mx.zeros((2, 64, 128), mx.float16)
    )
    if broken == "extent":
        pool._pool_extent = 36
    else:
        shape = {
            "batch": (1, 64, 128),
            "dim": (2, 64, 64),
            "dtype": (2, 64, 128),
            "capacity": (2, 36, 128),
        }[broken]
        pool._pool_buf = mx.zeros(
            shape, mx.float32 if broken == "dtype" else mx.float16
        )
    storage, stride = small._pool_storage(keys, pool)
    assert storage is keys
    assert stride == 37 * 128


@pytest.mark.parametrize(
    "broken",
    [
        "rank",
        "head_dim",
        "keys_dim",
        "weight_shape",
        "dtype",
        "keys_dtype",
        "weights_dtype",
        "negative_before",
        "plen",
        "row_count",
    ],
)
def test_score_validation_rejects_before_kernel(monkeypatch, small, broken):
    # Metadata-only inputs cannot accidentally perform BF16 arithmetic on CPU.
    def metadata(shape, dtype=mx.float16):
        return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype)

    q, keys, weights = (
        metadata((2, 3, 32, 128)),
        metadata((2, 37, 128)),
        metadata((2, 3, 32)),
    )
    before, plen = [140, 91], [37, 23]
    if broken == "rank":
        q = metadata((2, 3, 128))
    elif broken == "head_dim":
        q = metadata((2, 3, 32, 127))
    elif broken == "keys_dim":
        keys = metadata((2, 37, 64))
    elif broken == "weight_shape":
        weights = metadata((2, 2, 32))
    elif broken == "dtype":
        q.dtype = mx.float32
    elif broken == "keys_dtype":
        keys.dtype = mx.float32
    elif broken == "weights_dtype":
        weights.dtype = mx.float32
    elif broken == "negative_before":
        before[0] = -1
    elif broken == "plen":
        plen[0] = 38
    else:
        before.pop()
    monkeypatch.setattr(small, "_get_batched_kernel", _forbidden)
    with pytest.raises(ValueError):
        small.small_scores_batched(q, keys, weights, before, plen, 4)


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="실제 Metal 점수 커널 시험입니다"
)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("width", [1, 3, 8])
def test_metal_scores_match_fp32_reference(small, dtype, width):
    mx.random.seed(0)
    heads, dim, extent, ratio = 32, 128, 37, 4
    q = mx.random.normal((1, width, heads, dim)).astype(dtype)
    keys = mx.random.normal((1, extent, dim)).astype(dtype)
    weights = (mx.random.normal((1, width, heads)) * 0.1).astype(dtype)
    before = 140
    with mx.stream(mx.gpu):
        got = small.small_scores(q, keys, weights, before, ratio).astype(mx.float32)
        ref = mx.einsum(
            "shd,pd->shp", q[0].astype(mx.float32), keys[0].astype(mx.float32)
        )
        ref = (mx.maximum(ref, 0) * weights[0].astype(mx.float32)[..., None]).sum(
            axis=1
        )
        valid = (mx.arange(extent)[None] + 1) * ratio - 1 <= before + mx.arange(width)[
            :, None
        ]
        ref = ref.astype(dtype).astype(mx.float32)
        assert got.shape == (1, width, extent)
        assert mx.array_equal(got[0] <= -1e29, ~valid).item()
        assert mx.isfinite(mx.where(valid, got[0], 0)).all().item()
        # test_glm53_dsa_decode_port.py at 40157fa3 uses these score tolerances.
        assert mx.allclose(
            mx.where(valid, got[0], 0), mx.where(valid, ref, 0), atol=0.02, rtol=0.01
        ).item()


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="실제 Metal 배치 점수 커널 시험입니다"
)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_metal_batched_scores_bitwise_match_scalar_with_storage(small, dtype):
    mx.random.seed(7)
    batch, width, heads, dim, ratio = 3, 4, 32, 128, 4
    plen, before = [37, 12, 29], [150, 51, 118]
    extent = max(plen)
    with mx.stream(mx.gpu):
        q = mx.random.normal((batch, width, heads, dim)).astype(dtype)
        backing = mx.random.normal((batch, 64, dim)).astype(dtype)
        keys = backing[:, :extent]
        weights = (mx.random.normal((batch, width, heads)) * 0.1).astype(dtype)
        pool = SimpleNamespace(_pool_buf=backing, _pool_extent=extent)
        got = small.small_scores_batched(q, keys, weights, before, plen, ratio, pool)
        assert got.shape == (batch, width, extent)
        assert got.dtype == dtype
        for b in range(batch):
            ref = small.small_scores(
                q[b : b + 1],
                keys[b : b + 1, : plen[b]],
                weights[b : b + 1],
                before[b],
                ratio,
            )
            # Preserve the original batched score kernel's bitwise contract.
            assert mx.array_equal(got[b : b + 1, :, : plen[b]], ref).item(), b
            assert (got[b, :, plen[b] :] <= -1e29).all().item()


@pytest.mark.skipif(
    not mx.metal.is_available(), reason="실제 native selector 순서 시험입니다"
)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_metal_native_order_on_identical_scores(language, small, dtype):
    if not fast.has_symbol("dsa_topk_indices"):
        pytest.skip("native dsa_topk_indices 심볼이 없습니다")
    with mx.stream(mx.gpu):
        scores = mx.broadcast_to((mx.arange(513) % 17)[None, None], (2, 3, 513)).astype(
            dtype
        )
        expected = fast.dsa_topk_indices(scores[:, None], 512)[:, 0]
        selected = small.native_topk_required(
            language.Glm5NextIndexer._native_topk, scores, 512
        )
        assert selected.dtype == mx.int32
        assert mx.array_equal(selected, expected).item()
