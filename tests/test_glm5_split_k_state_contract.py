# SPDX-License-Identifier: Apache-2.0
"""Exact accepted cache bits and persistent row-address fault contracts."""

import copy
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache

from omlx.patches.deepseek_v4.cache_extras import BatchPoolingCache, PoolingCache
from tests import test_glm5_split_k_attention as source


def test_nonoverlap_carry_difference_only_is_ignored():
    batch, rows = _rows()
    pool = batch[7][1]
    pool.prev_win_kv = mx.ones((2, 1, pool.ratio, 4), mx.bfloat16)
    pool.prev_win_gate = mx.zeros_like(pool.prev_win_kv)
    pool._prev_valid = [True, True]
    source._assert_cache_rows(batch, rows, 0, "verify", overlap=False)
    with pytest.raises(AssertionError, match="prev_win_kv"):
        source._assert_cache_rows(batch, rows, 0, "verify", overlap=True)


def test_nonoverlap_actual_indexer_flag_change_fails():
    source.glm5_next_vlm_runtime.apply()
    host = source._host()
    assert source._assert_glm_nonoverlap(host) is False
    host.model.layers[3].self_attn.indexer.overlap = True
    with pytest.raises(AssertionError, match="overlap changed"):
        source._assert_glm_nonoverlap(host)


def test_nonoverlap_other_state_bits_stay_exact():
    batch, rows = _rows()
    value = batch[7][1].buf_kv
    raw = value.view(mx.uint16)
    mask = mx.zeros(raw.shape, mx.uint16)
    mask[1, 0, 0] = 1
    batch[7][1].buf_kv = (raw ^ mask).view(mx.bfloat16)
    with pytest.raises(source.ContinuityMismatchError, match="cache state"):
        source._assert_cache_rows(batch, rows, 0, "verify", overlap=False)


def _rows():
    rows = []
    for row, length in enumerate((4, 5)):
        kv = KVCache()
        keys = (mx.arange(length * 4).reshape(1, 1, length, 4) + row * 100).astype(
            mx.bfloat16
        )
        kv.update_and_fetch(keys, keys)
        pool = PoolingCache(128)
        pool.remainder = length
        pool.buf_kv = mx.ones((1, length, 4), mx.bfloat16)
        pool.buf_gate = mx.zeros((1, length, 4), mx.bfloat16)
        rows.append(
            [CacheList(copy.deepcopy(kv), copy.deepcopy(pool)) for _ in range(8)]
        )
    batch = [
        CacheList(
            BatchKVCache.merge([r[layer][0] for r in rows]),
            BatchPoolingCache.merge([r[layer][1] for r in rows]),
        )
        for layer in range(8)
    ]
    for layer in batch:
        layer[0].keys = mx.concatenate(
            [layer[0].keys, mx.full((2, 1, 2, 4), 999, mx.bfloat16)], axis=2
        )
    return batch, rows


def test_attention_argument_mapping_does_not_change_persistent_kv():
    batch, rows = _rows()
    original = mx.array(batch[7][0].keys)
    wrong_argument = mx.take(batch[7][0].keys, mx.array([0, 0], mx.int32), axis=0)
    assert not mx.array_equal(original, wrong_argument).item()
    assert mx.array_equal(
        original.view(mx.uint16), batch[7][0].keys.view(mx.uint16)
    ).item()
    source._assert_cache_rows(batch, rows, 1, "verify")


@pytest.mark.parametrize("fault", ["cache", "rollback", "indexing"])
def test_persistent_fault_is_detected_by_actual_kv_check(monkeypatch, fault):
    batch, rows = _rows()
    trace = SimpleNamespace(state_checks=[])
    monkeypatch.setattr(source.language, "_glm40_test_trace", trace, raising=False)
    before = mx.array(batch[7][0].keys)
    if fault == "indexing":
        mutation = source._damage_row_mapping_kv(batch, 7)
        assert mutation == {
            "layer": 7,
            "row": 1,
            "source_row": 0,
            "physical_target": 4,
            "logical_target": 4,
            "physical_source": 4,
            "kind": "KV values",
        }
    else:
        source._damage_accepted_kv(batch, 7, 3, fault == "rollback")
    assert mx.array_equal(
        before[0].view(mx.uint16), batch[7][0].keys[0].view(mx.uint16)
    ).item()
    assert mx.array_equal(
        before[1, :, :4].view(mx.uint16), batch[7][0].keys[1, :, :4].view(mx.uint16)
    ).item()
    assert mx.array_equal(
        before[1, :, 5:].view(mx.uint16), batch[7][0].keys[1, :, 5:].view(mx.uint16)
    ).item()
    with pytest.raises(source.ContinuityMismatchError) as caught:
        source._assert_cache_rows(batch, rows, 1, "verify")
    assert caught.value.kind == "KV values" and caught.value.context == (1, "verify", 1)
    assert (
        trace.state_checks[-1]["layer"] == 7
        and not trace.state_checks[-1]["raw_bits_equal"]
    )


def test_retained_pool_bits_are_required():
    batch, rows = _rows()
    batch[7][1].buf_kv[1, 0, 0] = 2
    with pytest.raises(source.ContinuityMismatchError, match="cache state"):
        source._assert_cache_rows(batch, rows, 1, "verify")


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_recurrent_cache_requires_exact_bits(dtype):
    batch = ArraysCache(2)
    batch[0], batch[1] = mx.ones((2, 2, 4), dtype), mx.zeros((2, 2, 4), dtype)
    rows = [[batch.extract(i)] for i in range(2)]
    source._assert_cache_rows([batch], rows, 1, "verify")
    batch[1][1, 0, 0] = -0.0
    with pytest.raises(source.ContinuityMismatchError, match="cache state"):
        source._assert_cache_rows([batch], rows, 1, "verify")


def _c0_pool():
    # The tiny-model row0 prefill plus its first accepted verify token ends a ratio-4 window.
    length = source.TINY_TEXT_CONFIG["index_kpool"]
    width = (
        128  # _host's index_head_dim; a real pooled key and its gate have this width.
    )
    pool = PoolingCache(length)
    pool.pooled = mx.ones((1, 2052 // length, width), mx.bfloat16)
    pool.remainder = 0
    pool.buf_kv = mx.ones((1, length, width), mx.bfloat16)
    pool.buf_gate = mx.zeros((1, length, width), mx.bfloat16)
    return pool


@pytest.mark.parametrize("name", ["buf_kv", "buf_gate"])
def test_c0_logical_zero_allocation_and_reset_are_equivalent(name):
    pool = _c0_pool()
    live = source._pool_live_buffer(pool, name)
    assert live.shape == (1, 0, pool.pooled.shape[-1])
    assert source._normalize_zero_pool_views(None, live) == (None, None)
    setattr(pool, name, mx.zeros((1, 0, pool.pooled.shape[-1]), mx.bfloat16))
    live = source._pool_live_buffer(pool, name)
    assert live.shape == (1, 0, pool.pooled.shape[-1])
    assert source._normalize_zero_pool_views(live, None) == (None, None)
    setattr(pool, name, None)
    assert source._pool_live_buffer(pool, name) is None


@pytest.mark.parametrize(
    "case", ["nonempty", "rank", "batch", "dtype", "different_width", "different_dtype"]
)
def test_zero_normalizer_rejects_invalid_logical_view(case):
    a = mx.zeros((1, 0, 128), mx.bfloat16)
    b = None
    if case == "nonempty":
        a = mx.zeros((1, 1, 128), mx.bfloat16)
    elif case == "rank":
        a = a[0]
    elif case == "batch":
        a = mx.zeros((2, 0, 128), mx.bfloat16)
    elif case == "dtype":
        a = a.astype(mx.float16)
    elif case == "different_width":
        b = mx.zeros((1, 0, 127), mx.bfloat16)
    else:
        b = mx.zeros((1, 0, 128), mx.float32)
    with pytest.raises(AssertionError):
        source._normalize_zero_pool_views(a, b)


def test_other_nonempty_previous_window_is_not_normalized():
    batch, rows = _rows()
    pool = batch[7][1]
    pool.prev_win_kv = mx.ones((2, 1, pool.ratio, 4), mx.bfloat16)
    pool.prev_win_gate = mx.zeros_like(pool.prev_win_kv)
    pool._prev_valid = [True, False]
    with pytest.raises(AssertionError, match="prev_win_kv"):
        source._assert_cache_rows(batch, rows, 0, "verify")


@pytest.mark.parametrize(
    "case",
    [
        "missing_live",
        "too_short",
        "rank",
        "batch",
        "width",
        "dtype",
        "capacity",
        "negative_remainder",
    ],
)
def test_pool_buffer_rejects_invalid_live_range_and_storage(case):
    pool = _c0_pool()
    if case == "missing_live":
        pool.remainder, pool.buf_kv = 1, None
    elif case == "too_short":
        pool.remainder, pool.buf_kv = 1, pool.buf_kv[:, :0]
    elif case == "rank":
        pool.buf_kv = pool.buf_kv[0]
    elif case == "batch":
        pool.buf_kv = mx.concatenate([pool.buf_kv, pool.buf_kv])
    elif case == "width":
        pool.buf_kv = pool.buf_kv[:, :, :-1]
    elif case == "dtype":
        pool.buf_kv = pool.buf_kv.astype(mx.float32)
    elif case == "capacity":
        pool.buf_kv = mx.concatenate([pool.buf_kv, pool.buf_kv[:, :1]], axis=1)
    else:
        pool.remainder = -1
    with pytest.raises(AssertionError):
        source._pool_live_buffer(pool, "buf_kv")


@pytest.mark.parametrize(
    "case", ["remainder", "counter", "logical_length", "kv_offset", "ratio", "live_bit"]
)
def test_pool_and_kv_metadata_and_live_bits_remain_exact(case):
    batch, rows = _rows()
    expected = rows[1][7][1]
    if case == "remainder":
        expected.remainder += 1
    elif case == "counter":
        batch[7][1]._processed[1] += 1
    elif case == "logical_length":
        batch[7][1]._pool_lengths[1] += 1
    elif case == "kv_offset":
        rows[1][7][0].offset -= 1
    elif case == "ratio":
        expected.ratio += 1
    else:
        bits = batch[7][1].buf_kv.view(mx.uint16)
        bits[1, 0, 0] = bits[1, 0, 0] + 1
        batch[7][1].buf_kv = bits.view(mx.bfloat16)
    with pytest.raises(AssertionError):
        source._assert_cache_rows(batch, rows, 1, "verify")


@pytest.mark.parametrize("fault", ["cache", "rollback", "indexing"])
def test_state_fault_direct_preservation_record(fault):
    batch, rows = _rows()
    record = source._inject_state_fault(batch, 7, 3, fault)
    assert record["row0_bits_unchanged"] and record["prior_accepted_bits_unchanged"]
    assert record["rejected_tail_bits_unchanged"] and record["accepted_prefix"] == 3
    assert record["changed_word_count"] == sum(
        a != b
        for a, b in zip(record["target_bits_before"], record["target_bits_after"])
    )
    with pytest.raises(source.ContinuityMismatchError, match="KV values"):
        source._assert_cache_rows(batch, rows, 1, "verify")
