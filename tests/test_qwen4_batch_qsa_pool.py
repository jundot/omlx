# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp batched QSA cache: allocation-only changes stay bit-exact.

Covers the MLX-pool fixes for 2+ row (Lightning MTP) batches:

* ``BatchQSAKVCache.extract`` rows round their first append to one step
  instead of geometrically doubling the whole row.
* ``BatchKVCache.merge`` reserves the first append step, producing exactly
  the physical layout the old exact merge + first append produced.
* ``BatchQSAKVCache`` indexer appends write into a stepped buffer; every
  update/trim/ragged-finalize sequence matches the old concatenate path
  bit-for-bit, and the backing width only changes on step crossings.
* The pool-reclaim hint raised after a batch row-cache rebuild becomes one
  ordinary deferred Metal clear.
"""

import random
import threading

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.mlx_vlm_qwen4_exp_compat import (  # noqa: E402
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp.cache import BatchKVCache, dynamic_roll  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    BatchQSAKVCache,
    QSAKVCache,
    _append_indexer_positions,
)

H, D, DI = 1, 8, 4


def _bits(a):
    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    return np.array(a)


def _assert_same(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    np.testing.assert_array_equal(_bits(a), _bits(b))


def _rnd(shape, key):
    return mx.random.normal(shape, key=mx.random.key(key)).astype(mx.bfloat16)


def _row(length, key):
    cache = QSAKVCache()
    cache.update_and_fetch(
        _rnd((1, H, length, D), key), _rnd((1, H, length, D), key + 1)
    )
    cache.update_indexer(
        _rnd((1, length, DI), key + 2), mx.arange(length, dtype=mx.int32)[None]
    )
    return cache


class _ExactIndexer:
    """The pre-patch BatchQSAKVCache indexer bookkeeping (exact widths)."""

    def __init__(self, keys, positions):
        self.index_keys = keys
        self.index_position_ids = positions
        self.index_offset = keys.shape[1]

    def update(self, keys, positions):
        self.index_keys = mx.concatenate([self.index_keys, keys], axis=1)
        self.index_position_ids = _append_indexer_positions(
            self.index_position_ids, positions
        )
        self.index_offset = self.index_keys.shape[1]

    def trim(self, n):
        self.index_offset -= n
        self.index_keys = self.index_keys[:, : self.index_offset]
        self.index_position_ids = self.index_position_ids[..., : self.index_offset]

    def roll(self, right_padding):
        self.index_keys = dynamic_roll(self.index_keys, right_padding, axis=1)
        if self.index_position_ids.ndim == 3:
            self.index_position_ids = dynamic_roll(
                self.index_position_ids, right_padding[None], axis=2
            )
        else:
            self.index_position_ids = dynamic_roll(
                self.index_position_ids, right_padding, axis=1
            )


def test_batch_extract_rounds_first_append_to_one_step():
    length = 5000  # 2*L > one 8192 step, so doubling would show
    batch = BatchQSAKVCache.merge([_row(length, 1), _row(length - 3, 11)])
    row = batch.extract(0)
    assert row._geometric_capacity_managed is False
    row.update_and_fetch(_rnd((1, H, 1, D), 21), _rnd((1, H, 1, D), 22))
    assert row.keys.shape[2] == 8192  # was max(8192, 2 * 5000) = 16384


def test_merge_reserves_first_step_with_the_old_first_append_layout():
    rows = [_row(300, 1), _row(250, 11)]
    merged = BatchKVCache.merge(rows)
    assert merged.size() == 300
    assert merged.keys.shape[2] == 300 + BatchKVCache.step

    # Old behaviour: exact-width merge, then the first append concatenates.
    old = BatchKVCache([0, 50])
    old.state = (
        merged.keys[..., :300, :],
        merged.values[..., :300, :],
        merged.offset,
        merged.left_padding,
    )
    k, v = _rnd((2, H, 4, D), 31), _rnd((2, H, 4, D), 32)
    merged.update_and_fetch(k, v)
    old.update_and_fetch(k, v)
    assert merged.keys.shape == old.keys.shape
    _assert_same(merged.keys, old.keys)
    _assert_same(merged.values, old.values)
    assert merged._idx == old._idx
    np.testing.assert_array_equal(np.array(merged.offset), np.array(old.offset))


@pytest.mark.parametrize("mrope", [False, True])
def test_indexer_capacity_matches_concatenate_bit_for_bit(mrope):
    rng = random.Random(7)
    base = 700
    rows = [_row(base, 1), _row(base - 9, 11)]
    batch = BatchQSAKVCache.merge(rows)
    ref = _ExactIndexer(batch.index_keys, batch.index_position_ids)
    widths = set()
    pos = [base, base - 9]
    for step in range(300):
        m = 4
        keys = _rnd((2, m, DI), 1000 + step)
        text = mx.array([[p + i for i in range(m)] for p in pos], dtype=mx.int32)
        positions = mx.broadcast_to(text[None], (3, 2, m)) if mrope else text
        batch.kv_cache.update_and_fetch(
            _rnd((2, H, m, D), 5000 + step), _rnd((2, H, m, D), 9000 + step)
        )
        got_k, got_p = batch.update_indexer(keys, positions)
        ref.update(keys, positions)
        _assert_same(got_k, ref.index_keys)
        _assert_same(got_p, ref.index_position_ids)

        retained = [rng.randint(1, m) for _ in range(2)]
        keep = max(retained)
        batch.trim(m - keep)
        ref.trim(m - keep)
        right = [keep - r for r in retained]
        if any(right):
            batch.prepare(right_padding=right)
            batch.finalize()
            ref.roll(mx.array(right))
        mx.eval(batch._index_keys, batch._index_position_ids)
        _assert_same(batch.index_keys, ref.index_keys)
        _assert_same(batch.index_position_ids, ref.index_position_ids)
        assert batch.index_offset == ref.index_offset == batch.kv_cache.size()
        widths.add(int(batch._index_keys.shape[1]))
        pos = [p + r for p, r in zip(pos, retained)]

    # Stepped width: only a handful of distinct backing widths.
    assert all(w % BatchQSAKVCache.index_step == 0 for w in widths)
    assert len(widths) <= 2
    _assert_same(batch.state[1], ref.index_keys)


def test_externally_assigned_indexer_keeps_plain_attribute_semantics():
    batch = BatchQSAKVCache([0])
    keys = _rnd((1, 4, DI), 3)
    batch.index_keys = keys
    batch.index_position_ids = mx.arange(4, dtype=mx.int32)[None]
    # index_offset intentionally left at 0, as the old attribute allowed.
    _assert_same(batch.index_keys, keys)
    batch.update_indexer(_rnd((1, 1, DI), 4), mx.array([[4]], dtype=mx.int32))
    assert batch.index_offset == 5
    assert batch.index_keys.shape == (1, 5, DI)


def test_pool_reclaim_hint_is_thread_local_and_one_shot():
    from omlx.utils.metal_sync import (
        consume_pool_reclaim_request,
        request_pool_reclaim,
    )

    assert consume_pool_reclaim_request() is False
    request_pool_reclaim()
    seen = []
    worker = threading.Thread(
        target=lambda: seen.append(consume_pool_reclaim_request())
    )
    worker.start()
    worker.join()
    assert seen == [False]
    assert consume_pool_reclaim_request() is True
    assert consume_pool_reclaim_request() is False


def test_replace_cache_rows_raises_hint_and_scheduler_defers_clear(
    mock_model, mock_tokenizer
):
    from types import SimpleNamespace

    from omlx.patches.mlx_lm_mtp import batch_generator as bg
    from omlx.scheduler import Scheduler
    from omlx.utils.metal_sync import consume_pool_reclaim_request

    consume_pool_reclaim_request()
    gen = SimpleNamespace(uids=[1, 2], prompt_cache=None)
    rows = {0: [_row(20, 1)], 1: [_row(18, 11)]}
    bg._replace_cache_rows(gen, rows)
    assert isinstance(gen.prompt_cache[0], BatchQSAKVCache)

    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    assert scheduler._deferred_clear_at is None
    scheduler._schedule_hinted_pool_reclaim()
    assert scheduler._deferred_clear_at == (
        scheduler._step_counter + Scheduler._DEFERRED_CLEAR_DELAY
    )
    scheduler._deferred_clear_at = None
    scheduler._schedule_hinted_pool_reclaim()
    assert scheduler._deferred_clear_at is None


def test_filter_after_managed_appends_keeps_keys_and_positions_aligned():
    batch = BatchQSAKVCache.merge([_row(40, 1), _row(36, 11)])
    ref = _ExactIndexer(batch.index_keys, batch.index_position_ids)
    for step in range(3):
        keys = _rnd((2, 2, DI), 70 + step)
        pos = mx.array([[40 + 2 * step, 41 + 2 * step]] * 2, dtype=mx.int32)
        batch.kv_cache.update_and_fetch(
            _rnd((2, H, 2, D), 80 + step), _rnd((2, H, 2, D), 90 + step)
        )
        batch.update_indexer(keys, pos)
        ref.update(keys, pos)
    assert batch._index_capacity_managed
    batch.filter(mx.array([1]))
    assert batch.index_keys.shape[1] == batch.index_position_ids.shape[-1]
    assert batch.index_offset == batch.kv_cache.size()
    _assert_same(batch.index_keys, ref.index_keys[1:2, 4:])
    _assert_same(batch.index_position_ids, ref.index_position_ids[1:2, 4:])
