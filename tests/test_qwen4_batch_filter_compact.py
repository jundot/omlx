# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp batch ``filter`` copies survivors once, onto the ladder.

A finishing row used to cost two survivor banks: ``filter`` gathered the
kept rows at full capacity and sliced the shared left padding off as a
strided view, and the next append copied that view again at an exact width
(``capacity - min_left``) that differs on every finish, so MLX's pool could
never hand it back. With a 147K-token row re-joining every few seconds that
stranded ~8 GB per finish until a pool clear.

Only allocation changes: every observable matches the pre-change layout
bit-for-bit (``_ConcatBatchKV`` / ``_ExactIndexer`` references).
"""

import mlx.core as mx
import numpy as np
import pytest

from tests.test_qwen4_batch_kv_capacity import (
    H,
    D,
    _ConcatBatchKV,
    _assert_equivalent,
    _bits,
    _rnd,
    _row,
    _same,
)
from tests.test_qwen4_batch_qsa_pool import DI, _ExactIndexer
from tests.test_qwen4_batch_qsa_pool import _row as _qsa_row

from mlx_vlm.models.qwen4_exp.cache import BatchKVCache, _ladder_capacity  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache  # noqa: E402


def _pair(lengths, key=1):
    rows = [_row(n, key + 10 * i) for i, n in enumerate(lengths)]
    return BatchKVCache.merge(rows), _ConcatBatchKV.merge(rows)


def _append(new, ref, m, key):
    B = int(new.offset.shape[0])
    k, v = _rnd((B, H, m, D), key), _rnd((B, H, m, D), key + 1)
    got, want = new.update_and_fetch(k, v), ref.update_and_fetch(k, v)
    _same(got[0], want[0])
    _same(got[1], want[1])


@pytest.mark.parametrize("keep", [[0, 1], [0, 2], [1, 2], [1]])
def test_filter_drops_shared_padding_in_one_ladder_copy(keep):
    # Row 2 is the long re-joining row: dropping it shifts out the others'
    # shared left padding.
    new, ref = _pair([900, 40, 1500])
    for step in range(3):
        _append(new, ref, 4, 100 + 2 * step)
    new.filter(mx.array(keep))
    ref.filter(mx.array(keep))
    mx.eval(new.keys, new.values)
    _assert_equivalent(new, ref)
    capacity = int(new.keys.shape[2])
    assert capacity == _ladder_capacity(new._width, BatchKVCache.step)
    # The survivors decode in place: no second copy on the first append.
    for step in range(16):
        _append(new, ref, 4, 300 + 2 * step)
        assert int(new.keys.shape[2]) == capacity
    _assert_equivalent(new, ref)


def test_identity_filter_keeps_the_bank():
    new, ref = _pair([300, 300])
    _append(new, ref, 2, 7)
    keys = new.keys
    new.filter(mx.array([0, 1]))
    ref.filter(mx.array([0, 1]))
    assert new.keys is keys
    _assert_equivalent(new, ref)


def test_filter_of_exact_width_bank_matches_gather():
    # ``state`` assignment adopts an exact-width bank (``_width is None``).
    lengths = [60, 20, 45]
    keys = _rnd((3, H, 60, D), 3)
    offset = mx.array(lengths)
    pads = mx.array([60 - n for n in lengths])
    new, ref = BatchKVCache(pads.tolist()), _ConcatBatchKV(pads.tolist())
    new.state = ref.state = (keys, keys, offset, pads)
    new.filter(mx.array([1, 2]))
    ref.filter(mx.array([1, 2]))
    _assert_equivalent(new, ref)
    _append(new, ref, 3, 9)
    _assert_equivalent(new, ref)


def test_finish_capacity_is_reused_across_turns():
    # The compacted width is piecewise constant, so a later finish of a
    # slightly longer batch lands on the buffer size the last one released.
    shapes = set()
    for turn in range(4):
        new, ref = _pair([4000 + 30 * turn, 900, 4200 + 150 * turn], key=turn)
        new.filter(mx.array([0, 1]))
        ref.filter(mx.array([0, 1]))
        _assert_equivalent(new, ref)
        shapes.add(tuple(new.keys.shape))
    assert len(shapes) == 1


@pytest.mark.parametrize("mrope", [False, True])
def test_qsa_filter_lands_indexer_on_ladder(mrope):
    batch = BatchQSAKVCache.merge(
        [_qsa_row(90, 1), _qsa_row(30, 11), _qsa_row(140, 21)]
    )
    positions = batch.index_position_ids
    if mrope:
        positions = mx.broadcast_to(positions[None], (3, *positions.shape))
        batch.index_position_ids = positions
    ref = _ExactIndexer(batch.index_keys, batch.index_position_ids)

    def append(rows, key):
        keys = _rnd((rows, 2, DI), key)
        text = mx.broadcast_to(mx.array([[key, key + 1]], dtype=mx.int32), (rows, 2))
        pos = mx.broadcast_to(text[None], (3, rows, 2)) if mrope else text
        batch.kv_cache.update_and_fetch(
            _rnd((rows, 1, 2, 8), key + 1), _rnd((rows, 1, 2, 8), key + 2)
        )
        got_k, got_p = batch.update_indexer(keys, pos)
        ref.update(keys, pos)
        _same(got_k, ref.index_keys)
        _same(got_p, ref.index_position_ids)

    append(3, 500)
    batch.filter(mx.array([0, 1]))
    shift = 140 - 90
    ref.index_keys = ref.index_keys[:2, shift:]
    ref.index_position_ids = ref.index_position_ids[..., :2, shift:]
    ref.index_offset -= shift
    mx.eval(batch._index_keys, batch._index_position_ids)
    _same(batch.index_keys, ref.index_keys)
    _same(batch.index_position_ids, ref.index_position_ids)
    assert batch.index_offset == batch.kv_cache.size() == ref.index_offset
    assert batch._index_capacity_managed
    width = int(batch._index_keys.shape[1])
    assert not np.any(_bits(batch._index_keys[:, batch.index_offset :]))
    for step in range(6):
        append(2, 600 + 4 * step)
        assert int(batch._index_keys.shape[1]) == width
