# SPDX-License-Identifier: Apache-2.0
"""A reserved QSA prompt horizon sizes the singleton KV buffer once.

The scheduler reserves the whole prompt on every QSA cache before the first
prefill chunk (``reserve_index_capacity``). The indexer already allocated that
horizon at once; the K/V buffer grew by fixed 8192-token steps, so a cold
135K prefill reallocated 16 times and copied its growing prefix every time
(the old and new buffers coexisting at each step). K/V now allocates the
reserved horizon on the first append too. Only capacity changes: every
returned K/V array and every logical state is bit-identical.
"""

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.mlx_vlm_qwen4_exp_compat import (  # noqa: E402
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp.language import QSAKVCache  # noqa: E402

H, D, DI = 2, 16, 8
STEP = 64  # instance override keeps the allocations test-sized


def _bits(a):
    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    return np.array(a)


def _rnd(shape, key):
    return mx.random.normal(shape, key=mx.random.key(key)).astype(mx.bfloat16)


def _cache(reserve=0):
    cache = QSAKVCache()
    cache.step = STEP
    cache.index_step = STEP
    if reserve:
        cache.reserve_index_capacity(reserve)
    return cache


def _capacity(cache):
    return int(cache.keys.shape[2])


def _feed(cache, start, stop, key=0):
    n = stop - start
    k = _rnd((1, H, n, D), key + start)
    v = _rnd((1, H, n, D), key + 7919 + start)
    got = cache.update_and_fetch(k, v)
    got += cache.update_indexer(
        _rnd((1, n, DI), key + 104729 + start),
        mx.arange(start, stop, dtype=mx.int32)[None],
    )
    mx.eval(*got)
    return got


def _prefill(cache, total, chunk, key=0):
    capacities, outputs = [], []
    for start in range(0, total, chunk):
        outputs.append(_feed(cache, start, min(total, start + chunk), key))
        capacities.append(_capacity(cache))
    return capacities, outputs


@pytest.mark.parametrize("chunk", [7, 30, 64, 200])
def test_reserved_cold_prefill_allocates_kv_once(chunk):
    cache = _cache(reserve=200)

    capacities, _ = _prefill(cache, 200, chunk)

    # ceil(200 / 64) * 64 from the first chunk on: no regrowth copies.
    assert capacities == [256] * len(capacities)


@pytest.mark.parametrize("chunk", [7, 30, 64, 200])
def test_reserved_prefill_is_bit_identical_to_unreserved(chunk):
    reserved, plain = _cache(reserve=200), _cache()

    _, got = _prefill(reserved, 200, chunk)
    _, want = _prefill(plain, 200, chunk)

    for got_step, want_step in zip(got, want):
        for a, b in zip(got_step, want_step):
            assert a.shape == b.shape and a.dtype == b.dtype
            np.testing.assert_array_equal(_bits(a), _bits(b))
    for a, b in zip(reserved.state, plain.state):
        np.testing.assert_array_equal(_bits(a), _bits(b))
    # Past the logical prefix the reserved buffer is zero, as after a grow.
    assert not np.any(_bits(reserved.keys[..., 200:, :]))


def test_decode_past_the_reservation_grows_by_one_step():
    cache = _cache(reserve=200)
    _prefill(cache, 200, 50)

    _feed(cache, 200, 256)
    assert _capacity(cache) == 256
    _feed(cache, 256, 257)

    # Beyond the horizon growth is the plain fixed step (256 -> 320).
    assert _capacity(cache) == 320


def test_unreserved_growth_keeps_fixed_steps():
    cache = _cache()

    capacities, _ = _prefill(cache, 200, 30)

    assert capacities == [64, 64, 128, 128, 192, 192, 256]


def test_reservation_below_the_current_capacity_is_ignored():
    cache = _cache()
    _prefill(cache, 130, 130)
    assert _capacity(cache) == 192

    cache.reserve_index_capacity(150)
    _feed(cache, 130, 200)

    # 192 >= 150: the reservation is already covered, so plain steps apply.
    assert _capacity(cache) == 256


def test_reservation_after_a_restore_sizes_the_first_regrowth():
    source = _cache()
    _prefill(source, 100, 100)
    restored = QSAKVCache()
    restored.state = source.state
    restored.step = STEP
    restored.reserve_index_capacity(300)

    _feed(restored, 100, 101, key=5)

    assert _capacity(restored) == 320
    np.testing.assert_array_equal(
        _bits(restored.keys[..., :100, :]), _bits(source.state[0])
    )
