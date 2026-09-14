"""QSA capacity reservation and prefix preservation tests."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib

import mlx.core as mx

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")

_STEP = 64  # instance override keeps the allocations test-sized


def _cache(reserve: int = 0):
    cache = language.QSAKVCache()
    cache.index_step = _STEP
    if reserve:
        cache.reserve_index_capacity(reserve)
    return cache


def _append(cache, start: int, stop: int) -> None:
    length = stop - start
    raw = mx.arange(1 * stop * 8, dtype=mx.float32).reshape(1, stop, 8)
    cache.update_indexer(raw[:, start:stop], mx.arange(start, stop, dtype=mx.int32)[None])


def _capacity(cache) -> int:
    return 0 if cache._index_keys is None else int(cache._index_keys.shape[1])


def test_reservation_lands_the_first_allocation_on_the_final_size():
    cache = _cache(reserve=200)

    _append(cache, 0, 8)

    # ceil(200 / 64) * 64 — the whole prompt fits in one allocation.
    assert _capacity(cache) == 256


def test_reserved_indexer_never_doubles_past_the_reservation():
    cache = _cache(reserve=200)

    _append(cache, 0, 8)
    _append(cache, 8, 200)
    assert _capacity(cache) == 256  # no grow while staying inside the horizon

    _append(cache, 200, 257)  # decode crossed the reserved horizon

    # A step (256 -> 320), not a doubling (which would be 512).
    assert _capacity(cache) == 320
    assert int(cache.index_keys.shape[1]) == 257  # logical view tracks length


def test_unreserved_growth_still_doubles():
    cache = _cache()

    _append(cache, 0, 8)
    assert _capacity(cache) == _STEP  # seed
    _append(cache, 8, 65)
    assert _capacity(cache) == 128
    _append(cache, 65, 129)

    # Unknown horizon: 128 -> 256 doubling is still the amortizing choice.
    assert _capacity(cache) == 256


def test_reservation_preserves_the_prefix_across_the_single_grow():
    cache = _cache(reserve=200)

    _append(cache, 0, 8)
    first = mx.array(cache.index_keys[:, :8, :])
    _append(cache, 8, 200)
    mx.eval(first, cache.index_keys)

    assert mx.array_equal(first, mx.array(cache.index_keys[:, :8, :]))


def test_restored_cache_below_the_reservation_lands_on_it_not_double():
    # A prefix hit restores capacity sized for the cached prefix. The legacy
    # policy doubled from there (measured on a warm 212k turn: 2.84 -> 5.59 GB
    # in one chunk, capacity for 229,376 tokens on a 212,068-token prompt), and
    # kept climbing the ladder whenever the prompt outgrew the doubling.
    cache = _cache(reserve=2000)

    capacity = cache._next_capacity(1088, 1089, _STEP)

    assert capacity == 2048  # ceil(2000 / 64) * 64
    assert capacity != 2176  # what doubling from 1088 would have produced


def test_next_capacity_policy_table():
    cache = _cache(reserve=200)

    # Tokens, first grow: land on the reservation.
    assert cache._next_capacity(0, 8, _STEP) == 256
    # Tokens, past the reservation: plain steps.
    assert cache._next_capacity(256, 257, _STEP) == 320
    # Block units (pooled bank): the reservation arrives pre-divided.
    assert cache._next_capacity(0, 10, 8, reserve=12) == 16
    assert cache._next_capacity(16, 17, 8, reserve=12) == 24
    # No reservation: unchanged legacy policy.
    unreserved = _cache()
    assert unreserved._next_capacity(64, 65, _STEP) == 128


def test_scheduler_reservation_skips_non_qsa_caches():
    from omlx.scheduler import Scheduler

    class _Plain:
        pass

    class _QSA:
        def __init__(self):
            self.reserved = 0

        def reserve_index_capacity(self, tokens):
            self.reserved = tokens

    qsa_a, qsa_b = _QSA(), _QSA()
    count = Scheduler._reserve_qsa_index_capacity(
        object(), [qsa_a, _Plain(), qsa_b], 12345
    )

    assert count == 2
    assert (qsa_a.reserved, qsa_b.reserved) == (12345, 12345)
    assert Scheduler._reserve_qsa_index_capacity(object(), [qsa_a], 0) == 0
