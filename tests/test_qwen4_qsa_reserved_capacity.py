"""QSA capacity reservation and prefix preservation tests."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from omlx.scheduler import Scheduler

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


def test_restored_index_reserves_full_horizon_on_first_growth():
    cache = language.QSAKVCache()
    cache.index_step = 64
    cache.state = (None, None, mx.ones((1, 1088, 8)), mx.arange(1088)[None])
    cache.reserve_index_capacity(2000)
    cache.update_indexer(mx.ones((1, 1, 8)), mx.array([[1088]]))
    mx.eval(cache.index_keys)
    assert cache._index_keys.shape[1] == 2048
    assert mx.all(cache.index_keys == 1).item()


@pytest.mark.parametrize("snapshots_enabled", [False, True])
def test_chunked_reservation_includes_restored_prefix(snapshots_enabled):
    cache = language.QSAKVCache()
    cache.state = (
        mx.zeros((1, 1, 128, 4)),
        mx.zeros((1, 1, 128, 4)),
        mx.ones((1, 128, 8)),
        mx.arange(128)[None],
    )
    ns = SimpleNamespace(
        model=object(),
        config=SimpleNamespace(paged_cache_block_size=64),
        block_aware_cache=object() if snapshots_enabled else None,
        _stream=mx.default_stream(mx.gpu),
        _prefill_tail_plan=lambda *a: (None, False),
        _turboquant_kv_bits=None,
    )
    state = Scheduler._begin_prefill(
        ns,
        SimpleNamespace(cached_tokens=128, request_id="reservation"),
        [1] * 129,
        [cache, SimpleNamespace(offset=128)],
    )
    ns._prefill_step_size_for_progress = lambda *a: 64
    ns._reserve_qsa_index_capacity = (
        lambda caches, tokens: Scheduler._reserve_qsa_index_capacity(ns, caches, tokens)
    )

    def stop(*args, **kwargs):
        raise RuntimeError("reservation complete")

    ns._adaptive_chunk_size = stop
    with pytest.raises(RuntimeError, match="reservation complete"):
        Scheduler._step_prefill_chunk(ns, state)
    assert cache._index_reserved_tokens >= 256


def test_reserved_qsa_prefill_and_decode_match_unreserved_model():
    from mlx_vlm.models.qwen4_exp.language import LanguageModel
    from test_mlx_vlm_qwen4_exp_compat import _tiny_config

    mx.random.seed(3455)
    config = _tiny_config()
    model = LanguageModel(config.text_config, config)
    ordinary, reserved = model.make_cache(), model.make_cache()
    for cache in ordinary + reserved:
        if hasattr(cache, "index_step"):
            cache.index_step = 8
    Scheduler._reserve_qsa_index_capacity(object(), reserved, 36)
    for start in range(0, 36, 6):
        tokens = (mx.arange(start, start + 6) % 60 + 1)[None]
        expected = model(tokens, cache=ordinary).logits
        actual = model(tokens, cache=reserved).logits
        mx.eval(expected, actual)
        assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
    for _ in range(5):
        token = mx.argmax(expected[:, -1], axis=-1)[:, None]
        assert mx.array_equal(token, mx.argmax(actual[:, -1], axis=-1)[:, None]).item()
        expected = model(token, cache=ordinary).logits
        actual = model(token, cache=reserved).logits
        mx.eval(expected, actual)
        assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
