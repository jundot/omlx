# SPDX-License-Identifier: Apache-2.0
"""Compare the patched generic cache with the installed, unmodified source."""

import importlib.util
import random

import mlx.core as mx
import pytest
from mlx_lm.models import cache as lm_cache
from mlx_vlm.models import cache as vlm_cache

import omlx.scheduler  # noqa: F401
from omlx.cache.type_handlers import KVCacheHandler


def pristine(module):
    spec = importlib.util.spec_from_file_location("_stock_cache", module.__file__)
    stock = importlib.util.module_from_spec(spec)
    stock.__package__ = module.__package__
    spec.loader.exec_module(stock)
    return stock


@pytest.fixture(params=[vlm_cache, lm_cache], ids=["vlm", "lm-restored"])
def backend(request):
    return pristine(request.param), request.param


def make(module, lengths):
    rows = []
    for row, length in enumerate(lengths):
        c = module.KVCache()
        x = (mx.arange(2 * length * 8).reshape(1, 2, length, 8) % 29 + row).astype(
            mx.bfloat16
        )
        c.update_and_fetch(x, x / 4)
        rows.append(c)
    return module.BatchKVCache.merge(rows)


def equal(a, b):
    for x, y in zip(a.state, b.state):
        if x is None or y is None:
            assert x is y
        elif isinstance(x, mx.array):
            assert x.shape == y.shape
            assert mx.array_equal(x, y).item()
        else:
            assert x == y
    assert a._idx == b._idx
    if a.keys is None:
        return
    assert b._logical_width() == a.keys.shape[2]
    assert mx.array_equal(a.keys, b.keys[..., : b._logical_width(), :]).item()
    assert mx.array_equal(a.values, b.values[..., : b._logical_width(), :]).item()
    # Include masked padding, stale speculative columns and the actual SDPA.
    q = mx.ones((a.keys.shape[0], 2, 1, 8), mx.bfloat16) / 8
    outputs = []
    for c in (a, b):
        mask = (
            mx.arange(c._idx)[None, None, None, :]
            >= c.left_padding[:, None, None, None]
        )
        outputs.append(
            mx.fast.scaled_dot_product_attention(
                q,
                c.keys[..., : c._idx, :],
                c.values[..., : c._idx, :],
                scale=0.25,
                mask=mask,
            )
        )
    assert mx.array_equal(*outputs).item()
    for i in range(a.keys.shape[0]):
        for x, y in zip(a.extract(i).state, b.extract(i).state):
            assert mx.array_equal(x, y).item()


@pytest.mark.parametrize("seed", range(8))
def test_cache_and_attention_match_stock_through_lifecycle(seed, backend):
    stock, patched = backend
    rng = random.Random(seed)
    a, b = make(stock, [271, 269, 134]), make(patched, [271, 269, 134])
    equal(a, b)
    for step in range(40):
        operation = rng.choice(
            ["append", "merge", "trim", "rollback", "filter", "extend", "restore"]
        )
        if operation in ("append", "rollback"):
            n = rng.randint(1, 17)
            x = mx.full((a.keys.shape[0], 2, n, 8), step + 1, mx.bfloat16)
            for c in (a, b):
                c.update_and_fetch(x, x / 4)
            if operation == "rollback":
                padding = [rng.randrange(n) for _ in range(a.keys.shape[0])]
                for c in (a, b):
                    c.prepare(right_padding=padding)
                    c.finalize()
        elif operation == "trim":
            n = min(rng.randint(1, 7), max(0, min(a.offset.tolist()) - 1))
            assert a.trim(n) == b.trim(n)
        elif operation == "merge":
            a = stock.BatchKVCache.merge([a.extract(i) for i in range(a.keys.shape[0])])
            b = patched.BatchKVCache.merge(
                [b.extract(i) for i in range(b.keys.shape[0])]
            )
        elif operation == "filter":
            kept = rng.sample(range(a.keys.shape[0]), rng.randint(1, a.keys.shape[0]))
            a.filter(kept)
            b.filter(kept)
        elif operation == "extend" and a.keys.shape[0] < 6:
            lengths = [rng.randint(1, 290)]
            a.extend(make(stock, lengths))
            b.extend(make(patched, lengths))
        elif operation == "restore":
            # The prefix restore contract assigns serialized state to a fresh cache.
            restored = []
            for module, c in ((stock, a), (patched, b)):
                clone = module.BatchKVCache([0] * c.keys.shape[0])
                clone.state = tuple(
                    mx.array(x) if isinstance(x, mx.array) else x for x in c.state
                )
                restored.append(clone)
            a, b = restored
        equal(a, b)


def test_merge_reserves_append_without_changing_logical_width(backend):
    _, patched = backend
    c = make(patched, [8192, 8187])
    assert c._logical_width() == 8192
    capacity = c.keys.shape[2]
    assert 8192 < capacity <= 8192 * 1.125 + 512
    for _ in range(16):
        x = mx.ones((2, 2, 5, 8), mx.bfloat16)
        c.update_and_fetch(x, x)
        assert c.keys.shape[2] == capacity
    row = c.extract(0)
    if patched is vlm_cache:
        assert row.keys.shape[2] > row.offset
    else:
        assert row.keys.shape[2] == row.offset


def test_prefix_handler_restore_uses_capacity_managed_cache():
    handler = KVCacheHandler()
    source = make(vlm_cache, [513]).extract(0)
    state = handler.extract_state(source)
    parts = [handler.slice_state(state, 0, 256), handler.slice_state(state, 256, 513)]
    restored = handler.reconstruct_cache(handler.concatenate_states(parts))
    # The actual SSD restore handler chooses mlx-lm, even for a VLM model.
    assert type(restored) is lm_cache.KVCache
    batch = restored.merge([restored])
    assert type(batch) is lm_cache.BatchKVCache
    assert batch.keys.shape[2] > batch._logical_width() == 513
    stock = pristine(lm_cache)
    expected = stock.KVCache()
    expected.state = (*source.state, source.offset)
    expected = stock.BatchKVCache.merge([expected])
    equal(expected, batch)
    x = mx.ones((1, 2, 17, 8), mx.bfloat16)
    for c in (expected, batch):
        c.update_and_fetch(x, x / 4)
    equal(expected, batch)


@pytest.mark.parametrize("empty_first", [False, True])
def test_empty_extend_preserves_backend_state_and_dtype(backend, empty_first):
    results = []
    for module in backend:
        empty = module.BatchKVCache([0])
        full = make(module, [17])
        a, b = (empty, full) if empty_first else (full, empty)
        a.extend(b)
        results.append(a.state)
    for x, y in zip(*results):
        if isinstance(x, mx.array):
            assert x.dtype == y.dtype
            assert mx.array_equal(x, y).item()
        else:
            assert x == y


def test_cold_and_restored_backends_can_share_a_batch(backend):
    stock, patched = backend
    other = lm_cache if patched is vlm_cache else vlm_cache
    a, b = make(stock, [271, 269]), make(patched, [271, 269])
    a.extend(make(pristine(other), [513]))
    b.extend(make(other, [513]))
    equal(a, b)
    x = mx.ones((3, 2, 7, 8), mx.bfloat16)
    for c in (a, b):
        c.update_and_fetch(x, x / 4)
    equal(a, b)
