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
from omlx.patches import vlm_batch_kv_capacity as capacity
from omlx.patches.mlx_vlm_qwen4_exp_compat import (
    apply_mlx_vlm_qwen4_exp_compat_patch,
)
from omlx.patches.vlm_batch_kv_capacity import _refill_order

apply_mlx_vlm_qwen4_exp_compat_patch()

# qwen4_exp carries its own BatchKVCache; importing its language module puts
# that class on the same capacity methods.
from mlx_vlm.models.qwen4_exp import cache as qwen4_cache  # noqa: E402
from mlx_vlm.models.qwen4_exp import language as _qwen4_language  # noqa: E402


def pristine(module):
    spec = importlib.util.spec_from_file_location("_stock_cache", module.__file__)
    stock = importlib.util.module_from_spec(spec)
    stock.__package__ = module.__package__
    spec.loader.exec_module(stock)
    return stock


@pytest.fixture(
    params=[vlm_cache, lm_cache, qwen4_cache], ids=["vlm", "lm-restored", "qwen4_exp"]
)
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
            [
                "append",
                "merge",
                "trim",
                "rollback",
                "filter",
                "finish",
                "extend",
                "restore",
            ]
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
        elif operation == "finish":
            # The row order GenerationBatch.filter passes after rows finish.
            rows = a.keys.shape[0]
            kept = _refill_order(sorted(rng.sample(range(rows), rng.randint(1, rows))))
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
    assert 8192 < capacity <= 8192 * 1.25 + 512
    for _ in range(16):
        x = mx.ones((2, 2, 5, 8), mx.bfloat16)
        c.update_and_fetch(x, x)
        assert c.keys.shape[2] == capacity
    row = c.extract(0)
    if patched is not lm_cache:
        assert row.keys.shape[2] > row.offset
    else:
        assert row.keys.shape[2] == row.offset


def test_finished_row_is_refilled_without_copying_the_bank(backend):
    stock, patched = backend
    lengths = [4000, 3990, 3999, 3000, 3998, 3997, 3996, 3995]
    a, b = make(stock, lengths), make(patched, lengths)
    x = mx.ones((8, 2, 3, 8), mx.bfloat16)
    kept = _refill_order([0, 1, 2, 4, 5, 6, 7])
    for c in (a, b):
        c.update_and_fetch(x, x)
        c.filter(kept)
        c.update_and_fetch(x[:7], x[:7])
    # The bank keeps its eight rows; the freed slot holds the last row.
    assert b._keys.shape[0] == 8 and b.keys.shape[0] == 7
    equal(a, b)


def test_setting_keys_alone_after_a_refill_keeps_rows_aligned(backend):
    stock, patched = backend
    a, b = make(stock, [40, 39, 38, 37]), make(patched, [40, 39, 38, 37])
    x = mx.ones((3, 2, 2, 8), mx.bfloat16)
    for c in (a, b):
        c.filter(_refill_order([0, 1, 3]))
        c.keys = mx.depends(c.keys, c.values)
        c.update_and_fetch(x, x)
    assert b.keys.shape[0] == b.values.shape[0] == 3
    equal(a, b)


def test_join_fills_a_freed_row_and_shifts_live_rows_in_place(backend):
    stock, patched = backend
    lengths = [300, 290, 299, 200]
    a, b = make(stock, lengths), make(patched, lengths)
    # A row shorter than the batch, then one longer that moves the live rows right.
    for joined in (120, 700):
        for c in (a, b):
            c.filter(_refill_order([0, 1, 2]))
        bank = b._keys
        a.extend(make(stock, [joined]))
        b.extend(make(patched, [joined]))
        assert b._keys is bank
        equal(a, b)
    assert b._idx == 700


def test_rows_freed_down_to_one_stay_for_the_next_joins(backend):
    stock, patched = backend
    lengths = [300, 290, 299, 280]
    a, b = make(stock, lengths), make(patched, lengths)
    for c in (a, b):
        c.filter(_refill_order([2]))
    bank = b._keys
    assert bank.shape[0] == 4 and b.keys.shape[0] == 1
    a.extend(make(stock, [250, 310]))
    b.extend(make(patched, [250, 310]))
    assert b._keys is bank
    equal(a, b)


def test_join_that_needs_more_columns_keeps_the_spare_rows(backend):
    stock, patched = backend
    lengths = [300, 290, 299, 280]
    a, b = make(stock, lengths), make(patched, lengths)
    for c in (a, b):
        c.filter(_refill_order([0, 1]))
    a.extend(make(stock, [2000]))
    b.extend(make(patched, [2000]))
    bank = b._keys
    assert bank.shape[0] == 4 and b.keys.shape[0] == 3
    equal(a, b)
    a.extend(make(stock, [500]))
    b.extend(make(patched, [500]))
    assert b._keys is bank
    equal(a, b)


@pytest.mark.parametrize("kept, rows", [([0, 1], 4), ([2], 1)])
def test_early_growth_moves_one_bank_per_allowance(backend, monkeypatch, kept, rows):
    stock, patched = backend
    monkeypatch.setattr(capacity, "_EARLY_GROWTH_MIN_BYTES", 0)
    monkeypatch.setattr(capacity, "_early_growth", {"allowed": True, "grown": False})
    lengths = [20000, 19990, 19999, 19980]
    a, b, other = make(stock, lengths), make(patched, lengths), make(patched, lengths)
    for c in (a, b, other):
        c.filter(_refill_order(kept))
    x = mx.ones((len(kept), 2, 64, 8), mx.bfloat16)
    for _ in range(100):
        banks = b._keys, other._keys
        for c in (a, b, other):
            c.update_and_fetch(x, x)
        if capacity.take_early_kv_growth():
            break
    else:
        pytest.fail("no bank grew early")
    # Only the first bank to need columns grew; the next waits for an allowance.
    assert b._keys is not banks[0] and other._keys is banks[1]
    assert b._keys.shape[0] == rows and b._keys.shape[2] > banks[0].shape[2]
    equal(a, b)
    equal(a, other)


def test_finish_of_the_longest_row_drops_the_padding_in_place(backend):
    stock, patched = backend
    lengths = [300, 290, 299, 200]
    a, b = make(stock, lengths), make(patched, lengths)
    for c in (a, b):
        c.filter(_refill_order([1, 2, 3]))
    # The spare row stays for the next join.
    assert b._keys.shape[0] == 4 and b.keys.shape[0] == 3
    equal(a, b)


def test_ragged_rollback_keeps_the_spare_rows(backend):
    stock, patched = backend
    lengths = [300, 290, 299, 200]
    a, b = make(stock, lengths), make(patched, lengths)
    x = mx.ones((3, 2, 9, 8), mx.bfloat16)
    for c in (a, b):
        c.filter(_refill_order([0, 1, 2]))
        c.update_and_fetch(x, x)
        c.prepare(right_padding=[0, 6, 3])
        c.finalize()
    assert b._keys.shape[0] == 4 and b.keys.shape[0] == 3
    equal(a, b)


def test_join_after_an_in_place_drop_leaves_no_stale_columns(backend):
    stock, patched = backend
    lengths = [3000, 2500, 2600, 2800]
    a, b = make(stock, lengths), make(patched, lengths)
    x = mx.ones((4, 2, 1, 8), mx.bfloat16)
    for c, module in ((a, stock), (b, patched)):
        c.filter(_refill_order([1, 2, 3]))
        c.extend(make(module, [1000]))
        # Growth inside the capacity turns spare columns into logical ones.
        c.update_and_fetch(x, x)
    equal(a, b)


def test_finish_that_drops_most_of_the_width_releases_the_bank(backend):
    stock, patched = backend
    lengths = [8192, 500, 480, 490]
    a, b = make(stock, lengths), make(patched, lengths)
    for c in (a, b):
        c.filter(_refill_order([1, 2, 3]))
    assert b._keys.shape[0] == 3 and b._keys.shape[2] < 2048
    equal(a, b)


@pytest.mark.parametrize("hold_view", [False, True])
@pytest.mark.parametrize("shift", [6, 20])
def test_ragged_rollback_matches_stock_on_every_roll_path(backend, shift, hold_view):
    """Native in place, native copy (a view holds the bank) and the gather fallback."""
    stock, patched = backend
    lengths = [300, 290, 299]
    a, b = make(stock, lengths), make(patched, lengths)
    x = mx.ones((3, 2, 24, 8), mx.bfloat16)
    for c in (a, b):
        c.update_and_fetch(x, x)
    held = b._keys[:1] if hold_view else None
    for c in (a, b):
        c.prepare(right_padding=[0, shift, 3])
        c.finalize()
    equal(a, b)
    del held


def test_setting_the_bank_back_keeps_its_logical_width(backend):
    stock, patched = backend
    a, b = make(stock, [300, 290, 280]), make(patched, [300, 290, 280])
    # DSA indexers and MTP heads write the bank back through mx.depends each step.
    for _ in range(3):
        for c, module in ((a, stock), (b, patched)):
            c.keys = mx.depends(c.keys, c.values)
            c.extend(make(module, [100]))
            c.filter([0, 1, 2])
        equal(a, b)


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


def test_qwen4_qsa_batch_decode_appends_in_place():
    rows = []
    for row, length in enumerate([600, 550]):
        c = _qwen4_language.QSAKVCache()
        x = mx.full((1, 1, length, 8), row + 1, mx.bfloat16)
        c.update_and_fetch(x, x / 4)
        c.update_indexer(
            mx.ones((1, length, 4), mx.bfloat16),
            mx.arange(length, dtype=mx.int32)[None],
        )
        rows.append(c)
    batch = _qwen4_language.BatchQSAKVCache.merge(rows)
    capacity = batch.keys.shape[2]
    # 400 tokens cross a 256-column step, where the stock class concatenates.
    for step in range(100):
        x = mx.full((2, 1, 4, 8), step, mx.bfloat16)
        keys, _ = batch.update_and_fetch(x, x / 4)
        assert keys.shape[2] == 600 + 4 * (step + 1)
        assert batch.keys.shape[2] == capacity
