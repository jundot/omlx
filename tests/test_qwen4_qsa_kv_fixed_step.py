# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp singleton QSA KV grows by fixed steps, never by doubling.

``QSAKVCache`` / ``QSAQuantizedKVCache`` used to grow to
``max(ceil_step(L + S), 2 * capacity)``. A row that crossed a power of two
(a 135K prompt, say) then reserved ~2x its KV. Growth now rounds the
requirement up to one step. Only the backing capacity changes: the logical
prefix ``keys[..., :offset]`` is bit-identical to a plain concatenation of
every appended chunk.
"""

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.mlx_vlm_qwen4_exp_compat import (  # noqa: E402
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    BatchQSAKVCache,
    QSAKVCache,
    QSAQuantizedKVCache,
)

H, D, DI = 1, 8, 4
STEP = 64  # instance override keeps the allocations test-sized


def _bits(a):
    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    return np.array(a)


def _rnd(shape, key):
    return mx.random.normal(shape, key=mx.random.key(key)).astype(mx.bfloat16)


def _ceil(n, step=STEP):
    return ((n + step - 1) // step) * step


def _capacity(cache):
    keys = cache.keys[0] if isinstance(cache.keys, tuple) else cache.keys
    return int(keys.shape[2])


@pytest.mark.parametrize("chunk", [1, 7, 16, 64, 100])
def test_prefill_capacity_is_the_requirement_rounded_to_one_step(chunk):
    cache = QSAKVCache()
    cache.step = STEP
    appended_k, appended_v = [], []
    total = 0
    for i in range(0, 300, chunk):
        n = min(chunk, 300 - i)
        k, v = _rnd((1, H, n, D), 10 + i), _rnd((1, H, n, D), 5000 + i)
        got_k, got_v = cache.update_and_fetch(k, v)
        appended_k.append(k)
        appended_v.append(v)
        total += n
        # Never more than one step of slack (doubling reached 2x).
        assert _capacity(cache) == _ceil(total)
        np.testing.assert_array_equal(
            _bits(got_k), _bits(mx.concatenate(appended_k, axis=2))
        )
        np.testing.assert_array_equal(
            _bits(got_v), _bits(mx.concatenate(appended_v, axis=2))
        )


def test_crossing_a_power_of_two_does_not_double():
    cache = QSAKVCache()
    cache.step = STEP
    cache.update_and_fetch(_rnd((1, H, 256, D), 1), _rnd((1, H, 256, D), 2))
    assert _capacity(cache) == 256
    cache.update_and_fetch(_rnd((1, H, 14, D), 3), _rnd((1, H, 14, D), 4))
    # 270 tokens: one step past 256 (the old policy reserved 512).
    assert _capacity(cache) == 320


def test_default_step_matches_the_8192_contract():
    assert QSAKVCache.step == 8192
    cache = QSAKVCache()
    cache.update_and_fetch(_rnd((1, H, 16384, D), 1), _rnd((1, H, 16384, D), 2))
    assert _capacity(cache) == 16384
    cache.update_and_fetch(_rnd((1, H, 1, D), 3), _rnd((1, H, 1, D), 4))
    # 16385 tokens: ceil8192 = 24576 (doubling gave 32768).
    assert _capacity(cache) == 24576


def test_restored_state_grows_by_one_step():
    source = QSAKVCache()
    source.update_and_fetch(_rnd((1, H, 300, D), 1), _rnd((1, H, 300, D), 2))
    source.update_indexer(_rnd((1, 300, DI), 3), mx.arange(300, dtype=mx.int32)[None])
    restored = QSAKVCache()
    restored.state = source.state
    restored.step = STEP
    restored.update_and_fetch(_rnd((1, H, 1, D), 4), _rnd((1, H, 1, D), 5))
    assert _capacity(restored) == _ceil(301)
    np.testing.assert_array_equal(
        _bits(restored.keys[..., :300, :]), _bits(source.state[0])
    )


def test_batch_extracted_row_grows_by_one_step():
    rows = []
    for i, n in enumerate((5000, 4997)):
        row = QSAKVCache()
        row.update_and_fetch(_rnd((1, H, n, D), 10 * i), _rnd((1, H, n, D), 10 * i + 1))
        row.update_indexer(
            _rnd((1, n, DI), 10 * i + 2), mx.arange(n, dtype=mx.int32)[None]
        )
        rows.append(row)
    batch = BatchQSAKVCache.merge(rows)
    row = batch.extract(0)
    before = mx.array(row.keys)
    row.update_and_fetch(_rnd((1, H, 1, D), 21), _rnd((1, H, 1, D), 22))
    assert _capacity(row) == 8192
    np.testing.assert_array_equal(_bits(row.keys[..., :5000, :]), _bits(before))


def test_quantized_cache_grows_by_one_step():
    dq = 32  # smallest supported quantization group
    cache = QSAQuantizedKVCache(group_size=dq, bits=8)
    cache.step = STEP
    cache.update_and_fetch(_rnd((1, H, 256, dq), 1), _rnd((1, H, 256, dq), 2))
    assert _capacity(cache) == 256
    cache.update_and_fetch(_rnd((1, H, 14, dq), 3), _rnd((1, H, 14, dq), 4))
    assert _capacity(cache) == 320
