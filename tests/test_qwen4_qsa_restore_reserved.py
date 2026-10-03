# SPDX-License-Identifier: Apache-2.0
"""A paged QSA prefix restores straight into buffers sized for the prompt.

Restoring used to concatenate every stored block into an exact-width prefix,
and the first prefill chunk then copied that prefix into a buffer grown to
the reserved prompt horizon: blocks, concatenated prefix and grown buffer
coexisted (3x the prefix KV for a moment). ``reconstruct_reserved`` writes
each block straight into the final buffer. Only allocation changes: the
restored state, every later append's outputs, attention over them, the
pooled indexer bank and even the post-append backing buffers are
bit-identical to the concatenate-then-grow path.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from omlx.cache.type_handlers import Qwen4QSAKVCacheHandler
from omlx.patches.mlx_vlm_qwen4_exp_compat import (  # noqa: E402
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp.language import QSAKVCache  # noqa: E402

H, D, DI = 2, 32, 16
STEP = 64
MiB = 1024**2


@pytest.fixture
def small_steps(monkeypatch):
    monkeypatch.setattr(QSAKVCache, "step", STEP)
    monkeypatch.setattr(QSAKVCache, "index_step", STEP)


def _bits(a):
    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    return np.array(a)


def _same(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    np.testing.assert_array_equal(_bits(a), _bits(b))


def _rnd(shape, key):
    return mx.random.normal(shape, key=mx.random.key(key)).astype(mx.bfloat16)


def _block(start, length, channels, key, h=H, d=D, di=DI):
    """One stored block in the serialized ``[B, C, S]`` position layout."""
    pos = mx.arange(start, start + length, dtype=mx.int32)[None, None]
    pos = mx.concatenate([pos + c for c in range(channels)], axis=1)
    elements = (
        _rnd((1, h, length, d), key),
        _rnd((1, h, length, d), key + 1),
        _rnd((1, length, di), key + 2),
        pos,
    )
    mx.eval(*elements)
    return {"states": elements}


def _blocks(sizes, channels):
    blocks, start = [], 0
    for i, size in enumerate(sizes):
        c = channels[i % len(channels)]
        blocks.append(_block(start, size, c, 1000 * i + 1))
        start += size
    return blocks


def _old_restore(states, reserve):
    handler = Qwen4QSAKVCacheHandler()
    cache = handler.reconstruct_cache(handler.concatenate_states(states))
    cache.reserve_index_capacity(reserve)
    return cache


def _new_restore(states, reserve):
    return Qwen4QSAKVCacheHandler().reconstruct_reserved(states, reserve)


def _logical(cache):
    keys, values, index_keys, positions = cache.state
    return [keys, values, index_keys, positions], cache.offset, cache._index_offset


def _step(cache, start, length, key, mrope, h=H, d=D, di=DI, reads=True):
    """One prefill/decode append plus the reads attention and QSA make of it."""
    k, v = cache.update_and_fetch(
        _rnd((1, h, length, d), key), _rnd((1, h, length, d), key + 1)
    )
    pos = mx.arange(start, start + length, dtype=mx.int32)[None]
    if mrope:
        pos = mx.stack([pos, pos + 1, pos + 2])
    ik, ip = cache.update_indexer(_rnd((1, length, di), key + 2), pos)
    if not reads:
        mx.eval(k, v, ik, ip)
        return [k, v, ik, ip]
    q = _rnd((1, h, length, d), key + 3)
    attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5)
    pooled = cache.pooled_indexer_keys(
        4,
        lambda x: x * 2,
        lambda x, _positions: x + 1,
    )
    out = [k, v, ik, ip, attn, pooled]
    mx.eval(*out)
    return out


SIZES = [(96,), (64, 64, 64), (64, 17), (128, 128, 64, 30)]


@pytest.mark.parametrize("sizes", SIZES)
@pytest.mark.parametrize("channels", [(1,), (3,), (1, 3)])
@pytest.mark.parametrize("extra", [0, 1, 45, 200])
def test_reserved_restore_is_bit_identical_through_prefill_and_decode(
    small_steps, sizes, channels, extra
):
    states = _blocks(sizes, channels)
    length = sum(sizes)
    reserve = length + extra
    mrope = 3 in channels
    old, new = _old_restore(states, reserve), _new_restore(states, reserve)

    assert type(new) is QSAKVCache
    old_arrays, old_offset, old_index = _logical(old)
    new_arrays, new_offset, new_index = _logical(new)
    assert (new_offset, new_index) == (old_offset, old_index) == (length, length)
    for a, b in zip(old_arrays, new_arrays):
        _same(a, b)
    assert new._index_reserved_tokens == reserve

    # Resume prefill up to the prompt in uneven chunks, then decode past it.
    start, key = length, 7
    plan = [min(29, extra)] if extra else []
    while plan and start + sum(plan) < reserve:
        plan.append(min(29, reserve - start - sum(plan)))
    plan += [1, 1, 3]
    for n in plan:
        for a, b in zip(
            _step(old, start, n, key, mrope), _step(new, start, n, key, mrope)
        ):
            _same(a, b)
        if n == plan[0]:
            # After the first append both paths hold the very same buffers.
            _same(old.keys, new.keys)
            _same(old.values, new.values)
            _same(old._index_keys, new._index_keys)
            _same(old._index_position_ids, new._index_position_ids)
        start += n
        key += 11
    for a, b in zip(_logical(old)[0], _logical(new)[0]):
        _same(a, b)


def test_reserved_restore_prefills_to_the_prompt_without_regrowing(small_steps):
    states = _blocks((64, 64, 40), (1,))
    cache = _new_restore(states, 300)

    capacities = {(int(cache.keys.shape[2]), int(cache._index_keys.shape[1]))}
    start = 168
    while start < 300:
        n = min(50, 300 - start)
        _step(cache, start, n, start, False)
        capacities.add((int(cache.keys.shape[2]), int(cache._index_keys.shape[1])))
        start += n

    assert capacities == {(320, 320)}


def test_exact_hit_trim_then_append_matches(small_steps):
    states = _blocks((64, 64), (1,))
    old, new = _old_restore(states, 128), _new_restore(states, 128)
    # Exact prefix hit: the scheduler rewinds one token and re-feeds it.
    assert old.trim(1) == new.trim(1) == 1
    for a, b in zip(_step(old, 127, 1, 3, False), _step(new, 127, 1, 3, False)):
        _same(a, b)
    assert int(new.keys.shape[2]) == 128  # exact width: nothing to reserve


def test_mismatched_block_dtypes_fall_back_to_concatenation():
    states = _blocks((64, 64), (1,))
    keys, values, index_keys, positions = states[1]["states"]
    states[1] = {"states": (keys.astype(mx.float16), values, index_keys, positions)}

    assert Qwen4QSAKVCacheHandler().reconstruct_reserved(states, 200) is None


def _prefix_cache(tmp_path, layers):
    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    ssd = PagedSSDCacheManager(cache_dir=tmp_path / "ssd", max_size_bytes=1024**3)
    prefix = BlockAwarePrefixCache(
        model=SimpleNamespace(
            layers=[None] * layers, args=SimpleNamespace(num_hidden_layers=layers)
        ),
        paged_cache_manager=PagedCacheManager(
            block_size=64, max_blocks=64, model_name="qsa-test", initial_blocks=64
        ),
        paged_ssd_cache_manager=ssd,
    )
    return prefix, ssd


def test_prefix_cache_restore_reserves_the_prompt(tmp_path):
    from omlx.scheduler import Scheduler

    source = []
    for layer in range(2):
        cache = QSAKVCache()
        cache.update_and_fetch(
            _rnd((1, H, 200, D), 10 * layer), _rnd((1, H, 200, D), 10 * layer + 1)
        )
        cache.update_indexer(
            _rnd((1, 200, DI), 10 * layer + 2), mx.arange(200, dtype=mx.int32)[None]
        )
        source.append(cache)
    cache_data, config = Scheduler._extract_cache_states(
        SimpleNamespace(model_name="qsa-test"), source
    )
    mx.eval(*Scheduler._collect_arrays_from_extracted_cache(cache_data))
    prefix, ssd = _prefix_cache(tmp_path, 2)
    table = prefix.store_cache("req", list(range(200)), cache_data, config)
    assert table.num_tokens == 192

    plain = prefix.reconstruct_cache(table)
    reserved = prefix.reconstruct_cache(table, reserve_tokens=20000)
    ssd.close()

    for old, new in zip(plain, reserved):
        assert int(old.keys.shape[2]) == 192
        assert int(new.keys.shape[2]) == 24576  # ceil8192(20000)
        assert int(new._index_keys.shape[1]) == 24576
        for a, b in zip(old.state, new.state):
            _same(a, b)


def _restore_then_first_chunk(states, reserve, chunk, **dims):
    """The scheduler's order: restore, reserve, then the first prefill chunk."""
    handler = Qwen4QSAKVCacheHandler()
    cache = None
    reserved = getattr(handler, "reconstruct_reserved", None)
    if reserved is not None:
        cache = reserved(states, reserve)
    if cache is None:
        cache = handler.reconstruct_cache(handler.concatenate_states(states))
    cache.reserve_index_capacity(reserve)
    length = cache.offset
    return cache, _step(cache, length, chunk, 99, False, reads=False, **dims)


def test_restore_never_holds_a_concatenated_prefix():
    # Production widths and steps: 4 blocks of 4096 tokens (16 MiB of K/V).
    h, d, di = 2, 256, 128
    states = []
    for i in range(4):
        states.append(_block(4096 * i, 4096, 1, 50 * i, h=h, d=d, di=di))
    mx.synchronize()
    mx.clear_cache()
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()
    cache, outputs = _restore_then_first_chunk(
        states, 16384 + 2048, 512, h=h, d=d, di=di
    )
    mx.synchronize()
    peak = mx.get_peak_memory() - baseline
    buffers = sum(
        a.nbytes
        for a in (
            cache.keys,
            cache.values,
            cache._index_keys,
            cache._index_position_ids,
        )
    )
    extra = 4 * MiB  # the appended chunk, its indexer rows and RNG temporaries
    prefix_kv = sum(s["states"][0].nbytes * 2 for s in states)  # 16 MiB

    # A concatenated K/V prefix (or a regrowth copy) would add >= 8 MiB.
    assert peak < buffers + extra
    assert peak + prefix_kv // 2 > buffers  # the bound is not vacuous
