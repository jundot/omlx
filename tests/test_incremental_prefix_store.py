# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache blocks stored while a prefill runs (Scheduler._maybe_incremental_store)."""

import concurrent.futures
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

import omlx.scheduler as sch
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.utils.metal_sync import _mx_buffer_access_lock, mx_buffer_lock_released

BLOCK = 4


def _live(length, seed=3):
    mx.random.seed(seed)
    cache = KVCache()
    cache.update_and_fetch(
        mx.random.normal((1, 2, length, 8)).astype(mx.bfloat16),
        mx.random.normal((1, 2, length, 8)).astype(mx.bfloat16),
    )
    mx.eval(cache.state)
    return cache


def _grow(cache, n, seed=5):
    mx.random.seed(seed)
    cache.update_and_fetch(
        mx.random.normal((1, 2, n, 8)).astype(mx.bfloat16),
        mx.random.normal((1, 2, n, 8)).astype(mx.bfloat16),
    )
    mx.eval(cache.state)


def _data(cache):
    # What Scheduler._extract_cache_states gives for a KVCache: the filled part.
    n = cache.offset
    return {
        "state": (cache.keys[:, :, :n, :], cache.values[:, :, :n, :]),
        "class_name": "KVCache",
        "cache_type": "KVCache",
        "meta_state": getattr(cache, "meta_state", ""),
    }


@pytest.fixture
def prefix_factory(tmp_path):
    managers = []

    def make():
        ssd = PagedSSDCacheManager(
            cache_dir=tmp_path / str(len(managers)),
            max_size_bytes=10 * 1024**2,
            hot_cache_max_bytes=0,
            expected_model_name="inc-test",
            expected_num_layers=1,
            expected_block_size=BLOCK,
        )
        ssd.set_expected_layer_signature(["KVCache"])
        managers.append(ssd)
        paged = PagedCacheManager(
            block_size=BLOCK, max_blocks=64, initial_blocks=64, model_name="inc-test"
        )
        paged.set_paged_ssd_cache_manager(ssd)
        prefix = BlockAwarePrefixCache(SimpleNamespace(layers=[object()]), paged, ssd)
        return prefix, paged

    yield make
    for manager in managers:
        manager.close()


class _Executor:
    """Runs submissions inline, or holds them until ``drain`` (run=False)."""

    def __init__(self, run=True):
        self.run = run
        self.queue = []

    def submit(self, fn, *args):
        future = concurrent.futures.Future()
        self.queue.append((future, fn, args))
        if self.run:
            self.drain()
        return future

    def drain(self):
        while self.queue:
            future, fn, args = self.queue.pop(0)
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn(*args))
            except Exception as exc:  # noqa: BLE001
                future.set_exception(exc)


def _scheduler(prefix, executor=None, **overrides):
    s = sch.Scheduler
    sched = SimpleNamespace(
        block_aware_cache=prefix,
        config=SimpleNamespace(paged_cache_block_size=BLOCK),
        _incremental_stored={},
        _incremental_futures={},
        _incremental_released=set(),
        _incremental_lock=threading.Lock(),
        _INCREMENTAL_RELEASE_WAIT_S=5.0,
        _boundary_cache_snapshots={},
        _boundary_snapshot_store=None,
        paged_ssd_cache_manager=None,
        _PREFILL_SNAPSHOT_MARKER=s._PREFILL_SNAPSHOT_MARKER,
        _stream=None,
        _store_cache_executor=executor or _Executor(),
        _admission_paused=False,
        _inflight_store_futures={},
        _turboquant_kv_bits=None,
        _model_has_unreconstructible_cache=lambda: False,
        _bypass_hot_cache_under_pressure=lambda: False,
        _turboquant_eligible=lambda cache: True,
        _extract_cache_states=lambda cache: ([_data(c) for c in cache], None),
        _extract_snapshot_cache_states=lambda snap: (snap, None),
        _collect_arrays_from_extracted_cache=lambda data: [
            part for layer in data for part in layer["state"]
        ],
    )
    for key, value in overrides.items():
        setattr(sched, key, value)
    for name in (
        "_incremental_store_allowed",
        "_maybe_incremental_store",
        "_incremental_store_worker",
        "_note_incremental_release",
        "_incremental_converts_to_turboquant",
        "_incremental_turboquant_payload",
    ):
        setattr(sched, name, types.MethodType(getattr(s, name), sched))
    return sched


def _request(n, rid="r", **extra):
    fields = dict(
        request_id=rid,
        prompt_token_ids=list(range(100, 100 + n)),
        vlm_extra_keys_for_cache=None,
        vlm_extra_key_ranges_for_cache=None,
        skip_cache_store=False,
        specprefill_indices=None,
    )
    fields.update(extra)
    return SimpleNamespace(**fields)


@pytest.fixture(autouse=True)
def _one_block(monkeypatch):
    monkeypatch.setattr(sch, "_INCREMENTAL_STORE_MIN_BLOCKS", 1)


def _kv(prefix, rid, tokens):
    table, _ = prefix.fetch_cache(rid, tokens + [1])
    assert table is not None
    restored = prefix.reconstruct_cache(table)
    prefix.release_cache(rid)
    return table.num_tokens, restored


def test_pure_attention_blocks_are_cut_from_the_live_kv(prefix_factory):
    """Plain KV, no snapshots: blocks cut during the prefill, then the finish
    store's tail, restore the same KV as one store of the whole prompt."""
    prefix, paged = prefix_factory()
    live = _live(8)
    request = _request(12)
    sched = _scheduler(prefix)
    sched._maybe_incremental_store(request, [live], 8, False)
    assert sched._incremental_stored == {"r": 8}
    stored = list(paged.get_block_table("r").block_ids)
    assert len(stored) == 2
    # A new question finds them while the prefill still runs.
    assert _kv(prefix, "reader", request.prompt_token_ids[:8])[0] == 8

    _grow(live, 4)
    final = [_data(live)]
    assert prefix.trim_store_payload("r", final) == 8
    table = prefix.store_cache("r", request.prompt_token_ids, final)
    assert table.block_ids[:2] == stored and table.num_tokens == 12

    other, _ = prefix_factory()
    whole = _live(8)
    _grow(whole, 4)
    assert other.store_cache("w", request.prompt_token_ids, [_data(whole)])
    n_a, kv_a = _kv(prefix, "a", request.prompt_token_ids)
    n_b, kv_b = _kv(other, "b", request.prompt_token_ids)
    assert n_a == n_b == 12
    a, b = kv_a[0], kv_b[0]
    assert type(a) is type(b) and a.offset == b.offset == 12
    for left, right in ((a.keys, b.keys), (a.values, b.values)):
        left, right = left[:, :, :12, :], right[:, :, :12, :]
        assert left.dtype == right.dtype and left.shape == right.shape
        assert mx.array_equal(left, right).item()


def test_nothing_new_or_off_the_grid_stores_nothing(prefix_factory):
    prefix, paged = prefix_factory()
    live = _live(8)
    sched = _scheduler(prefix)
    request = _request(12)
    sched._maybe_incremental_store(request, [live], 6, False)  # off the grid
    assert paged.get_block_table("r") is None
    sched._maybe_incremental_store(request, [live], 8, False)
    sched._maybe_incremental_store(request, [live], 8, False)  # nothing new
    assert len(paged.get_block_table("r").block_ids) == 2
    assert not sched._store_cache_executor.queue


@pytest.mark.parametrize(
    "case",
    [
        "skip_cache_store",
        "specprefill",
        "vlm",
        "admission_paused",
        "memory_pressure",
        "unreconstructible",
        "other_store_in_flight",
        "own_store_in_flight",
    ],
)
def test_finish_store_rules_skip_it(prefix_factory, case):
    prefix, paged = prefix_factory()
    request = _request(12)
    executor = _Executor()
    sched = _scheduler(prefix, executor)
    if case == "skip_cache_store":
        request.skip_cache_store = True
    elif case == "specprefill":
        request.specprefill_indices = [0, 1]
    elif case == "vlm":
        request.vlm_extra_keys_for_cache = [(0, "img")]
    elif case == "admission_paused":
        sched._admission_paused = True
    elif case == "memory_pressure":
        sched._bypass_hot_cache_under_pressure = lambda: True
    elif case == "unreconstructible":
        sched._model_has_unreconstructible_cache = lambda: True
    elif case == "other_store_in_flight":
        sched._inflight_store_futures = {"other": concurrent.futures.Future()}
    elif case == "own_store_in_flight":
        sched._incremental_futures = {"r": concurrent.futures.Future()}
    sched._maybe_incremental_store(request, [_live(8)], 8, False)
    assert paged.get_block_table("r") is None
    assert not executor.queue


def test_affine_prefill_layers_are_not_skipped_as_turboquant(prefix_factory):
    prefix, _ = prefix_factory()
    sched = _scheduler(prefix, _turboquant_kv_bits=8.0, _turboquant_kv_scheme="affine8")
    assert sched._incremental_store_allowed(_request(12), [_live(8)])


def test_snapshot_layers_come_from_the_new_block_ends():
    """With recurrent or windowed layers the store carries the boundary
    snapshots of the blocks it adds, and only those."""
    prefix = MagicMock()
    prefix.stored_prefix_tokens.return_value = 4
    prefix.copy_payload_prefix.side_effect = lambda data, end, start: data
    sched = _scheduler(prefix)
    marker = sch.Scheduler._PREFILL_SNAPSHOT_MARKER
    snaps = {tc: (marker, [{"state": (mx.zeros((1,)),)}]) for tc in (4, 8, 12, 16)}
    sched._boundary_cache_snapshots = {"r": snaps}
    sched._maybe_incremental_store(_request(20), [_live(12)], 12, True)
    provider = prefix.store_cache.call_args.kwargs["boundary_snapshots"]
    assert sorted(provider._valid_tcs) == [8, 12]


def test_release_before_the_store_runs_skips_it(prefix_factory):
    prefix, paged = prefix_factory()
    executor = _Executor(run=False)
    sched = _scheduler(prefix, executor)
    sched._maybe_incremental_store(_request(12), [_live(8)], 8, False)
    assert len(executor.queue) == 1
    sched._note_incremental_release("r")
    executor.drain()
    assert paged.get_block_table("r") is None
    assert not sched._incremental_stored and not sched._incremental_futures
    assert not sched._incremental_released


def test_release_waits_for_a_running_store_so_its_blocks_are_returned(
    prefix_factory, monkeypatch
):
    prefix, paged = prefix_factory()
    started, go = threading.Event(), threading.Event()
    store = prefix.store_cache

    def slow_store(*args, **kwargs):
        started.set()
        go.wait(5)
        return store(*args, **kwargs)

    monkeypatch.setattr(prefix, "store_cache", slow_store)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        sched = _scheduler(prefix, pool)
        sched._maybe_incremental_store(_request(12), [_live(8)], 8, False)
        assert started.wait(5)
        threading.Timer(0.2, go.set).start()
        t0 = time.monotonic()
        sched._note_incremental_release("r")
        assert time.monotonic() - t0 >= 0.15
        table = paged.get_block_table("r")
        assert table is not None and len(table.block_ids) == 2
        blocks = list(table.block_ids)
        prefix.release_cache("r")  # what the caller does next
        assert paged.get_block_table("r") is None
        assert all(
            paged.allocated_blocks[b].ref_count <= 1
            for b in blocks
            if b in paged.allocated_blocks
        )
        assert not sched._incremental_futures and not sched._incremental_released
    finally:
        pool.shutdown(wait=True)


def test_lock_is_let_go_while_a_store_waits_on_io():
    """A store waiting for room in the SSD write queue does not hold the
    lock every prefill chunk's clear_cache takes."""
    got = []
    with _mx_buffer_access_lock:  # noqa: SIM117 - held twice on purpose
        with _mx_buffer_access_lock:  # held twice, as store_cache nests it
            with mx_buffer_lock_released():
                t = threading.Thread(
                    target=lambda: got.append(_mx_buffer_access_lock.acquire(timeout=1))
                    or _mx_buffer_access_lock.release()
                )
                t.start()
                t.join(2)
            assert _mx_buffer_access_lock._is_owned()
    assert got == [True]
    assert not _mx_buffer_access_lock._is_owned()
    with mx_buffer_lock_released():  # not held: a no-op
        pass


def test_turboquant_blocks_match_the_converted_cache(tmp_path):
    """TurboQuant converts dense prefill layers at the end. It quantizes each
    token on its own, so blocks converted during the prefill are the same
    bytes as the same blocks of the converted cache."""
    from mlx_vlm.turboquant import TurboQuantKVCache

    def prefix_for(name):
        ssd = PagedSSDCacheManager(
            cache_dir=tmp_path / name,
            max_size_bytes=20 * 1024**2,
            hot_cache_max_bytes=0,
            expected_model_name="inc-test",
            expected_num_layers=1,
            expected_block_size=BLOCK,
        )
        ssd.set_expected_layer_signature(["TurboQuantKVCache"], turboquant_kv_bits=4)
        paged = PagedCacheManager(
            block_size=BLOCK, max_blocks=64, initial_blocks=64, model_name="inc-test"
        )
        paged.set_paged_ssd_cache_manager(ssd)
        return ssd, BlockAwarePrefixCache(
            SimpleNamespace(layers=[object()]), paged, ssd
        )

    mx.random.seed(11)
    keys = mx.random.normal((1, 2, 12, 128)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, 12, 128)).astype(mx.bfloat16)
    live = KVCache()
    live.update_and_fetch(keys[:, :, :8], values[:, :, :8])
    mx.eval(live.state)
    ssd_a, prefix_a = prefix_for("a")
    ssd_b, prefix_b = prefix_for("b")
    try:
        stub = SimpleNamespace(
            model_name="inc-test",
            _normalize_rotating_snapshot_state=lambda *a, **k: a[0] if a else None,
        )
        extract = types.MethodType(sch.Scheduler._extract_cache_states, stub)
        sched = _scheduler(
            prefix_a,
            _turboquant_kv_bits=4.0,
            _turboquant_kv_scheme="turboquant",
            _turboquant_skip_last=True,
            _extract_cache_states=extract,
            _collect_arrays_from_extracted_cache=lambda data: [],
        )
        request = _request(12)
        sched._maybe_incremental_store(request, [live], 8, False)
        assert prefix_a.paged_cache.get_block_table("r") is not None

        whole = KVCache()
        whole.update_and_fetch(keys, values)
        converted, _ = extract([TurboQuantKVCache.from_cache(whole, bits=4)])
        assert prefix_b.store_cache("w", request.prompt_token_ids, converted)

        for prefix, rid in ((prefix_a, "a"), (prefix_b, "b")):
            table, _ = prefix.fetch_cache(rid, request.prompt_token_ids[:8] + [1])
            assert table is not None and table.num_tokens == 8
        restored_a = prefix_a.reconstruct_cache(
            prefix_a.paged_cache.get_block_table("a")
        )
        restored_b = prefix_b.reconstruct_cache(
            prefix_b.paged_cache.get_block_table("b")
        )
        assert restored_b is not None, "whole-store reconstruct"
        assert restored_a is not None, "incremental reconstruct"
        assert type(restored_a[0]).__name__ == type(restored_b[0]).__name__
        for left, right in zip(restored_a[0].state, restored_b[0].state):
            left, right = getattr(left, "_state", left), getattr(right, "_state", right)
            for field in left._fields:
                x, y = getattr(left, field), getattr(right, field)
                n = min(x.shape[-2] if x.ndim == 4 else x.shape[-1], 8)
                x = x[..., :n, :] if x.ndim == 4 else x[..., :n]
                y = y[..., :n, :] if y.ndim == 4 else y[..., :n]
                assert x.shape == y.shape and mx.array_equal(x, y).item(), field
    finally:
        ssd_a.close()
        ssd_b.close()
