# SPDX-License-Identifier: Apache-2.0
"""Packed V4.1 block persistence and physical SSD roundtrip regressions."""

import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41 import tiny

from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from omlx.cache.deepseek_v41_delta import compact_state, restore_chain
from omlx.cache.paged_cache import BlockTable, PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.patches.deepseek_v41 import apply_patch
from omlx.patches.deepseek_v41.language import LanguageModel
from omlx.scheduler import Scheduler

META = ("deepseek_v41", "3", "2")


def _state(end):
    return (
        mx.array([end]),
        mx.zeros((1, 4, 8), mx.uint8),
        mx.zeros((1, end // 2, 8), mx.uint8),
        mx.zeros((1, end // 2, 4), mx.uint8),
        mx.zeros((1, 0, 8), mx.bfloat16),
        mx.zeros((1, 0, 8), mx.float32),
        mx.zeros((1, 0), mx.int64),
    )


def _marker(start, end):
    return (
        "__nstate__",
        "DeepseekV41Delta",
        compact_state(_state(end), META, start, end),
    )


def _wait_empty(pending):
    deadline = time.monotonic() + 10
    while pending() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not pending(), "SSD writer did not finish"


def test_delta_growth_is_linear_and_terminal_partial_preserves_state():
    total = 0
    for end in range(3, 301, 3):
        state = _state(end)
        delta = compact_state(state, META, end - 3, end)
        assert compact_state(delta, META, end - 3, end) is delta
        for slot in (0, 1, 4, 5, 6):
            assert delta[slot] is state[slot]
        total += delta[2].shape[1]
    assert total == 150
    markers = [_marker(0, 3), _marker(3, 6), _marker(6, 7)]
    restored = restore_chain(markers, [META] * 3, 7)
    assert restored.size() == 7
    assert restored.state[2].shape[1] == 3


@pytest.mark.parametrize("indices", [(1,), (0, 2), (1, 0), (0, 0)])
def test_missing_or_reordered_blocks_are_rejected(indices):
    markers = [_marker(0, 3), _marker(3, 6), _marker(6, 9)]
    with pytest.raises(ValueError):
        restore_chain([markers[i] for i in indices], [META] * len(indices), 9)


def test_legacy_anchor_and_invalid_metadata():
    old = ("__nstate__", "DeepseekV41Cache", _state(3))
    restored = restore_chain([old, _marker(3, 6)], [("deepseek_v41", "2"), META], 6)
    assert restored.state[2].shape[1] == 3
    with pytest.raises(ValueError):
        compact_state(_state(3), ("deepseek_v41", "4"), 0, 3)
    with pytest.raises(ValueError):
        compact_state(_marker(0, 3)[2], META, 3, 6)
    with pytest.raises(ValueError):
        restore_chain([_marker(0, 3)], [("deepseek_v41", "3", "4")], 3)


@pytest.mark.parametrize("block_size", [3, 4])
@pytest.mark.parametrize("engram", [False, True])
@pytest.mark.parametrize("extend_prefix", [False, True])
def test_real_ssd_reopen_full_and_partial_prefix(
    tmp_path, block_size, engram, extend_prefix
):
    apply_patch()
    mx.random.seed(18)
    cfg_args = (
        dict(
            engram_layer_ids=(1,),
            engram_num_embeddings=(72,),
            engram_vocab_size=5,
            engram_max_ngram_size=4,
            engram_n_heads=2,
            engram_head_dim=32,
            engram_compressed_vocab_size=64,
        )
        if engram
        else {}
    )
    model = LanguageModel(tiny(**cfg_args))
    if engram:
        model.set_token_map(np.arange(64))
    state = model.make_cache()
    tokens = list(range(3, 15))
    name = "deepseek-v41-test"
    paged = PagedCacheManager(
        block_size=block_size, max_blocks=100, initial_blocks=100, model_name=name
    )

    def open_ssd():
        return PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=0,
            hot_cache_only=False,
            expected_model_name=name,
        )

    def open_prefix(ssd):
        return BlockAwarePrefixCache(
            model=model,
            paged_cache_manager=paged,
            paged_ssd_cache_manager=ssd,
            gdn_ssd_split_enabled=True,
        )

    def extract(cache):
        return Scheduler._extract_cache_states(SimpleNamespace(model_name=name), cache)

    ssd = open_ssd()
    store = BoundarySnapshotSSDStore(tmp_path / "boundary")
    try:
        prefix = open_prefix(ssd)
        assert not prefix._gdn_split_layout_supported(["DeepseekV41Cache"] * 5)
        for end in range(block_size, 13, block_size):
            mx.eval(model(mx.array([tokens[end - block_size : end]]), cache=state))
            assert store.save("probe", end, state, extract, block_size=block_size)
        _wait_empty(lambda: store._pending_writes)
        snapshots = {
            end: store.load("probe", end) for end in range(block_size, 13, block_size)
        }
        full, cfg = extract(state)
        # Empty packed arrays must keep uint8 through the temporary SSD file.
        for layers in snapshots.values():
            for layer in layers:
                assert layer["state"][2].dtype == mx.uint8
                assert layer["state"][3].dtype == mx.uint8
        request_id = "probe"
        if extend_prefix:
            prefix.store_cache(
                request_id,
                tokens[: 2 * block_size],
                snapshots[2 * block_size],
                model_cache_config=cfg,
                boundary_snapshots=snapshots,
                hot_cache_write_back=False,
            )
            _wait_empty(lambda: ssd._pending_write_hashes)
            request_id = "extend"
            matched, remaining = prefix.fetch_cache(request_id, tokens)
            assert matched.num_tokens == 2 * block_size
            assert remaining == tokens[2 * block_size :]
            # Scheduler hands the final compact boundary to store_cache after
            # a prefix hit. Only newly traversed boundaries are captured.
            full = snapshots[12]
            snapshots = {
                end: value for end, value in snapshots.items() if end > 2 * block_size
            }
        table = prefix.store_cache(
            request_id,
            tokens,
            full,
            model_cache_config=cfg,
            boundary_snapshots=snapshots,
            hot_cache_write_back=False,
        )
        assert table is not None
        assert table.num_tokens == 12
        _wait_empty(lambda: ssd._pending_write_hashes)
        ssd.close()
        del prefix, snapshots, full, state
        ssd = open_ssd()
        prefix = open_prefix(ssd)
        for count in (block_size, 2 * block_size, 12):
            partial = BlockTable(
                "restore", table.block_ids[: count // block_size], count
            )
            restored = prefix.reconstruct_cache(partial)
            assert restored is not None
            for cache in restored:
                assert cache.state[2].dtype == mx.uint8
                assert cache.state[3].dtype == mx.uint8
            actual = model(mx.array([[20, 21]]), cache=restored)
            fresh = model(mx.array([tokens[:count] + [20, 21]]))[:, -2:]
            np.testing.assert_allclose(actual, fresh, atol=1e-5)
    finally:
        store.shutdown()
        ssd.close()


def test_in_memory_boundary_compacts_and_rejects_corrupt_record():
    from omlx.scheduler import _compact_boundary_snapshot_value

    layer = {"class_name": "DeepseekV41Cache", "state": _state(6), "meta_state": META}
    value = ("__prefill_extracted__", [layer])
    assert (
        _compact_boundary_snapshot_value(value, 6, 3, mx.default_stream(mx.gpu))
        is value
    )
    assert layer["state"][2].shape[1] == 2
    bad = list(layer["state"])
    bad[7] = mx.array([1, 3, 6, 0, 3, 2], mx.int64)
    with pytest.raises(ValueError, match="out-of-order"):
        restore_chain(
            [_marker(0, 3), ("__nstate__", "DeepseekV41Delta", bad)], [META] * 2, 6
        )


def test_cache_size_single_row_and_unaligned_batch():
    from omlx.patches.deepseek_v41.cache import DeepseekV41Cache

    cache = DeepseekV41Cache()
    assert cache.size() == 0
    cache.cache[0] = mx.array([4096], mx.int32)
    assert cache.size() == 4096
    cache.cache[0] = mx.array([4096, 17, 8192], mx.int32)
    assert cache.size() == 8192
    assert cache.extract(1).size() == 17
    restored = DeepseekV41Cache.from_state(cache.state, cache.meta_state)
    assert restored.size() == 8192
