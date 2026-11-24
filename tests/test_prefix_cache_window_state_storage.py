# SPDX-License-Identifier: Apache-2.0
"""Regression guards for turn-end-only sliding-window storage (OMLX_SWA_TAIL_ONLY).

Inkling 2 keeps each sliding visit's window K/V in an ArraysCache. A boundary
snapshot of that state is the whole window (~170 MB for Inkling 2 Mini), so
storing one in every block fills a 5 GB hot cache near 48k tokens and the
prefix chain is evicted. With the flag on, window state is stored only in a
store's last block (the turn end) and its last full block; the newest few
such full blocks keep it. The lookup restores from the longest point that
still has window state.

Tests:
- test_window_state_kept_only_in_last_full_block — interior blocks get a
  window placeholder, the turn end keeps the window, full-attention KV stays.
- test_flag_off_keeps_stock_storage — without the flag every snapshot block
  keeps its window state.
- test_state_without_window_is_unaffected — conv-only and recurrent
  (GDN-style: 3D conv + 4D state) ArraysCache layers are stored as before
  even with the flag on.
- test_lookup_falls_back_to_block_with_window — a longer match ending on a
  stripped block restores from the newest block that has window state.
- test_lookup_reads_payload_when_marker_is_unknown — same, with the
  in-memory markers cleared (as after a restart).
- test_only_newest_full_blocks_keep_window — older full blocks are stripped
  beyond the cap; turn-end tails are not counted.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

BLOCK_SIZE = 4
WINDOW = 4


class MockModel:
    def __init__(self, num_layers: int = 2):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        a = MagicMock()
        a.num_hidden_layers = self._num_layers
        return a


@pytest.fixture
def tail_only(monkeypatch):
    monkeypatch.setenv("OMLX_SWA_TAIL_ONLY", "1")


def _make_cache(tmp_path):
    paged_cache = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=200,
        model_name="test-model",
        initial_blocks=200,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / "ssd_cache",
        max_size_bytes=100 * 1024**2,
        hot_cache_max_bytes=20 * 1024**2,
        hot_cache_only=True,
    )
    paged_cache.set_paged_ssd_cache_manager(ssd)  # as the scheduler does
    cache = BlockAwarePrefixCache(
        model=MockModel(num_layers=2),
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=ssd,
    )
    return cache, ssd


def _layers(seq_len, window=True):
    """A sliding layer (ArraysCache: window K/V + conv, Inkling 2 style) and a
    full-attention KVCache layer. ``window=False`` gives a conv-only
    ArraysCache, ``window="gdn"`` a recurrent one (3D conv + 4D state)."""
    sliding = [mx.ones((1, 3, 8)), mx.ones((1, 3, 8))]
    if window == "gdn":
        sliding = [mx.ones((1, 3, 8)), mx.ones((1, 2, 8, 8))]
    elif window:
        sliding = [mx.ones((1, 2, WINDOW, 8)), mx.ones((1, 2, WINDOW, 8))] + sliding
    return [
        {
            "state": sliding,
            "cache_type": "ArraysCache",
            "class_name": "ArraysCache",
            "meta_state": (),
        },
        {
            "state": (mx.ones((1, 2, seq_len, 8)), mx.ones((1, 2, seq_len, 8))),
            "cache_type": "KVCache",
            "class_name": "KVCache",
            "meta_state": (str(seq_len),),
        },
    ]


def _store(cache, rid, num_tokens, window=True, tail_terminal=False):
    tokens = list(range(num_tokens))
    snapshots = {
        b: _layers(b, window) for b in range(BLOCK_SIZE, num_tokens, BLOCK_SIZE)
    }
    table = cache.store_cache(
        rid,
        tokens,
        _layers(num_tokens, window),
        boundary_snapshots=snapshots,
        _store_tail_terminal=tail_terminal,
    )
    assert table is not None
    return table


def _hash(cache, table, idx):
    return cache.paged_cache.allocated_blocks[table.block_ids[idx]].block_hash


def _sliding_has_window(ssd, block_hash):
    data, meta = ssd.load_block_with_metadata(block_hash)
    assert data is not None and meta is not None
    layer = data[0]
    if isinstance(layer, tuple) and len(layer) == 3 and layer[0] == "__nstate__":
        return any(getattr(e, "ndim", 0) == 4 for e in layer[2])
    return False


def _kv_len(ssd, block_hash):
    data, _ = ssd.load_block_with_metadata(block_hash)
    return int(data[1][0].shape[2])


def test_window_state_kept_only_in_last_full_block(tmp_path, tail_only):
    cache, ssd = _make_cache(tmp_path)
    table = _store(cache, "a", 4 * BLOCK_SIZE)
    has_window = [_sliding_has_window(ssd, _hash(cache, table, i)) for i in range(4)]
    assert has_window == [False, False, False, True]
    assert all(_kv_len(ssd, _hash(cache, table, i)) == BLOCK_SIZE for i in range(4))


def test_flag_off_keeps_stock_storage(tmp_path, monkeypatch):
    monkeypatch.delenv("OMLX_SWA_TAIL_ONLY", raising=False)
    cache, ssd = _make_cache(tmp_path)
    table = _store(cache, "a", 4 * BLOCK_SIZE)
    assert all(_sliding_has_window(ssd, _hash(cache, table, i)) for i in range(4))


@pytest.mark.parametrize("kind", [False, "gdn"], ids=["conv-only", "recurrent"])
def test_state_without_window_is_unaffected(tmp_path, tail_only, kind):
    cache, ssd = _make_cache(tmp_path)
    table = _store(cache, "a", 4 * BLOCK_SIZE, window=kind)
    for i in range(4):
        data, _ = ssd.load_block_with_metadata(_hash(cache, table, i))
        assert tuple(data[0][0].shape) == (1, 3, 8)  # real conv state, no placeholder
    assert not getattr(cache.paged_cache, "_swa_active", False)


def _two_turns(cache):
    # Turn 1 ends at block 1 (window kept); turn 2 extends to block 3,
    # so block 2 is stored with a window placeholder.
    _store(cache, "turn-1", 2 * BLOCK_SIZE)
    return _store(cache, "turn-2", 4 * BLOCK_SIZE)


def test_lookup_falls_back_to_block_with_window(tmp_path, tail_only):
    cache, _ = _make_cache(tmp_path)
    _two_turns(cache)
    prompt = list(range(3 * BLOCK_SIZE + 1))
    blocks, n = cache.paged_cache.get_computed_blocks(prompt)
    assert n == 2 * BLOCK_SIZE
    assert len(blocks) == 2


def test_lookup_reads_payload_when_marker_is_unknown(tmp_path, tail_only):
    cache, _ = _make_cache(tmp_path)
    _two_turns(cache)
    cache.paged_cache._swa_full_hashes.clear()
    cache.paged_cache._swa_no_window.clear()
    prompt = list(range(3 * BLOCK_SIZE + 1))
    _, n = cache.paged_cache.get_computed_blocks(prompt)
    assert n == 2 * BLOCK_SIZE


def test_only_newest_full_blocks_keep_window(tmp_path, tail_only):
    cache, ssd = _make_cache(tmp_path)
    keep = BlockAwarePrefixCache._SWA_KEEP_FULL_BLOCKS
    turns = keep + 3
    # Each turn ends 2 tokens into a new block: the tail is the turn end,
    # the block before it is the store's last full block.
    for t in range(1, turns + 1):
        table = _store(cache, f"t{t}", t * BLOCK_SIZE + 2, tail_terminal=True)
    full = [_sliding_has_window(ssd, _hash(cache, table, i)) for i in range(turns)]
    assert full[-keep:] == [True] * keep
    assert not any(full[: turns - keep])
    assert _sliding_has_window(ssd, _hash(cache, table, turns))  # turn-end tail
