# SPDX-License-Identifier: Apache-2.0
"""Regression guards for V4-style cache storage through prefix_cache.

DeepSeek V4 has zero sliceable KVCache layers — only RotatingKVCache
(sliding window) and PoolingCache (compressed). Before the fix,
``_get_cache_seq_len`` fell through to step 2 and returned the
RotatingKVCache window length (e.g. 128). The continuity check then
rejected every block past index 128, so multi-block prefill produced
exactly one stored block (tokens [0:512]) and discarded the rest —
even though boundary snapshots covered them.

The fix moves the boundary-snapshot lookup ahead of the continuity
check and skips the check when a snapshot exists. Snapshots are
self-contained (they carry the full state for their boundary), so
the live-cache seq_len gate does not apply to them.

Tests:
- test_v4_all_blocks_stored_with_snapshots — V4 mock cache + snapshots
  for every block, expect all N blocks saved.
- test_continuity_check_still_blocks_when_no_snapshot — same V4 cache
  shape, no snapshots, expect the gate to fire (regression guard).
- test_kvcache_only_unaffected — Llama-style sliceable cache, expect
  step 1 to find full seq_len and continuity check to pass naturally.
- test_hybrid_kvcache_rotating_unaffected — Gemma3-style hybrid: step
  1 picks the KVCache full seq_len; bypass path is irrelevant.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache


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
def mx():
    try:
        import mlx.core as _mx

        return _mx
    except ImportError:
        pytest.skip("MLX not available")


def _make_cache_with_ssd():
    """A prefix cache wired to a mock SSD that always succeeds."""
    paged_cache = PagedCacheManager(
        block_size=4,
        max_blocks=100,
        model_name="test-model",
        initial_blocks=100,
    )
    mock_ssd = MagicMock()
    mock_ssd.save_block.return_value = True
    cache = BlockAwarePrefixCache(
        model=MockModel(num_layers=2),
        paged_cache_manager=paged_cache,
        paged_ssd_cache_manager=mock_ssd,
    )
    return cache, mock_ssd


def _v4_layer_state(mx, seq_len_window=4):
    """A RotatingKVCache-shaped layer state: 4D, seq dim limited to window."""
    return {
        "state": (
            mx.ones((1, 1, seq_len_window, 32)),
            mx.ones((1, 1, seq_len_window, 32)),
        ),
        "cache_type": "RotatingKVCache",
        "class_name": "RotatingKVCache",
        "meta_state": ("0", str(seq_len_window), "16", str(seq_len_window)),
    }


def _kvcache_layer_state(mx, seq_len):
    """A standard sliceable KVCache layer state."""
    return {
        "state": (
            mx.ones((1, 4, seq_len, 64)),
            mx.ones((1, 4, seq_len, 64)),
        ),
        "cache_type": "KVCache",
        "class_name": "KVCache",
        "meta_state": (str(seq_len),),
    }


def _snapshot_layer(mx, layer_state):
    """Wrap a layer state into a boundary-snapshot entry shape."""
    return {
        "state": layer_state["state"],
        "meta_state": layer_state["meta_state"],
        "class_name": layer_state["class_name"],
        "cache_type": layer_state["cache_type"],
    }


def test_v4_all_blocks_stored_with_snapshots(mx):
    """V4-shape cache + boundary snapshot per block → every block saved.

    Pre-fix this used to save only block 1 because
    ``_get_cache_seq_len`` returned 4 (RotatingKVCache window) and the
    continuity check broke at cache_start=4 for block 2 onwards.

    Uses RotatingKVCache-only (no sliceable KVCache) to reproduce the
    "all-non-sliceable model" pattern that triggers the bug. The fix
    is orthogonal to which non-sliceable type is involved — once
    snapshots exist for a block, the live cache_seq_len gate must not
    fire. PoolingCache state preservation is covered separately by
    test_cache_ntuple_state.py and test_v4_multi_session.py.
    """
    cache, mock_ssd = _make_cache_with_ssd()

    block_size = 4
    num_blocks = 4
    tokens = list(range(block_size * num_blocks))  # 16 tokens

    # Live cache shows window=4 — this is what _get_cache_seq_len
    # returns via the step-2 fallback, gating allocation past index 4.
    cache_data = [_v4_layer_state(mx, seq_len_window=4)]

    # Snapshots at every block boundary — these are self-contained
    # and must let the storage proceed regardless of live cache length.
    boundary_snapshots = {
        block_size * (i + 1): [_snapshot_layer(mx, _v4_layer_state(mx, 4))]
        for i in range(num_blocks)
    }

    result = cache.store_cache(
        "req-v4-all-blocks",
        tokens,
        cache_data,
        boundary_snapshots=boundary_snapshots,
    )

    assert result is not None
    assert (
        len(result.block_ids) == num_blocks
    ), f"expected {num_blocks} blocks, got {len(result.block_ids)}"
    assert mock_ssd.save_block.call_count == num_blocks


def test_continuity_check_still_blocks_when_no_snapshot(mx):
    """Same V4-shape cache without snapshots → continuity check still fires.

    Regression guard: the bypass is gated on snapshot existence. With
    no snapshots, the original behavior must remain (live-cache
    seq_len gates block allocation past the window).
    """
    cache, mock_ssd = _make_cache_with_ssd()

    block_size = 4
    tokens = list(range(block_size * 4))  # 16 tokens

    cache_data = [_v4_layer_state(mx, seq_len_window=4)]

    result = cache.store_cache(
        "req-v4-no-snapshot",
        tokens,
        cache_data,
        boundary_snapshots=None,
    )

    assert result is not None
    # Block 0 [0:4] passes (cache_start=0 < 4). Block 1 [4:8] fails
    # (cache_start=4 >= 4). Stop.
    assert (
        len(result.block_ids) == 1
    ), f"expected continuity break after 1 block, got {len(result.block_ids)}"


def test_kvcache_only_unaffected(mx):
    """Llama-style cache (full sliceable KVCache) — step 1 finds full
    seq_len; bypass path never engages; behavior unchanged.
    """
    cache, mock_ssd = _make_cache_with_ssd()

    block_size = 4
    num_blocks = 4
    tokens = list(range(block_size * num_blocks))  # 16 tokens

    # Step 1 of _get_cache_seq_len locates this 4D KVCache and returns
    # 16, so cache_start (4, 8, 12) all pass continuity check.
    cache_data = [_kvcache_layer_state(mx, seq_len=16)]

    result = cache.store_cache(
        "req-kv-only", tokens, cache_data, boundary_snapshots=None
    )

    assert result is not None
    assert len(result.block_ids) == num_blocks


def test_hybrid_kvcache_rotating_unaffected(mx):
    """Gemma3-style hybrid (KVCache + RotatingKVCache) — step 1 still
    picks up the full-attention KVCache layer's seq_len; rotating
    layer's window does not gate.
    """
    cache, mock_ssd = _make_cache_with_ssd()

    block_size = 4
    num_blocks = 4
    tokens = list(range(block_size * num_blocks))

    # KVCache full layer with seq_len=16 → step 1 returns 16 → no gate.
    cache_data = [
        _kvcache_layer_state(mx, seq_len=16),
        _v4_layer_state(mx, seq_len_window=4),
    ]

    result = cache.store_cache(
        "req-hybrid", tokens, cache_data, boundary_snapshots=None
    )

    assert result is not None
    assert len(result.block_ids) == num_blocks
