# SPDX-License-Identifier: Apache-2.0
"""Tests for gdn_snapshot_interval (recurrent-state snapshot thinning).

Hybrid models (KVCache + ArraysCache/GDN layers) store a full recurrent-state
snapshot with every cached block, although restoring an append-only
conversation only needs the newest one. With gdn_snapshot_interval = N, only
every N-th full block (absolute block index) and each store's last block keep
real state; other blocks keep their KV and a placeholder, and restores walk
back to the newest real-state block and re-prefill the rest.

Snapshots in these tests are filled with the token count they were taken at,
so every stored block shows which snapshot (if any) it received.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.settings import GlobalSettings, is_valid_gdn_snapshot_interval

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

BS = 4  # block size (tokens)


class _MockModel:
    def __init__(self, num_layers: int = 2):
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        a = MagicMock()
        a.num_hidden_layers = len(self.layers)
        return a


def _config():
    from omlx.cache.hybrid_cache import ModelCacheConfig

    return ModelCacheConfig.from_type_list(["KVCache", "ArraysCache"], model_name="test-model")


def _kv(seq_len):
    # Distinct value per token position, so KV slices are checkable.
    pos = mx.arange(seq_len, dtype=mx.float32).reshape(1, 1, seq_len, 1)
    k = mx.broadcast_to(pos, (1, 2, seq_len, 8))
    return (k, k + 1000.0)


def _hybrid_state(seq_len, fill):
    return [
        {"state": _kv(seq_len), "cache_type": "KVCache", "class_name": "KVCache"},
        {
            "state": (mx.full((1, 3, 8), fill), mx.full((1, 2, 4, 8), fill)),
            "cache_type": "ArraysCache",
            "class_name": "ArraysCache",
        },
    ]


def _snapshots(total_tokens, start=0):
    """Boundary snapshots at every block end in (start, total_tokens]."""
    out = {}
    end = (start // BS + 1) * BS
    while end <= total_tokens:
        out[end] = _hybrid_state(end, float(end))
        end += BS
    return out


def _mock_cache(interval):
    paged = PagedCacheManager(block_size=BS, max_blocks=200, model_name="test-model", initial_blocks=200)
    ssd = MagicMock()
    ssd.save_block.return_value = True
    ssd.has_block.return_value = False
    ssd.iter_tail_blocks.return_value = []
    cache = BlockAwarePrefixCache(
        model=_MockModel(), paged_cache_manager=paged, paged_ssd_cache_manager=ssd,
        gdn_snapshot_interval=interval,
    )
    return cache, ssd


def _stored_states(ssd):
    """Per save_block call: snapshot fill value, or None for a placeholder."""
    out = []
    for call in ssd.save_block.call_args_list:
        arrays = call.kwargs["cache_data"][1]
        first = arrays[0]
        out.append(None if tuple(first.shape) == (1,) else float(first.reshape(-1)[0]))
    return out


# --- config -------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, 2.5, "4", True, None])
def test_invalid_interval_rejected(bad):
    paged = PagedCacheManager(block_size=BS, max_blocks=10, model_name="m", initial_blocks=10)
    with pytest.raises(ValueError):
        BlockAwarePrefixCache(model=_MockModel(), paged_cache_manager=paged, gdn_snapshot_interval=bad)
    assert not is_valid_gdn_snapshot_interval(bad)


def test_settings_roundtrip_and_validation(tmp_path):
    s = GlobalSettings(base_path=tmp_path)
    assert s.cache.gdn_snapshot_interval == 1
    s.cache.gdn_snapshot_interval = 4
    s.save()
    loaded = GlobalSettings.load(base_path=tmp_path)
    assert loaded.cache.gdn_snapshot_interval == 4
    assert loaded.to_scheduler_config().gdn_snapshot_interval == 4
    assert not [e for e in loaded.validate() if "gdn_snapshot_interval" in e]
    for bad in (0, -3, 1.5, "2"):
        loaded.cache.gdn_snapshot_interval = bad
        assert any("gdn_snapshot_interval" in e for e in loaded.validate())


# --- which blocks keep real state ----------------------------------------------


@pytest.mark.parametrize("interval", [1, 2, 3, 4, 8])
def test_grid_and_tip_keep_real_state(interval):
    """8 full blocks: real state on the N-grid and the last block only."""
    cache, ssd = _mock_cache(interval)
    tokens = list(range(8 * BS))
    assert cache.store_cache(
        "r", tokens, _hybrid_state(len(tokens), float(len(tokens))),
        model_cache_config=_config(), boundary_snapshots=_snapshots(len(tokens)),
    ) is not None
    ends = [(i + 1) * BS for i in range(8)]
    expected = [
        float(end) if (end % (interval * BS) == 0 or end == ends[-1]) else None
        for end in ends
    ]
    assert _stored_states(ssd) == expected


@pytest.mark.parametrize("interval", [1, 3, 4])
def test_tail_store_keeps_tail_last_full_block_and_grid(interval):
    """Tail-terminal store: the tail and the last full block keep state; others follow the grid."""
    cache, ssd = _mock_cache(interval)
    tokens = list(range(7 * BS + 2))
    snaps = _snapshots(7 * BS)
    snaps[len(tokens)] = _hybrid_state(len(tokens), float(len(tokens)))
    assert cache.store_cache(
        "r", tokens, _hybrid_state(len(tokens), float(len(tokens))),
        model_cache_config=_config(), boundary_snapshots=snaps, _store_tail_terminal=True,
    ) is not None
    states = _stored_states(ssd)
    assert states[-1] == float(len(tokens))  # tail
    ends = [(i + 1) * BS for i in range(7)]
    assert states[:-1] == [
        float(end) if (end % (interval * BS) == 0 or end == ends[-1]) else None
        for end in ends
    ]


def test_extension_is_aligned_to_absolute_block_index():
    """A second store extending the chain uses absolute positions for the grid."""
    cache, ssd = _mock_cache(3)
    first = list(range(5 * BS))
    cache.store_cache("t1", first, _hybrid_state(len(first), float(len(first))),
                      model_cache_config=_config(), boundary_snapshots=_snapshots(len(first)))
    n_first = ssd.save_block.call_count
    second = list(range(10 * BS))
    cache.store_cache("t2", second, _hybrid_state(len(second), float(len(second))),
                      model_cache_config=_config(), boundary_snapshots=_snapshots(len(second), start=len(first)))
    new_states = _stored_states(ssd)[n_first:]
    # New blocks end at 24..40: grid (3 blocks = 12 tokens) hits 24 and 36; 40 is the tip.
    assert new_states == [24.0, None, None, 36.0, 40.0]


def test_backfill_skips_off_grid_blocks():
    """Dedup backfill must not re-add snapshots that thinning drops."""
    cache, ssd = _mock_cache(4)
    # First store: no intermediate snapshots at all (placeholders everywhere but the tip).
    first = list(range(8 * BS))
    cache.store_cache("t1", first, _hybrid_state(len(first), float(len(first))),
                      model_cache_config=_config(), boundary_snapshots={len(first): _hybrid_state(len(first), float(len(first)))})
    calls = []
    with patch.object(cache, "_maybe_backfill_dedup_block_payload",
                      side_effect=lambda h, end, data, **kw: calls.append(end) or False):
        longer = list(range(9 * BS))
        cache.store_cache("t2", longer, _hybrid_state(len(longer), float(len(longer))),
                          model_cache_config=_config(), boundary_snapshots=_snapshots(len(longer)))
    # Re-used blocks end at 4..32; only grid ends (16, 32) may be backfilled.
    assert calls and set(calls) <= {16, 32}


# --- restore: walk-back and data-level equivalence (U2.3 / U2.4) -----------------


def _real_cache(tmp_path, interval):
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager

    paged = PagedCacheManager(block_size=BS, max_blocks=200, model_name="test-model", initial_blocks=200)
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / f"ssd{interval}", max_size_bytes=100 * 1024**2,
        hot_cache_max_bytes=100 * 1024**2, hot_cache_only=True,
    )
    cache = BlockAwarePrefixCache(
        model=_MockModel(), paged_cache_manager=paged, paged_ssd_cache_manager=ssd,
        gdn_snapshot_interval=interval,
    )
    return cache, ssd


def _restore(cache, request_id, query):
    table, _remaining = cache.fetch_cache(request_id, query)
    if table is None:
        return 0, None
    caches = cache.reconstruct_cache(table)
    return (table.num_tokens if caches is not None else 0), caches


def _leaf_arrays(obj):
    """All arrays in a (possibly nested) cache state, in order."""
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return [obj]
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            out.extend(_leaf_arrays(item))
        return out
    return []


def _states(caches):
    return [_leaf_arrays(c.state) for c in caches]


@pytest.mark.parametrize("interval", [2, 3, 4, 8])
@pytest.mark.parametrize("diverge_block", [1, 3, 5, 6, 7])
def test_thinned_restore_matches_full_restore_at_walkback_point(tmp_path, interval, diverge_block):
    """Restoring a prompt that diverges inside block `diverge_block` (0-based):
    the thinned cache walks back to the newest grid block before it, and the
    restored tensors are bit-identical to an N=1 cache restoring the same prefix."""
    tokens = list(range(8 * BS))
    query = tokens[: diverge_block * BS + 1] + [99999] * (BS + 1)
    # Usable full blocks before the divergence: blocks 0..diverge_block-1.
    matched_blocks = diverge_block
    walkback_blocks = (matched_blocks // interval) * interval

    thin, thin_ssd = _real_cache(tmp_path, interval)
    thin.store_cache("s", tokens, _hybrid_state(len(tokens), float(len(tokens))),
                     model_cache_config=_config(), boundary_snapshots=_snapshots(len(tokens)))
    got_tokens, got = _restore(thin, "q", query)
    assert got_tokens == walkback_blocks * BS

    if walkback_blocks == 0:
        assert got is None or got_tokens == 0
    else:
        full, full_ssd = _real_cache(tmp_path / "full", 1)
        full.store_cache("s", tokens, _hybrid_state(len(tokens), float(len(tokens))),
                         model_cache_config=_config(), boundary_snapshots=_snapshots(len(tokens)))
        # Same usable prefix for the reference: diverge right after the walk-back point.
        ref_query = tokens[: walkback_blocks * BS] + [99999] * (BS + 1)
        ref_tokens, ref = _restore(full, "q", ref_query)
        assert ref_tokens == walkback_blocks * BS
        got_states, ref_states = _states(got), _states(ref)
        assert len(got_states) == len(ref_states)
        for layer_got, layer_ref in zip(got_states, ref_states):
            assert layer_got and len(layer_got) == len(layer_ref)
            for a, b in zip(layer_got, layer_ref):
                assert a.shape == b.shape and bool(mx.array_equal(a, b))
        full_ssd.close()
    thin_ssd.close()
