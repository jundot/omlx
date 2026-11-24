# SPDX-License-Identifier: Apache-2.0
"""Tests for gdn_snapshot_interval in the SSD sidecar layout.

With the SSD tier on, GDN state is staged to SSD at every block boundary
during prefill and committed as one sidecar file per block. With interval
N > 1, only boundaries on the N-block grid, the prompt's final boundary and
each store's last block get a sidecar; other blocks keep KV only and restores
walk back to the newest block that has a sidecar. Default 1 = unchanged.
"""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.scheduler import _BoundarySnapshotProvider

BS = 4
LAYER_TYPES = ["KVCache", "ArraysCache"]


class _HybridModel:
    def __init__(self):
        self.layers = [MagicMock(), MagicMock()]

    def make_cache(self):
        from mlx_lm.models.cache import ArraysCache, KVCache

        return [KVCache(), ArraysCache(size=2)]


def _extracted(token_count: int):
    value = float(token_count)
    return [
        {
            "state": (mx.full((1, 2, token_count, 8), value), mx.full((1, 2, token_count, 8), value + 1)),
            "class_name": "KVCache", "cache_type": "KVCache", "meta_state": (token_count,),
        },
        {
            "state": (mx.full((1, 3, 8), value), mx.full((1, 2, 4, 8), value)),
            "class_name": "ArraysCache", "cache_type": "ArraysCache", "meta_state": (),
        },
    ]


class _Env:
    def __init__(self, tmp_path, interval):
        cache_dir = tmp_path / f"cache{interval}"
        self.paged = PagedCacheManager(block_size=BS, max_blocks=200, model_name="hybrid-model", initial_blocks=200)
        self.ssd = PagedSSDCacheManager(
            cache_dir=cache_dir, max_size_bytes=100 * 1024**2, expected_model_name="hybrid-model",
            expected_num_layers=2, expected_block_size=BS, expected_layer_cache_types=LAYER_TYPES,
            gdn_ssd_split_enabled=True,
        )
        self.boundary = BoundarySnapshotSSDStore(cache_dir, pending_max_bytes=1024**2)
        self.prefix = BlockAwarePrefixCache(
            model=_HybridModel(), paged_cache_manager=self.paged, paged_ssd_cache_manager=self.ssd,
            gdn_ssd_split_enabled=True, gdn_snapshot_interval=interval,
        )
        self.prefix.set_gdn_checkpoint_loader(self.boundary.load_file)
        self.interval = interval
        self.signature = self.ssd.gdn_cache_signature_for(
            model_name="hybrid-model", num_layers=2, block_size=BS, layer_cache_types=LAYER_TYPES
        )

    def store(self, request_id, total_tokens, start=0):
        """Stage what the scheduler would stage (grid + final), then store."""
        kept = [
            t for t in range((start // BS + 1) * BS, total_tokens + 1, BS)
            if t % (self.interval * BS) == 0 or t == total_tokens
        ]
        for t in kept:
            ex = _extracted(t)
            assert self.boundary.save(request_id, t, [MagicMock()], lambda _s, ex=ex: (ex, None))
        provider = _BoundarySnapshotProvider(
            self.boundary, request_id, [t for t in kept if t != total_tokens], {}, paged_ssd_manager=self.ssd
        )
        table = self.prefix.store_cache(request_id, list(range(total_tokens)), _extracted(total_tokens),
                                        boundary_snapshots=provider)
        assert table is not None and table.num_tokens == total_tokens
        return table

    def hashes(self, table):
        return [self.paged.allocated_blocks[b].block_hash for b in table.block_ids]

    def restore(self, request_id, query):
        table, _ = self.prefix.fetch_cache(request_id, query)
        if table is None:
            return 0, None
        caches = self.prefix.reconstruct_cache(table)
        out = (table.num_tokens if caches is not None else 0), caches
        self.prefix.release_cache(request_id)
        return out

    def close(self):
        self.ssd.close()


@pytest.mark.parametrize("interval", [1, 2, 4])
def test_sidecars_only_on_grid_and_last_block(tmp_path, interval):
    env = _Env(tmp_path, interval)
    try:
        table = env.store("s", 8 * BS)
        hashes = env.hashes(table)
        assert len(hashes) == 8  # every block's KV is stored
        has = [env.ssd.has_gdn_checkpoint(h, env.signature) for h in hashes]
        ends = [(i + 1) * BS for i in range(8)]
        assert has == [e % (interval * BS) == 0 or e == ends[-1] for e in ends]
    finally:
        env.close()


@pytest.mark.parametrize("diverge_block", [3, 5, 6])
def test_restore_walks_back_to_grid_block(tmp_path, diverge_block):
    env = _Env(tmp_path, 2)
    try:
        env.store("s", 8 * BS)
        # Full continuation restores through the last block's sidecar.
        n, caches = env.restore("r1", list(range(8 * BS)) + [999])
        assert n == 8 * BS and float(caches[1].cache[0][0, 0, 0]) == pytest.approx(32.0)
        # Diverging inside block `diverge_block` walks back to the newest grid block before it.
        query = list(range(diverge_block * BS + 1)) + [999] * (BS + 1)
        n, caches = env.restore("r2", query)
        expected = (diverge_block // 2) * 2 * BS
        assert n == expected
        assert float(caches[1].cache[0][0, 0, 0]) == pytest.approx(float(expected))
        assert caches[0].keys_and_values()[0].shape[2] == expected
    finally:
        env.close()


def test_extending_a_thinned_chain_is_not_stopped(tmp_path):
    """Dedup'd off-grid blocks without a sidecar must not stop the next store."""
    env = _Env(tmp_path, 2)
    try:
        first = env.store("t1", 5 * BS)   # sidecars at 8, 16, 20 (last)
        second = env.store("t2", 9 * BS, start=5 * BS)
        assert len(second.block_ids) == 9
        hashes = env.hashes(second)
        has = [env.ssd.has_gdn_checkpoint(h, env.signature) for h in hashes]
        # ends 4..36: grid 8,16,24,32; 20 was the previous tip; 36 is the new tip
        assert has == [False, True, False, True, True, True, False, True, True]
        n, _ = env.restore("r", list(range(9 * BS)) + [999])
        assert n == 9 * BS
        del first
    finally:
        env.close()
