# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for SSD-only GDN sidecars plus normal KV blocks."""

from unittest.mock import MagicMock

import mlx.core as mx

from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.scheduler import _BoundarySnapshotProvider

BLOCK_SIZE = 4
LAYER_TYPES = ["KVCache", "ArraysCache"]


class _HybridModel:
    def __init__(self):
        self.layers = [MagicMock(), MagicMock()]


def _hybrid_extracted(token_count: int, recurrent_value: float):
    return [
        {
            "state": (
                mx.full((1, 2, token_count, 8), recurrent_value),
                mx.full((1, 2, token_count, 8), recurrent_value + 1),
            ),
            "class_name": "KVCache",
            "cache_type": "KVCache",
            "meta_state": (token_count,),
        },
        {
            "state": (
                mx.full((1, 3, 8), recurrent_value),
                mx.full((1, 2, 4, 8), recurrent_value),
            ),
            "class_name": "ArraysCache",
            "cache_type": "ArraysCache",
            "meta_state": (),
        },
    ]


def _block_hashes(prefix_cache, table):
    return [
        prefix_cache.paged_cache.allocated_blocks[block_id].block_hash
        for block_id in table.block_ids
    ]


def test_split_store_restores_one_sidecar_and_walks_back(tmp_path):
    cache_dir = tmp_path / "cache"
    paged = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=100,
        model_name="hybrid-model",
        initial_blocks=100,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=100 * 1024**2,
        expected_model_name="hybrid-model",
        expected_num_layers=2,
        expected_block_size=BLOCK_SIZE,
        expected_layer_cache_types=LAYER_TYPES,
        gdn_ssd_split_enabled=True,
    )
    boundary = BoundarySnapshotSSDStore(cache_dir, pending_max_bytes=1024**2)
    prefix = BlockAwarePrefixCache(
        model=_HybridModel(),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=True,
    )
    prefix.set_gdn_checkpoint_loader(boundary.load_file)

    try:
        request_id = "store-request"
        boundaries = [4, 8, 12]
        for token_count in boundaries:
            extracted = _hybrid_extracted(token_count, float(token_count))
            assert boundary.save(
                request_id,
                token_count,
                [MagicMock()],
                lambda _snapshot, extracted=extracted: (extracted, None),
            )

        provider = _BoundarySnapshotProvider(
            boundary,
            request_id,
            boundaries[:-1],
            {},
            paged_ssd_manager=ssd,
        )
        tokens = list(range(12))
        stored = prefix.store_cache(
            request_id,
            tokens,
            _hybrid_extracted(12, 12.0),
            boundary_snapshots=provider,
        )
        assert stored is not None and stored.num_tokens == 12
        hashes = _block_hashes(prefix, stored)
        assert all(block_hash is not None for block_hash in hashes)

        signature = ssd.cache_signature_for(
            model_name="hybrid-model",
            num_layers=2,
            block_size=BLOCK_SIZE,
            layer_cache_types=LAYER_TYPES,
        )
        assert all(
            ssd.has_gdn_checkpoint(block_hash, signature)
            for block_hash in hashes
        )
        # Main blocks contain only structural Arrays placeholders; recurrent
        # states never enter the hot cache or ordinary block payload.
        for block_hash in hashes:
            block_data, _ = ssd.load_block_with_metadata(block_hash)
            assert tuple(block_data[1][0].shape) == (1,)
            assert tuple(block_data[1][1].shape) == (1,)

        hit_table, remaining = prefix.fetch_cache("restore-latest", tokens)
        assert hit_table is not None and remaining == []
        restored = prefix.reconstruct_cache(hit_table)
        assert restored is not None
        assert hit_table.num_tokens == 12
        assert restored[0].state[0].shape[2] == 12
        assert restored[1].size() == 12
        assert float(restored[1].state[0][0, 0, 0]) == 12.0
        latest_diagnostic = prefix.get_stats_dict()["gdn_last_restore"]
        assert latest_diagnostic["chosen_endpoint_tokens"] == 12
        assert latest_diagnostic["walkback_blocks"] == 0
        assert latest_diagnostic["checkpoint_load_latency_ms"] >= 0
        assert latest_diagnostic["source_block_hash"] == hashes[-1].hex()[:16]
        prefix.release_cache("restore-latest")

        # If the newest recurrent checkpoint was evicted independently, the
        # contiguous KV chain remains useful up to the newest older sidecar.
        assert ssd.forget_gdn_checkpoint(hashes[-1], signature)
        hit_table, remaining = prefix.fetch_cache("restore-walkback", tokens)
        assert hit_table is not None and remaining == []
        restored = prefix.reconstruct_cache(hit_table)
        assert restored is not None
        assert hit_table.num_tokens == 8
        assert restored[0].state[0].shape[2] == 8
        assert restored[1].size() == 8
        assert float(restored[1].state[0][0, 0, 0]) == 8.0
        assert prefix._gdn_checkpoint_loads == 2
        assert prefix._gdn_checkpoint_walkbacks == 1
        walkback_diagnostic = prefix.get_stats_dict()["gdn_last_restore"]
        assert walkback_diagnostic["chosen_endpoint_tokens"] == 8
        assert walkback_diagnostic["walkback_blocks"] == 1
        assert walkback_diagnostic["source_block_hash"] == hashes[-2].hex()[:16]
        prefix.reset_stats()
        assert prefix.get_stats_dict()["gdn_last_restore"] is None
        assert prefix.get_stats_dict()["gdn_checkpoint_loads"] == 0
        assert prefix.get_stats_dict()["gdn_checkpoint_walkbacks"] == 0
        prefix.release_cache("restore-walkback")
    finally:
        boundary.shutdown()
        ssd.close()


def test_split_store_commits_single_final_sidecar_outside_provider_index(tmp_path):
    cache_dir = tmp_path / "cache"
    paged = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=100,
        model_name="hybrid-model",
        initial_blocks=100,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=100 * 1024**2,
        expected_model_name="hybrid-model",
        expected_num_layers=2,
        expected_block_size=BLOCK_SIZE,
        expected_layer_cache_types=LAYER_TYPES,
        gdn_ssd_split_enabled=True,
    )
    boundary = BoundarySnapshotSSDStore(cache_dir, pending_max_bytes=1024**2)
    prefix = BlockAwarePrefixCache(
        model=_HybridModel(),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=True,
    )
    prefix.set_gdn_checkpoint_loader(boundary.load_file)

    try:
        request_id = "single-final-boundary"
        extracted = _hybrid_extracted(BLOCK_SIZE, float(BLOCK_SIZE))
        assert boundary.save(
            request_id,
            BLOCK_SIZE,
            [MagicMock()],
            lambda _snapshot: (extracted, None),
        )
        # Scheduler excludes the latest snapshot from the provider's mapping;
        # it is still staged and must be committed for the final block.
        provider = _BoundarySnapshotProvider(
            boundary,
            request_id,
            [],
            {},
            paged_ssd_manager=ssd,
        )
        tokens = list(range(BLOCK_SIZE))
        stored = prefix.store_cache(
            request_id,
            tokens,
            extracted,
            boundary_snapshots=provider,
        )

        assert stored is not None and stored.num_tokens == BLOCK_SIZE
        block_hash = _block_hashes(prefix, stored)[0]
        signature = ssd.cache_signature_for(
            model_name="hybrid-model",
            num_layers=2,
            block_size=BLOCK_SIZE,
            layer_cache_types=LAYER_TYPES,
        )
        assert ssd.has_gdn_checkpoint(block_hash, signature)
    finally:
        boundary.shutdown()
        ssd.close()


def test_split_dedup_recreates_evicted_sidecar(tmp_path):
    cache_dir = tmp_path / "cache"
    paged = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=100,
        model_name="hybrid-model",
        initial_blocks=100,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=100 * 1024**2,
        expected_model_name="hybrid-model",
        expected_num_layers=2,
        expected_block_size=BLOCK_SIZE,
        expected_layer_cache_types=LAYER_TYPES,
        gdn_ssd_split_enabled=True,
    )
    boundary = BoundarySnapshotSSDStore(cache_dir, pending_max_bytes=1024**2)
    prefix = BlockAwarePrefixCache(
        model=_HybridModel(),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=True,
    )
    prefix.set_gdn_checkpoint_loader(boundary.load_file)

    try:
        tokens = list(range(12))
        boundaries = [4, 8, 12]
        for token_count in boundaries:
            extracted = _hybrid_extracted(token_count, float(token_count))
            assert boundary.save(
                "dedup-original",
                token_count,
                [MagicMock()],
                lambda _snapshot, extracted=extracted: (extracted, None),
            )
        original_provider = _BoundarySnapshotProvider(
            boundary,
            "dedup-original",
            boundaries[:-1],
            {},
            paged_ssd_manager=ssd,
        )
        original = prefix.store_cache(
            "dedup-original",
            tokens,
            _hybrid_extracted(12, 12.0),
            boundary_snapshots=original_provider,
        )
        assert original is not None and original.num_tokens == 12
        hashes = _block_hashes(prefix, original)
        signature = ssd.cache_signature_for(
            model_name="hybrid-model",
            num_layers=2,
            block_size=BLOCK_SIZE,
            layer_cache_types=LAYER_TYPES,
        )
        assert ssd.forget_gdn_checkpoint(hashes[-1], signature)

        replacement = _hybrid_extracted(12, 12.0)
        assert boundary.save(
            "dedup-repair",
            12,
            [MagicMock()],
            lambda _snapshot: (replacement, None),
        )
        repair_provider = _BoundarySnapshotProvider(
            boundary,
            "dedup-repair",
            [],
            {},
            paged_ssd_manager=ssd,
        )
        repaired = prefix.store_cache(
            "dedup-repair",
            tokens,
            replacement,
            boundary_snapshots=repair_provider,
        )

        assert repaired is not None and repaired.num_tokens == 12
        assert _block_hashes(prefix, repaired) == hashes
        assert ssd.has_gdn_checkpoint(hashes[-1], signature)
        hit_table, remaining = prefix.fetch_cache("dedup-restored", tokens)
        assert hit_table is not None and remaining == []
        restored = prefix.reconstruct_cache(hit_table)
        assert restored is not None
        assert hit_table.num_tokens == 12
        assert float(restored[1].state[0][0, 0, 0]) == 12.0
        prefix.release_cache("dedup-restored")
    finally:
        boundary.shutdown()
        ssd.close()


def test_split_restore_walks_back_from_structurally_invalid_sidecar(tmp_path):
    cache_dir = tmp_path / "cache"
    paged = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=100,
        model_name="hybrid-model",
        initial_blocks=100,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=100 * 1024**2,
        expected_model_name="hybrid-model",
        expected_num_layers=2,
        expected_block_size=BLOCK_SIZE,
        expected_layer_cache_types=LAYER_TYPES,
        gdn_ssd_split_enabled=True,
    )
    boundary = BoundarySnapshotSSDStore(cache_dir, pending_max_bytes=1024**2)
    prefix = BlockAwarePrefixCache(
        model=_HybridModel(),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=True,
    )

    try:
        request_id = "store-invalid-newest"
        boundaries = [4, 8, 12]
        for token_count in boundaries:
            extracted = _hybrid_extracted(token_count, float(token_count))
            assert boundary.save(
                request_id,
                token_count,
                [MagicMock()],
                lambda _snapshot, extracted=extracted: (extracted, None),
            )

        provider = _BoundarySnapshotProvider(
            boundary,
            request_id,
            boundaries,
            {},
            paged_ssd_manager=ssd,
        )
        tokens = list(range(12))
        stored = prefix.store_cache(
            request_id,
            tokens,
            _hybrid_extracted(12, 12.0),
            boundary_snapshots=provider,
        )
        assert stored is not None and stored.num_tokens == 12
        hashes = _block_hashes(prefix, stored)
        signature = ssd.cache_signature_for(
            model_name="hybrid-model",
            num_layers=2,
            block_size=BLOCK_SIZE,
            layer_cache_types=LAYER_TYPES,
        )
        newest_path = ssd.get_gdn_checkpoint_file(hashes[-1], signature)
        assert newest_path is not None

        def load_with_invalid_newest(path):
            snapshot = boundary.load_file(path)
            assert snapshot is not None
            if path == newest_path:
                # The container is readable, but the recurrent state is not.
                snapshot[1]["state"] = (snapshot[1]["state"][0],)
            return snapshot

        prefix.set_gdn_checkpoint_loader(load_with_invalid_newest)
        hit_table, remaining = prefix.fetch_cache("restore-invalid-newest", tokens)
        assert hit_table is not None and remaining == []
        restored = prefix.reconstruct_cache(hit_table)

        assert restored is not None
        assert hit_table.num_tokens == 8
        assert restored[0].state[0].shape[2] == 8
        assert restored[1].size() == 8
        assert float(restored[1].state[0][0, 0, 0]) == 8.0
        assert not ssd.has_gdn_checkpoint(hashes[-1], signature)
        diagnostic = prefix.get_stats_dict()["gdn_last_restore"]
        assert diagnostic["chosen_endpoint_tokens"] == 8
        assert diagnostic["walkback_blocks"] == 1
        prefix.release_cache("restore-invalid-newest")
    finally:
        boundary.shutdown()
        ssd.close()


def test_split_store_rejects_placeholder_when_checkpoint_commit_fails(tmp_path):
    cache_dir = tmp_path / "cache"
    paged = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=100,
        model_name="hybrid-model",
        initial_blocks=100,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=100 * 1024**2,
        expected_model_name="hybrid-model",
        expected_num_layers=2,
        expected_block_size=BLOCK_SIZE,
        expected_layer_cache_types=LAYER_TYPES,
        gdn_ssd_split_enabled=True,
    )
    boundary = BoundarySnapshotSSDStore(cache_dir, pending_max_bytes=1024**2)
    prefix = BlockAwarePrefixCache(
        model=_HybridModel(),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=True,
    )

    try:
        request_id = "failed-commit"
        extracted = _hybrid_extracted(BLOCK_SIZE, float(BLOCK_SIZE))
        assert boundary.save(
            request_id,
            BLOCK_SIZE,
            [MagicMock()],
            lambda _snapshot: (extracted, None),
        )
        provider = _BoundarySnapshotProvider(
            boundary,
            request_id,
            [BLOCK_SIZE],
            {},
            paged_ssd_manager=ssd,
        )
        provider.commit_gdn_checkpoint = MagicMock(return_value=False)
        allocated_before = set(paged.allocated_blocks)

        stored = prefix.store_cache(
            request_id,
            list(range(BLOCK_SIZE)),
            extracted,
            boundary_snapshots=provider,
        )

        assert stored is not None
        assert stored.block_ids == []
        assert set(paged.allocated_blocks) == allocated_before
        provider.commit_gdn_checkpoint.assert_called_once()
        assert ssd.get_stats().num_files == 0
    finally:
        boundary.shutdown()
        ssd.close()


def test_split_exact_prefix_fails_closed_before_placeholder_allocation(tmp_path):
    cache_dir = tmp_path / "cache"
    paged = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=100,
        model_name="hybrid-model",
        initial_blocks=100,
    )
    ssd = PagedSSDCacheManager(
        cache_dir=cache_dir,
        max_size_bytes=100 * 1024**2,
        expected_model_name="hybrid-model",
        expected_num_layers=2,
        expected_block_size=BLOCK_SIZE,
        expected_layer_cache_types=LAYER_TYPES,
        gdn_ssd_split_enabled=True,
    )
    prefix = BlockAwarePrefixCache(
        model=_HybridModel(),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=True,
    )

    try:
        allocated_before = set(paged.allocated_blocks)
        stored = prefix.store_exact_prefix(
            "split-exact-prefix",
            list(range(BLOCK_SIZE + 1)),
            _hybrid_extracted(BLOCK_SIZE + 1, float(BLOCK_SIZE + 1)),
        )

        assert stored is None
        assert set(paged.allocated_blocks) == allocated_before
        assert "split-exact-prefix" not in paged.request_tables
        assert ssd.get_stats().num_files == 0
        assert prefix.get_stats().exact_prefix_store_failures == 1
    finally:
        ssd.close()
