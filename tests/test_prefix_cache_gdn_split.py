# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for SSD-only GDN sidecars plus normal KV blocks."""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from omlx.cache.paged_cache import BlockTable, CacheBlock, PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.scheduler import Scheduler, SchedulerConfig, _BoundarySnapshotProvider

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


def test_unsupported_gdn_layout_logs_embedded_fallback_once(tmp_path, caplog):
    paged = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=8,
        model_name="hybrid-model",
        initial_blocks=8,
    )
    prefix = BlockAwarePrefixCache(
        model=_HybridModel(),
        paged_cache_manager=paged,
        gdn_ssd_split_enabled=True,
    )

    with caplog.at_level("INFO"):
        assert not prefix._gdn_split_layout_supported(["ArraysCache", "CacheList"])
        assert not prefix._gdn_split_layout_supported(["ArraysCache", "CacheList"])

    assert caplog.text.count("falling back to embedded GDN snapshots") == 1


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
        gdn_sidecar_state_dtype="rht_int8",
    )
    boundary = BoundarySnapshotSSDStore(
        cache_dir,
        pending_max_bytes=1024**2,
        gdn_sidecar_state_dtype="rht_int8",
    )
    prefix = BlockAwarePrefixCache(
        model=_HybridModel(),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=True,
    )
    prefix.set_gdn_checkpoint_loader(
        boundary.load_file,
        dequantization_counter=lambda: boundary.gdn_state_dequantizations,
    )

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

        signature = ssd.gdn_cache_signature_for(
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
        assert float(restored[1].state[0][0, 0, 0]) == pytest.approx(12.0)
        latest_diagnostic = prefix.get_stats_dict()["gdn_last_restore"]
        assert latest_diagnostic["chosen_endpoint_tokens"] == 12
        assert latest_diagnostic["walkback_blocks"] == 0
        assert latest_diagnostic["checkpoint_load_latency_ms"] >= 0
        assert latest_diagnostic["source_block_hash"] == hashes[-1].hex()[:16]
        assert latest_diagnostic["requested_state_dtype"] == "rht_int8"
        assert (
            latest_diagnostic["effective_state_codec"]
            == "rht_int8_rowwise_last_axis_v1"
        )
        assert latest_diagnostic["used_legacy_fp32_fallback"] is False
        assert latest_diagnostic["dequantized_state_count"] == 1
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
        assert float(restored[1].state[0][0, 0, 0]) == pytest.approx(8.0)
        assert prefix._gdn_checkpoint_loads == 2
        assert prefix._gdn_checkpoint_walkbacks == 1
        walkback_diagnostic = prefix.get_stats_dict()["gdn_last_restore"]
        assert walkback_diagnostic["chosen_endpoint_tokens"] == 8
        assert walkback_diagnostic["walkback_blocks"] == 1
        assert walkback_diagnostic["source_block_hash"] == hashes[-2].hex()[:16]
        assert walkback_diagnostic["requested_state_dtype"] == "rht_int8"
        assert (
            walkback_diagnostic["effective_state_codec"]
            == "rht_int8_rowwise_last_axis_v1"
        )
        assert walkback_diagnostic["used_legacy_fp32_fallback"] is False
        assert walkback_diagnostic["dequantized_state_count"] == 1
        prefix.reset_stats()
        assert prefix.get_stats_dict()["gdn_last_restore"] is None
        assert prefix.get_stats_dict()["gdn_checkpoint_loads"] == 0
        assert prefix.get_stats_dict()["gdn_checkpoint_walkbacks"] == 0
        prefix.release_cache("restore-walkback")

        # Reintroduce only the newest endpoint as a legacy FP32 sidecar.  The
        # RHT request must expose that the successful restore used fallback.
        legacy_boundary = BoundarySnapshotSSDStore(
            tmp_path / "legacy-boundary",
            pending_max_bytes=1024**2,
            gdn_sidecar_state_dtype="fp32",
        )
        try:
            assert legacy_boundary.save(
                "legacy-request",
                12,
                [MagicMock()],
                lambda _snapshot: (_hybrid_extracted(12, 12.0), None),
            )
            legacy_staged = legacy_boundary.take_staged_file(
                "legacy-request", 12
            )
            assert legacy_staged is not None
            legacy_signature = ssd.cache_signature_for(
                model_name="hybrid-model",
                num_layers=2,
                block_size=BLOCK_SIZE,
                layer_cache_types=LAYER_TYPES,
            )
            assert (
                ssd.commit_gdn_checkpoint_file(
                    hashes[-1],
                    legacy_staged,
                    token_count=12,
                    model_name="hybrid-model",
                    cache_signature=legacy_signature,
                    block_size=BLOCK_SIZE,
                )
                is not None
            )

            hit_table, remaining = prefix.fetch_cache(
                "restore-legacy-fallback", tokens
            )
            assert hit_table is not None and remaining == []
            restored = prefix.reconstruct_cache(hit_table)
            assert restored is not None
            assert hit_table.num_tokens == 12
            fallback_diagnostic = prefix.get_stats_dict()["gdn_last_restore"]
            assert fallback_diagnostic["requested_state_dtype"] == "rht_int8"
            assert fallback_diagnostic["effective_state_codec"] == "fp32"
            assert fallback_diagnostic["used_legacy_fp32_fallback"] is True
            assert fallback_diagnostic["dequantized_state_count"] == 0
            assert ssd.gdn_legacy_fp32_fallbacks == 1
            prefix.release_cache("restore-legacy-fallback")
        finally:
            legacy_boundary.shutdown()
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


def test_scheduler_exact_split_hit_reprefills_only_last_block():
    """Exact GDN hits walk back one block before reconstruction (N-1 safety)."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(
        paged_cache_block_size=BLOCK_SIZE,
        gdn_ssd_split_enabled=True,
    )
    scheduler._prefix_cache_prepared = set()
    scheduler._p34_try_adopt_retained_chain = MagicMock(return_value=False)
    scheduler._gdn_split_active = MagicMock(return_value=True)
    scheduler._bypass_hot_cache_under_pressure = MagicMock(return_value=False)
    scheduler._align_minimax_m3_partial_cache_to_prefill_step = MagicMock(
        return_value=False
    )
    scheduler._cache_list_needs_boundary_snapshot = MagicMock(return_value=True)
    scheduler._log_prefix_divergence = MagicMock()
    scheduler._try_specprefill_scoring = MagicMock()

    table = BlockTable(
        request_id="exact-hit",
        block_ids=[1, 2, 3],
        num_tokens=12,
    )
    last = CacheBlock(block_id=3, ref_count=2, token_count=4)
    scheduler.paged_cache_manager = MagicMock()
    scheduler.paged_cache_manager.allocated_blocks = {3: last}
    scheduler.block_aware_cache = MagicMock()
    scheduler.block_aware_cache.fetch_cache.return_value = (table, [])
    scheduler.block_aware_cache.reconstruct_cache.return_value = ["restored"]

    request = MagicMock()
    request.request_id = "exact-hit"
    request.prompt_token_ids = list(range(12))
    request.vlm_extra_keys_for_cache = None
    request.vlm_extra_key_token_start_for_cache = None
    request.vlm_extra_key_ranges_for_cache = None

    scheduler._prepare_prefix_cache_for_request(request)

    assert table.block_ids == [1, 2]
    assert table.num_tokens == 8
    assert request.cached_tokens == 8
    assert request.remaining_tokens == [8, 9, 10, 11]
    scheduler.paged_cache_manager.free_block.assert_called_once_with(3)
    scheduler.block_aware_cache.reconstruct_cache.assert_called_once_with(table)


def test_split_restore_retries_legacy_candidate_at_the_same_endpoint(tmp_path):
    """A corrupt current sidecar falls back in place instead of walking back.

    ``forget_gdn_checkpoint`` drops only the first matching namespace, so the
    retry at the same block finds the legacy FP32 candidate. Recovering the
    endpoint costs one extra lookup; walking back would cost a whole block of
    re-prefill. The substitution stays visible via ``used_legacy_fp32_fallback``.
    """
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
        gdn_sidecar_state_dtype="rht_int8",
    )
    boundary = BoundarySnapshotSSDStore(
        cache_dir,
        pending_max_bytes=1024**2,
        gdn_sidecar_state_dtype="rht_int8",
    )
    legacy_boundary = BoundarySnapshotSSDStore(
        tmp_path / "legacy-boundary",
        pending_max_bytes=1024**2,
        gdn_sidecar_state_dtype="fp32",
    )
    prefix = BlockAwarePrefixCache(
        model=_HybridModel(),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=True,
    )

    try:
        request_id = "retry-request"
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
        rht_signature = ssd.gdn_cache_signature_for(
            model_name="hybrid-model",
            num_layers=2,
            block_size=BLOCK_SIZE,
            layer_cache_types=LAYER_TYPES,
        )
        legacy_signature = ssd.cache_signature_for(
            model_name="hybrid-model",
            num_layers=2,
            block_size=BLOCK_SIZE,
            layer_cache_types=LAYER_TYPES,
        )

        # Same endpoint, both namespaces populated.
        assert legacy_boundary.save(
            "legacy-request",
            12,
            [MagicMock()],
            lambda _snapshot: (_hybrid_extracted(12, 12.0), None),
        )
        legacy_staged = legacy_boundary.take_staged_file("legacy-request", 12)
        assert legacy_staged is not None
        assert (
            ssd.commit_gdn_checkpoint_file(
                hashes[-1],
                legacy_staged,
                token_count=12,
                model_name="hybrid-model",
                cache_signature=legacy_signature,
                block_size=BLOCK_SIZE,
            )
            is not None
        )

        corrupt_path = ssd.get_gdn_checkpoint_file(hashes[-1], rht_signature)
        assert corrupt_path is not None

        loaded_paths = []

        def load_rejecting_current(path):
            loaded_paths.append(path)
            if path == corrupt_path:
                # Stands in for any fail-closed decode: load_file returns None.
                return None
            return boundary.load_file(path)

        prefix.set_gdn_checkpoint_loader(
            load_rejecting_current,
            dequantization_counter=lambda: boundary.gdn_state_dequantizations,
        )

        hit_table, remaining = prefix.fetch_cache("restore-retry", tokens)
        assert hit_table is not None and remaining == []
        restored = prefix.reconstruct_cache(hit_table)
        assert restored is not None

        # The endpoint is kept, not walked back.
        assert hit_table.num_tokens == 12
        assert restored[1].size() == 12
        assert float(restored[1].state[0][0, 0, 0]) == pytest.approx(12.0)

        diagnostic = prefix.get_stats_dict()["gdn_last_restore"]
        assert diagnostic["chosen_endpoint_tokens"] == 12
        assert diagnostic["walkback_blocks"] == 0
        assert diagnostic["source_block_hash"] == hashes[-1].hex()[:16]
        assert diagnostic["requested_state_dtype"] == "rht_int8"
        assert diagnostic["effective_state_codec"] == "fp32"
        assert diagnostic["used_legacy_fp32_fallback"] is True
        assert ssd.gdn_legacy_fp32_fallbacks == 1
        assert prefix._gdn_checkpoint_walkbacks == 0
        # Exactly two attempts at this block: the rejected one and the legacy.
        assert len(loaded_paths) == 2
        assert loaded_paths[0] == corrupt_path
        assert loaded_paths[1] != corrupt_path
        # The rejected sidecar is gone; the legacy one survives.
        assert ssd.has_gdn_checkpoint(hashes[-1], legacy_signature)
        prefix.release_cache("restore-retry")
    finally:
        legacy_boundary.shutdown()
        boundary.shutdown()
        ssd.close()


def test_split_restore_retry_budget_is_one_per_block(tmp_path):
    """When every candidate fails the loop still advances to older blocks."""
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
        gdn_sidecar_state_dtype="rht_int8",
    )
    boundary = BoundarySnapshotSSDStore(
        cache_dir,
        pending_max_bytes=1024**2,
        gdn_sidecar_state_dtype="rht_int8",
    )
    prefix = BlockAwarePrefixCache(
        model=_HybridModel(),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=True,
    )

    try:
        request_id = "retry-budget"
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
        signature = ssd.gdn_cache_signature_for(
            model_name="hybrid-model",
            num_layers=2,
            block_size=BLOCK_SIZE,
            layer_cache_types=LAYER_TYPES,
        )
        newest_path = ssd.get_gdn_checkpoint_file(hashes[-1], signature)

        attempts = []

        def load_rejecting_newest(path):
            attempts.append(path)
            if path == newest_path:
                return None
            return boundary.load_file(path)

        prefix.set_gdn_checkpoint_loader(
            load_rejecting_newest,
            dequantization_counter=lambda: boundary.gdn_state_dequantizations,
        )

        hit_table, remaining = prefix.fetch_cache("restore-budget", tokens)
        assert hit_table is not None and remaining == []
        restored = prefix.reconstruct_cache(hit_table)
        assert restored is not None

        # No legacy candidate exists, so the newest block is attempted once and
        # the loop falls back to the previous boundary.
        assert hit_table.num_tokens == 8
        assert float(restored[1].state[0][0, 0, 0]) == pytest.approx(8.0)
        assert attempts.count(newest_path) == 1
        assert prefix._gdn_checkpoint_walkbacks == 1
        diagnostic = prefix.get_stats_dict()["gdn_last_restore"]
        assert diagnostic["walkback_blocks"] == 1
        assert diagnostic["used_legacy_fp32_fallback"] is False
        prefix.release_cache("restore-budget")
    finally:
        boundary.shutdown()
        ssd.close()
