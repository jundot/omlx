# SPDX-License-Identifier: Apache-2.0
"""Split-GDN restore when the longest stored chain lost its recurrent checkpoints.

Real traffic (Qwen3.8-27B through Qwen Code, block 2048, SSD cache at its size
limit): after a compaction, a new context shares the 14,337-token system prefix
plus a few tokens with an older stored chain. The grid walk follows that chain;
its recurrent sidecars had been evicted (sidecars share the SSD LRU with KV
blocks but are touched only when chosen as the restore endpoint, while the KV
blocks are touched by every hit), so the walk-back found none and the whole
prompt was re-prefilled ("Split GDN restore found no compatible recurrent
checkpoint", 28-36K tokens, 40-50 s). The shared system-prefix tail, touched by
every new context, still had its checkpoint.
"""

from unittest.mock import MagicMock

import pytest

from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.scheduler import Scheduler, SchedulerConfig, _BoundarySnapshotProvider

from tests.test_prefix_cache_gdn_split import (
    BLOCK_SIZE,
    LAYER_TYPES,
    _HybridModel,
    _block_hashes,
    _hybrid_extracted,
)

SYSTEM_END = 13  # 3 full blocks + a 1-token tail, like 14,337 = 7 x 2048 + 1
SYSTEM = list(range(100, 100 + SYSTEM_END))
CONVERSATION = SYSTEM + list(range(500, 517))  # 30 tokens
# A new context: the system prompt plus tokens the conversation also has, so the
# grid walk follows the conversation's blocks to 20 tokens, then diverges.
NEW_CONTEXT = CONVERSATION[:21] + [900, 901, 902, 903]


@pytest.fixture
def caches(tmp_path):
    cache_dir = tmp_path / "cache"
    paged = PagedCacheManager(
        block_size=BLOCK_SIZE,
        max_blocks=200,
        model_name="hybrid-model",
        initial_blocks=200,
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
    # As in serving: blocks freed from the hot tier are lazily re-registered from SSD.
    paged.set_paged_ssd_cache_manager(ssd)
    boundary = BoundarySnapshotSSDStore(cache_dir, pending_max_bytes=1024**2)
    prefix = BlockAwarePrefixCache(
        model=_HybridModel(),
        paged_cache_manager=paged,
        paged_ssd_cache_manager=ssd,
        gdn_ssd_split_enabled=True,
    )
    prefix.set_gdn_checkpoint_loader(boundary.load_file)
    try:
        yield paged, ssd, boundary, prefix
    finally:
        boundary.shutdown()
        ssd.close()


def _save_snapshots(boundary, request_id, token_counts):
    for token_count in token_counts:
        extracted = _hybrid_extracted(token_count, float(token_count))
        assert boundary.save(
            request_id,
            token_count,
            [MagicMock()],
            lambda _snapshot, extracted=extracted: (extracted, None),
        )


def _store_conversation_and_system_tail(paged, ssd, boundary, prefix):
    """Store a conversation and its shared system tail as the scheduler does,
    then evict every recurrent checkpoint except the system tail's."""
    snapshots = list(range(BLOCK_SIZE, len(CONVERSATION), BLOCK_SIZE))
    _save_snapshots(boundary, "conversation", snapshots + [len(CONVERSATION)])
    stored = prefix.store_cache(
        "conversation",
        CONVERSATION,
        _hybrid_extracted(len(CONVERSATION), float(len(CONVERSATION))),
        boundary_snapshots=_BoundarySnapshotProvider(
            boundary,
            "conversation",
            snapshots,
            {},
            paged_ssd_manager=ssd,
            tail_terminal_token_count=len(CONVERSATION),
        ),
        _store_tail_terminal=True,
    )
    assert stored is not None and stored.num_tokens == len(CONVERSATION)
    conversation_hashes = _block_hashes(prefix, stored)
    paged.release_for_eviction(stored.block_ids)
    paged.request_tables.pop("conversation", None)
    prefix.clear_request_entry("conversation")

    # Scheduler._store_system_prefix: start from the cached grid, add the tail.
    store_id = "conversation#system"
    grid = SYSTEM_END // BLOCK_SIZE * BLOCK_SIZE
    table, _ = prefix.fetch_cache(store_id, SYSTEM[:grid], record_stats=False)
    assert table is not None and table.num_tokens == grid
    _save_snapshots(boundary, store_id, [SYSTEM_END])
    system = prefix.store_cache(
        store_id,
        SYSTEM,
        _hybrid_extracted(SYSTEM_END, float(SYSTEM_END)),
        boundary_snapshots=_BoundarySnapshotProvider(
            boundary,
            store_id,
            [],
            {},
            paged_ssd_manager=ssd,
            tail_terminal_token_count=SYSTEM_END,
        ),
        _store_tail_terminal=True,
        _store_shared_tail=True,
    )
    assert system is not None and system.num_tokens == SYSTEM_END
    system_tail_hash = _block_hashes(prefix, system)[-1]
    paged.release_for_eviction(system.block_ids)
    paged.request_tables.pop(store_id, None)
    prefix.clear_request_entry(store_id)

    signature = ssd.gdn_cache_signature_for(
        model_name="hybrid-model",
        num_layers=2,
        block_size=BLOCK_SIZE,
        layer_cache_types=LAYER_TYPES,
    )
    for block_hash in conversation_hashes:
        ssd.forget_gdn_checkpoint(block_hash, signature)
        assert ssd.has_block(block_hash)  # the KV block itself survives
    assert ssd.has_gdn_checkpoint(system_tail_hash, signature)


def test_evicted_chain_checkpoints_miss_while_system_tail_restores(caches):
    """The mechanism: the longest chain has KV but no checkpoint; the system
    prefix alone reaches the shared tail and restores."""
    paged, ssd, boundary, prefix = caches
    _store_conversation_and_system_tail(paged, ssd, boundary, prefix)

    table, _ = prefix.fetch_cache("new-context", NEW_CONTEXT, record_stats=False)
    assert table is not None and table.num_tokens == 20
    assert prefix.reconstruct_cache(table) is None
    paged.delete_block_table("new-context")

    table, _ = prefix.fetch_cache(
        "new-context", NEW_CONTEXT[:SYSTEM_END], record_stats=False
    )
    assert table is not None and table.num_tokens == SYSTEM_END
    restored = prefix.reconstruct_cache(table)
    assert restored is not None
    assert restored[0].keys_and_values()[0].shape[2] == SYSTEM_END
    assert float(restored[1].cache[0][0, 0, 0]) == pytest.approx(float(SYSTEM_END))


def _scheduler(paged, prefix):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = _HybridModel()
    scheduler.config = SchedulerConfig(
        paged_cache_block_size=BLOCK_SIZE,
        gdn_ssd_split_enabled=True,
    )
    scheduler._prefix_cache_prepared = set()
    scheduler._gdn_split_active = MagicMock(return_value=True)
    scheduler._bypass_hot_cache_under_pressure = MagicMock(return_value=False)
    scheduler._align_minimax_m3_partial_cache_to_prefill_step = MagicMock(
        return_value=False
    )
    scheduler._log_prefix_divergence = MagicMock()
    scheduler._try_specprefill_scoring = MagicMock()
    scheduler.paged_cache_manager = paged
    scheduler.block_aware_cache = prefix
    return scheduler


def _request(system_prefix_end):
    request = MagicMock()
    request.request_id = "new-context"
    request.prompt_token_ids = list(NEW_CONTEXT)
    request.system_prefix_end = system_prefix_end
    request.vlm_extra_keys_for_cache = None
    request.vlm_extra_key_token_start_for_cache = None
    request.vlm_extra_key_ranges_for_cache = None
    request.memory_requeues = 0
    return request


def test_scheduler_falls_back_to_shared_system_tail(caches):
    paged, ssd, boundary, prefix = caches
    _store_conversation_and_system_tail(paged, ssd, boundary, prefix)
    request = _request(SYSTEM_END)

    _scheduler(paged, prefix)._prepare_prefix_cache_for_request(request)

    assert request.cached_tokens == SYSTEM_END
    assert request.remaining_tokens == NEW_CONTEXT[SYSTEM_END:]
    assert request.block_table.num_tokens == SYSTEM_END


def test_scheduler_without_system_prefix_still_misses(caches):
    """No known system prefix: nothing to fall back to, full prefill as before."""
    paged, ssd, boundary, prefix = caches
    _store_conversation_and_system_tail(paged, ssd, boundary, prefix)
    request = _request(0)

    _scheduler(paged, prefix)._prepare_prefix_cache_for_request(request)

    assert request.remaining_tokens == NEW_CONTEXT
    assert "new-context" not in paged.request_tables


def _held_blocks(paged, request):
    """Blocks still referenced outside the request's own table (leaks)."""
    own = set(request.block_table.block_ids) if request.block_table else set()
    return [
        block_id
        for block_id, block in paged.allocated_blocks.items()
        if block.ref_count > 0 and block_id not in own and not block.is_null
    ]


def test_fallback_releases_the_abandoned_chain(caches):
    paged, ssd, boundary, prefix = caches
    _store_conversation_and_system_tail(paged, ssd, boundary, prefix)
    request = _request(SYSTEM_END)

    _scheduler(paged, prefix)._prepare_prefix_cache_for_request(request)

    assert request.cached_tokens == SYSTEM_END
    assert _held_blocks(paged, request) == []


def test_scheduler_misses_cleanly_when_the_system_tail_is_gone_too(caches):
    paged, ssd, boundary, prefix = caches
    _store_conversation_and_system_tail(paged, ssd, boundary, prefix)
    signature = ssd.gdn_cache_signature_for(
        model_name="hybrid-model",
        num_layers=2,
        block_size=BLOCK_SIZE,
        layer_cache_types=LAYER_TYPES,
    )
    table, _ = prefix.fetch_cache("probe", SYSTEM, record_stats=False)
    for block_hash in _block_hashes(prefix, table):
        ssd.forget_gdn_checkpoint(block_hash, signature)
    paged.delete_block_table("probe")
    request = _request(SYSTEM_END)

    _scheduler(paged, prefix)._prepare_prefix_cache_for_request(request)

    assert request.remaining_tokens == NEW_CONTEXT
    assert "new-context" not in paged.request_tables
    assert [b for b in paged.allocated_blocks.values() if b.ref_count > 0 and not b.is_null] == []


def test_longer_sibling_tail_without_checkpoint_falls_back_to_system_tail(caches):
    """The other real case: no longer grid chain, but a longer tail under the
    same parent (14,344 tokens vs the system tail's 14,337) lost its checkpoint
    and wins the longest-tail match."""
    paged, ssd, boundary, prefix = caches
    _store_conversation_and_system_tail(paged, ssd, boundary, prefix)
    sibling = SYSTEM + [600, 601]  # 3 blocks + a 3-token tail
    grid = len(sibling) // BLOCK_SIZE * BLOCK_SIZE
    table, _ = prefix.fetch_cache("sibling", sibling[:grid], record_stats=False)
    assert table is not None and table.num_tokens == grid
    _save_snapshots(boundary, "sibling", [len(sibling)])
    stored = prefix.store_cache(
        "sibling",
        sibling,
        _hybrid_extracted(len(sibling), float(len(sibling))),
        boundary_snapshots=_BoundarySnapshotProvider(
            boundary, "sibling", [], {}, paged_ssd_manager=ssd,
            tail_terminal_token_count=len(sibling),
        ),
        _store_tail_terminal=True,
    )
    assert stored is not None and stored.num_tokens == len(sibling)
    signature = ssd.gdn_cache_signature_for(
        model_name="hybrid-model",
        num_layers=2,
        block_size=BLOCK_SIZE,
        layer_cache_types=LAYER_TYPES,
    )
    ssd.forget_gdn_checkpoint(_block_hashes(prefix, stored)[-1], signature)
    paged.release_for_eviction(stored.block_ids)
    paged.request_tables.pop("sibling", None)
    prefix.clear_request_entry("sibling")

    request = _request(SYSTEM_END)
    request.prompt_token_ids = sibling + [700, 701, 702]
    _scheduler(paged, prefix)._prepare_prefix_cache_for_request(request)

    assert request.cached_tokens == SYSTEM_END
    assert request.remaining_tokens == [600, 601, 700, 701, 702]
    assert _held_blocks(paged, request) == []
