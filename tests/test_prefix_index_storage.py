# SPDX-License-Identifier: Apache-2.0
"""Prefix-index ID storage must scale with blocks, not prefix depths."""

import sys
from types import SimpleNamespace

import pytest

import omlx.cache.prefix_cache as prefix_module
from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache


def indexed(tokens, block_size=4):
    count = (len(tokens) + block_size - 1) // block_size
    paged = PagedCacheManager(
        block_size=block_size,
        max_blocks=count + 8,
        initial_blocks=count + 8,
        model_name="synthetic-index-storage",
    )
    cache = BlockAwarePrefixCache(SimpleNamespace(layers=[]), paged)
    blocks = paged.get_new_blocks(count)
    ids = [block.block_id for block in blocks]
    for block, start in zip(blocks, range(0, len(tokens), block_size)):
        block.token_count = min(block_size, len(tokens) - start)
    cache._update_prefix_index(tokens, ids)
    return cache, paged, ids


@pytest.mark.parametrize("count", [128, 1024])
def test_prefix_index_retains_linear_id_storage(count):
    cache, _, ids = indexed(list(range(count * 4)))
    snapshots = {id(entry[1]): entry[1] for entry in cache._prefix_index.values()}
    retained = sum(sys.getsizeof(value) for value in snapshots.values())
    assert retained <= sys.getsizeof(tuple(ids))
    assert len(cache._prefix_index) == count


@pytest.mark.parametrize("depth", [1, 3, 8])
def test_lookup_returns_only_matched_ids_after_input_mutation(depth):
    tokens = list(range(32))
    cache, _, ids = indexed(tokens)
    expected = tuple(ids[:depth])
    ids[:] = [9999]
    result = cache._find_best_prefix_match(tokens[: depth * 4] + [99999])
    assert result is not None
    length, matched, count, hashes = result
    assert (length, matched, count) == (depth * 4, expected, depth)
    assert len(hashes) == depth


def test_overlapping_stores_do_not_mutate_previous_branch():
    tokens = list(range(16))
    cache, paged, first_ids = indexed(tokens)
    branch = tokens[:8] + [91, 92, 93, 94, 95, 96, 97, 98]
    blocks = paged.get_new_blocks(2)
    branch_ids = first_ids[:2] + [block.block_id for block in blocks]
    for block in blocks:
        block.token_count = 4
    cache._update_prefix_index(branch, branch_ids)
    branch_ids[:] = []
    for request, prompt, expected in (
        ("original", tokens, first_ids),
        ("branch", branch, first_ids[:2] + [block.block_id for block in blocks]),
    ):
        table, remaining = cache.fetch_cache(request, prompt)
        assert table.block_ids == expected
        assert remaining == []


def test_missing_later_block_does_not_escape_lookup():
    cache, _, ids = indexed(list(range(8)))
    cache._prefix_index.clear()
    cache._update_prefix_index(list(range(12)), ids + [9999])
    result = cache._find_best_prefix_match(list(range(12)))
    assert result is not None
    assert result[:3] == (8, tuple(ids), 2)


def test_update_iterates_the_immutable_snapshot(monkeypatch):
    tokens = list(range(8))
    cache, paged, ids = indexed(tokens)
    expected = tuple(ids)
    cache._prefix_index.clear()
    for block_id in ids:
        paged.allocated_blocks[block_id].block_hash = None
    original_hash = prefix_module.compute_block_hash

    def mutate_caller(*args, **kwargs):
        ids[1] = ids[0]
        return original_hash(*args, **kwargs)

    monkeypatch.setattr(prefix_module, "compute_block_hash", mutate_caller)
    cache._update_prefix_index(tokens, ids)
    assert cache._find_best_prefix_match(tokens)[1] == expected
    assert paged.allocated_blocks[expected[1]].block_hash is not None


def test_store_cache_shares_snapshot_across_prefix_depths():
    import mlx.core as mx

    tokens = list(range(512))
    paged = PagedCacheManager(
        block_size=4,
        max_blocks=140,
        initial_blocks=140,
        model_name="synthetic-store-index",
    )
    cache = BlockAwarePrefixCache(SimpleNamespace(layers=[object()]), paged)
    state = mx.ones((1, 1, len(tokens), 8))
    mx.eval(state)
    table = cache.store_cache(
        "stored",
        tokens,
        [{"state": (state, state), "cache_type": "KVCache", "class_name": "KVCache"}],
    )
    assert table is not None
    assert len(cache._prefix_index) == 128
    snapshots = {id(entry[1]) for entry in cache._prefix_index.values()}
    assert len(snapshots) == 1
    assert cache._find_best_prefix_match(tokens)[1] == tuple(table.block_ids)
