# SPDX-License-Identifier: Apache-2.0
"""Bounded LRU peeks preserve main/sidecar victim ordering and index state."""

from collections import OrderedDict
from pathlib import Path

import pytest

from omlx.cache.paged_ssd_cache import (
    GDNCheckpointIndex,
    GDNCheckpointMetadata,
    PagedSSDBlockMetadata,
    PagedSSDCacheIndex,
    PagedSSDCacheManager,
)


class CountingLRU(OrderedDict):
    """Count visited keys, not wall time, to catch an unbounded snapshot."""

    visited = 0

    def __iter__(self):
        for key in super().__iter__():
            self.visited += 1
            yield key

    def keys(self):
        return iter(self)


def make_index(sidecar, size=20):
    index = (GDNCheckpointIndex if sidecar else PagedSSDCacheIndex)(2**30)
    entries = []
    for i in range(size):
        common = dict(
            file_path=Path(f"synthetic-{i}"),
            file_size=100,
            token_count=256,
            model_name="synthetic",
            block_size=256,
            created_at=float(i),
            last_access=float(i),
        )
        if sidecar:
            entry = GDNCheckpointMetadata(
                source_block_hash=i.to_bytes(32, "big"),
                cache_signature_digest="synthetic-signature",
                **common,
            )
        else:
            entry = PagedSSDBlockMetadata(
                block_hash=i.to_bytes(32, "big"), num_layers=1, **common
            )
        index.add(entry)
        entries.append(entry)
    index._lru = CountingLRU(index._lru)
    return index, entries


@pytest.mark.parametrize("sidecar", [False, True])
@pytest.mark.parametrize("count", [0, 1, 3, 20, 25, -1, -25, None, 2**100, -(2**100)])
def test_lru_peek_visits_only_selected_prefix(sidecar, count):
    index, entries = make_index(sidecar)
    expected = entries[:count]
    actual = index.get_lru_entries(count)
    assert all(a is b for a, b in zip(actual, expected))
    assert len(actual) == len(expected)
    assert index._lru.visited == len(expected)
    assert index.count == 20
    assert index.total_size == 2000


@pytest.mark.parametrize("sidecar", [False, True])
def test_lru_peek_follows_touch_and_remove_without_consuming(sidecar):
    index, entries = make_index(sidecar, 4)
    if sidecar:
        index.touch(entries[0].source_block_hash, entries[0].cache_signature_digest)
        index.remove_key(entries[1].key)
    else:
        index.touch(entries[0].block_hash)
        index.remove(entries[1].block_hash)
    for _ in range(2):
        assert index.get_lru_entries(1)[0] is entries[2]
    assert index._lru.visited == 2
    assert index.count == 3
    assert index.total_size == 300


@pytest.mark.parametrize("sidecar", [False, True])
def test_lru_peek_keeps_missing_entry_prefix_semantics(sidecar):
    index, entries = make_index(sidecar, 4)
    key = entries[0].key if sidecar else entries[0].block_hash
    # Preserve existing behavior if an inconsistent index reaches this path:
    # skip the missing entry, but do not read beyond the requested LRU prefix.
    del index._index[key]
    assert index.get_lru_entries(1) == []
    assert index._lru.visited == 1
    assert index.get_lru_entries(2) == [entries[1]]
    assert index._lru.visited == 3


def test_global_eviction_uses_bounded_peeks_and_preserves_ties():
    manager = object.__new__(PagedSSDCacheManager)
    manager._index, compatible = make_index(False, 20)
    manager._incompatible_index, incompatible = make_index(False, 20)
    manager._gdn_sidecar_index, sidecars = make_index(True, 20)
    evicted = manager._evict_tracked_until_size(0, max_count=4)
    assert [(id(index), id(entry)) for index, entry in evicted] == [
        (id(manager._index), id(compatible[0])),
        (id(manager._incompatible_index), id(incompatible[0])),
        (id(manager._gdn_sidecar_index), id(sidecars[0])),
        (id(manager._index), id(compatible[1])),
    ]
    for index in (
        manager._index,
        manager._incompatible_index,
        manager._gdn_sidecar_index,
    ):
        assert index._lru.visited == 4
    assert manager._tracked_ssd_size() == 5600
