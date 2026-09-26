# SPDX-License-Identifier: Apache-2.0
"""The read-back probe must not keep what it looked at.

Canonical recovery only counts a boundary as published after asking the
serving cache, through the same ``fetch_cache`` a real request takes, whether
the boundary is actually reachable. That question acquires a reference on
every block it matches. The probe stores nothing, so the ordinary release path
— which frees through the request table only ``store_cache`` writes — has
nothing to free, and the references stay held.

The cost is not a slow leak. Every boundary the recovery publishes pins a
longer chain than the one before it, and pinned blocks are not evictable, so
the cache fills with exactly the state the recovery was meant to make
reusable.
"""

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.canonical_recovery import CanonicalRecoveryJob
from omlx.scheduler import Scheduler, SchedulerConfig

BLOCK = 4


class _Model:
    def __init__(self, num_layers: int = 1):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        args = MagicMock()
        args.num_hidden_layers = self._num_layers
        return args


def _cache_data(num_tokens: int):
    keys = mx.arange(num_tokens, dtype=mx.float32).reshape(1, 1, num_tokens, 1)
    values = (keys + 100).astype(mx.float32)
    return [{"state": (keys, values), "cache_type": "KVCache", "class_name": "KVCache"}]


@pytest.fixture
def prefix_cache():
    paged = PagedCacheManager(
        block_size=BLOCK, max_blocks=64, model_name="test-model", initial_blocks=64
    )
    return BlockAwarePrefixCache(
        model=_Model(), paged_cache_manager=paged, paged_ssd_cache_manager=None
    )


@pytest.fixture
def scheduler(prefix_cache):
    model = MagicMock()
    model.layers = []
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            paged_cache_block_size=BLOCK,
            canonical_state_recovery_enabled=True,
        ),
    )
    sched.block_aware_cache = prefix_cache
    return sched


def _ref_counts(prefix_cache, block_ids):
    """The publish holds a reference of its own, so the contract under test
    is that the probe returns to this, not that it reaches zero."""
    return {
        block_id: prefix_cache.paged_cache.allocated_blocks[block_id].ref_count
        for block_id in block_ids
    }


def _publish(prefix_cache, tokens):
    """Write a canonical prefix the way the recovery publish does."""
    table = prefix_cache.store_cache(
        "canonical-recovery:s1", tokens, _cache_data(len(tokens))
    )
    assert table is not None and table.block_ids, "the publish itself failed"
    prefix_cache.clear_request_entry("canonical-recovery:s1")
    return table


def test_readback_returns_every_reference_it_took(scheduler, prefix_cache):
    tokens = list(range(8))
    table = _publish(prefix_cache, tokens)
    before = _ref_counts(prefix_cache, table.block_ids)

    job = CanonicalRecoveryJob(
        session_key="s1",
        tokens=list(tokens),
        target_tokens=len(tokens),
        block_size=BLOCK,
    )
    restorable = scheduler._canonical_recovery_readback_tokens(job, tokens)

    # A probe that matched nothing would hold nothing, and would pass this
    # test while proving none of it.
    assert restorable > 0
    assert _ref_counts(prefix_cache, table.block_ids) == before


def test_repeated_readbacks_do_not_accumulate(scheduler, prefix_cache):
    """Each boundary of a session probes again; the pin must not compound."""
    tokens = list(range(8))
    table = _publish(prefix_cache, tokens)
    before = _ref_counts(prefix_cache, table.block_ids)

    job = CanonicalRecoveryJob(
        session_key="s1",
        tokens=list(tokens),
        target_tokens=len(tokens),
        block_size=BLOCK,
    )
    for _ in range(5):
        assert scheduler._canonical_recovery_readback_tokens(job, tokens) > 0

    assert _ref_counts(prefix_cache, table.block_ids) == before


def test_the_probe_leaves_no_block_table_behind(scheduler, prefix_cache):
    tokens = list(range(8))
    _publish(prefix_cache, tokens)
    job = CanonicalRecoveryJob(
        session_key="s1",
        tokens=list(tokens),
        target_tokens=len(tokens),
        block_size=BLOCK,
    )
    scheduler._canonical_recovery_readback_tokens(job, tokens)

    probe_id = "canonical-recovery-readback:s1"
    assert probe_id not in prefix_cache.paged_cache.request_tables
    assert probe_id not in prefix_cache._request_tables
