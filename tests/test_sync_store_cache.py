# SPDX-License-Identifier: Apache-2.0
"""Tests for OMLX_SYNC_STORE_CACHE.

The env flag disables the async store-cache executor so store_cache (including
the cached_block_hash_to_block index registration) runs synchronously on the
inference thread via the existing fallback path, making the prefix cache
immediately consistent for back-to-back / rapid requests.
"""

import pytest

import omlx.scheduler as scheduler_module
from omlx.scheduler import Scheduler, SchedulerConfig


@pytest.fixture
def make_scheduler(mock_model, mock_tokenizer, tmp_path):
    """Factory that builds a Scheduler with the paged SSD cache enabled (so the
    store-cache executor branch is exercised) and shuts each one down on teardown
    (the default branch spins up a real ThreadPoolExecutor + SSD writer thread).

    Returned as a factory so the test can set OMLX_SYNC_STORE_CACHE *before*
    construction.
    """
    created = []

    def _make():
        if not scheduler_module.HAS_TIERED_CACHE:
            pytest.skip("tiered cache modules are unavailable")
        config = SchedulerConfig(paged_ssd_cache_dir=str(tmp_path / "cache"))
        scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer, config=config)
        if scheduler.block_aware_cache is None:
            scheduler.shutdown()
            pytest.skip("paged SSD cache did not initialize in this environment")
        created.append(scheduler)
        return scheduler

    yield _make

    for scheduler in created:
        scheduler.shutdown()


class TestSyncStoreCacheFlag:
    def test_default_uses_async_executor(self, make_scheduler, monkeypatch):
        """Unset (default): the async store-cache executor is created."""
        monkeypatch.delenv("OMLX_SYNC_STORE_CACHE", raising=False)
        scheduler = make_scheduler()
        assert scheduler._store_cache_executor is not None

    def test_flag_disables_executor(self, make_scheduler, monkeypatch):
        """OMLX_SYNC_STORE_CACHE=1: executor is disabled -> store_cache runs
        synchronously via the fallback path, while the cache stays enabled."""
        monkeypatch.setenv("OMLX_SYNC_STORE_CACHE", "1")
        scheduler = make_scheduler()
        assert scheduler._store_cache_executor is None
        assert scheduler.block_aware_cache is not None

    def test_non_one_value_keeps_async(self, make_scheduler, monkeypatch):
        """Only the exact value "1" enables sync mode; other values are async."""
        monkeypatch.setenv("OMLX_SYNC_STORE_CACHE", "0")
        scheduler = make_scheduler()
        assert scheduler._store_cache_executor is not None
