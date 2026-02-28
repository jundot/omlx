# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-memory hot cache tier in PagedSSDCacheManager."""

import threading
import time
from pathlib import Path
from typing import List

import pytest

from omlx.cache.paged_ssd_cache import (
    PagedSSDCacheManager,
    _extract_tensor_bytes,
)


def _has_mlx() -> bool:
    """Check if MLX is available."""
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_mlx(), reason="MLX not available")
class TestHotCacheDisabled:
    """Verify that hot_cache_max_bytes=0 preserves existing behaviour."""

    @pytest.fixture
    def mx(self):
        import mlx.core as mx

        return mx

    @pytest.fixture
    def manager(self, tmp_path):
        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=0,
        )
        yield mgr
        mgr.close()

    def test_hot_cache_disabled_by_default(self, manager):
        """hot_cache_max_bytes=0 means hot cache is disabled."""
        assert manager._hot_cache_enabled is False
        assert manager._hot_cache_max_bytes == 0

    def test_save_load_works_without_hot_cache(self, manager, mx):
        """Save/load should work even when hot cache is disabled."""
        block_hash = b"disabled_hot_cache_test"
        cache_data = [
            (mx.zeros((1, 8, 64, 64)), mx.zeros((1, 8, 64, 64)))
            for _ in range(4)
        ]
        result = manager.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=64,
            model_name="test-model",
            layer_cache_types=["KVCache"] * 4,
        )
        assert result is True
        assert manager.has_block(block_hash)

        loaded = manager.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 4

    def test_stats_hot_cache_zero_when_disabled(self, manager):
        """Hot cache stats should be zero when disabled."""
        stats = manager.get_stats()
        assert stats.hot_cache_entries == 0
        assert stats.hot_cache_size_bytes == 0
        assert stats.hot_cache_max_bytes == 0
        assert stats.hot_cache_hits == 0
        assert stats.hot_cache_evictions == 0
        assert stats.hot_cache_promotions == 0


@pytest.mark.skipif(not _has_mlx(), reason="MLX not available")
class TestHotCacheEnabled:
    """Test hot cache with in-memory caching active."""

    @pytest.fixture
    def mx(self):
        import mlx.core as mx

        return mx

    @pytest.fixture
    def manager(self, tmp_path):
        # 10 MB hot cache — generous for test blocks
        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=10 * 1024**2,
        )
        yield mgr
        mgr.close()

    def _make_cache_data(self, mx, num_layers=4, seq_len=32, heads=4, head_dim=32):
        """Create test cache data."""
        return [
            (
                mx.zeros((1, heads, seq_len, head_dim)),
                mx.zeros((1, heads, seq_len, head_dim)),
            )
            for _ in range(num_layers)
        ]

    def _save_block(self, manager, mx, block_hash, num_layers=4, model="test-model"):
        """Save a test block and return True on success."""
        cache_data = self._make_cache_data(mx, num_layers=num_layers)
        return manager.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=64,
            model_name=model,
            layer_cache_types=["KVCache"] * num_layers,
        )

    def test_save_stores_in_hot_cache(self, manager, mx):
        """After save_block(), the entry should be in hot cache."""
        block_hash = b"hot_cache_save_test1"
        self._save_block(manager, mx, block_hash)

        # Verify hot cache has the entry
        entry = manager._hot_cache_get(block_hash)
        assert entry is not None
        assert 'tensors_raw' in entry
        assert entry['num_layers'] == 4

    def test_load_from_hot_cache(self, manager, mx):
        """load_block() should return data from hot cache without SSD I/O."""
        block_hash = b"hot_cache_load_test1"
        self._save_block(manager, mx, block_hash)

        loaded = manager.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 4

        stats = manager.get_stats()
        assert stats.hot_cache_hits >= 1

    def test_hot_cache_hit_updates_stats(self, manager, mx):
        """Hot cache hit should increment hot_cache_hits counter."""
        block_hash = b"hot_cache_stats_test1"
        self._save_block(manager, mx, block_hash)

        initial_stats = manager.get_stats()
        initial_hits = initial_stats.hot_cache_hits

        manager.load_block(block_hash)
        manager.load_block(block_hash)

        stats = manager.get_stats()
        assert stats.hot_cache_hits >= initial_hits + 2

    def test_hot_cache_size_tracking(self, manager, mx):
        """Hot cache should track total size in bytes."""
        block_hash = b"hot_cache_size_test1"
        self._save_block(manager, mx, block_hash)

        stats = manager.get_stats()
        assert stats.hot_cache_entries == 1
        assert stats.hot_cache_size_bytes > 0
        assert stats.hot_cache_max_bytes == 10 * 1024**2

    def test_delete_block_removes_from_hot_cache(self, manager, mx):
        """delete_block() should remove entry from hot cache."""
        block_hash = b"hot_cache_delete_test"
        self._save_block(manager, mx, block_hash)

        # Verify it's in hot cache
        assert manager._hot_cache_get(block_hash) is not None

        manager.delete_block(block_hash)

        # Verify it's gone from hot cache
        assert manager._hot_cache_get(block_hash) is None

    def test_close_clears_hot_cache(self, tmp_path, mx):
        """close() should clear all hot cache entries."""
        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "close_test",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=10 * 1024**2,
        )
        block_hash = b"hot_cache_close_test1"
        cache_data = self._make_cache_data(mx)
        mgr.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=64,
            model_name="test",
            layer_cache_types=["KVCache"] * 4,
        )
        assert len(mgr._hot_cache) > 0

        mgr.close()

        assert len(mgr._hot_cache) == 0
        assert mgr._hot_cache_total_bytes == 0


@pytest.mark.skipif(not _has_mlx(), reason="MLX not available")
class TestHotCacheLRU:
    """Test LRU eviction behaviour of the hot cache."""

    @pytest.fixture
    def mx(self):
        import mlx.core as mx

        return mx

    def _make_cache_data(self, mx, num_layers=2, seq_len=16, heads=2, head_dim=16):
        """Create small test cache data."""
        return [
            (
                mx.zeros((1, heads, seq_len, head_dim)),
                mx.zeros((1, heads, seq_len, head_dim)),
            )
            for _ in range(num_layers)
        ]

    def _entry_size(self, mx, num_layers=2, seq_len=16, heads=2, head_dim=16):
        """Estimate the raw byte size of one entry."""
        # Each tensor: 1 * heads * seq_len * head_dim * 4 bytes (float32)
        # 2 tensors (keys + values) per layer, num_layers layers
        return num_layers * 2 * 1 * heads * seq_len * head_dim * 4

    def test_lru_eviction(self, tmp_path, mx):
        """Old entries should be evicted when capacity is exceeded."""
        entry_size = self._entry_size(mx)
        # Allow room for exactly 2 entries
        max_bytes = entry_size * 2 + 100

        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "lru_test",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=max_bytes,
        )

        try:
            # Save 3 blocks — the first should be evicted
            for i in range(3):
                block_hash = f"lru_block_{i}".encode()
                cache_data = self._make_cache_data(mx)
                mgr.save_block(
                    block_hash=block_hash,
                    cache_data=cache_data,
                    token_count=16,
                    model_name="test",
                    layer_cache_types=["KVCache"] * 2,
                )

            # Block 0 should have been evicted (LRU)
            assert mgr._hot_cache_get(b"lru_block_0") is None
            # Blocks 1 and 2 should still be in hot cache
            assert mgr._hot_cache_get(b"lru_block_1") is not None
            assert mgr._hot_cache_get(b"lru_block_2") is not None

            stats = mgr.get_stats()
            assert stats.hot_cache_evictions >= 1
        finally:
            mgr.close()

    def test_lru_access_refreshes_order(self, tmp_path, mx):
        """Accessing a block should move it to MRU position."""
        entry_size = self._entry_size(mx)
        max_bytes = entry_size * 2 + 100

        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "lru_order_test",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=max_bytes,
        )

        try:
            # Save blocks 0 and 1
            for i in range(2):
                block_hash = f"order_block_{i}".encode()
                cache_data = self._make_cache_data(mx)
                mgr.save_block(
                    block_hash=block_hash,
                    cache_data=cache_data,
                    token_count=16,
                    model_name="test",
                    layer_cache_types=["KVCache"] * 2,
                )

            # Access block 0 to refresh its LRU position
            mgr.load_block(b"order_block_0")

            # Save block 2 — should evict block 1 (LRU), not block 0
            cache_data = self._make_cache_data(mx)
            mgr.save_block(
                block_hash=b"order_block_2",
                cache_data=cache_data,
                token_count=16,
                model_name="test",
                layer_cache_types=["KVCache"] * 2,
            )

            # Block 0 was accessed so should still be present
            assert mgr._hot_cache_get(b"order_block_0") is not None
            # Block 1 was LRU and should be evicted
            assert mgr._hot_cache_get(b"order_block_1") is None
            # Block 2 was just added
            assert mgr._hot_cache_get(b"order_block_2") is not None
        finally:
            mgr.close()


@pytest.mark.skipif(not _has_mlx(), reason="MLX not available")
class TestHotCachePromotion:
    """Test promotion from SSD to hot cache on load."""

    @pytest.fixture
    def mx(self):
        import mlx.core as mx

        return mx

    def _make_cache_data(self, mx, num_layers=4, seq_len=32, heads=4, head_dim=32):
        return [
            (
                mx.zeros((1, heads, seq_len, head_dim)),
                mx.zeros((1, heads, seq_len, head_dim)),
            )
            for _ in range(num_layers)
        ]

    def test_ssd_load_promotes_to_hot_cache(self, tmp_path, mx):
        """Loading a block from SSD should promote it to hot cache."""
        # Use hot_cache disabled to write directly to SSD first
        mgr_cold = PagedSSDCacheManager(
            cache_dir=tmp_path / "promote_test",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=0,
        )

        block_hash = b"promote_test_block1"
        cache_data = self._make_cache_data(mx)
        mgr_cold.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=64,
            model_name="test",
            layer_cache_types=["KVCache"] * 4,
        )
        # Wait for background SSD write to complete
        time.sleep(0.5)
        mgr_cold.close()

        # Now open with hot cache enabled — block is on SSD only
        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "promote_test",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=10 * 1024**2,
        )

        try:
            assert mgr._hot_cache_get(block_hash) is None

            # Load from SSD — should promote to hot cache
            loaded = mgr.load_block(block_hash)
            assert loaded is not None
            assert len(loaded) == 4

            # Verify promotion happened
            assert mgr._hot_cache_get(block_hash) is not None
            stats = mgr.get_stats()
            assert stats.hot_cache_promotions >= 1
        finally:
            mgr.close()

    def test_promotion_does_not_happen_when_disabled(self, tmp_path, mx):
        """No promotion when hot cache is disabled."""
        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "no_promote_test",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=0,
        )

        try:
            block_hash = b"no_promote_block1__"
            cache_data = self._make_cache_data(mx)
            mgr.save_block(
                block_hash=block_hash,
                cache_data=cache_data,
                token_count=64,
                model_name="test",
                layer_cache_types=["KVCache"] * 4,
            )

            # Wait for background write
            time.sleep(0.5)

            # Clear the temporary buffer (simulates what happens after write completes)
            mgr._hot_cache_remove(block_hash)

            # Load from SSD
            loaded = mgr.load_block(block_hash)
            assert loaded is not None

            stats = mgr.get_stats()
            assert stats.hot_cache_promotions == 0
        finally:
            mgr.close()


@pytest.mark.skipif(not _has_mlx(), reason="MLX not available")
class TestHotCacheCacheTypes:
    """Test hot cache with various cache types (KVCache, CacheList)."""

    @pytest.fixture
    def mx(self):
        import mlx.core as mx

        return mx

    @pytest.fixture
    def manager(self, tmp_path):
        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "types_test",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=10 * 1024**2,
        )
        yield mgr
        mgr.close()

    def test_cache_list_blocks(self, manager, mx):
        """Hot cache should handle CacheList blocks correctly."""
        block_hash = b"cache_list_hot_test"

        sub_keys1 = mx.zeros((1, 8, 32, 64))
        sub_values1 = mx.ones((1, 8, 32, 64))
        sub_keys2 = mx.zeros((1, 4, 32, 64))
        sub_values2 = mx.ones((1, 4, 32, 64))

        cache_data = [
            (
                "__cache_list__",
                [(sub_keys1, sub_values1), (sub_keys2, sub_values2)],
            ),
            (mx.zeros((1, 8, 32, 64)), mx.ones((1, 8, 32, 64))),
        ]

        result = manager.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=32,
            model_name="test",
            layer_cache_types=["CacheList", "KVCache"],
        )
        assert result is True

        # Load from hot cache
        loaded = manager.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 2
        # First layer is CacheList
        assert isinstance(loaded[0], list)
        assert len(loaded[0]) == 2
        # Second layer is KVCache tuple
        assert isinstance(loaded[1], tuple)


class TestHotCacheConcurrency:
    """Test thread safety of hot cache internal operations."""

    def test_concurrent_put_get(self, tmp_path):
        """Hot cache put/get should be thread-safe under concurrent access."""
        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "concurrent_test",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=50 * 1024**2,
        )

        errors: List[Exception] = []
        num_threads = 8
        ops_per_thread = 20

        def worker(thread_id):
            try:
                for i in range(ops_per_thread):
                    block_hash = f"conc_{thread_id}_{i}____".encode()
                    # Create a fake hot cache entry with raw bytes
                    raw_data = bytes(1024)  # 1KB of zeros
                    entry = {
                        'tensors_raw': {
                            'layer_0_keys': (raw_data, 'float32', [1, 2, 16, 8]),
                            'layer_0_values': (raw_data, 'float32', [1, 2, 16, 8]),
                        },
                        'file_metadata': {},
                        'num_layers': 1,
                        'layer_cache_types': ['KVCache'],
                        'block_metadata': None,
                    }
                    mgr._hot_cache_put(block_hash, entry)

                # Read back
                for i in range(ops_per_thread):
                    block_hash = f"conc_{thread_id}_{i}____".encode()
                    result = mgr._hot_cache_get(block_hash)
                    # May be None if evicted by another thread, that's OK
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(t,)) for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        mgr.close()

        assert len(errors) == 0, f"Concurrent errors: {errors}"


@pytest.mark.skipif(not _has_mlx(), reason="MLX not available")
class TestHotCacheStatsAccuracy:
    """Test that hot cache statistics are accurate."""

    @pytest.fixture
    def mx(self):
        import mlx.core as mx

        return mx

    def test_all_stats_counters(self, tmp_path, mx):
        """Verify hot cache stats counters are correctly maintained."""
        entry_size = 2 * 2 * 1 * 2 * 16 * 16 * 4  # ~4096 bytes per entry
        # Room for 2 entries
        max_bytes = entry_size * 2 + 100

        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "stats_test",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=max_bytes,
        )

        try:

            def save(idx):
                block_hash = f"stats_block_{idx}__".encode()
                cache_data = [
                    (mx.zeros((1, 2, 16, 16)), mx.zeros((1, 2, 16, 16)))
                    for _ in range(2)
                ]
                mgr.save_block(
                    block_hash=block_hash,
                    cache_data=cache_data,
                    token_count=16,
                    model_name="test",
                    layer_cache_types=["KVCache"] * 2,
                )

            # Save 2 blocks: fits in hot cache
            save(0)
            save(1)
            stats = mgr.get_stats()
            assert stats.hot_cache_entries == 2
            assert stats.hot_cache_evictions == 0

            # Save 3rd block: triggers eviction of block 0
            save(2)
            stats = mgr.get_stats()
            assert stats.hot_cache_entries == 2
            assert stats.hot_cache_evictions >= 1

            # Load block 1 (hot cache hit)
            mgr.load_block(b"stats_block_1__")
            stats = mgr.get_stats()
            assert stats.hot_cache_hits >= 1

            # Verify size tracking is positive
            assert stats.hot_cache_size_bytes > 0
            assert stats.hot_cache_max_bytes == max_bytes
        finally:
            mgr.close()


@pytest.mark.skipif(not _has_mlx(), reason="MLX not available")
class TestHotCacheWriteBack:
    """Test write-back behavior: no SSD writes until eviction or shutdown."""

    @pytest.fixture
    def mx(self):
        import mlx.core as mx

        return mx

    def _make_cache_data(self, mx, num_layers=2, seq_len=16, heads=2, head_dim=16):
        return [
            (
                mx.zeros((1, heads, seq_len, head_dim)),
                mx.zeros((1, heads, seq_len, head_dim)),
            )
            for _ in range(num_layers)
        ]

    def test_save_does_not_write_to_ssd(self, tmp_path, mx):
        """With hot cache enabled, save_block should not create SSD files."""
        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "wb_test",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=10 * 1024**2,
        )

        try:
            block_hash = b"wb_no_ssd_write_t1"
            cache_data = self._make_cache_data(mx)
            mgr.save_block(
                block_hash=block_hash,
                cache_data=cache_data,
                token_count=16,
                model_name="test",
                layer_cache_types=["KVCache"] * 2,
            )

            # Block should be in hot cache
            assert mgr._hot_cache_get(block_hash) is not None

            # No SSD file should exist yet
            time.sleep(0.3)
            ssd_files = list((tmp_path / "wb_test").rglob("*.safetensors"))
            assert len(ssd_files) == 0, f"Unexpected SSD files: {ssd_files}"
        finally:
            mgr.close()

    def test_eviction_writes_to_ssd(self, tmp_path, mx):
        """When hot cache evicts, the evicted block should be written to SSD."""
        entry_size = 2 * 2 * 1 * 2 * 16 * 16 * 4
        max_bytes = entry_size * 2 + 100

        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "wb_evict_test",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=max_bytes,
        )

        try:
            # Save 2 blocks (fits in hot cache)
            for i in range(2):
                block_hash = f"wb_evict_blk_{i}__".encode()
                cache_data = self._make_cache_data(mx)
                mgr.save_block(
                    block_hash=block_hash,
                    cache_data=cache_data,
                    token_count=16,
                    model_name="test",
                    layer_cache_types=["KVCache"] * 2,
                )

            # No SSD files yet
            time.sleep(0.3)
            ssd_files = list((tmp_path / "wb_evict_test").rglob("*.safetensors"))
            assert len(ssd_files) == 0

            # Save 3rd block → evicts block 0 → should trigger SSD write
            cache_data = self._make_cache_data(mx)
            mgr.save_block(
                block_hash=b"wb_evict_blk_2__",
                cache_data=cache_data,
                token_count=16,
                model_name="test",
                layer_cache_types=["KVCache"] * 2,
            )

            # Wait for background writer to process the evicted block
            time.sleep(0.5)
            ssd_files = list((tmp_path / "wb_evict_test").rglob("*.safetensors"))
            assert len(ssd_files) >= 1, "Evicted block should be written to SSD"
        finally:
            mgr.close()

    def test_close_flushes_hot_cache_to_ssd(self, tmp_path, mx):
        """close() should flush all hot cache entries to SSD."""
        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "wb_flush_test",
            max_size_bytes=100 * 1024**2,
            hot_cache_max_bytes=10 * 1024**2,
        )

        block_hashes = []
        for i in range(3):
            block_hash = f"wb_flush_blk_{i}__".encode()
            block_hashes.append(block_hash)
            cache_data = self._make_cache_data(mx)
            mgr.save_block(
                block_hash=block_hash,
                cache_data=cache_data,
                token_count=16,
                model_name="test",
                layer_cache_types=["KVCache"] * 2,
            )

        # No SSD files before close
        time.sleep(0.3)
        ssd_files = list((tmp_path / "wb_flush_test").rglob("*.safetensors"))
        assert len(ssd_files) == 0

        # Close flushes to SSD
        mgr.close()

        ssd_files = list((tmp_path / "wb_flush_test").rglob("*.safetensors"))
        assert len(ssd_files) == 3, (
            f"Expected 3 SSD files after flush, got {len(ssd_files)}"
        )
