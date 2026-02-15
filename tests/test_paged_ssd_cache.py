# SPDX-License-Identifier: Apache-2.0
"""
Tests for PagedSSDCacheManager and related components.

This module tests SSD-based storage for paged KV cache blocks,
enabling larger effective cache sizes than GPU memory allows.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from omlx.cache.paged_ssd_cache import (
    PagedSSDBlockMetadata,
    PagedSSDCacheIndex,
    PagedSSDCacheManager,
    parse_size,
)


class TestParseSize:
    """Tests for parse_size utility function."""

    def test_parse_bytes(self):
        """Test parsing plain bytes."""
        assert parse_size("1024") == 1024
        assert parse_size("0") == 0

    def test_parse_kb(self):
        """Test parsing kilobytes."""
        assert parse_size("1KB") == 1024
        assert parse_size("10kb") == 10 * 1024
        assert parse_size("1.5KB") == int(1.5 * 1024)

    def test_parse_mb(self):
        """Test parsing megabytes."""
        assert parse_size("1MB") == 1024**2
        assert parse_size("100mb") == 100 * 1024**2

    def test_parse_gb(self):
        """Test parsing gigabytes."""
        assert parse_size("1GB") == 1024**3
        assert parse_size("16gb") == 16 * 1024**3
        assert parse_size("0.5GB") == int(0.5 * 1024**3)

    def test_parse_tb(self):
        """Test parsing terabytes."""
        assert parse_size("1TB") == 1024**4
        assert parse_size("2tb") == 2 * 1024**4

    def test_parse_with_whitespace(self):
        """Test parsing with whitespace."""
        assert parse_size("  100MB  ") == 100 * 1024**2

    def test_invalid_format(self):
        """Test invalid format raises ValueError."""
        with pytest.raises(ValueError):
            parse_size("invalid")
        with pytest.raises(ValueError):
            parse_size("MB100")


class TestPagedSSDBlockMetadata:
    """Tests for PagedSSDBlockMetadata dataclass."""

    def test_creation(self):
        """Test creating metadata."""
        metadata = PagedSSDBlockMetadata(
            block_hash=b"test_hash_bytes_1234",
            file_path=Path("/tmp/cache/test.safetensors"),
            file_size=1024,
            token_count=64,
            created_at=time.time(),
            last_access=time.time(),
            num_layers=32,
            model_name="test-model",
        )

        assert metadata.block_hash == b"test_hash_bytes_1234"
        assert metadata.file_size == 1024
        assert metadata.token_count == 64
        assert metadata.num_layers == 32
        assert metadata.model_name == "test-model"

    def test_touch(self):
        """Test touch updates last_access."""
        metadata = PagedSSDBlockMetadata(
            block_hash=b"test_hash_bytes_1234",
            file_path=Path("/tmp/test.safetensors"),
            file_size=1024,
            token_count=64,
            created_at=1000.0,
            last_access=1000.0,
            num_layers=32,
        )

        old_access = metadata.last_access
        time.sleep(0.01)
        metadata.touch()

        assert metadata.last_access > old_access

    def test_to_dict(self):
        """Test converting to dictionary."""
        now = time.time()
        metadata = PagedSSDBlockMetadata(
            block_hash=b"test_hash_bytes_1234",
            file_path=Path("/tmp/test.safetensors"),
            file_size=1024,
            token_count=64,
            created_at=now,
            last_access=now,
            num_layers=32,
            model_name="test-model",
            layer_cache_types=["KVCache", "ArraysCache"],
            layer_meta_states=[(0,), (1, 2, 3, 4)],
        )

        d = metadata.to_dict()

        assert d["block_hash"] == b"test_hash_bytes_1234".hex()
        assert d["file_path"] == "/tmp/test.safetensors"
        assert d["file_size"] == 1024
        assert d["token_count"] == 64
        assert d["num_layers"] == 32
        assert d["model_name"] == "test-model"
        assert d["layer_cache_types"] == ["KVCache", "ArraysCache"]
        assert d["layer_meta_states"] == [[0], [1, 2, 3, 4]]

    def test_from_dict(self):
        """Test creating from dictionary."""
        d = {
            "block_hash": b"test_hash_bytes_1234".hex(),
            "file_path": "/tmp/test.safetensors",
            "file_size": 1024,
            "token_count": 64,
            "created_at": 1000.0,
            "last_access": 1000.0,
            "num_layers": 32,
            "model_name": "test-model",
            "layer_cache_types": ["KVCache", "RotatingKVCache"],
            "layer_meta_states": [[0], [1, 2, 3, 4]],
        }

        metadata = PagedSSDBlockMetadata.from_dict(d)

        assert metadata.block_hash == b"test_hash_bytes_1234"
        assert metadata.file_path == Path("/tmp/test.safetensors")
        assert metadata.file_size == 1024
        assert metadata.layer_cache_types == ["KVCache", "RotatingKVCache"]
        assert metadata.layer_meta_states == [(0,), (1, 2, 3, 4)]

    def test_from_dict_without_optional_fields(self):
        """Test creating from dict without optional fields."""
        d = {
            "block_hash": b"test_hash".hex(),
            "file_path": "/tmp/test.safetensors",
            "file_size": 512,
            "token_count": 32,
            "created_at": 1000.0,
            "last_access": 1000.0,
            "num_layers": 16,
        }

        metadata = PagedSSDBlockMetadata.from_dict(d)

        assert metadata.model_name == ""
        assert metadata.layer_cache_types is None
        assert metadata.layer_meta_states is None


class TestPagedSSDCacheIndex:
    """Tests for PagedSSDCacheIndex (in-memory index)."""

    def test_empty_index(self):
        """Test empty index."""
        index = PagedSSDCacheIndex(max_size_bytes=1024**3)
        assert index.count == 0
        assert index.total_size == 0

    def test_add(self):
        """Test adding metadata."""
        index = PagedSSDCacheIndex(max_size_bytes=1024**3)

        metadata = PagedSSDBlockMetadata(
            block_hash=b"hash1_bytes_padding",
            file_path=Path("/tmp/1.safetensors"),
            file_size=1024,
            token_count=64,
            created_at=time.time(),
            last_access=time.time(),
            num_layers=32,
        )

        index.add(metadata)

        assert index.count == 1
        assert index.total_size == 1024

    def test_add_updates_existing(self):
        """Test adding with same hash updates existing entry."""
        index = PagedSSDCacheIndex(max_size_bytes=1024**3)
        block_hash = b"same_hash_bytes_pad"

        metadata1 = PagedSSDBlockMetadata(
            block_hash=block_hash,
            file_path=Path("/tmp/1.safetensors"),
            file_size=1024,
            token_count=64,
            created_at=time.time(),
            last_access=time.time(),
            num_layers=32,
        )

        metadata2 = PagedSSDBlockMetadata(
            block_hash=block_hash,
            file_path=Path("/tmp/2.safetensors"),
            file_size=2048,
            token_count=128,
            created_at=time.time(),
            last_access=time.time(),
            num_layers=32,
        )

        index.add(metadata1)
        assert index.total_size == 1024

        index.add(metadata2)
        # Should update, not add
        assert index.count == 1
        assert index.total_size == 2048

    def test_get(self):
        """Test getting metadata by hash."""
        index = PagedSSDCacheIndex(max_size_bytes=1024**3)
        block_hash = b"test_get_hash_bytes"

        metadata = PagedSSDBlockMetadata(
            block_hash=block_hash,
            file_path=Path("/tmp/test.safetensors"),
            file_size=1024,
            token_count=64,
            created_at=time.time(),
            last_access=time.time(),
            num_layers=32,
        )

        index.add(metadata)

        retrieved = index.get(block_hash)
        assert retrieved is metadata

        # Non-existent
        assert index.get(b"nonexistent_hash_by") is None

    def test_remove(self):
        """Test removing metadata."""
        index = PagedSSDCacheIndex(max_size_bytes=1024**3)
        block_hash = b"test_remove_hash_by"

        metadata = PagedSSDBlockMetadata(
            block_hash=block_hash,
            file_path=Path("/tmp/test.safetensors"),
            file_size=1024,
            token_count=64,
            created_at=time.time(),
            last_access=time.time(),
            num_layers=32,
        )

        index.add(metadata)
        assert index.count == 1

        removed = index.remove(block_hash)
        assert removed is metadata
        assert index.count == 0
        assert index.total_size == 0

    def test_remove_nonexistent(self):
        """Test removing nonexistent entry returns None."""
        index = PagedSSDCacheIndex(max_size_bytes=1024**3)
        result = index.remove(b"nonexistent_hash_by")
        assert result is None

    def test_touch(self):
        """Test touching updates LRU order."""
        index = PagedSSDCacheIndex(max_size_bytes=1024**3)

        # Add multiple entries
        for i in range(3):
            metadata = PagedSSDBlockMetadata(
                block_hash=f"hash_{i}_bytes_padding".encode()[:20],
                file_path=Path(f"/tmp/{i}.safetensors"),
                file_size=1024,
                token_count=64,
                created_at=time.time(),
                last_access=time.time(),
                num_layers=32,
            )
            index.add(metadata)
            time.sleep(0.01)  # Ensure different access times

        # Touch first entry (should move to end of LRU)
        first_hash = b"hash_0_bytes_padding"[:20]
        index.touch(first_hash)

        # Get LRU entries - first hash should not be first anymore
        lru_entries = index.get_lru_entries(3)
        lru_hashes = [e.block_hash for e in lru_entries]
        assert lru_hashes[0] != first_hash

    def test_get_lru_entries(self):
        """Test getting LRU entries."""
        index = PagedSSDCacheIndex(max_size_bytes=1024**3)

        # Add entries
        for i in range(5):
            metadata = PagedSSDBlockMetadata(
                block_hash=f"hash_{i}_bytes_padding".encode()[:20],
                file_path=Path(f"/tmp/{i}.safetensors"),
                file_size=1024,
                token_count=64,
                created_at=time.time(),
                last_access=time.time(),
                num_layers=32,
            )
            index.add(metadata)
            time.sleep(0.001)

        lru_entries = index.get_lru_entries(3)
        assert len(lru_entries) == 3

    def test_evict_until_size(self):
        """Test evicting until size limit."""
        index = PagedSSDCacheIndex(max_size_bytes=10240)

        # Add 5 entries of 1024 bytes each = 5120 total
        for i in range(5):
            metadata = PagedSSDBlockMetadata(
                block_hash=f"hash_{i}_bytes_padding".encode()[:20],
                file_path=Path(f"/tmp/{i}.safetensors"),
                file_size=1024,
                token_count=64,
                created_at=time.time(),
                last_access=time.time(),
                num_layers=32,
            )
            index.add(metadata)

        assert index.total_size == 5120

        # Evict until size is below 3000
        evicted = index.evict_until_size(3000)

        assert len(evicted) >= 2  # At least 2 entries evicted
        assert index.total_size <= 3000

    def test_contains(self):
        """Test checking if block exists."""
        index = PagedSSDCacheIndex(max_size_bytes=1024**3)
        block_hash = b"contains_test_hash1"

        assert not index.contains(block_hash)

        metadata = PagedSSDBlockMetadata(
            block_hash=block_hash,
            file_path=Path("/tmp/test.safetensors"),
            file_size=1024,
            token_count=64,
            created_at=time.time(),
            last_access=time.time(),
            num_layers=32,
        )
        index.add(metadata)

        assert index.contains(block_hash)

    def test_properties(self):
        """Test index properties."""
        max_size = 1024**3
        index = PagedSSDCacheIndex(max_size_bytes=max_size)

        assert index.max_size == max_size
        assert index.count == 0
        assert index.total_size == 0

        # Add some entries
        for i in range(3):
            metadata = PagedSSDBlockMetadata(
                block_hash=f"hash_{i}_bytes_padding".encode()[:20],
                file_path=Path(f"/tmp/{i}.safetensors"),
                file_size=1024,
                token_count=64,
                created_at=time.time(),
                last_access=time.time(),
                num_layers=32,
            )
            index.add(metadata)

        assert index.count == 3
        assert index.total_size == 3072

    def test_get_all_hashes(self):
        """Test getting all indexed hashes."""
        index = PagedSSDCacheIndex(max_size_bytes=1024**3)

        hashes = []
        for i in range(3):
            block_hash = f"hash_{i}_bytes_padding".encode()[:20]
            hashes.append(block_hash)
            metadata = PagedSSDBlockMetadata(
                block_hash=block_hash,
                file_path=Path(f"/tmp/{i}.safetensors"),
                file_size=1024,
                token_count=64,
                created_at=time.time(),
                last_access=time.time(),
                num_layers=32,
            )
            index.add(metadata)

        all_hashes = index.get_all_hashes()
        assert len(all_hashes) == 3
        for h in hashes:
            assert h in all_hashes


class TestPagedSSDCacheManager:
    """Tests for PagedSSDCacheManager."""

    def test_initialization(self, tmp_path: Path):
        """Test manager initialization."""
        cache_dir = tmp_path / "ssd_cache"

        manager = PagedSSDCacheManager(
            cache_dir=cache_dir,
            max_size_bytes=1024**3,
        )

        assert cache_dir.exists()
        # Check subdirectories created
        for char in "0123456789abcdef":
            assert (cache_dir / char).exists()

    def test_has_block(self, tmp_path: Path):
        """Test checking if block exists."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        # Non-existent block
        assert not manager.has_block(b"nonexistent_hash_by")

    def test_delete_block(self, tmp_path: Path):
        """Test deleting a block."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        # Delete non-existent
        result = manager.delete_block(b"nonexistent_hash_by")
        assert result is False

    def test_clear(self, tmp_path: Path):
        """Test clearing all cache."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        count = manager.clear()
        assert count == 0  # Empty cache

    def test_get_stats(self, tmp_path: Path):
        """Test getting statistics."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        stats = manager.get_stats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.saves == 0
        assert stats.loads == 0
        assert stats.errors == 0

    def test_get_stats_dict(self, tmp_path: Path):
        """Test getting statistics as dictionary."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        stats_dict = manager.get_stats_dict()

        assert "cache_dir" in stats_dict
        assert "max_size" in stats_dict
        assert "total_size" in stats_dict
        assert "num_files" in stats_dict
        assert "utilization" in stats_dict

    def test_cache_manager_interface(self, tmp_path: Path):
        """Test CacheManager ABC interface."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        # Test fetch (miss)
        value, hit = manager.fetch(b"nonexistent_key_byt")
        assert hit is False
        assert value is None

        # Test evict
        result = manager.evict(b"nonexistent_key_byt")
        assert result is False

        # Test size and max_size
        assert manager.size == 0
        assert manager.max_size == 1024**3

    def test_prefix_index_save_and_load(self, tmp_path: Path):
        """Test saving and loading prefix index."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        # Save prefix index
        entries = [
            {"block_hash": "abc123", "token_count": 64},
            {"block_hash": "def456", "token_count": 128},
        ]
        result = manager.save_prefix_index(entries, model_name="test-model")
        assert result is True

        # Load prefix index
        loaded = manager.load_prefix_index(model_name="test-model")
        assert len(loaded) == 2
        assert loaded[0]["block_hash"] == "abc123"

    def test_prefix_index_model_mismatch(self, tmp_path: Path):
        """Test loading prefix index with model mismatch."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        # Save with one model
        entries = [{"block_hash": "abc123", "token_count": 64}]
        manager.save_prefix_index(entries, model_name="model-a")

        # Load with different model
        loaded = manager.load_prefix_index(model_name="model-b")
        assert len(loaded) == 0  # Should be empty due to mismatch

    def test_close(self, tmp_path: Path):
        """Test closing the manager."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        # Should not raise
        manager.close()

    def test_repr(self, tmp_path: Path):
        """Test string representation."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        repr_str = repr(manager)
        assert "PagedSSDCacheManager" in repr_str
        assert "ssd_cache" in repr_str

    def test_file_path_generation(self, tmp_path: Path):
        """Test file path generation uses hash-based subdirectory."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        # Test internal path generation
        block_hash = bytes.fromhex("abc123def456" + "00" * 26)  # 32 bytes
        file_path = manager._get_file_path(block_hash)

        # First hex char of hash determines subdirectory
        assert file_path.parent.name == "a"
        assert file_path.suffix == ".safetensors"

    def test_enforce_size_limit(self, tmp_path: Path):
        """Test enforcing size limit."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        # Should return 0 when under limit
        freed = manager.enforce_size_limit()
        assert freed == 0


class TestPagedSSDCacheManagerWithMLX:
    """Tests for PagedSSDCacheManager that require MLX.

    These tests are skipped if MLX is not available.
    """

    @pytest.fixture
    def mock_mlx(self):
        """Mock MLX module for testing save/load without actual tensors."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    def test_save_and_load_block(self, tmp_path: Path, mock_mlx):
        """Test saving and loading a block with actual tensors."""
        mx = mock_mlx

        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        # Create test cache data
        block_hash = b"test_save_load_hash1"
        cache_data = [
            (mx.zeros((1, 8, 64, 64)), mx.zeros((1, 8, 64, 64)))
            for _ in range(4)  # 4 layers
        ]

        # Save
        result = manager.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=64,
            model_name="test-model",
            layer_cache_types=["KVCache"] * 4,
        )
        assert result is True
        assert manager.has_block(block_hash)

        # Load
        loaded = manager.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 4

        # Verify shapes
        for keys, values in loaded:
            assert keys.shape == (1, 8, 64, 64)
            assert values.shape == (1, 8, 64, 64)

    def test_load_block_with_metadata(self, tmp_path: Path, mock_mlx):
        """Test loading block with metadata."""
        mx = mock_mlx

        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        block_hash = b"test_load_meta_hash"
        cache_data = [
            (mx.zeros((1, 8, 64, 64)), mx.zeros((1, 8, 64, 64)))
            for _ in range(2)
        ]

        manager.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=64,
            model_name="test-model",
            layer_cache_types=["KVCache", "RotatingKVCache"],
            layer_meta_states=[(0,), (1, 256, 64, 0)],
        )

        # Load with metadata
        loaded_data, loaded_meta = manager.load_block_with_metadata(block_hash)

        assert loaded_data is not None
        assert loaded_meta is not None
        assert loaded_meta["num_layers"] == 2
        assert loaded_meta["token_count"] == 64
        assert loaded_meta["model_name"] == "test-model"
        assert loaded_meta["layer_cache_types"] == ["KVCache", "RotatingKVCache"]

    def test_get_block_metadata(self, tmp_path: Path, mock_mlx):
        """Test getting block metadata without loading data."""
        mx = mock_mlx

        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        block_hash = b"test_get_metadata_h"
        cache_data = [(mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))]

        manager.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=32,
            model_name="test-model",
        )

        metadata = manager.get_block_metadata(block_hash)

        assert metadata is not None
        assert metadata.block_hash == block_hash
        assert metadata.token_count == 32
        assert metadata.num_layers == 1
        assert metadata.model_name == "test-model"

    def test_save_existing_block_touches(self, tmp_path: Path, mock_mlx):
        """Test saving existing block just touches (updates LRU)."""
        mx = mock_mlx

        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        block_hash = b"test_touch_existing"
        cache_data = [(mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))]

        # First save
        manager.save_block(block_hash, cache_data, 32)
        initial_saves = manager._stats["saves"]

        # Second save (should just touch)
        manager.save_block(block_hash, cache_data, 32)

        # saves count should not increase (just hit)
        assert manager._stats["saves"] == initial_saves
        assert manager._stats["hits"] >= 1


class TestPagedSSDCacheManagerCacheList:
    """Tests for CacheList support in PagedSSDCacheManager."""

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def ssd_cache(self, tmp_path):
        """Create a PagedSSDCacheManager for testing."""
        return PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=100 * 1024**2,
        )

    def test_save_load_cache_list_block(self, ssd_cache, mx):
        """Test saving and loading a block with CacheList data."""
        block_hash = b"cache_list_test_hash"
        # Build cache_data with CacheList marker
        sub_keys1 = mx.zeros((1, 8, 32, 64))
        sub_values1 = mx.ones((1, 8, 32, 64))
        sub_keys2 = mx.zeros((1, 4, 32, 64))
        sub_values2 = mx.ones((1, 4, 32, 64))

        cache_data = [
            ('__cache_list__', [(sub_keys1, sub_values1), (sub_keys2, sub_values2)]),
            (mx.zeros((1, 8, 32, 64)), mx.ones((1, 8, 32, 64))),  # Standard KVCache layer
        ]

        layer_cache_types = ["CacheList", "KVCache"]

        result = ssd_cache.save_block(
            block_hash, cache_data, token_count=32,
            model_name="test", layer_cache_types=layer_cache_types,
        )
        assert result is True

        # Load back
        loaded = ssd_cache.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 2

        # First layer should be List[Tuple] (CacheList)
        assert isinstance(loaded[0], list)
        assert len(loaded[0]) == 2
        assert loaded[0][0][0].shape == (1, 8, 32, 64)
        assert loaded[0][1][0].shape == (1, 4, 32, 64)

        # Second layer should be tuple (KVCache)
        assert isinstance(loaded[1], tuple)
        assert loaded[1][0].shape == (1, 8, 32, 64)

    def test_save_load_cache_list_placeholder(self, ssd_cache, mx):
        """Test saving and loading placeholder CacheList block."""
        block_hash = b"placeholder_cl_hash_"
        # Non-last block: CacheList gets standard placeholder
        cache_data = [
            (mx.zeros((1,)), mx.zeros((1,))),  # CacheList placeholder
            (mx.zeros((1, 8, 32, 64)), mx.ones((1, 8, 32, 64))),  # KVCache
        ]

        layer_cache_types = ["CacheList", "KVCache"]

        result = ssd_cache.save_block(
            block_hash, cache_data, token_count=32,
            model_name="test", layer_cache_types=layer_cache_types,
        )
        assert result is True

        # Load back — CacheList placeholder loads as standard (keys, values) tuple
        loaded = ssd_cache.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 2
        # Placeholder has no sub_count, so loads as standard tuple
        assert isinstance(loaded[0], tuple)
        assert loaded[0][0].shape == (1,)

    def test_load_block_with_metadata_cache_list(self, ssd_cache, mx):
        """Test load_block_with_metadata for CacheList blocks."""
        block_hash = b"cl_metadata_test_ha_"
        sub_keys = mx.zeros((1, 8, 64, 64))
        sub_values = mx.ones((1, 8, 64, 64))

        cache_data = [
            ('__cache_list__', [(sub_keys, sub_values)]),
        ]
        layer_cache_types = ["CacheList"]
        layer_meta_states = [
            (["KVCache"], [(64,)]),  # CacheList meta_state format
        ]

        ssd_cache.save_block(
            block_hash, cache_data, token_count=64,
            model_name="test",
            layer_cache_types=layer_cache_types,
            layer_meta_states=layer_meta_states,
        )

        loaded_data, metadata = ssd_cache.load_block_with_metadata(block_hash)
        assert loaded_data is not None
        assert metadata is not None
        assert len(loaded_data) == 1
        assert isinstance(loaded_data[0], list)
        assert len(loaded_data[0]) == 1
        assert loaded_data[0][0][0].shape == (1, 8, 64, 64)
        assert metadata["layer_cache_types"] == ["CacheList"]

    def test_save_load_cache_list_with_zero_dim_values(self, ssd_cache, mx):
        """Test round-trip for CacheList where sub-cache has zero-dim values.

        This covers the deepseek_v32 / GLM-5 case where the DSA indexer
        sub-cache stores values with shape (B, 1, N, 0) — head_dim=0.
        """
        block_hash = b"zero_dim_cl_test_ha_"
        sub_keys1 = mx.zeros((1, 1, 64, 512))   # Main attention kv_latent
        sub_values1 = mx.zeros((1, 1, 64, 64))   # Main attention k_pe
        sub_keys2 = mx.zeros((1, 1, 64, 128))    # Indexer keys
        sub_values2 = mx.zeros((1, 1, 64, 0))    # Indexer values (zero head_dim)

        cache_data = [
            ('__cache_list__', [
                (sub_keys1, sub_values1),
                (sub_keys2, sub_values2),
            ]),
        ]
        layer_cache_types = ["CacheList"]

        result = ssd_cache.save_block(
            block_hash, cache_data, token_count=64,
            model_name="test", layer_cache_types=layer_cache_types,
        )
        assert result is True

        # Load back and verify round-trip correctness
        loaded = ssd_cache.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 1
        assert isinstance(loaded[0], list)
        assert len(loaded[0]) == 2

        # Sub-cache 0: normal tensors preserved
        assert loaded[0][0][0].shape == (1, 1, 64, 512)
        assert loaded[0][0][1].shape == (1, 1, 64, 64)

        # Sub-cache 1: keys normal, values zero-dim reconstructed
        assert loaded[0][1][0].shape == (1, 1, 64, 128)
        assert loaded[0][1][1].shape == (1, 1, 64, 0)

    def test_save_load_zero_dim_with_load_block_with_metadata(self, ssd_cache, mx):
        """Test load_block_with_metadata also handles zero-dim tensors."""
        block_hash = b"zero_dim_meta_test_h"
        sub_keys = mx.zeros((1, 1, 32, 128))
        sub_values = mx.zeros((1, 1, 32, 0))

        cache_data = [
            ('__cache_list__', [(sub_keys, sub_values)]),
        ]
        layer_cache_types = ["CacheList"]
        layer_meta_states = [
            (["KVCache"], [(32,)]),
        ]

        ssd_cache.save_block(
            block_hash, cache_data, token_count=32,
            model_name="test",
            layer_cache_types=layer_cache_types,
            layer_meta_states=layer_meta_states,
        )

        loaded_data, metadata = ssd_cache.load_block_with_metadata(block_hash)
        assert loaded_data is not None
        assert metadata is not None
        assert len(loaded_data) == 1
        assert isinstance(loaded_data[0], list)
        assert loaded_data[0][0][0].shape == (1, 1, 32, 128)
        assert loaded_data[0][0][1].shape == (1, 1, 32, 0)


class TestAsyncWriteAndTimeoutLoad:
    """Tests for the async write / timeout load deadlock fix.

    These tests verify:
    - save_block() returns immediately (non-blocking)
    - Pending writes are served on load (zero I/O)
    - Load timeout returns None (cache miss) instead of blocking
    - Writer thread errors clean up index entries
    - close() gracefully shuts down background threads
    """

    @pytest.fixture
    def mx(self):
        """Import MLX or skip."""
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def ssd_cache(self, tmp_path):
        """Create a PagedSSDCacheManager for testing."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=100 * 1024**2,
        )
        yield manager
        manager.close()

    def test_save_block_non_blocking(self, ssd_cache, mx, tmp_path):
        """Verify save_block() returns immediately and file appears async."""
        block_hash = b"async_save_test_hash"
        cache_data = [
            (mx.zeros((1, 8, 64, 64)), mx.zeros((1, 8, 64, 64)))
            for _ in range(4)
        ]

        t0 = time.time()
        result = ssd_cache.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=64,
            model_name="test-model",
            layer_cache_types=["KVCache"] * 4,
        )
        elapsed = time.time() - t0

        assert result is True
        # save_block should return almost instantly (< 1s),
        # not wait for disk I/O
        assert elapsed < 1.0

        # Block should be in index (optimistic update)
        assert ssd_cache.has_block(block_hash)

        # Wait for background writer to finish
        import time as time_mod
        for _ in range(50):  # Wait up to 5s
            file_path = ssd_cache._get_file_path(block_hash)
            if file_path.exists():
                break
            time_mod.sleep(0.1)

        assert file_path.exists(), "File should appear after background write"

    def test_pending_writes_served_on_load(self, ssd_cache, mx):
        """Verify that a block saved then immediately loaded is served from memory."""
        block_hash = b"pending_load_test_ha"
        cache_data = [
            (mx.zeros((1, 8, 32, 64)), mx.ones((1, 8, 32, 64)))
            for _ in range(2)
        ]

        ssd_cache.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=32,
            model_name="test-model",
            layer_cache_types=["KVCache", "KVCache"],
        )

        # Immediately load — should come from _pending_writes, not disk
        loaded = ssd_cache.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0][0].shape == (1, 8, 32, 64)
        assert loaded[0][1].shape == (1, 8, 32, 64)

    def test_pending_writes_served_on_load_with_metadata(self, ssd_cache, mx):
        """Verify load_block_with_metadata also reads from pending writes."""
        block_hash = b"pending_meta_test_ha"
        cache_data = [
            (mx.zeros((1, 4, 16, 32)), mx.zeros((1, 4, 16, 32)))
        ]

        ssd_cache.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=16,
            model_name="test-model",
            layer_cache_types=["KVCache"],
            layer_meta_states=[(16,)],
        )

        loaded_data, metadata = ssd_cache.load_block_with_metadata(block_hash)
        assert loaded_data is not None
        assert metadata is not None
        assert metadata["num_layers"] == 1
        assert metadata["token_count"] == 16
        assert metadata["model_name"] == "test-model"
        assert metadata["layer_cache_types"] == ["KVCache"]

    def test_load_error_returns_none(self, ssd_cache, mx):
        """Verify that a corrupted file returns None and cleans up index."""
        block_hash = b"error_test_hash_1234"
        cache_data = [
            (mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))
        ]

        # Save and wait for background write to complete
        ssd_cache.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=32,
        )
        import time as time_mod
        for _ in range(50):
            with ssd_cache._pending_writes_lock:
                if block_hash not in ssd_cache._pending_writes:
                    break
            time_mod.sleep(0.1)

        # Mock mx.load to simulate a corrupted file
        with patch("mlx.core.load", side_effect=OSError("corrupted file")):
            loaded = ssd_cache.load_block(block_hash)
            assert loaded is None  # Should return None, not raise

        # Block should be removed from index (corrupted entry cleanup)
        assert not ssd_cache.has_block(block_hash)

    def test_load_no_executor_deadlock(self, ssd_cache, mx):
        """Regression test: _load_executor must not exist (prevents deadlock)."""
        # The old implementation used ThreadPoolExecutor(max_workers=1) which
        # caused deadlocks when mx.load() in a worker thread contested Metal
        # GPU resources with the main inference thread. Verify it's gone.
        assert not hasattr(ssd_cache, '_load_executor'), (
            "_load_executor should not exist — it causes Metal GPU deadlocks"
        )

    def test_sequential_loads_no_queue_blocking(self, ssd_cache, mx):
        """Regression test: consecutive loads must not block each other."""
        import time as time_mod

        # Save 5 different blocks
        hashes = []
        for i in range(5):
            block_hash = f"seq_load_test_{i:04d}_".encode()[:20]
            hashes.append(block_hash)
            cache_data = [
                (mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))
            ]
            ssd_cache.save_block(block_hash, cache_data, token_count=32)

        # Wait for all pending writes to flush
        for _ in range(100):
            with ssd_cache._pending_writes_lock:
                if not ssd_cache._pending_writes:
                    break
            time_mod.sleep(0.1)

        # Load all 5 blocks sequentially — should complete quickly
        t0 = time_mod.time()
        for block_hash in hashes:
            loaded = ssd_cache.load_block(block_hash)
            assert loaded is not None, f"Failed to load {block_hash!r}"
            assert len(loaded) == 1
        elapsed = time_mod.time() - t0

        # 5 loads from SSD should complete in well under 5s
        # (each ~2ms read + reconstruction)
        assert elapsed < 5.0, (
            f"Sequential loads took {elapsed:.1f}s — possible queue blocking"
        )

    def test_writer_error_handling(self, ssd_cache, mx):
        """Verify that writer errors clean up the index."""
        block_hash = b"writer_error_test_ha"
        cache_data = [
            (mx.zeros((1, 4, 16, 32)), mx.zeros((1, 4, 16, 32)))
        ]

        # Temporarily break the save path by making the file path unwritable
        file_path = ssd_cache._get_file_path(block_hash)
        subdir = file_path.parent

        # Create a file where the directory should be (prevents writing)
        # Instead, we'll patch mx.save_safetensors to raise
        import time as time_mod

        with patch("mlx.core.save_safetensors", side_effect=OSError("Disk full")):
            result = ssd_cache.save_block(
                block_hash=block_hash,
                cache_data=cache_data,
                token_count=16,
            )
            # With temp-file approach, mx.save_safetensors runs on the main
            # thread in save_block. If it raises, save_block returns False
            # immediately and the block is never added to the index.
            assert result is False

        # Block should never have been added to the index
        assert not ssd_cache.has_block(block_hash)

    def test_graceful_shutdown(self, tmp_path, mx):
        """Verify close() stops the writer thread."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "shutdown_cache",
            max_size_bytes=100 * 1024**2,
        )

        # Save a block to ensure writer is active
        block_hash = b"shutdown_test_hash_1"
        cache_data = [(mx.zeros((1, 4, 16, 32)), mx.zeros((1, 4, 16, 32)))]
        manager.save_block(block_hash, cache_data, 16)

        # Close should stop the writer thread
        manager.close()

        assert not manager._writer_thread.is_alive()

    def test_save_existing_block_still_touches(self, ssd_cache, mx):
        """Verify saving an existing block just touches LRU (unchanged behavior)."""
        block_hash = b"touch_existing_test_"
        cache_data = [(mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))]

        ssd_cache.save_block(block_hash, cache_data, 32)
        initial_saves = ssd_cache._stats["saves"]

        # Second save should just touch, not re-enqueue
        ssd_cache.save_block(block_hash, cache_data, 32)
        assert ssd_cache._stats["saves"] == initial_saves
        assert ssd_cache._stats["hits"] >= 1

    def test_save_and_load_round_trip_after_flush(self, ssd_cache, mx):
        """Verify full round-trip: save -> flush -> load from disk."""
        import time as time_mod

        block_hash = b"round_trip_flush_tes"
        cache_data = [
            (mx.zeros((1, 8, 64, 64)), mx.ones((1, 8, 64, 64)))
            for _ in range(4)
        ]

        ssd_cache.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=64,
            model_name="test-model",
            layer_cache_types=["KVCache"] * 4,
        )

        # Wait for background write to complete
        for _ in range(50):
            with ssd_cache._pending_writes_lock:
                if block_hash not in ssd_cache._pending_writes:
                    break
            time_mod.sleep(0.1)

        # Now load should come from disk, not pending writes
        loaded = ssd_cache.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 4
        for keys, values in loaded:
            assert keys.shape == (1, 8, 64, 64)
            assert values.shape == (1, 8, 64, 64)
