# SPDX-License-Identifier: Apache-2.0
"""
Tests for PagedSSDCacheManager and related components.

This module tests SSD-based storage for paged KV cache blocks,
enabling larger effective cache sizes than GPU memory allows.
"""

import errno
import json
import logging
import queue
import shutil
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from omlx.cache.paged_ssd_cache import (
    PagedSSDBlockMetadata,
    PagedSSDCacheIndex,
    PagedSSDCacheManager,
    SharedHotCacheBudget,
    _cache_compat_signature,
    _extract_tensor_bytes,
    _restore_tensor_from_bytes,
    _write_safetensors_no_mx,
    parse_size,
)


def _has_mlx() -> bool:
    """Check if MLX is available."""
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


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
            block_size=2048,
            cache_signature="sig",
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
        assert d["block_size"] == 2048
        assert d["cache_signature"] == "sig"
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
            "block_size": 2048,
            "cache_signature": "sig",
            "layer_cache_types": ["KVCache", "RotatingKVCache"],
            "layer_meta_states": [[0], [1, 2, 3, 4]],
        }

        metadata = PagedSSDBlockMetadata.from_dict(d)

        assert metadata.block_hash == b"test_hash_bytes_1234"
        assert metadata.file_path == Path("/tmp/test.safetensors")
        assert metadata.file_size == 1024
        assert metadata.block_size == 2048
        assert metadata.cache_signature == "sig"
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
        assert metadata.block_size == 0
        assert metadata.cache_signature == ""
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
            expected_model_name="test-model",
            expected_block_size=64,
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
            (mx.zeros((1, 8, 64, 64)), mx.zeros((1, 8, 64, 64))) for _ in range(2)
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
        assert loaded_meta["block_size"] == 64
        assert loaded_meta["cache_signature"]
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

    def test_save_writes_format_version(self, tmp_path: Path, mock_mlx):
        """Saved blocks tag the file with the current format version."""
        import time as time_mod

        from omlx.cache.paged_ssd_cache import _CACHE_FORMAT_VERSION

        mx = mock_mlx

        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
        )

        block_hash = b"test_format_version_save"
        cache_data = [(mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))]
        assert manager.save_block(block_hash, cache_data, 32) is True

        # Wait for the background writer to flush the file to disk.
        file_path = manager._get_file_path(block_hash)
        for _ in range(50):
            if file_path.exists():
                break
            time_mod.sleep(0.1)
        assert file_path.exists(), "background writer never produced the file"

        _, file_metadata = mx.load(str(file_path), return_metadata=True)
        assert file_metadata.get("omlx_cache_format_version") == _CACHE_FORMAT_VERSION

    def test_unversioned_block_is_rejected_at_index_scan(
        self, tmp_path: Path, mock_mlx
    ):
        """Pre-fix blocks (no version marker) are skipped during scan.

        Older builds saved RotatingKVCache layers zero-padded to max_size.
        Loading those after the fix would leak zero positions into
        attention via BatchRotatingKVCache.merge(). Treat them as a cache
        miss by rejecting blocks without the format version.
        """
        mx = mock_mlx

        cache_dir = tmp_path / "ssd_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Hand-write a cache file without the version tag, mirroring what
        # an old-format save_block() would produce.
        block_hash = b"\x01" * 32
        block_hash_hex = block_hash.hex()
        # Match the manager's per-prefix subdirectory layout.
        sub_dir = cache_dir / block_hash_hex[:2]
        sub_dir.mkdir(parents=True, exist_ok=True)
        legacy_file = sub_dir / f"{block_hash_hex}.safetensors"

        mx.save_safetensors(
            str(legacy_file),
            {
                "layer_0_keys": mx.zeros((1, 8, 32, 64)),
                "layer_0_values": mx.zeros((1, 8, 32, 64)),
            },
            metadata={
                # Intentionally missing omlx_cache_format_version.
                "block_hash": block_hash_hex,
                "token_count": "32",
                "num_layers": "1",
                "model_name": "legacy-model",
                "created_at": "0",
            },
        )

        manager_after_scan = PagedSSDCacheManager(
            cache_dir=cache_dir,
            max_size_bytes=1024**3,
        )

        # Index scan ran in __init__. The legacy file should not appear.
        assert not manager_after_scan.has_block(block_hash)

    def _write_versioned_fixture_block(
        self,
        cache_dir: Path,
        mx,
        block_hash: bytes,
        *,
        num_layers: int,
        model_name: str,
        block_size: int = 256,
        layer_cache_types: list[str] | None = None,
    ) -> Path:
        """Drop a minimally-valid versioned block on disk so we can exercise
        the startup scan without relying on the background writer."""
        from omlx.cache.paged_ssd_cache import _CACHE_FORMAT_VERSION

        cache_dir.mkdir(parents=True, exist_ok=True)
        block_hash_hex = block_hash.hex()
        sub_dir = cache_dir / block_hash_hex[0]
        sub_dir.mkdir(parents=True, exist_ok=True)
        file_path = sub_dir / f"{block_hash_hex}.safetensors"

        tensors = {}
        for i in range(num_layers):
            tensors[f"layer_{i}_keys"] = mx.zeros((1, 8, 32, 64))
            tensors[f"layer_{i}_values"] = mx.zeros((1, 8, 32, 64))

        if layer_cache_types is None:
            layer_cache_types = ["KVCache"] * num_layers

        mx.save_safetensors(
            str(file_path),
            tensors,
            metadata={
                "omlx_cache_format_version": _CACHE_FORMAT_VERSION,
                "block_hash": block_hash_hex,
                "token_count": "32",
                "num_layers": str(num_layers),
                "model_name": model_name,
                "block_size": str(block_size),
                "cache_signature": _cache_compat_signature(
                    model_name=model_name,
                    num_layers=num_layers,
                    block_size=block_size,
                    layer_cache_types=layer_cache_types,
                ),
                "layer_cache_types": json.dumps(layer_cache_types),
                "created_at": "0",
            },
        )
        return file_path

    def test_scan_skips_layer_count_mismatch_without_unlinking(
        self, tmp_path: Path, mock_mlx
    ):
        """Blocks with num_layers != expected_num_layers are not indexed.

        Models that change their effective layer count across versions (e.g.,
        #1404 attaching MTPModule changed 30 -> 40) should not hit the
        layer-mismatch reject path on every prefix lookup. The file is still
        left on disk so shared cache directories are non-destructive.
        """
        mx = mock_mlx
        cache_dir = tmp_path / "ssd_cache"

        stale_hash = b"\x10" + b"\x00" * 31
        fresh_hash = b"\x20" + b"\x00" * 31
        stale_path = self._write_versioned_fixture_block(
            cache_dir, mx, stale_hash, num_layers=30, model_name="qwen3.6"
        )
        fresh_path = self._write_versioned_fixture_block(
            cache_dir, mx, fresh_hash, num_layers=40, model_name="qwen3.6"
        )

        manager = PagedSSDCacheManager(
            cache_dir=cache_dir,
            max_size_bytes=1024**3,
            expected_model_name="qwen3.6",
            expected_num_layers=40,
            expected_block_size=256,
        )

        assert stale_path.exists()
        assert fresh_path.exists()
        assert not manager.has_block(stale_hash)
        assert manager.has_block(fresh_hash)

    def test_scan_skips_model_name_mismatch_without_unlinking(
        self, tmp_path: Path, mock_mlx
    ):
        """Blocks from a different model stay on disk but are not indexed."""
        mx = mock_mlx
        cache_dir = tmp_path / "ssd_cache"

        other_hash = b"\x30" + b"\x00" * 31
        match_hash = b"\x40" + b"\x00" * 31
        other_path = self._write_versioned_fixture_block(
            cache_dir, mx, other_hash, num_layers=40, model_name="llama"
        )
        match_path = self._write_versioned_fixture_block(
            cache_dir, mx, match_hash, num_layers=40, model_name="qwen3.6"
        )

        manager = PagedSSDCacheManager(
            cache_dir=cache_dir,
            max_size_bytes=1024**3,
            expected_model_name="qwen3.6",
            expected_num_layers=40,
            expected_block_size=256,
        )

        assert other_path.exists()
        assert match_path.exists()
        assert not manager.has_block(other_hash)
        assert manager.has_block(match_hash)

    def test_scan_skips_block_size_mismatch_without_unlinking(
        self, tmp_path: Path, mock_mlx
    ):
        """Blocks with another paged cache block size are not indexed."""
        mx = mock_mlx
        cache_dir = tmp_path / "ssd_cache"

        wrong_hash = b"\x41" + b"\x00" * 31
        match_hash = b"\x42" + b"\x00" * 31
        wrong_path = self._write_versioned_fixture_block(
            cache_dir,
            mx,
            wrong_hash,
            num_layers=40,
            model_name="qwen3.6",
            block_size=2048,
        )
        match_path = self._write_versioned_fixture_block(
            cache_dir,
            mx,
            match_hash,
            num_layers=40,
            model_name="qwen3.6",
            block_size=256,
        )

        manager = PagedSSDCacheManager(
            cache_dir=cache_dir,
            max_size_bytes=1024**3,
            expected_model_name="qwen3.6",
            expected_num_layers=40,
            expected_block_size=256,
        )

        assert wrong_path.exists()
        assert match_path.exists()
        assert not manager.has_block(wrong_hash)
        assert manager.has_block(match_hash)

    def test_scan_keeps_blocks_when_expected_fields_unset(
        self, tmp_path: Path, mock_mlx
    ):
        """Backwards compatibility: callers that omit the new init args see
        no behavior change. All blocks survive scan regardless of metadata."""
        mx = mock_mlx
        cache_dir = tmp_path / "ssd_cache"

        h1 = b"\x50" + b"\x00" * 31
        h2 = b"\x60" + b"\x00" * 31
        p1 = self._write_versioned_fixture_block(
            cache_dir, mx, h1, num_layers=30, model_name="a"
        )
        p2 = self._write_versioned_fixture_block(
            cache_dir, mx, h2, num_layers=40, model_name="b"
        )

        manager = PagedSSDCacheManager(
            cache_dir=cache_dir,
            max_size_bytes=1024**3,
        )

        assert p1.exists()
        assert p2.exists()
        assert manager.has_block(h1)
        assert manager.has_block(h2)

    def test_scan_logs_skipped_incompatible_count(
        self, tmp_path: Path, mock_mlx, caplog
    ):
        """The completion log line surfaces incompatible blocks skipped at scan."""
        import logging

        mx = mock_mlx
        cache_dir = tmp_path / "ssd_cache"

        for i in range(3):
            self._write_versioned_fixture_block(
                cache_dir,
                mx,
                bytes([0x70 + i]) + b"\x00" * 31,
                num_layers=30,
                model_name="old",
            )

        with caplog.at_level(logging.INFO, logger="omlx.cache.paged_ssd_cache"):
            PagedSSDCacheManager(
                cache_dir=cache_dir,
                max_size_bytes=1024**3,
                expected_model_name="old",
                expected_num_layers=40,
                expected_block_size=256,
            )

        scan_lines = [
            r.message for r in caplog.records if "SSD cache scan complete" in r.message
        ]
        assert scan_lines, "scan completion log not emitted"
        assert "skipped_incompatible=3 blocks" in scan_lines[-1]


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
            ("__cache_list__", [(sub_keys1, sub_values1), (sub_keys2, sub_values2)]),
            (
                mx.zeros((1, 8, 32, 64)),
                mx.ones((1, 8, 32, 64)),
            ),  # Standard KVCache layer
        ]

        layer_cache_types = ["CacheList", "KVCache"]

        result = ssd_cache.save_block(
            block_hash,
            cache_data,
            token_count=32,
            model_name="test",
            layer_cache_types=layer_cache_types,
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
            block_hash,
            cache_data,
            token_count=32,
            model_name="test",
            layer_cache_types=layer_cache_types,
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
            ("__cache_list__", [(sub_keys, sub_values)]),
        ]
        layer_cache_types = ["CacheList"]
        layer_meta_states = [
            (["KVCache"], [(64,)]),  # CacheList meta_state format
        ]

        ssd_cache.save_block(
            block_hash,
            cache_data,
            token_count=64,
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
        sub_keys1 = mx.zeros((1, 1, 64, 512))  # Main attention kv_latent
        sub_values1 = mx.zeros((1, 1, 64, 64))  # Main attention k_pe
        sub_keys2 = mx.zeros((1, 1, 64, 128))  # Indexer keys
        sub_values2 = mx.zeros((1, 1, 64, 0))  # Indexer values (zero head_dim)

        cache_data = [
            (
                "__cache_list__",
                [
                    (sub_keys1, sub_values1),
                    (sub_keys2, sub_values2),
                ],
            ),
        ]
        layer_cache_types = ["CacheList"]

        result = ssd_cache.save_block(
            block_hash,
            cache_data,
            token_count=64,
            model_name="test",
            layer_cache_types=layer_cache_types,
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
            ("__cache_list__", [(sub_keys, sub_values)]),
        ]
        layer_cache_types = ["CacheList"]
        layer_meta_states = [
            (["KVCache"], [(32,)]),
        ]

        ssd_cache.save_block(
            block_hash,
            cache_data,
            token_count=32,
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
            (mx.zeros((1, 8, 64, 64)), mx.zeros((1, 8, 64, 64))) for _ in range(4)
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
            (mx.zeros((1, 8, 32, 64)), mx.ones((1, 8, 32, 64))) for _ in range(2)
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
        cache_data = [(mx.zeros((1, 4, 16, 32)), mx.zeros((1, 4, 16, 32)))]

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
        cache_data = [(mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))]

        # Save and wait for background write to complete
        ssd_cache.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=32,
        )
        import time as time_mod

        for _ in range(50):
            with ssd_cache._pending_write_hashes_lock:
                if block_hash not in ssd_cache._pending_write_hashes:
                    break
            time_mod.sleep(0.1)

        # Remove from hot cache buffer so load goes to disk
        ssd_cache._hot_cache_remove(block_hash)

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
        assert not hasattr(
            ssd_cache, "_load_executor"
        ), "_load_executor should not exist — it causes Metal GPU deadlocks"

    def test_sequential_loads_no_queue_blocking(self, ssd_cache, mx):
        """Regression test: consecutive loads must not block each other."""
        import time as time_mod

        # Save 5 different blocks
        hashes = []
        for i in range(5):
            block_hash = f"seq_load_test_{i:04d}_".encode()[:20]
            hashes.append(block_hash)
            cache_data = [(mx.zeros((1, 8, 32, 64)), mx.zeros((1, 8, 32, 64)))]
            ssd_cache.save_block(block_hash, cache_data, token_count=32)

        # Wait for all pending writes to flush
        for _ in range(100):
            with ssd_cache._pending_write_hashes_lock:
                if not ssd_cache._pending_write_hashes:
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
        assert (
            elapsed < 5.0
        ), f"Sequential loads took {elapsed:.1f}s — possible queue blocking"

    def test_writer_error_handling(self, ssd_cache, mx):
        """Verify that background writer errors clean up the index."""
        block_hash = b"writer_error_test_ha"
        cache_data = [(mx.zeros((1, 4, 16, 32)), mx.zeros((1, 4, 16, 32)))]

        # Patch _write_safetensors_no_mx to simulate disk error in background writer
        import time as time_mod

        with patch(
            "omlx.cache.paged_ssd_cache._write_safetensors_no_mx",
            side_effect=OSError("Disk full"),
        ):
            result = ssd_cache.save_block(
                block_hash=block_hash,
                cache_data=cache_data,
                token_count=16,
            )
            # save_block() succeeds (bytes extracted, queued for background write)
            assert result is True

            # Wait for background writer to process and fail
            for _ in range(50):
                if ssd_cache._write_queue.empty():
                    break
                time_mod.sleep(0.05)
            time_mod.sleep(0.1)

        # Background writer should have removed the block from index on error
        assert not ssd_cache.has_block(block_hash)
        # And from pending write hashes
        with ssd_cache._pending_write_hashes_lock:
            assert block_hash not in ssd_cache._pending_write_hashes

    def test_writer_enospc_logs_disk_full(self, ssd_cache, mx, caplog):
        """ENOSPC errors should log 'disk full' warning, not generic error."""
        block_hash = b"enospc_test_hash_123"
        cache_data = [(mx.zeros((1, 4, 16, 32)), mx.zeros((1, 4, 16, 32)))]

        enospc = OSError("No space left on device")
        enospc.errno = errno.ENOSPC

        import time as time_mod

        with (
            patch(
                "omlx.cache.paged_ssd_cache._write_safetensors_no_mx",
                side_effect=enospc,
            ),
            caplog.at_level(logging.WARNING),
        ):
            ssd_cache.save_block(
                block_hash=block_hash,
                cache_data=cache_data,
                token_count=16,
            )
            for _ in range(50):
                if ssd_cache._write_queue.empty():
                    break
                time_mod.sleep(0.05)
            time_mod.sleep(0.1)

        assert "SSD cache disk full" in caplog.text

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
            (mx.zeros((1, 8, 64, 64)), mx.ones((1, 8, 64, 64))) for _ in range(4)
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
            with ssd_cache._pending_write_hashes_lock:
                if block_hash not in ssd_cache._pending_write_hashes:
                    break
            time_mod.sleep(0.1)

        # Remove from hot cache buffer so load goes to disk
        ssd_cache._hot_cache_remove(block_hash)

        # Now load should come from disk, not pending writes
        loaded = ssd_cache.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 4
        for keys, values in loaded:
            assert keys.shape == (1, 8, 64, 64)
            assert values.shape == (1, 8, 64, 64)


# =============================================================================
# Async Background Write Tests
# =============================================================================


@pytest.mark.skipif(not _has_mlx(), reason="MLX not available")
class TestAsyncBackgroundWrite:
    """Tests for the async background write pipeline (no-mx safetensors)."""

    @pytest.fixture
    def mx(self):
        import mlx.core as mx

        return mx

    def test_extract_and_restore_float32(self, mx):
        """Round-trip test for float32 tensors."""
        original = mx.random.normal((2, 4, 8))
        mx.eval(original)
        raw, dtype_str, shape = _extract_tensor_bytes(original)
        assert dtype_str == "F32"
        assert shape == [2, 4, 8]
        restored = _restore_tensor_from_bytes(raw, dtype_str, shape)
        assert restored.dtype == mx.float32
        assert restored.shape == (2, 4, 8)
        assert mx.allclose(original, restored).item()

    def test_extract_and_restore_float16(self, mx):
        """Round-trip test for float16 tensors."""
        original = mx.random.normal((3, 5)).astype(mx.float16)
        mx.eval(original)
        raw, dtype_str, shape = _extract_tensor_bytes(original)
        assert dtype_str == "F16"
        restored = _restore_tensor_from_bytes(raw, dtype_str, shape)
        assert restored.dtype == mx.float16
        assert mx.allclose(original, restored).item()

    def test_extract_and_restore_bfloat16(self, mx):
        """Round-trip test for bfloat16 tensors (the key dtype for this feature)."""
        original = mx.random.normal((4, 8, 16)).astype(mx.bfloat16)
        mx.eval(original)
        raw, dtype_str, shape = _extract_tensor_bytes(original)
        assert dtype_str == "BF16"
        assert shape == [4, 8, 16]
        restored = _restore_tensor_from_bytes(raw, dtype_str, shape)
        assert restored.dtype == mx.bfloat16
        assert restored.shape == (4, 8, 16)
        # Compare as float32 to avoid bfloat16 precision issues
        assert mx.allclose(
            original.astype(mx.float32), restored.astype(mx.float32)
        ).item()

    def test_extract_and_restore_int_types(self, mx):
        """Round-trip test for integer dtypes."""
        for mx_dtype, st_str in [
            (mx.int8, "I8"),
            (mx.int32, "I32"),
            (mx.uint8, "U8"),
        ]:
            original = mx.array([1, 2, 3, 4], dtype=mx_dtype)
            mx.eval(original)
            raw, dtype_str, shape = _extract_tensor_bytes(original)
            assert dtype_str == st_str
            restored = _restore_tensor_from_bytes(raw, dtype_str, shape)
            assert restored.dtype == mx_dtype
            assert mx.array_equal(original, restored).item()

    def test_extract_materializes_lazy_slice(self, mx):
        """_extract_tensor_bytes handles lazy block slices."""
        base = mx.arange(1 * 2 * 16 * 4, dtype=mx.float32).reshape(1, 2, 16, 4)
        mx.eval(base)
        lazy_slice = base[:, :, 3:11, :]

        raw, dtype_str, shape = _extract_tensor_bytes(lazy_slice)

        restored = _restore_tensor_from_bytes(raw, dtype_str, shape)
        expected = base[:, :, 3:11, :]
        mx.eval(expected)
        assert dtype_str == "F32"
        assert shape == [1, 2, 8, 4]
        assert mx.allclose(expected, restored).item()

    def test_extract_materializes_lazy_bfloat16_slice(self, mx):
        """_extract_tensor_bytes handles lazy bf16 slices and uint16 views."""
        base = mx.arange(1 * 2 * 12 * 4, dtype=mx.float32).reshape(1, 2, 12, 4)
        base = base.astype(mx.bfloat16)
        mx.eval(base)
        lazy_slice = base[:, :, 2:10, :]

        raw, dtype_str, shape = _extract_tensor_bytes(lazy_slice)

        restored = _restore_tensor_from_bytes(raw, dtype_str, shape)
        expected = base[:, :, 2:10, :]
        mx.eval(expected)
        assert dtype_str == "BF16"
        assert shape == [1, 2, 8, 4]
        assert restored.dtype == mx.bfloat16
        assert mx.allclose(
            expected.astype(mx.float32), restored.astype(mx.float32)
        ).item()

    def test_extract_materializes_lazy_clone(self, mx):
        """_extract_tensor_bytes handles block-like lazy clone/copy tensors."""
        base = mx.arange(1 * 2 * 16 * 4, dtype=mx.float32).reshape(1, 2, 16, 4)
        mx.eval(base)
        tensor = base[:, :, 4:12, :]
        if hasattr(mx, "copy"):
            cloned = mx.copy(tensor)
        elif hasattr(tensor, "copy"):
            cloned = tensor.copy()
        else:
            cloned = mx.array(tensor)

        raw, dtype_str, shape = _extract_tensor_bytes(cloned)

        restored = _restore_tensor_from_bytes(raw, dtype_str, shape)
        expected = base[:, :, 4:12, :]
        mx.eval(expected)
        assert dtype_str == "F32"
        assert shape == [1, 2, 8, 4]
        assert mx.allclose(expected, restored).item()

    def test_write_safetensors_no_mx_roundtrip(self, mx, tmp_path):
        """Write safetensors without mx API, then load with mx.load()."""
        t1 = mx.random.normal((2, 3, 4))
        t2 = mx.ones((5,), dtype=mx.float16)
        mx.eval(t1, t2)

        tensors_raw = {
            "tensor_a": _extract_tensor_bytes(t1),
            "tensor_b": _extract_tensor_bytes(t2),
        }
        metadata = {"test_key": "test_value", "block_hash": "abc123"}

        out_path = str(tmp_path / "test.safetensors")
        file_size = _write_safetensors_no_mx(out_path, tensors_raw, metadata)
        assert file_size > 0

        # Load with mx.load and verify
        loaded_arrays, loaded_meta = mx.load(out_path, return_metadata=True)
        assert "tensor_a" in loaded_arrays
        assert "tensor_b" in loaded_arrays
        assert loaded_meta["test_key"] == "test_value"
        assert loaded_meta["block_hash"] == "abc123"
        assert mx.allclose(t1, loaded_arrays["tensor_a"]).item()
        assert mx.allclose(t2, loaded_arrays["tensor_b"]).item()

    def test_write_safetensors_bfloat16_roundtrip(self, mx, tmp_path):
        """Verify bfloat16 safetensors file is loadable by mx.load."""
        original = mx.random.normal((8, 16, 32)).astype(mx.bfloat16)
        mx.eval(original)

        tensors_raw = {"kv_cache": _extract_tensor_bytes(original)}
        out_path = str(tmp_path / "bf16_test.safetensors")
        _write_safetensors_no_mx(out_path, tensors_raw)

        loaded, _ = mx.load(out_path, return_metadata=True)
        assert loaded["kv_cache"].dtype == mx.bfloat16
        assert loaded["kv_cache"].shape == (8, 16, 32)
        assert mx.allclose(
            original.astype(mx.float32),
            loaded["kv_cache"].astype(mx.float32),
        ).item()

    def test_save_block_uses_background_write(self, tmp_path, mx):
        """Verify save_block enqueues bytes for background writer (no mx.save_safetensors)."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "async_test",
            max_size_bytes=100 * 1024**2,
        )

        block_hash = b"async_write_test_hsh"
        cache_data = [(mx.ones((1, 4, 16, 32)), mx.zeros((1, 4, 16, 32)))]

        # Patch mx.save_safetensors to ensure it's NOT called
        with patch("mlx.core.save_safetensors") as mock_save:
            result = manager.save_block(
                block_hash=block_hash,
                cache_data=cache_data,
                token_count=16,
            )
            assert result is True
            # mx.save_safetensors should NOT be called (we use _write_safetensors_no_mx)
            mock_save.assert_not_called()

        # Hot cache buffer should store tensors_raw (bytes), not arrays (mx.array)
        with manager._hot_cache_lock:
            pending = manager._hot_cache.get(block_hash)
        assert pending is not None
        assert "tensors_raw" in pending
        assert "arrays" not in pending  # Old key should not exist

        # Wait for background write and verify file exists
        for _ in range(50):
            file_path = manager._get_file_path(block_hash)
            if file_path.exists():
                break
            time.sleep(0.05)
        assert file_path.exists()

        # Verify file is loadable by mx.load. V3 stores state elements as
        # ``layer_{i}_state_{k}`` keys with a ``layer_{i}_state_count`` meta
        # entry, polyfilled from V2 ``(keys, values)`` 2-tuples on save.
        loaded, meta = mx.load(str(file_path), return_metadata=True)
        assert "layer_0_state_0" in loaded
        assert "layer_0_state_1" in loaded
        assert meta.get("layer_0_state_count") == "2"
        assert meta["block_hash"] == block_hash.hex()

        manager.close()

    def test_pending_writes_bytes_readback(self, tmp_path, mx):
        """Verify load_block can restore mx.arrays from bytes-based pending_writes."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "readback_test",
            max_size_bytes=100 * 1024**2,
        )

        block_hash = b"readback_test_hash__"
        original_keys = mx.random.normal((1, 8, 32, 64))
        original_values = mx.random.normal((1, 8, 32, 64))
        mx.eval(original_keys, original_values)
        cache_data = [(original_keys, original_values)]

        manager.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=32,
        )

        # Load immediately from pending_writes (before background write completes)
        loaded = manager.load_block(block_hash)
        assert loaded is not None
        assert len(loaded) == 1
        keys, values = loaded[0]
        assert mx.allclose(original_keys, keys).item()
        assert mx.allclose(original_values, values).item()

        manager.close()

    def test_index_update_file_size(self):
        """Verify PagedSSDCacheIndex.update_file_size works correctly."""
        index = PagedSSDCacheIndex(max_size_bytes=1000)
        block_hash = b"size_update_test____"
        metadata = PagedSSDBlockMetadata(
            block_hash=block_hash,
            file_path=Path("/tmp/test.safetensors"),
            file_size=100,
            token_count=16,
            created_at=time.time(),
            last_access=time.time(),
            num_layers=1,
        )
        index.add(metadata)
        assert index.total_size == 100

        # Update to actual size
        index.update_file_size(block_hash, 150)
        assert index.total_size == 150

        # Non-existent hash should be no-op
        index.update_file_size(b"nonexistent_hash____", 999)
        assert index.total_size == 150


class TestEffectiveMaxSize:
    """Tests for dynamic effective max size based on disk free space."""

    def _make_disk_usage(self, total: int, used: int, free: int):
        """Create a mock disk_usage result."""
        return shutil._ntuple_diskusage(total, used, free)

    def test_effective_max_size_disk_sufficient(self, tmp_path: Path):
        """When disk has plenty of free space, effective = configured max."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=100 * 1024**3,  # 100GB configured
        )

        # Mock: 500GB free, cache is empty (0 bytes)
        mock_usage = self._make_disk_usage(
            total=1000 * 1024**3, used=500 * 1024**3, free=500 * 1024**3
        )
        with patch("shutil.disk_usage", return_value=mock_usage):
            effective = manager._get_effective_max_size()

        # disk_available = 0 + 500GB = 500GB, disk_limit = 495GB
        # effective = min(100GB, 495GB) = 100GB
        assert effective == 100 * 1024**3

    def test_effective_max_size_disk_low(self, tmp_path: Path):
        """When disk is low, effective shrinks below configured max."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=110 * 1024**3,  # 110GB configured
        )

        # Simulate: cache currently has 10GB, disk free is 90GB
        # So disk_available = 10GB + 90GB = 100GB
        manager._index._total_size = 10 * 1024**3

        mock_usage = self._make_disk_usage(
            total=500 * 1024**3, used=410 * 1024**3, free=90 * 1024**3
        )
        with patch("shutil.disk_usage", return_value=mock_usage):
            effective = manager._get_effective_max_size()

        # disk_limit = int(100GB * 0.99) = 99GB
        # effective = min(110GB, 99GB) = 99GB
        expected = int(100 * 1024**3 * 0.99)
        assert effective == expected

    def test_effective_max_size_oserror_fallback(self, tmp_path: Path):
        """When disk_usage fails, fall back to configured max."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=50 * 1024**3,
        )

        with patch("shutil.disk_usage", side_effect=OSError("disk error")):
            effective = manager._get_effective_max_size()

        assert effective == 50 * 1024**3

    def test_effective_max_size_cache_30s(self, tmp_path: Path):
        """disk_usage result is cached for 30 seconds."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=100 * 1024**3,
        )

        mock_usage = self._make_disk_usage(
            total=1000 * 1024**3, used=500 * 1024**3, free=500 * 1024**3
        )
        with patch("shutil.disk_usage", return_value=mock_usage) as mock_du:
            # First call — should invoke disk_usage
            manager._get_effective_max_size()
            assert mock_du.call_count == 1

            # Second call within 30s — should use cache
            manager._get_effective_max_size()
            assert mock_du.call_count == 1

            # Expire cache by rewinding timestamp
            manager._disk_usage_cache_time -= 31.0

            # Third call — should invoke disk_usage again
            manager._get_effective_max_size()
            assert mock_du.call_count == 2

    def test_utilization_never_exceeds_1(self, tmp_path: Path):
        """Utilization should never exceed 1.0 with effective max size."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=100 * 1024**3,
        )

        # Simulate: cache has 50GB, but disk only has 10GB free
        # So disk_available = 50GB + 10GB = 60GB, disk_limit = ~59.4GB
        manager._index._total_size = 50 * 1024**3

        mock_usage = self._make_disk_usage(
            total=200 * 1024**3, used=190 * 1024**3, free=10 * 1024**3
        )
        with patch("shutil.disk_usage", return_value=mock_usage):
            stats = manager.get_stats_dict()

        assert stats["utilization"] <= 1.0
        assert stats["max_size"] < stats["configured_max_size"]

    def test_stats_includes_effective_and_configured(self, tmp_path: Path):
        """Stats should include both effective and configured max sizes."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=100 * 1024**3,
        )

        mock_usage = self._make_disk_usage(
            total=500 * 1024**3, used=450 * 1024**3, free=50 * 1024**3
        )
        with patch("shutil.disk_usage", return_value=mock_usage):
            stats_dict = manager.get_stats_dict()
            stats_obj = manager.get_stats()

        # Dict format
        assert "configured_max_size" in stats_dict
        assert stats_dict["configured_max_size"] == 100 * 1024**3

        # Dataclass format
        assert stats_obj.configured_max_size_bytes == 100 * 1024**3
        assert stats_obj.max_size_bytes > 0

    def test_max_size_property_returns_effective(self, tmp_path: Path):
        """max_size property should return effective (not configured) value."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=200 * 1024**3,
        )

        # disk_available = 0 + 50GB = 50GB, disk_limit = ~49.5GB
        mock_usage = self._make_disk_usage(
            total=500 * 1024**3, used=450 * 1024**3, free=50 * 1024**3
        )
        with patch("shutil.disk_usage", return_value=mock_usage):
            assert manager.max_size < 200 * 1024**3
            assert manager.configured_max_size == 200 * 1024**3

    def test_oserror_fallback_logs_warning(self, tmp_path: Path, caplog):
        """disk_usage failure should log a warning, not fail silently."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=50 * 1024**3,
        )
        # Expire cache so next call hits disk_usage
        manager._disk_usage_cache_time -= 31.0

        with (
            patch("shutil.disk_usage", side_effect=OSError("mount gone")),
            caplog.at_level(logging.WARNING),
        ):
            effective = manager._get_effective_max_size()

        assert effective == 50 * 1024**3
        assert "Failed to check disk usage" in caplog.text

    def test_disk_pressure_warning(self, tmp_path: Path, caplog):
        """Warn when effective max drops below 10% of configured max."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=100 * 1024**3,
        )

        # Simulate nearly full disk: only 5GB free, cache has 0 bytes
        mock_usage = self._make_disk_usage(
            total=500 * 1024**3, used=495 * 1024**3, free=5 * 1024**3
        )
        with (
            patch("shutil.disk_usage", return_value=mock_usage),
            caplog.at_level(logging.WARNING),
        ):
            manager._enforce_size_limit_for_new_block()

        assert "disk pressure" in caplog.text
        assert "disk nearly full" in caplog.text


class TestPreloadMatchedBlocks:
    """Tests for parallel block preloading into hot cache."""

    @pytest.fixture
    def mx(self):
        try:
            import mlx.core as mx

            return mx
        except ImportError:
            pytest.skip("MLX not available")

    @pytest.fixture
    def manager_with_hot_cache(self, tmp_path, mx):
        """Create a manager with hot cache enabled."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=512 * 1024**2,
        )
        yield manager
        manager.close()

    def _save_test_blocks(
        self,
        manager,
        mx,
        count=4,
        layers=2,
        hot_cache_max_bytes=512 * 1024**2,
        hot_cache_budget=None,
    ):
        """Save test blocks and flush them to SSD (not hot cache)."""
        hashes = []
        for i in range(count):
            block_hash = f"preload_test_block_{i:04d}".encode()
            cache_data = [
                (
                    mx.zeros((1, 4, 64, 64)),
                    mx.zeros((1, 4, 64, 64)),
                )
                for _ in range(layers)
            ]
            manager.save_block(
                block_hash=block_hash,
                cache_data=cache_data,
                token_count=64,
                model_name="test-model",
                layer_cache_types=["KVCache"] * layers,
            )
            hashes.append(block_hash)

        # Flush writer to ensure blocks are on SSD
        manager.close()

        # Re-open manager (cold start — hot cache is empty)
        new_manager = PagedSSDCacheManager(
            cache_dir=manager._cache_dir,
            max_size_bytes=1024**3,
            hot_cache_max_bytes=hot_cache_max_bytes,
            hot_cache_budget=hot_cache_budget,
        )
        return new_manager, hashes

    def test_preload_promotes_to_hot_cache(self, tmp_path, mx):
        """After preload, blocks are found in hot cache."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=512 * 1024**2,
        )
        manager2, hashes = self._save_test_blocks(manager, mx, count=4)

        # Verify blocks are NOT in hot cache before preload
        for h in hashes:
            assert manager2._hot_cache_get(h) is None

        # Preload
        loaded = manager2.preload_matched_blocks(hashes)
        assert loaded == 4

        # Verify blocks ARE in hot cache after preload
        for h in hashes:
            assert manager2._hot_cache_get(h) is not None

        manager2.close()

    def test_clean_promoted_block_eviction_skips_ssd_write(self, tmp_path, mx):
        """Blocks loaded from SSD are clean and should not be re-written."""
        entry_size = 2 * 2 * 1 * 4 * 64 * 64 * 4
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=512 * 1024**2,
        )
        manager2, hashes = self._save_test_blocks(
            manager,
            mx,
            count=2,
            hot_cache_max_bytes=entry_size + 1024,
        )

        try:
            assert manager2.load_block(hashes[0]) is not None
            first_entry = manager2._hot_cache_get(hashes[0])
            assert first_entry is not None
            assert first_entry["dirty"] is False

            with patch.object(manager2, "_enqueue_ssd_write") as enqueue_write:
                assert manager2.load_block(hashes[1]) is not None
                enqueue_write.assert_not_called()

            assert manager2._hot_cache_get(hashes[0]) is None
            assert manager2._hot_cache_get(hashes[1]) is not None
        finally:
            manager2.close()

    def test_preload_partial_failure(self, tmp_path, mx):
        """If one block file is missing, others still load."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=512 * 1024**2,
        )
        manager2, hashes = self._save_test_blocks(manager, mx, count=5)

        # Delete one block file from SSD to simulate failure
        metadata = manager2._index.get(hashes[1])
        metadata.file_path.unlink()

        loaded = manager2.preload_matched_blocks(hashes)

        # 4 of 5 should succeed (1 deleted)
        assert loaded == 4
        assert manager2._hot_cache_get(hashes[0]) is not None
        assert manager2._hot_cache_get(hashes[1]) is None  # deleted file
        assert manager2._hot_cache_get(hashes[2]) is not None

        manager2.close()

    def test_preload_skips_hot_cache_blocks(self, tmp_path, mx):
        """Blocks already in hot cache are not re-loaded."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=512 * 1024**2,
        )
        manager2, hashes = self._save_test_blocks(manager, mx, count=5)

        # Load one block into hot cache manually
        manager2.load_block(hashes[0])
        assert manager2._hot_cache_get(hashes[0]) is not None
        promotions_before = manager2._stats["hot_cache_promotions"]

        # Preload all — should only load the 4 cold blocks
        loaded = manager2.preload_matched_blocks(hashes)
        assert loaded == 4

        # Promotion count should increase by exactly 4 (not 5)
        assert manager2._stats["hot_cache_promotions"] == promotions_before + 4

        manager2.close()

    def test_preload_unknown_hashes_ignored(self, tmp_path, mx):
        """Hashes not in the SSD index are silently skipped."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=512 * 1024**2,
        )
        manager2, hashes = self._save_test_blocks(manager, mx, count=5)

        all_hashes = hashes + [b"nonexistent_hash_01", b"nonexistent_hash_02"]
        loaded = manager2.preload_matched_blocks(all_hashes)
        assert loaded == 5  # only the real blocks

        manager2.close()

    def test_preload_noop_without_hot_cache(self, tmp_path, mx):
        """Preload returns 0 when hot cache is disabled."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=0,  # hot cache disabled
        )
        block_hash = b"preload_no_hot_test"
        cache_data = [(mx.zeros((1, 4, 32, 64)), mx.zeros((1, 4, 32, 64)))]
        manager.save_block(block_hash, cache_data, 32, layer_cache_types=["KVCache"])
        manager.close()

        manager2 = PagedSSDCacheManager(
            cache_dir=manager._cache_dir,
            max_size_bytes=1024**3,
            hot_cache_max_bytes=0,
        )
        loaded = manager2.preload_matched_blocks([block_hash])
        assert loaded == 0

        manager2.close()

    def test_preload_skips_when_hot_cache_full(self, tmp_path, mx):
        """Preload returns 0 when hot cache has no remaining capacity."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=1024,  # tiny hot cache
        )
        manager2, hashes = self._save_test_blocks(manager, mx, count=2)

        # Fill hot cache to capacity
        manager2._hot_cache_total_bytes = manager2._hot_cache_max_bytes

        loaded = manager2.preload_matched_blocks(hashes)
        assert loaded == 0

        manager2.close()

    def test_preload_skips_when_shared_hot_cache_budget_full(self, tmp_path, mx):
        """Preload uses remaining shared budget, not only local hot cache bytes."""
        budget = SharedHotCacheBudget(1024)
        filler = PagedSSDCacheManager(
            cache_dir=tmp_path / "budget_filler",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=budget.max_bytes,
            hot_cache_only=True,
            hot_cache_budget=budget,
        )
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=0,
        )
        manager2 = None

        try:
            filler._hot_cache_put(
                b"preload_budget_filler",
                {
                    "tensors_raw": {
                        "layer_0_keys": (bytes(1024), "uint8", [1024]),
                    },
                    "file_metadata": {},
                    "num_layers": 1,
                    "layer_cache_types": ["KVCache"],
                    "block_metadata": None,
                },
            )
            assert budget.remaining_bytes == 0

            manager2, hashes = self._save_test_blocks(
                manager,
                mx,
                count=4,
                hot_cache_max_bytes=budget.max_bytes,
                hot_cache_budget=budget,
            )

            loaded = manager2.preload_matched_blocks(hashes)
            assert loaded == 0
            assert budget.total_bytes == budget.max_bytes
            for h in hashes:
                assert manager2._hot_cache_get(h) is None
        finally:
            if manager2 is not None:
                manager2.close()
            else:
                manager.close()
            filler.close()

    def test_preload_empty_list(self, manager_with_hot_cache):
        """Empty hash list returns 0 immediately."""
        loaded = manager_with_hot_cache.preload_matched_blocks([])
        assert loaded == 0

    def test_preload_skips_below_threshold(self, tmp_path, mx):
        """Preload skips when fewer than 4 cold blocks need loading."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=512 * 1024**2,
        )
        manager2, hashes = self._save_test_blocks(manager, mx, count=3)

        loaded = manager2.preload_matched_blocks(hashes)
        assert loaded == 0
        for bh in hashes:
            assert manager2._hot_cache_get(bh) is None

        manager2.close()

    def test_preload_updates_stats(self, tmp_path, mx):
        """Preload increments preload-specific stats counters."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=512 * 1024**2,
        )
        manager2, hashes = self._save_test_blocks(manager, mx, count=4)

        manager2.preload_matched_blocks(hashes)

        assert manager2._stats["preload_blocks_loaded"] == 4
        assert manager2._stats["preload_calls"] == 1
        assert manager2._stats["preload_time_ms"] > 0

        manager2.close()

    def test_preloaded_blocks_load_correctly(self, tmp_path, mx):
        """After preload, load_block returns correct data from hot cache."""
        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=512 * 1024**2,
        )
        manager2, hashes = self._save_test_blocks(manager, mx, count=5, layers=3)

        # Preload blocks into hot cache
        manager2.preload_matched_blocks(hashes)

        # Now load_block should hit hot cache
        hot_hits_before = manager2._stats["hot_cache_hits"]
        for h in hashes:
            data = manager2.load_block(h)
            assert data is not None
            assert len(data) == 3  # 3 layers
            for keys, values in data:
                assert keys.shape == (1, 4, 64, 64)
                assert values.shape == (1, 4, 64, 64)

        # All loads should be hot cache hits (not SSD reads)
        assert manager2._stats["hot_cache_hits"] == hot_hits_before + 5

        manager2.close()

    def test_concurrent_preload_and_load(self, tmp_path, mx):
        """Preload and load_block don't race on hot cache."""
        import threading

        manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=512 * 1024**2,
        )
        manager2, hashes = self._save_test_blocks(manager, mx, count=8)

        results = {"preload": None, "loads": []}
        errors = []

        def do_preload():
            try:
                results["preload"] = manager2.preload_matched_blocks(hashes)
            except Exception as e:
                errors.append(f"preload: {e}")

        def do_loads():
            try:
                for h in hashes:
                    data = manager2.load_block(h)
                    results["loads"].append(data is not None)
            except Exception as e:
                errors.append(f"load: {e}")

        t1 = threading.Thread(target=do_preload)
        t2 = threading.Thread(target=do_loads)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"Concurrent errors: {errors}"
        assert not t1.is_alive(), "Preload thread hung"
        assert not t2.is_alive(), "Load thread hung"

        manager2.close()


class TestPreloadBlocks:
    """Tests for BlockAwarePrefixCache.preload_blocks()."""

    @pytest.fixture
    def mx(self):
        try:
            import mlx.core as mx

            return mx
        except ImportError:
            pytest.skip("MLX not available")

    def test_preload_blocks_calls_ssd_preload(self, tmp_path, mx):
        """preload_blocks extracts hashes from BlockTable and calls SSD preload."""
        from unittest.mock import MagicMock

        from omlx.cache.paged_cache import PagedCacheManager

        # Set up real SSD manager with blocks
        ssd_manager = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=256 * 1024**2,
        )

        hashes = []
        for i in range(5):
            bh = f"preload_blocks_test_{i:04d}".encode()
            cache_data = [(mx.zeros((1, 4, 32, 64)), mx.zeros((1, 4, 32, 64)))]
            ssd_manager.save_block(bh, cache_data, 32, layer_cache_types=["KVCache"])
            hashes.append(bh)
        ssd_manager.close()

        # Re-open cold
        ssd_manager2 = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1024**3,
            hot_cache_max_bytes=256 * 1024**2,
        )

        # Set up paged cache with allocated blocks
        paged_cache = PagedCacheManager(block_size=256, max_blocks=100)
        block_ids = []
        for bh in hashes:
            block = paged_cache.allocate_block()
            block.block_hash = bh
            block.token_count = 32
            block_ids.append(block.block_id)

        # Create BlockAwarePrefixCache
        from omlx.cache.prefix_cache import BlockAwarePrefixCache, BlockTable

        model = MagicMock()
        prefix_cache = BlockAwarePrefixCache(model, paged_cache, ssd_manager2)

        # Create a BlockTable
        bt = BlockTable(
            request_id="test-req",
            block_ids=block_ids,
            num_tokens=32 * len(block_ids),
        )

        # Call preload_blocks
        loaded = prefix_cache.preload_blocks(bt)
        assert loaded == 5

        # Verify blocks are in hot cache
        for bh in hashes:
            assert ssd_manager2._hot_cache_get(bh) is not None

        ssd_manager2.close()


class TestComputeMaxPendingWrites:
    """Pin the block-size-aware cap formula.

    The pending-writes queue holds raw KV bytes that the background
    SSD writer hasn't drained yet — so the cap must scale by bytes per
    slot, not just host RAM. Cap should shrink when each block is
    expensive (large block_size or fat KV) and grow when each block is
    cheap, while the hard byte budget bounds the soft floor.
    """

    def test_soft_floor_applies_when_hard_budget_allows(self):
        """The soft floor gives large-block workloads burst headroom
        when the byte hard cap still has room."""
        from omlx.cache.paged_ssd_cache import _compute_max_pending_writes

        def fake_sysconf(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            if name == "SC_PHYS_PAGES":
                return (64 * 1024**3) // 4096
            raise ValueError(name)

        with patch("omlx.cache.paged_ssd_cache.os.sysconf", side_effect=fake_sysconf):
            cap = _compute_max_pending_writes(
                block_size_tokens=2048,
                kv_bytes_per_token=200_000,
            )
        assert cap == 32

    def test_hard_budget_bounds_soft_floor(self):
        """The soft floor must not force the pending pool above 30%
        of host RAM for very expensive blocks."""
        from omlx.cache.paged_ssd_cache import _compute_max_pending_writes

        def fake_sysconf(name):
            if name == "SC_PAGE_SIZE":
                return 4096
            if name == "SC_PHYS_PAGES":
                return (64 * 1024**3) // 4096
            raise ValueError(name)

        with patch("omlx.cache.paged_ssd_cache.os.sysconf", side_effect=fake_sysconf):
            cap = _compute_max_pending_writes(
                block_size_tokens=2048,
                kv_bytes_per_token=500_000,
            )
        assert cap == 20

    def test_ceiling_clamps_high_end(self):
        """A 512 GB host with tiny blocks still tops out at 256
        rather than pinning thousands of in-flight writes."""
        from omlx.cache.paged_ssd_cache import _compute_max_pending_writes

        # Tiny per-slot cost → math would produce a huge number;
        # the ceiling must hold at 256.
        cap = _compute_max_pending_writes(
            block_size_tokens=1,
            kv_bytes_per_token=1,
        )
        assert cap == 256

    def test_larger_block_shrinks_cap(self):
        """Doubling block_size_tokens halves the cap target (until
        floor/ceiling clamp)."""
        from omlx.cache.paged_ssd_cache import _compute_max_pending_writes

        cap_256 = _compute_max_pending_writes(block_size_tokens=256)
        cap_1024 = _compute_max_pending_writes(block_size_tokens=1024)

        # 4× block size → ¼ the target. Both should be in the
        # in-band range (above floor, below ceiling) for typical
        # Macs; if the smaller one is at the ceiling the test
        # degenerates but the ordering still holds.
        assert cap_1024 <= cap_256, (
            "Larger blocks must yield smaller cap so worst-case "
            "pinned bytes stay bounded"
        )

    def test_larger_kv_bytes_shrinks_cap(self):
        """Heavier per-token KV (bigger model) shrinks the cap so
        the byte budget is preserved across model sizes."""
        from omlx.cache.paged_ssd_cache import _compute_max_pending_writes

        cap_small = _compute_max_pending_writes(kv_bytes_per_token=50_000)
        cap_large = _compute_max_pending_writes(kv_bytes_per_token=400_000)

        assert cap_large <= cap_small

    def test_default_args_produce_sensible_cap(self):
        """The default-args path used by static callers must produce
        a cap inside the bounded range."""
        from omlx.cache.paged_ssd_cache import _compute_max_pending_writes

        cap = _compute_max_pending_writes()
        assert 1 <= cap <= 256

    def test_manager_picks_up_per_instance_cap(self, tmp_path):
        """The PagedSSDCacheManager must recompute the cap from its
        constructor args, not just inherit the module-level constant.
        Otherwise a non-default block size silently uses a cap sized
        for the default block size."""
        from omlx.cache.paged_ssd_cache import (
            PagedSSDCacheManager,
            _compute_max_pending_writes,
        )

        mgr_small = PagedSSDCacheManager(
            cache_dir=tmp_path / "small",
            max_size_bytes=1 << 30,
            expected_block_size_tokens=256,
            expected_kv_bytes_per_token=200_000,
        )
        mgr_large = PagedSSDCacheManager(
            cache_dir=tmp_path / "large",
            max_size_bytes=1 << 30,
            expected_block_size_tokens=2048,
            expected_kv_bytes_per_token=200_000,
        )

        # Large blocks must produce a cap no larger than small blocks.
        # Both should equal what the standalone function computes for
        # the same inputs (no surprise rescaling inside __init__).
        assert mgr_small._max_pending_writes == _compute_max_pending_writes(
            block_size_tokens=256, kv_bytes_per_token=200_000
        )
        assert mgr_large._max_pending_writes == _compute_max_pending_writes(
            block_size_tokens=2048, kv_bytes_per_token=200_000
        )
        assert mgr_large._max_pending_writes <= mgr_small._max_pending_writes

        # The write queue must use the instance cap, not the module
        # default — otherwise the formula change is ineffective.
        assert mgr_small._write_queue.maxsize == mgr_small._max_pending_writes
        assert mgr_large._write_queue.maxsize == mgr_large._max_pending_writes

        mgr_small.close()
        mgr_large.close()


class TestSchedulerPlumbsBlockSizeToSSDCache:
    """The Scheduler construction path must plumb its final
    ``paged_cache_block_size`` and a model-derived KV bytes-per-token
    estimate into ``PagedSSDCacheManager.__init__`` — otherwise the
    block-size-aware queue cap silently falls back to defaults sized
    for the wrong workload.

    Regression guard for jundot's review on PR #1627:

      > PagedSSDCacheManager now accepts ``expected_block_size_tokens``
      > / ``expected_kv_bytes_per_token``, but the real scheduler
      > construction path still does not pass them.
    """

    def _make_scheduler(self, tmp_path, block_size_tokens, model_layers):
        """Build a Scheduler with paged SSD cache enabled at the given
        block size and a model whose config exposes ``model_layers``
        layers (so the memory monitor produces a real per-token KV
        estimate rather than its default)."""
        from unittest.mock import MagicMock

        from omlx.scheduler import Scheduler, SchedulerConfig

        # Real numbers so MemoryMonitor.set_model_info accepts them.
        class _Config:
            num_hidden_layers = model_layers
            num_key_value_heads = 8
            num_attention_heads = 32
            head_dim = 192
            hidden_size = 6144

        model = MagicMock()
        model.layers = []
        model.config = _Config()

        tokenizer = MagicMock()
        tokenizer.eos_token_id = 2

        config = SchedulerConfig(
            max_num_seqs=4,
            prefill_step_size=2048,
            paged_cache_block_size=block_size_tokens,
            paged_ssd_cache_dir=str(tmp_path),
            paged_ssd_cache_max_size=1 << 30,
            hot_cache_max_size=0,
            model_name="test-model",
        )
        return Scheduler(model=model, tokenizer=tokenizer, config=config)

    def test_block_size_plumbed_to_ssd_manager(self, tmp_path):
        """Final scheduler block size must reach the SSD cache manager
        — not the 256-token default."""
        model_layers = 32
        sched = self._make_scheduler(
            tmp_path / "blk1024",
            block_size_tokens=1024,
            model_layers=model_layers,
        )

        mgr = sched.paged_ssd_cache_manager

        # The manager stashes the constructor inputs as
        # ``_expected_block_size_tokens`` / ``_expected_kv_bytes_per_token``
        # so the plumbing-regression check is bulletproof regardless
        # of the test host's RAM tier (where the cap math might clamp
        # both default and plumbed inputs to the same floor/ceiling).
        assert mgr._expected_block_size_tokens == 1024, (
            "Scheduler must plumb its final paged_cache_block_size "
            "into PagedSSDCacheManager"
        )

        # The auto-init in ``Scheduler.__init__`` (paired with
        # ``_set_model_info_for_monitor()``) must populate dims from
        # the fixture's model.config so the writer-queue cap formula
        # weighs a real per-token cost. If either step regresses the
        # SSD manager would silently fall back to either the monitor's
        # ~128 KB 7B-class fiction or the manager's own 200 KB default
        # — both miscalibrate the cap on the real model. Assert the
        # construction order didn't regress.
        assert sched.memory_monitor is not None, (
            "Scheduler.__init__ must auto-init MemoryMonitor before "
            "_init_tiered_cache; if this regresses the SSD manager "
            "silently uses its 200 KB default and the cap math is "
            "miscalibrated on every workload"
        )
        assert sched.memory_monitor.has_model_info(), (
            "_set_model_info_for_monitor must populate dims from "
            "model.config BEFORE _init_tiered_cache constructs the "
            "PagedSSDCacheManager; otherwise estimate_block_memory(1) "
            "returns its 7B-class default fiction"
        )

        # The model-derived per-token KV is what should reach the
        # manager. Compute it inline from the fixture's known dims so
        # this assertion can't be satisfied by a tautological pass-
        # through of whatever ``estimate_block_memory`` happens to
        # return — and so it doesn't equal the 200 KB manager default,
        # making a silent fallback to the default visible as a numeric
        # diff in the failure message.
        per_token_bytes = (
            1  # block_size token
            * 8  # num_kv_heads in _make_scheduler fixture
            * 192  # head_dim
            * 2  # dtype_size (float16)
            * 2  # keys + values
        ) * model_layers
        assert mgr._expected_kv_bytes_per_token == per_token_bytes, (
            f"PagedSSDCacheManager received "
            f"expected_kv_bytes_per_token={mgr._expected_kv_bytes_per_token} "
            f"but the model-derived value is {per_token_bytes}; "
            f"a value of 200000 indicates a silent fallback to the "
            f"manager default"
        )
        # And the value the manager received must match
        # ``memory_monitor.estimate_block_memory(1)`` — the documented
        # source. Keep both checks: the inline calc above catches
        # tautological pass-throughs, this one catches drift between
        # the monitor's calc and the scheduler's wiring.
        assert mgr._expected_kv_bytes_per_token == (
            sched.memory_monitor.estimate_block_memory(1)
        ), (
            "Scheduler must plumb memory_monitor.estimate_block_memory"
            "(1) as expected_kv_bytes_per_token"
        )

        # And the cap computed from those plumbed inputs must drive the
        # write queue's maxsize — the cap is only useful if the queue
        # actually enforces it.
        from omlx.cache.paged_ssd_cache import _compute_max_pending_writes

        expected_cap = _compute_max_pending_writes(
            block_size_tokens=mgr._expected_block_size_tokens,
            kv_bytes_per_token=mgr._expected_kv_bytes_per_token,
        )
        assert mgr._max_pending_writes == expected_cap
        assert mgr._write_queue.maxsize == mgr._max_pending_writes

        mgr.close()

    def test_larger_block_size_shrinks_scheduler_cap(self, tmp_path):
        """The same model on the same Mac with a bigger block size
        must produce a smaller pending-writes cap — proves the wiring
        is not stuck at the default for every scheduler instance."""
        sched_small = self._make_scheduler(
            tmp_path / "blk256",
            block_size_tokens=256,
            model_layers=32,
        )
        sched_large = self._make_scheduler(
            tmp_path / "blk2048",
            block_size_tokens=2048,
            model_layers=32,
        )

        cap_small = sched_small.paged_ssd_cache_manager._max_pending_writes
        cap_large = sched_large.paged_ssd_cache_manager._max_pending_writes
        assert cap_large <= cap_small, (
            f"Larger blocks must shrink the cap; got "
            f"cap(2048)={cap_large} > cap(256)={cap_small}"
        )

        sched_small.paged_ssd_cache_manager.close()
        sched_large.paged_ssd_cache_manager.close()

    def test_factory_path_also_plumbs_block_size(self, tmp_path):
        """The cache factory path (used by direct/test callers) must
        also pass ``config.block_size`` through — otherwise direct
        callers silently get the 256-token default regardless of what
        they configured."""
        from omlx.cache.factory import CacheConfig, CacheFactory

        cfg = CacheConfig(
            block_size=1024,
            paged_ssd_cache_dir=tmp_path,
            max_paged_ssd_cache_size=1 << 30,
        )
        mgr = CacheFactory.create_paged_ssd_cache(cfg, model_name="m")
        assert mgr is not None

        from omlx.cache.paged_ssd_cache import _compute_max_pending_writes

        # Factory has no memory monitor, so it leaves
        # ``kv_bytes_per_token`` at the 200 KB default — that's the
        # documented contract for non-Scheduler callers.
        expected = _compute_max_pending_writes(
            block_size_tokens=1024, kv_bytes_per_token=200_000
        )
        assert mgr._max_pending_writes == expected
        mgr.close()


class TestInlineLRUUnlinks:
    """LRU eviction must unlink inline on the calling thread, not enqueue
    ``("unlink", path)`` tasks onto ``_write_queue``.

    The original async-queued design routed eviction unlinks through the
    same bounded queue that carries pending writes. Under sustained save
    pressure, the queue saturated, ``save_block``'s pre-eviction
    ``_write_queue.full()`` short-circuit fired before eviction could
    run, and the cache stayed permanently full once the queue saturated.
    Inlining removes the bounded-queue contention.
    """

    @pytest.fixture
    def mx(self):
        try:
            import mlx.core as mx

            return mx
        except ImportError:
            pytest.skip("MLX not available")

    def _entry_size(self, num_layers=2, seq_len=16, heads=2, head_dim=16):
        # 2 tensors (K+V) per layer, batch=1, float32=4
        return num_layers * 2 * 1 * heads * seq_len * head_dim * 4

    def _save_block(self, mgr, mx, block_hash, num_layers=2):
        cache_data = [
            (mx.zeros((1, 2, 16, 16)), mx.zeros((1, 2, 16, 16)))
            for _ in range(num_layers)
        ]
        return mgr.save_block(
            block_hash=block_hash,
            cache_data=cache_data,
            token_count=16,
            model_name="test-model",
            layer_cache_types=["KVCache"] * num_layers,
        )

    def test_save_waits_on_full_queue_before_inline_fallback(self, tmp_path, mx):
        """A full queue must not short-circuit before tensor extraction.

        The bounded wait gives the writer a chance to drain transient bursts;
        sustained saturation writes the block inline.
        """
        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "full_queue_wait",
            max_size_bytes=1 << 30,
        )
        original_put = mgr._write_queue.put
        original_full = mgr._write_queue.full
        calls: list[float | None] = []

        def fake_put(item, timeout=None, *args, **kwargs):
            calls.append(timeout)
            raise queue.Full

        try:
            mgr._write_queue.full = lambda: True  # type: ignore[method-assign]
            mgr._write_queue.put = fake_put  # type: ignore[method-assign]

            block_hash = b"full_queue_wait"
            assert self._save_block(mgr, mx, block_hash) is True

            assert calls == [1.0]
            stats = mgr.get_stats()
            assert stats.ssd_write_drops == 0
            assert stats.ssd_inline_write_fallbacks == 1
            assert stats.saves == 1
            assert stats.saves_persisted == 1
            assert mgr.load_block(block_hash) is not None
        finally:
            mgr._write_queue.put = original_put  # type: ignore[method-assign]
            mgr._write_queue.full = original_full  # type: ignore[method-assign]
            mgr.close()

    def test_eviction_does_not_enqueue_unlink_tasks(self, tmp_path, mx):
        """Force eviction; assert no ``("unlink", ...)`` items ever enter
        ``_write_queue``. Regression for the original async-queued
        design."""
        entry_size = self._entry_size()
        # Room for ~2 entries; the third save forces eviction of the first.
        max_bytes = entry_size * 2 + 100

        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "inline_eviction",
            max_size_bytes=max_bytes,
        )

        # Sentinel: intercept put_nowait and reject any unlink-shaped tuple.
        original_put_nowait = mgr._write_queue.put_nowait
        unlink_attempts: list = []

        def guard_put_nowait(item):
            if isinstance(item, tuple) and item and item[0] == "unlink":
                unlink_attempts.append(item)
            return original_put_nowait(item)

        mgr._write_queue.put_nowait = guard_put_nowait  # type: ignore[assignment]

        try:
            for i in range(5):
                self._save_block(mgr, mx, f"inline_evict_{i:04d}".encode())

            assert unlink_attempts == [], (
                "Eviction must unlink inline, not enqueue. Found queued "
                f"unlink attempts: {unlink_attempts!r}"
            )
        finally:
            mgr.close()

    def test_eviction_frees_capacity_under_pressure(self, tmp_path, mx):
        """Even when the writer thread is paused (mimicking the
        saturation scenario), eviction must keep the index size within
        the configured cap."""
        entry_size = self._entry_size()
        max_bytes = entry_size * 2 + 100  # holds exactly 2 entries

        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "inline_pressure",
            max_size_bytes=max_bytes,
        )
        try:
            # Save more entries than the cap allows; eviction must keep
            # the index within ``max_bytes``.
            for i in range(8):
                self._save_block(mgr, mx, f"pressure_{i:04d}".encode())

            # Wait briefly for in-flight writes to settle so the index
            # accounting reflects post-eviction state.
            time.sleep(0.05)

            assert mgr._index.total_size <= max_bytes + entry_size, (
                f"Eviction failed to keep total_size ({mgr._index.total_size}) "
                f"near cap ({max_bytes})"
            )
        finally:
            mgr.close()

    def test_inline_eviction_burst_is_capped(self, tmp_path, mx):
        """A large forced eviction is bounded by
        ``_MAX_INLINE_UNLINKS_PER_SAVE``; deferred entries reinsert into
        the index so subsequent saves drain the remainder."""
        from omlx.cache.paged_ssd_cache import _MAX_INLINE_UNLINKS_PER_SAVE

        # Use a large cap initially, then shrink to force a mass-eviction.
        entry_size = self._entry_size()
        n_entries = _MAX_INLINE_UNLINKS_PER_SAVE + 16
        initial_max = entry_size * (n_entries + 2)

        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "inline_burst",
            max_size_bytes=initial_max,
        )
        try:
            for i in range(n_entries):
                self._save_block(mgr, mx, f"burst_{i:04d}".encode())

            # Wait for writes to flush so file_size in the index matches
            # what's on disk.
            time.sleep(0.05)
            count_before = mgr._index.count

            # Shrink the effective cap dramatically. Next eviction must
            # cap its inline burst at _MAX_INLINE_UNLINKS_PER_SAVE.
            mgr._max_size = entry_size  # cap at 1 entry

            # Trigger eviction via a fresh save.
            self._save_block(mgr, mx, b"burst_trigger___")

            # After one save, the index should have shed at most
            # _MAX_INLINE_UNLINKS_PER_SAVE entries. The rest must have
            # been reinserted so subsequent saves can drain them.
            time.sleep(0.05)
            count_after = mgr._index.count
            removed = count_before + 1 - count_after  # +1 for the new save
            assert removed <= _MAX_INLINE_UNLINKS_PER_SAVE, (
                f"Inline burst removed {removed} entries (cap "
                f"{_MAX_INLINE_UNLINKS_PER_SAVE}); ENOSPC-storm protection "
                f"is not in effect"
            )
            assert removed > 0, (
                "No entries were evicted despite the new save crossing "
                "the (shrunken) cap"
            )
        finally:
            mgr.close()

    def test_deferred_eviction_preserves_lru_order(self, tmp_path, monkeypatch):
        """Deferred eviction entries remain older than survivor entries."""
        from omlx.cache import paged_ssd_cache as ssd_cache_module

        monkeypatch.setattr(ssd_cache_module, "_MAX_INLINE_UNLINKS_PER_SAVE", 2)

        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "deferred_lru_order",
            max_size_bytes=1024**2 + 20,
            hot_cache_only=True,
        )
        try:
            for i in range(6):
                block_hash = f"lru_{i}".encode()
                mgr._index.add(
                    PagedSSDBlockMetadata(
                        block_hash=block_hash,
                        file_path=tmp_path / f"{block_hash.hex()}.safetensors",
                        file_size=10,
                        token_count=1,
                        created_at=float(i),
                        last_access=float(i),
                        num_layers=1,
                    )
                )

            mgr._get_effective_max_size = (  # type: ignore[method-assign]
                lambda: 1024**2 + 20
            )
            mgr._enforce_size_limit_for_new_block()

            remaining_lru = [
                metadata.block_hash
                for metadata in mgr._index.get_lru_entries(mgr._index.count)
            ]
            assert remaining_lru == [b"lru_2", b"lru_3", b"lru_4", b"lru_5"]
        finally:
            mgr.close()

    def test_unlink_failure_increments_counter(self, tmp_path, mx):
        """When ``Path.unlink`` raises ``OSError``, the eviction loop
        records the failure in ``evict_unlink_failures`` instead of
        silently dropping the signal."""
        entry_size = self._entry_size()
        max_bytes = entry_size + 100

        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "unlink_fail",
            max_size_bytes=max_bytes,
        )
        try:
            self._save_block(mgr, mx, b"unlink_fail_0001")
            time.sleep(0.05)

            # Patch unlink to raise OSError on the next eviction attempt.
            from pathlib import Path as _Path

            def boom_unlink(self, *args, **kwargs):
                raise OSError("simulated unlink failure")

            with patch.object(_Path, "unlink", boom_unlink):
                # Save a second block; eviction of the first triggers the
                # patched unlink.
                self._save_block(mgr, mx, b"unlink_fail_0002")
                time.sleep(0.05)

            assert mgr._stats["evict_unlink_failures"] >= 1
        finally:
            mgr.close()


class TestSharedHotCacheBudgetClearAllOwners:
    """clear_all_owners reaches managers the budget still pins (orphaned)."""

    class _FakeOwner:
        def __init__(self, budget, n):
            self._budget = budget
            self._n = n
            self.cleared = False

        def clear_hot_cache(self):
            self._budget.forget_owner(self)
            self.cleared = True
            return self._n

    def test_clears_every_tracked_owner(self):
        budget = SharedHotCacheBudget(1 << 20)
        o1 = self._FakeOwner(budget, 3)
        o2 = self._FakeOwner(budget, 4)
        budget.put(o1, b"h1", 100)
        budget.put(o1, b"h2", 100)
        budget.put(o2, b"h3", 200)

        cleared = budget.clear_all_owners()

        assert cleared == 7
        assert o1.cleared and o2.cleared
        assert len(budget._entries) == 0

    def test_empty_budget_is_noop(self):
        assert SharedHotCacheBudget(1 << 20).clear_all_owners() == 0


class TestLayerSignatureSweep:
    """Tests for ``adopt_layer_signature_if_unset`` /
    ``invalidate_stale_layer_signature``.

    The bug: blocks saved before a cache-config change (e.g., TurboQuant
    toggle) carry stale ``layer_cache_types`` metadata, and the
    per-block compatibility check in ``reconstruct_cache`` uses block 0
    as the reference — so a stale block 0 defines the reference forever
    and every newer correctly-typed block trips the mismatch.
    """

    def _make_meta(
        self,
        *,
        block_hash: bytes,
        model_name: str,
        layer_cache_types: list[str] | None,
        num_layers: int = 40,
        block_size: int = 2048,
    ) -> PagedSSDBlockMetadata:
        now = time.time()
        return PagedSSDBlockMetadata(
            block_hash=block_hash,
            file_path=Path("/tmp/never-touched.safetensors"),
            file_size=1024,
            token_count=block_size,
            created_at=now,
            last_access=now,
            num_layers=num_layers,
            model_name=model_name,
            block_size=block_size,
            layer_cache_types=layer_cache_types,
        )

    def _make_manager(
        self,
        tmp_path: Path,
        *,
        model_name: str = "test-model",
        expected_layer_cache_types: list[str] | None = None,
    ) -> PagedSSDCacheManager:
        return PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1 << 30,
            expected_model_name=model_name,
            expected_num_layers=40,
            expected_block_size=2048,
            expected_layer_cache_types=expected_layer_cache_types,
        )

    def test_adopt_sets_signature_first_call(self, tmp_path: Path):
        mgr = self._make_manager(tmp_path)
        assert mgr._expected_layer_cache_types is None

        sig = ["ArraysCache", "ArraysCache", "ArraysCache", "TurboQuantKVCache"]
        adopted = mgr.adopt_layer_signature_if_unset(sig)

        assert adopted is True
        assert mgr._expected_layer_cache_types == sig

    def test_adopt_noop_when_already_set(self, tmp_path: Path):
        original = ["ArraysCache", "KVCache"]
        mgr = self._make_manager(tmp_path, expected_layer_cache_types=original)

        adopted = mgr.adopt_layer_signature_if_unset(
            ["ArraysCache", "TurboQuantKVCache"]
        )

        assert adopted is False
        assert mgr._expected_layer_cache_types == original

    def test_adopt_noop_when_empty(self, tmp_path: Path):
        mgr = self._make_manager(tmp_path)
        assert mgr.adopt_layer_signature_if_unset(None) is False
        assert mgr.adopt_layer_signature_if_unset([]) is False
        assert mgr._expected_layer_cache_types is None

    def test_sweep_drops_blocks_with_stale_signature(self, tmp_path: Path):
        stock = ["ArraysCache", "ArraysCache", "ArraysCache", "KVCache"]
        turbo = [
            "ArraysCache",
            "ArraysCache",
            "ArraysCache",
            "TurboQuantKVCache",
        ]
        mgr = self._make_manager(tmp_path, expected_layer_cache_types=turbo)

        mgr._index.add(
            self._make_meta(
                block_hash=b"01" * 10,
                model_name="test-model",
                layer_cache_types=stock,
                num_layers=4,
            )
        )
        mgr._index.add(
            self._make_meta(
                block_hash=b"02" * 10,
                model_name="test-model",
                layer_cache_types=turbo,
                num_layers=4,
            )
        )

        dropped = mgr.invalidate_stale_layer_signature()

        assert dropped == 1
        assert mgr._index.get(b"01" * 10) is None
        assert mgr._index.get(b"02" * 10) is not None

    def test_sweep_canonicalizes_sized_arrays_cache(self, tmp_path: Path):
        # SizedArraysCache and ArraysCache must compare equal — both are
        # the same on-disk format. The sweep must NOT drop blocks just
        # because one side uses the wrapper class name.
        a = ["ArraysCache", "ArraysCache", "KVCache"]
        b = ["SizedArraysCache", "SizedArraysCache", "KVCache"]
        mgr = self._make_manager(tmp_path, expected_layer_cache_types=a)
        mgr._index.add(
            self._make_meta(
                block_hash=b"aa" * 10,
                model_name="test-model",
                layer_cache_types=b,
                num_layers=3,
            )
        )

        dropped = mgr.invalidate_stale_layer_signature()

        assert dropped == 0
        assert mgr._index.get(b"aa" * 10) is not None

    def test_sweep_leaves_other_models_alone(self, tmp_path: Path):
        turbo = ["ArraysCache", "TurboQuantKVCache"]
        stock = ["ArraysCache", "KVCache"]
        mgr = self._make_manager(
            tmp_path, model_name="model-A", expected_layer_cache_types=turbo
        )

        # Other-model block with the "wrong" signature for THIS manager
        # but perfectly valid for model-B.
        mgr._index.add(
            self._make_meta(
                block_hash=b"bb" * 10,
                model_name="model-B",
                layer_cache_types=stock,
                num_layers=2,
            )
        )
        # Same-model stale block.
        mgr._index.add(
            self._make_meta(
                block_hash=b"cc" * 10,
                model_name="model-A",
                layer_cache_types=stock,
                num_layers=2,
            )
        )

        dropped = mgr.invalidate_stale_layer_signature()

        assert dropped == 1
        assert mgr._index.get(b"bb" * 10) is not None, (
            "model-B block must not be touched by model-A's sweep"
        )
        assert mgr._index.get(b"cc" * 10) is None

    def test_sweep_skips_legacy_unnamed_blocks(self, tmp_path: Path):
        turbo = ["ArraysCache", "TurboQuantKVCache"]
        mgr = self._make_manager(
            tmp_path, model_name="model-A", expected_layer_cache_types=turbo
        )
        # No model_name — cannot safely attribute.
        mgr._index.add(
            self._make_meta(
                block_hash=b"dd" * 10,
                model_name="",
                layer_cache_types=["ArraysCache", "KVCache"],
                num_layers=2,
            )
        )

        dropped = mgr.invalidate_stale_layer_signature()

        assert dropped == 0
        assert mgr._index.get(b"dd" * 10) is not None

    def test_sweep_skips_blocks_without_layer_metadata(self, tmp_path: Path):
        turbo = ["ArraysCache", "TurboQuantKVCache"]
        mgr = self._make_manager(
            tmp_path, model_name="model-A", expected_layer_cache_types=turbo
        )
        # Pre-signature block: no layer_cache_types recorded.
        mgr._index.add(
            self._make_meta(
                block_hash=b"ee" * 10,
                model_name="model-A",
                layer_cache_types=None,
                num_layers=2,
            )
        )

        dropped = mgr.invalidate_stale_layer_signature()

        assert dropped == 0
        assert mgr._index.get(b"ee" * 10) is not None

    def test_sweep_idempotent(self, tmp_path: Path):
        turbo = ["ArraysCache", "TurboQuantKVCache"]
        mgr = self._make_manager(
            tmp_path, model_name="model-A", expected_layer_cache_types=turbo
        )
        mgr._index.add(
            self._make_meta(
                block_hash=b"ff" * 10,
                model_name="model-A",
                layer_cache_types=["ArraysCache", "KVCache"],
                num_layers=2,
            )
        )

        assert mgr.invalidate_stale_layer_signature() == 1
        # Subsequent calls do nothing — flag is set.
        assert mgr.invalidate_stale_layer_signature() == 0

    def test_sweep_noop_without_signature(self, tmp_path: Path):
        mgr = self._make_manager(tmp_path)  # signature unset
        mgr._index.add(
            self._make_meta(
                block_hash=b"99" * 10,
                model_name="test-model",
                layer_cache_types=["ArraysCache", "KVCache"],
                num_layers=2,
            )
        )
        assert mgr.invalidate_stale_layer_signature() == 0
        assert mgr._index.get(b"99" * 10) is not None

    def test_sweep_noop_without_model_name(self, tmp_path: Path):
        # Without an expected_model_name we cannot scope safely.
        mgr = PagedSSDCacheManager(
            cache_dir=tmp_path / "ssd_cache",
            max_size_bytes=1 << 30,
            expected_model_name="",  # explicit unset
            expected_layer_cache_types=["ArraysCache", "KVCache"],
        )
        mgr._index.add(
            self._make_meta(
                block_hash=b"77" * 10,
                model_name="some-model",
                layer_cache_types=["ArraysCache", "TurboQuantKVCache"],
                num_layers=2,
            )
        )
        assert mgr.invalidate_stale_layer_signature() == 0
        assert mgr._index.get(b"77" * 10) is not None

    def test_adopt_resets_sweep_flag(self, tmp_path: Path):
        # If a previous signature got adopted-and-swept, but somehow a new
        # signature gets adopted later (currently the manager treats this
        # as a no-op; we still want the flag to reset if it ever does).
        mgr = self._make_manager(tmp_path)
        mgr.adopt_layer_signature_if_unset(["ArraysCache", "KVCache"])
        mgr._signature_sweep_completed = True
        # Same call signature — already set, returns False, flag stays.
        assert mgr.adopt_layer_signature_if_unset(["ArraysCache", "KVCache"]) is False
        assert mgr._signature_sweep_completed is True
