# SPDX-License-Identifier: Apache-2.0
"""
Paged SSD Cache Manager for oMLX KV cache.

This module implements SSD-based storage for paged KV cache blocks,
enabling larger effective cache sizes than GPU memory allows.

Key features:
- Block-level safetensors serialization (compatible with mlx-lm)
- Hash-based subdirectory structure for scalability
- LRU-based paged SSD cache size management
- Startup scan to reuse existing cache files

Reference: mlx-lm/mlx_lm/models/cache.py (save_prompt_cache, load_prompt_cache)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import shutil
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from omlx.utils.formatting import format_bytes
from .interface import CacheManager
from .stats import BaseCacheStats, PagedSSDCacheStats

logger = logging.getLogger(__name__)

# Check for MLX
try:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None


# --- Async I/O constants ---
def _compute_max_pending_writes() -> int:
    """Compute max pending writes queue depth based on system memory.

    The background writer now handles full safetensors file writes (not just
    renames), so the queue needs to be deeper to absorb burst saves from
    large requests (e.g., 64 blocks per 4096-token request).

    Scales proportionally: 512GB = 256, 32GB = 32, minimum 32.
    """
    try:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        total_gb = total_bytes / (1024 ** 3)
        return max(32, min(256, int(total_gb / 2)))
    except (ValueError, OSError):
        return 32  # Safe default


_MAX_PENDING_WRITES = _compute_max_pending_writes()


def _has_zero_dim(tensor: Any) -> bool:
    """Check if a tensor has any zero-dimension axis (unsupported by safetensors)."""
    return hasattr(tensor, "shape") and any(d == 0 for d in tensor.shape)


def _encode_shape(shape) -> str:
    """Encode tensor shape as comma-separated string for safetensors metadata."""
    return ",".join(str(d) for d in shape)


def _decode_shape(shape_str: str) -> tuple:
    """Decode shape string back to tuple of ints."""
    return tuple(int(d) for d in shape_str.split(","))


# --- Safetensors dtype mapping for background-thread-safe serialization ---
# These mappings enable writing safetensors files without any mx/Metal API,
# bypassing the bfloat16 limitation that blocked PR #16 v2 (numpy doesn't
# support bfloat16, but safetensors format natively does via "BF16" dtype).

_MX_TO_ST_DTYPE: Dict[Any, str] = {}
_ST_TO_MX_DTYPE: Dict[str, Any] = {}
_ST_DTYPE_TO_NP: Dict[str, Any] = {}

if HAS_MLX:
    _MX_TO_ST_DTYPE = {
        mx.float16: "F16",
        mx.float32: "F32",
        mx.bfloat16: "BF16",
        mx.int8: "I8",
        mx.int16: "I16",
        mx.int32: "I32",
        mx.int64: "I64",
        mx.uint8: "U8",
        mx.uint16: "U16",
        mx.uint32: "U32",
        mx.uint64: "U64",
        mx.bool_: "BOOL",
    }
    _ST_TO_MX_DTYPE = {v: k for k, v in _MX_TO_ST_DTYPE.items()}

_ST_DTYPE_TO_NP = {
    "F16": np.float16,
    "F32": np.float32,
    "BF16": np.uint16,  # bfloat16 handled via uint16 view
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "BOOL": np.bool_,
}


def _extract_tensor_bytes(arr: "mx.array") -> Tuple[bytes, str, List[int]]:
    """Extract raw bytes from an evaluated mx.array.

    Must be called from the inference thread after mx.eval() (Metal-safe).
    The returned bytes can be written to disk from any thread without mx API.

    For bfloat16 arrays, uses view(uint16) trick since Python's buffer
    protocol doesn't support bfloat16 directly.

    Args:
        arr: Evaluated MLX array (must have been mx.eval'd).

    Returns:
        Tuple of (raw_bytes, safetensors_dtype_string, shape_list).
    """
    dtype_str = _MX_TO_ST_DTYPE[arr.dtype]
    shape = list(arr.shape)
    if arr.dtype == mx.bfloat16:
        u16 = arr.view(mx.uint16)
        mx.eval(u16)  # noqa: S307
        raw = bytes(memoryview(u16))
    else:
        raw = bytes(memoryview(arr))
    return raw, dtype_str, shape


def _restore_tensor_from_bytes(
    raw: bytes, dtype_str: str, shape: List[int]
) -> "mx.array":
    """Restore an mx.array from raw bytes extracted by _extract_tensor_bytes.

    No Metal API required — uses numpy as intermediary.

    Args:
        raw: Raw tensor bytes.
        dtype_str: Safetensors dtype string (e.g., "F16", "BF16").
        shape: Tensor shape as list of ints.

    Returns:
        Restored mx.array with correct dtype and shape.
    """
    np_dtype = _ST_DTYPE_TO_NP[dtype_str]
    np_arr = np.frombuffer(raw, dtype=np_dtype)
    arr = mx.array(np_arr)
    if dtype_str == "BF16":
        arr = arr.view(mx.bfloat16)
    return arr.reshape(shape)


def _write_safetensors_no_mx(
    path: str,
    tensors_raw: Dict[str, Tuple[bytes, str, List[int]]],
    metadata: Optional[Dict[str, str]] = None,
) -> int:
    """Write a safetensors file without any mx/Metal API calls.

    Safe to call from background threads. Produces files fully compatible
    with mx.load(path, return_metadata=True).

    The safetensors binary format:
      [8 bytes: header_size as little-endian uint64]
      [header_size bytes: JSON header]
      [remaining bytes: concatenated tensor data]

    Args:
        path: Output file path (must include .safetensors extension).
        tensors_raw: Dict of {name: (raw_bytes, dtype_str, shape)}.
        metadata: Optional string-to-string metadata dict.

    Returns:
        Total file size in bytes.
    """
    offset = 0
    header_tensors = {}
    all_data = []

    for name, (raw, dtype_str, shape) in tensors_raw.items():
        header_tensors[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        all_data.append(raw)
        offset += len(raw)

    header_dict = dict(header_tensors)
    if metadata:
        header_dict["__metadata__"] = metadata

    header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    # Safetensors spec: header must be 8-byte aligned
    pad = (8 - len(header_json) % 8) % 8
    header_json += b" " * pad

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for d in all_data:
            f.write(d)

    return 8 + len(header_json) + offset


def parse_size(size_str: str) -> int:
    """
    Parse a human-readable size string to bytes.

    Args:
        size_str: Size string like "100GB", "50MB", "1TB"

    Returns:
        Size in bytes.
    """
    size_str = size_str.strip().upper()

    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }

    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                value = float(size_str[: -len(unit)])
                return int(value * multiplier)
            except ValueError:
                pass

    # Try parsing as plain number (bytes)
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size string: {size_str}")


@dataclass
class PagedSSDBlockMetadata:
    """
    Metadata for a block stored on SSD.

    Attributes:
        block_hash: Content hash (SHA256) for identification
        file_path: Full path to safetensors file
        file_size: Size in bytes
        token_count: Number of tokens in this block
        created_at: Timestamp when saved
        last_access: Last access time for LRU tracking
        num_layers: Number of model layers
        model_name: Model name for cache isolation between different models
        layer_cache_types: Per-layer cache type names (e.g., ["KVCache", "ArraysCache"])
        layer_meta_states: Per-layer meta_state tuples for reconstruction
    """

    block_hash: bytes
    file_path: Path
    file_size: int
    token_count: int
    created_at: float
    last_access: float
    num_layers: int
    model_name: str = ""
    layer_cache_types: Optional[List[str]] = None
    layer_meta_states: Optional[List[Tuple]] = None

    def touch(self) -> None:
        """Update last access time."""
        self.last_access = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "block_hash": self.block_hash.hex(),
            "file_path": str(self.file_path),
            "file_size": self.file_size,
            "token_count": self.token_count,
            "created_at": self.created_at,
            "last_access": self.last_access,
            "num_layers": self.num_layers,
            "model_name": self.model_name,
        }
        if self.layer_cache_types:
            result["layer_cache_types"] = self.layer_cache_types
        if self.layer_meta_states:
            # Convert tuples to lists for JSON serialization
            result["layer_meta_states"] = [list(m) for m in self.layer_meta_states]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PagedSSDBlockMetadata":
        """Create from dictionary."""
        # Parse layer_meta_states back to tuples
        layer_meta_states = None
        if "layer_meta_states" in data and data["layer_meta_states"]:
            layer_meta_states = [tuple(m) for m in data["layer_meta_states"]]

        return cls(
            block_hash=bytes.fromhex(data["block_hash"]),
            file_path=Path(data["file_path"]),
            file_size=data["file_size"],
            token_count=data["token_count"],
            created_at=data["created_at"],
            last_access=data["last_access"],
            num_layers=data["num_layers"],
            model_name=data.get("model_name", ""),
            layer_cache_types=data.get("layer_cache_types"),
            layer_meta_states=layer_meta_states,
        )


class PagedSSDCacheIndex:
    """
    In-memory index of SSD cache files.

    Provides O(1) lookup by block_hash and LRU tracking for size management.
    Thread-safe for concurrent access.
    """

    def __init__(self, max_size_bytes: int):
        """
        Initialize the SSD cache index.

        Args:
            max_size_bytes: Maximum total size of SSD cache files.
        """
        self._index: Dict[bytes, PagedSSDBlockMetadata] = {}
        self._lru: OrderedDict[bytes, float] = OrderedDict()
        self._total_size: int = 0
        self._max_size: int = max_size_bytes
        self._lock = threading.RLock()

    def add(self, metadata: PagedSSDBlockMetadata) -> None:
        """
        Add a block to the index.

        Args:
            metadata: Block metadata to add.
        """
        with self._lock:
            # Remove existing entry if present
            if metadata.block_hash in self._index:
                old_meta = self._index[metadata.block_hash]
                self._total_size -= old_meta.file_size
                del self._lru[metadata.block_hash]

            self._index[metadata.block_hash] = metadata
            self._lru[metadata.block_hash] = metadata.last_access
            self._total_size += metadata.file_size

    def get(self, block_hash: bytes) -> Optional[PagedSSDBlockMetadata]:
        """
        Get block metadata by hash.

        Args:
            block_hash: Block content hash.

        Returns:
            PagedSSDBlockMetadata if found, None otherwise.
        """
        with self._lock:
            return self._index.get(block_hash)

    def remove(self, block_hash: bytes) -> Optional[PagedSSDBlockMetadata]:
        """
        Remove a block from the index.

        Args:
            block_hash: Block content hash.

        Returns:
            Removed metadata if found, None otherwise.
        """
        with self._lock:
            if block_hash not in self._index:
                return None

            metadata = self._index.pop(block_hash)
            del self._lru[block_hash]
            self._total_size -= metadata.file_size
            return metadata

    def touch(self, block_hash: bytes) -> None:
        """
        Update last access time (move to end of LRU).

        Args:
            block_hash: Block content hash.
        """
        with self._lock:
            if block_hash in self._index:
                self._index[block_hash].touch()
                self._lru.move_to_end(block_hash)
                self._lru[block_hash] = self._index[block_hash].last_access

    def get_lru_entries(self, count: int) -> List[PagedSSDBlockMetadata]:
        """
        Get least recently used entries.

        Args:
            count: Maximum number of entries to return.

        Returns:
            List of LRU metadata entries.
        """
        with self._lock:
            result = []
            for block_hash in list(self._lru.keys())[:count]:
                if block_hash in self._index:
                    result.append(self._index[block_hash])
            return result

    def evict_until_size(self, target_size: int) -> List[PagedSSDBlockMetadata]:
        """
        Evict LRU entries until total size is below target.

        Args:
            target_size: Target total size in bytes.

        Returns:
            List of evicted metadata (files need to be deleted by caller).
        """
        with self._lock:
            evicted = []
            while self._total_size > target_size and self._lru:
                # Get LRU entry (first in OrderedDict)
                block_hash = next(iter(self._lru))
                metadata = self.remove(block_hash)
                if metadata:
                    evicted.append(metadata)
            return evicted

    def contains(self, block_hash: bytes) -> bool:
        """Check if block exists in index."""
        with self._lock:
            return block_hash in self._index

    @property
    def total_size(self) -> int:
        """Get total size of indexed files."""
        with self._lock:
            return self._total_size

    @property
    def max_size(self) -> int:
        """Get maximum allowed size."""
        return self._max_size

    @property
    def count(self) -> int:
        """Get number of indexed blocks."""
        with self._lock:
            return len(self._index)

    def update_file_size(self, block_hash: bytes, actual_size: int) -> None:
        """Update file size for a block after background write completes.

        Args:
            block_hash: Block content hash.
            actual_size: Actual file size in bytes.
        """
        with self._lock:
            entry = self._index.get(block_hash)
            if entry is not None:
                self._total_size += (actual_size - entry.file_size)
                entry.file_size = actual_size

    def get_all_hashes(self) -> List[bytes]:
        """Get all indexed block hashes."""
        with self._lock:
            return list(self._index.keys())


class PagedSSDCacheManager(CacheManager):
    """
    Manages SSD storage for KV cache blocks.

    Features:
    - Block-level safetensors serialization
    - Hash-based subdirectory structure (single level: /a/, /b/, etc.)
    - LRU-based SSD cache size management

    Implements the CacheManager ABC interface for consistency with other
    cache implementations in oMLX.

    Example:
        >>> manager = PagedSSDCacheManager(
        ...     cache_dir=Path("/tmp/ssd_cache"),
        ...     max_size_bytes=100 * 1024**3,  # 100GB
        ... )
        >>> manager.save_block(block_hash, cache_data, token_count=64)
        >>> loaded = manager.load_block(block_hash)
    """

    # Subdirectory prefixes (hash first char)
    SUBDIR_CHARS = "0123456789abcdef"

    def __init__(
        self,
        cache_dir: Path,
        max_size_bytes: int,
    ):
        """
        Initialize the SSD cache manager.

        Args:
            cache_dir: Directory for SSD cache files.
            max_size_bytes: Maximum total size of SSD cache.
        """
        self._cache_dir = Path(cache_dir)
        self._max_size = max_size_bytes
        self._index = PagedSSDCacheIndex(max_size_bytes)
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "saves": 0,
            "loads": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "errors": 0,
        }

        # Initialize directory structure and scan existing files
        self._init_directories()
        self._scan_existing_files()

        # --- Background writer for non-blocking saves ---
        self._write_queue: queue.Queue = queue.Queue(maxsize=_MAX_PENDING_WRITES)
        self._pending_writes: Dict[bytes, Dict] = {}
        self._pending_writes_lock = threading.Lock()
        self._writer_shutdown = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="ssd-cache-writer",
            daemon=True,
        )
        self._writer_thread.start()

        logger.info(
            f"PagedSSDCacheManager initialized: dir={self._cache_dir}, "
            f"max_size={format_bytes(max_size_bytes)}, "
            f"existing_files={self._index.count}"
        )

    def _init_directories(self) -> None:
        """Create cache directory structure."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for first hex character
        for char in self.SUBDIR_CHARS:
            subdir = self._cache_dir / char
            subdir.mkdir(exist_ok=True)

    def _get_file_path(self, block_hash: bytes) -> Path:
        """
        Get file path for a block hash.

        Uses first character of hex hash as subdirectory.

        Args:
            block_hash: Block content hash.

        Returns:
            Path to the safetensors file.
        """
        hash_hex = block_hash.hex()
        subdir = hash_hex[0]  # First character
        filename = f"{hash_hex}.safetensors"
        return self._cache_dir / subdir / filename

    def _scan_existing_files(self) -> None:
        """Scan cache directory for existing files and build index."""
        logger.info(f"Scanning SSD cache directory: {self._cache_dir}")

        scanned = 0
        indexed = 0
        errors = 0

        for subdir in self.SUBDIR_CHARS:
            subdir_path = self._cache_dir / subdir
            if not subdir_path.exists():
                continue

            for file_path in subdir_path.glob("*.safetensors"):
                scanned += 1
                try:
                    metadata = self._read_file_metadata(file_path)
                    if metadata:
                        self._index.add(metadata)
                        indexed += 1
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
                    errors += 1

        logger.info(
            f"SSD cache scan complete: scanned={scanned}, indexed={indexed}, "
            f"errors={errors}, total_size={format_bytes(self._index.total_size)}"
        )

    def _read_file_metadata(self, file_path: Path) -> Optional[PagedSSDBlockMetadata]:
        """
        Read metadata from an existing cache file.

        Args:
            file_path: Path to safetensors file.

        Returns:
            PagedSSDBlockMetadata if valid, None otherwise.
        """
        if not HAS_MLX:
            return None

        try:
            # Load just the metadata without loading tensors
            _, metadata = mx.load(str(file_path), return_metadata=True)

            block_hash_hex = metadata.get("block_hash", "")
            if not block_hash_hex:
                return None

            file_stat = file_path.stat()

            # Parse cache type information if present
            layer_cache_types = None
            layer_meta_states = None

            if "layer_cache_types" in metadata and metadata["layer_cache_types"]:
                try:
                    layer_cache_types = json.loads(metadata["layer_cache_types"])
                except (json.JSONDecodeError, TypeError):
                    pass

            if "layer_meta_states" in metadata and metadata["layer_meta_states"]:
                try:
                    raw_meta_states = json.loads(metadata["layer_meta_states"])
                    layer_meta_states = [tuple(m) if m else () for m in raw_meta_states]
                except (json.JSONDecodeError, TypeError):
                    pass

            return PagedSSDBlockMetadata(
                block_hash=bytes.fromhex(block_hash_hex),
                file_path=file_path,
                file_size=file_stat.st_size,
                token_count=int(metadata.get("token_count", 0)),
                created_at=file_stat.st_ctime,
                last_access=file_stat.st_mtime,
                num_layers=int(metadata.get("num_layers", 0)),
                model_name=metadata.get("model_name", ""),
                layer_cache_types=layer_cache_types,
                layer_meta_states=layer_meta_states,
            )
        except Exception as e:
            logger.debug(f"Failed to read metadata from {file_path}: {e}")
            return None

    def _writer_loop(self) -> None:
        """Background writer that drains the write queue.

        Runs in a dedicated daemon thread. Writes full safetensors files
        using pure Python I/O (no mx/Metal API calls), then atomically
        renames temp files to their final paths.

        This is safe because save_block() extracts tensor data as raw bytes
        on the inference thread (Metal-safe), and this thread only performs
        standard file I/O operations.
        """
        while not self._writer_shutdown.is_set():
            try:
                item = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:  # Sentinel for shutdown
                break

            block_hash, tensors_raw, metadata, file_path = item
            temp_path = None

            try:
                # Write safetensors file using pure Python (no mx/Metal API)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = file_path.with_name(
                    file_path.stem + "_tmp.safetensors"
                )
                actual_size = _write_safetensors_no_mx(
                    str(temp_path), tensors_raw, metadata
                )

                # Atomic rename to final path
                os.rename(str(temp_path), str(file_path))

                # Update index with actual file size
                self._index.update_file_size(block_hash, actual_size)

                # Check if block was evicted while write was pending
                if not self._index.contains(block_hash):
                    logger.debug(
                        f"Block {block_hash.hex()[:16]} evicted during write, "
                        f"cleaning up file"
                    )
                    try:
                        file_path.unlink()
                    except Exception:
                        pass

            except Exception as e:
                logger.error(
                    f"Background write failed for {block_hash.hex()[:16]}: {e}"
                )
                self._stats["errors"] += 1
                # Remove from index since file wasn't written
                self._index.remove(block_hash)
                # Clean up temp and final files
                for p in (temp_path, file_path):
                    try:
                        if p is not None and isinstance(p, Path) and p.exists():
                            p.unlink()
                    except Exception:
                        pass
            finally:
                # Always remove from pending writes after file is on disk
                with self._pending_writes_lock:
                    self._pending_writes.pop(block_hash, None)

    def save_block(
        self,
        block_hash: bytes,
        cache_data: List[Any],
        token_count: int,
        model_name: str = "",
        layer_cache_types: Optional[List[str]] = None,
        layer_meta_states: Optional[List[Tuple]] = None,
    ) -> bool:
        """
        Save a KV cache block to SSD storage (non-blocking).

        Data is enqueued for background writing. The block is immediately
        available for reads via the in-memory pending-writes buffer.

        Args:
            block_hash: Content hash for the block.
            cache_data: List of per-layer data. Each element is either:
                - (keys, values) tuple for standard caches (KVCache, etc.)
                - ('__cache_list__', sub_tensors) marker tuple for CacheList layers,
                  where sub_tensors is List[Tuple[keys, values]] per sub-cache.
            token_count: Number of tokens in the block.
            model_name: Model name for cache isolation between different models.
            layer_cache_types: Optional list of cache type names per layer
                (e.g., ["KVCache", "ArraysCache", "KVCache", "CacheList"]).
            layer_meta_states: Optional list of meta_state tuples per layer
                for reconstruction (e.g., [(offset,), (keep, max_size, offset, _idx)]).

        Returns:
            True if enqueued successfully, False otherwise.
        """
        if not HAS_MLX:
            logger.error("MLX not available, cannot save block")
            return False

        # Check if already exists in index (thread-safe)
        if self._index.contains(block_hash):
            self._index.touch(block_hash)
            self._stats["hits"] += 1
            return True

        # Also check pending writes
        with self._pending_writes_lock:
            if block_hash in self._pending_writes:
                self._stats["hits"] += 1
                return True

        # Check queue capacity before doing expensive GPU/disk work
        if self._write_queue.full():
            logger.warning(
                f"SSD cache write queue full, skipping save for "
                f"{block_hash.hex()[:16]}"
            )
            return False

        file_path = self._get_file_path(block_hash)

        try:
            # Enforce size limit before saving
            self._enforce_size_limit_for_new_block()

            # Prepare arrays for safetensors
            arrays = {}
            cache_list_meta = {}  # Temporary dict for CacheList sub_count
            for i, layer_data in enumerate(cache_data):
                if (isinstance(layer_data, tuple) and len(layer_data) == 2
                        and isinstance(layer_data[0], str)
                        and layer_data[0] == '__cache_list__'):
                    # CacheList: sub-indexed tensors
                    sub_tensors = layer_data[1]
                    for j, (sub_keys, sub_values) in enumerate(sub_tensors):
                        if _has_zero_dim(sub_keys):
                            arrays[f"layer_{i}_sub_{j}_keys"] = mx.zeros((1,))
                            cache_list_meta[f"layer_{i}_sub_{j}_keys_zero_dim"] = (
                                _encode_shape(sub_keys.shape)
                            )
                        else:
                            arrays[f"layer_{i}_sub_{j}_keys"] = sub_keys
                        if _has_zero_dim(sub_values):
                            arrays[f"layer_{i}_sub_{j}_values"] = mx.zeros((1,))
                            cache_list_meta[f"layer_{i}_sub_{j}_values_zero_dim"] = (
                                _encode_shape(sub_values.shape)
                            )
                        else:
                            arrays[f"layer_{i}_sub_{j}_values"] = sub_values
                    cache_list_meta[f"layer_{i}_sub_count"] = str(len(sub_tensors))
                else:
                    keys, values = layer_data
                    if _has_zero_dim(keys):
                        arrays[f"layer_{i}_keys"] = mx.zeros((1,))
                        cache_list_meta[f"layer_{i}_keys_zero_dim"] = (
                            _encode_shape(keys.shape)
                        )
                    else:
                        arrays[f"layer_{i}_keys"] = keys
                    if _has_zero_dim(values):
                        arrays[f"layer_{i}_values"] = mx.zeros((1,))
                        cache_list_meta[f"layer_{i}_values_zero_dim"] = (
                            _encode_shape(values.shape)
                        )
                    else:
                        arrays[f"layer_{i}_values"] = values

            # Prepare metadata
            metadata = {
                "block_hash": block_hash.hex(),
                "token_count": str(token_count),
                "num_layers": str(len(cache_data)),
                "model_name": model_name,
                "created_at": str(time.time()),
            }

            # Add cache type information if provided
            if layer_cache_types:
                metadata["layer_cache_types"] = json.dumps(layer_cache_types)
            if layer_meta_states:
                metadata["layer_meta_states"] = json.dumps(
                    [list(m) if m else [] for m in layer_meta_states]
                )

            # Merge CacheList sub_count metadata
            metadata.update(cache_list_meta)

            # Materialize lazy arrays on the inference thread (Metal-safe).
            if arrays:
                mx.eval(*arrays.values())  # noqa: S307 — MLX tensor eval, not Python eval

            # Extract raw bytes from evaluated tensors on the inference thread.
            # This is Metal-safe because it uses memoryview() on evaluated arrays.
            # For bfloat16, we use view(uint16) trick since Python's buffer
            # protocol doesn't support bfloat16 directly.
            # The background writer thread then writes the safetensors file
            # using pure Python I/O — no mx/Metal API calls needed.
            tensors_raw = {}
            for name, arr in arrays.items():
                tensors_raw[name] = _extract_tensor_bytes(arr)

            # Estimate file size from raw bytes (actual size set by background writer)
            estimated_size = (
                sum(len(raw) for raw, _, _ in tensors_raw.values()) + 1024
            )

            # Add to index with estimated file size
            now = time.time()
            block_metadata = PagedSSDBlockMetadata(
                block_hash=block_hash,
                file_path=file_path,
                file_size=estimated_size,
                token_count=token_count,
                created_at=now,
                last_access=now,
                num_layers=len(cache_data),
                model_name=model_name,
                layer_cache_types=layer_cache_types,
                layer_meta_states=layer_meta_states,
            )
            self._index.add(block_metadata)

            # Store in pending writes for immediate read-back.
            # Uses raw bytes (not mx tensors) so mx.arrays can be GC'd,
            # releasing Metal resources sooner.
            with self._pending_writes_lock:
                self._pending_writes[block_hash] = {
                    'tensors_raw': tensors_raw,
                    'file_metadata': metadata,
                    'num_layers': len(cache_data),
                    'layer_cache_types': layer_cache_types,
                    'block_metadata': block_metadata,
                }

            # Enqueue full file write for background thread
            try:
                self._write_queue.put_nowait(
                    (block_hash, tensors_raw, metadata, file_path)
                )
            except queue.Full:
                logger.warning(
                    f"SSD cache write queue full, dropping write for "
                    f"{block_hash.hex()[:16]}"
                )
                self._index.remove(block_hash)
                with self._pending_writes_lock:
                    self._pending_writes.pop(block_hash, None)
                return False

            self._stats["saves"] += 1
            logger.debug(
                f"Enqueued block for SSD cache write: {block_hash.hex()[:16]}..., "
                f"size={format_bytes(estimated_size)}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to prepare block for SSD cache: {e}")
            self._stats["errors"] += 1
            return False

    def _reconstruct_cache_data(
        self,
        arrays: Dict[str, Any],
        file_metadata: Dict[str, str],
        num_layers: int,
        layer_cache_types: Optional[List[str]] = None,
    ) -> Optional[List[Any]]:
        """Reconstruct cache_data list from flattened arrays and metadata.

        Shared helper for load_block(), load_block_with_metadata(), and
        pending-writes read path to avoid code duplication.

        Args:
            arrays: Flattened tensor dict (layer_i_keys, layer_i_values, ...).
            file_metadata: Safetensors metadata dict (string values).
            num_layers: Number of model layers.
            layer_cache_types: Per-layer cache type names.

        Returns:
            Reconstructed cache_data list, or None on error.
        """
        cache_data = []

        for i in range(num_layers):
            cache_type = (
                layer_cache_types[i]
                if layer_cache_types and i < len(layer_cache_types)
                else None
            )

            if cache_type == 'CacheList':
                sub_count_key = f"layer_{i}_sub_count"
                sub_count = 0
                if file_metadata and sub_count_key in file_metadata:
                    try:
                        sub_count = int(file_metadata[sub_count_key])
                    except (ValueError, TypeError):
                        pass

                if sub_count > 0:
                    sub_tensors = []
                    for j in range(sub_count):
                        sk_key = f"layer_{i}_sub_{j}_keys"
                        sv_key = f"layer_{i}_sub_{j}_values"
                        if sk_key not in arrays or sv_key not in arrays:
                            logger.error(
                                f"Missing sub-cache {j} for CacheList layer {i}"
                            )
                            return None
                        sub_keys = arrays[sk_key]
                        sub_values = arrays[sv_key]
                        sk_zd = f"layer_{i}_sub_{j}_keys_zero_dim"
                        sv_zd = f"layer_{i}_sub_{j}_values_zero_dim"
                        if file_metadata and sk_zd in file_metadata:
                            sub_keys = mx.zeros(_decode_shape(file_metadata[sk_zd]))
                        if file_metadata and sv_zd in file_metadata:
                            sub_values = mx.zeros(_decode_shape(file_metadata[sv_zd]))
                        sub_tensors.append((sub_keys, sub_values))
                    cache_data.append(sub_tensors)
                else:
                    keys_key = f"layer_{i}_keys"
                    values_key = f"layer_{i}_values"
                    if keys_key not in arrays or values_key not in arrays:
                        logger.error(f"Missing keys/values for layer {i}")
                        return None
                    cache_data.append((arrays[keys_key], arrays[values_key]))
            else:
                keys_key = f"layer_{i}_keys"
                values_key = f"layer_{i}_values"

                if keys_key not in arrays or values_key not in arrays:
                    logger.error(f"Missing keys/values for layer {i}")
                    return None

                keys = arrays[keys_key]
                values = arrays[values_key]
                k_zd = f"layer_{i}_keys_zero_dim"
                v_zd = f"layer_{i}_values_zero_dim"
                if file_metadata and k_zd in file_metadata:
                    keys = mx.zeros(_decode_shape(file_metadata[k_zd]))
                if file_metadata and v_zd in file_metadata:
                    values = mx.zeros(_decode_shape(file_metadata[v_zd]))
                cache_data.append((keys, values))

        return cache_data

    @staticmethod
    def _arrays_from_tensors_raw(
        tensors_raw: Dict[str, Tuple[bytes, str, List[int]]]
    ) -> Dict[str, "mx.array"]:
        """Convert raw bytes dict back to mx.array dict for _reconstruct_cache_data.

        Args:
            tensors_raw: Dict of {name: (raw_bytes, dtype_str, shape)}.

        Returns:
            Dict of {name: mx.array} with correct dtypes and shapes.
        """
        arrays = {}
        for name, (raw, dtype_str, shape) in tensors_raw.items():
            arrays[name] = _restore_tensor_from_bytes(raw, dtype_str, shape)
        return arrays

    def load_block(
        self,
        block_hash: bytes,
    ) -> Optional[List[Any]]:
        """
        Load a KV cache block from SSD storage.

        Checks pending writes first (in-memory, no I/O), then falls back to disk
        read with a timeout to prevent inference deadlocks.

        Args:
            block_hash: Content hash for the block.

        Returns:
            List of per-layer data, or None if not found/timed out.
            Each element is either:
            - (keys, values) tuple for standard caches
            - List[Tuple[keys, values]] for CacheList layers
        """
        if not HAS_MLX:
            logger.error("MLX not available, cannot load block")
            return None

        # Check pending writes first (in-memory, no I/O)
        with self._pending_writes_lock:
            pending = self._pending_writes.get(block_hash)

        if pending is not None:
            arrays = self._arrays_from_tensors_raw(pending['tensors_raw'])
            cache_data = self._reconstruct_cache_data(
                arrays, pending['file_metadata'],
                pending['num_layers'], pending['layer_cache_types'],
            )
            if cache_data is not None:
                self._index.touch(block_hash)
                self._stats["loads"] += 1
                self._stats["hits"] += 1
                logger.debug(
                    f"Loaded block from pending writes: {block_hash.hex()[:16]}..."
                )
            return cache_data

        # Check index
        metadata = self._index.get(block_hash)
        if metadata is None:
            self._stats["misses"] += 1
            return None

        file_path = metadata.file_path

        if not file_path.exists():
            logger.warning(f"SSD cache file missing: {file_path}")
            self._index.remove(block_hash)
            self._stats["misses"] += 1
            return None

        try:
            # Load directly on the inference thread (Metal-safe).
            # SSD read for a ~10MB block takes ~2ms @ 5GB/s — negligible.
            # Previous executor-based approach caused deadlocks when
            # mx.load() in a worker thread contested Metal GPU resources
            # with the main inference thread.
            arrays, file_metadata = mx.load(str(file_path), return_metadata=True)

            # Get layer_cache_types for CacheList detection
            layer_cache_types = metadata.layer_cache_types
            if not layer_cache_types and file_metadata and "layer_cache_types" in file_metadata:
                try:
                    layer_cache_types = json.loads(file_metadata["layer_cache_types"])
                except (json.JSONDecodeError, TypeError):
                    layer_cache_types = None

            cache_data = self._reconstruct_cache_data(
                arrays, file_metadata, metadata.num_layers, layer_cache_types,
            )
            if cache_data is None:
                return None

            # Update access time
            self._index.touch(block_hash)
            self._stats["loads"] += 1
            self._stats["hits"] += 1

            logger.debug(f"Loaded block from SSD cache: {block_hash.hex()[:16]}...")
            return cache_data

        except Exception as e:
            logger.error(f"Failed to load block from SSD cache: {e}")
            self._stats["errors"] += 1
            # Remove corrupted entry
            self._index.remove(block_hash)
            try:
                file_path.unlink()
            except Exception:
                pass
            return None

    def load_block_with_metadata(
        self,
        block_hash: bytes,
    ) -> Tuple[Optional[List[Any]], Optional[Dict[str, Any]]]:
        """
        Load a KV cache block with its metadata from SSD storage.

        Checks pending writes first (zero I/O), then falls back to disk
        read with a timeout to prevent inference deadlocks.

        Args:
            block_hash: Content hash for the block.

        Returns:
            Tuple of (cache_data, metadata_dict) where:
            - cache_data: List of per-layer data, or None.
              Each element is either (keys, values) or List[Tuple[keys, values]]
              for CacheList layers.
            - metadata_dict: Dictionary with cache type info, or None
              {
                  "layer_cache_types": List[str],  # per-layer type names
                  "layer_meta_states": List[Tuple],  # per-layer meta states
                  "num_layers": int,
                  "token_count": int,
              }
        """
        if not HAS_MLX:
            logger.error("MLX not available, cannot load block")
            return None, None

        # Check pending writes first (in-memory, no I/O)
        with self._pending_writes_lock:
            pending = self._pending_writes.get(block_hash)

        if pending is not None:
            blk_meta = pending['block_metadata']
            arrays = self._arrays_from_tensors_raw(pending['tensors_raw'])
            cache_data = self._reconstruct_cache_data(
                arrays, pending['file_metadata'],
                pending['num_layers'], pending['layer_cache_types'],
            )
            if cache_data is None:
                return None, None

            metadata_dict = {
                "num_layers": pending['num_layers'],
                "token_count": blk_meta.token_count,
                "model_name": blk_meta.model_name,
                "layer_cache_types": pending['layer_cache_types'],
                "layer_meta_states": blk_meta.layer_meta_states,
            }

            self._index.touch(block_hash)
            self._stats["loads"] += 1
            self._stats["hits"] += 1
            logger.debug(
                f"Loaded block with metadata from pending writes: "
                f"{block_hash.hex()[:16]}..."
            )
            return cache_data, metadata_dict

        # Check index
        block_metadata = self._index.get(block_hash)
        if block_metadata is None:
            self._stats["misses"] += 1
            return None, None

        file_path = block_metadata.file_path

        if not file_path.exists():
            logger.warning(f"SSD cache file missing: {file_path}")
            self._index.remove(block_hash)
            self._stats["misses"] += 1
            return None, None

        try:
            # Load directly on the inference thread (Metal-safe).
            # See load_block() for rationale on removing the executor.
            arrays, file_metadata = mx.load(str(file_path), return_metadata=True)

            # Parse layer_cache_types early for CacheList detection
            layer_cache_types = block_metadata.layer_cache_types
            if not layer_cache_types and file_metadata and "layer_cache_types" in file_metadata:
                try:
                    layer_cache_types = json.loads(
                        file_metadata["layer_cache_types"]
                    )
                except (json.JSONDecodeError, TypeError):
                    layer_cache_types = None

            cache_data = self._reconstruct_cache_data(
                arrays, file_metadata, block_metadata.num_layers, layer_cache_types,
            )
            if cache_data is None:
                return None, None

            # Build metadata dict for reconstruction
            metadata_dict = {
                "num_layers": block_metadata.num_layers,
                "token_count": block_metadata.token_count,
                "model_name": block_metadata.model_name,
                "layer_cache_types": layer_cache_types,
                "layer_meta_states": block_metadata.layer_meta_states,
            }

            if not metadata_dict["layer_meta_states"] and file_metadata:
                if "layer_meta_states" in file_metadata:
                    try:
                        raw = json.loads(file_metadata["layer_meta_states"])
                        metadata_dict["layer_meta_states"] = [
                            tuple(m) if m else () for m in raw
                        ]
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Update access time
            self._index.touch(block_hash)
            self._stats["loads"] += 1
            self._stats["hits"] += 1

            logger.debug(
                f"Loaded block with metadata from SSD cache: {block_hash.hex()[:16]}..."
            )
            return cache_data, metadata_dict

        except Exception as e:
            logger.error(f"Failed to load block from SSD cache: {e}")
            self._stats["errors"] += 1
            # Remove corrupted entry
            self._index.remove(block_hash)
            try:
                file_path.unlink()
            except Exception:
                pass
            return None, None

    def get_block_metadata(self, block_hash: bytes) -> Optional[PagedSSDBlockMetadata]:
        """
        Get metadata for a block without loading the data.

        Args:
            block_hash: Content hash for the block.

        Returns:
            PagedSSDBlockMetadata if found, None otherwise.
        """
        return self._index.get(block_hash)

    def has_block(self, block_hash: bytes) -> bool:
        """
        Check if a block exists in SSD storage.

        Args:
            block_hash: Content hash for the block.

        Returns:
            True if block exists in SSD cache.
        """
        return self._index.contains(block_hash)

    def delete_block(self, block_hash: bytes) -> bool:
        """
        Delete a block from SSD storage.

        Args:
            block_hash: Content hash for the block.

        Returns:
            True if deleted successfully.
        """
        with self._lock:
            # Also remove from pending writes
            with self._pending_writes_lock:
                self._pending_writes.pop(block_hash, None)

            metadata = self._index.remove(block_hash)
            if metadata is None:
                return False

            try:
                if metadata.file_path.exists():
                    metadata.file_path.unlink()
                    logger.debug(f"Deleted SSD cache file: {metadata.file_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete SSD cache file: {e}")
                return False

    def _enforce_size_limit_for_new_block(self) -> None:
        """Enforce size limit before adding a new block."""
        # Estimate average block size (use 1MB as conservative estimate)
        estimated_new_size = 1 * 1024 * 1024

        target_size = self._max_size - estimated_new_size
        if target_size < 0:
            target_size = int(self._max_size * 0.9)

        if self._index.total_size > target_size:
            evicted = self._index.evict_until_size(target_size)
            for metadata in evicted:
                # Clean up pending writes for evicted blocks
                with self._pending_writes_lock:
                    self._pending_writes.pop(metadata.block_hash, None)
                try:
                    if metadata.file_path.exists():
                        metadata.file_path.unlink()
                        self._stats["evictions"] += 1
                        logger.debug(f"Evicted SSD cache file: {metadata.file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete evicted file: {e}")

    def enforce_size_limit(self) -> int:
        """
        Enforce SSD cache size limit by evicting LRU files.

        Returns:
            Number of bytes freed.
        """
        with self._lock:
            initial_size = self._index.total_size

            if initial_size <= self._max_size:
                return 0

            target_size = int(self._max_size * 0.9)  # 90% of max
            evicted = self._index.evict_until_size(target_size)

            for metadata in evicted:
                # Clean up pending writes for evicted blocks
                with self._pending_writes_lock:
                    self._pending_writes.pop(metadata.block_hash, None)
                try:
                    if metadata.file_path.exists():
                        metadata.file_path.unlink()
                        self._stats["evictions"] += 1
                except Exception as e:
                    logger.warning(f"Failed to delete evicted file: {e}")

            freed = initial_size - self._index.total_size
            logger.info(
                f"SSD cache size enforcement: freed {format_bytes(freed)}, "
                f"evicted {len(evicted)} files"
            )
            return freed

    def clear(self) -> int:
        """
        Clear all SSD cache files.

        Returns:
            Number of files deleted.
        """
        with self._lock:
            count = 0
            for block_hash in self._index.get_all_hashes():
                if self.delete_block(block_hash):
                    count += 1

            logger.info(f"Cleared SSD cache: deleted {count} files")
            return count

    def get_stats(self) -> PagedSSDCacheStats:
        """
        Get SSD cache statistics.

        Returns:
            PagedSSDCacheStats with cache metrics.
        """
        with self._lock:
            return PagedSSDCacheStats(
                hits=self._stats["hits"],
                misses=self._stats["misses"],
                evictions=self._stats["evictions"],
                saves=self._stats["saves"],
                loads=self._stats["loads"],
                errors=self._stats["errors"],
                total_size_bytes=self._index.total_size,
                num_files=self._index.count,
            )

    def get_stats_dict(self) -> Dict[str, Any]:
        """
        Get SSD cache statistics as a dictionary.

        This method provides the legacy dictionary format for compatibility.

        Returns:
            Dictionary with cache statistics.
        """
        with self._lock:
            return {
                "cache_dir": str(self._cache_dir),
                "max_size": self._max_size,
                "max_size_formatted": format_bytes(self._max_size),
                "total_size": self._index.total_size,
                "total_size_formatted": format_bytes(self._index.total_size),
                "utilization": (
                    self._index.total_size / self._max_size
                    if self._max_size > 0
                    else 0.0
                ),
                "num_files": self._index.count,
                **self._stats,
            }

    def close(self) -> None:
        """Close the SSD cache manager, flushing pending writes."""
        logger.info("Shutting down PagedSSDCacheManager...")

        # Signal writer thread to stop
        self._writer_shutdown.set()

        # Send sentinel to unblock the writer if it's waiting on the queue
        try:
            self._write_queue.put_nowait(None)
        except queue.Full:
            pass  # Writer will check shutdown flag on next iteration

        # Wait for writer to finish pending writes (increased timeout for full
        # file writes instead of just renames)
        self._writer_thread.join(timeout=60)
        if self._writer_thread.is_alive():
            logger.warning("SSD cache writer thread did not stop within 60s")

        logger.debug("PagedSSDCacheManager closed")

    def __repr__(self) -> str:
        return (
            f"PagedSSDCacheManager(dir={self._cache_dir}, "
            f"size={format_bytes(self._index.total_size)}/"
            f"{format_bytes(self._max_size)}, "
            f"files={self._index.count})"
        )

    # =========================================================================
    # CacheManager ABC Interface Implementation
    # =========================================================================

    def fetch(self, key: Any) -> Tuple[Optional[Any], bool]:
        """
        Fetch a cached block from SSD storage.

        Args:
            key: Block hash (bytes) to look up.

        Returns:
            Tuple of (cache_data, True) if found, (None, False) otherwise.
        """
        if not isinstance(key, bytes):
            return None, False

        cache_data = self.load_block(key)
        if cache_data is not None:
            return cache_data, True
        return None, False

    def store(self, key: Any, value: Any) -> bool:
        """
        Store a block in SSD cache.

        Args:
            key: Block hash (bytes).
            value: Tuple of (cache_data, token_count) or just cache_data.

        Returns:
            True if stored successfully.
        """
        if not isinstance(key, bytes):
            return False

        if isinstance(value, tuple) and len(value) >= 2:
            cache_data, token_count = value[0], value[1]
            model_name = value[2] if len(value) > 2 else ""
        else:
            cache_data = value
            token_count = 0
            model_name = ""

        return self.save_block(key, cache_data, token_count, model_name)

    def evict(self, key: Any) -> bool:
        """
        Evict a specific block from SSD cache.

        Args:
            key: Block hash (bytes) to evict.

        Returns:
            True if evicted, False if not found.
        """
        if not isinstance(key, bytes):
            return False

        return self.delete_block(key)

    @property
    def size(self) -> int:
        """
        Get the current number of cached blocks.

        Returns:
            Number of cached blocks.
        """
        return self._index.count

    @property
    def max_size(self) -> int:
        """
        Get the maximum cache size in bytes.

        Returns:
            Maximum cache size in bytes.
        """
        return self._max_size


