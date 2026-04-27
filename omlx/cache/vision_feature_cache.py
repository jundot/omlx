# SPDX-License-Identifier: Apache-2.0
"""
Vision feature cache with memory LRU and SSD persistence.

Caches the output of vision_tower + projector (image features projected
into language model space) keyed by (model_name, image_hash). This avoids
re-running the vision encoder when the same image appears with different
text contexts across multi-turn conversations.

Two-tier caching:
- In-memory LRU (OrderedDict): fast lookup for recently seen images
- SSD persistence (safetensors): survives engine restarts

Uses the same safetensors serialization infrastructure as PagedSSDCacheManager
for consistency and bfloat16 support.
"""

import errno
import hashlib
import json
import logging
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from .paged_ssd_cache import _extract_tensor_bytes, _write_safetensors_no_mx

logger = logging.getLogger(__name__)

# Hex chars for subdirectory bucketing
_SUBDIR_CHARS = "0123456789abcdef"


def _composite_key(model_name: str, image_hash: str) -> str:
    """Build a composite cache key from model name and image hash."""
    return f"{model_name}:{image_hash}"


def _composite_hash(model_name: str, image_hash: str) -> str:
    """Compute a SHA256 hex digest for SSD file naming.

    Using a hash avoids filesystem issues with long model paths
    and ensures uniform directory distribution.
    """
    return hashlib.sha256(
        f"{model_name}:{image_hash}".encode()
    ).hexdigest()


@dataclass
class VisionFeatureSSDEntry:
    """Metadata for a cached vision feature stored on SSD."""

    image_hash: str
    model_name: str
    file_path: Path
    file_size: int
    created_at: float
    last_access: float
    num_tensors: int = 1  # 1 for single image, N for multi-image list


class VisionFeatureSSDCache:
    """Two-tier vision feature cache: in-memory LRU + SSD persistence.

    Args:
        cache_dir: SSD storage directory. None for memory-only mode.
        max_size_bytes: Maximum SSD cache size in bytes (default 10GB).
        max_memory_entries: Maximum in-memory LRU entries (default 20).
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size_bytes: int = 10 * 1024**3,
        max_memory_entries: int = 20,
    ):
        self._cache_dir = cache_dir
        self._max_size_bytes = max_size_bytes
        self._max_memory_entries = max_memory_entries

        # In-memory LRU cache: composite_key -> mx.array (or list[mx.array])
        self._memory_cache: OrderedDict[str, Any] = OrderedDict()
        self._memory_lock = threading.Lock()

        # SSD index: composite_key -> VisionFeatureSSDEntry
        self._ssd_index: OrderedDict[str, VisionFeatureSSDEntry] = OrderedDict()
        self._ssd_lock = threading.RLock()
        self._ssd_total_size: int = 0

        # Background writer
        self._write_queue: queue.Queue = queue.Queue(maxsize=32)
        self._writer_shutdown = threading.Event()
        self._pending_write_keys: set = set()
        self._pending_lock = threading.Lock()

        # Stats
        self._stats: Dict[str, int] = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "ssd_loads": 0,
            "errors": 0,
        }

        # Initialize SSD directory and scan existing files
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._scan_existing_files()

        # Start background writer thread
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="vision-cache-writer"
        )
        self._writer_thread.start()

    def get(self, image_hash: str, model_name: str) -> Optional[Any]:
        """Look up cached vision features.

        Checks memory LRU first, then SSD. Returns None on miss.

        Args:
            image_hash: SHA256 hash from compute_image_hash().
            model_name: Model path for cache isolation.

        Returns:
            Cached mx.array features, or None on miss.
        """
        key = _composite_key(model_name, image_hash)

        # Check memory LRU
        with self._memory_lock:
            if key in self._memory_cache:
                self._memory_cache.move_to_end(key)
                self._stats["hits"] += 1
                return self._memory_cache[key]

        # Check SSD
        if self._cache_dir is not None:
            features = self._load_from_ssd(key)
            if features is not None:
                # Promote to memory cache
                with self._memory_lock:
                    self._memory_put(key, features)
                self._stats["hits"] += 1
                self._stats["ssd_loads"] += 1
                return features

        self._stats["misses"] += 1
        return None

    def put(
        self,
        image_hash: str,
        model_name: str,
        features: Any,
    ) -> None:
        """Store vision features in the cache.

        Must be called after mx.eval(features) on the MLX executor thread.

        Args:
            image_hash: SHA256 hash from compute_image_hash().
            model_name: Model path for cache isolation.
            features: Evaluated mx.array (or list of mx.array for multi-image).
        """
        key = _composite_key(model_name, image_hash)

        # Store in memory LRU
        with self._memory_lock:
            self._memory_put(key, features)

        # Enqueue SSD write
        if self._cache_dir is not None:
            self._enqueue_ssd_write(key, image_hash, model_name, features)

        self._stats["saves"] += 1

    def close(self) -> None:
        """Shut down the background writer and flush pending writes."""
        self._writer_shutdown.set()
        # Send sentinel to unblock the writer
        try:
            self._write_queue.put_nowait(None)
        except queue.Full:
            pass
        self._writer_thread.join(timeout=10.0)
        logger.debug(
            "Vision feature cache closed: %s",
            {k: v for k, v in self._stats.items() if v > 0},
        )

    @property
    def stats(self) -> Dict[str, int]:
        """Return a copy of cache statistics."""
        return dict(self._stats)

    # ── Memory LRU helpers ──────────────────────────────────────────

    def _memory_put(self, key: str, features: Any) -> None:
        """Insert into memory LRU, evicting oldest if over limit.

        Caller must hold _memory_lock.
        """
        if key in self._memory_cache:
            self._memory_cache.move_to_end(key)
            self._memory_cache[key] = features
            return

        # Evict oldest if over limit
        while len(self._memory_cache) >= self._max_memory_entries:
            self._memory_cache.popitem(last=False)

        self._memory_cache[key] = features

    # ── SSD persistence ─────────────────────────────────────────────

    def _file_path_for_key(self, key: str) -> Path:
        """Compute SSD file path from composite key."""
        h = hashlib.sha256(key.encode()).hexdigest()
        subdir = h[0]
        return self._cache_dir / subdir / f"{h}.safetensors"

    def _enqueue_ssd_write(
        self,
        key: str,
        image_hash: str,
        model_name: str,
        features: Any,
    ) -> None:
        """Extract tensor bytes and enqueue background SSD write."""
        with self._pending_lock:
            if key in self._pending_write_keys:
                return  # Already pending
            self._pending_write_keys.add(key)

        # Check if already on SSD
        with self._ssd_lock:
            if key in self._ssd_index:
                self._ssd_index[key].last_access = time.time()
                self._ssd_index.move_to_end(key)
                with self._pending_lock:
                    self._pending_write_keys.discard(key)
                return

        try:
            # Extract raw bytes on the Metal-safe thread
            tensors_raw: Dict[str, Tuple[bytes, str, List[int]]] = {}
            num_tensors = 1

            if isinstance(features, list):
                num_tensors = len(features)
                for i, feat in enumerate(features):
                    tensors_raw[f"feature_{i}"] = _extract_tensor_bytes(feat)
            else:
                tensors_raw["feature"] = _extract_tensor_bytes(features)

            metadata = {
                "image_hash": image_hash,
                "model_name": model_name,
                "num_tensors": str(num_tensors),
                "created_at": str(time.time()),
            }

            file_path = self._file_path_for_key(key)

            # Estimate file size for index
            estimated_size = sum(len(raw) for raw, _, _ in tensors_raw.values())

            # Add to index immediately (size updated after write)
            now = time.time()
            entry = VisionFeatureSSDEntry(
                image_hash=image_hash,
                model_name=model_name,
                file_path=file_path,
                file_size=estimated_size,
                created_at=now,
                last_access=now,
                num_tensors=num_tensors,
            )
            with self._ssd_lock:
                self._ssd_index[key] = entry
                self._ssd_total_size += estimated_size
                # Evict old entries if over limit
                self._evict_ssd_if_needed()

            # Enqueue write
            try:
                self._write_queue.put_nowait(
                    (key, tensors_raw, metadata, file_path)
                )
            except queue.Full:
                logger.debug("Vision cache write queue full, dropping write for %s", key[:32])
                with self._ssd_lock:
                    if key in self._ssd_index:
                        self._ssd_total_size -= self._ssd_index[key].file_size
                        del self._ssd_index[key]
                with self._pending_lock:
                    self._pending_write_keys.discard(key)

        except Exception as e:
            logger.debug("Failed to prepare vision feature for SSD write: %s", e)
            with self._pending_lock:
                self._pending_write_keys.discard(key)

    def _evict_ssd_if_needed(self) -> None:
        """Evict oldest SSD entries until total size is under limit.

        Caller must hold _ssd_lock.
        """
        while self._ssd_total_size > self._max_size_bytes and self._ssd_index:
            _, oldest = self._ssd_index.popitem(last=False)
            self._ssd_total_size -= oldest.file_size
            # Delete file in background (non-blocking)
            try:
                if oldest.file_path.exists():
                    oldest.file_path.unlink()
            except Exception:
                pass

    def _load_from_ssd(self, key: str) -> Optional[Any]:
        """Load cached features from SSD.

        Args:
            key: Composite cache key.

        Returns:
            mx.array (or list[mx.array]) if found, None otherwise.
        """
        with self._ssd_lock:
            entry = self._ssd_index.get(key)
            if entry is None:
                return None
            file_path = entry.file_path
            num_tensors = entry.num_tensors

        try:
            if not file_path.exists():
                # File was deleted externally
                with self._ssd_lock:
                    if key in self._ssd_index:
                        self._ssd_total_size -= self._ssd_index[key].file_size
                        del self._ssd_index[key]
                return None

            arrays = mx.load(str(file_path))

            if num_tensors == 1 and "feature" in arrays:
                features = arrays["feature"]
            else:
                features = []
                for i in range(num_tensors):
                    tensor_key = f"feature_{i}"
                    if tensor_key in arrays:
                        features.append(arrays[tensor_key])
                    else:
                        logger.warning(
                            "Missing tensor %s in %s", tensor_key, file_path
                        )
                        return None

            # Update access time
            with self._ssd_lock:
                if key in self._ssd_index:
                    self._ssd_index[key].last_access = time.time()
                    self._ssd_index.move_to_end(key)

            return features

        except Exception as e:
            logger.warning("Failed to load vision features from %s: %s", file_path, e)
            self._stats["errors"] += 1
            # Remove corrupted entry
            with self._ssd_lock:
                if key in self._ssd_index:
                    self._ssd_total_size -= self._ssd_index[key].file_size
                    del self._ssd_index[key]
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass
            return None

    def _scan_existing_files(self) -> None:
        """Scan SSD cache directory for existing files and rebuild index."""
        if self._cache_dir is None:
            return

        scanned = 0
        indexed = 0
        errors = 0

        for subdir_char in _SUBDIR_CHARS:
            subdir_path = self._cache_dir / subdir_char
            if not subdir_path.exists():
                continue

            for file_path in subdir_path.glob("*.safetensors"):
                scanned += 1
                try:
                    _, metadata = mx.load(str(file_path), return_metadata=True)
                    image_hash = metadata.get("image_hash", "")
                    model_name = metadata.get("model_name", "")
                    num_tensors = int(metadata.get("num_tensors", "1"))

                    if not image_hash or not model_name:
                        errors += 1
                        continue

                    key = _composite_key(model_name, image_hash)
                    file_stat = file_path.stat()

                    entry = VisionFeatureSSDEntry(
                        image_hash=image_hash,
                        model_name=model_name,
                        file_path=file_path,
                        file_size=file_stat.st_size,
                        created_at=file_stat.st_ctime,
                        last_access=file_stat.st_mtime,
                        num_tensors=num_tensors,
                    )
                    self._ssd_index[key] = entry
                    self._ssd_total_size += file_stat.st_size
                    indexed += 1

                except Exception as e:
                    logger.debug("Failed to read vision cache file %s: %s", file_path, e)
                    errors += 1

        if scanned > 0:
            logger.info(
                "Vision feature SSD cache scan: scanned=%d, indexed=%d, errors=%d, "
                "total_size=%.1fMB",
                scanned, indexed, errors, self._ssd_total_size / (1024 * 1024),
            )

    # ── Background writer ───────────────────────────────────────────

    def _writer_loop(self) -> None:
        """Background writer thread. Writes safetensors files using pure Python I/O."""
        while True:
            try:
                item = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                if self._writer_shutdown.is_set():
                    break
                continue

            if item is None:  # Sentinel for shutdown
                break

            key, tensors_raw, metadata, file_path = item
            temp_path = None

            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = file_path.with_name(
                    file_path.stem + "_tmp.safetensors"
                )
                actual_size = _write_safetensors_no_mx(
                    str(temp_path), tensors_raw, metadata
                )

                # Atomic rename
                os.rename(str(temp_path), str(file_path))

                # Update index with actual file size
                with self._ssd_lock:
                    if key in self._ssd_index:
                        old_size = self._ssd_index[key].file_size
                        self._ssd_index[key].file_size = actual_size
                        self._ssd_total_size += actual_size - old_size

            except Exception as e:
                if isinstance(e, OSError) and e.errno in (errno.ENOSPC, errno.EDQUOT):
                    logger.warning("Vision cache disk full: %s", e)
                else:
                    logger.warning("Vision cache background write failed: %s", e)
                self._stats["errors"] += 1
                # Remove from index since file wasn't written
                with self._ssd_lock:
                    if key in self._ssd_index:
                        self._ssd_total_size -= self._ssd_index[key].file_size
                        del self._ssd_index[key]
                # Clean up temp/final files
                for p in (temp_path, file_path):
                    try:
                        if p is not None and p.exists():
                            p.unlink()
                    except Exception:
                        pass
            finally:
                with self._pending_lock:
                    self._pending_write_keys.discard(key)
