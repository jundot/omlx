# SPDX-License-Identifier: Apache-2.0
"""
Boundary Snapshot SSD Store for oMLX.

Stores non-sliceable cache layer snapshots (e.g. ArraysCache) to SSD during
prefill, freeing GPU memory immediately.  At request completion the snapshots
are loaded back one block at a time for final SSD cache storage.

Uses the same async-write pattern as PagedSSDCacheManager: tensors are
serialized to raw bytes on the inference thread (Metal-safe), buffered in
``_pending_writes`` for instant read-back, and flushed to disk by a
background writer thread via ``_write_safetensors_no_mx``.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .paged_ssd_cache import (
    HAS_MLX,
    _extract_tensor_bytes,
    _has_zero_dim,
    _encode_shape,
    _restore_tensor_from_bytes,
    _write_safetensors_no_mx,
)

if HAS_MLX:
    import mlx.core as mx

logger = logging.getLogger(__name__)

# Max pending writes before save() blocks.
_MAX_PENDING_WRITES = 128


class BoundarySnapshotSSDStore:
    """Temporary SSD storage for boundary cache snapshots.

    Stores ArraysCache/RotatingKVCache boundary snapshots to SSD during
    prefill to avoid GPU memory accumulation.  Files are ephemeral and
    cleaned up when the request completes or aborts.

    Parameters
    ----------
    base_dir : Path
        Parent directory for the SSD cache (typically ``paged_ssd_cache_dir``).
        Snapshots are stored under ``base_dir/_boundary_snapshots/``.
    """

    def __init__(self, base_dir: Path) -> None:
        self._snapshot_dir = base_dir / "_boundary_snapshots"
        # Clean up orphaned files from previous crashes.
        if self._snapshot_dir.exists():
            try:
                shutil.rmtree(self._snapshot_dir)
            except Exception as e:
                logger.warning(
                    "Failed to clean up orphaned boundary snapshots: %s", e
                )
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        # request_id -> {token_count -> file_path}
        self._file_registry: Dict[str, Dict[int, Path]] = {}
        self._registry_lock = threading.Lock()

        # Pending writes buffer — raw bytes for instant read-back.
        # key: (request_id, token_count)
        self._pending_writes: Dict[Tuple[str, int], Dict] = {}
        self._pending_lock = threading.Lock()

        # Background writer thread.
        self._write_queue: queue.Queue = queue.Queue(maxsize=_MAX_PENDING_WRITES)
        self._shutdown = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="boundary-snapshot-writer",
            daemon=True,
        )
        self._writer_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        request_id: str,
        token_count: int,
        snapshot_cache: List[Any],
        extract_cache_states_fn: Callable,
    ) -> bool:
        """Serialize snapshot to SSD (non-blocking).

        Must be called from the inference thread (Metal-safe for mx.eval).

        Parameters
        ----------
        request_id : str
            Unique request identifier.
        token_count : int
            Token boundary count.
        snapshot_cache : list
            Per-layer cache objects (None for skipped sliceable layers).
        extract_cache_states_fn : callable
            ``Scheduler._extract_cache_states`` — converts raw cache objects
            to ``List[Dict[str, Any]]``.

        Returns
        -------
        bool
            True if successfully enqueued for writing.
        """
        if not HAS_MLX:
            return False

        try:
            # 1. Extract dict-format states on inference thread.
            extracted, model_cache_config = extract_cache_states_fn(snapshot_cache)
            if not extracted:
                return False

            # 2. Flatten tensors + metadata for safetensors serialization.
            tensors_raw, metadata = self._serialize_extracted(
                extracted, request_id, token_count
            )

            # 3. Buffer in pending writes for instant read-back.
            pw_key = (request_id, token_count)
            with self._pending_lock:
                self._pending_writes[pw_key] = {
                    "tensors_raw": tensors_raw,
                    "metadata": metadata,
                    "extracted": extracted,  # keep for cheap read-back
                }

            # 4. Compute file path and register.
            file_path = self._file_path(request_id, token_count)
            with self._registry_lock:
                self._file_registry.setdefault(request_id, {})[token_count] = file_path

            # 5. Enqueue for background write.
            try:
                self._write_queue.put_nowait(
                    (pw_key, tensors_raw, metadata, file_path)
                )
            except queue.Full:
                logger.warning(
                    "Boundary snapshot write queue full, snapshot %s/%d "
                    "stays in memory only",
                    request_id,
                    token_count,
                )
                # Still returns True — data is in pending_writes for read-back.

            return True

        except Exception as e:
            logger.debug("Failed to save boundary snapshot: %s", e)
            return False

    def load(
        self,
        request_id: str,
        token_count: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Load a snapshot, returning extracted cache state dicts.

        Checks the in-memory pending-writes buffer first (zero I/O), then
        falls back to reading the safetensors file from disk.

        Returns
        -------
        list or None
            List of per-layer dicts matching ``_extract_cache_states`` output
            format, or None on failure.
        """
        pw_key = (request_id, token_count)

        # Fast path: still in pending writes buffer.
        with self._pending_lock:
            pending = self._pending_writes.get(pw_key)
            if pending is not None:
                extracted = pending.get("extracted")
                if extracted is not None:
                    return extracted

                # Fallback: reconstruct from raw bytes.
                tensors_raw = pending.get("tensors_raw")
                metadata = pending.get("metadata")
                if tensors_raw and metadata:
                    return self._deserialize(tensors_raw, metadata)

        # Slow path: read from disk.
        file_path = self._file_path(request_id, token_count)
        if not file_path.exists():
            return None

        try:
            data = mx.load(str(file_path), return_metadata=True)
            if isinstance(data, tuple) and len(data) == 2:
                arrays, metadata = data
            else:
                return None
            return self._reconstruct_from_safetensors(arrays, metadata)
        except Exception as e:
            logger.debug(
                "Failed to load boundary snapshot %s/%d: %s",
                request_id, token_count, e,
            )
            return None

    def has(self, request_id: str, token_count: int) -> bool:
        """Check if a snapshot exists (in memory or on disk)."""
        pw_key = (request_id, token_count)
        with self._pending_lock:
            if pw_key in self._pending_writes:
                return True
        with self._registry_lock:
            req_files = self._file_registry.get(request_id)
            if req_files and token_count in req_files:
                return True
        return False

    def cleanup_request(self, request_id: str) -> None:
        """Delete all snapshot files and pending writes for a request."""
        # Remove from pending writes.
        keys_to_remove = []
        with self._pending_lock:
            for key in self._pending_writes:
                if key[0] == request_id:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                del self._pending_writes[key]

        # Remove from registry.
        with self._registry_lock:
            self._file_registry.pop(request_id, None)

        # Remove files.
        req_dir = self._snapshot_dir / request_id
        if req_dir.exists():
            try:
                shutil.rmtree(req_dir)
            except Exception as e:
                logger.debug("Failed to clean up snapshots for %s: %s", request_id, e)

    def cleanup_all(self) -> None:
        """Delete all snapshot files (for reset/startup)."""
        with self._pending_lock:
            self._pending_writes.clear()
        with self._registry_lock:
            self._file_registry.clear()

        if self._snapshot_dir.exists():
            try:
                shutil.rmtree(self._snapshot_dir)
            except Exception as e:
                logger.debug("Failed to clean up all boundary snapshots: %s", e)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    def shutdown(self) -> None:
        """Stop background writer thread."""
        self._shutdown.set()
        try:
            self._write_queue.put_nowait(None)  # Sentinel
        except queue.Full:
            pass
        self._writer_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _file_path(self, request_id: str, token_count: int) -> Path:
        return self._snapshot_dir / request_id / f"{token_count}.safetensors"

    def _writer_loop(self) -> None:
        """Background thread that writes safetensors files."""
        while not self._shutdown.is_set():
            try:
                item = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:  # Sentinel
                break

            pw_key, tensors_raw, metadata, file_path = item
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = file_path.with_name(
                    file_path.stem + "_tmp.safetensors"
                )
                _write_safetensors_no_mx(str(temp_path), tensors_raw, metadata)
                os.rename(str(temp_path), str(file_path))
            except Exception as e:
                logger.debug("Background snapshot write failed: %s", e)
                for p in (temp_path, file_path):
                    try:
                        if p is not None and p.exists():
                            p.unlink()
                    except Exception:
                        pass
            finally:
                # Remove extracted cache objects from pending writes to free
                # memory, but keep tensors_raw for read-back until file is on
                # disk.
                with self._pending_lock:
                    pending = self._pending_writes.get(pw_key)
                    if pending is not None:
                        pending.pop("extracted", None)
                    # If file was written successfully, remove entirely.
                    if file_path.exists():
                        self._pending_writes.pop(pw_key, None)

    def _serialize_extracted(
        self,
        extracted: List[Dict[str, Any]],
        request_id: str,
        token_count: int,
    ) -> Tuple[Dict[str, Tuple[bytes, str, List[int]]], Dict[str, str]]:
        """Convert extracted cache states to tensors_raw + metadata.

        Must be called on the inference thread (for mx.eval / _extract_tensor_bytes).
        """
        arrays: Dict[str, Any] = {}  # name -> mx.array
        layer_info: List[Dict[str, str]] = []

        for i, layer_state in enumerate(extracted):
            class_name = layer_state.get("class_name", "KVCache")
            cache_type = layer_state.get("cache_type", "KVCache")
            meta_state = layer_state.get("meta_state", ())
            state = layer_state.get("state", ())

            info: Dict[str, str] = {
                "class_name": class_name,
                "cache_type": cache_type,
                "meta_state": json.dumps(
                    list(meta_state) if meta_state else []
                ),
            }

            if isinstance(state, (list, tuple)) and len(state) >= 2:
                first, second = state[0], state[1]
                has_tensors = hasattr(first, "shape") or hasattr(second, "shape")
                if has_tensors:
                    info["has_state"] = "true"
                    if hasattr(first, "shape"):
                        if _has_zero_dim(first):
                            arrays[f"layer_{i}_0"] = mx.zeros((1,))
                            info["zero_dim_0"] = _encode_shape(first.shape)
                        else:
                            arrays[f"layer_{i}_0"] = first
                    if hasattr(second, "shape"):
                        if _has_zero_dim(second):
                            arrays[f"layer_{i}_1"] = mx.zeros((1,))
                            info["zero_dim_1"] = _encode_shape(second.shape)
                        else:
                            arrays[f"layer_{i}_1"] = second
                else:
                    info["has_state"] = "false"
            else:
                info["has_state"] = "false"

            layer_info.append(info)

        # Materialize lazy tensors on inference thread.
        if arrays:
            mx.eval(*arrays.values())

        # Extract raw bytes (Metal-safe memoryview copy).
        tensors_raw = {}
        for name, arr in arrays.items():
            tensors_raw[name] = _extract_tensor_bytes(arr)

        metadata = {
            "request_id": request_id,
            "token_count": str(token_count),
            "num_layers": str(len(extracted)),
            "layer_info": json.dumps(layer_info),
        }

        return tensors_raw, metadata

    def _deserialize(
        self,
        tensors_raw: Dict[str, Tuple[bytes, str, List[int]]],
        metadata: Dict[str, str],
    ) -> Optional[List[Dict[str, Any]]]:
        """Reconstruct extracted cache states from raw bytes + metadata."""
        try:
            num_layers = int(metadata["num_layers"])
            layer_info = json.loads(metadata["layer_info"])
        except (KeyError, ValueError, json.JSONDecodeError):
            return None

        result: List[Dict[str, Any]] = []
        for i in range(num_layers):
            info = layer_info[i] if i < len(layer_info) else {}
            class_name = info.get("class_name", "KVCache")
            cache_type = info.get("cache_type", "KVCache")
            meta_state_json = info.get("meta_state", "[]")
            try:
                meta_state = tuple(json.loads(meta_state_json))
            except (ValueError, json.JSONDecodeError):
                meta_state = ()

            if info.get("has_state") == "true":
                first = None
                second = None
                key_0 = f"layer_{i}_0"
                key_1 = f"layer_{i}_1"

                if key_0 in tensors_raw:
                    raw, dtype_str, shape = tensors_raw[key_0]
                    if f"zero_dim_0" in info:
                        # Restore zero-dim tensor shape.
                        zd_shape = tuple(
                            int(d) for d in info["zero_dim_0"].split(",")
                        )
                        first = _restore_tensor_from_bytes(raw, dtype_str, [1])
                        first = mx.zeros(zd_shape, dtype=first.dtype)
                    else:
                        first = _restore_tensor_from_bytes(raw, dtype_str, shape)
                if key_1 in tensors_raw:
                    raw, dtype_str, shape = tensors_raw[key_1]
                    if f"zero_dim_1" in info:
                        zd_shape = tuple(
                            int(d) for d in info["zero_dim_1"].split(",")
                        )
                        second = _restore_tensor_from_bytes(raw, dtype_str, [1])
                        second = mx.zeros(zd_shape, dtype=second.dtype)
                    else:
                        second = _restore_tensor_from_bytes(raw, dtype_str, shape)

                state = (first, second) if first is not None else ()
                result.append({
                    "state": state,
                    "meta_state": meta_state,
                    "class_name": class_name,
                    "cache_type": cache_type,
                })
            else:
                # Placeholder for skipped sliceable layers.
                result.append({
                    "state": (),
                    "meta_state": meta_state,
                    "class_name": class_name,
                    "cache_type": cache_type,
                })

        return result

    def _reconstruct_from_safetensors(
        self,
        arrays: Dict[str, Any],
        metadata: Dict[str, str],
    ) -> Optional[List[Dict[str, Any]]]:
        """Reconstruct from mx.load() result (arrays dict + metadata)."""
        try:
            num_layers = int(metadata["num_layers"])
            layer_info = json.loads(metadata["layer_info"])
        except (KeyError, ValueError, json.JSONDecodeError):
            return None

        result: List[Dict[str, Any]] = []
        for i in range(num_layers):
            info = layer_info[i] if i < len(layer_info) else {}
            class_name = info.get("class_name", "KVCache")
            cache_type = info.get("cache_type", "KVCache")
            meta_state_json = info.get("meta_state", "[]")
            try:
                meta_state = tuple(json.loads(meta_state_json))
            except (ValueError, json.JSONDecodeError):
                meta_state = ()

            if info.get("has_state") == "true":
                first = arrays.get(f"layer_{i}_0")
                second = arrays.get(f"layer_{i}_1")

                # Restore zero-dim tensors.
                if "zero_dim_0" in info and first is not None:
                    zd_shape = tuple(
                        int(d) for d in info["zero_dim_0"].split(",")
                    )
                    first = mx.zeros(zd_shape, dtype=first.dtype)
                if "zero_dim_1" in info and second is not None:
                    zd_shape = tuple(
                        int(d) for d in info["zero_dim_1"].split(",")
                    )
                    second = mx.zeros(zd_shape, dtype=second.dtype)

                state = (first, second) if first is not None else ()
                result.append({
                    "state": state,
                    "meta_state": meta_state,
                    "class_name": class_name,
                    "cache_type": cache_type,
                })
            else:
                result.append({
                    "state": (),
                    "meta_state": meta_state,
                    "class_name": class_name,
                    "cache_type": cache_type,
                })

        return result
