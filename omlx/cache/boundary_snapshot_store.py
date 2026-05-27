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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .paged_ssd_cache import (
    HAS_MLX,
    _encode_shape,
    _extract_tensor_bytes,
    _has_zero_dim,
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

    # Timeouts applied when acquiring _writer_busy from each cleanup
    # path. cleanup_request is called from the scheduler's abort hot
    # path (~3 sites) and must yield faster than cleanup_all, which
    # also runs at startup / reset where blocking longer is tolerable
    # in exchange for a stronger orphan-avoidance guarantee. The
    # worst-case impact on the timeout fallback is identical in both
    # paths — an orphan file in the recreated dir until the next
    # constructor cleanup — so the only knob is per-call latency.
    _CLEANUP_ALL_TIMEOUT_S = 5.0
    _CLEANUP_REQUEST_TIMEOUT_S = 2.0

    def __init__(self, base_dir: Path) -> None:
        self._snapshot_dir = base_dir / "_boundary_snapshots"
        # Clean up orphaned files from previous crashes.
        if self._snapshot_dir.exists():
            try:
                shutil.rmtree(self._snapshot_dir)
            except Exception as e:
                logger.warning("Failed to clean up orphaned boundary snapshots: %s", e)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        # request_id -> {token_count -> file_path}
        self._file_registry: dict[str, dict[int, Path]] = {}
        self._registry_lock = threading.Lock()

        # Pending writes buffer — raw bytes for instant read-back.
        # key: (request_id, token_count)
        self._pending_writes: dict[tuple[str, int], dict] = {}
        self._pending_lock = threading.Lock()

        # Cancelled requests with remaining queue item counts. Writer
        # thread decrements on each skip; entry is deleted when count
        # reaches zero, preventing unbounded growth. All access is
        # guarded by ``_cancelled_lock`` — the dict was previously
        # mutated unlocked from cleanup_request, cleanup_all, and the
        # writer thread, creating lost-cancellation and counter-
        # underflow races.
        self._cancelled_requests: dict[str, int] = {}
        self._cancelled_lock = threading.Lock()

        # Background writer thread.
        self._write_queue: queue.Queue = queue.Queue(maxsize=_MAX_PENDING_WRITES)
        self._shutdown = threading.Event()
        # Held by the writer for the duration of each item's processing.
        # cleanup_all() acquires it after draining the queue so the writer
        # can't be mid-item (creating files inside the just-cleaned dir)
        # when rmtree runs.
        self._writer_busy = threading.Lock()
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
        snapshot_cache: list[Any],
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
                self._write_queue.put_nowait((pw_key, tensors_raw, metadata, file_path))
            except queue.Full:
                # Roll back the pending + registry entries: with no
                # queue item the writer can never decrement
                # _cancelled_requests for this entry, so if a later
                # cleanup_request counts it the rid stays pinned in
                # _cancelled_requests forever and every subsequent
                # save under that rid is silently discarded by the
                # _is_cancelled gates. The previous "stays in memory
                # only" promise was already broken because cleanup
                # discards the in-memory copy anyway.
                logger.warning(
                    "Boundary snapshot write queue full, dropping "
                    "snapshot %s/%d",
                    request_id,
                    token_count,
                )
                with self._pending_lock:
                    self._pending_writes.pop(pw_key, None)
                with self._registry_lock:
                    req_files = self._file_registry.get(request_id)
                    if req_files is not None:
                        req_files.pop(token_count, None)
                        if not req_files:
                            self._file_registry.pop(request_id, None)
                return False

            return True

        except Exception as e:
            logger.debug("Failed to save boundary snapshot: %s", e)
            return False

    def load(
        self,
        request_id: str,
        token_count: int,
    ) -> list[dict[str, Any]] | None:
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
                request_id,
                token_count,
                e,
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
        """Delete all snapshot files and pending writes for a request.

        Caller must guarantee no async store_cache worker is still reading
        snapshots for this request — concurrent ``rmtree`` here would race
        the worker's :meth:`load` calls and silently strip block storage.
        :class:`omlx.scheduler.Scheduler` defers this call until the
        ``store_future`` for ``request_id`` is done.

        Acquires ``_writer_busy`` after marking the request cancelled so
        the writer thread can finish any item it is mid-processing first.
        Without this barrier the writer can pull an item, ``mkdir`` the
        request directory, write its temp file, then ``os.rename`` it
        into the final path *after* we have rmtree'd — leaving an
        orphaned file behind. The ``_cancelled_requests`` counter (held
        under ``_cancelled_lock``) catches the late-rename case if
        ``_writer_busy.acquire`` times out.

        Bounded with a timeout so a stuck I/O on the writer thread
        cannot deadlock request abort paths (called from scheduler's
        hot path at ~3 sites).

        The cancelled-counter is bumped additively and only when at
        least one pending item exists for the rid — see the inline
        comment at the bump site for the two distinct bugs that
        rules out (stale ``rid: 0`` after a timeout for an empty
        cleanup, and overwrites racing with re-entrant cleanup_request
        calls for the same rid).
        """
        if self._shutdown.is_set():
            # After shutdown the writer no longer reacquires
            # _writer_busy per-item, so cleanup_request cannot
            # synchronise with it. Best-effort: just drop in-memory
            # state. Files (if any leaked through shutdown) are removed
            # by the next constructor cleanup_all.
            with self._pending_lock:
                for k in [k for k in self._pending_writes if k[0] == request_id]:
                    del self._pending_writes[k]
            with self._registry_lock:
                self._file_registry.pop(request_id, None)
            logger.warning(
                "cleanup_request(%s) called after shutdown — running "
                "in-memory-only", request_id,
            )
            return

        # Atomically: count pending items for this rid, drop them, mark
        # the rid cancelled. Holding both locks during the snapshot is
        # required to keep the counter consistent with what the writer
        # will see — a save() call from another thread cannot interleave
        # an enqueue between our count and our cancellation mark.
        #
        # The bump is additive (``get + count``) and skipped entirely
        # when ``count == 0``. Both rules close real bugs:
        #   * Skip-on-zero: cleanup_request("X") for an rid with no
        #     pending items previously wrote ``cancelled[X] = 0`` then
        #     popped it on the acquired path. On the timeout fallback
        #     the pop never runs and the ``X: 0`` entry lingers for
        #     the process lifetime — every subsequent save() under
        #     that rid (or any later reuse of the same string) is
        #     discarded by the writer's ``_is_cancelled`` gates,
        #     which check key membership not value > 0. The counter
        #     must only exist when there is at least one in-flight
        #     item to drain it.
        #   * Additive: a re-entrant cleanup_request("X") for an rid
        #     that already has an in-flight cancellation must NOT
        #     overwrite the previous count. The writer's
        #     ``cleared_by_cleanup`` branch + ``_writer_busy`` lock
        #     together close the file-write race today, but the
        #     per-item dec_cancelled bookkeeping still has to balance.
        #     Overwriting drops the remaining decs on the floor; on
        #     the next ``save()`` under the same rid the writer would
        #     see a non-zero counter from the earlier batch and
        #     silently discard the new item.
        with self._pending_lock:
            keys_to_remove = [k for k in self._pending_writes if k[0] == request_id]
            count = len(keys_to_remove)
            for key in keys_to_remove:
                del self._pending_writes[key]
            if count > 0:
                with self._cancelled_lock:
                    self._cancelled_requests[request_id] = (
                        self._cancelled_requests.get(request_id, 0) + count
                    )

        # Remove from registry.
        with self._registry_lock:
            self._file_registry.pop(request_id, None)

        # Wait briefly for the writer to finish any item it had already
        # pulled. If it's genuinely stuck (slow disk, dead thread) fall
        # back to the cancelled-counter rescue rather than blocking the
        # caller.
        acquired = self._writer_busy.acquire(
            timeout=self._CLEANUP_REQUEST_TIMEOUT_S
        )
        try:
            # Remove files.
            req_dir = self._snapshot_dir / request_id
            if req_dir.exists():
                try:
                    shutil.rmtree(req_dir)
                except Exception as e:
                    logger.debug(
                        "Failed to clean up snapshots for %s: %s", request_id, e
                    )
        finally:
            if acquired:
                self._writer_busy.release()
                # Counter entry has done its job — we own the lock so all
                # _is_cancelled-gated work has either run or skipped. Drop
                # the counter so a future racing save() can't leave it
                # elevated forever. CRITICAL: only pop on the acquired
                # path. On timeout the writer is still mid-item and may
                # not yet have consulted ``_is_cancelled``; popping here
                # would defeat the late-rename rescue that the docstring
                # advertises as the timeout-fallback safety net.
                with self._cancelled_lock:
                    self._cancelled_requests.pop(request_id, None)
            else:
                logger.warning(
                    "cleanup_request(%s): writer thread did not yield "
                    "within %.1fs; relying on cancelled-counter rescue "
                    "for late-rename safety",
                    request_id,
                    self._CLEANUP_REQUEST_TIMEOUT_S,
                )

    def cleanup_all(self) -> None:
        """Delete all snapshot files (for reset/startup).

        Synchronizes with the background writer: we drain the queue to
        prevent it from starting a new item, then acquire ``_writer_busy``
        to wait until any item it had already pulled finishes. Without
        this barrier the writer can create ``req-X/temp.safetensors``
        and ``os.rename`` it to its final path *after* we've already
        rmtree'd and recreated the snapshot directory, leaving an
        orphaned file behind.

        Threading: concurrent ``save()`` is safe because the writer
        consults ``_pending_writes`` and ``_is_cancelled`` while
        holding ``_writer_busy``, and ``cleanup_all`` clears both
        under the same lock before rmtree. The earlier "must run on
        the save() thread" constraint is therefore no longer required.

        Invariant enforcement: ``cleanup_all`` must run BEFORE
        ``shutdown()`` to actually synchronise with the writer. Once
        ``_shutdown`` is set the writer drops the per-item
        ``_writer_busy`` acquire, so a post-shutdown ``cleanup_all``
        cannot block on the writer and degrades to an in-memory
        clear. Callers that need both should pass ``shutdown(
        cleanup=True)`` instead of sequencing the calls themselves.
        """
        if self._shutdown.is_set():
            # See cleanup_request: best-effort in-memory clear only.
            with self._pending_lock:
                self._pending_writes.clear()
            with self._registry_lock:
                self._file_registry.clear()
            with self._cancelled_lock:
                self._cancelled_requests.clear()
            logger.warning(
                "cleanup_all called after shutdown — running in-memory-only; "
                "callers wanting on-disk cleanup should use "
                "shutdown(cleanup=True) instead"
            )
            return

        # Drain write queue so the writer thread doesn't process stale
        # items after the directory is deleted. Put_nowait the sentinel
        # back so shutdown still sees it; on Full just drop and let
        # shutdown re-issue.
        while True:
            try:
                item = self._write_queue.get_nowait()
                if item is None:  # Sentinel — put it back for shutdown.
                    try:
                        self._write_queue.put_nowait(item)
                    except queue.Full:
                        # Drop the sentinel; shutdown will re-enqueue.
                        # If cleanup_all is the LAST call before process
                        # exit without an explicit shutdown(), the writer
                        # thread will only be reaped on daemon teardown.
                        logger.debug(
                            "cleanup_all: dropped writer-sentinel on Full"
                        )
                    break
            except queue.Empty:
                break

        # Wait for the writer to finish any item it had already pulled.
        # When we own _writer_busy the writer is between items, and we
        # just drained the queue so no new item can start. Bounded so a
        # stuck writer (slow disk, dead thread) cannot deadlock callers
        # — scheduler calls cleanup_all() from its abort / reset hot
        # path. After the timeout we proceed anyway: the worst case is
        # an orphaned file in the recreated directory, which next
        # startup's cleanup_all() will clear.
        acquired = self._writer_busy.acquire(
            timeout=self._CLEANUP_ALL_TIMEOUT_S
        )
        try:
            if not acquired:
                logger.warning(
                    "cleanup_all: writer thread did not yield within "
                    "%.1fs; proceeding with rmtree — late-rename may "
                    "orphan a file under the recreated snapshot dir "
                    "until next startup.",
                    self._CLEANUP_ALL_TIMEOUT_S,
                )
            with self._pending_lock:
                self._pending_writes.clear()
            with self._registry_lock:
                self._file_registry.clear()
            with self._cancelled_lock:
                # Only safe to clear when we own _writer_busy — otherwise
                # a writer mid-_dec_cancelled would race. On timeout we
                # leave the counter intact so the rescue path stays
                # effective for in-flight items.
                if acquired:
                    self._cancelled_requests.clear()

            if self._snapshot_dir.exists():
                try:
                    shutil.rmtree(self._snapshot_dir)
                except Exception as e:
                    logger.debug(
                        "Failed to clean up all boundary snapshots: %s", e
                    )
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        finally:
            if acquired:
                self._writer_busy.release()

    def shutdown(self, *, cleanup: bool = False) -> None:
        """Stop background writer thread.

        Parameters
        ----------
        cleanup : bool
            When True, run ``cleanup_all()`` first, then signal shutdown.
            This enforces the cleanup-before-shutdown ordering invariant
            in one call. Callers that pass ``cleanup=False`` (the
            default) and *also* want cleanup MUST call ``cleanup_all()``
            themselves before ``shutdown()``.

        Invariant: if a caller wants to combine ``cleanup_all()`` with
        shutdown, the cleanup MUST run BEFORE ``_shutdown.set()`` /
        sentinel-enqueue. Once the sentinel is in the queue and the
        writer has consumed it, the writer no longer reacquires
        ``_writer_busy`` after each item, so a subsequent
        ``cleanup_all`` would wait its full 5s timeout if the writer is
        already mid-final-item, then proceed unsynchronised. The
        ``cleanup=True`` path handles this ordering; the warning at
        the top of ``cleanup_all`` catches misordered callers.
        """
        if cleanup:
            self.cleanup_all()
        self._shutdown.set()
        try:
            self._write_queue.put_nowait(None)  # Sentinel
        except queue.Full:
            pass
        self._writer_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_cancelled(self, request_id: str) -> bool:
        """Thread-safe check for cancellation."""
        with self._cancelled_lock:
            return request_id in self._cancelled_requests

    def _dec_cancelled(self, request_id: str) -> None:
        """Decrement cancelled counter under lock; remove entry when
        exhausted. Atomic read-modify-write closes the underflow race
        between two writer-thread iterations / cleanup_all clears."""
        with self._cancelled_lock:
            remaining = self._cancelled_requests.get(request_id, 0) - 1
            if remaining <= 0:
                self._cancelled_requests.pop(request_id, None)
            else:
                self._cancelled_requests[request_id] = remaining

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

            # Hold _writer_busy for the entire item's lifetime so
            # cleanup_all() can serialize with us — otherwise it can
            # rmtree the snapshot directory while we're mid-write and
            # we'd recreate ``req-X/`` underneath it, leaving an
            # orphaned file after the cleanup returns.
            with self._writer_busy:
                self._process_write_item(item)

    def _process_write_item(self, item) -> None:
        """Process one (pw_key, tensors_raw, metadata, file_path) queue item.

        Extracted from ``_writer_loop`` so the busy-lock can wrap it
        cleanly. Called only on the writer thread.
        """
        pw_key, tensors_raw, metadata, file_path = item

        # If cleanup_all or cleanup_request cleared this key from
        # _pending_writes while the item was in the writer's local hand
        # (i.e. between ``get()`` and entering ``with _writer_busy``),
        # treat the write as cancelled. This closes the late-rename
        # window where cleanup runs entirely between the writer's pull
        # and its busy-lock acquisition.
        with self._pending_lock:
            cleared_by_cleanup = pw_key not in self._pending_writes
        if cleared_by_cleanup:
            # If a timed-out cleanup_request bumped ``_cancelled_requests``
            # before clearing pending_writes, this item is one of the N
            # the counter is waiting on. Without this decrement the
            # counter would never reach zero, leaving the rid pinned in
            # ``_cancelled_requests`` for the process lifetime and
            # causing every subsequent write under that rid (or any
            # later reuse of the same string) to be silently discarded.
            if self._is_cancelled(pw_key[0]):
                self._dec_cancelled(pw_key[0])
            return

        # Skip writes for cancelled/cleaned-up requests.
        if self._is_cancelled(pw_key[0]):
            with self._pending_lock:
                self._pending_writes.pop(pw_key, None)
            try:
                req_dir = file_path.parent
                if req_dir.exists():
                    shutil.rmtree(req_dir)
            except Exception:
                pass
            self._dec_cancelled(pw_key[0])
            return

        temp_path = None
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = file_path.with_name(file_path.stem + "_tmp.safetensors")
            _write_safetensors_no_mx(str(temp_path), tensors_raw, metadata)

            # Request may have been cleaned up while serializing.
            if self._is_cancelled(pw_key[0]):
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                with self._pending_lock:
                    self._pending_writes.pop(pw_key, None)
                self._dec_cancelled(pw_key[0])
                return

            os.rename(str(temp_path), str(file_path))

            # Cleanup may race with a queued write; remove any late file.
            if self._is_cancelled(pw_key[0]):
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception:
                    pass
                req_dir = file_path.parent
                try:
                    if req_dir.exists():
                        shutil.rmtree(req_dir)
                except Exception:
                    pass
                self._dec_cancelled(pw_key[0])
        except Exception as e:
            logger.debug("Background snapshot write failed: %s", e)
            for p in (temp_path, file_path):
                try:
                    if p is not None and p.exists():
                        p.unlink()
                except Exception:
                    pass
            # Same bookkeeping invariant as the early-return path: if
            # cleanup_request bumped the counter and the failure was a
            # side-effect of that cleanup (e.g. its rmtree pulled the
            # parent dir out from under our temp write), we still owe
            # one decrement. The _is_cancelled rescue blocks above all
            # return before this except clause runs, so we cannot
            # double-decrement.
            if self._is_cancelled(pw_key[0]):
                self._dec_cancelled(pw_key[0])
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
        extracted: list[dict[str, Any]],
        request_id: str,
        token_count: int,
    ) -> tuple[dict[str, tuple[bytes, str, list[int]]], dict[str, str]]:
        """Convert extracted cache states to tensors_raw + metadata.

        Must be called on the inference thread (for mx.eval / _extract_tensor_bytes).
        """
        arrays: dict[str, Any] = {}  # name -> mx.array
        layer_info: list[dict[str, str]] = []

        for i, layer_state in enumerate(extracted):
            class_name = layer_state.get("class_name", "KVCache")
            cache_type = layer_state.get("cache_type", "KVCache")
            meta_state = layer_state.get("meta_state", ())
            state = layer_state.get("state", ())

            info: dict[str, str] = {
                "class_name": class_name,
                "cache_type": cache_type,
                "meta_state": json.dumps(list(meta_state) if meta_state else []),
            }

            if (
                isinstance(state, list)
                and len(state) >= 1
                and all(isinstance(s, (list, tuple)) for s in state)
            ):
                # CacheList layer: ``state`` is a list of nested sub-state
                # tuples (one per sub-cache, e.g. RotatingKVCache +
                # PoolingCache for DeepSeek V4). Flatten as
                # ``layer_{i}_sub_{j}_state_{k}`` keys so reconstruction
                # can rebuild the nested shape.
                info["has_state"] = "true"
                info["sub_count"] = str(len(state))
                for j, sub_state in enumerate(state):
                    info[f"sub_{j}_count"] = str(len(sub_state))
                    for k, elem in enumerate(sub_state):
                        if not hasattr(elem, "shape"):
                            info[f"sub_{j}_missing_{k}"] = "1"
                            continue
                        if _has_zero_dim(elem):
                            arrays[f"layer_{i}_sub_{j}_state_{k}"] = mx.zeros((1,))
                            info[f"sub_{j}_zero_dim_{k}"] = _encode_shape(elem.shape)
                        else:
                            arrays[f"layer_{i}_sub_{j}_state_{k}"] = elem
            elif isinstance(state, (list, tuple)) and len(state) >= 1:
                # Flat N-tuple state (KVCache, RotatingKVCache, PoolingCache,
                # BatchKVCache). Store every element under
                # ``layer_{i}_state_{k}`` regardless of tuple length.
                has_tensors = any(hasattr(elem, "shape") for elem in state)
                if has_tensors:
                    info["has_state"] = "true"
                    info["state_count"] = str(len(state))
                    for k, elem in enumerate(state):
                        if not hasattr(elem, "shape"):
                            # Non-tensor element (None, scalar). Mark it so
                            # _deserialize can restore the gap.
                            info[f"missing_{k}"] = "1"
                            continue
                        if _has_zero_dim(elem):
                            arrays[f"layer_{i}_state_{k}"] = mx.zeros((1,))
                            info[f"zero_dim_{k}"] = _encode_shape(elem.shape)
                        else:
                            arrays[f"layer_{i}_state_{k}"] = elem
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
        tensors_raw: dict[str, tuple[bytes, str, list[int]]],
        metadata: dict[str, str],
    ) -> list[dict[str, Any]] | None:
        """Reconstruct extracted cache states from raw bytes + metadata."""
        try:
            num_layers = int(metadata["num_layers"])
            layer_info = json.loads(metadata["layer_info"])
        except (KeyError, ValueError, json.JSONDecodeError):
            return None

        result: list[dict[str, Any]] = []
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
                # V3 path: state_count meta + layer_{i}_state_{k} keys.
                # V2 fallback: legacy layer_{i}_0/1 + zero_dim_0/1 keys
                # for snapshots written before the N-tuple migration.
                state = self._read_state_tuple_raw(tensors_raw, info, i)
                result.append(
                    {
                        "state": state,
                        "meta_state": meta_state,
                        "class_name": class_name,
                        "cache_type": cache_type,
                    }
                )
            else:
                # Placeholder for skipped sliceable layers.
                result.append(
                    {
                        "state": (),
                        "meta_state": meta_state,
                        "class_name": class_name,
                        "cache_type": cache_type,
                    }
                )

        return result

    def _read_state_tuple_raw(
        self,
        tensors_raw: dict[str, tuple[bytes, str, list[int]]],
        info: dict[str, str],
        layer_idx: int,
    ) -> Any:
        """Read state for one layer from raw tensor bytes.

        Returns:
            - ``list`` of nested sub-state tuples for CacheList layers
              (``sub_count`` in info), or
            - ``tuple`` of N elements for flat layers (``state_count`` in
              info, V3 layout), or
            - 2-tuple from V2 polyfill (``layer_{i}_0`` / ``layer_{i}_1``).

        Missing elements come back as ``None``.
        """
        if "sub_count" in info:
            try:
                sub_count = int(info["sub_count"])
            except (ValueError, TypeError):
                return []
            sub_states: list[tuple[Any, ...]] = []
            for j in range(sub_count):
                count_key = f"sub_{j}_count"
                try:
                    count = int(info.get(count_key, "0"))
                except (ValueError, TypeError):
                    count = 0
                sub_elements: list[Any] = []
                for k in range(count):
                    if info.get(f"sub_{j}_missing_{k}") == "1":
                        sub_elements.append(None)
                        continue
                    key = f"layer_{layer_idx}_sub_{j}_state_{k}"
                    if key not in tensors_raw:
                        sub_elements.append(None)
                        continue
                    raw, dtype_str, shape = tensors_raw[key]
                    zd_marker = f"sub_{j}_zero_dim_{k}"
                    if zd_marker in info:
                        zd_shape = tuple(int(d) for d in info[zd_marker].split(","))
                        restored = _restore_tensor_from_bytes(raw, dtype_str, [1])
                        sub_elements.append(mx.zeros(zd_shape, dtype=restored.dtype))
                    else:
                        sub_elements.append(
                            _restore_tensor_from_bytes(raw, dtype_str, shape)
                        )
                sub_states.append(tuple(sub_elements))
            return sub_states

        if "state_count" in info:
            try:
                count = int(info["state_count"])
            except (ValueError, TypeError):
                return ()
            elements: list[Any] = []
            for k in range(count):
                if info.get(f"missing_{k}") == "1":
                    elements.append(None)
                    continue
                key = f"layer_{layer_idx}_state_{k}"
                if key not in tensors_raw:
                    elements.append(None)
                    continue
                raw, dtype_str, shape = tensors_raw[key]
                zd_marker = f"zero_dim_{k}"
                if zd_marker in info:
                    zd_shape = tuple(int(d) for d in info[zd_marker].split(","))
                    restored = _restore_tensor_from_bytes(raw, dtype_str, [1])
                    elements.append(mx.zeros(zd_shape, dtype=restored.dtype))
                else:
                    elements.append(_restore_tensor_from_bytes(raw, dtype_str, shape))
            return tuple(elements)

        # V2 polyfill — legacy 2-tuple snapshot.
        first = None
        second = None
        key_0 = f"layer_{layer_idx}_0"
        key_1 = f"layer_{layer_idx}_1"
        if key_0 in tensors_raw:
            raw, dtype_str, shape = tensors_raw[key_0]
            if "zero_dim_0" in info:
                zd_shape = tuple(int(d) for d in info["zero_dim_0"].split(","))
                first = _restore_tensor_from_bytes(raw, dtype_str, [1])
                first = mx.zeros(zd_shape, dtype=first.dtype)
            else:
                first = _restore_tensor_from_bytes(raw, dtype_str, shape)
        if key_1 in tensors_raw:
            raw, dtype_str, shape = tensors_raw[key_1]
            if "zero_dim_1" in info:
                zd_shape = tuple(int(d) for d in info["zero_dim_1"].split(","))
                second = _restore_tensor_from_bytes(raw, dtype_str, [1])
                second = mx.zeros(zd_shape, dtype=second.dtype)
            else:
                second = _restore_tensor_from_bytes(raw, dtype_str, shape)
        return (first, second) if first is not None else ()

    def _reconstruct_from_safetensors(
        self,
        arrays: dict[str, Any],
        metadata: dict[str, str],
    ) -> list[dict[str, Any]] | None:
        """Reconstruct from mx.load() result (arrays dict + metadata)."""
        try:
            num_layers = int(metadata["num_layers"])
            layer_info = json.loads(metadata["layer_info"])
        except (KeyError, ValueError, json.JSONDecodeError):
            return None

        result: list[dict[str, Any]] = []
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
                state = self._read_state_tuple_arrays(arrays, info, i)
                result.append(
                    {
                        "state": state,
                        "meta_state": meta_state,
                        "class_name": class_name,
                        "cache_type": cache_type,
                    }
                )
            else:
                result.append(
                    {
                        "state": (),
                        "meta_state": meta_state,
                        "class_name": class_name,
                        "cache_type": cache_type,
                    }
                )

        return result

    def _read_state_tuple_arrays(
        self,
        arrays: dict[str, Any],
        info: dict[str, str],
        layer_idx: int,
    ) -> Any:
        """N-tuple aware safetensors-loaded variant of
        ``_read_state_tuple_raw`` — sources tensors from a pre-decoded
        ``mx.array`` dict instead of raw bytes. Returns a list of nested
        tuples for CacheList layers (``sub_count`` in info) or a flat
        tuple otherwise.
        """
        if "sub_count" in info:
            try:
                sub_count = int(info["sub_count"])
            except (ValueError, TypeError):
                return []
            sub_states: list[tuple[Any, ...]] = []
            for j in range(sub_count):
                count_key = f"sub_{j}_count"
                try:
                    count = int(info.get(count_key, "0"))
                except (ValueError, TypeError):
                    count = 0
                sub_elements: list[Any] = []
                for k in range(count):
                    if info.get(f"sub_{j}_missing_{k}") == "1":
                        sub_elements.append(None)
                        continue
                    key = f"layer_{layer_idx}_sub_{j}_state_{k}"
                    tensor = arrays.get(key)
                    if tensor is None:
                        sub_elements.append(None)
                        continue
                    zd_marker = f"sub_{j}_zero_dim_{k}"
                    if zd_marker in info:
                        zd_shape = tuple(int(d) for d in info[zd_marker].split(","))
                        sub_elements.append(mx.zeros(zd_shape, dtype=tensor.dtype))
                    else:
                        sub_elements.append(tensor)
                sub_states.append(tuple(sub_elements))
            return sub_states

        if "state_count" in info:
            try:
                count = int(info["state_count"])
            except (ValueError, TypeError):
                return ()
            elements: list[Any] = []
            for k in range(count):
                if info.get(f"missing_{k}") == "1":
                    elements.append(None)
                    continue
                key = f"layer_{layer_idx}_state_{k}"
                tensor = arrays.get(key)
                if tensor is None:
                    elements.append(None)
                    continue
                zd_marker = f"zero_dim_{k}"
                if zd_marker in info:
                    zd_shape = tuple(int(d) for d in info[zd_marker].split(","))
                    elements.append(mx.zeros(zd_shape, dtype=tensor.dtype))
                else:
                    elements.append(tensor)
            return tuple(elements)

        # V2 polyfill.
        first = arrays.get(f"layer_{layer_idx}_0")
        second = arrays.get(f"layer_{layer_idx}_1")
        if "zero_dim_0" in info and first is not None:
            zd_shape = tuple(int(d) for d in info["zero_dim_0"].split(","))
            first = mx.zeros(zd_shape, dtype=first.dtype)
        if "zero_dim_1" in info and second is not None:
            zd_shape = tuple(int(d) for d in info["zero_dim_1"].split(","))
            second = mx.zeros(zd_shape, dtype=second.dtype)
        return (first, second) if first is not None else ()
