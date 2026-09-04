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

import hashlib
import json
import logging
import math
import os
import queue
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager, suppress
from functools import lru_cache
from pathlib import Path
from typing import Any

from .paged_ssd_cache import (
    HAS_MLX,
    _encode_shape,
    _extract_tensor_bytes,
    _fsync_parent_dir,
    _has_zero_dim,
    _restore_tensor_from_bytes,
    _write_safetensors_no_mx,
)
from .pooling_delta import compact_pooling_cache_snapshot

if HAS_MLX:
    import mlx.core as mx

logger = logging.getLogger(__name__)

# Max pending writes before save() blocks.
_MAX_PENDING_WRITES = 128
_DEFAULT_PENDING_MAX_BYTES = 512 * 1024 * 1024
_PENDING_RESERVATION_TIMEOUT_S = 2.0
_GDN_SIDECAR_FORMAT_VERSION = "2"
_GDN_STATE_DTYPES = frozenset(
    {"fp32", "bf16", "int8", "rht_int8", "rht_int16"}
)
_GDN_INT8_CODEC = "int8_rowwise_last_axis_v1"
_GDN_RHT_INT8_CODEC = "rht_int8_rowwise_last_axis_v1"
_GDN_RHT_INT16_CODEC = "rht_int16_rowwise_last_axis_v1"
_GDN_RHT_SEED = 0
_GDN_BF16_CODEC = "bf16_v1"
_GDN_INT8_CODECS = frozenset({_GDN_INT8_CODEC, _GDN_RHT_INT8_CODEC})
_GDN_INT16_CODECS = frozenset({_GDN_RHT_INT16_CODEC})
_GDN_INTEGER_CODECS = frozenset(_GDN_INT8_CODECS | _GDN_INT16_CODECS)
_GDN_REDUCED_CODECS = frozenset(_GDN_INTEGER_CODECS | {_GDN_BF16_CODEC})
# Reduced codecs only ever encode the fp32 recurrent member, so a sidecar
# claiming any other source dtype was produced by a different (or tampered)
# writer and must not be silently reinterpreted as fp32.
_GDN_REQUIRED_ORIGINAL_DTYPE = "float32"


def _gdn_rht_dimension_supported(dim: int) -> bool:
    """Report whether the RHT codec can rotate this last-axis width.

    Pure metadata: the normalized Hadamard inverse is only exact for the
    power-of-two sizes this codec versions its sign diagonal for.
    """
    return isinstance(dim, int) and dim > 0 and not (dim & (dim - 1))


# One entry per distinct GDN width. A model contributes a single width, so the
# bound only exists to keep a pathological caller from growing the cache
# without limit; exceeding it costs recomputation, never correctness.
@lru_cache(maxsize=32)
def _gdn_rht_sign_values(dim: int, seed: int) -> tuple[float, ...]:
    """Return a version-stable random-sign diagonal without touching MLX RNG."""
    if not _gdn_rht_dimension_supported(dim):
        raise ValueError(
            f"GDN RHT requires a positive power-of-two last dimension, got {dim}"
        )
    prefix = f"omlx-gdn-rht-v1:{seed}:{dim}:".encode()
    return tuple(
        1.0
        if hashlib.sha256(prefix + index.to_bytes(4, "little")).digest()[0] & 1
        else -1.0
        for index in range(dim)
    )


def reset_boundary_snapshot_root(base_dir: Path) -> None:
    """Remove all boundary snapshot sessions for a server lifecycle boundary."""
    snapshot_root = base_dir / "_boundary_snapshots"
    if snapshot_root.is_symlink():
        logger.warning(
            "Refusing to reset symlinked boundary snapshot root: %s",
            snapshot_root,
        )
        return
    if snapshot_root.exists():
        try:
            shutil.rmtree(snapshot_root)
        except Exception as e:
            logger.warning("Failed to reset boundary snapshots: %s", e)
    snapshot_root.mkdir(parents=True, exist_ok=True)


class BoundarySnapshotSSDStore:
    """Temporary SSD storage for boundary cache snapshots.

    Stores ArraysCache/RotatingKVCache boundary snapshots to SSD during
    prefill to avoid GPU memory accumulation.  Files are ephemeral and
    cleaned up when the request completes or aborts.

    Parameters
    ----------
    base_dir : Path
        Parent directory for the SSD cache (typically ``paged_ssd_cache_dir``).
        Snapshots are stored under
        ``base_dir/_boundary_snapshots/<session_id>/<opaque_request_dir>/``.
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

    def __init__(
        self,
        base_dir: Path,
        *,
        pending_max_bytes: int = _DEFAULT_PENDING_MAX_BYTES,
        gdn_sidecar_state_dtype: str = "fp32",
    ) -> None:
        state_dtype = str(gdn_sidecar_state_dtype).lower()
        if state_dtype not in _GDN_STATE_DTYPES:
            raise ValueError(
                "gdn_sidecar_state_dtype must be one of: "
                "fp32, bf16, int8, rht_int8, rht_int16"
            )
        self._gdn_sidecar_state_dtype = state_dtype
        self._gdn_state_dequantizations = 0
        self._gdn_encode_failures = 0
        self._gdn_decode_failures = 0
        self._gdn_capability_fallbacks = 0
        self._gdn_capability_reported = False
        self._gdn_dequant_lock = threading.Lock()
        self._snapshot_root = base_dir / "_boundary_snapshots"
        if self._snapshot_root.is_symlink():
            raise ValueError(
                f"Refusing to use symlinked boundary snapshot root: {self._snapshot_root}"
            )
        self._session_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._snapshot_dir = self._snapshot_root / self._session_id
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        if self._snapshot_dir.is_symlink():
            raise ValueError(
                f"Refusing to use symlinked boundary snapshot session: {self._snapshot_dir}"
            )

        # request_id -> {token_count -> file_path}
        self._file_registry: dict[str, dict[int, Path]] = {}
        self._registry_lock = threading.Lock()

        # Pending writes buffer — raw bytes for instant read-back.
        # key: (request_id, token_count)
        self._pending_writes: dict[tuple[str, int], dict] = {}
        self._pending_lock = threading.Lock()
        self._pending_cond = threading.Condition(self._pending_lock)
        self._pending_max_bytes = max(1, int(pending_max_bytes))
        self._pending_bytes = 0
        self._pending_peak_bytes = 0
        self._backpressure_ms = 0.0

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
        *,
        block_size: int | None = None,
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
        block_size : int, optional
            When provided, compact append-only PoolingCache state to the
            current block's delta before buffering or writing it.

        Returns
        -------
        bool
            True if successfully enqueued for writing.
        """
        if not HAS_MLX:
            return False

        try:
            pw_key = (request_id, token_count)
            file_path = self._file_path(request_id, token_count)
            with self._pending_lock:
                if pw_key in self._pending_writes:
                    return True

            # 1. Extract dict-format states on inference thread.
            extracted, model_cache_config = extract_cache_states_fn(snapshot_cache)
            if not extracted:
                return False
            if block_size is not None:
                compact_pooling_cache_snapshot(extracted, token_count, block_size)

            # 2. Flatten tensors + metadata for safetensors serialization.
            tensors_raw, metadata = self._serialize_extracted(
                extracted, request_id, token_count
            )

            # The extracted MLX arrays and their raw byte copies must not both
            # live in the pending queue.  Keeping ``extracted`` here used to
            # pin one complete recurrent snapshot per queued boundary on top
            # of the serialized bytes.  Read-back can reconstruct from raw
            # bytes while the writer is in flight, so drop the MLX container
            # immediately after serialization.
            del extracted

            raw_size = sum(len(raw) for raw, _dtype, _shape in tensors_raw.values())

            # 3. Reserve a byte-bounded pending slot.  A single checkpoint is
            # allowed to exceed the configured cap when the queue is empty;
            # this guarantees forward progress for unusually wide recurrent
            # states without allowing a second checkpoint to accumulate.
            temp_path = file_path.with_name(file_path.stem + "_tmp.safetensors")
            if not self._is_safe_snapshot_path(file_path) or temp_path.is_symlink():
                raise ValueError("unsafe boundary snapshot staging path")
            inline_write = False
            wait_started = time.monotonic()
            deadline = wait_started + _PENDING_RESERVATION_TIMEOUT_S
            with self._pending_cond:
                # The same request/token boundary is deterministic. Coalesce a
                # duplicate capture with the generation already queued instead
                # of retaining two raw buffers or letting stale writer cleanup
                # race a replacement generation.
                if pw_key in self._pending_writes:
                    return True
                while (
                    self._pending_writes
                    and self._pending_bytes + raw_size > self._pending_max_bytes
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        inline_write = True
                        break
                    self._pending_cond.wait(timeout=remaining)

                self._backpressure_ms += max(
                    0.0, (time.monotonic() - wait_started) * 1000.0
                )
                if not inline_write:
                    pending = {
                        "tensors_raw": tensors_raw,
                        "metadata": metadata,
                        "raw_size": raw_size,
                        "reservation_released": False,
                    }
                    self._pending_writes[pw_key] = pending
                    self._pending_bytes += raw_size
                    self._pending_peak_bytes = max(
                        self._pending_peak_bytes, self._pending_bytes
                    )
                else:
                    # Publish an inline write as a pending item too.  This
                    # gives cleanup_request an atomic cancellation point
                    # before _write_inline acquires _writer_busy.  Inline
                    # bytes were already excluded from the reservation, so
                    # keep raw_size at zero here; the local tensors_raw is
                    # the only extra reference held by this synchronous path.
                    pending = {
                        "tensors_raw": tensors_raw,
                        "metadata": metadata,
                        "raw_size": 0,
                        "inline": True,
                        "reservation_released": False,
                    }
                    self._pending_writes[pw_key] = pending

                # Publish the registry entry while the pending marker is
                # still locked.  cleanup_request takes the same lock before
                # deciding which writes to cancel, so it cannot remove the
                # marker and then race this registration.
                with self._registry_lock:
                    self._file_registry.setdefault(request_id, {})[
                        token_count
                    ] = file_path

            # 5. Enqueue for background write.  Saturation never falls back
            # to retaining MLX state in RAM: after the bounded wait above we
            # write this one snapshot inline and return only once it is on
            # disk (or fail the checkpoint cleanly).
            if inline_write:
                return self._write_inline(pw_key, pending, file_path)

            try:
                self._write_queue.put_nowait((pw_key, pending, file_path))
            except queue.Full:
                logger.warning(
                    "Boundary snapshot write queue full, writing "
                    "snapshot %s/%d inline",
                    request_id,
                    token_count,
                )
                return self._write_inline(pw_key, pending, file_path)

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
        try:
            file_path = self._file_path(request_id, token_count)
        except (TypeError, ValueError):
            return None
        if not self._is_safe_snapshot_path(file_path):
            logger.warning("Refusing to load unsafe boundary snapshot path: %s", file_path)
            return None

        # Fast path: still in pending writes buffer.
        with self._pending_lock:
            pending = self._pending_writes.get(pw_key)
            if pending is not None:
                tensors_raw = pending.get("tensors_raw")
                metadata = pending.get("metadata")
                if tensors_raw and metadata:
                    try:
                        return self._deserialize(tensors_raw, metadata)
                    except Exception as e:
                        logger.debug(
                            "Failed to decode pending boundary snapshot %s/%d: %s",
                            request_id,
                            token_count,
                            e,
                        )
                        return None

        # Slow path: read from disk.
        if not file_path.is_file():
            return None

        try:
            data = mx.load(str(file_path), return_metadata=True)
            if isinstance(data, tuple) and len(data) == 2:
                arrays, metadata = data
            else:
                return None
            reconstructed = self._reconstruct_from_safetensors(arrays, metadata)
            if reconstructed is None:
                return None
            # mx.load returns file-backed lazy arrays. Materialize them while
            # this ephemeral snapshot still exists so the reconstructed cache
            # cannot retain a read primitive past cleanup or promotion.
            if arrays:
                mx.eval(*arrays.values())
            return reconstructed
        except Exception as e:
            logger.debug(
                "Failed to load boundary snapshot %s/%d: %s",
                request_id,
                token_count,
                e,
            )
            return None

    def load_file(self, file_path: Path) -> list[dict[str, Any]] | None:
        """Load a committed recurrent checkpoint from an explicit path."""
        file_path = Path(file_path)
        try:
            file_path.resolve(strict=True).relative_to(
                self._snapshot_root.parent.resolve(strict=True)
            )
        except (OSError, ValueError):
            return None
        if (
            not HAS_MLX
            or file_path.is_symlink()
            or file_path.parent.is_symlink()
            or not file_path.is_file()
        ):
            return None
        try:
            data = mx.load(str(file_path), return_metadata=True)
            if not (isinstance(data, tuple) and len(data) == 2):
                return None
            arrays, metadata = data
            reconstructed = self._reconstruct_from_safetensors(arrays, metadata)
            if reconstructed is None:
                return None
            # ``take_staged_file`` transfers this path to a caller that may
            # move or unlink it immediately after load_file returns. Detach
            # every lazy input from the file before returning cache state.
            if arrays:
                mx.eval(*arrays.values())
            return reconstructed
        except Exception as e:
            logger.debug("Failed to load committed boundary snapshot %s: %s", file_path, e)
            return None

    def take_staged_file(
        self,
        request_id: str,
        token_count: int,
        *,
        timeout_s: float = 30.0,
    ) -> Path | None:
        """Detach a completed staging file for durable sidecar promotion.

        The returned path remains owned by the caller.  It is removed from
        this store's request registry so later request cleanup cannot race an
        atomic rename into the durable cache namespace.
        """
        pw_key = (request_id, token_count)
        deadline = time.monotonic() + max(0.0, timeout_s)
        try:
            file_path = self._file_path(request_id, token_count)
        except (TypeError, ValueError):
            return None

        with self._pending_cond:
            # Wait for the latest generation, even if an older generation has
            # already materialized the shared final path.
            while pw_key in self._pending_writes:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._pending_cond.wait(timeout=remaining)

        if (
            not self._is_safe_snapshot_path(file_path)
            or file_path.is_symlink()
            or not file_path.is_file()
        ):
            return None

        with self._registry_lock:
            req_files = self._file_registry.get(request_id)
            if req_files is not None:
                req_files.pop(token_count, None)
                if not req_files:
                    self._file_registry.pop(request_id, None)

        # Move out of the request directory before returning ownership.
        # cleanup_request() removes that directory recursively, so registry
        # detachment alone is not enough to prevent a completion/abort race.
        # Keep caller-owned promotion files outside the session directory:
        # cleanup_all() removes and recreates that directory atomically.
        # reset_boundary_snapshot_root() still reclaims abandoned promotions
        # at the next server lifecycle boundary.
        detached_dir = self._snapshot_root / "_promote"
        detached_path = detached_dir / f"{uuid.uuid4().hex}.safetensors"
        try:
            if not self._is_safe_snapshot_path(detached_dir):
                return None
            detached_dir.mkdir(parents=True, exist_ok=True)
            if not self._is_safe_snapshot_path(detached_path):
                return None
            os.replace(file_path, detached_path)
            with suppress(OSError):
                file_path.parent.rmdir()
            return detached_path
        except Exception as e:
            logger.debug(
                "Failed to detach staged boundary snapshot %s/%d: %s",
                request_id,
                token_count,
                e,
            )
            return None

    @property
    def pending_bytes(self) -> int:
        with self._pending_lock:
            return self._pending_bytes

    @property
    def pending_peak_bytes(self) -> int:
        with self._pending_lock:
            return self._pending_peak_bytes

    @property
    def backpressure_ms(self) -> float:
        with self._pending_lock:
            return self._backpressure_ms

    @property
    def gdn_sidecar_state_dtype(self) -> str:
        return self._gdn_sidecar_state_dtype

    @property
    def gdn_state_dequantizations(self) -> int:
        """Number of recurrent tensors restored from reduced precision."""
        with self._gdn_dequant_lock:
            return self._gdn_state_dequantizations

    @property
    def gdn_encode_failures(self) -> int:
        """Recurrent tensors refused at the storage boundary (non-finite)."""
        with self._gdn_dequant_lock:
            return self._gdn_encode_failures

    @property
    def gdn_decode_failures(self) -> int:
        """Sidecar states rejected on restore (corrupt or mixed-writer)."""
        with self._gdn_dequant_lock:
            return self._gdn_decode_failures

    @property
    def gdn_capability_fallbacks(self) -> int:
        """Recurrent tensors stored as fp32 because the codec cannot fit them."""
        with self._gdn_dequant_lock:
            return self._gdn_capability_fallbacks

    def has(self, request_id: str, token_count: int) -> bool:
        """Check if a snapshot exists (in memory or on disk)."""
        pw_key = (request_id, token_count)
        try:
            file_path = self._file_path(request_id, token_count)
        except (TypeError, ValueError):
            return False
        if not self._is_safe_snapshot_path(file_path):
            return False
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
        with self._pending_cond:
            keys_to_remove = [k for k in self._pending_writes if k[0] == request_id]
            count = len(keys_to_remove)
            for key in keys_to_remove:
                # The queue/inline owner still holds the raw buffer and releases
                # its reservation when it observes cancellation.
                self._remove_pending_locked(key, release_reservation=False)
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
            req_dir = self._request_dir(request_id)
            if self._is_safe_snapshot_path(req_dir) and req_dir.exists():
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
        """Delete all snapshot files for this store session.

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
        """
        # Drain write queue so the writer thread doesn't process stale
        # items after the directory is deleted. Put_nowait the sentinel
        # back so shutdown still sees it; on Full just drop and let
        # shutdown re-issue.
        drained_items = []
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
                drained_items.append(item)
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
            with self._pending_cond:
                for pw_key, pending, _file_path in drained_items:
                    self._remove_pending_if_current_locked(
                        pw_key,
                        pending,
                        release_reservation=False,
                    )
                    self._release_pending_reservation_locked(pending)
                # Remaining entries are owned by an in-flight writer or an
                # inline caller waiting on _writer_busy. Hide them from loads,
                # but retain their reservations until those owners release the
                # raw buffers.
                self._pending_writes.clear()
                self._pending_cond.notify_all()
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

    def shutdown(self) -> None:
        """Stop background writer thread."""
        self._shutdown.set()
        with suppress(queue.Full):
            self._write_queue.put_nowait(None)  # Sentinel
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

    @staticmethod
    def _validate_token_count(token_count: int) -> int:
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise ValueError("token_count must be a non-negative integer")
        return token_count

    def _request_dir(self, request_id: str) -> Path:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        request_digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        return self._snapshot_dir / request_digest

    def _file_path(self, request_id: str, token_count: int) -> Path:
        token_count = self._validate_token_count(token_count)
        return self._request_dir(request_id) / f"{token_count}.safetensors"

    def _is_safe_snapshot_path(self, path: Path) -> bool:
        """Return whether a store-owned path has no symlinked component."""
        try:
            relative_path = path.relative_to(self._snapshot_root)
        except ValueError:
            return False

        current = self._snapshot_root
        if current.is_symlink():
            return False
        for component in relative_path.parts:
            current /= component
            if current.is_symlink():
                return False
        return True

    def _release_pending_reservation_locked(self, pending: dict) -> None:
        """Release one generation's byte reservation exactly once."""
        if pending.get("reservation_released", False):
            return
        pending["reservation_released"] = True
        self._pending_bytes = max(
            0, self._pending_bytes - int(pending.get("raw_size", 0))
        )
        self._pending_cond.notify_all()

    def _remove_pending_locked(
        self,
        pw_key: tuple[str, int],
        *,
        release_reservation: bool = True,
    ) -> dict | None:
        """Remove one pending entry while ``_pending_lock`` is held."""
        pending = self._pending_writes.pop(pw_key, None)
        if pending is not None and release_reservation:
            self._release_pending_reservation_locked(pending)
        return pending

    def _remove_pending_if_current_locked(
        self,
        pw_key: tuple[str, int],
        pending: dict,
        *,
        release_reservation: bool = True,
    ) -> bool:
        """Remove ``pending`` only if it is still this key's generation."""
        if self._pending_writes.get(pw_key) is not pending:
            return False
        self._remove_pending_locked(
            pw_key, release_reservation=release_reservation
        )
        return True

    def _drop_registry_entry(self, request_id: str, token_count: int) -> None:
        with self._registry_lock:
            req_files = self._file_registry.get(request_id)
            if req_files is None:
                return
            req_files.pop(token_count, None)
            if not req_files:
                self._file_registry.pop(request_id, None)

    def _write_inline(
        self,
        pw_key: tuple[str, int],
        pending: dict,
        file_path: Path,
    ) -> bool:
        """Write one staging file synchronously without retaining a fallback."""
        tensors_raw = pending["tensors_raw"]
        metadata = pending["metadata"]
        temp_path = file_path.with_name(file_path.stem + "_tmp.safetensors")
        owns_pending = False
        wrote_file = False
        try:
            with self._writer_busy:
                with self._pending_cond:
                    candidate = self._pending_writes.get(pw_key)
                    owns_pending = candidate is not None and candidate.get(
                        "tensors_raw"
                    ) is tensors_raw
                if not owns_pending:
                    # cleanup_request may have removed the inline marker and
                    # consumed its cancellation counter before this writer
                    # reached _writer_busy.  In that case this write must not
                    # recreate the request directory after cleanup returns.
                    if self._is_cancelled(pw_key[0]):
                        self._dec_cancelled(pw_key[0])
                    return False
                if self._is_cancelled(pw_key[0]):
                    return False
                if (
                    not self._is_safe_snapshot_path(file_path)
                    or temp_path.is_symlink()
                ):
                    logger.warning(
                        "Refusing to stage boundary snapshot through symlink: %s",
                        file_path,
                    )
                    return False
                file_path.parent.mkdir(parents=True, exist_ok=True)
                if (
                    not self._is_safe_snapshot_path(file_path)
                    or temp_path.is_symlink()
                ):
                    logger.warning(
                        "Refusing to stage boundary snapshot through symlink: %s",
                        file_path,
                    )
                    return False
                _write_safetensors_no_mx(str(temp_path), tensors_raw, metadata)
                if self._is_cancelled(pw_key[0]):
                    with suppress(OSError):
                        temp_path.unlink()
                    self._dec_cancelled(pw_key[0])
                    return False
                if (
                    not self._is_safe_snapshot_path(file_path)
                    or temp_path.is_symlink()
                ):
                    with suppress(OSError):
                        temp_path.unlink()
                    logger.warning(
                        "Refusing to commit boundary snapshot through symlink: %s",
                        file_path,
                    )
                    return False
                os.replace(str(temp_path), str(file_path))
                _fsync_parent_dir(file_path)
                wrote_file = True
                if self._is_cancelled(pw_key[0]):
                    with suppress(OSError):
                        file_path.unlink()
                    self._dec_cancelled(pw_key[0])
                    wrote_file = False
                    return False
                with self._pending_cond:
                    current = self._pending_writes.get(pw_key)
                    if current is pending:
                        self._remove_pending_locked(pw_key)
                return True
        except Exception as e:
            logger.warning(
                "Inline boundary snapshot write failed for %s/%d: %s",
                pw_key[0],
                pw_key[1],
                e,
            )
            for path in (temp_path, file_path):
                try:
                    if path.is_symlink() or path.is_file():
                        path.unlink()
                except Exception:
                    pass
            return False
        finally:
            removed = False
            with self._pending_cond:
                if not wrote_file:
                    removed = self._remove_pending_if_current_locked(
                        pw_key,
                        pending,
                        release_reservation=False,
                    )
                self._release_pending_reservation_locked(pending)
            if removed:
                self._drop_registry_entry(pw_key[0], pw_key[1])

    def _writer_loop(self) -> None:
        """Background thread that writes safetensors files."""
        while not self._shutdown.is_set():
            item = None
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
            try:
                with self._writer_busy:
                    self._process_write_item(item)
            finally:
                # ``item`` contains ``tensors_raw``. If the thread waits for
                # the next queue entry without clearing this local, the frame
                # can pin a whole boundary snapshot after pending_writes was
                # cleaned up.
                item = None

    def _process_write_item(self, item) -> None:
        """Process one (pw_key, pending, file_path) queue item.

        Extracted from ``_writer_loop`` so the busy-lock can wrap it
        cleanly. Called only on the writer thread.
        """
        pw_key, pending, file_path = item
        tensors_raw = pending["tensors_raw"]
        metadata = pending["metadata"]

        # If cleanup cleared this key, or a newer save superseded this queue
        # item, do not write or clear the latest generation.
        with self._pending_lock:
            owns_pending = self._pending_writes.get(pw_key) is pending
        if not owns_pending:
            # If a timed-out cleanup_request bumped ``_cancelled_requests``
            # before clearing pending_writes, this item is one of the N
            # the counter is waiting on. Without this decrement the
            # counter would never reach zero, leaving the rid pinned in
            # ``_cancelled_requests`` for the process lifetime and
            # causing every subsequent write under that rid (or any
            # later reuse of the same string) to be silently discarded.
            if self._is_cancelled(pw_key[0]):
                self._dec_cancelled(pw_key[0])
            with self._pending_cond:
                self._release_pending_reservation_locked(pending)
            return

        # Skip writes for cancelled/cleaned-up requests.
        if self._is_cancelled(pw_key[0]):
            with self._pending_cond:
                self._remove_pending_if_current_locked(
                    pw_key, pending, release_reservation=False
                )
                self._release_pending_reservation_locked(pending)
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
            if not self._is_safe_snapshot_path(file_path):
                logger.warning(
                    "Refusing to stage boundary snapshot through symlink: %s",
                    file_path,
                )
                return
            file_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = file_path.with_name(file_path.stem + "_tmp.safetensors")
            if (
                not self._is_safe_snapshot_path(file_path)
                or temp_path.is_symlink()
            ):
                logger.warning(
                    "Refusing to stage boundary snapshot through symlink: %s",
                    file_path,
                )
                return
            _write_safetensors_no_mx(str(temp_path), tensors_raw, metadata)

            # Request may have been cleaned up while serializing.
            if self._is_cancelled(pw_key[0]):
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except Exception:
                    pass
                with self._pending_cond:
                    self._remove_pending_if_current_locked(
                        pw_key, pending, release_reservation=False
                    )
                self._dec_cancelled(pw_key[0])
                return

            if not self._is_safe_snapshot_path(file_path) or temp_path.is_symlink():
                with suppress(OSError):
                    temp_path.unlink()
                logger.warning(
                    "Refusing to commit boundary snapshot through symlink: %s",
                    file_path,
                )
                return
            os.rename(str(temp_path), str(file_path))
            _fsync_parent_dir(file_path)

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
                    if p is not None and (p.is_symlink() or p.is_file()):
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
            # Raw bytes are only an in-flight read-back buffer.  Once the
            # write attempt finishes they must be released even on failure;
            # retaining them would recreate the unbounded RAM fallback this
            # store exists to avoid.
            removed = False
            with self._pending_cond:
                removed = self._remove_pending_if_current_locked(
                    pw_key,
                    pending,
                    release_reservation=False,
                )
                self._release_pending_reservation_locked(pending)
            if removed and (
                not self._is_safe_snapshot_path(file_path)
                or not file_path.is_file()
            ):
                self._drop_registry_entry(pw_key[0], pw_key[1])

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
        # (reason, shape, unevaluated flag) per encoded recurrent tensor,
        # verified in one batch below rather than one sync per layer.
        gdn_checks: list[tuple[str, tuple[int, ...], Any]] = []

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
            pooling_delta_ranges = layer_state.get("pooling_delta_ranges")
            if pooling_delta_ranges:
                info["pooling_delta_ranges"] = json.dumps(pooling_delta_ranges)

            if (
                isinstance(state, list)
                and len(state) >= 1
                and all(isinstance(s, (list, tuple)) for s in state)
            ):
                # CacheList layer: ``state`` is a list of nested sub-state
                # tuples (one per sub-cache, e.g. RotatingKVCache +
                # PoolingCache for DeepSeek V4). Flatten as
                # ``layer_{i}_sub_{j}_state_{k}`` keys so reconstruction
                # can rebuild the nested shape. Mirror the flat branch's
                # has_tensors gate: a CacheList whose subs hold no tensor
                # at all (empty KV + untouched conv slots) carries no
                # restorable state.
                has_tensors = any(
                    hasattr(elem, "shape") for sub in state for elem in sub
                )
                if not has_tensors:
                    info["has_state"] = "false"
                    layer_info.append(info)
                    continue
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
                            key = f"layer_{i}_state_{k}"
                            if self._should_quantize_gdn_state(
                                class_name,
                                cache_type,
                                k,
                                elem,
                            ):
                                stored, scale, codec = self._encode_gdn_state(
                                    elem, gdn_checks
                                )
                                arrays[key] = stored
                                if scale is not None:
                                    arrays[f"{key}__scale"] = scale
                                info[f"state_{k}_storage_codec"] = codec
                                info[f"state_{k}_original_dtype"] = "float32"
                                if codec in {
                                    _GDN_RHT_INT8_CODEC,
                                    _GDN_RHT_INT16_CODEC,
                                }:
                                    info[f"state_{k}_rht_seed"] = str(
                                        _GDN_RHT_SEED
                                    )
                                    info[f"state_{k}_rht_dim"] = str(elem.shape[-1])
                            else:
                                arrays[key] = elem
                else:
                    info["has_state"] = "false"
            else:
                info["has_state"] = "false"

            layer_info.append(info)

        # Reject a corrupt recurrent state before materializing the payload it
        # would have been written into.
        self._verify_gdn_encode_checks(gdn_checks)

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
            "gdn_sidecar_format_version": _GDN_SIDECAR_FORMAT_VERSION,
        }

        return tensors_raw, metadata

    def _should_quantize_gdn_state(
        self,
        class_name: str,
        cache_type: str,
        state_index: int,
        tensor: Any,
    ) -> bool:
        """Select only the fp32 recurrent member of Arrays-family caches."""
        eligible = bool(
            self._gdn_sidecar_state_dtype != "fp32"
            and state_index == 1
            and (
                str(class_name).endswith("ArraysCache")
                or str(cache_type).endswith("ArraysCache")
            )
            and getattr(tensor, "dtype", None) == mx.float32
            and len(getattr(tensor, "shape", ())) >= 1
        )
        if not eligible or self._gdn_sidecar_state_dtype not in {
            "rht_int8",
            "rht_int16",
        }:
            return eligible
        return self._rht_supports(tensor)

    def _rht_supports(self, tensor: Any) -> bool:
        """Gate the RHT codec on a width it can invert exactly.

        Unlike bf16 and plain int8, the rotation is only defined for
        power-of-two widths. An unsupported tensor is stored as raw fp32
        instead: it carries no codec key, so the existing "no codec" restore
        path returns it unchanged. Failing the whole checkpoint instead would
        make a cache-precision setting able to break inference, and silently
        dropping it (the prior behaviour) disabled GDN sidecars for the model
        with only a debug line to show for it.
        """
        shape = getattr(tensor, "shape", ())
        dim = int(shape[-1]) if shape else 0
        if _gdn_rht_dimension_supported(dim) and hasattr(mx, "hadamard_transform"):
            return True
        with self._gdn_dequant_lock:
            self._gdn_capability_fallbacks += 1
            first = not self._gdn_capability_reported
            self._gdn_capability_reported = True
        if first:
            # Once per store: a 48-layer model would otherwise log this on
            # every layer of every checkpoint.
            logger.error(
                "GDN RHT sidecars unsupported for last dimension %s "
                "(needs a positive power of two); storing recurrent state as "
                "fp32 instead. Set gdn_sidecar_state_dtype=int8 for a codec "
                "with no width constraint, or use fp32 for this shape.",
                dim,
            )
        return False

    def _encode_gdn_state(
        self, tensor: Any, checks: list[tuple[str, tuple[int, ...], Any]]
    ) -> tuple[Any, Any | None, str]:
        """Encode an ArraysCache recurrent tensor for storage only.

        int8 uses one symmetric scale per row of the last axis. This is the
        axis measured by ``test/session116_gdn_state_distribution.py`` and is
        1.15x-2.70x more accurate than reducing the penultimate axis on sampled
        27B production sidecars. Scales stay fp32; their footprint is tiny and
        this keeps the experiment focused on state quantization error.

        rht_int8 and rht_int16 first apply a normalized randomized Hadamard
        transform on the last axis. The transform is orthogonal and inverted
        immediately on restore, so the live recurrent state still runs in fp32
        in its original basis. A fixed, versioned sign diagonal avoids mutating
        the generation RNG and keeps sidecars reproducible across processes.

        A non-finite source must not be encoded. NaN survives ``round``/``clip``
        and lands in int8 as an undefined value, and the row scale it produces
        may still look finite, so a corrupt recurrent state would otherwise be
        written as a well-formed sidecar and restored later as silent garbage.

        The validity flags are appended to ``checks`` as unevaluated arrays
        rather than tested here. Forcing ``bool()`` per layer costs a
        GPU->CPU sync each time (+19 ms over the 48-layer 27B state, ~19x the
        codec itself); the caller evaluates all flags in one batch instead.
        """
        checks.append(
            (
                "non-finite GDN recurrent state",
                tuple(getattr(tensor, "shape", ())),
                mx.all(mx.isfinite(tensor)),
            )
        )
        if self._gdn_sidecar_state_dtype == "bf16":
            return tensor.astype(mx.bfloat16), None, _GDN_BF16_CODEC

        encoded = tensor
        codec = _GDN_INT8_CODEC
        qmax = 127
        payload_dtype = mx.int8
        if self._gdn_sidecar_state_dtype == "rht_int8":
            encoded = self._gdn_rht_forward(tensor, _GDN_RHT_SEED)
            codec = _GDN_RHT_INT8_CODEC
        elif self._gdn_sidecar_state_dtype == "rht_int16":
            encoded = self._gdn_rht_forward(tensor, _GDN_RHT_SEED)
            codec = _GDN_RHT_INT16_CODEC
            qmax = 32767
            payload_dtype = mx.int16

        scale = mx.max(mx.abs(encoded), axis=-1, keepdims=True) / qmax
        scale = mx.maximum(scale, mx.array(1e-12, dtype=mx.float32))
        # A finite source is not sufficient: the RHT is not magnitude
        # preserving per element, so a row near the fp32 maximum can sum to inf
        # under the Hadamard butterfly. Checking the (tiny) scale catches that
        # before the payload is written.
        checks.append(
            (
                "non-finite GDN row scale (encoded state overflowed fp32)",
                tuple(getattr(encoded, "shape", ())),
                mx.all(mx.isfinite(scale) & (scale > 0)),
            )
        )
        quantized = mx.clip(mx.round(encoded / scale), -qmax, qmax).astype(
            payload_dtype
        )
        return quantized, scale.astype(mx.float32), codec

    def _verify_gdn_encode_checks(
        self, checks: list[tuple[str, tuple[int, ...], Any]]
    ) -> None:
        """Evaluate every deferred encode check with a single sync."""
        if not checks:
            return
        flags = mx.stack([flag for _reason, _shape, flag in checks])
        mx.eval(flags)
        if bool(mx.all(flags)):
            return
        reason, shape, _flag = next(
            check for check, ok in zip(checks, flags.tolist()) if not ok
        )
        with self._gdn_dequant_lock:
            self._gdn_encode_failures += 1
        # save() degrades every failure to ``False`` at debug level, so warn
        # here: refusing a recurrent state is a model-level event, not a
        # routine checkpoint miss.
        logger.warning(
            "Skipping GDN sidecar: %s (shape=%s, codec=%s)",
            reason,
            shape,
            self._gdn_sidecar_state_dtype,
        )
        raise ValueError(f"refusing to encode {reason} (shape={shape})")

    @staticmethod
    def _gdn_rht_forward(tensor: Any, seed: int) -> Any:
        dim = int(tensor.shape[-1])
        signs = mx.array(_gdn_rht_sign_values(dim, seed), dtype=mx.float32)
        return mx.hadamard_transform(
            tensor.astype(mx.float32) * signs,
            scale=1.0 / math.sqrt(dim),
        )

    @staticmethod
    def _gdn_rht_inverse(tensor: Any, seed: int) -> Any:
        dim = int(tensor.shape[-1])
        signs = mx.array(_gdn_rht_sign_values(dim, seed), dtype=mx.float32)
        return (
            mx.hadamard_transform(
                tensor.astype(mx.float32),
                scale=1.0 / math.sqrt(dim),
            )
            * signs
        )

    @staticmethod
    def _gdn_rht_metadata(
        info: dict[str, str], state_idx: int, tensor: Any
    ) -> tuple[int, int]:
        raw_seed = info.get(f"state_{state_idx}_rht_seed")
        raw_dim = info.get(f"state_{state_idx}_rht_dim")
        if raw_seed is None or raw_dim is None:
            raise ValueError("missing GDN RHT metadata")
        # ``int()`` accepts surrounding whitespace, a sign and underscore
        # separators, so require a plain decimal literal before converting.
        # Anything else means the metadata was not written by this codec.
        if not (
            isinstance(raw_seed, str)
            and isinstance(raw_dim, str)
            and raw_seed.isdigit()
            and raw_dim.isdigit()
        ):
            raise ValueError(
                f"invalid GDN RHT metadata literals: seed={raw_seed!r}, dim={raw_dim!r}"
            )
        seed = int(raw_seed)
        dim = int(raw_dim)
        if seed != _GDN_RHT_SEED:
            raise ValueError(f"unsupported GDN RHT seed: {seed}")
        if dim <= 0 or dim & (dim - 1):
            raise ValueError(f"invalid GDN RHT dimension: {dim} is not a power of two")
        if dim != int(tensor.shape[-1]):
            raise ValueError(
                f"invalid GDN RHT dimension: expected {tensor.shape[-1]}, got {dim}"
            )
        _gdn_rht_sign_values(dim, seed)
        return seed, dim

    @staticmethod
    def _validate_gdn_codec_metadata(
        info: dict[str, str], state_idx: int, codec: str
    ) -> None:
        """Require codec and its metadata to agree on the same state index.

        Guards two mixed-writer shapes that would otherwise decode: a plain
        int8 payload carrying RHT metadata (restored without the inverse
        rotation, i.e. wrong basis), and a reduced codec whose source dtype is
        not the fp32 recurrent member this codec is defined for.
        """
        original = info.get(f"state_{state_idx}_original_dtype")
        if original != _GDN_REQUIRED_ORIGINAL_DTYPE:
            raise ValueError(
                f"invalid GDN source dtype for codec {codec}: "
                f"expected {_GDN_REQUIRED_ORIGINAL_DTYPE}, got {original!r}"
            )
        if codec in {_GDN_RHT_INT8_CODEC, _GDN_RHT_INT16_CODEC}:
            return
        stray = [
            key
            for key in (
                f"state_{state_idx}_rht_seed",
                f"state_{state_idx}_rht_dim",
            )
            if key in info
        ]
        if stray:
            raise ValueError(
                f"GDN RHT metadata present under non-RHT codec {codec}: {stray}"
            )

    def _note_gdn_dequantization(self) -> None:
        with self._gdn_dequant_lock:
            self._gdn_state_dequantizations += 1

    @contextmanager
    def _gdn_decode_guard(self, codec: str) -> Any:
        """Count and surface a rejected sidecar, then re-raise.

        Callers turn the exception into a cache miss, so without this the
        only trace of a corrupt or mixed-writer sidecar would be a silent
        re-prefill.
        """
        try:
            yield
        except Exception as exc:
            with self._gdn_dequant_lock:
                self._gdn_decode_failures += 1
            logger.warning("Rejected GDN sidecar (codec=%s): %s", codec, exc)
            raise

    def _decode_gdn_state_raw(
        self,
        tensors_raw: dict[str, tuple[bytes, str, list[int]]],
        info: dict[str, str],
        layer_idx: int,
        state_idx: int,
        tensor: Any,
    ) -> Any:
        codec = info.get(f"state_{state_idx}_storage_codec")
        if not codec:
            return tensor
        with self._gdn_decode_guard(codec):
            # Reject an unknown codec before reading any of its metadata: a
            # future or foreign codec says nothing about what the accompanying
            # keys mean, so it is the more fundamental rejection.
            if codec not in _GDN_REDUCED_CODECS:
                raise ValueError(f"unsupported GDN storage codec: {codec}")
            self._validate_gdn_codec_metadata(info, state_idx, codec)
            if codec == _GDN_BF16_CODEC:
                self._note_gdn_dequantization()
                return tensor.astype(mx.float32)
            if codec in _GDN_INTEGER_CODECS:
                scale_key = f"layer_{layer_idx}_state_{state_idx}__scale"
                scale_raw = tensors_raw.get(scale_key)
                if scale_raw is None:
                    kind = "int16" if codec in _GDN_INT16_CODECS else "int8"
                    raise ValueError(
                        f"missing GDN {kind} scale tensor: {scale_key}"
                    )
                raw, dtype_str, shape = scale_raw
                scale = _restore_tensor_from_bytes(raw, dtype_str, shape)
                payload_dtype = (
                    mx.int16 if codec in _GDN_INT16_CODECS else mx.int8
                )
                self._validate_gdn_scale(
                    tensor, scale, scale_key, payload_dtype=payload_dtype
                )
                restored = tensor.astype(mx.float32) * scale
                if codec in {_GDN_RHT_INT8_CODEC, _GDN_RHT_INT16_CODEC}:
                    seed, _dim = self._gdn_rht_metadata(info, state_idx, tensor)
                    restored = self._gdn_rht_inverse(restored, seed)
                self._note_gdn_dequantization()
                return restored
            raise ValueError(f"unsupported GDN storage codec: {codec}")

    def _decode_gdn_state_array(
        self,
        arrays: dict[str, Any],
        info: dict[str, str],
        layer_idx: int,
        state_idx: int,
        tensor: Any,
    ) -> Any:
        codec = info.get(f"state_{state_idx}_storage_codec")
        if not codec:
            return tensor
        with self._gdn_decode_guard(codec):
            # Reject an unknown codec before reading any of its metadata: a
            # future or foreign codec says nothing about what the accompanying
            # keys mean, so it is the more fundamental rejection.
            if codec not in _GDN_REDUCED_CODECS:
                raise ValueError(f"unsupported GDN storage codec: {codec}")
            self._validate_gdn_codec_metadata(info, state_idx, codec)
            if codec == _GDN_BF16_CODEC:
                self._note_gdn_dequantization()
                return tensor.astype(mx.float32)
            if codec in _GDN_INTEGER_CODECS:
                scale_key = f"layer_{layer_idx}_state_{state_idx}__scale"
                scale = arrays.get(scale_key)
                if scale is None:
                    kind = "int16" if codec in _GDN_INT16_CODECS else "int8"
                    raise ValueError(
                        f"missing GDN {kind} scale tensor: {scale_key}"
                    )
                payload_dtype = (
                    mx.int16 if codec in _GDN_INT16_CODECS else mx.int8
                )
                self._validate_gdn_scale(
                    tensor, scale, scale_key, payload_dtype=payload_dtype
                )
                restored = tensor.astype(mx.float32) * scale
                if codec in {_GDN_RHT_INT8_CODEC, _GDN_RHT_INT16_CODEC}:
                    seed, _dim = self._gdn_rht_metadata(info, state_idx, tensor)
                    restored = self._gdn_rht_inverse(restored, seed)
                self._note_gdn_dequantization()
                return restored
            raise ValueError(f"unsupported GDN storage codec: {codec}")

    @staticmethod
    def _validate_gdn_scale(
        tensor: Any,
        scale: Any,
        scale_key: str,
        *,
        payload_dtype: Any = None,
    ) -> None:
        if payload_dtype is None:
            payload_dtype = mx.int8
        kind = "int16" if payload_dtype == mx.int16 else "int8"
        expected_shape = tuple(tensor.shape[:-1]) + (1,)
        if tuple(scale.shape) != expected_shape:
            raise ValueError(
                f"invalid GDN {kind} scale shape for {scale_key}: "
                f"expected {expected_shape}, got {tuple(scale.shape)}"
            )
        if getattr(tensor, "dtype", None) != payload_dtype:
            raise ValueError(f"invalid GDN {kind} payload dtype for {scale_key}")
        # The codec contract stores scales as fp32. Accepting a narrower dtype
        # would silently change the reconstruction of every row, so reject it
        # rather than upcast whatever the file happens to carry.
        if getattr(scale, "dtype", None) != mx.float32:
            raise ValueError(
                f"invalid GDN {kind} scale dtype for {scale_key}: "
                f"expected float32, got {getattr(scale, 'dtype', None)}"
            )
        if not bool(mx.all(mx.isfinite(scale) & (scale > 0))):
            raise ValueError(f"invalid GDN {kind} scale values for {scale_key}")

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
        if metadata.get("gdn_sidecar_format_version", "1") not in {"1", "2"}:
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

            if "pooling_delta_ranges" in info:
                try:
                    result[-1]["pooling_delta_ranges"] = json.loads(
                        info["pooling_delta_ranges"]
                    )
                except (TypeError, json.JSONDecodeError):
                    return None

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
                    restored = _restore_tensor_from_bytes(raw, dtype_str, shape)
                    elements.append(
                        self._decode_gdn_state_raw(
                            tensors_raw,
                            info,
                            layer_idx,
                            k,
                            restored,
                        )
                    )
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
        if metadata.get("gdn_sidecar_format_version", "1") not in {"1", "2"}:
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

            if "pooling_delta_ranges" in info:
                try:
                    result[-1]["pooling_delta_ranges"] = json.loads(
                        info["pooling_delta_ranges"]
                    )
                except (TypeError, json.JSONDecodeError):
                    return None

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
                    elements.append(
                        self._decode_gdn_state_array(
                            arrays,
                            info,
                            layer_idx,
                            k,
                            tensor,
                        )
                    )
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
