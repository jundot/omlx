# SPDX-License-Identifier: Apache-2.0
"""Worker-local telemetry for the pinned MLX-LM distributed server."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
import struct
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .performance import (
    DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES,
    ExecutionSettings,
)
from .planner import PipelineAssignment

logger = logging.getLogger(__name__)

# Bounded read for the coordinator's cancel-request file.
_MAX_CANCEL_FILE_BYTES = 64 * 1024

# A shared request contains HTTP arguments and may include multimodal payloads,
# but it must never be able to turn one corrupt transport word into an
# unbounded worker allocation.
_MAX_SHARED_REQUEST_BYTES = 256 * 1024 * 1024
_SHARED_OBJECT_MAGIC = 0x4F4D4C58  # ASCII "OMLX"
_SHARED_OBJECT_HEADER_BYTES = 64

# Runtime markers are capped at 64 KiB.  A normal deployment admits eight
# concurrent requests, but keep enough bounded detail for deliberately wider
# serving configurations without letting telemetry grow with an unbounded
# queue.  ``active_requests`` remains the authoritative total when truncated.
_MAX_ACTIVE_REQUEST_METRICS = 64
_MAX_TARGETED_CANCEL_REQUESTS = 256

# Rank-zero cancellation decisions travel in the reliable control stream as an
# explicit epoch rather than an untyped UID list. The epoch is process-local:
# the control-plane sequence resets with the rank lifetime too, while a newer
# coordinator marker epoch is folded in when one exists.
_CANCEL_VOTE_KIND = "omlx.cancel_vote"
_CANCEL_VOTE_SCHEMA_VERSION = 1

# How often a rank refreshes its marker with nothing to report.
#
# Every other publish is request-driven, so an idle deployment's marker simply
# stopped ageing, and the peer watchdog — which reads that timestamp — declared
# a healthy cluster lost 60 s after the last token and called ``os._exit(1)``.
# A conversation is mostly idle, so this fired on the way from one turn to the
# next, and each turn paid for a full model reload.
#
# Comfortably under ``liveness._DEFAULT_STALE_AFTER`` (45 s) so several
# heartbeats can be missed before anything is called stale, and long enough that
# it never competes with generation for the lock.
_DEFAULT_HEARTBEAT_INTERVAL = 10.0
_MAX_TRANSPORT_REQUEST_ID_BYTES = 128
_INTERNAL_REQUEST_PREFIX = "omlx-internal-"


def _agreed_snapshot_capacity_charge(
    local_charge: int | None,
    *,
    world_size: int,
    rank: int,
    control_plane: Any | None,
) -> int | None:
    """Largest rank charge, or None when safe agreement is unavailable.

    The reliable control plane is intentional. Tiny int32 collectives are a
    known-bad fit for JACCL control traffic (lost completion can poison the
    following model collective), so an unmanaged multi-rank worker skips the
    snapshot rather than reintroducing that fallback.
    """

    valid = local_charge is not None and int(local_charge) > 0
    local = int(local_charge or 0)
    if world_size <= 1:
        return local if valid else None
    if control_plane is None:
        return None
    sizes: list[int] = []
    for source in range(world_size):
        payload = (
            struct.pack("!BQ", int(valid), local) if rank == source else None
        )
        packet = control_plane.broadcast_owned_bytes(
            payload,
            source_rank=source,
            expected_size=9,
        )
        ok, size = struct.unpack("!BQ", packet)
        if ok != 1 or size <= 0:
            return None
        sizes.append(int(size))
    return max(sizes) if sizes else None


def _transport_request_id(value: Any) -> str | None:
    """Validate the private coordinator-to-rank request correlation id."""

    if not isinstance(value, str) or not value:
        return None
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        return None
    if len(encoded) > _MAX_TRANSPORT_REQUEST_ID_BYTES:
        return None
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:"
    )
    return value if all(character in allowed for character in value) else None


def _python_token_id(value: Any) -> int:
    """Normalize a generated token to a Python signed-32-bit index.

    MLX samplers may surface a uint32 scalar (or a one-element nested list
    after ``tolist``). Those are valid model values but some MLX releases
    reject them as ``array[index]`` objects. The private server indexes the
    response logprob row after BatchGenerator.next(), so normalize at that
    queue boundary without touching the device-side token graph.
    """

    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("generated token must be a scalar")
        value = value[0]
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (int, bool)):
        value = item()
    if isinstance(value, bool):
        raise ValueError("generated token must be an integer")
    try:
        token = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("generated token must be an integer") from exc
    if not 0 <= token <= 2**31 - 1:
        raise ValueError("generated token is outside signed int32 range")
    return token


def _capture_prompt_boundary_cache(
    prompt_cache: Any,
    model_key: Any,
    batch_generator: Any,
    uid: Any,
) -> int:
    """Store an immutable prompt-side cache before its seed token is consumed.

    DS4 has sliding-window layers whose post-decode state cannot be trimmed
    backward. Capturing at the prompt-batch→generation boundary yields the
    ``prompt[:-1]`` state MLX-LM needs for its next request.
    """

    if uid is None or prompt_cache is None or model_key is None:
        return 0
    try:
        extracted = batch_generator.extract_cache([uid]).get(uid)
    except Exception:
        return 0
    if (
        not isinstance(extracted, tuple)
        or len(extracted) != 2
        or not isinstance(extracted[1], list)
        or not extracted[1]
    ):
        return 0
    cache, cache_key = extracted
    prompt_cache.insert_cache(
        model_key,
        cache_key[:],
        # ``extract_cache`` returns rank-local views. Generation mutates those
        # same cache objects after the prompt boundary, which otherwise turns
        # a correctly keyed prefix into a post-assistant cache by the next
        # request. MLX-LM already uses deepcopy on fetch; freeze the stored
        # boundary with the matching operation.
        copy.deepcopy(cache),
        cache_type="user",
    )
    return len(cache_key)


class MarkerWriter(Protocol):
    """Small RuntimeMarker surface used by the telemetry observer."""

    def update(self, phase: str, **extra: Any) -> None: ...


@dataclass
class _RequestSample:
    request_id: int
    started_at: float
    updated_at: float
    internal: bool = False
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    first_token_at: float | None = None
    prefill_started_at: float | None = None
    prefill_updated_at: float | None = None
    prefill_processed_tokens: int = 0
    prefill_total_tokens: int = 0
    # ``prefill_speed`` is the most recent MLX-LM chunk. It is useful for ETA
    # and for exposing genuine long-context slowdown, but is too noisy to
    # present as the request's headline throughput.
    prefill_speed: float = 0.0
    prefill_average_speed: float = 0.0


class RuntimeTelemetry:
    """Aggregate bounded, end-to-end pipeline metrics for one worker rank."""

    def __init__(
        self,
        marker: MarkerWriter,
        *,
        clock: Any = time.perf_counter,
        publish_interval: float = 1.0,
        execution: ExecutionSettings | None = None,
        assignment: PipelineAssignment | None = None,
        heartbeat_interval: float = _DEFAULT_HEARTBEAT_INTERVAL,
        cancel_path: Path | None = None,
        cancel_deployment_id: str = "",
        cancel_plan_hash: str = "",
        cancel_epoch_floor: int = 0,
        prompt_cache_ssd_enabled: bool = False,
        prompt_cache_ssd_max_bytes: int = 0,
    ) -> None:
        if publish_interval < 0:
            raise ValueError("publish_interval must be non-negative")
        if heartbeat_interval < 0:
            raise ValueError("heartbeat_interval must be non-negative")
        self._marker = marker
        self._clock = clock
        self._publish_interval = publish_interval
        self._execution = execution
        self._assignment = assignment
        self._heartbeat_interval = float(heartbeat_interval)
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._maintenance_callback: Any | None = None
        self._lock = threading.Lock()
        self._started_at = float(self._clock())
        self._last_step_finished_at = self._started_at
        self._next_request_id = 0
        self._requests: dict[int, _RequestSample] = {}
        self._uid_to_request: dict[Any, int] = {}
        self._request_to_uid: dict[int, Any] = {}
        self._request_contexts: dict[int, Any] = {}
        self._request_queues: dict[int, Any] = {}
        self._transport_to_request: dict[str, int] = {}
        self._request_to_transport: dict[int, str] = {}
        self._pending_transport_cancels: set[str] = set()
        self._cancel_requested_requests: set[int] = set()
        self._pending_uid = threading.local()
        self._last_completed: dict[str, Any] | None = None
        self._last_publish_at = float("-inf")
        self._requests_completed = 0
        self._requests_failed = 0
        self._requests_cancelled = 0
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0
        self._cached_tokens_total = 0
        # Sum of per-request decode windows (first-token -> finish) for
        # finished requests. This produces the average per-request rate. True
        # aggregate throughput uses generation rows / batch-step wall time;
        # summing overlapping request windows would divide concurrency away.
        self._decode_seconds_total = 0.0
        self._decode_step_tokens_total = 0
        self._decode_step_seconds_total = 0.0
        self._batch_steps = 0
        self._busy_seconds = 0.0
        self._idle_seconds = 0.0
        self._last_batch: dict[str, Any] | None = None
        self._cache_lookups = 0
        self._cache_hits = 0
        self._cache_tokens_reused = 0
        self._cache_entries = 0
        self._cache_bytes = 0
        self._cache_memory_entries = 0
        self._cache_memory_bytes = 0
        self._cache_memory_hits = 0
        self._cache_ssd_entries = 0
        self._cache_ssd_bytes = 0
        self._cache_ssd_hits = 0
        self._cache_ssd_enabled = bool(prompt_cache_ssd_enabled)
        self._cache_ssd_max_bytes = max(0, int(prompt_cache_ssd_max_bytes))
        self._cache_ssd_capacity_bytes = 0
        self._cache_ssd_evictions = 0
        self._cache_ssd_capacity_drops = 0
        self._cache_ssd_pending_bytes = 0
        self._cache_ssd_pending_max_bytes = 0
        self._cache_ssd_write_failures = 0
        self._phase_split: dict[str, Any] | None = None
        # Rank-side force-cancel surface. The coordinator drops a cancel file
        # next to the runtime markers; the heartbeat picks it up and removes
        # every active uid through BatchGenerator.remove — MLX-LM's own
        # cancel path, which lands at a batch step boundary and is shared
        # with peer ranks, so cancellation never severs a collective
        # mid-request.
        self._batch_generator: Any | None = None
        self._cancel_path = cancel_path
        self._cancel_deployment_id = cancel_deployment_id
        self._cancel_plan_hash = cancel_plan_hash
        self._cancel_epoch_floor = max(0, int(cancel_epoch_floor))
        self._last_cancel_epoch = max(0, self._cancel_epoch_floor - 1)
        self._accepted_cancel_vote_epoch = 0
        self._last_cancel_poll_at = float("-inf")
        # Cancellation is an edge-triggered control event, not durable desired
        # state. A file left by a previous rank lifetime must be the startup
        # watermark, never work applied to the first request of this process.
        existing_cancel = self._read_cancel_request()
        if existing_cancel is not None:
            self._last_cancel_epoch = max(
                self._last_cancel_epoch,
                int(existing_cancel["epoch"]),
            )

    def heartbeat(self) -> None:
        """Refresh the marker with nothing new to say.

        "Stale" has to mean *stalled*. Every other publish here is driven by a
        request, so without this an idle rank's marker just stopped ageing and
        was indistinguishable from a rank wedged inside a collective.
        """

        now = float(self._clock())
        # A coordinator cancel request is picked up here even when no
        # request event is driving publishes (e.g. a wedged handler).
        self.poll_cancel_requests(min_interval=0.0)
        maintenance = self._maintenance_callback
        if maintenance is not None:
            maintenance()
        with self._lock:
            self._publish_locked(now, force=True)

    def set_maintenance_callback(self, callback: Any | None) -> None:
        """Install the rank-local, heartbeat-driven maintenance poll."""

        self._maintenance_callback = callback

    def start_heartbeat(self) -> None:
        """Begin refreshing the marker on a fixed interval. Idempotent."""

        if self._heartbeat_interval <= 0 or self._heartbeat_thread is not None:
            return
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="omlx-cluster-telemetry-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop_heartbeat(self, *, timeout: float = 2.0) -> None:
        """Stop the heartbeat thread. Safe to call when it never started."""

        self._heartbeat_stop.set()
        thread, self._heartbeat_thread = self._heartbeat_thread, None
        if thread is not None:
            thread.join(timeout=timeout)

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self._heartbeat_interval):
            try:
                self.heartbeat()
            except Exception as exc:  # pragma: no cover - defensive
                # Visibility is fail-soft, and this thread outlives every
                # request: one bad write must not silently end the heartbeat
                # and reintroduce the self-kill it exists to prevent.
                logger.debug("Runtime marker heartbeat failed: %s", exc)

    def begin_request(self, transport_request_id: str | None = None) -> int:
        now = float(self._clock())
        transport_request_id = _transport_request_id(transport_request_id)
        with self._lock:
            self._next_request_id += 1
            request_id = self._next_request_id
            self._requests[request_id] = _RequestSample(
                request_id=request_id,
                started_at=now,
                updated_at=now,
                internal=bool(
                    transport_request_id
                    and transport_request_id.startswith(_INTERNAL_REQUEST_PREFIX)
                ),
            )
            if transport_request_id is not None:
                self._transport_to_request[transport_request_id] = request_id
                self._request_to_transport[request_id] = transport_request_id
            self._publish_locked(now, force=True)
            return request_id

    def observe_context(
        self,
        request_id: int,
        *,
        prompt_tokens: int,
        cached_tokens: int,
    ) -> None:
        now = float(self._clock())
        with self._lock:
            sample = self._requests.get(request_id)
            if sample is None:
                return
            sample.prompt_tokens = max(0, int(prompt_tokens))
            sample.cached_tokens = max(
                0,
                min(sample.prompt_tokens, int(cached_tokens)),
            )
            sample.prefill_started_at = now
            sample.prefill_updated_at = now
            sample.prefill_processed_tokens = 0
            sample.prefill_total_tokens = max(
                0,
                sample.prompt_tokens - sample.cached_tokens,
            )
            sample.prefill_speed = 0.0
            sample.prefill_average_speed = 0.0
            sample.updated_at = now
            self._publish_locked(now, force=True)

    def observe_prefill_progress(
        self,
        uid: Any,
        *,
        processed_tokens: int,
        total_tokens: int,
    ) -> None:
        """Publish MLX-LM's real chunk-level prompt progress.

        ``BatchGenerator`` already reports ``(processed, total)`` for every
        prefill chunk. The normal oMLX dashboard consumes the same signal; the
        distributed worker previously threw it away and could therefore show
        only a completed prefill rate after the first generated token.
        """

        now = float(self._clock())
        with self._lock:
            request_id = self._uid_to_request.get(uid)
            sample = self._requests.get(request_id) if request_id is not None else None
            if sample is None:
                return

            total = max(0, int(total_tokens))
            processed = max(0, min(total, int(processed_tokens)))
            previous_processed = sample.prefill_processed_tokens
            previous_at = sample.prefill_updated_at
            if sample.prefill_started_at is None:
                sample.prefill_started_at = sample.started_at
            if (
                previous_at is not None
                and now > previous_at
                and processed > previous_processed
            ):
                sample.prefill_speed = (
                    processed - previous_processed
                ) / (now - previous_at)
            elif (
                processed > 0
                and now > sample.prefill_started_at
                and sample.prefill_speed <= 0
            ):
                sample.prefill_speed = processed / (
                    now - sample.prefill_started_at
                )
            prefill_elapsed = now - sample.prefill_started_at
            if processed > 0 and prefill_elapsed > 0:
                sample.prefill_average_speed = processed / prefill_elapsed

            sample.prefill_processed_tokens = processed
            sample.prefill_total_tokens = total
            sample.prefill_updated_at = now
            sample.updated_at = now
            self._publish_locked(now, force=True)

    def has_active_prefill(self) -> bool:
        """Whether a request still has prompt chunks left on this rank."""

        with self._lock:
            return any(
                sample.first_token_at is None
                and sample.prefill_total_tokens > 0
                and sample.prefill_processed_tokens < sample.prefill_total_tokens
                for sample in self._requests.values()
            )

    def active_request_count(self) -> int:
        """Current rank-local requests, for quiescence-gated maintenance."""

        with self._lock:
            return len(self._requests)

    def observe_phase_handoff(
        self,
        *,
        tensor_bytes: int,
        array_count: int,
        elapsed_seconds: float,
        queue_depth: int = 0,
    ) -> None:
        """Publish one completed full-replica prefill→decode cache handoff."""

        now = float(self._clock())
        elapsed = max(0.0, float(elapsed_seconds))
        nbytes = max(0, int(tensor_bytes))
        with self._lock:
            self._phase_split = {
                "handoffs_completed": int(
                    (self._phase_split or {}).get("handoffs_completed", 0)
                )
                + 1,
                "last_handoff_bytes": nbytes,
                "last_handoff_arrays": max(0, int(array_count)),
                "last_handoff_seconds": elapsed,
                "last_handoff_bytes_per_second": (
                    nbytes / elapsed if elapsed > 0 else 0.0
                ),
                "queue_depth": max(0, int(queue_depth)),
            }
            self._publish_locked(now, force=True)

    def cancel_request(self, request_id: int) -> None:
        """Retire one request cancelled by a serving adapter without a batch UID."""

        now = float(self._clock())
        with self._lock:
            if self._finish_locked(request_id, now=now, status="cancelled"):
                self._publish_locked(now, force=True)

    def make_cancel_vote(self, uids: Any) -> dict[str, Any]:
        """Create the next rank-zero-owned, monotonically ordered cancel vote."""

        try:
            candidates = tuple(uids)
        except TypeError as exc:
            raise RuntimeError("distributed cancel vote UIDs are not iterable") from exc
        normalized: list[int] = []
        for uid in candidates:
            if not isinstance(uid, int) or isinstance(uid, bool) or uid < 0:
                raise RuntimeError("distributed cancel vote contains an invalid UID")
            if uid not in normalized:
                normalized.append(uid)
        if not normalized or len(normalized) > _MAX_TARGETED_CANCEL_REQUESTS:
            raise RuntimeError("distributed cancel vote has an invalid UID count")
        with self._lock:
            epoch = max(
                self._accepted_cancel_vote_epoch + 1,
                self._last_cancel_epoch,
                1,
            )
        return {
            "kind": _CANCEL_VOTE_KIND,
            "schema_version": _CANCEL_VOTE_SCHEMA_VERSION,
            "epoch": epoch,
            "uids": normalized,
        }

    def accept_cancel_vote(self, vote: Any) -> tuple[int, list[int]]:
        """Validate one shared cancel epoch before any rank mutates its batch."""

        if (
            not isinstance(vote, dict)
            or vote.get("kind") != _CANCEL_VOTE_KIND
            or vote.get("schema_version") != _CANCEL_VOTE_SCHEMA_VERSION
        ):
            raise RuntimeError("distributed cancel vote envelope is invalid")
        epoch = vote.get("epoch")
        uids = vote.get("uids")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
            raise RuntimeError("distributed cancel vote epoch is invalid")
        if not isinstance(uids, list):
            raise RuntimeError("distributed cancel vote UID list is invalid")
        normalized: list[int] = []
        for uid in uids:
            if not isinstance(uid, int) or isinstance(uid, bool) or uid < 0:
                raise RuntimeError("distributed cancel vote contains an invalid UID")
            if uid in normalized:
                raise RuntimeError("distributed cancel vote contains duplicate UIDs")
            normalized.append(uid)
        if not normalized or len(normalized) > _MAX_TARGETED_CANCEL_REQUESTS:
            raise RuntimeError("distributed cancel vote has an invalid UID count")
        with self._lock:
            if epoch <= self._accepted_cancel_vote_epoch:
                raise RuntimeError(
                    "distributed cancel vote epoch did not advance: "
                    f"previous={self._accepted_cancel_vote_epoch}, received={epoch}"
                )
            self._accepted_cancel_vote_epoch = epoch
        return epoch, normalized

    def drain_cancel_boundary(self, epoch: int, uids: Any) -> None:
        """Drain rank-local Metal work for a validated shared cancellation."""

        with self._lock:
            generator = self._batch_generator
        drain = getattr(generator, "_omlx_drain_cancel_boundary", None)
        if not callable(drain):
            raise RuntimeError("distributed cancel vote has no live batch generator")
        drain(epoch, uids)

    def arm_cancel_boundary(self, epoch: int, uids: Any) -> None:
        """Allow exactly the agreed UID removal after the rank rendezvous."""

        with self._lock:
            generator = self._batch_generator
        arm = getattr(generator, "_omlx_arm_cancel_boundary", None)
        if not callable(arm):
            raise RuntimeError("distributed cancel vote has no live batch generator")
        arm(epoch, uids)

    def observe_token(self, request_id: int) -> None:
        now = float(self._clock())
        with self._lock:
            sample = self._requests.get(request_id)
            if sample is None:
                return
            sample.completion_tokens += 1
            sample.updated_at = now
            first = sample.first_token_at is None
            if first:
                sample.first_token_at = now
                sample.prefill_processed_tokens = sample.prefill_total_tokens
            self._publish_locked(now, force=first)

    def finish_request(self, request_id: int, *, failed: bool = False) -> None:
        now = float(self._clock())
        with self._lock:
            status = "failed" if failed else "completed"
            if self._finish_locked(request_id, now=now, status=status):
                self._publish_locked(now, force=True)

    def mark_pending_uid(self, request_id: int) -> None:
        """Remember the queue request immediately preceding batch insertion."""

        self._pending_uid.request_id = request_id

    def register_context(self, request_id: int, context: Any) -> None:
        """Retain the server context used by its synchronized cancel path."""

        cancel_immediately = False
        with self._lock:
            if request_id in self._requests:
                self._request_contexts[request_id] = context
                transport_request_id = self._request_to_transport.get(request_id)
                if transport_request_id in self._pending_transport_cancels:
                    self._pending_transport_cancels.discard(transport_request_id)
                    self._cancel_requested_requests.add(request_id)
                    cancel_immediately = True
        if cancel_immediately:
            stop = getattr(context, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception as exc:
                    logger.warning("Rank-side pending context cancel failed: %s", exc)

    def register_response_queue(self, request_id: int, queue: Any) -> None:
        """Retain the raw rank-local queue so shared removal can terminate it."""

        with self._lock:
            if request_id in self._requests:
                self._request_queues[request_id] = queue

    def bind_pending_uid(self, uids: Any) -> None:
        """Bind MLX-LM's generated batch UID to the observed queue request."""

        request_id = getattr(self._pending_uid, "request_id", None)
        self._pending_uid.request_id = None
        if request_id is None:
            return
        try:
            uid_values = tuple(uids)
        except TypeError:
            return
        if len(uid_values) != 1:
            return
        uid = uid_values[0]
        with self._lock:
            if request_id not in self._requests:
                return
            self._uid_to_request[uid] = request_id
            self._request_to_uid[request_id] = uid

    def cancel_uids(self, uids: Any) -> None:
        """Close requests removed by MLX-LM without a terminal queue item."""

        try:
            uid_values = tuple(uids)
        except TypeError:
            return
        now = float(self._clock())
        changed = False
        queues_to_close: list[Any] = []
        with self._lock:
            for uid in uid_values:
                request_id = self._uid_to_request.pop(uid, None)
                if request_id is None:
                    continue
                self._cancel_requested_requests.discard(request_id)
                self._request_to_uid.pop(request_id, None)
                queue = self._request_queues.get(request_id)
                finished = self._finish_locked(
                    request_id,
                    now=now,
                    status="cancelled",
                )
                if finished and queue is not None:
                    queues_to_close.append(queue)
                changed = finished or changed
            if changed:
                self._publish_locked(now, force=True)
        # Wake the rank-zero HTTP collector only after releasing telemetry's
        # lock. Worker dummy queues receive the same terminal sentinel, which
        # is harmless and keeps queue semantics symmetric.
        for queue in queues_to_close:
            try:
                queue.put(None)
            except Exception as exc:
                logger.debug("Could not terminate cancelled response queue: %s", exc)

    def register_batch_generator(self, generator: Any) -> None:
        """Remember the live generator for telemetry, never rank-local removal."""

        with self._lock:
            self._batch_generator = generator

    def merge_requested_cancel_uids(self, uids: Any) -> list[Any]:
        """Inject requested UIDs into MLX-LM's rank-zero cancellation vote.

        This runs on the generation thread immediately before its existing
        ``_share_object(uids_to_remove)`` broadcast.  It is therefore safe for
        distributed collectives: rank zero remains the single producer and
        every peer receives/removes the same list at the same batch boundary.
        The heartbeat thread never calls ``BatchGenerator.remove`` directly.
        """

        try:
            merged = list(uids)
        except TypeError:
            merged = []
        with self._lock:
            requested = [
                self._request_to_uid[request_id]
                for request_id in self._cancel_requested_requests
                if request_id in self._request_to_uid
                and request_id in self._requests
            ]
        for uid in requested:
            if uid not in merged:
                merged.append(uid)
        return merged

    def force_cancel_all(self, *, reason: str = "coordinator cancel") -> int:
        """Request cancellation through MLX-LM's shared server-loop path.

        Directly calling ``BatchGenerator.remove`` here is rank-local and may
        run on the heartbeat thread.  Rank zero would discard its UID while
        peers kept decoding, producing mismatched tensor collectives. Setting
        the real ``GenerationContext`` stop flag lets the pinned server append
        the UID at its next prompt/token boundary, broadcast that list, and
        remove it on every rank together. Process teardown remains the backstop
        for a rank genuinely wedged inside a collective.
        """

        with self._lock:
            self._cancel_requested_requests.update(self._requests)
            contexts = [
                context
                for request_id, context in self._request_contexts.items()
                if request_id in self._requests
            ]
        if not contexts:
            return 0
        requested = 0
        for context in contexts:
            stop = getattr(context, "stop", None)
            if not callable(stop):
                continue
            try:
                stop()
            except Exception as exc:
                logger.warning("Rank-side context cancel failed: %s", exc)
                continue
            requested += 1
        if not requested:
            return 0
        logger.warning(
            "Requested synchronized cancellation for %d active request(s): %s",
            requested,
            reason,
        )
        return requested

    def force_cancel_request(
        self,
        transport_request_id: str,
        *,
        reason: str = "coordinator cancel",
    ) -> int:
        """Stop exactly one request identified by the private HTTP transport."""

        normalized = _transport_request_id(transport_request_id)
        if normalized is None:
            return 0
        with self._lock:
            request_id = self._transport_to_request.get(normalized)
            context = self._request_contexts.get(request_id)
            active = request_id in self._requests if request_id is not None else False
            if active and request_id is not None:
                self._cancel_requested_requests.add(request_id)
        if not active or context is None:
            with self._lock:
                if len(self._pending_transport_cancels) < _MAX_TARGETED_CANCEL_REQUESTS:
                    self._pending_transport_cancels.add(normalized)
            return 0
        stop = getattr(context, "stop", None)
        if not callable(stop):
            return 0
        try:
            stop()
        except Exception as exc:
            logger.warning("Rank-side targeted context cancel failed: %s", exc)
            return 0
        logger.warning(
            "Requested synchronized cancellation for request %s: %s",
            normalized,
            reason,
        )
        return 1

    def _read_cancel_request(self) -> dict[str, Any] | None:
        path = self._cancel_path
        if path is None:
            return None
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        if len(raw.encode()) > _MAX_CANCEL_FILE_BYTES:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return None
        if (
            self._cancel_deployment_id
            and payload.get("deployment_id") != self._cancel_deployment_id
        ):
            return None
        if self._cancel_plan_hash and payload.get("plan_hash") != self._cancel_plan_hash:
            return None
        epoch = payload.get("epoch")
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            return None
        if epoch < self._cancel_epoch_floor:
            return None
        return payload

    def _write_cancel_ack(self, *, epoch: int, cancelled: int) -> None:
        path = self._cancel_path
        if path is None:
            return
        ack = path.with_name(path.stem + "-ack.json")
        payload = {
            "schema_version": 1,
            "deployment_id": self._cancel_deployment_id,
            "plan_hash": self._cancel_plan_hash,
            "epoch": epoch,
            "cancelled": cancelled,
            "at": time.time(),
        }
        try:
            temporary = ack.with_name(ack.name + ".tmp")
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True)
            os.replace(temporary, ack)
        except OSError:
            # The ack is advisory; the marker's active_requests count is the
            # authoritative drain evidence.
            return

    def poll_cancel_requests(self, *, min_interval: float = 1.0) -> int:
        """Consume the coordinator's cancel file, at most one read a second.

        Each epoch is consumed once. The ack file lets the coordinator
        confirm the cancel reached the rank without guessing.
        """

        now = float(self._clock())
        if now - self._last_cancel_poll_at < max(0.0, min_interval):
            return 0
        self._last_cancel_poll_at = now
        payload = self._read_cancel_request()
        if payload is None:
            return 0
        epoch = payload["epoch"]
        if epoch <= self._last_cancel_epoch:
            return 0
        self._last_cancel_epoch = epoch
        reason = str(payload.get("reason") or "coordinator cancel")
        if payload.get("scope") == "request":
            cancelled = self.force_cancel_request(
                payload.get("request_id"),
                reason=reason,
            )
        elif payload.get("scope") == "requests":
            request_ids = payload.get("request_ids")
            if not isinstance(request_ids, list):
                cancelled = 0
            else:
                normalized = []
                for request_id in request_ids[:_MAX_TARGETED_CANCEL_REQUESTS]:
                    request_id = _transport_request_id(request_id)
                    if request_id is not None and request_id not in normalized:
                        normalized.append(request_id)
                cancelled = sum(
                    self.force_cancel_request(request_id, reason=reason)
                    for request_id in normalized
                )
        elif payload.get("scope") == "all":
            cancelled = self.force_cancel_all(reason=reason)
        else:
            cancelled = 0
        self._write_cancel_ack(epoch=epoch, cancelled=cancelled)
        return cancelled

    def observe_batch_step(
        self,
        *,
        prompt_responses: int,
        generation_responses: int,
        elapsed_seconds: float,
    ) -> None:
        """Record one coalesced MLX-LM continuous-batching step."""

        # Throttled to one read a second; gives a coordinator cancel ~1 s
        # latency while the batch loop is actively stepping.
        self.poll_cancel_requests()
        now = float(self._clock())
        elapsed = max(0.0, float(elapsed_seconds))
        with self._lock:
            idle = max(0.0, now - self._last_step_finished_at - elapsed)
            self._idle_seconds += idle
            self._busy_seconds += elapsed
            self._last_step_finished_at = now
            self._batch_steps += 1
            generation_rows = max(0, int(generation_responses))
            prompt_rows = max(0, int(prompt_responses))
            internal_only = bool(self._requests) and all(
                sample.internal for sample in self._requests.values()
            )
            visible_generation_rows = 0 if internal_only else generation_rows
            visible_prompt_rows = 0 if internal_only else prompt_rows
            if visible_generation_rows > 0 and elapsed > 0.0:
                self._decode_step_tokens_total += visible_generation_rows
                self._decode_step_seconds_total += elapsed
            coalesced = max(visible_prompt_rows, visible_generation_rows)
            self._last_batch = {
                "step_seconds": elapsed,
                "prompt_responses": visible_prompt_rows,
                "generation_responses": visible_generation_rows,
                "coalesced_batch_size": max(0, int(coalesced)),
            }
            self._publish_locked(now, force=False)

    def observe_cache_lookup(
        self,
        *,
        prompt_tokens: int,
        remaining_tokens: int,
        entries: int,
        nbytes: int,
        memory_entries: int | None = None,
        memory_bytes: int | None = None,
        ssd_entries: int | None = None,
        ssd_bytes: int | None = None,
        ssd_max_bytes: int | None = None,
        ssd_capacity_bytes: int | None = None,
        ssd_evictions: int | None = None,
        ssd_capacity_drops: int | None = None,
        ssd_pending_bytes: int | None = None,
        ssd_pending_max_bytes: int | None = None,
        ssd_write_failures: int | None = None,
        hit_tier: str | None = None,
    ) -> None:
        now = float(self._clock())
        prompt = max(0, int(prompt_tokens))
        remaining = max(0, min(prompt, int(remaining_tokens)))
        reused = prompt - remaining
        with self._lock:
            self._cache_lookups += 1
            if reused > 0:
                self._cache_hits += 1
                self._cache_tokens_reused += reused
                if hit_tier == "ssd":
                    self._cache_ssd_hits += 1
                else:
                    # The optional argument preserves the old in-memory-only
                    # caller contract while making durable restores explicit.
                    self._cache_memory_hits += 1
            self._cache_entries = max(0, int(entries))
            self._cache_bytes = max(0, int(nbytes))
            if memory_entries is None and ssd_entries is None:
                # Backward-compatible callers report the volatile tier only.
                memory_entries = entries
                memory_bytes = nbytes
            if memory_entries is not None:
                self._cache_memory_entries = max(0, int(memory_entries))
            if memory_bytes is not None:
                self._cache_memory_bytes = max(0, int(memory_bytes))
            if ssd_entries is not None:
                self._cache_ssd_entries = max(0, int(ssd_entries))
            if ssd_bytes is not None:
                self._cache_ssd_bytes = max(0, int(ssd_bytes))
            self._observe_ssd_capacity_locked(
                max_bytes=ssd_max_bytes,
                capacity_bytes=ssd_capacity_bytes,
                evictions=ssd_evictions,
                capacity_drops=ssd_capacity_drops,
                pending_bytes=ssd_pending_bytes,
                pending_max_bytes=ssd_pending_max_bytes,
                write_failures=ssd_write_failures,
            )
            self._publish_locked(now, force=False)

    def observe_cache_state(
        self,
        *,
        entries: int,
        nbytes: int,
        memory_entries: int | None = None,
        memory_bytes: int | None = None,
        ssd_entries: int | None = None,
        ssd_bytes: int | None = None,
        ssd_max_bytes: int | None = None,
        ssd_capacity_bytes: int | None = None,
        ssd_evictions: int | None = None,
        ssd_capacity_drops: int | None = None,
        ssd_pending_bytes: int | None = None,
        ssd_pending_max_bytes: int | None = None,
        ssd_write_failures: int | None = None,
    ) -> None:
        now = float(self._clock())
        with self._lock:
            self._cache_entries = max(0, int(entries))
            self._cache_bytes = max(0, int(nbytes))
            if memory_entries is None and ssd_entries is None:
                memory_entries = entries
                memory_bytes = nbytes
            if memory_entries is not None:
                self._cache_memory_entries = max(0, int(memory_entries))
            if memory_bytes is not None:
                self._cache_memory_bytes = max(0, int(memory_bytes))
            if ssd_entries is not None:
                self._cache_ssd_entries = max(0, int(ssd_entries))
            if ssd_bytes is not None:
                self._cache_ssd_bytes = max(0, int(ssd_bytes))
            self._observe_ssd_capacity_locked(
                max_bytes=ssd_max_bytes,
                capacity_bytes=ssd_capacity_bytes,
                evictions=ssd_evictions,
                capacity_drops=ssd_capacity_drops,
                pending_bytes=ssd_pending_bytes,
                pending_max_bytes=ssd_pending_max_bytes,
                write_failures=ssd_write_failures,
            )
            self._publish_locked(now, force=False)

    def _observe_ssd_capacity_locked(
        self,
        *,
        max_bytes: int | None,
        capacity_bytes: int | None,
        evictions: int | None,
        capacity_drops: int | None,
        pending_bytes: int | None,
        pending_max_bytes: int | None,
        write_failures: int | None,
    ) -> None:
        values = {
            "_cache_ssd_max_bytes": max_bytes,
            "_cache_ssd_capacity_bytes": capacity_bytes,
            "_cache_ssd_evictions": evictions,
            "_cache_ssd_capacity_drops": capacity_drops,
            "_cache_ssd_pending_bytes": pending_bytes,
            "_cache_ssd_pending_max_bytes": pending_max_bytes,
            "_cache_ssd_write_failures": write_failures,
        }
        for name, value in values.items():
            if value is not None:
                setattr(self, name, max(0, int(value)))

    def _finish_locked(
        self,
        request_id: int,
        *,
        now: float,
        status: str,
    ) -> bool:
        sample = self._requests.pop(request_id, None)
        if sample is None:
            return False
        self._cancel_requested_requests.discard(request_id)
        if getattr(self._pending_uid, "request_id", None) == request_id:
            self._pending_uid.request_id = None
        uid = self._request_to_uid.pop(request_id, None)
        self._request_contexts.pop(request_id, None)
        self._request_queues.pop(request_id, None)
        transport_request_id = self._request_to_transport.pop(request_id, None)
        if transport_request_id is not None:
            self._transport_to_request.pop(transport_request_id, None)
            self._pending_transport_cancels.discard(transport_request_id)
        if uid is not None:
            self._uid_to_request.pop(uid, None)
        sample.updated_at = now
        if sample.internal:
            # Readiness canaries prove the ranks but are not user inference.
            # Do not turn a four-token load probe into Request #1, poison the
            # displayed prefill rate, or dilute lifetime aggregate throughput.
            return True
        if status == "failed":
            self._requests_failed += 1
        elif status == "cancelled":
            self._requests_cancelled += 1
        else:
            self._requests_completed += 1
        self._prompt_tokens_total += sample.prompt_tokens
        self._completion_tokens_total += sample.completion_tokens
        self._cached_tokens_total += sample.cached_tokens
        if sample.first_token_at is not None and sample.completion_tokens > 0:
            self._decode_seconds_total += max(0.0, now - sample.first_token_at)
        self._last_completed = self._sample_snapshot(
            sample,
            now=now,
            status=status,
        )
        return True

    def snapshot(self) -> dict[str, Any]:
        now = float(self._clock())
        with self._lock:
            return self._snapshot_locked(now)

    def _sample_snapshot(
        self,
        sample: _RequestSample,
        *,
        now: float,
        status: str,
    ) -> dict[str, Any]:
        elapsed = max(0.0, now - sample.started_at)
        first = sample.first_token_at
        ttft = max(0.0, first - sample.started_at) if first is not None else None
        uncached_prompt = max(0, sample.prompt_tokens - sample.cached_tokens)
        prefill_tps = (
            uncached_prompt / ttft
            if ttft is not None and ttft > 0
            else sample.prefill_speed
        )
        prefill_total = sample.prefill_total_tokens or uncached_prompt
        prefill_processed = min(
            prefill_total,
            max(0, sample.prefill_processed_tokens),
        )
        prefill_started_at = sample.prefill_started_at or sample.started_at
        prefill_active = status == "running" and first is None
        # Freeze the prefill clock when prefill is done. ``now`` is the
        # snapshot/finish instant, so an unfrozen elapsed keeps growing through
        # the whole decode phase (observed: elapsed 25.3 s with ttft 2.49 s)
        # and average_speed collapses to a few tok/s on every long decode.
        # The last progress callback lands with the final chunk; when no
        # callback ever arrived, the first token is the honest upper bound.
        if prefill_active:
            prefill_end = now
        elif (
            prefill_total > 0
            and prefill_processed >= prefill_total
            and sample.prefill_updated_at is not None
        ):
            prefill_end = sample.prefill_updated_at
        elif first is not None:
            prefill_end = first
        else:
            prefill_end = now
        prefill_elapsed = max(0.0, prefill_end - prefill_started_at)
        prefill_average_speed = (
            prefill_processed / prefill_elapsed
            if prefill_processed > 0 and prefill_elapsed > 0
            else sample.prefill_average_speed
        )
        if ttft is None:
            # Before the first token, the only truthful request-level rate is
            # processed prompt tokens divided by total time so far. The recent
            # chunk remains separate below for ETA and diagnostics.
            prefill_tps = prefill_average_speed or sample.prefill_speed
        prefill_remaining = max(0, prefill_total - prefill_processed)
        prefill_eta = (
            prefill_remaining / sample.prefill_speed
            if prefill_active and sample.prefill_speed > 0
            else None
        )
        decode_seconds = max(0.0, now - first) if first is not None else 0.0
        decode_intervals = max(0, sample.completion_tokens - 1)
        decode_tps = (
            decode_intervals / decode_seconds
            if decode_intervals and decode_seconds > 0
            else 0.0
        )
        end_to_end_tps = sample.completion_tokens / elapsed if elapsed > 0 else 0.0
        return {
            "request_id": sample.request_id,
            "status": status,
            "prompt_tokens": sample.prompt_tokens,
            "cached_tokens": sample.cached_tokens,
            "completion_tokens": sample.completion_tokens,
            "elapsed_seconds": elapsed,
            "ttft_seconds": ttft,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "end_to_end_tps": end_to_end_tps,
            "prefill_progress": {
                "active": prefill_active,
                "processed": prefill_processed,
                "total": prefill_total,
                "speed": sample.prefill_speed,
                "average_speed": prefill_average_speed,
                "eta": prefill_eta,
                "elapsed": prefill_elapsed,
            },
        }

    def _snapshot_locked(self, now: float) -> dict[str, Any]:
        active_samples = sorted(
            (
                sample
                for sample in self._requests.values()
                if not sample.internal
            ),
            key=lambda item: item.request_id,
        )
        active_sample = (
            max(active_samples, key=lambda item: item.updated_at)
            if active_samples
            else None
        )
        current = (
            self._sample_snapshot(active_sample, now=now, status="running")
            if active_sample is not None
            else self._last_completed
        )
        active_completion_tokens = sum(
            sample.completion_tokens for sample in active_samples
        )
        uptime = max(0.0, now - self._started_at)
        total_generated = self._completion_tokens_total + active_completion_tokens
        active_decode_seconds = sum(
            now - sample.first_token_at
            for sample in active_samples
            if sample.first_token_at is not None
        )
        decode_seconds = self._decode_seconds_total + active_decode_seconds
        utilization_denominator = self._busy_seconds + self._idle_seconds
        pipeline_utilization = (
            self._busy_seconds / utilization_denominator
            if utilization_denominator > 0
            else 0.0
        )
        cache_hit_rate = (
            self._cache_hits / self._cache_lookups if self._cache_lookups else 0.0
        )
        result = {
            "scope": "end_to_end_pipeline",
            "active_requests": len(active_samples),
            "requests_completed": self._requests_completed,
            "requests_failed": self._requests_failed,
            "requests_cancelled": self._requests_cancelled,
            "prompt_tokens_total": self._prompt_tokens_total,
            "completion_tokens_total": self._completion_tokens_total,
            "cached_tokens_total": self._cached_tokens_total,
            # ``last_request`` is retained for older dashboards.  New clients
            # use this stable request-id ordered collection so concurrent
            # prompt/decode rates never overwrite one another before the API
            # boundary.
            "active_request_metrics": [
                self._sample_snapshot(sample, now=now, status="running")
                for sample in active_samples[:_MAX_ACTIVE_REQUEST_METRICS]
            ],
            "active_request_metrics_truncated": max(
                0,
                len(active_samples) - _MAX_ACTIVE_REQUEST_METRICS,
            ),
            "aggregate_decode_tps": (
                self._decode_step_tokens_total / self._decode_step_seconds_total
                if self._decode_step_seconds_total > 0
                else 0.0
            ),
            "average_request_decode_tps": (
                total_generated / decode_seconds if decode_seconds > 0 else 0.0
            ),
            # The pre-fix semantic (tokens / process uptime, idle included),
            # kept under an honest name for capacity planning views.
            "aggregate_wall_tps": (total_generated / uptime if uptime > 0 else 0.0),
            "cache": {
                "affinity": (
                    "deployment"
                    if self._execution is not None and self._execution.cache_affinity
                    else "none"
                ),
                "lookups": self._cache_lookups,
                "hits": self._cache_hits,
                "misses": self._cache_lookups - self._cache_hits,
                "hit_rate": cache_hit_rate,
                "tokens_reused": self._cache_tokens_reused,
                "entries": self._cache_entries,
                "bytes": self._cache_bytes,
                "ssd_enabled": self._cache_ssd_enabled,
                "memory": {
                    "entries": self._cache_memory_entries,
                    "bytes": self._cache_memory_bytes,
                    "hits": self._cache_memory_hits,
                },
                "ssd": {
                    "entries": self._cache_ssd_entries,
                    "bytes": self._cache_ssd_bytes,
                    "hits": self._cache_ssd_hits,
                    "max_bytes": self._cache_ssd_max_bytes,
                    "capacity_bytes": self._cache_ssd_capacity_bytes,
                    "evictions": self._cache_ssd_evictions,
                    "capacity_drops": self._cache_ssd_capacity_drops,
                    "pending_bytes": self._cache_ssd_pending_bytes,
                    "pending_max_bytes": self._cache_ssd_pending_max_bytes,
                    "write_failures": self._cache_ssd_write_failures,
                },
            },
            "pipeline": {
                "batch_steps": self._batch_steps,
                "busy_seconds": self._busy_seconds,
                "idle_seconds": self._idle_seconds,
                "utilization": pipeline_utilization,
                "microbatch_target": (
                    self._execution.pipeline_microbatch_size
                    if self._execution is not None
                    else 1
                ),
                "async_overlap": (
                    self._execution.async_overlap
                    if self._execution is not None
                    else False
                ),
                "last_batch": self._last_batch,
            },
            "last_request": current,
        }
        if self._phase_split is not None:
            result["phase_split"] = copy.deepcopy(self._phase_split)
        try:
            from omlx.patches.mlx_lm_mtp.batch_generator import (
                mtp_runtime_stats_snapshot,
            )

            mtp_stats = mtp_runtime_stats_snapshot()
        except Exception:
            mtp_stats = None
        if mtp_stats is not None:
            result["mtp"] = mtp_stats
        if self._execution is not None:
            result["execution"] = self._execution.to_dict()
        if self._assignment is not None:
            result["stage"] = {
                "rank": self._assignment.rank,
                "predicted_compute_seconds": (
                    self._assignment.predicted_compute_seconds
                ),
                "predicted_send_seconds": self._assignment.predicted_send_seconds,
                "predicted_stage_seconds": self._assignment.predicted_stage_seconds,
                "observed_step_seconds": (
                    self._last_batch["step_seconds"]
                    if self._last_batch is not None
                    else None
                ),
            }
        return result

    def _publish_locked(self, now: float, *, force: bool) -> None:
        if not force and now - self._last_publish_at < self._publish_interval:
            return
        snapshot = self._snapshot_locked(now)
        if not _finite_metrics(snapshot):
            return
        self._last_publish_at = now
        try:
            self._marker.update("ready", metrics=snapshot)
        except OSError:
            # Runtime visibility is deliberately fail-soft. A full disk or
            # unavailable state directory must never interrupt generation.
            return


def _finite_metrics(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_metrics(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_metrics(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value) and value >= 0
    return True


class _TelemetryQueue:
    """Observe MLX-LM's rank-local response queue without changing its API."""

    def __init__(
        self,
        queue: Any,
        telemetry: RuntimeTelemetry,
        transport_request_id: str | None = None,
    ) -> None:
        self._queue = queue
        self._telemetry = telemetry
        self._request_id = telemetry.begin_request(transport_request_id)
        telemetry.register_response_queue(self._request_id, queue)
        self._finished = False

    def put(self, item: Any, *args: Any, **kwargs: Any) -> Any:
        if not self._finished:
            if item is None:
                self._finished = True
                self._telemetry.finish_request(self._request_id)
            elif isinstance(item, BaseException):
                self._finished = True
                self._telemetry.finish_request(self._request_id, failed=True)
            elif hasattr(item, "prompt") and hasattr(item, "prompt_cache_count"):
                prompt = getattr(item, "prompt", ())
                cache_count = getattr(item, "prompt_cache_count", 0)
                self._telemetry.observe_context(
                    self._request_id,
                    prompt_tokens=len(prompt),
                    cached_tokens=max(0, int(cache_count)),
                )
                self._telemetry.mark_pending_uid(self._request_id)
                self._telemetry.register_context(self._request_id, item)
            elif hasattr(item, "token") and hasattr(item, "finish_reason"):
                self._telemetry.observe_token(self._request_id)
        return self._queue.put(item, *args, **kwargs)


@contextmanager
def install_server_telemetry(
    marker: MarkerWriter,
    *,
    execution: ExecutionSettings | None = None,
    assignment: PipelineAssignment | None = None,
    prefill_guard: Any | None = None,
    heartbeat_interval: float = _DEFAULT_HEARTBEAT_INTERVAL,
    ssd_cache_dir: str | None = None,
    ssd_max_entries: int = 512,
    ssd_max_bytes: int | None = None,
    ssd_cache_persistent: bool = False,
    ssd_write_behind: bool = False,
    prefill_step_size: int = 2048,
    control_plane: Any | None = None,
) -> Iterator[RuntimeTelemetry]:
    """Patch the pinned worker's generator at its rank-local queue boundary.

    Also starts the idle heartbeat for as long as the rank is serving, so a
    marker that has stopped ageing means the rank has stopped, not that nobody
    has asked it anything.

    When ``ssd_cache_dir`` is set, both generation paths additionally snapshot
    the prompt cache to SSD at prefill boundaries (the batched generator as its
    per-uid progress crosses each aligned boundary, the sequential path through
    a chained progress callback) and restore from it on an in-memory miss, so a
    model whose per-layer state cannot be sliced still reuses a long prefix
    instead of recomputing it. The restore boundary is agreed across ranks so a
    disk write that failed on one rank cannot desync the pipeline. With
    ``ssd_cache_persistent`` the compact plan-scoped index survives rank
    restarts; otherwise this context retains the legacy process-lifetime tier.
    """

    import mlx.core as mx
    import mlx_lm.server as mlx_server

    original = mlx_server.ResponseGenerator
    original_batch_generator = mlx_server.BatchGenerator
    original_prompt_cache = mlx_server.LRUPromptCache
    original_generation_context = mlx_server.GenerationContext
    original_time_budget = mlx_server.TimeBudget
    original_handle_completion = mlx_server.APIHandler.handle_completion
    original_do_post = mlx_server.APIHandler.do_POST
    cancellation_state = threading.local()
    marker_path = getattr(marker, "path", None)
    marker_payload = getattr(marker, "payload", None)
    marker_deployment_id = (
        str(marker_payload.get("deployment_id") or "")
        if isinstance(marker_payload, dict)
        else ""
    )
    marker_plan_hash = (
        str(marker_payload.get("plan_hash") or "")
        if isinstance(marker_payload, dict)
        else ""
    )
    worker_cancel_epoch_floor = int(time.time() * 1000)
    telemetry = RuntimeTelemetry(
        marker,
        execution=execution,
        assignment=assignment,
        heartbeat_interval=heartbeat_interval,
        cancel_path=(
            Path(marker_path).parent / f"{marker_deployment_id}-cancel.json"
            if marker_path is not None and marker_deployment_id
            else None
        ),
        cancel_deployment_id=marker_deployment_id,
        cancel_plan_hash=marker_plan_hash,
        cancel_epoch_floor=worker_cancel_epoch_floor,
        prompt_cache_ssd_enabled=bool(ssd_cache_dir),
        prompt_cache_ssd_max_bytes=(
            int(ssd_max_bytes)
            if ssd_max_bytes is not None
            else execution.prompt_cache_ssd_max_bytes
            if execution is not None
            else DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES
        )
        if ssd_cache_dir
        else 0,
    )

    try:
        group = mx.distributed.init()
        world_size = int(group.size())
        rank = int(group.rank())
    except Exception:
        world_size = 1
        rank = 0

    def agree_snapshot_capacity_charge(local_charge: int | None) -> int | None:
        """Largest rank file charge, or None if any rank cannot snapshot.

        All ranks account the same largest charge even though their layer
        slices have different physical sizes. The common charge plus the
        operation-ordered LRU makes byte eviction rank-symmetric.
        """

        return _agreed_snapshot_capacity_charge(
            local_charge,
            world_size=world_size,
            rank=rank,
            control_plane=control_plane,
        )

    snapshot_ctx = threading.local()
    prompt_cache_instances: list[Any] = []
    snapshot_step = max(1, int(prefill_step_size))
    ssd_store = None
    if ssd_cache_dir:
        from .prompt_snapshot_cache import (
            SSDPromptSnapshotStore,
            agreed_boundary,
            candidate_boundaries,
        )

        resolved_ssd_max_bytes = int(
            ssd_max_bytes
            or (
                execution.prompt_cache_ssd_max_bytes
                if execution is not None
                else DEFAULT_PROMPT_CACHE_SSD_MAX_BYTES
            )
        )
        ssd_store = SSDPromptSnapshotStore(
            ssd_cache_dir,
            step=snapshot_step,
            max_entries=ssd_max_entries,
            max_bytes=resolved_ssd_max_bytes,
            persistent=ssd_cache_persistent,
            write_behind=ssd_write_behind,
            capacity_agreement=agree_snapshot_capacity_charge,
        )

    def ssd_store_observability() -> dict[str, int]:
        if ssd_store is None:
            return {}
        return {
            "ssd_max_bytes": ssd_store.max_bytes,
            "ssd_capacity_bytes": ssd_store.capacity_bytes,
            "ssd_evictions": ssd_store.evictions,
            "ssd_capacity_drops": ssd_store.capacity_drops,
            "ssd_pending_bytes": ssd_store.pending_bytes,
            "ssd_pending_max_bytes": ssd_store.pending_max_bytes,
            "ssd_write_failures": ssd_store.write_failures,
        }

    def prompt_cache_positions(cache: Any) -> tuple[int, ...]:
        """Best-effort logical token positions for a rank-local cache copy."""

        positions: list[int] = []
        for item in cache or ():
            nested = getattr(item, "caches", None)
            entries = nested if isinstance(nested, (list, tuple)) else (item,)
            for entry in entries:
                # DS4 PoolingCache.offset counts compressed rows rather than
                # source tokens.  Reconstruct the logical source position so
                # a failed arbitrary trim cannot masquerade as a valid hit.
                ratio = getattr(entry, "ratio", None)
                remainder = getattr(entry, "remainder", None)
                pooled = getattr(entry, "_pool_len", None)
                if all(isinstance(value, int) for value in (ratio, remainder, pooled)):
                    positions.append(int(pooled) * int(ratio) + int(remainder))
                    continue
                offset = getattr(entry, "offset", None)
                if isinstance(offset, int):
                    positions.append(offset)
        return tuple(positions)

    def prompt_cache_position_trace(cache: Any) -> tuple[dict[str, Any], ...]:
        """Describe cached offsets for an operator-only DS4 reuse diagnosis."""

        rows: list[dict[str, Any]] = []
        for item in cache or ():
            nested = getattr(item, "caches", None)
            entries = nested if isinstance(nested, (list, tuple)) else (item,)
            for entry in entries:
                row: dict[str, Any] = {"type": type(entry).__name__}
                for name in ("ratio", "remainder", "_pool_len", "offset", "max_size"):
                    value = getattr(entry, name, None)
                    if isinstance(value, int):
                        row[name] = value
                rows.append(row)
        return tuple(rows)

    def agree_prompt_cache_plan(
        cache: Any,
        tokens: list[int],
        rest: list[int],
    ) -> tuple[Any, list[int]]:
        """Use a cache only when every rank will prefill the same suffix.

        Pipeline stages can have different cache types.  In particular, one
        DS4 stage may be able to trim a longer cached conversation to a common
        prefix while another stage containing a recurrent cache cannot.  The
        stock rank-local lookup then gives the stages different input lengths;
        JACCL reports ``IBV_WC_LOC_LEN_ERR`` when the activation send reaches a
        receive posted for the other length.

        Exchange both the reused-prefix and suffix lengths before the first
        model call.  Any disagreement makes every rank discard its local copy
        and perform the same full prefill.  The reliable TCP control plane is
        preferred because these are owned scalar values, not tensor
        reductions; the fixed-shape reduction is retained for older launchers.
        """

        if world_size <= 1:
            return cache, rest

        reused = len(tokens) - len(rest) if cache is not None else 0
        suffix = len(rest) if cache is not None else len(tokens)
        positions = prompt_cache_positions(cache)
        incoherent = int(any(position != reused for position in positions))
        if (
            incoherent
            and os.environ.get("OMLX_CLUSTER_CACHE_TRACE", "0") == "1"
        ):
            logger.warning(
                "Prompt-cache position diagnostic rank=%s reused=%s positions=%s details=%s",
                rank,
                reused,
                positions,
                prompt_cache_position_trace(cache),
            )
        local = (reused, suffix, incoherent)
        plans: list[tuple[int, int, int]] = []
        if control_plane is not None:
            for source in range(world_size):
                payload = (
                    struct.pack("!QQQ", *local) if rank == source else None
                )
                packet = control_plane.broadcast_owned_bytes(
                    payload,
                    source_rank=source,
                    expected_size=24,
                )
                plans.append(struct.unpack("!QQQ", packet))
        else:
            fields = [0] * (3 * world_size)
            fields[3 * rank : 3 * rank + 3] = local
            agreed = mx.distributed.all_sum(
                mx.array(fields, dtype=mx.int32)
            )
            mx.eval(agreed)
            values = [int(value) for value in agreed.tolist()]
            plans = [
                tuple(values[3 * source : 3 * source + 3])
                for source in range(world_size)
            ]

        if not any(plan[2] for plan in plans) and all(
            plan == plans[0] for plan in plans[1:]
        ):
            return cache, rest

        logger.warning(
            "Prompt-cache plan diverged across ranks (%s); rebuilding the "
            "request with a full synchronized prefill",
            plans,
        )
        return None, list(tokens)

    def agree_ssd_boundary(model: Any, tokens: list[int]) -> int:
        """Longest prefix boundary every rank can restore for ``tokens``, or 0.

        Each rank votes the boundaries it holds on disk; a boundary is taken
        only when the whole group has it, so a rank whose snapshot write failed
        simply drops that boundary rather than reading a prefix the others
        recompute. This is the same collective-agreement shape the prefill guard
        already uses, and it runs only on the sequential path where every rank
        reaches it in lockstep.
        """

        # Capped at len - 1: the pinned batched server cannot insert a request
        # whose segments were all consumed (insert_segments indexes seq[-1])
        # and the sequential generate_step rejects an empty prompt, so a full
        # hit must leave the last token unprocessed. The cap is computed from
        # the broadcast prompt length, so it is identical on every rank.
        local = set(ssd_store.present_boundaries(model, tokens))
        candidates = candidate_boundaries(len(tokens) - 1, snapshot_step)
        if not candidates:
            return 0
        if world_size <= 1:
            usable = local.intersection(candidates)
            return max(usable) if usable else 0
        vote = mx.array([1 if c in local else 0 for c in candidates], dtype=mx.int32)
        agreed = mx.distributed.all_sum(vote).tolist()
        return agreed_boundary(candidates, agreed, world_size)

    class TelemetryBatchGenerator(original_batch_generator):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if execution is not None:
                microbatch = execution.pipeline_microbatch_size
                if "completion_batch_size" in kwargs:
                    kwargs["completion_batch_size"] = min(
                        int(kwargs["completion_batch_size"]),
                        microbatch,
                    )
                if "prefill_batch_size" in kwargs:
                    kwargs["prefill_batch_size"] = min(
                        int(kwargs["prefill_batch_size"]),
                        microbatch,
                    )
                if execution.max_kv_size is not None:
                    kwargs.setdefault("max_kv_size", execution.max_kv_size)
            super().__init__(*args, **kwargs)
            # Full token sequence per in-flight uid, so a boundary snapshot can
            # be keyed while the batched prefill is still running.
            self._omlx_tokens: dict[Any, list[int]] = {}
            self._omlx_batch_trace_steps = 0
            self._omlx_prepared_cancel_vote: tuple[int, tuple[int, ...]] | None = None
            telemetry.register_batch_generator(self)

        @staticmethod
        def _at_generation_boundary(sequence: Any) -> bool:
            """Whether one staged sequence has only its decode seed token."""

            return bool(
                isinstance(sequence, (tuple, list))
                and len(sequence) == 1
                and isinstance(sequence[0], (tuple, list))
                and len(sequence[0]) == 1
            )

        def _about_to_enter_generation(self) -> bool:
            current = getattr(self, "_currently_processing", ())
            if any(
                self._at_generation_boundary(item[0])
                for item in current
                if isinstance(item, (tuple, list)) and item
            ):
                return True
            pending = getattr(self, "_unprocessed_sequences", ())
            return any(
                self._at_generation_boundary(item[1])
                for item in pending
                if isinstance(item, (tuple, list)) and len(item) > 1
            )

        def _generation_boundary_uids(self) -> tuple[Any, ...]:
            """Prompt-batch UIDs whose final token moves to generation next."""

            current = getattr(self, "_currently_processing", ())
            prompt_batch = getattr(self, "_prompt_batch", None)
            uids = tuple(getattr(prompt_batch, "uids", ()) or ())
            return tuple(
                uids[index]
                for index, item in enumerate(current)
                if index < len(uids)
                and isinstance(item, (tuple, list))
                and item
                and self._at_generation_boundary(item[0])
            )

        def _omlx_drain_cancel_boundary(self, epoch: int, uids: Any) -> None:
            """Finish all scheduled tensor work before a cancel rendezvous."""

            if self._omlx_prepared_cancel_vote is not None:
                raise RuntimeError("a distributed cancel vote is already armed")
            # ``BatchGenerator.next`` normally materializes cache state, but a
            # discarded final-layer/indexer graph can otherwise outlive the
            # outer prompt call. Removing/filtering its cache while that graph
            # is live can let the peer issue the next differently shaped
            # collective first. Synchronizing the generation stream is paid
            # only for a real cancellation, never on the decode hot path.
            stream = getattr(self, "stream", None)
            if stream is None:
                mx.synchronize()
            else:
                mx.synchronize(stream)

        def _omlx_arm_cancel_boundary(self, epoch: int, uids: Any) -> None:
            normalized = tuple(int(uid) for uid in uids)
            if self._omlx_prepared_cancel_vote is not None:
                raise RuntimeError("a distributed cancel vote is already armed")
            self._omlx_prepared_cancel_vote = (int(epoch), normalized)

        def insert_segments(self, *args: Any, **kwargs: Any) -> Any:
            uids = super().insert_segments(*args, **kwargs)
            telemetry.bind_pending_uid(uids)
            segments = kwargs.get("segments")
            all_tokens = kwargs.get("all_tokens")
            if isinstance(segments, list):
                for index, uid in enumerate(uids):
                    prefix = (
                        list(all_tokens[index])
                        if isinstance(all_tokens, list)
                        and index < len(all_tokens)
                        and all_tokens[index]
                        else []
                    )
                    body = [
                        token for segment in segments[index] for token in segment
                    ]
                    self._omlx_tokens[uid] = prefix + body
            return uids

        def _make_batch(self, n: int) -> Any:
            batch = super()._make_batch(n)
            totals = {
                uid: len(self._omlx_tokens[uid])
                for uid in batch.uids
                if uid in self._omlx_tokens
            }
            if totals:
                existing = dict(
                    getattr(
                        self._prompt_batch,
                        "_omlx_total_prompt_lengths",
                        {},
                    )
                )
                existing.update(totals)
                self._prompt_batch._omlx_total_prompt_lengths = existing
            return batch

        def remove(self, uids: Any) -> Any:
            uid_values = list(uids)
            if world_size > 1:
                prepared = self._omlx_prepared_cancel_vote
                if prepared is None or prepared[1] != tuple(uid_values):
                    raise RuntimeError(
                        "distributed UID removal was not armed by the shared "
                        "cancel epoch"
                    )
                self._omlx_prepared_cancel_vote = None
            # Mutate the MLX-LM batch first. Only then publish cancellation and
            # wake the HTTP collector; an exception during cache filtering must
            # not advertise a request as safely removed when its graph remains.
            result = super().remove(uid_values)
            for uid in uid_values:
                self._omlx_tokens.pop(uid, None)
            telemetry.cancel_uids(uid_values)
            return result

        def _omlx_snapshot_boundary(self, response: Any) -> None:
            """Save the batched prompt cache when prefill crosses a boundary.

            The batched prefill advances by ``prefill_step_size``, so a report at
            an aligned position is exactly a reusable prefix. The cache is pulled
            out per uid (a copy) and keyed by the tokens it now holds.
            """

            uid = getattr(response, "uid", None)
            full = self._omlx_tokens.get(uid)
            progress = getattr(response, "progress", None)
            if (
                full is None
                or not isinstance(progress, (tuple, list))
                or len(progress) != 2
            ):
                return
            processed, total = int(progress[0]), int(progress[1])
            absolute = len(full) - total + processed
            if (
                ssd_store is not None
                and absolute > 0
                and absolute % snapshot_step == 0
            ):
                model = getattr(snapshot_ctx, "model", None)
                if model is not None:
                    try:
                        extracted = self.extract_cache([uid]).get(uid)
                    except Exception:
                        extracted = None
                    # Every rank must enter capacity agreement for every
                    # aligned boundary. A rank-local extraction failure is
                    # represented as an invalid payload so all peers reject
                    # the checkpoint together instead of one blocking in the
                    # tiny size exchange.
                    cache_payload = extracted[0] if extracted is not None else None
                    ssd_store.put(model, full[:absolute], cache_payload)
            if getattr(response, "end_of_prompt", False):
                self._omlx_tokens.pop(uid, None)

        def next(self) -> Any:
            at_boundary = self._about_to_enter_generation()
            captured_boundaries: tuple[tuple[Any, int], ...] = ()
            if at_boundary:
                prompt_cache = getattr(snapshot_ctx, "prompt_cache", None)
                model_key = getattr(snapshot_ctx, "model_key", None)
                captured_boundaries = tuple(
                    (uid, captured)
                    for uid in self._generation_boundary_uids()
                    if (
                        captured := _capture_prompt_boundary_cache(
                            prompt_cache,
                            model_key,
                            self,
                            uid,
                        )
                    )
                )
                if (
                    captured_boundaries
                    and os.environ.get("OMLX_CLUSTER_CACHE_TRACE", "0") == "1"
                ):
                    logger.warning(
                        "Captured predecode prompt cache rank=%s boundaries=%s",
                        rank,
                        captured_boundaries,
                    )
            if (
                os.environ.get("OMLX_CLUSTER_TRACE_COLLECTIVES", "0")
                .strip()
                .lower()
                in {"1", "true", "on", "yes"}
                and self._omlx_batch_trace_steps < 10
            ):
                self._omlx_batch_trace_steps += 1
                current = getattr(self, "_currently_processing", ())
                pending = getattr(self, "_unprocessed_sequences", ())
                print(
                    "OMLX_BATCH_TRACE:"
                    + json.dumps(
                        {
                            "rank": int(mx.distributed.init().rank()),
                            "step": self._omlx_batch_trace_steps,
                            "at_generation_boundary": at_boundary,
                            "prompt_uids": list(
                                getattr(
                                    getattr(self, "_prompt_batch", None),
                                    "uids",
                                    (),
                                )
                            ),
                            "generation_uids": list(
                                getattr(
                                    getattr(self, "_generation_batch", None),
                                    "uids",
                                    (),
                                )
                            ),
                            "current_segments": [
                                [len(segment) for segment in item[0]]
                                for item in current
                                if isinstance(item, (tuple, list)) and item
                            ],
                            "pending_segments": [
                                [len(segment) for segment in item[1]]
                                for item in pending
                                if isinstance(item, (tuple, list)) and len(item) > 1
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            started = time.perf_counter()
            prompt_responses, generation_responses = super().next()
            elapsed = time.perf_counter() - started
            for response in generation_responses:
                response.token = _python_token_id(response.token)
            for response in prompt_responses:
                progress = getattr(response, "progress", None)
                uid = getattr(response, "uid", None)
                if (
                    uid is None
                    or not isinstance(progress, (tuple, list))
                    or len(progress) != 2
                ):
                    continue
                telemetry.observe_prefill_progress(
                    uid,
                    processed_tokens=progress[0],
                    total_tokens=progress[1],
                )
                # This routine also retires the per-UID token bookkeeping at
                # end_of_prompt; it is not only an SSD-cache operation.
                self._omlx_snapshot_boundary(response)
            telemetry.observe_batch_step(
                prompt_responses=len(prompt_responses),
                generation_responses=len(generation_responses),
                elapsed_seconds=elapsed,
            )
            return prompt_responses, generation_responses

    class TelemetryPromptCache(original_prompt_cache):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            prompt_cache_instances.append(self)

        def _omlx_entry_trace(self, model: Any) -> dict[str, tuple[tuple[int, bool], ...]]:
            """Bounded LRU shape trace for operator cache-reuse diagnosis."""

            lru = getattr(self, "_lru", None)
            rows = getattr(lru, "_lrus", {})
            if not isinstance(rows, dict):
                return {}
            result: dict[str, tuple[tuple[int, bool], ...]] = {}
            for cache_type, entries in rows.items():
                result[str(cache_type)] = tuple(
                    (len(tokens), stored_model == model)
                    for stored_model, tokens in entries
                    if isinstance(tokens, list)
                )[:8]
            return result

        def _omlx_cache_inventory(
            self,
        ) -> tuple[int, int, int, int, int, int]:
            """Combined totals plus observable memory and SSD tier splits."""

            memory_entries = len(self)
            memory_bytes = self.nbytes
            disk_entries = len(ssd_store) if ssd_store is not None else 0
            disk_bytes = ssd_store.nbytes if ssd_store is not None else 0
            return (
                memory_entries + disk_entries,
                memory_bytes + disk_bytes,
                memory_entries,
                memory_bytes,
                disk_entries,
                disk_bytes,
            )

        def _fetch_observed(self, model: Any, tokens: list[int]) -> Any:
            return super().fetch_nearest_cache(model, tokens)

        def _lookup(self, model: Any, tokens: list[int]) -> Any:
            cache, rest = self._fetch_observed(model, tokens)
            if os.environ.get("OMLX_CLUSTER_CACHE_TRACE", "0") == "1":
                logger.warning(
                    "Prompt-cache lookup rank=%s tokens=%s found=%s rest=%s lru=%s",
                    rank,
                    len(tokens),
                    cache is not None,
                    len(rest),
                    self._omlx_entry_trace(model),
                )
            hit_tier = "memory" if cache is not None else None
            if cache is not None and not rest and tokens:
                # MLX-LM's exact-hit branch returns an empty rest, unlike its
                # shorter/longer branches which cap the prefix at len - 1. The
                # pinned batched server dies inserting a fully consumed
                # request (insert_segments indexes seq[-1]) and the sequential
                # generate_step rejects an empty prompt, so hand back the last
                # token: trimmed off the hit when the cache supports it,
                # recomputed from scratch when it does not.
                from mlx_lm.models.cache import (
                    can_trim_prompt_cache,
                    trim_prompt_cache,
                )

                if can_trim_prompt_cache(cache):
                    trim_prompt_cache(cache, 1)
                    rest = list(tokens[-1:])
                else:
                    cache, rest = None, list(tokens)
                    hit_tier = None
            # Record the full prompt so the boundary-snapshot callback can key
            # its writes; it runs later on this same generation thread.
            snapshot_ctx.model = model
            snapshot_ctx.prompt = list(tokens)
            if ssd_store is not None:
                # The collective is taken on every request, hit or miss, so all
                # ranks reach it the same number of times regardless of their
                # in-memory state; the agreed boundary is only used when the
                # in-memory tier missed. It also protects restore if a transient
                # write failure leaves one rank without an otherwise symmetric
                # byte-budget entry.
                boundary = agree_ssd_boundary(model, tokens)
                if cache is None and boundary > 0:
                    loaded = ssd_store.load(model, tokens, boundary)
                    if loaded is not None:
                        cache, rest = loaded, list(tokens[boundary:])
                        hit_tier = "ssd"
            cache, rest = agree_prompt_cache_plan(cache, tokens, rest)
            if cache is None:
                hit_tier = None
            (
                entries,
                nbytes,
                memory_entries,
                memory_bytes,
                ssd_entries,
                ssd_bytes,
            ) = self._omlx_cache_inventory()
            # Observe the final agreed tier. Recording the volatile lookup
            # before SSD restore made a real durable hit appear as a miss on
            # the cluster dashboard even while per-request cached_tokens was
            # correct.
            telemetry.observe_cache_lookup(
                prompt_tokens=len(tokens),
                remaining_tokens=len(rest),
                entries=entries,
                nbytes=nbytes,
                memory_entries=memory_entries,
                memory_bytes=memory_bytes,
                ssd_entries=ssd_entries,
                ssd_bytes=ssd_bytes,
                hit_tier=hit_tier,
                **ssd_store_observability(),
            )
            return cache, rest

        def prefetch_nearest_cache(self, model: Any, tokens: list[int]) -> Any:
            """Look up once during caught preflight, then hand it to MLX-LM."""

            result = self._lookup(model, tokens)
            self._omlx_prefetched: Any = ((model, tuple(tokens)), result)
            return result

        def discard_prefetched_cache(self) -> None:
            self._omlx_prefetched = None

        def fetch_nearest_cache(self, model: Any, tokens: list[int]) -> Any:
            # This is the entry MLX-LM itself always calls, on both the batched
            # and the sequential path. The preflight lookup only happens when a
            # prefill guard is installed, so the SSD tier and the snapshot
            # context must live here too or a guardless deployment would
            # silently lose the whole feature.
            key = (model, tuple(tokens))
            prefetched = getattr(self, "_omlx_prefetched", None)
            self._omlx_prefetched = None
            if prefetched is not None and prefetched[0] == key:
                return prefetched[1]
            return self._lookup(model, tokens)

        def insert_cache(self, *args: Any, **kwargs: Any) -> Any:
            result = super().insert_cache(*args, **kwargs)
            (
                entries,
                nbytes,
                memory_entries,
                memory_bytes,
                ssd_entries,
                ssd_bytes,
            ) = self._omlx_cache_inventory()
            telemetry.observe_cache_state(
                entries=entries,
                nbytes=nbytes,
                memory_entries=memory_entries,
                memory_bytes=memory_bytes,
                ssd_entries=ssd_entries,
                ssd_bytes=ssd_bytes,
                **ssd_store_observability(),
            )
            return result

    class CoordinatedGenerationContext(original_generation_context):
        """Make sequential-request cancellation a rank-agreed decision.

        MLX-LM already shares batch cancellations through ``_share_object``.
        Its sequential distributed path instead raises ``NotImplementedError``
        on rank zero while peers keep decoding. The generation thread marks
        this context as sequential, so every rank contributes its local stop
        bit at the same token boundary and all ranks leave together.
        """

        @property
        def _should_stop(self) -> bool:
            local_stop = bool(self.__dict__.get("_omlx_local_should_stop", False))
            if not getattr(cancellation_state, "sequential", False):
                return local_stop
            agreed = mx.distributed.all_sum(mx.array(int(local_stop))).item()
            return bool(agreed)

        @_should_stop.setter
        def _should_stop(self, value: Any) -> None:
            self.__dict__["_omlx_local_should_stop"] = bool(value)

    class PrefillCancellationTimeBudget(original_time_budget):
        """Expose one synchronized cancel/admission boundary per prompt chunk.

        MLX-LM's distributed TimeBudget normally batches 25 ``next()`` calls
        before sharing ``uids_to_remove``.  With 2,048-token prompt chunks that
        deferred a disconnect by 51,200 tokens.  Every rank observes the same
        prompt lifecycle, so limiting only active-prefill slices to one step is
        symmetric and lets the existing rank-zero cancellation vote run after
        at most the in-progress chunk. Decode keeps the amortized 25-step slice.
        """

        def __next__(self) -> Any:
            if (
                telemetry.has_active_prefill()
                and getattr(self, "_current_iterations", 0) >= 1
            ):
                raise StopIteration()
            return super().__next__()

    class TelemetryResponseGenerator(original):
        @staticmethod
        def _finish_shared_cancel_vote(shared: Any) -> Any:
            if not (
                isinstance(shared, dict)
                and shared.get("kind") == _CANCEL_VOTE_KIND
            ):
                return shared
            epoch, uids = telemetry.accept_cancel_vote(shared)
            telemetry.drain_cancel_boundary(epoch, uids)
            if control_plane is not None:
                # TCP sendall is not a rendezvous: rank zero can otherwise
                # filter its caches while the worker is still draining an
                # indexer gather from the just-finished prompt chunk.
                control_plane.barrier()
            # ``remove`` is fail-closed in TP mode and accepts only this exact
            # epoch/list after every peer completed the rendezvous.
            telemetry.arm_cancel_boundary(epoch, uids)
            return uids

        def _share_object(self, obj: Any) -> Any:
            if not bool(getattr(self, "_is_distributed", False)):
                return super()._share_object(obj)
            # MLX-LM calls this with the rank-zero-owned ``uids_to_remove``
            # list once per batch-loop slice. Merge heartbeat/transport aborts
            # here, on the generation thread, so cancellation cannot depend on
            # cross-thread mutation of GenerationContext and cannot remove a
            # UID on only one rank.
            if int(getattr(self, "_rank", 0)) == 0 and isinstance(obj, list):
                obj = telemetry.merge_requested_cancel_uids(obj)
                if obj:
                    obj = telemetry.make_cancel_vote(obj)
            if control_plane is not None:
                shared = control_plane.broadcast_object(obj)
                return self._finish_shared_cancel_vote(shared)

            # MLX-LM implements every rank-zero object broadcast as two
            # all-sums: the producer contributes pickle length/data and every
            # worker contributes zeros. Tiny JACCL reductions are the wrong
            # primitive for an owned byte stream and can corrupt either the
            # request or later cancellation lists. Direct send/recv is also
            # unsuitable here: MLX-LM polls and retires batches on slightly
            # different host timelines, and a lost JACCL send completion then
            # leaves one rank in teardown. Use identically ordered all-gathers
            # with a fixed, checksummed control envelope instead.
            import pickle
            import struct
            import zlib

            group = mx.distributed.init()
            rank = int(getattr(self, "_rank", group.rank()))
            if rank == 0:
                payload = pickle.dumps(obj) if obj is not None else b""
                if len(payload) > _MAX_SHARED_REQUEST_BYTES:
                    raise RuntimeError(
                        "distributed object exceeds the 256 MiB safety bound"
                    )
                checksum = zlib.crc32(payload)
                header_bytes = struct.pack(
                    ">III", _SHARED_OBJECT_MAGIC, len(payload), checksum
                ).ljust(_SHARED_OBJECT_HEADER_BYTES, b"\0")
                local_header = mx.array(header_bytes, dtype=mx.uint8)
            else:
                payload = b""
                local_header = mx.zeros(
                    (_SHARED_OBJECT_HEADER_BYTES,), dtype=mx.uint8
                )

            gathered_header = mx.distributed.all_gather(local_header, group=group)
            rank_zero_header = bytes(
                gathered_header[:_SHARED_OBJECT_HEADER_BYTES].tolist()
            )
            try:
                magic, size, checksum = struct.unpack(">III", rank_zero_header[:12])
            except struct.error as exc:
                raise RuntimeError(
                    "distributed object broadcast has a malformed header"
                ) from exc
            if magic != _SHARED_OBJECT_MAGIC:
                raise RuntimeError(
                    "distributed object broadcast failed its magic-header check"
                )
            if size > _MAX_SHARED_REQUEST_BYTES:
                raise RuntimeError(
                    "distributed object broadcast has an invalid byte length: "
                    f"{size}"
                )
            if size == 0:
                return self._finish_shared_cancel_vote(None)

            local_data = (
                mx.array(payload, dtype=mx.uint8)
                if rank == 0
                else mx.zeros((size,), dtype=mx.uint8)
            )
            gathered_data = mx.distributed.all_gather(local_data, group=group)
            rank_zero_data = bytes(gathered_data[:size].tolist())
            if zlib.crc32(rank_zero_data) != checksum:
                raise RuntimeError(
                    "distributed object broadcast failed its CRC32 integrity check"
                )
            if rank == 0:
                shared = obj
            else:
                try:
                    shared = pickle.loads(rank_zero_data)
                except Exception as exc:
                    raise RuntimeError(
                        "distributed object broadcast failed integrity decoding"
                    ) from exc
            return self._finish_shared_cancel_vote(shared)

        def _share_request(self, request: Any) -> Any:
            shared = super()._share_request(request)
            if shared is None:
                return None
            response_queue, *rest = shared
            transport_request_id = (
                _transport_request_id(
                    getattr(rest[0], "_omlx_transport_request_id", None)
                )
                if rest
                else None
            )
            return (
                _TelemetryQueue(
                    response_queue,
                    telemetry,
                    transport_request_id=transport_request_id,
                ),
                *rest,
            )

        def _tokenize(self, tokenizer: Any, request: Any, args: Any) -> Any:
            tokenized = super()._tokenize(tokenizer, request, args)
            snapshot_ctx.prompt_cache = self.prompt_cache
            snapshot_ctx.model_key = self.model_provider.model_key
            if prefill_guard is None:
                return tokenized

            prompt = tokenized[0]
            _cache, rest = self.prompt_cache.prefetch_nearest_cache(
                self.model_provider.model_key,
                prompt,
            )
            try:
                # MLX-LM catches tokenization failures for batched requests.
                # Keeping the rank vote inside that boundary rejects just this
                # request; raising from its later cache lookup kills the whole
                # generation thread.
                prefill_guard.check_collective(
                    len(prompt),
                    cached_tokens=len(prompt) - len(rest),
                    mx_module=mx,
                )
            except Exception:
                self.prompt_cache.discard_prefetched_cache()
                raise
            return tokenized

        def _serve_single(self, request: Any) -> Any:
            # The coordinated context above makes every rank observe the same
            # cancellation. Suppress only MLX-LM's obsolete distributed guard,
            # whose entire behavior is to raise NotImplementedError after that
            # agreement; model collectives remain distributed.
            was_distributed = self._is_distributed
            previous = getattr(cancellation_state, "sequential", False)
            cancellation_state.sequential = was_distributed
            self._is_distributed = False
            try:
                return super()._serve_single(request)
            finally:
                self._is_distributed = was_distributed
                cancellation_state.sequential = previous

    original_stream_generate = mlx_server.stream_generate

    def request_correlated_handle_completion(
        handler: Any,
        request: Any,
        stop_words: Any,
    ) -> Any:
        """Attach the private coordinator id before ranks share the request."""

        request_id = _transport_request_id(
            handler.headers.get("X-oMLX-Request-ID")
        )
        if request_id is not None:
            request._omlx_transport_request_id = request_id
        return original_handle_completion(handler, request, stop_words)

    def clear_rank_caches(*, clear_ssd: bool, clear_hot: bool) -> dict[str, Any]:
        active = telemetry.active_request_count()
        if active:
            raise RuntimeError(f"requests are active ({active})")
        deleted = 0
        cleared = 0
        if clear_ssd and ssd_store is not None:
            deleted = ssd_store.clear(timeout=30.0)
        if clear_hot:
            for cache in tuple(prompt_cache_instances):
                before = len(cache)
                cache.trim_to(n_sequences=0, n_bytes=0)
                cleared += before
        memory_entries = sum(len(cache) for cache in prompt_cache_instances)
        memory_bytes = sum(cache.nbytes for cache in prompt_cache_instances)
        disk_entries = len(ssd_store) if ssd_store is not None else 0
        disk_bytes = ssd_store.nbytes if ssd_store is not None else 0
        telemetry.observe_cache_state(
            entries=memory_entries + disk_entries,
            nbytes=memory_bytes + disk_bytes,
            memory_entries=memory_entries,
            memory_bytes=memory_bytes,
            ssd_entries=disk_entries,
            ssd_bytes=disk_bytes,
            **ssd_store_observability(),
        )
        return {
            "status": "ok",
            "rank": rank,
            "ssd_deleted": deleted,
            "hot_cleared": cleared,
        }

    def write_maintenance_ack(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"), sort_keys=True)
        os.replace(temporary, path)

    maintenance_epoch_floor = time.time_ns()
    maintenance_last_epoch = maintenance_epoch_floor
    maintenance_request_path = (
        Path(marker_path).parent / f"{marker_deployment_id}-cache-clear.json"
        if marker_path is not None and marker_deployment_id
        else None
    )
    maintenance_ack_path = (
        Path(marker_path).parent
        / f"{marker_deployment_id}-cache-clear-rank-{rank}.json"
        if marker_path is not None and marker_deployment_id
        else None
    )

    def poll_cache_maintenance() -> None:
        nonlocal maintenance_last_epoch
        path = maintenance_request_path
        ack_path = maintenance_ack_path
        if path is None or ack_path is None or not path.is_file():
            return
        try:
            if path.stat().st_size > _MAX_CANCEL_FILE_BYTES:
                return
            request = json.loads(path.read_text(encoding="utf-8"))
            epoch = int(request.get("epoch", 0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return
        if (
            epoch <= maintenance_last_epoch
            or epoch < maintenance_epoch_floor
            or request.get("deployment_id") != marker_deployment_id
            or request.get("plan_hash") != marker_plan_hash
        ):
            return
        maintenance_last_epoch = epoch
        try:
            result = clear_rank_caches(
                clear_ssd=bool(request.get("ssd")),
                clear_hot=bool(request.get("hot")),
            )
            ack = result | {"epoch": epoch}
        except Exception as exc:
            ack = {
                "status": "error",
                "rank": rank,
                "epoch": epoch,
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
        write_maintenance_ack(ack_path, ack)
        with suppress(OSError):
            path.unlink()

    telemetry.set_maintenance_callback(poll_cache_maintenance)

    def maintenance_do_post(handler: Any) -> Any:
        """Private rank-local cache maintenance endpoint.

        The public coordinator invokes this on every rank. It is bound to
        loopback by the worker, authenticated with the signed plan hash, and
        refuses to race a live request. Keeping the store clear in-process is
        essential: deleting only coordinator files leaves peer manifests and
        write-behind queues able to resurrect stale snapshots.
        """

        modes = {
            "/omlx/internal/cache/ssd/clear": (True, False),
            "/omlx/internal/cache/hot/clear": (False, True),
            "/omlx/internal/cache/all/clear": (True, True),
        }
        selected = modes.get(handler.path)
        if selected is None:
            return original_do_post(handler)
        if handler.headers.get("X-oMLX-Plan-Hash") != marker_plan_hash:
            handler._set_completion_headers(403)
            handler.end_headers()
            handler.wfile.write(b'{"error":"invalid maintenance token"}')
            return None
        clear_ssd, clear_hot = selected
        try:
            result = clear_rank_caches(
                clear_ssd=clear_ssd,
                clear_hot=clear_hot,
            )
        except (RuntimeError, TimeoutError) as exc:
            status = 409 if "requests are active" in str(exc) else 503
            handler._set_completion_headers(status)
            handler.end_headers()
            handler.wfile.write(json.dumps({"error": str(exc)}).encode())
            return None
        body = json.dumps(result, separators=(",", ":")).encode()
        handler._set_completion_headers(200)
        handler.end_headers()
        handler.wfile.write(body)
        return None

    def snapshotting_stream_generate(*args: Any, **kwargs: Any) -> Any:
        """Deposit a snapshot at each prefill boundary before it is overwritten.

        A ``RotatingKVCache`` window and a recurrent gated-delta-net state are
        gone once prefill moves past a boundary, so the whole cache is saved the
        moment the boundary is reached rather than at the end. The callback is
        chained onto whatever MLX-LM passed, so progress reporting is unchanged.
        """

        prompt_cache = kwargs.get("prompt_cache")
        rest = kwargs.get("prompt")
        model = getattr(snapshot_ctx, "model", None)
        full = getattr(snapshot_ctx, "prompt", None)
        if (
            ssd_store is None
            or prompt_cache is None
            or not isinstance(rest, list)
            or model is None
            or full is None
        ):
            yield from original_stream_generate(*args, **kwargs)
            return

        base = len(full) - len(rest)
        user_callback = kwargs.get("prompt_progress_callback")

        def _snapshot(processed: int, total: int) -> None:
            absolute = base + int(processed)
            # Only aligned boundaries are reusable: an unaligned base leaves the
            # next request's prefill off the grid, so those writes would never be
            # found and are skipped.
            if absolute > 0 and absolute % snapshot_step == 0:
                ssd_store.put(model, full[:absolute], prompt_cache)
            if user_callback is not None:
                user_callback(processed, total)

        kwargs["prompt_progress_callback"] = _snapshot
        yield from original_stream_generate(*args, **kwargs)

    mlx_server.BatchGenerator = TelemetryBatchGenerator
    mlx_server.ResponseGenerator = TelemetryResponseGenerator
    mlx_server.LRUPromptCache = TelemetryPromptCache
    mlx_server.GenerationContext = CoordinatedGenerationContext
    mlx_server.TimeBudget = PrefillCancellationTimeBudget
    mlx_server.APIHandler.handle_completion = request_correlated_handle_completion
    mlx_server.APIHandler.do_POST = maintenance_do_post
    if ssd_store is not None:
        mlx_server.stream_generate = snapshotting_stream_generate
        # A persistent manifest can already contain useful snapshots before
        # the first lookup. Publish that inventory as soon as serving begins
        # so "enabled but idle" is distinguishable from "not wired".
        telemetry.observe_cache_state(
            entries=len(ssd_store),
            nbytes=ssd_store.nbytes,
            memory_entries=0,
            memory_bytes=0,
            ssd_entries=len(ssd_store),
            ssd_bytes=ssd_store.nbytes,
            **ssd_store_observability(),
        )
    # Started here rather than by the caller: this block is exactly the span
    # during which a rank is alive and expected to look alive, and a heartbeat
    # a caller can forget to start is a heartbeat that will be forgotten.
    telemetry.start_heartbeat()
    try:
        yield telemetry
    finally:
        telemetry.stop_heartbeat()
        mlx_server.ResponseGenerator = original
        mlx_server.BatchGenerator = original_batch_generator
        mlx_server.LRUPromptCache = original_prompt_cache
        mlx_server.GenerationContext = original_generation_context
        mlx_server.TimeBudget = original_time_budget
        mlx_server.APIHandler.handle_completion = original_handle_completion
        mlx_server.APIHandler.do_POST = original_do_post
        mlx_server.stream_generate = original_stream_generate
        snapshot_writer_closed = True
        if ssd_store is not None:
            snapshot_writer_closed = ssd_store.close(timeout=10.0)
            if not snapshot_writer_closed:
                logger.warning(
                    "Prompt snapshot writer did not drain within 10 seconds; "
                    "leaving its directory intact for failure-safe teardown"
                )
        if (
            ssd_store is not None
            and snapshot_writer_closed
            and not ssd_cache_persistent
        ):
            # Legacy process-lifetime mode: a hard crash skips this and the
            # next nonpersistent store reclaims the leftovers instead.
            shutil.rmtree(ssd_store.directory, ignore_errors=True)
