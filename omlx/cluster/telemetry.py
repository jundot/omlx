# SPDX-License-Identifier: Apache-2.0
"""Worker-local telemetry for the pinned MLX-LM distributed server."""

from __future__ import annotations

import logging
import math
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from .performance import ExecutionSettings
from .planner import PipelineAssignment

logger = logging.getLogger(__name__)

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


class MarkerWriter(Protocol):
    """Small RuntimeMarker surface used by the telemetry observer."""

    def update(self, phase: str, **extra: Any) -> None: ...


@dataclass
class _RequestSample:
    request_id: int
    started_at: float
    updated_at: float
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
        self._lock = threading.Lock()
        self._started_at = float(self._clock())
        self._last_step_finished_at = self._started_at
        self._next_request_id = 0
        self._requests: dict[int, _RequestSample] = {}
        self._uid_to_request: dict[Any, int] = {}
        self._request_to_uid: dict[int, Any] = {}
        self._pending_uid = threading.local()
        self._last_completed: dict[str, Any] | None = None
        self._last_publish_at = float("-inf")
        self._requests_completed = 0
        self._requests_failed = 0
        self._requests_cancelled = 0
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0
        self._cached_tokens_total = 0
        self._batch_steps = 0
        self._busy_seconds = 0.0
        self._idle_seconds = 0.0
        self._last_batch: dict[str, Any] | None = None
        self._cache_lookups = 0
        self._cache_hits = 0
        self._cache_tokens_reused = 0
        self._cache_entries = 0
        self._cache_bytes = 0

    def heartbeat(self) -> None:
        """Refresh the marker with nothing new to say.

        "Stale" has to mean *stalled*. Every other publish here is driven by a
        request, so without this an idle rank's marker just stopped ageing and
        was indistinguishable from a rank wedged inside a collective.
        """

        now = float(self._clock())
        with self._lock:
            self._publish_locked(now, force=True)

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

    def begin_request(self) -> int:
        now = float(self._clock())
        with self._lock:
            self._next_request_id += 1
            request_id = self._next_request_id
            self._requests[request_id] = _RequestSample(
                request_id=request_id,
                started_at=now,
                updated_at=now,
            )
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
        with self._lock:
            for uid in uid_values:
                request_id = self._uid_to_request.pop(uid, None)
                if request_id is None:
                    continue
                self._request_to_uid.pop(request_id, None)
                changed = (
                    self._finish_locked(
                        request_id,
                        now=now,
                        status="cancelled",
                    )
                    or changed
                )
            if changed:
                self._publish_locked(now, force=True)

    def observe_batch_step(
        self,
        *,
        prompt_responses: int,
        generation_responses: int,
        elapsed_seconds: float,
    ) -> None:
        """Record one coalesced MLX-LM continuous-batching step."""

        now = float(self._clock())
        elapsed = max(0.0, float(elapsed_seconds))
        with self._lock:
            idle = max(0.0, now - self._last_step_finished_at - elapsed)
            self._idle_seconds += idle
            self._busy_seconds += elapsed
            self._last_step_finished_at = now
            self._batch_steps += 1
            coalesced = max(prompt_responses, generation_responses)
            self._last_batch = {
                "step_seconds": elapsed,
                "prompt_responses": max(0, int(prompt_responses)),
                "generation_responses": max(0, int(generation_responses)),
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
            self._cache_entries = max(0, int(entries))
            self._cache_bytes = max(0, int(nbytes))
            self._publish_locked(now, force=False)

    def observe_cache_state(self, *, entries: int, nbytes: int) -> None:
        now = float(self._clock())
        with self._lock:
            self._cache_entries = max(0, int(entries))
            self._cache_bytes = max(0, int(nbytes))
            self._publish_locked(now, force=False)

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
        if getattr(self._pending_uid, "request_id", None) == request_id:
            self._pending_uid.request_id = None
        uid = self._request_to_uid.pop(request_id, None)
        if uid is not None:
            self._uid_to_request.pop(uid, None)
        sample.updated_at = now
        if status == "failed":
            self._requests_failed += 1
        elif status == "cancelled":
            self._requests_cancelled += 1
        else:
            self._requests_completed += 1
        self._prompt_tokens_total += sample.prompt_tokens
        self._completion_tokens_total += sample.completion_tokens
        self._cached_tokens_total += sample.cached_tokens
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
        prefill_elapsed = max(0.0, now - prefill_started_at)
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
        prefill_active = status == "running" and first is None
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
        active_sample = (
            max(self._requests.values(), key=lambda item: item.updated_at)
            if self._requests
            else None
        )
        current = (
            self._sample_snapshot(active_sample, now=now, status="running")
            if active_sample is not None
            else self._last_completed
        )
        active_completion_tokens = sum(
            sample.completion_tokens for sample in self._requests.values()
        )
        uptime = max(0.0, now - self._started_at)
        total_generated = self._completion_tokens_total + active_completion_tokens
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
            "active_requests": len(self._requests),
            "requests_completed": self._requests_completed,
            "requests_failed": self._requests_failed,
            "requests_cancelled": self._requests_cancelled,
            "prompt_tokens_total": self._prompt_tokens_total,
            "completion_tokens_total": self._completion_tokens_total,
            "cached_tokens_total": self._cached_tokens_total,
            "aggregate_decode_tps": (total_generated / uptime if uptime > 0 else 0.0),
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

    def __init__(self, queue: Any, telemetry: RuntimeTelemetry) -> None:
        self._queue = queue
        self._telemetry = telemetry
        self._request_id = telemetry.begin_request()
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
    ssd_max_entries: int = 64,
    prefill_step_size: int = 2048,
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
    disk write that failed on one rank cannot desync the pipeline. The snapshot
    directory is process-lifetime: the store starts it clean and this context's
    teardown removes it.
    """

    import mlx.core as mx
    import mlx_lm.server as mlx_server

    original = mlx_server.ResponseGenerator
    original_batch_generator = mlx_server.BatchGenerator
    original_prompt_cache = mlx_server.LRUPromptCache
    original_generation_context = mlx_server.GenerationContext
    cancellation_state = threading.local()
    telemetry = RuntimeTelemetry(
        marker,
        execution=execution,
        assignment=assignment,
        heartbeat_interval=heartbeat_interval,
    )

    snapshot_ctx = threading.local()
    snapshot_step = max(1, int(prefill_step_size))
    ssd_store = None
    if ssd_cache_dir:
        from .prompt_snapshot_cache import (
            SSDPromptSnapshotStore,
            agreed_boundary,
            candidate_boundaries,
        )

        ssd_store = SSDPromptSnapshotStore(
            ssd_cache_dir, step=snapshot_step, max_entries=ssd_max_entries
        )
    try:
        world_size = int(mx.distributed.init().size())
    except Exception:
        world_size = 1

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

        def insert_segments(self, *args: Any, **kwargs: Any) -> Any:
            uids = super().insert_segments(*args, **kwargs)
            telemetry.bind_pending_uid(uids)
            if ssd_store is not None:
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

        def remove(self, uids: Any) -> Any:
            telemetry.cancel_uids(uids)
            for uid in uids:
                self._omlx_tokens.pop(uid, None)
            return super().remove(uids)

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
            if absolute > 0 and absolute % snapshot_step == 0:
                model = getattr(snapshot_ctx, "model", None)
                if model is not None:
                    try:
                        extracted = self.extract_cache([uid]).get(uid)
                    except Exception:
                        extracted = None
                    if extracted is not None:
                        ssd_store.put(model, full[:absolute], extracted[0])
            if getattr(response, "end_of_prompt", False):
                self._omlx_tokens.pop(uid, None)

        def next(self) -> Any:
            started = time.perf_counter()
            prompt_responses, generation_responses = super().next()
            elapsed = time.perf_counter() - started
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
                if ssd_store is not None:
                    self._omlx_snapshot_boundary(response)
            telemetry.observe_batch_step(
                prompt_responses=len(prompt_responses),
                generation_responses=len(generation_responses),
                elapsed_seconds=elapsed,
            )
            return prompt_responses, generation_responses

    class TelemetryPromptCache(original_prompt_cache):
        def _fetch_observed(self, model: Any, tokens: list[int]) -> Any:
            cache, rest = super().fetch_nearest_cache(model, tokens)
            telemetry.observe_cache_lookup(
                prompt_tokens=len(tokens),
                remaining_tokens=len(rest),
                entries=len(self),
                nbytes=self.nbytes,
            )
            return cache, rest

        def _lookup(self, model: Any, tokens: list[int]) -> Any:
            cache, rest = self._fetch_observed(model, tokens)
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
            # Record the full prompt so the boundary-snapshot callback can key
            # its writes; it runs later on this same generation thread.
            snapshot_ctx.model = model
            snapshot_ctx.prompt = list(tokens)
            if ssd_store is not None:
                # The collective is taken on every request, hit or miss, so all
                # ranks reach it the same number of times regardless of their
                # in-memory state; the agreed boundary is only used when the
                # in-memory tier missed. This is what lets SSD serve the batched
                # path, whose byte-based eviction can diverge across ranks.
                boundary = agree_ssd_boundary(model, tokens)
                if cache is None and boundary > 0:
                    loaded = ssd_store.load(model, tokens, boundary)
                    if loaded is not None:
                        cache, rest = loaded, list(tokens[boundary:])
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
            telemetry.observe_cache_state(entries=len(self), nbytes=self.nbytes)
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

    class TelemetryResponseGenerator(original):
        def _share_request(self, request: Any) -> Any:
            shared = super()._share_request(request)
            if shared is None:
                return None
            response_queue, *rest = shared
            return _TelemetryQueue(response_queue, telemetry), *rest

        def _tokenize(self, tokenizer: Any, request: Any, args: Any) -> Any:
            tokenized = super()._tokenize(tokenizer, request, args)
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
    if ssd_store is not None:
        mlx_server.stream_generate = snapshotting_stream_generate
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
        mlx_server.stream_generate = original_stream_generate
        if ssd_store is not None:
            # Snapshots are process-lifetime; a hard crash skips this and the
            # next store on the same directory reclaims the leftovers instead.
            shutil.rmtree(ssd_store.directory, ignore_errors=True)
