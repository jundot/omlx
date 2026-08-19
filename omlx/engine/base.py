# SPDX-License-Identifier: Apache-2.0
"""
Base engine interface for oMLX inference.
"""

import asyncio
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import mlx.core as mx

from omlx.engine_core import get_mlx_executor

_preflight_logger = logging.getLogger("omlx.engine.preflight")

_PREFLIGHT_CLEANUP_WAIT_TIMEOUT_S = 4.0
_PREFLIGHT_CLEANUP_POLL_INTERVAL_S = 0.05

# Per-process record of (engine_class_name, method) pairs that have
# already logged a "scheduler unreachable" warning. The warning marks a
# wrapper-chain misconfiguration — a deployment bug rather than a
# runtime condition — so once-per-pair is enough to alert oncall
# without flooding the journal at request rate.
_PREFLIGHT_UNREACHABLE_WARNED: set[tuple[str, str]] = set()


def _clear_teardown_references(
    engine: object,
    *,
    none_attrs: tuple[str, ...],
    false_attrs: tuple[str, ...] = (),
) -> None:
    """Clear wrapper-side references in a consistent stop() teardown pass."""
    for attr in none_attrs:
        setattr(engine, attr, None)
    for attr in false_attrs:
        setattr(engine, attr, False)


def _warn_scheduler_unreachable_once(
    engine: object, method: str, detail: str = ""
) -> None:
    """Emit a one-shot WARNING when the wrapper chain doesn't expose a
    scheduler. Subsequent calls with the same (engine type, method) pair
    are silent so a misconfigured engine doesn't spam logs at request
    rate.
    """
    key = (type(engine).__name__, method)
    if key in _PREFLIGHT_UNREACHABLE_WARNED:
        return
    _PREFLIGHT_UNREACHABLE_WARNED.add(key)
    suffix = f" — {detail}" if detail else ""
    _preflight_logger.warning(
        "%s.%s: scheduler unreachable via _engine.engine.scheduler"
        "%s; preflight check skipped (further occurrences suppressed)",
        type(engine).__name__,
        method,
        suffix,
    )


async def _run_scheduler_preflight_with_cleanup_retry(
    scheduler: Any,
    *,
    num_prompt_tokens: int,
    request_id: str | None,
    eviction_callback: Any | None,
    executor: Any | None = None,
) -> None:
    """Run route preflight after transient post-request cleanup settles.

    A finished request can remain resident while its asynchronous cache store
    owns the extracted KV and while the scheduler's deferred Metal clear is
    pending. Charging a new request against that temporary footprint produces
    a false rejection. Only defer a preflight that would otherwise need
    eviction or rejection; requests that already fit keep the fast path.

    Cleanup remains owned by ``Scheduler.step()``. This helper never drains
    request state, synchronizes a stream, or clears the Metal pool, so active
    batched work is not interrupted.
    """
    deadline = time.monotonic() + _PREFLIGHT_CLEANUP_WAIT_TIMEOUT_S
    waited_for_cleanup = False

    while True:
        eviction_request = scheduler.preflight_eviction_request(
            num_prompt_tokens=num_prompt_tokens,
            request_id=request_id,
        )
        if eviction_request is None:
            scheduler.preflight_or_raise(
                num_prompt_tokens=num_prompt_tokens,
                request_id=request_id,
            )
            return

        cleanup_pending_fn = getattr(
            scheduler, "has_pending_route_preflight_cleanup", None
        )
        cleanup_pending = (
            cleanup_pending_fn() is True if callable(cleanup_pending_fn) else False
        )
        now = time.monotonic()
        if cleanup_pending and now < deadline:
            if not waited_for_cleanup:
                waited_for_cleanup = True
                _preflight_logger.debug(
                    "Deferring preflight rejection for request %s until "
                    "post-request cache cleanup settles",
                    request_id or "preflight",
                )
            await asyncio.sleep(_PREFLIGHT_CLEANUP_POLL_INTERVAL_S)
            continue

        # Dropping the last Request/KV references and clearing MLX's pool do
        # not make macOS phys_footprint settle atomically. Once a transient
        # rejection has observed pending cleanup, keep re-measuring for the
        # same bounded window even after the scheduler bookkeeping is clear.
        # This also avoids evicting an idle model for bytes that IOKit is
        # already returning asynchronously.
        if waited_for_cleanup and now < deadline:
            refresh_usage = getattr(
                scheduler, "refresh_route_preflight_usage", None
            )
            if executor is not None and callable(refresh_usage):
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(executor, refresh_usage)
            await asyncio.sleep(_PREFLIGHT_CLEANUP_POLL_INTERVAL_S)
            continue

        if cleanup_pending and waited_for_cleanup:
            _preflight_logger.warning(
                "Post-request cache cleanup did not settle within %.1fs before "
                "preflight for request %s; using the current memory snapshot",
                _PREFLIGHT_CLEANUP_WAIT_TIMEOUT_S,
                request_id or "preflight",
            )

        if eviction_callback is not None:
            _preflight_logger.info(
                "Running preflight LRU eviction for request %s",
                eviction_request.request_id,
            )
            await eviction_callback(eviction_request)
        scheduler.preflight_or_raise(
            num_prompt_tokens=num_prompt_tokens,
            request_id=request_id,
        )
        return


@dataclass
class GenerationOutput:
    """
    Output from generation.

    Compatible with both simple and batched engines.
    """

    text: str
    tokens: List[int] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = "stop"
    # For streaming
    new_text: str = ""
    finished: bool = True
    # For tool calling (Harmony and other models)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # Prefix cache stats
    cached_tokens: int = 0
    # Optional engine-native throughput stats. Diffusion models report
    # generation after prefill separately from end-to-end request time.
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    diffusion_canvas_tokens: int = 0
    diffusion_denoising_steps: int = 0
    diffusion_work_tokens: int = 0
    diffusion_canvas_tps: float = 0.0
    diffusion_work_tps: float = 0.0
    generated_at: Optional[float] = None
    generated_until: Optional[float] = None
    first_token_at: Optional[float] = None
    # Internal scheduler trace populated only by local benchmark requests.
    benchmark_prefill_chunks: List[int] = field(default_factory=list)
    benchmark_requested_steps: List[int] = field(default_factory=list)
    benchmark_boundary_enabled: bool = False
    benchmark_cache_block_size: int = 0


class BaseEngine(ABC):
    """
    Abstract base class for inference engines.

    Both SimpleEngine and BatchedEngine implement this interface,
    allowing the server to use either without code changes.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate a complete response (non-streaming).

        Args:
            prompt: Input text or token IDs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with complete text
        """
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream generation token by token.

        Args:
            prompt: Input text or token IDs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            stop: Stop sequences
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> GenerationOutput:
        """
        Chat completion (non-streaming).

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            tools: Optional tool definitions
            **kwargs: Additional model-specific parameters

        Returns:
            GenerationOutput with assistant response
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: Optional[List[dict]] = None,
        **kwargs,
    ) -> AsyncIterator[GenerationOutput]:
        """
        Stream chat completion token by token.

        Args:
            messages: List of chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling (0 = disabled)
            repetition_penalty: Repetition penalty (1.0 = disabled)
            tools: Optional tool definitions
            **kwargs: Additional model-specific parameters

        Yields:
            GenerationOutput with incremental text
        """
        pass

    @property
    @abstractmethod
    def model_type(self) -> Optional[str]:
        """Get the model type from config.json (e.g., 'gpt_oss', 'llama', 'qwen2').

        This can be used to apply model-specific processing.

        Returns:
            Model type string or None if not available.
        """
        pass

    @property
    def grammar_compiler(self):
        """Return the grammar compiler for this engine, or ``None``.

        Subclasses that support xgrammar should override this with a
        lazy-initializing property.
        """
        return None

    @property
    def prefix_cache_enabled(self) -> bool:
        """Whether automatic prefix caching is active on this engine.

        Subclasses that wire up a BlockAwarePrefixCache should override this.
        """
        return False

    def has_active_requests(self) -> bool:
        """Check if the engine has active in-flight requests.

        Used by EnginePool.check_ttl_expirations() to prevent unloading
        a model while requests are still being processed.

        Returns:
            True if there are active requests, False otherwise.
        """
        return False

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary containing engine statistics.
        """
        pass

    @abstractmethod
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics, or None if not applicable.
        """
        pass

    async def preflight_chat(
        self,
        messages: list,
        tools: Optional[list] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Optional prefill-memory preflight check for chat requests.

        Default no-op; engines that implement the prefill memory guard
        (``BatchedEngine``, ``VLMBatchedEngine``) override this with the
        actual estimation logic. The base no-op lets simpler engines
        (SimpleEngine, embedding/reranker engines, test stubs) be
        invoked from the server endpoints without additional wrapping.
        """
        return None

    async def preflight_completion(
        self,
        prompt: str,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Optional prefill-memory preflight check for completion requests.

        See :meth:`preflight_chat` for the rationale.
        """
        return None


class ActivityTrackingMixin:
    """In-flight operation tracking for admin visibility.

    Engines that don't run requests through a Scheduler (non-streaming
    engines, DFlashEngine) have no scheduler snapshot for the admin
    Active Models card to read, so they track their own operations here
    and expose them via get_activity_snapshot().
    """

    def __init__(self):
        super().__init__()
        self._active_count = 0
        self._active_lock = threading.Lock()
        self._activities: Dict[str, Dict[str, Any]] = {}

    def has_active_requests(self) -> bool:
        """Check if the engine has active in-flight requests."""
        with self._active_lock:
            return self._active_count > 0

    def _reset_activity_tracking(self) -> None:
        """Clear the in-flight activity counter + records on engine teardown.

        #1595: the memory-enforcer's immediate-abort eviction stops the engine WITHOUT
        running the normal per-request completion callbacks (_end_activity), so the
        ``_active_count`` atomic counter can be left non-zero. That phantom 'busy' count
        both corrupts the status API and (via has_active_requests()) can make a stale
        engine look permanently non-evictable. Called from EnginePool._unload_engine().
        """
        with self._active_lock:
            self._active_count = 0
            self._activities.clear()

    _ACTIVITY_RESERVED_KEYS = {
        "request_id",
        "kind",
        "detail",
        "started_at",
        "last_activity_at",
        "total_items",
    }

    def _sanitize_activity_metadata(
        self, metadata: Dict[str, Any] | None
    ) -> Dict[str, Any]:
        """Drop reserved activity keys from caller-provided metadata.

        Timing keys are owned by the tracker: _begin_activity sets them and
        _update_activity always advances last_activity_at to "now".
        """
        if not metadata:
            return {}
        return {
            key: value
            for key, value in metadata.items()
            if key not in self._ACTIVITY_RESERVED_KEYS
        }

    def _begin_activity(
        self,
        kind: str,
        detail: str | None = None,
        total_items: int | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        """Track a non-streaming operation for admin visibility."""
        activity_id = str(uuid.uuid4())
        now = time.monotonic()
        with self._active_lock:
            self._active_count += 1
            activity = {
                "request_id": activity_id,
                "kind": kind,
                "detail": detail or kind,
                "started_at": now,
                "last_activity_at": now,
                "total_items": total_items,
            }
            activity.update(self._sanitize_activity_metadata(metadata))
            self._activities[activity_id] = activity
        return activity_id

    def _update_activity(self, activity_id: str, **updates: Any) -> None:
        """Update tracked non-streaming operation metadata."""
        with self._active_lock:
            activity = self._activities.get(activity_id)
            if activity is None:
                return
            activity.update(self._sanitize_activity_metadata(updates))
            activity["last_activity_at"] = time.monotonic()

    def _end_activity(self, activity_id: str) -> None:
        """End an activity."""
        with self._active_lock:
            removed = self._activities.pop(activity_id, None)
            if removed is None:
                raise RuntimeError(
                    f"Activity {activity_id} ended more than once or was never started"
                )
            self._active_count -= 1
            if self._active_count < 0:
                raise RuntimeError("Active request count became negative")

    async def _finish_activity(self, activity_id: str) -> None:
        """End an activity and clear the Metal buffer pool.

        Always clears per request. Gating the clear on `_active_count == 0`
        caused unbounded Metal pool growth under concurrent workloads (#684),
        because indexing clients keep the active count above zero indefinitely.
        `mx.synchronize()` is required before `mx.clear_cache()` to avoid
        Metal buffer races on M3/M4 (#300, #888, #1106).
        """
        self._end_activity(activity_id)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(),
            lambda: (mx.synchronize(), mx.clear_cache()),
        )

    def get_activity_snapshot(self) -> Dict[str, Any]:
        """Return active non-streaming operations for admin display."""
        now = time.monotonic()
        with self._active_lock:
            activities = []
            for activity in self._activities.values():
                item = dict(activity)
                started_at = item.pop("started_at", None)
                last_activity_at = item.pop("last_activity_at", None)
                item["elapsed_seconds"] = (
                    max(0.0, now - started_at) if started_at is not None else None
                )
                item["last_activity_age_seconds"] = (
                    max(0.0, now - last_activity_at)
                    if last_activity_at is not None
                    else None
                )
                activities.append(item)
            return {
                "active_requests": self._active_count,
                "activities": activities,
            }


class BaseNonStreamingEngine(ActivityTrackingMixin, ABC):
    """Base class for non-streaming engines (embedding, reranker).

    These engines compute outputs in a single forward pass and don't
    support streaming or chat completion interfaces.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the engine (load model if not loaded)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the engine and cleanup resources."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary containing engine statistics.
        """
        pass
