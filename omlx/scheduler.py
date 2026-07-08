# SPDX-License-Identifier: Apache-2.0
"""
Scheduler for oMLX continuous batching.

This module provides a Scheduler class that manages request scheduling
using mlx-lm's BatchGenerator for efficient continuous batching.

The scheduler follows vLLM's design with:
- Waiting queue for pending requests
- Running set for active requests
- Continuous batching via BatchGenerator
"""

import concurrent.futures
import copy
import gc
import importlib
import logging
import os
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple, Optional

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator,
    GenerationBatch,
    PromptProcessingBatch,
    SequenceStateMachine,
    generation_stream,
)
from mlx_lm.models.cache import (
    KVCache as _MLXKVCache,
)
from mlx_lm.models.cache import (
    RotatingKVCache as _MLXRotatingKVCache,
)
from mlx_lm.models.cache import (
    make_prompt_cache,
)
from mlx_lm.sample_utils import make_logits_processors

from .cache.observability import CacheRateTracker
from .cache.paged_cache import PagedCacheManager
from .cache.prefix_cache import BlockAwarePrefixCache
from .exceptions import PrefillMemoryExceededError, is_cache_corruption_error
from .prefill_progress import get_prefill_tracker
from .prefill_transient_tracker import PrefillTransientTracker
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .speculative.vlm_mtp import VLMMTPDrafter, run_vlm_mtp_decode
from .utils.fatal import FATAL_TEARDOWN_TIMEOUT_S, fatal_exit
from .utils.generation_config import load_generation_config_token_ids
from .utils.proc_memory import get_phys_footprint
from .utils.sampling import make_sampler as omlx_make_sampler
from .utils.tokenizer import create_streaming_detokenizer

# Module-level alias so Scheduler.__init__ can fall back to mlx-lm's default
# stream when no per-engine stream is provided.
_default_generation_stream = generation_stream


def _apply_suppress_token_ids(logits: Any, suppress_token_ids: tuple[int, ...]) -> Any:
    if suppress_token_ids:
        logits[..., list(suppress_token_ids)] = mx.array(float("-inf"))
    return logits


def _make_suppress_logits_processor(
    suppress_token_ids: set[int],
) -> Callable[[Any, Any], Any] | None:
    suppress_tuple = tuple(sorted(int(t) for t in suppress_token_ids))
    if not suppress_tuple:
        return None

    def _suppress_logits(tokens: Any, logits: Any) -> Any:
        return _apply_suppress_token_ids(logits, suppress_tuple)

    return _suppress_logits


def _make_suppressing_sampler(
    sampler: Callable[[Any], Any],
    suppress_token_ids: set[int],
) -> Callable[[Any], Any]:
    suppress_tuple = tuple(sorted(int(t) for t in suppress_token_ids))
    if not suppress_tuple:
        return sampler

    def _sample(logits: Any) -> Any:
        return sampler(_apply_suppress_token_ids(logits, suppress_tuple))

    return _sample


@dataclass
class _PreflightRejection:
    """Typed return for ``_preflight_memory_check`` / its token-count
    helper. Carries the human-readable diagnostic plus the numeric
    estimated / limit bytes so callers can populate
    ``PrefillMemoryExceededError`` without parsing the string.

    Same shim as ``preflight_or_raise`` (restored on main after the
    upstream merge dropped it); the typed shape is what
    ``tests/test_scheduler_prefill_memory_guard.py`` asserts against,
    and what PR #1452 carries upstream.
    """

    message: str
    estimated_bytes: int
    limit_bytes: int


@dataclass
class _VLMMTPDecodeState:
    """Per-request state for vlm_mtp decode that bypasses BatchGenerator.

    The wrapper generator yields plain Python ints (single-request mode).
    Scheduler iterates it one token per ``step()`` and feeds each token
    into ``_process_batch_responses`` via a synthesized ``_VLMMTPResponse``.
    """

    generator: Any  # Generator[int, None, None] from run_vlm_mtp_decode
    request: Request
    prompt_cache: list[Any]
    sampler: Callable[[Any], Any]
    state_machine: Any
    max_tokens: int
    # Plain stop-token set (EOS + request-specific) for direct membership
    # check; mlx-lm's SequenceStateMachine doesn't expose a "did the last
    # token finish" helper, so we keep a copy.
    stop_token_ids: set[int] = field(default_factory=set)
    emitted: int = 0
    finished: bool = False


@dataclass
class _VLMMTPResponse:
    """BatchGenerator.Response shim emitted by the vlm_mtp decode loop.

    Same field surface used by ``_process_batch_responses``: ``uid``,
    ``token``, ``finish_reason``, ``logprobs``, and an optional
    ``prompt_cache`` returned on the terminal yield so paged-cache reuse
    keeps working.
    """

    uid: int
    token: int
    finish_reason: Optional[str] = None
    logprobs: Any = None
    prompt_cache: Any = None


# Serializes Metal buffer-protocol access from the async store-cache worker
# against inference-thread mx.clear_cache / mx.synchronize calls that can
# invalidate the underlying buffer pool. Closes a SIGABRT path where
# _async_store_cache_worker reads tensor bytes via memoryview while the
# inference thread concurrently issues a reclaim-triggering mx op.
# See: https://github.com/jundot/omlx/issues/1106
_mx_buffer_access_lock = threading.RLock()


def _sync_and_clear_cache(stream=None):
    """Synchronize in-flight GPU work before clearing the Metal buffer cache.

    Without synchronization, mx.clear_cache() can release Metal buffers that
    are still referenced by in-flight command buffers submitted via
    mx.async_eval(). This causes the GPU driver to hit a
    'completeMemory() prepare count underflow' kernel panic on M4 hardware
    (and SIGSEGV/SIGABRT on M3).

    Held under _mx_buffer_access_lock so the async store-cache worker cannot
    observe a half-reclaimed Metal buffer pool while it is in the middle of
    reading tensor bytes via the Python buffer protocol (#1106).

    See: https://github.com/jundot/omlx/issues/300, #888, #1106
    """
    with _mx_buffer_access_lock:
        # The engine stream may not have in-flight work on the current thread
        # (e.g. external prefill submits to the default stream). On some MLX
        # builds mx.synchronize raises "There is no Stream(gpu, 0) in current
        # thread" in that case; swallow it since there is nothing to drain.
        target = stream if stream is not None else _default_generation_stream
        try:
            mx.synchronize(target)
        except RuntimeError:
            pass
        mx.synchronize()  # default stream
        mx.clear_cache()


def _safe_sync_stream(stream=None):
    """mx.synchronize(stream) that tolerates cross-thread calls.

    The per-engine stream is owned by the engine's executor thread. Teardown
    paths that run on the main thread (via EngineCore.close) hit "no Stream in
    current thread" RuntimeError. Swallow that specific case so cleanup can
    proceed; re-raise anything else so real GPU errors stay visible.
    """
    target = stream if stream is not None else _default_generation_stream
    try:
        mx.synchronize(target)
    except RuntimeError as e:
        if "no Stream" not in str(e):
            raise


def _env_int(name: str, default: int = 0) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignoring invalid integer env %s=%r", name, value)
        return default


def _collect_mx_arrays(value, out: list[mx.array]) -> None:
    if isinstance(value, mx.array):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_mx_arrays(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_mx_arrays(item, out)


def _eval_generation_batch_cache(batch_generator) -> int:
    generation_batch = getattr(batch_generator, "_generation_batch", None)
    prompt_cache = getattr(generation_batch, "prompt_cache", None)
    if not prompt_cache:
        return 0
    arrays: list[mx.array] = []
    for cache in prompt_cache:
        state = getattr(cache, "state", None)
        if state is not None:
            _collect_mx_arrays(state, arrays)
    if arrays:
        mx.eval(*arrays)
    return len(arrays)


class _StoreCacheGate:
    """Non-blocking counter that bounds in-flight store-cache submissions.

    Tracks how many KV caches are alive in the post-completion store-cache
    pipeline. _cleanup_finished records each submission with note_submitted()
    and _drain_pending_async_removes clears it with note_done() after the
    deferred batch_generator.remove() releases the request cache references;
    neither blocks the generation step. Backpressure is applied at admission
    instead — _schedule_waiting declines to admit new prefills while
    in_flight >= cap (see has_capacity), so token generation never stalls
    waiting for an SSD write (#1496).

    cap still bounds the concurrent extracted-KV count, which is the OOM
    guard for the burst-finish RAM growth reported in #1383. It is adjusted
    at runtime from ProcessMemoryEnforcer so the pipeline shrinks under
    memory pressure on smaller systems.
    """

    def __init__(self, cap: int) -> None:
        self._cap = max(1, cap)
        self._in_flight = 0
        self._lock = threading.Lock()

    def note_submitted(self) -> None:
        """Record a store-cache job handed to the executor (never blocks)."""
        with self._lock:
            self._in_flight += 1

    def note_done(self) -> None:
        """Record a store-cache job finished (future done callback)."""
        with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1

    def set_cap(self, cap: int) -> None:
        with self._lock:
            self._cap = max(1, cap)

    @property
    def cap(self) -> int:
        with self._lock:
            return self._cap

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def has_capacity(self) -> bool:
        """True when another submission would stay within cap.

        Read by _schedule_waiting to decide whether to admit a new prefill.
        """
        with self._lock:
            return self._in_flight < self._cap


# Import tiered cache components
try:
    from .cache.boundary_snapshot_store import BoundarySnapshotSSDStore
    from .cache.paged_ssd_cache import PagedSSDCacheManager
    from .memory_monitor import MemoryMonitor, estimate_mla_kv_bytes_per_token

    HAS_TIERED_CACHE = True
except ImportError:
    PagedSSDCacheManager = None
    BoundarySnapshotSSDStore = None
    MemoryMonitor = None
    estimate_mla_kv_bytes_per_token = None
    HAS_TIERED_CACHE = False

# Import cache type handlers for hybrid cache support
try:
    from .cache.hybrid_cache import ModelCacheConfig
    from .cache.type_registry import CacheTypeRegistry

    HAS_CACHE_TYPE_HANDLERS = True
except ImportError:
    CacheTypeRegistry = None
    ModelCacheConfig = None
    HAS_CACHE_TYPE_HANDLERS = False

# Import protocol-specific output parser support
try:
    from .adapter.output_parser import (
        OutputParserFactory,
        OutputParserSession,
        detect_output_parser,
    )

    HAS_OUTPUT_PARSER = True
except ImportError:
    OutputParserFactory = None
    OutputParserSession = None
    detect_output_parser = None
    HAS_OUTPUT_PARSER = False

logger = logging.getLogger(__name__)


class _PrefillAbortedError(Exception):
    """Raised when prefill is interrupted by a pending abort."""

    def __init__(self, aborted_uids: list[int], processed_tokens: int):
        self.aborted_uids = aborted_uids
        self.processed_tokens = processed_tokens
        super().__init__(
            f"Prefill aborted for UIDs {aborted_uids} " f"at {processed_tokens} tokens"
        )


@dataclass
class PrefillEvictionRequest:
    """Internal request for async LRU model eviction before prefill."""

    request_id: str
    model_id: str
    current_bytes: int
    target_cap_bytes: int
    predicted_transient_bytes: int
    requested_tokens: int
    reason: str


class _PrefillEvictionNeeded(Exception):
    """Raised inside scheduler.step() to pause and request async eviction."""

    def __init__(self, request: PrefillEvictionRequest):
        super().__init__(request.reason)
        self.request = request


def _prefill_memory_error_output(
    request_id: str,
    message: str,
    *,
    estimated_bytes: int | None = None,
    limit_bytes: int | None = None,
) -> RequestOutput:
    metadata: dict[str, int | str] = {"request_id": request_id}
    if estimated_bytes is not None:
        metadata["estimated_bytes"] = estimated_bytes
    if limit_bytes is not None:
        metadata["limit_bytes"] = limit_bytes
    return RequestOutput(
        request_id=request_id,
        finished=True,
        finish_reason="error",
        error=message,
        error_code="prefill_memory_exceeded",
        error_metadata=metadata,
    )


def _prefill_memory_exception_output(
    request_id: str,
    exc: PrefillMemoryExceededError,
) -> RequestOutput:
    return _prefill_memory_error_output(
        request_id,
        str(exc),
        estimated_bytes=exc.estimated_bytes,
        limit_bytes=exc.limit_bytes,
    )


@dataclass
class _PrefillState:
    """Intermediate state for a request undergoing chunked prefill.

    When chunked_prefill=True, a long prefill is spread across multiple
    step() calls (one prefill_step_size chunk per step). This dataclass
    holds all the state needed to resume prefill between steps.
    """

    request: Any
    cache: list  # Accumulated prompt_cache (mutated in-place by each chunk)
    tokens_remaining: Any  # mx.array shape (1, N) — tokens not yet prefilled
    last_token: list  # tokens[-1:] — passed to batch_generator.insert()
    tokens_processed: int  # Cumulative count for boundary snapshot math
    base_size: int  # Prefix cache offset at prefill start (for alignment)
    emitted_boundaries: dict  # {request_id: int} — last emitted boundary count
    boundary_enabled: bool  # Whether boundary snapshots are active
    block_size: int  # Copied from config.paged_cache_block_size
    total_length: int  # len(original tokens) for completeness
    # Pre-built insert-time params (set by _schedule_waiting before enqueuing)
    sampler: Any = None
    sm: Any = None
    per_row_lps: Any = None


@dataclass
class _InflightStoreInfo:
    tokens: list[int]
    extra_keys: tuple[Any, ...] | None = None
    extra_key_token_start: int | None = None
    extra_key_ranges: list[tuple[int, tuple[Any, ...]]] | None = None


@dataclass
class _CacheFreshnessWait:
    store_request_id: str
    future: concurrent.futures.Future
    common_prefix: int
    prompt_len: int
    deadline_s: float


# ---------------------------------------------------------------------------
# Monkey-patch GenerationBatch._step to call grammar accept_token() after
# sampling.  In the pipelined _step(), logits processors fill the bitmask
# (constrain NEXT token) but can't know which token was just sampled.
# After _original_step returns, self._next_tokens holds the freshly sampled
# tokens.  We eval them synchronously and accept in grammar processors.
# ---------------------------------------------------------------------------
# Authoritative per-uid row state for the generation batch.
#
# mlx-lm keeps ``samplers`` / ``logits_processors`` as positional lists that
# must stay aligned with ``uids``.  Heterogeneous continuous batching
# (extend/filter/split across prompt and generation batches) can leave stale
# or offset row slots behind; #1799 made the step crash-safe by normalising
# ``None`` slots, but a misaligned row silently runs the WRONG sampler and
# logits processors (e.g. a grammar/thinking-budget request decoding with no
# constraints at all).  The registry below records, at insert time, what each
# uid is supposed to run; the step chokepoint realigns the positional lists
# from it.  Bounded so a missing cleanup can never grow it unbounded.
class _RegisteredRow(NamedTuple):
    """What a uid is supposed to run, recorded at request insert."""

    sampler: Any
    logits_processors: list


_UID_ROW_REGISTRY_MAX = 4096
# Keyed by (id(model), uid): mlx-lm's BatchGenerator numbers uids per
# instance starting at 0, so two engines serving concurrently (or an engine
# reload) produce colliding uid sequences. The model object is the one
# identity both the insert sites and the step chokepoint can see.
_uid_row_registry: "OrderedDict[tuple[int, int], _RegisteredRow]" = OrderedDict()
# Engines run on separate executor threads and share this module-level
# registry; a plain OrderedDict is not safe under concurrent mutation.
_uid_row_registry_lock = threading.Lock()
# Drift corrections are worth one log line each, but a pathological batching
# pattern could correct on every merge; cap the WARNING rate and route the
# rest to DEBUG so the signal survives without flooding the logs.
_UID_ROW_DRIFT_WARNING_INTERVAL_S = 60.0
_uid_row_drift_last_warning = float("-inf")


def _register_uid_rows(model, uids, samplers, lps_rows) -> None:
    """Record the sampler and logits processors each freshly-inserted uid must run.

    Each (model, uid) key is inserted exactly once per request, so plain
    insertion order is enough for the oldest-first backstop eviction.
    """
    with _uid_row_registry_lock:
        for uid, sampler, lps in zip(uids, samplers, lps_rows):
            _uid_row_registry[(id(model), uid)] = _RegisteredRow(
                sampler, list(lps or ())
            )
        while len(_uid_row_registry) > _UID_ROW_REGISTRY_MAX:
            _uid_row_registry.popitem(last=False)


def _unregister_uid_row(model, uid) -> None:
    """Drop a finished request's row so heavy processors are not pinned
    until FIFO eviction; the bounded size stays as the backstop."""
    with _uid_row_registry_lock:
        _uid_row_registry.pop((id(model), uid), None)


def _unregister_uid_rows_for_model(model) -> None:
    """Drop every registry row for a model (generator reset, recovery, shutdown).

    The recovery and reset paths clear the uid maps wholesale instead of
    finishing requests one by one; releasing by model covers them, and leaves
    nothing behind that a later engine load could match if ``id(model)`` were
    recycled.
    """
    model_id = id(model)
    with _uid_row_registry_lock:
        for key in [key for key in _uid_row_registry if key[0] == model_id]:
            del _uid_row_registry[key]


def _row_drifted(current_lps, expected_lps) -> bool:
    """True when a slot's processors genuinely differ from the registered row.

    Two distinct empty lists are equivalent — the #1799 normalisation mints
    fresh ``[]`` objects every step — so only differing content counts. The
    caller's identity check is the steady-state fast path; this only runs
    past it.
    """
    if not current_lps and not expected_lps:
        return False
    return current_lps != expected_lps


def _log_drift_correction(uids, slot_count) -> None:
    """Log a corrected drift: one WARNING per window, the rest at DEBUG."""
    global _uid_row_drift_last_warning
    now = time.monotonic()
    rate_limited = now - _uid_row_drift_last_warning < _UID_ROW_DRIFT_WARNING_INTERVAL_S
    if not rate_limited:
        _uid_row_drift_last_warning = now
    (logger.debug if rate_limited else logger.warning)(
        "Realigned generation-batch row state from the uid registry "
        f"(uids={list(uids)}, had {slot_count} processor slots); "
        "stale or offset slots would have run the wrong sampler/processors."
    )


def _realigned_rows(model, uids, cur_samplers, cur_lps):
    """Rebuild the positional row lists in uid order from the registry.

    Registered uids take their recorded row; unregistered uids keep their
    current slot (the #1799 fallback), padding when the lists are shorter
    than ``uids``. Returns ``(samplers, logits_processors, drift)`` — drift
    only drives logging, the rebuilt lists are always installed. In steady
    state the slots already are the registry lists, so the identity check
    skips any comparison work.
    """
    model_id = id(model)
    with _uid_row_registry_lock:
        rows = [_uid_row_registry.get((model_id, uid)) for uid in uids]

    drift = len(cur_lps) != len(uids)
    samplers, lps = [], []
    for i, row in enumerate(rows):
        if row is not None:
            if not drift:
                if i >= len(cur_samplers):
                    drift = row.sampler is not None
                elif cur_samplers[i] is not row.sampler:
                    drift = True
            if (
                not drift
                and i < len(cur_lps)
                and cur_lps[i] is not row.logits_processors
            ):
                drift = _row_drifted(cur_lps[i], row.logits_processors)
            samplers.append(row.sampler)
            lps.append(row.logits_processors)
        else:
            samplers.append(cur_samplers[i] if i < len(cur_samplers) else None)
            lps.append(cur_lps[i] if i < len(cur_lps) else [])
    return samplers, lps, drift


def _omlx_realign_generation_batch_rows(self) -> None:
    """Realign positional row state with ``uids`` before any decode path reads it."""
    if self.logits_processors is None:
        self.logits_processors = []
    else:
        self.logits_processors = [
            procs if procs is not None else [] for procs in self.logits_processors
        ]

    uids = getattr(self, "uids", None) or []
    if not uids:
        return

    new_samplers, new_lps, drift = _realigned_rows(
        getattr(self, "model", None),
        uids,
        getattr(self, "samplers", None) or [],
        self.logits_processors,
    )
    if drift:
        _log_drift_correction(uids, len(self.logits_processors))
    self.logits_processors = new_lps
    self.samplers = new_samplers


_original_generation_batch_step = GenerationBatch._step


def _patched_generation_batch_step(self):
    # Build per-batch mRoPE deltas from UID mapping before each step.
    # This handles batch size changes during prompt split/generate.
    model = self.model
    if (
        getattr(model, "_uses_mrope", False)
        and getattr(model, "_uid_rope_deltas", None)
        and self.uids
    ):
        deltas = [model._uid_rope_deltas.get(uid, 0.0) for uid in self.uids]
        model.set_batch_rope_deltas(mx.array(deltas))

    # Defensive: mlx-lm's GenerationBatch._step does `any(self.logits_processors)`
    # and `for p in self.logits_processors[e]`, both of which crash when a row
    # slot is None.  Normalise the whole list AND every per-row slot to [] here,
    # at the single consumption chokepoint, so the original step and the
    # grammar-accept loop below are both safe regardless of slot origin.
    #
    # The insert call sites already wrap each request's processors as a list,
    # but that is not enough: on a heterogeneous continuous-batch merge,
    # mlx-lm's GenerationBatch.extend() re-introduces None slots via
    # `if not any(self.logits_processors): self.logits_processors =
    # [None] * len(self.uids)`.  `any([[], []])` is False, so empty-list slots
    # collapse back to None whenever a batch with no *active* processor merges
    # with a grammar-constrained one (e.g. a plain chat request joining a batch
    # that is serving a structured json_schema request).  Per-row normalisation
    # at this chokepoint is the only place that covers both insert and merge.
    # See #934 / #1747.
    _omlx_realign_generation_batch_rows(self)

    result = _original_generation_batch_step(self)

    # self._next_tokens contains the just-sampled tokens (async eval pending).
    # We need to accept them NOW so the next __call__ fills the correct bitmask.
    if any(self.logits_processors):
        from .api.grammar import GrammarConstraintProcessor

        has_grammar = any(
            isinstance(p, GrammarConstraintProcessor)
            for procs in self.logits_processors
            for p in procs
        )
        if has_grammar:
            # Force eval of the sampled tokens so we can read them.
            mx.eval(self._next_tokens)
            sampled = self._next_tokens.tolist()
            for e in range(len(self.uids)):
                for proc in self.logits_processors[e]:
                    if isinstance(proc, GrammarConstraintProcessor):
                        proc.accept_token(sampled[e])

    return result


GenerationBatch._omlx_realign_rows = _omlx_realign_generation_batch_rows
GenerationBatch._step = _patched_generation_batch_step


# ---------------------------------------------------------------------------
# Monkey-patch GenerationBatch.filter to keep logits_processors aligned with
# uids.  mlx-lm's filter only reindexes the processor list when at least one
# row has an active processor:
#
#     if any(self.logits_processors):
#         self.logits_processors = [self.logits_processors[idx] for idx in keep]
#
# There is no else branch (unlike the prompt-batch class, which resets to
# ``[[]] * len(keep)``), so when every slot is empty — the normal state after
# serving requests without per-request processors — the stale list survives
# while uids/tokens shrink.  A later extend() then appends the next request's
# processors BEHIND its own row index: the row reads a leftover empty slot and
# the real processor (thinking budget, grammar constraint) is silently never
# applied.  Which requests are affected depends on insertion/removal order,
# and alignment self-heals once the broken request finishes, so the symptom
# is an intermittently ignored thinking_budget or grammar.  See #934/#1747
# for the sibling None-slot collapse handled in _patched_generation_batch_step.
_original_generation_batch_filter = GenerationBatch.filter


def _patched_generation_batch_filter(self, keep):
    lps = self.logits_processors
    lps_inert = not lps or not any(lps)
    if lps is None:
        # ``any(None)`` inside the original filter raises TypeError.
        self.logits_processors = []
    _original_generation_batch_filter(self, keep)
    if lps_inert:
        # Original filter skipped the reindex; reset to one empty slot per
        # surviving row so extend() appends at the correct indices.
        self.logits_processors = [[] for _ in keep]


GenerationBatch.filter = _patched_generation_batch_filter


_TQ_SINGLETON_CACHE_TYPE: type[Any] | None = None

# Monkey-patch TurboQuantKVCache.merge so _merge_caches() works
try:
    from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache

    from .turboquant_kv import BatchTurboQuantKVCache as _BTQCache

    _TQ_SINGLETON_CACHE_TYPE = _TQCache
    if not hasattr(_TQCache, "merge"):
        _TQCache.merge = _BTQCache.merge
except ImportError:
    pass


# Regular singleton KV caches are already the fastest decode representation.
# mlx-lm's default _merge_caches([cache]) turns them into BatchKVCache even
# when there is only one active row, which slows text-only VLM decode. Install
# the minimal BatchGenerator methods needed while the row count remains one;
# _patched_extend_cache converts them back to batched caches before a second
# row is appended.
def _batch_indices_len(batch_indices: Any) -> int:
    try:
        return len(batch_indices)
    except TypeError:
        return int(getattr(batch_indices, "shape", (0,))[0] or 0)


def _regular_kv_filter_singleton(self, batch_indices):
    n = _batch_indices_len(batch_indices)
    if n == 0:
        self.keys = None
        self.values = None
        self.offset = 0
        return
    if n == 1:
        return
    raise NotImplementedError(
        f"{type(self).__name__}.filter only supports singleton pass-through; "
        "convert to a batched cache before keeping multiple rows."
    )


def _regular_rotating_kv_filter_singleton(self, batch_indices):
    n = _batch_indices_len(batch_indices)
    if n == 0:
        self.keys = None
        self.values = None
        self.offset = 0
        self._idx = 0
        return
    if n == 1:
        return
    raise NotImplementedError(
        f"{type(self).__name__}.filter only supports singleton pass-through; "
        "convert to a batched cache before keeping multiple rows."
    )


def _regular_cache_extract_singleton(self, idx: int):
    if int(idx) != 0:
        raise IndexError(f"{type(self).__name__} singleton cache only has row 0")
    return self


def _regular_cache_extend_singleton(self, other):
    raise NotImplementedError(
        f"{type(self).__name__}.extend requires batched conversion first"
    )


def _turboquant_filter_singleton(self, batch_indices):
    n = _batch_indices_len(batch_indices)
    if n == 0:
        self.keys = None
        self.values = None
        self.offset = 0
        self._cached_state = None
        self._cached_state_offset = -1
        if hasattr(self, "_shadow_keys"):
            self._shadow_keys = None
        if hasattr(self, "_shadow_values"):
            self._shadow_values = None
        return
    if n == 1:
        return
    raise NotImplementedError(
        f"{type(self).__name__}.filter only supports singleton pass-through; "
        "convert to a batched cache before keeping multiple rows."
    )


if not hasattr(_MLXKVCache, "filter"):
    _MLXKVCache.filter = _regular_kv_filter_singleton
if not hasattr(_MLXKVCache, "extract"):
    _MLXKVCache.extract = _regular_cache_extract_singleton
if not hasattr(_MLXKVCache, "extend"):
    _MLXKVCache.extend = _regular_cache_extend_singleton

if not hasattr(_MLXRotatingKVCache, "filter"):
    _MLXRotatingKVCache.filter = _regular_rotating_kv_filter_singleton
if not hasattr(_MLXRotatingKVCache, "extract"):
    _MLXRotatingKVCache.extract = _regular_cache_extract_singleton
if not hasattr(_MLXRotatingKVCache, "extend"):
    _MLXRotatingKVCache.extend = _regular_cache_extend_singleton

if _TQ_SINGLETON_CACHE_TYPE is not None:
    if not hasattr(_TQ_SINGLETON_CACHE_TYPE, "filter"):
        _TQ_SINGLETON_CACHE_TYPE.filter = _turboquant_filter_singleton
    if not hasattr(_TQ_SINGLETON_CACHE_TYPE, "extract"):
        _TQ_SINGLETON_CACHE_TYPE.extract = _regular_cache_extract_singleton
    if not hasattr(_TQ_SINGLETON_CACHE_TYPE, "extend"):
        _TQ_SINGLETON_CACHE_TYPE.extend = _regular_cache_extend_singleton

_mlx_lm_generate_module = importlib.import_module("mlx_lm.generate")
_original_merge_caches = _mlx_lm_generate_module._merge_caches
_original_ppb_split = PromptProcessingBatch.split
_REGULAR_SINGLETON_CACHE_TYPES = (_MLXKVCache, _MLXRotatingKVCache)


def _cache_layer_supports_singleton_passthrough(cache_obj: Any) -> bool:
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        return all(_cache_layer_supports_singleton_passthrough(c) for c in sub_caches)
    return hasattr(cache_obj, "filter") and hasattr(cache_obj, "extract")


def _to_batched_cache_layer(cache_obj: Any) -> Any:
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        converted = tuple(_to_batched_cache_layer(c) for c in sub_caches)
        if all(a is b for a, b in zip(sub_caches, converted)):
            return cache_obj
        return type(cache_obj)(*converted)
    if isinstance(cache_obj, _REGULAR_SINGLETON_CACHE_TYPES):
        return cache_obj.merge([cache_obj])
    if (
        _TQ_SINGLETON_CACHE_TYPE is not None
        and type(cache_obj) is _TQ_SINGLETON_CACHE_TYPE
    ):
        return cache_obj.merge([cache_obj])
    return cache_obj


def _extend_cache_layer(cache_a: Any, cache_b: Any) -> Any:
    sub_a = getattr(cache_a, "caches", None)
    sub_b = getattr(cache_b, "caches", None)
    if isinstance(sub_a, (list, tuple)) and isinstance(sub_b, (list, tuple)):
        cache_a.caches = tuple(
            _extend_cache_layer(ca, cb) for ca, cb in zip(sub_a, sub_b)
        )
        return cache_a

    cache_a = _to_batched_cache_layer(cache_a)
    cache_b = _to_batched_cache_layer(cache_b)
    cache_a.extend(cache_b)
    return cache_a


def _patched_merge_caches(caches):
    if not caches:
        return []
    if len(caches) == 1:
        merged = []
        for layer_cache in caches[0]:
            if _cache_layer_supports_singleton_passthrough(layer_cache):
                merged.append(layer_cache)
            elif hasattr(layer_cache, "merge"):
                merged.append(layer_cache.merge([layer_cache]))
            else:
                raise ValueError(
                    f"{type(layer_cache)} does not yet support batching with history"
                )
        return merged
    return _original_merge_caches(caches)


def _patched_extend_cache(cache_a, cache_b):
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    return [_extend_cache_layer(ca, cb) for ca, cb in zip(cache_a, cache_b)]


def _patched_ppb_split(self, indices):
    sorted_indices = sorted(indices)
    if sorted_indices and sorted_indices == list(range(len(self.uids))):
        new_batch = self.__class__.__new__(self.__class__)
        new_batch.model = self.model
        new_batch.uids = self.uids
        new_batch.prompt_cache = self.prompt_cache
        new_batch.tokens = self.tokens
        new_batch.prefill_step_size = self.prefill_step_size
        new_batch.samplers = self.samplers
        new_batch.fallback_sampler = self.fallback_sampler
        # Defensive: normalise None → [] to avoid mlx-lm crash in _step
        lps = self.logits_processors if self.logits_processors is not None else []
        new_batch.logits_processors = lps
        new_batch.state_machines = self.state_machines
        new_batch.max_tokens = self.max_tokens
        if hasattr(self, "_omlx_glm_dsa_adaptive_prefill"):
            new_batch._omlx_glm_dsa_adaptive_prefill = (
                self._omlx_glm_dsa_adaptive_prefill
            )

        self.uids = []
        self.prompt_cache = []
        self.tokens = []
        self.samplers = []
        self.logits_processors = []
        self.state_machines = []
        self.max_tokens = []
        return new_batch
    return _original_ppb_split(self, indices)


_mlx_lm_generate_module._merge_caches = _patched_merge_caches
_mlx_lm_generate_module._extend_cache = _patched_extend_cache
PromptProcessingBatch.split = _patched_ppb_split


# Monkey-patch ChunkedKVCache for Llama-4 (Scout / Maverick): mlx_lm's
# ChunkedKVCache lacks the batch-aware methods (`merge`, `filter`, `extract`,
# `size`, `extend`) that BatchGenerator's continuous-batching code path
# expects, so any chat completion targeting a Llama-4 model raises
# `Cache corruption not recoverable: <ChunkedKVCache> does not yet support
# batching with history` and returns 500.
#
# Real continuous batching with chunked attention is unimplemented upstream;
# this patch installs batch=1 pass-throughs so serialized requests work.
# Run the server with `--max-concurrent-requests 1` to honor the assumption.
try:
    from mlx_lm.models.cache import ChunkedKVCache as _CKVCache

    _ckvcache_methods_skipped: list[str] = []

    if not hasattr(_CKVCache, "merge"):

        @classmethod
        def _ckvcache_merge_passthrough(cls, caches):
            if len(caches) == 1:
                return caches[0]
            raise NotImplementedError(
                "ChunkedKVCache.merge for batch_size > 1 is not implemented. "
                "Run with --max-concurrent-requests 1 when serving Llama-4."
            )

        _CKVCache.merge = _ckvcache_merge_passthrough
    else:
        _ckvcache_methods_skipped.append("merge")

    if not hasattr(_CKVCache, "filter"):

        def _ckvcache_filter_passthrough(self, batch_indices):
            try:
                n = len(batch_indices)
            except TypeError:
                n = int(getattr(batch_indices, "shape", (0,))[0] or 0)
            if n == 0:
                self.keys = None
                self.values = None
                self.offset = 0
                self.start_position = 0
                return
            if n == 1:
                return
            raise NotImplementedError(
                f"ChunkedKVCache.filter with batch_size={n} > 1 is not "
                "implemented. Run with --max-concurrent-requests 1 when "
                "serving Llama-4."
            )

        _CKVCache.filter = _ckvcache_filter_passthrough
    else:
        _ckvcache_methods_skipped.append("filter")

    if not hasattr(_CKVCache, "extract"):

        def _ckvcache_extract_passthrough(self, idx):
            return self

        _CKVCache.extract = _ckvcache_extract_passthrough
    else:
        _ckvcache_methods_skipped.append("extract")

    if not hasattr(_CKVCache, "size"):

        def _ckvcache_size(self):
            return max(0, self.offset - self.start_position)

        _CKVCache.size = _ckvcache_size
    else:
        _ckvcache_methods_skipped.append("size")

    if not hasattr(_CKVCache, "extend"):

        def _ckvcache_extend_passthrough(self, other):
            if other is None or other.empty():
                return
            if self.empty():
                self.keys = other.keys
                self.values = other.values
                self.offset = other.offset
                self.start_position = other.start_position
                return
            raise NotImplementedError(
                "ChunkedKVCache.extend across non-empty caches is not "
                "supported. Run with --max-concurrent-requests 1."
            )

        _CKVCache.extend = _ckvcache_extend_passthrough
    else:
        _ckvcache_methods_skipped.append("extend")

    if _ckvcache_methods_skipped:
        # Upstream may have landed implementations between mlx_lm upgrades.
        # Surface which ones so a regression in Llama-4 batching is visible
        # to operators without diffing the patch against installed mlx_lm.
        logger.info(
            "ChunkedKVCache patch: methods already present upstream, " "skipped: %s",
            ", ".join(_ckvcache_methods_skipped),
        )
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Monkey-patch PromptProcessingBatch.prompt to set mRoPE deltas before the
# prompt processing loop.  Without this, batched VLM prompt processing
# (e.g. the 1-token final prompt after external prefill) would use
# per-request offsets without rope_deltas, corrupting attention masks
# for concurrent VLM requests.
# ---------------------------------------------------------------------------
_original_ppb_prompt = PromptProcessingBatch.prompt


def _patched_ppb_prompt(self, tokens):
    model = self.model
    if (
        getattr(model, "_uses_mrope", False)
        and getattr(model, "_uid_rope_deltas", None)
        and self.uids
    ):
        deltas = [model._uid_rope_deltas.get(uid, 0.0) for uid in self.uids]
        model.set_batch_rope_deltas(mx.array(deltas))
    return _original_ppb_prompt(self, tokens)


PromptProcessingBatch.prompt = _patched_ppb_prompt


# Cache class names known to be sliceable (no boundary snapshots needed).
# ChunkedKVCache is included once the batch=1 patch above installs its
# extract/filter/size pass-throughs; without it Llama-4 requests fall
# back to the snapshot path unnecessarily.
_KNOWN_SLICEABLE_CACHE_TYPES = frozenset(
    {
        "KVCache",
        "BatchKVCache",
        "QuantizedKVCache",
        "TurboQuantKVCache",
        "BatchTurboQuantKVCache",
        "ChunkedKVCache",
        "MiniMaxM3KVCache",
    }
)


_TURBOQUANT_KV_CACHE_TYPES = frozenset(
    {
        "TurboQuantKVCache",
        "BatchTurboQuantKVCache",
    }
)


def _is_turboquant_kv_cache(cache_obj: Any) -> bool:
    return type(cache_obj).__name__ in _TURBOQUANT_KV_CACHE_TYPES


def _is_turboquant_kv_family_cache(cache_obj: Any) -> bool:
    """Cache layer counted by TurboQuant's skip-last full-attention rule."""
    return isinstance(cache_obj, _MLXKVCache) or _is_turboquant_kv_cache(cache_obj)


class _BoundaryStoreUnavailable(Exception):
    """No boundary-aligned snapshot exists for non-sliceable cache state.

    Raised inside the final-response store path when the model needs
    boundary snapshots but none were captured (e.g. every capture was
    skipped by the speculative-decode skew guard). Storing the live
    state instead would label off-boundary recurrent/rotating state
    with a block-boundary token count and corrupt later prefix hits;
    the handler skips the store and releases the block references.
    """


def _first_leaf_cache_offset(cache_obj: Any) -> int | None:
    """First integer ``offset`` found walking into composite caches.

    CacheList-style wrappers (DeepSeek-V4 / GLM per-layer caches) expose
    no ``offset`` themselves — the token position lives on their leading
    sub-cache (RotatingKVCache / KVCache, whose ``offset`` counts
    forwarded tokens). Walk depth-first so that leading leaf decides;
    later sub-caches may count something else entirely (PoolingCache's
    ``offset`` is the pooled-window count).
    """
    subs = getattr(cache_obj, "caches", None)
    if isinstance(subs, (list, tuple)):
        for sub in subs:
            offset = _first_leaf_cache_offset(sub)
            if offset is not None:
                return offset
        return None
    offset = getattr(cache_obj, "offset", None)
    if offset is None:
        return None
    try:
        return int(offset)
    except Exception:
        return None


def _prompt_cache_needs_snapshots(prompt_cache: list[Any]) -> bool:
    """Return True if any layer cache is non-sliceable (needs snapshots).

    Checks the cache objects created during prefill. If all layers
    are known-sliceable types (e.g. KVCache), boundary snapshots
    are unnecessary and can be skipped entirely.
    """
    for cache_obj in prompt_cache:
        sub_caches = getattr(cache_obj, "caches", None)
        if isinstance(sub_caches, (list, tuple)):
            for sub in sub_caches:
                if type(sub).__name__ not in _KNOWN_SLICEABLE_CACHE_TYPES:
                    return True
        elif type(cache_obj).__name__ not in _KNOWN_SLICEABLE_CACHE_TYPES:
            return True
    return False


def _batch_generator_all_tokens(request: Any) -> list[int]:
    """Seed tokens for mlx-lm's TokenBuffer before the kickoff token."""
    token_ids = getattr(request, "prompt_token_ids", None)
    if token_ids is None:
        return []
    return list(token_ids[:-1])


def _cache_layer_token_count(cache_obj: Any) -> int:
    """Return the number of tokens stored in a single cache layer."""
    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)) and sub_caches:
        return max(_cache_layer_token_count(sub_cache) for sub_cache in sub_caches)

    offset = getattr(cache_obj, "offset", None)
    if isinstance(offset, (int, float)):
        return int(offset)

    size_fn = getattr(cache_obj, "size", None)
    if callable(size_fn):
        try:
            return int(size_fn())
        except Exception:
            return 0

    return 0


def _cache_base_sizes(caches: list[Any]) -> int:
    """Return the base token count of a single-request cache list."""
    if not caches:
        return 0
    try:
        return max(_cache_layer_token_count(c) for c in caches)
    except Exception:
        return 0


def _collect_cache_storage_arrays(cache_obj: Any) -> list[mx.array]:
    """Collect concrete backing arrays from cache objects, not state slices."""
    arrays: list[mx.array] = []

    if isinstance(cache_obj, mx.array):
        return [cache_obj]

    sub_caches = getattr(cache_obj, "caches", None)
    if isinstance(sub_caches, (list, tuple)):
        for sub_cache in sub_caches:
            arrays.extend(_collect_cache_storage_arrays(sub_cache))

    array_cache = getattr(cache_obj, "cache", None)
    if isinstance(array_cache, (list, tuple)):
        for item in array_cache:
            arrays.extend(_collect_cache_storage_arrays(item))

    for attr in ("keys", "values", "left_padding", "lengths"):
        value = getattr(cache_obj, attr, None)
        if isinstance(value, mx.array):
            arrays.append(value)

    return arrays


def _materialize_cache_storage(cache_list: list[Any]) -> None:
    """Force restored cache backing arrays concrete before decode begins."""
    arrays: list[mx.array] = []
    for cache_obj in cache_list:
        arrays.extend(_collect_cache_storage_arrays(cache_obj))
    if arrays:
        with _mx_buffer_access_lock:
            mx.eval(*arrays)


def _seed_text_only_mrope_delta_for_cached_prefill(model: Any, request: Any) -> None:
    """Seed zero mRoPE delta after clearing text-only cached-prefix state."""
    if getattr(request, "cached_tokens", 0) <= 0:
        return
    lm = getattr(model, "_language_model", None)
    if lm is None or not hasattr(lm, "_rope_deltas"):
        return
    lm._rope_deltas = mx.zeros((1, 1), dtype=mx.int64)


def _vlm_extra_seq_slice(val: mx.array, s: slice) -> mx.array:
    """Slice a VLM extra tensor along its seq dimension.

    Standard layout (batch=1, seq, ...): seq at dim 1.
    Special layout (e.g. mRoPE (3, batch, seq)): seq at last dim.
    """
    if val.ndim >= 3 and val.shape[0] == 1:
        return val[:, s]
    if val.ndim >= 3:
        return val[..., s]
    return val[:, s]


def _slice_vlm_extra(extra: dict[str, Any], n: int) -> dict[str, Any]:
    """Slice VLM extra kwargs to first n tokens along seq dimension."""
    sliced: dict[str, Any] = {}
    for key, val in extra.items():
        if isinstance(val, mx.array) and val.ndim >= 2:
            sliced[key] = _vlm_extra_seq_slice(val, slice(None, n))
        else:
            sliced[key] = val
    return sliced


def _advance_vlm_extra(extra: dict[str, Any], n: int) -> dict[str, Any]:
    """Advance VLM extra kwargs past first n tokens along seq dimension."""
    advanced: dict[str, Any] = {}
    for key, val in extra.items():
        if isinstance(val, mx.array) and val.ndim >= 2:
            advanced[key] = _vlm_extra_seq_slice(val, slice(n, None))
        else:
            advanced[key] = val
    return advanced


def _get_attr_or_key(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    try:
        value = getattr(obj, name)
    except Exception:
        return None
    if type(value).__module__.startswith("unittest.mock"):
        return None
    return value


def _model_declares_llama4(model: Any) -> bool:
    """Return True if the loaded model/config tree declares Llama 4."""
    seen: set[int] = set()
    stack = [model]
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if _get_attr_or_key(obj, "model_type") == "llama4":
            return True

        for attr in ("config", "args", "text_config", "language_config", "llm_config"):
            child = _get_attr_or_key(obj, attr)
            if child is not None and not isinstance(
                child, (str, bytes, int, float, bool)
            ):
                stack.append(child)
    return False


class SchedulingPolicy(Enum):
    """Scheduling policy for request ordering."""

    FCFS = "fcfs"  # First-Come-First-Served
    PRIORITY = "priority"  # Priority-based


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    # Maximum number of concurrent requests in the batch
    max_num_seqs: int = 256
    # Maximum tokens to process per step (for prefill chunking)
    max_num_batched_tokens: int = 8192
    # Scheduling policy
    policy: SchedulingPolicy = SchedulingPolicy.FCFS
    # BatchGenerator settings (passed directly to mlx-lm)
    completion_batch_size: int = 32
    # Per-forward embedding input chunk size
    embedding_batch_size: int = 32
    prefill_step_size: int = 2048
    # When True, long prefills are processed one chunk per step() call,
    # interleaved with decode steps for already-running requests. This
    # reduces TTFT for concurrent requests but adds per-step overhead.
    chunked_prefill: bool = False

    # Paged cache settings (internal defaults)
    paged_cache_block_size: int = 256  # Tokens per block
    max_cache_blocks: int | None = (
        None  # Auto-calculated from available KV cache memory
    )
    initial_cache_blocks: int = (
        256  # Starting blocks (grows dynamically to max_cache_blocks)
    )

    # paged SSD cache settings (oMLX only supports paged SSD-based caching)
    # When paged_ssd_cache_dir is set, oMLX stores KV cache on paged SSD for prefix reuse.
    # When None, no oMLX caching (mlx-lm BatchGenerator manages KV internally).
    paged_ssd_cache_dir: str | None = (
        None  # Path for paged SSD cache storage (None = disabled)
    )
    hot_cache_only: bool = False
    paged_ssd_cache_max_size: int = 100 * 1024 * 1024 * 1024  # 100GB default
    hot_cache_max_size: int = 0  # In-memory hot cache size in bytes (0 = disabled)
    hot_cache_budget: Any | None = None  # Shared process-wide hot cache budget

    # Model identification (for cache isolation between different models)
    model_name: str = ""  # OpenAI API model name (e.g., "mlx-community/Llama-3.2-3B")

    # GC/cleanup settings (memory optimization)
    gc_cleanup_interval: int = 0  # Steps between gc.collect() calls (0=disabled)
    mlx_cache_cleanup_interval: int = 512  # Steps between mx.clear_cache() calls


@dataclass
class SchedulerOutput:
    """
    Output from a scheduling step.

    Contains information about what was scheduled and results.
    """

    # Requests scheduled in this step
    scheduled_request_ids: list[str] = field(default_factory=list)
    # Total tokens scheduled
    num_scheduled_tokens: int = 0
    # Requests that finished in this step
    finished_request_ids: set[str] = field(default_factory=set)
    # Request outputs (tokens generated)
    outputs: list[RequestOutput] = field(default_factory=list)
    # Internal signal consumed by EngineCore; not part of any API response.
    prefill_eviction_request: PrefillEvictionRequest | None = None
    # Whether any work was done
    has_work: bool = False


class _BoundarySnapshotProvider:
    """Dict-like loader for extracted boundary snapshots.

    Used by ``store_cache()`` to load snapshots from SSD one block at a time
    or serve pre-extracted in-memory snapshots. In-memory snapshots must already
    be in the ``_extract_cache_states`` dict format so this provider can be used
    safely from the async store-cache worker without touching raw MLX cache
    objects on the wrong thread.
    """

    def __init__(
        self,
        store: Any,  # Optional[BoundarySnapshotSSDStore]
        request_id: str,
        valid_tcs: list[int],
        in_memory_snapshots: dict[int, Any],
    ) -> None:
        self._store = store
        self._request_id = request_id
        self._valid_tcs = set(valid_tcs)
        self._in_memory = in_memory_snapshots

    def __contains__(self, tc: int) -> bool:
        return tc in self._valid_tcs

    def __getitem__(self, tc: int) -> Any:
        snap = self._in_memory.get(tc)
        if snap is not None:
            return snap
        if self._store is not None:
            return self._store.load(self._request_id, tc)
        return None

    def __len__(self) -> int:
        return len(self._valid_tcs)

    def __bool__(self) -> bool:
        return bool(self._valid_tcs)

    def iter_in_memory_extracted(self):
        """Yield pre-extracted in-memory snapshots for pre-evaluation."""
        for tc in sorted(self._valid_tcs):
            snap = self._in_memory.get(tc)
            if snap is not None:
                yield snap


class Scheduler:
    """
    Scheduler for continuous batching using mlx-lm BatchGenerator.

    This scheduler manages the lifecycle of requests:
    1. Requests arrive and are added to the waiting queue
    2. Scheduler moves requests from waiting to running (via BatchGenerator)
    3. BatchGenerator processes all running requests together
    4. Finished requests are removed and outputs returned

    .. note::

       ``_DEFERRED_CLEAR_DELAY`` controls how many generation steps to wait
       after the last request completion before calling ``mx.clear_cache()``.
       Immediate clearing races with IOKit's asynchronous ``completeMemory()``
       callbacks, causing 'prepare count underflow' kernel panics (#435).
       8 steps (~10-40 ms at typical generation speeds) gives IOKit ample
       time to process those callbacks while still reclaiming Metal buffers
       fast enough to prevent TTFT spikes (#411).

    The key insight is that mlx-lm's BatchGenerator already implements
    continuous batching at the token level, so we use it as the backend.
    """

    _DEFERRED_CLEAR_DELAY: int = 8
    _GENERATION_OVERFLOW_PATTERN = "__next_prime overflow"
    _MAX_GENERATION_OVERFLOW_RETRIES = 1

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: SchedulerConfig | None = None,
        stream: Any | None = None,
    ):
        """
        Initialize the scheduler.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Scheduler configuration
            stream: Optional mx.Stream for this engine. Falls back to the
                module-level _default_generation_stream when not provided.
        """
        self.model = model
        # Deep-copy the tokenizer so the scheduler owns an independent Rust
        # tokenizer backend.  Without this, concurrent access from the asyncio
        # event loop (encode/apply_chat_template in engine handlers) and the
        # MLX executor thread (scheduler.step) causes
        # "RuntimeError: Already borrowed" from the HuggingFace tokenizers
        # Rust RefCell.  See: https://github.com/huggingface/tokenizers/issues/537
        self.tokenizer = copy.deepcopy(tokenizer)
        self.config = copy.copy(config) if config else SchedulerConfig()
        self._stream = stream if stream is not None else _default_generation_stream
        self._serialize_llama4_requests = _model_declares_llama4(model)
        if self._serialize_llama4_requests and self.config.max_num_seqs > 1:
            logger.info(
                "Llama 4 detected; serializing requests because ChunkedKVCache "
                "does not support multi-row batching yet"
            )

        # Load additional EOS tokens from generation_config.json.
        # Some models (e.g. GLM-4.6V) define multiple EOS tokens there
        # that are not in tokenizer.eos_token_id.
        self._generation_config_eos: set[int] | None = (
            self._load_generation_config_eos()
        )

        # Load generation_config.suppress_tokens once and apply them on every
        # sampling path. Gemma 4 uses this to suppress multimodal close markers.
        self._model_suppress_tokens: set[int] = self._load_model_suppress_tokens()

        # For strict RotatingKVCache reuse, align paged cache block size to
        # the model's rotating window size when paged cache is enabled.
        self._align_block_size_with_rotating_window()
        # For ArraysCache-only models (no RotatingKVCache), use a larger block
        # size to reduce boundary snapshot overhead during prefill.
        self._enlarge_block_size_for_arrays_cache()

        # TurboQuant KV cache (set by engine if model_settings has it enabled)
        self._turboquant_kv_bits: float | None = None
        self._turboquant_skip_last: bool = True
        # Memoized MLA-architecture detection (see _model_uses_mla / #1613).
        self._mla_model: bool | None = None
        self._glm_dsa_adaptive_prefill = None
        try:
            from .patches.glm_moe_dsa.generate_patch import (
                _glm_dsa_adaptive_prefill_config,
            )

            self._glm_dsa_adaptive_prefill = _glm_dsa_adaptive_prefill_config(
                model, self.config.prefill_step_size
            )
        except Exception:
            logger.debug("GLM DSA adaptive prefill config unavailable", exc_info=True)
        if self._glm_dsa_adaptive_prefill is not None:
            logger.info(
                "GLM DSA adaptive scheduler prefill enabled: step=%d after=%d "
                "min_remaining=%d",
                self._glm_dsa_adaptive_prefill.step_size,
                self._glm_dsa_adaptive_prefill.after,
                self._glm_dsa_adaptive_prefill.min_remaining,
            )
        self._minimax_m3_adaptive_prefill = None
        try:
            from .patches.minimax_m3.generate_patch import (
                _minimax_m3_adaptive_prefill_config,
            )

            self._minimax_m3_adaptive_prefill = _minimax_m3_adaptive_prefill_config(
                model,
                self.config.prefill_step_size,
                getattr(self.config, "model_name", None),
            )
        except Exception:
            logger.debug(
                "MiniMax M3 adaptive prefill config unavailable", exc_info=True
            )
        if self._minimax_m3_adaptive_prefill is not None:
            logger.info(
                "MiniMax M3 adaptive scheduler prefill enabled: step=%d after=%d "
                "min_remaining=%d",
                self._minimax_m3_adaptive_prefill.step_size,
                self._minimax_m3_adaptive_prefill.after,
                self._minimax_m3_adaptive_prefill.min_remaining,
            )

        # Request management - following vLLM's design
        self.waiting: deque[Request] = deque()  # Waiting queue (FCFS)
        self.running: dict[str, Request] = {}  # Running requests by ID
        # Chunked prefill queue: requests whose prefill spans multiple steps.
        # Populated when chunked_prefill=True and prompt exceeds prefill_step_size.
        self.prefilling: deque[Request] = deque()
        self._prefill_states: dict[str, _PrefillState] = {}
        self.requests: dict[str, Request] = {}  # All requests by ID
        self.finished_req_ids: set[str] = set()  # Recently finished
        self._generation_overflow_recovery_ids: set[str] = set()

        # Thread-safe set for deferred aborts (main thread → executor thread)
        # CPython GIL guarantees set.add() and `x in set` are atomic.
        self._pending_abort_ids: set[str] = set()

        # Deferred between-turn Metal reclaim, requested by the (asyncio-thread)
        # ProcessMemoryEnforcer under pinned-model memory pressure. A bare bool
        # is GIL-atomic to set; it is drained on the inference thread at the top
        # of step() (same cross-thread idiom as _pending_abort_ids) because the
        # enforcer must never touch Metal directly.
        self._pending_reclaim_request: bool = False

        # Lock-free admin snapshot. Published at the end of each step() while
        # the engine thread is the sole writer of running/waiting; the admin
        # endpoint reads the dict reference atomically (GIL) and never iterates
        # the live mutable structures.
        self._admin_snapshot: dict[str, Any] = {
            "running_by_id": {},
            "waiting": [],
        }

        # Memory limits for inline prefill checking.
        # Set by ProcessMemoryEnforcer; propagated to BatchGenerator.
        self._memory_limit_bytes: int = 0  # soft limit (dynamic, jittery)
        self._memory_hard_limit_bytes: int = 0  # dynamic ceiling (throttle target)
        # Stable physical cap = min(static_ceiling, metal_cap). Used ONLY to
        # abort an in-flight prefill, so a transient dynamic-ceiling dip can't
        # kill a near-complete request that actually fits. 0 => fall back to
        # _memory_hard_limit_bytes (pre-propagation / old enforcer).
        self._memory_abort_limit_bytes: int = 0
        # Last mx.get_active_memory() sample taken on this scheduler's MLX
        # executor thread. The background memory enforcer reads this cached
        # value during active decode instead of touching MLX/Metal directly.
        self._last_mlx_active_memory_bytes: int = 0
        # Component ceilings — propagated alongside the hard limit so the
        # rejection-path error message can identify which constraint is
        # binding and suggest the right remedy (close apps / raise tier /
        # raise iogpu.wired_limit_mb / reduce context). 0 = not set yet.
        self._memory_static_ceiling_bytes: int = 0
        self._memory_dynamic_ceiling_bytes: int = 0
        self._memory_metal_cap_bytes: int = 0
        self._memory_hot_cache_reserved_bytes: int = 0
        # Tier name propagated alongside the breakdown. For ``custom`` the
        # "dynamic" ceiling is the user-pinned ``custom_ceiling_bytes``
        # rather than computed reclaimable memory, so the advice ladder
        # must steer the user to that knob instead of "close other apps".
        self._memory_guard_tier: str = "balanced"
        self._prefill_memory_guard: bool = False  # set by ProcessMemoryEnforcer
        # Set to True by ProcessMemoryEnforcer when phys_footprint crosses
        # soft_threshold. Schedulers stop admitting new prefills while this is
        # set; in-flight requests proceed.
        self._admission_paused: bool = False
        # Adaptive prefill throttle params, propagated from enforcer.
        # Until set, _adaptive_chunk_size is a no-op (returns requested as-is).
        self._prefill_headroom_safety: float = self._PREFILL_HEADROOM_SAFETY
        self._prefill_safe_zone_ratio: float = 0.80
        self._prefill_min_chunk_tokens: int = 256
        self._prefill_abort_margin: float = self._PREFILL_ABORT_MARGIN
        self._pending_prefill_eviction_request: PrefillEvictionRequest | None = None
        self._memory_admission_blocked_request_id: str | None = None
        self._memory_admission_blocked_since: float = 0.0
        self._store_cache_admission_blocked_request_id: str | None = None
        self._store_cache_admission_blocked_since: float = 0.0
        # EWMA estimator of per-token chunk transient bytes, used by
        # _adaptive_chunk_size in the caution zone. Owned per-scheduler.
        _tracker_model_id = ""
        if config is not None and config.model_name:
            _tracker_model_id = os.path.basename(config.model_name.rstrip("/"))
        self._prefill_transient_tracker = PrefillTransientTracker(
            model_id=_tracker_model_id
        )

        # SpecPrefill: draft model for attention-based sparse prefill
        self._specprefill_draft_model: Any | None = None
        # Track active specprefill request for RoPE cleanup
        self._specprefill_active_request_id: str | None = None

        # DEBUG-only prefix-cache divergence probe (issue #1003): recent
        # stored cache sequences, so a miss can be traced to the exact
        # token where the new prompt diverges from what was cached.
        # Populated only when debug logging is enabled — zero cost otherwise.
        self._cache_probe_seqs: deque[tuple[str, list[int]]] = deque(maxlen=4)

        model_name_lower = (self.config.model_name or "").lower()
        default_kv_eval_interval = 256 if "minimax" in model_name_lower else 0
        self._decode_eval_kv_cache_interval: int = max(
            0,
            _env_int(
                "OMLX_DECODE_EVAL_KV_CACHE_INTERVAL",
                default_kv_eval_interval,
            ),
        )
        self._tokens_since_kv_cache_eval: int = 0
        if self._decode_eval_kv_cache_interval > 0:
            logger.info(
                "Decode KV cache materialization interval set to %d tokens",
                self._decode_eval_kv_cache_interval,
            )

        # VLM MTP: gemma4_assistant drafter attached by VLMBatchedEngine.
        # When set, eligible requests bypass mlx-lm BatchGenerator for decode
        # and run through mlx-vlm's _mtp_rounds round loop instead.
        self._vlm_mtp_drafter: VLMMTPDrafter | None = None
        # Active vlm_mtp decode generators keyed by synthesized negative uid
        # (negative to make collision with BatchGenerator uids impossible).
        self._vlm_mtp_active: dict[int, _VLMMTPDecodeState] = {}
        self._vlm_mtp_next_uid: int = -1
        # Per-request settings snapshot for vlm_mtp routing (block size etc.).
        # Injected by VLMBatchedEngine.set_vlm_mtp_drafter alongside the drafter.
        self._vlm_mtp_draft_block_size: int | None = None

        # Phase timing instrumentation for cache-on overhead diagnostics.
        # Accumulated wall-time per phase + invocation count, dumped at request
        # end or via get_phase_stats(). Adds ~100ns per measurement.
        self._phase_total_ms: dict[str, float] = defaultdict(float)
        self._phase_count: dict[str, int] = defaultdict(int)

        # Async store_cache executor (G2-async). Offloads the post-finish
        # bulk memcpy (28GB+ per 32k request) off the inference thread so
        # response streaming isn't blocked by it.
        self._store_cache_executor: concurrent.futures.ThreadPoolExecutor | None = None
        # Gate that caps in-flight store-cache submissions. Set only when
        # tiered cache is enabled (alongside _store_cache_executor).
        self._store_cache_gate: _StoreCacheGate | None = None
        # Pending (uid, request_id, future) entries waiting for async store
        # to finish before batch_generator.remove() can safely run. Drained
        # at the start of every step.
        self._pending_async_removes: deque = deque()
        # Track in-flight store futures per request_id for lookup wait /
        # shutdown wait.
        self._inflight_store_futures: dict[str, concurrent.futures.Future] = {}
        self._inflight_store_info: dict[str, _InflightStoreInfo] = {}
        # Admission-only cache freshness waits. A waiting request can pause at
        # the front of the queue for a relevant in-flight store without
        # blocking the scheduler step that continues existing decode/prefill.
        self._cache_freshness_waits: dict[str, _CacheFreshnessWait] = {}
        self._prefix_cache_prepared: set[str] = set()

        # Mapping between our request IDs and BatchGenerator UIDs
        self.request_id_to_uid: dict[str, int] = {}
        self.uid_to_request_id: dict[int, str] = {}

        # BatchGenerator - the actual batching engine
        self.batch_generator: BatchGenerator | None = None
        self._current_sampler_params: tuple | None = None
        # Boundary cache snapshots for stateful non-sliceable caches (e.g., ArraysCache).
        # request_id -> {token_count -> snapshot_cache_or_None}
        # Multiple snapshots per request to support per-block ArraysCache state storage.
        # Values are None when offloaded to SSD via _boundary_snapshot_store.
        self._boundary_cache_snapshots: dict[str, dict[int, Any]] = {}
        # Lazy detection flag: True/False once determined, None before first check.
        self._boundary_snapshot_required: bool | None = None
        # SSD store for offloading boundary snapshots (initialized in _init_tiered_cache).
        self._boundary_snapshot_store: BoundarySnapshotSSDStore | None = None

        # paged SSD cache for KV state persistence (oMLX only supports paged SSD-based caching)
        self.paged_cache_manager: PagedCacheManager | None = None
        self.block_aware_cache: BlockAwarePrefixCache | None = None
        self.paged_ssd_cache_manager: PagedSSDCacheManager | None = None
        self._cache_rate_tracker = CacheRateTracker()
        # Prefill-peak estimator used by ``_preflight_memory_check`` /
        # ``preflight_or_raise``. Only the estimator path is exercised
        # here (it reads head_dim / num_layers / num_kv_heads via
        # ``set_model_info`` below); ``eviction_enabled=False`` so the
        # monitor does not gate on ``max_kv_cache_memory`` — paged SSD
        # mode never wants this monitor making eviction decisions, and
        # we have no real value to pass for that field at this point.
        #
        # This auto-init was wired up in b6a69c4 then silently dropped
        # by an upstream merge (same pattern as ``preflight_or_raise``
        # in d40ab80). Without it the guard short-circuits at the
        # ``memory_monitor is None`` gate for every request — Pi prompts
        # that should be rejected by the configured hard limit instead
        # sail straight into chunked prefill and OOM at the Metal cap.
        if MemoryMonitor is not None:
            self.memory_monitor: MemoryMonitor | None = MemoryMonitor(
                max_kv_cache_memory=None,
                eviction_enabled=False,
            )
            self._set_model_info_for_monitor()
        else:
            self.memory_monitor: MemoryMonitor | None = None

        # Initialize paged SSD cache if paged_ssd_cache_dir is specified
        if self.config.paged_ssd_cache_dir:
            # Calculate max_blocks automatically if not specified
            if self.config.max_cache_blocks is not None:
                max_blocks = self.config.max_cache_blocks
            else:
                max_blocks = self._calculate_max_blocks()

            # Initialize paged cache manager for block metadata
            self.paged_cache_manager = PagedCacheManager(
                block_size=self.config.paged_cache_block_size,
                max_blocks=max_blocks,
                model_name=self.config.model_name,
                initial_blocks=self.config.initial_cache_blocks,
            )
            self.block_aware_cache = BlockAwarePrefixCache(
                model=model,
                paged_cache_manager=self.paged_cache_manager,
            )

            # Initialize paged SSD cache. If the backing directory is not
            # usable (for example, an external cache drive is disconnected),
            # continue with cache disabled instead of leaving partial state.
            cache_initialized = self._init_tiered_cache()

            # Set cold restore callback for prefix cache
            if cache_initialized and self.paged_ssd_cache_manager is not None:
                self.block_aware_cache.set_cold_restore_callback(
                    self._restore_block_from_cold
                )
                if self.config.hot_cache_only:
                    logger.info(
                        f"hot-cache-only mode enabled: "
                        f"block_size={self.config.paged_cache_block_size}, "
                        f"max_blocks={max_blocks}"
                    )
                else:
                    logger.info(
                        f"paged SSD cache enabled: {self.config.paged_ssd_cache_dir}, "
                        f"block_size={self.config.paged_cache_block_size}, "
                        f"max_blocks={max_blocks}"
                    )

                # Async store_cache executor: single worker so submissions are
                # serialized (matches the original synchronous order) and we
                # never have two stores racing on the same paged_ssd index.
                self._store_cache_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="omlx-store-cache",
                )
                # Gate caps the post-completion store-cache pipeline so a burst
                # of finishes cannot pile up unbounded KV caches in memory while
                # the single writer drains. Cap starts at max_concurrent_requests
                # and is shrunk by ProcessMemoryEnforcer under pressure (#1383).
                self._store_cache_gate = _StoreCacheGate(cap=self.config.max_num_seqs)
            else:
                self._disable_paged_cache_components()
                logger.info(
                    "oMLX cache disabled after paged SSD cache initialization failed"
                )
        else:
            logger.info(
                "oMLX cache disabled (mlx-lm BatchGenerator manages KV internally)"
            )

        # Streaming detokenizers for proper UTF-8 handling (one per active request)
        # NOTE: No pooling - each request gets a fresh instance to prevent state contamination
        self._request_detokenizers: dict[str, Any] = (
            {}
        )  # request_id → active detokenizer

        # Protocol-specific output parser support (e.g. Harmony, Gemma 4)
        self._output_parser_factory: OutputParserFactory | None = None
        self._output_parser_kind: str | None = None
        self._output_parser_sessions: dict[str, OutputParserSession] = {}
        self._is_harmony_model: bool = False
        if HAS_OUTPUT_PARSER and detect_output_parser is not None:
            try:
                model_config = None
                if hasattr(model, "config"):
                    # model.config may be a Pydantic model or dict
                    try:
                        if hasattr(model.config, "model_dump"):
                            model_config = model.config.model_dump()
                        elif hasattr(model.config, "dict"):
                            model_config = model.config.dict()
                        elif isinstance(model.config, dict):
                            model_config = model.config
                        else:
                            # Try to convert to dict via __dict__
                            model_config = getattr(model.config, "__dict__", None)
                    except Exception as e:
                        logger.debug(f"Failed to extract model.config: {e}")
                elif hasattr(model, "args"):
                    try:
                        if hasattr(model.args, "model_dump"):
                            model_config = model.args.model_dump()
                        elif hasattr(model.args, "__dict__"):
                            model_config = model.args.__dict__
                    except Exception as e:
                        logger.debug(f"Failed to extract model.args: {e}")

                self._output_parser_factory = detect_output_parser(
                    self.config.model_name,
                    self.tokenizer,
                    model_config,
                )
                if self._output_parser_factory is not None:
                    self._output_parser_kind = self._output_parser_factory.kind
                    self._is_harmony_model = self._output_parser_kind == "harmony"
                    logger.info(
                        "Output parser detected: %s for %s, stop_tokens=%s",
                        self._output_parser_kind,
                        self.config.model_name,
                        sorted(self._output_parser_factory.stop_token_ids),
                    )
            except Exception as e:
                logger.warning(f"Error detecting output parser: {e}, assuming none")
                self._output_parser_factory = None
                self._output_parser_kind = None
                self._is_harmony_model = False

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Step counter for periodic cleanup
        self._step_counter = 0
        # Deferred Metal cache cleanup after request completion.
        # Immediate mx.clear_cache() after request completion races with
        # IOKit's asynchronous completeMemory() callbacks, causing
        # 'prepare count underflow' kernel panics. Deferring the clear
        # by a few generation steps gives IOKit time to process callbacks.
        #
        # Stored as the absolute step number at which the clear should fire,
        # rather than a countdown integer.  This avoids the burst-completion
        # bug (#557): with max_num_seqs > 1 two requests can finish in the
        # same batch.  The old "only set if None" guard meant the second
        # completion never extended the window, so the first request's KV
        # cache blocks could be re-allocated before IOKit finished its
        # completeMemory() callbacks.  Using max() ensures the window always
        # covers the *latest* completion.
        # None = no deferred clear pending; int = step at which to fire.
        self._deferred_clear_at: int | None = None

        # Cache XTC special tokens (newline + EOS) — stable per tokenizer.
        # Must be after _is_harmony_model / _generation_config_eos init
        # since _get_xtc_special_tokens() delegates to _get_stop_tokens().
        self._xtc_special_tokens: list[int] = self._get_xtc_special_tokens()

        # Drop transient aliases after ownership moves to scheduler fields;
        # close()/deep_reset() clear those fields during teardown.
        model = None
        tokenizer = None

    @contextmanager
    def _phase_timer(self, phase: str):
        """Lightweight wall-time accumulator for cache-on overhead diagnostics.

        Tracks total ms and invocation count per named phase. Intended for
        boundary capture / store_cache / hot cache eviction hot paths.
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._phase_total_ms[phase] += (time.perf_counter() - t0) * 1000.0
            self._phase_count[phase] += 1

    def get_phase_stats(self) -> dict[str, dict[str, float]]:
        """Return accumulated phase timings for diagnostics.

        Returns dict of phase -> {total_ms, count, avg_ms}.
        """
        result = {}
        for phase, total in self._phase_total_ms.items():
            count = self._phase_count.get(phase, 0)
            result[phase] = {
                "total_ms": total,
                "count": count,
                "avg_ms": total / count if count else 0.0,
            }
        return result

    def _periodic_clear_threshold_bytes(self) -> int:
        """Cache-bytes threshold above which the periodic clear runs.

        Defaults to memory_limit/3 when a process memory limit is set,
        otherwise an absolute 2 GiB floor. Each periodic clear releases
        the entire MLX buffer pool in one batch; gating it on accumulated
        bytes avoids producing IOGPUFamily refcount bursts when the pool
        is already small.
        """
        if self._memory_limit_bytes > 0:
            return max(self._memory_limit_bytes // 3, 2 * 1024**3)
        return 2 * 1024**3

    def _should_periodic_clear_cache(self) -> bool:
        """Decide whether the per-step periodic clear should fire.

        Returns False unless ``mlx_cache_cleanup_interval`` is configured,
        the step counter just landed on the interval boundary, AND the
        MLX buffer pool exceeds the threshold. See #978 / #1040 for the
        kernel panic class this gating is meant to mitigate.
        """
        interval = self.config.mlx_cache_cleanup_interval
        if interval <= 0 or self._step_counter % interval != 0:
            return False
        return mx.get_cache_memory() > self._periodic_clear_threshold_bytes()

    @staticmethod
    def _collect_arrays_from_extracted_cache(
        extracted_cache: list[Any],
    ) -> list[Any]:
        """Collect lazy mx.array references from an _extracted_cache payload.

        Used by G2-async to force a single batched mx.eval on the inference
        thread before handing the cache off to the store_cache worker. The
        worker can then call _extract_tensor_bytes safely (no further Metal
        graph evaluation needed for non-bfloat16, no-op for already-evaluated).

        Walks the per-layer dict format produced by _extract_cache_states:
        each layer is {state, meta_state, class_name, cache_type}, where
        state is a tuple of mx.arrays (or nested for CacheList / TurboQuant).
        """
        arrays: list[Any] = []
        for layer in extracted_cache or []:
            if not isinstance(layer, dict):
                continue
            state = layer.get("state", ())
            if isinstance(state, mx.array):
                arrays.append(state)
                continue
            if not isinstance(state, (list, tuple)):
                continue
            for item in state:
                if isinstance(item, mx.array):
                    arrays.append(item)
                elif isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, mx.array):
                            arrays.append(sub)
                        elif hasattr(sub, "_fields"):
                            # NamedTuple state (TurboQuant). Walk fields.
                            for fname in sub._fields:
                                val = getattr(sub, fname, None)
                                if isinstance(val, mx.array):
                                    arrays.append(val)
                elif hasattr(item, "_fields"):
                    for fname in item._fields:
                        val = getattr(item, fname, None)
                        if isinstance(val, mx.array):
                            arrays.append(val)
        return arrays

    def _async_store_cache_worker(
        self,
        request_id: str,
        token_sequence_to_store: list[int],
        cache_to_store: list[Any],
        model_cache_config: Any | None,
        intermediate_snapshots: dict[int, list[Any]] | None,
        extra_keys: tuple[Any, ...] | None,
        extra_key_token_start: int | None,
        extra_key_ranges: list[tuple[int, tuple[Any, ...]]] | None,
        hot_cache_write_back: bool = True,
    ) -> None:
        """Run store_cache + paged_cache cleanup off the inference thread.

        Pre-conditions enforced by the caller (_cleanup_finished):
        - mx.eval() (a FULL, blocking eval — NOT async_eval) was called on
          the inference thread for all KV cache arrays in cache_to_store, so
          they are fully materialized concrete buffers before this worker
          runs. This is the load-bearing invariant: MLX streams ARE
          thread-local (each engine's generation stream is created via
          mx.new_thread_local_stream on the inference thread). A KV array
          left lazy and bound to self._stream cannot be materialized on this
          worker thread — _extract_block_tensor_slice (slices) and
          _extract_tensor_bytes (bf16 -> uint16 view) would re-dispatch the
          source op to self._stream's index, which does NOT exist on this
          thread, aborting the process ("There is no Stream(gpu, N) in
          current thread"). Boundary snapshots get the same treatment via
          _eval_snapshot_cache at capture time.
        - Because the sources are concrete, the worker's own slice/view ops
          consume materialized buffers and bind their new ops to the
          always-present default stream (gpu,0). The _safe_sync_stream call
          below is now belt-and-suspenders (the owner already drained the
          work); correctness no longer depends on its cross-thread
          mx.synchronize, whose "no Stream" RuntimeError it tolerantly
          swallows.
        - All mx-buffer access here is held under _mx_buffer_access_lock,
          serializing the bf16 view+eval and the buffer-protocol reads
          against inference-thread _sync_and_clear_cache (which also takes
          that lock), so mx.clear_cache cannot reclaim a buffer mid-read.
        - batch_generator.remove(uid) is deferred until this worker
          completes (handled by _drain_pending_async_removes).

        paged_cache_manager and block_aware_cache rely on
        threading.RLock so concurrent access from main and worker is safe.
        """
        try:
            # Hold _mx_buffer_access_lock across the worker's mx-buffer
            # access. store_cache eventually drives _extract_tensor_bytes,
            # which reads raw bytes via the buffer protocol; serializing
            # against inference-thread mx.clear_cache / mx.synchronize calls
            # prevents a SIGABRT when those reclaim the underlying Metal
            # buffer pool mid-read (#1106).
            with _mx_buffer_access_lock:
                with self._phase_timer("store_cache_worker_sync"):
                    _safe_sync_stream(self._stream)
                if hot_cache_write_back:
                    block_table = self.block_aware_cache.store_cache(
                        request_id,
                        token_sequence_to_store,
                        cache_to_store,
                        model_cache_config=model_cache_config,
                        boundary_snapshots=intermediate_snapshots,
                        extra_keys=extra_keys,
                        extra_key_token_start=extra_key_token_start,
                        extra_key_ranges=extra_key_ranges,
                    )
                else:
                    block_table = self.block_aware_cache.store_cache(
                        request_id,
                        token_sequence_to_store,
                        cache_to_store,
                        model_cache_config=model_cache_config,
                        boundary_snapshots=intermediate_snapshots,
                        extra_keys=extra_keys,
                        extra_key_token_start=extra_key_token_start,
                        extra_key_ranges=extra_key_ranges,
                        hot_cache_write_back=False,
                    )
            if block_table is None and self.paged_cache_manager is not None:
                block_table = self.paged_cache_manager.get_block_table(request_id)
            if block_table and self.paged_cache_manager is not None:
                self.paged_cache_manager.release_for_eviction(block_table.block_ids)
            if self.block_aware_cache is not None:
                self.block_aware_cache.clear_request_entry(request_id)
        except Exception as e:
            logger.warning("Async store_cache failed for %s: %s", request_id, e)

    def _drain_pending_async_removes(self) -> bool:
        """Process deferred batch_generator.remove() calls from prior steps.

        Called at the start of every step. For each pending entry whose async
        store_cache future has finished, perform batch_generator.remove() on
        the inference thread (Metal-safe) and finalize cleanup state. Entries
        whose futures are still in flight are kept for a later step, but they
        do not block later completed entries from releasing cache references.
        """
        if not self._pending_async_removes:
            return False
        drained = False
        pending: deque = deque()
        while self._pending_async_removes:
            uid, request_id, future = self._pending_async_removes.popleft()
            if future is not None and not future.done():
                # Worker still busy. Keep it for the next step, but continue
                # scanning so later completed futures can release memory now.
                pending.append((uid, request_id, future))
                continue
            # Surface worker exceptions for visibility (don't crash step loop).
            if future is not None:
                try:
                    exc = future.exception()
                except concurrent.futures.CancelledError:
                    logger.warning("Async store_cache for %s was cancelled", request_id)
                else:
                    if exc is not None:
                        logger.warning(
                            "Async store_cache for %s raised: %s", request_id, exc
                        )
            try:
                # Run batch_generator.remove on the inference thread.
                try:
                    _safe_sync_stream(self._stream)
                    self._remove_uid_from_active_batch(uid)
                    if hasattr(self.model, "unregister_rope_delta"):
                        self.model.unregister_rope_delta(uid)
                except Exception as e:
                    logger.warning(
                        "Deferred batch_generator.remove(uid=%s) failed: %s",
                        uid,
                        e,
                    )
                # Cleanup uid maps now that the slot is reclaimable.
                _unregister_uid_row(self.model, uid)
                if uid in self.uid_to_request_id:
                    del self.uid_to_request_id[uid]
                if request_id in self.request_id_to_uid:
                    del self.request_id_to_uid[request_id]
                self._inflight_store_futures.pop(request_id, None)
                self._inflight_store_info.pop(request_id, None)
                self._clear_request_admission_bookkeeping(request_id)
                # Boundary snapshots were kept on disk for the worker; safe to
                # delete now that the future has completed. Cleanup was
                # deferred from _cleanup_finished to avoid racing the worker's
                # boundary_snapshot_store.load() calls with rmtree.
                if self._boundary_snapshot_store is not None:
                    self._boundary_snapshot_store.cleanup_request(request_id)
                # Worker no longer holds extracted_cache — pop request from
                # self.requests and drop the cache buffer references so MLX
                # arrays can be freed.
                req_to_remove = self.requests.pop(request_id, None)
                if req_to_remove is not None:
                    req_to_remove._extracted_cache = None
                    req_to_remove.prompt_cache = None
            finally:
                gate = self._store_cache_gate
                if gate is not None:
                    gate.note_done()
                drained = True
        self._pending_async_removes = pending
        return drained

    def _calculate_max_blocks(self) -> int:
        """
        Calculate maximum cache blocks for paged SSD-only mode.

        In paged SSD-only mode, blocks don't consume GPU memory (data is on paged SSD),
        so we use a large default that can be limited by SSD capacity.

        Returns:
            Maximum number of cache blocks to allocate.
        """
        # In paged SSD-only mode, use a large default since blocks don't consume GPU memory
        # The actual limit is SSD capacity (paged_ssd_cache_max_size)
        max_blocks = 100000  # Large default for paged SSD-only mode

        block_size = self.config.paged_cache_block_size
        logger.info(
            f"paged SSD-only mode: max_blocks={max_blocks}, block_size={block_size} tokens"
        )

        return max_blocks

    def _collect_rotating_window_sizes(
        self,
        cache_obj: Any,
        window_sizes: set[int],
    ) -> None:
        """Collect rotating window sizes recursively from cache objects."""
        sub_caches = getattr(cache_obj, "caches", None)
        if isinstance(sub_caches, (list, tuple)):
            for sub_cache in sub_caches:
                self._collect_rotating_window_sizes(sub_cache, window_sizes)

        class_name = type(cache_obj).__name__
        is_rotating_cache = class_name in ("RotatingKVCache", "BatchRotatingKVCache")
        if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
            is_rotating_cache = (
                is_rotating_cache or CacheTypeRegistry.is_rotating_family(class_name)
            )
        if is_rotating_cache:
            max_size = getattr(cache_obj, "max_size", 0)
            if isinstance(max_size, int) and max_size > 0:
                window_sizes.add(max_size)

    def _detect_rotating_window_sizes(self) -> set[int]:
        """Detect rotating window sizes from model.make_cache() if available."""
        if not hasattr(self.model, "make_cache"):
            return set()

        try:
            cache_list = self.model.make_cache()
        except Exception as e:
            logger.debug(f"Failed to inspect model rotating window sizes: {e}")
            return set()

        if cache_list is None:
            return set()

        window_sizes: set[int] = set()
        for cache_obj in cache_list:
            self._collect_rotating_window_sizes(cache_obj, window_sizes)

        return window_sizes

    # Target range for RotatingKVCache block size alignment.
    # Using a multiple of window_size within this range reduces SSD I/O
    # overhead (fewer, larger block files) while keeping cache restore
    # reprocessing reasonable.
    _ROTATING_BLOCK_SIZE_MIN = 512
    _ROTATING_BLOCK_SIZE_MAX = 1024

    def _align_block_size_with_rotating_window(self) -> None:
        """
        Align paged cache block size to a multiple of RotatingKVCache
        window size, targeting 512-1024 tokens per block.

        Block size must be a multiple of window_size so that block
        boundaries align with rotation boundaries. When window_size is
        small (e.g. 128), using it directly as block_size creates too
        many small files. Instead we pick the smallest multiple of
        window_size that falls within [_ROTATING_BLOCK_SIZE_MIN,
        _ROTATING_BLOCK_SIZE_MAX].
        """
        if not self.config.paged_ssd_cache_dir:
            return

        window_sizes = self._detect_rotating_window_sizes()
        if not window_sizes:
            return

        if len(window_sizes) > 1:
            raise ValueError(
                "Multiple RotatingKVCache window sizes detected "
                f"({sorted(window_sizes)}). Set a single aligned block size or "
                "disable paged cache for this model."
            )

        window_size = next(iter(window_sizes))

        # Find the smallest multiple of window_size >= _ROTATING_BLOCK_SIZE_MIN.
        # If window_size itself is already >= max, just use window_size.
        lo = self._ROTATING_BLOCK_SIZE_MIN
        hi = self._ROTATING_BLOCK_SIZE_MAX

        if window_size >= hi or window_size >= lo:
            target_block_size = window_size
        else:
            # window_size < lo: pick smallest multiple in [lo, hi]
            multiplier = (lo + window_size - 1) // window_size  # ceil(lo / ws)
            target_block_size = multiplier * window_size
            if target_block_size > hi:
                # Fall back to largest multiple <= hi
                target_block_size = (hi // window_size) * window_size
                if target_block_size < window_size:
                    target_block_size = window_size

        if self.config.paged_cache_block_size != target_block_size:
            logger.info(
                "Aligning paged cache block_size=%s to %s "
                "(RotatingKVCache window_size=%s, multiplier=%sx)",
                self.config.paged_cache_block_size,
                target_block_size,
                window_size,
                target_block_size // window_size,
            )
            self.config.paged_cache_block_size = target_block_size

    # Default block size for ArraysCache-only hybrid models.
    # Match prefill_step_size (2048) so that boundary caching ON/OFF
    # produces identical prefill chunk sizes, eliminating float32↔dtype
    # roundtrip differences in GatedDeltaNet recurrent state.
    _ARRAYS_CACHE_BLOCK_SIZE = 2048

    def _enlarge_block_size_for_arrays_cache(self) -> None:
        """Enlarge block size for ArraysCache-only hybrid models.

        When a model uses ArraysCache (GatedDeltaNet) but not RotatingKVCache,
        a larger block size reduces the number of boundary snapshot stops during
        prefill while still storing valid per-block recurrent state.

        This is skipped if RotatingKVCache was already detected (block size was
        aligned to its window size) or if the user explicitly set a block size
        larger than the default.
        """
        if not self.config.paged_ssd_cache_dir:
            return

        # Skip if RotatingKVCache already adjusted block size.
        rotating_sizes = self._detect_rotating_window_sizes()
        if rotating_sizes:
            return

        # Detect ArraysCache from model.make_cache()
        if not hasattr(self.model, "make_cache"):
            return

        try:
            cache_list = self.model.make_cache()
        except Exception:
            return

        if cache_list is None:
            return

        has_arrays_cache = any(
            self._cache_tree_has_arrays_cache(cache_obj) for cache_obj in cache_list
        )
        if not has_arrays_cache:
            return

        target = self._ARRAYS_CACHE_BLOCK_SIZE
        if self.config.paged_cache_block_size >= target:
            return

        logger.info(
            "Enlarging paged cache block_size=%s to %s for "
            "ArraysCache hybrid model (reduces boundary snapshot overhead)",
            self.config.paged_cache_block_size,
            target,
        )
        self.config.paged_cache_block_size = target

    @staticmethod
    def _cache_tree_has_arrays_cache(cache_obj: Any) -> bool:
        """Return True if cache_obj contains ArraysCache (recursively)."""
        sub_caches = getattr(cache_obj, "caches", None)
        if isinstance(sub_caches, (list, tuple)):
            return any(
                Scheduler._cache_tree_has_arrays_cache(sub) for sub in sub_caches
            )
        return type(cache_obj).__name__ in ("ArraysCache", "SizedArraysCache")

    def _load_generation_config_eos(self) -> set[int] | None:
        """Load EOS token IDs from generation_config.json if available."""
        try:
            model_ref = getattr(self.tokenizer, "name_or_path", None) or getattr(
                self.config, "model_name", None
            )
            if not model_ref:
                return None

            result = load_generation_config_token_ids(model_ref, "eos_token_id")
            if result is None:
                return None

            # Only return if there are tokens beyond what tokenizer already provides
            tokenizer_eos = getattr(self.tokenizer, "eos_token_id", None)
            if tokenizer_eos is not None:
                existing = (
                    {tokenizer_eos}
                    if isinstance(tokenizer_eos, int)
                    else set(tokenizer_eos)
                )
                extra = result - existing
                if extra:
                    logger.info(
                        f"Loaded {len(extra)} additional EOS token(s) from "
                        f"generation_config.json: {extra}"
                    )
                    return result
            return result
        except Exception as e:
            logger.debug(f"Could not load generation_config.json: {e}")
            return None

    def _load_model_suppress_tokens(self) -> set[int]:
        """Load suppress_tokens from generation_config.json if available.

        These tokens are set to -inf during generation. For Gemma 4 unified,
        generation_config marks the multimodal close markers (<image|>,
        <audio|>) this way.
        """
        try:
            model_ref = getattr(self.tokenizer, "name_or_path", None) or getattr(
                self.config, "model_name", None
            )
            if not model_ref:
                return set()

            result = load_generation_config_token_ids(model_ref, "suppress_tokens")
            if not result:
                return set()
            logger.info(
                f"Loaded {len(result)} suppress token(s) from "
                f"generation_config.json: {result}"
            )
            return result
        except Exception as e:
            logger.debug(f"Could not load suppress_tokens from generation_config: {e}")
            return set()

    def _get_stop_tokens(self) -> set[int]:
        """Get stop token IDs from tokenizer and generation_config."""
        stop_tokens = set()
        if (
            hasattr(self.tokenizer, "eos_token_id")
            and self.tokenizer.eos_token_id is not None
        ):
            if isinstance(self.tokenizer.eos_token_id, list):
                stop_tokens.update(self.tokenizer.eos_token_id)
            else:
                stop_tokens.add(self.tokenizer.eos_token_id)
        if (
            hasattr(self.tokenizer, "eos_token_ids")
            and self.tokenizer.eos_token_ids is not None
        ):
            eos_ids = self.tokenizer.eos_token_ids
            if isinstance(eos_ids, int):
                stop_tokens.add(eos_ids)
            else:
                stop_tokens.update(eos_ids)

        # Include end-of-turn token for models that use turn-based
        # conversation delimiters (e.g. Gemma 4 with <turn|>).  Without
        # this the model generates the full next turn after its response.
        eot_token_id = getattr(self.tokenizer, "eot_token_id", None)
        if eot_token_id is not None:
            if isinstance(eot_token_id, list):
                stop_tokens.update(eot_token_id)
            else:
                stop_tokens.add(eot_token_id)
        elif hasattr(self.tokenizer, "eot_token") and self.tokenizer.eot_token:
            # Encode the string value if eot_token_id isn't directly exposed
            try:
                encoded = self.tokenizer.encode(
                    self.tokenizer.eot_token, add_special_tokens=False
                )
                if encoded:
                    stop_tokens.update(encoded)
            except Exception:
                pass

        # Read additional EOS tokens from generation_config.json.
        # Some models (e.g. GLM-4.6V) define multiple EOS tokens there
        # that are not reflected in tokenizer.eos_token_id.
        if self._generation_config_eos is not None:
            stop_tokens.update(self._generation_config_eos)

        # Protocol parsers need to observe their own stop tokens so they can
        # apply channel-aware handling (for example, Harmony analysis end
        # should continue into the final channel).

        return stop_tokens

    # _update_stop_tokens deleted — per-request stop tokens are now
    # handled via SequenceStateMachine passed to insert().

    def _get_detokenizer(self, request_id: str):
        """Get or create a streaming detokenizer for a request.

        This enables proper UTF-8 handling for multi-byte characters
        (Korean, Chinese, Japanese, etc.) during streaming.

        NOTE: Each request gets a fresh detokenizer instance. Pooling was removed
        because internal state (byte buffers) can leak between requests even after
        finalize()/reset(), causing text corruption (e.g., spaces inserted in paths,
        character swaps like 'features' -> 'featurse').
        """
        if request_id not in self._request_detokenizers:
            # Always create a fresh detokenizer - no pooling to prevent state contamination
            detok = create_streaming_detokenizer(
                self.tokenizer,
                model_path=self.config.model_name,
            )
            if detok is None:
                # Fallback: return None, we'll use decode([token])
                return None
            detok.reset()
            self._request_detokenizers[request_id] = detok
        return self._request_detokenizers[request_id]

    def _cleanup_detokenizer(self, request_id: str):
        """Clean up detokenizer for a finished request.

        NOTE: Detokenizers are NOT pooled - each request gets a fresh instance
        to prevent state contamination that causes text corruption.
        """
        self._request_detokenizers.pop(request_id, None)
        # Let GC collect - no pooling to prevent state contamination

    def _get_output_parser_session(
        self, request_id: str
    ) -> Optional["OutputParserSession"]:
        """Get or create a protocol-specific output parser session."""
        if self._output_parser_factory is None:
            return None

        if request_id not in self._output_parser_sessions:
            self._output_parser_sessions[request_id] = (
                self._output_parser_factory.create_session(self.tokenizer)
            )
        return self._output_parser_sessions[request_id]

    def _cleanup_output_parser_session(self, request_id: str):
        """Remove any per-request protocol parser session."""
        self._output_parser_sessions.pop(request_id, None)

    def _get_xtc_special_tokens(self) -> list[int]:
        """Get special tokens to exclude from XTC sampling.

        Parser-owned stop tokens stay out of BatchGenerator stop-token matching
        so protocol parsers can handle them channel-aware, but XTC must still
        protect them from diversity masking.
        """
        tokens = self.tokenizer.encode("\n")
        tokens.extend(self._get_stop_tokens())
        if self._output_parser_factory is not None:
            tokens.extend(self._output_parser_factory.stop_token_ids)
        return tokens

    def _create_batch_generator(
        self, sampling_params: SamplingParams
    ) -> BatchGenerator:
        """Create a BatchGenerator with the given sampling parameters."""
        sampler = omlx_make_sampler(
            temp=sampling_params.temperature,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
            top_k=sampling_params.top_k,
            xtc_probability=sampling_params.xtc_probability,
            xtc_threshold=sampling_params.xtc_threshold,
            xtc_special_tokens=self._xtc_special_tokens,
        )

        # Create logits processors for repetition/presence/frequency penalties
        logits_processors = make_logits_processors(
            repetition_penalty=(
                sampling_params.repetition_penalty
                if sampling_params.repetition_penalty != 1.0
                else None
            ),
            presence_penalty=(
                sampling_params.presence_penalty
                if sampling_params.presence_penalty != 0.0
                else None
            ),
            frequency_penalty=(
                sampling_params.frequency_penalty
                if sampling_params.frequency_penalty != 0.0
                else None
            ),
        )

        suppress_processor = _make_suppress_logits_processor(
            self._model_suppress_tokens
        )
        if suppress_processor is not None:
            logits_processors.append(suppress_processor)

        # Convert stop tokens from Set[int] to Sequence[Sequence[int]]
        # for the new BatchGenerator API (each stop token is a sequence).
        stop_tokens_set = self._get_stop_tokens()
        if sampling_params.stop_token_ids:
            stop_tokens_set.update(sampling_params.stop_token_ids)
        stop_tokens_seq = [[t] for t in stop_tokens_set] if stop_tokens_set else None

        bg = BatchGenerator(
            model=self.model,
            max_tokens=sampling_params.max_tokens,
            stop_tokens=stop_tokens_seq,
            sampler=sampler,
            logits_processors=logits_processors if logits_processors else [],
            prefill_batch_size=1,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
            stream=self._stream,
        )

        return bg

    def _on_prompt_progress(self, updates: list[tuple[int, int, int]]) -> None:
        """Callback from BatchGenerator's prefill loop.

        Called once per prefill chunk (default 2048 tokens) with a list of
        (uid, processed_tokens, total_tokens) tuples.  Updates the global
        PrefillProgressTracker so the admin dashboard can display per-request
        prefill progress.  Only touches CPU counters — zero GPU overhead.
        """
        tracker = get_prefill_tracker()
        # model_name is a full path; use basename to match engine_pool model_id.
        model_id = os.path.basename(self.config.model_name.rstrip("/"))
        for uid, processed, total in updates:
            request_id = self.uid_to_request_id.get(uid)
            if request_id is None:
                continue
            tracker.update(
                request_id=request_id,
                processed=processed,
                total=total,
                model_id=model_id,
            )

    # ------------------------------------------------------------------
    # External prefill (composition pattern — replaces _process_prompts)
    # ------------------------------------------------------------------

    def _model_uses_mla(self) -> bool:
        """Detect Multi-head Latent Attention models (DeepSeek-V2/V3/V4,
        GLM-4-MoE / GLM-4.7-Flash, Kimi-K2, ...).

        MLA compresses K/V into a low-rank latent plus a separate rope key and
        reads the *fetched* cache tensors directly — e.g.
        ``kv_latent, k_pe = cache.update_and_fetch(...)`` then
        ``k_pe.swapaxes(-1, -2)`` (mlx_lm/models/glm4_moe_lite.py). TurboQuant
        replaces the cache state with quantized NamedTuples that have no array
        methods, so that ``.swapaxes`` raises ``AttributeError`` (#1613). MLA
        also stores keys/values with mismatched head dims, which the codec does
        not support. Such models stay fp16 — no crash, no TurboQuant.

        Result is memoized: the model never changes for a scheduler instance.
        """
        cached = getattr(self, "_mla_model", None)
        if cached is not None:
            return cached

        detected = False
        model = getattr(self, "model", None)

        # kv_lora_rank is the defining MLA hyperparameter and is an int on real
        # models. It may sit on the top-level config or be nested under a
        # text/LM sub-config (VLM MLA, e.g. kimi_vl -> text_config). The
        # isinstance(int) check guards against mocks where it is a sentinel.
        def _cfg_has_kv_lora(cfg: Any, depth: int = 0) -> bool:
            if cfg is None or depth > 3:
                return False
            if isinstance(getattr(cfg, "kv_lora_rank", None), int):
                return True
            return any(
                _cfg_has_kv_lora(getattr(cfg, sub, None), depth + 1)
                for sub in (
                    "text_config",
                    "llm_config",
                    "language_config",
                    "thinker_config",
                )
            )

        # Config signal. For VLMs the scheduler sees VLMModelAdapter, whose
        # .args delegates to the language model; also probe (_)language_model.
        for holder in (
            model,
            getattr(model, "_language_model", None),
            getattr(model, "language_model", None),
        ):
            if holder is None:
                continue
            if _cfg_has_kv_lora(getattr(holder, "args", None)) or _cfg_has_kv_lora(
                getattr(holder, "config", None)
            ):
                detected = True
                break

        # Architecture signal: an attention submodule carrying the MLA
        # down-projection, latent layernorm, or latent rank. Covers models
        # whose config does not surface kv_lora_rank where the scheduler can
        # see it (e.g. a directly-loaded VLM with a nested text config).
        if not detected and model is not None and hasattr(model, "modules"):
            try:
                for m in model.modules():
                    if (
                        hasattr(m, "kv_a_proj_with_mqa")
                        or hasattr(m, "kv_a_layernorm")
                        or isinstance(getattr(m, "kv_lora_rank", None), int)
                    ):
                        detected = True
                        break
            except Exception:
                pass

        if detected:
            logger.info(
                "TurboQuant disabled: model uses Multi-head Latent Attention "
                "(MLA), which is incompatible with quantized KV cache states; "
                "keeping fp16 KV cache (#1613)."
            )
        self._mla_model = detected
        return detected

    def _model_uses_attention_sinks(self) -> bool:
        """Detect models whose attention path passes sink logits to SDPA.

        TurboQuant's quantized attention kernels currently do not implement the
        sink term used by attention-sink models. Ignoring it silently changes
        the model's attention distribution, so these models must keep fp16 KV
        unless the attention patch falls back to dequantized sink-aware SDPA.
        """
        cached = getattr(self, "_attention_sink_model", None)
        if cached is not None:
            return cached

        detected = False
        model = getattr(self, "model", None)

        def _has_real_sink_attr(obj: Any) -> bool:
            for name in ("sinks", "attention_sink_bias", "attn_sink"):
                value = None
                if isinstance(obj, dict):
                    value = obj.get(name)
                if value is None:
                    data = getattr(obj, "__dict__", {})
                    if isinstance(data, dict):
                        value = data.get(name)
                if isinstance(value, mx.array):
                    return True
                if value is not None and isinstance(value, (int, float, list, tuple)):
                    return True
            return False

        try:
            modules = getattr(model, "modules", None)
        except Exception:
            modules = None
        if type(modules).__module__.startswith("unittest.mock"):
            modules = None
        if not detected and callable(modules):
            try:
                for m in modules():
                    if _has_real_sink_attr(m):
                        detected = True
                        break
            except Exception:
                pass

        if detected:
            logger.info(
                "TurboQuant disabled: model uses attention sinks, which are "
                "not supported by TurboQuant's quantized attention kernels; "
                "keeping fp16 KV cache."
            )
        self._attention_sink_model = detected
        return detected

    def _turboquant_eligible(self, prompt_cache: list[Any]) -> bool:
        """True if this cache layout can safely mix TQ and pass-through caches.

        Plain KVCache layers are TurboQuant-convertible. State-array caches and
        rotating/sliding-window caches are pass-through: they stay in their
        native form while adjacent full-attention KVCache layers are converted.

        MLA models (DeepSeek / GLM-4.7-Flash) and attention-sink models are
        excluded because their attention paths need semantics TurboQuant's
        quantized cache states/kernels do not currently provide.
        """
        from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

        if self._model_uses_mla():
            return False
        if self._model_uses_attention_sinks():
            return False

        def _ok(c: Any) -> bool:
            if isinstance(c, KVCache):
                return True
            if isinstance(c, ArraysCache):
                return True
            class_name = type(c).__name__
            if class_name in (
                "SizedArraysCache",
                "RotatingKVCache",
                "BatchRotatingKVCache",
                "PrefillReadyRotatingKVCache",
                "BufferedRotatingKVCache",
                "TurboQuantKVCache",
                "BatchTurboQuantKVCache",
            ):
                return True
            if class_name in ("MiniMaxM3KVCache", "MiniMaxM3BatchKVCache"):
                return False
            if isinstance(c, CacheList):
                return all(_ok(inner) for inner in c.caches)
            return False

        return bool(prompt_cache) and all(_ok(c) for c in prompt_cache)

    def _apply_turboquant_kv_empty(self, prompt_cache: list[Any]) -> None:
        """Replace empty KVCache layers with empty TurboQuantKVCache.

        Tokens are quantized on the fly during update_and_fetch, avoiding
        the peak memory spike from storing full-precision KV then converting.
        Used only when there is no prefill history to preserve (the single
        last token is quantized during insert()'s prompt step). Skips the
        last KVCache layer if turboquant_skip_last is set.
        """
        from mlx_lm.models.cache import CacheList, KVCache
        from mlx_vlm.turboquant import TurboQuantKVCache

        kv_indices = [
            i for i, c in enumerate(prompt_cache) if _is_turboquant_kv_family_cache(c)
        ]
        skip_last = self._turboquant_skip_last and len(kv_indices) > 1
        last_kv_idx = kv_indices[-1] if skip_last else -1

        converted = 0
        bits = float(self._turboquant_kv_bits)
        for i, cache_obj in enumerate(prompt_cache):
            if isinstance(cache_obj, KVCache):
                if i == last_kv_idx:
                    continue
                prompt_cache[i] = TurboQuantKVCache(bits=bits)
                converted += 1
            elif isinstance(cache_obj, CacheList):
                new_caches = []
                for c in cache_obj.caches:
                    if isinstance(c, KVCache):
                        new_caches.append(TurboQuantKVCache(bits=bits))
                        converted += 1
                    else:
                        new_caches.append(c)
                cache_obj.caches = tuple(new_caches)
        if converted > 0:
            skip_msg = ", skipped last KVCache layer" if skip_last else ""
            logger.info(
                f"TurboQuant: {converted}/{len(prompt_cache)} "
                f"cache layers set to {bits}-bit{skip_msg}"
            )

    def _apply_turboquant_kv_convert(self, prompt_cache: list[Any]) -> None:
        """Convert populated KVCache data to TurboQuantKVCache via from_cache().

        Called AFTER fp16 prefill completes (or on an SSD-restored fp16
        cache): the completed full-precision KV is quantized once, so prefill
        hidden states stay exact and quantization error only enters at
        decode-time reads. This is the key difference from #717/#771, which
        quantized on the fly during prefill and corrupted hidden states.
        """
        from mlx_lm.models.cache import CacheList, KVCache
        from mlx_vlm.turboquant import TurboQuantKVCache

        kv_indices = [
            i for i, c in enumerate(prompt_cache) if _is_turboquant_kv_family_cache(c)
        ]
        skip_last = self._turboquant_skip_last and len(kv_indices) > 1
        last_kv_idx = kv_indices[-1] if skip_last else -1

        converted = 0
        bits = float(self._turboquant_kv_bits)
        for i, cache_obj in enumerate(prompt_cache):
            if isinstance(cache_obj, KVCache):
                if i == last_kv_idx:
                    continue
                prompt_cache[i] = TurboQuantKVCache.from_cache(cache_obj, bits=bits)
                converted += 1
            elif isinstance(cache_obj, CacheList):
                new_caches = []
                for c in cache_obj.caches:
                    if isinstance(c, KVCache):
                        new_caches.append(TurboQuantKVCache.from_cache(c, bits=bits))
                        converted += 1
                    else:
                        new_caches.append(c)
                cache_obj.caches = tuple(new_caches)
        if converted > 0:
            skip_msg = ", skipped last KVCache layer" if skip_last else ""
            logger.info(
                f"TurboQuant: converted {converted}/{len(prompt_cache)} "
                f"cache layers to {bits}-bit{skip_msg}"
            )

    def _do_external_prefill(
        self,
        request: "Request",
        tokens: list[int],
        existing_cache: list[Any] | None,
        vlm_embeds: tuple[mx.array, dict[str, Any], int] | None = None,
    ) -> tuple[list[Any], list[int]]:
        """Run prefill externally (outside BatchGenerator) for a single request.

        Processes tokens[0:N-1] through the model. The last token tokens[N-1]
        is NOT processed here — it will be passed to BatchGenerator.insert()
        so that the first decode step produces the correct logit.

        Args:
            request: The request being prefilled.
            tokens: Full token list to prefill.
            existing_cache: Restored cache from paged SSD (or None).
            vlm_embeds: Optional (inputs_embeds, extra_kwargs, start_offset)
                tuple for VLM requests.

        Returns:
            (prefilled_cache, last_token_list) where last_token_list contains
            the single last token to pass to insert().

        Raises:
            _PrefillAbortedError: If prefill is interrupted by a pending abort.
            RuntimeError: If memory limit exceeded during prefill.
        """
        n_tokens = len(tokens)
        if n_tokens <= 1:
            # Nothing to prefill, return cache + tokens as-is.
            cache = existing_cache or make_prompt_cache(self.model)
            # TurboQuant: a TQ cache here makes _merge_caches() build a
            # BatchTurboQuantKVCache (via the monkey-patched merge), so the
            # one decode token quantizes against TQ history. An empty fresh
            # cache gets empty TQ layers; a restored cache preserves its data.
            if self._turboquant_kv_bits is not None and self._turboquant_eligible(
                cache
            ):
                if existing_cache is None:
                    self._apply_turboquant_kv_empty(cache)
                else:
                    self._apply_turboquant_kv_convert(cache)
            return cache, tokens

        # Create or reuse cache
        if existing_cache is not None:
            prompt_cache = existing_cache
        else:
            prompt_cache = make_prompt_cache(self.model)

        # Fresh TurboQuant requests run fp16 during the cold prefill loop and
        # are quantized once at the end. Restored TurboQuant prefix caches stay
        # quantized while pre-filling the uncached suffix, then keep using TQ for
        # decode. Rotating/sliding-window layers remain native pass-through
        # caches; only full-attention KVCache layers are converted.

        # Clear stale mRoPE position state for text-only requests.
        if vlm_embeds is None and hasattr(self.model, "clear_vlm_position_state"):
            self.model.clear_vlm_position_state()
            _seed_text_only_mrope_delta_for_cached_prefill(self.model, request)

        # Boundary snapshot setup
        block_size = self.config.paged_cache_block_size
        boundary_enabled = (
            block_size > 0
            and self.block_aware_cache is not None
            and _prompt_cache_needs_snapshots(prompt_cache)
        )
        base_size = _cache_base_sizes(prompt_cache) if boundary_enabled else 0
        # Sanity check: base_size from cache offsets should match the number
        # of tokens actually cached. A mismatch indicates stale meta_state
        # in a restored RotatingKVCache (e.g. shared layer_meta_states from
        # an earlier store_cache bug). Use cached_tokens which is always
        # derived from block_table.num_tokens and therefore trustworthy.
        if (
            boundary_enabled
            and hasattr(request, "cached_tokens")
            and request.cached_tokens > 0
        ):
            if base_size != request.cached_tokens:
                logger.debug(
                    "Cache base_size mismatch: computed %d, expected %d "
                    "(cached_tokens). Using cached_tokens for boundary "
                    "alignment.",
                    base_size,
                    request.cached_tokens,
                )
                base_size = request.cached_tokens

        # Prepare VLM embeddings for prefill
        embeds_array: mx.array | None = None
        extra_kwargs: dict[str, Any] | None = None
        if vlm_embeds is not None:
            embeds_array, extra_kwargs, start_offset = vlm_embeds
            embeds_array = embeds_array[:, start_offset:]  # skip cached portion
            if start_offset > 0 and extra_kwargs:
                extra_kwargs = _advance_vlm_extra(extra_kwargs, start_offset)
            # Force _position_ids path in language model for cached VLM
            # prefill. Without this, the delta approach gives sequential
            # positions to image tokens that need 3D mRoPE positions.
            # Setting _rope_deltas=None makes the language model use
            # _position_ids (set by get_input_embeddings) instead.
            # Saved and restored after prefill for decode rope_deltas capture.
            # Only applies to mRoPE VLMs (Qwen2-VL, Qwen2.5-VL, GLM-4V, etc.);
            # non-mRoPE VLMs like Gemma 4 have no _rope_deltas attribute.
            _saved_rope_deltas = None
            if start_offset > 0:
                lm = getattr(self.model, "_language_model", None)
                if lm is not None and hasattr(lm, "_rope_deltas"):
                    _saved_rope_deltas = lm._rope_deltas
                    lm._rope_deltas = None
            # Stash so the #1405 requeue path can restore it if this prefill
            # raises before the normal restore below runs.
            request._prefill_saved_rope_deltas = _saved_rope_deltas

        # Prefill tokens[0:N-1] (leave last token for insert())
        prefill_tokens = tokens[:-1]
        last_token = tokens[-1:]
        total_length = len(tokens)

        input_arr = mx.array(prefill_tokens)[None]  # (1, seq_len)
        processed_tokens = 0
        uid = self.request_id_to_uid.get(request.request_id)

        emitted_boundaries: dict[int, int] = {}

        while input_arr.shape[1] > 0:
            remaining = input_arr.shape[1]
            prefill_step_size = self._prefill_step_size_for_progress(
                processed_tokens, remaining
            )
            n_to_process = min(prefill_step_size, remaining)

            if processed_tokens == 0:
                _sync_and_clear_cache(self._stream)

            # Boundary-limited step size
            if boundary_enabled and block_size > 0:
                current_total = base_size + processed_tokens
                next_boundary = ((current_total // block_size) + 1) * block_size
                target_boundary_prefill = next_boundary - base_size
                delta = target_boundary_prefill - processed_tokens
                if delta > 0:
                    n_to_process = min(n_to_process, delta)
                n_to_process = max(1, n_to_process)

            # Adaptive throttle: shrink chunk when entering the caution zone
            # so the hard cap is honored before the chunk-end check. Raises
            # RuntimeError if the min chunk would exceed the cap — the
            # #1405 cleanup path catches it and emits an error to the client.
            n_to_process = self._adaptive_chunk_size(
                n_to_process,
                request_id=request.request_id,
                loop_label="external",
                kv_len=base_size + processed_tokens,
            )

            # Pre-chunk safety guard: NEVER submit a chunk whose predicted peak
            # would breach the prefill safety cap. The Metal command-buffer
            # OOM is an async, uncatchable SIGABRT, so it must be prevented
            # before submission — a post-chunk check is too late. Falls back to
            # min_chunk after a reclaim; raises gracefully only if even the
            # floor can't fit (caught by the #1405 path → requeue/clean error).
            n_to_process = self._guard_prefill_chunk(
                n_to_process,
                kv_len=base_size + processed_tokens,
                progress=processed_tokens,
                loop_label="external",
                request_id=request.request_id,
            )

            model_kwargs: dict[str, Any] = {}
            if embeds_array is not None and embeds_array.shape[1] > 0:
                model_kwargs["inputs_embeds"] = embeds_array[:, :n_to_process]
                if extra_kwargs:
                    model_kwargs["vlm_extra_kwargs"] = _slice_vlm_extra(
                        extra_kwargs, n_to_process
                    )

            _throttle_pre = get_phys_footprint()
            self.model(input_arr[:, :n_to_process], cache=prompt_cache, **model_kwargs)
            mx.eval([c.state for c in prompt_cache])
            _throttle_post = get_phys_footprint()
            self._record_chunk_transient(
                n_to_process,
                _throttle_pre,
                _throttle_post,
                request_id=request.request_id,
                loop_label="external",
            )

            input_arr = input_arr[:, n_to_process:]
            if embeds_array is not None:
                embeds_array = embeds_array[:, n_to_process:]
                if extra_kwargs:
                    extra_kwargs = _advance_vlm_extra(extra_kwargs, n_to_process)
            processed_tokens += n_to_process

            # Progress callback
            if uid is not None:
                self._on_prompt_progress([(uid, processed_tokens, total_length)])

            # Boundary snapshot emission
            if boundary_enabled:
                total_tokens = base_size + processed_tokens
                if (
                    total_tokens > 0
                    and total_tokens % block_size == 0
                    and emitted_boundaries.get(request.request_id, -1) < total_tokens
                ):
                    self._emit_prefill_boundary_snapshot(
                        request, prompt_cache, total_tokens
                    )
                    emitted_boundaries[request.request_id] = total_tokens

            # Memory monitoring — use max(active, phys_footprint) so MLX
            # cache pool and IOAccelerator-backed allocations that don't
            # show in mx.get_active_memory() still trigger the guard.
            # See utils/proc_memory.py for why phys_footprint matters.
            if self._memory_limit_bytes > 0:
                current = self._current_usage_bytes()
                _hard = self._memory_hard_limit_bytes
                _soft = self._memory_limit_bytes
                # Only log when crossing the soft watermark — that's the
                # caution zone where adaptive throttle decisions matter.
                # Skipped on healthy traffic to keep the log quiet.
                if current > _soft:
                    logger.debug(
                        "[memcheck:external] rid=%s n=%d processed=%d "
                        "current=%.3fGB soft=%.3fGB hard=%.3fGB %s",
                        request.request_id,
                        n_to_process,
                        processed_tokens,
                        current / 1024**3,
                        _soft / 1024**3,
                        _hard / 1024**3,
                        "OVER_HARD" if _hard > 0 and current > _hard else "OVER_SOFT",
                    )
                # Abort decision uses the STABLE physical cap, not the jittery
                # dynamic ceiling: only kill an in-flight prefill if it would
                # breach what Metal actually allows. Throttling above still
                # targets the dynamic ceiling. Falls back to the dynamic hard
                # limit if the abort limit hasn't been propagated yet.
                _abort = self._memory_abort_limit_bytes or self._memory_hard_limit_bytes
                if _abort > 0 and current > _abort:
                    # Reclaim the just-computed chunk's Metal transients before
                    # giving up — they are still resident at this pre-clear
                    # check and are usually what tipped us over the cap.
                    current = self._reclaim_prefill_headroom()
                    if current > _abort:
                        logger.warning(
                            f"Prefill force-stopped at {processed_tokens} "
                            f"tokens: memory {current / 1024**3:.1f}GB "
                            f"exceeds physical cap "
                            f"{_abort / 1024**3:.1f}GB (after reclaim)"
                        )
                        raise RuntimeError("Memory limit exceeded during prefill")
                    logger.info(
                        "Prefill recovered after reclaim at %d tokens "
                        "(%.1fGB <= cap %.1fGB)",
                        processed_tokens,
                        current / 1024**3,
                        _abort / 1024**3,
                    )
                elif current > self._memory_limit_bytes:
                    logger.warning(
                        f"Prefill above max_bytes at "
                        f"{processed_tokens} tokens: "
                        f"{current / 1024**3:.1f}GB > "
                        f"{self._memory_limit_bytes / 1024**3:.1f}GB "
                        f"(ceiling: "
                        f"{self._memory_hard_limit_bytes / 1024**3:.1f}GB)"
                    )

            # Check for pending aborts between prefill chunks.
            abort_uids = self._check_pending_aborts_for_uids(
                [uid] if uid is not None else []
            )
            if abort_uids:
                logger.info(
                    f"Prefill interrupted at {processed_tokens}/"
                    f"{total_length} tokens: "
                    f"{len(abort_uids)} request(s) aborted"
                )
                if vlm_embeds is not None and _saved_rope_deltas is not None:
                    self.model._language_model._rope_deltas = _saved_rope_deltas
                request._prefill_saved_rope_deltas = None

                # Drop partial-prefill references before clearing the Metal pool.
                # Otherwise the traceback frame can keep large KV/cache arrays
                # alive until after the abort handler returns.
                input_arr = None
                embeds_array = None
                extra_kwargs = None
                model_kwargs = {}
                prompt_cache = None
                _sync_and_clear_cache(self._stream)
                raise _PrefillAbortedError(abort_uids, processed_tokens)

            # Reclaim Metal intermediates between prefill chunks.
            _sync_and_clear_cache(self._stream)

        # Emit final boundary snapshot if prompt lands exactly on boundary.
        if boundary_enabled:
            total_tokens = base_size + processed_tokens
            if (
                total_tokens > 0
                and total_tokens % block_size == 0
                and emitted_boundaries.get(request.request_id, -1) < total_tokens
            ):
                self._emit_prefill_boundary_snapshot(
                    request, prompt_cache, total_tokens
                )

        _sync_and_clear_cache(self._stream)

        # Restore _rope_deltas after cached VLM prefill (for decode capture)
        if vlm_embeds is not None and _saved_rope_deltas is not None:
            self.model._language_model._rope_deltas = _saved_rope_deltas
        request._prefill_saved_rope_deltas = None

        # Quantize the completed fp16 KV cache to TurboQuant for decode.
        # Done here (after the prefill loop, after boundary snapshots are
        # captured fp16) so prefill hidden states stay exact and the paged-SSD
        # format is unchanged. _merge_caches() then builds a
        # BatchTurboQuantKVCache when this request is inserted. Gated to dense
        # KVCache models — chunked/rotating caches stay fp16.
        if self._turboquant_kv_bits is not None and self._turboquant_eligible(
            prompt_cache
        ):
            self._apply_turboquant_kv_convert(prompt_cache)

        if getattr(request, "cached_tokens", 0) > 0:
            with mx.stream(self._stream):
                _materialize_cache_storage(prompt_cache)

        return prompt_cache, last_token

    # ------------------------------------------------------------------
    # Adaptive prefill throttle
    # ------------------------------------------------------------------

    # Discrete step sizes used by the watermark-based throttle. Each tier
    # halves SDPA-fallback transient (∝ query_len × kv_len), so crossing
    # one tier under memory pressure roughly doubles the available
    # headroom for the next chunk's intermediates.
    _PREFILL_STEP_TIERS: tuple[int, ...] = (1024, 512)

    # Safety margin applied to the headroom (hard_cap - current) when sizing
    # a chunk predictively. The remaining 10% absorbs estimator error and
    # Metal command-buffer overhead above the modeled SDPA + KV growth.
    _PREFILL_HEADROOM_SAFETY: float = 0.90

    # Default fraction of the physical abort cap we allow a chunk's predicted
    # PEAK to reach. ProcessMemoryEnforcer can override this per tier. The
    # remaining headroom is reserved for Metal command-buffer overhead: a chunk
    # whose peak lands on the wired limit can make Metal abort the command
    # buffer asynchronously (kIOGPUCommandBufferCallbackError OutOfMemory) —
    # an uncatchable SIGABRT — so we keep a hard margin below it.
    _PREFILL_ABORT_MARGIN: float = 0.90

    # Safety multiplier on the predicted per-chunk transient. The transient
    # scales with query_len * kv_len, so per-token cost grows with context
    # length; this covers one chunk's worth of growth + measurement noise.
    _PREFILL_TRANSIENT_SAFETY: float = 1.3
    _MEMORY_ADMISSION_STALL_TIMEOUT_S: float = 60.0
    _STORE_CACHE_ADMISSION_STALL_TIMEOUT_S: float = 60.0

    def _predicted_chunk_transient(self, n_tokens: int, kv_len: int) -> float:
        """Conservative predicted Metal peak growth for one prefill chunk.

        The per-chunk SDPA/MoE transient scales with ``query_len * kv_len``, so
        the per-token cost GROWS with context length. A long-run EWMA average
        lags that growth and underestimates the next chunk — the cause of the
        Metal command-buffer OOM crash at large kv_len. We therefore take the
        MAX of three signals and apply a safety factor:
          - the most recently MEASURED per-token growth (last_delta / last_n)
            — anchored on reality at the current kv_len regime,
          - the long-run EWMA (model-specific constants the static misses),
          - the kv_len-aware static estimate (SDPA transient + this chunk's
            newly allocated KV).
        Returns 0 only when nothing is known (first chunk, no model info).
        """
        if n_tokens <= 0:
            return 0.0
        per_token = 0.0
        tracker = self._prefill_transient_tracker
        if tracker is not None:
            if tracker.last_n_tokens > 0 and tracker.last_delta_bytes > 0:
                per_token = max(
                    per_token, tracker.last_delta_bytes / tracker.last_n_tokens
                )
            if tracker.bytes_per_token > 0:
                per_token = max(per_token, tracker.bytes_per_token)
        if self.memory_monitor is not None:
            static = self.memory_monitor.estimate_chunk_transient_bytes(
                n_tokens, kv_len + n_tokens
            )
            static += self.memory_monitor.estimate_prompt_kv_bytes(n_tokens)
            per_token = max(per_token, float(static) / n_tokens)
        return per_token * n_tokens * self._PREFILL_TRANSIENT_SAFETY

    def _prefill_abort_cap(self) -> int:
        """Safety cap a chunk's predicted peak must stay under.

        Uses the stable abort limit (min(static, metal_cap)) with a margin so
        we never submit a chunk that could trip the async Metal OOM. Falls back
        to the dynamic hard limit before the abort limit is propagated.
        """
        cap = self._memory_abort_limit_bytes or self._memory_hard_limit_bytes
        return int(cap * self._prefill_abort_margin) if cap > 0 else 0

    def _prefill_abort_description(self) -> tuple[int, int, float]:
        """Return (base cap, safety cap, margin) for diagnostics."""
        base_cap = self._memory_abort_limit_bytes or self._memory_hard_limit_bytes
        safety_cap = self._prefill_abort_cap()
        return base_cap, safety_cap, self._prefill_abort_margin

    _MAX_PREFILL_EVICTION_RETRIES = 1

    def _raise_prefill_eviction_if_available(
        self,
        *,
        request_id: str,
        current: int,
        target_cap: int,
        predicted_transient: int,
        requested_tokens: int,
        reason: str,
    ) -> None:
        """Pause a request once so EngineCore can evict idle LRU models."""
        request = self.requests.get(request_id)
        if request is None:
            return
        max_retries = getattr(
            self,
            "_MAX_PREFILL_EVICTION_RETRIES",
            Scheduler._MAX_PREFILL_EVICTION_RETRIES,
        )
        if request.prefill_eviction_retries >= max_retries:
            return
        if target_cap <= 0 or predicted_transient <= 0:
            return

        request.prefill_eviction_retries += 1
        config = getattr(self, "config", None)
        eviction_request = PrefillEvictionRequest(
            request_id=request_id,
            model_id=getattr(config, "model_name", ""),
            current_bytes=int(current),
            target_cap_bytes=int(target_cap),
            predicted_transient_bytes=int(predicted_transient),
            requested_tokens=int(requested_tokens),
            reason=reason,
        )
        logger.info(
            "Request %s needs prefill headroom before throttling "
            "(reason=%s, current=%.2fGB, predicted=%.2fGB, target=%.2fGB)",
            request_id,
            reason,
            current / 1024**3,
            predicted_transient / 1024**3,
            target_cap / 1024**3,
        )
        raise _PrefillEvictionNeeded(eviction_request)

    def _guard_prefill_chunk(
        self,
        n_tokens: int,
        *,
        kv_len: int,
        progress: int,
        loop_label: str,
        request_id: str | None = None,
    ) -> int:
        """Clamp/abort a prefill chunk so its predicted peak can never reach
        the physical Metal cap (the uncatchable async OOM crash).

        Returns a chunk size whose predicted peak fits under the margined cap
        (possibly shrunk from ``n_tokens``). If even the minimum chunk would
        not fit after a reclaim, raises a clean RuntimeError — the context is
        genuinely too large for available memory. That message intentionally
        does NOT contain "Memory limit exceeded", so ``_requeue_or_fail_prefill``
        fails it fast with a clear error rather than looping a doomed retry.
        """
        base_cap, cap, margin = self._prefill_abort_description()
        if cap <= 0:
            return n_tokens
        min_chunk = max(1, self._prefill_min_chunk_tokens)
        current = self._current_usage_bytes()
        if current + self._predicted_chunk_transient(n_tokens, kv_len) <= cap:
            return n_tokens

        # Predicted to breach — reclaim transients and re-measure once.
        current = self._reclaim_prefill_headroom()
        min_transient = self._predicted_chunk_transient(min_chunk, kv_len)
        if current + min_transient > cap:
            maybe_raise_eviction = getattr(
                self, "_raise_prefill_eviction_if_available", None
            )
            if request_id is not None and callable(maybe_raise_eviction):
                maybe_raise_eviction(
                    request_id=request_id,
                    current=current,
                    target_cap=cap,
                    predicted_transient=int(min_transient),
                    requested_tokens=min_chunk,
                    reason="prefill_safety_cap",
                )
            logger.warning(
                "[guard:%s] context too large at progress=%d kv_len=%d: "
                "%.2fGB + min-chunk transient exceeds prefill safety cap "
                "%.2fGB (%d%% of effective ceiling %.2fGB)",
                loop_label,
                progress,
                kv_len,
                current / 1024**3,
                cap / 1024**3,
                round(margin * 100),
                base_cap / 1024**3,
            )
            message = (
                "Prefill context too large for available memory "
                f"(pre-chunk guard at {progress} tokens, kv_len={kv_len}): "
                "predicted peak would exceed prefill safety cap "
                f"{cap / 1024**3:.1f}GB "
                f"({round(margin * 100)}% of effective ceiling "
                f"{base_cap / 1024**3:.1f}GB)"
            )
            raise PrefillMemoryExceededError(
                message=message,
                request_id=request_id,
                estimated_bytes=int(current + min_transient),
                limit_bytes=int(cap),
            )

        # The floor fits — pick the largest chunk that still fits under the cap.
        per_token = self._predicted_chunk_transient(n_tokens, kv_len) / n_tokens
        safe_n = int((cap - current) / per_token) if per_token > 0 else n_tokens
        n_fit = max(min_chunk, min(n_tokens, safe_n))
        if n_fit < n_tokens:
            logger.debug(
                "[guard:%s] shrink %d -> %d at progress=%d kv_len=%d "
                "(current=%.2fGB cap=%.2fGB)",
                loop_label,
                n_tokens,
                n_fit,
                progress,
                kv_len,
                current / 1024**3,
                cap / 1024**3,
            )
        return n_fit

    def _adaptive_chunk_size(
        self,
        requested: int,
        *,
        request_id: str,
        loop_label: str,
        kv_len: int = 0,
    ) -> int:
        """Size the next prefill chunk so its predicted peak stays under a
        safety margin below the hard cap.

        The chunk is sized so that ``current + predicted_transient(n) <=
        hard_cap * safety``. If the full requested chunk already fits, it runs
        unchanged — no behavior change on healthy traffic. Crucially the gate
        is on the *predicted peak*, not on current memory crossing the soft
        watermark: a single large chunk's transient (e.g. MoE prefill at tens
        of MB/token) can blow the ceiling from a low baseline before current
        ever reaches the watermark, which is the failure this prevents.

        Two predictors feed the sizing:
          - Measured: once the per-scheduler EWMA has samples, use its
            ``bytes_per_token`` (× the same 1.2 safety factor ``predict()``
            applies) — this is measurement-based and model-agnostic.
          - First chunk (no samples yet): fall back to the static SDPA + KV
            growth estimate for the requested candidate chunk. ``kv_len`` is
            the current context span (cached prefix + already-prefilled
            tokens), so a large prefix-cache hit with a small suffix is
            throttled correctly without classifying large prefill chunks as
            vector-path traffic.

        The discrete watermark tiers are retained as a *secondary clamp* —
        they only ever shrink further, never enlarge the predicted size.

        The chunk-end memory check (``self._memory_hard_limit_bytes``
        comparison in the prefill loops) remains the safety net: if memory
        still exceeds the cap after this shrink, the loop attempts reclaim
        (``_reclaim_prefill_headroom``) and, failing that, raises so the
        #1405 cleanup path can requeue or emit ``finish_reason="error"``.

        Args:
            requested: The chunk size the caller would have used without
                throttle (already clamped by boundary alignment).
            request_id: For debug log correlation.
            loop_label: "external" or "chunked_step", used only for debug
                log identification.
            kv_len: Current context span (base/cached + processed tokens)
                used for the first-chunk static peak-growth estimate.

        Returns:
            The chunk size to actually process (>= 1, <= requested).
        """
        soft_base = self._memory_limit_bytes
        hard_cap = self._memory_hard_limit_bytes
        if soft_base <= 0 or hard_cap <= 0 or requested <= 0:
            return requested

        current = self._current_usage_bytes()
        min_chunk = max(1, self._prefill_min_chunk_tokens)

        # Conservative per-token peak growth (measured-last / EWMA / static, ×
        # safety) — see _predicted_chunk_transient. Anchored on the most recent
        # measurement so it tracks growth with kv_len instead of lagging behind
        # a long-run average.
        per_token = self._predicted_chunk_transient(requested, kv_len) / requested
        predictor = "measured" if per_token > 0 else "none"

        # Keep each chunk's predicted peak under the LOWER of the dynamic
        # throttle target and the prefill safety cap, so the peak can never
        # reach the Metal wall (the uncatchable async OOM).
        headroom_safety = getattr(
            self, "_prefill_headroom_safety", self._PREFILL_HEADROOM_SAFETY
        )
        safe_target = int(hard_cap * headroom_safety)
        abort_cap = self._prefill_abort_cap()
        target = min(safe_target, abort_cap) if abort_cap > 0 else safe_target
        soft_watermark = int(soft_base * self._prefill_safe_zone_ratio)

        if per_token <= 0:
            # No usable predictor (e.g. model info unavailable). Fall back to
            # the legacy watermark gate so we never run unbounded.
            if current < soft_watermark:
                return requested
            n_fit = requested
        else:
            # Predicted-peak gate: if the FULL requested chunk fits under the
            # target it runs unchanged (covers all healthy traffic). Gated on
            # the predicted peak, not on current crossing the soft watermark —
            # a single big chunk's transient can blow the cap from a low
            # baseline (MoE prefill at tens of MB/token), the failure this
            # prevents.
            if current + per_token * requested <= target:
                return requested
            maybe_raise_eviction = getattr(
                self, "_raise_prefill_eviction_if_available", None
            )
            if callable(maybe_raise_eviction):
                maybe_raise_eviction(
                    request_id=request_id,
                    current=current,
                    target_cap=target,
                    predicted_transient=int(per_token * requested),
                    requested_tokens=requested,
                    reason="adaptive_prefill_throttle",
                )
            headroom = max(target - current, 0)
            n_fit = int(headroom / per_token)

        n = max(min_chunk, min(requested, n_fit))

        # Secondary clamp: once in the watermark caution zone, cap by the
        # discrete tiers so a mispredicting EWMA can't run an oversized chunk
        # in deep pressure. Skipped below the watermark so a low-baseline chunk
        # with ample headroom isn't needlessly shrunk.
        band_ratio = -1.0
        if current >= soft_watermark and hard_cap > soft_watermark:
            band = hard_cap - soft_watermark
            band_ratio = max(0.0, min(1.0, (current - soft_watermark) / band))
            if band_ratio < 0.50:
                bucket = self._PREFILL_STEP_TIERS[0]  # 1024
            else:
                bucket = self._PREFILL_STEP_TIERS[1]  # 512
            n = max(min_chunk, min(n, bucket))

        if n < requested:
            logger.debug(
                "[throttle:%s] shrink rid=%s chunk %d -> %d "
                "(predictor=%s per_token=%.1fKB current=%.2fGB "
                "safe_target=%.2fGB ceiling=%.2fGB kv_len=%d band_ratio=%.2f)",
                loop_label,
                request_id,
                requested,
                n,
                predictor,
                per_token / 1024,
                current / 1024**3,
                safe_target / 1024**3,
                hard_cap / 1024**3,
                kv_len,
                band_ratio,
            )
        return n

    def get_cached_mlx_active_memory_bytes(self) -> int:
        """Return the last MLX active-memory sample taken on the executor."""
        return self._last_mlx_active_memory_bytes

    def _hot_cache_cpu_bytes(self) -> int:
        """Return serialized hot-cache bytes safe to exclude from phys guard."""
        config = getattr(self, "config", None)
        budget = getattr(config, "hot_cache_budget", None)
        if budget is not None:
            try:
                return max(0, int(getattr(budget, "total_bytes", 0)))
            except Exception:
                logger.debug("Failed to read shared hot-cache byte budget")
                return 0

        manager = getattr(self, "paged_ssd_cache_manager", None)
        if manager is None:
            return 0

        try:
            stats = manager.get_stats()
            return max(0, int(getattr(stats, "hot_cache_size_bytes", 0)))
        except Exception:
            try:
                return max(0, int(getattr(manager, "_hot_cache_total_bytes", 0)))
            except Exception:
                logger.debug("Failed to read local hot-cache byte counter")
                return 0

    def _current_usage_bytes(self, *, refresh_mlx_active: bool = True) -> int:
        """Current memory usage for scheduler-side guard checks.

        Scheduler steps run on the MLX executor thread, so they can refresh
        mx.get_active_memory() safely. Event-loop callers such as early
        preflight use the cached executor sample and phys_footprint instead.
        """
        active = self._last_mlx_active_memory_bytes
        if refresh_mlx_active:
            active = max(0, int(mx.get_active_memory()))
            self._last_mlx_active_memory_bytes = active
        hot_cache_cpu_bytes = getattr(self, "_hot_cache_cpu_bytes", None)
        if callable(hot_cache_cpu_bytes):
            hot_cache_bytes = hot_cache_cpu_bytes()
        else:
            hot_cache_bytes = Scheduler._hot_cache_cpu_bytes(self)
        phys = max(0, int(get_phys_footprint()) - hot_cache_bytes)
        return max(active, phys)

    def get_active_hot_cache_block_hashes(self) -> set[bytes]:
        """Return hot-cache block hashes owned by active in-flight requests."""
        manager = getattr(self, "paged_cache_manager", None)
        if manager is None:
            return set()

        hashes: set[bytes] = set()
        active_requests = list(self.running.values()) + list(self.prefilling)
        for request in active_requests:
            block_table = getattr(request, "block_table", None)
            if block_table is None:
                continue
            for block_id in getattr(block_table, "block_ids", []) or []:
                try:
                    block = manager.blocks[block_id]
                    block_hash = getattr(block, "block_hash", None)
                except Exception:
                    continue
                if block_hash is not None:
                    hashes.add(bytes(block_hash))
        return hashes

    def _clear_memory_admission_blocker(self, request_id: str | None = None) -> None:
        if (
            request_id is not None
            and request_id != self._memory_admission_blocked_request_id
        ):
            return
        self._memory_admission_blocked_request_id = None
        self._memory_admission_blocked_since = 0.0

    def _clear_store_cache_admission_blocker(
        self, request_id: str | None = None
    ) -> None:
        if (
            request_id is not None
            and request_id != self._store_cache_admission_blocked_request_id
        ):
            return
        self._store_cache_admission_blocked_request_id = None
        self._store_cache_admission_blocked_since = 0.0

    def _clear_request_admission_bookkeeping(self, request_id: str) -> None:
        self._cache_freshness_waits.pop(request_id, None)
        self._prefix_cache_prepared.discard(request_id)
        self._clear_memory_admission_blocker(request_id)
        self._clear_store_cache_admission_blocker(request_id)

    def _memory_admission_stall_output(self, reason: str) -> RequestOutput | None:
        """Fail one head-of-line request after persistent memory admission stall."""
        if not self.waiting:
            self._clear_memory_admission_blocker()
            return None

        request = self.waiting[0]
        request_id = request.request_id
        now = time.monotonic()
        if request_id != self._memory_admission_blocked_request_id:
            self._memory_admission_blocked_request_id = request_id
            self._memory_admission_blocked_since = now
            return None

        timeout = getattr(
            self,
            "_MEMORY_ADMISSION_STALL_TIMEOUT_S",
            Scheduler._MEMORY_ADMISSION_STALL_TIMEOUT_S,
        )
        if now - self._memory_admission_blocked_since < timeout:
            return None

        stalled_for = now - self._memory_admission_blocked_since
        self.waiting.popleft()
        self._release_paged_cache_for_request(request_id)
        self.requests.pop(request_id, None)
        self._clear_request_admission_bookkeeping(request_id)
        get_prefill_tracker().remove(request_id)
        self._clear_memory_admission_blocker(request_id)

        message = (
            "Request could not be admitted because memory pressure persisted "
            f"for {stalled_for:.1f}s ({reason}). Reduce context length, free "
            "memory, lower hot_cache_max_size, or loosen memory_guard_tier."
        )
        logger.warning("Memory admission stalled for %s: %s", request_id, message)
        return RequestOutput(
            request_id=request_id,
            finished=True,
            finish_reason="error",
            error=message,
            error_code="memory_admission_stalled",
            error_metadata={
                "request_id": request_id,
                "reason": reason,
                "stalled_seconds": int(stalled_for),
            },
        )

    def _store_cache_admission_stall_output(
        self,
        reason: str,
        *,
        gate_in_flight: int,
        gate_cap: int,
        pending_cleanups: int,
    ) -> RequestOutput | None:
        """Fail one head-of-line request after persistent store-cache stall."""
        if not self.waiting:
            self._clear_store_cache_admission_blocker()
            return None

        request = self.waiting[0]
        request_id = request.request_id
        now = time.monotonic()
        if request_id != self._store_cache_admission_blocked_request_id:
            self._store_cache_admission_blocked_request_id = request_id
            self._store_cache_admission_blocked_since = now
            return None

        timeout = getattr(
            self,
            "_STORE_CACHE_ADMISSION_STALL_TIMEOUT_S",
            Scheduler._STORE_CACHE_ADMISSION_STALL_TIMEOUT_S,
        )
        if now - self._store_cache_admission_blocked_since < timeout:
            return None

        stalled_for = now - self._store_cache_admission_blocked_since
        self.waiting.popleft()
        self._release_paged_cache_for_request(request_id)
        self.requests.pop(request_id, None)
        self._clear_request_admission_bookkeeping(request_id)
        get_prefill_tracker().remove(request_id)

        message = (
            "Request could not be admitted because store-cache cleanup stayed "
            f"full for {stalled_for:.1f}s ({reason}). The previous response "
            "cache is still being persisted; retry after the cache writer drains "
            "or reduce cache/write pressure."
        )
        logger.warning(
            "Store-cache admission stalled for %s: %s "
            "(in_flight=%d pending_cleanups=%d cap=%d)",
            request_id,
            message,
            gate_in_flight,
            pending_cleanups,
            gate_cap,
        )
        return RequestOutput(
            request_id=request_id,
            finished=True,
            finish_reason="error",
            error=message,
            error_code="store_cache_admission_stalled",
            error_metadata={
                "request_id": request_id,
                "reason": reason,
                "stalled_seconds": int(stalled_for),
                "store_cache_in_flight": gate_in_flight,
                "pending_store_cleanups": pending_cleanups,
                "store_cache_cap": gate_cap,
            },
        )

    def _bypass_hot_cache_under_pressure(self) -> bool:
        """Return True when SSD-backed hot-cache acceleration should be bypassed."""
        if not self._prefill_memory_guard:
            return False
        if self._memory_limit_bytes <= 0:
            return False
        config = getattr(self, "config", None)
        if config is None:
            return False
        if getattr(config, "hot_cache_only", False):
            return False
        if int(getattr(config, "hot_cache_max_size", 0) or 0) <= 0:
            return False
        if getattr(self, "paged_ssd_cache_manager", None) is None:
            return False
        try:
            current = self._current_usage_bytes()
        except Exception:
            logger.debug("Failed to sample memory for hot-cache pressure bypass")
            return False
        return current >= self._memory_limit_bytes

    def _record_chunk_transient(
        self,
        n_tokens: int,
        pre_bytes: int,
        post_bytes: int,
        *,
        request_id: str,
        loop_label: str,
    ) -> None:
        """Feed one chunk's measured transient into the EWMA tracker."""
        delta = post_bytes - pre_bytes
        min_chunk = max(1, self._prefill_min_chunk_tokens)
        if n_tokens < min_chunk:
            logger.debug(
                "[throttle:%s] measure rid=%s n=%d delta=%.2fMB "
                "(skipped: tail < min_chunk=%d)",
                loop_label,
                request_id,
                n_tokens,
                delta / 1024**2,
                min_chunk,
            )
            return
        if delta <= 0:
            logger.debug(
                "[throttle:%s] measure rid=%s n=%d delta=%dB (skipped: <=0)",
                loop_label,
                request_id,
                n_tokens,
                delta,
            )
            return
        self._prefill_transient_tracker.update(n_tokens, delta)
        logger.debug(
            "[throttle:%s] measure rid=%s n=%d transient=%.2fMB "
            "per_token=%.1fKB ewma=%.1fKB samples=%d",
            loop_label,
            request_id,
            n_tokens,
            delta / 1024**2,
            (delta / max(n_tokens, 1)) / 1024,
            self._prefill_transient_tracker.bytes_per_token / 1024,
            self._prefill_transient_tracker.samples,
        )

    def _reclaim_prefill_headroom(self) -> int:
        """Reclaim Metal headroom mid-prefill and return the re-measured usage.

        The prefill loops measure the hard-limit at the chunk boundary, which
        is *before* the per-chunk ``_sync_and_clear_cache`` runs — so the
        just-completed forward pass's SDPA intermediates are still resident
        when the limit is checked. Synchronizing and clearing the Metal buffer
        cache here releases those transients, which is exactly the spike that
        drives prefill OOM (observed: 42.8GB at the check → 24.6GB after the
        buffers are reclaimed). This is the only lever that actually lowers
        the physical footprint: paged-cache block eviction merely recycles
        ``CacheBlock`` metadata back into the free queue (the pool never
        shrinks, see ``PagedCacheManager._grow_blocks``), so it is deliberately
        not attempted here — it would drop reusable prefix-cache entries for no
        memory benefit.

        Returns:
            ``max(active, phys_footprint)`` after reclaim.
        """
        _sync_and_clear_cache(self._stream)
        return self._current_usage_bytes()

    # ------------------------------------------------------------------
    # Chunked prefill helpers (used when config.chunked_prefill=True)
    # ------------------------------------------------------------------

    def _prefill_step_size_for_progress(
        self, processed_tokens: int, remaining_tokens: int
    ) -> int:
        """Return the scheduler prefill chunk size for the current progress."""
        adaptive_prefill = self._glm_dsa_adaptive_prefill
        if adaptive_prefill is not None:
            from .patches.glm_moe_dsa.generate_patch import (
                _prefill_step_size_for_progress,
            )

            return _prefill_step_size_for_progress(
                self.config.prefill_step_size,
                processed_tokens,
                remaining_tokens,
                adaptive_prefill,
            )

        adaptive_prefill = getattr(self, "_minimax_m3_adaptive_prefill", None)
        if adaptive_prefill is None:
            return self.config.prefill_step_size
        from .patches.minimax_m3.generate_patch import (
            _prefill_step_size_for_progress as _minimax_prefill_step_size,
        )

        return _minimax_prefill_step_size(
            self.config.prefill_step_size,
            processed_tokens,
            remaining_tokens,
            adaptive_prefill,
        )

    def _begin_prefill(
        self,
        request: "Request",
        tokens: list[int],
        existing_cache: "list[Any] | None",
    ) -> _PrefillState:
        """Initialise a _PrefillState for a non-VLM request.

        Performs all once-per-request setup (cache creation, boundary config,
        token splitting) without running any model forward passes.
        """
        if hasattr(self.model, "clear_vlm_position_state"):
            self.model.clear_vlm_position_state()
            _seed_text_only_mrope_delta_for_cached_prefill(self.model, request)

        prompt_cache = (
            existing_cache
            if existing_cache is not None
            else make_prompt_cache(self.model)
        )

        block_size = self.config.paged_cache_block_size
        boundary_enabled = (
            block_size > 0
            and self.block_aware_cache is not None
            and _prompt_cache_needs_snapshots(prompt_cache)
        )
        base_size = _cache_base_sizes(prompt_cache) if boundary_enabled else 0
        if (
            boundary_enabled
            and hasattr(request, "cached_tokens")
            and request.cached_tokens > 0
            and base_size != request.cached_tokens
        ):
            logger.debug(
                "Cache base_size mismatch: computed %d, expected %d "
                "(cached_tokens). Using cached_tokens for boundary alignment.",
                base_size,
                request.cached_tokens,
            )
            base_size = request.cached_tokens

        prefill_tokens = tokens[:-1]
        last_token = tokens[-1:]
        input_arr = mx.array(prefill_tokens)[None]  # (1, N-1)

        return _PrefillState(
            request=request,
            cache=prompt_cache,
            tokens_remaining=input_arr,
            last_token=last_token,
            tokens_processed=0,
            base_size=base_size,
            emitted_boundaries={},
            boundary_enabled=boundary_enabled,
            block_size=block_size,
            total_length=len(tokens),
        )

    def _step_prefill_chunk(self, state: _PrefillState) -> bool:
        """Process one prefill chunk from *state*.

        Runs the model on at most prefill_step_size tokens, evals the cache,
        emits any due boundary snapshot, updates the prefill progress
        tracker, and clears Metal intermediates.

        Returns:
            True when all tokens_remaining have been consumed (prefill done).

        Raises:
            RuntimeError: If the hard memory limit is exceeded.
        """
        if state.tokens_remaining.shape[1] == 0:
            return True

        remaining = state.tokens_remaining.shape[1]
        prefill_step_size = self._prefill_step_size_for_progress(
            state.tokens_processed, remaining
        )
        n = min(prefill_step_size, remaining)

        if state.tokens_processed == 0:
            _sync_and_clear_cache(self._stream)

        # Clamp to the next block boundary so boundary snapshots fire exactly.
        if state.boundary_enabled and state.block_size > 0:
            current_total = state.base_size + state.tokens_processed
            next_boundary = ((current_total // state.block_size) + 1) * state.block_size
            delta = (next_boundary - state.base_size) - state.tokens_processed
            if delta > 0:
                n = min(n, delta)
            n = max(1, n)

        # Adaptive throttle — see _adaptive_chunk_size docstring. Raises
        # if even prefill_min_chunk_tokens would exceed the cap; #1405
        # cleanup paths in _schedule_waiting / _advance_chunked_prefills
        # convert that into a finish_reason="error" output for the client.
        n = self._adaptive_chunk_size(
            n,
            request_id=state.request.request_id,
            loop_label="chunked_step",
            kv_len=state.base_size + state.tokens_processed,
        )

        # Pre-chunk safety guard (mirrors the external loop): never submit a
        # chunk whose predicted peak would trip the uncatchable async Metal OOM.
        n = self._guard_prefill_chunk(
            n,
            kv_len=state.base_size + state.tokens_processed,
            progress=state.tokens_processed,
            loop_label="chunked_step",
            request_id=state.request.request_id,
        )

        chunk = state.tokens_remaining[:, :n]
        state.tokens_remaining = state.tokens_remaining[:, n:]
        _throttle_pre = get_phys_footprint()
        self.model(chunk, cache=state.cache)
        mx.eval([c.state for c in state.cache])
        _throttle_post = get_phys_footprint()
        self._record_chunk_transient(
            n,
            _throttle_pre,
            _throttle_post,
            request_id=state.request.request_id,
            loop_label="chunked_step",
        )
        state.tokens_processed += n

        # Boundary snapshot
        if state.boundary_enabled:
            total_tokens = state.base_size + state.tokens_processed
            rid = state.request.request_id
            if (
                total_tokens > 0
                and total_tokens % state.block_size == 0
                and state.emitted_boundaries.get(rid, -1) < total_tokens
            ):
                self._emit_prefill_boundary_snapshot(
                    state.request, state.cache, total_tokens
                )
                state.emitted_boundaries[rid] = total_tokens

        # Progress callback so the admin UI prefilling list advances during
        # chunked prefill. _do_external_prefill calls _on_prompt_progress
        # via the temp_uid mapping; the chunked path has no temp uid so we
        # talk to the tracker directly with the request_id.
        get_prefill_tracker().update(
            state.request.request_id,
            state.tokens_processed,
            state.total_length - 1,
            (
                os.path.basename(self.config.model_name.rstrip("/"))
                if self.config.model_name
                else ""
            ),
        )

        # Memory monitoring — use max(active, phys_footprint) so MLX cache
        # pool and IOAccelerator-backed allocations that don't show up in
        # mx.get_active_memory() still trigger the guard. Matches the
        # _do_external_prefill check; on macOS jetsam watches
        # phys_footprint, so the active-only check could miss the page
        # before the kernel kills us.
        if self._memory_limit_bytes > 0:
            current = self._current_usage_bytes()
            _hard = self._memory_hard_limit_bytes
            _soft = self._memory_limit_bytes
            # Caution-zone-only memcheck log (see external loop counterpart).
            if current > _soft:
                logger.debug(
                    "[memcheck:chunked_step] rid=%s n=%d processed=%d/%d "
                    "current=%.3fGB soft=%.3fGB hard=%.3fGB %s",
                    state.request.request_id,
                    n,
                    state.tokens_processed,
                    state.total_length - 1,
                    current / 1024**3,
                    _soft / 1024**3,
                    _hard / 1024**3,
                    "OVER_HARD" if _hard > 0 and current > _hard else "OVER_SOFT",
                )
            # Abort on the stable physical cap, not the jittery dynamic ceiling
            # (mirrors the external prefill loop).
            _abort = self._memory_abort_limit_bytes or self._memory_hard_limit_bytes
            if _abort > 0 and current > _abort:
                # Reclaim the just-computed chunk's Metal transients before
                # giving up (mirrors the external prefill loop).
                current = self._reclaim_prefill_headroom()
                if current > _abort:
                    raise RuntimeError(
                        f"Memory limit exceeded during chunked prefill at "
                        f"{state.tokens_processed}/{state.total_length - 1} tokens: "
                        f"{current / 1024**3:.1f}GB exceeds physical cap "
                        f"{_abort / 1024**3:.1f}GB (after reclaim)"
                    )
                logger.info(
                    "Chunked prefill recovered after reclaim at %d/%d tokens "
                    "(%.1fGB <= cap %.1fGB)",
                    state.tokens_processed,
                    state.total_length - 1,
                    current / 1024**3,
                    _abort / 1024**3,
                )
            elif current > self._memory_limit_bytes:
                logger.warning(
                    f"Chunked prefill above max_bytes at "
                    f"{state.tokens_processed} tokens: "
                    f"{current / 1024**3:.1f}GB > "
                    f"{self._memory_limit_bytes / 1024**3:.1f}GB "
                    f"(ceiling: "
                    f"{self._memory_hard_limit_bytes / 1024**3:.1f}GB)"
                )

        _sync_and_clear_cache(self._stream)
        return state.tokens_remaining.shape[1] == 0

    def _emit_final_boundary_if_needed(self, state: _PrefillState) -> None:
        """Emit a final boundary snapshot if the prefill landed on a boundary."""
        if not state.boundary_enabled:
            return
        total_tokens = state.base_size + state.tokens_processed
        rid = state.request.request_id
        if (
            total_tokens > 0
            and total_tokens % state.block_size == 0
            and state.emitted_boundaries.get(rid, -1) < total_tokens
        ):
            self._emit_prefill_boundary_snapshot(
                state.request, state.cache, total_tokens
            )

    def _finalize_chunked_prefill_cache_for_insert(
        self, request: "Request", prompt_cache: list[Any] | None
    ) -> None:
        """Mirror external prefill's post-prefill cache epilogue."""
        if not prompt_cache or self._turboquant_kv_bits is None:
            return
        if not self._turboquant_eligible(prompt_cache):
            return

        self._apply_turboquant_kv_convert(prompt_cache)
        if getattr(request, "cached_tokens", 0) > 0:
            with mx.stream(self._stream):
                _materialize_cache_storage(prompt_cache)
        _sync_and_clear_cache(self._stream)

    def _insert_prefilled_request(
        self,
        request: "Request",
        state: _PrefillState,
        scheduled: "list[Request]",
    ) -> None:
        """Insert a fully-prefilled request into BatchGenerator.

        Handles the batch_generator.insert() call, uid bookkeeping, and moving
        the request to self.running. Called from both the inline chunked path
        (first chunk completed immediately) and _advance_chunked_prefills()
        (last chunk completed across steps).

        Precondition: state.sampler, state.sm, state.per_row_lps are set.
        """
        self._finalize_chunked_prefill_cache_for_insert(request, state.cache)

        if request.sampling_params.seed is not None:
            mx.random.seed(request.sampling_params.seed)

        per_row_lps = state.per_row_lps if state.per_row_lps is not None else []
        uids = self.batch_generator.insert(
            [state.last_token],
            max_tokens=[request.sampling_params.max_tokens],
            caches=[state.cache] if state.cache else None,
            all_tokens=[_batch_generator_all_tokens(request)],
            samplers=[state.sampler],
            logits_processors=[per_row_lps],
            state_machines=[state.sm],
        )
        if uids:
            _register_uid_rows(self.model, uids, [state.sampler], [per_row_lps])
            uid = uids[0]
            self.request_id_to_uid[request.request_id] = uid
            self.uid_to_request_id[uid] = request.request_id
            now = time.monotonic()
            request.batch_uid = uid
            request.status = RequestStatus.RUNNING
            request.generation_started_at = now
            request.last_activity_at = now
            self.running[request.request_id] = request
            scheduled.append(request)

            if hasattr(self.model, "register_rope_delta"):
                self.model.register_rope_delta(uid, request.rope_deltas)

            self.total_prompt_tokens += request.num_prompt_tokens
            cache_info = (
                f", {request.cached_tokens} cached" if request.cached_tokens > 0 else ""
            )
            logger.debug(
                "Scheduled chunked-prefill request %s (uid=%d) "
                "with %d tokens (%d total)%s",
                request.request_id,
                uid,
                len(state.last_token),
                request.num_prompt_tokens,
                cache_info,
            )

    def _advance_chunked_prefills(
        self,
        scheduled: "list[Request]",
        rejected: "list[RequestOutput]",
    ) -> None:
        """Process one prefill chunk per in-flight chunked-prefill request.

        Called at the start of each step() before _schedule_waiting(). Each
        call advances every request in self.prefilling by one prefill_step_size
        chunk. When a request's prefill completes it is inserted into
        BatchGenerator and moved to self.running.

        Args:
            scheduled: The step's running list of newly-scheduled requests;
                completed chunked-prefill requests are appended here.
            rejected: Per-step rejected outputs. A chunked prefill that hits
                the memory hard limit emits a finish_reason="error" entry
                here so the engine can surface the failure to the client.
        """
        if not self.prefilling:
            return

        pending_prefills = list(self.prefilling)
        still_prefilling: deque[Request] = deque()

        for index, request in enumerate(pending_prefills):
            rid = request.request_id
            state = self._prefill_states.get(rid)

            # State missing means the request was aborted and cleaned up by
            # _do_abort_request() between steps — just skip it.
            if state is None:
                continue

            try:
                done = self._step_prefill_chunk(state)
            except _PrefillAbortedError:
                # Request aborted mid-chunk. Discard state; the abort will
                # be fully processed by _process_pending_aborts() next step.
                self._prefill_states.pop(rid, None)
                _sync_and_clear_cache(self._stream)
                continue
            except _PrefillEvictionNeeded as e:
                self._pending_prefill_eviction_request = e.request
                still_prefilling.append(request)
                still_prefilling.extend(pending_prefills[index + 1 :])
                logger.info(
                    "Paused chunked prefill request %s for LRU eviction " "(reason=%s)",
                    rid,
                    e.request.reason,
                )
                break
            except PrefillMemoryExceededError as e:
                logger.error("Chunked prefill capacity rejected for %s: %s", rid, e)
                self._prefill_states.pop(rid, None)
                self._release_paged_cache_for_request(rid)
                self.requests.pop(rid, None)
                self._clear_request_admission_bookkeeping(rid)
                get_prefill_tracker().remove(rid)
                _sync_and_clear_cache()
                rejected.append(_prefill_memory_exception_output(rid, e))
                continue
            except RuntimeError as e:
                logger.error("Chunked prefill failed for %s: %s", rid, e)
                self._prefill_states.pop(rid, None)
                self._release_paged_cache_for_request(rid)
                self.requests.pop(rid, None)
                self._clear_request_admission_bookkeeping(rid)
                get_prefill_tracker().remove(rid)
                # Drop Metal cache pool buffers held by the aborted chunk's
                # forward / mx.eval transients. Without this, enforcer keeps
                # seeing the burst footprint until the next mx.clear_cache().
                _sync_and_clear_cache()
                # Try a bounded requeue before surfacing the failure: a
                # memory-pressure prefill gets a fresh, better-throttled
                # attempt. Only after the retry budget is exhausted (or for
                # non-memory errors) do we emit the client-facing error.
                if self._requeue_or_fail_prefill(request, e):
                    continue
                # Surface the failure to the engine. Without this, the
                # request is silently dropped and the client hangs.
                rejected.append(
                    RequestOutput(
                        request_id=rid,
                        finished=True,
                        finish_reason="error",
                        error=str(e),
                    )
                )
                continue

            if not done:
                still_prefilling.append(request)
                continue

            # Prefill complete — emit final boundary snapshot and insert.
            self._prefill_states.pop(rid, None)
            self._emit_final_boundary_if_needed(state)
            _sync_and_clear_cache(self._stream)

            # Ensure a BatchGenerator exists (may not if all requests were
            # previously in chunked prefill with no running decode).
            self._ensure_batch_generator(request.sampling_params)
            if self.batch_generator is None:
                # Unlikely, but if BG creation fails put request back.
                logger.error(
                    "BatchGenerator unavailable at chunked-prefill completion "
                    "for %s; requeueing.",
                    rid,
                )
                still_prefilling.append(request)
                self._prefill_states[rid] = state
                continue

            # Clean up the prefill-progress tracker entry.
            get_prefill_tracker().remove(rid)

            self._insert_prefilled_request(request, state, scheduled)

        self.prefilling = still_prefilling

    def _build_state_machine(self, request: "Request") -> SequenceStateMachine:
        """Build a SequenceStateMachine for per-request stop tokens.

        Combines base stop tokens (EOS, Harmony) with request-specific
        stop_token_ids and tokenized stop strings into a single state
        machine that tells BatchGenerator when to stop generating for
        this request.
        """
        stop_tokens_set = self._get_stop_tokens()
        if request.sampling_params.stop_token_ids:
            stop_tokens_set.update(request.sampling_params.stop_token_ids)

        transitions: dict[str, list] = {
            "normal": [([t], None) for t in stop_tokens_set]
        }

        # Tokenize stop strings into token sequences. mlx-lm's
        # SequenceStateMachine uses Aho-Corasick, so per-token match
        # cost stays O(1) regardless of how many sequences are added.
        # BPE merge edge cases (where a stop string boundary lands
        # mid-token) may miss; that is a known limitation.
        for stop_str in request.sampling_params.stop or []:
            if not isinstance(stop_str, str) or not stop_str:
                continue
            try:
                seq = self.tokenizer.encode(stop_str, add_special_tokens=False)
            except TypeError:
                seq = self.tokenizer.encode(stop_str)
            if seq:
                transitions["normal"].append((list(seq), None))

        if transitions["normal"]:
            return SequenceStateMachine(transitions, initial="normal")
        return SequenceStateMachine({}, initial="normal")

    def _emit_prefill_boundary_snapshot(
        self,
        request: "Request",
        prompt_cache: list[Any],
        total_tokens: int,
    ) -> None:
        """Capture boundary snapshot from individual (non-batch) cache.

        During external prefill we have direct access to per-layer cache
        objects (not BatchKVCache). Extract non-sliceable layers for
        boundary snapshot storage.

        Pass ``request_id`` directly. The request is mid-prefill and has
        not been inserted into ``BatchGenerator`` yet, so
        ``request_id_to_uid`` has no entry for it. The earlier shape
        routed through ``self.request_id_to_uid.get(request_id, -1)`` →
        ``uid_to_request_id.get(-1)`` → ``None`` → silent return,
        dropping every snapshot. For ArraysCache / GDN / hybrid models
        that made every non-last block store a placeholder, and the
        next identical-prompt request rejected the cache and re-
        prefilled from scratch.
        """
        snapshot_cache = [
            c if type(c).__name__ not in _KNOWN_SLICEABLE_CACHE_TYPES else None
            for c in prompt_cache
        ]
        self._on_prefill_boundary_snapshot(
            request.request_id,
            snapshot_cache,
            total_tokens,
        )

    def _build_sampler_and_processors(
        self, sampling_params: SamplingParams, request: Any = None
    ) -> tuple[Callable[[mx.array], mx.array], list[Callable]]:
        """Build per-request sampler and logits processors."""
        # Use omlx.utils.sampling.make_sampler instead of mlx_lm.sample_utils.
        # The mlx-lm version decorates categorical_sampling and apply_* with
        # @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state),
        # which fails to advance the RNG state after the first call in this
        # server environment. Identical prompts then produce identical output
        # even at temperature > 1.
        sampler = omlx_make_sampler(
            temp=sampling_params.temperature,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
            top_k=sampling_params.top_k,
            xtc_probability=sampling_params.xtc_probability,
            xtc_threshold=sampling_params.xtc_threshold,
            xtc_special_tokens=self._xtc_special_tokens,
        )
        logits_processors = make_logits_processors(
            repetition_penalty=(
                sampling_params.repetition_penalty
                if sampling_params.repetition_penalty != 1.0
                else None
            ),
            presence_penalty=(
                sampling_params.presence_penalty
                if sampling_params.presence_penalty != 0.0
                else None
            ),
            frequency_penalty=(
                sampling_params.frequency_penalty
                if sampling_params.frequency_penalty != 0.0
                else None
            ),
        )

        suppress_processor = _make_suppress_logits_processor(
            self._model_suppress_tokens
        )
        if suppress_processor is not None:
            logits_processors.append(suppress_processor)

        # Add thinking budget processor for reasoning models
        if (
            sampling_params.thinking_budget is not None
            and request is not None
            and (
                getattr(request, "needs_think_prefix", False)
                or self._get_output_parser_thinking_end_text() is not None
            )
        ):
            think_end_ids = self._resolve_think_end_token_ids()
            if think_end_ids:
                from .api.thinking import ThinkingBudgetProcessor

                think_start_id = self._get_think_token_id("think_start_id")
                leading_ids, trailing_ids = self._resolve_think_close_pattern(
                    self._get_output_parser_thinking_end_text()
                )
                parser_trailing_ids = (
                    self._resolve_output_parser_thinking_trailing_ids()
                )
                if parser_trailing_ids is not None:
                    trailing_ids = parser_trailing_ids
                processor = ThinkingBudgetProcessor(
                    think_end_token_ids=think_end_ids,
                    budget=sampling_params.thinking_budget,
                    think_start_token_id=think_start_id,
                    leading_token_ids=leading_ids,
                    trailing_token_ids=trailing_ids,
                    token_to_piece=self._thinking_budget_token_to_piece,
                )
                logits_processors.append(processor)

        # Add grammar constraint processor for structured output.
        # Phase awareness (thinking vs output) is handled by the compiled
        # grammar itself via xgrammar structural tags, so we don't need
        # think_end_ids here.
        if sampling_params.compiled_grammar is not None:
            try:
                from .api.grammar import GrammarConstraintProcessor

                vocab_size = self._get_model_vocab_size()
                if vocab_size is not None:
                    processor = GrammarConstraintProcessor(
                        compiled_grammar=sampling_params.compiled_grammar,
                        vocab_size=vocab_size,
                    )
                    logits_processors.append(processor)
                else:
                    logger.warning(
                        "Cannot determine vocab_size; skipping grammar constraint"
                    )
            except ImportError:
                logger.warning("xgrammar not installed; skipping grammar constraint")

        return sampler, logits_processors

    def _get_model_vocab_size(self) -> int | None:
        """Return vocab_size from model config, or None if unavailable."""
        from .utils.tokenizer import resolve_vocab_size

        return resolve_vocab_size(self.model)

    def _get_think_token_id(self, attr: str) -> int | None:
        """Safely read a think token id from the tokenizer.

        mlx-lm tokenizers expose ``think_start_id`` / ``think_end_id`` as
        properties that may raise ``ValueError`` (multi-token sequence) or
        ``TypeError`` (``_think_start_tokens`` is ``None`` for models without
        thinking support, e.g. context-1 / harmony parser).

        Returns the token id, or ``None`` when unavailable.
        """
        try:
            return getattr(self.tokenizer, attr, None)
        except (ValueError, TypeError):
            return None

    def _get_output_parser_thinking_end_text(self) -> str | None:
        """Return parser-provided thinking close text, if the parser has one."""
        factory = getattr(self, "_output_parser_factory", None)
        if factory is None:
            return None
        return getattr(factory, "thinking_end_text", None)

    def _get_output_parser_thinking_start_text(self) -> str | None:
        """Return parser-provided thinking open text, if the parser has one."""
        factory = getattr(self, "_output_parser_factory", None)
        if factory is None:
            return None
        return getattr(factory, "thinking_start_text", None)

    def _get_output_parser_thinking_start_output_text(self) -> str | None:
        """Return normalized text to prepend when parser thinking starts in prompt."""
        factory = getattr(self, "_output_parser_factory", None)
        if factory is None:
            return None
        return getattr(factory, "thinking_start_output_text", None)

    def _encode_thinking_marker(self, text: str) -> list[int] | None:
        """Encode a parser/tokenizer thinking marker into token IDs."""
        try:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            try:
                ids = self.tokenizer.encode(text)
            except Exception:
                return None
        except Exception:
            return None

        if ids:
            return list(ids)
        return None

    def _thinking_budget_token_to_piece(self, token_id: int) -> str | bytes | None:
        """Best-effort token piece lookup for UTF-8-safe budget forcing."""
        try:
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            if token is not None:
                byte_piece = self._token_piece_to_bytes(token)
                return byte_piece if byte_piece is not None else token
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

        try:
            return self.tokenizer.decode([token_id], skip_special_tokens=False)
        except TypeError:
            try:
                return self.tokenizer.decode([token_id])
            except Exception:
                return None
        except Exception:
            return None

    def _token_piece_to_bytes(self, token: str) -> bytes | None:
        """Convert byte-fallback tokenizer pieces to raw bytes when possible."""
        import re

        byte_fallback = re.fullmatch(r"(?:<0x[0-9A-Fa-f]{2}>)+", token)
        if byte_fallback is not None:
            return bytes(
                int(match.group(1), 16)
                for match in re.finditer(r"<0x([0-9A-Fa-f]{2})>", token)
            )

        byte_decoder = getattr(self.tokenizer, "byte_decoder", None)
        if isinstance(byte_decoder, dict) and token:
            try:
                return bytes(byte_decoder[ch] for ch in token)
            except (KeyError, TypeError, ValueError):
                pass

        return None

    def _resolve_output_parser_thinking_trailing_ids(self) -> list[int] | None:
        """Resolve parser-provided tokens that should follow a forced close."""
        factory = getattr(self, "_output_parser_factory", None)
        if factory is None:
            return None

        trailing_text = getattr(factory, "thinking_end_trailing_text", None)
        if not trailing_text:
            return None

        return self._encode_thinking_marker(trailing_text)

    def _resolve_think_end_token_ids(self) -> list[int] | None:
        """Resolve token ID(s) for the close-think tag.

        Uses mlx-lm's built-in think_end_id which supports both
        </think> and </longcat_think> automatically.
        """
        parser_think_end = self._get_output_parser_thinking_end_text()
        if parser_think_end is not None:
            return self._encode_thinking_marker(parser_think_end)

        # Tier 1: mlx-lm tokenizer attribute (covers all known think variants)
        think_end_id = self._get_think_token_id("think_end_id")
        if think_end_id is not None:
            return [think_end_id]

        # Tier 2: encode the think_end string
        think_end_str = getattr(self.tokenizer, "think_end", "</think>")
        try:
            ids = self.tokenizer.encode(think_end_str, add_special_tokens=False)
            if ids:
                return list(ids)
        except Exception:
            pass

        # Tier 3: direct token lookup
        try:
            tid = self.tokenizer.convert_tokens_to_ids("</think>")
            if tid != getattr(self.tokenizer, "unk_token_id", None):
                return [tid]
        except (AttributeError, KeyError, TypeError):
            pass

        return None

    def _resolve_think_close_pattern(
        self, think_end_str: str | None = None
    ) -> tuple[list[int] | None, list[int] | None]:
        """Detect leading/trailing tokens around </think> from the chat template.

        Different models use different patterns:
        - Qwen3/3.5, MiniMax: ``\\n</think>\\n\\n``
        - DeepSeek V3.2, GLM-5: ``</think>`` (no newlines)
        - GLM-4.6V: ``</think>\\n``
        - Step-3.5-Flash: ``\\n</think>\\n``

        Returns (leading_token_ids, trailing_token_ids) or (None, None).
        """
        import re

        if think_end_str is None:
            think_end_str = getattr(self.tokenizer, "think_end", None) or "</think>"

        # Try to get the chat template text
        template_text = self._get_chat_template_text()
        if not template_text:
            return None, None

        # Find the close pattern in the template, e.g. \n</think>\n\n
        # Look for the think_end_str surrounded by whitespace/newlines in string literals
        escaped = re.escape(think_end_str)
        # Match patterns like: \n</think>\n\n or </think> in template strings
        match = re.search(
            r"(\\n|\\r|[\n\r])*" + escaped + r"((?:\\n|\\r|[\n\r])*)",
            template_text,
        )
        if not match:
            return None, None

        # Extract raw leading/trailing whitespace, converting \n escapes to actual newlines
        raw_leading = (
            match.group(0)
            .split(think_end_str)[0]
            .replace("\\n", "\n")
            .replace("\\r", "\r")
        )
        raw_trailing = (
            match.group(0)
            .split(think_end_str)[1]
            .replace("\\n", "\n")
            .replace("\\r", "\r")
        )

        # Encode to token IDs
        leading_ids = None
        trailing_ids = None
        if raw_leading:
            try:
                ids = self.tokenizer.encode(raw_leading, add_special_tokens=False)
                if ids:
                    leading_ids = list(ids)
            except Exception:
                pass
        if raw_trailing:
            try:
                ids = self.tokenizer.encode(raw_trailing, add_special_tokens=False)
                if ids:
                    trailing_ids = list(ids)
            except Exception:
                pass

        return leading_ids, trailing_ids

    def _get_chat_template_text(self) -> str | None:
        """Get chat template text from the tokenizer or model directory."""
        # Try tokenizer's chat_template attribute (Jinja string)
        ct = getattr(self.tokenizer, "_chat_template", None)
        if ct:
            return ct if isinstance(ct, str) else str(ct)
        ct = getattr(self.tokenizer, "chat_template", None)
        if ct:
            return ct if isinstance(ct, str) else str(ct)

        # Try reading the .jinja file from model directory
        import os

        model_path = getattr(self.config, "model_name", None) or ""
        jinja_path = os.path.join(model_path, "chat_template.jinja")
        if os.path.isfile(jinja_path):
            try:
                with open(jinja_path, encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass

        return None

    def _detect_needs_think_prefix(self, request: "Request") -> bool:
        """Detect if prompt ends with an open <think> tag (thinking enabled).

        Returns False for disabled-thinking patterns like <think></think>
        where </think> immediately follows <think> in the prompt tail.
        """
        think_start_ids = None
        think_start_id = self._get_think_token_id("think_start_id")
        if think_start_id is not None:
            think_start_ids = [think_start_id]
        else:
            think_start_text = (
                self._get_output_parser_thinking_start_text() or "<think>"
            )
            try:
                token_id = self.tokenizer.convert_tokens_to_ids(think_start_text)
                if isinstance(token_id, int) and token_id != getattr(
                    self.tokenizer, "unk_token_id", None
                ):
                    think_start_ids = [token_id]
            except (AttributeError, KeyError, TypeError):
                think_start_ids = None

            if think_start_ids is None:
                think_start_ids = self._encode_thinking_marker(think_start_text)

        if not think_start_ids or not request.prompt_token_ids:
            return False

        lookback = max(3, len(think_start_ids) + 2)
        last_tokens = list(request.prompt_token_ids[-lookback:])
        last_idx = None
        for idx in range(len(last_tokens) - len(think_start_ids), -1, -1):
            if last_tokens[idx : idx + len(think_start_ids)] == think_start_ids:
                last_idx = idx
                break
        if last_idx is None:
            return False

        # <think> found. Check if </think> follows it (disabled thinking pattern).
        after_start = last_tokens[last_idx + len(think_start_ids) :]

        if after_start:
            think_end_ids = self._resolve_think_end_token_ids()
            if think_end_ids and len(after_start) >= len(think_end_ids):
                for idx in range(len(after_start) - len(think_end_ids) + 1):
                    if after_start[idx : idx + len(think_end_ids)] == think_end_ids:
                        return False
            elif think_end_ids and think_end_ids[0] in after_start:
                return False

        return True

    def _ensure_batch_generator(self, sampling_params: SamplingParams) -> None:
        """Ensure BatchGenerator exists with compatible settings."""
        # Only create once; per-request samplers are passed at insert time.
        if self.batch_generator is None:
            self.batch_generator = self._create_batch_generator(sampling_params)

        # Track latest params for debugging/metrics.
        self._current_sampler_params = (
            sampling_params.temperature,
            sampling_params.top_p,
            sampling_params.min_p,
            sampling_params.top_k,
            sampling_params.repetition_penalty,
        )

    def _cache_tree_has_stateful_non_sliceable(self, cache_obj: Any) -> bool:
        """Detect non-sliceable recurrent cache layers requiring snapshots."""
        # None placeholders from boundary snapshots (sliceable layers replaced).
        if cache_obj is None:
            return False

        # CacheList nests multiple cache objects.
        sub_caches = getattr(cache_obj, "caches", None)
        if isinstance(sub_caches, (list, tuple)):
            return any(
                self._cache_tree_has_stateful_non_sliceable(sub_cache)
                for sub_cache in sub_caches
            )

        class_name = type(cache_obj).__name__

        # Known sliceable cache types — no boundary snapshots needed.
        if class_name in (
            "KVCache",
            "BatchKVCache",
            "QuantizedKVCache",
        ):
            return False

        # Stateful non-sliceable caches require boundary-safe snapshots.
        is_rotating_cache = class_name in (
            "RotatingKVCache",
            "BatchRotatingKVCache",
        )
        if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
            is_rotating_cache = (
                is_rotating_cache or CacheTypeRegistry.is_rotating_family(class_name)
            )

        if is_rotating_cache or class_name in ("ArraysCache", "SizedArraysCache"):
            return True

        if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
            handler = CacheTypeRegistry.get_handler_by_class_name(class_name)
            if not handler.supports_block_slicing:
                return True

        # Best-effort fallback for unknown recurrent cache structures.
        state_list = getattr(cache_obj, "cache", None)
        if isinstance(state_list, list):
            return True

        return False

    def _cache_list_needs_boundary_snapshot(self, cache_list: list[Any]) -> bool:
        """Return True if any layer cache requires boundary snapshots."""
        if not cache_list:
            return False
        return any(
            self._cache_tree_has_stateful_non_sliceable(layer_cache)
            for layer_cache in cache_list
        )

    def _eval_snapshot_cache(self, snapshot_cache: list[Any]) -> None:
        """Force the leaf KV tensors of an in-memory boundary snapshot concrete.

        Runs on the capturing (owner/inference) thread. The store-cache worker
        later re-extracts and slices these via _BoundarySnapshotProvider; MLX
        streams are thread-local, so any leaf op still lazy at that point would
        re-dispatch to this thread's stream index, which does not exist on the
        worker -> SIGABRT. Extracting + eval'ing the leaves here (under this
        thread's stream, where that stream lives) makes the worker's slicing
        operate exclusively on already-materialized buffers, which bind their
        new ops to the always-present default stream (gpu,0).
        """
        if not snapshot_cache:
            return
        extracted, _ = self._extract_cache_states(snapshot_cache)
        leaves = self._collect_arrays_from_extracted_cache(extracted)
        if leaves:
            with mx.stream(self._stream):
                mx.eval(*leaves)

    def _on_prefill_boundary_snapshot(
        self,
        request_id: str,
        snapshot_cache: list[Any],
        token_count: int,
    ) -> None:
        """Record boundary snapshots captured during prefill processing.

        Called from ``_emit_prefill_boundary_snapshot`` at each block
        boundary crossed during prefill. Keyed by ``request_id`` rather
        than ``uid`` because the request has not been inserted into
        ``BatchGenerator`` yet and the uid mapping does not exist —
        routing through it dropped every snapshot silently (#TBD).
        """
        if self.block_aware_cache is None:
            return

        block_size = self.config.paged_cache_block_size
        if block_size <= 0 or token_count <= 0 or token_count % block_size != 0:
            return

        if not self._cache_list_needs_boundary_snapshot(snapshot_cache):
            return

        if request_id not in self._boundary_cache_snapshots:
            self._boundary_cache_snapshots[request_id] = {}

        # Skip if we already have a snapshot at this token count
        if token_count in self._boundary_cache_snapshots[request_id]:
            return

        # Offload snapshot to SSD if store is available, keeping only a
        # None marker in the dict.  Falls back to in-memory storage when
        # the SSD store is unavailable or the write fails.
        if self._boundary_snapshot_store is not None:
            saved = self._boundary_snapshot_store.save(
                request_id,
                token_count,
                snapshot_cache,
                self._extract_cache_states,
            )
            if saved:
                self._boundary_cache_snapshots[request_id][token_count] = None
            else:
                # In-memory fallback: this snapshot will be sliced later on the
                # store-cache worker thread (via _BoundarySnapshotProvider ->
                # _extract_cache_states). MLX streams are thread-local, so the
                # worker cannot materialize a lazy op bound to THIS (owner) thread's
                # stream. Force it concrete now, on the capturing thread, so the
                # worker only ever slices already-evaluated buffers.
                self._eval_snapshot_cache(snapshot_cache)
                self._boundary_cache_snapshots[request_id][token_count] = snapshot_cache
        else:
            self._eval_snapshot_cache(snapshot_cache)
            self._boundary_cache_snapshots[request_id][token_count] = snapshot_cache

        self._boundary_snapshot_required = True
        logger.debug(
            "Captured prefill boundary cache snapshot for %s at %s tokens",
            request_id,
            token_count,
        )

    def _detect_boundary_snapshot_need(self) -> bool:
        """
        Determine whether boundary snapshots are needed for the current model.

        Evaluated lazily by inspecting model.make_cache() output instead of
        the active batch (which no longer exists in the new API).
        """
        if self._boundary_snapshot_required is not None:
            return self._boundary_snapshot_required

        if not hasattr(self.model, "make_cache"):
            self._boundary_snapshot_required = False
            return False

        try:
            cache_list = self.model.make_cache()
        except Exception:
            self._boundary_snapshot_required = False
            return False

        if not cache_list:
            self._boundary_snapshot_required = False
            return False

        try:
            self._boundary_snapshot_required = any(
                self._cache_tree_has_stateful_non_sliceable(layer_cache)
                for layer_cache in cache_list
            )
        except TypeError:
            # make_cache() returned something non-iterable (stub models in
            # tests); treat as not snapshot-needing.
            self._boundary_snapshot_required = False
            return False

        if self._boundary_snapshot_required:
            logger.info(
                "Enabled boundary cache snapshots for stateful non-sliceable "
                "cache layers"
            )
            # Ask the speculative (MTP) decode path to land its commits on
            # block boundaries: its emit queue otherwise leaves the cache a
            # few tokens ahead of the emitted count when a boundary token
            # surfaces, which forces the consistency guard in
            # _extract_boundary_snapshot to skip most captures.
            block = int(self.config.paged_cache_block_size or 0)
            if block > 0:
                try:
                    self.model._omlx_mtp_commit_align = block
                except Exception:
                    pass
        else:
            logger.debug(
                "Boundary cache snapshots disabled (no stateful non-sliceable "
                "cache layers detected)"
            )

        return self._boundary_snapshot_required

    def _extract_boundary_snapshot(
        self, uid: int, expected_tokens: int | None = None
    ) -> list[Any] | None:
        """Extract a per-request prompt cache snapshot via extract_cache().

        Uses BatchGenerator.extract_cache() which returns
        Dict[uid, (cache_list, tokens_list)].

        ``expected_tokens`` guards positional consistency: a snapshot labeled
        "state at N tokens" must be extracted while the cache holds exactly N
        forwarded tokens. The standard decode step always satisfies this at
        emit time, but speculative (MTP) decode advances the cache in bursts
        and emits from a queue, so the cache can be a few tokens ahead of —
        or one behind — the emitted count when the boundary token surfaces.
        A skewed snapshot would pair block-aligned KV with recurrent (SSM)
        state from a different position and corrupt later prefix-cache hits
        on hybrid models; skipping the capture merely costs a reuse
        opportunity.
        """
        if self.batch_generator is None:
            return None

        try:
            # Synchronize pending engine stream operations before
            # accessing batch cache tensors.
            with self._phase_timer("boundary_capture_sync"):
                _safe_sync_stream(self._stream)
            with self._phase_timer("boundary_capture_extract"):
                with mx.stream(self._stream):
                    result = self.batch_generator.extract_cache([uid])
                    if uid not in result:
                        return None
                    cache_list, _tokens = result[uid]
                    if expected_tokens is not None:
                        # Walk into CacheList wrappers: DeepSeek-V4/GLM layer
                        # caches carry their token offset on a sub-cache, and
                        # checking only the wrapper would silently pass a
                        # skewed capture through.
                        for c in cache_list:
                            offset = _first_leaf_cache_offset(c)
                            if offset is None:
                                continue
                            if offset != expected_tokens:
                                logger.debug(
                                    "Skipping boundary snapshot for uid=%s: "
                                    "cache offset %d != boundary %d "
                                    "(speculative decode skew)",
                                    uid,
                                    offset,
                                    expected_tokens,
                                )
                                return None
                            break
                    # Only extract non-sliceable layers to avoid costly
                    # deep-copy accumulation (same rationale as prefill path).
                    return [
                        (
                            c
                            if type(c).__name__ not in _KNOWN_SLICEABLE_CACHE_TYPES
                            else None
                        )
                        for c in cache_list
                    ]
        except Exception as e:
            logger.debug(
                f"Failed to extract boundary cache snapshot for uid={uid}: {e}"
            )
            return None

    def _maybe_capture_boundary_snapshot(self, request: Request, uid: int) -> None:
        """Capture cache snapshot exactly at block boundaries for safe reuse."""
        if self.block_aware_cache is None:
            return

        block_size = self.config.paged_cache_block_size
        if block_size <= 0:
            return

        total_tokens = request.num_tokens
        if total_tokens <= 0 or total_tokens % block_size != 0:
            return

        if not self._detect_boundary_snapshot_need():
            return

        snapshot_cache = self._extract_boundary_snapshot(
            uid, expected_tokens=total_tokens
        )
        if not snapshot_cache:
            return

        if request.request_id not in self._boundary_cache_snapshots:
            self._boundary_cache_snapshots[request.request_id] = {}

        # Offload to SSD with in-memory fallback.
        if self._boundary_snapshot_store is not None:
            with self._phase_timer("boundary_snapshot_save"):
                saved = self._boundary_snapshot_store.save(
                    request.request_id,
                    total_tokens,
                    snapshot_cache,
                    self._extract_cache_states,
                )
            if saved:
                self._boundary_cache_snapshots[request.request_id][total_tokens] = None
            else:
                # In-memory fallback: the store-cache worker slices this snapshot
                # off-thread (via _BoundarySnapshotProvider -> _extract_cache_states).
                # MLX streams are thread-local, so force the leaves concrete now on
                # the capturing thread; otherwise the worker re-dispatches a lazy op
                # to this thread's stream index -> "no Stream(gpu, N)" -> SIGABRT.
                # Mirrors _on_prefill_boundary_snapshot's in-memory fallback.
                self._eval_snapshot_cache(snapshot_cache)
                self._boundary_cache_snapshots[request.request_id][
                    total_tokens
                ] = snapshot_cache
        else:
            self._eval_snapshot_cache(snapshot_cache)
            self._boundary_cache_snapshots[request.request_id][
                total_tokens
            ] = snapshot_cache

        logger.debug(
            f"Captured boundary cache snapshot for {request.request_id} at "
            f"{total_tokens} tokens"
        )

    def _get_boundary_store_override(
        self,
        request_id: str,
        full_token_sequence: list[int],
    ) -> (
        tuple[
            list[int],
            list[dict[str, Any]],
            Optional["ModelCacheConfig"],
            dict[int, list[dict[str, Any]]],
        ]
        | None
    ):
        """
        Return boundary-aligned cache payload when final request ends on partial block.

        Returns:
            Tuple of (truncated_tokens, extracted_cache, model_cache_config,
            intermediate_snapshots) where intermediate_snapshots maps
            token_count -> extracted cache states for per-block storage.
        """
        snapshots = self._boundary_cache_snapshots.get(request_id)
        if not snapshots:
            return None

        total_tokens = len(full_token_sequence)
        block_size = self.config.paged_cache_block_size

        # Find all valid boundary-aligned snapshot token counts
        valid_counts = sorted(
            tc
            for tc in snapshots.keys()
            if 0 < tc <= total_tokens and tc % block_size == 0
        )
        if not valid_counts:
            return None

        # Find the latest snapshot that leaves trailing partial tokens
        # (or equals total if it's block-aligned).
        latest_tc = valid_counts[-1]
        if latest_tc < total_tokens:
            # Trailing partial tokens exist — use this snapshot for truncation
            pass
        elif latest_tc == total_tokens and total_tokens % block_size == 0:
            # Exactly block-aligned — no truncation needed but we still
            # provide intermediate snapshots for per-block storage.
            latest_tc = total_tokens
        else:
            return None

        # Load latest snapshot — may be on SSD (None marker) or in memory.
        #
        # In-memory snapshots are raw mlx-lm cache objects. They must be
        # converted to the extracted dict format here on the engine MLX thread.
        # The async store-cache worker does not own the generation stream; if it
        # touches raw Rotating/Arrays cache state, MLX can abort with
        # "There is no Stream(gpu, X) in current thread" (#1568).
        latest_snapshot = snapshots[latest_tc]
        if latest_snapshot is None and self._boundary_snapshot_store is not None:
            # Offloaded to SSD — load back.
            extracted_cache = self._boundary_snapshot_store.load(request_id, latest_tc)
            if not extracted_cache:
                return None
            # Build model_cache_config from the main request cache config
            # since the SSD snapshot doesn't carry it.
            model_cache_config = getattr(
                self.requests.get(request_id), "_model_cache_config", None
            )
        elif latest_snapshot is not None:
            extracted_cache, model_cache_config = self._extract_cache_states(
                latest_snapshot
            )
            if not extracted_cache:
                return None
        else:
            return None

        # Build provider for intermediate snapshots. SSD-backed snapshots remain
        # lazy-loaded, but in-memory snapshots are extracted eagerly on this
        # engine thread before the provider is handed to the async worker.
        intermediate_tcs = [tc for tc in valid_counts if tc != latest_tc]
        provider_tcs: list[int] = []
        extracted_in_memory: dict[int, list[dict[str, Any]]] = {}
        for tc in intermediate_tcs:
            snap = snapshots.get(tc)
            if snap is None:
                if self._boundary_snapshot_store is not None:
                    provider_tcs.append(tc)
                continue

            extracted_snapshot, _ = self._extract_cache_states(snap)
            if extracted_snapshot:
                extracted_in_memory[tc] = extracted_snapshot
                provider_tcs.append(tc)

        intermediate_snapshots = _BoundarySnapshotProvider(
            store=self._boundary_snapshot_store,
            request_id=request_id,
            valid_tcs=provider_tcs,
            in_memory_snapshots=extracted_in_memory,
        )

        token_sequence = (
            full_token_sequence[:latest_tc]
            if latest_tc < total_tokens
            else full_token_sequence
        )

        return (
            token_sequence,
            extracted_cache,
            model_cache_config,
            intermediate_snapshots,
        )

    @staticmethod
    def _merge_boundary_with_full_cache(
        boundary_cache: list[dict[str, Any]],
        full_cache: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fill placeholder layers in boundary cache from full extracted cache.

        Boundary snapshots skip sliceable (KVCache) layers to save memory,
        leaving them as ``{'state': (), ...}`` placeholders.  For block
        storage the KV tensors are needed, so we copy them from the full
        extracted cache (which contains the complete sequence).
        """
        if not full_cache or len(boundary_cache) != len(full_cache):
            return boundary_cache

        merged = []
        for bc, fc in zip(boundary_cache, full_cache):
            state = bc.get("state", ())
            # Placeholder layers have state == () (empty tuple).
            if isinstance(state, tuple) and len(state) == 0:
                # Take full cache layer instead.
                merged.append(fc)
            else:
                merged.append(bc)
        return merged

    @staticmethod
    def _is_empty_boundary_placeholder(layer_state: Any) -> bool:
        if not isinstance(layer_state, dict):
            return False
        state = layer_state.get("state", ())
        return isinstance(state, tuple) and len(state) == 0

    @staticmethod
    def _extracted_layer_type_name(layer_state: dict[str, Any]) -> str:
        class_name = str(layer_state.get("class_name") or "")
        if class_name:
            return class_name
        return str(layer_state.get("cache_type") or "")

    @staticmethod
    def _is_sliceable_extracted_layer(layer_state: dict[str, Any]) -> bool:
        type_name = Scheduler._extracted_layer_type_name(layer_state)
        if type_name in _KNOWN_SLICEABLE_CACHE_TYPES:
            return True

        if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None and type_name:
            try:
                if type_name not in CacheTypeRegistry.list_known_class_names():
                    return False
                handler = CacheTypeRegistry.get_handler_by_class_name(type_name)
                return bool(handler.supports_block_slicing)
            except Exception:
                return False

        return False

    def _fill_boundary_placeholders_from_live_cache(
        self,
        boundary_cache: list[dict[str, Any]],
        live_cache: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        """Fill boundary placeholders only with proven sliceable live layers."""
        if (
            not boundary_cache
            or not live_cache
            or len(boundary_cache) != len(live_cache)
        ):
            return None

        merged: list[dict[str, Any]] = []
        for layer_idx, (boundary_layer, live_layer) in enumerate(
            zip(boundary_cache, live_cache)
        ):
            if not self._is_empty_boundary_placeholder(boundary_layer):
                merged.append(boundary_layer)
                continue

            if not isinstance(
                live_layer, dict
            ) or not self._is_sliceable_extracted_layer(live_layer):
                logger.debug(
                    "Cannot fill boundary placeholder for layer %s from non-sliceable "
                    "live cache type %s",
                    layer_idx,
                    (
                        self._extracted_layer_type_name(live_layer)
                        if isinstance(live_layer, dict)
                        else type(live_layer).__name__
                    ),
                )
                return None

            merged.append(live_layer)

        if any(self._is_empty_boundary_placeholder(layer) for layer in merged):
            return None
        return merged

    def _extract_live_request_cache_for_store(
        self,
        request_id: str,
        uid: int,
        expected_tokens: list[int],
    ) -> tuple[list[dict[str, Any]], Optional["ModelCacheConfig"]] | None:
        """Extract live cache only when its token prefix matches exactly."""
        if self.batch_generator is None or uid is None or uid < 0:
            return None

        try:
            _safe_sync_stream(self._stream)
            with mx.stream(self._stream):
                result = self.batch_generator.extract_cache([uid])
            if uid not in result:
                logger.debug(
                    "Cannot extract live cache for %s: uid %s not present",
                    request_id,
                    uid,
                )
                return None

            live_cache, live_tokens = result[uid]
            live_tokens_list = list(live_tokens) if live_tokens is not None else []
            if len(live_tokens_list) < len(expected_tokens) or (
                live_tokens_list[: len(expected_tokens)] != expected_tokens
            ):
                logger.debug(
                    "Skipping parser-stop cache store for %s: live cache tokens do "
                    "not match prompt boundary prefix (%s/%s tokens)",
                    request_id,
                    min(len(live_tokens_list), len(expected_tokens)),
                    len(expected_tokens),
                )
                return None

            extracted_cache, model_cache_config = self._extract_cache_states(live_cache)
            if not extracted_cache:
                return None
            return extracted_cache, model_cache_config
        except Exception as e:
            logger.debug(
                "Failed to extract live cache for parser-stop cache store %s: %s",
                request_id,
                e,
            )
            return None

    def _prepare_prompt_boundary_cache_store(
        self,
        request_id: str,
        request: Request,
        uid: int,
    ) -> (
        tuple[
            list[int],
            list[dict[str, Any]],
            Optional["ModelCacheConfig"],
            Any | None,
        ]
        | None
    ):
        """Prepare a prompt-only cache payload when a scheduler stop skipped it.

        Parser-side stops (for example tool-call end markers) can finish a request
        after a normal streaming token response. That response has no
        ``prompt_cache``, so the usual final-response cache extraction never runs.
        This fallback stores only prompt tokens up to a prefill block boundary,
        never generated output tokens.
        """
        if self.block_aware_cache is None:
            return None
        if request.specprefill_indices is not None:
            return None

        block_size = self.config.paged_cache_block_size
        if block_size <= 0:
            return None

        prompt_tokens = list(request.prompt_token_ids or [])
        boundary_len = (len(prompt_tokens) // block_size) * block_size
        if boundary_len <= 0:
            return None

        token_sequence = prompt_tokens[:boundary_len]

        boundary_override = self._get_boundary_store_override(request_id, prompt_tokens)
        if boundary_override is not None:
            (
                token_sequence,
                boundary_cache,
                boundary_model_config,
                intermediate_snapshots,
            ) = boundary_override

            live_payload = None
            if any(
                self._is_empty_boundary_placeholder(layer) for layer in boundary_cache
            ):
                live_payload = self._extract_live_request_cache_for_store(
                    request_id,
                    uid,
                    token_sequence,
                )
                if live_payload is None:
                    return None
                live_cache, live_model_config = live_payload
                cache_to_store = self._fill_boundary_placeholders_from_live_cache(
                    boundary_cache,
                    live_cache,
                )
                if cache_to_store is None:
                    return None
                model_cache_config = boundary_model_config or live_model_config
            else:
                cache_to_store = boundary_cache
                model_cache_config = boundary_model_config

            logger.info(
                "Using prompt boundary cache snapshot for %s: storing %s/%s prompt "
                "tokens after scheduler-side stop (skipping output tokens, %s "
                "intermediate snapshots)",
                request_id,
                len(token_sequence),
                len(prompt_tokens),
                len(intermediate_snapshots) if intermediate_snapshots else 0,
            )
            return (
                token_sequence,
                cache_to_store,
                model_cache_config,
                intermediate_snapshots,
            )

        # Pure sliceable cache models do not need boundary snapshots. For models
        # that do need snapshots, a missing snapshot means the non-sliceable state
        # at this boundary is unavailable, so skip rather than storing an unsafe
        # live decode tail.
        if self._detect_boundary_snapshot_need():
            return None

        live_payload = self._extract_live_request_cache_for_store(
            request_id,
            uid,
            token_sequence,
        )
        if live_payload is None:
            return None

        live_cache, live_model_config = live_payload
        if not all(self._is_sliceable_extracted_layer(layer) for layer in live_cache):
            logger.debug(
                "Skipping parser-stop cache store for %s: live cache has "
                "non-sliceable layers but no boundary snapshot",
                request_id,
            )
            return None

        logger.info(
            "Using live prompt cache for %s: storing %s/%s prompt tokens after "
            "scheduler-side stop (skipping output tokens)",
            request_id,
            len(token_sequence),
            len(prompt_tokens),
        )
        return token_sequence, live_cache, live_model_config, None

    def _validate_cache(self, cache: Any) -> bool:
        """
        Validate that a cache object is usable.

        This prevents NoneType errors when mlx-lm's BatchKVCache
        contains invalid/stale references.

        Args:
            cache: The cache object to validate

        Returns:
            True if cache is valid and usable
        """
        if cache is None:
            return False

        # Check if it's a list of cache layers
        if isinstance(cache, list):
            if len(cache) == 0:
                return False
            # Check each layer
            for layer_cache in cache:
                if layer_cache is None:
                    return False
                # Check if layer has expected structure
                # RotatingKVCache may have keys=None (legacy) or zero-length
                # keys (hybrid window padding). Both are valid empty states
                # that will be filled during padding reprocessing.
                if hasattr(layer_cache, "keys") and layer_cache.keys is None:
                    if hasattr(layer_cache, "max_size"):
                        continue  # Valid empty RotatingKVCache (keys=None)
                    return False
                if hasattr(layer_cache, "values") and layer_cache.values is None:
                    if hasattr(layer_cache, "max_size"):
                        continue  # Valid empty RotatingKVCache (values=None)
                    return False

        # Check BatchKVCache structure
        if hasattr(cache, "caches"):
            if cache.caches is None:
                return False
            for c in cache.caches:
                if c is None:
                    return False

        return True

    def _normalize_rotating_snapshot_state(
        self,
        layer_cache: Any,
        state: tuple[Any, Any],
        meta_state: Any,
        layer_idx: int | None = None,
    ) -> tuple[tuple[Any, Any], tuple[str, str, str, str]]:
        """
        Normalize RotatingKVCache state into merge-safe canonical form.

        Boundary snapshots captured mid-prefill can expose oversized rotating
        buffers (e.g., max_size + chunk_size - 1). Those states are valid for
        in-flight prefill but break BatchRotatingKVCache.merge() after SSD
        restore because merge expects per-request rotating buffers capped to
        max_size. This method canonicalizes to the latest max_size tokens.
        """
        if not isinstance(state, (list, tuple)) or len(state) < 2:
            return state, (
                tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ()
            )

        keys = state[0]
        values = state[1]
        if keys is None or values is None or not hasattr(keys, "shape"):
            return state, (
                tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ()
            )

        try:
            keep = (
                int(meta_state[0])
                if meta_state and len(meta_state) >= 1
                else int(getattr(layer_cache, "keep", 0))
            )
            max_size = (
                int(meta_state[1])
                if meta_state and len(meta_state) >= 2
                else int(getattr(layer_cache, "max_size", keys.shape[2]))
            )
            offset = (
                int(meta_state[2])
                if meta_state and len(meta_state) >= 3
                else int(getattr(layer_cache, "offset", keys.shape[2]))
            )
            idx = (
                int(meta_state[3])
                if meta_state and len(meta_state) >= 4
                else int(getattr(layer_cache, "_idx", keys.shape[2]))
            )
        except Exception:
            return state, (
                tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ()
            )

        ordered_keys = keys
        ordered_values = values
        temporal_order = getattr(layer_cache, "_temporal_order", None)
        if callable(temporal_order):
            try:
                ordered_keys = temporal_order(keys)
                ordered_values = temporal_order(values)
            except Exception:
                ordered_keys = keys
                ordered_values = values

        original_len = int(ordered_keys.shape[2]) if len(ordered_keys.shape) >= 3 else 0
        normalized_keys = ordered_keys
        normalized_values = ordered_values

        if max_size > 0 and original_len > max_size:
            if keep > 0 and keep < max_size:
                tail_len = max_size - keep
                normalized_keys = mx.concatenate(
                    [
                        ordered_keys[..., :keep, :],
                        ordered_keys[..., -tail_len:, :],
                    ],
                    axis=2,
                )
                normalized_values = mx.concatenate(
                    [
                        ordered_values[..., :keep, :],
                        ordered_values[..., -tail_len:, :],
                    ],
                    axis=2,
                )
            else:
                normalized_keys = ordered_keys[..., -max_size:, :]
                normalized_values = ordered_values[..., -max_size:, :]

            try:
                normalized_keys = mx.contiguous(normalized_keys)
                normalized_values = mx.contiguous(normalized_values)
            except Exception:
                pass

        normalized_len = (
            int(normalized_keys.shape[2]) if len(normalized_keys.shape) >= 3 else 0
        )
        # Force case 1 of _temporal_order: _idx == keys.shape[2] means the
        # buffer is already in temporal order (which is exactly what the
        # oversized trim above produces — the contiguous tail of the most
        # recent tokens). Anything else lets _temporal_order re-slice the
        # buffer in the rotated branch (case 2), which is wasted work and
        # obscures the merge contract. See cache.py:431-447 for the branches.
        normalized_idx = normalized_len

        normalized_meta = (
            str(keep),
            str(max_size),
            str(offset),
            str(normalized_idx),
        )

        if original_len != normalized_len or idx != normalized_idx:
            layer_tag = f"layer {layer_idx}: " if layer_idx is not None else ""
            logger.debug(
                "%sNormalized RotatingKVCache snapshot: len %s->%s, idx %s->%s, "
                "offset=%s, max_size=%s",
                layer_tag,
                original_len,
                normalized_len,
                idx,
                normalized_idx,
                offset,
                max_size,
            )

        return (normalized_keys, normalized_values), normalized_meta

    def _extract_cache_states(
        self,
        raw_cache: list[Any],
    ) -> tuple[list[dict[str, Any]], Optional["ModelCacheConfig"]]:
        """
        Extract actual tensor state from each layer cache.

        This extracts the real KV data using mlx-lm's cache.state property,
        allowing the data to be stored and reconstructed later even after
        the BatchGenerator is recreated.

        Also creates a ModelCacheConfig with per-layer type information to
        support hybrid cache models (e.g., KVCache + ArraysCache).

        Args:
            raw_cache: List of cache objects from mlx-lm (KVCache, ArraysCache, etc.)

        Returns:
            Tuple of:
            - List of dicts with {state, meta_state, class_name, cache_type}
            - ModelCacheConfig with per-layer type information (or None)
        """
        if not raw_cache:
            return [], None

        # Build ModelCacheConfig for type information.
        # Skip if raw_cache contains None entries (boundary snapshots with
        # sliceable layers replaced by None) — from_cache_list expects real
        # cache objects and would log noisy NoneType warnings.
        model_cache_config = None
        has_none_layers = any(c is None for c in raw_cache)
        if (
            HAS_CACHE_TYPE_HANDLERS
            and ModelCacheConfig is not None
            and not has_none_layers
        ):
            try:
                model_cache_config = ModelCacheConfig.from_cache_list(
                    raw_cache,
                    model_name=self.model_name if hasattr(self, "model_name") else "",
                )
            except Exception as e:
                logger.debug(f"Failed to build ModelCacheConfig: {e}")

        extracted = []
        for layer_idx, layer_cache in enumerate(raw_cache):
            # Boundary snapshots may contain None for sliceable layers
            # (KVCache) that were skipped during capture to save memory.
            # Insert a placeholder to preserve layer index alignment.
            if layer_cache is None:
                extracted.append(
                    {
                        "state": (),
                        "meta_state": (),
                        "class_name": "KVCache",
                        "cache_type": "KVCache",
                    }
                )
                continue
            try:
                class_name = type(layer_cache).__name__

                # Determine cache type using registry if available
                cache_type_name = class_name
                handler = None
                if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
                    try:
                        cache_type = CacheTypeRegistry.detect_cache_type(layer_cache)
                        cache_type_name = cache_type.value
                        handler = CacheTypeRegistry.get_handler(cache_type)
                    except Exception:
                        pass

                # CacheList: composite cache with multiple sub-caches
                if cache_type_name == "CacheList" or class_name == "CacheList":
                    if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
                        try:
                            handler = CacheTypeRegistry.get_handler_by_class_name(
                                "CacheList"
                            )
                            state_dict = handler.extract_state(layer_cache)
                            sub_states = list(state_dict.get("sub_states", []))
                            sub_class_names = list(
                                state_dict.get("sub_class_names", [])
                            )
                            sub_meta_states = list(
                                state_dict.get("sub_meta_states", [])
                            )

                            sub_caches = getattr(layer_cache, "caches", ())
                            for sub_idx, sub_cache in enumerate(sub_caches):
                                if sub_idx >= len(sub_states):
                                    break

                                sub_class_name = type(sub_cache).__name__
                                if sub_class_name in (
                                    "RotatingKVCache",
                                    "BatchRotatingKVCache",
                                    "PrefillReadyRotatingKVCache",
                                ):
                                    normalized_state, normalized_meta = (
                                        self._normalize_rotating_snapshot_state(
                                            sub_cache,
                                            sub_states[sub_idx],
                                            (
                                                sub_meta_states[sub_idx]
                                                if sub_idx < len(sub_meta_states)
                                                else getattr(
                                                    sub_cache, "meta_state", ()
                                                )
                                            ),
                                            layer_idx=layer_idx,
                                        )
                                    )
                                    sub_states[sub_idx] = normalized_state
                                    if sub_idx < len(sub_meta_states):
                                        sub_meta_states[sub_idx] = normalized_meta

                            extracted.append(
                                {
                                    "state": sub_states,
                                    "meta_state": (
                                        sub_class_names,
                                        sub_meta_states,
                                    ),
                                    "class_name": "CacheList",
                                    "cache_type": "CacheList",
                                }
                            )
                        except Exception as e:
                            logger.debug(f"CacheList handler extraction failed: {e}")
                            extracted.append(
                                {
                                    "state": [],
                                    "meta_state": ([], []),
                                    "class_name": "CacheList",
                                    "cache_type": "CacheList",
                                }
                            )
                    else:
                        # Fallback: extract sub-cache state/meta without handlers
                        # MUST append to extracted to prevent layer count mismatch (Issue #1)
                        sub_caches = getattr(layer_cache, "caches", ())
                        sub_states = []
                        sub_class_names = []
                        sub_meta_states = []
                        for sc in sub_caches:
                            sub_states.append(sc.state if hasattr(sc, "state") else ())
                            sub_class_names.append(type(sc).__name__)
                            sub_meta_states.append(getattr(sc, "meta_state", ()))
                        extracted.append(
                            {
                                "state": sub_states,
                                "meta_state": (sub_class_names, sub_meta_states),
                                "class_name": "CacheList",
                                "cache_type": "CacheList",
                            }
                        )
                    continue

                if hasattr(layer_cache, "state"):
                    if handler is not None and class_name in (
                        "MiniMaxM3KVCache",
                        "MiniMaxM3BatchKVCache",
                    ):
                        state = handler.serialize_state(layer_cache)
                        meta = handler.serialize_meta_state(layer_cache)
                    else:
                        state = layer_cache.state
                        meta = getattr(layer_cache, "meta_state", ())

                    is_rotating_cache = class_name in (
                        "RotatingKVCache",
                        "BatchRotatingKVCache",
                        "PrefillReadyRotatingKVCache",
                        "BufferedRotatingKVCache",
                    )
                    if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
                        is_rotating_cache = (
                            is_rotating_cache
                            or CacheTypeRegistry.is_rotating_family(class_name)
                        )
                    if is_rotating_cache:
                        state, meta = self._normalize_rotating_snapshot_state(
                            layer_cache,
                            state,
                            meta,
                            layer_idx=layer_idx,
                        )

                    # Preserve the full state tuple regardless of length.
                    # Legacy 2-tuple caches (KVCache, RotatingKVCache, ...)
                    # surface as (keys, values); 3-tuple caches like
                    # PoolingCache surface as (buf_kv, buf_gate, pooled);
                    # 4-tuple caches like BatchKVCache surface with the
                    # extra offset/padding metadata. Downstream
                    # serialization (paged_ssd_cache, boundary_snapshot)
                    # is N-tuple aware after the cache architecture
                    # refactor — see Section 6 of the implementation
                    # plan.
                    if isinstance(state, (list, tuple)) and len(state) >= 1:
                        # Validate non-None for legacy KV-style caches only.
                        # PoolingCache's buf_kv may legitimately be None
                        # (fresh cache before any update), so skip the
                        # null guard for non-KV cache classes.
                        if (
                            class_name in ("KVCache", "RotatingKVCache", "BatchKVCache")
                            or (
                                HAS_CACHE_TYPE_HANDLERS
                                and CacheTypeRegistry is not None
                                and CacheTypeRegistry.is_rotating_family(class_name)
                            )
                        ) and len(state) >= 2:
                            if state[0] is None or state[1] is None:
                                logger.debug(
                                    f"Layer {layer_idx} ({class_name}) has None keys/values, "
                                    f"skipping cache extraction"
                                )
                                return [], None  # Return empty - cache is corrupted

                        extracted.append(
                            {
                                "state": tuple(state),
                                "meta_state": meta,
                                "class_name": class_name,
                                "cache_type": cache_type_name,
                            }
                        )
                    else:
                        # Unexpected state format (e.g. a non-tuple scalar).
                        logger.debug(
                            f"Layer {layer_idx} ({class_name}) has unexpected state format"
                        )
                        meta = getattr(layer_cache, "meta_state", ())
                        # Wrap the scalar so downstream code still gets a
                        # tuple-shaped state. This path is essentially dead
                        # in practice — kept defensive only.
                        extracted.append(
                            {
                                "state": (state,),
                                "meta_state": meta,
                                "class_name": class_name,
                                "cache_type": cache_type_name,
                            }
                        )
                elif hasattr(layer_cache, "cache"):
                    # ArraysCache style: state stored in .cache list
                    cache_list = layer_cache.cache
                    if isinstance(cache_list, list) and len(cache_list) >= 2:
                        state = (cache_list[0], cache_list[1])
                        meta = getattr(layer_cache, "meta_state", ())
                        extracted.append(
                            {
                                "state": state,
                                "meta_state": meta,
                                "class_name": class_name,
                                "cache_type": cache_type_name,
                            }
                        )
                    else:
                        logger.debug(
                            f"Layer {layer_idx} ({class_name}) has invalid cache list"
                        )
                        continue
                else:
                    logger.debug(
                        f"Layer {layer_idx} ({class_name}) has no state or cache attribute"
                    )
                    continue

            except Exception as e:
                logger.debug(
                    f"Failed to extract state from cache layer {layer_idx}: {e}"
                )
                continue

        if len(extracted) != len(raw_cache):
            logger.debug(
                f"Incomplete cache extraction: {len(extracted)}/{len(raw_cache)} layers"
            )
            return [], None

        return extracted, model_cache_config

    @staticmethod
    def _common_prefix_len(a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    def _log_prefix_divergence(self, request: Request) -> None:
        """DEBUG-only prefix-cache miss diagnostic (issue #1003).

        Compares the new prompt against recently stored cache sequences and
        logs the first divergent token offset with decoded context on both
        sides, so an always-miss report can be traced to the exact prompt
        position (template re-render drift, client echo changes, eviction)
        instead of guessing from hit counters.
        """
        prompt = request.prompt_token_ids or []
        if not prompt or not self._cache_probe_seqs:
            return
        best_id, best_seq, best_p = None, None, -1
        for ref_id, seq in list(self._cache_probe_seqs):
            p = self._common_prefix_len(prompt, seq)
            if p > best_p:
                best_id, best_seq, best_p = ref_id, seq, p
        if best_seq is None:
            return
        cached = request.cached_tokens or 0
        reusable = min(len(prompt), len(best_seq))
        block = self.config.paged_cache_block_size
        logger.debug(
            f"Request {request.request_id}: prefix probe vs stored {best_id}: "
            f"common_prefix={best_p}/{reusable} tokens "
            f"(~{best_p // max(1, block)} blocks of {block}), "
            f"served cached_tokens={cached}, prompt={len(prompt)}"
        )
        if best_p < reusable:
            lo = max(0, best_p - 12)
            hi = best_p + 12
            try:
                stored_ctx = self.tokenizer.decode(best_seq[lo:hi])
                prompt_ctx = self.tokenizer.decode(prompt[lo:hi])
            except Exception:
                stored_ctx = prompt_ctx = "<decode failed>"
            logger.debug(
                f"Request {request.request_id}: first divergence at token "
                f"{best_p}: stored=...{stored_ctx!r} vs prompt=...{prompt_ctx!r}"
            )

    _CACHE_FRESHNESS_WAIT_MIN_PROMPT_TOKENS = 8192
    _CACHE_FRESHNESS_WAIT_MIN_COMMON_TOKENS = 8192
    _CACHE_FRESHNESS_WAIT_MIN_PROMPT_RATIO = 0.30
    _CACHE_FRESHNESS_WAIT_TIMEOUT_S = 4.0

    @staticmethod
    def _store_extra_keys_match(
        info: _InflightStoreInfo,
        request: Request,
    ) -> bool:
        return (
            info.extra_keys == request.vlm_extra_keys_for_cache
            and info.extra_key_token_start
            == request.vlm_extra_key_token_start_for_cache
            and info.extra_key_ranges == request.vlm_extra_key_ranges_for_cache
        )

    def _find_relevant_inflight_store(
        self,
        request: Request,
    ) -> tuple[str, concurrent.futures.Future, int] | None:
        """Find a pending store_cache job worth waiting for before lookup."""
        if not self._inflight_store_futures:
            return None

        prompt = request.prompt_token_ids or []
        if len(prompt) < self._CACHE_FRESHNESS_WAIT_MIN_PROMPT_TOKENS:
            return None

        best_rid: str | None = None
        best_future: concurrent.futures.Future | None = None
        best_common = 0
        for rid, future in list(self._inflight_store_futures.items()):
            if future.done():
                continue
            info = self._inflight_store_info.get(rid)
            if info is None or not self._store_extra_keys_match(info, request):
                continue

            common = self._common_prefix_len(prompt, info.tokens)
            if common > best_common:
                best_rid = rid
                best_future = future
                best_common = common

        if best_future is None or best_rid is None:
            return None
        if not (
            best_common >= self._CACHE_FRESHNESS_WAIT_MIN_COMMON_TOKENS
            or best_common / len(prompt) >= self._CACHE_FRESHNESS_WAIT_MIN_PROMPT_RATIO
        ):
            return None

        return best_rid, best_future, best_common

    def _should_defer_for_cache_freshness(self, request: Request) -> bool:
        """Defer only this waiting request until a relevant store is visible.

        This intentionally does not call Future.result(). The scheduler step can
        return immediately, so running decode rows and chunked prefills continue
        on subsequent steps while the waiting head holds admission order.
        """
        if self.block_aware_cache is None:
            return False

        now = time.monotonic()
        wait = self._cache_freshness_waits.get(request.request_id)
        if wait is not None:
            if wait.future.done():
                self._cache_freshness_waits.pop(request.request_id, None)
                try:
                    exc = wait.future.exception()
                except concurrent.futures.CancelledError:
                    logger.debug(
                        "Cache freshness deferral saw cancelled store_cache %s "
                        "before prefix lookup for %s",
                        wait.store_request_id,
                        request.request_id,
                    )
                else:
                    if exc is None:
                        logger.debug(
                            "Completed cache freshness deferral for store_cache %s "
                            "before prefix lookup for %s",
                            wait.store_request_id,
                            request.request_id,
                        )
                    else:
                        logger.debug(
                            "Cache freshness deferral saw failed store_cache %s "
                            "before prefix lookup for %s: %s",
                            wait.store_request_id,
                            request.request_id,
                            exc,
                        )
                return False
            if now >= wait.deadline_s:
                self._cache_freshness_waits.pop(request.request_id, None)
                logger.debug(
                    "Timed out cache freshness deferral for store_cache %s before "
                    "prefix lookup for %s (common_prefix=%d/%d)",
                    wait.store_request_id,
                    request.request_id,
                    wait.common_prefix,
                    wait.prompt_len,
                )
                return False
            return True

        match = self._find_relevant_inflight_store(request)
        if match is None:
            return False

        store_request_id, future, common_prefix = match
        prompt_len = len(request.prompt_token_ids or [])
        timeout = self._CACHE_FRESHNESS_WAIT_TIMEOUT_S
        self._cache_freshness_waits[request.request_id] = _CacheFreshnessWait(
            store_request_id=store_request_id,
            future=future,
            common_prefix=common_prefix,
            prompt_len=prompt_len,
            deadline_s=now + timeout,
        )

        logger.debug(
            "Deferring admission up to %.1fs for in-flight store_cache %s before "
            "prefix lookup for %s (common_prefix=%d/%d running=%d prefilling=%d)",
            timeout,
            store_request_id,
            request.request_id,
            common_prefix,
            prompt_len,
            len(self.running),
            len(self.prefilling),
        )
        return True

    def _prepare_prefix_cache_for_request(self, request: Request) -> None:
        if request.request_id in self._prefix_cache_prepared:
            return

        # Check prefix cache for cached KV state
        if self.block_aware_cache is not None:
            # Use paged cache
            block_table, remaining = self.block_aware_cache.fetch_cache(
                request.request_id,
                request.prompt_token_ids,
                extra_keys=request.vlm_extra_keys_for_cache,
                extra_key_token_start=request.vlm_extra_key_token_start_for_cache,
                extra_key_ranges=request.vlm_extra_key_ranges_for_cache,
            )
            if block_table and block_table.num_tokens > 0:
                bypass_hot_cache = self._bypass_hot_cache_under_pressure()
                if bypass_hot_cache:
                    logger.info(
                        "Skipping hot-cache preload for %s under memory pressure",
                        request.request_id,
                    )
                else:
                    self.block_aware_cache.preload_blocks(block_table)
                # Reconstruct actual KVCache objects from stored tensor data
                # Note: reconstruct_cache may modify block_table in-place if
                # partial reconstruction occurs (some blocks invalid)
                original_tokens = block_table.num_tokens
                if bypass_hot_cache:
                    reconstructed = self.block_aware_cache.reconstruct_cache(
                        block_table,
                        promote_to_hot_cache=False,
                    )
                else:
                    reconstructed = self.block_aware_cache.reconstruct_cache(
                        block_table
                    )
                if reconstructed:
                    request.prompt_cache = reconstructed
                    request.block_table = block_table
                    request.cached_tokens = block_table.num_tokens
                    request.shared_prefix_blocks = len(block_table.block_ids)
                    # Recalculate remaining_tokens in case block_table was truncated
                    request.remaining_tokens = request.prompt_token_ids[
                        block_table.num_tokens :
                    ]
                    if self._align_minimax_m3_partial_cache_to_prefill_step(request):
                        request.cached_tokens = block_table.num_tokens
                        request.shared_prefix_blocks = len(block_table.block_ids)
                        request.remaining_tokens = request.prompt_token_ids[
                            block_table.num_tokens :
                        ]
                    # For exact prefix hits we need cache state at (N-1) and the
                    # last prompt token as input to produce the first decode logit.
                    # Reusing cache state at N and feeding the last token again
                    # shifts the model state and can change greedy output.
                    if len(request.remaining_tokens) == 0 and request.cached_tokens > 0:
                        if self._cache_list_needs_boundary_snapshot(
                            request.prompt_cache
                        ):
                            # Stateful non-sliceable caches (Rotating/Arrays)
                            # cannot be safely converted from N to N-1 state
                            # without cache-type-specific logic.
                            if self.paged_cache_manager is not None:
                                self.paged_cache_manager.delete_block_table(
                                    request.request_id
                                )
                            request.prompt_cache = None
                            request.block_table = None
                            request.cached_tokens = 0
                            request.shared_prefix_blocks = 0
                            request.remaining_tokens = request.prompt_token_ids
                            logger.debug(
                                f"Request {request.request_id}: exact cache hit with "
                                f"stateful cache type, falling back to full prefill "
                                f"for deterministic kickoff"
                            )
                        elif self._trim_prompt_cache_for_generation(
                            request.prompt_cache
                        ):
                            request.cached_tokens = max(0, request.cached_tokens - 1)
                            request.remaining_tokens = request.prompt_token_ids[-1:]
                            logger.debug(
                                f"Request {request.request_id}: exact cache hit adjusted "
                                f"to N-1 state for generation kickoff "
                                f"(cached_tokens={request.cached_tokens}, "
                                f"remaining={len(request.remaining_tokens)})"
                            )
                        else:
                            # Fallback to full recompute when cache layers cannot
                            # be safely trimmed by one token (e.g., non-trimmable
                            # recurrent state caches).
                            if self.paged_cache_manager is not None:
                                self.paged_cache_manager.delete_block_table(
                                    request.request_id
                                )
                            request.prompt_cache = None
                            request.block_table = None
                            request.cached_tokens = 0
                            request.shared_prefix_blocks = 0
                            request.remaining_tokens = request.prompt_token_ids
                            logger.debug(
                                f"Request {request.request_id}: exact cache hit could "
                                f"not be trimmed safely, falling back to full prefill"
                            )
                    if block_table.num_tokens < original_tokens:
                        logger.debug(
                            f"Request {request.request_id}: partial cache hit, "
                            f"{request.cached_tokens} tokens in {request.shared_prefix_blocks} blocks "
                            f"(originally {original_tokens} tokens), "
                            f"{len(request.remaining_tokens)} tokens remaining"
                        )
                    else:
                        logger.debug(
                            f"Request {request.request_id}: paged cache hit, "
                            f"{request.cached_tokens} tokens in {request.shared_prefix_blocks} blocks, "
                            f"{len(request.remaining_tokens)} tokens remaining, cache reconstructed"
                        )
                else:
                    # Reconstruction failed, treat as cache miss
                    if self.paged_cache_manager is not None:
                        self.paged_cache_manager.delete_block_table(request.request_id)
                    request.remaining_tokens = request.prompt_token_ids
                    logger.debug(
                        f"Request {request.request_id}: paged cache reconstruction failed, "
                        "released shared blocks"
                    )
            else:
                request.remaining_tokens = request.prompt_token_ids
        else:
            # No paged SSD cache configured - process all tokens
            request.remaining_tokens = request.prompt_token_ids

        # DEBUG-only: trace where this prompt diverges from recently stored
        # cache sequences (issue #1003 always-miss diagnosis).
        if logger.isEnabledFor(logging.DEBUG):
            self._log_prefix_divergence(request)

        # SpecPrefill: score remaining tokens with draft model if applicable.
        # Must run AFTER prefix cache check (scoring applies only to uncached suffix).
        self._try_specprefill_scoring(request)
        self._prefix_cache_prepared.add(request.request_id)

    def add_request(self, request: Request) -> None:
        """
        Add a new request to the scheduler.

        Raises SchedulerQueueFullError when the waiting queue is at or above
        the configured cap (max(max_num_seqs * 4, 32)). Server layer maps
        this to HTTP 503 + Retry-After.

        Args:
            request: The request to add
        """
        if request.request_id in self.requests:
            raise ValueError(f"Request {request.request_id} already exists")

        # Cap the waiting queue so client-side polling can't accumulate
        # unbounded work and the scheduler can apply backpressure via 503.
        max_waiting = max(self.config.max_num_seqs * 4, 32)
        if len(self.waiting) >= max_waiting:
            from .exceptions import SchedulerQueueFullError

            raise SchedulerQueueFullError(
                current_depth=len(self.waiting),
                max_depth=max_waiting,
            )

        # Tokenize if needed
        if request.prompt_token_ids is None:
            if isinstance(request.prompt, str):
                request.prompt_token_ids = self.tokenizer.encode(request.prompt)
            else:
                request.prompt_token_ids = list(request.prompt)
            request.num_prompt_tokens = len(request.prompt_token_ids)

        # Prefix-cache lookup is intentionally delayed until admission. That
        # lets a same-prefix request wait for a relevant in-flight store_cache
        # without blocking the scheduler lane that continues decode/prefill.
        #
        # Keep the immediate preflight only when no prefix cache lookup can
        # change cached_tokens. With a block-aware cache, the in-stream
        # _preflight_memory_check runs after lookup with the final cache state.
        if self.block_aware_cache is None:
            request.remaining_tokens = request.prompt_token_ids
            try:
                self.preflight_or_raise(
                    num_prompt_tokens=request.num_prompt_tokens,
                    cached_tokens=request.cached_tokens or 0,
                    request_id=request.request_id,
                )
            except Exception:
                self._release_paged_cache_for_request(request.request_id)
                raise

        # Add to tracking
        self.requests[request.request_id] = request
        self.waiting.append(request)

        logger.debug(
            f"Added request {request.request_id} with {request.num_prompt_tokens} prompt tokens"
        )

    def set_specprefill_draft_model(
        self, draft_model: Any, draft_model_name: str | None = None
    ) -> None:
        """Set the draft model for SpecPrefill scoring.

        Creates a separate BlockAwarePrefixCache for the draft model
        using the existing paged SSD cache infrastructure. The model_name
        in compute_block_hash() naturally isolates draft blocks from target.
        """
        self._specprefill_draft_model = draft_model
        self._draft_prefix_cache: Any | None = None

        if (
            self.paged_cache_manager is not None
            and self.paged_ssd_cache_manager is not None
        ):
            try:
                from .cache.paged_cache import PagedCacheManager
                from .cache.prefix_cache import BlockAwarePrefixCache

                name = draft_model_name or "specprefill-draft"
                draft_paged = PagedCacheManager(
                    block_size=self.config.paged_cache_block_size,
                    max_blocks=self.paged_cache_manager.max_blocks,
                    model_name=name,
                )
                self._draft_prefix_cache = BlockAwarePrefixCache(
                    model=draft_model,
                    paged_cache_manager=draft_paged,
                    paged_ssd_cache_manager=self.paged_ssd_cache_manager,
                )
                self._draft_prefix_cache.set_cold_restore_callback(
                    self._restore_block_from_cold
                )
                logger.info(
                    f"SpecPrefill: draft model set with SSD cache (model_name={name})"
                )
            except Exception as e:
                logger.warning(f"SpecPrefill: draft SSD cache setup failed: {e}")
                logger.info("SpecPrefill: draft model set (no SSD cache)")
        else:
            logger.info("SpecPrefill: draft model set (no SSD cache)")

    def set_vlm_mtp_drafter(
        self,
        drafter: VLMMTPDrafter | None,
        draft_block_size: int | None = None,
    ) -> None:
        """Attach an MTP drafter for VLM MTP speculative decode.

        Called by ``VLMBatchedEngine.set_vlm_mtp_drafter`` once the drafter
        artifact is loaded.  Supports any drafter that mlx-vlm's
        ``load_drafter()`` resolves to ``kind="mtp"`` (gemma4_assistant,
        qwen3_5_mtp, etc.).  ``None`` clears the toggle.
        """
        self._vlm_mtp_drafter = drafter
        self._vlm_mtp_draft_block_size = draft_block_size
        if drafter is not None:
            logger.info(
                "VLM MTP drafter attached to scheduler (block_size=%s)",
                draft_block_size,
            )

    def _route_to_vlm_mtp(
        self,
        request: Request,
        prefilled_cache: list[Any],
        last_tokens: list[int],
        sampler: Callable[[Any], Any],
        state_machine: Any,
    ) -> int | None:
        """Bypass BatchGenerator and stand up a vlm_mtp generator instead.

        Runs the final forward on ``last_tokens`` with ``return_hidden=True``
        and ``return_shared_kv=True`` so the drafter has the targets it
        needs, samples the first bonus token from the resulting logits, and
        returns a synthesized uid that ``step()`` will drive.

        Returns ``None`` if the eligibility check fails at the last second
        (drafter missing, language model lacks rollback hook, etc.) so the
        caller can fall back to the normal BatchGenerator path.
        """
        drafter = self._vlm_mtp_drafter
        if drafter is None:
            return None

        # Gemma4AssistantDraftModel keeps ``_shared_kv`` / ``_input_embed`` on
        # the module instance, so multiple in-flight ``_mtp_rounds`` generators
        # share one drafter and effectively serialize on it: each round has
        # to ``set_shared_kv`` for its own request before ``draft_block`` runs.
        # Output stays correct because target-side verify is the source of
        # truth in speculative decoding (a stale-drafter round just rejects
        # everything and falls back to a target-only step), but the
        # per-request tok/s is roughly halved under concurrency. Empirically
        # at 4 concurrent, vlm_mtp gives ~14 tok/s each vs BatchGenerator's
        # ~27 tok/s each — BG's batched matmul beats serialized speculative
        # rounds. So we route only the first eligible request through
        # vlm_mtp and let subsequent concurrent requests fall back. A future
        # commit can swap this gate for true batched MTP via
        # ``_mtp_rounds_batch`` if and when omlx prefill exposes batched
        # hidden/shared_kv outputs.
        if self._vlm_mtp_active:
            logger.info(
                "vlm_mtp routing skipped for %s: drafter is busy with %d "
                "request(s); falling back to BatchGenerator",
                request.request_id,
                len(self._vlm_mtp_active),
            )
            return None

        lm = getattr(self.model, "_language_model", None)
        if lm is None or not hasattr(lm, "rollback_speculative_cache"):
            logger.warning(
                "vlm_mtp toggle on but model lacks _language_model with "
                "rollback_speculative_cache (model=%s); falling back to "
                "standard decode for request %s",
                type(self.model).__name__,
                request.request_id,
            )
            return None
        target_model = self.model

        if not last_tokens:
            logger.warning(
                "vlm_mtp routing skipped: last_tokens empty for request %s",
                request.request_id,
            )
            return None

        mtp_sampler = _make_suppressing_sampler(sampler, self._model_suppress_tokens)
        last_arr = mx.array(last_tokens)[None]  # (1, len_last)
        try:
            with mx.stream(self._stream):
                set_batch_rope = getattr(target_model, "set_batch_rope_deltas", None)
                if callable(set_batch_rope):
                    set_batch_rope(mx.array([request.rope_deltas]))
                out = target_model(
                    last_arr,
                    cache=prefilled_cache,
                    return_hidden=True,
                    return_shared_kv=True,
                )
                mx.eval([c.state for c in prefilled_cache])
        except Exception as e:
            logger.warning(
                "vlm_mtp final-prefill forward failed (%s); falling back "
                "to standard decode for request %s",
                e,
                request.request_id,
            )
            return None

        # Handle current LanguageModelOutput and legacy tuple
        # (logits, hidden, gdn_states) MTP runtime patch returns.
        if isinstance(out, tuple):
            logits = out[0][:, -1, :]
            hidden_raw = out[1]
        else:
            logits = out.logits[:, -1, :]
            hidden_raw = out.hidden_states

        first_bonus_arr = mtp_sampler(logits)  # mx.array shape [1]
        mx.eval(first_bonus_arr)

        if isinstance(hidden_raw, list):
            hidden = hidden_raw[-1]
        else:
            hidden = hidden_raw
        # Slice to last position so the drafter sees a [B, 1, H] tensor
        # regardless of how many tokens this forward processed.
        if hidden.shape[1] > 1:
            hidden = hidden[:, -1:, :]

        # Combine base stop tokens (EOS, Harmony, generation_config) with
        # request-specific stop_token_ids — same shape as _build_state_machine.
        eos_ids: set[int] = self._get_stop_tokens()
        if request.sampling_params.stop_token_ids:
            eos_ids.update(request.sampling_params.stop_token_ids)

        try:
            generator = run_vlm_mtp_decode(
                target_language_model=target_model,
                drafter=drafter,
                prompt_cache=prefilled_cache,
                hidden=hidden,
                shared_kv_states=(
                    getattr(out, "shared_kv_states", {})
                    if not isinstance(out, tuple)
                    else {}
                ),
                first_bonus=int(first_bonus_arr.item()),
                max_tokens=request.sampling_params.max_tokens,
                sampler=mtp_sampler,
                draft_block_size=self._vlm_mtp_draft_block_size,
                token_dtype=mx.int32,
                eos_token_ids=eos_ids or None,
            )
        except Exception as e:
            logger.warning(
                "vlm_mtp generator setup failed (%s); falling back for %s",
                e,
                request.request_id,
            )
            return None

        uid = self._vlm_mtp_next_uid
        self._vlm_mtp_next_uid -= 1
        self._vlm_mtp_active[uid] = _VLMMTPDecodeState(
            generator=generator,
            request=request,
            prompt_cache=prefilled_cache,
            sampler=mtp_sampler,
            state_machine=state_machine,
            max_tokens=request.sampling_params.max_tokens,
            stop_token_ids=set(eos_ids),
        )
        logger.info(
            "vlm_mtp decode started: request=%s uid=%d block_size=%s",
            request.request_id,
            uid,
            self._vlm_mtp_draft_block_size,
        )
        return uid

    def _log_vlm_mtp_stats(
        self, state: "_VLMMTPDecodeState", finish_reason: str
    ) -> None:
        """Emit one INFO line per finished vlm_mtp request with the drafter
        acceptance rate measured for that request.

        Reads ``Gemma4AssistantDraftModel.accept_lens`` — a list of accepted
        draft counts per round, populated inside mlx-vlm's ``_mtp_rounds``.
        The drafter mutates this in place and ``reset()`` (called at the
        start of every new round-loop entry) clears it, so we have to read
        before the next eligible request lands. The serialized routing in
        ``_route_to_vlm_mtp`` guarantees one in-flight vlm_mtp generator
        at a time, so the value we read here belongs to ``state.request``.
        """
        drafter = self._vlm_mtp_drafter
        if drafter is None:
            return
        accept_lens = getattr(drafter.model, "accept_lens", None)
        if not accept_lens:
            return
        try:
            lens = [int(x) for x in accept_lens]
        except Exception:
            return
        rounds = len(lens)
        if rounds == 0:
            return
        total_accepted = sum(lens)
        block_size = self._vlm_mtp_draft_block_size or int(
            getattr(drafter.model.config, "block_size", 4)
        )
        max_per_round = max(1, block_size - 1)
        acceptance_rate = total_accepted / (rounds * max_per_round)
        avg_tokens_per_round = (total_accepted + rounds) / rounds
        logger.info(
            "vlm_mtp stats: request=%s finish=%s rounds=%d "
            "accepted=%d/%d (%.1f%%) tokens_per_round=%.2f "
            "emitted=%d block_size=%d",
            state.request.request_id,
            finish_reason,
            rounds,
            total_accepted,
            rounds * max_per_round,
            acceptance_rate * 100,
            avg_tokens_per_round,
            state.emitted,
            block_size,
        )

    def _step_vlm_mtp(self) -> list[_VLMMTPResponse]:
        """Advance every active vlm_mtp generator by one yield.

        Returns the synthesized responses for ``_process_batch_responses``.
        Mirrors mlx-lm BatchGenerator's per-step contract: one
        ``GenerationBatch.Response``-shaped object per active uid.
        """
        if not self._vlm_mtp_active:
            return []

        responses: list[_VLMMTPResponse] = []
        for uid, state in list(self._vlm_mtp_active.items()):
            try:
                with mx.stream(self._stream):
                    token_val = next(state.generator)
            except StopIteration:
                # Round loop exited naturally — terminate with prompt cache
                # so the prefix-cache layer can keep using it.
                self._log_vlm_mtp_stats(state, "length")
                responses.append(
                    _VLMMTPResponse(
                        uid=uid,
                        token=0,
                        finish_reason="length",
                        prompt_cache=state.prompt_cache,
                    )
                )
                state.finished = True
                continue

            # Single-request mode yields ints; batch mode (not yet routed
            # by omlx) would yield a list. Guard so the path stays robust
            # if we widen routing later.
            if isinstance(token_val, list):
                # Take the first row (we only route singles for now).
                tok = next((t for t in token_val if t is not None), None)
                if tok is None:
                    responses.append(
                        _VLMMTPResponse(
                            uid=uid,
                            token=0,
                            finish_reason="length",
                            prompt_cache=state.prompt_cache,
                        )
                    )
                    state.finished = True
                    continue
                token = int(tok)
            else:
                token = int(token_val)

            state.emitted += 1
            finish_reason: str | None = None
            if state.stop_token_ids and token in state.stop_token_ids:
                finish_reason = "stop"
            elif state.emitted >= state.max_tokens:
                finish_reason = "length"

            if finish_reason is not None:
                self._log_vlm_mtp_stats(state, finish_reason)

            responses.append(
                _VLMMTPResponse(
                    uid=uid,
                    token=token,
                    finish_reason=finish_reason,
                    prompt_cache=(
                        state.prompt_cache if finish_reason is not None else None
                    ),
                )
            )
            if finish_reason is not None:
                state.finished = True

        # Drop finished entries.
        for uid in [u for u, s in self._vlm_mtp_active.items() if s.finished]:
            del self._vlm_mtp_active[uid]

        return responses

    def _try_specprefill_scoring(self, request: Request) -> None:
        """Score tokens with draft model if SpecPrefill is applicable.

        Uses paged SSD cache for the draft model: if the prompt prefix
        was already scored in a previous turn, the draft cache is restored
        and only the new suffix is prefilled through the draft model.
        """
        if self._specprefill_draft_model is None:
            return

        specprefill_enabled = getattr(request, "_specprefill_enabled", False)
        if not specprefill_enabled:
            return

        if request.vlm_inputs_embeds is not None:
            return

        remaining = request.remaining_tokens or request.prompt_token_ids
        if remaining is None:
            return

        n_remaining = len(remaining)
        from .patches.specprefill import DEFAULT_KEEP_RATE, DEFAULT_THRESHOLD

        threshold = (
            getattr(request, "_specprefill_threshold", None) or DEFAULT_THRESHOLD
        )
        keep_pct = getattr(request, "_specprefill_keep_pct", None) or DEFAULT_KEEP_RATE

        # Threshold check on TOTAL remaining (not after system exclusion)
        if n_remaining <= threshold:
            return

        # System prompt protection: exclude system tokens from scoring.
        # If paged cache already covered the system prompt, remaining
        # won't include it (effective_system = 0).
        system_end = request.specprefill_system_end
        effective_system = max(0, system_end - request.cached_tokens)
        tokens_to_score = (
            remaining[effective_system:] if effective_system > 0 else remaining
        )
        n_to_score = len(tokens_to_score)

        # If conversation portion is below threshold after system exclusion,
        # skip SpecPrefill (system will be full-prefilled by normal path)
        if n_to_score <= threshold:
            return

        tracker = get_prefill_tracker()
        model_id = os.path.basename(self.config.model_name.rstrip("/"))

        try:
            from .patches.specprefill import score_tokens, select_chunks

            # Draft prefix cache lookup
            draft_cache = None
            draft_cached_tokens = 0
            if self._draft_prefix_cache is not None:
                try:
                    block_table, draft_remaining = self._draft_prefix_cache.fetch_cache(
                        request.request_id, tokens_to_score
                    )
                    if block_table and block_table.num_tokens > 0:
                        self._draft_prefix_cache.preload_blocks(block_table)
                        reconstructed = self._draft_prefix_cache.reconstruct_cache(
                            block_table
                        )
                        if reconstructed:
                            draft_cache = reconstructed
                            draft_cached_tokens = block_table.num_tokens
                except Exception as e:
                    logger.debug(f"SpecPrefill: draft cache fetch failed: {e}")

            spec_extra = {
                "prompt_tokens": request.num_prompt_tokens,
                "system_tokens": request.specprefill_system_end,
                "conversation_tokens": request.num_prompt_tokens
                - request.specprefill_system_end,
                "cached_tokens": request.cached_tokens,
            }

            def _score_progress(processed: int, total: int, phase: str) -> None:
                tracker.update(
                    request.request_id,
                    min(processed, total - 1),
                    total,
                    model_id,
                    phase=f"specprefill_{phase}",
                    detail="scoring draft tokens",
                    extra=spec_extra,
                )

            # Register tracker entry and stream draft scoring progress so the
            # dashboard shows movement during long SpecPrefill scoring pauses.
            tracker.update(
                request.request_id,
                0,
                n_to_score,
                model_id,
                phase="specprefill_scoring",
                detail="scoring draft tokens",
                extra=spec_extra,
            )

            t0 = time.monotonic()
            importance, used_cache = score_tokens(
                self._specprefill_draft_model,
                tokens_to_score,
                prefill_step_size=self.config.prefill_step_size,
                existing_cache=draft_cache,
                progress_callback=_score_progress,
            )
            selected = select_chunks(importance, keep_pct=keep_pct)
            t_score = time.monotonic() - t0

            n_selected = selected.shape[0]
            request.specprefill_indices = selected
            request.specprefill_total_tokens = n_to_score
            request.specprefill_position_offset = (
                request.cached_tokens + effective_system
            )
            request._specprefill_system_tokens = effective_system

            extras = []
            if draft_cached_tokens > 0:
                extras.append(f"draft cache hit {draft_cached_tokens}")
            total_prompt = request.num_prompt_tokens
            system_total = request.specprefill_system_end
            cached = request.cached_tokens
            extras.append(
                f"prompt {total_prompt} = "
                f"system {system_total} + conv {total_prompt - system_total}, "
                f"cached {cached}"
            )

            tracker.update(
                request.request_id,
                n_to_score - 1,
                n_to_score,
                model_id,
                phase="specprefill_selected",
                detail="selected sparse tokens",
                extra={
                    **spec_extra,
                    "scored_tokens": n_to_score,
                    "selected_tokens": n_selected,
                    "keep_percent": round(n_selected / n_to_score * 100),
                },
            )

            logger.info(
                f"SpecPrefill: scored {n_to_score} tokens in {t_score:.1f}s, "
                f"selected {n_selected}/{n_to_score} "
                f"(keep={n_selected/n_to_score*100:.0f}%, {', '.join(extras)})"
            )

            # Save draft cache for next turn
            if self._draft_prefix_cache is not None and used_cache is not None:
                try:
                    extracted, mcc = self._extract_cache_states(used_cache)
                    if extracted:
                        self._draft_prefix_cache.store_cache(
                            request.request_id,
                            tokens_to_score,
                            extracted,
                            model_cache_config=mcc,
                        )
                except Exception as e:
                    logger.debug(f"SpecPrefill: draft cache store failed: {e}")

            # Free draft cache from memory.  Use _sync_and_clear_cache() so
            # the engine stream is drained before Metal buffers are
            # returned to the pool — a bare mx.clear_cache() here can race
            # with in-flight async evals and trigger a kernel panic (#557).
            del used_cache
            _sync_and_clear_cache(self._stream)

            # Mark scoring complete (auto-removes tracker entry).
            tracker.update(request.request_id, n_to_score, n_to_score, model_id)

        except Exception as e:
            logger.error(
                f"SpecPrefill scoring failed, falling back to normal path: {e}"
            )
            request.specprefill_indices = None
            tracker.remove(request.request_id)

    def _cleanup_specprefill(self, request_id: str) -> None:
        """Clean up SpecPrefill RoPE patches when a request finishes."""
        if self._specprefill_active_request_id == request_id:
            from .patches.specprefill import cleanup_rope

            cleanup_rope(self.model)
            self._specprefill_active_request_id = None
            logger.debug(
                f"SpecPrefill: RoPE restored for finished request {request_id}"
            )

    def _trim_prompt_cache_for_generation(self, cache_list: list[Any]) -> bool:
        """Trim each cache layer by one token for exact-hit generation kickoff."""
        return self._trim_prompt_cache_by_tokens(cache_list, 1)

    def _trim_prompt_cache_by_tokens(self, cache_list: list[Any], n: int) -> bool:
        """Trim each cache layer by n tokens."""
        if not cache_list:
            return False
        if n <= 0:
            return True

        for cache_obj in cache_list:
            if not self._trim_cache_tree_by_tokens(cache_obj, n):
                return False
        return True

    def _trim_cache_tree_by_one(self, cache_obj: Any) -> bool:
        """Trim one token from cache object (recursively for CacheList)."""
        return self._trim_cache_tree_by_tokens(cache_obj, 1)

    def _trim_cache_tree_by_tokens(self, cache_obj: Any, n: int) -> bool:
        """Trim n tokens from cache object (recursively for CacheList)."""
        sub_caches = getattr(cache_obj, "caches", None)
        if isinstance(sub_caches, (list, tuple)):
            return all(
                self._trim_cache_tree_by_tokens(sub_cache, n)
                for sub_cache in sub_caches
            )

        trim_fn = getattr(cache_obj, "trim", None)
        if not callable(trim_fn):
            return False

        try:
            trimmed = trim_fn(n)
            if trimmed is None:
                return True
            return int(trimmed) >= n
        except Exception:
            return False

    def _cache_tree_has_class_name(
        self,
        cache_obj: Any,
        class_names: frozenset[str],
    ) -> bool:
        """Return True when a cache tree contains one of the named cache classes."""
        if type(cache_obj).__name__ in class_names:
            return True
        sub_caches = getattr(cache_obj, "caches", None)
        if isinstance(sub_caches, (list, tuple)):
            return any(
                self._cache_tree_has_class_name(sub_cache, class_names)
                for sub_cache in sub_caches
            )
        return False

    def _align_minimax_m3_partial_cache_to_prefill_step(
        self,
        request: "Request",
    ) -> bool:
        """Align MiniMax M3 partial hits to external prefill chunk boundaries."""
        cache_list = request.prompt_cache
        block_table = request.block_table
        prompt_tokens = request.prompt_token_ids or []
        if not cache_list or block_table is None or not block_table.block_ids:
            return False
        if block_table.num_tokens <= 0 or block_table.num_tokens >= len(prompt_tokens):
            return False

        minimax_m3_names = frozenset({"MiniMaxM3KVCache"})
        has_minimax_m3 = any(
            self._cache_tree_has_class_name(cache_obj, minimax_m3_names)
            for cache_obj in cache_list
        )
        if not has_minimax_m3:
            return False

        block_size = int(getattr(self.config, "paged_cache_block_size", 0) or 0)
        prefill_step = int(getattr(self.config, "prefill_step_size", 0) or 0)
        if block_size <= 0 or prefill_step <= block_size:
            return False

        aligned_tokens = (block_table.num_tokens // prefill_step) * prefill_step
        aligned_tokens = (aligned_tokens // block_size) * block_size
        if aligned_tokens <= 0 or aligned_tokens >= block_table.num_tokens:
            return False

        target_block_count = 0
        target_tokens = 0
        for block_id in block_table.block_ids:
            block = (
                self.paged_cache_manager.allocated_blocks.get(block_id)
                if self.paged_cache_manager is not None
                else None
            )
            token_count = int(getattr(block, "token_count", block_size) or block_size)
            if target_tokens + token_count > aligned_tokens:
                break
            target_tokens += token_count
            target_block_count += 1

        if target_tokens != aligned_tokens:
            logger.debug(
                "MiniMax M3 partial cache alignment skipped for %s: cannot align "
                "block table from %d to %d tokens",
                request.request_id,
                block_table.num_tokens,
                aligned_tokens,
            )
            return False

        trim_tokens = block_table.num_tokens - aligned_tokens
        if not self._trim_prompt_cache_by_tokens(cache_list, trim_tokens):
            logger.debug(
                "MiniMax M3 partial cache alignment skipped for %s: cache trim "
                "by %d tokens failed",
                request.request_id,
                trim_tokens,
            )
            return False

        dropped_block_ids = block_table.block_ids[target_block_count:]
        if self.paged_cache_manager is not None:
            for block_id in dropped_block_ids:
                self.paged_cache_manager.free_block(block_id)
        block_table.block_ids = block_table.block_ids[:target_block_count]
        block_table.num_tokens = aligned_tokens

        logger.info(
            "MiniMax M3 partial cache aligned to prefill step for %s: "
            "%d -> %d tokens, dropped %d block(s)",
            request.request_id,
            aligned_tokens + trim_tokens,
            aligned_tokens,
            len(dropped_block_ids),
        )
        return True

    def _remove_uid_from_active_batch(self, uid: int) -> None:
        """Remove UID from BatchGenerator safely.

        vlm_mtp uses negative uids that BatchGenerator never sees; the
        per-uid generator state is owned by ``_vlm_mtp_active`` and gets
        dropped when ``_step_vlm_mtp`` marks the entry finished.
        """
        if uid < 0:
            return
        if self.batch_generator is None:
            return

        self.batch_generator.remove([uid])

    def _check_pending_aborts_for_uids(self, uids: list[int]) -> list[int]:
        """Return UIDs that have pending aborts.

        Called during prefill to detect aborted
        requests between chunks. GIL guarantees thread-safe reads of
        _pending_abort_ids from the executor thread.
        """
        if not self._pending_abort_ids:
            return []
        aborted = []
        for uid in uids:
            request_id = self.uid_to_request_id.get(uid)
            if request_id and request_id in self._pending_abort_ids:
                aborted.append(uid)
        return aborted

    def abort_request(self, request_id: str) -> bool:
        """
        Enqueue a request for deferred abort.

        The actual abort is processed at the start of the next step() call,
        ensuring thread safety with the hybrid executor pattern. CPython GIL
        guarantees set.add() is atomic.

        Args:
            request_id: The request ID to abort

        Returns:
            True (abort is always enqueued)
        """
        self._pending_abort_ids.add(request_id)
        logger.debug(f"Enqueued deferred abort for request {request_id}")
        return True

    def _process_pending_aborts(self) -> None:
        """Drain and process pending abort requests.

        Called from step() to ensure aborts are processed in the same
        execution context as generation (thread-safe).
        """
        while self._pending_abort_ids:
            request_id = self._pending_abort_ids.pop()
            self._do_abort_request(request_id)

    def _cleanup_prefill_abort_request(
        self, request: "Request", temp_uid: int | None = None
    ) -> None:
        """Finish cleanup for a request aborted while it was being prefetched.

        External prefill removes the request from ``waiting`` before it has a
        real BatchGenerator UID. If a client abort arrives at that point, the
        normal next-step deferred abort can be stranded because ``has_requests``
        no longer sees queued work. Finish it synchronously on the scheduler
        thread instead.
        """
        if temp_uid is not None:
            self.uid_to_request_id.pop(temp_uid, None)
            self.request_id_to_uid.pop(request.request_id, None)

        self._pending_abort_ids.discard(request.request_id)
        self._do_abort_request(request.request_id)

    def request_idle_reclaim(self) -> None:
        """Enqueue a between-turn Metal reclaim (thread-safe, no Metal touch).

        Called by ProcessMemoryEnforcer (asyncio thread) when memory pressure
        is hard but every loaded model is pinned and no load is in progress —
        the case where there is nothing to evict. Setting the flag is
        GIL-atomic; the actual ``_sync_and_clear_cache`` runs on the inference
        thread when step() drains it, and only when the scheduler is idle.
        """
        self._pending_reclaim_request = True

    def _process_pending_reclaim(self) -> None:
        """Drain a deferred idle reclaim request (inference-thread side).

        Only reclaims when truly idle (no running / prefilling / waiting work)
        so we never clear Metal buffers an in-flight decode or prefill still
        references.
        """
        if not self._pending_reclaim_request:
            return
        self._pending_reclaim_request = False
        if self.running or self.prefilling or self.waiting:
            return
        before = self._current_usage_bytes()
        after = self._reclaim_prefill_headroom()
        logger.info(
            "Idle reclaim: trimmed Metal transients between turns "
            "(%.1fGB -> %.1fGB)",
            before / 1024**3,
            after / 1024**3,
        )

    def _do_abort_request(self, request_id: str) -> bool:
        """
        Actually abort a request. Must be called from the step() context.

        Args:
            request_id: The request ID to abort

        Returns:
            True if request was found and aborted, False otherwise
        """
        request = self.requests.get(request_id)
        if request is None:
            return False

        self._clear_request_admission_bookkeeping(request_id)

        # Remove from waiting queue
        if request.status == RequestStatus.WAITING:
            try:
                self.waiting.remove(request)
            except ValueError:
                pass

        # Remove from chunked-prefill queue (if mid-prefill)
        if request_id in self._prefill_states:
            self._prefill_states.pop(request_id, None)
            self.prefilling = deque(
                r for r in self.prefilling if r.request_id != request_id
            )

        # Remove from running (BatchGenerator)
        if request.request_id in self.request_id_to_uid:
            uid = self.request_id_to_uid[request.request_id]
            # Synchronize in-flight GPU work before modifying batch state.
            # batch_generator.remove() triggers lazy KV cache array slicing
            # that replaces references to arrays still used by in-flight
            # Metal command buffers.  Without this barrier the Metal driver
            # can hit 'completeMemory() prepare count underflow'.
            _safe_sync_stream(self._stream)
            self._remove_uid_from_active_batch(uid)
            if hasattr(self.model, "unregister_rope_delta"):
                self.model.unregister_rope_delta(uid)
            if uid < 0:
                mtp_state = self._vlm_mtp_active.pop(uid, None)
                if mtp_state is not None:
                    close = getattr(mtp_state.generator, "close", None)
                    if callable(close):
                        close()
            _unregister_uid_row(self.model, uid)
            del self.uid_to_request_id[uid]
            del self.request_id_to_uid[request.request_id]

        if request_id in self.running:
            del self.running[request_id]

        # Release blocks for eviction (same as _cleanup_finished)
        if self.paged_cache_manager is not None:
            block_table = self.paged_cache_manager.get_block_table(request_id)
            if block_table is None and hasattr(request, "block_table"):
                block_table = request.block_table
            if block_table:
                released = self.paged_cache_manager.release_for_eviction(
                    block_table.block_ids
                )
                if released > 0:
                    logger.debug(
                        f"Released {released} blocks for eviction on abort "
                        f"(request {request_id})"
                    )

        # Clear request entry from block_aware_cache
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear_request_entry(request_id)

        # Clean up streaming detokenizer to prevent state contamination
        self._cleanup_detokenizer(request_id)

        # Clean up protocol-specific output parser session
        self._cleanup_output_parser_session(request_id)

        # Clean up VLM adapter state to prevent contamination
        if hasattr(self.model, "clear_vlm_position_state"):
            self.model.clear_vlm_position_state()
        if hasattr(self.model, "clear_pending_embeddings"):
            self.model.clear_pending_embeddings()

        # Drop any boundary snapshot for this request.
        self._boundary_cache_snapshots.pop(request_id, None)
        if self._boundary_snapshot_store is not None:
            self._boundary_snapshot_store.cleanup_request(request_id)

        # Remove from prefill progress tracker.
        get_prefill_tracker().remove(request_id)

        # Mark as aborted
        request.set_finished(RequestStatus.FINISHED_ABORTED)
        self.finished_req_ids.add(request_id)

        # Remove from requests dict and clear cache references to release
        # MLX arrays promptly (mirrors _cleanup_finished behavior).
        # _cleanup_request (engine_core) no longer calls remove_finished_request,
        # so this is the single cleanup point for aborted requests.
        req_to_remove = self.requests.pop(request_id, None)
        if req_to_remove is not None:
            req_to_remove._extracted_cache = None
            req_to_remove.prompt_cache = None

        logger.debug(f"Aborted request {request_id}")
        return True

    def has_requests(self) -> bool:
        """Check if there are any pending or running requests.

        Also returns True when a deferred Metal cache clear is pending,
        so that the engine loop keeps calling step() until the clear fires.
        Without this, an idle server would never reach the target step and
        stale buffers would accumulate indefinitely.
        """
        return bool(
            self.waiting
            or self.prefilling
            or self.running
            or self._pending_async_removes
            or self._deferred_clear_at is not None
        )

    def _refresh_generation_overflow_recovery_ids(self) -> None:
        """Drop serial-retry markers once the affected requests leave the scheduler."""
        if not self._generation_overflow_recovery_ids:
            return
        active_ids = set(self.running)
        active_ids.update(request.request_id for request in self.waiting)
        active_ids.update(request.request_id for request in self.prefilling)
        self._generation_overflow_recovery_ids.intersection_update(active_ids)

    def _effective_max_num_seqs(self) -> int:
        """Current admission cap, narrowed for models that require serial decode."""
        self._refresh_generation_overflow_recovery_ids()
        if self._serialize_llama4_requests or self._generation_overflow_recovery_ids:
            return 1
        return max(1, self.config.max_num_seqs)

    def fail_all_requests(self) -> list[str]:
        """Remove all running and waiting requests after unrecoverable error.

        Used as a safety net by engine_core when step() raises an
        unexpected exception, to prevent infinite loops.

        Only resets batch_generator (not full cache) because this method
        is called for non-corruption errors — corruption is already
        handled inside step().

        Returns:
            List of failed request IDs.
        """
        failed_ids: list[str] = []
        for request_id in list(self.running):
            failed_ids.append(request_id)
            req = self.requests.pop(request_id, None)
            self._clear_request_admission_bookkeeping(request_id)
            if req is not None:
                req._extracted_cache = None
                req.prompt_cache = None
        self.running.clear()
        for request in list(self.prefilling):
            failed_ids.append(request.request_id)
            req = self.requests.pop(request.request_id, None)
            self._clear_request_admission_bookkeeping(request.request_id)
            if req is not None:
                req._extracted_cache = None
                req.prompt_cache = None
        self.prefilling.clear()
        self._prefill_states.clear()
        for request in list(self.waiting):
            failed_ids.append(request.request_id)
            req = self.requests.pop(request.request_id, None)
            self._clear_request_admission_bookkeeping(request.request_id)
            if req is not None:
                req._extracted_cache = None
                req.prompt_cache = None
        self.waiting.clear()
        # Catch in-flight orphans: a request popped from self.waiting but
        # not yet added to self.running (or self.prefilling) sits as a
        # local in _schedule_waiting. If _do_external_prefill raises, the
        # request is unreachable through the three queues but still lives
        # in self.requests (and the engine_core collector / finished_event
        # for its id is still waiting). Without this pass, fail_all_requests
        # returns an incomplete list and the HTTP request hangs forever.
        #
        # Exclude finished requests still awaiting async cache-store cleanup
        # (those have an entry in ``_inflight_store_futures`` — see
        # ``_cleanup_finished`` line ~5267). They have already emitted a
        # ``finished=True`` output to their collector; ``_drain_pending_async_removes``
        # pops them from ``self.requests`` after the store future completes.
        # Failing them here would append an error output that wins over the
        # success for non-streaming ``generate()`` callers (engine_core
        # returns the last queued output).
        for request_id in list(self.requests):
            if request_id in self._inflight_store_futures:
                continue
            failed_ids.append(request_id)
            req = self.requests.pop(request_id, None)
            self._clear_request_admission_bookkeeping(request_id)
            if req is not None:
                req._extracted_cache = None
                req.prompt_cache = None
        # Clear stale uid mappings for every failed id. Running requests hold
        # real uids; the in-flight orphan above holds the temp_uid assigned at
        # _schedule_waiting (id(request)) that its success-path cleanup never
        # reached. batch_generator is reset below, so these mappings are dead
        # either way. failed_ids excludes _inflight_store_futures ids, so the
        # async-cleanup uids that _drain_pending_async_removes still needs are
        # left intact.
        for rid in failed_ids:
            uid = self.request_id_to_uid.pop(rid, None)
            if uid is not None:
                self.uid_to_request_id.pop(uid, None)
        self._generation_overflow_recovery_ids.difference_update(failed_ids)
        # Reset batch generator only (cache is not corrupted). Every row dies
        # with it; survivors re-register at re-insert.
        _unregister_uid_rows_for_model(self.model)
        self.batch_generator = None
        self._current_sampler_params = None
        # Reclaim fragmented Metal buffers after generation failure.
        # Without this, subsequent requests may hit the same resource
        # limit even though Python references have been cleared.
        # Wrapped in try-except because Metal may already be in an error
        # state — mx.synchronize() or mx.clear_cache() can throw a C++
        # exception that causes SIGABRT if uncaught (#435).
        try:
            _sync_and_clear_cache(self._stream)
        except Exception as e:
            logger.warning(f"Metal cache clear failed during error recovery: {e}")
        return failed_ids

    def get_num_waiting(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """Get number of running requests."""
        return len(self.running)

    def _num_admitted_requests(self) -> int:
        """Return requests already occupying scheduler capacity."""
        return len(self.running) + len(self.prefilling)

    def _preflight_memory_check(
        self, request: "Request"
    ) -> "_PreflightRejection | None":
        """
        Estimate whether prefill would exceed memory limits.

        Computes worst-case peak memory for the last prefill chunk
        (model weights + KV cache + SDPA activation/scratch) and rejects
        if it would exceed the hard limit.

        Mirrors MLX SDPA dispatch closely enough that unsupported prefill
        head dimensions are charged for the unfused fp32 score matrix.

        Returns:
            ``_PreflightRejection`` carrying the message + numeric
            estimated / limit bytes if the request should be rejected,
            otherwise ``None``. The structured return lets the server
            layer populate ``PrefillMemoryExceededError.estimated_bytes``
            / ``limit_bytes`` without parsing the human string.
        """
        if not self._prefill_memory_guard:
            return None
        if self._memory_hard_limit_bytes <= 0:
            return None
        if self.memory_monitor is None:
            return None

        prompt_tokens = request.num_prompt_tokens
        cached_tokens = request.cached_tokens or 0
        new_tokens = max(prompt_tokens - cached_tokens, 0)

        if new_tokens == 0:
            return None

        peak = self.memory_monitor.estimate_prefill_peak_bytes(
            new_tokens, self.config.prefill_step_size, cached_tokens=cached_tokens
        )
        if peak == 0:
            return None  # can't estimate, skip

        current = self._current_usage_bytes()
        estimated = current + peak
        hard_limit = self._memory_hard_limit_bytes

        if estimated > hard_limit:
            # Try LRU eviction first (upstream's predictive-throttle
            # path): if eviction can free enough headroom this raises
            # ``_PrefillEvictionNeeded`` and the request is paused for
            # retry. If eviction can't help (already retried, no idle
            # models), the call is a no-op and we fall through to the
            # typed rejection.
            self._raise_prefill_eviction_if_available(
                request_id=request.request_id,
                current=current,
                target_cap=hard_limit,
                predicted_transient=peak,
                requested_tokens=min(new_tokens, self.config.prefill_step_size),
                reason="prefill_preflight",
            )

            message = self._format_rejection_message(
                estimated=estimated,
                current=current,
                peak=peak,
                hard_limit=hard_limit,
            )
            return _PreflightRejection(
                message=message,
                estimated_bytes=int(estimated),
                limit_bytes=int(hard_limit),
            )
        safety_rejection = self._preflight_safety_rejection(
            num_prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            current_usage_bytes=current,
        )
        if safety_rejection is not None:
            requested_tokens = min(max(1, self._prefill_min_chunk_tokens), new_tokens)
            self._raise_prefill_eviction_if_available(
                request_id=request.request_id,
                current=current,
                target_cap=safety_rejection.limit_bytes,
                predicted_transient=max(
                    0, int(safety_rejection.estimated_bytes) - int(current)
                ),
                requested_tokens=requested_tokens,
                reason="prefill_safety_cap",
            )
            return safety_rejection
        return None

    def _memory_component_limit_for_rejection(self, component_limit: int) -> int:
        if component_limit <= 0:
            return 0
        hot_reserved = max(0, int(self._memory_hot_cache_reserved_bytes or 0))
        if hot_reserved <= 0:
            return component_limit
        return max(1, component_limit - hot_reserved)

    def _format_rejection_message(
        self,
        *,
        estimated: int,
        current: int,
        peak: int,
        hard_limit: int,
    ) -> str:
        """Build the prefill-rejection diagnostic.

        Identifies which of static / dynamic / metal_cap is binding so the
        message can steer the user to the right remedy (close apps for
        dynamic, raise sysctl for metal_cap, raise tier or reduce context
        for static). Component ceilings are propagated by
        ``ProcessMemoryEnforcer._propagate_memory_limit``; if a caller
        wired this scheduler outside that path the components stay 0 and
        we fall back to a generic message.
        """
        from .utils.hardware import format_bytes

        static = self._memory_static_ceiling_bytes
        dynamic = self._memory_dynamic_ceiling_bytes
        metal_cap = self._memory_metal_cap_bytes

        binding: list[str] = []
        if static and self._memory_component_limit_for_rejection(static) == hard_limit:
            binding.append("static")
        if (
            dynamic
            and self._memory_component_limit_for_rejection(dynamic) == hard_limit
        ):
            binding.append("dynamic")
        if (
            metal_cap
            and self._memory_component_limit_for_rejection(metal_cap) == hard_limit
        ):
            binding.append("metal_cap")
        binding_str = "/".join(binding) if binding else "effective"

        # Order remedies by likelihood of helping for the binding cause.
        # Dynamic-bound on a reclaim tier (safe/balanced/aggressive) means
        # reclaimable memory is low right now even though the static cap
        # has room — closing apps raises ``free`` / ``inactive`` and a
        # more aggressive ``memory_guard_tier`` raises the active-reclaim
        # ratio. Dynamic-bound under ``custom`` means the user pinned the
        # ceiling there; the only knob that helps is raising
        # ``custom_ceiling_bytes`` itself. Metal-cap bound means the
        # kernel sysctl is the ceiling, so raising ``iogpu.wired_limit_mb``
        # is the only knob that helps. Static-bound (or no breakdown
        # available) leaves ``memory_guard_tier`` / context length as the
        # levers.
        is_custom = self._memory_guard_tier == "custom"
        if "dynamic" in binding and is_custom:
            advice = (
                f"raise custom_ceiling_bytes in admin Memory settings "
                f"(currently pinned at {format_bytes(dynamic)}), "
                f"or reduce context length"
            )
        elif "dynamic" in binding and static and static > dynamic:
            headroom = max(0, dynamic - current)
            advice = (
                f"close other apps to free RAM "
                f"(static cap is {format_bytes(static)} but only "
                f"{format_bytes(headroom)} is reclaimable right now), "
                f"raise memory_guard_tier (safe → balanced → aggressive), "
                f"or reduce context length"
            )
        elif "metal_cap" in binding:
            advice = (
                f"raise kernel iogpu.wired_limit_mb in Terminal "
                f"(currently caps Metal at {format_bytes(metal_cap)}), "
                f"or reduce context length"
            )
        else:
            advice = (
                "reduce context length or raise memory_guard_tier "
                "(safe → balanced → aggressive)"
            )
        advice = advice[:1].upper() + advice[1:]

        return (
            f"Prefill would require ~{format_bytes(estimated)} peak "
            f"(current {format_bytes(current)} + KV+SDPA {format_bytes(peak)}) "
            f"but {binding_str} ceiling is {format_bytes(hard_limit)}. "
            f"{advice}."
        )

    def preflight_or_raise(
        self,
        *,
        num_prompt_tokens: int,
        cached_tokens: int = 0,
        request_id: str | None = None,
    ) -> None:
        """Pre-StreamingResponse prefill memory check.

        Called from the engine's ``preflight_chat`` / ``preflight_completion``
        before the FastAPI route wraps the body in a ``StreamingResponse``,
        so the typed exception can be mapped to HTTP 400 by the registered
        handler. A no-op when the guard is disabled or the request fits.

        Mirrors the ``_preflight_memory_check`` math but takes token counts
        directly (no Request object) and raises instead of returning a
        message — the in-stream re-check inside ``_schedule_waiting``
        remains as defense-in-depth.
        """
        if not self._prefill_memory_guard:
            return
        if self._memory_hard_limit_bytes <= 0:
            return
        if self.memory_monitor is None:
            return

        new_tokens = max(int(num_prompt_tokens) - max(int(cached_tokens), 0), 0)
        if new_tokens == 0:
            return

        peak = self.memory_monitor.estimate_prefill_peak_bytes(
            new_tokens, self.config.prefill_step_size, cached_tokens=cached_tokens
        )
        if peak == 0:
            return

        current = self._current_usage_bytes(refresh_mlx_active=False)
        if not request_id:
            import uuid as _uuid

            request_id = f"preflight-{_uuid.uuid4().hex[:8]}"

        if current + peak > self._memory_hard_limit_bytes:
            message = self._format_rejection_message(
                estimated=current + peak,
                current=current,
                peak=peak,
                hard_limit=self._memory_hard_limit_bytes,
            )

            logger.warning(
                "Preflight rejected (%d tokens, cached=%d, request_id=%s): %s",
                num_prompt_tokens,
                cached_tokens,
                request_id,
                message,
            )
            raise PrefillMemoryExceededError(
                message=message,
                request_id=request_id,
                estimated_bytes=int(current + peak),
                limit_bytes=int(self._memory_hard_limit_bytes),
            )

        safety_rejection = self._preflight_safety_rejection(
            num_prompt_tokens=num_prompt_tokens,
            cached_tokens=cached_tokens,
            current_usage_bytes=current,
        )
        if safety_rejection is None:
            return

        logger.warning(
            "Preflight safety-cap rejected (%d tokens, cached=%d, "
            "request_id=%s): %s",
            num_prompt_tokens,
            cached_tokens,
            request_id,
            safety_rejection.message,
        )
        raise PrefillMemoryExceededError(
            message=safety_rejection.message,
            request_id=request_id,
            estimated_bytes=safety_rejection.estimated_bytes,
            limit_bytes=safety_rejection.limit_bytes,
        )

    def preflight_eviction_request(
        self,
        *,
        num_prompt_tokens: int,
        cached_tokens: int = 0,
        request_id: str | None = None,
    ) -> PrefillEvictionRequest | None:
        """Return an idle-model eviction request for route-level preflight.

        ``preflight_or_raise`` runs before a ``Request`` is admitted, so it
        cannot use the request-bound ``_raise_prefill_eviction_if_available``.
        The API-facing engines call this first, run the async pool callback if
        needed, then call ``preflight_or_raise`` to re-measure and reject only
        if eviction did not create enough headroom.
        """
        if not self._prefill_memory_guard:
            return None
        if self._memory_hard_limit_bytes <= 0:
            return None
        if self.memory_monitor is None:
            return None

        new_tokens = max(int(num_prompt_tokens) - max(int(cached_tokens), 0), 0)
        if new_tokens == 0:
            return None

        current = self._current_usage_bytes(refresh_mlx_active=False)
        request_id = request_id or "preflight"

        peak = self.memory_monitor.estimate_prefill_peak_bytes(
            new_tokens, self.config.prefill_step_size, cached_tokens=cached_tokens
        )
        if peak and current + peak > self._memory_hard_limit_bytes:
            return PrefillEvictionRequest(
                request_id=request_id,
                model_id=getattr(self.config, "model_name", ""),
                current_bytes=int(current),
                target_cap_bytes=int(self._memory_hard_limit_bytes),
                predicted_transient_bytes=int(peak),
                requested_tokens=int(min(new_tokens, self.config.prefill_step_size)),
                reason="prefill_preflight",
            )

        safety_rejection = self._preflight_safety_rejection(
            num_prompt_tokens=num_prompt_tokens,
            cached_tokens=cached_tokens,
            current_usage_bytes=current,
        )
        if safety_rejection is None:
            return None

        requested_tokens = min(max(1, self._prefill_min_chunk_tokens), new_tokens)
        return PrefillEvictionRequest(
            request_id=request_id,
            model_id=getattr(self.config, "model_name", ""),
            current_bytes=int(current),
            target_cap_bytes=int(safety_rejection.limit_bytes),
            predicted_transient_bytes=max(
                0, int(safety_rejection.estimated_bytes) - int(current)
            ),
            requested_tokens=int(requested_tokens),
            reason="prefill_safety_cap",
        )

    def _preflight_safety_rejection(
        self,
        *,
        num_prompt_tokens: int,
        cached_tokens: int = 0,
        current_usage_bytes: int,
    ) -> _PreflightRejection | None:
        """Predict whether even the safety floor chunk cannot fit.

        This mirrors the mid-prefill ``_guard_prefill_chunk`` rejection, but
        runs before the route returns a ``StreamingResponse``. It charges the
        resident KV that will be allocated by the prompt plus the minimum
        chunk transient at the full prompt context length.
        """
        if self.memory_monitor is None:
            return None
        base_cap, cap, margin = self._prefill_abort_description()
        if cap <= 0:
            return None

        new_tokens = max(int(num_prompt_tokens) - max(int(cached_tokens), 0), 0)
        if new_tokens == 0:
            return None

        floor_chunk = min(max(1, self._prefill_min_chunk_tokens), new_tokens)
        kv_len = max(int(num_prompt_tokens) - 1, 1)
        kv_growth = self.memory_monitor.estimate_prompt_kv_bytes(new_tokens)
        min_transient = self._predicted_chunk_transient(floor_chunk, kv_len)
        if kv_growth <= 0 and min_transient <= 0:
            return None

        estimated = int(current_usage_bytes + kv_growth + min_transient)
        if estimated <= cap:
            return None

        from .utils.hardware import format_bytes

        message = (
            "Prefill context too large for available memory "
            f"(preflight safety guard, kv_len={kv_len}, "
            f"min_chunk={floor_chunk}): predicted peak would require "
            f"~{format_bytes(estimated)} "
            f"(current {format_bytes(current_usage_bytes)} + "
            f"KV {format_bytes(kv_growth)} + "
            f"min-chunk transient {format_bytes(min_transient)}) "
            f"but prefill safety cap is {format_bytes(cap)} "
            f"({round(margin * 100)}% of effective ceiling "
            f"{format_bytes(base_cap)}). Reduce context length, free system "
            "memory, or loosen memory_guard_tier (safe → balanced → aggressive)."
        )
        return _PreflightRejection(
            message=message,
            estimated_bytes=estimated,
            limit_bytes=int(cap),
        )

    def _schedule_waiting(
        self,
    ) -> tuple[list["Request"], list[RequestOutput]]:
        """
        Move requests from waiting queue to running.

        Each request is prefilled externally before being inserted into
        BatchGenerator, so prefill_batch_size=1 is always used. Cache
        status homogeneity tracking is kept for safety since it affects
        how we handle the existing_cache argument.

        Returns:
            Tuple of (scheduled requests, rejected error outputs)
        """
        scheduled = []
        rejected_outputs: list[RequestOutput] = []

        # Track cache status of first scheduled request to ensure homogeneity
        # None = not determined yet, True = has cache, False = no cache
        batch_cache_status: bool | None = None
        # Track VLM status: VLM and text-only requests cannot be in the same prefill batch
        # None = not determined yet, True = VLM request, False = text-only request
        batch_vlm_status: bool | None = None
        # Track SpecPrefill: these requests must be alone (RoPE patching affects whole model)
        batch_specprefill_status: bool | None = None

        while (
            self.waiting
            and self._num_admitted_requests() < self._effective_max_num_seqs()
        ):
            # Admission pause: set by ProcessMemoryEnforcer when phys
            # crosses soft_threshold. New prefills wait; in-flight requests
            # continue. First request always passes (no admitted work yet)
            # so admission can recover by completing the current generation.
            admitted = self._num_admitted_requests()
            if self._admission_paused and admitted:
                logger.debug(
                    "Admission paused by memory pressure, %d admitted",
                    admitted,
                )
                stalled = self._memory_admission_stall_output("admission_paused")
                if stalled is not None:
                    rejected_outputs.append(stalled)
                break

            # Store-cache backpressure: when the post-completion pipeline is
            # at its cleanup cap, defer admitting new prefills instead of
            # blocking the generation step on the store-cache write (#1496).
            # The cap bounds concurrent extracted-KV copies (the #1383 OOM
            # guard) and shrinks under memory pressure via
            # adjust_store_cache_cap. This also applies between sequential
            # turns: a new prefill must not start while async store-cache
            # cleanup still owns too many large cache payloads (#1684).
            gate = self._store_cache_gate
            pending_store_cleanups = len(self._pending_async_removes)
            if gate is not None and (
                not gate.has_capacity or pending_store_cleanups >= gate.cap
            ):
                logger.debug(
                    "Admission deferred: store-cache pipeline full "
                    "(in_flight=%d pending_cleanups=%d cap=%d), %d running",
                    gate.in_flight,
                    pending_store_cleanups,
                    gate.cap,
                    len(self.running),
                )
                memory_related_gate = self._admission_paused
                if (
                    not memory_related_gate
                    and self._prefill_memory_guard
                    and self._memory_limit_bytes > 0
                ):
                    try:
                        memory_related_gate = (
                            self._current_usage_bytes() >= self._memory_limit_bytes
                        )
                    except Exception:
                        memory_related_gate = False
                if memory_related_gate:
                    if self.waiting:
                        self._clear_store_cache_admission_blocker(
                            self.waiting[0].request_id
                        )
                    stalled = self._memory_admission_stall_output(
                        "store_cache_backpressure"
                    )
                    if stalled is not None:
                        rejected_outputs.append(stalled)
                else:
                    if self.waiting:
                        self._clear_memory_admission_blocker(self.waiting[0].request_id)
                    stalled = self._store_cache_admission_stall_output(
                        "store_cache_backpressure",
                        gate_in_flight=gate.in_flight,
                        gate_cap=gate.cap,
                        pending_cleanups=pending_store_cleanups,
                    )
                    if stalled is not None:
                        rejected_outputs.append(stalled)
                break

            # Generation memory guard: when requests are already admitted,
            # defer scheduling if memory pressure is high to prevent
            # Metal allocation failures during batch_generator.next().
            # First request always passes (no admitted work yet).
            if self._prefill_memory_guard and self._memory_limit_bytes > 0 and admitted:
                current = self._current_usage_bytes()
                if current > self._memory_limit_bytes:
                    logger.debug(
                        "Generation memory guard: deferring scheduling "
                        "(%s > %s), %d admitted",
                        current,
                        self._memory_limit_bytes,
                        admitted,
                    )
                    stalled = self._memory_admission_stall_output(
                        "generation_memory_guard"
                    )
                    if stalled is not None:
                        rejected_outputs.append(stalled)
                    break

            request = self.waiting[0]
            self._clear_memory_admission_blocker(request.request_id)
            self._clear_store_cache_admission_blocker(request.request_id)
            if self._should_defer_for_cache_freshness(request):
                break

            request = self.waiting.popleft()
            self._cache_freshness_waits.pop(request.request_id, None)
            self._clear_memory_admission_blocker(request.request_id)
            self._clear_store_cache_admission_blocker(request.request_id)

            # Ensure we have a batch generator
            self._ensure_batch_generator(request.sampling_params)

            if self.batch_generator is None:
                # Put back and try again later
                self.waiting.appendleft(request)
                break

            self._prepare_prefix_cache_for_request(request)

            # Determine tokens to process and cache to use
            # Note: Don't use `remaining_tokens or prompt_token_ids` because empty list
            # is falsy in Python. For exact cache match, remaining_tokens=[] but we should
            # pass just the last token so BatchGenerator can start generation.
            if (
                request.remaining_tokens is not None
                and len(request.remaining_tokens) == 0
            ):
                # Exact cache match - pass only last token for generation kickoff
                tokens_to_process = request.prompt_token_ids[-1:]
            elif request.remaining_tokens:
                tokens_to_process = request.remaining_tokens
            else:
                tokens_to_process = request.prompt_token_ids
            cache_to_use = request.prompt_cache  # May be None

            # Validate cache before using it
            if cache_to_use is not None and not self._validate_cache(cache_to_use):
                logger.debug(
                    f"Request {request.request_id}: invalid cache detected, "
                    f"proceeding without cache"
                )
                cache_to_use = None
                request.prompt_cache = None
                request.cached_tokens = 0
                request.remaining_tokens = request.prompt_token_ids
                tokens_to_process = request.prompt_token_ids

            # SpecPrefill requests must be alone in the batch (RoPE patching
            # affects the entire model). Also block scheduling if another
            # specprefill request is already running (offset RoPE active).
            request_is_specprefill = request.specprefill_indices is not None
            if (
                self._specprefill_active_request_id is not None
                and not request_is_specprefill
            ):
                # A specprefill request is running — defer all others until it finishes
                self.waiting.appendleft(request)
                break
            if batch_specprefill_status is None:
                batch_specprefill_status = request_is_specprefill
            elif batch_specprefill_status != request_is_specprefill:
                self.waiting.appendleft(request)
                break
            if request_is_specprefill and len(scheduled) > 0:
                # SpecPrefill request must be alone
                self.waiting.appendleft(request)
                break

            # Check VLM status homogeneity: VLM and text-only requests use
            # different prefill paths (embeddings vs token IDs)
            request_is_vlm = request.vlm_inputs_embeds is not None
            if batch_vlm_status is None:
                batch_vlm_status = request_is_vlm
            elif batch_vlm_status != request_is_vlm:
                # VLM status mismatch - defer this request to next batch
                self.waiting.appendleft(request)
                logger.debug(
                    f"Deferring request {request.request_id} to next batch "
                    f"(VLM status mismatch: batch={batch_vlm_status}, request={request_is_vlm})"
                )
                break

            # Check cache status homogeneity (kept for consistent prefill behavior)
            request_has_cache = cache_to_use is not None
            if batch_cache_status is None:
                batch_cache_status = request_has_cache
            elif batch_cache_status != request_has_cache:
                # Cache status mismatch - defer this request to next batch
                self.waiting.appendleft(request)
                logger.debug(
                    f"Deferring request {request.request_id} to next batch "
                    f"(cache status mismatch: batch={batch_cache_status}, request={request_has_cache})"
                )
                break

            # Mark as Harmony model if applicable (before think detection)
            if self._is_harmony_model:
                request.is_harmony_model = True

            # Check if prompt ends with <think> token for reasoning models.
            # Must happen before _build_sampler_and_processors so the thinking
            # budget processor can check needs_think_prefix.
            if self._detect_needs_think_prefix(request):
                request.needs_think_prefix = True

            # Per-request sampler/logits processors to avoid BatchGenerator recreation.
            sampler, logits_processors = self._build_sampler_and_processors(
                request.sampling_params, request
            )

            # Pre-flight memory guard: estimate peak memory for this request
            # and reject if it would exceed the hard limit. The check
            # may raise ``_PrefillEvictionNeeded`` (upstream's
            # predictive-throttle path) to pause and retry under
            # eviction headroom; only if eviction can't help does the
            # typed rejection propagate.
            try:
                preflight_rejection = self._preflight_memory_check(request)
            except _PrefillEvictionNeeded as e:
                self._pause_for_prefill_eviction(request, e.request)
                break
            if preflight_rejection is not None:
                logger.warning(
                    f"Request {request.request_id} rejected by prefill "
                    f"memory guard: {preflight_rejection.message}"
                )
                self._release_paged_cache_for_request(request.request_id)
                self.requests.pop(request.request_id, None)
                self._clear_request_admission_bookkeeping(request.request_id)
                rejected_outputs.append(
                    _prefill_memory_error_output(
                        request.request_id,
                        preflight_rejection.message,
                        estimated_bytes=preflight_rejection.estimated_bytes,
                        limit_bytes=preflight_rejection.limit_bytes,
                    )
                )
                continue

            # SpecPrefill: replace tokens with selected subset and pre-fill
            # cache via sparse_prefill before inserting into BatchGenerator.
            #
            # Key design: sparse_prefill processes selected tokens (excluding
            # the last prompt token). BatchGenerator then processes the last
            # prompt token to produce generation logits. This avoids:
            #   - Double-processing the last token (Bug #2)
            #   - Off-by-one RoPE positions (Bug #1)
            #
            # Position math:
            #   sparse_prefill: N' tokens, adjustment = M - N'
            #   We subtract 1: adjustment = M - N' - 1
            #   BatchGenerator last token: pos = N' + (M - N' - 1) = M - 1
            #   First gen token: pos = (N'+1) + (M - N' - 1) = M
            if request.specprefill_indices is not None:
                tracker = get_prefill_tracker()
                model_id = os.path.basename(self.config.model_name.rstrip("/"))
                total_pp = 0
                try:
                    from .patches.specprefill import (
                        _find_attention_layers,
                        _get_attn_module,
                        _OffsetAdjustedRoPE,
                        cleanup_rope,
                        sparse_prefill,
                    )

                    t0 = time.monotonic()

                    sp_cache = make_prompt_cache(self.model)
                    all_tokens = tokens_to_process
                    sys_count = getattr(request, "_specprefill_system_tokens", 0)

                    # Register tracker entry so the dashboard shows the PP
                    # indicator throughout sys + sparse prefill. Denominator
                    # mirrors the last-token removal applied below so the bar
                    # ends cleanly at 100%.
                    sel_list_pre = request.specprefill_indices.tolist()
                    m_pre = len(all_tokens) - sys_count
                    n_eff = len(sel_list_pre) - (
                        1 if (m_pre - 1) in sel_list_pre else 0
                    )
                    total_pp = sys_count + n_eff
                    tracker.update(request.request_id, 0, total_pp, model_id)

                    def _check_specprefill_abort(processed: int) -> None:
                        if request.request_id in self._pending_abort_ids:
                            logger.info(
                                f"SpecPrefill interrupted at {processed}/{total_pp} "
                                f"tokens: request aborted"
                            )
                            tracker.remove(request.request_id)
                            self.waiting.appendleft(request)
                            raise _PrefillAbortedError([], processed)

                    # Phase 1: system prompt full prefill (if not cached)
                    if sys_count > 0:
                        sys_arr = mx.array(all_tokens[:sys_count])
                        step = self.config.prefill_step_size
                        sys_processed = 0
                        spec_sparse_extra = {
                            "prompt_tokens": request.num_prompt_tokens,
                            "system_tokens": request.specprefill_system_end,
                            "conversation_tokens": request.num_prompt_tokens
                            - request.specprefill_system_end,
                            "cached_tokens": request.cached_tokens,
                            "scored_tokens": m_pre,
                            "selected_tokens": n_eff,
                            "keep_percent": (
                                round(n_eff / m_pre * 100) if m_pre > 0 else 0
                            ),
                        }
                        while sys_arr.size > step:
                            _check_specprefill_abort(sys_processed)
                            tracker.update(
                                request.request_id,
                                sys_processed,
                                total_pp,
                                model_id,
                                phase="specprefill_system",
                                detail="system prompt prefill",
                                extra=spec_sparse_extra,
                            )
                            self.model(sys_arr[:step][None], cache=sp_cache)
                            mx.eval([c.state for c in sp_cache])
                            sys_processed += step
                            _check_specprefill_abort(sys_processed)
                            tracker.update(
                                request.request_id,
                                min(sys_processed, total_pp - 1),
                                total_pp,
                                model_id,
                                phase="specprefill_system",
                                detail="system prompt prefill",
                                extra=spec_sparse_extra,
                            )
                            sys_arr = sys_arr[step:]
                            # Use _sync_and_clear_cache() instead of bare
                            # mx.clear_cache() to flush the engine stream
                            # before releasing Metal buffers.  A bare call here
                            # can race with in-flight command buffers submitted
                            # by the preceding mx.eval(), triggering the same
                            # 'completeMemory() prepare count underflow' kernel
                            # panic that #435 fixed elsewhere (#557).
                            _sync_and_clear_cache(self._stream)
                        if sys_arr.size > 0:
                            _check_specprefill_abort(sys_processed)
                            final_sys = int(sys_arr.size)
                            tracker.update(
                                request.request_id,
                                sys_processed,
                                total_pp,
                                model_id,
                                phase="specprefill_system",
                                detail="system prompt prefill",
                                extra=spec_sparse_extra,
                            )
                            self.model(sys_arr[None], cache=sp_cache)
                            mx.eval([c.state for c in sp_cache])
                            sys_processed += final_sys
                            _check_specprefill_abort(sys_processed)
                            tracker.update(
                                request.request_id,
                                min(sys_processed, total_pp - 1),
                                total_pp,
                                model_id,
                                phase="specprefill_system",
                                detail="system prompt prefill",
                                extra=spec_sparse_extra,
                            )
                        logger.info(
                            f"SpecPrefill: system prompt {sys_count} tokens full prefill"
                        )

                    # Phase 2: conversation sparse prefill
                    conv_tokens = all_tokens[sys_count:]
                    selected = request.specprefill_indices
                    conv_len = len(conv_tokens)
                    pos_offset = request.specprefill_position_offset
                    last_idx = conv_len - 1

                    # Remove last token from selected set — BatchGenerator
                    # will process it separately for generation kickoff.
                    selected_list = selected.tolist()
                    if last_idx in selected_list:
                        selected_list.remove(last_idx)
                        selected = mx.array(sorted(selected_list))

                    def _sparse_progress(processed: int, total: int) -> None:
                        _check_specprefill_abort(sys_count + processed)
                        tracker.update(
                            request.request_id,
                            min(sys_count + processed, total_pp - 1),
                            total_pp,
                            model_id,
                            phase="specprefill_sparse",
                            detail="sparse target prefill",
                            extra={
                                "scored_tokens": conv_len,
                                "selected_tokens": int(selected.shape[0]),
                                "keep_percent": (
                                    round(int(selected.shape[0]) / conv_len * 100)
                                    if conv_len > 0
                                    else 0
                                ),
                                "prompt_tokens": request.num_prompt_tokens,
                                "system_tokens": request.specprefill_system_end,
                                "conversation_tokens": request.num_prompt_tokens
                                - request.specprefill_system_end,
                                "cached_tokens": request.cached_tokens,
                            },
                        )

                    sparse_prefill(
                        self.model,
                        conv_tokens,
                        selected,
                        sp_cache,
                        step_size=self.config.prefill_step_size,
                        position_offset=pos_offset,
                        progress_callback=_sparse_progress,
                    )
                    # sparse_prefill installs _OffsetAdjustedRoPE with
                    # adjustment = conv_len - selected_len'. Subtract 1 to account for the
                    # extra token BatchGenerator will process.
                    for _, layer in _find_attention_layers(self.model):
                        attn = _get_attn_module(layer)
                        if (
                            attn
                            and hasattr(attn, "rope")
                            and isinstance(attn.rope, _OffsetAdjustedRoPE)
                        ):
                            attn.rope._adjustment -= 1

                    selected_len = int(selected.shape[0])
                    t_prefill = time.monotonic() - t0
                    total_prompt = request.num_prompt_tokens
                    cached = request.cached_tokens
                    logger.info(
                        f"SpecPrefill: sparse prefill {selected_len}/{conv_len} conv tokens in {t_prefill:.1f}s "
                        f"(total {total_prompt}, cached {cached}, "
                        f"system {sys_count} full, conv {conv_len} sparse)"
                    )

                    # Set up request as if we had a prefix cache hit
                    cache_to_use = sp_cache
                    # Last token for generation kickoff
                    tokens_to_process = all_tokens[-1:]
                    self._specprefill_active_request_id = request.request_id

                    # Mark spec-prefill complete (auto-removes tracker entry).
                    tracker.update(request.request_id, total_pp, total_pp, model_id)

                except _PrefillAbortedError:
                    cleanup_rope(self.model)
                    request.specprefill_indices = None
                    tracker.remove(request.request_id)
                    sp_cache = None
                    sys_arr = None
                    conv_tokens = None
                    selected = None
                    _sync_and_clear_cache(self._stream)
                    self._cleanup_prefill_abort_request(request)
                    continue
                except Exception as e:
                    logger.error(f"SpecPrefill sparse prefill failed: {e}")
                    cleanup_rope(self.model)
                    request.specprefill_indices = None
                    tracker.remove(request.request_id)
                    # Fall through to normal prefill

            # External prefill: process tokens[0:N-1] outside BatchGenerator.
            # Only the last token goes to insert() for the first decode step.
            # SpecPrefill already handled its own prefill above, so skip for those.
            if request.specprefill_indices is None and len(tokens_to_process) > 1:
                vlm_embeds = None
                if request.vlm_inputs_embeds is not None:
                    vlm_embeds = (
                        request.vlm_inputs_embeds,
                        request.vlm_extra_kwargs or {},
                        request.cached_tokens,
                    )

                # Chunked prefill: non-VLM prompts longer than one step are
                # spread across multiple step() calls. The first chunk is run
                # here; subsequent chunks run in _advance_chunked_prefills().
                if (
                    self.config.chunked_prefill
                    and vlm_embeds is None
                    and len(tokens_to_process) > self.config.prefill_step_size + 1
                ):
                    sm = self._build_state_machine(request)
                    per_row_lps = list(logits_processors) if logits_processors else []
                    state = self._begin_prefill(
                        request, tokens_to_process, cache_to_use
                    )
                    state.sampler = sampler
                    state.sm = sm
                    state.per_row_lps = per_row_lps

                    try:
                        done = self._step_prefill_chunk(state)
                    except _PrefillAbortedError:
                        _sync_and_clear_cache(self._stream)
                        self._cleanup_prefill_abort_request(request)
                        continue
                    except _PrefillEvictionNeeded as e:
                        self._release_paged_cache_for_request(request.request_id)
                        self._pause_for_prefill_eviction(
                            request,
                            e.request,
                            reset_chunked_state=True,
                        )
                        break
                    except PrefillMemoryExceededError as e:
                        logger.error(
                            "Chunked prefill (first chunk) capacity rejected "
                            "for %s: %s",
                            request.request_id,
                            e,
                        )
                        self._release_paged_cache_for_request(request.request_id)
                        self.requests.pop(request.request_id, None)
                        self._clear_request_admission_bookkeeping(request.request_id)
                        get_prefill_tracker().remove(request.request_id)
                        _sync_and_clear_cache()
                        rejected_outputs.append(
                            _prefill_memory_exception_output(request.request_id, e)
                        )
                        continue
                    except RuntimeError as e:
                        # Hard memory limit hit on the first chunk.
                        # _step_prefill_chunk updates the PrefillProgressTracker
                        # before the limit check, so without this catch the
                        # tracker entry leaks and stays in the dashboard
                        # forever (#1405). Mirrors the cleanup in
                        # _advance_chunked_prefills (d736bfd).
                        logger.error(
                            "Chunked prefill (first chunk) failed for %s: %s",
                            request.request_id,
                            e,
                        )
                        self._release_paged_cache_for_request(request.request_id)
                        self.requests.pop(request.request_id, None)
                        self._clear_request_admission_bookkeeping(request.request_id)
                        get_prefill_tracker().remove(request.request_id)
                        # Drop Metal cache pool buffers held by the aborted
                        # first chunk's forward / mx.eval transients.
                        _sync_and_clear_cache()
                        if self._requeue_or_fail_prefill(request, e):
                            continue
                        rejected_outputs.append(
                            RequestOutput(
                                request_id=request.request_id,
                                finished=True,
                                finish_reason="error",
                                error=str(e),
                            )
                        )
                        continue

                    if done:
                        self._emit_final_boundary_if_needed(state)
                        _sync_and_clear_cache(self._stream)
                        get_prefill_tracker().remove(request.request_id)
                        self._insert_prefilled_request(request, state, scheduled)
                    else:
                        self.prefilling.append(request)
                        self._prefill_states[request.request_id] = state
                    continue  # Skip normal prefill + insert path

                # Normal (non-chunked) full prefill path.
                # Assign a temporary UID so progress callbacks can map
                # uid→request_id during external prefill. Replaced by the
                # real UID returned from insert().
                temp_uid = id(request)  # unique, won't collide with BatchGenerator UIDs
                self.request_id_to_uid[request.request_id] = temp_uid
                self.uid_to_request_id[temp_uid] = request.request_id

                try:
                    prefilled_cache, last_token = self._do_external_prefill(
                        request,
                        tokens_to_process,
                        cache_to_use,
                        vlm_embeds=vlm_embeds,
                    )
                except _PrefillAbortedError:
                    self._cleanup_prefill_abort_request(request, temp_uid=temp_uid)
                    continue
                except _PrefillEvictionNeeded as e:
                    self.uid_to_request_id.pop(temp_uid, None)
                    self.request_id_to_uid.pop(request.request_id, None)
                    self._release_paged_cache_for_request(request.request_id)
                    get_prefill_tracker().remove(request.request_id)
                    self._pause_for_prefill_eviction(request, e.request)
                    break
                except PrefillMemoryExceededError as e:
                    logger.error(
                        "Prefill capacity rejected for %s: %s",
                        request.request_id,
                        e,
                    )
                    self.uid_to_request_id.pop(temp_uid, None)
                    self.request_id_to_uid.pop(request.request_id, None)
                    self._release_paged_cache_for_request(request.request_id)
                    self.requests.pop(request.request_id, None)
                    self._clear_request_admission_bookkeeping(request.request_id)
                    get_prefill_tracker().remove(request.request_id)
                    _sync_and_clear_cache()
                    rejected_outputs.append(
                        _prefill_memory_exception_output(request.request_id, e)
                    )
                    continue
                except RuntimeError as e:
                    # Hard memory limit hit during external prefill. Without
                    # this catch, the exception bubbles up to step() and then
                    # engine_core's fail_all_requests(), which pops
                    # self.requests but cannot reach the PrefillProgressTracker
                    # singleton, so the dashboard entry leaks across model
                    # reload (#1405). Mirrors the cleanup in
                    # _advance_chunked_prefills (d736bfd).
                    logger.error("Prefill failed for %s: %s", request.request_id, e)
                    self.uid_to_request_id.pop(temp_uid, None)
                    self.request_id_to_uid.pop(request.request_id, None)
                    self._release_paged_cache_for_request(request.request_id)
                    self.requests.pop(request.request_id, None)
                    self._clear_request_admission_bookkeeping(request.request_id)
                    get_prefill_tracker().remove(request.request_id)
                    # Drop Metal cache pool buffers held by the aborted
                    # chunk's forward / mx.eval transients.
                    _sync_and_clear_cache()
                    if self._requeue_or_fail_prefill(request, e):
                        continue
                    rejected_outputs.append(
                        RequestOutput(
                            request_id=request.request_id,
                            finished=True,
                            finish_reason="error",
                            error=str(e),
                        )
                    )
                    continue

                # Clean up temp UID mapping
                del self.uid_to_request_id[temp_uid]
                del self.request_id_to_uid[request.request_id]

                # Prefill complete: remove from progress tracker so dashboard
                # shows "generating" instead of "PP" during decode.
                get_prefill_tracker().remove(request.request_id)

                cache_to_use = prefilled_cache
                tokens_to_process = last_token

            # Capture per-request mRoPE rope_deltas for decode.
            # Prefer _captured_rope_deltas from per-request extra_kwargs
            # (set during get_input_embeddings), since the global
            # _rope_deltas may be stale when explicit position_ids are used.
            if request.vlm_inputs_embeds is not None:
                extra = request.vlm_extra_kwargs or {}
                captured = extra.get("_captured_rope_deltas")
                if captured is not None:
                    if hasattr(captured, "item"):
                        request.rope_deltas = float(captured.item())
                    else:
                        request.rope_deltas = float(captured)
                elif hasattr(self.model, "get_last_rope_deltas"):
                    request.rope_deltas = self.model.get_last_rope_deltas()

            # Build per-request state machine for stop tokens
            sm = self._build_state_machine(request)

            # Set random seed for reproducible generation (best-effort).
            # This affects global MLX random state, so concurrent requests
            # may interfere. Matches OpenAI's best-effort seed semantics.
            if request.sampling_params.seed is not None:
                mx.random.seed(request.sampling_params.seed)

            # TurboQuant KV is quantized at the end of _do_external_prefill
            # (fp16 prefill → quantize once); _merge_caches() turns the per
            # request TQ cache into a BatchTurboQuantKVCache on insert.

            # VLM MTP routing: if a gemma4_assistant drafter is attached, run
            # an extra last-token forward to capture hidden + shared_kv_states,
            # sample the first bonus, and hand the request to a vlm_mtp
            # generator instead of BatchGenerator. Falls through on any
            # eligibility issue so other speculative paths stay intact.
            if self._vlm_mtp_drafter is not None and cache_to_use is not None:
                vlm_mtp_uid = self._route_to_vlm_mtp(
                    request, cache_to_use, tokens_to_process, sampler, sm
                )
                if vlm_mtp_uid is not None:
                    self.request_id_to_uid[request.request_id] = vlm_mtp_uid
                    self.uid_to_request_id[vlm_mtp_uid] = request.request_id
                    now = time.monotonic()
                    request.batch_uid = vlm_mtp_uid
                    request.status = RequestStatus.RUNNING
                    request.generation_started_at = now
                    request.last_activity_at = now
                    self.running[request.request_id] = request
                    scheduled.append(request)
                    self.total_prompt_tokens += request.num_prompt_tokens
                    logger.debug(
                        f"Scheduled request {request.request_id} via vlm_mtp "
                        f"(uid={vlm_mtp_uid}, {request.num_prompt_tokens} prompt tokens)"
                    )
                    continue

            # Insert into BatchGenerator with pre-filled cache + last token.
            # BatchGenerator only handles decode from here.
            #
            # IMPORTANT: ``logits_processors`` MUST be passed as a per-row
            # list (possibly empty), never None.  mlx-lm's
            # GenerationBatch._step does ``for p in self.logits_processors[e]``
            # in any branch where ``any(self.logits_processors)`` is True
            # (e.g., heterogeneous merge with another row that has a
            # processor).  A None slot crashes that loop with
            # ``TypeError: 'NoneType' object is not iterable``, which then
            # bubbles into the engine retry loop and presents as a hang.
            # See vllm-mlx-patched commit 8d4052b for the same root cause
            # in a sibling project, and #934 for the user-visible symptom.
            per_row_lps = list(logits_processors) if logits_processors else []
            uids = self.batch_generator.insert(
                [tokens_to_process],
                max_tokens=[request.sampling_params.max_tokens],
                caches=[cache_to_use] if cache_to_use else None,
                all_tokens=[_batch_generator_all_tokens(request)],
                samplers=[sampler],
                logits_processors=[per_row_lps],
                state_machines=[sm],
            )
            if uids:
                _register_uid_rows(self.model, uids, [sampler], [per_row_lps])
                uid = uids[0]
                self.request_id_to_uid[request.request_id] = uid
                self.uid_to_request_id[uid] = request.request_id
                now = time.monotonic()
                request.batch_uid = uid
                request.status = RequestStatus.RUNNING
                request.generation_started_at = now
                request.last_activity_at = now
                self.running[request.request_id] = request
                scheduled.append(request)

                # Register per-UID rope_delta for mRoPE decode.
                if hasattr(self.model, "register_rope_delta"):
                    self.model.register_rope_delta(uid, request.rope_deltas)

                self.total_prompt_tokens += request.num_prompt_tokens
                cache_info = (
                    f", {request.cached_tokens} cached"
                    if request.cached_tokens > 0
                    else ""
                )
                cache_used = "with cache" if cache_to_use else "no cache"
                logger.debug(
                    f"Scheduled request {request.request_id} (uid={uid}) "
                    f"with {len(tokens_to_process)} tokens to process "
                    f"({request.num_prompt_tokens} total){cache_info}, {cache_used}"
                )

        return scheduled, rejected_outputs

    def _process_batch_responses(
        self, responses: list[Any]
    ) -> tuple[list[RequestOutput], set[str]]:
        """
        Process responses from BatchGenerator.

        Args:
            responses: List of BatchGenerator.Response objects

        Returns:
            Tuple of (outputs, finished_request_ids)
        """
        outputs = []
        finished_ids = set()

        step_now = time.monotonic()
        generated_at = time.perf_counter()
        for response in responses:
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                continue

            request = self.running.get(request_id)
            if request is None:
                continue

            request.last_activity_at = step_now
            completion_tokens_before = request.num_output_tokens

            # Release VLM embeddings after first decode token (prefill is done)
            if request.vlm_inputs_embeds is not None:
                request.vlm_inputs_embeds = None
                request.vlm_extra_kwargs = None

            # Check finish reason first - don't include EOS token in output
            # (following mlx-lm's batch_generate behavior)
            is_stop = response.finish_reason == "stop"
            is_length = response.finish_reason == "length"
            is_finished = response.finish_reason is not None

            # Only append token if not stopping due to EOS token
            new_text = ""

            # Check if this request uses a protocol-specific output parser
            parser_session = self._get_output_parser_session(request_id)

            if parser_session is not None and not is_stop:
                parser_result = parser_session.process_token(response.token)
                new_text = parser_result.stream_text
                if parser_result.visible_text:
                    request.output_text += parser_result.visible_text

                # Parser-defined stop token can override finish reason
                if parser_result.is_stop and not is_finished:
                    is_finished = True
                    is_stop = True
                    response.finish_reason = "stop"

                should_record_token = (
                    parser_result.record_token
                    if parser_result.record_token is not None
                    else not is_stop
                )
                if should_record_token:
                    request.append_output_token(response.token)

            elif not is_stop:
                # Standard processing without a protocol parser
                request.append_output_token(response.token)

                # Decode the new token using streaming detokenizer for proper UTF-8 handling
                detokenizer = self._get_detokenizer(request_id)
                if detokenizer is not None:
                    detokenizer.add_token(response.token)
                    new_text = detokenizer.last_segment
                else:
                    # Fallback to single-token decode
                    new_text = self.tokenizer.decode([response.token])

                # Text-level stop-string fallback. Catches BPE edge cases
                # where the tokenized stop sequence does not match the
                # model's actual output tokens (e.g. " delta" vs "delta").
                # Only scans the tail to keep cost O(stop_len) per step.
                stop_strs = request.sampling_params.stop or []
                if stop_strs and not is_finished and detokenizer is not None:
                    full_text = detokenizer.text
                    prev_len = len(full_text) - len(new_text)
                    for ss in stop_strs:
                        if not ss:
                            continue
                        scan_start = max(0, prev_len - len(ss) + 1)
                        idx_in_tail = full_text.find(ss, scan_start)
                        if idx_in_tail < 0:
                            continue
                        is_finished = True
                        is_stop = True
                        response.finish_reason = "stop"
                        if idx_in_tail >= prev_len:
                            new_text = new_text[: idx_in_tail - prev_len]
                        else:
                            new_text = ""
                        break

            # Prepend <think> tag for first chunk if this is a reasoning model.
            # Protocol parsers may expose a normalized prefix when their prompt
            # uses a model-specific open-think marker (e.g. MiniMax <mm:think>).
            if getattr(request, "needs_think_prefix", False):
                if not getattr(request, "think_prefix_sent", False):
                    if parser_session is None:
                        think_tag = getattr(self.tokenizer, "think_start", "<think>")
                        prefix_text = think_tag + "\n"
                    else:
                        prefix_text = (
                            self._get_output_parser_thinking_start_output_text() or ""
                        )
                    if prefix_text:
                        new_text = prefix_text + new_text
                        if parser_session is not None:
                            request.output_text = prefix_text + request.output_text
                    request.think_prefix_sent = True

            # Immediately discard logprobs if not requested to free memory (~800KB per response)
            # This prevents accumulation of large MLX arrays during streaming
            if (
                hasattr(response, "logprobs")
                and response.logprobs is not None
                and not request.sampling_params.logprobs
            ):
                response.logprobs = None

            # Create output
            output_generated_at = (
                generated_at
                if request.num_output_tokens > completion_tokens_before
                else None
            )
            output = RequestOutput(
                request_id=request_id,
                new_token_ids=[response.token] if not is_stop else [],
                new_text=new_text,
                output_token_ids=list(request.output_token_ids),
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
                generated_at=output_generated_at,
                generated_until=output_generated_at,
                cached_tokens=request.cached_tokens,
            )

            if not is_finished:
                self._maybe_capture_boundary_snapshot(request, response.uid)

            # Handle finished requests
            if is_finished:
                if is_stop:
                    request.set_finished(RequestStatus.FINISHED_STOPPED)
                elif is_length:
                    request.set_finished(RequestStatus.FINISHED_LENGTH_CAPPED)

                output.finished = True
                output.finish_reason = response.finish_reason
                finished_ids.add(request_id)

                if parser_session is not None:
                    final_result = parser_session.finalize()
                    if final_result.stream_text:
                        output.new_text += final_result.stream_text
                    if final_result.visible_text:
                        request.output_text += final_result.visible_text
                    if final_result.output_text_prefix:
                        request.output_text = (
                            final_result.output_text_prefix + request.output_text
                        )
                    if final_result.tool_calls:
                        output.tool_calls = final_result.tool_calls
                    if final_result.finish_reason:
                        output.finish_reason = final_result.finish_reason
                    output.output_text = request.output_text
                else:
                    # Standard finalization without a protocol parser
                    # Finalize detokenizer to flush any remaining bytes
                    detokenizer = self._get_detokenizer(request_id)
                    if detokenizer is not None:
                        detokenizer.finalize()
                        final_segment = detokenizer.last_segment
                        if final_segment:
                            output.new_text += final_segment

                    # Decode full output
                    output.output_text = self.tokenizer.decode(request.output_token_ids)
                    request.output_text = output.output_text

                    # Trim accumulated output text at the first stop string
                    # match so non-streaming responses do not include the
                    # stop sequence itself (matches OpenAI semantics).
                    if is_stop:
                        stop_strs = request.sampling_params.stop or []
                        for ss in stop_strs:
                            if not ss:
                                continue
                            cut = output.output_text.find(ss)
                            if cut >= 0:
                                output.output_text = output.output_text[:cut]
                                request.output_text = output.output_text
                                break

                # Extract cache for future reuse.
                # In the new API, prompt_cache is a direct value (not callable).
                raw_cache = getattr(response, "prompt_cache", None)
                if raw_cache is not None:
                    try:
                        # SpecPrefill: sparse KV data can't be stored in
                        # paged cache (hash mismatch with full token IDs).
                        if request.specprefill_indices is not None:
                            raw_cache = None

                        # For paged cache, extract actual tensor states
                        # This allows cache to survive BatchGenerator recreation
                        elif self.block_aware_cache is not None:
                            extracted_cache, model_cache_config = (
                                self._extract_cache_states(raw_cache)
                            )
                            if extracted_cache:
                                request._extracted_cache = extracted_cache
                                request._model_cache_config = model_cache_config
                                logger.debug(
                                    f"Extracted {len(extracted_cache)} layer states "
                                    f"for request {request_id}"
                                )
                        else:
                            # Standard cache stores object references
                            request._extracted_cache = raw_cache
                            request._model_cache_config = None
                    except Exception as e:
                        logger.debug(f"Failed to extract cache for {request_id}: {e}")

                self.total_completion_tokens += request.num_output_tokens
                self.num_requests_processed += 1

                logger.debug(
                    f"Request {request_id} finished: {response.finish_reason}, "
                    f"{request.num_output_tokens} tokens"
                )
                logger.log(
                    5, "Request %s generated text:\n%s", request_id, output.output_text
                )

            outputs.append(output)

        return outputs, finished_ids

    def _release_paged_cache_for_request(self, request_id: str) -> None:
        """Drop a request's paged-cache footprint on rejection paths.

        ``add_request`` routes through ``block_aware_cache.fetch_cache``
        which records the request in ``_request_tables`` and increments
        ref counts on every prefix-matched paged-cache block. The normal
        completion path releases that state in ``_cleanup_finished``;
        the prefill-rejection paths in ``_advance_chunked_prefills`` /
        ``_schedule_waiting`` must do the same or rejected requests
        leak block refs (pinning the paged cache and compounding the
        very memory pressure that triggered the rejection) and orphan
        ``_request_tables`` entries.
        """
        if self.block_aware_cache is not None:
            self.block_aware_cache.release_cache(request_id)
        elif self.paged_cache_manager is not None:
            self.paged_cache_manager.delete_block_table(request_id)

        # SpecPrefill primes an independent ``_draft_prefix_cache`` in
        # ``_try_specprefill_scoring`` whose block refs are tracked
        # separately from the target ``block_aware_cache``. Without
        # releasing it on the rejection path a rejected SpecPrefill
        # request leaks every draft-block ref symmetric to the
        # target-cache leak the main branch above guards against.
        draft_cache = getattr(self, "_draft_prefix_cache", None)
        if draft_cache is not None:
            try:
                draft_cache.release_cache(request_id)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Draft prefix cache release_cache(%s) raised; ignoring",
                    request_id,
                    exc_info=True,
                )

    def _cleanup_finished(self, finished_ids: set[str]) -> None:
        """Clean up finished requests and store caches for reuse."""
        # Synchronize pending engine stream operations before cache storage.
        # store_cache -> mx.save_safetensors triggers implicit mx.eval() which
        # can conflict with async Metal operations on the generation stream.
        if finished_ids:
            with self._phase_timer("cleanup_finished_sync"):
                _safe_sync_stream(self._stream)

        # SpecPrefill: restore original RoPE if active request finished
        for rid in finished_ids:
            self._cleanup_specprefill(rid)

        # Remove finished requests from prefill progress tracker.
        tracker = get_prefill_tracker()
        for rid in finished_ids:
            tracker.remove(rid)

        for request_id in finished_ids:
            request = self.running.get(request_id)

            # Store cache for future reuse (G2-async): submit to background
            # executor so the post-finish 28GB+ memcpy doesn't block response
            # streaming. The inference thread does mx.synchronize +
            # boundary merge + a single batched mx.eval here; the worker
            # handles _extract_tensor_bytes (CPU memcpy) + index/queue
            # registration. batch_generator.remove(uid) is deferred and
            # picked up at the next step's _drain_pending_async_removes.
            store_future = None
            if request is not None and request.prompt_token_ids:
                if self.block_aware_cache is not None:
                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ):
                        prompt_boundary_store = None
                    else:
                        uid_for_store = self.request_id_to_uid.get(request_id, -1)
                        prompt_boundary_store = (
                            self._prepare_prompt_boundary_cache_store(
                                request_id,
                                request,
                                uid_for_store,
                            )
                        )

                    if (
                        hasattr(request, "_extracted_cache")
                        and request._extracted_cache is not None
                    ) or prompt_boundary_store is not None:
                        try:
                            full_token_sequence = list(request.prompt_token_ids) + list(
                                request.output_token_ids
                            )

                            if prompt_boundary_store is not None:
                                (
                                    token_sequence_to_store,
                                    cache_to_store,
                                    model_cache_config,
                                    intermediate_snapshots,
                                ) = prompt_boundary_store
                                cacheable_sequence = list(token_sequence_to_store)
                            else:
                                # For reasoning models, only cache prompt tokens.
                                # Output contains <think> tokens that the API layer
                                # strips before the next turn, so they never match.
                                if getattr(request, "needs_think_prefix", False):
                                    cacheable_sequence = list(request.prompt_token_ids)
                                else:
                                    cacheable_sequence = full_token_sequence
                                token_sequence_to_store = cacheable_sequence
                                cache_to_store = request._extracted_cache
                                model_cache_config = getattr(
                                    request, "_model_cache_config", None
                                )
                                intermediate_snapshots = None

                            # Inference-thread store_cache prep, timed as
                            # three sub-phases (boundary / collect / dispatch)
                            # mirroring boundary_capture_* granularity.
                            # The dispatch phase does a FULL mx.eval (not
                            # async_eval) so the KV arrays are concrete on THIS
                            # thread before the store-cache worker slices/views
                            # them. MLX streams are thread-local; a lazy op left
                            # for the worker to materialize re-dispatches to this
                            # thread's stream index, which is absent on the worker
                            # -> SIGABRT. See the dispatch-phase comment below.
                            with mx.stream(self._stream):
                                with self._phase_timer("store_cache_main_boundary"):
                                    if prompt_boundary_store is None:
                                        boundary_override = (
                                            self._get_boundary_store_override(
                                                request_id,
                                                cacheable_sequence,
                                            )
                                        )
                                        if (
                                            boundary_override is None
                                            and self._detect_boundary_snapshot_need()
                                        ):
                                            # Non-sliceable cache state is only
                                            # storable from boundary-aligned
                                            # snapshots; the live state sits at
                                            # the current decode offset (which
                                            # speculative decode can leave off
                                            # the emitted count entirely).
                                            raise _BoundaryStoreUnavailable()
                                        if boundary_override is not None:
                                            (
                                                token_sequence_to_store,
                                                boundary_cache,
                                                boundary_model_config,
                                                intermediate_snapshots,
                                            ) = boundary_override
                                            cache_to_store = (
                                                self._merge_boundary_with_full_cache(
                                                    boundary_cache,
                                                    request._extracted_cache,
                                                )
                                            )
                                            if boundary_model_config is not None:
                                                model_cache_config = (
                                                    boundary_model_config
                                                )
                                            logger.info(
                                                f"Using boundary cache snapshot for {request_id}: "
                                                f"storing {len(token_sequence_to_store)}/"
                                                f"{len(full_token_sequence)} tokens "
                                                f"(skipping trailing partial block, "
                                                f"{len(intermediate_snapshots) if intermediate_snapshots else 0} "
                                                f"intermediate snapshots)"
                                            )
                                # DEBUG-only divergence probe (issue #1003).
                                # Record the exact token sequence submitted to
                                # store_cache, including boundary truncation.
                                if logger.isEnabledFor(logging.DEBUG):
                                    self._cache_probe_seqs.append(
                                        (
                                            request.request_id,
                                            list(token_sequence_to_store),
                                        )
                                    )
                                with self._phase_timer("store_cache_main_collect"):
                                    pre_eval_arrays = (
                                        self._collect_arrays_from_extracted_cache(
                                            cache_to_store
                                        )
                                    )
                                    if intermediate_snapshots is not None:
                                        for (
                                            snapshot_cache
                                        ) in (
                                            intermediate_snapshots.iter_in_memory_extracted()
                                        ):
                                            pre_eval_arrays.extend(
                                                self._collect_arrays_from_extracted_cache(
                                                    snapshot_cache
                                                )
                                            )
                                with self._phase_timer("store_cache_main_dispatch"):
                                    if pre_eval_arrays:
                                        # FULL eval (not async_eval) on the owner
                                        # thread. MLX streams are thread-local:
                                        # these KV arrays carry self._stream
                                        # (a per-engine ThreadLocalStream created
                                        # on THIS thread). The store-cache worker
                                        # later slices them (_extract_block_tensor_slice)
                                        # and views bf16->uint16 (_extract_tensor_bytes);
                                        # if the source op is still LAZY at that
                                        # point, materializing it on the worker
                                        # re-dispatches to self._stream's index,
                                        # which does not exist on the worker thread
                                        # -> "There is no Stream(gpu, N) in current
                                        # thread" -> std::terminate -> SIGABRT.
                                        # Forcing concrete materialization here
                                        # means every downstream worker op consumes
                                        # an already-evaluated buffer and binds its
                                        # own new ops to the always-present default
                                        # stream (gpu,0). The big host memcpy
                                        # (bytes(memoryview(...))) and the disk
                                        # write stay on the worker — only the GPU
                                        # completion fence moves onto this thread.
                                        mx.eval(*pre_eval_arrays)

                            hot_cache_write_back = (
                                not self._bypass_hot_cache_under_pressure()
                            )
                            if not hot_cache_write_back:
                                logger.info(
                                    "Using SSD write-through for %s "
                                    "under memory pressure",
                                    request_id,
                                )

                            if self._store_cache_executor is not None:
                                # Hand host memcpy and disk write to the
                                # background executor after the owner thread
                                # has materialized KV arrays. The gate counts
                                # cleanup slots that still own extracted cache
                                # references; backpressure is applied at
                                # admission in _schedule_waiting so cache
                                # persistence does not wait in the token loop
                                # after submission (#1496). note_submitted is
                                # called before submit, and note_done happens in
                                # _drain_pending_async_removes after the request
                                # cache references are released.
                                gate = self._store_cache_gate
                                if gate is not None:
                                    gate.note_submitted()
                                try:
                                    store_future = self._store_cache_executor.submit(
                                        self._async_store_cache_worker,
                                        request_id,
                                        token_sequence_to_store,
                                        cache_to_store,
                                        model_cache_config,
                                        intermediate_snapshots,
                                        request.vlm_extra_keys_for_cache,
                                        request.vlm_extra_key_token_start_for_cache,
                                        request.vlm_extra_key_ranges_for_cache,
                                        hot_cache_write_back,
                                    )
                                except BaseException:
                                    if gate is not None:
                                        gate.note_done()
                                    raise
                                self._inflight_store_futures[request_id] = store_future
                                self._inflight_store_info[request_id] = (
                                    _InflightStoreInfo(
                                        tokens=list(token_sequence_to_store),
                                        extra_keys=request.vlm_extra_keys_for_cache,
                                        extra_key_token_start=(
                                            request.vlm_extra_key_token_start_for_cache
                                        ),
                                        extra_key_ranges=(
                                            request.vlm_extra_key_ranges_for_cache
                                        ),
                                    )
                                )
                            else:
                                # Executor unavailable — synchronous fallback.
                                self._async_store_cache_worker(
                                    request_id,
                                    token_sequence_to_store,
                                    cache_to_store,
                                    model_cache_config,
                                    intermediate_snapshots,
                                    request.vlm_extra_keys_for_cache,
                                    request.vlm_extra_key_token_start_for_cache,
                                    request.vlm_extra_key_ranges_for_cache,
                                    hot_cache_write_back,
                                )
                            logger.debug(
                                f"Submitted async store_cache for {request_id} "
                                f"({len(token_sequence_to_store)} tokens, "
                                f"{len(full_token_sequence)} total: "
                                f"{len(request.prompt_token_ids)} prompt + "
                                f"{len(request.output_token_ids)} output)"
                            )
                        except _BoundaryStoreUnavailable:
                            logger.debug(
                                "Skipping cache store for %s: no boundary-aligned "
                                "snapshot for non-sliceable cache state (all "
                                "captures skipped, e.g. by the speculative-decode "
                                "skew guard); storing live state would corrupt "
                                "later prefix hits",
                                request_id,
                            )
                            block_table = None
                            if self.paged_cache_manager:
                                block_table = self.paged_cache_manager.get_block_table(
                                    request_id
                                )
                            if block_table and self.paged_cache_manager:
                                self.paged_cache_manager.release_for_eviction(
                                    block_table.block_ids
                                )
                            self.block_aware_cache.clear_request_entry(request_id)
                        except Exception as e:
                            logger.debug(
                                f"Failed to submit async store for {request_id}: {e}"
                            )
                    else:
                        # No extracted_cache to store, but ensure block leak guard.
                        block_table = None
                        if self.paged_cache_manager:
                            block_table = self.paged_cache_manager.get_block_table(
                                request_id
                            )
                            if block_table is None and hasattr(request, "block_table"):
                                block_table = request.block_table
                        if block_table and self.paged_cache_manager:
                            self.paged_cache_manager.release_for_eviction(
                                block_table.block_ids
                            )
                        self.block_aware_cache.clear_request_entry(request_id)

            # Remove from running
            if request_id in self.running:
                del self.running[request_id]

            # batch_generator.remove(uid): defer until the async store_cache
            # worker finishes so the BatchKVCache slot isn't reused while the
            # worker is still reading buffer references via cache_to_store.
            # _drain_pending_async_removes (next step) handles the actual
            # mx.synchronize + remove + uid_maps cleanup. If we have no async
            # store (no extracted_cache, executor missing, fallback fail),
            # fall back to immediate remove for back-compat behavior.
            if request_id in self.request_id_to_uid:
                uid = self.request_id_to_uid[request_id]
                if store_future is not None:
                    self._pending_async_removes.append((uid, request_id, store_future))
                else:
                    # Synchronize in-flight GPU work before modifying batch state.
                    # batch_generator.remove() triggers lazy KV cache array slicing
                    # (BatchKVCache.filter) that replaces references to arrays still
                    # used by in-flight Metal command buffers from the previous
                    # batch_generator.next() call.  Without this barrier the Metal
                    # driver can hit 'completeMemory() prepare count underflow'.
                    _safe_sync_stream(self._stream)
                    self._remove_uid_from_active_batch(uid)
                    if hasattr(self.model, "unregister_rope_delta"):
                        self.model.unregister_rope_delta(uid)
                    _unregister_uid_row(self.model, uid)
                    if uid in self.uid_to_request_id:
                        del self.uid_to_request_id[uid]
                    del self.request_id_to_uid[request_id]

            # Clean up streaming detokenizer
            self._cleanup_detokenizer(request_id)

            # Clean up protocol-specific output parser session
            self._cleanup_output_parser_session(request_id)

            # Clean up VLM adapter state (position_ids, rope_deltas, pending embeddings)
            if hasattr(self.model, "clear_vlm_position_state"):
                self.model.clear_vlm_position_state()
            if hasattr(self.model, "clear_pending_embeddings"):
                self.model.clear_pending_embeddings()

            # Drop any boundary snapshot for this request. The in-memory
            # dict pop is safe — the async store worker holds its own
            # reference to the snapshot dict via _BoundarySnapshotProvider.
            self._boundary_cache_snapshots.pop(request_id, None)
            # cleanup_request rmtree's the on-disk snapshot directory and
            # races the worker's boundary_snapshot_store.load() calls. If
            # an async store_future is in flight, defer cleanup until the
            # worker finishes (handled in _drain_pending_async_removes).
            if self._boundary_snapshot_store is not None and store_future is None:
                self._boundary_snapshot_store.cleanup_request(request_id)

            # Track as finished
            self.finished_req_ids.add(request_id)

            # Remove from requests dict to prevent memory leak.
            # When async store_cache is in flight, keep _extracted_cache alive
            # until the worker finishes — the worker holds a reference via
            # cache_to_store argument, but request._extracted_cache pointing
            # to the same data is the canonical owner. We pop here only when
            # no future is pending; the future's done callback (or
            # _drain_pending_async_removes) clears the request later.
            if store_future is None:
                req_to_remove = self.requests.pop(request_id, None)
                self._clear_request_admission_bookkeeping(request_id)
                if req_to_remove is not None:
                    req_to_remove._extracted_cache = None
                    req_to_remove.prompt_cache = None
            else:
                # Drop request from running but keep in self.requests so the
                # async worker keeps the cache buffers alive via reachability.
                # Cleanup happens in _drain_pending_async_removes.
                pass

        # Emit phase timing diagnostics when accumulated counts are meaningful.
        # Helps diagnose cache-on overhead (boundary capture / store_cache /
        # hot cache eviction). Logged at info level so operators can see it
        # without enabling debug.
        if finished_ids and self._phase_total_ms:
            stats_parts = []
            for phase, total_ms in sorted(self._phase_total_ms.items()):
                count = self._phase_count.get(phase, 0)
                if count == 0:
                    continue
                stats_parts.append(f"{phase}={total_ms:.1f}ms/{count}")
            if stats_parts:
                logger.info("Cache phase timings: %s", ", ".join(stats_parts))

        # Schedule deferred Metal cache cleanup after request completion.
        if finished_ids:
            # Schedule deferred Metal cache cleanup instead of clearing immediately.
            # Immediate mx.clear_cache() after request completion races with IOKit's
            # asynchronous completeMemory() callbacks — the kernel-level GPU memory
            # reference counting can still be in-flight even after mx.synchronize()
            # returns, causing 'prepare count underflow' kernel panics (#435).
            # Deferring by _DEFERRED_CLEAR_DELAY generation steps (~10-40 ms) gives
            # IOKit time to process callbacks while still reclaiming buffers fast
            # enough to prevent TTFT spikes from pool bloat (#411).
            #
            # Use max() so that concurrent completions (max_num_seqs > 1) each get
            # a full _DEFERRED_CLEAR_DELAY window counted from *their own* finish
            # step.  The old "only set if None" guard meant the second request's
            # window was anchored to the first request's finish step, allowing the
            # second request's KV cache blocks to be re-allocated before IOKit
            # finished their completeMemory() callbacks (#557).
            target = self._step_counter + self._DEFERRED_CLEAR_DELAY
            if self._deferred_clear_at is None or target > self._deferred_clear_at:
                self._deferred_clear_at = target

    def _is_cache_corruption_error(self, error: Exception) -> bool:
        """Check if an error indicates cache corruption."""
        return is_cache_corruption_error(error)

    def _is_generation_overflow_error(self, error: Exception) -> bool:
        """Check for MLX/libc++ unordered-container overflow during decode."""
        return isinstance(
            error, OverflowError
        ) and self._GENERATION_OVERFLOW_PATTERN in str(error)

    def _recover_from_cache_error(self) -> None:
        """Recover from cache corruption error."""
        # Clear batch generator (this is the source of the corruption)
        self.batch_generator = None
        self._current_sampler_params = None
        self._boundary_cache_snapshots.clear()
        if self._boundary_snapshot_store is not None:
            self._boundary_snapshot_store.cleanup_all()
        self._boundary_snapshot_required = None

        # Clear stale VLM position state to prevent re-corruption on retry
        if hasattr(self.model, "clear_vlm_position_state"):
            self.model.clear_vlm_position_state()

        # Clear pending VLM embeddings
        if hasattr(self.model, "clear_pending_embeddings"):
            self.model.clear_pending_embeddings()

        # Clear caches
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()
        self._cache_rate_tracker.clear()

        # Clear UID mappings
        _unregister_uid_rows_for_model(self.model)
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()

        # Cancel any pending deferred Metal cache clear
        self._deferred_clear_at = None

        # Clear detokenizer state to prevent contamination after recovery
        self._request_detokenizers.clear()

        # Clear protocol-specific output parser sessions
        self._output_parser_sessions.clear()

        logger.info("Cache recovery completed")

    def _recover_from_generation_overflow_error(self) -> None:
        """Reset decode state after MLX __next_prime overflow."""
        self.batch_generator = None
        self._current_sampler_params = None
        self._boundary_snapshot_required = None

        active_specprefill = self._specprefill_active_request_id
        if active_specprefill is not None:
            self._cleanup_specprefill(active_specprefill)

        if hasattr(self.model, "clear_vlm_position_state"):
            self.model.clear_vlm_position_state()
        if hasattr(self.model, "clear_pending_embeddings"):
            self.model.clear_pending_embeddings()

        _unregister_uid_rows_for_model(self.model)
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()
        self._deferred_clear_at = None
        self._request_detokenizers.clear()
        self._output_parser_sessions.clear()

        try:
            _sync_and_clear_cache(self._stream)
        except Exception as e:
            logger.warning(
                "Metal cache clear failed during generation overflow recovery: %s",
                e,
            )

        logger.info("Generation overflow recovery completed")

    def _reset_request_for_reprefill(self, request: Request) -> None:
        """Reset request-owned decode state so it can be prefilled again."""
        request.status = RequestStatus.WAITING
        request.batch_uid = None
        request.prompt_cache = None
        request.cached_tokens = 0
        request.remaining_tokens = request.prompt_token_ids
        request.block_table = None
        request.shared_prefix_blocks = 0
        request.output_token_ids = []
        request.output_text = ""
        request.num_computed_tokens = 0
        request._extracted_cache = None
        request._model_cache_config = None
        request.think_prefix_sent = False

    def _reschedule_running_requests(
        self, is_corruption: bool = False, max_corruption_retries: int = 3
    ) -> list[str]:
        """Move running requests back to waiting queue for retry.

        Args:
            is_corruption: If True, increment corruption retry counter and
                fail requests that exceed max_corruption_retries.
            max_corruption_retries: Max corruption retries before failing a request.

        Returns:
            List of request IDs that exceeded max retries (corruption only).
        """
        failed_ids: list[str] = []
        count = 0
        for request_id, request in list(self.running.items()):
            if is_corruption:
                request.cache_corruption_retries += 1
                if request.cache_corruption_retries > max_corruption_retries:
                    failed_ids.append(request_id)
                    del self.running[request_id]
                    # Clean up from requests dict (prevent memory leak)
                    req = self.requests.pop(request_id, None)
                    self._clear_request_admission_bookkeeping(request_id)
                    if req is not None:
                        req._extracted_cache = None
                        req.prompt_cache = None
                    continue

            self._reset_request_for_reprefill(request)

            # Move to waiting queue (at front for priority)
            self.waiting.appendleft(request)
            del self.running[request_id]
            count += 1

        if count > 0:
            logger.info(f"Rescheduled {count} requests for re-prefill")
        return failed_ids

    def _reschedule_generation_overflow_requests(
        self,
        max_generation_overflow_retries: int = _MAX_GENERATION_OVERFLOW_RETRIES,
    ) -> list[str]:
        """Retry active requests serially after MLX generation overflow."""
        retry_candidates: list[Request] = []
        seen: set[str] = set()
        prefilling_ids = {request.request_id for request in self.prefilling}
        waiting_ids = {request.request_id for request in self.waiting}

        def collect(request: Request | None) -> None:
            if request is None:
                return
            request_id = request.request_id
            if request_id in seen or request_id in self._inflight_store_futures:
                return
            if request.is_finished():
                return
            seen.add(request_id)
            retry_candidates.append(request)

        for request in self.running.values():
            collect(request)
        for request in self.prefilling:
            collect(request)
        for request_id, request in self.requests.items():
            if request_id in self.running or request_id in prefilling_ids:
                continue
            if request_id in waiting_ids:
                continue
            collect(request)

        collected_ids = {request.request_id for request in retry_candidates}
        for request_id in collected_ids:
            self.running.pop(request_id, None)
            self._prefill_states.pop(request_id, None)
        if collected_ids:
            self.prefilling = deque(
                request
                for request in self.prefilling
                if request.request_id not in collected_ids
            )

        failed_ids: list[str] = []
        retryable: list[Request] = []
        self._generation_overflow_recovery_ids.clear()
        for request in retry_candidates:
            request.generation_overflow_retries += 1
            request_id = request.request_id
            self._boundary_cache_snapshots.pop(request_id, None)
            if self._boundary_snapshot_store is not None:
                self._boundary_snapshot_store.cleanup_request(request_id)
            get_prefill_tracker().remove(request_id)
            if request.generation_overflow_retries > max_generation_overflow_retries:
                failed_ids.append(request_id)
                req = self.requests.pop(request_id, None)
                self._clear_request_admission_bookkeeping(request_id)
                if req is not None:
                    req._extracted_cache = None
                    req.prompt_cache = None
                continue

            self._reset_request_for_reprefill(request)
            retryable.append(request)
            self._generation_overflow_recovery_ids.add(request_id)

        for request in reversed(retryable):
            self.waiting.appendleft(request)

        if retryable:
            logger.info(
                "Rescheduled %d request(s) for serial generation-overflow retry",
                len(retryable),
            )
        return failed_ids

    # Max times a single request is requeued after a prefill memory-pressure
    # failure before we give up and emit a clean error to the client.
    _MAX_PREFILL_OOM_RETRIES = 2

    def _requeue_or_fail_prefill(self, request: "Request", error: Exception) -> bool:
        """Decide whether to requeue a prefill that hit the memory ceiling.

        The three #1405 catch sites have already torn the request down
        (released paged cache, popped ``self.requests``, removed the prefill
        tracker entry, cleared Metal). This either resets the request and puts
        it back on the waiting queue for a fresh attempt (returns ``True`` —
        caller continues without emitting an error), or — when the retry
        budget is exhausted or the failure is not a memory-pressure error —
        returns ``False`` so the caller emits the clean
        ``finish_reason="error"``.

        Only memory-limit failures are retried; any other RuntimeError fails
        immediately so genuine model errors don't loop.
        """
        if "Memory limit exceeded" not in str(error):
            return False
        if request.prefill_oom_retries >= self._MAX_PREFILL_OOM_RETRIES:
            logger.warning(
                "Prefill for %s exhausted %d memory-pressure retries; "
                "failing with a clean error.",
                request.request_id,
                self._MAX_PREFILL_OOM_RETRIES,
            )
            return False
        request.prefill_oom_retries += 1

        # Reclaim before requeue so the retry starts from a lower baseline.
        self._reclaim_prefill_headroom()

        # Clear any SpecPrefill RoPE patch tied to this request so the retry
        # re-scores cleanly.
        if self._specprefill_active_request_id == request.request_id:
            self._specprefill_active_request_id = None

        # Restore mRoPE deltas if an external VLM prefill was interrupted before
        # its own restore ran (value stashed on the request in
        # _do_external_prefill). Benign for non-VLM requests (stash is None).
        saved = getattr(request, "_prefill_saved_rope_deltas", None)
        if saved is not None:
            lm = getattr(self.model, "_language_model", None)
            if lm is not None and hasattr(lm, "_rope_deltas"):
                lm._rope_deltas = saved
            request._prefill_saved_rope_deltas = None

        # Reset scheduling + cache + output state to a clean pre-prefill state
        # (mirrors _reschedule_running_requests). We deliberately drop
        # cached_tokens / block_table so the retry does a cold full prefill and
        # does not re-attach the large cached prefix that produced the same
        # oversized SDPA span. VLM inputs/embeds are preserved.
        request.status = RequestStatus.WAITING
        request.batch_uid = None
        request.prompt_cache = None
        request.cached_tokens = 0
        request.remaining_tokens = request.prompt_token_ids
        request.block_table = None
        request.shared_prefix_blocks = 0
        request.output_token_ids = []
        request.output_text = ""
        request.num_computed_tokens = 0
        request._extracted_cache = None
        request._model_cache_config = None
        request.think_prefix_sent = False

        # Re-register (the catch site popped it) and requeue at the front. The
        # retry is throttled from its first chunk by the now-populated transient
        # EWMA, so it is strictly better-informed than this attempt.
        self.requests[request.request_id] = request
        self.waiting.appendleft(request)
        logger.warning(
            "Requeued %s for prefill retry %d/%d after memory pressure.",
            request.request_id,
            request.prefill_oom_retries,
            self._MAX_PREFILL_OOM_RETRIES,
        )
        return True

    def _pause_for_prefill_eviction(
        self,
        request: "Request",
        eviction: PrefillEvictionRequest,
        *,
        reset_chunked_state: bool = False,
    ) -> None:
        """Hold a request until EngineCore can evict idle models asynchronously."""
        self._pending_prefill_eviction_request = eviction
        request.status = RequestStatus.WAITING
        request.batch_uid = None
        if reset_chunked_state:
            self._prefill_states.pop(request.request_id, None)
            try:
                self.prefilling.remove(request)
            except ValueError:
                pass
            request.prompt_cache = None
            request.cached_tokens = 0
            request.remaining_tokens = request.prompt_token_ids
            request.block_table = None
            request.shared_prefix_blocks = 0
            get_prefill_tracker().remove(request.request_id)
        self.waiting.appendleft(request)
        logger.info(
            "Paused request %s for prefill LRU eviction (reason=%s)",
            request.request_id,
            eviction.reason,
        )

    def step(self) -> SchedulerOutput:
        """
        Execute one scheduling step with automatic error recovery.

        This method:
        1. Schedules waiting requests into the batch
        2. Runs one generation step via BatchGenerator
        3. Processes outputs and handles finished requests
        4. On cache corruption: clears all cache and reschedules requests
           for re-prefill (no error raised to caller)

        Returns:
            SchedulerOutput with results of this step
        """
        output = SchedulerOutput()

        # Process pending aborts FIRST (thread-safe with hybrid executor)
        self._process_pending_aborts()

        # Drain a deferred between-turn reclaim requested by the memory
        # enforcer (only acts when the scheduler is idle).
        self._process_pending_reclaim()

        # Drain async store_cache completions from prior steps. Each completed
        # entry triggers the deferred batch_generator.remove(uid) on the
        # inference thread. Inflight entries are left for a later step.
        drained_async_removes = self._drain_pending_async_removes()
        if drained_async_removes:
            output.has_work = True

        # Check memory pressure and evict if needed (tiered cache)
        if self.memory_monitor is not None:
            self._check_memory_pressure()

        try:
            # Advance in-flight chunked prefills (one chunk per request).
            # Must run before _schedule_waiting() so that completing prefills
            # are inserted into BatchGenerator before the decode step.
            chunked_scheduled: list[Request] = []
            chunked_rejected: list[RequestOutput] = []
            if self.prefilling:
                self._advance_chunked_prefills(chunked_scheduled, chunked_rejected)

            # Schedule waiting requests
            scheduled, rejected = self._schedule_waiting()
            # Merge chunked-prefill completions into the scheduled list.
            if chunked_scheduled:
                scheduled = chunked_scheduled + scheduled
            output.scheduled_request_ids = [r.request_id for r in scheduled]
            output.num_scheduled_tokens = sum(r.num_prompt_tokens for r in scheduled)
            if chunked_rejected:
                output.outputs.extend(chunked_rejected)
                output.has_work = True
            if rejected:
                output.outputs.extend(rejected)
                output.has_work = True
            if self._pending_prefill_eviction_request is not None:
                output.prefill_eviction_request = self._pending_prefill_eviction_request
                self._pending_prefill_eviction_request = None
                output.has_work = True

            # Run generation step if we have running requests.
            # Use next_generated() which returns only GenerationBatch.Response
            # objects (prefill is handled externally before insert).
            if (
                self.batch_generator is not None or self._vlm_mtp_active
            ) and self.running:
                if self.batch_generator is not None:
                    responses = list(self.batch_generator.next_generated())
                else:
                    responses = []
                # Drive vlm_mtp generators alongside BatchGenerator. Order
                # matters only for log determinism; _process_batch_responses
                # is per-uid.
                if self._vlm_mtp_active:
                    responses.extend(self._step_vlm_mtp())
                output.has_work = True

                if responses:
                    outputs, finished_ids = self._process_batch_responses(responses)
                    output.outputs.extend(outputs)
                    output.finished_request_ids.update(finished_ids)

                    # Periodic decode cache materialization for models whose
                    # KV cache update graph can otherwise grow for thousands of
                    # tokens. MiniMax-M3 has one lazy cache-update chain per
                    # layer; evaluating the cache state periodically cuts those
                    # references before Metal's resource-count limit is hit.
                    self._tokens_since_kv_cache_eval = getattr(
                        self, "_tokens_since_kv_cache_eval", 0
                    ) + len(responses)
                    kv_eval_interval = self._decode_eval_kv_cache_interval
                    if (
                        kv_eval_interval > 0
                        and self._tokens_since_kv_cache_eval >= kv_eval_interval
                    ):
                        with mx.stream(self._stream):
                            evaluated = _eval_generation_batch_cache(
                                self.batch_generator
                            )
                        logger.debug(
                            "Materialized decode KV cache state: %d arrays",
                            evaluated,
                        )
                        self._tokens_since_kv_cache_eval = 0

                    self._cleanup_finished(finished_ids)

                    # Periodic Metal allocator cleanup during long decodes.
                    # mx.random.categorical inside the sampler allocates a
                    # tiny scalar via gumbel → uniform on every call.
                    # omlx ships its own non-compiled sampler
                    # (omlx/utils/sampling.py) so that RNG state actually
                    # advances in the server, but the trade-off is that
                    # those scalars accumulate in the IOGPU residency set
                    # — macOS aborts at ~4096 entries. Long contexts
                    # (50k+) decoding thousands of tokens hit that limit
                    # mid-stream. Synchronise the generation stream first
                    # so any in-flight Metal command buffer that still
                    # references buffers we're about to drop has
                    # completed; the allocator only releases pool entries
                    # whose ref count is zero, but the sync guarantees
                    # there is no race window. Decode-only path —
                    # next_generated() returns nothing during prefill, so
                    # we never disrupt prefill activation buffers.
                    self._tokens_since_clear_cache = getattr(
                        self, "_tokens_since_clear_cache", 0
                    ) + len(responses)
                    if self._tokens_since_clear_cache >= 1024:
                        _sync_and_clear_cache(self._stream)
                        self._tokens_since_clear_cache = 0

        except _PrefillAbortedError:
            # Prefill was interrupted by a pending abort.
            # BatchGenerator is in an inconsistent state (partial
            # prefill), so reset it entirely. Pending aborts will
            # be processed at the start of the next step().
            self.batch_generator = None
            self._current_sampler_params = None
            self._boundary_cache_snapshots.clear()
            if self._boundary_snapshot_store is not None:
                self._boundary_snapshot_store.cleanup_all()
            self._boundary_snapshot_required = None
            # Move any running requests back to waiting so they
            # can be rescheduled with a fresh BatchGenerator.
            self._reschedule_running_requests()

        except (TypeError, AttributeError, ValueError) as e:
            if self._is_cache_corruption_error(e):
                import traceback

                logger.warning(
                    f"Cache corruption detected: {e}, "
                    f"clearing cache and re-prefilling..."
                )
                logger.debug(f"Cache corruption traceback:\n{traceback.format_exc()}")
                # Full reset: clear batch generator, all caches, VLM state
                self._recover_from_cache_error()
                # Reschedule requests for re-prefill from scratch.
                # Requests exceeding max corruption retries are failed.
                failed_ids = self._reschedule_running_requests(is_corruption=True)
                for rid in failed_ids:
                    output.outputs.append(
                        RequestOutput(
                            request_id=rid,
                            finished=True,
                            finish_reason="error",
                            error=(
                                f"Cache corruption not recoverable "
                                f"after retries: {e}"
                            ),
                        )
                    )
                    output.finished_request_ids.add(rid)
            else:
                raise

        except OverflowError as e:
            if self._is_generation_overflow_error(e):
                import traceback

                logger.warning(
                    "Generation overflow detected: %s; resetting decode state "
                    "and retrying affected requests serially",
                    e,
                )
                logger.debug(
                    "Generation overflow traceback:\n%s", traceback.format_exc()
                )
                self._recover_from_generation_overflow_error()
                failed_ids = self._reschedule_generation_overflow_requests()
                for rid in failed_ids:
                    output.outputs.append(
                        RequestOutput(
                            request_id=rid,
                            finished=True,
                            finish_reason="error",
                            error=(
                                "Generation overflow not recoverable after "
                                f"serial retry: {e}"
                            ),
                        )
                    )
                    output.finished_request_ids.add(rid)
                output.has_work = True
            else:
                raise

        except Exception as e:
            import traceback

            logger.error(
                f"Error in batch generation step: {e}\n" f"{traceback.format_exc()}"
            )
            raise

        # Clear finished tracking for next step
        self.finished_req_ids = set()
        self._refresh_generation_overflow_recovery_ids()

        # Periodic Metal cache cleanup
        self._step_counter += 1
        should_clear = self._should_periodic_clear_cache()
        # Deferred post-completion cleanup: fire once the step counter reaches
        # the target set by _cleanup_finished() (#435, #557).
        if (
            self._deferred_clear_at is not None
            and self._step_counter >= self._deferred_clear_at
        ):
            should_clear = True
            self._deferred_clear_at = None
        if should_clear:
            _sync_and_clear_cache(self._stream)
        if (
            self.config.gc_cleanup_interval > 0
            and self._step_counter % self.config.gc_cleanup_interval == 0
        ):
            gc.collect()

        self._publish_admin_snapshot()

        return output

    def _publish_admin_snapshot(self) -> None:
        """Atomically publish a fresh admin-visible snapshot.

        Called from step() on the engine thread, where running/waiting are
        not concurrently mutated. The admin endpoint reads the reference via
        snapshot_for_admin() and never iterates the live structures.
        """
        self._admin_snapshot = {
            "running_by_id": dict(self.running),
            "waiting": list(self.waiting),
        }

    def snapshot_for_admin(self) -> dict[str, Any]:
        """Return the most recently published admin snapshot.

        Reference read is GIL-atomic; the dict itself is no longer mutated
        after publication. May be one step stale, which is fine for dashboard
        polling.
        """
        return self._admin_snapshot

    def get_request(self, request_id: str) -> Request | None:
        """Get a request by ID."""
        return self.requests.get(request_id)

    def remove_finished_request(self, request_id: str) -> Request | None:
        """Remove a finished request from tracking."""
        request = self.requests.pop(request_id, None)
        self._clear_request_admission_bookkeeping(request_id)
        return request

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        stats = {
            "num_waiting": len(self.waiting),
            "num_prefilling": len(self.prefilling),
            "num_running": len(self.running),
            "num_requests_processed": self.num_requests_processed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }
        # Include cache stats
        if self.block_aware_cache is not None:
            stats["ssd_cache"] = self.block_aware_cache.get_stats()
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get cache statistics."""
        if self.block_aware_cache is not None:
            return self.block_aware_cache.get_stats()
        return None

    def reset(self) -> None:
        """Reset the scheduler state."""
        # Drain any pending deferred aborts
        self._pending_abort_ids.clear()

        # Abort all requests directly (reset is synchronous)
        for request_id in list(self.requests.keys()):
            self._do_abort_request(request_id)

        self.waiting.clear()
        self.prefilling.clear()
        self._prefill_states.clear()
        self.running.clear()
        self.requests.clear()
        self.finished_req_ids.clear()
        _unregister_uid_rows_for_model(self.model)
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()
        self._generation_overflow_recovery_ids.clear()
        # Async store_cache bookkeeping. shutdown() drains these before us,
        # but clear here too so reset() is safe to call standalone (e.g. tests
        # or recovery paths) without leaking Request refs through stale futures.
        self._pending_async_removes.clear()
        self._inflight_store_futures.clear()
        self._inflight_store_info.clear()
        self._cache_freshness_waits.clear()
        self._prefix_cache_prepared.clear()
        self.batch_generator = None
        self._current_sampler_params = None
        self._boundary_cache_snapshots.clear()
        if self._boundary_snapshot_store is not None:
            self._boundary_snapshot_store.cleanup_all()
        self._boundary_snapshot_required = None

        # Clear caches
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()
        self._cache_rate_tracker.clear()

        # Clear detokenizers
        self._request_detokenizers.clear()

        # Clear protocol-specific output parser sessions
        self._output_parser_sessions.clear()

        # Cancel any pending deferred Metal cache clear
        self._deferred_clear_at = None

    def deep_reset(self) -> None:
        """
        Deep reset that clears ALL cache state including model-level caches.

        This is more aggressive than reset() and should be used when
        switching engines or recovering from errors.
        """
        # Standard reset first
        self.reset()

        # Clear any model-level cache state
        # MLX models may have internal cache references
        if hasattr(self.model, "cache"):
            self.model.cache = None

        # Some MLX models store cache in layers
        if hasattr(self.model, "layers"):
            for layer in self.model.layers:
                if hasattr(layer, "cache"):
                    layer.cache = None
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "cache"):
                    layer.self_attn.cache = None

        # Release model and tokenizer references for GC
        self.model = None
        self.tokenizer = None

        # Release all cache-related references for GC
        self.paged_cache_manager = None
        self.block_aware_cache = None
        self.memory_monitor = None
        self._boundary_snapshot_store = None

        # Force garbage collection of any lingering cache objects
        import gc

        gc.collect()

        logger.info("Deep reset completed - all caches cleared")

    def shutdown(self) -> None:
        """
        Graceful shutdown.

        Flushes hot cache to SSD and closes the background writer.
        paged SSD cache files are NOT cleared to allow reuse on reload.
        """
        logger.info("Scheduler shutdown initiated...")
        # The store-cache gate is a non-blocking counter (#1496), so there is
        # no step-thread caller to wake here. Inflight futures are drained
        # below before the executor is asked to shut down.
        # Wait for any inflight async store_cache futures + drain pending
        # batch_generator removes so the writer thread / underlying paged SSD
        # cache see all blocks before close().
        if self._store_cache_executor is not None:
            try:
                inflight = list(self._inflight_store_futures.values())
                if inflight:
                    logger.info(
                        "Waiting for %d inflight async store_cache future(s)...",
                        len(inflight),
                    )
                    _done, not_done = concurrent.futures.wait(
                        inflight, timeout=FATAL_TEARDOWN_TIMEOUT_S
                    )
                    if not_done:
                        fatal_exit(
                            "Scheduler shutdown timed out after "
                            f"{FATAL_TEARDOWN_TIMEOUT_S:.0f}s waiting for "
                            f"{len(not_done)} async store_cache future(s)"
                        )
                self._drain_pending_async_removes()
                self._store_cache_executor.shutdown(wait=False)
                # Final drain after the bounded wait. If all workers finished
                # before the timeout, skipped entries are now drainable. If not,
                # fatal_exit() above terminates the process instead of leaving
                # a partially torn-down engine alive.
                self._drain_pending_async_removes()
            except Exception as e:
                logger.warning(f"Async store_cache shutdown error: {e}")
            self._store_cache_executor = None
            self._store_cache_gate = None
            self._inflight_store_futures.clear()
            self._inflight_store_info.clear()
            self._cache_freshness_waits.clear()
            self._prefix_cache_prepared.clear()
        if self.paged_ssd_cache_manager is not None:
            self.paged_ssd_cache_manager.close()
            self.paged_ssd_cache_manager = None
        # Release whatever the per-path unregisters did not reach, so nothing
        # survives this engine in the module-level row registry.
        _unregister_uid_rows_for_model(self.model)
        logger.info("Scheduler shutdown completed")

    def adjust_store_cache_cap(self, pressure_level: str) -> None:
        """Resize the store-cache gate based on memory pressure (#1383).

        Called from ProcessMemoryEnforcer on every poll. The cap walks one
        step per poll toward its target so transient spikes don't oscillate
        the cap. Bounded by [1, max_num_seqs]:
        - ok pressure: grow cap back toward max_num_seqs.
        - soft/hard pressure: shrink cap so KV cache backlog fits the system.
        """
        gate = self._store_cache_gate
        if gate is None:
            return
        current = gate.cap
        if pressure_level == "ok":
            new = min(self.config.max_num_seqs, current + 1)
        else:
            new = max(1, current - 1)
        if new != current:
            gate.set_cap(new)
            logger.debug(
                "store-cache queue cap: %d -> %d (pressure=%s)",
                current,
                new,
                pressure_level,
            )

    # =========================================================================
    # SSD Cache Methods
    # =========================================================================

    def _set_model_info_for_monitor(self) -> None:
        """Extract model info and set it on memory monitor for estimation."""
        if self.memory_monitor is None:
            return

        try:
            # Try to get model config
            config = None
            if hasattr(self.model, "config"):
                config = self.model.config
            elif hasattr(self.model, "args"):
                config = self.model.args

            if config is None:
                logger.debug("Could not extract model config for memory estimation")
                return

            def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)

            # VLM / multimodal configs (e.g. Qwen3.6-VL, Gemma-4) nest the
            # language-model dimensions under a sub-config. Prefer
            # ``text_config`` / ``language_config`` / ``llm_config`` when ANY
            # of them exposes the LM layer count, even if the top-level config
            # also has one — on some VLM packs (older Gemma-3, certain Llava /
            # HF auto-wrappers) the top-level field refers to the *vision
            # encoder*, not the LM, and accepting it silently miscalibrates
            # the SDPA-peak estimate by a constant factor (a 40-layer LM
            # wrapped in a 33-layer vision tower under-estimates by ~20 %).
            # Probe both ``num_hidden_layers`` and the legacy ``n_layer`` alias
            # so a GPT-style nested config is also picked up. Falls back to the
            # top-level config only when no sub-config has either field.
            for sub_attr in ("text_config", "language_config", "llm_config"):
                sub = _cfg_get(config, sub_attr)
                if sub is not None and (
                    _cfg_get(sub, "num_hidden_layers") or _cfg_get(sub, "n_layer")
                ):
                    config = sub
                    break

            # Extract KV cache dimensions
            num_layers = _cfg_get(config, "num_hidden_layers") or _cfg_get(
                config, "n_layer"
            )
            num_kv_heads = (
                _cfg_get(config, "num_key_value_heads")
                or _cfg_get(config, "num_attention_heads")
                or _cfg_get(config, "n_head")
            )
            head_dim = _cfg_get(config, "head_dim")
            hidden_size = _cfg_get(config, "hidden_size") or _cfg_get(config, "n_embd")

            # Calculate head_dim if not directly available
            if head_dim is None and hidden_size and num_kv_heads:
                num_heads = _cfg_get(config, "num_attention_heads") or num_kv_heads
                head_dim = hidden_size // num_heads

            # Determine base dtype size for uncompressed KV cache elements.
            base_dtype_size: float = 2  # Default float16/bfloat16
            if hasattr(self.model, "dtype"):
                if self.model.dtype == mx.float32:
                    base_dtype_size = 4
                elif self.model.dtype == mx.bfloat16:
                    base_dtype_size = 2
            dtype_size = base_dtype_size

            # Extract num_attention_heads (query heads) for SDPA peak estimation
            num_attention_heads = (
                _cfg_get(config, "num_attention_heads")
                or _cfg_get(config, "n_head")
                or num_kv_heads
            )

            # Count KVCache layers for hybrid models
            cache_list_for_tq = None
            actual_kv_cache_layers = None
            num_kv_cache_layers = num_layers
            if not hasattr(self.model, "make_cache"):
                actual_kv_cache_layers = num_layers
            else:
                try:
                    cache_list = self.model.make_cache()
                    cache_list_for_tq = cache_list
                    from mlx_lm.models.cache import CacheList, KVCache

                    def _count_kv(c: Any) -> int:
                        if type(c) is KVCache:
                            return 1
                        if isinstance(c, CacheList):
                            return sum(_count_kv(inner) for inner in c.caches)
                        return 0

                    actual_kv_cache_layers = sum(_count_kv(c) for c in cache_list)
                    num_kv_cache_layers = actual_kv_cache_layers
                    if num_kv_cache_layers == 0:
                        num_kv_cache_layers = num_layers  # fallback
                except Exception:
                    pass

            if (
                self._turboquant_kv_bits is not None
                and isinstance(head_dim, int)
                and not isinstance(head_dim, bool)
                and head_dim > 0
                and isinstance(actual_kv_cache_layers, int)
                and actual_kv_cache_layers > 0
                and (
                    self._turboquant_eligible(cache_list_for_tq)
                    if cache_list_for_tq is not None
                    else not (
                        self._model_uses_mla() or self._model_uses_attention_sinks()
                    )
                )
            ):
                tq_dtype_size = float(self._turboquant_kv_bits) / 8.0 + (2.0 / head_dim)
                if (
                    self._turboquant_skip_last
                    and not isinstance(actual_kv_cache_layers, bool)
                    and actual_kv_cache_layers > 1
                ):
                    dtype_size = (
                        (actual_kv_cache_layers - 1) * tq_dtype_size + base_dtype_size
                    ) / actual_kv_cache_layers
                else:
                    dtype_size = tq_dtype_size

            kv_bytes_per_token = (
                estimate_mla_kv_bytes_per_token(
                    config,
                    cache_list_for_tq,
                    base_dtype_size,
                )
                if estimate_mla_kv_bytes_per_token is not None
                else None
            )

            # Truthiness alone isn't enough — MagicMock proxies leaking
            # through the descent (test scaffolds that don't fully spec
            # ``model.config``) are truthy but fail any later numeric
            # comparison (``> 128`` etc.) deep inside MemoryMonitor.
            # Insist on real positive integers before calling.
            def _pos_int(v: Any) -> bool:
                return isinstance(v, int) and not isinstance(v, bool) and v > 0

            if _pos_int(num_layers) and _pos_int(num_kv_heads) and _pos_int(head_dim):
                self.memory_monitor.set_model_info(
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    dtype_size=dtype_size,
                    num_attention_heads=num_attention_heads,
                    num_kv_cache_layers=num_kv_cache_layers,
                    # SDPA scores are materialized at the compute/activation
                    # dtype, not the (possibly fractional TurboQuant) KV width.
                    compute_dtype_size=base_dtype_size,
                    kv_bytes_per_token=kv_bytes_per_token,
                )
                logger.debug(
                    f"Model info for memory estimation: "
                    f"layers={num_layers} ({num_kv_cache_layers} KVCache), "
                    f"kv_heads={num_kv_heads}, q_heads={num_attention_heads}, "
                    f"head_dim={head_dim}, dtype_size={dtype_size}"
                )
            else:
                logger.debug(
                    f"Incomplete model info: layers={num_layers}, "
                    f"kv_heads={num_kv_heads}, head_dim={head_dim}"
                )

        except Exception as e:
            logger.debug(f"Failed to extract model info: {e}")

    def _infer_live_layer_cache_types(self) -> list[str] | None:
        """Infer the layer-cache signature that future SSD saves will use."""
        if not HAS_CACHE_TYPE_HANDLERS or ModelCacheConfig is None:
            return None

        make_cache = getattr(self.model, "make_cache", None)
        if not callable(make_cache):
            return None

        try:
            cache_list = make_cache()
        except Exception as e:
            logger.debug("Failed to build cache list for SSD signature: %s", e)
            return None

        if not isinstance(cache_list, (list, tuple)) or not cache_list:
            return None

        cache_list = list(cache_list)
        try:
            model_cache_config = ModelCacheConfig.from_cache_list(
                cache_list,
                model_name=self.config.model_name or "",
            )
            layer_cache_types = model_cache_config.get_type_names()
        except Exception as e:
            logger.debug("Failed to infer SSD layer cache signature: %s", e)
            return None

        if not layer_cache_types:
            return None

        if self._turboquant_kv_bits is None:
            return layer_cache_types

        try:
            if not self._turboquant_eligible(cache_list):
                return layer_cache_types
        except Exception as e:
            logger.debug("Failed to evaluate TurboQuant SSD signature: %s", e)
            return layer_cache_types

        kv_indices = [
            i for i, c in enumerate(cache_list) if _is_turboquant_kv_family_cache(c)
        ]
        skip_last = self._turboquant_skip_last and len(kv_indices) > 1
        last_kv_idx = kv_indices[-1] if skip_last else -1
        for idx in kv_indices:
            if idx != last_kv_idx and idx < len(layer_cache_types):
                layer_cache_types[idx] = "TurboQuantKVCache"

        return layer_cache_types

    def refresh_ssd_layer_signature(self) -> list[str] | None:
        """Set the SSD manager's live layer signature before prefix lookup."""
        manager = self.paged_ssd_cache_manager
        if manager is None:
            return None

        layer_cache_types = self._infer_live_layer_cache_types()
        if not layer_cache_types:
            return None

        try:
            set_signature = getattr(manager, "set_expected_layer_signature", None)
            if callable(set_signature):
                set_signature(layer_cache_types)
            else:
                manager.adopt_layer_signature_if_unset(layer_cache_types)
            manager.invalidate_stale_layer_signature()
        except Exception as e:
            logger.warning("Failed to refresh SSD layer cache signature: %s", e)
            return None

        return layer_cache_types

    def _init_tiered_cache(self) -> bool:
        """Initialize paged SSD cache components if configured.

        In paged SSD-only mode:
        - All KV cache data is stored on paged SSD via PagedSSDCacheManager
        - PagedCacheManager only stores block metadata (no GPU memory for cache data)
        - BatchGenerator handles GPU memory for active inference
        """
        if not HAS_TIERED_CACHE:
            if self.config.paged_ssd_cache_dir:
                logger.warning(
                    "paged SSD cache requested but ssd_cache/memory_monitor modules "
                    "not available. Install required dependencies."
                )
            return False

        # In paged SSD-only mode, paged_ssd_cache_dir is required
        if not self.config.paged_ssd_cache_dir:
            logger.debug(
                "paged SSD cache not configured (no --ssd-cache-dir specified)"
            )
            return False

        try:
            cache_dir = (
                Path(self.config.paged_ssd_cache_dir)
                if self.config.paged_ssd_cache_dir
                else None
            )

            # Pass current model identity so stale blocks from a prior model
            # version (e.g., 30-layer cache after an upgrade to 40 layers via
            # #1404) are unlinked at startup instead of triggering a layer
            # mismatch reject on every prefix lookup. See #1413.
            expected_num_layers = (
                self.block_aware_cache.expected_num_layers
                if self.block_aware_cache is not None
                else 0
            )

            # Pending-writes queue sizing depends on per-block bytes
            # (block_size × per-token KV). Pass the *final* scheduler
            # block size — possibly adjusted from the config default by
            # RotatingKVCache / ArraysCache logic earlier in
            # ``__init__`` — and a model-derived per-token KV estimate
            # from the memory monitor.
            #
            # Gate on ``has_model_info()`` rather than just non-None:
            # ``estimate_block_memory(1)`` silently substitutes a
            # 7B-class fiction (32 layers × 8 KV heads × 128 head_dim
            # ≈ 128 KB/token) when dims were never set, and feeding
            # that "default" value into the writer-queue formula gives
            # the wrong cap on real workloads. When dims are missing
            # (test fixtures with skeletal model.config, unusual VLM
            # packs the nested-config walk doesn't recognise), pass
            # the PagedSSDCacheManager's 200 KB default explicitly so
            # the cap math degrades to a known constant instead of a
            # model-class fiction. The auto-init in ``Scheduler.__init__``
            # paired with ``_set_model_info_for_monitor()`` means the
            # happy path here is ``has_model_info() is True``; this
            # else branch only fires for skeletal test fixtures.
            if self.memory_monitor is not None and self.memory_monitor.has_model_info():
                # ``estimate_block_memory(1)`` returns all-layers K+V
                # bytes for a single token at the dtype the monitor was
                # configured with — exactly the per-token cost the
                # queue cap needs to weigh.
                expected_kv_bytes_per_token = self.memory_monitor.estimate_block_memory(
                    1
                )
            else:
                expected_kv_bytes_per_token = 200_000  # PagedSSDCacheManager default

            # Initialize paged SSD cache manager for SSD storage
            self.paged_ssd_cache_manager = PagedSSDCacheManager(
                cache_dir=cache_dir,
                max_size_bytes=self.config.paged_ssd_cache_max_size,
                hot_cache_max_bytes=self.config.hot_cache_max_size,
                hot_cache_only=self.config.hot_cache_only,
                hot_cache_budget=self.config.hot_cache_budget,
                expected_model_name=self.config.model_name or "",
                expected_num_layers=expected_num_layers,
                expected_block_size=self.config.paged_cache_block_size,
                expected_block_size_tokens=self.config.paged_cache_block_size,
                expected_kv_bytes_per_token=expected_kv_bytes_per_token,
            )

            # Connect paged SSD cache manager to PagedCacheManager
            if self.paged_cache_manager is not None:
                self.paged_cache_manager.set_paged_ssd_cache_manager(
                    self.paged_ssd_cache_manager
                )

            # Connect paged SSD cache manager to BlockAwarePrefixCache for paged SSD-only mode
            if self.block_aware_cache is not None:
                self.block_aware_cache.set_paged_ssd_cache_manager(
                    self.paged_ssd_cache_manager
                )

            # Initialize boundary snapshot SSD store for offloading
            # non-sliceable cache snapshots during prefill.
            # Skip in hot_cache_only mode since snapshots would never be written.
            if BoundarySnapshotSSDStore is not None and not self.config.hot_cache_only:
                try:
                    self._boundary_snapshot_store = BoundarySnapshotSSDStore(
                        base_dir=Path(self.config.paged_ssd_cache_dir)
                    )
                except Exception as e:
                    logger.debug(
                        "Failed to initialize boundary snapshot SSD store: %s", e
                    )

            if self.config.hot_cache_only:
                logger.info(
                    f"hot-cache-only mode enabled: "
                    f"hot_cache_max={self._format_bytes(self.config.hot_cache_max_size)}, "
                    f"block_size={self.config.paged_cache_block_size} tokens"
                )
            else:
                logger.info(
                    f"paged SSD cache enabled: "
                    f"cache_dir={self.config.paged_ssd_cache_dir}, "
                    f"max_size={self._format_bytes(self.config.paged_ssd_cache_max_size)}, "
                    f"block_size={self.config.paged_cache_block_size} tokens"
                )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize paged SSD cache: {e}")
            self.paged_ssd_cache_manager = None
            return False

    def _disable_paged_cache_components(self) -> None:
        """Clear paged-cache runtime state after SSD cache setup fails."""
        if self.paged_ssd_cache_manager is not None:
            try:
                self.paged_ssd_cache_manager.close()
            except Exception as e:
                logger.debug("Failed to close paged SSD cache manager: %s", e)
        self.paged_ssd_cache_manager = None
        self.paged_cache_manager = None
        self.block_aware_cache = None
        self._boundary_snapshot_store = None

    def _check_memory_pressure(self) -> None:
        """Check memory and evict blocks if needed.

        In paged SSD-only mode, memory pressure is not monitored since
        KV cache data is stored on paged SSD, not GPU memory.
        """
        # In paged SSD-only mode, memory_monitor is not used
        # All KV cache data is on paged SSD, so no GPU memory pressure from PagedCache
        pass

    def _evict_blocks_permanently(self, bytes_to_free: int) -> int:
        """
        Evict LRU blocks permanently (metadata cleanup).

        In paged SSD-only mode, blocks don't store data in GPU memory.
        This method just removes block metadata to free up slots.

        Args:
            bytes_to_free: Target bytes to free (used for estimation).

        Returns:
            Number of bytes freed (estimated).
        """
        if self.paged_cache_manager is None or self.memory_monitor is None:
            return 0

        # Estimate how many blocks to evict
        block_size = self.config.paged_cache_block_size
        num_blocks_to_evict = self.memory_monitor.estimate_blocks_to_free(
            bytes_to_free, block_size
        )

        # Get evictable blocks in LRU order
        evictable = self.paged_cache_manager.get_evictable_blocks(num_blocks_to_evict)

        if not evictable:
            logger.debug("No evictable blocks found for permanent eviction")
            return 0

        freed = 0
        evicted_count = 0

        for block in evictable:
            # In paged SSD-only mode, just clear metadata (data is on paged SSD)
            if self.paged_cache_manager.evict_block_permanently(block.block_id):
                freed += self.memory_monitor.estimate_block_memory(block_size)
                evicted_count += 1

            if freed >= bytes_to_free:
                break

        if evicted_count > 0:
            logger.info(
                f"Evicted {evicted_count} blocks permanently "
                f"(~{self._format_bytes(freed)} estimated)"
            )

        return freed

    def _evict_blocks_to_cold(self, bytes_to_free: int) -> int:
        """
        Evict LRU blocks (with paged SSD cache configured).

        In paged SSD-only mode, data is already on paged SSD, so this just evicts
        block metadata from the index. The data remains on paged SSD and can
        be re-discovered if the same token sequence is requested.

        Args:
            bytes_to_free: Target bytes to free (used for estimation).

        Returns:
            Number of bytes freed (estimated).
        """
        if self.paged_cache_manager is None or self.paged_ssd_cache_manager is None:
            return 0

        if self.memory_monitor is None:
            return 0

        # Estimate how many blocks to evict
        block_size = self.config.paged_cache_block_size
        num_blocks_to_evict = self.memory_monitor.estimate_blocks_to_free(
            bytes_to_free, block_size
        )

        # Get evictable blocks in LRU order
        evictable = self.paged_cache_manager.get_evictable_blocks(num_blocks_to_evict)

        if not evictable:
            logger.debug("No evictable blocks found")
            return 0

        evicted_count = 0

        for block in evictable:
            # In paged SSD-only mode, data is already on paged SSD
            # Just evict the block metadata
            if self.paged_cache_manager.evict_block_permanently(block.block_id):
                evicted_count += 1

        # Estimate bytes freed based on block count
        estimated_freed = evicted_count * self.memory_monitor.estimate_block_memory(
            block_size
        )

        if evicted_count > 0:
            logger.info(
                f"Evicted {evicted_count} blocks from index "
                f"(data preserved on paged SSD, ~{self._format_bytes(estimated_freed)} metadata freed)"
            )

        return estimated_freed

    def _restore_block_from_cold(self, block_id: int, block_hash: bytes) -> bool:
        """
        Restore a block from cold storage (deprecated in paged SSD-only mode).

        In paged SSD-only mode, blocks don't store cache_data. Data is loaded
        directly from SSD when needed via reconstruct_cache().

        Kept for API compatibility.

        Args:
            block_id: Block ID to restore.
            block_hash: Block's content hash.

        Returns:
            True if block exists in cold storage.
        """
        if self.paged_ssd_cache_manager is None or self.paged_cache_manager is None:
            return False

        # In paged SSD-only mode, just verify block exists on paged SSD
        if not self.paged_ssd_cache_manager.has_block(block_hash):
            logger.warning(f"Block {block_id} not found in cold storage")
            return False

        # Touch the block to update LRU
        block = (
            self.paged_cache_manager.blocks[block_id]
            if block_id < len(self.paged_cache_manager.blocks)
            else None
        )
        if block:
            block.touch()

        logger.debug(
            f"Block {block_id} verified on paged SSD (hash={block_hash.hex()[:16]}...)"
        )
        return True

    def restore_cold_blocks_for_request(self, request_id: str) -> int:
        """
        Verify all blocks needed for a request exist on paged SSD.

        In paged SSD-only mode, blocks don't store cache_data. This method
        just verifies that blocks exist on paged SSD.

        Args:
            request_id: Request ID.

        Returns:
            Number of blocks verified on paged SSD.
        """
        if self.paged_cache_manager is None or self.paged_ssd_cache_manager is None:
            return 0

        if self.block_aware_cache is None:
            return 0

        # Get block table for request
        block_table = self.paged_cache_manager.request_tables.get(request_id)
        if block_table is None:
            return 0

        verified = 0
        for block_id in block_table.block_ids:
            block = self.paged_cache_manager.blocks[block_id]
            if block.block_hash is not None:
                if self._restore_block_from_cold(block_id, block.block_hash):
                    verified += 1

        return verified

    def _collect_cache_counters(self) -> dict[str, int] | None:
        if self.block_aware_cache is None:
            return None

        prefix_stats = self.block_aware_cache.get_stats()
        counters = {
            "prefix_hits": prefix_stats.hits,
            "prefix_misses": prefix_stats.misses,
            "prefix_tokens_matched": prefix_stats.tokens_matched_total,
            "prefix_tokens_requested": prefix_stats.tokens_requested_total,
            "prefix_tokens_saved": prefix_stats.tokens_saved,
            "evictions": prefix_stats.evictions,
        }

        if self.paged_ssd_cache_manager is not None:
            ssd = self.paged_ssd_cache_manager.get_stats()
            hot_hits = ssd.hot_cache_hits
            total_loads = ssd.loads
            counters.update(
                {
                    "ssd_hot_hits": hot_hits,
                    "ssd_disk_loads": max(0, total_loads - hot_hits),
                    "ssd_saves": ssd.saves,
                    "ssd_errors": ssd.errors,
                    "hot_cache_evictions": ssd.hot_cache_evictions,
                    "hot_cache_promotions": ssd.hot_cache_promotions,
                }
            )

        return counters

    def get_ssd_cache_stats(self) -> dict[str, Any] | None:
        """Get paged SSD + prefix cache observability statistics."""
        stats = {}

        if self.paged_ssd_cache_manager is not None:
            stats["ssd_cache"] = self.paged_ssd_cache_manager.get_stats()

        if self.paged_cache_manager is not None:
            stats["indexed_blocks"] = self.paged_cache_manager.cold_block_count
            stats["block_size"] = self.config.paged_cache_block_size

        if self.block_aware_cache is not None:
            stats["prefix_cache"] = self.block_aware_cache.get_stats_dict()

        counters = self._collect_cache_counters()
        if counters:
            stats["cache_rates"] = self._cache_rate_tracker.snapshot_and_get_rates(
                counters
            )

        return stats if stats else None

    # Alias for backwards compatibility
    get_tiered_cache_stats = get_ssd_cache_stats

    @staticmethod
    def _format_bytes(bytes_value: int) -> str:
        """Format bytes as human-readable string."""
        if bytes_value >= 1024**3:
            return f"{bytes_value / 1024**3:.2f} GB"
        elif bytes_value >= 1024**2:
            return f"{bytes_value / 1024**2:.2f} MB"
        elif bytes_value >= 1024:
            return f"{bytes_value / 1024:.2f} KB"
        else:
            return f"{bytes_value} B"
