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
import logging
import os
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator,
    GenerationBatch,
    PromptProcessingBatch,
    SequenceStateMachine,
    generation_stream,
)
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors

from .cache.paged_cache import PagedCacheManager
from .cache.prefix_cache import BlockAwarePrefixCache
from .exceptions import is_cache_corruption_error
from .prefill_progress import get_prefill_tracker
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .utils.sampling import make_sampler as omlx_make_sampler


def _sync_and_clear_cache():
    """Synchronize in-flight GPU work before clearing the Metal buffer cache.

    Without synchronization, mx.clear_cache() can release Metal buffers that
    are still referenced by in-flight command buffers submitted via
    mx.async_eval(). This causes the GPU driver to hit a
    'completeMemory() prepare count underflow' kernel panic on M4 hardware
    (and SIGSEGV/SIGABRT on M3).

    See: https://github.com/jundot/omlx/issues/300, #888
    """
    # Generation_stream may not have in-flight work on the current thread
    # (e.g. external prefill submits to the default stream). On some MLX
    # builds mx.synchronize raises "There is no Stream(gpu, 0) in current
    # thread" in that case; swallow it since there is nothing to drain.
    try:
        mx.synchronize(generation_stream)
    except RuntimeError:
        pass
    mx.synchronize()  # default stream
    mx.clear_cache()


# Import tiered cache components
try:
    from .cache.boundary_snapshot_store import BoundarySnapshotSSDStore
    from .cache.paged_ssd_cache import PagedSSDCacheManager
    from .memory_monitor import MemoryMonitor

    HAS_TIERED_CACHE = True
except ImportError:
    PagedSSDCacheManager = None
    BoundarySnapshotSSDStore = None
    MemoryMonitor = None
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

# Import streaming detokenizer for proper UTF-8 handling
try:
    from mlx_lm.tokenizer_utils import NaiveStreamingDetokenizer
except ImportError:
    NaiveStreamingDetokenizer = None

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


# ---------------------------------------------------------------------------
# Monkey-patch GenerationBatch._step to call grammar accept_token() after
# sampling.  In the pipelined _step(), logits processors fill the bitmask
# (constrain NEXT token) but can't know which token was just sampled.
# After _original_step returns, self._next_tokens holds the freshly sampled
# tokens.  We eval them synchronously and accept in grammar processors.
# ---------------------------------------------------------------------------
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


GenerationBatch._step = _patched_generation_batch_step


# Monkey-patch TurboQuantKVCache.merge so _merge_caches() works
try:
    from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache

    from .turboquant_kv import BatchTurboQuantKVCache as _BTQCache

    if not hasattr(_TQCache, "merge"):
        _TQCache.merge = _BTQCache.merge
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
_KNOWN_SLICEABLE_CACHE_TYPES = frozenset(
    {
        "KVCache",
        "BatchKVCache",
        "QuantizedKVCache",
        "TurboQuantKVCache",
        "BatchTurboQuantKVCache",
    }
)


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
    prefill_step_size: int = 2048

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
    # Whether any work was done
    has_work: bool = False


class _BoundarySnapshotProvider:
    """Dict-like lazy loader for boundary snapshots.

    Used by ``store_cache()`` to load snapshots from SSD one block at a time
    instead of extracting all intermediate snapshots into memory at once.
    Implements ``__bool__``, ``__contains__``, and ``__getitem__`` to be a
    drop-in replacement for ``Dict[int, List[Dict[str, Any]]]``.
    """

    def __init__(
        self,
        store: Any,  # Optional[BoundarySnapshotSSDStore]
        request_id: str,
        valid_tcs: list[int],
        in_memory_snapshots: dict[int, Any],
        extract_fn: Any,  # Callable — Scheduler._extract_cache_states
    ) -> None:
        self._store = store
        self._request_id = request_id
        self._valid_tcs = set(valid_tcs)
        self._in_memory = in_memory_snapshots
        self._extract_fn = extract_fn

    def __contains__(self, tc: int) -> bool:
        return tc in self._valid_tcs

    def __getitem__(self, tc: int) -> Any:
        snap = self._in_memory.get(tc)
        if snap is not None:
            # In-memory fallback (SSD write failed).
            extracted, _ = self._extract_fn(snap)
            return extracted
        if self._store is not None:
            return self._store.load(self._request_id, tc)
        return None

    def __len__(self) -> int:
        return len(self._valid_tcs)

    def __bool__(self) -> bool:
        return bool(self._valid_tcs)


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

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: SchedulerConfig | None = None,
    ):
        """
        Initialize the scheduler.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Scheduler configuration
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

        # Load additional EOS tokens from generation_config.json.
        # Some models (e.g. GLM-4.6V) define multiple EOS tokens there
        # that are not in tokenizer.eos_token_id.
        self._generation_config_eos: set[int] | None = (
            self._load_generation_config_eos()
        )

        # For strict RotatingKVCache reuse, align paged cache block size to
        # the model's rotating window size when paged cache is enabled.
        self._align_block_size_with_rotating_window()
        # For ArraysCache-only models (no RotatingKVCache), use a larger block
        # size to reduce boundary snapshot overhead during prefill.
        self._enlarge_block_size_for_arrays_cache()

        # TurboQuant KV cache (set by engine if model_settings has it enabled)
        self._turboquant_kv_bits: float | None = None
        self._turboquant_skip_last: bool = True

        # Request management - following vLLM's design
        self.waiting: deque[Request] = deque()  # Waiting queue (FCFS)
        self.running: dict[str, Request] = {}  # Running requests by ID
        self.requests: dict[str, Request] = {}  # All requests by ID
        self.finished_req_ids: set[str] = set()  # Recently finished

        # Thread-safe set for deferred aborts (main thread → executor thread)
        # CPython GIL guarantees set.add() and `x in set` are atomic.
        self._pending_abort_ids: set[str] = set()

        # Memory limits for inline prefill checking.
        # Set by ProcessMemoryEnforcer; propagated to BatchGenerator.
        self._memory_limit_bytes: int = 0  # soft limit
        self._memory_hard_limit_bytes: int = 0  # hard limit (system_ram - 4GB)
        self._prefill_memory_guard: bool = False  # set by ProcessMemoryEnforcer

        # SpecPrefill: draft model for attention-based sparse prefill
        self._specprefill_draft_model: Any | None = None
        # Track active specprefill request for RoPE cleanup
        self._specprefill_active_request_id: str | None = None

        # Phase timing instrumentation for cache-on overhead diagnostics.
        # Accumulated wall-time per phase + invocation count, dumped at request
        # end or via get_phase_stats(). Adds ~100ns per measurement.
        self._phase_total_ms: dict[str, float] = defaultdict(float)
        self._phase_count: dict[str, int] = defaultdict(int)

        # Async store_cache executor (G2-async). Offloads the post-finish
        # bulk memcpy (28GB+ per 32k request) off the inference thread so
        # response streaming isn't blocked by it.
        self._store_cache_executor: concurrent.futures.ThreadPoolExecutor | None = None
        # Pending (uid, request_id, future) entries waiting for async store
        # to finish before batch_generator.remove() can safely run. Drained
        # at the start of every step.
        self._pending_async_removes: deque = deque()
        # Track in-flight store futures per request_id for lookup wait /
        # shutdown wait.
        self._inflight_store_futures: dict[str, concurrent.futures.Future] = {}

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

            # Initialize paged SSD cache
            self._init_tiered_cache()

            # Set cold restore callback for prefix cache
            if self.paged_ssd_cache_manager is not None:
                self.block_aware_cache.set_cold_restore_callback(
                    self._restore_block_from_cold
                )
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
    ) -> None:
        """Run store_cache + paged_cache cleanup off the inference thread.

        Pre-conditions enforced by the caller (_cleanup_finished):
        - All mx.array references in cache_to_store have already been
          mx.eval()'d on the inference thread, so save_block's internal
          mx.eval is a no-op.
        - bfloat16 view+eval inside _extract_tensor_bytes runs on this
          worker's default mx stream, isolated from generation_stream;
          the underlying buffer is read-only at this point.
        - batch_generator.remove(uid) is deferred until this worker
          completes (handled by _drain_pending_async_removes).

        paged_cache_manager and block_aware_cache rely on
        threading.RLock so concurrent access from main and worker is safe.
        """
        try:
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
            if block_table is None and self.paged_cache_manager is not None:
                block_table = self.paged_cache_manager.get_block_table(request_id)
            if block_table and self.paged_cache_manager is not None:
                self.paged_cache_manager.release_for_eviction(block_table.block_ids)
            if self.block_aware_cache is not None:
                self.block_aware_cache.clear_request_entry(request_id)
        except Exception as e:
            logger.warning("Async store_cache failed for %s: %s", request_id, e)

    def _drain_pending_async_removes(self) -> None:
        """Process deferred batch_generator.remove() calls from prior steps.

        Called at the start of every step. For each pending entry, if the
        async store_cache future has finished, perform the
        batch_generator.remove() on the inference thread (Metal-safe) and
        finalize cleanup state. Entries whose futures are still in flight
        are left at the head of the deque for a later step.
        """
        if not self._pending_async_removes:
            return
        while self._pending_async_removes:
            uid, request_id, future = self._pending_async_removes[0]
            if future is not None and not future.done():
                # Worker still busy. Stop draining; check again next step.
                # Inflight entry stays at deque head to preserve order.
                break
            self._pending_async_removes.popleft()
            # Surface worker exceptions for visibility (don't crash step loop).
            if future is not None:
                exc = future.exception()
                if exc is not None:
                    logger.warning(
                        "Async store_cache for %s raised: %s", request_id, exc
                    )
            # Run batch_generator.remove on the inference thread.
            try:
                mx.synchronize(generation_stream)
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
            if uid in self.uid_to_request_id:
                del self.uid_to_request_id[uid]
            if request_id in self.request_id_to_uid:
                del self.request_id_to_uid[request_id]
            self._inflight_store_futures.pop(request_id, None)
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
        if class_name in ("RotatingKVCache", "BatchRotatingKVCache"):
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
            model_path = getattr(self.tokenizer, "name_or_path", None)
            if not model_path:
                return None
            import json
            import os

            gc_path = os.path.join(model_path, "generation_config.json")
            if not os.path.exists(gc_path):
                # name_or_path may be a HuggingFace repo ID (e.g. for VLM
                # tokenizers loaded via AutoProcessor).  Try the HF cache.
                try:
                    from huggingface_hub import try_to_load_from_cache

                    cached = try_to_load_from_cache(
                        model_path, "generation_config.json"
                    )
                    if cached and isinstance(cached, str):
                        gc_path = cached
                    else:
                        return None
                except (ImportError, Exception):
                    return None
            with open(gc_path) as f:
                gc = json.load(f)
            eos = gc.get("eos_token_id")
            if eos is None:
                return None
            if isinstance(eos, list):
                result = set(eos)
            else:
                result = {eos}
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

        # Read additional EOS tokens from generation_config.json.
        # Some models (e.g. GLM-4.6V) define multiple EOS tokens there
        # that are not reflected in tokenizer.eos_token_id.
        if self._generation_config_eos is not None:
            stop_tokens.update(self._generation_config_eos)

        # Add protocol-specific stop tokens (e.g. Harmony action stops)
        if self._output_parser_factory is not None:
            stop_tokens.update(self._output_parser_factory.stop_token_ids)

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
            if hasattr(self.tokenizer, "detokenizer"):
                detok = self.tokenizer.detokenizer
            elif NaiveStreamingDetokenizer is not None:
                detok = NaiveStreamingDetokenizer(self.tokenizer)
            else:
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
        detok = self._request_detokenizers.pop(request_id, None)
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
        """Get special tokens to exclude from XTC sampling (newline + EOS).

        Reuses _get_stop_tokens() for EOS coverage (includes generation_config.json
        tokens) so XTC exclusions stay consistent with stop-token logic.
        """
        tokens = self.tokenizer.encode("\n")
        tokens.extend(self._get_stop_tokens())
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
            logits_processors=logits_processors if logits_processors else None,
            prefill_batch_size=1,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
        )

        return bg

    def _on_prompt_progress(self, updates: list[tuple[int, int, int]]) -> None:
        """Callback from BatchGenerator's prefill loop.

        Called once per prefill chunk (default 2048 tokens) with a list of
        (uid, processed_tokens, total_tokens) tuples.  Updates the global
        PrefillProgressTracker so the admin dashboard can display per-request
        prefill progress.  Only touches CPU counters — zero GPU overhead.
        """
        import os

        from .prefill_progress import get_prefill_tracker

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

    def _apply_turboquant_kv_empty(self, prompt_cache: list[Any]) -> None:
        """Replace KVCache with empty TurboQuantKVCache before prefill.

        NOTE: Not currently called -- see #771. Kept for future use when
        TurboQuantKVCache implements merge()/maybe_trim_front().

        Tokens are quantized on the fly during update_and_fetch, avoiding
        the peak memory spike from storing full-precision KV then converting.
        Skips the last KVCache layer if turboquant_skip_last is set.
        """
        from mlx_lm.models.cache import CacheList, KVCache
        from mlx_vlm.turboquant import TurboQuantKVCache

        kv_indices = [i for i, c in enumerate(prompt_cache) if isinstance(c, KVCache)]
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
        """Convert existing KVCache data to TurboQuantKVCache via from_cache().

        NOTE: Not currently called -- see #771. Kept for future use when
        TurboQuantKVCache implements merge()/maybe_trim_front().

        Used when an existing cache is provided (e.g. from SSD prefix cache).
        Uses from_cache() to quantize the existing KV data.
        """
        from mlx_lm.models.cache import CacheList, KVCache
        from mlx_vlm.turboquant import TurboQuantKVCache

        kv_indices = [i for i, c in enumerate(prompt_cache) if isinstance(c, KVCache)]
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
            # Nothing to prefill, return cache + tokens as-is
            cache = existing_cache or make_prompt_cache(self.model)
            # NOTE: Do NOT apply TurboQuant here. TurboQuantKVCache does not
            # support merge(), which is called by _merge_caches() inside
            # BatchGenerator when insert() creates a PromptProcessingBatch.
            # TurboQuant conversion must happen inside BatchGenerator after
            # the batch cache is created, not on individual per-request caches.
            return cache, tokens

        # Create or reuse cache
        if existing_cache is not None:
            prompt_cache = existing_cache
        else:
            prompt_cache = make_prompt_cache(self.model)

        # NOTE: TurboQuant conversion is NOT applied during external prefill.
        # TurboQuantKVCache does not support merge() or maybe_trim_front(),
        # so passing it to insert() would fail in _merge_caches() or cause
        # AttributeError in chunked-attention models (e.g. Llama-4-Scout).
        # Additionally, on-the-fly quantization during prefill causes
        # precision loss that corrupts hidden states across layers (#771).
        # Prefill runs with standard KVCache; TurboQuant quantization
        # happens inside BatchGenerator during the decode phase.

        # Clear stale mRoPE position state for text-only requests.
        if vlm_embeds is None and hasattr(self.model, "clear_vlm_position_state"):
            self.model.clear_vlm_position_state()

        # Boundary snapshot setup
        block_size = self.config.paged_cache_block_size
        boundary_enabled = (
            block_size > 0
            and self.block_aware_cache is not None
            and _prompt_cache_needs_snapshots(prompt_cache)
        )
        all_boundaries = (
            boundary_enabled  # always stop at every boundary for hybrid models
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

        # Prefill tokens[0:N-1] (leave last token for insert())
        prefill_tokens = tokens[:-1]
        last_token = tokens[-1:]
        total_length = len(tokens)

        input_arr = mx.array(prefill_tokens)[None]  # (1, seq_len)
        processed_tokens = 0
        prefill_step_size = self.config.prefill_step_size
        uid = self.request_id_to_uid.get(request.request_id)

        emitted_boundaries: dict[int, int] = {}

        while input_arr.shape[1] > 0:
            remaining = input_arr.shape[1]
            n_to_process = min(prefill_step_size, remaining)

            # Boundary-limited step size
            if boundary_enabled and block_size > 0:
                current_total = base_size + processed_tokens
                next_boundary = ((current_total // block_size) + 1) * block_size
                target_boundary_prefill = next_boundary - base_size
                delta = target_boundary_prefill - processed_tokens
                if delta > 0:
                    n_to_process = min(n_to_process, delta)
                n_to_process = max(1, n_to_process)

            model_kwargs: dict[str, Any] = {}
            if embeds_array is not None and embeds_array.shape[1] > 0:
                model_kwargs["inputs_embeds"] = embeds_array[:, :n_to_process]
                if extra_kwargs:
                    model_kwargs["vlm_extra_kwargs"] = _slice_vlm_extra(
                        extra_kwargs, n_to_process
                    )

            self.model(input_arr[:, :n_to_process], cache=prompt_cache, **model_kwargs)
            mx.eval([c.state for c in prompt_cache])

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

            # Memory monitoring
            if self._memory_limit_bytes > 0:
                active = mx.get_active_memory()
                if (
                    self._memory_hard_limit_bytes > 0
                    and active > self._memory_hard_limit_bytes
                ):
                    logger.warning(
                        f"Prefill force-stopped at {processed_tokens} "
                        f"tokens: memory {active / 1024**3:.1f}GB "
                        f"exceeds hard limit "
                        f"{self._memory_hard_limit_bytes / 1024**3:.1f}GB"
                    )
                    raise RuntimeError("Memory limit exceeded during prefill")
                elif active > self._memory_limit_bytes:
                    logger.warning(
                        f"Prefill memory soft limit exceeded at "
                        f"{processed_tokens} tokens: "
                        f"{active / 1024**3:.1f}GB > "
                        f"{self._memory_limit_bytes / 1024**3:.1f}GB "
                        f"(hard limit: "
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
                raise _PrefillAbortedError(abort_uids, processed_tokens)

            # Reclaim Metal intermediates between prefill chunks.
            _sync_and_clear_cache()

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

        _sync_and_clear_cache()

        # Restore _rope_deltas after cached VLM prefill (for decode capture)
        if vlm_embeds is not None and _saved_rope_deltas is not None:
            self.model._language_model._rope_deltas = _saved_rope_deltas

        return prompt_cache, last_token

    def _build_state_machine(self, request: "Request") -> SequenceStateMachine:
        """Build a SequenceStateMachine for per-request stop tokens.

        Combines base stop tokens (EOS, Harmony) with request-specific
        stop_token_ids into a single state machine that tells
        BatchGenerator when to stop generating for this request.
        """
        stop_tokens_set = self._get_stop_tokens()
        if request.sampling_params.stop_token_ids:
            stop_tokens_set.update(request.sampling_params.stop_token_ids)

        if stop_tokens_set:
            # Each stop token is a single-element sequence.
            transitions = {"normal": [([t], None) for t in stop_tokens_set]}
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
        """
        snapshot_cache = [
            c if type(c).__name__ not in _KNOWN_SLICEABLE_CACHE_TYPES else None
            for c in prompt_cache
        ]
        self._on_prefill_boundary_snapshot(
            self.request_id_to_uid.get(request.request_id, -1),
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

        # Add thinking budget processor for reasoning models
        if (
            sampling_params.thinking_budget is not None
            and request is not None
            and getattr(request, "needs_think_prefix", False)
            and not getattr(request, "is_harmony_model", False)
        ):
            think_end_ids = self._resolve_think_end_token_ids()
            if think_end_ids:
                from .api.thinking import ThinkingBudgetProcessor

                think_start_id = self._get_think_token_id("think_start_id")
                leading_ids, trailing_ids = self._resolve_think_close_pattern()
                processor = ThinkingBudgetProcessor(
                    think_end_token_ids=think_end_ids,
                    budget=sampling_params.thinking_budget,
                    think_start_token_id=think_start_id,
                    leading_token_ids=leading_ids,
                    trailing_token_ids=trailing_ids,
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

    def _resolve_think_end_token_ids(self) -> list[int] | None:
        """Resolve token ID(s) for the close-think tag.

        Uses mlx-lm's built-in think_end_id which supports both
        </think> and </longcat_think> automatically.
        """
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

    def _resolve_think_close_pattern(self) -> tuple[list[int] | None, list[int] | None]:
        """Detect leading/trailing tokens around </think> from the chat template.

        Different models use different patterns:
        - Qwen3/3.5, MiniMax: ``\\n</think>\\n\\n``
        - DeepSeek V3.2, GLM-5: ``</think>`` (no newlines)
        - GLM-4.6V: ``</think>\\n``
        - Step-3.5-Flash: ``\\n</think>\\n``

        Returns (leading_token_ids, trailing_token_ids) or (None, None).
        """
        import re

        think_end_str = getattr(self.tokenizer, "think_end", "</think>")

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
        think_start_id = self._get_think_token_id("think_start_id")
        if think_start_id is None:
            try:
                think_start_id = self.tokenizer.convert_tokens_to_ids("<think>")
                if think_start_id == getattr(self.tokenizer, "unk_token_id", None):
                    return False
            except (AttributeError, KeyError, TypeError):
                return False

        if not think_start_id or not request.prompt_token_ids:
            return False

        last_tokens = list(request.prompt_token_ids[-3:])
        if think_start_id not in last_tokens:
            return False

        # <think> found. Check if </think> follows it (disabled thinking pattern).
        last_idx = len(last_tokens) - 1 - last_tokens[::-1].index(think_start_id)
        after_start = last_tokens[last_idx + 1 :]

        if after_start:
            think_end_ids = self._resolve_think_end_token_ids()
            if think_end_ids and think_end_ids[0] in after_start:
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
        if class_name in (
            "RotatingKVCache",
            "BatchRotatingKVCache",
            "ArraysCache",
            "SizedArraysCache",
        ):
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

    def _on_prefill_boundary_snapshot(
        self,
        uid: int,
        snapshot_cache: list[Any],
        token_count: int,
    ) -> None:
        """Record boundary snapshots captured during prefill processing."""
        if self.block_aware_cache is None:
            return

        block_size = self.config.paged_cache_block_size
        if block_size <= 0 or token_count <= 0 or token_count % block_size != 0:
            return

        request_id = self.uid_to_request_id.get(uid)
        if request_id is None:
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
                self._boundary_cache_snapshots[request_id][token_count] = snapshot_cache
        else:
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

        self._boundary_snapshot_required = any(
            self._cache_tree_has_stateful_non_sliceable(layer_cache)
            for layer_cache in cache_list
        )

        if self._boundary_snapshot_required:
            logger.info(
                "Enabled boundary cache snapshots for stateful non-sliceable "
                "cache layers"
            )
        else:
            logger.debug(
                "Boundary cache snapshots disabled (no stateful non-sliceable "
                "cache layers detected)"
            )

        return self._boundary_snapshot_required

    def _extract_boundary_snapshot(self, uid: int) -> list[Any] | None:
        """Extract a per-request prompt cache snapshot via extract_cache().

        Uses BatchGenerator.extract_cache() which returns
        Dict[uid, (cache_list, tokens_list)].
        """
        if self.batch_generator is None:
            return None

        try:
            # Synchronize pending generation_stream operations before
            # accessing batch cache tensors.
            with self._phase_timer("boundary_capture_sync"):
                mx.synchronize(generation_stream)
            with self._phase_timer("boundary_capture_extract"):
                with mx.stream(generation_stream):
                    result = self.batch_generator.extract_cache([uid])
                    if uid not in result:
                        return None
                    cache_list, _tokens = result[uid]
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

        snapshot_cache = self._extract_boundary_snapshot(uid)
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
                self._boundary_cache_snapshots[request.request_id][
                    total_tokens
                ] = snapshot_cache
        else:
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

        # Build lazy-loading provider for intermediate snapshots.
        # Each snapshot is loaded from SSD one-at-a-time during
        # store_cache() instead of extracting all at once.
        intermediate_tcs = [tc for tc in valid_counts if tc != latest_tc]
        intermediate_snapshots = _BoundarySnapshotProvider(
            store=self._boundary_snapshot_store,
            request_id=request_id,
            valid_tcs=intermediate_tcs,
            in_memory_snapshots=snapshots,
            extract_fn=self._extract_cache_states,
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
                if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
                    try:
                        cache_type = CacheTypeRegistry.detect_cache_type(layer_cache)
                        cache_type_name = cache_type.value
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
                            extracted.append(
                                {
                                    "state": state_dict.get("sub_states", []),
                                    "meta_state": (
                                        state_dict.get("sub_class_names", []),
                                        state_dict.get("sub_meta_states", []),
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
                    state = layer_cache.state
                    meta = getattr(layer_cache, "meta_state", ())

                    if class_name in ("RotatingKVCache", "BatchRotatingKVCache"):
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
                            and len(state) >= 2
                        ):
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

    def add_request(self, request: Request) -> None:
        """
        Add a new request to the scheduler.

        Args:
            request: The request to add
        """
        if request.request_id in self.requests:
            raise ValueError(f"Request {request.request_id} already exists")

        # Tokenize if needed
        if request.prompt_token_ids is None:
            if isinstance(request.prompt, str):
                request.prompt_token_ids = self.tokenizer.encode(request.prompt)
            else:
                request.prompt_token_ids = list(request.prompt)
            request.num_prompt_tokens = len(request.prompt_token_ids)

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
                # Reconstruct actual KVCache objects from stored tensor data
                # Note: reconstruct_cache may modify block_table in-place if
                # partial reconstruction occurs (some blocks invalid)
                original_tokens = block_table.num_tokens
                reconstructed = self.block_aware_cache.reconstruct_cache(block_table)
                if reconstructed:
                    request.prompt_cache = reconstructed
                    request.block_table = block_table
                    request.cached_tokens = block_table.num_tokens
                    request.shared_prefix_blocks = len(block_table.block_ids)
                    # Recalculate remaining_tokens in case block_table was truncated
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

        # SpecPrefill: score remaining tokens with draft model if applicable.
        # Must run AFTER prefix cache check (scoring applies only to uncached suffix).
        self._try_specprefill_scoring(request)

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
                "conversation_tokens": request.num_prompt_tokens - request.specprefill_system_end,
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
            # the generation_stream is drained before Metal buffers are
            # returned to the pool — a bare mx.clear_cache() here can race
            # with in-flight async evals and trigger a kernel panic (#557).
            del used_cache
            _sync_and_clear_cache()

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
        if not cache_list:
            return False

        for cache_obj in cache_list:
            if not self._trim_cache_tree_by_one(cache_obj):
                return False
        return True

    def _trim_cache_tree_by_one(self, cache_obj: Any) -> bool:
        """Trim one token from cache object (recursively for CacheList)."""
        sub_caches = getattr(cache_obj, "caches", None)
        if isinstance(sub_caches, (list, tuple)):
            return all(
                self._trim_cache_tree_by_one(sub_cache) for sub_cache in sub_caches
            )

        trim_fn = getattr(cache_obj, "trim", None)
        if not callable(trim_fn):
            return False

        try:
            trimmed = trim_fn(1)
            if trimmed is None:
                return True
            return int(trimmed) >= 1
        except Exception:
            return False

    def _remove_uid_from_active_batch(self, uid: int) -> None:
        """Remove UID from BatchGenerator safely."""
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

        # Remove from waiting queue
        if request.status == RequestStatus.WAITING:
            try:
                self.waiting.remove(request)
            except ValueError:
                pass

        # Remove from running (BatchGenerator)
        if request.request_id in self.request_id_to_uid:
            uid = self.request_id_to_uid[request.request_id]
            # Synchronize in-flight GPU work before modifying batch state.
            # batch_generator.remove() triggers lazy KV cache array slicing
            # that replaces references to arrays still used by in-flight
            # Metal command buffers.  Without this barrier the Metal driver
            # can hit 'completeMemory() prepare count underflow'.
            mx.synchronize(generation_stream)
            self._remove_uid_from_active_batch(uid)
            if hasattr(self.model, "unregister_rope_delta"):
                self.model.unregister_rope_delta(uid)
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
        from .prefill_progress import get_prefill_tracker

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
        return bool(self.waiting or self.running or self._deferred_clear_at is not None)

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
            if req is not None:
                req._extracted_cache = None
                req.prompt_cache = None
        self.running.clear()
        for request in list(self.waiting):
            failed_ids.append(request.request_id)
            req = self.requests.pop(request.request_id, None)
            if req is not None:
                req._extracted_cache = None
                req.prompt_cache = None
        self.waiting.clear()
        # Reset batch generator only (cache is not corrupted)
        self.batch_generator = None
        self._current_sampler_params = None
        # Reclaim fragmented Metal buffers after generation failure.
        # Without this, subsequent requests may hit the same resource
        # limit even though Python references have been cleared.
        # Wrapped in try-except because Metal may already be in an error
        # state — mx.synchronize() or mx.clear_cache() can throw a C++
        # exception that causes SIGABRT if uncaught (#435).
        try:
            _sync_and_clear_cache()
        except Exception as e:
            logger.warning(f"Metal cache clear failed during error recovery: {e}")
        return failed_ids

    def get_num_waiting(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """Get number of running requests."""
        return len(self.running)

    def _preflight_memory_check(self, request: "Request") -> str | None:
        """
        Estimate whether prefill would exceed memory limits.

        Computes worst-case peak memory for the last prefill chunk
        (model weights + KV cache + SDPA attention matrix) and rejects
        if it would exceed the hard limit.

        For head_dim > 128, MLX SDPA uses a fallback that materializes
        the full attention matrix [B, n_q, chunk, kv_len] in float32.
        For head_dim <= 128, MLX uses a fused kernel with O(n) memory.

        Returns:
            Error message string if request should be rejected, None if OK.
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
            new_tokens, self.config.prefill_step_size
        )
        if peak == 0:
            return None  # can't estimate, skip

        current = mx.get_active_memory()

        if current + peak > self._memory_hard_limit_bytes:
            from .utils.hardware import format_bytes

            return (
                f"Prefill would require ~{format_bytes(current + peak)} peak "
                f"(model {format_bytes(current)} + KV+SDPA {format_bytes(peak)}) "
                f"but limit is {format_bytes(self._memory_hard_limit_bytes)}. "
                f"Reduce context length or increase --max-process-memory."
            )
        return None

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

        while self.waiting and len(self.running) < self.config.max_num_seqs:
            # Generation memory guard: when requests are already running,
            # defer scheduling if memory pressure is high to prevent
            # Metal allocation failures during batch_generator.next().
            # First request always passes (self.running is empty).
            if (
                self._prefill_memory_guard
                and self._memory_limit_bytes > 0
                and self.running
            ):
                active = mx.get_active_memory()
                if active > self._memory_limit_bytes:
                    logger.debug(
                        "Generation memory guard: deferring scheduling "
                        "(%s > %s), %d running",
                        active,
                        self._memory_limit_bytes,
                        len(self.running),
                    )
                    break

            request = self.waiting.popleft()

            # Ensure we have a batch generator
            self._ensure_batch_generator(request.sampling_params)

            if self.batch_generator is None:
                # Put back and try again later
                self.waiting.appendleft(request)
                break

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
            # and reject if it would exceed the hard limit.
            preflight_error = self._preflight_memory_check(request)
            if preflight_error:
                logger.warning(
                    f"Request {request.request_id} rejected by prefill "
                    f"memory guard: {preflight_error}"
                )
                self.requests.pop(request.request_id, None)
                rejected_outputs.append(
                    RequestOutput(
                        request_id=request.request_id,
                        finished=True,
                        finish_reason="error",
                        error=preflight_error,
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
                            "conversation_tokens": request.num_prompt_tokens - request.specprefill_system_end,
                            "cached_tokens": request.cached_tokens,
                            "scored_tokens": m_pre,
                            "selected_tokens": n_eff,
                            "keep_percent": round(n_eff / m_pre * 100)
                            if m_pre > 0
                            else 0,
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
                            # mx.clear_cache() to flush the generation_stream
                            # before releasing Metal buffers.  A bare call here
                            # can race with in-flight command buffers submitted
                            # by the preceding mx.eval(), triggering the same
                            # 'completeMemory() prepare count underflow' kernel
                            # panic that #435 fixed elsewhere (#557).
                            _sync_and_clear_cache()
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
                    M = len(conv_tokens)
                    pos_offset = request.specprefill_position_offset
                    last_idx = M - 1

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
                                "scored_tokens": M,
                                "selected_tokens": int(selected.shape[0]),
                                "keep_percent": round(int(selected.shape[0]) / M * 100)
                                if M > 0
                                else 0,
                                "prompt_tokens": request.num_prompt_tokens,
                                "system_tokens": request.specprefill_system_end,
                                "conversation_tokens": request.num_prompt_tokens - request.specprefill_system_end,
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
                    # adjustment = M - N'. Subtract 1 to account for the
                    # extra token BatchGenerator will process.
                    for _, layer in _find_attention_layers(self.model):
                        attn = _get_attn_module(layer)
                        if (
                            attn
                            and hasattr(attn, "rope")
                            and isinstance(attn.rope, _OffsetAdjustedRoPE)
                        ):
                            attn.rope._adjustment -= 1

                    N = int(selected.shape[0])
                    t_prefill = time.monotonic() - t0
                    total_prompt = request.num_prompt_tokens
                    cached = request.cached_tokens
                    logger.info(
                        f"SpecPrefill: sparse prefill {N}/{M} conv tokens in {t_prefill:.1f}s "
                        f"(total {total_prompt}, cached {cached}, "
                        f"system {sys_count} full, conv {M} sparse)"
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
                    raise
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
                # Assign UID early so progress callbacks can map uid->request_id
                # during external prefill. Use a temporary UID that will be replaced
                # by the real one from insert().
                temp_uid = id(request)  # unique, won't collide with BatchGenerator UIDs
                self.request_id_to_uid[request.request_id] = temp_uid
                self.uid_to_request_id[temp_uid] = request.request_id

                vlm_embeds = None
                if request.vlm_inputs_embeds is not None:
                    vlm_embeds = (
                        request.vlm_inputs_embeds,
                        request.vlm_extra_kwargs or {},
                        request.cached_tokens,
                    )

                prefilled_cache, last_token = self._do_external_prefill(
                    request,
                    tokens_to_process,
                    cache_to_use,
                    vlm_embeds=vlm_embeds,
                )

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

            # NOTE: TurboQuant KV conversion is not applied during prefill.
            # See _do_external_prefill() comment for rationale (#771).

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
                samplers=[sampler],
                logits_processors=[per_row_lps],
                state_machines=[sm],
            )

            if uids:
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
        for response in responses:
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                continue

            request = self.running.get(request_id)
            if request is None:
                continue

            request.last_activity_at = step_now

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

            if parser_session is not None:
                parser_result = parser_session.process_token(response.token)
                new_text = parser_result.stream_text
                if parser_result.visible_text:
                    request.output_text += parser_result.visible_text

                # Parser-defined stop token can override finish reason
                if parser_result.is_stop and not is_finished:
                    is_finished = True
                    is_stop = True

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

            # Prepend <think> tag for first chunk if this is a reasoning model
            # (skip when a protocol parser already manages reasoning formatting)
            if parser_session is None and getattr(request, "needs_think_prefix", False):
                if not getattr(request, "think_prefix_sent", False):
                    think_tag = getattr(self.tokenizer, "think_start", "<think>")
                    new_text = think_tag + "\n" + new_text
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
            output = RequestOutput(
                request_id=request_id,
                new_token_ids=[response.token] if not is_stop else [],
                new_text=new_text,
                output_token_ids=list(request.output_token_ids),
                prompt_tokens=request.num_prompt_tokens,
                completion_tokens=request.num_output_tokens,
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

    def _cleanup_finished(self, finished_ids: set[str]) -> None:
        """Clean up finished requests and store caches for reuse."""
        # Synchronize pending generation_stream operations before cache storage.
        # store_cache -> mx.save_safetensors triggers implicit mx.eval() which
        # can conflict with async Metal operations on the generation stream.
        if finished_ids:
            with self._phase_timer("cleanup_finished_sync"):
                mx.synchronize(generation_stream)

        # SpecPrefill: restore original RoPE if active request finished
        for rid in finished_ids:
            self._cleanup_specprefill(rid)

        # Remove finished requests from prefill progress tracker.
        from .prefill_progress import get_prefill_tracker

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
                        try:
                            full_token_sequence = list(request.prompt_token_ids) + list(
                                request.output_token_ids
                            )
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

                            # Boundary merge + a batched mx.eval all on the
                            # inference thread. The eval forces every array
                            # we hand to the worker into a materialized state
                            # so the worker's _extract_tensor_bytes is a pure
                            # memcpy (mx.eval inside save_block becomes a
                            # no-op). bfloat16 view+eval inside the worker
                            # then runs on its own default stream, isolated
                            # from generation_stream.
                            with (
                                self._phase_timer("store_cache_main_prep"),
                                mx.stream(generation_stream),
                            ):
                                boundary_override = self._get_boundary_store_override(
                                    request_id,
                                    cacheable_sequence,
                                )
                                if boundary_override is not None:
                                    (
                                        token_sequence_to_store,
                                        boundary_cache,
                                        boundary_model_config,
                                        intermediate_snapshots,
                                    ) = boundary_override
                                    cache_to_store = (
                                        self._merge_boundary_with_full_cache(
                                            boundary_cache, request._extracted_cache
                                        )
                                    )
                                    if boundary_model_config is not None:
                                        model_cache_config = boundary_model_config
                                    logger.info(
                                        f"Using boundary cache snapshot for {request_id}: "
                                        f"storing {len(token_sequence_to_store)}/"
                                        f"{len(full_token_sequence)} tokens "
                                        f"(skipping trailing partial block, "
                                        f"{len(intermediate_snapshots) if intermediate_snapshots else 0} "
                                        f"intermediate snapshots)"
                                    )
                                pre_eval_arrays = (
                                    self._collect_arrays_from_extracted_cache(
                                        cache_to_store
                                    )
                                )
                                if pre_eval_arrays:
                                    with self._phase_timer("store_cache_main_eval"):
                                        mx.eval(*pre_eval_arrays)

                            if self._store_cache_executor is not None:
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
                                )
                                self._inflight_store_futures[request_id] = store_future
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
                                )
                            logger.debug(
                                f"Submitted async store_cache for {request_id} "
                                f"({len(token_sequence_to_store)} tokens, "
                                f"{len(full_token_sequence)} total: "
                                f"{len(request.prompt_token_ids)} prompt + "
                                f"{len(request.output_token_ids)} output)"
                            )
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
                    mx.synchronize(generation_stream)
                    self._remove_uid_from_active_batch(uid)
                    if hasattr(self.model, "unregister_rope_delta"):
                        self.model.unregister_rope_delta(uid)
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

        # Clear UID mappings
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()

        # Cancel any pending deferred Metal cache clear
        self._deferred_clear_at = None

        # Clear detokenizer state to prevent contamination after recovery
        self._request_detokenizers.clear()

        # Clear protocol-specific output parser sessions
        self._output_parser_sessions.clear()

        logger.info("Cache recovery completed")

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
                    if req is not None:
                        req._extracted_cache = None
                        req.prompt_cache = None
                    continue

            # Reset scheduling state
            request.status = RequestStatus.WAITING
            request.batch_uid = None

            # Reset cache state
            request.prompt_cache = None
            request.cached_tokens = 0
            request.remaining_tokens = request.prompt_token_ids
            request.block_table = None
            request.shared_prefix_blocks = 0

            # Reset generation output (prevent duplicate tokens on re-prefill)
            request.output_token_ids = []
            request.output_text = ""
            request.num_computed_tokens = 0

            # Reset extracted cache (prevent GPU memory leak)
            request._extracted_cache = None
            request._model_cache_config = None

            # Reset reasoning model state
            request.think_prefix_sent = False

            # Move to waiting queue (at front for priority)
            self.waiting.appendleft(request)
            del self.running[request_id]
            count += 1

        if count > 0:
            logger.info(f"Rescheduled {count} requests for re-prefill")
        return failed_ids

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

        # Drain async store_cache completions from prior steps. Each completed
        # entry triggers the deferred batch_generator.remove(uid) on the
        # inference thread. Inflight entries are left for a later step.
        self._drain_pending_async_removes()

        # Check memory pressure and evict if needed (tiered cache)
        if self.memory_monitor is not None:
            self._check_memory_pressure()

        try:
            # Schedule waiting requests
            scheduled, rejected = self._schedule_waiting()
            output.scheduled_request_ids = [r.request_id for r in scheduled]
            output.num_scheduled_tokens = sum(r.num_prompt_tokens for r in scheduled)
            if rejected:
                output.outputs.extend(rejected)
                output.has_work = True

            # Run generation step if we have running requests.
            # Use next_generated() which returns only GenerationBatch.Response
            # objects (prefill is handled externally before insert).
            if self.batch_generator is not None and self.running:
                responses = self.batch_generator.next_generated()
                output.has_work = True

                if responses:
                    outputs, finished_ids = self._process_batch_responses(responses)
                    output.outputs = outputs
                    output.finished_request_ids = finished_ids
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
                    self._tokens_since_clear_cache = (
                        getattr(self, "_tokens_since_clear_cache", 0)
                        + len(responses)
                    )
                    if self._tokens_since_clear_cache >= 1024:
                        mx.synchronize(generation_stream)
                        mx.clear_cache()
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

        except Exception as e:
            import traceback

            logger.error(
                f"Error in batch generation step: {e}\n" f"{traceback.format_exc()}"
            )
            raise

        # Clear finished tracking for next step
        self.finished_req_ids = set()

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
            _sync_and_clear_cache()
        if (
            self.config.gc_cleanup_interval > 0
            and self._step_counter % self.config.gc_cleanup_interval == 0
        ):
            gc.collect()

        return output

    def get_request(self, request_id: str) -> Request | None:
        """Get a request by ID."""
        return self.requests.get(request_id)

    def remove_finished_request(self, request_id: str) -> Request | None:
        """Remove a finished request from tracking."""
        return self.requests.pop(request_id, None)

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        stats = {
            "num_waiting": len(self.waiting),
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
        self.running.clear()
        self.requests.clear()
        self.finished_req_ids.clear()
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()
        self.batch_generator = None
        self._current_sampler_params = None
        self._boundary_cache_snapshots.clear()
        if self._boundary_snapshot_store is not None:
            self._boundary_snapshot_store.cleanup_all()
        self._boundary_snapshot_required = None

        # Clear caches
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()

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
                    concurrent.futures.wait(inflight, timeout=30.0)
                self._drain_pending_async_removes()
                self._store_cache_executor.shutdown(wait=True)
            except Exception as e:
                logger.warning(f"Async store_cache shutdown error: {e}")
            self._store_cache_executor = None
        if self.paged_ssd_cache_manager is not None:
            self.paged_ssd_cache_manager.close()
            self.paged_ssd_cache_manager = None
        logger.info("Scheduler shutdown completed")

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

            # Extract KV cache dimensions
            num_layers = getattr(config, "num_hidden_layers", None) or getattr(
                config, "n_layer", None
            )
            num_kv_heads = (
                getattr(config, "num_key_value_heads", None)
                or getattr(config, "num_attention_heads", None)
                or getattr(config, "n_head", None)
            )
            head_dim = getattr(config, "head_dim", None)
            hidden_size = getattr(config, "hidden_size", None) or getattr(
                config, "n_embd", None
            )

            # Calculate head_dim if not directly available
            if head_dim is None and hidden_size and num_kv_heads:
                num_heads = getattr(config, "num_attention_heads", None) or num_kv_heads
                head_dim = hidden_size // num_heads

            # Determine dtype size
            dtype_size = 2  # Default float16
            if hasattr(self.model, "dtype"):
                if self.model.dtype == mx.float32:
                    dtype_size = 4
                elif self.model.dtype == mx.bfloat16:
                    dtype_size = 2

            # Extract num_attention_heads (query heads) for SDPA peak estimation
            num_attention_heads = (
                getattr(config, "num_attention_heads", None)
                or getattr(config, "n_head", None)
                or num_kv_heads
            )

            # Count KVCache layers for hybrid models
            num_kv_cache_layers = num_layers
            if hasattr(self.model, "make_cache"):
                try:
                    cache_list = self.model.make_cache()
                    from mlx_lm.models.cache import KVCache

                    num_kv_cache_layers = sum(
                        1 for c in cache_list if type(c) is KVCache
                    )
                    if num_kv_cache_layers == 0:
                        num_kv_cache_layers = num_layers  # fallback
                except Exception:
                    pass

            if num_layers and num_kv_heads and head_dim:
                self.memory_monitor.set_model_info(
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    dtype_size=dtype_size,
                    num_attention_heads=num_attention_heads,
                    num_kv_cache_layers=num_kv_cache_layers,
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

    def _init_tiered_cache(self) -> None:
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
            return

        # In paged SSD-only mode, paged_ssd_cache_dir is required
        if not self.config.paged_ssd_cache_dir:
            logger.debug(
                "paged SSD cache not configured (no --ssd-cache-dir specified)"
            )
            return

        try:
            cache_dir = (
                Path(self.config.paged_ssd_cache_dir)
                if self.config.paged_ssd_cache_dir
                else None
            )

            # Initialize paged SSD cache manager for SSD storage
            self.paged_ssd_cache_manager = PagedSSDCacheManager(
                cache_dir=cache_dir,
                max_size_bytes=self.config.paged_ssd_cache_max_size,
                hot_cache_max_bytes=self.config.hot_cache_max_size,
                hot_cache_only=self.config.hot_cache_only,
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

            logger.info(
                f"paged SSD cache enabled: "
                f"cache_dir={self.config.paged_ssd_cache_dir}, "
                f"max_size={self._format_bytes(self.config.paged_ssd_cache_max_size)}, "
                f"block_size={self.config.paged_cache_block_size} tokens"
            )

        except Exception as e:
            logger.error(f"Failed to initialize paged SSD cache: {e}")
            self.paged_ssd_cache_manager = None

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

    def get_ssd_cache_stats(self) -> dict[str, Any] | None:
        """Get paged SSD + prefix cache observability statistics."""
        stats = {}

        if self.paged_ssd_cache_manager is not None:
            stats["ssd_cache"] = self.paged_ssd_cache_manager.get_stats()

        if self.paged_cache_manager is not None:
            # In paged SSD-only mode, all cache data is on paged SSD
            stats["indexed_blocks"] = self.paged_cache_manager.cold_block_count
            stats["block_size"] = self.config.paged_cache_block_size

        if self.block_aware_cache is not None:
            # Expose prefix-cache observability so UI can distinguish
            # "0 indexed blocks" from "sub-block cached (<block_size)".
            stats["prefix_cache"] = self.block_aware_cache.get_stats_dict()

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
