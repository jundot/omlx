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

import gc
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import mlx.core as mx
from mlx_lm.generate import (
    Batch as MLXBatch,
    BatchGenerator,
    _left_pad_prompts,
    _make_cache,
    _merge_caches,
    _right_pad_prompts,
    generation_stream,
)
from mlx_lm.sample_utils import make_sampler, make_logits_processors

from pathlib import Path

from .cache.paged_cache import PagedCacheManager
from .cache.prefix_cache import BlockAwarePrefixCache
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .exceptions import is_cache_corruption_error

# Import tiered cache components
try:
    from .cache.paged_ssd_cache import PagedSSDCacheManager
    from .memory_monitor import MemoryMonitor

    HAS_TIERED_CACHE = True
except ImportError:
    PagedSSDCacheManager = None
    MemoryMonitor = None
    HAS_TIERED_CACHE = False

# Import cache type handlers for hybrid cache support
try:
    from .cache.type_registry import CacheTypeRegistry
    from .cache.hybrid_cache import ModelCacheConfig
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

# Import Harmony adapter for gpt-oss models
try:
    from .adapter.harmony import HarmonyStreamingParser, parse_tool_calls_from_tokens
    from .utils.tokenizer import is_harmony_model

    HAS_HARMONY_ADAPTER = True
except ImportError:
    HarmonyStreamingParser = None
    parse_tool_calls_from_tokens = None
    is_harmony_model = None
    HAS_HARMONY_ADAPTER = False

logger = logging.getLogger(__name__)


class _BoundarySnapshotBatchGenerator(BatchGenerator):
    """BatchGenerator with boundary-aligned prefill snapshot callbacks."""

    def __init__(
        self,
        *args: Any,
        boundary_block_size: int = 0,
        prefill_boundary_callback: Optional[Callable[[int, List[Any], int], None]] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._boundary_block_size = max(0, int(boundary_block_size))
        self._prefill_boundary_callback = prefill_boundary_callback

    def _boundary_capture_enabled(self) -> bool:
        return (
            self._boundary_block_size > 0
            and self._prefill_boundary_callback is not None
        )

    @staticmethod
    def _cache_layer_token_count(cache_obj: Any) -> int:
        sub_caches = getattr(cache_obj, "caches", None)
        if isinstance(sub_caches, (list, tuple)) and sub_caches:
            return max(
                _BoundarySnapshotBatchGenerator._cache_layer_token_count(sub_cache)
                for sub_cache in sub_caches
            )

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

    @staticmethod
    def _cache_base_sizes(caches: Tuple[List[Any], ...]) -> List[int]:
        base_sizes: List[int] = []
        for cache_list in caches:
            try:
                base_sizes.append(
                    max(
                        _BoundarySnapshotBatchGenerator._cache_layer_token_count(c)
                        for c in cache_list
                    )
                )
            except Exception:
                base_sizes.append(0)
        return base_sizes

    def _next_boundary_limited_step(
        self,
        processed_tokens: int,
        lengths: List[int],
        base_sizes: List[int],
        max_allowed: int,
        target_boundaries: List[Optional[int]],
    ) -> int:
        n_to_process = min(self.prefill_step_size, max_allowed)
        if n_to_process <= 1:
            return max(1, n_to_process)

        next_delta: Optional[int] = None

        for length, base, target_boundary in zip(lengths, base_sizes, target_boundaries):
            if target_boundary is None:
                continue

            prefill_limit = max(length - 1, 0)
            if processed_tokens >= prefill_limit:
                continue

            current_prefill = min(processed_tokens, prefill_limit)
            current_total = base + current_prefill
            if current_total >= target_boundary:
                continue

            target_prefill = target_boundary - base
            delta = target_prefill - processed_tokens
            if delta <= 0:
                continue

            next_delta = delta if next_delta is None else min(next_delta, delta)

        if next_delta is not None:
            n_to_process = min(n_to_process, next_delta)

        return max(1, n_to_process)

    def _emit_boundary_snapshots(
        self,
        *,
        uids: List[int],
        lengths: List[int],
        base_sizes: List[int],
        prompt_cache: List[Any],
        emitted: Dict[int, int],
        processed_tokens: Optional[int] = None,
        use_full_prompt_lengths: bool = False,
    ) -> None:
        if not self._boundary_capture_enabled() or self._prefill_boundary_callback is None:
            return

        for idx, uid in enumerate(uids):
            length = lengths[idx]
            base = base_sizes[idx]
            if use_full_prompt_lengths:
                total_tokens = base + max(length, 0)
            else:
                if processed_tokens is None:
                    continue
                prefill_done = min(processed_tokens, max(length - 1, 0))
                total_tokens = base + prefill_done

            if total_tokens <= 0 or total_tokens % self._boundary_block_size != 0:
                continue
            if emitted.get(uid, -1) >= total_tokens:
                continue

            try:
                snapshot_cache = [c.extract(idx) for c in prompt_cache]
                self._prefill_boundary_callback(uid, snapshot_cache, total_tokens)
                emitted[uid] = total_tokens
            except Exception as e:
                logger.debug(
                    f"Prefill boundary snapshot callback failed for uid={uid}: {e}"
                )

    def _process_prompts(self, prompts):
        uids, inputs, max_tokens, caches, samplers, logits_processors = zip(*prompts)

        lengths = [len(p) for p in inputs]
        max_length = max(lengths)
        padding = [max_length - l for l in lengths]

        self._stats.prompt_tokens += sum(lengths)

        tokens = [mx.array(inp) for inp in inputs]
        processed_tokens = 0
        boundary_enabled = self._boundary_capture_enabled()
        base_sizes = self._cache_base_sizes(caches) if boundary_enabled else []
        target_boundaries: List[Optional[int]] = []
        if boundary_enabled:
            block_size = self._boundary_block_size
            for length, base in zip(lengths, base_sizes):
                full_total = base + max(length, 0)
                if full_total <= 0 or full_total % block_size == 0:
                    target_boundaries.append(None)
                    continue

                # Capture only the last full boundary below the full prompt
                # length. This avoids splitting prefill at every boundary.
                target_boundary = (full_total // block_size) * block_size
                target_boundaries.append(target_boundary if target_boundary > base else None)

        emitted_boundaries: Dict[int, int] = {uid: -1 for uid in uids}

        # New prompts so
        #   1. Left-pad the inputs
        #   2. Process
        if all(c[0].empty() for c in caches):
            inputs = _left_pad_prompts(inputs, max_length=max_length)
            prompt_cache = _make_cache(self.model, padding, self.max_kv_size)

            while inputs.shape[1] > 1:
                max_allowed = inputs.shape[1] - 1
                if boundary_enabled:
                    n_to_process = self._next_boundary_limited_step(
                        processed_tokens,
                        lengths,
                        base_sizes,
                        max_allowed,
                        target_boundaries,
                    )
                else:
                    n_to_process = min(self.prefill_step_size, max_allowed)

                self.model(inputs[:, :n_to_process], cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])
                inputs = inputs[:, n_to_process:]
                processed_tokens += n_to_process
                self.prompt_progress_callback(
                    [
                        (uid, processed_tokens, length)
                        for uid, length in zip(uids, lengths)
                    ]
                )
                self._emit_boundary_snapshots(
                    uids=list(uids),
                    lengths=lengths,
                    base_sizes=base_sizes,
                    prompt_cache=prompt_cache,
                    emitted=emitted_boundaries,
                    processed_tokens=processed_tokens,
                )
                mx.clear_cache()

        # Further prompt processing so we need to
        #   1. Merge the KV caches and prepare for right padded prompts
        #   2. Right pad the inputs
        #   2. Process
        #   3. Finalize the KV caches so they are left padded again
        else:
            last_inputs = mx.array([p[-1:] for p in inputs])
            inputs = _right_pad_prompts(inputs, max_length=max_length)
            prompt_cache = _merge_caches(caches)

            for c in prompt_cache:
                # subtract one from lengths since we don't process the last token during prefill
                c.prepare(lengths=[l - 1 for l in lengths], right_padding=padding)

            while inputs.shape[1] > 1:
                max_allowed = inputs.shape[1] - 1
                if boundary_enabled:
                    n_to_process = self._next_boundary_limited_step(
                        processed_tokens,
                        lengths,
                        base_sizes,
                        max_allowed,
                        target_boundaries,
                    )
                else:
                    n_to_process = min(self.prefill_step_size, max_allowed)

                self.model(inputs[:, :n_to_process], cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])
                inputs = inputs[:, n_to_process:]
                processed_tokens += n_to_process
                self.prompt_progress_callback(
                    [
                        (uid, processed_tokens, length)
                        for uid, length in zip(uids, lengths)
                    ]
                )
                self._emit_boundary_snapshots(
                    uids=list(uids),
                    lengths=lengths,
                    base_sizes=base_sizes,
                    prompt_cache=prompt_cache,
                    emitted=emitted_boundaries,
                    processed_tokens=processed_tokens,
                )
                mx.clear_cache()

            mx.eval([c.state for c in prompt_cache])
            inputs = last_inputs

        for c in prompt_cache:
            c.finalize()
        mx.clear_cache()

        y, logprobs = self._step(
            inputs, prompt_cache, samplers, logits_processors, tokens
        )

        self._emit_boundary_snapshots(
            uids=list(uids),
            lengths=lengths,
            base_sizes=base_sizes,
            prompt_cache=prompt_cache,
            emitted=emitted_boundaries,
            use_full_prompt_lengths=True,
        )

        mx.async_eval(y, logprobs)

        return MLXBatch(
            list(uids),
            y,
            logprobs,
            list(max_tokens),
            [0] * len(uids),
            prompt_cache,
            list(samplers),
            list(logits_processors),
            tokens,
        )


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
    prefill_batch_size: int = 8
    completion_batch_size: int = 32
    prefill_step_size: int = 2048

    # Paged cache settings (internal defaults)
    paged_cache_block_size: int = 256  # Tokens per block
    max_cache_blocks: Optional[int] = None  # Auto-calculated from available KV cache memory
    initial_cache_blocks: int = 256  # Starting blocks (grows dynamically to max_cache_blocks)

    # paged SSD cache settings (oMLX only supports paged SSD-based caching)
    # When paged_ssd_cache_dir is set, oMLX stores KV cache on paged SSD for prefix reuse.
    # When None, no oMLX caching (mlx-lm BatchGenerator manages KV internally).
    paged_ssd_cache_dir: Optional[str] = None  # Path for paged SSD cache storage (None = disabled)
    paged_ssd_cache_max_size: int = 100 * 1024 * 1024 * 1024  # 100GB default

    # Model identification (for cache isolation between different models)
    model_name: str = ""  # OpenAI API model name (e.g., "mlx-community/Llama-3.2-3B")

    # GC/cleanup settings (memory optimization)
    gc_cleanup_interval: int = 0  # Steps between gc.collect() calls (0=disabled)
    mlx_cache_cleanup_interval: int = 32  # Steps between mx.clear_cache() calls


@dataclass
class SchedulerOutput:
    """
    Output from a scheduling step.

    Contains information about what was scheduled and results.
    """

    # Requests scheduled in this step
    scheduled_request_ids: List[str] = field(default_factory=list)
    # Total tokens scheduled
    num_scheduled_tokens: int = 0
    # Requests that finished in this step
    finished_request_ids: Set[str] = field(default_factory=set)
    # Request outputs (tokens generated)
    outputs: List[RequestOutput] = field(default_factory=list)
    # Whether any work was done
    has_work: bool = False


class Scheduler:
    """
    Scheduler for continuous batching using mlx-lm BatchGenerator.

    This scheduler manages the lifecycle of requests:
    1. Requests arrive and are added to the waiting queue
    2. Scheduler moves requests from waiting to running (via BatchGenerator)
    3. BatchGenerator processes all running requests together
    4. Finished requests are removed and outputs returned

    The key insight is that mlx-lm's BatchGenerator already implements
    continuous batching at the token level, so we use it as the backend.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[SchedulerConfig] = None,
    ):
        """
        Initialize the scheduler.

        Args:
            model: The MLX model
            tokenizer: The tokenizer
            config: Scheduler configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SchedulerConfig()

        # For strict RotatingKVCache reuse, align paged cache block size to
        # the model's rotating window size when paged cache is enabled.
        self._align_block_size_with_rotating_window()

        # Request management - following vLLM's design
        self.waiting: deque[Request] = deque()  # Waiting queue (FCFS)
        self.running: Dict[str, Request] = {}  # Running requests by ID
        self.requests: Dict[str, Request] = {}  # All requests by ID
        self.finished_req_ids: Set[str] = set()  # Recently finished

        # Thread-safe set for deferred aborts (main thread → executor thread)
        # CPython GIL guarantees set.add() and `x in set` are atomic.
        self._pending_abort_ids: Set[str] = set()

        # Mapping between our request IDs and BatchGenerator UIDs
        self.request_id_to_uid: Dict[str, int] = {}
        self.uid_to_request_id: Dict[int, str] = {}

        # BatchGenerator - the actual batching engine
        self.batch_generator: Optional[BatchGenerator] = None
        self._current_sampler_params: Optional[Tuple] = None
        # Boundary cache snapshots for stateful non-sliceable caches (e.g., ArraysCache).
        # request_id -> (snapshot_cache, token_count_at_snapshot)
        self._boundary_cache_snapshots: Dict[str, Tuple[List[Any], int]] = {}
        # Lazy detection flag: True/False once determined, None before first check.
        self._boundary_snapshot_required: Optional[bool] = None

        # paged SSD cache for KV state persistence (oMLX only supports paged SSD-based caching)
        self.paged_cache_manager: Optional[PagedCacheManager] = None
        self.block_aware_cache: Optional[BlockAwarePrefixCache] = None
        self.paged_ssd_cache_manager: Optional["PagedSSDCacheManager"] = None
        self.memory_monitor: Optional["MemoryMonitor"] = None

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
        else:
            logger.info("oMLX cache disabled (mlx-lm BatchGenerator manages KV internally)")

        # Streaming detokenizers for proper UTF-8 handling (one per active request)
        # NOTE: No pooling - each request gets a fresh instance to prevent state contamination
        self._request_detokenizers: Dict[str, Any] = {}  # request_id → active detokenizer

        # Harmony model support (gpt-oss)
        self._harmony_parser: Optional["HarmonyStreamingParser"] = None
        self._is_harmony_model: bool = False
        self._harmony_parsers: Dict[str, "HarmonyStreamingParser"] = {}  # Per-request parsers
        self._harmony_stop_tokens: Optional[Set[int]] = None  # Cached stop tokens
        if HAS_HARMONY_ADAPTER and is_harmony_model is not None:
            # Check if model uses Harmony format
            try:
                model_config = None
                if hasattr(model, 'config'):
                    # model.config may be a Pydantic model or dict
                    try:
                        if hasattr(model.config, 'model_dump'):
                            model_config = model.config.model_dump()
                        elif hasattr(model.config, 'dict'):
                            model_config = model.config.dict()
                        elif isinstance(model.config, dict):
                            model_config = model.config
                        else:
                            # Try to convert to dict via __dict__
                            model_config = getattr(model.config, '__dict__', None)
                    except Exception as e:
                        logger.debug(f"Failed to extract model.config: {e}")
                elif hasattr(model, 'args'):
                    try:
                        if hasattr(model.args, 'model_dump'):
                            model_config = model.args.model_dump()
                        elif hasattr(model.args, '__dict__'):
                            model_config = model.args.__dict__
                    except Exception as e:
                        logger.debug(f"Failed to extract model.args: {e}")

                if is_harmony_model(self.config.model_name, model_config):
                    self._is_harmony_model = True
                    # Cache Harmony stop tokens to avoid creating temporary parsers
                    temp_parser = HarmonyStreamingParser(self.tokenizer)
                    self._harmony_stop_tokens = temp_parser.get_stop_token_ids()
                    logger.info(
                        f"Harmony model detected: {self.config.model_name}, "
                        f"stop_tokens={self._harmony_stop_tokens}, "
                        "streaming parser will be used for each request"
                    )
            except Exception as e:
                logger.warning(f"Error detecting Harmony model: {e}, assuming non-Harmony")
                self._is_harmony_model = False

        # Statistics
        self.num_requests_processed = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        # Step counter for periodic cleanup
        self._step_counter = 0

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
        window_sizes: Set[int],
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

    def _detect_rotating_window_sizes(self) -> Set[int]:
        """Detect rotating window sizes from model.make_cache() if available."""
        if not hasattr(self.model, "make_cache"):
            return set()

        try:
            cache_list = self.model.make_cache()
        except Exception as e:
            logger.debug(f"Failed to inspect model rotating window sizes: {e}")
            return set()

        window_sizes: Set[int] = set()
        for cache_obj in cache_list:
            self._collect_rotating_window_sizes(cache_obj, window_sizes)

        return window_sizes

    def _align_block_size_with_rotating_window(self) -> None:
        """
        Align paged cache block size with RotatingKVCache window size.

        Strict rotating snapshot restore assumes block boundaries coincide with
        the rotating window size.
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

        target_block_size = next(iter(window_sizes))
        if self.config.paged_cache_block_size != target_block_size:
            logger.warning(
                "Aligning paged cache block_size=%s to RotatingKVCache "
                "window_size=%s for strict boundary snapshots",
                self.config.paged_cache_block_size,
                target_block_size,
            )
            self.config.paged_cache_block_size = target_block_size

    def _get_stop_tokens(self) -> Set[int]:
        """Get stop token IDs from tokenizer."""
        stop_tokens = set()
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, list):
                stop_tokens.update(self.tokenizer.eos_token_id)
            else:
                stop_tokens.add(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'eos_token_ids'):
            stop_tokens.update(self.tokenizer.eos_token_ids)

        # Add Harmony stop tokens for gpt-oss models (use cached tokens)
        if self._is_harmony_model and self._harmony_stop_tokens:
            stop_tokens.update(self._harmony_stop_tokens)

        return stop_tokens

    def _update_stop_tokens(self):
        """
        Update BatchGenerator.stop_tokens to union of all running requests.

        This prevents stop token pollution where tokens from completed requests
        persist and affect new requests.
        """
        if self.batch_generator is None:
            return

        # Start with base stop tokens from tokenizer
        stop_tokens = self._get_stop_tokens()

        # Add custom stop tokens from all currently running requests
        for request in self.running.values():
            if request.sampling_params.stop_token_ids:
                stop_tokens.update(request.sampling_params.stop_token_ids)

        # Update BatchGenerator's stop tokens
        self.batch_generator.stop_tokens = stop_tokens

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
            if hasattr(self.tokenizer, 'detokenizer'):
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

    def _get_harmony_parser(self, request_id: str) -> Optional["HarmonyStreamingParser"]:
        """Get or create a Harmony parser for a request.

        Each request gets its own parser instance to maintain independent state.
        """
        if not self._is_harmony_model or not HAS_HARMONY_ADAPTER:
            return None

        if request_id not in self._harmony_parsers:
            self._harmony_parsers[request_id] = HarmonyStreamingParser(self.tokenizer)
        return self._harmony_parsers[request_id]

    def _get_harmony_detokenizer(self, request_id: str) -> Any:
        """Get or create a streaming detokenizer for Harmony.

        Uses a single detokenizer per request since we process tokens sequentially.
        For final channel where stream_token == visible_token, we decode once
        and reuse the result.

        This ensures proper UTF-8 handling for multi-byte characters.
        """
        key = f"{request_id}_harmony"

        if key not in self._request_detokenizers:
            if NaiveStreamingDetokenizer is not None:
                detok = NaiveStreamingDetokenizer(self.tokenizer)
            elif hasattr(self.tokenizer, 'detokenizer'):
                detok = self.tokenizer.detokenizer
            else:
                return None

            detok.reset()
            self._request_detokenizers[key] = detok

        return self._request_detokenizers.get(key)

    def _cleanup_harmony_parser(self, request_id: str):
        """Clean up Harmony parser and detokenizer for a finished request."""
        parser = self._harmony_parsers.pop(request_id, None)
        if parser is not None:
            try:
                # Ensure parser is finalized before cleanup
                parser.finalize()
            except Exception as e:
                logger.debug(f"Error finalizing Harmony parser for {request_id}: {e}")
        self._request_detokenizers.pop(f"{request_id}_harmony", None)

    def _create_batch_generator(self, sampling_params: SamplingParams) -> BatchGenerator:
        """Create a BatchGenerator with the given sampling parameters."""
        sampler = make_sampler(
            temp=sampling_params.temperature,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
            top_k=sampling_params.top_k,
        )

        # Create logits processors for repetition penalty
        logits_processors = make_logits_processors(
            repetition_penalty=sampling_params.repetition_penalty
            if sampling_params.repetition_penalty != 1.0
            else None,
        )

        stop_tokens = self._get_stop_tokens()
        # Add custom stop token IDs
        if sampling_params.stop_token_ids:
            stop_tokens.update(sampling_params.stop_token_ids)

        return _BoundarySnapshotBatchGenerator(
            model=self.model,
            max_tokens=sampling_params.max_tokens,
            stop_tokens=stop_tokens,
            sampler=sampler,
            logits_processors=logits_processors if logits_processors else None,
            prefill_batch_size=self.config.prefill_batch_size,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
            boundary_block_size=self.config.paged_cache_block_size,
            prefill_boundary_callback=(
                self._on_prefill_boundary_snapshot
                if self.block_aware_cache is not None
                else None
            ),
        )

    def _build_sampler_and_processors(
        self, sampling_params: SamplingParams
    ) -> Tuple[Callable[[mx.array], mx.array], List[Callable]]:
        """Build per-request sampler and logits processors."""
        sampler = make_sampler(
            temp=sampling_params.temperature,
            top_p=sampling_params.top_p,
            min_p=sampling_params.min_p,
            top_k=sampling_params.top_k,
        )
        logits_processors = make_logits_processors(
            repetition_penalty=sampling_params.repetition_penalty
            if sampling_params.repetition_penalty != 1.0
            else None,
        )
        return sampler, logits_processors

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
        # CacheList nests multiple cache objects.
        sub_caches = getattr(cache_obj, "caches", None)
        if isinstance(sub_caches, (list, tuple)):
            return any(
                self._cache_tree_has_stateful_non_sliceable(sub_cache)
                for sub_cache in sub_caches
            )

        class_name = type(cache_obj).__name__

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

    def _cache_list_needs_boundary_snapshot(self, cache_list: List[Any]) -> bool:
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
        snapshot_cache: List[Any],
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

        prev_snapshot = self._boundary_cache_snapshots.get(request_id)
        if prev_snapshot is not None and prev_snapshot[1] >= token_count:
            return

        self._boundary_cache_snapshots[request_id] = (snapshot_cache, token_count)
        self._boundary_snapshot_required = True
        logger.debug(
            "Captured prefill boundary cache snapshot for %s at %s tokens",
            request_id,
            token_count,
        )

    def _detect_boundary_snapshot_need(self) -> bool:
        """
        Determine whether boundary snapshots are needed for the current model.

        This is evaluated lazily from the active batch cache structure.
        """
        if self._boundary_snapshot_required is not None:
            return self._boundary_snapshot_required

        if self.batch_generator is None or self.batch_generator.active_batch is None:
            return False

        batch_cache = getattr(self.batch_generator.active_batch, "cache", None)
        if not batch_cache:
            self._boundary_snapshot_required = False
            return False

        self._boundary_snapshot_required = any(
            self._cache_tree_has_stateful_non_sliceable(layer_cache)
            for layer_cache in batch_cache
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

    def _extract_boundary_snapshot(self, uid: int) -> Optional[List[Any]]:
        """Extract a per-request prompt cache snapshot from the active batch."""
        if self.batch_generator is None or self.batch_generator.active_batch is None:
            return None

        batch = self.batch_generator.active_batch
        uids = getattr(batch, "uids", None)
        if not isinstance(uids, list):
            return None

        try:
            idx = uids.index(uid)
        except ValueError:
            return None

        try:
            # Synchronize pending generation_stream operations before
            # accessing batch cache tensors.
            # BatchGenerator.next() uses mx.async_eval on generation_stream,
            # and accessing the same batch.cache arrays from a different
            # stream causes Metal command buffer conflicts.
            mx.synchronize(generation_stream)
            # Keep extraction on generation_stream as well. ArraysCache layers
            # can return tensor views that alias active batch buffers; creating
            # and later evaluating those views on a different stream can still
            # trigger Metal command buffer races.
            with mx.stream(generation_stream):
                return batch.extract_cache(idx)
        except Exception as e:
            logger.debug(f"Failed to extract boundary cache snapshot for uid={uid}: {e}")
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

        self._boundary_cache_snapshots[request.request_id] = (
            snapshot_cache,
            total_tokens,
        )
        logger.debug(
            f"Captured boundary cache snapshot for {request.request_id} at "
            f"{total_tokens} tokens"
        )

    def _get_boundary_store_override(
        self,
        request_id: str,
        full_token_sequence: List[int],
    ) -> Optional[Tuple[List[int], List[Dict[str, Any]], Optional["ModelCacheConfig"]]]:
        """
        Return boundary-aligned cache payload when final request ends on partial block.
        """
        snapshot = self._boundary_cache_snapshots.get(request_id)
        if not snapshot:
            return None

        snapshot_cache, snapshot_token_count = snapshot
        total_tokens = len(full_token_sequence)
        block_size = self.config.paged_cache_block_size

        # Only apply when final sequence has trailing partial tokens.
        if snapshot_token_count <= 0 or snapshot_token_count >= total_tokens:
            return None
        if snapshot_token_count % block_size != 0:
            return None

        extracted_cache, model_cache_config = self._extract_cache_states(snapshot_cache)
        if not extracted_cache:
            return None

        return (
            full_token_sequence[:snapshot_token_count],
            extracted_cache,
            model_cache_config,
        )

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
                if hasattr(layer_cache, 'keys') and layer_cache.keys is None:
                    if hasattr(layer_cache, 'max_size'):
                        continue  # Valid empty RotatingKVCache (keys=None)
                    return False
                if hasattr(layer_cache, 'values') and layer_cache.values is None:
                    if hasattr(layer_cache, 'max_size'):
                        continue  # Valid empty RotatingKVCache (values=None)
                    return False

        # Check BatchKVCache structure
        if hasattr(cache, 'caches'):
            if cache.caches is None:
                return False
            for c in cache.caches:
                if c is None:
                    return False

        return True

    def _normalize_rotating_snapshot_state(
        self,
        layer_cache: Any,
        state: Tuple[Any, Any],
        meta_state: Any,
    ) -> Tuple[Tuple[Any, Any], Tuple[str, str, str, str]]:
        """
        Normalize RotatingKVCache state into merge-safe canonical form.

        Boundary snapshots captured mid-prefill can expose oversized rotating
        buffers (e.g., max_size + chunk_size - 1). Those states are valid for
        in-flight prefill but break BatchRotatingKVCache.merge() after SSD
        restore because merge expects per-request rotating buffers capped to
        max_size. This method canonicalizes to the latest max_size tokens.
        """
        if not isinstance(state, (list, tuple)) or len(state) < 2:
            return state, tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ()

        keys = state[0]
        values = state[1]
        if keys is None or values is None or not hasattr(keys, "shape"):
            return state, tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ()

        try:
            keep = int(meta_state[0]) if meta_state and len(meta_state) >= 1 else int(getattr(layer_cache, "keep", 0))
            max_size = int(meta_state[1]) if meta_state and len(meta_state) >= 2 else int(getattr(layer_cache, "max_size", keys.shape[2]))
            offset = int(meta_state[2]) if meta_state and len(meta_state) >= 3 else int(getattr(layer_cache, "offset", keys.shape[2]))
            idx = int(meta_state[3]) if meta_state and len(meta_state) >= 4 else int(getattr(layer_cache, "_idx", keys.shape[2]))
        except Exception:
            return state, tuple(meta_state) if isinstance(meta_state, (list, tuple)) else ()

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

        normalized_len = int(normalized_keys.shape[2]) if len(normalized_keys.shape) >= 3 else 0
        effective_offset = max(0, offset)
        if max_size > 0 and effective_offset >= max_size:
            normalized_idx = min(normalized_len, max_size)
        elif effective_offset > 0:
            normalized_idx = min(normalized_len, effective_offset)
        else:
            normalized_idx = min(normalized_len, max(0, idx))

        normalized_meta = (
            str(keep),
            str(max_size),
            str(offset),
            str(normalized_idx),
        )

        if original_len != normalized_len or idx != normalized_idx:
            logger.debug(
                "Normalized RotatingKVCache snapshot: len %s->%s, idx %s->%s, "
                "offset=%s, max_size=%s",
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
        raw_cache: List[Any],
    ) -> Tuple[List[Dict[str, Any]], Optional["ModelCacheConfig"]]:
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

        # Build ModelCacheConfig for type information
        model_cache_config = None
        if HAS_CACHE_TYPE_HANDLERS and ModelCacheConfig is not None:
            try:
                model_cache_config = ModelCacheConfig.from_cache_list(
                    raw_cache, model_name=self.model_name if hasattr(self, 'model_name') else ""
                )
            except Exception as e:
                logger.debug(f"Failed to build ModelCacheConfig: {e}")

        extracted = []
        for layer_idx, layer_cache in enumerate(raw_cache):
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
                if cache_type_name == 'CacheList' or class_name == 'CacheList':
                    if HAS_CACHE_TYPE_HANDLERS and CacheTypeRegistry is not None:
                        try:
                            handler = CacheTypeRegistry.get_handler_by_class_name('CacheList')
                            state_dict = handler.extract_state(layer_cache)
                            extracted.append({
                                'state': state_dict.get('sub_states', []),
                                'meta_state': (
                                    state_dict.get('sub_class_names', []),
                                    state_dict.get('sub_meta_states', []),
                                ),
                                'class_name': 'CacheList',
                                'cache_type': 'CacheList',
                            })
                        except Exception as e:
                            logger.debug(f"CacheList handler extraction failed: {e}")
                            extracted.append({
                                'state': [],
                                'meta_state': ([], []),
                                'class_name': 'CacheList',
                                'cache_type': 'CacheList',
                            })
                    else:
                        # Fallback: extract sub-cache state/meta without handlers
                        # MUST append to extracted to prevent layer count mismatch (Issue #1)
                        sub_caches = getattr(layer_cache, 'caches', ())
                        sub_states = []
                        sub_class_names = []
                        sub_meta_states = []
                        for sc in sub_caches:
                            sub_states.append(sc.state if hasattr(sc, 'state') else ())
                            sub_class_names.append(type(sc).__name__)
                            sub_meta_states.append(getattr(sc, 'meta_state', ()))
                        extracted.append({
                            'state': sub_states,
                            'meta_state': (sub_class_names, sub_meta_states),
                            'class_name': 'CacheList',
                            'cache_type': 'CacheList',
                        })
                    continue

                if hasattr(layer_cache, 'state'):
                    state = layer_cache.state
                    meta = getattr(layer_cache, 'meta_state', ())

                    if class_name in ('RotatingKVCache', 'BatchRotatingKVCache'):
                        state, meta = self._normalize_rotating_snapshot_state(
                            layer_cache,
                            state,
                            meta,
                        )

                    # Handle different state formats
                    if isinstance(state, (list, tuple)) and len(state) >= 2:
                        # Standard KVCache: (keys, values)
                        # Or ArraysCache: [conv_state, ssm_state]
                        first, second = state[0], state[1]

                        # Validate non-None for KVCache types
                        if class_name in ('KVCache', 'RotatingKVCache', 'BatchKVCache'):
                            if first is None or second is None:
                                logger.debug(
                                    f"Layer {layer_idx} ({class_name}) has None keys/values, "
                                    f"skipping cache extraction"
                                )
                                return [], None  # Return empty - cache is corrupted

                        extracted.append({
                            'state': state,
                            'meta_state': meta,
                            'class_name': class_name,
                            'cache_type': cache_type_name,
                        })
                    else:
                        # Unexpected state format
                        logger.debug(
                            f"Layer {layer_idx} ({class_name}) has unexpected state format"
                        )
                        meta = getattr(layer_cache, 'meta_state', ())
                        extracted.append({
                            'state': (state, state),  # Duplicate for compatibility
                            'meta_state': meta,
                            'class_name': class_name,
                            'cache_type': cache_type_name,
                        })
                elif hasattr(layer_cache, 'cache'):
                    # ArraysCache style: state stored in .cache list
                    cache_list = layer_cache.cache
                    if isinstance(cache_list, list) and len(cache_list) >= 2:
                        state = (cache_list[0], cache_list[1])
                        meta = getattr(layer_cache, 'meta_state', ())
                        extracted.append({
                            'state': state,
                            'meta_state': meta,
                            'class_name': class_name,
                            'cache_type': cache_type_name,
                        })
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
                logger.debug(f"Failed to extract state from cache layer {layer_idx}: {e}")
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
                    request.remaining_tokens = request.prompt_token_ids[block_table.num_tokens:]
                    # For exact prefix hits we need cache state at (N-1) and the
                    # last prompt token as input to produce the first decode logit.
                    # Reusing cache state at N and feeding the last token again
                    # shifts the model state and can change greedy output.
                    if len(request.remaining_tokens) == 0 and request.cached_tokens > 0:
                        if self._cache_list_needs_boundary_snapshot(request.prompt_cache):
                            # Stateful non-sliceable caches (Rotating/Arrays)
                            # cannot be safely converted from N to N-1 state
                            # without cache-type-specific logic.
                            if self.paged_cache_manager is not None:
                                self.paged_cache_manager.delete_block_table(request.request_id)
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
                        elif self._trim_prompt_cache_for_generation(request.prompt_cache):
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
                                self.paged_cache_manager.delete_block_table(request.request_id)
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

        # Add to tracking
        self.requests[request.request_id] = request
        self.waiting.append(request)

        logger.debug(
            f"Added request {request.request_id} with {request.num_prompt_tokens} prompt tokens"
        )

    def _trim_prompt_cache_for_generation(self, cache_list: List[Any]) -> bool:
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
            return all(self._trim_cache_tree_by_one(sub_cache) for sub_cache in sub_caches)

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
        """Remove UID from active batch safely on generation_stream."""
        if self.batch_generator is None:
            return

        batch = self.batch_generator.active_batch
        if batch is None:
            return

        batch_uids = getattr(batch, "uids", None)
        if not isinstance(batch_uids, list) or uid not in batch_uids:
            return

        self.batch_generator.remove([uid])

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
            self._remove_uid_from_active_batch(uid)
            del self.uid_to_request_id[uid]
            del self.request_id_to_uid[request.request_id]

        if request_id in self.running:
            del self.running[request_id]

        # Release blocks for eviction (same as _cleanup_finished)
        if self.paged_cache_manager is not None:
            block_table = self.paged_cache_manager.get_block_table(request_id)
            if block_table is None and hasattr(request, 'block_table'):
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

        # Clean up Harmony parser
        self._cleanup_harmony_parser(request_id)

        # Drop any boundary snapshot for this request.
        self._boundary_cache_snapshots.pop(request_id, None)

        # Mark as aborted
        request.set_finished(RequestStatus.FINISHED_ABORTED)
        self.finished_req_ids.add(request_id)

        logger.debug(f"Aborted request {request_id}")
        return True

    def has_requests(self) -> bool:
        """Check if there are any pending or running requests."""
        return bool(self.waiting or self.running)

    def get_num_waiting(self) -> int:
        """Get number of waiting requests."""
        return len(self.waiting)

    def get_num_running(self) -> int:
        """Get number of running requests."""
        return len(self.running)

    def _schedule_waiting(self) -> List[Request]:
        """
        Move requests from waiting queue to running.

        Note: mlx-lm's _merge_caches has a bug where mixing cached and non-cached
        requests causes TypeError. To work around this, we only schedule requests
        with the same cache status (all with cache or all without) in a single batch.

        Returns:
            List of requests that were scheduled
        """
        scheduled = []

        # Track cache status of first scheduled request to ensure homogeneity
        # None = not determined yet, True = has cache, False = no cache
        batch_cache_status: Optional[bool] = None

        while self.waiting and len(self.running) < self.config.max_num_seqs:
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
            if request.remaining_tokens is not None and len(request.remaining_tokens) == 0:
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

            # Check cache status homogeneity to avoid mlx-lm _merge_caches bug
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

            # Per-request sampler/logits processors to avoid BatchGenerator recreation.
            sampler, logits_processors = self._build_sampler_and_processors(
                request.sampling_params
            )

            # Insert into BatchGenerator with optional cache
            uids = self.batch_generator.insert(
                [tokens_to_process],
                max_tokens=[request.sampling_params.max_tokens],
                caches=[cache_to_use] if cache_to_use else None,
                samplers=[sampler],
                logits_processors=[logits_processors],
            )

            if uids:
                uid = uids[0]
                self.request_id_to_uid[request.request_id] = uid
                self.uid_to_request_id[uid] = request.request_id
                request.batch_uid = uid
                request.status = RequestStatus.RUNNING
                self.running[request.request_id] = request
                scheduled.append(request)

                # Mark as Harmony model if applicable
                if self._is_harmony_model:
                    request.is_harmony_model = True

                # Check if prompt ends with <think> token for reasoning models
                # The chat template may end with "<think>\n" so check last few tokens
                if hasattr(self.tokenizer, 'has_thinking') and self.tokenizer.has_thinking:
                    think_start_id = getattr(self.tokenizer, 'think_start_id', None)
                    if think_start_id and request.prompt_token_ids:
                        # Check last 3 tokens (covers "<think>\n" case)
                        last_tokens = request.prompt_token_ids[-3:]
                        if think_start_id in last_tokens:
                            request.needs_think_prefix = True

                self.total_prompt_tokens += request.num_prompt_tokens
                cache_info = f", {request.cached_tokens} cached" if request.cached_tokens > 0 else ""
                cache_used = "with cache" if cache_to_use else "no cache"
                logger.debug(
                    f"Scheduled request {request.request_id} (uid={uid}) "
                    f"with {len(tokens_to_process)} tokens to process "
                    f"({request.num_prompt_tokens} total){cache_info}, {cache_used}"
                )

        # Update stop tokens to reflect currently running requests
        if scheduled:
            self._update_stop_tokens()

        return scheduled

    def _process_batch_responses(
        self, responses: List[Any]
    ) -> Tuple[List[RequestOutput], Set[str]]:
        """
        Process responses from BatchGenerator.

        Args:
            responses: List of BatchGenerator.Response objects

        Returns:
            Tuple of (outputs, finished_request_ids)
        """
        outputs = []
        finished_ids = set()

        for response in responses:
            request_id = self.uid_to_request_id.get(response.uid)
            if request_id is None:
                continue

            request = self.running.get(request_id)
            if request is None:
                continue

            # Check finish reason first - don't include EOS token in output
            # (following mlx-lm's batch_generate behavior)
            is_stop = response.finish_reason == "stop"
            is_length = response.finish_reason == "length"
            is_finished = response.finish_reason is not None

            # Only append token if not stopping due to EOS token
            new_text = ""

            # Check if this request uses Harmony format
            harmony_parser = self._get_harmony_parser(request_id)

            if harmony_parser is not None:
                # Harmony model: process token through parser
                # Returns: (control_text, stream_token, visible_token, is_stop)
                control_text, stream_token, visible_token, harmony_is_stop = (
                    harmony_parser.process_token(response.token)
                )

                # Start with control text (e.g., <think>, </think>)
                new_text = control_text

                # Get Harmony detokenizer for proper UTF-8 handling
                harmony_detok = self._get_harmony_detokenizer(request_id)

                # Decode stream token if present
                if stream_token is not None:
                    if harmony_detok is not None:
                        harmony_detok.add_token(stream_token)
                        decoded_text = harmony_detok.last_segment
                    else:
                        # Fallback to single-token decode (may break UTF-8)
                        decoded_text = self.tokenizer.decode([stream_token])

                    new_text += decoded_text

                    # For final channel, stream_token == visible_token
                    # Reuse decoded text instead of decoding again
                    if visible_token is not None:
                        request.output_text += decoded_text
                elif visible_token is not None:
                    # This case shouldn't happen in current design but handle it
                    if harmony_detok is not None:
                        harmony_detok.add_token(visible_token)
                        request.output_text += harmony_detok.last_segment
                    else:
                        request.output_text += self.tokenizer.decode([visible_token])

                # Harmony stop token can override finish reason
                if harmony_is_stop and not is_finished:
                    is_finished = True
                    is_stop = True

                # For Harmony, always track all tokens including stop tokens
                # This is needed for tool call parsing (parse_tool_calls_from_tokens)
                # which requires the complete token sequence including <|call|>/<|return|>
                request.append_output_token(response.token)

            elif not is_stop:
                # Non-Harmony: standard processing
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
            # (skip for Harmony models - parser handles <think> for analysis channel)
            if harmony_parser is None and getattr(request, 'needs_think_prefix', False):
                if not getattr(request, 'think_prefix_sent', False):
                    think_tag = getattr(self.tokenizer, 'think_start', '<think>')
                    new_text = think_tag + "\n" + new_text
                    request.think_prefix_sent = True

            # Immediately discard logprobs if not requested to free memory (~800KB per response)
            # This prevents accumulation of large MLX arrays during streaming
            if (
                hasattr(response, 'logprobs')
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

                if harmony_parser is not None:
                    # Harmony: finalize parser and get any remaining stream output
                    logger.info(
                        f"Harmony finished: channel={harmony_parser.current_channel}, "
                        f"output_text='{request.output_text[:100]}...'"
                    )
                    final_control = harmony_parser.finalize()
                    if final_control:
                        output.new_text += final_control

                    # Finalize Harmony detokenizer to flush any remaining bytes
                    harmony_detok = self._get_harmony_detokenizer(request_id)
                    if harmony_detok is not None:
                        harmony_detok.finalize()
                        final_text = harmony_detok.last_segment
                        if final_text:
                            output.new_text += final_text
                            # If we were in final channel, also add to output_text
                            if harmony_parser.current_channel == "final":
                                request.output_text += final_text

                    # Extract tool calls using parse_tool_calls_from_tokens (non-streaming)
                    if parse_tool_calls_from_tokens is not None:
                        token_ids = list(request.output_token_ids)
                        logger.info(
                            f"Harmony parse_tool_calls_from_tokens: {len(token_ids)} tokens"
                        )
                        _, tool_calls = parse_tool_calls_from_tokens(token_ids)
                        logger.info(f"Harmony tool_calls extracted: {tool_calls}")
                        if tool_calls:
                            output.tool_calls = tool_calls
                            output.finish_reason = "tool_calls"

                    # For Harmony, output_text is already accumulated (final channel only)
                    output.output_text = request.output_text
                else:
                    # Non-Harmony: standard finalization
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

                # Extract cache for future reuse
                if hasattr(response, 'prompt_cache'):
                    try:
                        # prompt_cache may be callable or direct attribute
                        if callable(response.prompt_cache):
                            raw_cache = response.prompt_cache()
                        else:
                            raw_cache = response.prompt_cache

                        if raw_cache:
                            # For paged cache, extract actual tensor states
                            # This allows cache to survive BatchGenerator recreation
                            if self.block_aware_cache is not None:
                                extracted_cache, model_cache_config = self._extract_cache_states(raw_cache)
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
                logger.log(5, "Request %s generated text:\n%s", request_id, output.output_text)

            outputs.append(output)

        return outputs, finished_ids

    def _cleanup_finished(self, finished_ids: Set[str]) -> None:
        """Clean up finished requests and store caches for reuse."""
        # Synchronize pending generation_stream operations before cache storage.
        # store_cache -> mx.save_safetensors triggers implicit mx.eval() which
        # can conflict with async Metal operations on the generation stream.
        # This is needed even when active_batch is None, because _next() sets
        # active_batch = None after mx.async_eval when all requests finish.
        if finished_ids and self.block_aware_cache is not None:
            mx.synchronize(generation_stream)

        for request_id in finished_ids:
            request = self.running.get(request_id)

            # Store cache for future reuse
            if request is not None and request.prompt_token_ids:
                if self.block_aware_cache is not None:
                    # Store in paged cache
                    # Key includes both prompt and output tokens for multi-turn chat caching
                    block_table = None
                    if hasattr(request, '_extracted_cache') and request._extracted_cache is not None:
                        try:
                            full_token_sequence = list(request.prompt_token_ids) + list(request.output_token_ids)
                            token_sequence_to_store = full_token_sequence
                            cache_to_store = request._extracted_cache
                            # Get model cache config if available (for hybrid cache support)
                            model_cache_config = getattr(request, '_model_cache_config', None)

                            # Keep all tensor-touching cache store work on the
                            # generation stream to avoid cross-stream conflicts
                            # with arrays extracted from BatchGenerator caches.
                            with mx.stream(generation_stream):
                                boundary_override = self._get_boundary_store_override(
                                    request_id,
                                    full_token_sequence,
                                )
                                if boundary_override is not None:
                                    (
                                        token_sequence_to_store,
                                        cache_to_store,
                                        boundary_model_config,
                                    ) = boundary_override
                                    if boundary_model_config is not None:
                                        model_cache_config = boundary_model_config
                                    logger.info(
                                        f"Using boundary cache snapshot for {request_id}: "
                                        f"storing {len(token_sequence_to_store)}/"
                                        f"{len(full_token_sequence)} tokens "
                                        f"(skipping trailing partial block)"
                                    )

                                block_table = self.block_aware_cache.store_cache(
                                    request_id,
                                    token_sequence_to_store,
                                    cache_to_store,
                                    model_cache_config=model_cache_config,
                                )
                            logger.debug(
                                f"Stored paged cache for request {request_id} "
                                f"({len(token_sequence_to_store)} tokens stored, "
                                f"{len(full_token_sequence)} total: "
                                f"{len(request.prompt_token_ids)} prompt + "
                                f"{len(request.output_token_ids)} output)"
                            )
                            # Immediately release _extracted_cache to free copy #1
                            # (store_cache already cloned to PagedCache blocks)
                            request._extracted_cache = None
                        except Exception as e:
                            logger.debug(f"Failed to store paged cache for {request_id}: {e}")

                    # ALWAYS release blocks for eviction, even if store_cache() failed
                    # This prevents ref_count leak when _extracted_cache is None or exception occurs
                    if block_table is None and self.paged_cache_manager:
                        # Try to get existing block_table from paged cache or request
                        block_table = self.paged_cache_manager.get_block_table(request_id)
                        if block_table is None and hasattr(request, 'block_table'):
                            block_table = request.block_table

                    if block_table and self.paged_cache_manager:
                        released = self.paged_cache_manager.release_for_eviction(
                            block_table.block_ids
                        )
                        if released > 0:
                            logger.debug(
                                f"Released {released} blocks for eviction "
                                f"(request {request_id})"
                            )

                    # ALWAYS clear request entry to prevent memory leak
                    self.block_aware_cache.clear_request_entry(request_id)

            # Remove from running
            if request_id in self.running:
                del self.running[request_id]

            # Remove from BatchGenerator to free internal KV cache
            if request_id in self.request_id_to_uid:
                uid = self.request_id_to_uid[request_id]
                self._remove_uid_from_active_batch(uid)
                if uid in self.uid_to_request_id:
                    del self.uid_to_request_id[uid]
                del self.request_id_to_uid[request_id]

            # Clean up streaming detokenizer
            self._cleanup_detokenizer(request_id)

            # Clean up Harmony parser
            self._cleanup_harmony_parser(request_id)

            # Drop any boundary snapshot for this request.
            self._boundary_cache_snapshots.pop(request_id, None)

            # Track as finished
            self.finished_req_ids.add(request_id)

            # Remove from requests dict to prevent memory leak
            # Also clear cache references to release MLX arrays
            req_to_remove = self.requests.pop(request_id, None)
            if req_to_remove is not None:
                req_to_remove._extracted_cache = None
                req_to_remove.prompt_cache = None

        # Update stop tokens after cleaning up finished requests
        if finished_ids:
            self._update_stop_tokens()

    def _is_cache_corruption_error(self, error: Exception) -> bool:
        """Check if an error indicates cache corruption."""
        return is_cache_corruption_error(error)

    def _recover_from_cache_error(self) -> None:
        """Recover from cache corruption error."""
        # Clear batch generator (this is the source of the corruption)
        self.batch_generator = None
        self._current_sampler_params = None
        self._boundary_cache_snapshots.clear()
        self._boundary_snapshot_required = None

        # Clear caches
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()

        # Clear UID mappings
        self.request_id_to_uid.clear()
        self.uid_to_request_id.clear()

        # Clear detokenizer state to prevent contamination after recovery
        self._request_detokenizers.clear()

        # Clear Harmony parsers
        self._harmony_parsers.clear()

        logger.info("Cache recovery completed")

    def _reschedule_running_requests(self) -> None:
        """Move running requests back to waiting queue for retry."""
        count = len(self.running)
        for request_id, request in list(self.running.items()):
            # Reset request state
            request.status = RequestStatus.WAITING
            request.batch_uid = None
            request.prompt_cache = None
            request.cached_tokens = 0
            request.remaining_tokens = request.prompt_token_ids

            # Move to waiting queue (at front for priority)
            self.waiting.appendleft(request)
            del self.running[request_id]

        if count > 0:
            logger.info(f"Rescheduled {count} requests for retry")

    def step(self, max_retries: int = 1) -> SchedulerOutput:
        """
        Execute one scheduling step with automatic error recovery.

        This method:
        1. Schedules waiting requests into the batch
        2. Runs one generation step via BatchGenerator
        3. Processes outputs and handles finished requests
        4. Automatically recovers from cache corruption errors

        Args:
            max_retries: Number of times to retry on cache errors (default 1)

        Returns:
            SchedulerOutput with results of this step
        """
        output = SchedulerOutput()

        # Process pending aborts FIRST (thread-safe with hybrid executor)
        self._process_pending_aborts()

        # Check memory pressure and evict if needed (tiered cache)
        if self.memory_monitor is not None:
            self._check_memory_pressure()

        for attempt in range(max_retries + 1):
            try:
                # Schedule waiting requests
                scheduled = self._schedule_waiting()
                output.scheduled_request_ids = [r.request_id for r in scheduled]
                output.num_scheduled_tokens = sum(r.num_prompt_tokens for r in scheduled)

                # Run generation step if we have running requests
                if self.batch_generator is not None and self.running:
                    responses = self.batch_generator.next()
                    output.has_work = True

                    if responses:
                        outputs, finished_ids = self._process_batch_responses(responses)
                        output.outputs = outputs
                        output.finished_request_ids = finished_ids
                        self._cleanup_finished(finished_ids)

                # Success - break out of retry loop
                break

            except TypeError as e:
                # Catch the NoneType error specifically
                if self._is_cache_corruption_error(e):
                    if attempt < max_retries:
                        import traceback
                        logger.warning(
                            f"Cache corruption detected (attempt {attempt + 1}): {e}, "
                            f"performing recovery and retry..."
                        )
                        logger.debug(f"Cache corruption traceback:\n{traceback.format_exc()}")
                        # Deep reset to recover
                        self._recover_from_cache_error()
                        # Re-add any running requests back to waiting
                        self._reschedule_running_requests()
                    else:
                        logger.error(
                            f"Cache corruption not recoverable after "
                            f"{max_retries + 1} attempts"
                        )
                        raise
                else:
                    raise
            except Exception as e:
                import traceback
                logger.error(f"Error in batch generation step: {e}\n{traceback.format_exc()}")
                raise

        # Clear finished tracking for next step
        self.finished_req_ids = set()

        # Periodic Metal cache cleanup
        self._step_counter += 1
        if (
            self.config.mlx_cache_cleanup_interval > 0
            and self._step_counter % self.config.mlx_cache_cleanup_interval == 0
        ):
            mx.clear_cache()
        if (
            self.config.gc_cleanup_interval > 0
            and self._step_counter % self.config.gc_cleanup_interval == 0
        ):
            gc.collect()

        return output

    def get_request(self, request_id: str) -> Optional[Request]:
        """Get a request by ID."""
        return self.requests.get(request_id)

    def remove_finished_request(self, request_id: str) -> Optional[Request]:
        """Remove a finished request from tracking."""
        return self.requests.pop(request_id, None)

    def get_stats(self) -> Dict[str, Any]:
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

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
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
        self._boundary_snapshot_required = None

        # Clear caches
        if self.block_aware_cache is not None:
            self.block_aware_cache.clear()

        # Clear detokenizers
        self._request_detokenizers.clear()

        # Clear Harmony parsers
        self._harmony_parsers.clear()

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
        if hasattr(self.model, 'cache'):
            self.model.cache = None

        # Some MLX models store cache in layers
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer, 'cache'):
                    layer.cache = None
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'cache'):
                    layer.self_attn.cache = None

        # Force garbage collection of any lingering cache objects
        import gc
        gc.collect()

        logger.info("Deep reset completed - all caches cleared")

    def shutdown(self) -> None:
        """
        Graceful shutdown.

        paged SSD cache is NOT cleared here to allow reuse when the same model
        is reloaded. paged SSD cache cleanup happens on server startup instead.
        """
        logger.info("Scheduler shutdown initiated...")
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
            if hasattr(self.model, 'config'):
                config = self.model.config
            elif hasattr(self.model, 'args'):
                config = self.model.args

            if config is None:
                logger.debug("Could not extract model config for memory estimation")
                return

            # Extract KV cache dimensions
            num_layers = getattr(config, 'num_hidden_layers', None) or getattr(config, 'n_layer', None)
            num_kv_heads = getattr(config, 'num_key_value_heads', None) or getattr(config, 'num_attention_heads', None) or getattr(config, 'n_head', None)
            head_dim = getattr(config, 'head_dim', None)
            hidden_size = getattr(config, 'hidden_size', None) or getattr(config, 'n_embd', None)

            # Calculate head_dim if not directly available
            if head_dim is None and hidden_size and num_kv_heads:
                num_heads = getattr(config, 'num_attention_heads', None) or num_kv_heads
                head_dim = hidden_size // num_heads

            # Determine dtype size
            dtype_size = 2  # Default float16
            if hasattr(self.model, 'dtype'):
                import mlx.core as mx
                if self.model.dtype == mx.float32:
                    dtype_size = 4
                elif self.model.dtype == mx.bfloat16:
                    dtype_size = 2

            if num_layers and num_kv_heads and head_dim:
                self.memory_monitor.set_model_info(
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    dtype_size=dtype_size,
                )
                logger.debug(
                    f"Model info for memory estimation: "
                    f"layers={num_layers}, kv_heads={num_kv_heads}, "
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
            logger.debug("paged SSD cache not configured (no --ssd-cache-dir specified)")
            return

        try:
            # Initialize paged SSD cache manager for SSD storage
            self.paged_ssd_cache_manager = PagedSSDCacheManager(
                cache_dir=Path(self.config.paged_ssd_cache_dir),
                max_size_bytes=self.config.paged_ssd_cache_max_size,
            )

            # Connect paged SSD cache manager to PagedCacheManager
            if self.paged_cache_manager is not None:
                self.paged_cache_manager.set_paged_ssd_cache_manager(self.paged_ssd_cache_manager)

            # Connect paged SSD cache manager to BlockAwarePrefixCache for paged SSD-only mode
            if self.block_aware_cache is not None:
                self.block_aware_cache.set_paged_ssd_cache_manager(self.paged_ssd_cache_manager)

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
        estimated_freed = evicted_count * self.memory_monitor.estimate_block_memory(block_size)

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
        block = self.paged_cache_manager.blocks[block_id] if block_id < len(self.paged_cache_manager.blocks) else None
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

    def get_ssd_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get paged SSD cache statistics."""
        stats = {}

        if self.paged_ssd_cache_manager is not None:
            stats["ssd_cache"] = self.paged_ssd_cache_manager.get_stats()

        if self.paged_cache_manager is not None:
            # In paged SSD-only mode, all cache data is on paged SSD
            stats["indexed_blocks"] = self.paged_cache_manager.cold_block_count
            stats["block_size"] = self.config.paged_cache_block_size

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
