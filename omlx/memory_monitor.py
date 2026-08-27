# SPDX-License-Identifier: Apache-2.0
"""
Memory Monitor for oMLX paged SSD-based KV cache.

This module provides memory utilities for paged SSD-based KV cache management
on Apple Silicon unified memory.

Key features:
- GPU memory utilization tracking via MLX Metal API
- Block memory estimation for cache management
"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from omlx.cache.paged_cache import PagedCacheManager

from omlx.exceptions import PrefillMemoryExceededError, describe_ceiling_binding
from omlx.patches.deepseek_v4.indexer_dispatch import native_indexer_eligible
from omlx.utils.hardware import format_bytes, get_max_working_set_bytes

logger = logging.getLogger(__name__)

# Check if MLX Metal is available
try:
    import mlx.core as mx

    HAS_MLX_METAL = mx.metal.is_available()
except ImportError:
    HAS_MLX_METAL = False
    mx = None

# Mirrors MLX Metal ScaledDotProductAttention::use_fallback for the
# generation/inference path. Full prefill and short vector kernels support
# different head dimensions; unsupported cases fall back to an unfused
# score-matrix allocation.
_SDPA_VECTOR_QUERY_TOKEN_THRESHOLD = 8
_SDPA_FULL_SUPPORTED_HEAD_DIMS = frozenset({64, 80, 128})
_SDPA_VECTOR_SUPPORTED_HEAD_DIMS = frozenset({64, 96, 128, 256})
# Default bytes/elem for the materialized unfused score matrix when the model's
# compute dtype is unknown. MLX softmax accumulates in fp32, but the dominant
# scratch buffer is allocated at the model's compute dtype, not fp32 — measured
# ~2.1-2.2 bytes/elem on MLX 0.31.2 for a head_dim=256 prefill (fp16/bf16),
# ~4.4 for fp32. Callers that know the model dtype pass it via
# ``set_model_info(compute_dtype_size=...)``; this default covers the rare
# dim-less path and matches the fp16/bf16 majority of MLX inference models.
_SDPA_FALLBACK_SCORE_DTYPE_SIZE = 2

# Head dims whose multi-token prefill is routed to an O(L) tiled/online-softmax
# kernel instead of the unfused O(L^2) score-matrix fallback. Populated at
# runtime by the kernel patch that installs the route (see
# omlx/patches/sdpa256_attention.py); empty otherwise, so the estimate stays
# O(L^2) when no such kernel is active. Maps head_dim -> kv_tile (the kernel's
# KV block width, which bounds the per-chunk score transient).
_SDPA_TILED_PREFILL_HEAD_DIMS: dict[int, int] = {}
_SDPA_TILED_MIN_KV_LEN = 8192


def register_tiled_prefill_head_dim(
    head_dim: int, *, min_kv_len: int = 8192, kv_tile: int = 1024
) -> None:
    """Register a head_dim whose long-context prefill now uses an O(L) tiled
    kernel, so the prefill memory estimate stops charging the O(L^2) score
    matrix for it. Must be called in lockstep with installing the kernel route,
    or the guard keeps rejecting valid long-context requests."""
    global _SDPA_TILED_MIN_KV_LEN
    _SDPA_TILED_PREFILL_HEAD_DIMS[int(head_dim)] = int(kv_tile)
    _SDPA_TILED_MIN_KV_LEN = int(min_kv_len)


# Bytes/elem of a model-built additive attention bias materialized as a
# full [n_q_heads, query_tokens, kv_len] tensor per attention call (e.g.
# inkling's banded relative-position mask). None when the loaded model
# builds no such tensor. The fused-SDPA head_dim check cannot see this
# allocation — it happens in model code before SDPA — so admission would
# otherwise under-count exactly the long-context prefills that OOM.
_ATTENTION_BIAS_TRANSIENT_DTYPE_SIZE: float | None = None


def register_attention_bias_transient(dtype_size: float | None) -> None:
    """Register (or clear with ``None``) a per-call additive attention-bias
    materialization so prefill admission prices it. Call in lockstep with
    model load/swap: the setting is process-wide, like the tiled head_dim
    registry above."""
    global _ATTENTION_BIAS_TRANSIENT_DTYPE_SIZE
    _ATTENTION_BIAS_TRANSIENT_DTYPE_SIZE = float(dtype_size) if dtype_size else None


def estimate_unfused_sdpa_call_bytes(
    n_q_heads: int,
    query_tokens: int,
    kv_len: int,
    head_dim: int,
    score_dtype_size: float = _SDPA_FALLBACK_SCORE_DTYPE_SIZE,
) -> int:
    """Transient bytes for ONE SDPA call taking the unfused fallback: the
    materialized ``[n_q, query_tokens, kv_len]`` score matrix plus the fp32
    output. Shared by the per-request prefill-peak estimate
    (``MemoryMonitor._estimate_sdpa_activation_bytes``) and the sdpa256 route
    gate (``patches/sdpa256_attention._tiled_route_required``) so the guard
    and the router price the unfused path with the same math (issue #2204)."""
    scores = n_q_heads * query_tokens * kv_len * score_dtype_size
    output = n_q_heads * query_tokens * head_dim * 4
    return int(scores + output)


@dataclass
class MemoryInfo:
    """
    Current GPU memory state.

    Attributes:
        total_bytes: Total available GPU memory
        used_bytes: Currently used memory (estimated)
        available_bytes: Available memory
        utilization: Memory utilization ratio (0.0 to 1.0)
    """

    total_bytes: int
    used_bytes: int
    available_bytes: int
    utilization: float


class PrefillMemoryProfile(Protocol):
    """Model-specific prefill memory strategy used by ``MemoryMonitor``.

    Most models use the monitor's uniform KV/SDPA formulas. Architectures
    whose cache and attention shapes cannot be represented by those formulas
    can provide this small internal strategy instead.
    """

    def estimate_resident_kv_bytes(
        self, num_tokens: int, *, chunk_tokens: int = 1
    ) -> int: ...

    def estimate_prefill_transient_bytes(
        self, query_tokens: int, kv_len: int
    ) -> int: ...


class MemoryMonitor:
    """
    Memory monitor for paged SSD-based KV cache.

    In paged SSD-only mode, KV cache data is stored on paged SSD, not GPU memory.
    This class provides memory estimation utilities for block management
    but does not trigger GPU memory-based eviction.

    Example:
        >>> monitor = MemoryMonitor(max_kv_cache_memory=2 * 1024**3)
        >>> block_mem = monitor.estimate_block_memory(64)  # 64 tokens
    """

    def __init__(
        self,
        max_kv_cache_memory: int | None,
        check_interval: float = 1.0,
        *,
        eviction_enabled: bool = True,
    ):
        """
        Initialize the memory monitor.

        Args:
            max_kv_cache_memory: Maximum memory for KV cache in bytes.
                Required when ``eviction_enabled=True``. May be ``None``
                (or 0) when the monitor is used only for prefill-peak
                estimation and no eviction/pressure decisions are made
                against this limit.
            check_interval: Minimum seconds between memory checks (for throttling).
            eviction_enabled: When False, ``max_kv_cache_memory`` is not
                consulted and estimation methods that depend on it raise.
                Set False on schedulers in paged-SSD-only mode where the
                monitor exists solely for prefill-peak estimation.
        """
        if eviction_enabled and (
            max_kv_cache_memory is None or max_kv_cache_memory <= 0
        ):
            raise ValueError(
                "max_kv_cache_memory must be positive when "
                f"eviction_enabled=True, got {max_kv_cache_memory}"
            )

        self._max_kv_cache_memory = max_kv_cache_memory or 0
        self._eviction_enabled = eviction_enabled
        # Public accessor — callers (Scheduler._evict_blocks_*) need a way
        # to skip the eviction code path without reaching into a private
        # attribute and without triggering a RuntimeError from
        # estimate_blocks_to_free().
        self._check_interval = check_interval
        self._max_memory = self._get_max_memory()

        self._last_check_time = 0.0
        self._last_memory_info: Optional[MemoryInfo] = None
        self._lock = threading.Lock()

        # Model info for memory estimation (set by scheduler)
        self._num_layers: Optional[int] = None
        self._num_kv_heads: Optional[int] = None
        self._head_dim: Optional[int] = None
        # KV storage width; may be fractional with TurboQuant.
        self._dtype_size: float = 2
        self._kv_bytes_per_token_override: float | None = None
        # SDPA score-matrix width = model compute/activation dtype, distinct from
        # _dtype_size (which the scheduler may override to a fractional TurboQuant
        # KV width). Set via set_model_info(compute_dtype_size=...).
        self._score_dtype_size: float = _SDPA_FALLBACK_SCORE_DTYPE_SIZE
        self._num_attention_heads: Optional[int] = None
        self._num_kv_cache_layers: Optional[int] = None
        # Sliding-window (RotatingKVCache-family) layers as (count, window)
        # groups. Their resident KV is bounded at window + chunk - 1 tokens
        # per layer, so they need a capped term instead of the linear
        # full-attention formula. Empty for non-hybrid models.
        self._rotating_layer_specs: tuple[tuple[int, int], ...] = ()
        self._prefill_memory_profile: PrefillMemoryProfile | None = None
        # Fixed-shape ANE prefill I/O surfaces (issue #2841); set via
        # set_model_info, 0 unless the Qwen ANE prefill backend is attached.
        self._ane_prefill_transient_bytes: int = 0
        # Fixed per-sequence recurrent state (GDN/Mamba ArraysCache),
        # measured once from a live cache after the first prefill chunk.
        self._fixed_state_bytes: int = 0

        # PagedCacheManager for KV cache memory measurement
        self._paged_cache_manager: Optional["PagedCacheManager"] = None
        self._block_size: int = 256  # Default block size

        # Baseline memory (model weights) - set after model load
        self._baseline_memory: int = 0

        # Request stats (set by scheduler for logging)
        self._running_requests: int = 0
        self._waiting_requests: int = 0

        if self._eviction_enabled:
            logger.info(
                "MemoryMonitor initialized: max_kv_cache=%s",
                format_bytes(self._max_kv_cache_memory),
            )
        else:
            logger.info("MemoryMonitor initialized (estimator-only, eviction disabled)")

    def _get_max_memory(self) -> int:
        """
        Get max_recommended_working_set_size from MLX Metal.

        Falls back to system memory heuristic if MLX Metal unavailable.

        Returns:
            Maximum memory in bytes that can be used.
        """
        return get_max_working_set_bytes()

    def set_paged_cache_manager(
        self, manager: "PagedCacheManager", block_size: int = 64
    ) -> None:
        """
        Set PagedCacheManager for memory monitoring.

        Args:
            manager: PagedCacheManager instance
            block_size: Number of tokens per block
        """
        self._paged_cache_manager = manager
        self._block_size = block_size
        logger.info(
            f"PagedCacheManager connected for memory monitoring "
            f"(block_size={block_size})"
        )

    def set_baseline_memory(self) -> None:
        """
        Set baseline memory after model load.

        Call this after loading the model to capture the baseline memory usage
        (model weights, etc.). The KV cache memory is calculated as:
        active_memory - baseline_memory

        This allows accurate detection of memory pressure from KV cache growth
        while ignoring static model memory.
        """
        if HAS_MLX_METAL:
            try:
                self._baseline_memory = mx.get_active_memory()
                logger.info(
                    f"Baseline memory set: {format_bytes(self._baseline_memory)}"
                )
            except Exception as e:
                logger.warning(f"Failed to set baseline memory: {e}")
                self._baseline_memory = 0
        else:
            self._baseline_memory = 0
            logger.warning("MLX Metal not available, baseline memory set to 0")

    def set_request_stats(self, running: int, waiting: int) -> None:
        """
        Update request stats for logging.

        Args:
            running: Number of currently running requests
            waiting: Number of waiting requests
        """
        self._running_requests = running
        self._waiting_requests = waiting

    def _get_current_memory_usage(self) -> int:
        """
        Get current KV cache memory usage.

        In paged SSD-only mode, returns 0 since KV cache data is stored on paged SSD,
        not GPU memory. PagedCacheManager only holds metadata.

        Returns:
            0 in paged SSD-only mode (no GPU memory used for KV cache).
        """
        # In paged SSD-only mode, PagedCache doesn't hold GPU memory
        # All KV cache data is on paged SSD
        return 0

    def _get_process_rss(self) -> int:
        """
        Get process RSS memory (fallback method).

        Returns:
            Process resident set size in bytes.
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return 0

    def get_memory_info(self) -> MemoryInfo:
        """
        Get current memory state.

        Returns:
            MemoryInfo with current memory statistics.
        """
        with self._lock:
            current_time = time.time()

            # Throttle checks to avoid overhead
            if (
                self._last_memory_info is not None
                and current_time - self._last_check_time < self._check_interval
            ):
                return self._last_memory_info

            used = self._get_current_memory_usage()
            available = max(0, self._max_memory - used)
            utilization = used / self._max_memory if self._max_memory > 0 else 0.0

            self._last_memory_info = MemoryInfo(
                total_bytes=self._max_memory,
                used_bytes=used,
                available_bytes=available,
                utilization=utilization,
            )
            self._last_check_time = current_time

            return self._last_memory_info

    def is_under_pressure(self) -> bool:
        """
        Check if memory pressure exists.

        In paged SSD-only mode, always returns False since KV cache data
        is stored on paged SSD, not GPU memory.

        Returns:
            False in paged SSD-only mode.
        """
        return False

    def bytes_to_free(self) -> int:
        """
        Calculate bytes needed to free.

        In paged SSD-only mode, always returns 0 since KV cache data
        is stored on paged SSD, not GPU memory.

        Returns:
            0 in paged SSD-only mode.
        """
        # In paged SSD-only mode, no memory to free from KV cache
        return 0

    def clear_ane_prefill_transient(self) -> None:
        """Stop reserving ANE prefill I/O surfaces after the banks are shed.

        The reservation is snapshotted at load; once the runtime headroom
        rung releases the banks the surfaces are gone, so keeping the term
        would make every later admission pass pause for memory that can no
        longer be reclaimed. The next load re-prices it via set_model_info.
        """
        self._ane_prefill_transient_bytes = 0

    def set_model_info(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype_size: float = 2,
        num_attention_heads: Optional[int] = None,
        num_kv_cache_layers: Optional[int] = None,
        compute_dtype_size: Optional[float] = None,
        kv_bytes_per_token: Optional[float] = None,
        rotating_layer_specs: Sequence[tuple[int, int]] | None = None,
        prefill_memory_profile: PrefillMemoryProfile | None = None,
        ane_prefill_transient_bytes: int = 0,
    ) -> None:
        """
        Set model information for memory estimation.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV attention heads
            head_dim: Dimension per attention head
            dtype_size: Bytes per element of the *stored KV cache*. This may
                be fractional for quantized (e.g. TurboQuant) KV layouts.
            num_attention_heads: Number of query attention heads (for SDPA
                peak estimation). Defaults to num_kv_heads if not set.
            num_kv_cache_layers: Number of layers that use KVCache
                (full attention). For hybrid models this may be less than
                num_layers. Defaults to num_layers.
            compute_dtype_size: Bytes per element of the model's
                compute/activation dtype (2 for fp16/bf16, 4 for fp32). Used
                for the unfused SDPA score-matrix transient, which is allocated
                at the activation dtype regardless of KV quantization. Defaults
                to the fp16/bf16 fallback when unknown.
            kv_bytes_per_token: Optional exact resident KV-cache bytes added
                per token. Use for compressed-cache architectures such as MLA
                where stored cache tensors are not representable as uniform
                ``num_kv_heads * head_dim * 2`` K/V tensors.
            rotating_layer_specs: Sliding-window layer groups as
                ``(layer_count, window_tokens)`` pairs, used by
                ``estimate_resident_kv_bytes`` for the window-capped KV term.
                These layers are excluded from ``num_kv_cache_layers``.
            prefill_memory_profile: Optional model-specific strategy for cache
                and prefill transient shapes that the uniform estimator cannot
                represent.
        """
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._dtype_size = dtype_size
        self._score_dtype_size = (
            compute_dtype_size
            if compute_dtype_size and compute_dtype_size > 0
            else _SDPA_FALLBACK_SCORE_DTYPE_SIZE
        )
        self._kv_bytes_per_token_override = (
            float(kv_bytes_per_token)
            if kv_bytes_per_token is not None and kv_bytes_per_token > 0
            else None
        )
        self._num_attention_heads = num_attention_heads or num_kv_heads
        # ``is not None`` (not truthiness): a genuine 0 means "no
        # full-attention layers" for rotating-only hybrids, and coercing it
        # to num_layers would double-count on top of the rotating term.
        self._num_kv_cache_layers = (
            num_kv_cache_layers if num_kv_cache_layers is not None else num_layers
        )
        self._rotating_layer_specs = tuple(
            (int(count), int(window))
            for count, window in (rotating_layer_specs or ())
            if count > 0 and window > 0
        )
        self._prefill_memory_profile = prefill_memory_profile
        # ANE prefill I/O surfaces are dirtied by the first long prompt, on
        # top of the KV+SDPA peak, so admission must price them or the hard
        # watermark aborts the request mid-prefill (issue #2841).
        self._ane_prefill_transient_bytes = max(int(ane_prefill_transient_bytes), 0)
        # A new model's fixed state must be re-measured; a stale value from
        # the previous model would silently mis-charge admission.
        self._fixed_state_bytes = 0

        # Log estimated memory per block
        if num_layers and num_kv_heads and head_dim:
            sample_block_mem = self.estimate_block_memory(64)  # 64 tokens
            override_note = (
                ", KV override "
                f"{format_bytes(int(self._kv_bytes_per_token_override))}/tok"
                if self._kv_bytes_per_token_override
                else ""
            )
            rotating_note = (
                ", rotating "
                + "+".join(
                    f"{count}x@{window}" for count, window in self._rotating_layer_specs
                )
                if self._rotating_layer_specs
                else ""
            )
            logger.info(
                f"Model info set: {num_layers} layers "
                f"({self._num_kv_cache_layers} KVCache{rotating_note}), "
                f"{num_kv_heads} KV heads, "
                f"{self._num_attention_heads} Q heads, "
                f"{head_dim} head_dim. Estimated memory per 64-token block: "
                f"{format_bytes(sample_block_mem)}{override_note}"
            )

    def set_fixed_state_bytes(self, n: int) -> None:
        """Record the per-sequence fixed recurrent-state footprint.

        Measured once from live ArraysCache-family caches after the first
        prefill chunk (state shapes are unknown before the first forward).
        Cleared by ``set_model_info`` so a model swap never inherits the
        previous model's measurement.
        """
        self._fixed_state_bytes = max(0, int(n))

    @property
    def fixed_state_bytes(self) -> int:
        """Measured per-sequence recurrent state bytes (0 until measured)."""
        return self._fixed_state_bytes

    def has_model_info(self) -> bool:
        """Whether ``set_model_info`` has been called with real dims.

        ``estimate_block_memory`` silently substitutes ``32 layers / 8 KV
        heads / 128 head_dim`` (a 7B-class assumption) when dims are
        unset, which means callers can't tell "real model" from
        "estimator default" by inspecting the return value. Use this
        accessor when the difference matters — e.g. the PagedSSDCache
        writer-queue cap formula prefers its own 200 KB fallback over
        the monitor's 128 KB default-fiction.
        """
        return (
            self._num_layers is not None
            and self._num_layers > 0
            and self._num_kv_heads is not None
            and self._num_kv_heads > 0
            and self._head_dim is not None
            and self._head_dim > 0
        )

    def estimate_block_memory(
        self,
        block_size: int,
        num_layers: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dtype_size: Optional[float] = None,
    ) -> float:
        """
        Estimate memory usage for a KV cache block.

        The default layer count is the number of layers that retain KV state.
        When cache layout information is unavailable, it falls back to the
        configured transformer layer count.

        Args:
            block_size: Number of tokens in the block
            num_layers: Override the stored KV-cache layer count
            num_kv_heads: Override stored num_kv_heads
            head_dim: Override stored head_dim
            dtype_size: Override stored dtype_size

        Returns:
            Estimated memory in bytes for one block.
        """
        # A block grows with the layers that retain per-token KV state, not
        # with recurrent or other layers whose state is fixed per sequence.
        # Keep an explicit ``num_layers`` override for callers that need a
        # custom estimate, while preserving a genuine zero for rotating-only
        # models.
        if num_layers is not None:
            layers = num_layers
        elif self._num_kv_cache_layers is not None:
            layers = self._num_kv_cache_layers
        else:
            layers = self._num_layers or 32  # Default for ~7B model
        kv_heads = num_kv_heads or self._num_kv_heads or 8
        dim = head_dim or self._head_dim or 128
        dtype = dtype_size or self._dtype_size

        if (
            self._kv_bytes_per_token_override is not None
            and num_layers is None
            and num_kv_heads is None
            and head_dim is None
            and dtype_size is None
        ):
            return block_size * self._kv_bytes_per_token_override

        # Memory per layer: keys + values
        # Shape: (batch=1, kv_heads, block_size, head_dim)
        per_layer = block_size * kv_heads * dim * dtype * 2  # *2 for keys+values
        total = per_layer * layers

        return total

    def estimate_prompt_kv_bytes(self, num_tokens: int) -> float:
        """
        Estimate KV cache memory for a prompt of given length.

        Uses per-layer cache type info if available (hybrid models),
        otherwise falls back to uniform num_layers estimate.

        Args:
            num_tokens: Number of prompt tokens.

        Returns:
            Estimated KV cache memory in bytes.
        """
        if self._prefill_memory_profile is not None:
            return self._prefill_memory_profile.estimate_resident_kv_bytes(
                num_tokens,
                chunk_tokens=max(int(num_tokens), 1),
            )

        # A genuine 0 means the model has no full-attention KVCache layers.
        # Falling back by truthiness charges every rotating-only layer as a
        # full linear cache (issue #2521).
        layers = self._num_kv_cache_layers
        if layers is None:
            layers = self._num_layers or 0
        kv_heads = self._num_kv_heads or 0
        dim = self._head_dim or 0
        dtype = self._dtype_size

        if not (layers and kv_heads and dim):
            return 0

        if self._kv_bytes_per_token_override is not None:
            return num_tokens * self._kv_bytes_per_token_override

        # KVCache layers: memory grows with num_tokens
        per_token = layers * kv_heads * dim * dtype * 2  # keys + values
        return num_tokens * per_token

    def estimate_resident_kv_bytes(
        self, num_tokens: int, *, chunk_tokens: int = 1
    ) -> float:
        """Exact-shape resident KV bytes a prefill of ``num_tokens`` adds.

        Extends ``estimate_prompt_kv_bytes`` (full-attention layers only,
        linear in tokens) with the two layer groups that formula cannot see:

        - Sliding-window layers: each holds at most ``window + chunk - 1``
          tokens (``RotatingKVCache._update_concat`` concatenates the chunk
          before ``_trim``), so their term saturates instead of growing
          linearly. Priced at the base/compute dtype: rotating layers are
          pass-through for TurboQuant KV compression, so the fractional
          ``_dtype_size`` would under-count them ~4x in TQ configurations.
        - Fixed recurrent state (GDN/Mamba): a per-sequence constant,
          measured after the first chunk (0 before that — same as today).

        Admission-only: charge this once per request. Never call it from
        per-chunk sizing paths — the fixed-state term would be re-charged
        every chunk, and chunk sizing must stay on the frozen
        ``_predicted_chunk_transient`` signal.
        """
        if num_tokens <= 0:
            return 0
        if self._prefill_memory_profile is not None:
            return self._prefill_memory_profile.estimate_resident_kv_bytes(
                num_tokens, chunk_tokens=chunk_tokens
            )
        total = self.estimate_prompt_kv_bytes(num_tokens)

        if self._rotating_layer_specs:
            kv_heads = self._num_kv_heads or 0
            dim = self._head_dim or 0
            if kv_heads and dim:
                chunk = max(int(chunk_tokens), 1)
                per_token = kv_heads * dim * self._score_dtype_size * 2
                for count, window in self._rotating_layer_specs:
                    resident = min(num_tokens, window + chunk - 1)
                    total += count * resident * per_token

        return total + self._fixed_state_bytes

    def _uses_fused_sdpa(self, query_tokens: int, kv_len: int) -> bool:
        hd = self._head_dim or 0
        n_q = self._num_attention_heads or 0
        n_kv = self._num_kv_heads or n_q
        if n_q <= 0 or n_kv <= 0 or hd <= 0 or query_tokens <= 0:
            return False
        if kv_len < query_tokens:
            return False

        if query_tokens <= _SDPA_VECTOR_QUERY_TOKEN_THRESHOLD:
            gqa_factor = max(1, n_q // n_kv)
            return (
                hd in _SDPA_VECTOR_SUPPORTED_HEAD_DIMS
                and query_tokens * gqa_factor <= 32
            )

        return hd in _SDPA_FULL_SUPPORTED_HEAD_DIMS

    def _estimate_sdpa_activation_bytes(self, query_tokens: int, kv_len: int) -> int:
        hd = self._head_dim or 0
        n_q = self._num_attention_heads or 0
        if n_q == 0 or hd == 0 or query_tokens <= 0:
            return 0

        query_tokens = int(query_tokens)
        kv_len = max(int(kv_len), 0)

        # Model-built additive bias (e.g. inkling's banded mask) is
        # materialized regardless of which SDPA route runs.
        bias = 0
        if _ATTENTION_BIAS_TRANSIENT_DTYPE_SIZE is not None:
            bias = int(
                n_q * query_tokens * kv_len * _ATTENTION_BIAS_TRANSIENT_DTYPE_SIZE
            )

        output = n_q * query_tokens * hd * 4
        if self._uses_fused_sdpa(query_tokens, kv_len):
            return output + bias

        # O(L) tiled-prefill kernel active for this head_dim (e.g. the head_dim
        # 256 sdpa256 patch): the score matrix is never materialized. The peak
        # transient is the output plus one KV tile of scores, not the full
        # [n_q, query_tokens, kv_len] matrix. This matches the kernel's route
        # gate (query_len > 1, kv_len >= threshold); any query_len <= 1 already
        # returned above via the fused vector path.
        kv_tile = _SDPA_TILED_PREFILL_HEAD_DIMS.get(hd)
        if (
            kv_tile is not None
            and query_tokens > 1
            and kv_len >= _SDPA_TILED_MIN_KV_LEN
        ):
            tile_scores = (
                n_q * query_tokens * min(kv_tile, kv_len) * self._score_dtype_size
            )
            return output + tile_scores + bias

        return (
            estimate_unfused_sdpa_call_bytes(
                n_q, query_tokens, kv_len, hd, self._score_dtype_size
            )
            + bias
        )

    def estimate_prefill_peak_bytes(
        self, new_tokens: int, chunk_size: int, *, cached_tokens: int = 0
    ) -> float:
        """
        Estimate per-request prefill peak memory contribution (KV + SDPA).

        Returns only the part directly attributable to this request's prefill:
        KV cache for the new tokens being added + SDPA attention activation
        peak for the last chunk. Does NOT include model weights (already in
        active baseline), prefix-cached KV that is already resident, or MLX
        cache pool / python heap overhead (absorbed by enforcer's hard
        threshold margin — see MemorySettings.hard_threshold).

        MLX SDPA only uses fused full-attention kernels for the head dimensions
        supported by ``ScaledDotProductAttention::use_fallback``. Unsupported
        prefill chunks fall back to an unfused fp32 score matrix whose K
        dimension spans the full key/value context. With prefix-cache hits,
        that context is ``new_tokens + cached_tokens``, not just the new suffix.
        Passing only ``new_tokens`` here silently under-counts long-context
        prefill, exactly where prefix caching makes such requests possible.

        Args:
            new_tokens: Tokens being prefilled this request (prompt minus
                what the prefix cache already covers). Drives newly
                allocated KV and the last chunk's query length.
            chunk_size: Prefill step size (default 2048). Effective chunk
                is ``min(chunk_size, new_tokens)`` since the last chunk
                cannot be larger than the remaining new tokens.
            cached_tokens: Tokens served from prefix cache. Added to
                ``new_tokens`` for the SDPA scores K-dim because those
                positions still participate in attention. Keyword-only with
                a default of 0 so callers that don't know the cache state
                still typecheck — but they get the under-counting behavior
                this method was designed to fix, so always pass it when the
                value is available.

        Returns:
            Per-request peak contribution in bytes (KV + SDPA). Returns 0 if
            model info is not available. Caller compares this against
            `(hard_threshold * max_bytes) - current_usage_bytes` —
            the margin handles cache pool / python heap / compressed memory.
        """
        hd = self._head_dim or 0
        n_q = self._num_attention_heads or 0

        if n_q == 0 or hd == 0:
            return 0  # can't estimate

        if new_tokens <= 0:
            return 0

        # Effective chunk: bounded by the remaining new tokens. Short
        # prompts (smaller than chunk_size) would otherwise be charged the
        # full chunk_size width in the scores tensor, over-estimating by
        # chunk_size / new_tokens — a constant-factor over-count that
        # raised false-positive 400s on small prompts.
        eff_chunk = min(chunk_size, new_tokens)
        full_kv_len = new_tokens + max(cached_tokens, 0)
        attn = self._estimate_sdpa_activation_bytes(eff_chunk, full_kv_len)

        # KV growth attributable to this request: only the new tokens.
        # The cached portion is already counted in the caller's current-usage
        # baseline. Resident math includes window-capped sliding-window
        # layers and measured fixed state, not just full-attention KVCache.
        kv = self.estimate_resident_kv_bytes(new_tokens, chunk_tokens=eff_chunk)
        return attn + kv + self._ane_prefill_transient_bytes

    def estimate_chunk_transient_bytes(self, n_tokens: int, kv_len: int) -> int:
        """Transient SDPA activation bytes for ONE prefill chunk.

        Isolates the per-chunk attention transient — the spike that drives
        prefill OOM — for a chunk of ``n_tokens`` query tokens attending over
        ``kv_len`` total context tokens. Unlike ``estimate_prefill_peak_bytes``
        this excludes newly-allocated KV (that becomes resident and is counted
        in the caller's ``current`` baseline once eval'd); it is the quantity
        the adaptive throttle must keep under the remaining headroom.

        Fused MLX SDPA uses the output-buffer estimate. Unsupported
        query/head-dim combinations use the unfused fp32 score-matrix fallback
        and scale with total ``kv_len``.

        Returns 0 when model info is unavailable.
        """
        if self._prefill_memory_profile is not None:
            return self._prefill_memory_profile.estimate_prefill_transient_bytes(
                n_tokens, kv_len
            )
        return self._estimate_sdpa_activation_bytes(n_tokens, kv_len)

    def estimate_blocks_to_free(self, bytes_to_free: int, block_size: int) -> int:
        """
        Estimate number of blocks to evict to free the given bytes.

        Args:
            bytes_to_free: Target bytes to free
            block_size: Tokens per block

        Returns:
            Number of blocks to evict.
        """
        if not self._eviction_enabled:
            raise RuntimeError(
                "estimate_blocks_to_free called on a MemoryMonitor "
                "constructed with eviction_enabled=False"
            )
        block_mem = self.estimate_block_memory(block_size)
        if block_mem <= 0:
            return 0

        # Round up to ensure we free enough
        num_blocks = int((bytes_to_free + block_mem - 1) // block_mem)
        return max(1, num_blocks)

    @property
    def max_memory(self) -> int:
        """Get maximum system memory limit."""
        return self._max_memory

    @property
    def max_kv_cache_memory(self) -> int:
        """Get maximum KV cache memory limit."""
        return self._max_kv_cache_memory

    @property
    def eviction_enabled(self) -> bool:
        """Whether this monitor was built with eviction wiring.

        Paged-SSD-only mode passes ``eviction_enabled=False`` because
        the SDPA-peak / prefill-admission paths don't need KV eviction
        math. Callers (Scheduler._evict_blocks_*) check this before
        calling ``estimate_blocks_to_free``, which would otherwise
        raise ``RuntimeError``.
        """
        return self._eviction_enabled

    def get_stats(self) -> dict:
        """
        Get memory statistics as a dictionary.

        Returns:
            Dictionary with memory statistics.
        """
        info = self.get_memory_info()
        return {
            "total_bytes": info.total_bytes,
            "used_bytes": info.used_bytes,
            "available_bytes": info.available_bytes,
            "utilization": info.utilization,
            "max_kv_cache_memory": self._max_kv_cache_memory,
            "total_formatted": format_bytes(info.total_bytes),
            "used_formatted": format_bytes(info.used_bytes),
            "available_formatted": format_bytes(info.available_bytes),
        }

    def __repr__(self) -> str:
        info = self.get_memory_info()
        return (
            f"MemoryMonitor(max_kv_cache={format_bytes(self._max_kv_cache_memory)}, "
            f"used={format_bytes(info.used_bytes)})"
        )


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _pos_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _nonnegative_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


@dataclass(frozen=True)
class _DeepSeekV4PrefillMemoryProfile:
    """Exact-shape prefill estimator for DeepSeek V4's hybrid attention.

    Every layer keeps a 128-token local K-only rotating cache. Ratio-4 layers
    additionally keep main and indexer pools and use top-k sparse attention;
    ratio-128 layers attend over a much smaller dense pool. Treating these as
    ordinary full-context K/V + dense SDPA is the 81.25 GiB false positive in
    issue #2521.
    """

    local_layers: int
    ratio4_layers: int
    ratio128_layers: int
    num_attention_heads: int
    head_dim: int
    sliding_window: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    dtype_size: float
    wsdpa_dtype_supported: bool = False

    def _wsdpa_route_active(self, *, topk: bool = False) -> bool:
        """Match the live WSDPA route without a process-wide head-dim flag."""
        if (
            not self.wsdpa_dtype_supported
            or self.num_attention_heads != 64
            or self.head_dim != 512
        ):
            return False
        try:
            from omlx.patches.deepseek_v4.wsdpa_attention import (
                wsdpa_prefill_route_active,
            )

            return wsdpa_prefill_route_active(topk=topk)
        except Exception:
            return False

    def _wsdpa_attention_bytes(
        self,
        query_tokens: int,
        local_tokens: int,
        pooled_tokens: int = 0,
        *,
        selected_tokens: int = 0,
    ) -> int:
        """Bound the custom kernel's contiguous inputs and output.

        The kernel performs online softmax in registers and never materializes
        a score matrix. Count potentially copied query/KV/top-k inputs and keep
        the generic estimator's fp32-width output bound for conservatism.
        """
        query = (
            self.num_attention_heads * query_tokens * self.head_dim * self.dtype_size
        )
        keys = (local_tokens + pooled_tokens) * self.head_dim * self.dtype_size
        output = self.num_attention_heads * query_tokens * self.head_dim * 4
        topk = query_tokens * selected_tokens * 4
        return int(query + keys + output + topk)

    def _pool_cache_elements(
        self,
        num_tokens: int,
        *,
        ratio: int,
        pooled_dim: int,
        overlap: bool,
    ) -> int:
        pooled = (num_tokens // ratio) * pooled_dim
        projection_dim = pooled_dim * (2 if overlap else 1)
        # PoolingCache allocates full KV/gate remainder buffers on first use.
        buffers = 2 * ratio * projection_dim
        # Ratio-4 compression retains the previous raw window for overlap.
        carry = 2 * ratio * projection_dim if overlap and num_tokens >= ratio else 0
        return pooled + buffers + carry

    def estimate_resident_kv_bytes(
        self, num_tokens: int, *, chunk_tokens: int = 1
    ) -> int:
        if num_tokens <= 0:
            return 0

        num_tokens = int(num_tokens)
        chunk_tokens = max(1, min(int(chunk_tokens), num_tokens))
        local_tokens = min(
            num_tokens,
            self.sliding_window + chunk_tokens - 1,
        )
        total_elements = self.local_layers * local_tokens * self.head_dim

        if self.ratio4_layers:
            main_pool = self._pool_cache_elements(
                num_tokens,
                ratio=4,
                pooled_dim=self.head_dim,
                overlap=True,
            )
            index_pool = self._pool_cache_elements(
                num_tokens,
                ratio=4,
                pooled_dim=self.index_head_dim,
                overlap=True,
            )
            total_elements += self.ratio4_layers * (main_pool + index_pool)

        if self.ratio128_layers:
            pool = self._pool_cache_elements(
                num_tokens,
                ratio=128,
                pooled_dim=self.head_dim,
                overlap=False,
            )
            total_elements += self.ratio128_layers * pool

        return int(total_elements * self.dtype_size)

    def _indexer_fallback_bytes(self, query_tokens: int, pooled_tokens: int) -> int:
        if query_tokens <= 0 or pooled_tokens <= 0:
            return 0

        score_elements = self.index_n_heads * query_tokens * pooled_tokens
        query = self.index_n_heads * query_tokens * self.index_head_dim * 4
        keys = pooled_tokens * self.index_head_dim * 4
        weights = self.index_n_heads * query_tokens * 4
        # The model splits score matmuls to keep each tensor below MLX's int32
        # indexing limit, then concatenates the reduced shards lazily. MLX can
        # keep every per-head shard live while evaluating that graph, so the
        # aggregate peak still approaches the full [H, L, P] fp32 volume.
        materialized_scores = score_elements * 4
        reduced_scores = query_tokens * pooled_tokens * 4

        # _stable_topk_indices keeps the reduced fp32 scores while building
        # region/key/partition uint32 workspaces. Six score-width buffers is a
        # conservative bound for the unfused MLX path.
        topk_workspace = 6 * reduced_scores
        selected = query_tokens * min(self.index_topk, pooled_tokens) * 4
        return query + keys + weights + materialized_scores + topk_workspace + selected

    def _indexer_native_bytes(self, query_tokens: int, pooled_tokens: int) -> int:
        """Bound the fused native indexer score and deterministic top-k path."""
        query = (
            self.index_n_heads * query_tokens * self.index_head_dim * self.dtype_size
        )
        weights = self.index_n_heads * query_tokens * self.dtype_size
        scores = query_tokens * pooled_tokens * self.dtype_size
        selected = query_tokens * self.index_topk * 4
        return int(query + weights + scores + selected)

    def _sparse_attention_bytes(
        self, query_tokens: int, local_tokens: int, pooled_tokens: int
    ) -> int:
        selected = min(self.index_topk, pooled_tokens)
        gathered = query_tokens * selected * self.head_dim * self.dtype_size
        score_width = (
            self.num_attention_heads * query_tokens * (local_tokens + selected)
        )
        scores_and_weights = 2 * score_width * self.dtype_size
        query = (
            self.num_attention_heads * query_tokens * self.head_dim * self.dtype_size
        )
        output = self.num_attention_heads * query_tokens * self.head_dim * 4
        return int(gathered + scores_and_weights + query + output)

    def estimate_prefill_transient_bytes(self, query_tokens: int, kv_len: int) -> int:
        if query_tokens <= 0 or kv_len <= 0:
            return 0

        query_tokens = int(query_tokens)
        kv_len = int(kv_len)
        local_tokens = min(
            kv_len,
            self.sliding_window + query_tokens - 1,
        )
        candidates: list[int] = []

        if self.local_layers:
            if query_tokens > 1 and self._wsdpa_route_active():
                local_attention = self._wsdpa_attention_bytes(
                    query_tokens, local_tokens
                )
            else:
                local_attention = estimate_unfused_sdpa_call_bytes(
                    self.num_attention_heads,
                    query_tokens,
                    local_tokens,
                    self.head_dim,
                    self.dtype_size,
                )
            candidates.append(local_attention)

        if self.ratio128_layers:
            pooled_tokens = kv_len // 128
            attended = local_tokens + pooled_tokens
            projection = 2 * query_tokens * self.head_dim * self.dtype_size
            if query_tokens > 1 and self._wsdpa_route_active():
                attention = self._wsdpa_attention_bytes(
                    query_tokens,
                    local_tokens,
                    pooled_tokens,
                )
                candidates.append(int(projection) + attention)
            else:
                concat = attended * self.head_dim * self.dtype_size
                candidates.append(
                    int(projection + concat)
                    + estimate_unfused_sdpa_call_bytes(
                        self.num_attention_heads,
                        query_tokens,
                        attended,
                        self.head_dim,
                        self.dtype_size,
                    )
                )

        if self.ratio4_layers:
            pooled_tokens = kv_len // 4
            # Main and index compressors each project KV + gate at twice the
            # pooled width on overlap layers.
            projections = int(
                4
                * query_tokens
                * (self.head_dim + self.index_head_dim)
                * self.dtype_size
            )
            if native_indexer_eligible(
                query_tokens=query_tokens,
                pooled_tokens=pooled_tokens,
                n_heads=self.index_n_heads,
                head_dim=self.index_head_dim,
                index_topk=self.index_topk,
                dtype_supported=self.dtype_size == 2,
            ):
                indexer = self._indexer_native_bytes(query_tokens, pooled_tokens)
            else:
                indexer = self._indexer_fallback_bytes(query_tokens, pooled_tokens)
            if pooled_tokens <= self.index_topk:
                attended = local_tokens + pooled_tokens
                if query_tokens > 1 and self._wsdpa_route_active():
                    attention = self._wsdpa_attention_bytes(
                        query_tokens,
                        local_tokens,
                        pooled_tokens,
                    )
                else:
                    attention = estimate_unfused_sdpa_call_bytes(
                        self.num_attention_heads,
                        query_tokens,
                        attended,
                        self.head_dim,
                        self.dtype_size,
                    )
            else:
                if query_tokens > 4 and self._wsdpa_route_active(topk=True):
                    attention = self._wsdpa_attention_bytes(
                        query_tokens,
                        local_tokens,
                        pooled_tokens,
                        selected_tokens=self.index_topk,
                    )
                else:
                    attention = self._sparse_attention_bytes(
                        query_tokens,
                        local_tokens,
                        pooled_tokens,
                    )
            # Both are evaluated within the same layer graph. Summing them is
            # deliberately conservative for lazy MLX execution and remains far
            # below the impossible dense full-context SDPA charge.
            candidates.append(projections + indexer + attention)

        return max(candidates, default=0)


def make_prefill_memory_profile(
    config: Any,
    *,
    compute_dtype_size: float,
    wsdpa_dtype_supported: bool = False,
) -> PrefillMemoryProfile | None:
    """Build the one model-specific prefill strategy currently required."""
    model_type = str(_cfg_get(config, "model_type", "") or "")
    if not model_type.startswith("deepseek_v4"):
        return None

    num_layers = _cfg_get(config, "num_hidden_layers")
    ratios = _cfg_get(config, "compress_ratios")
    if (
        not _pos_int(num_layers)
        or not isinstance(ratios, Sequence)
        or isinstance(ratios, (str, bytes))
    ):
        return None
    ratios = tuple(ratios[:num_layers])
    if len(ratios) != num_layers or any(ratio not in (0, 4, 128) for ratio in ratios):
        return None

    num_attention_heads = _cfg_get(config, "num_attention_heads")
    head_dim = _cfg_get(config, "head_dim")
    sliding_window = _cfg_get(config, "sliding_window")
    index_n_heads = _cfg_get(config, "index_n_heads")
    index_head_dim = _cfg_get(config, "index_head_dim")
    index_topk = _cfg_get(config, "index_topk")
    required = (
        num_attention_heads,
        head_dim,
        sliding_window,
        index_n_heads,
        index_head_dim,
        index_topk,
    )
    if not all(_pos_int(value) for value in required):
        return None
    if not isinstance(compute_dtype_size, (int, float)) or compute_dtype_size <= 0:
        return None

    counts = Counter(ratios)
    return _DeepSeekV4PrefillMemoryProfile(
        local_layers=num_layers,
        ratio4_layers=counts[4],
        ratio128_layers=counts[128],
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        dtype_size=float(compute_dtype_size),
        wsdpa_dtype_supported=bool(wsdpa_dtype_supported),
    )


# Known rotating-cache class names, matching the scheduler-side sets
# (Scheduler._collect_rotating_window_sizes and the TurboQuant eligibility
# walk). Backstop only — duck-typing on (positive int max_size, keep attr)
# is the primary test so renamed omlx subclasses still classify.
_ROTATING_CACHE_CLASS_NAMES = frozenset(
    {
        "RotatingKVCache",
        "BatchRotatingKVCache",
        "PrefillReadyRotatingKVCache",
        "BufferedRotatingKVCache",
    }
)
# Fixed-state recurrent caches (GDN/Mamba). Matches
# Scheduler._cache_tree_has_arrays_cache plus mlx-lm's MambaCache.
_ARRAYS_CACHE_CLASS_NAMES = frozenset({"ArraysCache", "SizedArraysCache", "MambaCache"})
_FULL_KV_CACHE_CLASS_NAMES = frozenset(
    {"QSAKVCache", "QSAQuantizedKVCache", "BatchQSAKVCache"}
)


def collect_kv_layer_specs(
    cache_list: Any,
) -> tuple[int, list[tuple[int, int]], int]:
    """Classify a ``model.make_cache()`` result into KV-layer groups.

    Returns ``(full_kv_layers, rotating_specs, arrays_layers)`` where
    ``rotating_specs`` groups sliding-window layers as
    ``(layer_count, window_tokens)`` pairs. Full-attention layers keep the
    strict ``type(c) is KVCache`` test the hybrid-layer counting has always
    used; rotating layers are duck-typed (positive int ``max_size`` plus a
    ``keep`` attribute) with a class-name backstop, avoiding a dependency on
    scheduler-side registries. Returns all-zero on any failure so callers
    degrade to today's behavior.
    """
    if cache_list is None:
        return 0, [], 0
    try:
        from mlx_lm.models.cache import CacheList, KVCache
    except ImportError:
        return 0, [], 0

    full = 0
    arrays = 0
    windows: Counter[int] = Counter()

    def _walk(c: Any) -> None:
        nonlocal full, arrays
        if type(c) is KVCache or type(c).__name__ in _FULL_KV_CACHE_CLASS_NAMES:
            full += 1
            return
        if isinstance(c, CacheList):
            for inner in c.caches:
                _walk(inner)
            return
        name = type(c).__name__
        max_size = getattr(c, "max_size", None)
        if (_pos_int(max_size) and hasattr(c, "keep")) or (
            name in _ROTATING_CACHE_CLASS_NAMES
        ):
            if _pos_int(max_size):
                windows[max_size] += 1
            return
        if name in _ARRAYS_CACHE_CLASS_NAMES:
            arrays += 1

    try:
        for c in cache_list:
            _walk(c)
    except Exception:
        return 0, [], 0

    specs = [(count, window) for window, count in sorted(windows.items())]
    return full, specs, arrays


def estimate_qwen4_exp_kv_bytes_per_token(
    config: Any,
    cache_list: Any,
    dtype_size: float,
) -> float | None:
    """Price Qwen4 QSA K/V plus its raw index keys and MRoPE positions."""
    if not str(_cfg_get(config, "model_type", "")).startswith("qwen4_exp"):
        return None
    if cache_list is None:
        return None

    try:
        qsa_layers = sum(
            1
            for cache in cache_list
            if type(cache).__name__
            in {"QSAKVCache", "QSAQuantizedKVCache", "BatchQSAKVCache"}
        )
    except Exception:
        return None
    if qsa_layers <= 0:
        return None

    num_kv_heads = _cfg_get(config, "num_key_value_heads")
    head_dim = _cfg_get(config, "head_dim")
    indexer_head_dim = _cfg_get(config, "indexer_head_dim")
    if not all(_pos_int(value) for value in (num_kv_heads, head_dim, indexer_head_dim)):
        return None
    if not isinstance(dtype_size, (int, float)) or dtype_size <= 0:
        return None

    # QSA keeps ordinary K/V, one raw index-key vector, and up to three int64
    # MRoPE coordinates for every cached token. Text-only positions use one
    # coordinate, but charging all three keeps image requests conservative.
    per_layer = (
        2 * num_kv_heads * head_dim * float(dtype_size)
        + indexer_head_dim * float(dtype_size)
        + 3 * 8
    )
    return float(qsa_layers * per_layer)


def estimate_mla_kv_bytes_per_token(
    config: Any,
    cache_list: Any,
    dtype_size: float,
) -> float | None:
    """Estimate exact resident KV bytes/token for MLA-style caches.

    GLM/DeepSeek MLA models do not store expanded ``num_kv_heads * head_dim``
    K/V tensors. Their main cache stores a latent key and RoPE value
    (``kv_lora_rank + qk_rope_head_dim``) with a single KV head. GLM-5.2's DSA
    indexer adds a second cache on full-indexer layers. GLM-5.2 stores one
    ``index_head_dim`` key per token, while GLM-5.3 pools those keys by the
    cache's compression ratio. Falling back to the standard uniform KV formula
    over-counts these models by more than an order of magnitude.
    """
    kv_lora_rank = _cfg_get(config, "kv_lora_rank")
    rope_dim = _cfg_get(config, "qk_rope_head_dim")
    if not (_pos_int(kv_lora_rank) and _nonnegative_int(rope_dim)):
        return None

    if cache_list is None:
        return None

    main_cache_layers = 0
    indexer_cache_token_ratio = 0.0
    try:
        for layer_cache in cache_list:
            caches = getattr(layer_cache, "caches", None)
            if caches is None:
                continue
            n_caches = len(caches)
            if n_caches >= 1:
                main_cache_layers += 1
            if n_caches >= 2:
                indexer_cache = caches[1]
                ratio = getattr(indexer_cache, "ratio", 1)
                if not _pos_int(ratio):
                    ratio = 1
                indexer_cache_token_ratio += 1.0 / ratio
    except Exception:
        return None

    if main_cache_layers <= 0:
        return None

    index_head_dim = _cfg_get(config, "index_head_dim", 0) or 0
    if not _pos_int(index_head_dim):
        index_head_dim = 0

    elems_per_token = (
        main_cache_layers * (kv_lora_rank + rope_dim)
        + indexer_cache_token_ratio * index_head_dim
    )
    return float(elems_per_token) * float(dtype_size)


def _ane_prefill_transient_bytes(model: Any) -> int:
    """ANE prefill I/O surface bytes for ``model``, 0 when not attached.

    Defensive: the ANE patch is optional at runtime, so any import or lookup
    failure leaves the KV+SDPA estimate unchanged (issue #2841).
    """
    try:
        from omlx.patches.qwen35_ane_prefill import ane_prefill_transient_bytes

        return int(ane_prefill_transient_bytes(model))
    except Exception:  # noqa: BLE001 - patch optional; never break estimation
        return 0


def set_model_info_from_model(monitor: "MemoryMonitor", model: Any) -> None:
    """Populate ``monitor`` with KV/SDPA dims read from an mlx-lm ``model``.

    The engine-agnostic baseline used by engines that bypass the
    ``Scheduler`` (currently ``DFlashEngine``'s primary speculative path) so
    they can run the same prefill-peak estimate the scheduler-driven engines
    get. Best-effort: on any extraction failure the monitor is left dim-less
    and ``estimate_prefill_peak_bytes`` returns 0, making the guard a no-op
    rather than raising spuriously.

    Note this populates the *uncompressed* (base-dtype) KV size — it does not
    apply the TurboQuant fractional-byte adjustment that
    ``Scheduler._set_model_info_for_monitor`` layers on, because that depends
    on scheduler-side TurboQuant configuration. For a memory *guard* the
    uncompressed estimate is the conservative (never-under-count) choice.
    """
    try:
        # Try to get model config
        config = None
        if hasattr(model, "config"):
            config = model.config
        elif hasattr(model, "args"):
            config = model.args

        if config is None:
            logger.debug("Could not extract model config for memory estimation")
            return

        # VLM / multimodal configs (e.g. Qwen3.6-VL, Gemma-4) nest the
        # language-model dimensions under a sub-config. Prefer
        # ``text_config`` / ``language_config`` / ``llm_config`` when ANY of
        # them exposes the LM layer count, even if the top-level config also
        # has one — on some VLM packs the top-level field refers to the
        # *vision encoder*, not the LM, and accepting it silently miscalibrates
        # the SDPA-peak estimate. Probe ``num_hidden_layers`` and the legacy
        # ``n_layer`` alias. Falls back to the top-level config only when no
        # sub-config has either field.
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

        # Determine dtype size
        dtype_size = 2  # Default float16
        if hasattr(model, "dtype"):
            if model.dtype == mx.float32:
                dtype_size = 4
            elif model.dtype == mx.bfloat16:
                dtype_size = 2

        # Extract num_attention_heads (query heads) for SDPA peak estimation
        num_attention_heads = (
            _cfg_get(config, "num_attention_heads")
            or _cfg_get(config, "n_head")
            or num_kv_heads
        )

        # Classify layer cache types for hybrid models. Mirrors
        # Scheduler._set_model_info_for_monitor via the shared helper so the
        # DFlash guard and the scheduler guard price the same layer groups.
        cache_list = None
        num_kv_cache_layers = num_layers
        rotating_layer_specs: list[tuple[int, int]] = []
        if hasattr(model, "make_cache"):
            try:
                cache_list = model.make_cache()
                full_layers, rotating_layer_specs, arrays_layers = (
                    collect_kv_layer_specs(cache_list)
                )
                num_kv_cache_layers = full_layers
                # Fall back to charging every layer only when classification
                # found nothing at all. A genuine 0 full-attention count next
                # to rotating/arrays layers must stay 0, or the rotating term
                # would be double-counted on top of a linear all-layers term.
                if full_layers == 0 and not rotating_layer_specs and not arrays_layers:
                    num_kv_cache_layers = num_layers
            except Exception:
                pass

        kv_bytes_per_token = estimate_qwen4_exp_kv_bytes_per_token(
            config, cache_list, dtype_size
        ) or estimate_mla_kv_bytes_per_token(config, cache_list, dtype_size)

        # Truthiness alone isn't enough — MagicMock proxies leaking through the
        # descent (test scaffolds that don't fully spec ``model.config``) are
        # truthy but fail any later numeric comparison (``> 128`` etc.) deep
        # inside MemoryMonitor. Insist on real positive integers before calling.
        if _pos_int(num_layers) and _pos_int(num_kv_heads) and _pos_int(head_dim):
            monitor.set_model_info(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype_size=dtype_size,
                num_attention_heads=num_attention_heads,
                num_kv_cache_layers=num_kv_cache_layers,
                # This path uses the uncompressed base dtype for KV, so
                # dtype_size already equals the compute/activation dtype.
                compute_dtype_size=dtype_size,
                kv_bytes_per_token=kv_bytes_per_token,
                rotating_layer_specs=rotating_layer_specs,
                ane_prefill_transient_bytes=_ane_prefill_transient_bytes(model),
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


def raise_if_prefill_exceeds(
    monitor: "MemoryMonitor | None",
    *,
    prefill_memory_guard: bool,
    hard_limit_bytes: int,
    current_usage_bytes: int,
    prefill_step_size: int,
    num_prompt_tokens: int,
    cached_tokens: int = 0,
    request_id: str | None = None,
    static_ceiling_bytes: int = 0,
    dynamic_ceiling_bytes: int = 0,
    metal_cap_bytes: int = 0,
    memory_guard_tier: str = "",
) -> None:
    """Raise ``PrefillMemoryExceededError`` if a prompt's prefill peak would
    push memory past ``hard_limit_bytes``.

    The shared front-door guard, taking token counts + watermarks directly so
    an engine without a ``Scheduler`` (``DFlashEngine``) enforces with the
    same math ``Scheduler.preflight_or_raise`` uses. No-op when the guard is
    disabled, no limit is set, the monitor is missing, or the request fits.
    The caller supplies ``current_usage_bytes`` so HTTP/event-loop preflight
    paths can use cached executor telemetry plus physical footprint without
    calling MLX directly. Maps to HTTP 400 via the server's
    ``prefill_memory_exceeded_handler``.

    ``cached_tokens`` means prompt KV *already resident in current memory*
    (e.g. the scheduler's paged prefix cache) — not merely "tokens that hit
    a cache". A cache whose hits re-allocate KV (DFlash prefix snapshots)
    must pass 0.

    The component ceilings are optional: callers that receive the
    enforcer's breakdown (the DFlash prefill guard) pass them so the
    rejection names the binding constraint the same way the scheduler's
    does. Callers that do not fall back to generic advice.
    """
    if not prefill_memory_guard:
        return
    if hard_limit_bytes <= 0:
        return
    if monitor is None:
        return

    new_tokens = max(int(num_prompt_tokens) - max(int(cached_tokens), 0), 0)
    if new_tokens == 0:
        return

    peak = monitor.estimate_prefill_peak_bytes(
        new_tokens, prefill_step_size, cached_tokens=cached_tokens
    )
    if peak == 0:
        return

    current = max(0, int(current_usage_bytes))
    if current + peak <= hard_limit_bytes:
        return

    usage_gb = current / (1024**3)
    ceiling_gb = hard_limit_bytes / (1024**3)
    binding, advice = describe_ceiling_binding(
        static=static_ceiling_bytes,
        dynamic=dynamic_ceiling_bytes,
        metal_cap=metal_cap_bytes,
        tier=memory_guard_tier,
        current=current,
        fmt=format_bytes,
        tail="reduce context length",
    )
    message = (
        f"Prefill would require ~{format_bytes(current + peak)} peak "
        f"(current {format_bytes(current)} + KV+SDPA {format_bytes(peak)}) "
        f"but {binding} ceiling is {format_bytes(hard_limit_bytes)} "
        f"(usage {usage_gb:.1f} GB, ceiling {ceiling_gb:.1f} GB). "
        f"{advice}."
    )

    if not request_id:
        import uuid as _uuid

        request_id = f"preflight-{_uuid.uuid4().hex[:8]}"
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
        limit_bytes=int(hard_limit_bytes),
    )
