# SPDX-License-Identifier: Apache-2.0
"""
Custom exception hierarchy for oMLX.

This module provides a structured exception hierarchy for better error handling
and debugging throughout the codebase.

Usage:
    from omlx.exceptions import CacheCorruptionError, SchedulerError

    try:
        scheduler.step()
    except CacheCorruptionError as e:
        # Handle cache corruption specifically
        scheduler.recover()
    except SchedulerError as e:
        # Handle other scheduler errors
        logger.error(f"Scheduler error: {e}")
"""

from collections.abc import Callable
from typing import Any, Optional, Union

# Direction of the memory_guard_tier knob. Spelled out everywhere a message
# names the tier: the admin popup lists Safe first, so "lower the tier" reads
# as "move toward Safe" — the opposite of what a user hitting a ceiling needs.
MEMORY_GUARD_TIER_LADDER = "safe → balanced → aggressive"

# Used when the component breakdown is unavailable (standalone engine pool,
# enforcer not wired up yet) and we cannot name the binding ceiling.
DEFAULT_CEILING_ADVICE = (
    f"Free system memory or raise memory_guard_tier ({MEMORY_GUARD_TIER_LADDER})"
)


def _join_remedies(remedies: list[str]) -> str:
    """Join remedies as "a, b, or c"."""
    if len(remedies) == 1:
        return remedies[0]
    return f"{', '.join(remedies[:-1])}, or {remedies[-1]}"


def describe_ceiling_binding(
    *,
    static: int,
    dynamic: int,
    metal_cap: int,
    tier: str,
    current: int,
    fmt: Callable[[int], str],
    tail: Optional[Union[str, list[str]]] = None,
) -> tuple[str, str]:
    """Name the binding component ceiling and the remedy that actually moves it.

    The process ceiling is ``min(static, dynamic, metal_cap)`` and each
    component answers to a different knob, so a message that does not say
    which one is binding sends users to the wrong knob. The field case: a
    16 GB Mac on the ``safe`` tier is dynamic-bound near 7 GB while static
    and metal_cap both sit at 12 GB, so raising ``iogpu.wired_limit_mb``
    (the only advice the admin banner gives) changes nothing.

    Binding is the argmin over the components the caller passes, so it does
    not depend on the caller's own ceiling value being sampled at the same
    instant, nor on whether that ceiling has a hot-cache reservation
    subtracted. Pass 0 for a component a given cap never used — the
    prefill abort limit is min(static, metal_cap) by design, so it passes
    ``dynamic=0``.

    Returns ``(binding, advice)``, where ``binding`` is "effective" when no
    component was supplied. ``advice`` is capitalized and carries no
    trailing punctuation; ``tail`` appends one or more call-site remedies
    ("reduce context length" for prefill, "use a smaller model" for a load).
    """
    components = {
        name: value
        for name, value in (
            ("static", static),
            ("dynamic", dynamic),
            ("metal_cap", metal_cap),
        )
        if value > 0
    }
    effective = min(components.values()) if components else 0
    binding = [name for name, value in components.items() if value == effective]
    binding_str = "/".join(binding) if binding else "effective"

    # Order remedies by likelihood of helping for the binding cause.
    # Dynamic-bound on a reclaim tier (safe/balanced/aggressive) means
    # reclaimable memory is low right now even though the static cap has
    # room — closing apps raises ``free`` / ``inactive`` and a more
    # aggressive ``memory_guard_tier`` raises the active-reclaim ratio.
    # Dynamic-bound under ``custom`` means the user pinned the ceiling
    # there; the only knob that helps is raising ``custom_ceiling_bytes``
    # itself. Metal-cap bound means the kernel sysctl is the ceiling.
    # Static-bound (or no breakdown available) leaves the tier as the lever.
    remedies: list[str] = []
    is_custom = (tier or "").strip().lower() == "custom"
    if "dynamic" in binding and is_custom:
        remedies.append(
            f"raise custom_ceiling_bytes in admin Memory settings "
            f"(currently pinned at {fmt(dynamic)})"
        )
    elif "dynamic" in binding and static and static > dynamic:
        headroom = max(0, dynamic - current)
        remedies.append(
            f"close other apps to free RAM (static cap is {fmt(static)} but "
            f"only {fmt(headroom)} is reclaimable right now)"
        )
        remedies.append(f"raise memory_guard_tier ({MEMORY_GUARD_TIER_LADDER})")
    elif "metal_cap" in binding:
        remedies.append(
            f"raise kernel iogpu.wired_limit_mb in Terminal "
            f"(currently caps Metal at {fmt(metal_cap)})"
        )
    else:
        remedies.append(f"raise memory_guard_tier ({MEMORY_GUARD_TIER_LADDER})")
    if tail:
        remedies.extend([tail] if isinstance(tail, str) else tail)

    advice = _join_remedies(remedies)
    return binding_str, advice[:1].upper() + advice[1:]


class OMLXError(Exception):
    """
    Base exception for all oMLX errors.

    All custom exceptions in oMLX should inherit from this class to allow
    for easy catching of all oMLX-related errors.
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


# =============================================================================
# Cache-related Exceptions
# =============================================================================


class CacheError(OMLXError):
    """Base exception for cache-related errors."""

    pass


class CacheCorruptionError(CacheError):
    """
    KV cache data is corrupted or invalid.

    This error indicates that the cache contains invalid data that prevents
    normal operation. Recovery typically involves clearing the cache and
    rescheduling affected requests.

    Attributes:
        request_id: The request ID affected by corruption, if known.
        block_id: The block ID that is corrupted, if applicable.
    """

    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        block_id: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details)
        self.request_id = request_id
        self.block_id = block_id


class CacheMissError(CacheError):
    """
    Requested cache entry was not found.

    This is typically not a fatal error - it indicates a cache miss that
    should be handled by falling back to computation.
    """

    def __init__(
        self,
        message: str,
        key: Optional[Any] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details)
        self.key = key


class CacheEvictionError(CacheError):
    """
    Failed to evict cache entries.

    This can occur when eviction policies fail or when there are no
    evictable entries despite memory pressure.
    """

    pass


class CacheStorageError(CacheError):
    """
    Failed to store or retrieve cache data from storage (paged SSD/disk).

    This error indicates I/O issues with the paged SSD cache storage layer.
    """

    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details)
        self.path = path
        self.operation = operation


# =============================================================================
# Scheduler-related Exceptions
# =============================================================================


class SchedulerError(OMLXError):
    """Base exception for scheduler-related errors."""

    pass


class RequestError(SchedulerError):
    """
    Error related to a specific request.

    Attributes:
        request_id: The ID of the request that caused the error.
    """

    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details)
        self.request_id = request_id


class RequestNotFoundError(RequestError):
    """Request was not found in the scheduler."""

    pass


class RequestAbortedError(RequestError):
    """Request was aborted before completion."""

    pass


class BatchingError(SchedulerError):
    """
    Error during batch processing.

    This can occur when the BatchGenerator encounters issues during
    token generation or batch management.
    """

    pass


class SchedulerQueueFullError(SchedulerError):
    """
    Waiting queue depth cap exceeded.

    Raised when admission control rejects a request because the scheduler's
    waiting queue is already at the configured depth cap. Server layer maps
    this to HTTP 503 with a short Retry-After.

    Attributes:
        current_depth: Current number of waiting requests.
        max_depth: Configured queue depth cap.
    """

    def __init__(
        self,
        current_depth: int,
        max_depth: int,
        details: Optional[dict] = None,
    ):
        super().__init__(
            f"Scheduler waiting queue full: {current_depth} >= {max_depth}",
            details,
        )
        self.current_depth = current_depth
        self.max_depth = max_depth


# =============================================================================
# Model-related Exceptions
# =============================================================================


class ModelError(OMLXError):
    """Base exception for model-related errors."""

    pass


class ModelLoadError(ModelError):
    """
    Failed to load the model.

    Attributes:
        model_name: The name/path of the model that failed to load.
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details)
        self.model_name = model_name


class ModelInferenceError(ModelError):
    """Error during model inference/generation."""

    pass


class TokenizerError(ModelError):
    """Error related to tokenization."""

    pass


# =============================================================================
# API-related Exceptions
# =============================================================================


class APIError(OMLXError):
    """Base exception for API-related errors."""

    pass


class InvalidRequestError(APIError):
    """
    The API request is invalid.

    Attributes:
        field: The field that is invalid, if applicable.
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details)
        self.field = field


class RateLimitError(APIError):
    """Request was rate limited."""

    pass


class AuthenticationError(APIError):
    """Authentication failed."""

    pass


# =============================================================================
# Configuration-related Exceptions
# =============================================================================


class ConfigurationError(OMLXError):
    """
    Configuration is invalid or inconsistent.

    Attributes:
        config_key: The configuration key that is invalid.
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details)
        self.config_key = config_key


# =============================================================================
# Memory-related Exceptions
# =============================================================================


class OMLXMemoryError(OMLXError):
    """Base exception for memory-related errors."""

    pass


class OutOfMemoryError(OMLXMemoryError):
    """
    Out of memory error.

    This typically indicates that the system has run out of GPU/CPU memory
    and cannot allocate more resources.
    """

    def __init__(
        self,
        message: str,
        requested_bytes: Optional[int] = None,
        available_bytes: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details)
        self.requested_bytes = requested_bytes
        self.available_bytes = available_bytes


class PrefillMemoryExceededError(OMLXMemoryError):
    """
    Prefill would exceed memory limits.

    Raised by the pre-flight memory guard when the estimated peak memory
    for a prefill operation (model weights + KV cache + SDPA attention matrix)
    exceeds the hard memory limit. This prevents kernel panics from large
    context windows on memory-constrained systems.

    Attributes:
        request_id: The request that was rejected.
        estimated_bytes: Estimated peak memory in bytes.
        limit_bytes: Hard memory limit in bytes.
    """

    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        estimated_bytes: Optional[int] = None,
        limit_bytes: Optional[int] = None,
        details: Optional[dict] = None,
    ):
        super().__init__(message, details)
        self.request_id = request_id
        self.estimated_bytes = estimated_bytes
        self.limit_bytes = limit_bytes


class PrefillMemoryAbortedError(PrefillMemoryExceededError):
    """
    An admitted request was aborted mid-prefill by the memory enforcer.

    The pre-flight guard let this request in, but process usage crossed the
    hard watermark while it was prefilling, so the enforcer aborted it to
    keep the server alive. Subclassing PrefillMemoryExceededError is what
    routes this to the same HTTP 400 mapping and the JSON keepalive body;
    the separate type only exists so the wording can say "aborted
    mid-prefill" instead of "rejected this prompt".
    """


# =============================================================================
# Engine Pool Exceptions
# =============================================================================


class EnginePoolError(OMLXError):
    """Base exception for engine pool errors."""

    pass


class ModelNotFoundError(EnginePoolError):
    """Raised when a requested model is not found."""

    def __init__(self, model_id: str, available_models: list[str]):
        self.model_id = model_id
        self.available_models = available_models
        message = (
            f"Model '{model_id}' not found. "
            f"Available models: {', '.join(available_models) if available_models else '(none)'}"
        )
        super().__init__(message)


class ModelTooLargeError(EnginePoolError):
    """Raised when a model cannot fit under the current memory ceiling."""

    def __init__(
        self,
        model_id: str,
        model_size: int,
        ceiling: int,
        *,
        binding: Optional[str] = None,
        advice: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_size = model_size
        self.ceiling = ceiling
        self.binding = binding
        # Import here to avoid circular dependency
        from .model_discovery import format_size

        label = f"{binding} memory ceiling" if binding else "memory ceiling"
        message = (
            f"Model '{model_id}' ({format_size(model_size)}) "
            f"does not fit under the {label} ({format_size(ceiling)}). "
            f"{advice or DEFAULT_CEILING_ADVICE}."
        )
        super().__init__(message)


class InsufficientMemoryError(EnginePoolError):
    """Raised when there's not enough memory even after eviction."""

    def __init__(self, required: int, current: int, message: str):
        self.required = required
        self.current = current
        super().__init__(message)


class ModelLoadingError(EnginePoolError):
    """Raised when a model load is unavailable, blocked, or invalid."""

    def __init__(self, model_id: str, message: str | None = None):
        self.model_id = model_id
        super().__init__(message or f"Model '{model_id}' is already being loaded")


class ModelUnavailableError(EnginePoolError):
    """Raised when a previous load failure makes a model temporarily unavailable."""

    def __init__(self, model_id: str, message: str | None = None):
        self.model_id = model_id
        super().__init__(message or f"Model '{model_id}' is unavailable")


class ModelBusyError(EnginePoolError):
    """Raised when an operation would unload a model that is currently in use."""

    def __init__(self, model_id: str, operation: str):
        self.model_id = model_id
        self.operation = operation
        super().__init__(
            f"Model '{model_id}' is busy; cannot {operation} until active "
            "requests finish or are aborted"
        )


# =============================================================================
# MCP Errors
# =============================================================================


class MCPError(OMLXError):
    """Base exception for MCP-related errors."""

    pass


class MCPConnectionError(MCPError):
    """Failed to connect to MCP server."""

    pass


class MCPToolExecutionError(MCPError):
    """Failed to execute MCP tool."""

    pass


# =============================================================================
# Helper Functions
# =============================================================================

# Patterns that indicate cache corruption (used by scheduler recovery logic)
CACHE_CORRUPTION_PATTERNS = [
    "'NoneType' object is not subscriptable",
    # Heterogeneous-batch crash: when one row has logits_processors=[proc]
    # and another has None, mlx-lm's GenerationBatch._step does
    # ``for p in self.logits_processors[e]`` over a None slot and raises
    # this exact message.  Without matching it here, the error bubbles
    # past recovery and into engine_loop's bare except, presenting as a
    # request hang.  See vllm-mlx-patched commit 8d4052b for the same
    # root cause in a sibling project.  Issue #934.
    "'NoneType' object is not iterable",
    "BatchKVCache",
    "KVCache",
    "cache.keys",
    "cache.values",
    "'NoneType' object has no attribute",
    "not broadcastable",
    "cannot be broadcast",
    "shape mismatch",
]


def is_cache_corruption_error(error: Exception) -> bool:
    """
    Check if an error indicates cache corruption.

    This function examines the error message to determine if it matches
    known cache corruption patterns.

    Args:
        error: The exception to check.

    Returns:
        True if the error appears to be cache corruption.
    """
    error_str = str(error)
    return any(pattern in error_str for pattern in CACHE_CORRUPTION_PATTERNS)
