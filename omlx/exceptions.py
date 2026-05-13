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

from typing import Any, Optional


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


class MemoryPressureRejection(SchedulerError):
    """
    Request rejected due to current process memory pressure.

    Raised by the HTTP admission layer when the projected peak memory
    (current phys_footprint + estimated prefill peak) would exceed the
    hard limit. Server layer maps this to HTTP 503 with a Retry-After
    proportional to the pressure level.

    Attributes:
        pressure_level: "soft" or "hard".
        current_bytes: Current phys_footprint in bytes.
        estimated_peak_bytes: Estimated peak after admitting this request.
        limit_bytes: Configured hard limit.
    """

    def __init__(
        self,
        pressure_level: str,
        current_bytes: int,
        estimated_peak_bytes: int,
        limit_bytes: int,
        details: Optional[dict] = None,
    ):
        super().__init__(
            f"Memory pressure ({pressure_level}): projected peak "
            f"{estimated_peak_bytes / 1024**3:.1f}GB "
            f"(current {current_bytes / 1024**3:.1f}GB) "
            f"exceeds limit {limit_bytes / 1024**3:.1f}GB",
            details,
        )
        self.pressure_level = pressure_level
        self.current_bytes = current_bytes
        self.estimated_peak_bytes = estimated_peak_bytes
        self.limit_bytes = limit_bytes


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
    """Raised when a model exceeds the memory limit."""

    def __init__(self, model_id: str, model_size: int, max_memory: int):
        self.model_id = model_id
        self.model_size = model_size
        self.max_memory = max_memory
        # Import here to avoid circular dependency
        from .model_discovery import format_size

        message = (
            f"Model '{model_id}' ({format_size(model_size)}) "
            f"exceeds max-model-memory ({format_size(max_memory)})"
        )
        super().__init__(message)


class InsufficientMemoryError(EnginePoolError):
    """Raised when there's not enough memory even after eviction."""

    def __init__(self, required: int, current: int, message: str):
        self.required = required
        self.current = current
        super().__init__(message)


class ModelLoadingError(EnginePoolError):
    """Raised when a model is already being loaded."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' is already being loaded")


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
