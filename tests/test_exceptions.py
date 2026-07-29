# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.exceptions module."""

import pytest

from omlx.exceptions import (
    # Base exception
    OMLXError,
    # Cache exceptions
    CacheError,
    CacheCorruptionError,
    CacheMissError,
    CacheEvictionError,
    CacheStorageError,
    # Scheduler exceptions
    SchedulerError,
    RequestError,
    RequestNotFoundError,
    RequestAbortedError,
    BatchingError,
    # Model exceptions
    ModelError,
    ModelLoadError,
    ModelInferenceError,
    TokenizerError,
    # API exceptions
    APIError,
    InvalidRequestError,
    RateLimitError,
    AuthenticationError,
    # Configuration exceptions
    ConfigurationError,
    # Memory exceptions
    OMLXMemoryError,
    OutOfMemoryError,
    # Engine pool exceptions
    EnginePoolError,
    ModelNotFoundError,
    ModelTooLargeError,
    InsufficientMemoryError,
    describe_ceiling_binding,
    ModelLoadingError,
    # MCP exceptions
    MCPError,
    MCPConnectionError,
    MCPToolExecutionError,
    # Helper function
    is_cache_corruption_error,
    CACHE_CORRUPTION_PATTERNS,
)


class TestOMLXError:
    """Test cases for base OMLXError exception."""

    def test_basic_instantiation(self):
        """Test basic exception creation."""
        error = OMLXError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}

    def test_with_details(self):
        """Test exception with details dictionary."""
        details = {"key": "value", "count": 42}
        error = OMLXError("Error with details", details=details)
        assert error.details == details
        assert "details:" in str(error)
        assert "key" in str(error)

    def test_inheritance(self):
        """Test that OMLXError inherits from Exception."""
        error = OMLXError("Test")
        assert isinstance(error, Exception)


class TestCacheExceptions:
    """Test cases for cache-related exceptions."""

    def test_cache_error_inheritance(self):
        """Test CacheError inherits from OMLXError."""
        error = CacheError("Cache error")
        assert isinstance(error, OMLXError)

    def test_cache_corruption_error(self):
        """Test CacheCorruptionError with all attributes."""
        error = CacheCorruptionError(
            "Cache data corrupted",
            request_id="req-123",
            block_id=42,
            details={"reason": "invalid checksum"},
        )
        assert error.request_id == "req-123"
        assert error.block_id == 42
        assert isinstance(error, CacheError)

    def test_cache_miss_error(self):
        """Test CacheMissError with key attribute."""
        error = CacheMissError("Key not found", key="prefix_abc")
        assert error.key == "prefix_abc"
        assert isinstance(error, CacheError)

    def test_cache_eviction_error(self):
        """Test CacheEvictionError inheritance."""
        error = CacheEvictionError("No evictable entries")
        assert isinstance(error, CacheError)

    def test_cache_storage_error(self):
        """Test CacheStorageError with path and operation."""
        error = CacheStorageError(
            "Failed to write cache",
            path="/tmp/cache/block_42.safetensors",
            operation="write",
        )
        assert error.path == "/tmp/cache/block_42.safetensors"
        assert error.operation == "write"
        assert isinstance(error, CacheError)


class TestSchedulerExceptions:
    """Test cases for scheduler-related exceptions."""

    def test_scheduler_error_inheritance(self):
        """Test SchedulerError inherits from OMLXError."""
        error = SchedulerError("Scheduler error")
        assert isinstance(error, OMLXError)

    def test_request_error(self):
        """Test RequestError with request_id."""
        error = RequestError("Request failed", request_id="req-456")
        assert error.request_id == "req-456"
        assert isinstance(error, SchedulerError)

    def test_request_not_found_error(self):
        """Test RequestNotFoundError inheritance chain."""
        error = RequestNotFoundError("Request not found", request_id="req-789")
        assert isinstance(error, RequestError)
        assert isinstance(error, SchedulerError)
        assert isinstance(error, OMLXError)

    def test_request_aborted_error(self):
        """Test RequestAbortedError."""
        error = RequestAbortedError("Request aborted by user", request_id="req-abc")
        assert isinstance(error, RequestError)

    def test_batching_error(self):
        """Test BatchingError."""
        error = BatchingError("Batch processing failed")
        assert isinstance(error, SchedulerError)


class TestModelExceptions:
    """Test cases for model-related exceptions."""

    def test_model_error_inheritance(self):
        """Test ModelError inherits from OMLXError."""
        error = ModelError("Model error")
        assert isinstance(error, OMLXError)

    def test_model_load_error(self):
        """Test ModelLoadError with model_name."""
        error = ModelLoadError(
            "Failed to load model",
            model_name="llama-3.1-8b",
        )
        assert error.model_name == "llama-3.1-8b"
        assert isinstance(error, ModelError)

    def test_model_inference_error(self):
        """Test ModelInferenceError."""
        error = ModelInferenceError("Inference failed")
        assert isinstance(error, ModelError)

    def test_tokenizer_error(self):
        """Test TokenizerError."""
        error = TokenizerError("Tokenization failed")
        assert isinstance(error, ModelError)


class TestAPIExceptions:
    """Test cases for API-related exceptions."""

    def test_api_error_inheritance(self):
        """Test APIError inherits from OMLXError."""
        error = APIError("API error")
        assert isinstance(error, OMLXError)

    def test_invalid_request_error(self):
        """Test InvalidRequestError with field attribute."""
        error = InvalidRequestError("Invalid field value", field="temperature")
        assert error.field == "temperature"
        assert isinstance(error, APIError)

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Rate limit exceeded")
        assert isinstance(error, APIError)

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid API key")
        assert isinstance(error, APIError)


class TestConfigurationError:
    """Test cases for ConfigurationError."""

    def test_configuration_error(self):
        """Test ConfigurationError with config_key."""
        error = ConfigurationError(
            "Invalid configuration",
            config_key="max_tokens",
        )
        assert error.config_key == "max_tokens"
        assert isinstance(error, OMLXError)


class TestMemoryExceptions:
    """Test cases for memory-related exceptions."""

    def test_omlx_memory_error_inheritance(self):
        """Test OMLXMemoryError inherits from OMLXError."""
        error = OMLXMemoryError("Memory error")
        assert isinstance(error, OMLXError)

    def test_out_of_memory_error(self):
        """Test OutOfMemoryError with memory attributes."""
        error = OutOfMemoryError(
            "Out of GPU memory",
            requested_bytes=8 * 1024**3,
            available_bytes=2 * 1024**3,
        )
        assert error.requested_bytes == 8 * 1024**3
        assert error.available_bytes == 2 * 1024**3
        assert isinstance(error, OMLXMemoryError)


class TestEnginePoolExceptions:
    """Test cases for engine pool exceptions."""

    def test_engine_pool_error_inheritance(self):
        """Test EnginePoolError inherits from OMLXError."""
        error = EnginePoolError("Engine pool error")
        assert isinstance(error, OMLXError)

    def test_model_not_found_error(self):
        """Test ModelNotFoundError with available_models."""
        error = ModelNotFoundError(
            model_id="unknown-model",
            available_models=["llama-3.1-8b", "qwen2.5-32b"],
        )
        assert error.model_id == "unknown-model"
        assert error.available_models == ["llama-3.1-8b", "qwen2.5-32b"]
        assert "unknown-model" in str(error)
        assert "llama-3.1-8b" in str(error)
        assert isinstance(error, EnginePoolError)

    def test_model_not_found_error_empty_models(self):
        """Test ModelNotFoundError with no available models."""
        error = ModelNotFoundError(
            model_id="test-model",
            available_models=[],
        )
        assert "(none)" in str(error)

    def test_model_too_large_error(self):
        """Test ModelTooLargeError with size information."""
        error = ModelTooLargeError(
            model_id="huge-model",
            model_size=100 * 1024**3,
            ceiling=32 * 1024**3,
        )
        assert error.model_id == "huge-model"
        assert error.model_size == 100 * 1024**3
        assert error.ceiling == 32 * 1024**3
        assert isinstance(error, EnginePoolError)

    def test_insufficient_memory_error(self):
        """Test InsufficientMemoryError."""
        error = InsufficientMemoryError(
            required=16 * 1024**3,
            current=8 * 1024**3,
            message="Not enough memory after eviction",
        )
        assert error.required == 16 * 1024**3
        assert error.current == 8 * 1024**3
        assert isinstance(error, EnginePoolError)

    def test_model_loading_error(self):
        """Test ModelLoadingError."""
        error = ModelLoadingError(model_id="loading-model")
        assert error.model_id == "loading-model"
        assert "already being loaded" in str(error)
        assert isinstance(error, EnginePoolError)


class TestMCPExceptions:
    """Test cases for MCP-related exceptions."""

    def test_mcp_error_inheritance(self):
        """Test MCPError inherits from OMLXError."""
        error = MCPError("MCP error")
        assert isinstance(error, OMLXError)

    def test_mcp_connection_error(self):
        """Test MCPConnectionError."""
        error = MCPConnectionError("Failed to connect to MCP server")
        assert isinstance(error, MCPError)

    def test_mcp_tool_execution_error(self):
        """Test MCPToolExecutionError."""
        error = MCPToolExecutionError("Tool execution failed")
        assert isinstance(error, MCPError)


class TestIsCacheCorruptionError:
    """Test cases for is_cache_corruption_error helper function."""

    def test_none_type_pattern(self):
        """Test detection of NoneType subscript error."""
        error = TypeError("'NoneType' object is not subscriptable")
        assert is_cache_corruption_error(error) is True

    def test_batch_kv_cache_pattern(self):
        """Test detection of BatchKVCache error."""
        error = RuntimeError("BatchKVCache.keys returned None")
        assert is_cache_corruption_error(error) is True

    def test_kv_cache_pattern(self):
        """Test detection of KVCache error."""
        error = ValueError("KVCache data is invalid")
        assert is_cache_corruption_error(error) is True

    def test_cache_keys_pattern(self):
        """Test detection of cache.keys error."""
        error = AttributeError("cache.keys is not available")
        assert is_cache_corruption_error(error) is True

    def test_cache_values_pattern(self):
        """Test detection of cache.values error."""
        error = AttributeError("cache.values returned empty")
        assert is_cache_corruption_error(error) is True

    def test_attribute_error_none_attribute(self):
        """Test detection of NoneType attribute access error."""
        error = AttributeError("'NoneType' object has no attribute 'shape'")
        assert is_cache_corruption_error(error) is True

    def test_none_type_not_iterable_pattern(self):
        """Test detection of NoneType iterable error.

        Triggered by mlx-lm GenerationBatch._step at
        ``for p in self.logits_processors[e]`` when the merged
        logits_processors list contains a None slot from heterogeneous
        request mixing.  Without this pattern the error escapes
        recovery and presents as a hang. See issue #934.
        """
        error = TypeError("'NoneType' object is not iterable")
        assert is_cache_corruption_error(error) is True

    def test_value_error_not_broadcastable(self):
        """Test detection of shape broadcast error."""
        error = ValueError(
            "Shapes (1,8,256,128) and (1,8,512,128) are not broadcastable"
        )
        assert is_cache_corruption_error(error) is True

    def test_value_error_cannot_be_broadcast(self):
        """Test detection of MLX broadcast_shapes error (Issue #79)."""
        error = ValueError(
            "[broadcast_shapes] Shapes (8,1,625,2674) and (8,8,625,3697) "
            "cannot be broadcast."
        )
        assert is_cache_corruption_error(error) is True

    def test_value_error_shape_mismatch(self):
        """Test detection of shape mismatch error."""
        error = ValueError("shape mismatch in cache concatenation")
        assert is_cache_corruption_error(error) is True

    def test_non_corruption_error(self):
        """Test that non-corruption errors return False."""
        error = ValueError("Invalid temperature value")
        assert is_cache_corruption_error(error) is False

    def test_attribute_error_non_corruption(self):
        """Non-corruption AttributeError should not match."""
        error = AttributeError("'str' object has no attribute 'encode'")
        assert is_cache_corruption_error(error) is False

    def test_unrelated_error(self):
        """Test completely unrelated error."""
        error = FileNotFoundError("Model file not found")
        assert is_cache_corruption_error(error) is False


class TestCacheCorruptionPatterns:
    """Test cases for CACHE_CORRUPTION_PATTERNS constant."""

    def test_patterns_is_list(self):
        """Test that patterns is a list."""
        assert isinstance(CACHE_CORRUPTION_PATTERNS, list)

    def test_patterns_contains_expected(self):
        """Test that patterns contains expected strings."""
        expected_patterns = [
            "'NoneType' object is not subscriptable",
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
        for pattern in expected_patterns:
            assert pattern in CACHE_CORRUPTION_PATTERNS


class TestExceptionHierarchy:
    """Test the full exception hierarchy."""

    def test_all_exceptions_inherit_from_omlx_error(self):
        """Test that all custom exceptions inherit from OMLXError."""
        exceptions_to_test = [
            CacheError("test"),
            CacheCorruptionError("test"),
            CacheMissError("test"),
            CacheEvictionError("test"),
            CacheStorageError("test"),
            SchedulerError("test"),
            RequestError("test"),
            RequestNotFoundError("test"),
            RequestAbortedError("test"),
            BatchingError("test"),
            ModelError("test"),
            ModelLoadError("test"),
            ModelInferenceError("test"),
            TokenizerError("test"),
            APIError("test"),
            InvalidRequestError("test"),
            RateLimitError("test"),
            AuthenticationError("test"),
            ConfigurationError("test"),
            OMLXMemoryError("test"),
            OutOfMemoryError("test"),
            EnginePoolError("test"),
            MCPError("test"),
            MCPConnectionError("test"),
            MCPToolExecutionError("test"),
        ]

        for exc in exceptions_to_test:
            assert isinstance(exc, OMLXError), f"{type(exc).__name__} should inherit from OMLXError"
            assert isinstance(exc, Exception), f"{type(exc).__name__} should inherit from Exception"

    def test_engine_pool_special_exceptions(self):
        """Test engine pool exceptions that have special constructors."""
        # These have different constructor signatures
        exc1 = ModelNotFoundError("model", [])
        assert isinstance(exc1, OMLXError)

        exc2 = ModelTooLargeError("model", 100, 50)
        assert isinstance(exc2, OMLXError)

        exc3 = InsufficientMemoryError(100, 50, "message")
        assert isinstance(exc3, OMLXError)

        exc4 = ModelLoadingError("model")
        assert isinstance(exc4, OMLXError)


class TestCeilingBindingAdvice:
    """The memory-ceiling advice ladder.

    A 16 GB Mac on the `safe` tier is dynamic-bound around 7 GB while the
    static and Metal caps both sit at 12 GB. The load refusal used to say
    "free system memory or lower memory_guard_tier" with no breakdown, and
    the tier popup lists Safe first, so "lower" reads as "move to Safe" —
    which shrinks the ceiling further. Every message must name the binding
    ceiling and point up the ladder.
    """

    @staticmethod
    def _fmt(value: int) -> str:
        return f"{value / 1024**3:.2f}GB"

    def test_dynamic_binding_names_apps_and_ladder_direction(self):
        binding, advice = describe_ceiling_binding(
            static=12 * 1024**3,
            dynamic=7 * 1024**3,
            metal_cap=12 * 1024**3,
            tier="safe",
            current=1 * 1024**3,
            fmt=self._fmt,
            tail="use a smaller model",
        )
        assert binding == "dynamic"
        assert "close other apps" in advice.lower()
        assert "raise memory_guard_tier (safe → balanced → aggressive)" in advice
        assert "lower memory_guard_tier" not in advice
        # Reclaimable headroom, not the whole dynamic ceiling.
        assert "6.00GB is reclaimable" in advice
        assert advice.endswith("or use a smaller model")

    def test_metal_cap_binding_names_sysctl(self):
        binding, advice = describe_ceiling_binding(
            static=24 * 1024**3,
            dynamic=16 * 1024**3,
            metal_cap=8 * 1024**3,
            tier="balanced",
            current=0,
            fmt=self._fmt,
        )
        assert binding == "metal_cap"
        assert "iogpu.wired_limit_mb" in advice
        assert "close other apps" not in advice.lower()

    def test_dynamic_binding_under_custom_names_admin_setting(self):
        binding, advice = describe_ceiling_binding(
            static=24 * 1024**3,
            dynamic=4 * 1024**3,
            metal_cap=16 * 1024**3,
            tier="custom",
            current=0,
            fmt=self._fmt,
        )
        assert binding == "dynamic"
        assert "custom_ceiling_bytes" in advice
        assert "close other apps" not in advice.lower()

    def test_static_binding_falls_back_to_the_tier_lever(self):
        binding, advice = describe_ceiling_binding(
            static=8 * 1024**3,
            dynamic=16 * 1024**3,
            metal_cap=24 * 1024**3,
            tier="safe",
            current=0,
            fmt=self._fmt,
        )
        assert binding == "static"
        assert advice.startswith("Raise memory_guard_tier")
        assert "iogpu.wired_limit_mb" not in advice

    def test_a_component_the_cap_never_used_is_excluded(self):
        """The prefill abort limit is min(static, metal_cap) by design, so
        its rejection passes dynamic=0 — naming the dynamic ceiling there
        would point at a knob that cannot move that cap."""
        binding, advice = describe_ceiling_binding(
            static=24 * 1024**3,
            dynamic=0,
            metal_cap=8 * 1024**3,
            tier="safe",
            current=0,
            fmt=self._fmt,
        )
        assert binding == "metal_cap"
        assert "close other apps" not in advice.lower()

    def test_no_components_falls_back_to_generic_advice(self):
        """Guard disabled or breakdown never propagated."""
        binding, advice = describe_ceiling_binding(
            static=0,
            dynamic=0,
            metal_cap=0,
            tier="balanced",
            current=0,
            fmt=self._fmt,
            tail="reduce context length",
        )
        assert binding == "effective"
        assert advice == (
            "Raise memory_guard_tier (safe → balanced → aggressive), "
            "or reduce context length"
        )

    def test_multiple_tails_join_as_a_single_list(self):
        _binding, advice = describe_ceiling_binding(
            static=8 * 1024**3,
            dynamic=16 * 1024**3,
            metal_cap=24 * 1024**3,
            tier="safe",
            current=0,
            fmt=self._fmt,
            tail=["reduce context length", "lower hot_cache_max_size"],
        )
        assert advice.endswith("reduce context length, or lower hot_cache_max_size")

    def test_model_too_large_default_advice_points_up_the_ladder(self):
        message = str(ModelTooLargeError("m", 100 * 1024**3, 32 * 1024**3))
        assert "raise memory_guard_tier (safe → balanced → aggressive)" in message
        assert "lower memory_guard_tier" not in message

    def test_model_too_large_carries_binding_and_advice(self):
        error = ModelTooLargeError(
            "m",
            100 * 1024**3,
            32 * 1024**3,
            binding="dynamic",
            advice="Close other apps to free RAM",
        )
        assert error.binding == "dynamic"
        assert "dynamic memory ceiling" in str(error)
        assert str(error).endswith("Close other apps to free RAM.")
