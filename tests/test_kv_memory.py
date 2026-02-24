# SPDX-License-Identifier: Apache-2.0
"""Tests for KV cache memory budget utilities.

Note: max_kv_size only applies to models WITHOUT make_cache().
Models with make_cache() (MLA, hybrid, ArraysCache) are skipped at
the scheduler level, so per-token calculation only covers Standard/GQA.
"""

from unittest.mock import MagicMock, patch

import pytest

from omlx.utils.kv_memory import (
    MIN_MAX_KV_SIZE,
    _get_dtype_bytes,
    calculate_max_kv_size,
    calculate_per_token_kv_bytes,
    parse_memory_string,
    resolve_kv_cache_memory_bytes,
)


# ---------------------------------------------------------------------------
# Helpers to build mock models
# ---------------------------------------------------------------------------


def _make_model(
    num_hidden_layers=32,
    num_key_value_heads=None,
    num_attention_heads=32,
    head_dim=None,
    hidden_size=4096,
    dtype=None,
):
    """Build a mock model with the given config fields (no make_cache)."""
    config = MagicMock(spec=[])
    config.num_hidden_layers = num_hidden_layers
    config.num_key_value_heads = num_key_value_heads
    config.num_attention_heads = num_attention_heads
    config.head_dim = head_dim
    config.hidden_size = hidden_size

    model = MagicMock(spec=[])
    model.config = config

    if dtype is not None:
        model.dtype = dtype

    return model


# ---------------------------------------------------------------------------
# _get_dtype_bytes
# ---------------------------------------------------------------------------


class TestGetDtypeBytes:
    def test_default_is_2(self):
        model = MagicMock(spec=[])
        assert _get_dtype_bytes(model) == 2

    def test_float32(self):
        import mlx.core as mx

        model = MagicMock(spec=[])
        model.dtype = mx.float32
        assert _get_dtype_bytes(model) == 4

    def test_float16(self):
        import mlx.core as mx

        model = MagicMock(spec=[])
        model.dtype = mx.float16
        assert _get_dtype_bytes(model) == 2

    def test_bfloat16(self):
        import mlx.core as mx

        model = MagicMock(spec=[])
        model.dtype = mx.bfloat16
        assert _get_dtype_bytes(model) == 2


# ---------------------------------------------------------------------------
# calculate_per_token_kv_bytes
# ---------------------------------------------------------------------------


class TestCalculatePerTokenKVBytes:
    def test_standard_model(self):
        """Standard model: 32 layers, 32 KV heads, head_dim=128, float16."""
        model = _make_model(
            num_hidden_layers=32,
            num_key_value_heads=32,
            head_dim=128,
        )
        result = calculate_per_token_kv_bytes(model)
        # 32 layers × 32 heads × 128 dim × 2 bytes × 2 (K+V) = 524,288
        assert result == 32 * 32 * 128 * 2 * 2

    def test_gqa_model(self):
        """GQA model: Llama-3 style with 8 KV heads, 32 Q heads."""
        model = _make_model(
            num_hidden_layers=32,
            num_key_value_heads=8,
            num_attention_heads=32,
            head_dim=128,
        )
        result = calculate_per_token_kv_bytes(model)
        # 32 layers × 8 heads × 128 dim × 2 bytes × 2 (K+V) = 131,072
        assert result == 32 * 8 * 128 * 2 * 2

    def test_gqa_less_kv_than_standard(self):
        """GQA model should use less KV bytes than standard MHA."""
        standard = _make_model(num_key_value_heads=32, head_dim=128)
        gqa = _make_model(num_key_value_heads=8, num_attention_heads=32, head_dim=128)
        assert calculate_per_token_kv_bytes(gqa) < calculate_per_token_kv_bytes(standard)

    def test_head_dim_calculated_from_hidden_size(self):
        """head_dim derived from hidden_size / num_attention_heads."""
        model = _make_model(
            num_hidden_layers=32,
            num_key_value_heads=8,
            num_attention_heads=32,
            head_dim=None,
            hidden_size=4096,
        )
        result = calculate_per_token_kv_bytes(model)
        # head_dim = 4096 / 32 = 128
        assert result == 32 * 8 * 128 * 2 * 2

    def test_no_config_returns_none(self):
        model = MagicMock(spec=[])
        assert calculate_per_token_kv_bytes(model) is None

    def test_incomplete_config_returns_none(self):
        """Missing num_key_value_heads and num_attention_heads."""
        config = MagicMock(spec=[])
        config.num_hidden_layers = 32
        model = MagicMock(spec=[])
        model.config = config
        assert calculate_per_token_kv_bytes(model) is None


# ---------------------------------------------------------------------------
# parse_memory_string
# ---------------------------------------------------------------------------


class TestParseMemoryString:
    def test_gb(self):
        assert parse_memory_string("8GB") == 8 * 1024**3

    def test_gb_lowercase(self):
        assert parse_memory_string("8gb") == 8 * 1024**3

    def test_mb(self):
        assert parse_memory_string("512MB") == 512 * 1024**2

    def test_tb(self):
        assert parse_memory_string("1TB") == 1 * 1024**4

    def test_with_spaces(self):
        assert parse_memory_string("  16 GB  ") == 16 * 1024**3

    def test_float(self):
        assert parse_memory_string("1.5GB") == int(1.5 * 1024**3)

    def test_invalid(self):
        assert parse_memory_string("auto") is None
        assert parse_memory_string("disabled") is None
        assert parse_memory_string("abc") is None


# ---------------------------------------------------------------------------
# resolve_kv_cache_memory_bytes
# ---------------------------------------------------------------------------


class TestResolveKVCacheMemoryBytes:
    def test_disabled(self):
        assert resolve_kv_cache_memory_bytes("disabled") is None

    def test_explicit_size(self):
        result = resolve_kv_cache_memory_bytes("8GB")
        assert result == 8 * 1024**3

    def test_auto_with_override(self):
        """Auto mode with available_override."""
        available = 40 * 1024**3  # 40 GB
        model_weight = 16 * 1024**3  # 16 GB
        result = resolve_kv_cache_memory_bytes(
            "auto", model_weight_bytes=model_weight, available_override=available
        )
        # (40 - 16) * 0.9 = 21.6 GB
        expected = int((available - model_weight) * 0.9)
        assert result == expected

    @patch("omlx.utils.kv_memory.get_system_available_memory_bytes")
    def test_auto_system(self, mock_sys_mem):
        """Auto mode using system memory detection."""
        mock_sys_mem.return_value = 56 * 1024**3  # 56 GB available (64 - 8)
        model_weight = 16 * 1024**3
        result = resolve_kv_cache_memory_bytes("auto", model_weight_bytes=model_weight)
        expected = int((56 * 1024**3 - model_weight) * 0.9)
        assert result == expected

    def test_auto_minimum_1gb(self):
        """Auto mode should return at least 1 GB."""
        result = resolve_kv_cache_memory_bytes(
            "auto",
            model_weight_bytes=100 * 1024**3,  # Huge model
            available_override=10 * 1024**3,
        )
        assert result == 1 * 1024**3

    def test_unknown_falls_back_to_auto(self):
        """Unrecognized value treated as auto."""
        result = resolve_kv_cache_memory_bytes(
            "whatever", model_weight_bytes=0, available_override=20 * 1024**3
        )
        assert result == int(20 * 1024**3 * 0.9)


# ---------------------------------------------------------------------------
# calculate_max_kv_size
# ---------------------------------------------------------------------------


class TestCalculateMaxKVSize:
    def test_basic(self):
        """Standard model, 8GB budget, 8 concurrent."""
        model = _make_model(
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
        )
        budget = 8 * 1024**3  # 8 GB
        result = calculate_max_kv_size(model, budget, max_concurrent=8)
        per_token = 32 * 8 * 128 * 2 * 2  # 131,072 bytes
        expected = (budget // 8) // per_token
        assert result == expected

    def test_minimum_512(self):
        """Should return at least 512 even with tiny budget."""
        model = _make_model(
            num_hidden_layers=32,
            num_key_value_heads=32,
            head_dim=128,
        )
        result = calculate_max_kv_size(model, budget_bytes=1024, max_concurrent=8)
        assert result == MIN_MAX_KV_SIZE

    def test_none_if_no_config(self):
        model = MagicMock(spec=[])
        assert calculate_max_kv_size(model, budget_bytes=8 * 1024**3, max_concurrent=8) is None

    def test_zero_concurrent(self):
        model = _make_model(num_hidden_layers=32, num_key_value_heads=8, head_dim=128)
        assert calculate_max_kv_size(model, budget_bytes=8 * 1024**3, max_concurrent=0) is None
