# SPDX-License-Identifier: Apache-2.0
"""Tests for memory_monitor module (SSD-only mode)."""

import pytest
from unittest.mock import patch, MagicMock

from omlx.memory_monitor import MemoryMonitor, MemoryInfo
from omlx.utils.hardware import format_bytes


class TestMemoryInfo:
    """Tests for MemoryInfo dataclass."""

    def test_create_memory_info(self):
        """Test creating MemoryInfo."""
        info = MemoryInfo(
            total_bytes=16 * 1024**3,
            used_bytes=8 * 1024**3,
            available_bytes=8 * 1024**3,
            utilization=0.5,
        )
        assert info.total_bytes == 16 * 1024**3
        assert info.used_bytes == 8 * 1024**3
        assert info.available_bytes == 8 * 1024**3
        assert info.utilization == 0.5

    def test_memory_info_zero_usage(self):
        """Test MemoryInfo with zero usage."""
        info = MemoryInfo(
            total_bytes=16 * 1024**3,
            used_bytes=0,
            available_bytes=16 * 1024**3,
            utilization=0.0,
        )
        assert info.used_bytes == 0
        assert info.utilization == 0.0


class TestMemoryMonitor:
    """Test MemoryMonitor class for SSD-only mode."""

    def test_init_with_required_params(self):
        """Test initialization with required parameters."""
        max_kv_cache = 2 * 1024**3  # 2GB
        monitor = MemoryMonitor(max_kv_cache_memory=max_kv_cache)
        assert monitor.max_kv_cache_memory == max_kv_cache

    def test_init_invalid_max_kv_cache_memory_zero(self):
        """Test initialization with zero max_kv_cache_memory."""
        with pytest.raises(ValueError, match="max_kv_cache_memory"):
            MemoryMonitor(max_kv_cache_memory=0)

    def test_init_invalid_max_kv_cache_memory_negative(self):
        """Test initialization with negative max_kv_cache_memory."""
        with pytest.raises(ValueError, match="max_kv_cache_memory"):
            MemoryMonitor(max_kv_cache_memory=-1)

    def test_get_memory_info(self):
        """Test get_memory_info returns valid data."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)
        info = monitor.get_memory_info()

        assert isinstance(info, MemoryInfo)
        assert info.total_bytes == monitor.max_memory
        # In SSD-only mode, used_bytes is always 0
        assert info.used_bytes == 0
        assert info.available_bytes == monitor.max_memory
        assert info.utilization == 0.0

    def test_get_memory_info_throttling(self):
        """Test that memory info checks are throttled."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3, check_interval=10.0)

        # First call
        info1 = monitor.get_memory_info()
        # Second call within interval should return cached value
        info2 = monitor.get_memory_info()

        # Should be the same object (cached)
        assert info1 is info2

    def test_is_under_pressure_always_false(self):
        """Test is_under_pressure always returns False in SSD-only mode."""
        monitor = MemoryMonitor(max_kv_cache_memory=10000)
        # In SSD-only mode, always returns False
        assert not monitor.is_under_pressure()

    def test_bytes_to_free_always_zero(self):
        """Test bytes_to_free always returns 0 in SSD-only mode."""
        monitor = MemoryMonitor(max_kv_cache_memory=10000)
        # In SSD-only mode, always returns 0
        assert monitor.bytes_to_free() == 0

    def test_set_model_info(self):
        """Test setting model information."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)

        monitor.set_model_info(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,
        )

        # Internal state should be set
        assert monitor._num_layers == 32
        assert monitor._num_kv_heads == 8
        assert monitor._head_dim == 128
        assert monitor._dtype_size == 2

    def test_estimate_block_memory(self):
        """Test block memory estimation."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)

        # Set model info
        monitor.set_model_info(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,
        )

        # Estimate for 64 tokens
        estimate = monitor.estimate_block_memory(64)
        # Expected: 64 * 8 * 128 * 2 * 2 (keys+values) * 32 layers
        expected = 64 * 8 * 128 * 2 * 2 * 32
        assert estimate == expected

    def test_estimate_block_memory_default_values(self):
        """Test block memory estimation with default values."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)

        # Without setting model info, should use defaults
        estimate = monitor.estimate_block_memory(64)
        # Default: 32 layers, 8 kv_heads, 128 head_dim, 2 dtype_size
        expected = 64 * 8 * 128 * 2 * 2 * 32
        assert estimate == expected

    def test_estimate_block_memory_with_overrides(self):
        """Test block memory estimation with parameter overrides."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)
        monitor.set_model_info(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,
        )

        # Override some parameters
        estimate = monitor.estimate_block_memory(
            block_size=32,
            num_layers=16,  # Override
            dtype_size=4,  # Override
        )
        expected = 32 * 8 * 128 * 4 * 2 * 16
        assert estimate == expected

    def test_estimate_blocks_to_free(self):
        """Test estimation of blocks to free."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)
        monitor.set_model_info(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,
        )

        block_size = 64
        block_mem = monitor.estimate_block_memory(block_size)

        # Need to free 10 blocks worth
        bytes_to_free = block_mem * 10
        num_blocks = monitor.estimate_blocks_to_free(bytes_to_free, block_size)
        assert num_blocks == 10

    def test_estimate_blocks_to_free_rounds_up(self):
        """Test that blocks to free rounds up."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)
        monitor.set_model_info(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,
        )

        block_size = 64
        block_mem = monitor.estimate_block_memory(block_size)

        # Need to free slightly more than 9 blocks
        bytes_to_free = block_mem * 9 + 1
        num_blocks = monitor.estimate_blocks_to_free(bytes_to_free, block_size)
        assert num_blocks == 10  # Should round up

    def test_get_stats(self):
        """Test get_stats returns dict with expected keys."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)
        stats = monitor.get_stats()

        assert "total_bytes" in stats
        assert "used_bytes" in stats
        assert "available_bytes" in stats
        assert "utilization" in stats
        assert "max_kv_cache_memory" in stats
        assert "total_formatted" in stats
        assert "used_formatted" in stats
        assert "available_formatted" in stats
        # In SSD-only mode, used_bytes should be 0
        assert stats["used_bytes"] == 0

    def test_format_bytes(self):
        """Test format_bytes utility function."""
        assert "1.00 KB" == format_bytes(1024)
        assert "1.00 MB" == format_bytes(1024 * 1024)
        assert "1.00 GB" == format_bytes(1024 * 1024 * 1024)
        assert "512 B" == format_bytes(512)

    def test_repr(self):
        """Test string representation."""
        monitor = MemoryMonitor(max_kv_cache_memory=2 * 1024**3)
        repr_str = repr(monitor)
        assert "MemoryMonitor" in repr_str
        assert "max_kv_cache" in repr_str
        assert "used" in repr_str

    def test_properties(self):
        """Test property accessors."""
        max_kv_cache = 2 * 1024**3
        monitor = MemoryMonitor(max_kv_cache_memory=max_kv_cache)

        assert monitor.max_kv_cache_memory == max_kv_cache
        assert monitor.max_memory > 0

    def test_set_paged_cache_manager(self):
        """Test setting paged cache manager."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)

        mock_manager = MagicMock()
        monitor.set_paged_cache_manager(mock_manager, block_size=128)

        assert monitor._paged_cache_manager is mock_manager
        assert monitor._block_size == 128

    def test_set_baseline_memory(self):
        """Test setting baseline memory."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)

        # This should not raise (uses MLX if available, otherwise sets to 0)
        monitor.set_baseline_memory()

    def test_set_request_stats(self):
        """Test setting request stats."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)

        monitor.set_request_stats(running=5, waiting=10)

        assert monitor._running_requests == 5
        assert monitor._waiting_requests == 10

    def test_check_interval_parameter(self):
        """Test check_interval parameter."""
        monitor = MemoryMonitor(
            max_kv_cache_memory=1024**3,
            check_interval=5.0,
        )

        assert monitor._check_interval == 5.0


class TestEstimatePrefillPeakBytes:
    """Tests for estimate_prefill_peak_bytes (KV + SDPA only)."""

    def _make_monitor(self, head_dim=128, n_attn=32, n_kv=4, n_layers=62):
        m = MemoryMonitor(max_kv_cache_memory=10 * 1024**3)
        m.set_model_info(
            num_layers=n_layers,
            num_kv_heads=n_kv,
            head_dim=head_dim,
            dtype_size=2,
            num_attention_heads=n_attn,
        )
        return m

    def test_returns_zero_when_model_info_missing(self):
        m = MemoryMonitor(max_kv_cache_memory=10 * 1024**3)
        assert m.estimate_prefill_peak_bytes(32768, 2048) == 0

    def test_fused_kernel_below_head_dim_128(self):
        # head_dim<=128 → fused tiled kernel, SDPA peak is just output buffer
        m = self._make_monitor(head_dim=128, n_attn=32, n_kv=4, n_layers=62)
        peak = m.estimate_prefill_peak_bytes(32768, 2048)
        # KV: 62 layers * 4 kv_heads * 128 dim * 2 bytes * 2 (k+v) * 32768 ≈ 4.0 GB
        # SDPA fused: n_attn * chunk * head_dim * 4 = 32*2048*128*4 ≈ 32 MB
        # Total ≈ 4 GB
        assert 3 * 1024**3 < peak < 5 * 1024**3

    def test_fallback_path_above_head_dim_128(self):
        # head_dim>128 → full attention matrix materialized in float32
        m = self._make_monitor(head_dim=256, n_attn=8, n_kv=4, n_layers=48)
        peak = m.estimate_prefill_peak_bytes(32768, 2048)
        # SDPA fallback: n_attn * chunk * total_tokens * 4 = 8*2048*32768*4 = 2 GB
        # + output buffer 8*2048*256*4 ≈ 16 MB
        # KV: 48 * 4 * 256 * 2 * 2 * 32768 ≈ 6 GB
        # Total ≈ 8 GB
        assert 7 * 1024**3 < peak < 9 * 1024**3

    def test_scales_linearly_with_token_count(self):
        m = self._make_monitor()
        p8k = m.estimate_prefill_peak_bytes(8 * 1024, 2048)
        p32k = m.estimate_prefill_peak_bytes(32 * 1024, 2048)
        # KV grows linearly with tokens; SDPA fused doesn't depend on
        # total_tokens. KV dominates here, so 32k/8k ≈ 4x.
        assert p32k > p8k
        ratio = p32k / p8k
        assert 3.5 < ratio < 4.5

    def test_sdpa_fallback_scales_quadratically(self):
        # head_dim>128 fallback: SDPA peak ∝ chunk * total_tokens.
        # When chunk is fixed (2048), peak grows linearly with total_tokens
        # plus KV grows linearly too. Doubling tokens should ~double peak.
        m = self._make_monitor(head_dim=256, n_attn=8, n_kv=4, n_layers=48)
        p16k = m.estimate_prefill_peak_bytes(16 * 1024, 2048)
        p32k = m.estimate_prefill_peak_bytes(32 * 1024, 2048)
        ratio = p32k / p16k
        assert 1.8 < ratio < 2.2

    def test_no_python_overhead_constant(self):
        # estimator must NOT include cache_pool_overhead or python_overhead
        # magic constants — those are absorbed by enforcer hard_threshold.
        # If a small prompt returns >2 GB on a small model, that's a sign
        # someone added back the magic constants.
        m = self._make_monitor(head_dim=128, n_attn=8, n_kv=2, n_layers=8)
        peak = m.estimate_prefill_peak_bytes(512, 2048)
        # KV: 8*2*128*2*2*512 ≈ 4 MB. SDPA fused: 8*2048*128*4 ≈ 8 MB. Total ≈ 12 MB.
        assert peak < 100 * 1024**2, f"unexpected large peak: {peak / 1024**2:.1f} MB"
