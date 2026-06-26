# SPDX-License-Identifier: Apache-2.0
"""Tests for memory_monitor module (SSD-only mode)."""

from unittest.mock import MagicMock

import pytest

from omlx.memory_monitor import (
    _SDPA_FALLBACK_SCORE_DTYPE_SIZE,
    _SDPA_FULL_SUPPORTED_HEAD_DIMS,
    _SDPA_VECTOR_QUERY_TOKEN_THRESHOLD,
    _SDPA_VECTOR_SUPPORTED_HEAD_DIMS,
    MemoryInfo,
    MemoryMonitor,
)
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

    def test_eviction_enabled_property_default_true(self):
        """The default ``eviction_enabled=True`` makes the
        public-facing predicate True so the existing tiered-cache
        path keeps working without changes."""
        monitor = MemoryMonitor(max_kv_cache_memory=1024**3)
        assert monitor.eviction_enabled is True

    def test_eviction_enabled_property_false_in_ssd_only_mode(self):
        """Paged-SSD-only mode passes ``eviction_enabled=False``; the
        public predicate must surface that so Scheduler can branch on
        it (avoiding the RuntimeError from estimate_blocks_to_free)."""
        monitor = MemoryMonitor(max_kv_cache_memory=None, eviction_enabled=False)
        assert monitor.eviction_enabled is False

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

    def _expected_output_sdpa(self, n_q, query_tokens, head_dim):
        return n_q * query_tokens * head_dim * 4

    def _expected_fallback_sdpa(self, n_q, query_tokens, kv_len, head_dim):
        scores = n_q * query_tokens * kv_len * _SDPA_FALLBACK_SCORE_DTYPE_SIZE
        output = n_q * query_tokens * head_dim * 4
        return scores + output

    def test_returns_zero_when_model_info_missing(self):
        m = MemoryMonitor(max_kv_cache_memory=10 * 1024**3)
        assert m.estimate_prefill_peak_bytes(32768, 2048) == 0

    def test_returns_zero_when_no_new_tokens(self):
        # Fully-prefix-cached request: nothing to prefill, peak is 0.
        m = self._make_monitor()
        assert m.estimate_prefill_peak_bytes(0, 2048, cached_tokens=32768) == 0

    def test_fused_full_prefill_head_dim_128(self):
        # head_dim=128 is supported by the fused full prefill kernel.
        m = self._make_monitor(head_dim=128, n_attn=32, n_kv=4, n_layers=62)
        peak = m.estimate_prefill_peak_bytes(32768, 2048)
        # KV: 62 layers * 4 kv_heads * 128 dim * 2 bytes * 2 (k+v) * 32768 ≈ 4.0 GB
        # SDPA fused: n_attn * chunk * head_dim * 4 = 32*2048*128*4 ≈ 32 MB
        # Total ≈ 4 GB
        assert 3 * 1024**3 < peak < 5 * 1024**3

    def test_prefill_head_dim_256_uses_full_score_fallback(self):
        # head_dim=256 is vector-kernel-supported, but not full-prefill-supported.
        m = self._make_monitor(head_dim=256, n_attn=8, n_kv=4, n_layers=48)
        peak = m.estimate_prefill_peak_bytes(32768, 2048)
        expected_sdpa = self._expected_fallback_sdpa(8, 2048, 32768, 256)
        expected_kv = m.estimate_prompt_kv_bytes(32768)
        assert peak == expected_sdpa + expected_kv
        assert expected_sdpa > 8 * 2048 * 256 * 2

    def test_sdpa_fallback_scores_track_compute_dtype(self):
        # The unfused score matrix is materialized at the model's compute
        # dtype, not fp32 and not the (possibly fractional TurboQuant) KV width.
        # fp32 model -> 4 bytes/elem; bf16/fp16 -> 2.
        def _scores(monitor, n_q, chunk, kv, hd):
            out = n_q * chunk * hd * 4
            return monitor._estimate_sdpa_activation_bytes(chunk, kv) - out

        n_q, chunk, kv, hd = 8, 2048, 32768, 256
        m_bf16 = MemoryMonitor(max_kv_cache_memory=10 * 1024**3)
        m_bf16.set_model_info(
            num_layers=48, num_kv_heads=4, head_dim=hd,
            num_attention_heads=n_q, compute_dtype_size=2,
        )
        m_fp32 = MemoryMonitor(max_kv_cache_memory=10 * 1024**3)
        m_fp32.set_model_info(
            num_layers=48, num_kv_heads=4, head_dim=hd,
            num_attention_heads=n_q, compute_dtype_size=4,
        )
        assert _scores(m_bf16, n_q, chunk, kv, hd) == n_q * chunk * kv * 2
        assert _scores(m_fp32, n_q, chunk, kv, hd) == n_q * chunk * kv * 4

    def test_sdpa_score_dtype_ignores_fractional_kv_width(self):
        # TurboQuant sets a fractional KV dtype_size; the score matrix must
        # still be charged at the compute dtype, not ~0.5 bytes/elem.
        n_q, chunk, kv, hd = 8, 2048, 32768, 256
        m = MemoryMonitor(max_kv_cache_memory=10 * 1024**3)
        m.set_model_info(
            num_layers=48, num_kv_heads=4, head_dim=hd, dtype_size=0.5,
            num_attention_heads=n_q, compute_dtype_size=2,
        )
        out = n_q * chunk * hd * 4
        scores = m._estimate_sdpa_activation_bytes(chunk, kv) - out
        assert scores == n_q * chunk * kv * 2

    def test_sdpa_fallback_accounts_for_cached_kv_span(self):
        """Regression for M3: SDPA fallback spans the FULL prompt (cached + new),
        not just new_tokens. A heavily-cached long-context request previously
        slipped through with under-counted peak.
        """
        m = self._make_monitor(head_dim=256, n_attn=8, n_kv=4, n_layers=48)
        # Same total prompt (100k), different cache split:
        # - All-new: cached=0, new=100k
        # - Heavy cache: cached=99k, new=1k
        all_new = m.estimate_prefill_peak_bytes(100 * 1024, 2048)
        heavy_cache = m.estimate_prefill_peak_bytes(1024, 2048, cached_tokens=99 * 1024)
        expected_heavy_sdpa = self._expected_fallback_sdpa(8, 1024, 100 * 1024, 256)
        expected_heavy = expected_heavy_sdpa + m.estimate_prompt_kv_bytes(1024)
        assert heavy_cache == expected_heavy
        assert (
            heavy_cache > 900 * 1024**2
        ), f"heavy-cache peak under-counted: {heavy_cache / 1024**2:.0f} MB"
        # And the all-new case (larger eff_chunk = 2048 but same kv_len)
        # should be larger overall because both KV growth and scores
        # widen with new_tokens.
        assert all_new > heavy_cache

    def test_scales_linearly_with_token_count(self):
        m = self._make_monitor()
        p8k = m.estimate_prefill_peak_bytes(8 * 1024, 2048)
        p32k = m.estimate_prefill_peak_bytes(32 * 1024, 2048)
        # KV grows linearly with tokens; SDPA fused doesn't depend on
        # total_tokens. KV dominates here, so 32k/8k ≈ 4x.
        assert p32k > p8k
        ratio = p32k / p8k
        assert 3.5 < ratio < 4.5

    def test_sdpa_fallback_scales_with_context_length(self):
        # Unsupported full-prefill head dims: SDPA peak ∝ query_len * total_tokens.
        # When chunk is fixed (2048), peak grows linearly with total_tokens
        # plus KV grows linearly too. Doubling tokens should ~double peak.
        m = self._make_monitor(head_dim=256, n_attn=8, n_kv=4, n_layers=48)
        p16k = m.estimate_prefill_peak_bytes(16 * 1024, 2048)
        p32k = m.estimate_prefill_peak_bytes(32 * 1024, 2048)
        ratio = p32k / p16k
        assert 1.8 < ratio < 2.2

    def test_eff_chunk_capped_at_new_tokens(self):
        """Short prompts (smaller than chunk_size) must not be charged
        the full chunk_size width — the effective chunk is bounded by
        the number of remaining new tokens. Regression for the constant-
        factor over-count on small prompts.
        """
        m = self._make_monitor(head_dim=256, n_attn=8, n_kv=4, n_layers=48)
        # 100-token prompt; chunk_size=2048. eff_chunk should be 100,
        # not 2048 — so the query width is 100, not the default step size.
        peak = m.estimate_prefill_peak_bytes(100, 2048)
        # KV: 48*4*256*2*2*100 ≈ 19 MB. SDPA is small here. Total < 25 MB.
        assert peak < 25 * 1024**2, (
            f"short-prompt peak suggests chunk wasn't clamped: "
            f"{peak / 1024**2:.0f} MB"
        )

    def test_no_python_overhead_constant(self):
        # estimator must NOT include cache_pool_overhead or python_overhead
        # magic constants — those are absorbed by enforcer hard_threshold.
        # If a small prompt returns >2 GB on a small model, that's a sign
        # someone added back the magic constants.
        m = self._make_monitor(head_dim=128, n_attn=8, n_kv=2, n_layers=8)
        peak = m.estimate_prefill_peak_bytes(512, 2048)
        # KV: 8*2*128*2*2*512 ≈ 4 MB. SDPA fused: 8*512*128*4 ≈ 2 MB. Total ≈ 6 MB.
        assert peak < 100 * 1024**2, f"unexpected large peak: {peak / 1024**2:.1f} MB"

    def test_cached_tokens_extends_sdpa_span(self):
        # Unsupported full-prefill head dims span cached+new tokens.
        # A request with a big prefix-cache hit (small new suffix) must still
        # estimate the SDPA transient over the full span, not just new_tokens.
        m = self._make_monitor(head_dim=256, n_attn=16, n_kv=2, n_layers=40)
        # 2k new on top of 30k cached → SDPA span is 32k, query is 2k.
        with_cache = m.estimate_prefill_peak_bytes(2048, 2048, cached_tokens=30 * 1024)
        # Same new_tokens, no cache → SDPA span is only 2k.
        without_cache = m.estimate_prefill_peak_bytes(2048, 2048, cached_tokens=0)
        # The output buffer and KV growth are identical; only the score-matrix
        # K dimension changes.
        sdpa_with = self._expected_fallback_sdpa(16, 2048, 2048 + 30 * 1024, 256)
        sdpa_without = self._expected_fallback_sdpa(16, 2048, 2048, 256)
        assert with_cache - without_cache == sdpa_with - sdpa_without
        assert with_cache > without_cache * 2

    def test_cached_tokens_default_matches_no_cache(self):
        # Omitting cached_tokens must reproduce the pre-change behavior so the
        # no-cache path (cached=0) is a strict regression guard.
        m = self._make_monitor(head_dim=256, n_attn=8, n_kv=4, n_layers=48)
        assert m.estimate_prefill_peak_bytes(
            32768, 2048
        ) == m.estimate_prefill_peak_bytes(32768, 2048, cached_tokens=0)

    def test_query_len_capped_at_new_tokens(self):
        # When new_tokens < chunk_size the last (only) chunk's query length is
        # new_tokens, not the full step size.
        m = self._make_monitor(head_dim=256, n_attn=8, n_kv=4, n_layers=48)
        # 512 new on top of 10k cached: query=512, span=10k+512.
        peak = m.estimate_prefill_peak_bytes(512, 2048, cached_tokens=10 * 1024)
        expected_sdpa = self._expected_fallback_sdpa(8, 512, 512 + 10 * 1024, 256)
        expected_kv = m.estimate_prompt_kv_bytes(512)
        assert peak == expected_sdpa + expected_kv

    def test_sdpa_dispatch_constants_match_mlx_use_fallback(self):
        assert _SDPA_VECTOR_QUERY_TOKEN_THRESHOLD == 8
        assert frozenset({64, 80, 128}) == _SDPA_FULL_SUPPORTED_HEAD_DIMS
        assert frozenset({64, 96, 128, 256}) == _SDPA_VECTOR_SUPPORTED_HEAD_DIMS

    def test_vector_path_head_dim_256_is_output_only_for_short_query(self):
        m = self._make_monitor(head_dim=256, n_attn=8, n_kv=4, n_layers=48)
        assert m.estimate_chunk_transient_bytes(4, 10_000) == (
            self._expected_output_sdpa(8, 4, 256)
        )

    def test_vector_path_head_dim_80_falls_back(self):
        m = self._make_monitor(head_dim=80, n_attn=8, n_kv=4, n_layers=48)
        assert m.estimate_chunk_transient_bytes(4, 10_000) == (
            self._expected_fallback_sdpa(8, 4, 10_000, 80)
        )

    def test_full_prefill_head_dim_80_is_output_only(self):
        m = self._make_monitor(head_dim=80, n_attn=8, n_kv=4, n_layers=48)
        assert m.estimate_chunk_transient_bytes(512, 10_000) == (
            self._expected_output_sdpa(8, 512, 80)
        )

    def test_full_prefill_head_dim_96_falls_back(self):
        m = self._make_monitor(head_dim=96, n_attn=8, n_kv=4, n_layers=48)
        assert m.estimate_chunk_transient_bytes(512, 10_000) == (
            self._expected_fallback_sdpa(8, 512, 10_000, 96)
        )

    def test_vector_path_gqa_limit_falls_back(self):
        m = self._make_monitor(head_dim=256, n_attn=64, n_kv=1, n_layers=48)
        assert m.estimate_chunk_transient_bytes(1, 10_000) == (
            self._expected_fallback_sdpa(64, 1, 10_000, 256)
        )
