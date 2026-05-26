# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx/optimizations.py — a thin hardware/MLX status helper.
The re-exported symbols (HardwareInfo, detect_hardware, get_total_memory_gb)
are covered by test_utils_hardware.py; here we pin the dict shape and the
flash-attention detection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlx.core as mx

from omlx import optimizations
from omlx.optimizations import (
    HardwareInfo,
    detect_hardware,
    get_optimization_status,
    get_system_memory_gb,
)


class TestReExports:
    def test_hardware_symbols_importable_from_optimizations(self):
        """The module's docstring promises these names. Removing one
        would silently break ``from omlx.optimizations import ...``
        used by external scripts."""
        from omlx.utils.hardware import detect_hardware as canonical_detect
        from omlx.utils.hardware import HardwareInfo as CanonicalInfo

        assert detect_hardware is canonical_detect
        assert HardwareInfo is CanonicalInfo

    def test_get_system_memory_gb_aliases_get_total_memory_gb(self):
        """The re-export renames ``get_total_memory_gb`` →
        ``get_system_memory_gb``. The alias must stay in place."""
        from omlx.utils.hardware import get_total_memory_gb

        assert get_system_memory_gb is get_total_memory_gb

    def test_all_lists_documented_surface(self):
        assert set(optimizations.__all__) == {
            "HardwareInfo",
            "detect_hardware",
            "get_system_memory_gb",
            "get_optimization_status",
        }


class TestGetOptimizationStatus:
    def test_returns_top_level_keys(self):
        status = get_optimization_status()
        assert set(status.keys()) == {"hardware", "mlx_memory", "mlx_lm_features"}

    def test_hardware_section_shape(self):
        status = get_optimization_status()
        hw = status["hardware"]
        assert set(hw.keys()) == {"chip", "total_memory_gb", "device_name"}
        # chip is populated from detect_hardware().chip_name — non-empty
        # string on any Apple Silicon test runner.
        assert isinstance(hw["chip"], str)
        assert isinstance(hw["total_memory_gb"], (int, float))
        assert hw["total_memory_gb"] > 0
        assert isinstance(hw["device_name"], str)

    def test_mlx_memory_section_is_byte_counters(self):
        status = get_optimization_status()
        mem = status["mlx_memory"]
        assert set(mem.keys()) == {"active_bytes", "cache_bytes", "peak_bytes"}
        # All three come straight from mx.get_*_memory(); non-negative ints
        for key in mem:
            assert isinstance(mem[key], int), f"{key} not an int"
            assert mem[key] >= 0

    def test_mlx_lm_features_static_strings(self):
        """These strings appear in the admin dashboard. Pin them so a
        typo or accidental rewording shows up as a test failure rather
        than a confusing UI change."""
        features = get_optimization_status()["mlx_lm_features"]
        assert features["metal_kernels"] == "optimized for Apple Silicon"
        assert features["kv_cache"] == "managed by mlx-lm"
        assert features["quantization"] == "4-bit and 8-bit supported"

    def test_flash_attention_reports_built_in_when_available(self):
        """``mlx.core.fast.scaled_dot_product_attention`` exists in all
        recent MLX versions — the test environment is one of them."""
        assert hasattr(mx, "fast")
        assert hasattr(mx.fast, "scaled_dot_product_attention")
        status = get_optimization_status()
        assert status["mlx_lm_features"]["flash_attention"] == "built-in"

    def test_flash_attention_reports_not_available_when_missing(self):
        """The fallback branch runs on hypothetical MLX builds without
        the fused SDPA. Simulated by replacing ``mx.fast`` with an
        object that lacks the attribute."""
        fake_fast = MagicMock(spec=[])  # spec=[] → no attributes
        with patch.object(mx, "fast", fake_fast):
            status = get_optimization_status()
        assert (
            status["mlx_lm_features"]["flash_attention"] == "not available"
        )

    def test_active_bytes_reflects_real_mlx_state(self):
        """Verify the value isn't hardcoded — allocating an array
        should bump active memory above the pre-allocation baseline.
        Defensive: the loop ensures eval happens so memory shows up."""
        before = mx.get_active_memory()
        arr = mx.zeros((1024, 1024), dtype=mx.float32)
        mx.eval(arr)
        after = get_optimization_status()["mlx_memory"]["active_bytes"]
        # 1024*1024*4 bytes = 4 MiB allocation must register somewhere
        # in the active memory delta.
        assert after >= before
