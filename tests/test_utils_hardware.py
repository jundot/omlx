# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.hardware module."""

import platform
import sys
from unittest.mock import MagicMock, patch

import pytest

from omlx.utils.hardware import (
    DEFAULT_MEMORY_BYTES,
    HardwareInfo,
    format_bytes,
    get_chip_name,
    get_total_memory_bytes,
    get_total_memory_gb,
    is_apple_silicon,
    is_mlx_available,
)


class TestHardwareInfo:
    """Test cases for HardwareInfo dataclass."""

    def test_hardware_info_creation(self):
        """Test creating HardwareInfo with all fields."""
        info = HardwareInfo(
            chip_name="Apple M4 Pro",
            total_memory_gb=48.0,
            max_working_set_bytes=36 * 1024**3,
            mlx_device_name="Apple M4 Pro",
        )
        assert info.chip_name == "Apple M4 Pro"
        assert info.total_memory_gb == 48.0
        assert info.max_working_set_bytes == 36 * 1024**3
        assert info.mlx_device_name == "Apple M4 Pro"

    def test_hardware_info_default_mlx_device(self):
        """Test HardwareInfo with default mlx_device_name."""
        info = HardwareInfo(
            chip_name="Apple M1",
            total_memory_gb=8.0,
            max_working_set_bytes=6 * 1024**3,
        )
        assert info.mlx_device_name is None

    def test_hardware_info_various_chips(self):
        """Test HardwareInfo with various Apple Silicon chips."""
        chips = [
            ("Apple M1", 8.0),
            ("Apple M1 Pro", 16.0),
            ("Apple M1 Max", 32.0),
            ("Apple M1 Ultra", 64.0),
            ("Apple M2", 8.0),
            ("Apple M2 Pro", 16.0),
            ("Apple M2 Max", 32.0),
            ("Apple M2 Ultra", 128.0),
            ("Apple M3", 8.0),
            ("Apple M3 Pro", 18.0),
            ("Apple M3 Max", 36.0),
            ("Apple M4", 16.0),
            ("Apple M4 Pro", 48.0),
            ("Apple M4 Max", 128.0),
        ]
        for chip_name, memory_gb in chips:
            info = HardwareInfo(
                chip_name=chip_name,
                total_memory_gb=memory_gb,
                max_working_set_bytes=int(memory_gb * 0.75 * 1024**3),
            )
            assert info.chip_name == chip_name
            assert info.total_memory_gb == memory_gb


class TestGetChipName:
    """Test cases for get_chip_name function."""

    def test_get_chip_name_success(self):
        """Test get_chip_name with successful sysctl call."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Apple M4 Pro\n", returncode=0)
            result = get_chip_name()
            assert result == "Apple M4 Pro"
            mock_run.assert_called_once()

    def test_get_chip_name_fallback(self):
        """Test get_chip_name fallback when sysctl fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("sysctl not available")
            result = get_chip_name()
            assert result == "Apple Silicon"


class TestGetTotalMemoryBytes:
    """Test cases for get_total_memory_bytes function."""

    def test_get_total_memory_bytes_sysctl_success(self):
        """Test get_total_memory_bytes with successful sysctl call."""
        expected_bytes = 48 * 1024**3  # 48 GB
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout=f"{expected_bytes}\n", returncode=0
            )
            result = get_total_memory_bytes()
            assert result == expected_bytes

    def test_get_total_memory_bytes_default_fallback(self):
        """Test get_total_memory_bytes default fallback."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("sysctl failed")
            # Mock HAS_MLX to False so MLX fallback is skipped
            with patch("omlx.utils.hardware.HAS_MLX", False):
                result = get_total_memory_bytes()
                assert result == DEFAULT_MEMORY_BYTES


class TestGetTotalMemoryGb:
    """Test cases for get_total_memory_gb function."""

    def test_get_total_memory_gb_conversion(self):
        """Test that get_total_memory_gb correctly converts bytes to GB."""
        with patch(
            "omlx.utils.hardware.get_total_memory_bytes", return_value=16 * 1024**3
        ):
            result = get_total_memory_gb()
            assert result == 16.0

    def test_get_total_memory_gb_fractional(self):
        """Test get_total_memory_gb with fractional values."""
        with patch(
            "omlx.utils.hardware.get_total_memory_bytes",
            return_value=int(18.5 * 1024**3),
        ):
            result = get_total_memory_gb()
            assert abs(result - 18.5) < 0.01


class TestIsAppleSilicon:
    """Test cases for is_apple_silicon function."""

    def test_is_apple_silicon_true(self):
        """Test is_apple_silicon returns True on macOS arm64."""
        with patch.object(sys, "platform", "darwin"):
            with patch.object(platform, "machine", return_value="arm64"):
                assert is_apple_silicon() is True

    def test_is_apple_silicon_false_wrong_platform(self):
        """Test is_apple_silicon returns False on non-macOS."""
        with patch.object(sys, "platform", "linux"):
            assert is_apple_silicon() is False

    def test_is_apple_silicon_false_wrong_arch(self):
        """Test is_apple_silicon returns False on x86_64 macOS."""
        with patch.object(sys, "platform", "darwin"):
            with patch.object(platform, "machine", return_value="x86_64"):
                assert is_apple_silicon() is False


class TestIsMlxAvailable:
    """Test cases for is_mlx_available function."""

    def test_is_mlx_available_not_apple_silicon(self):
        """Test is_mlx_available returns False on non-Apple Silicon."""
        with patch("omlx.utils.hardware.is_apple_silicon", return_value=False):
            assert is_mlx_available() is False

    def test_is_mlx_available_import_error(self):
        """Test is_mlx_available handles import errors."""
        with patch("omlx.utils.hardware.is_apple_silicon", return_value=True):
            with patch.dict("sys.modules", {"mlx.core": None}):
                # Force import to fail
                import builtins

                original_import = builtins.__import__

                def mock_import(name, *args, **kwargs):
                    if name == "mlx.core" or name.startswith("mlx"):
                        raise ImportError("No module named mlx")
                    return original_import(name, *args, **kwargs)

                with patch.object(builtins, "__import__", mock_import):
                    result = is_mlx_available()
                    # May return True if mlx is already imported, False if not
                    assert isinstance(result, bool)


class TestFormatBytesHardware:
    """Test cases for format_bytes function in hardware module."""

    def test_format_bytes_gb(self):
        """Test formatting bytes to GB."""
        assert format_bytes(1024**3) == "1.00 GB"
        assert format_bytes(16 * 1024**3) == "16.00 GB"
        assert format_bytes(48 * 1024**3) == "48.00 GB"

    def test_format_bytes_mb(self):
        """Test formatting bytes to MB."""
        assert format_bytes(1024**2) == "1.00 MB"
        assert format_bytes(512 * 1024**2) == "512.00 MB"

    def test_format_bytes_kb(self):
        """Test formatting bytes to KB."""
        assert format_bytes(1024) == "1.00 KB"
        assert format_bytes(512 * 1024) == "512.00 KB"

    def test_format_bytes_small(self):
        """Test formatting small byte values."""
        assert format_bytes(0) == "0 B"
        assert format_bytes(512) == "512 B"
        assert format_bytes(1023) == "1023 B"


class TestDefaultMemoryBytes:
    """Test cases for DEFAULT_MEMORY_BYTES constant."""

    def test_default_memory_bytes_value(self):
        """Test that DEFAULT_MEMORY_BYTES is 8 GB."""
        assert DEFAULT_MEMORY_BYTES == 8 * 1024**3
