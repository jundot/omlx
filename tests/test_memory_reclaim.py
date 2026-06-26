# SPDX-License-Identifier: Apache-2.0
"""Tests for process memory reclamation helpers."""

from unittest.mock import MagicMock, patch

from omlx.utils import memory_reclaim


def _reset_resolution_state():
    memory_reclaim._resolved = False
    memory_reclaim._pressure_relief_fn = None
    memory_reclaim._default_zone_fn = None


def test_release_free_malloc_pages_noops_off_macos():
    """Non-macOS platforms should return 0 without touching ctypes."""
    _reset_resolution_state()
    with patch.object(memory_reclaim.sys, "platform", "linux"), \
         patch.object(memory_reclaim.ctypes, "CDLL") as mock_cdll:
        assert memory_reclaim.release_free_malloc_pages() == 0
    mock_cdll.assert_not_called()


def test_release_free_malloc_pages_noops_when_symbol_unavailable():
    """Missing malloc pressure-relief symbols should be a safe no-op."""
    _reset_resolution_state()
    mock_lib = MagicMock()
    del mock_lib.malloc_zone_pressure_relief
    with patch.object(memory_reclaim.sys, "platform", "darwin"), \
         patch.object(memory_reclaim.ctypes, "CDLL", return_value=mock_lib):
        assert memory_reclaim.release_free_malloc_pages() == 0


def test_release_free_malloc_pages_calls_default_zone_pressure_relief():
    """macOS helper should pressure the default malloc zone."""
    _reset_resolution_state()
    mock_default_zone = MagicMock(return_value=1234)
    mock_pressure_relief = MagicMock(return_value=4096)
    mock_lib = MagicMock(
        malloc_default_zone=mock_default_zone,
        malloc_zone_pressure_relief=mock_pressure_relief,
    )

    with patch.object(memory_reclaim.sys, "platform", "darwin"), \
         patch.object(memory_reclaim.ctypes, "CDLL", return_value=mock_lib):
        assert memory_reclaim.release_free_malloc_pages() == 4096

    mock_pressure_relief.assert_called_once_with(1234, 0)
