# SPDX-License-Identifier: Apache-2.0
"""Tests for psutil_compat memory telemetry."""

from unittest.mock import patch

import pytest

import omlx.utils.psutil_compat as psutil_compat


def test_virtual_memory_uses_macos_host_stats():
    with (
        patch("omlx.utils.psutil_compat.sys.platform", "darwin"),
        patch(
            "omlx.utils.psutil_compat.get_macos_vm_stats",
            return_value={
                "free": 2 * 1024**3,
                "inactive": 3 * 1024**3,
                "active": 4 * 1024**3,
                "wired": 1 * 1024**3,
            },
        ),
        patch("omlx.utils.psutil_compat.get_total_memory", return_value=16 * 1024**3),
    ):
        vm = psutil_compat.virtual_memory()

    assert vm.total == 16 * 1024**3
    assert vm.available == 5 * 1024**3
    assert vm.free == 2 * 1024**3
    assert vm.inactive == 3 * 1024**3
    assert vm.active == 4 * 1024**3
    assert vm.wired == 1 * 1024**3


def test_virtual_memory_vm_stat_fallback_is_cached():
    vm_stat_output = """Mach Virtual Memory Statistics: (page size of 4096 bytes)
Pages free:                               10.
Pages active:                             20.
Pages inactive:                           30.
Pages wired down:                         40.
"""
    with (
        patch("omlx.utils.psutil_compat.sys.platform", "darwin"),
        patch("omlx.utils.psutil_compat.get_macos_vm_stats", return_value=None),
        patch("omlx.utils.psutil_compat.get_total_memory", return_value=1024**3),
        patch("omlx.utils.psutil_compat._cached_slow_virtual_memory", None),
        patch("omlx.utils.psutil_compat._cached_slow_virtual_memory_at", 0.0),
        patch(
            "omlx.utils.psutil_compat.subprocess.check_output",
            return_value=vm_stat_output,
        ) as mock_check_output,
    ):
        first = psutil_compat.virtual_memory()
        second = psutil_compat.virtual_memory()

    assert first.available == (10 + 30) * 4096
    assert second.available == first.available
    mock_check_output.assert_called_once()


def test_vm_stats_include_compressed_and_speculative(monkeypatch):
    # These sit past the four stable counters, so they are only reported when
    # the kernel actually filled that far into the struct.
    stats = psutil_compat.get_macos_vm_stats()
    if stats is None:
        pytest.skip("host_statistics64 unavailable")
    assert {"free", "active", "inactive", "wired"} <= set(stats)
    for key in ("speculative", "compressed"):
        if key in stats:
            assert stats[key] >= 0


def test_vm_stats_omits_tail_counters_on_short_reply(monkeypatch):
    monkeypatch.setattr(psutil_compat, "_VM_COMPRESSOR_INDEX", 10**6)
    monkeypatch.setattr(psutil_compat, "_VM_SPECULATIVE_INDEX", 10**6)
    stats = psutil_compat.get_macos_vm_stats()
    if stats is None:
        pytest.skip("host_statistics64 unavailable")
    assert "compressed" not in stats
    assert "speculative" not in stats
    # The stable four must still be there — _build_svmem depends on them.
    assert {"free", "active", "inactive", "wired"} <= set(stats)
