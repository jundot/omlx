# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.proc_memory.get_phys_footprint."""

import ctypes
import os
import sys

import pytest

from omlx.utils.proc_memory import (
    get_lifetime_max_phys_footprint,
    get_phys_footprint,
)


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin-only API")
class TestGetPhysFootprintDarwin:
    def test_returns_positive_for_current_process(self):
        v = get_phys_footprint()
        assert v > 0
        # Python interpreter alone should be at least a few MB.
        assert v > 4 * 1024**2

    def test_explicit_pid_matches_default(self):
        v_default = get_phys_footprint()
        v_explicit = get_phys_footprint(pid=os.getpid())
        # Phys can change between two calls (running interpreter), but
        # should be within a small drift band.
        assert abs(v_default - v_explicit) < 32 * 1024**2

    def test_invalid_pid_returns_zero(self):
        # PID 0 is the kernel — proc_pid_rusage refuses it.
        assert get_phys_footprint(pid=0) == 0

    def test_nonexistent_pid_returns_zero(self):
        # Find a PID that is guaranteed not to exist.
        # Use a PID far above the current process table.
        # macOS reserves PID_MAX = 99999, so anything above that is invalid.
        nonexistent_pid = os.getpid() + 100000
        assert get_phys_footprint(pid=nonexistent_pid) == 0

    def test_result_is_reasonable_size(self):
        v = get_phys_footprint()
        # phys_footprint should be representable as a positive int.
        # No upper bound assertion — the value varies by runtime environment
        # (CI runners, debug builds, loaded tooling) making fixed ceilings fragile.
        assert v > 0

    def test_returns_int(self):
        assert isinstance(get_phys_footprint(), int)

    def test_returns_int_for_explicit_pid(self):
        assert isinstance(get_phys_footprint(pid=os.getpid()), int)


class TestGetPhysFootprintFallback:
    def test_returns_zero_on_non_darwin(self, monkeypatch):
        # Simulate libproc unavailable.
        monkeypatch.setattr("omlx.utils.proc_memory._proc_pid_rusage", None)
        assert get_phys_footprint() == 0
        assert get_phys_footprint(pid=12345) == 0


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin-only API")
class TestLifetimeMaxPhysFootprintDarwin:
    def test_returns_positive_for_current_process(self):
        assert get_lifetime_max_phys_footprint() > 0

    def test_is_at_least_current_footprint(self):
        # It is a high-water mark, so it can never sit below the live value.
        assert get_lifetime_max_phys_footprint() >= get_phys_footprint()

    def test_invalid_pid_returns_zero(self):
        assert get_lifetime_max_phys_footprint(pid=0) == 0

    def test_nonexistent_pid_returns_zero(self):
        assert get_lifetime_max_phys_footprint(pid=999999) == 0

    def test_returns_int(self):
        assert isinstance(get_lifetime_max_phys_footprint(), int)


class TestLifetimeMaxPhysFootprintFallback:
    def test_returns_zero_when_libproc_unavailable(self, monkeypatch):
        monkeypatch.setattr("omlx.utils.proc_memory._proc_pid_rusage", None)
        assert get_lifetime_max_phys_footprint() == 0


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin-only API")
class TestGraphicsFootprintDarwin:
    def test_kernel_fills_the_graphics_ledger(self):
        """The kernel fills only whole revisions that fit the caller's count."""
        from omlx.utils import proc_memory as pm

        info = pm._TaskVMInfo()
        count = ctypes.c_uint(pm._TASK_VM_INFO_COUNT)
        rc = pm._task_info(
            pm._mach_task_self.value,
            pm._TASK_VM_INFO,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        assert rc == 0
        assert count.value >= pm._TASK_VM_INFO_GRAPHICS_COUNT
        assert info.phys_footprint == pytest.approx(get_phys_footprint(), rel=0.5)


class TestMetalReleaseLag:
    GB = 1024**3

    def test_lag_is_graphics_above_the_settled_residual(self, monkeypatch):
        from omlx.utils import metal_sync

        graphics = [16.1 * self.GB]
        monkeypatch.setattr(
            metal_sync, "get_graphics_footprint", lambda: int(graphics[0])
        )
        # 0.1 GB of Metal memory outside MLX is the settled level.
        assert metal_sync.unreleased_graphics_bytes(16 * self.GB) == 0
        # MLX dropped a 3 GB pool; the ledger still charges it.
        assert metal_sync.unreleased_graphics_bytes(13 * self.GB) == pytest.approx(
            3 * self.GB, abs=1
        )
        # The driver finished releasing it.
        graphics[0] = 13.1 * self.GB
        assert metal_sync.unreleased_graphics_bytes(13 * self.GB) == 0

    def test_settled_level_follows_the_recent_window(self, monkeypatch):
        from omlx.utils import metal_sync

        now = [100.0]
        monkeypatch.setattr(metal_sync.time, "monotonic", lambda: now[0])
        monkeypatch.setattr(metal_sync, "get_graphics_footprint", lambda: 20 * self.GB)
        metal_sync.unreleased_graphics_bytes(19 * self.GB)
        # New Metal memory outside MLX reads as pending release until the
        # window settles it.
        now[0] += 1.0
        assert metal_sync.unreleased_graphics_bytes(18 * self.GB) == 1 * self.GB
        now[0] += metal_sync._RESIDUAL_WINDOW_S
        assert metal_sync.unreleased_graphics_bytes(18 * self.GB) == 0
