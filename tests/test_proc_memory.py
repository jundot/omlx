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
