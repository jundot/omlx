# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.utils.proc_memory.get_phys_footprint."""

import os
import sys

import pytest

from omlx.utils.proc_memory import get_phys_footprint


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

    def test_includes_python_heap(self):
        baseline = get_phys_footprint()
        # Allocate a large buffer to force phys growth. The allocation is
        # deliberately much larger than the >32MB assertion threshold:
        # under a full-suite run a warm allocator can satisfy a smaller
        # buffer from already-resident cached pages, so phys would not grow
        # and the test would flake. 256MB against a 32MB floor leaves wide
        # headroom for allocator page reuse while still proving the heap
        # is counted.
        big = bytearray(256 * 1024 * 1024)
        # Touch it so pages become resident.
        for i in range(0, len(big), 4096):
            big[i] = 1
        after = get_phys_footprint()
        # phys should have grown by a meaningful fraction of the allocation.
        assert after - baseline > 32 * 1024 * 1024
        del big


class TestGetPhysFootprintFallback:
    def test_returns_zero_on_non_darwin(self, monkeypatch):
        # Simulate libproc unavailable.
        monkeypatch.setattr("omlx.utils.proc_memory._proc_pid_rusage", None)
        assert get_phys_footprint() == 0
        assert get_phys_footprint(pid=12345) == 0
