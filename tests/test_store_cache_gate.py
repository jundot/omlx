# SPDX-License-Identifier: Apache-2.0
"""Tests for _StoreCacheGate.

The gate bounds how many KV caches can be alive in the post-completion
store-cache pipeline at once, preventing the unbounded RAM growth reported
in #1383 on burst-finish workloads.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from omlx.scheduler import Scheduler, _StoreCacheGate


class TestAcquireRelease:
    def test_acquire_below_cap_does_not_block(self):
        gate = _StoreCacheGate(cap=3)
        assert gate.acquire() is True
        assert gate.acquire() is True
        assert gate.in_flight == 2

    def test_acquire_blocks_at_cap_until_release(self):
        gate = _StoreCacheGate(cap=1)
        assert gate.acquire() is True
        unblocked = threading.Event()

        def waiter():
            assert gate.acquire() is True
            unblocked.set()

        t = threading.Thread(target=waiter)
        t.start()
        # waiter should still be blocked
        assert not unblocked.wait(0.05)
        gate.release()
        assert unblocked.wait(1.0)
        t.join()

    def test_release_does_not_underflow(self):
        gate = _StoreCacheGate(cap=2)
        gate.release()
        gate.release()
        assert gate.in_flight == 0
        assert gate.acquire() is True
        assert gate.in_flight == 1


class TestSetCap:
    def test_clamps_to_minimum_one(self):
        gate = _StoreCacheGate(cap=4)
        gate.set_cap(0)
        assert gate.cap == 1
        gate.set_cap(-5)
        assert gate.cap == 1

    def test_shrink_does_not_wake_waiters(self):
        gate = _StoreCacheGate(cap=2)
        gate.acquire()
        gate.acquire()
        unblocked = threading.Event()

        def waiter():
            gate.acquire()
            unblocked.set()

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        gate.set_cap(1)  # shrink under load — waiter must stay blocked
        assert not unblocked.wait(0.05)
        # release twice — only after second release should waiter wake up
        gate.release()
        assert not unblocked.wait(0.05)
        gate.release()
        assert unblocked.wait(1.0)
        t.join()

    def test_grow_wakes_waiters(self):
        gate = _StoreCacheGate(cap=1)
        gate.acquire()
        unblocked = threading.Event()

        def waiter():
            gate.acquire()
            unblocked.set()

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        assert not unblocked.is_set()
        gate.set_cap(3)  # grow — waiter has room now
        assert unblocked.wait(1.0)
        t.join()

    def test_no_op_when_cap_unchanged(self):
        gate = _StoreCacheGate(cap=4)
        gate.set_cap(4)
        assert gate.cap == 4


class TestShutdown:
    def test_shutdown_unblocks_waiters_with_false(self):
        gate = _StoreCacheGate(cap=1)
        gate.acquire()
        result = {}

        def waiter():
            result["value"] = gate.acquire()

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        gate.shutdown()
        t.join(timeout=1.0)
        assert result["value"] is False

    def test_acquire_after_shutdown_returns_false(self):
        gate = _StoreCacheGate(cap=2)
        gate.shutdown()
        assert gate.acquire() is False


class TestAdjustStoreCacheCap:
    """Tests for Scheduler.adjust_store_cache_cap pressure mapping."""

    def _fake_scheduler(self, cap, max_num_seqs=8):
        sched = MagicMock(spec=["config", "_store_cache_gate"])
        sched.config = MagicMock(spec=["max_num_seqs"])
        sched.config.max_num_seqs = max_num_seqs
        sched._store_cache_gate = _StoreCacheGate(cap=cap)
        return sched

    def test_ok_grows_by_one(self):
        sched = self._fake_scheduler(cap=4, max_num_seqs=8)
        Scheduler.adjust_store_cache_cap(sched, "ok")
        assert sched._store_cache_gate.cap == 5

    def test_ok_clamps_to_max_num_seqs(self):
        sched = self._fake_scheduler(cap=8, max_num_seqs=8)
        Scheduler.adjust_store_cache_cap(sched, "ok")
        assert sched._store_cache_gate.cap == 8

    def test_soft_shrinks_by_one(self):
        sched = self._fake_scheduler(cap=5)
        Scheduler.adjust_store_cache_cap(sched, "soft")
        assert sched._store_cache_gate.cap == 4

    def test_hard_shrinks_by_one(self):
        sched = self._fake_scheduler(cap=3)
        Scheduler.adjust_store_cache_cap(sched, "hard")
        assert sched._store_cache_gate.cap == 2

    def test_shrink_floor_at_one(self):
        sched = self._fake_scheduler(cap=1)
        Scheduler.adjust_store_cache_cap(sched, "hard")
        assert sched._store_cache_gate.cap == 1

    def test_no_op_when_gate_missing(self):
        sched = MagicMock(spec=["config", "_store_cache_gate"])
        sched.config = MagicMock(spec=["max_num_seqs"])
        sched.config.max_num_seqs = 8
        sched._store_cache_gate = None
        # Should not raise.
        Scheduler.adjust_store_cache_cap(sched, "ok")


class TestThreadSafety:
    @pytest.mark.timeout(5)
    def test_bounded_under_contention(self):
        """Many threads acquiring/releasing should never exceed cap."""
        gate = _StoreCacheGate(cap=4)
        peak = {"value": 0}
        peak_lock = threading.Lock()

        def worker():
            for _ in range(50):
                assert gate.acquire() is True
                with peak_lock:
                    peak["value"] = max(peak["value"], gate.in_flight)
                time.sleep(0.001)
                gate.release()

        threads = [threading.Thread(target=worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert peak["value"] <= 4
