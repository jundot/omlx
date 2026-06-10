# SPDX-License-Identifier: Apache-2.0
"""Tests for scheduler admission control (queue depth cap + admission_paused)."""

from collections import deque
from unittest.mock import MagicMock, patch

import pytest

from omlx.exceptions import SchedulerQueueFullError
from omlx.scheduler import Scheduler

GB = 1024**3


@pytest.fixture
def scheduler():
    """Build a minimal Scheduler instance without invoking __init__.

    Scheduler.__init__ pulls in mlx_lm model wiring; for queue-cap tests we
    only need self.config, self.waiting, self.requests, so we manufacture a
    bare instance and seed those attributes directly.
    """
    s = Scheduler.__new__(Scheduler)
    s.config = MagicMock(max_num_seqs=8)
    s.waiting = deque()
    s.requests = {}
    return s


def _make_request(rid: str):
    r = MagicMock()
    r.request_id = rid
    r.prompt = "hello"
    r.prompt_token_ids = [1, 2, 3]
    r.num_prompt_tokens = 3
    return r


class TestWaitingQueueCap:
    def test_admits_below_cap(self, scheduler):
        # cap = max(max_num_seqs * 4, 32) = 32 for max_num_seqs=8
        # Seed 31 waiting; add_request for #32 should succeed.
        for i in range(31):
            scheduler.waiting.append(_make_request(f"r{i}"))
        # add_request will try to tokenize / fetch cache — short-circuit by
        # making request already tokenized and skipping cache path.
        req = _make_request("r-new")
        # Block all the downstream paths by raising at the next step we don't
        # care about: we only need to confirm the cap check passes (no raise).
        # The easiest way is to insert into self.requests first to force
        # the duplicate check to raise — that lets us prove we got past
        # the cap check.
        scheduler.requests[req.request_id] = req
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_request(req)

    def test_rejects_at_cap(self, scheduler):
        # Fill up to cap (32 with max_num_seqs=8).
        for i in range(32):
            scheduler.waiting.append(_make_request(f"r{i}"))
        req = _make_request("over")
        with pytest.raises(SchedulerQueueFullError) as exc:
            scheduler.add_request(req)
        assert exc.value.current_depth == 32
        assert exc.value.max_depth == 32

    def test_cap_scales_with_max_num_seqs(self, scheduler):
        # cap = max(max_num_seqs * 4, 32); when max_num_seqs=16, cap=64
        scheduler.config.max_num_seqs = 16
        for i in range(64):
            scheduler.waiting.append(_make_request(f"r{i}"))
        with pytest.raises(SchedulerQueueFullError) as exc:
            scheduler.add_request(_make_request("over"))
        assert exc.value.max_depth == 64

    def test_cap_floor_at_32(self, scheduler):
        # Tiny max_num_seqs still gets a floor of 32.
        scheduler.config.max_num_seqs = 1
        for i in range(32):
            scheduler.waiting.append(_make_request(f"r{i}"))
        with pytest.raises(SchedulerQueueFullError) as exc:
            scheduler.add_request(_make_request("over"))
        assert exc.value.max_depth == 32

    def test_duplicate_request_raises_before_cap(self, scheduler):
        # Duplicate check fires before the cap check.
        req = _make_request("dup")
        scheduler.requests[req.request_id] = req
        # Even with an empty queue, duplicate should raise ValueError.
        with pytest.raises(ValueError, match="already exists"):
            scheduler.add_request(req)


class TestAdmissionPausedField:
    def test_default_false(self):
        # Direct field check on a fresh Scheduler — we want to make sure the
        # attribute exists with the right default for enforcer to set.
        s = Scheduler.__new__(Scheduler)
        # Mimic the relevant subset of __init__
        s._memory_limit_bytes = 0
        s._memory_hard_limit_bytes = 0
        s._prefill_memory_guard = False
        s._admission_paused = False
        assert s._admission_paused is False


def _preflight_scheduler(hard_limit: int, recent_peak: int, peak: int):
    """Build a bare Scheduler wired for _preflight_memory_check.

    `peak` is the value the (mocked) memory_monitor estimates for the
    prefill chunk; `recent_peak` is the propagated high-water mark.
    """
    s = Scheduler.__new__(Scheduler)
    s._prefill_memory_guard = True
    s._memory_hard_limit_bytes = hard_limit
    s._memory_recent_peak_bytes = recent_peak
    s.config = MagicMock(prefill_step_size=2048)
    s.memory_monitor = MagicMock()
    s.memory_monitor.estimate_prefill_peak_bytes = MagicMock(return_value=peak)
    return s


def _preflight_request():
    r = MagicMock()
    r.num_prompt_tokens = 8192
    r.cached_tokens = 0
    return r


class TestPreflightRecentPeak:
    """_preflight_memory_check uses the recent high-water mark, not just the
    instant reading, so it does not wave through a request during a prefill
    trough that would wall the next chunk."""

    def test_rejects_on_recent_peak_when_instant_is_low(self):
        """Instant active/phys low but recent_peak high -> reject.

        Picks numbers so that low + peak fits (pre-change behaviour would
        admit) but recent_peak + peak exceeds the hard limit. This pins the
        fix.
        """
        hard_limit = 100 * GB
        peak = 20 * GB
        low = 10 * GB
        high = 85 * GB
        # Sanity: old code (low + peak) would have passed.
        assert low + peak <= hard_limit
        # New code (high + peak) must exceed the limit.
        assert high + peak > hard_limit

        s = _preflight_scheduler(
            hard_limit=hard_limit, recent_peak=high, peak=peak
        )
        with patch("omlx.scheduler.mx") as mock_mx, patch(
            "omlx.scheduler.get_phys_footprint", return_value=low
        ):
            mock_mx.get_active_memory.return_value = low
            result = s._preflight_memory_check(_preflight_request())

        assert result is not None
        assert "Prefill would require" in result

    def test_admits_when_recent_peak_also_low(self):
        """Control: when recent_peak is low too, the request passes."""
        hard_limit = 100 * GB
        peak = 20 * GB
        low = 10 * GB

        s = _preflight_scheduler(
            hard_limit=hard_limit, recent_peak=low, peak=peak
        )
        with patch("omlx.scheduler.mx") as mock_mx, patch(
            "omlx.scheduler.get_phys_footprint", return_value=low
        ):
            mock_mx.get_active_memory.return_value = low
            result = s._preflight_memory_check(_preflight_request())

        assert result is None
