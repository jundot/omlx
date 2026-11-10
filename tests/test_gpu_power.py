# SPDX-License-Identifier: Apache-2.0
"""GPU power limiter: lowest cap wins, delayed removal, crash recovery."""

import asyncio
import os

import pytest

from omlx.utils import gpu_power
from omlx.utils.gpu_power import GpuPowerLimiter


class FakeCap:
    def __init__(self, supported=True, write_ok=True):
        self.supported = supported
        self.write_ok = write_ok
        self.value = 181795
        self.writes = []

    def read_mw(self):
        return self.value

    def write_mw(self, milliwatts):
        self.writes.append(milliwatts)
        if self.write_ok:
            self.value = milliwatts
        return self.write_ok


def _limiter(tmp_path, cap, release_delay_s=0.02):
    # A long check interval keeps the hold loop out of these tests.
    return GpuPowerLimiter(
        tmp_path, cap=cap, release_delay_s=release_delay_s, check_interval_s=60
    )


@pytest.mark.asyncio
async def test_lowest_active_cap_applies_and_is_removed_after_delay(tmp_path):
    cap = FakeCap()
    limiter = _limiter(tmp_path, cap)
    marker = tmp_path / gpu_power.MARKER_NAME

    limiter.acquire(20)
    limiter.acquire(10)
    limiter.release(10)
    assert cap.writes == [20000, 10000, 20000]
    assert marker.exists()

    limiter.release(20)
    assert cap.writes[-1] == 20000
    await asyncio.sleep(0.05)
    assert cap.writes[-1] == -1000
    assert limiter.target_watts is None
    assert not marker.exists()
    await limiter.close()


@pytest.mark.asyncio
async def test_request_within_release_delay_keeps_the_cap(tmp_path):
    cap = FakeCap()
    limiter = _limiter(tmp_path, cap, release_delay_s=0.05)

    limiter.acquire(10)
    limiter.release(10)
    limiter.acquire(10)
    await asyncio.sleep(0.08)
    assert cap.writes == [10000]
    limiter.release(10)
    await limiter.close()
    assert cap.writes == [10000, -1000]


@pytest.mark.asyncio
async def test_benchmark_drops_only_an_idle_cap(tmp_path):
    cap = FakeCap()
    limiter = _limiter(tmp_path, cap, release_delay_s=60)
    limiter.acquire(10)
    limiter.drop_idle_cap()  # a capped request is still running
    assert cap.writes == [10000]
    limiter.release(10)
    limiter.drop_idle_cap()  # only the release delay holds it
    assert cap.writes == [10000, -1000]
    await limiter.close()


@pytest.mark.asyncio
async def test_raised_or_removed_cap_is_rewritten_but_lower_limit_stays(tmp_path):
    cap = FakeCap()
    limiter = _limiter(tmp_path, cap)
    limiter.acquire(10)

    cap.value = 55000  # thermal controller raised the limit
    assert limiter._check(0) == 1
    assert cap.writes[-1] == 10000
    assert limiter._check(1) == 0  # rewrite held
    cap.value = -1000  # someone removed the cap
    assert limiter._check(0) == 1
    cap.value = 8000  # a stricter thermal limit
    assert limiter._check(0) == 0
    assert cap.writes == [10000, 10000, 10000]
    assert [gpu_power.rewrite_delay_steps(m) for m in range(5)] == [1, 1, 2, 3, 3]
    await limiter.close()


@pytest.mark.asyncio
async def test_cap_set_again_in_the_release_tick_keeps_a_hold_task(tmp_path):
    limiter = _limiter(tmp_path, FakeCap())
    limiter.acquire(10)
    limiter._clear()
    limiter.acquire(10)
    assert limiter._hold_task is not None and not limiter._hold_task.cancelled()
    await limiter.close()


def test_recover_removes_cap_left_by_dead_server_only(tmp_path):
    cap = FakeCap()
    marker = tmp_path / gpu_power.MARKER_NAME
    marker.write_text(str(os.getppid()))  # a live server still owns its cap
    GpuPowerLimiter(tmp_path, cap=cap).recover()
    assert cap.writes == []

    marker.write_text("0")
    cap.write_ok = False  # removal failed; retry on the next start
    GpuPowerLimiter(tmp_path, cap=cap).recover()
    assert marker.exists()

    cap.write_ok = True
    GpuPowerLimiter(tmp_path, cap=cap).recover()
    assert cap.writes[-1] == -1000
    assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [FakeCap(supported=False), FakeCap(write_ok=False)])
async def test_unusable_cap_leaves_requests_uncapped(tmp_path, cap):
    limiter = _limiter(tmp_path, cap)
    limiter.acquire(10)
    limiter.acquire(10)
    limiter.release(10)
    limiter.release(10)
    await limiter.close()
    assert cap.writes in ([], [10000])
    assert not cap.supported
    assert not (tmp_path / gpu_power.MARKER_NAME).exists()
