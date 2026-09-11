# SPDX-License-Identifier: Apache-2.0
"""The accuracy queue waits for the kernel ledger to settle between two items.

Measured 03/09: the unload reported `freed=106.72GB, active_memory: 433.66MB
(settled)` at 15:06:25.491 and the queue asked for the next model in the SAME
millisecond. Admission compares the cap against
`max(active_memory, phys_footprint, accounted)`, and phys_footprint -- the
macOS ledger -- was still counting 75.91 GB of reclaimable pages from the model
that had just left. Two items died with InsufficientMemoryError, with the
machine genuinely empty; 92 s later the same load went through on its own.
"""

import asyncio

import pytest

from omlx.admin import accuracy_benchmark as ab


@pytest.mark.asyncio
async def test_waits_until_the_ledger_falls(monkeypatch):
    """While the footprint is high the queue holds; once it falls, it goes on."""
    readings = iter([80 * 1024**3, 60 * 1024**3, 30 * 1024**3, 1 * 1024**3])
    seen = []

    def _footprint():
        v = next(readings)
        seen.append(v)
        return v

    monkeypatch.setattr("omlx.utils.proc_memory.get_phys_footprint", _footprint)
    _sleep_real = asyncio.sleep
    monkeypatch.setattr(ab.asyncio, "sleep", lambda _s: _sleep_real(0))

    class _MX:
        @staticmethod
        def get_active_memory():
            return 400 * 1024**2

    monkeypatch.setitem(__import__("sys").modules, "mlx.core", _MX)
    await ab._wait_for_memory_to_settle(None, timeout_s=5.0)
    # waited out the high readings and stopped at the settled one (1 GB <= 0.4 + 8)
    assert len(seen) == 4, seen


@pytest.mark.asyncio
async def test_does_not_wait_when_it_is_already_low(monkeypatch):
    """With the ledger already low it costs nothing -- the queue loses no time."""
    calls = []

    def _footprint():
        calls.append(1)
        return 500 * 1024**2

    monkeypatch.setattr("omlx.utils.proc_memory.get_phys_footprint", _footprint)

    class _MX:
        @staticmethod
        def get_active_memory():
            return 400 * 1024**2

    monkeypatch.setitem(__import__("sys").modules, "mlx.core", _MX)
    await ab._wait_for_memory_to_settle(None, timeout_s=5.0)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_gives_up_when_it_stops_falling(monkeypatch):
    """If the ledger stabilizes high, insisting does not help: admission has its
    own eviction path, and holding the queue forever is worse."""
    readings = iter([80 * 1024**3, 79 * 1024**3, 79 * 1024**3, 79 * 1024**3])
    seen = []

    def _footprint():
        v = next(readings)
        seen.append(v)
        return v

    monkeypatch.setattr("omlx.utils.proc_memory.get_phys_footprint", _footprint)
    _sleep_real = asyncio.sleep
    monkeypatch.setattr(ab.asyncio, "sleep", lambda _s: _sleep_real(0))

    class _MX:
        @staticmethod
        def get_active_memory():
            return 400 * 1024**2

    monkeypatch.setitem(__import__("sys").modules, "mlx.core", _MX)
    await ab._wait_for_memory_to_settle(None, timeout_s=5.0)
    assert len(seen) == 3, seen
