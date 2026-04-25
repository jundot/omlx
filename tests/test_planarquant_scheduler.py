# SPDX-License-Identifier: Apache-2.0
"""Scheduler smoke tests for PlanarQuant memory-pressure wiring."""

from __future__ import annotations

from types import SimpleNamespace

from omlx.cache.planarquant.kv_cache import PlanarQuantKVCache
from omlx.scheduler import Scheduler


class _PressureMonitor:
    def is_under_pressure(self) -> bool:
        return True


def test_memory_pressure_enables_planarquant_tiled_mode_on_running_caches():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.memory_monitor = _PressureMonitor()
    scheduler._planarquant_kv_bits = 3

    cache = PlanarQuantKVCache()
    scheduler.running = {"req": SimpleNamespace(prompt_cache=[cache])}

    scheduler._check_memory_pressure()

    assert cache.memory_pressure is True
    assert cache.tile_size == 4096
