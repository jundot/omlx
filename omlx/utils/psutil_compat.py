# SPDX-License-Identifier: Apache-2.0
"""Small psutil-compatible memory helpers for macOS VM stat churn.

macOS 27 changed the HOST_VM_INFO64 tail layout before psutil had a matching
adapter in some bundled environments. oMLX only needs a small subset of
``psutil.virtual_memory()`` for memory guard and dashboard decisions, so keep a
local implementation for macOS and use psutil on other platforms.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import re
import subprocess
import sys
import time
from collections import namedtuple
from typing import Any

logger = logging.getLogger(__name__)

_HOST_VM_INFO64 = 4
_HOST_INFO64_MAX_COUNT = 256
_VM_STATS_MIN_COUNT = 4
_VM_PAGE_SIZE = 16384
_SYSCTL = "/usr/sbin/sysctl"
_VM_STAT = "/usr/bin/vm_stat"
_SLOW_FALLBACK_TTL_S = 1.0

svmem = namedtuple(
    "svmem",
    ["total", "available", "percent", "used", "free", "active", "inactive", "wired"],
)

_cached_total_memory: int | None = None
_cached_slow_virtual_memory: Any | None = None
_cached_slow_virtual_memory_at = 0.0

if sys.platform == "darwin":
    try:
        _libc = ctypes.CDLL(ctypes.util.find_library("c"))
        _libc.mach_host_self.restype = ctypes.c_uint
        _MACH_HOST = _libc.mach_host_self()
        _ps = ctypes.c_uint(0)
        _libc.host_page_size(_MACH_HOST, ctypes.byref(_ps))
        if _ps.value > 0:
            _VM_PAGE_SIZE = int(_ps.value)
    except Exception:  # noqa: BLE001
        _libc = None
        _MACH_HOST = None
else:
    _libc = None
    _MACH_HOST = None


def get_total_memory() -> int:
    """Return total physical memory in bytes."""
    global _cached_total_memory

    if _cached_total_memory is not None:
        return _cached_total_memory

    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = int(pages) * int(page_size)
        if total > 0:
            _cached_total_memory = total
            return total
    except (AttributeError, ValueError, OSError):
        pass

    if sys.platform == "darwin":
        try:
            out = subprocess.check_output([_SYSCTL, "-n", "hw.memsize"], text=True)
            total = int(out.strip())
            if total > 0:
                _cached_total_memory = total
                return total
        except (subprocess.SubprocessError, ValueError, OSError):
            pass

    try:
        import psutil  # type: ignore

        total = int(psutil.virtual_memory().total)
        if total > 0:
            _cached_total_memory = total
            return total
    except Exception as exc:  # noqa: BLE001
        logger.debug("psutil total memory fallback failed: %s", exc)

    return 0


def get_macos_vm_stats() -> dict[str, int] | None:
    """Return macOS vm_statistics64 page counters in bytes.

    The first four counters are stable across SDK versions. The oversized
    host_info64_t buffer avoids binding oMLX to an SDK-specific struct tail.
    """
    if _libc is None or _MACH_HOST is None:
        return None
    try:
        stats = (ctypes.c_int * _HOST_INFO64_MAX_COUNT)()
        count = ctypes.c_uint(_HOST_INFO64_MAX_COUNT)
        rc = _libc.host_statistics64(
            _MACH_HOST, _HOST_VM_INFO64, stats, ctypes.byref(count)
        )
        if rc != 0 or count.value < _VM_STATS_MIN_COUNT:
            return None
        ps = _VM_PAGE_SIZE
        return {
            "free": int(stats[0]) * ps,
            "active": int(stats[1]) * ps,
            "inactive": int(stats[2]) * ps,
            "wired": int(stats[3]) * ps,
        }
    except Exception:  # noqa: BLE001
        return None


def _build_svmem(stats: dict[str, int], total: int | None = None) -> Any:
    free = max(0, int(stats.get("free", 0) or 0))
    active = max(0, int(stats.get("active", 0) or 0))
    inactive = max(0, int(stats.get("inactive", 0) or 0))
    wired = max(0, int(stats.get("wired", 0) or 0))
    available = max(0, free + inactive)
    total_bytes = int(total or get_total_memory() or 0)
    if total_bytes <= 0:
        total_bytes = max(available + active + wired, free + active + inactive + wired)
    used = max(0, total_bytes - available)
    percent = (used / total_bytes * 100.0) if total_bytes > 0 else 0.0
    return svmem(
        total=total_bytes,
        available=available,
        percent=percent,
        used=used,
        free=free,
        active=active,
        inactive=inactive,
        wired=wired,
    )


def _parse_vm_stat_output(output: str) -> dict[str, int] | None:
    page_size = _VM_PAGE_SIZE
    first_line = output.splitlines()[0] if output else ""
    match = re.search(r"page size of (\d+) bytes", first_line)
    if match is not None:
        page_size = int(match.group(1))

    pages: dict[str, int] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value = raw_value.strip().rstrip(".").replace(" ", "")
        if not value.isdigit():
            continue
        pages[key.strip()] = int(value)

    if not pages:
        return None
    return {
        "free": pages.get("Pages free", 0) * page_size,
        "active": pages.get("Pages active", 0) * page_size,
        "inactive": pages.get("Pages inactive", 0) * page_size,
        "wired": pages.get("Pages wired down", 0) * page_size,
    }


def _virtual_memory_from_vm_stat_cached() -> Any | None:
    global _cached_slow_virtual_memory, _cached_slow_virtual_memory_at

    now = time.monotonic()
    if (
        _cached_slow_virtual_memory is not None
        and now - _cached_slow_virtual_memory_at < _SLOW_FALLBACK_TTL_S
    ):
        return _cached_slow_virtual_memory

    try:
        out = subprocess.check_output([_VM_STAT], text=True)
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("vm_stat fallback failed: %s", exc)
        return None

    stats = _parse_vm_stat_output(out)
    if stats is None:
        return None

    vm = _build_svmem(stats)
    _cached_slow_virtual_memory = vm
    _cached_slow_virtual_memory_at = now
    return vm


def virtual_memory(*, allow_slow_fallback: bool = True) -> Any:
    """Return a psutil-like virtual memory snapshot.

    On macOS this avoids psutil's HOST_VM_INFO64 adapter. On other platforms it
    delegates to psutil.
    """
    if sys.platform == "darwin":
        stats = get_macos_vm_stats()
        if stats is not None:
            return _build_svmem(stats)
        if allow_slow_fallback:
            vm = _virtual_memory_from_vm_stat_cached()
            if vm is not None:
                return vm
        raise RuntimeError("macOS VM statistics unavailable")

    import psutil  # type: ignore

    return psutil.virtual_memory()
