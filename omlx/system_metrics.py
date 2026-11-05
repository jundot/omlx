# SPDX-License-Identifier: Apache-2.0
"""Host resource metrics behind ``GET /api/system-metrics``.

Mirrors what the macOS app's ``SystemStatsSampler`` reads on the local Mac, so
a client pointed at a remote server can show the server host's CPU, GPU,
memory, load, uptime and thermal state. Public sources only: psutil, sysctl,
``ioreg`` and ``NSProcessInfo``. Every leaf is ``None`` when unknown; no probe
failure ever raises out of :func:`collect_system_metrics`.

CPU percentages come from deltas between two calls, so the first response of a
process has ``None`` for them. Later calls less than ``_CPU_MIN_INTERVAL``
after the previous sample reuse it, so a burst of requests cannot shrink the
measurement window to noise.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

_IS_DARWIN = sys.platform == "darwin"

_SUBPROCESS_TIMEOUT = 2.0
_IOREG_TTL = 1.0
_VM_STAT_TTL = 1.0
_CPU_MIN_INTERVAL = 0.5

# NSProcessInfoThermalState raw values, in order.
_THERMAL_STATES = ("nominal", "fair", "serious", "critical")

_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}
_cpu_primed = False
_cpu_sample_at = 0.0
_cpu_last: list[float] | None = None


def _cached(key: str, ttl: float, probe: Callable[[], Any]) -> Any:
    """Return ``probe()``, reusing the last result (including None) for ``ttl`` s."""
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit is not None and now - hit[0] < ttl:
            return hit[1]
    value = probe()
    with _lock:
        _cache[key] = (time.monotonic(), value)
    return value


def _run(*cmd: str) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
            check=True,
        )
        return result.stdout
    except Exception as exc:  # noqa: BLE001 - probes must never raise
        logger.debug("system metrics: %s failed: %s", cmd[0], exc)
        return None


def _sysctl_int(name: str) -> int | None:
    if not _IS_DARWIN:
        return None
    out = _run("sysctl", "-n", name)
    try:
        return int(out.strip()) if out else None
    except ValueError:
        return None


# --- CPU -------------------------------------------------------------------


def efficiency_core_count() -> int:
    """Logical CPUs in the E cluster; 0 when unknown (every core counts as P).

    Same rule as ``SystemStatsSampler.readECoreCount``: Apple Silicon lists the
    E cluster first, and ``hw.perflevel1`` is that cluster whenever two
    performance levels exist.
    """
    levels = _cached("nperflevels", 3600, lambda: _sysctl_int("hw.nperflevels"))
    if levels is None or levels < 2:
        return 0
    count = _cached("perflevel1", 3600, lambda: _sysctl_int("hw.perflevel1.logicalcpu"))
    return count or 0


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _clamp_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return round(min(100.0, max(0.0, float(value))), 1)


def split_cpu_clusters(per_cpu: Sequence[float], e_core_count: int) -> dict[str, Any]:
    """Split per-CPU busy percentages into the efficiency and performance clusters.

    Indices ``[0, e_core_count)`` are E cores, the rest are P cores, as in the
    Swift sampler. Percentages are plain means: every CPU is measured over the
    same window, so this equals the Swift tick-weighted average. With
    ``e_core_count == 0`` there is no E cluster and its percentage is None.
    """
    e_values = list(per_cpu[:e_core_count])
    p_values = list(per_cpu[e_core_count:])
    return {
        "total_percent": _clamp_percent(_mean(list(per_cpu))),
        "efficiency_percent": _clamp_percent(_mean(e_values)),
        "performance_percent": _clamp_percent(_mean(p_values)),
        "efficiency_cores": len(e_values),
        "performance_cores": len(p_values),
    }


def _sample_per_cpu() -> list[float] | None:
    """Per-CPU percentages since the previous sample; None until primed."""
    global _cpu_primed, _cpu_sample_at, _cpu_last
    import psutil

    with _lock:
        now = time.monotonic()
        if _cpu_primed and now - _cpu_sample_at < _CPU_MIN_INTERVAL:
            return _cpu_last
        values = psutil.cpu_percent(percpu=True, interval=None)
        if not _cpu_primed:
            # The first call reports 0.0 for everything: not a measurement.
            _cpu_primed = True
            _cpu_sample_at = now
            return None
        _cpu_sample_at = now
        _cpu_last = list(values)
        return _cpu_last


def _cpu_metrics() -> dict[str, Any]:
    out: dict[str, Any] = {
        "total_percent": None,
        "efficiency_percent": None,
        "performance_percent": None,
        "efficiency_cores": None,
        "performance_cores": None,
        "load_averages": None,
    }
    try:
        import psutil

        e_count = efficiency_core_count()
        logical = psutil.cpu_count(logical=True)
        if logical:
            # Known even before the first CPU delta exists.
            out["efficiency_cores"] = min(e_count, logical)
            out["performance_cores"] = logical - out["efficiency_cores"]
        per_cpu = _sample_per_cpu()
        if per_cpu:
            out.update(split_cpu_clusters(per_cpu, e_count))
    except Exception as exc:  # noqa: BLE001
        logger.debug("system metrics: cpu probe failed: %s", exc)
    try:
        out["load_averages"] = [round(v, 2) for v in os.getloadavg()]
    except Exception as exc:  # noqa: BLE001
        logger.debug("system metrics: loadavg failed: %s", exc)
    return out


# --- GPU -------------------------------------------------------------------

_GPU_UTIL_RE = re.compile(r'"Device Utilization %"\s*=\s*(\d+)')
_GPU_MEM_RE = re.compile(r'"In use system memory"\s*=\s*(\d+)')


def parse_ioreg_gpu(text: str) -> dict[str, Any]:
    """Pull GPU utilization and in-use memory out of ``ioreg -c IOAccelerator``.

    Reads the same ``PerformanceStatistics`` keys as the Swift sampler. The
    ``"In use system memory (driver)"`` key does not match: the closing quote
    must follow the name. The first accelerator that reports either value wins.
    """
    for line in text.splitlines():
        if '"PerformanceStatistics"' not in line:
            continue
        util = _GPU_UTIL_RE.search(line)
        mem = _GPU_MEM_RE.search(line)
        if util or mem:
            return {
                "utilization_percent": (
                    _clamp_percent(float(util.group(1))) if util else None
                ),
                "memory_in_use_bytes": int(mem.group(1)) if mem else None,
            }
    return {"utilization_percent": None, "memory_in_use_bytes": None}


def _gpu_metrics() -> dict[str, Any]:
    empty = {"utilization_percent": None, "memory_in_use_bytes": None}
    if not _IS_DARWIN:
        return empty

    def probe() -> dict[str, Any] | None:
        out = _run("ioreg", "-r", "-d", "1", "-c", "IOAccelerator")
        return parse_ioreg_gpu(out) if out else None

    try:
        return _cached("ioreg", _IOREG_TTL, probe) or empty
    except Exception as exc:  # noqa: BLE001
        logger.debug("system metrics: gpu probe failed: %s", exc)
        return empty


# --- Memory ----------------------------------------------------------------

_VM_STAT_PAGE_RE = re.compile(r"page size of (\d+) bytes")
_VM_STAT_COMPRESSOR_RE = re.compile(r"Pages occupied by compressor:\s*(\d+)")


def parse_vm_stat_compressed(text: str) -> int | None:
    """Bytes held by the memory compressor from ``vm_stat`` output."""
    page = _VM_STAT_PAGE_RE.search(text)
    comp = _VM_STAT_COMPRESSOR_RE.search(text)
    if not page or not comp:
        return None
    return int(comp.group(1)) * int(page.group(1))


def _compressed_bytes() -> int | None:
    if not _IS_DARWIN:
        return None

    def probe() -> int | None:
        out = _run("vm_stat")
        return parse_vm_stat_compressed(out) if out else None

    return _cached("vm_stat", _VM_STAT_TTL, probe)


def _memory_metrics() -> dict[str, Any]:
    out: dict[str, Any] = {
        "total_bytes": None,
        "wired_bytes": None,
        "active_bytes": None,
        "inactive_bytes": None,
        "compressed_bytes": None,
        "free_bytes": None,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        out["total_bytes"] = int(vm.total)
        out["wired_bytes"] = _int_attr(vm, "wired")
        out["active_bytes"] = _int_attr(vm, "active")
        out["inactive_bytes"] = _int_attr(vm, "inactive")
    except Exception as exc:  # noqa: BLE001
        logger.debug("system metrics: memory probe failed: %s", exc)
    try:
        out["compressed_bytes"] = _compressed_bytes()
    except Exception as exc:  # noqa: BLE001
        logger.debug("system metrics: compressor probe failed: %s", exc)

    # Same definition as the Swift sampler: free = total - (wired + active +
    # compressed), so the legend rows always sum to the machine total.
    parts = (out["total_bytes"], out["wired_bytes"], out["active_bytes"])
    if all(p is not None for p in parts) and out["compressed_bytes"] is not None:
        used = out["wired_bytes"] + out["active_bytes"] + out["compressed_bytes"]
        out["free_bytes"] = max(0, out["total_bytes"] - used)
    return out


def _int_attr(obj: Any, name: str) -> int | None:
    value = getattr(obj, name, None)
    return int(value) if value is not None else None


# --- Thermal ---------------------------------------------------------------


def _thermal_state() -> str | None:
    """``NSProcessInfo.thermalState`` through the Objective-C runtime (macOS)."""
    if not _IS_DARWIN:
        return None
    try:
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        addr = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        send_ptr = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
            addr
        )
        send_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
            addr
        )
        info = send_ptr(
            objc.objc_getClass(b"NSProcessInfo"),
            objc.sel_registerName(b"processInfo"),
        )
        raw = send_long(info, objc.sel_registerName(b"thermalState"))
        return _THERMAL_STATES[raw] if 0 <= raw < len(_THERMAL_STATES) else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("system metrics: thermal probe failed: %s", exc)
        return None


# --- Entry point -----------------------------------------------------------


def _uptime_seconds() -> float | None:
    try:
        import psutil

        return round(max(0.0, time.time() - psutil.boot_time()), 1)
    except Exception as exc:  # noqa: BLE001
        logger.debug("system metrics: uptime failed: %s", exc)
        return None


def collect_system_metrics() -> dict[str, Any]:
    """Snapshot of this host's resource use; never raises."""
    return {
        "timestamp": time.time(),
        "uptime_seconds": _uptime_seconds(),
        "cpu": _cpu_metrics(),
        "gpu": _gpu_metrics(),
        "memory": _memory_metrics(),
        "thermal_state": _thermal_state(),
    }
