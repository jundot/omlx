"""Host telemetry sampling for benchmark runs.

Samples CPU utilization (split by performance/efficiency cluster), GPU
utilization, thermal pressure, and memory at a fixed interval so a benchmark
can report the conditions its numbers were produced under.

Everything here is public API: Mach host statistics, the IOKit registry,
sysctl, ioreg, and the OSThermalNotification pressure level. Die temperature,
power draw, and core frequencies would need private frameworks and are out of
scope; the thermal pressure level answers the question that actually matters
for benchmark validity ("was this run throttled?").

Every reader degrades to None independently, so a failure in one subsystem
(or a non-Darwin platform) leaves the rest usable.
"""

import contextlib
import ctypes
import ctypes.util
import logging
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache

from .proc_memory import get_phys_footprint
from .psutil_compat import get_macos_vm_stats, get_total_memory

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:  # pragma: no cover - exercised only without MLX
    mx = None  # type: ignore[assignment]
    HAS_MLX = False

IS_DARWIN = sys.platform == "darwin"

# host_processor_info / PROCESSOR_CPU_LOAD_INFO
_PROCESSOR_CPU_LOAD_INFO = 2
_CPU_STATE_MAX = 4
_CPU_STATE_USER = 0
_CPU_STATE_SYSTEM = 1
_CPU_STATE_IDLE = 2
_CPU_STATE_NICE = 3

# /usr/sbin and /usr/bin are not on PATH under launchd (issue #1322).
_IOREG = "/usr/sbin/ioreg"

# 4 Hz. A single sample costs ~0.05 ms, so this is ~0.02% of one core — and
# benchmark tests are short: a 1024-token prompt with a 32-token generation on
# a small model finishes in well under a second, which at 1 Hz produced no
# usable window at all.
_DEFAULT_INTERVAL_S = 0.25
# 2 hours at 4 Hz. Each record is ~200 B, and the sampler is created per run
# and dropped afterwards, so the ceiling is ~5.8 MB while a bench is running.
_MAX_SAMPLES = 28800


@dataclass(frozen=True)
class _Sample:
    """One instant of host state. Fractions are 0..1, bytes are bytes."""

    t: float
    cpu_total: float | None
    cpu_p: float | None
    cpu_e: float | None
    gpu_util: float | None
    thermal: int | None
    phys_footprint: int
    mlx_active: int
    mlx_cache: int
    sys_used: int
    sys_wired: int


class _CPUReader:
    """Per-CPU tick deltas from host_processor_info, split by cluster."""

    def __init__(self) -> None:
        self._libc: ctypes.CDLL | None = None
        self._host = 0
        self._task = 0
        self._prev: list[tuple[int, int]] | None = None
        if not IS_DARWIN:
            return
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            libc.mach_host_self.restype = ctypes.c_uint
            libc.mach_task_self.restype = ctypes.c_uint
            libc.host_processor_info.argtypes = [
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                ctypes.POINTER(ctypes.c_uint),
            ]
            libc.host_processor_info.restype = ctypes.c_int
            libc.vm_deallocate.argtypes = [
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            libc.vm_deallocate.restype = ctypes.c_int
            self._libc = libc
            self._host = libc.mach_host_self()
            self._task = libc.mach_task_self()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"CPU sampling unavailable: {e}")
            self._libc = None

    def _raw_ticks(self) -> list[tuple[int, int]] | None:
        """Return per-CPU (busy, total) cumulative ticks."""
        if self._libc is None:
            return None
        cpu_count = ctypes.c_uint(0)
        info = ctypes.POINTER(ctypes.c_int)()
        info_count = ctypes.c_uint(0)
        rc = self._libc.host_processor_info(
            self._host,
            _PROCESSOR_CPU_LOAD_INFO,
            ctypes.byref(cpu_count),
            ctypes.byref(info),
            ctypes.byref(info_count),
        )
        if rc != 0:
            return None
        try:
            out: list[tuple[int, int]] = []
            for i in range(cpu_count.value):
                base = i * _CPU_STATE_MAX
                # The kernel counters are 32-bit natural_t; ctypes hands them
                # back signed, so reinterpret before doing arithmetic.
                user = ctypes.c_uint32(info[base + _CPU_STATE_USER]).value
                system = ctypes.c_uint32(info[base + _CPU_STATE_SYSTEM]).value
                idle = ctypes.c_uint32(info[base + _CPU_STATE_IDLE]).value
                nice = ctypes.c_uint32(info[base + _CPU_STATE_NICE]).value
                busy = user + system + nice
                out.append((busy, busy + idle))
            return out
        finally:
            # Without this the array leaks 4 bytes per state per CPU on every
            # sample — ~43 MB/day on a long-lived server at 1 Hz.
            self._libc.vm_deallocate(
                self._task,
                ctypes.cast(info, ctypes.c_void_p),
                info_count.value * ctypes.sizeof(ctypes.c_int),
            )

    def sample(self) -> tuple[float | None, float | None, float | None]:
        """Return (total, performance, efficiency) utilization since last call."""
        ticks = self._raw_ticks()
        if ticks is None:
            return (None, None, None)
        prev, self._prev = self._prev, ticks
        if prev is None or len(prev) != len(ticks):
            return (None, None, None)

        is_efficiency = _cluster_map(len(ticks))
        buckets: dict[str, list[int]] = {
            "all": [0, 0],
            "p": [0, 0],
            "e": [0, 0],
        }
        for i, ((busy, total), (pbusy, ptotal)) in enumerate(zip(ticks, prev)):
            d_busy = busy - pbusy
            d_total = total - ptotal
            # A 32-bit counter wrap makes the delta meaningless for this CPU;
            # drop it from both numerator and denominator rather than clamping.
            if d_total <= 0 or d_busy < 0:
                continue
            for key in ("all", "e" if is_efficiency[i] else "p"):
                buckets[key][0] += d_busy
                buckets[key][1] += d_total

        def frac(key: str) -> float | None:
            busy, total = buckets[key]
            return busy / total if total > 0 else None

        return (frac("all"), frac("p"), frac("e"))


@lru_cache(maxsize=1)
def _cluster_map(cpu_count: int) -> tuple[bool, ...]:
    """Map logical CPU index to "is an efficiency core".

    Reads the IODeviceTree cluster-type, because assuming the efficiency cores
    occupy a contiguous low-index prefix is wrong on fused-die parts: an M3
    Ultra reports 8 efficiency cores but lays them out as two groups of four at
    indices 0-3 and 16-19, with performance cores in between.
    """
    mapping = _cluster_map_from_ioreg()
    if mapping is not None and len(mapping) == cpu_count:
        return mapping
    # Contiguous-prefix fallback. Correct on every non-Ultra part.
    e_count = _sysctl_int("hw.perflevel1.logicalcpu")
    if e_count and 0 < e_count <= cpu_count:
        return tuple(i < e_count for i in range(cpu_count))
    return tuple(False for _ in range(cpu_count))


def _cluster_map_from_ioreg() -> tuple[bool, ...] | None:
    if not IS_DARWIN:
        return None
    try:
        out = subprocess.run(
            [_IOREG, "-lw0", "-p", "IODeviceTree", "-c", "IOPlatformDevice"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception as e:  # noqa: BLE001
        logger.debug(f"ioreg cluster probe failed: {e}")
        return None

    entries: dict[int, bool] = {}
    pending_id: int | None = None
    pending_type: bool | None = None
    for line in out.splitlines():
        m = re.search(r'"logical-cpu-id"\s*=\s*<?(\d+)', line)
        if m:
            pending_id = int(m.group(1))
        m = re.search(r'"cluster-type"\s*=\s*<"([EP])"', line)
        if m:
            pending_type = m.group(1) == "E"
        if pending_id is not None and pending_type is not None:
            entries[pending_id] = pending_type
            pending_id = None
            pending_type = None

    if not entries or set(entries) != set(range(len(entries))):
        return None
    return tuple(entries[i] for i in range(len(entries)))


def _sysctl_int(name: str) -> int | None:
    if not IS_DARWIN:
        return None
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


class _GPUReader:
    """IOAccelerator PerformanceStatistics via the IOKit registry."""

    def __init__(self) -> None:
        self._ok = False
        self._service = 0
        if not IS_DARWIN:
            return
        try:
            self._iokit = ctypes.CDLL(
                "/System/Library/Frameworks/IOKit.framework/IOKit"
            )
            self._cf = ctypes.CDLL(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )
            self._cf.CFStringCreateWithCString.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_uint32,
            ]
            self._cf.CFStringCreateWithCString.restype = ctypes.c_void_p
            self._cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self._cf.CFDictionaryGetValue.restype = ctypes.c_void_p
            self._cf.CFNumberGetValue.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            self._cf.CFNumberGetValue.restype = ctypes.c_bool
            self._cf.CFRelease.argtypes = [ctypes.c_void_p]
            self._iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
            self._iokit.IOServiceMatching.restype = ctypes.c_void_p
            self._iokit.IOServiceGetMatchingService.argtypes = [
                ctypes.c_uint,
                ctypes.c_void_p,
            ]
            self._iokit.IOServiceGetMatchingService.restype = ctypes.c_uint
            self._iokit.IORegistryEntryCreateCFProperty.argtypes = [
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            self._iokit.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
            self._iokit.IOObjectRelease.argtypes = [ctypes.c_uint]

            matching = self._iokit.IOServiceMatching(b"IOAccelerator")
            if not matching:
                return
            self._service = self._iokit.IOServiceGetMatchingService(0, matching)
            if not self._service:
                return
            # Cached: creating these per sample would dominate the sample cost.
            self._k_stats = self._cfstr("PerformanceStatistics")
            self._k_util = self._cfstr("Device Utilization %")
            self._ok = True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"GPU sampling unavailable: {e}")
            self._ok = False

    def _cfstr(self, value: str) -> ctypes.c_void_p:
        return self._cf.CFStringCreateWithCString(None, value.encode(), 0x08000100)

    def sample(self) -> float | None:
        if not self._ok:
            return None
        stats = None
        try:
            stats = self._iokit.IORegistryEntryCreateCFProperty(
                self._service, self._k_stats, None, 0
            )
            if not stats:
                return None
            num = self._cf.CFDictionaryGetValue(stats, self._k_util)
            if not num:
                return None
            out = ctypes.c_int64(0)
            if not self._cf.CFNumberGetValue(num, 4, ctypes.byref(out)):
                return None
            return max(0.0, min(1.0, out.value / 100.0))
        except Exception:  # noqa: BLE001
            return None
        finally:
            if stats:
                self._cf.CFRelease(stats)

    def close(self) -> None:
        if self._service:
            with contextlib.suppress(Exception):
                self._iokit.IOObjectRelease(self._service)
            self._service = 0
        self._ok = False


class _ThermalReader:
    """OSThermalPressureLevel via notify_get_state.

    Levels are 0 Nominal, 1 Moderate, 2 Heavy, 3 Trapping, 4 Sleeping. Note
    this is five-valued: Foundation's ProcessInfo.ThermalState collapses
    Trapping and Sleeping into `.critical`. The raw value is kept because a
    machine heading for a thermal sleep is exactly the run worth discarding.
    """

    def __init__(self) -> None:
        self._libc: ctypes.CDLL | None = None
        self._token = ctypes.c_int(0)
        if not IS_DARWIN:
            return
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            libc.notify_register_check.argtypes = [
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_int),
            ]
            libc.notify_register_check.restype = ctypes.c_uint32
            libc.notify_get_state.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint64),
            ]
            libc.notify_get_state.restype = ctypes.c_uint32
            libc.notify_cancel.argtypes = [ctypes.c_int]
            name = ctypes.c_char_p.in_dll(
                libc, "kOSThermalNotificationPressureLevelName"
            ).value
            if not name:
                return
            if libc.notify_register_check(name, ctypes.byref(self._token)) != 0:
                return
            self._libc = libc
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Thermal sampling unavailable: {e}")
            self._libc = None

    def sample(self) -> int | None:
        if self._libc is None:
            return None
        state = ctypes.c_uint64(0)
        if self._libc.notify_get_state(self._token, ctypes.byref(state)) != 0:
            return None
        return int(state.value)

    def close(self) -> None:
        if self._libc is not None:
            with contextlib.suppress(Exception):
                self._libc.notify_cancel(self._token)
            self._libc = None


class SystemSampler:
    """Background 1 Hz sampler with a rolling window.

    Runs on a plain thread rather than an asyncio task: the benchmark's hot
    path is a per-token `await` loop whose inter-token timing is what is being
    measured, and scheduling a coroutine into that loop every second would
    perturb it. The ctypes calls here release the GIL and never touch the event
    loop.

    Sampling is continuous for the whole run rather than restarted per test,
    because CPU utilization is a delta between two cumulative readings — a
    per-test sampler throws away its first sample every time.
    """

    def __init__(self, interval_s: float = _DEFAULT_INTERVAL_S) -> None:
        self.interval_s = interval_s
        self._samples: deque[_Sample] = deque(maxlen=_MAX_SAMPLES)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._cpu = _CPUReader()
        self._gpu = _GPUReader()
        self._thermal = _ThermalReader()
        self._total_ram = get_total_memory() or 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        # Prime the CPU deltas so the first recorded sample is already valid.
        self._cpu.sample()
        self._thread = threading.Thread(
            target=self._run, name="omlx-system-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=self.interval_s * 2 + 1.0)
        self._gpu.close()
        self._thermal.close()

    def _run(self) -> None:
        # Event.wait rather than sleep so stop() returns without waiting out
        # the remainder of the current tick.
        while not self._stop.wait(self.interval_s):
            try:
                sample = self._collect()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"System sample failed: {e}")
                continue
            with self._lock:
                self._samples.append(sample)

    def _collect(self) -> _Sample:
        cpu_total, cpu_p, cpu_e = self._cpu.sample()
        mlx_active = 0
        mlx_cache = 0
        if HAS_MLX:
            try:
                # Reads a process-global allocator counter. Deliberately never
                # calls reset_peak_memory() — that is also process-global and
                # would zero the benchmark's own peak mid-test.
                mlx_active = int(mx.get_active_memory())
                mlx_cache = int(mx.get_cache_memory())
            except Exception:  # noqa: BLE001
                pass
        sys_used = 0
        sys_wired = 0
        vm = get_macos_vm_stats()
        if vm:
            available = vm.get("free", 0) + vm.get("inactive", 0)
            # Matches psutil_compat's `used` definition (total - available), the
            # one the memory guard and dashboard already use. The menubar's
            # wired+active+compressed sum is a different number entirely.
            sys_used = max(0, self._total_ram - available) if self._total_ram else 0
            sys_wired = vm.get("wired", 0)
        return _Sample(
            t=time.monotonic(),
            cpu_total=cpu_total,
            cpu_p=cpu_p,
            cpu_e=cpu_e,
            gpu_util=self._gpu.sample(),
            thermal=self._thermal.sample(),
            phys_footprint=get_phys_footprint(),
            mlx_active=mlx_active,
            mlx_cache=mlx_cache,
            sys_used=sys_used,
            sys_wired=sys_wired,
        )

    def window(self, t0: float, t1: float) -> dict | None:
        """Aggregate samples in [t0, t1] into the upload payload shape.

        Returns None when no sample landed in the window, so the caller
        uploads null rather than a zero-filled object that would be averaged
        in as real data downstream.
        """
        with self._lock:
            selected = [s for s in self._samples if t0 <= s.t <= t1]
        return aggregate(selected, self.interval_s)

    def run_peak_footprint(self) -> int:
        with self._lock:
            return max((s.phys_footprint for s in self._samples), default=0)


def _gib(value: int) -> float:
    return round(value / (1024**3), 2)


def _pct(values: list[float], op) -> float | None:
    return round(op(values) * 100.0, 1) if values else None


def aggregate(samples: list[_Sample], interval_s: float) -> dict | None:
    """Reduce samples to averages and maxima. Pure — no OS calls.

    A single sample is enough: each one already carries a CPU delta measured
    over the preceding interval, so it is real data rather than a snapshot.
    An empty window returns None so the caller uploads null instead of a
    zero-filled object the leaderboard would average in as a measurement.
    """
    if not samples:
        return None

    def present(attr: str) -> list[float]:
        return [
            getattr(s, attr) for s in samples if getattr(s, attr) is not None
        ]

    cpu_total = present("cpu_total")
    cpu_p = present("cpu_p")
    cpu_e = present("cpu_e")
    gpu = present("gpu_util")
    thermal = [s.thermal for s in samples if s.thermal is not None]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    result: dict = {
        "sample_count": len(samples),
        "interval_s": interval_s,
        "cpu": {
            "total_avg": _pct(cpu_total, mean),
            "total_max": _pct(cpu_total, max),
            "p_avg": _pct(cpu_p, mean),
            "e_avg": _pct(cpu_e, mean),
        },
        "memory": {
            "phys_footprint_peak": _gib(max(s.phys_footprint for s in samples)),
            "mlx_active_peak": _gib(max(s.mlx_active for s in samples)),
            "mlx_cache_peak": _gib(max(s.mlx_cache for s in samples)),
            "system_used_peak": _gib(max(s.sys_used for s in samples)),
            "system_wired_peak": _gib(max(s.sys_wired for s in samples)),
            "total_ram": round((get_total_memory() or 0) / (1024**3)),
        },
    }
    if gpu:
        result["gpu"] = {
            "util_avg": _pct(gpu, mean),
            "util_max": _pct(gpu, max),
        }
    if thermal:
        result["thermal"] = {"start": thermal[0], "max": max(thermal)}
    return result
