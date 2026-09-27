# SPDX-License-Identifier: Apache-2.0
"""Apple Silicon process memory measurement via macOS phys_footprint ledger.

macOS jetsam compares against `phys_footprint` (per-process kernel ledger),
not `task_basic_info.resident_size`. psutil and similar tools use the latter,
which can underreport IOAccelerator-backed (Metal) memory on Apple Silicon
UMA systems. This module exposes `get_phys_footprint()` which returns the
exact value jetsam sees.

References:
- xnu kernel `bsd/kern/kern_memorystatus.c` uses phys_footprint ledger
- iOS Xcode memory gauge matches `vmInfo.phys_footprint` and includes
  Metal textures (Mozilla bugzilla 1786860)
- `proc_pid_rusage` with RUSAGE_INFO_V4 returns `rusage_info_v4` with
  `ri_phys_footprint` field
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys

logger = logging.getLogger(__name__)


# rusage_info_v4 layout from /usr/include/sys/resource.h.
# ri_phys_footprint is the field we care about (kernel ledger of physical
# memory pressure — includes anonymous, dirty file-backed, and IOAccelerator
# allocations).
class _RusageInfoV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


_RUSAGE_INFO_V4 = 4


# task_vm_info from /usr/include/mach/task_info.h (rev7). Metal buffers are
# charged to ledger_tag_graphics_footprint, so phys_footprint minus that tag
# is the CPU-side footprint. The kernel fills only whole revisions that fit
# the caller's count, so the struct must span a complete revision.
class _TaskVMInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("region_count", ctypes.c_int32),
        ("page_size", ctypes.c_int32),
        *[
            (name, ctypes.c_uint64)
            for name in (
                "resident_size",
                "resident_size_peak",
                "device",
                "device_peak",
                "internal",
                "internal_peak",
                "external",
                "external_peak",
                "reusable",
                "reusable_peak",
                "purgeable_volatile_pmap",
                "purgeable_volatile_resident",
                "purgeable_volatile_virtual",
                "compressed",
                "compressed_peak",
                "compressed_lifetime",
                "phys_footprint",
                "min_address",
                "max_address",
            )
        ],
        *[
            (name, ctypes.c_int64)
            for name in (
                "ledger_phys_footprint_peak",
                "ledger_purgeable_nonvolatile",
                "ledger_purgeable_novolatile_compressed",
                "ledger_purgeable_volatile",
                "ledger_purgeable_volatile_compressed",
                "ledger_tag_network_nonvolatile",
                "ledger_tag_network_nonvolatile_compressed",
                "ledger_tag_network_volatile",
                "ledger_tag_network_volatile_compressed",
                "ledger_tag_media_footprint",
                "ledger_tag_media_footprint_compressed",
                "ledger_tag_media_nofootprint",
                "ledger_tag_media_nofootprint_compressed",
                "ledger_tag_graphics_footprint",
                "ledger_tag_graphics_footprint_compressed",
                "ledger_tag_graphics_nofootprint",
                "ledger_tag_graphics_nofootprint_compressed",
                "ledger_tag_neural_footprint",
                "ledger_tag_neural_footprint_compressed",
                "ledger_tag_neural_nofootprint",
                "ledger_tag_neural_nofootprint_compressed",
            )
        ],
        ("limit_bytes_remaining", ctypes.c_uint64),
        ("decompressions", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
        ("ledger_swapins", ctypes.c_int64),
        ("ledger_tag_neural_nofootprint_total", ctypes.c_int64),
        ("ledger_tag_neural_nofootprint_peak", ctypes.c_int64),
    ]


_TASK_VM_INFO = 22
_TASK_VM_INFO_COUNT = ctypes.sizeof(_TaskVMInfo) // 4
# Smallest reply that includes the graphics ledger (end of rev3).
_TASK_VM_INFO_GRAPHICS_COUNT = (
    _TaskVMInfo.ledger_tag_neural_nofootprint_compressed.offset + 8
) // 4

_libproc: ctypes.CDLL | None = None
_proc_pid_rusage = None
_task_info = None
_mach_task_self: ctypes.c_uint | None = None

if sys.platform == "darwin":
    try:
        _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _proc_pid_rusage = _libproc.proc_pid_rusage
        _proc_pid_rusage.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        _proc_pid_rusage.restype = ctypes.c_int
    except OSError as e:
        logger.warning(f"libproc unavailable, phys_footprint will return 0: {e}")
        _libproc = None
        _proc_pid_rusage = None
    try:
        _libc = ctypes.CDLL("/usr/lib/libc.dylib")
        _task_info = _libc.task_info
        _task_info.argtypes = [
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint),
        ]
        _task_info.restype = ctypes.c_int
        _mach_task_self = ctypes.c_uint.in_dll(_libc, "mach_task_self_")
    except (OSError, ValueError) as e:
        logger.debug(f"task_info unavailable, graphics footprint will return 0: {e}")
        _task_info = None
        _mach_task_self = None


def get_phys_footprint(pid: int | None = None) -> int:
    """Return process phys_footprint in bytes.

    phys_footprint is the macOS kernel's per-process ledger of physical
    memory pressure. It includes anonymous memory, dirty file-backed pages,
    and IOAccelerator-backed (Metal) allocations on Apple Silicon. This is
    the metric jetsam compares against — the authoritative number for
    memory-pressure decisions.

    Args:
        pid: Process ID to query. Defaults to current process.

    Returns:
        Bytes of phys_footprint. Returns 0 on non-Darwin platforms or if
        the libproc call fails (so callers can safely use
        `max(active, get_phys_footprint())`).
    """
    if _proc_pid_rusage is None:
        return 0
    info = _RusageInfoV4()
    target_pid = pid if pid is not None else os.getpid()
    rc = _proc_pid_rusage(target_pid, _RUSAGE_INFO_V4, ctypes.byref(info))
    if rc != 0:
        return 0
    return info.ri_phys_footprint


def get_lifetime_max_phys_footprint(pid: int | None = None) -> int:
    """Return the highest phys_footprint the process has ever reached, in bytes.

    This is a high-water mark since process start, so it does not fall when
    memory is released. Callers measuring a single episode must snapshot it
    before and after and treat an unchanged value as "this episode did not set
    a new maximum" rather than as the episode's peak.

    Args:
        pid: Process ID to query. Defaults to current process.

    Returns:
        Bytes. Returns 0 on non-Darwin platforms or if the libproc call fails.
    """
    if _proc_pid_rusage is None:
        return 0
    info = _RusageInfoV4()
    target_pid = pid if pid is not None else os.getpid()
    rc = _proc_pid_rusage(target_pid, _RUSAGE_INFO_V4, ctypes.byref(info))
    if rc != 0:
        return 0
    return info.ri_lifetime_max_phys_footprint


def get_graphics_footprint() -> int:
    """Return this process's graphics (Metal/IOGPU) footprint in bytes.

    MLX buffers are charged to this ledger, so it tracks MLX active + cache
    plus other Metal allocations. Like phys_footprint, it drops only after
    the driver finishes releasing freed buffers (0.1-0.3s on macOS 27).

    Returns:
        Bytes, or 0 on non-Darwin platforms, older kernels without the rev3
        ledger fields, or if the task_info call fails.
    """
    if _task_info is None or _mach_task_self is None:
        return 0
    info = _TaskVMInfo()
    count = ctypes.c_uint(_TASK_VM_INFO_COUNT)
    rc = _task_info(
        _mach_task_self.value, _TASK_VM_INFO, ctypes.byref(info), ctypes.byref(count)
    )
    if rc != 0 or count.value < _TASK_VM_INFO_GRAPHICS_COUNT:
        return 0
    return max(0, int(info.ledger_tag_graphics_footprint))
