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

_libproc: ctypes.CDLL | None = None
_proc_pid_rusage = None

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
