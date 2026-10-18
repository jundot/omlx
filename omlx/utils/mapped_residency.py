# SPDX-License-Identifier: Apache-2.0
"""Fresh, conservative residency of live read-only checkpoint views on macOS.

VM_REGION_TOP_INFO reads object counters rather than walking every page. Only
whole-file mappings without COW/shadow objects qualify. Partial/unknown regions
receive zero credit. Weak exporters track sanitization/unload without retaining
weights; aliases of the same file share a single physical residency budget.
"""

import ctypes
import os
import threading
import weakref

_views = weakref.WeakValueDictionary()
_lock = threading.Lock()
_PAGE = os.sysconf("SC_PAGE_SIZE")
_lib = None


class _TopInfo(ctypes.Structure):
    _fields_ = [
        ("object_id", ctypes.c_uint32),
        ("ref_count", ctypes.c_uint32),
        ("private_pages", ctypes.c_uint32),
        ("shared_pages", ctypes.c_uint32),
        ("share_mode", ctypes.c_uint8),
    ]


def track(view, mapping, stat, lo, hi):
    """Register only a successful foreign-buffer import, never copied weights."""
    if mapping.offset != 0 or mapping.nbytes != stat.st_size:
        return  # object counters cannot attribute a partial-file mapping
    key = (stat.st_dev, stat.st_ino, mapping.nbytes, mapping.ctypes.data, lo, hi)
    with _lock:
        _views[key] = view


def _resident_file_bytes(address, length):
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(None)
    task = ctypes.c_uint32.in_dll(_lib, "mach_task_self_")
    start, size = ctypes.c_uint64(address), ctypes.c_uint64()
    info, count, obj = (
        _TopInfo(),
        ctypes.c_uint32(ctypes.sizeof(_TopInfo) // 4),
        ctypes.c_uint32(),
    )
    result = _lib.mach_vm_region(
        task,
        ctypes.byref(start),
        ctypes.byref(size),
        12,
        ctypes.byref(info),
        ctypes.byref(count),
        ctypes.byref(obj),
    )
    if obj.value:
        _lib.mach_port_deallocate(task, obj)
    extent = (length + _PAGE - 1) // _PAGE * _PAGE
    # TOP_INFO counts the whole backing object, not an arbitrary subrange.
    # An exact whole-file VM entry and a non-COW object are required.
    if (
        result
        or start.value != address
        or size.value != extent
        or info.share_mode not in (2, 4, 5)
    ):
        return 0
    resident = (info.private_pages + info.shared_pages) * _PAGE
    return resident if resident <= extent else 0


def resident_mapped_bytes():
    """Lower bound on current resident bytes backing live imported tensors.

    Subtract *all* non-live bytes from object residency, conservatively assuming
    they are resident first. This avoids page walks and per-tensor syscalls.
    No cache, stale sample, pinned-time heuristic or logical-byte credit is used.
    """
    with _lock:
        snapshot = list(_views.items())
    # Hold exporters across the native calls, which can release the GIL.
    files = {}
    for (device, inode, size, address, lo, hi), _view in snapshot:
        entries = files.setdefault((device, inode), [])
        entries.append((size, address, lo, hi, _view._mmap))
    total = 0
    for entries in files.values():
        sizes = {entry[0] for entry in entries}
        if len(sizes) != 1:
            continue  # checkpoint changed size while mapped
        size = sizes.pop()
        try:
            if entries[0][4].size() != size:
                continue  # object counters may include a newly appended tail
        except (OSError, ValueError):
            continue
        covered, end = 0, 0
        for lo, hi in sorted((entry[2], entry[3]) for entry in entries):
            covered += max(0, hi - max(lo, end))
            end = max(end, hi)
        resident = 0
        for address in {entry[1] for entry in entries}:
            try:
                resident = _resident_file_bytes(address, size)
            except (AttributeError, OSError, ValueError):
                resident = 0
            if resident:
                break
        extent = (size + _PAGE - 1) // _PAGE * _PAGE
        total += max(0, resident - (extent - covered))
    return total
