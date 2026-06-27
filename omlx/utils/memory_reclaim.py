# SPDX-License-Identifier: Apache-2.0
"""Best-effort process memory reclamation helpers."""

from __future__ import annotations

import ctypes
import logging
import sys
import threading

logger = logging.getLogger(__name__)
_PRESSURE_RELIEF_GOAL = ctypes.c_size_t(-1).value

_resolve_lock = threading.Lock()
_resolved = False
_pressure_relief_fn = None
_default_zone_fn = None
_get_all_zones_fn = None
_mach_task_self = 0


def _resolve_macos_malloc_fns():
    """Resolve macOS malloc pressure-relief functions once."""
    global _resolved, _pressure_relief_fn, _default_zone_fn
    global _get_all_zones_fn, _mach_task_self
    if _resolved:
        return _default_zone_fn, _pressure_relief_fn, _get_all_zones_fn
    with _resolve_lock:
        if _resolved:
            return _default_zone_fn, _pressure_relief_fn
        _resolved = True
        if sys.platform != "darwin":
            return None, None, None
        try:
            lib = ctypes.CDLL(None)
            default_zone = lib.malloc_default_zone
            default_zone.argtypes = []
            default_zone.restype = ctypes.c_void_p

            pressure_relief = lib.malloc_zone_pressure_relief
            pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            pressure_relief.restype = ctypes.c_size_t

            get_all_zones = lib.malloc_get_all_zones
            get_all_zones.argtypes = [
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
                ctypes.POINTER(ctypes.c_uint),
            ]
            get_all_zones.restype = ctypes.c_int

            _default_zone_fn = default_zone
            _pressure_relief_fn = pressure_relief
            _get_all_zones_fn = get_all_zones
            try:
                _mach_task_self = ctypes.c_uint.in_dll(lib, "mach_task_self_").value
            except (TypeError, ValueError, AttributeError):
                _mach_task_self = 0
        except (AttributeError, OSError) as e:
            logger.debug("macOS malloc pressure relief unavailable: %s", e)
            _default_zone_fn = None
            _pressure_relief_fn = None
            _get_all_zones_fn = None
        return _default_zone_fn, _pressure_relief_fn, _get_all_zones_fn


def _malloc_zones() -> list[int]:
    default_zone_fn, _, get_all_zones_fn = _resolve_macos_malloc_fns()
    zones: list[int] = []

    if get_all_zones_fn is not None and _mach_task_self:
        zone_ptr = ctypes.POINTER(ctypes.c_void_p)()
        count = ctypes.c_uint(0)
        try:
            result = get_all_zones_fn(
                _mach_task_self, None, ctypes.byref(zone_ptr), ctypes.byref(count)
            )
            if result == 0:
                for idx in range(count.value):
                    zone = zone_ptr[idx]
                    if zone:
                        zones.append(int(zone))
        except Exception as e:
            logger.debug("macOS malloc zone enumeration failed: %s", e)

    if not zones and default_zone_fn is not None:
        try:
            zone = default_zone_fn()
            if zone:
                zones.append(int(zone))
        except Exception as e:
            logger.debug("macOS default malloc zone lookup failed: %s", e)

    return list(dict.fromkeys(zones))


def release_free_malloc_pages() -> int:
    """Ask macOS malloc to return free pages from the default zone.

    This does not free live Python objects. It only pressures malloc to release
    empty pages it is already holding after large transient model loads.
    Non-macOS platforms and unsupported runtimes return 0.
    """
    _, pressure_relief_fn, _ = _resolve_macos_malloc_fns()
    if pressure_relief_fn is None:
        return 0
    released = 0
    for zone in _malloc_zones():
        try:
            for _ in range(2):
                released += int(pressure_relief_fn(zone, _PRESSURE_RELIEF_GOAL))
        except Exception as e:
            logger.debug("macOS malloc pressure relief failed: %s", e)
    return released
