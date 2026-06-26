# SPDX-License-Identifier: Apache-2.0
"""Best-effort process memory reclamation helpers."""

from __future__ import annotations

import ctypes
import logging
import sys
import threading

logger = logging.getLogger(__name__)

_resolve_lock = threading.Lock()
_resolved = False
_pressure_relief_fn = None
_default_zone_fn = None


def _resolve_macos_malloc_fns():
    """Resolve macOS malloc pressure-relief functions once."""
    global _resolved, _pressure_relief_fn, _default_zone_fn
    if _resolved:
        return _default_zone_fn, _pressure_relief_fn
    with _resolve_lock:
        if _resolved:
            return _default_zone_fn, _pressure_relief_fn
        _resolved = True
        if sys.platform != "darwin":
            return None, None
        try:
            lib = ctypes.CDLL(None)
            default_zone = lib.malloc_default_zone
            default_zone.argtypes = []
            default_zone.restype = ctypes.c_void_p

            pressure_relief = lib.malloc_zone_pressure_relief
            pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            pressure_relief.restype = ctypes.c_size_t

            _default_zone_fn = default_zone
            _pressure_relief_fn = pressure_relief
        except (AttributeError, OSError) as e:
            logger.debug("macOS malloc pressure relief unavailable: %s", e)
            _default_zone_fn = None
            _pressure_relief_fn = None
        return _default_zone_fn, _pressure_relief_fn


def release_free_malloc_pages() -> int:
    """Ask macOS malloc to return free pages from the default zone.

    This does not free live Python objects. It only pressures malloc to release
    empty pages it is already holding after large transient model loads.
    Non-macOS platforms and unsupported runtimes return 0.
    """
    default_zone_fn, pressure_relief_fn = _resolve_macos_malloc_fns()
    if default_zone_fn is None or pressure_relief_fn is None:
        return 0
    try:
        zone = default_zone_fn()
        if not zone:
            return 0
        return int(pressure_relief_fn(zone, 0))
    except Exception as e:
        logger.debug("macOS malloc pressure relief failed: %s", e)
        return 0
