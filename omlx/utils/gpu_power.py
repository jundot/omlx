"""System-wide GPU power cap for capped generation requests.

The AGXAccelerator registry entry takes ``SetMaxGPUAbsolutePower`` with an
``AbsoluteTarget`` in milliwatts from an unprivileged process, and the firmware
holds GPU power under that target. The cap is not a public API, affects the
whole Mac, and stays set until it is removed. m5gpu
(https://github.com/beamivalice/m5gpu) documented the property names and the
-1000 value that removes the cap. So:

- The limiter holds the lowest cap among active requests and removes it
  ``release_delay_s`` after the last one ends.
- A marker file holding the server PID exists while a cap is set, so the next
  start (or the macOS app, after a crash) can remove a cap that a killed
  server left behind.
- The macOS thermal controller writes its own limit into the same property
  while the GPU is hot. A held cap is checked and rewritten when that limit
  is higher than ours; a lower one stays.
"""

import asyncio
import contextlib
import ctypes
import logging
import os
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_WATTS = 2
MAX_WATTS = 1000
MARKER_NAME = "gpu_power_cap.active"

# Removes our cap. The system then publishes its own limit again.
_UNCAPPED_MW = -1000
_MAX_REWRITE_STEPS = 3
_K_CF_NUMBER_SINT64 = 4
_K_CF_STRING_UTF8 = 0x08000100


class AGXPowerCap:
    """Read and write ``MaxGPUAbsolutePower`` on the AGXAccelerator entry."""

    def __init__(self) -> None:
        self.supported = False
        self._service = 0
        if sys.platform != "darwin":
            return
        try:
            iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
            cf = ctypes.CDLL(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )
            vp = ctypes.c_void_p
            cf.CFStringCreateWithCString.argtypes = [
                vp,
                ctypes.c_char_p,
                ctypes.c_uint32,
            ]
            cf.CFStringCreateWithCString.restype = vp
            cf.CFNumberCreate.argtypes = [vp, ctypes.c_int, vp]
            cf.CFNumberCreate.restype = vp
            cf.CFNumberGetValue.argtypes = [vp, ctypes.c_int, vp]
            cf.CFNumberGetValue.restype = ctypes.c_bool
            cf.CFDictionaryCreateMutable.argtypes = [vp, ctypes.c_long, vp, vp]
            cf.CFDictionaryCreateMutable.restype = vp
            cf.CFDictionarySetValue.argtypes = [vp, vp, vp]
            cf.CFRelease.argtypes = [vp]
            iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
            iokit.IOServiceMatching.restype = vp
            iokit.IOServiceGetMatchingService.argtypes = [ctypes.c_uint, vp]
            iokit.IOServiceGetMatchingService.restype = ctypes.c_uint
            iokit.IORegistryEntryCreateCFProperty.argtypes = [
                ctypes.c_uint,
                vp,
                vp,
                ctypes.c_uint32,
            ]
            iokit.IORegistryEntryCreateCFProperty.restype = vp
            iokit.IORegistryEntrySetCFProperties.argtypes = [ctypes.c_uint, vp]
            iokit.IORegistryEntrySetCFProperties.restype = ctypes.c_int
            self._iokit, self._cf = iokit, cf

            matching = iokit.IOServiceMatching(b"AGXAccelerator")
            self._service = (
                iokit.IOServiceGetMatchingService(0, matching) if matching else 0
            )
            if not self._service:
                return
            # Created once; they live as long as the process.
            self._k_max = self._cfstr("MaxGPUAbsolutePower")
            self._k_set = self._cfstr("SetMaxGPUAbsolutePower")
            self._k_target = self._cfstr("AbsoluteTarget")
            self._true = vp.in_dll(cf, "kCFBooleanTrue")
            self._key_callbacks = ctypes.addressof(
                ctypes.c_char.in_dll(cf, "kCFTypeDictionaryKeyCallBacks")
            )
            self._value_callbacks = ctypes.addressof(
                ctypes.c_char.in_dll(cf, "kCFTypeDictionaryValueCallBacks")
            )
            self.supported = self.read_mw() is not None
        except Exception as e:  # noqa: BLE001
            logger.debug(f"GPU power cap unavailable: {e}")
            self.supported = False

    def _cfstr(self, value: str) -> int:
        return self._cf.CFStringCreateWithCString(
            None, value.encode(), _K_CF_STRING_UTF8
        )

    def read_mw(self) -> int | None:
        ref = self._iokit.IORegistryEntryCreateCFProperty(
            self._service, self._k_max, None, 0
        )
        if not ref:
            return None
        try:
            out = ctypes.c_int64(0)
            if not self._cf.CFNumberGetValue(
                ref, _K_CF_NUMBER_SINT64, ctypes.byref(out)
            ):
                return None
            return out.value
        finally:
            self._cf.CFRelease(ref)

    def write_mw(self, milliwatts: int) -> bool:
        props = self._cf.CFDictionaryCreateMutable(
            None, 0, self._key_callbacks, self._value_callbacks
        )
        value = ctypes.c_int64(milliwatts)
        number = self._cf.CFNumberCreate(None, _K_CF_NUMBER_SINT64, ctypes.byref(value))
        try:
            self._cf.CFDictionarySetValue(props, self._k_set, self._true)
            self._cf.CFDictionarySetValue(props, self._k_target, number)
            return self._iokit.IORegistryEntrySetCFProperties(self._service, props) == 0
        finally:
            self._cf.CFRelease(number)
            self._cf.CFRelease(props)


_agx: AGXPowerCap | None = None


def _get_agx() -> AGXPowerCap:
    global _agx
    if _agx is None:
        _agx = AGXPowerCap()
    return _agx


def gpu_power_cap_supported() -> bool:
    return _get_agx().supported


def rewrite_delay_steps(misses: int) -> int:
    """Check intervals to wait after ``misses`` consecutive overwrites."""
    return min(max(misses, 1), _MAX_REWRITE_STEPS)


class GpuPowerLimiter:
    """Hold the lowest requested GPU power cap while capped requests run."""

    def __init__(
        self,
        base_path: Path,
        cap=None,
        release_delay_s: float = 15.0,
        check_interval_s: float = 1.0,
    ) -> None:
        self._cap = cap if cap is not None else _get_agx()
        self._marker = Path(base_path) / MARKER_NAME
        self._release_delay_s = release_delay_s
        self._check_interval_s = check_interval_s
        self._active: Counter[int] = Counter()
        self._target_mw: int | None = None
        self._release_handle: asyncio.TimerHandle | None = None
        self._hold_task: asyncio.Task | None = None
        self._warned = False

    @property
    def target_watts(self) -> int | None:
        return None if self._target_mw is None else self._target_mw // 1000

    def recover(self) -> None:
        """Remove a cap that a killed server on this base path left behind."""
        try:
            owner = int(self._marker.read_text() or 0)
        except FileNotFoundError:
            return
        except (OSError, ValueError):
            owner = 0
        # The marker holds the owner's PID; a live owner still holds its cap.
        if owner and owner != os.getpid() and _pid_alive(owner):
            return
        if self._cap.supported:
            if not self._cap.write_mw(_UNCAPPED_MW):
                return
            logger.info("Removed a GPU power cap left by a previous server run")
        self._unmark()

    def acquire(self, watts: int) -> None:
        self._active[watts] += 1
        if self._release_handle is not None:
            self._release_handle.cancel()
            self._release_handle = None
        self._apply(min(self._active))

    def release(self, watts: int) -> None:
        self._active[watts] -= 1
        if self._active[watts] <= 0:
            del self._active[watts]
        if self._active:
            self._apply(min(self._active))
        elif self._target_mw is not None and self._release_handle is None:
            self._release_handle = asyncio.get_running_loop().call_later(
                self._release_delay_s, self._clear
            )

    def drop_idle_cap(self) -> None:
        """Remove a cap that is only waiting out the release delay."""
        if self._active or self._release_handle is None:
            return
        self._release_handle.cancel()
        self._clear()

    async def close(self) -> None:
        if self._release_handle is not None:
            self._release_handle.cancel()
            self._release_handle = None
        hold_task = self._hold_task
        self._active.clear()
        self._clear()
        if hold_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await hold_task

    def _apply(self, watts: int) -> None:
        if not self._cap.supported:
            if not self._warned:
                self._warned = True
                logger.warning(
                    "GPU power limit requested, but this Mac's GPU does not "
                    "accept a power cap; generating without it"
                )
            return
        target_mw = watts * 1000
        if target_mw == self._target_mw:
            return
        # Without the marker a crash could leave the cap set, so do not cap.
        if self._target_mw is None and not self._mark():
            return
        if not self._cap.write_mw(target_mw):
            self._cap.supported = False
            # Keep the marker while an earlier cap is still in place.
            if self._target_mw is None:
                self._unmark()
            logger.warning("GPU power cap write failed; disabling GPU power limits")
            return
        self._target_mw = target_mw
        logger.info(f"GPU power cap set to {watts} W")
        if self._hold_task is None:
            self._hold_task = asyncio.get_running_loop().create_task(self._hold())

    def _clear(self) -> None:
        self._release_handle = None
        if self._hold_task is not None:
            self._hold_task.cancel()
            self._hold_task = None
        if self._target_mw is None:
            return
        self._target_mw = None
        # On failure the marker stays so the next start retries the removal.
        if self._cap.write_mw(_UNCAPPED_MW):
            self._unmark()
            logger.info("GPU power cap removed")
        else:
            logger.warning("GPU power cap removal failed; the next start retries it")

    def _mark(self) -> bool:
        try:
            self._marker.parent.mkdir(parents=True, exist_ok=True)
            self._marker.write_text(str(os.getpid()))
        except OSError as e:
            logger.warning(f"Cannot write {self._marker}; not capping GPU power: {e}")
            return False
        return True

    def _unmark(self) -> None:
        with contextlib.suppress(OSError):
            self._marker.unlink(missing_ok=True)

    async def _hold(self) -> None:
        misses = 0
        while self._target_mw is not None:
            await asyncio.sleep(rewrite_delay_steps(misses) * self._check_interval_s)
            if self._target_mw is None:
                return
            misses = self._check(misses)

    def _check(self, misses: int) -> int:
        """Rewrite the cap if something raised or removed it.

        Returns the number of consecutive rewrites. A lower value is a
        thermal limit and stays in place.
        """
        current = self._cap.read_mw()
        if current is not None and 0 <= current <= self._target_mw:
            return 0
        self._cap.write_mw(self._target_mw)
        logger.debug(f"GPU power cap replaced by {current}; rewrote {self._target_mw}")
        return misses + 1


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
