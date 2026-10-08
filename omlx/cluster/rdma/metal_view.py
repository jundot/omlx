# SPDX-License-Identifier: Apache-2.0
"""MLX arrays of mailbox memory on a Mac: a Metal buffer over the reply half, imported through DLPack."""

from __future__ import annotations

import ctypes
import threading
from typing import Any

import numpy as np

# DLPack device and dtype codes.
_METAL = 8
_UINT = 1
# Frames at least this large move by GPU copy. Measured on Apple Silicon, a CPU copy of 28 MiB takes as long,
# and smaller frames copy faster on the CPU than a GPU round trip (0.02 ms against 0.19 ms at 1 MiB).
MIN_BYTES = 32 << 20


class _Device(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int32)]


class _DataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _Tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _Device),
        ("ndim", ctypes.c_int32),
        ("dtype", _DataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _Managed(ctypes.Structure):
    pass


_Deleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_Managed))
_Managed._fields_ = [
    ("dl_tensor", _Tensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _Deleter),
]
# Exports MLX still holds; its deleter call drops them.
_live: dict[int, Any] = {}
_live_lock = threading.Lock()


@_Deleter
def _release(managed: Any) -> None:
    with _live_lock:
        _live.pop(ctypes.addressof(managed.contents), None)


class _Export:
    """One DLPack export of `count` bytes at `offset` in a Metal buffer."""

    def __init__(self, buffer: int, offset: int, count: int) -> None:
        shape = (ctypes.c_int64 * 1)(count)
        managed = _Managed()
        managed.dl_tensor = _Tensor(
            buffer, _Device(_METAL, 0), 1, _DataType(_UINT, 8, 1), shape, None, offset
        )
        managed.manager_ctx = None
        managed.deleter = _release
        with _live_lock:
            _live[ctypes.addressof(managed)] = (managed, shape)
        self._address = ctypes.addressof(managed)

    def __dlpack_device__(self) -> tuple[int, int]:
        return (_METAL, 0)

    def __dlpack__(self, **_: Any) -> Any:
        new = ctypes.pythonapi.PyCapsule_New
        new.restype = ctypes.py_object
        new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        return new(self._address, b"dltensor", None)


class MetalWindow:
    """A no-copy Metal buffer over mailbox memory; its views are MLX arrays of that same memory."""

    def __init__(self, library: Any, buffer: int, address: int, length: int) -> None:
        self._library = library
        self._buffer = buffer
        self.address = address
        self.length = length

    @classmethod
    def open(
        cls, mx: Any, library: Any, address: int, length: int
    ) -> MetalWindow | None:
        """A window over [address, address + length), or None when the helper or MLX cannot share it."""
        wrap = getattr(library, "mcdma_rpc_metal_wrap", None)
        if wrap is None:
            return None
        wrap.restype = ctypes.c_void_p
        wrap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        library.mcdma_rpc_metal_release.argtypes = [ctypes.c_void_p]
        buffer = wrap(address, length)
        if not buffer:
            return None
        window = cls(library, buffer, address, length)
        try:
            probe = window.view(mx, 0, 16)
            shared = int(np.asarray(probe).ctypes.data) == address
        except Exception:
            shared = False
        if not shared:
            # MLX copied instead of sharing, so a view would read stale bytes.
            window.close()
            return None
        return window

    def view(self, mx: Any, offset: int, count: int) -> Any:
        """The uint8 MLX array of bytes [offset, offset + count), sharing the mailbox memory."""
        if offset < 0 or offset + count > self.length:
            raise ValueError("view lies outside the window")
        return mx.from_dlpack(_Export(self._buffer, offset, count), copy=False)

    def copy(self, mx: Any, offset: int, count: int, dtype: Any = None) -> Any:
        """A GPU copy of bytes [offset, offset + count), as `dtype` elements if given, finished on return."""
        view = self.view(mx, offset, count)
        # Materialized here, so nothing lazy is left tied to this thread's stream.
        copied = (view.view(dtype) if dtype is not None else view) * 1
        mx.eval(copied)
        return copied

    def close(self) -> None:
        if self._buffer:
            self._library.mcdma_rpc_metal_release(self._buffer)
            self._buffer = 0
