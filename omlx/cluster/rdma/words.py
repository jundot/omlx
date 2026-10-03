# SPDX-License-Identifier: Apache-2.0
"""Ordered loads and stores of mailbox words through MCDMA's libmcdma-rpc helper."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Protocol

LIBRARY_ENV = "OMLX_MCDMA_RPC_LIBRARY"
ABI_VERSION = 1
# Where MCDMA's installers put the helper on macOS and Linux.
_DEFAULT_LIBRARIES = (
    "/usr/local/lib/libmcdma-rpc.dylib",
    "/usr/local/lib/libmcdma-rpc.so",
    "/usr/lib/libmcdma-rpc.so",
)
# A waiter spins this long before backing off to short sleeps.
_SPIN_NS = 2_000_000


class WordOps(Protocol):
    """Acquire-ordered waits and release-ordered stores on 8-byte mailbox words."""

    def wait(self, address: int, seq: int, *, equal: bool, timeout_s: float) -> int:
        """Return the word once its sequence matches, or 0 after `timeout_s`."""
        ...

    def store(self, address: int, value: int) -> None:
        """Publish `value` after every earlier store to the mailbox."""
        ...


class NativeWordOps:
    """WordOps backed by libmcdma-rpc; ctypes releases the GIL while it spins."""

    def __init__(self, library: ctypes.CDLL, path: str) -> None:
        self.path = path
        # Also offers the optional Metal buffer calls on macOS builds.
        self.library = library
        self._wait = library.mcdma_rpc_wait_word
        self._wait.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self._wait.restype = ctypes.c_uint64
        self._store = library.mcdma_rpc_store_word
        self._store.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self._store.restype = None

    def wait(self, address: int, seq: int, *, equal: bool, timeout_s: float) -> int:
        timeout_ns = max(1, int(timeout_s * 1e9))
        return int(self._wait(address, seq, 1 if equal else 0, _SPIN_NS, timeout_ns))

    def store(self, address: int, value: int) -> None:
        self._store(address, value)


def load_word_ops(path: str | None = None) -> tuple[NativeWordOps | None, str]:
    """Load libmcdma-rpc; returns the ops, or None with the reason it is unusable."""
    candidates = [path] if path else []
    if not path and os.environ.get(LIBRARY_ENV):
        candidates.append(os.environ[LIBRARY_ENV])
    candidates.extend(_DEFAULT_LIBRARIES)
    for candidate in candidates:
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            library = ctypes.CDLL(candidate)
            abi = library.mcdma_rpc_abi
        except (OSError, AttributeError) as exc:
            return None, f"{candidate} is not a usable libmcdma-rpc: {exc}"
        abi.restype = ctypes.c_uint32
        if int(abi()) != ABI_VERSION:
            return None, f"{candidate} speaks ABI {abi()}, oMLX needs {ABI_VERSION}"
        return NativeWordOps(library, candidate), f"using {candidate}"
    return None, "libmcdma-rpc is not installed; install MCDMA's link daemon package"
