# SPDX-License-Identifier: Apache-2.0
"""Filesystem helpers for optional inspection sidecars.

Locks are shared by managers using the same cache directory in this process.
They cover commits/unlinks, not tensor serialization or text rendering.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path

_FILE_LOCKS = tuple(threading.RLock() for _ in range(256))
_SIDECAR_NAME = re.compile(r"^([0-9a-f]{64})\.(tokens|txt)$")
_TEMP_NAME = re.compile(r"^\.([0-9a-f]{64})\..*\.inspection-tmp$")


def block_file_lock(path: Path):
    return _FILE_LOCKS[hash(str(path.absolute())) % len(_FILE_LOCKS)]


def sidecar_paths(path: Path) -> tuple[Path, Path]:
    return path.with_suffix(".tokens"), path.with_suffix(".txt")


def sidecar_sizes(path: Path) -> tuple[int, bool]:
    total = 0
    complete = True
    for sidecar in sidecar_paths(path):
        try:
            # Never follow a symlink supplied in the cache directory.
            if sidecar.is_symlink():
                complete = False
                continue
            total += sidecar.stat().st_size
        except OSError:
            complete = False
    return total, complete


def atomic_write(path: Path, content: bytes) -> None:
    """Private permissions; unique temporary name; atomic per-file commit."""
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".inspection-tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def unlink_block_files(path: Path) -> None:
    """Remove inspection data first, so a failed unlink stays accountable."""
    with block_file_lock(path):
        for sidecar in sidecar_paths(path):
            sidecar.unlink(missing_ok=True)
        path.unlink(missing_ok=True)


def cleanup_orphans(directory: Path) -> None:
    """Remove only recognized inspection files with no corresponding KV file."""
    for path in directory.iterdir():
        match = _SIDECAR_NAME.fullmatch(path.name)
        temporary = _TEMP_NAME.fullmatch(path.name)
        if match is None and temporary is None:
            continue
        stem = (match or temporary).group(1)
        if stem[0] != directory.name:
            continue
        canonical = directory / f"{stem}.safetensors"
        with block_file_lock(canonical):
            if temporary or not canonical.exists():
                path.unlink(missing_ok=True)
