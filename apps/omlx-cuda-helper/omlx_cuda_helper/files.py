# SPDX-License-Identifier: Apache-2.0
"""Model copies the host pushes: one folder per model under the models dir."""

from __future__ import annotations

import os
import shutil
import zlib
from pathlib import Path
from typing import Any, BinaryIO

from .gpus import disk_free

# In every folder this store made for a copy; its time is the last use. Only
# those folders are copies: one the operator put here is used, never removed.
_COPY_MARKER = ".last_used"


class FileError(RuntimeError):
    pass


def _component(name: str) -> str:
    """A single path component; anything that could leave the folder is refused."""
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\0" in name:
        raise FileError(f"unsafe name: {name!r}")
    return name


class ModelStore:
    def __init__(self, root: Path):
        self.root = root
        self._open: tuple[Path, BinaryIO] | None = None

    def folder(self, name: str) -> Path:
        return self.root / _component(name)

    def probe(
        self, name: str, files: dict[str, int], running: str = ""
    ) -> dict[str, Any]:
        """What a copy of *name* still lacks, and the other copies here.

        An empty *name* only lists the copies. The running session's copy is
        never listed, so the host cannot offer to delete it.
        """
        copy = self.folder(name) if name else None
        missing, partial = [], {}
        for file, size in files.items():
            path = copy / _component(file)
            if not (path.is_file() and path.stat().st_size == size):
                missing.append(file)
                part = path.with_name(path.name + ".part")
                if part.is_file():
                    partial[file] = part.stat().st_size
        if copy is not None and files and (copy / _COPY_MARKER).exists():
            (copy / _COPY_MARKER).touch()
        copies = []
        for path in sorted(self.root.iterdir()) if self.root.is_dir() else []:
            marker = path / _COPY_MARKER
            if (
                path.name == name
                or path.is_symlink()
                or not marker.exists()
                or os.path.realpath(path) == running
            ):
                continue
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            used = marker.stat().st_mtime
            copies.append({"name": path.name, "bytes": size, "used": used})
        complete = copy is not None and bool(files) and not missing
        return {
            "found": str(copy) if complete else None,
            "copy": str(copy) if copy is not None else "",
            "missing": missing,
            "partial": partial,
            "copies": copies,
            "free": disk_free(self.root),
        }

    def write(
        self,
        name: str,
        file: str,
        size: int,
        offset: int,
        data: bytes | memoryview,
        crc: int,
        last: bool,
    ) -> None:
        """Append one chunk of a pushed file; the last chunk makes it visible."""
        if file == _COPY_MARKER:
            # A pushed marker would turn the operator's folder into a copy.
            raise FileError(f"unsafe name: {file!r}")
        path = self.folder(name) / _component(file)
        part = path.with_name(path.name + ".part")
        if zlib.crc32(data) != crc:
            raise FileError(f"{file}: a chunk arrived damaged")
        if self._open is None or self._open[0] != part:
            self.close()
            created = not part.parent.exists()
            part.parent.mkdir(parents=True, exist_ok=True)
            if created:
                (part.parent / _COPY_MARKER).touch()
            # "r+b" keeps what a stopped copy left; a fresh file starts empty.
            mode = "r+b" if offset and part.exists() else "wb"
            handle = open(part, mode)  # noqa: SIM115 - kept open across chunks
            self._open = (part, handle)
        handle = self._open[1]
        handle.seek(0, os.SEEK_END)
        if handle.tell() != offset:
            raise FileError(f"{file}: expected offset {handle.tell()}, got {offset}")
        handle.write(data)
        if not last:
            return
        handle.flush()
        os.fsync(handle.fileno())
        total = handle.tell()
        self.close()
        if total != size:
            part.unlink()
            raise FileError(f"{file}: got {total} of {size} bytes")
        os.replace(part, path)

    def close(self) -> None:
        if self._open is not None:
            self._open[1].close()
            self._open = None

    def delete(self, names: list[str], running: str = "") -> int:
        """Remove copies by folder name; returns the free bytes afterwards."""
        root = os.path.realpath(self.root)
        for name in names:
            path = self.folder(name)
            real = os.path.realpath(path)
            if (
                path.is_symlink()
                or os.path.dirname(real) != root
                or real == running
                or not (path / _COPY_MARKER).exists()
            ):
                continue
            shutil.rmtree(path, ignore_errors=True)
        return disk_free(self.root)
