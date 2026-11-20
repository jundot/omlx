# SPDX-License-Identifier: Apache-2.0
"""Verified external BF16 Engram storage for the pinned text-only conversion.

prepare_table_receipt streams the original files once, offline. Keep the
resulting private receipt under your control and the four inputs immutable.
open_tables rechecks the receipt, file identities and headers. By default shard
4 is mapped and wired; with first_mode="pread" it is read by row like shards 3,
2 and 1, which are NEVER mapped or scanned. Late tables belong on the internal
SSD for speed (a warning otherwise). close() releases mappings, locks and fds.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import mmap
import os
import plistlib
import re
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import warnings
import weakref
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx

_REVISION = "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"
_BLOCKS = (1, 11, 21, 31)
# Whole-file LFS sizes and SHA-256 oids, independent of the converted manifest.
_PINS = {
    1: (
        31678921112,
        "49ed344e78dc2b5aa5a9c085ef69f9fde65e24e8e31e9f69af0bfcc6b31e46bd",
    ),
    11: (
        31679809688,
        "679ebd686950fd8ba0634988df113ebcc3b86a9c8b2a0e531f4ae756271fe18b",
    ),
    21: (
        31680784664,
        "c857c5a16e2988e655950bb55334e787d1fdfd29ca6d0e7355d0cdc544f08f06",
    ),
    31: (
        31681717016,
        "042902eb804c14847dde80e57d4c4dc4a30663a296b2f87e421d93f841692c8f",
    ),
}
_READ = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_RECEIPT = "hy4-flash-table-receipt/v1"
_CHUNK = 4 << 20
_MAX_GATHER_BYTES = 256 << 20  # Larger requests must chunk their prefill.


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _unique_object(pairs):
    d = dict(pairs)
    _require(len(d) == len(pairs), "duplicate JSON key")
    return d


def _stamp(st):
    _require(stat.S_ISREG(st.st_mode), "expected an immutable regular file")
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def _read_exact(fd, length, offset):
    data = os.pread(fd, length, offset)
    if len(data) != length:
        raise OSError(
            f"short positional read: {len(data)} of {length} bytes at {offset}"
        )
    return data


def _document(path, *, private=False):
    fd = os.open(path, _READ)
    try:
        st = os.fstat(fd)
        before = _stamp(st)
        if private:
            _require(
                st.st_uid == os.geteuid() and not st.st_mode & 0o077,
                "expected a private, owned receipt",
            )
        _require(
            0 < st.st_size <= (65536 if private else 32 << 20),
            "invalid JSON document length",
        )
        data = _read_exact(fd, st.st_size, 0)
        _require(
            before == _stamp(os.fstat(fd)) == _stamp(os.lstat(path)),
            "JSON document changed",
        )
    finally:
        os.close(fd)
    doc = json.loads(data, object_pairs_hook=_unique_object)
    _require(isinstance(doc, dict), "expected a JSON object")
    return doc, hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class _Spec:
    block: int
    key: str
    shard: str
    rows: int
    offset: int
    length: int
    size: int
    sha: str
    payload_sha: str

    # Every original tensor at the pinned revision has this BF16 row layout.
    width = 96
    stride = 192


def table_specs(doc):
    source = doc.get("source", {})
    _require(
        doc.get("schema") == "hy4-flash-oqh/v1"
        and doc.get("text_only") is True
        and isinstance(source, dict)
        and source.get("repo") == "tencent/Hy4-flash"
        and source.get("revision") == _REVISION,
        "expected a converted text-only manifest for the pinned Hy4-flash revision",
    )
    tables = doc.get("external_tables", {})
    _require(
        isinstance(tables, dict) and tables.keys() == {str(b) for b in _BLOCKS},
        "expected exactly four external Engram blocks",
    )
    specs = []
    for i, block in enumerate(_BLOCKS):
        r = tables[str(block)]
        key = f"model.layers.{block}.engram_layer.multi_head_embedding.embedding.weight"
        shard = f"model-{4 - i:05d}-of-00148.safetensors"
        required = {
            "source_key",
            "source_shard",
            "shape",
            "dtype",
            "block",
            "row_stride",
            "source_offset",
            "byte_length",
            "file_size",
            "file_sha256",
            "payload_sha256",
        }
        _require(
            isinstance(r, dict) and required <= r.keys(),
            f"incomplete table locator for block {block}",
        )
        shape = r["shape"]
        _require(
            r["source_key"] == key
            and r["source_shard"] == shard
            and r["dtype"] == "BF16"
            and isinstance(shape, list)
            and len(shape) == 2
            and all(type(n) is int for n in shape)
            and shape[0] > 0
            and shape[1] == _Spec.width
            and all(
                type(r[k]) is int
                for k in (
                    "block",
                    "row_stride",
                    "source_offset",
                    "byte_length",
                    "file_size",
                )
            )
            and r["block"] == block
            and r["row_stride"] == _Spec.stride
            and 8 < r["source_offset"] <= 8 + 65536
            and r["source_offset"] % 8 == 0
            and r["byte_length"] == shape[0] * _Spec.stride
            and r["source_offset"] + r["byte_length"] == r["file_size"],
            f"invalid dedicated BF16 table locator for block {block}",
        )
        _require(
            (r["file_size"], r["file_sha256"]) == _PINS[block]
            and isinstance(r["payload_sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", r["payload_sha256"]),
            f"missing digest or wrong public LFS pin for block {block}",
        )
        specs.append(
            _Spec(
                block,
                key,
                shard,
                shape[0],
                r["source_offset"],
                r["byte_length"],
                r["file_size"],
                r["file_sha256"],
                r["payload_sha256"],
            )
        )
    return specs


def _internal_ssd_device():
    _require(sys.platform == "darwin", "verified internal SSD policy requires macOS")
    data = "/System/Volumes/Data"
    info = plistlib.loads(
        subprocess.check_output(
            ["/usr/sbin/diskutil", "info", "-plist", data], timeout=15
        )
    )
    _require(
        info.get("Internal") is True
        and info.get("SolidState") is True
        and info.get("MountPoint") == data
        and info.get("FilesystemType") == "apfs",
        "Data volume is not verified internal solid-state APFS storage",
    )
    dev = os.stat(data).st_dev
    node = os.stat(info["DeviceNode"])
    _require(
        stat.S_ISBLK(node.st_mode) and node.st_rdev == dev,
        "internal SSD device mismatch",
    )
    return dev


def _internal_ssd_device_or_none():
    try:
        return _internal_ssd_device()
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def _open_table(path):
    try:
        return os.open(path, _READ)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise ValueError(
                f"{path} is a symlink (e.g. a Hugging Face cache snapshot); "
                "download the tables with --local-dir so they are plain files"
            ) from e
        raise


@dataclass
class _TableFile:
    spec: _Spec
    path: Path
    fd: int
    stamp: list[int]
    header: bytes
    requested_rows: int = 0
    unique_rows: int = 0
    pread_calls: int = 0
    pread_bytes: int = 0

    def unchanged(self):
        _require(
            self.stamp == _stamp(os.fstat(self.fd)) == _stamp(os.lstat(self.path)),
            f"table changed or replaced: {self.spec.shard}",
        )


def _open_files(stack, specs, first_root, late_root, first_mode="mlock"):
    device = _internal_ssd_device_or_none()
    result = {}
    for spec in specs:
        hot = spec.block == _BLOCKS[0]
        path = (Path(first_root) if hot else Path(late_root)) / spec.shard
        # Capture an absolute pathname once, but never follow a final symlink.
        path = path.parent.resolve(strict=True) / path.name
        fd = _open_table(path)
        stack.callback(os.close, fd)
        stamp = _stamp(os.fstat(fd))
        _require(
            stamp[2] == spec.size, f"unexpected original file length: {spec.shard}"
        )
        if not hot and device is not None and stamp[0] != device:
            warnings.warn(
                f"late Engram table {spec.shard} is not on the internal SSD; "
                "row reads may be slower",
                stacklevel=3,
            )
        if not hot or first_mode == "pread":
            # Row reads only: no readahead clusters or page-cache growth.
            # Darwin F_RDAHEAD (not exported by all Python fcntl builds).
            fcntl.fcntl(fd, getattr(fcntl, "F_RDAHEAD", 45), 0)
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        prefix = _read_exact(fd, 8, 0)
        (n,) = struct.unpack("<Q", prefix)
        _require(
            n == spec.offset - 8, f"unexpected Safetensors header length: {spec.shard}"
        )
        body = _read_exact(fd, n, 8)
        h = json.loads(body, object_pairs_hook=_unique_object)
        _require(isinstance(h, dict), f"invalid Safetensors JSON header: {spec.shard}")
        h.pop("__metadata__", None)
        _require(
            h
            == {
                spec.key: {
                    "shape": [spec.rows, spec.width],
                    "dtype": "BF16",
                    "data_offsets": [0, spec.length],
                }
            },
            f"wrong dedicated Safetensors header: {spec.shard}",
        )
        t = _TableFile(spec, path, fd, stamp, prefix + body)
        t.unchanged()
        result[spec.block] = t
    return result


def prepare_table_receipt(manifest, *, first_root, late_root, receipt) -> Path:
    """Offline verification, never part of generate(). Publish a *new* receipt.

    Both whole-file (pinned LFS oid) and BF16 payload digests must match. The
    receipt binds the exact manifest and all four open file identities without
    embedding any machine-specific paths. Source and SSD roots are read-only.
    """
    out = Path(receipt)
    parent = out.parent.resolve(strict=True)
    for root in (first_root, late_root):
        _require(
            not parent.is_relative_to(Path(root).resolve(strict=True)),
            "receipt must be outside table roots",
        )
    out = parent / out.name
    if os.path.lexists(out):
        raise FileExistsError(f"refusing to replace table receipt: {out}")
    doc, digest = _document(manifest)
    with ExitStack() as stack:
        files = _open_files(stack, table_specs(doc), first_root, late_root)
        for t in files.values():
            payload, whole = hashlib.sha256(), hashlib.sha256(t.header)
            pos = t.spec.offset
            while pos < t.spec.size:
                chunk = _read_exact(t.fd, min(_CHUNK, t.spec.size - pos), pos)
                payload.update(chunk)
                whole.update(chunk)
                pos += len(chunk)
            _require(
                whole.hexdigest() == t.spec.sha
                and payload.hexdigest() == t.spec.payload_sha,
                f"whole-file or payload SHA-256 mismatch: {t.spec.shard}",
            )
        for t in files.values():
            t.unchanged()
        _require(
            _document(manifest)[1] == digest, "manifest changed during verification"
        )
        record = {
            "schema": _RECEIPT,
            "manifest_sha256": digest,
            "files": {str(b): t.stamp for b, t in files.items()},
        }
        with tempfile.NamedTemporaryFile(dir=parent, prefix=".hy4-receipt-") as tmp:
            tmp.write(json.dumps(record).encode())
            tmp.flush()
            os.fsync(tmp.fileno())
            os.link(
                tmp.name, out, follow_symlinks=False
            )  # atomic; never replace an existing receipt
    return out


def _vm_call(fn, *args):
    if fn(*args) != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"first Engram table {fn.__name__}: {os.strerror(error)}")


class _WiredMapping:
    """Read-only, file-backed. No MLX array ever refers to the mapping itself."""

    def __init__(self, stack, table):
        libc = ctypes.CDLL(None, use_errno=True)
        ptr, size_t = ctypes.c_void_p, ctypes.c_size_t
        for name in ("mlock", "munlock", "mincore"):
            fn = getattr(libc, name)
            fn.restype = ctypes.c_int
            fn.argtypes = [ptr, size_t] + (
                [ctypes.POINTER(ctypes.c_ubyte)] if name == "mincore" else []
            )
        self._mincore = libc.mincore
        self.size = table.spec.size
        self.pages = (self.size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        self.mapping = mmap.mmap(table.fd, self.size, access=mmap.ACCESS_READ)
        stack.callback(self.mapping.close)
        self.addr = np.frombuffer(self.mapping, dtype=np.uint8, count=1).ctypes.data
        _vm_call(libc.mlock, self.addr, self.size)
        stack.callback(_vm_call, libc.munlock, self.addr, self.size)
        self.check_residency()  # No best-effort fallback if mlock/mincore fails.

    def read_rows(self, spec, ids):
        view = np.ndarray(
            (spec.rows, spec.width),
            dtype="<u2",
            buffer=self.mapping,
            offset=spec.offset,
        )
        return view[ids]  # advanced indexing: owns just the requested rows

    def check_residency(self):
        # mincore returns a status byte per page. Bound even this scratch space.
        total = 0
        for start in range(0, self.pages, 65536):
            count = min(self.pages - start, 65536)
            status = (ctypes.c_ubyte * count)()
            length = min(self.size - start * mmap.PAGESIZE, count * mmap.PAGESIZE)
            _vm_call(self._mincore, self.addr + start * mmap.PAGESIZE, length, status)
            total += int(np.count_nonzero(np.ctypeslib.as_array(status) & 1))
        if total != self.pages:
            raise RuntimeError(
                f"first Engram table residency incomplete: {total}/{self.pages} pages"
            )
        return total


class EngramTables:
    """First table wired (or read by row) plus three positional-read fds. Use open_tables()."""

    def __init__(self, files, hot, stack):
        self._files = files
        self._hot = hot
        self._lock = threading.Lock()
        self._release = weakref.finalize(self, stack.pop_all().close)

    def _require_open(self):
        if not self._release.alive:
            raise RuntimeError("Engram table provider is closed")

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        with self._lock:
            self._release()

    def shape_for_block(self, block):
        spec = self._files[block].spec
        return (spec.rows, spec.width)

    def _read_rows(self, block, rows):
        """One worker per verified FD; caller holds the provider lifetime lock."""
        table = self._files[block]
        spec = table.spec
        unique, inverse = np.unique(rows.ravel(), return_inverse=True)
        table.unchanged()
        if block == _BLOCKS[0] and self._hot is not None:
            values = self._hot.read_rows(spec, unique)
        else:
            values = np.empty((len(unique), spec.width), dtype="<u2")
            for i, row in enumerate(unique):
                raw = _read_exact(
                    table.fd, spec.stride, spec.offset + int(row) * spec.stride
                )
                values[i] = np.frombuffer(raw, dtype="<u2")
                table.pread_calls += 1
                table.pread_bytes += spec.stride
        table.unchanged()
        table.requested_rows += rows.size
        table.unique_rows += len(unique)
        return values[inverse].reshape(*rows.shape, spec.width)

    def gather(self, block: int, row_ids: np.ndarray) -> mx.array:
        import mlx.core as mx

        with self._lock:
            self._require_open()
            if type(block) is not int or block not in self._files:
                raise ValueError("unknown external Engram block")
            t = self._files[block]
            s = t.spec
            ids = np.asarray(row_ids)
            if ids.dtype.kind not in "ui":
                raise TypeError("Engram row IDs must be integers")
            if ids.size * s.stride > _MAX_GATHER_BYTES:
                raise ValueError(
                    "Engram gather exceeds per-call byte limit; chunk the prefill"
                )
            if ids.size and (ids.min() < 0 or ids.max() >= s.rows):
                raise IndexError(f"Engram row outside block {block}")
            try:
                bits = self._read_rows(block, ids)
            except (OSError, ValueError):
                self._release()
                raise
            return mx.array(bits).view(mx.bfloat16)

    def stats(self):
        """Logical demand counters, not physical SSD sector or VM-cache I/O."""
        with self._lock:
            live = self._release.alive
            wired = self._hot is not None
            pages = self._hot.pages if live and wired else 0
            blocks = {}
            for block, t in self._files.items():
                s = t.spec
                blocks[str(block)] = dict(
                    mode="mlock" if block == _BLOCKS[0] and wired else "pread",
                    source_shard=s.shard,
                    source_offset=s.offset,
                    byte_length=s.length,
                    row_stride=s.stride,
                    device=t.stamp[0],
                    inode=t.stamp[1],
                    requested_rows=t.requested_rows,
                    unique_rows=t.unique_rows,
                    pread_calls=t.pread_calls,
                    pread_bytes=t.pread_bytes,
                )
            return dict(
                closed=not live,
                first_block=_BLOCKS[0],
                page_size=mmap.PAGESIZE,
                first_locked_bytes=pages * mmap.PAGESIZE,
                first_pages=pages,
                first_mode="mlock" if wired else "pread",
                first_resident_pages=(
                    self._hot.check_residency() if live and wired else 0
                ),
                blocks=blocks,
            )


def open_tables(
    manifest, *, first_root, late_root, receipt, first_mode="mlock"
) -> EngramTables:
    """Reopen previously verified immutable inputs, then wire only the first file.

    first_mode="pread" never maps or wires the first table either; it is read
    row by row like the late tables, trading a little I/O for ~31.7 GB of RAM.

    No external path or identity hint in a converted artifact chooses a root.
    No fallback from a bad/missing receipt, failed lock or missing SSD shard.
    """
    _require(first_mode in ("mlock", "pread"), "unknown first Engram table mode")
    doc, digest = _document(manifest)
    specs = table_specs(doc)
    saved, _ = _document(receipt, private=True)
    _require(
        saved.get("schema") == _RECEIPT and saved.get("manifest_sha256") == digest,
        "stale or wrong table receipt for this manifest",
    )
    with ExitStack() as stack:
        files = _open_files(stack, specs, first_root, late_root, first_mode)
        _require(
            saved.get("files") == {str(b): t.stamp for b, t in files.items()},
            "stale table receipt: file identity changed",
        )
        hot = _WiredMapping(stack, files[_BLOCKS[0]]) if first_mode == "mlock" else None
        for t in files.values():
            t.unchanged()
        return EngramTables(files, hot, stack)
