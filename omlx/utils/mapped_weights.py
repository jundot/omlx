# SPDX-License-Identifier: Apache-2.0
"""Mapped (zero-copy) model weights.

With ``mapped_weights_enabled`` a model's ``.safetensors`` shards are mapped
read-only and every tensor is handed to MLX as a view of the file
(``mx.from_dlpack(copy=False)``) instead of being read into a fresh Metal
buffer. The weights then live in the page cache: they are not part of the
process footprint, a reload does not copy them again, and pages the model is
not using (cold MoE experts, say) can be dropped by macOS and read back from
the SSD on demand.

Metal still counts a mapped buffer in full against the GPU working set
(``iogpu.wired_limit_mb``) once a kernel touches it, so this does not let a
model larger than the Metal cap run. What it removes is the duplicate copy,
the pinning, and the footprint.

Alignment
---------
A GPU kernel reads a typed buffer at the alignment of its element type. A
tensor whose file offset is not a multiple of its element size comes out of
the zero-copy wrap pointing at the right byte, but Metal reads it as if it
started at the aligned address below, so the values are silently wrong. MLX
does not check this. Checkpoints written by ``mx.save_safetensors`` are laid
out tensor by tensor with aligned relative offsets, but the JSON header is not
padded, so the data section (and therefore every tensor) is usually shifted by
1-7 bytes. The Hugging Face writer pads the header to 8 bytes and is aligned.

The loader never maps a misaligned tensor: it copies it, exactly like
``mx.load``. :func:`check_alignment` tells the admin UI how much of a model
can be mapped, and :func:`realign_model` rewrites the shards (in place or into
a new directory) so all of it can. A rewrite keeps the file a valid,
hole-free safetensors file: the header is padded with spaces, tensors are
ordered by element size, and the tensor bytes are unchanged, so every other
loader reads the result exactly as before.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import shutil
import struct
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

# safetensors dtype -> (numpy transport dtype, mlx dtype to view as). bf16 has
# no numpy equivalent and travels as uint16; fp8 has no mlx dtype and loads
# as raw uint8 bytes, the same thing ``mx.load`` returns for it.
_DTYPES: dict[str, tuple[Any, Any]] = {
    "BF16": (np.uint16, mx.bfloat16),
    "F16": (np.float16, None),
    "F32": (np.float32, None),
    "U32": (np.uint32, None),
    "I32": (np.int32, None),
    "U16": (np.uint16, None),
    "I16": (np.int16, None),
    "U8": (np.uint8, None),
    "I8": (np.int8, None),
    "U64": (np.uint64, None),
    "I64": (np.int64, None),
    "BOOL": (np.bool_, None),
    "F8_E4M3": (np.uint8, None),
    "F8_E5M2": (np.uint8, None),
}

# Header padding target. 8 is the safetensors convention and covers every
# element size above.
_HEADER_ALIGN = 8
_COPY_CHUNK = 64 << 20
_TMP_SUFFIX = ".omlx-align.tmp"


def _itemsize(dtype: str) -> int:
    entry = _DTYPES.get(dtype)
    return np.dtype(entry[0]).itemsize if entry else 1


def read_header(path: str | Path) -> tuple[int, dict[str, Any]]:
    """Return ``(data_start, header)`` of a safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return 8 + n, header


def _weight_files(model_path: str | Path) -> list[Path]:
    return sorted(Path(model_path).glob("*.safetensors"))


# ---------------------------------------------------------------------------
# Alignment check
# ---------------------------------------------------------------------------


@dataclass
class AlignmentReport:
    """How much of a model's checkpoint can be mapped without a copy."""

    model_path: str
    files: int = 0
    tensors: int = 0
    total_bytes: int = 0
    misaligned_files: list[str] = field(default_factory=list)
    misaligned_tensors: int = 0
    misaligned_bytes: int = 0
    # Files that are symlinks (the Hugging Face cache layout). They are not
    # rewritten in place: the blob behind them is shared and content-addressed.
    symlinked: bool = False
    writable: bool = True
    free_disk_bytes: int = 0
    # Bytes of the shards a rewrite has to produce (misaligned files only).
    rewrite_bytes: int = 0
    # Size of everything a copy writes: every file in the model directory.
    model_bytes: int = 0

    @property
    def aligned(self) -> bool:
        return self.files > 0 and self.misaligned_tensors == 0

    @property
    def can_realign_in_place(self) -> bool:
        return (
            not self.aligned
            and not self.symlinked
            and self.writable
            # A shard is rewritten next to itself, then swapped in.
            and self.free_disk_bytes > self._largest_misaligned_file()
        )

    @property
    def can_realign_copy(self) -> bool:
        return not self.aligned and self.free_disk_bytes > self.model_bytes

    def _largest_misaligned_file(self) -> int:
        sizes = [
            os.path.getsize(Path(self.model_path) / name)
            for name in self.misaligned_files
        ]
        return max(sizes, default=0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["aligned"] = self.aligned
        data["can_realign_in_place"] = self.can_realign_in_place
        data["can_realign_copy"] = self.can_realign_copy
        return data


def check_alignment(model_path: str | Path) -> AlignmentReport:
    """Scan the shard headers (no tensor data is read)."""
    model_path = Path(model_path)
    report = AlignmentReport(model_path=str(model_path))
    for shard in _weight_files(model_path):
        data_start, header = read_header(shard)
        report.files += 1
        report.symlinked = report.symlinked or shard.is_symlink()
        file_bad = False
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            report.tensors += 1
            report.total_bytes += end - start
            if (data_start + start) % _itemsize(info["dtype"]):
                report.misaligned_tensors += 1
                report.misaligned_bytes += end - start
                file_bad = True
        if file_bad:
            report.misaligned_files.append(shard.name)
            report.rewrite_bytes += shard.stat().st_size
    report.model_bytes = sum(
        f.stat().st_size for f in model_path.rglob("*") if f.is_file()
    )
    report.writable = os.access(model_path, os.W_OK)
    try:
        report.free_disk_bytes = shutil.disk_usage(model_path).free
    except OSError:
        report.free_disk_bytes = 0
    return report


# ---------------------------------------------------------------------------
# Realign
# ---------------------------------------------------------------------------


def _aligned_layout(header: dict[str, Any]) -> tuple[bytes, list[tuple[str, int]]]:
    """Header bytes and copy order for an aligned, hole-free rewrite.

    Tensors are ordered by element size (largest first), so each one ends
    on a multiple of the next one's element size and needs no gap; the
    header is padded with spaces so the data starts on an 8-byte boundary.
    Returns the encoded header (without the 8-byte length prefix) and the
    tensors as ``(name, original start offset)`` in write order.
    """
    meta = header.get("__metadata__")
    items = [(k, v) for k, v in header.items() if k != "__metadata__"]
    items.sort(key=lambda kv: (-_itemsize(kv[1]["dtype"]), kv[1]["data_offsets"][0]))
    new: dict[str, Any] = {}
    if meta is not None:
        new["__metadata__"] = meta
    pos = 0
    order = []
    for name, info in items:
        start, end = info["data_offsets"]
        new[name] = {**info, "data_offsets": [pos, pos + end - start]}
        order.append((name, start))
        pos += end - start
    encoded = json.dumps(new, separators=(",", ":")).encode()
    encoded += b" " * (-(8 + len(encoded)) % _HEADER_ALIGN)
    return encoded, order


def rewrite_aligned(
    src: str | Path,
    dst: str | Path,
    progress: Callable[[int], None] | None = None,
) -> None:
    """Write ``src`` to ``dst`` with every tensor aligned (see module doc)."""
    data_start, header = read_header(src)
    encoded, order = _aligned_layout(header)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fout.write(struct.pack("<Q", len(encoded)))
        fout.write(encoded)
        for name, start in order:
            s, e = header[name]["data_offsets"]
            fin.seek(data_start + start)
            remaining = e - s
            while remaining:
                chunk = fin.read(min(remaining, _COPY_CHUNK))
                if not chunk:
                    raise OSError(f"{src}: truncated tensor {name}")
                fout.write(chunk)
                remaining -= len(chunk)
                if progress is not None:
                    progress(len(chunk))
        fout.flush()
        os.fsync(fout.fileno())


def realign_model(
    model_path: str | Path,
    *,
    mode: str,
    dest: str | Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Make every tensor of a model mappable.

    ``mode="in_place"`` rewrites each misaligned shard next to itself and
    atomically replaces it; the other files are untouched. ``mode="copy"``
    writes a complete copy of the model directory to ``dest`` (aligned shards
    rewritten, everything else copied) and leaves the original alone.
    ``progress(done_bytes, total_bytes)`` reports the shard bytes written.
    Returns the directory that now holds the aligned model.
    """
    model_path = Path(model_path)
    report = check_alignment(model_path)
    if mode == "in_place":
        if report.symlinked:
            raise ValueError(
                "model files are symlinks (Hugging Face cache); realign with a copy"
            )
        if not report.writable:
            raise PermissionError(f"{model_path} is not writable")
        targets = [model_path / name for name in report.misaligned_files]
        out_dir = model_path
    elif mode == "copy":
        if dest is None:
            raise ValueError("copy mode needs a destination directory")
        out_dir = Path(dest)
        if out_dir.exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"{out_dir} already exists and is not empty")
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = _weight_files(model_path)
    else:
        raise ValueError(f"unknown realign mode: {mode!r}")

    def _payload(shard: Path) -> int:
        data_start, _header = read_header(shard)
        return shard.stat().st_size - data_start

    total = sum(_payload(t) for t in targets)
    done = 0

    def _tick(n: int) -> None:
        nonlocal done
        done += n
        if progress is not None:
            progress(done, total)

    misaligned = set(report.misaligned_files)
    for shard in targets:
        final = out_dir / shard.name
        if mode == "copy" and shard.name not in misaligned:
            shutil.copyfile(shard, final)
            _tick(_payload(final))
            continue
        tmp = final.with_name(final.name + _TMP_SUFFIX)
        try:
            rewrite_aligned(shard, tmp, progress=_tick)
            os.replace(tmp, final)
        finally:
            tmp.unlink(missing_ok=True)

    if mode == "copy":
        for entry in model_path.iterdir():
            if entry.suffix == ".safetensors" or entry.name.startswith("."):
                continue
            target = out_dir / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, target)
    logger.info(
        "Realigned %d shard(s) of %s (%s) -> %s",
        len(misaligned),
        model_path.name,
        mode,
        out_dir,
    )
    return out_dir


# ---------------------------------------------------------------------------
# Mapped loading
# ---------------------------------------------------------------------------


@dataclass
class _MappedFile:
    ref: weakref.ref  # numpy memmap of the whole file
    nbytes: int  # tensor bytes served from the mapping
    # File byte ranges [start, end) of the mapped tensors, sorted and merged.
    # Copied (misaligned) tensors are not in here, so their pages never count
    # toward mapped residency.
    spans: list[tuple[int, int]] = field(default_factory=list)


_lock = threading.Lock()
# Model directory -> number of loaded engines that mapped it.
_enabled_dirs: dict[str, int] = {}
_files: dict[str, list[_MappedFile]] = {}
_original_load: Callable[..., Any] | None = None
_wired_restore: int | None = None


def _dir_key(path: str | Path) -> str:
    return os.path.normpath(os.path.abspath(os.fspath(path)))


def _load_mapped(path: str) -> tuple[dict[str, mx.array], dict[str, str]]:
    data_start, header = read_header(path)
    metadata = header.pop("__metadata__", None) or {}
    if any(info["dtype"] not in _DTYPES for info in header.values()):
        raise _UnsupportedDtypeError
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out: dict[str, mx.array] = {}
    mapped = copied = 0
    spans: list[tuple[int, int]] = []
    for name, info in header.items():
        start, end = info["data_offsets"]
        np_dtype, mx_view = _DTYPES[info["dtype"]]
        view = mm[data_start + start : data_start + end].view(np_dtype)
        view = view.reshape(info["shape"])
        arr = None
        # Never wrap a misaligned pointer: Metal would read shifted data.
        if (data_start + start) % view.itemsize == 0:
            try:
                arr = mx.from_dlpack(view, copy=False)
            except Exception:  # noqa: BLE001
                logger.debug("zero-copy wrap of %s failed", name, exc_info=True)
        if arr is None:
            arr = mx.array(np.array(view))
            copied += end - start
        else:
            mapped += end - start
            spans.append((data_start + start, data_start + end))
        out[name] = arr.view(mx_view) if mx_view is not None else arr
    key = _dir_key(Path(path).parent)
    with _lock:
        _files.setdefault(key, []).append(
            _MappedFile(weakref.ref(mm, _mappings_changed), mapped, _merge_spans(spans))
        )
        _mappings_changed()
    if copied:
        logger.warning(
            "%s: %.2f GB mapped, %.2f GB copied (misaligned tensors)",
            Path(path).name,
            mapped / 1e9,
            copied / 1e9,
        )
    else:
        logger.debug("%s: %.2f GB mapped", Path(path).name, mapped / 1e9)
    return out, metadata


class _UnsupportedDtypeError(Exception):
    pass


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(s for s in spans if s[1] > s[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _patched_load(file, *args, **kwargs):
    path = os.fspath(file) if isinstance(file, (str, os.PathLike)) else None
    fmt = kwargs.get("format", args[0] if args else None)
    if (
        path is None
        or not path.endswith(".safetensors")
        or fmt not in (None, "safetensors")
        or kwargs.get("stream") is not None
        or _dir_key(Path(path).parent) not in _enabled_dirs
    ):
        return _original_load(file, *args, **kwargs)
    try:
        weights, metadata = _load_mapped(path)
    except _UnsupportedDtypeError:
        logger.info("%s: dtype not mappable, loading normally", Path(path).name)
        return _original_load(file, *args, **kwargs)
    return_metadata = kwargs.get("return_metadata", args[1] if len(args) > 1 else False)
    return (weights, metadata) if return_metadata else weights


def _install() -> None:
    global _original_load
    if _original_load is None:
        _original_load = mx.load
        mx.load = _patched_load


def enable(model_path: str | Path) -> None:
    """Map the weights of every later load from ``model_path``.

    Also lifts the MLX wired limit while any mapped model is enabled, so
    macOS can page out mapped weights the model is not using (MLX's limit is
    process wide; it is restored when the last mapped model is disabled).
    """
    global _wired_restore
    key = _dir_key(model_path)
    # The wired limit is process wide: change it under the same lock as the
    # set of mapped models, so a concurrent unload cannot restore it while
    # another mapped model is loading.
    with _lock:
        _install()
        first = not _enabled_dirs
        _enabled_dirs[key] = _enabled_dirs.get(key, 0) + 1
        if first:
            try:
                _wired_restore = int(mx.set_wired_limit(0))
            except Exception:  # noqa: BLE001
                _wired_restore = 0
    if first and _wired_restore:
        logger.info("Mapped weights: MLX wired limit lifted while loaded")


def disable(model_path: str | Path) -> None:
    """Release one engine's mapping of ``model_path`` (call on unload).

    Directories are reference counted: two model IDs can resolve to the same
    path, and unloading one must not unmap the other.
    """
    global _wired_restore
    key = _dir_key(model_path)
    with _lock:
        if key not in _enabled_dirs:
            return
        _enabled_dirs[key] -= 1
        if _enabled_dirs[key] > 0:
            return
        del _enabled_dirs[key]
        if not _enabled_dirs and _wired_restore:
            try:
                mx.set_wired_limit(_wired_restore)
            except Exception:  # noqa: BLE001
                logger.debug("restoring the wired limit failed", exc_info=True)
            _wired_restore = None


def is_enabled(model_path: str | Path) -> bool:
    return _dir_key(model_path) in _enabled_dirs


def _live_maps() -> list[tuple[np.memmap, _MappedFile]]:
    """Mappings still referenced by a loaded model (unloads drop them)."""
    live = []
    with _lock:
        for key in list(_files):
            entries = [e for e in _files[key] if e.ref() is not None]
            if entries:
                _files[key] = entries
            else:
                del _files[key]
            live.extend((e.ref(), e) for e in entries)
    return [(mm, e) for mm, e in live if mm is not None]


def mapped_bytes() -> int:
    """Tensor bytes currently served from file mappings."""
    return sum(e.nbytes for _mm, e in _live_maps())


_PAGE = 16384
_libc = None
_resident = {"bytes": 0, "at": 0.0, "running": False, "generation": -1}
# Bumped whenever a mapping appears or is released (weakref callback). A
# residency sample is only valid for the generation it measured, so a model
# swap never inherits the previous model's resident pages.
_generation = [0]


def _mappings_changed(_ref: object = None) -> None:
    # Weakref callbacks can run while _lock is held; a lost increment is
    # harmless because any increment invalidates the cached sample.
    _generation[0] += 1


_RESIDENT_MAX_AGE = 1.0


def cached_resident_mapped_bytes() -> int:
    """Last :func:`resident_mapped_bytes` sample, refreshed off-thread.

    ``mincore`` costs ~20 ms per GB mapped, too slow for the memory guard's
    poll, so the guard reads this cached value and a short-lived thread
    refreshes it when it is older than a second. Until a sample of the
    current set of mappings lands (first load, or after any model maps or
    releases weights) nothing is assumed resident, the conservative side for
    the guard.
    """
    total = mapped_bytes()
    if not total:
        return 0
    now = time.monotonic()
    with _lock:
        current = _resident["generation"] == _generation[0]
        stale = not current or now - _resident["at"] > _RESIDENT_MAX_AGE
        start = stale and not _resident["running"]
        if start:
            _resident["running"] = True
        value = min(_resident["bytes"], total) if current else 0
    if start:
        threading.Thread(
            target=_refresh_resident, name="mapped-weights-mincore", daemon=True
        ).start()
    return value


def _refresh_resident() -> None:
    # Tag the sample with the generation it started from: if mappings change
    # mid-measurement the tag is already stale and the next read re-samples.
    generation = _generation[0]
    try:
        value = resident_mapped_bytes()
    except Exception:  # noqa: BLE001
        # Unmeasured pages get no credit: the guard counts resident mapped
        # pages as reclaimable, so guessing high would overstate headroom.
        logger.warning("mincore sample of mapped weights failed", exc_info=True)
        value = 0
    with _lock:
        _resident.update(
            bytes=value, at=time.monotonic(), running=False, generation=generation
        )


def _span_bytes_per_page(spans: list[tuple[int, int]], pages: int) -> np.ndarray:
    """Bytes of ``spans`` that fall in each page of a page-aligned mapping."""
    per_page = np.zeros(pages, dtype=np.int64)
    for start, end in spans:
        first, last = start // _PAGE, (end - 1) // _PAGE
        if first == last:
            per_page[first] += end - start
            continue
        per_page[first] += (first + 1) * _PAGE - start
        per_page[first + 1 : last] += _PAGE
        per_page[last] += end - last * _PAGE
    return per_page


def resident_mapped_bytes() -> int:
    """Bytes of mapped tensors that are in RAM right now (``mincore``).

    Only the byte ranges of tensors served from the mapping count: pages of
    copied tensors in the same file may be cached too, but they back nothing
    the guard charges as mapped.
    """
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
    total = 0
    for mm, entry in _live_maps():
        if not entry.spans:
            continue
        addr = mm.ctypes.data
        if addr % _PAGE:
            continue  # np.memmap of a whole file is page aligned
        pages = (mm.nbytes + _PAGE - 1) // _PAGE
        vec = (ctypes.c_char * pages)()
        if _libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(pages * _PAGE), vec):
            continue
        resident = np.frombuffer(vec, dtype=np.uint8) & 1
        per_page = _span_bytes_per_page(entry.spans, pages)
        total += int(per_page[resident.astype(bool)].sum())
    return total
