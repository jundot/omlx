# SPDX-License-Identifier: Apache-2.0
"""Opt-in, read-only safetensors views during a synchronous model load.

MLX >= 0.32.4 prevents donation of external buffers. Import alignment is
conservative because loaders may reinterpret bytes as larger scalar types.
NumPy/DLPack owns each mapping until the last derived MLX array dies; there
is no unload registry.
"""

import importlib
import json
import logging
import math
import os
import struct
import threading
from contextlib import contextmanager
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
from packaging.version import Version

from .mapped_residency import track

logger = logging.getLogger(__name__)
_load_lock = threading.RLock()
# Transport BF16 as U16, then reinterpret; NumPy has no native BF16 dtype.
_DTYPES = {
    "BF16": ("uint16", mx.bfloat16),
    "F16": ("float16", None),
    "F32": ("float32", None),
    "U8": ("uint8", None),
    "I8": ("int8", None),
    "U16": ("uint16", None),
    "I16": ("int16", None),
    "U32": ("uint32", None),
    "I32": ("int32", None),
    "U64": ("uint64", None),
    "I64": ("int64", None),
    "BOOL": ("bool", None),
    "F8_E4M3": ("uint8", None),
}


class _NativeLoaderError(Exception):
    """The shard has an unknown dtype or no tensor that could be mapped."""


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate safetensors key: {key}")
        result[key] = value
    return result


def mapped_coverage(model_path):
    """Bytes the mapped loader can import without copying, from headers only.

    Returns ``(mappable_bytes, total_bytes)`` or None when the directory has
    no readable safetensors. Applies the loader's own rule: a known dtype at
    an absolute 8-byte file offset. Cached per file name, size and mtime.
    """
    try:
        directory = Path(model_path).resolve()
        files = tuple(
            (f.name, s.st_size, s.st_mtime_ns)
            for f in sorted(directory.glob("*.safetensors"))
            for s in (f.stat(),)
        )
    except OSError:
        return None
    if not files:
        return None
    return _coverage(str(directory), files)


@lru_cache(maxsize=256)
def _coverage(directory, files):
    mappable = total = 0
    try:
        for name, _size, _mtime in files:
            with open(Path(directory) / name, "rb") as source:
                raw = source.read(8)
                if len(raw) != 8:
                    return None
                length = struct.unpack("<Q", raw)[0]
                if length > 100_000_000:
                    return None
                header = json.loads(source.read(length))
            if not isinstance(header, dict):
                return None
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                offsets = info.get("data_offsets") if isinstance(info, dict) else None
                if (
                    not isinstance(offsets, list)
                    or len(offsets) != 2
                    or any(type(n) is not int for n in offsets)
                    or not 0 <= offsets[0] <= offsets[1]
                ):
                    return None
                lo, hi = offsets
                total += hi - lo
                if (
                    hi > lo
                    and info.get("dtype") in _DTYPES
                    and (8 + length + lo) % 8 == 0
                ):
                    mappable += hi - lo
    except (OSError, ValueError):
        return None
    return mappable, total


def _load_mapped(path):
    # Parse and map the same open file, even if its pathname is atomically
    # replaced during load. In-place truncation of a live checkpoint remains
    # unsupported, as for any file-backed model.
    with open(path, "rb") as source:
        source_stat = os.fstat(source.fileno())
        size = source_stat.st_size
        raw = source.read(8)
        if len(raw) != 8:
            raise ValueError("Incomplete safetensors header")
        length = struct.unpack("<Q", raw)[0]
        if length > 100_000_000 or length > size - 8:
            raise ValueError("Invalid safetensors header length")
        header = json.loads(source.read(length), object_pairs_hook=_unique_pairs)
        if not isinstance(header, dict):
            raise ValueError("Invalid safetensors header")
        metadata = header.pop("__metadata__", None)
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in metadata.items()
        ):
            raise ValueError("Safetensors metadata must contain strings")
        start = 8 + length
        spans = []
        for info in header.values():
            if not isinstance(info, dict):
                raise ValueError("Invalid safetensors tensor")
            if info.get("dtype") not in _DTYPES:
                raise _NativeLoaderError
            shape, offsets = info.get("shape"), info.get("data_offsets")
            if (
                not isinstance(shape, list)
                or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)
            ):
                raise ValueError("Invalid safetensors shape or offsets")
            lo, hi = offsets
            dtype = np.dtype(_DTYPES[info["dtype"]][0])
            if (
                hi < lo
                or hi > size - start
                or hi - lo != math.prod(shape) * dtype.itemsize
            ):
                raise ValueError("Safetensors tensor size mismatch")
            spans.append((lo, hi))
        # Padding between tensors is how aligned checkpoints are written, and
        # the native loader accepts it; overlapping tensors are rejected.
        cursor = 0
        for lo, hi in sorted(spans):
            if lo < cursor:
                raise ValueError("Safetensors tensors overlap")
            cursor = hi
        if cursor != size - start:
            raise ValueError("Safetensors data size mismatch")
        # A shard whose header leaves every tensor misaligned gains nothing
        # from a mapping, and the native loader copies it faster.
        if cursor and not any(hi > lo and (start + lo) % 8 == 0 for lo, hi in spans):
            logger.info(
                "Mapped weights %s: no aligned tensors, using the native loader",
                Path(path).name,
            )
            raise _NativeLoaderError
        # No memmap is needed for a shard containing only empty tensors.
        mapping = np.memmap(source, mode="r", dtype=np.uint8) if cursor else None
        weights, mapped, copied = {}, 0, 0
        for name, info in header.items():
            lo, hi = info["data_offsets"]
            dtype, reinterpret = _DTYPES[info["dtype"]]
            if hi == lo:
                array = mx.array(np.empty(info["shape"], dtype=dtype))
            else:
                view = (
                    mapping[start + lo : start + hi].view(dtype).reshape(info["shape"])
                )
                try:
                    # Include later U8 -> U32/U64 dtype views. MLX validates
                    # element alignment at import, not the future view dtype.
                    if (start + lo) % 8:
                        raise ValueError("unaligned external buffer")
                    array = mx.from_dlpack(view, copy=False)
                    track(view, mapping, source_stat, start + lo, start + hi)
                    mapped += hi - lo
                except (ValueError, RuntimeError):
                    # Old writers need not align the header. A virtual mapping
                    # cannot change a tensor's byte offset within its OS page.
                    array = mx.array(np.array(view, copy=True))
                    copied += hi - lo
            weights[name] = (
                array.view(reinterpret) if reinterpret is not None else array
            )
        logger.info(
            "Mapped weights %s: %.3f GB mapped, %.3f GB copied",
            Path(path).name,
            mapped / 1e9,
            copied / 1e9,
        )
        return weights, metadata


class _LoaderProxy:
    """Override load in a loader module without modifying mlx.core itself."""

    def __init__(self, core, load):
        self._core = core
        self.load = load

    def __getattr__(self, name):
        return getattr(self._core, name)


@contextmanager
def mapped_loading(model_path, *, enabled=True):
    """Scope the mlx-lm/mlx-vlm loader modules to this thread/directory.

    These loaders have no weight-loader callback. A temporary module proxy
    preserves their sanitization and construction paths while leaving the
    process-wide mlx.core.load function unchanged. Other threads/directories,
    formats and explicit streams still use the existing loader.
    """
    if not enabled:
        yield
        return
    if Version(version("mlx")) < Version("0.32.4"):
        raise RuntimeError(
            "Mapped weights require MLX >= 0.32.4 (foreign-buffer ownership fix)"
        )
    directory = Path(model_path).resolve()
    owner = threading.get_ident()

    def wrap(original):
        def load(file, *args, **kwargs):
            path = Path(file) if isinstance(file, (str, os.PathLike)) else None
            fmt = kwargs.get("format", args[0] if args else None)
            if (
                threading.get_ident() != owner
                or path is None
                or path.suffix != ".safetensors"
                or path.parent.resolve() != directory
                or fmt not in (None, "safetensors")
                or kwargs.get("stream") is not None
                or len(args) > 2
                or set(kwargs) - {"format", "return_metadata", "stream"}
            ):
                return original(file, *args, **kwargs)
            try:
                weights, metadata = _load_mapped(path)
            except _NativeLoaderError:
                return original(file, *args, **kwargs)
            return_metadata = kwargs.get(
                "return_metadata", args[1] if len(args) > 1 else False
            )
            return (weights, metadata) if return_metadata else weights

        return load

    with _load_lock:
        saved = []
        try:
            for name in ("mlx_lm.utils", "mlx_vlm.utils"):
                module = importlib.import_module(name)
                original = module.mx
                saved.append((module, original))
                module.mx = _LoaderProxy(original, wrap(original.load))
            yield
        finally:
            for module, original in reversed(saved):
                module.mx = original
