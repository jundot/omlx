# SPDX-License-Identifier: Apache-2.0
"""Selected-row Engram I/O, following Qwen4-Exp's _SafeTensorMMap pattern.

DeepSeek uses per-row/per-32-channel scales instead of Qwen's shared scale.
Copies leave the mapping under the lock, so close cannot invalidate a live view.
Resident tables retain their packed bytes; prefetch workers only copy CPU rows.
"""

import fcntl
import json
import math
import mmap
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

import mlx.core as mx
import mlx.nn as nn
import numpy as np

RESIDENT_READ_BYTES = 8 * 1024 * 1024
# safetensors dtype tag -> numpy transport dtype (bf16 travels as raw uint16).
SAFETENSORS_NUMPY_DTYPES = {
    "BF16": "<u2",
    "F16": "<f2",
    "F32": "<f4",
    "U32": "<u4",
    "U8": "u1",
    "I8": "i1",
    "F8_E4M3": "u1",
    "F8_E8M0": "u1",
    "F8_E4M3FN": "u1",
    "F8_E8M0FNU": "u1",
}
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
PAGE_PREFETCH_MIN_ROWS = 128
PAGE_IO_WORKERS = 48
_PAGE_IO_POOL = ThreadPoolExecutor(
    max_workers=PAGE_IO_WORKERS, thread_name_prefix="v41-page-io"
)


def _resident_buffer(shape, dtype):
    """Share packed Metal storage with CPU gathers, without a host copy."""
    types = {
        np.dtype("<u4"): mx.uint32,
        np.dtype("<u2"): mx.uint16,
        np.dtype("u1"): mx.uint8,
        np.dtype("i1"): mx.int8,
        np.dtype("<f4"): mx.float32,
        np.dtype("<f2"): mx.float16,
    }
    # Materialize on the GPU before exposing a NumPy view. Otherwise zeros
    # remains a scalar broadcast and NumPy materializes unwired CPU storage.
    value = mx.contiguous(mx.zeros(shape, dtype=types[dtype]))
    mx.eval(value)
    # Submit after allocation so newly created residency sets are attached.
    corner = tuple(slice(0, 1) for _ in shape)
    mx.eval(mx.sum(value[corner].astype(mx.float32)))
    mx.synchronize()
    return np.asarray(value)


class TensorFile:
    def __init__(self, path):
        self._lock = RLock()
        self._file = Path(path).open("rb")  # noqa: SIM115 -- owned until close()
        try:
            length = struct.unpack("<Q", self._file.read(8))[0]
            self.header = json.loads(self._file.read(length))
            self._start = length + 8
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self._file_size = os.fstat(self._file.fileno()).st_size
            self._seen_pages = None
            self._last_rearm = 0.0
            # Engram hashes select sparse rows throughout the table. Whole-tensor
            # resident reads use readinto below and do not use this mapping.
            if hasattr(self._mapping, "madvise"):
                self._mapping.madvise(mmap.MADV_RANDOM)
        except Exception:
            mapping = getattr(self, "_mapping", None)
            if mapping is not None:
                mapping.close()
            self._file.close()
            raise

    def read(self, key, rows=None, *, metal_backed=False):
        with self._lock:
            if self._mapping is None:
                raise RuntimeError("Engram tensor file is closed")
            entry = self.header[key]
            dtype = entry["dtype"]
            if dtype not in SAFETENSORS_NUMPY_DTYPES:
                raise ValueError(f"Unsupported source tensor dtype: {dtype}")
            dt = np.dtype(SAFETENSORS_NUMPY_DTYPES[dtype])
            start, end = entry["data_offsets"]
            if end - start != math.prod(entry["shape"]) * dt.itemsize:
                raise ValueError(f"Invalid tensor byte length: {key}")
            view = np.ndarray(
                entry["shape"],
                dtype=dt,
                buffer=self._mapping,
                offset=self._start + start,
            )
            if rows is None:
                # Avoid faulting a whole mmap alongside its resident copy.
                # Read directly into the destination with bounded I/O requests.
                result = (
                    _resident_buffer(view.shape, dt)
                    if metal_backed and view.size
                    else np.empty(view.shape, dtype=dt)
                )
                if not result.size:
                    return result, dtype
                target = memoryview(result).cast("B")
                # Resident tensors already have a full host-memory copy. Avoid
                # retaining a second copy in the macOS unified file cache.
                nocache = getattr(fcntl, "F_NOCACHE", None)
                if nocache is not None:
                    fcntl.fcntl(self._file.fileno(), nocache, 1)
                try:
                    self._file.seek(self._start + start)
                    offset = 0
                    while offset < len(target):
                        stop = min(offset + RESIDENT_READ_BYTES, len(target))
                        count = self._file.readinto(target[offset:stop])
                        if not count:
                            raise ValueError(f"Truncated tensor data: {key}")
                        offset += count
                finally:
                    if nocache is not None:
                        fcntl.fcntl(self._file.fileno(), nocache, 0)
                return result, dtype
            if rows is not None:
                rows = np.asarray(rows, dtype=np.intp)
                if rows.size and (rows.min() < 0 or rows.max() >= view.shape[0]):
                    raise IndexError("Engram row outside table")
                gather_start = None
                row_bytes = (end - start) // view.shape[0] if view.shape[0] else 0
                if (
                    rows.size > PAGE_PREFETCH_MIN_ROWS
                    and 0 < row_bytes <= PAGE_SIZE
                    and self._prefetch_pages(rows, self._start + start, row_bytes)
                ):
                    gather_start = time.perf_counter()
                copied = view[rows]  # Advanced indexing already copies the rows.
                if gather_start is not None:
                    elapsed = time.perf_counter() - gather_start
                    now = time.monotonic()
                    # As in Qwen4-Exp, a slow warm gather can indicate eviction.
                    # This only changes I/O scheduling, never the selected rows.
                    if (
                        elapsed > 0.0005 + 2e-6 * rows.size
                        and now - self._last_rearm >= 60
                    ):
                        self._seen_pages = None
                        self._last_rearm = now
                return copied, dtype
            return np.array(view, copy=True), dtype

    def _prefetch_pages(self, rows, base, row_bytes):
        """Read unseen pages concurrently before gathering from the mmap."""
        if self._seen_pages is None:
            self._seen_pages = np.zeros(
                (self._file_size + PAGE_SIZE - 1) // PAGE_SIZE, dtype=np.uint8
            )
        offsets = base + rows.reshape(-1) * row_bytes
        pages = np.unique(
            np.concatenate(
                (offsets // PAGE_SIZE, (offsets + row_bytes - 1) // PAGE_SIZE)
            )
        )
        fresh = pages[self._seen_pages[pages] == 0]
        if not fresh.size:
            return True
        fd = self._file.fileno()

        def touch(group):
            for page in group:
                offset = int(page) * PAGE_SIZE
                remaining = min(PAGE_SIZE, self._file_size - offset)
                while remaining:
                    data = os.pread(fd, remaining, offset)
                    if not data:
                        raise ValueError("Truncated Engram page")
                    offset += len(data)
                    remaining -= len(data)

        # Bound the queue as well as the active reads. The caller holds _lock;
        # drain every worker, including on error, before close can release fd.
        futures = []
        try:
            for group in np.array_split(fresh, min(PAGE_IO_WORKERS, fresh.size)):
                futures.append(_PAGE_IO_POOL.submit(touch, group))
        finally:
            wait(futures)
        for future in futures:
            future.result()
        self._seen_pages[fresh] = 1
        return False

    def close(self):
        with self._lock:
            self._seen_pages = None
            if self._mapping is not None:
                self._mapping.close()
                self._mapping = None
            self._file.close()


def decode_array(raw, dtype):
    if dtype == "BF16":
        return mx.array((raw.astype(np.uint32) << 16).view(np.float32)).astype(
            mx.bfloat16
        )
    if dtype.startswith("F8_E4M3"):
        return mx.from_fp8(mx.array(raw), dtype=mx.float32)
    if dtype.startswith("F8_E8M0"):
        if np.any(raw == 255):
            raise ValueError("NaN E8M0 scale in checkpoint")
        return mx.array(np.exp2(raw.astype(np.float32) - 127))
    return mx.array(raw)


class DiskEngramEmbedding(nn.Module):
    def __init__(
        self,
        path,
        weight_key,
        scale_key,
        scale_path=None,
        *,
        bias_key=None,
        bits=None,
        group_size=32,
    ):
        super().__init__()
        self._lock = RLock()
        self._weights = TensorFile(path)
        try:
            self._scales = (
                self._weights
                if scale_path is None or Path(scale_path) == Path(path)
                else TensorFile(scale_path)
            )
        except Exception:
            self._weights.close()
            raise
        self._weight_key, self._scale_key = weight_key, scale_key
        self._bias_key, self._bits, self._group_size = bias_key, bits, group_size
        if (bits is None) != (bias_key is None) or (
            bits is not None and scale_key is None
        ):
            self._weights.close()
            self._scales.close()
            raise ValueError("Affine Engram requires bits, scales and biases")
        self._closed = False
        self._resident = None
        self._prefetched = None

    def make_resident(self):
        """Keep packed tensors in Metal-managed RAM with shared CPU views."""
        with self._lock:
            resident = {
                self._weight_key: self._weights.read(
                    self._weight_key, metal_backed=True
                ),
            }
            if self._scale_key is not None:
                resident[self._scale_key] = self._scales.read(
                    self._scale_key, metal_backed=True
                )
            if self._bias_key is not None:
                resident[self._bias_key] = self._scales.read(
                    self._bias_key, metal_backed=True
                )
            self._resident = resident
            self._weights.close()
            if self._scales is not self._weights:
                self._scales.close()

    def selected_bytes(self, rows):
        total = 0
        for reader, key in (
            (self._weights, self._weight_key),
            (self._scales, self._scale_key),
            (self._scales, self._bias_key),
        ):
            if key is not None:
                entry = reader.header[key]
                start, end = entry["data_offsets"]
                total += (end - start) // entry["shape"][0] * rows
        return total

    def _read_rows(self, host):
        with self._lock:
            if self._closed:
                raise RuntimeError("Engram embedding is closed")
            result = []
            for reader, key in (
                (self._weights, self._weight_key),
                (self._scales, self._scale_key),
                (self._scales, self._bias_key),
            ):
                if key is None:
                    continue
                if self._resident is None:
                    result.append(reader.read(key, host.reshape(-1)))
                else:
                    raw, dtype = self._resident[key]
                    rows = host.reshape(-1)
                    if rows.size and (rows.min() < 0 or rows.max() >= raw.shape[0]):
                        raise IndexError("Engram row outside table")
                    result.append((raw[rows], dtype))
            return result

    def __call__(self, indices):
        host = np.asarray(indices).astype(np.int64)
        pending, self._prefetched = self._prefetched, None
        if pending is not None:
            requested, future = pending
            data = future.result()
            if not np.array_equal(requested, host):
                data = self._read_rows(host)
        else:
            data = self._read_rows(host)
        values = decode_array(*data[0])
        if self._bits is not None:
            values = mx.dequantize(
                values,
                decode_array(*data[1]),
                decode_array(*data[2]),
                bits=self._bits,
                group_size=self._group_size,
                mode="affine",
            )
        elif self._scale_key is not None:
            scales = decode_array(*data[1])
            values = (
                values.reshape(values.shape[0], -1, 32) * scales[..., None]
            ).reshape(values.shape)
        return values.reshape(*host.shape, values.shape[-1]).astype(mx.bfloat16)

    def close(self):
        with self._lock:
            self._closed = True
            self._resident = None
            self._weights.close()
            if self._scales is not self._weights:
                self._scales.close()


PREFETCH_BYTES = 16 * 1024 * 1024


class EngramPrefetch:
    """One pending CPU read per model, drained at each forward boundary."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="v41-engram"
        )
        self._pending = None
        self._closed = False

    def submit(self, embed, ids):
        self.drain()
        if self._closed:
            raise RuntimeError("Engram prefetch is closed")
        if not isinstance(embed, DiskEngramEmbedding) or embed._resident is not None:
            return
        host = np.asarray(ids, dtype=np.int64).copy()
        if embed.selected_bytes(host.size) + host.nbytes > PREFETCH_BYTES:
            return
        future = self._executor.submit(embed._read_rows, host)
        embed._prefetched = (host, future)
        self._pending = (embed, future)

    def drain(self):
        pending, self._pending = self._pending, None
        if pending is not None:
            embed, future = pending
            embed._prefetched = None
            if not future.cancel():
                future.result()

    @contextmanager
    def forward(self):
        try:
            yield self
        finally:
            self.drain()

    def close(self):
        self._closed = True
        try:
            self.drain()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
