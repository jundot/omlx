# SPDX-License-Identifier: Apache-2.0
"""Small row-addressable safetensors reader used by expert streaming."""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Hashable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")

_NUMPY_DTYPES = {
    "U32": np.uint32,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
}


@dataclass(frozen=True)
class TensorLocation:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    row_bytes: int

    @property
    def tensor_bytes(self) -> int:
        return self.row_bytes * self.shape[0]


class SafetensorExpertIndex:
    """Index stacked ``switch_mlp`` tensors without mapping their payloads."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path).expanduser().resolve()
        index_path = self.model_path / "model.safetensors.index.json"
        if not index_path.is_file():
            raise ValueError("Expert streaming requires model.safetensors.index.json")
        try:
            self.weight_map = json.loads(index_path.read_text())["weight_map"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid safetensors index: {exc}") from exc
        self._headers: dict[str, tuple[int, dict]] = {}
        self._layer_locations: dict[int, dict[tuple[str, str], TensorLocation]] = {}

    def _header(self, filename: str) -> tuple[int, dict]:
        cached = self._headers.get(filename)
        if cached is not None:
            return cached
        path = self.model_path / filename
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ValueError(f"Invalid safetensors header: {path}")
            header_size = struct.unpack("<Q", raw)[0]
            header = json.loads(handle.read(header_size))
        cached = (8 + header_size, header)
        self._headers[filename] = cached
        return cached

    def layer(self, layer: int) -> dict[tuple[str, str], TensorLocation]:
        cached = self._layer_locations.get(layer)
        if cached is not None:
            return cached
        marker = f"layers.{layer}.mlp.switch_mlp."
        locations: dict[tuple[str, str], TensorLocation] = {}
        for name, filename in self.weight_map.items():
            # MTP checkpoints can contain ``mtp.layers.0`` alongside the
            # backbone's real layer 0. Soft-REAP manifests describe backbone
            # experts only, so never let the auxiliary predictor overwrite
            # those locations.
            if marker not in name or name.startswith("mtp.") or ".mtp." in name:
                continue
            for projection in PROJECTIONS:
                for part in PARTS:
                    if not name.endswith(f".{projection}.{part}"):
                        continue
                    data_base, header = self._header(filename)
                    meta = header.get(name)
                    if meta is None:
                        raise ValueError(f"Tensor {name} is absent from {filename}")
                    dtype = str(meta["dtype"])
                    if dtype not in _NUMPY_DTYPES:
                        raise ValueError(
                            f"Unsupported streamed tensor dtype {dtype}: {name}"
                        )
                    shape = tuple(int(value) for value in meta["shape"])
                    if len(shape) < 2 or shape[0] <= 0:
                        raise ValueError(
                            f"Expert tensor is not row-addressable: {name}"
                        )
                    start, end = (int(value) for value in meta["data_offsets"])
                    total = end - start
                    if total % shape[0]:
                        raise ValueError(f"Expert tensor rows are uneven: {name}")
                    locations[(projection, part)] = TensorLocation(
                        name=name,
                        path=self.model_path / filename,
                        dtype=dtype,
                        shape=shape,
                        data_start=data_base + start,
                        row_bytes=total // shape[0],
                    )
        required = {(projection, part) for projection in PROJECTIONS for part in PARTS}
        missing = required - set(locations)
        if missing:
            raise ValueError(
                f"Layer {layer} lacks stacked quantized expert tensors: {sorted(missing)}"
            )
        self._layer_locations[layer] = locations
        return locations

    def expert_bytes(self, layer: int) -> int:
        return sum(location.row_bytes for location in self.layer(layer).values())

    def tensor_bytes(self, layers: list[int] | tuple[int, ...]) -> int:
        return sum(
            location.tensor_bytes
            for layer in layers
            for location in self.layer(layer).values()
        )


class ExpertReader:
    """Read selected expert rows with ``pread`` and bounded parallelism."""

    def __init__(self, index: SafetensorExpertIndex, workers: int = 9):
        self.index = index
        self._fds: dict[tuple[Path, bool], int] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, workers), thread_name_prefix="expert-ssd"
        )
        self.bytes_read = 0
        self.direct_bytes_read = 0
        self.file_cache_bytes_read = 0
        self.read_operations = 0
        self.direct_read_operations = 0
        self.file_cache_read_operations = 0
        self._closed = False

    def _fd(self, path: Path, *, use_file_cache: bool) -> int:
        key = (path, use_file_cache)
        fd = self._fds.get(key)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[key] = fd
            no_cache = getattr(os, "F_NOCACHE", None)
            if not use_file_cache and no_cache is not None:
                try:
                    import fcntl

                    fcntl.fcntl(fd, no_cache, 1)
                except OSError:
                    pass
        return fd

    @staticmethod
    def _ranges(
        rows: list[int], row_bytes: int, max_gap_bytes: int
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start = previous = rows[0]
        for row in rows[1:]:
            gap = (row - previous - 1) * row_bytes
            if gap > max_gap_bytes:
                ranges.append((start, previous + 1))
                start = row
            previous = row
        ranges.append((start, previous + 1))
        return ranges

    def read_rows(
        self,
        location: TensorLocation,
        expert_ids: list[int] | tuple[int, ...],
        *,
        max_gap_bytes: int = 0,
        use_file_cache: bool = False,
    ) -> mx.array:
        return self.read_many(
            {"rows": location},
            expert_ids,
            max_gap_bytes=max_gap_bytes,
            use_file_cache=use_file_cache,
        )["rows"]

    def read_many(
        self,
        locations: Mapping[Hashable, TensorLocation],
        expert_ids: list[int] | tuple[int, ...],
        *,
        max_gap_bytes: int = 0,
        use_file_cache: bool = False,
    ) -> dict[Hashable, mx.array]:
        """Read several expert components through one flat parallel I/O queue."""

        requested = [int(value) for value in expert_ids]
        if not requested:
            raise ValueError("At least one expert row must be requested")
        rows = sorted(set(requested))
        tasks: list[tuple[Hashable, TensorLocation, int, int, int]] = []
        for key, location in locations.items():
            if min(requested) < 0 or max(requested) >= location.shape[0]:
                raise ValueError(f"Expert row outside tensor {location.name}")
            fd = self._fd(location.path, use_file_cache=use_file_cache)
            for first, stop in self._ranges(rows, location.row_bytes, max_gap_bytes):
                tasks.append((key, location, fd, first, stop))

        def read_range(task) -> tuple[Hashable, int, bytes]:
            key, location, fd, first, stop = task
            size = (stop - first) * location.row_bytes
            payload = os.pread(
                fd, size, location.data_start + first * location.row_bytes
            )
            if len(payload) != size:
                raise OSError(f"Short expert read from {location.path.name}")
            return key, first, payload

        chunks = list(self._pool.map(read_range, tasks))
        bytes_read = sum(len(payload) for _, _, payload in chunks)
        operations = len(chunks)
        self.bytes_read += bytes_read
        self.read_operations += operations
        if use_file_cache:
            self.file_cache_bytes_read += bytes_read
            self.file_cache_read_operations += operations
        else:
            self.direct_bytes_read += bytes_read
            self.direct_read_operations += operations
        grouped: dict[Hashable, list[tuple[int, bytes]]] = {
            key: [] for key in locations
        }
        for key, first, payload in chunks:
            grouped[key].append((first, payload))

        result: dict[Hashable, mx.array] = {}
        for key, location in locations.items():
            row_shape = location.shape[1:]
            dtype = _NUMPY_DTYPES[location.dtype]
            selected: dict[int, np.ndarray] = {}
            for first, payload in grouped[key]:
                count = len(payload) // location.row_bytes
                values = np.frombuffer(payload, dtype=dtype).reshape(
                    (count, *row_shape)
                )
                for row in rows:
                    if first <= row < first + count:
                        selected[row] = values[row - first]
            output = np.stack([selected[row] for row in requested])
            array = mx.array(output)
            if location.dtype == "BF16":
                array = array.view(mx.bfloat16)
            result[key] = array
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
