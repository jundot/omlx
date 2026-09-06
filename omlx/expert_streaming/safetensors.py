# SPDX-License-Identifier: Apache-2.0
"""Small row-addressable safetensors reader used by expert streaming."""

from __future__ import annotations

import json
import os
import re
import struct
import time
from collections.abc import Hashable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from .fast_resource import (
        FastExpertLoad,
        FastExpertLoader,
        FastExpertLoadFuture,
    )

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")

_NUMPY_DTYPES = {
    "U32": np.uint32,
    "U8": np.uint8,
    "I8": np.int8,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
}

_MLX_DTYPES = {
    "U32": mx.uint32,
    "U8": mx.uint8,
    "I8": mx.int8,
    "F32": mx.float32,
    "F16": mx.float16,
    "BF16": mx.bfloat16,
}


@dataclass(frozen=True)
class TensorRowSource:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_bytes: int


@dataclass(frozen=True)
class TensorLocation:
    name: str
    path: Path
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    row_bytes: int
    storage_dtype: str | None = None
    storage_shape: tuple[int, ...] | None = None
    storage_row_bytes: int | None = None
    row_sources: tuple[TensorRowSource, ...] | None = None
    output_slice: tuple[int, int] | None = None

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
        self._layer_locations: dict[
            tuple[int, str], dict[tuple[str, str], TensorLocation]
        ] = {}

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

    @staticmethod
    def _mlx_dtype_name(array) -> str:
        if array.dtype == mx.uint32:
            return "U32"
        if array.dtype == mx.uint8:
            return "U8"
        if array.dtype == mx.int8:
            return "I8"
        if array.dtype == mx.float32:
            return "F32"
        if array.dtype == mx.float16:
            return "F16"
        if array.dtype == mx.bfloat16:
            return "BF16"
        raise ValueError(f"Unsupported streamed MLX dtype: {array.dtype}")

    @staticmethod
    def _projection_aliases(projection: str) -> tuple[str, ...]:
        aliases = {
            "gate_proj": ("gate_proj", "w1"),
            "down_proj": ("down_proj", "w2"),
            "up_proj": ("up_proj", "w3"),
        }
        return aliases[projection]

    @staticmethod
    def _part_aliases(part: str) -> tuple[str, ...]:
        return ("scales", "scale") if part == "scales" else (part,)

    def _source(self, name: str, filename: str) -> TensorRowSource:
        data_base, header = self._header(filename)
        meta = header.get(name)
        if meta is None:
            raise ValueError(f"Tensor {name} is absent from {filename}")
        dtype = str(meta["dtype"])
        if dtype not in _NUMPY_DTYPES:
            raise ValueError(f"Unsupported streamed tensor dtype {dtype}: {name}")
        shape = tuple(int(value) for value in meta["shape"])
        start, end = (int(value) for value in meta["data_offsets"])
        return TensorRowSource(
            name=name,
            path=self.model_path / filename,
            dtype=dtype,
            shape=shape,
            data_start=data_base + start,
            data_bytes=end - start,
        )

    @staticmethod
    def _is_backbone_expert_name(name: str, layer: int) -> bool:
        if name.startswith("mtp.") or ".mtp." in name:
            return False
        if f"layers.{layer}." not in name or ".shared_experts." in name:
            return False
        return ".switch_mlp." in name or ".experts." in name

    def expert_layer_ids(self) -> list[int]:
        result: set[int] = set()
        for name in self.weight_map:
            if name.startswith("mtp.") or ".mtp." in name or ".shared_experts." in name:
                continue
            match = re.search(r"(?:^|\.)layers\.(\d+)\.(?:mlp|ffn)\.", name)
            if match and (".switch_mlp." in name or ".experts." in name):
                result.add(int(match.group(1)))
        return sorted(result)

    def _tensor_storage_bytes(self, name: str, filename: str) -> int:
        return self._source(name, filename).data_bytes

    def streamed_storage_bytes(self, layer: int) -> int:
        """Checkpoint bytes occupied by all routed tensors in one layer."""
        names = [
            (name, filename)
            for name, filename in self.weight_map.items()
            if self._is_backbone_expert_name(name, layer)
        ]
        stacked = [item for item in names if ".switch_mlp." in item[0]]
        selected = stacked or [item for item in names if ".experts." in item[0]]
        return sum(self._tensor_storage_bytes(*item) for item in selected)

    def expert_storage_bytes(self, layer: int) -> int:
        """Resident bytes for one routed expert, without loading payloads."""
        total = self.streamed_storage_bytes(layer)
        stacked_experts = None
        for name, filename in self.weight_map.items():
            if self._is_backbone_expert_name(name, layer) and ".switch_mlp." in name:
                source = self._source(name, filename)
                if source.shape:
                    stacked_experts = source.shape[0]
                    break
        if stacked_experts:
            return total // stacked_experts
        expert_ids: set[int] = set()
        marker = re.compile(rf"layers\.{layer}\.(?:mlp|ffn)\.experts\.(\d+)\.")
        for name in self.weight_map:
            match = marker.search(name)
            if match and self._is_backbone_expert_name(name, layer):
                expert_ids.add(int(match.group(1)))
        if not expert_ids:
            raise ValueError(f"Layer {layer} has no indexed routed experts")
        return total // len(expert_ids)

    def _stacked_location(
        self,
        *,
        layer: int,
        container_name: str,
        projection: str,
        part: str,
        metadata: dict,
        num_experts: int,
    ) -> TensorLocation | None:
        source_projection = str(metadata.get("source_projection", projection))
        projection_names = [source_projection]
        if source_projection == "gate_up_proj":
            projection_names.extend(self._projection_aliases(projection))
        else:
            projection_names.extend(self._projection_aliases(projection))
        seen: set[str] = set()
        projection_names = [
            value for value in projection_names if not (value in seen or seen.add(value))
        ]
        for name, filename in self.weight_map.items():
            if name.startswith("mtp.") or ".mtp." in name:
                continue
            if f"layers.{layer}.{container_name}.switch_mlp." not in name:
                continue
            matched_projection = next(
                (
                    candidate
                    for candidate in projection_names
                    if any(
                        name.endswith(f".{candidate}.{alias}")
                        for alias in self._part_aliases(part)
                    )
                ),
                None,
            )
            if matched_projection is None:
                continue
            source = self._source(name, filename)
            if len(source.shape) < 2 or source.shape[0] != num_experts:
                raise ValueError(f"Expert tensor is not row-addressable: {name}")
            output_slice = None
            logical_shape = tuple(int(v) for v in metadata["arrays"][part].shape)
            if metadata.get("fused_half"):
                logical_shape = (
                    logical_shape[0],
                    logical_shape[1] // 2,
                    *logical_shape[2:],
                )
            if matched_projection == "gate_up_proj":
                midpoint = source.shape[1] // 2
                output_slice = (
                    (0, midpoint)
                    if metadata.get("fused_half") == "gate"
                    else (midpoint, source.shape[1])
                )
            logical_dtype = self._mlx_dtype_name(metadata["arrays"][part])
            logical_bytes = (
                int(np.prod(logical_shape[1:]))
                * np.dtype(_NUMPY_DTYPES[logical_dtype]).itemsize
            )
            return TensorLocation(
                name=name,
                path=source.path,
                dtype=logical_dtype,
                shape=logical_shape,
                data_start=source.data_start,
                row_bytes=logical_bytes,
                storage_dtype=source.dtype,
                storage_shape=source.shape,
                storage_row_bytes=source.data_bytes // num_experts,
                output_slice=output_slice,
            )
        return None

    def _fragmented_location(
        self,
        *,
        layer: int,
        container_name: str,
        projection: str,
        part: str,
        metadata: dict,
        num_experts: int,
    ) -> TensorLocation | None:
        sources: list[TensorRowSource] = []
        aliases = self._projection_aliases(projection)
        for expert in range(num_experts):
            marker = f"layers.{layer}.{container_name}.experts.{expert}."
            found = None
            for name, filename in self.weight_map.items():
                if name.startswith("mtp.") or ".mtp." in name or marker not in name:
                    continue
                if any(
                    name.endswith(f".{candidate}.{part_alias}")
                    for candidate in aliases
                    for part_alias in self._part_aliases(part)
                ):
                    found = self._source(name, filename)
                    break
            if found is None:
                return None
            sources.append(found)
        logical_array = metadata["arrays"][part]
        logical_dtype = self._mlx_dtype_name(logical_array)
        logical_shape = tuple(int(value) for value in logical_array.shape)
        if metadata.get("fused_half"):
            logical_shape = (
                logical_shape[0],
                logical_shape[1] // 2,
                *logical_shape[2:],
            )
        row_bytes = (
            int(np.prod(logical_shape[1:]))
            * np.dtype(_NUMPY_DTYPES[logical_dtype]).itemsize
        )
        for source in sources:
            if source.data_bytes != row_bytes:
                raise ValueError(
                    f"Expert tensor {source.name} has {source.data_bytes} bytes; "
                    f"expected {row_bytes}"
                )
        return TensorLocation(
            name=f"layers.{layer}.{container_name}.experts.*.{projection}.{part}",
            path=sources[0].path,
            dtype=logical_dtype,
            shape=logical_shape,
            data_start=0,
            row_bytes=row_bytes,
            row_sources=tuple(sources),
        )

    def layer(
        self,
        layer: int,
        *,
        container_name: str = "mlp",
        schema: dict[str, dict] | None = None,
        num_experts: int | None = None,
    ) -> dict[tuple[str, str], TensorLocation]:
        cache_key = (layer, container_name)
        cached = self._layer_locations.get(cache_key)
        if cached is not None:
            return cached
        if schema is not None:
            if num_experts is None:
                raise ValueError("A projection schema requires the expert count")
            locations: dict[tuple[str, str], TensorLocation] = {}
            for projection in PROJECTIONS:
                metadata = schema[projection]
                for part in metadata["parts"]:
                    location = self._stacked_location(
                        layer=layer,
                        container_name=container_name,
                        projection=projection,
                        part=part,
                        metadata=metadata,
                        num_experts=num_experts,
                    ) or self._fragmented_location(
                        layer=layer,
                        container_name=container_name,
                        projection=projection,
                        part=part,
                        metadata=metadata,
                        num_experts=num_experts,
                    )
                    if location is None:
                        raise ValueError(
                            f"Layer {layer} lacks {container_name} expert tensor "
                            f"{projection}.{part}"
                        )
                    locations[(projection, part)] = location
            self._layer_locations[cache_key] = locations
            return locations

        marker = f"layers.{layer}.{container_name}.switch_mlp."
        locations: dict[tuple[str, str], TensorLocation] = {}
        for name, filename in self.weight_map.items():
            # MTP checkpoints can contain ``mtp.layers.0`` alongside the
            # backbone's real layer 0. Streaming manifests describe backbone
            # experts only, so never let the auxiliary MTP head overwrite
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
                        storage_dtype=dtype,
                        storage_shape=shape,
                        storage_row_bytes=total // shape[0],
                    )
        required = {
            (projection, part)
            for projection in PROJECTIONS
            for part in ("weight", "scales")
        }
        missing = required - set(locations)
        if missing:
            raise ValueError(
                f"Layer {layer} lacks stacked quantized expert tensors: {sorted(missing)}"
            )
        self._layer_locations[cache_key] = locations
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

    def __init__(
        self,
        index: SafetensorExpertIndex,
        workers: int = 9,
        *,
        cold_max_gap_bytes: int = 64 * 1024,
        fast_resource_loading: bool | str = False,
        direct_io: bool = True,
    ):
        self.index = index
        self.cold_max_gap_bytes = max(0, int(cold_max_gap_bytes))
        self._fds: dict[tuple[Path, bool], int] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, workers), thread_name_prefix="expert-ssd"
        )
        self._request_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="expert-prefetch"
        )
        self.bytes_read = 0
        self.direct_bytes_read = 0
        self.file_cache_bytes_read = 0
        self.read_operations = 0
        self.direct_read_operations = 0
        self.file_cache_read_operations = 0
        self.io_seconds = 0.0
        self.decode_seconds = 0.0
        self.readahead_descriptors = 0
        self.no_cache_descriptors = 0
        self._closed = False
        self.fast_resource_loading = False
        self.fast_resource_loading_scope = "off"
        self._fast_loader: FastExpertLoader | None = None
        self.fast_loads = 0
        self.fast_bytes_read = 0
        self.fast_read_operations = 0
        self.fast_io_wait_seconds = 0.0
        self.fast_copy_seconds = 0.0
        self.fast_queue_acquire_seconds = 0.0
        self.fast_submission_to_completion_seconds = 0.0
        self.fast_overlapped_submissions = 0
        self.fast_max_inflight = 0
        self.fast_direct_loads = 0
        self.fast_direct_bytes = 0
        self.fast_direct_exact = False
        scope = (
            "all"
            if fast_resource_loading is True
            else str(fast_resource_loading).lower()
        )
        if scope not in {"off", "false", "scratch", "all"}:
            raise ValueError("Fast Resource Loading scope must be off, scratch, or all")
        if scope in {"scratch", "all"}:
            from .fast_resource import FastExpertLoader

            self._fast_loader = FastExpertLoader()
            self.fast_resource_loading = True
            self.fast_resource_loading_scope = scope
            self.fast_direct_exact = (
                bool(direct_io) and self._fast_loader.direct_available
            )

    def _record_fast_stats(self, stats: Mapping[str, int | float]) -> None:
        self.fast_loads += 1
        self.fast_bytes_read += int(stats["bytes"])
        self.fast_read_operations += int(stats["commands"])
        self.fast_io_wait_seconds += float(stats["io_wait_seconds"])
        self.fast_copy_seconds += float(stats.get("copy_seconds", 0.0))
        self.fast_queue_acquire_seconds += float(
            stats.get("queue_acquire_seconds", 0.0)
        )
        self.fast_submission_to_completion_seconds += float(
            stats.get("submission_to_completion_seconds", 0.0)
        )
        self.fast_overlapped_submissions += int(
            bool(stats.get("overlapped_submission", False))
        )
        self.fast_max_inflight = max(
            self.fast_max_inflight, int(stats.get("queue_max_inflight", 0))
        )
        self.bytes_read += int(stats["bytes"])
        self.read_operations += int(stats["commands"])
        self.file_cache_bytes_read += int(stats["bytes"])
        self.file_cache_read_operations += int(stats["commands"])

    def begin_fast_many(
        self,
        locations: Mapping[Hashable, TensorLocation],
        expert_ids: list[int] | tuple[int, ...],
    ) -> FastExpertLoad:
        if self._fast_loader is None:
            raise RuntimeError("Fast Resource Loading is not enabled")
        return self._fast_loader.begin(
            locations,
            expert_ids,
            max_gap_bytes=self.cold_max_gap_bytes,
        )

    @staticmethod
    def fast_resource_compatible(
        locations: Mapping[Hashable, TensorLocation],
    ) -> bool:
        """Return whether FRL can publish the checkpoint rows.

        Fast Resource Loading is a byte-copy path.  Some model sanitizers
        deliberately change expert metadata dtype while loading (DeepSeek V4
        converts routed affine scales/biases from checkpoint BF16 to runtime
        FP16).  Those rows require a numeric conversion and must not be
        published by a raw MTLIO blit.
        """

        supported = {("BF16", "F16")}
        return all(
            (storage := location.storage_dtype or location.dtype) == location.dtype
            or (storage, location.dtype) in supported
            for location in locations.values()
        )

    @staticmethod
    def fast_direct_compatible(
        locations: Mapping[Hashable, TensorLocation],
    ) -> bool:
        """Return whether direct MTLIO can byte-copy rows without conversion."""

        return all(
            (location.storage_dtype or location.dtype) == location.dtype
            for location in locations.values()
        )

    def finish_fast_many(
        self,
        load: FastExpertLoad,
        slots: list[int],
        targets: Mapping[Hashable, tuple[mx.array, int]],
        *,
        destinations_idle: bool = False,
    ) -> dict[str, int | float]:
        if self._fast_loader is None:
            raise RuntimeError("Fast Resource Loading is not enabled")
        stats = self._fast_loader.finish_into(
            load,
            slots,
            targets,
            destinations_idle=destinations_idle,
        )
        self._record_fast_stats(stats)
        if bool(stats.get("direct", False)):
            self.fast_direct_loads += 1
            self.fast_direct_bytes += int(stats["bytes"])
        return stats

    def begin_fast_direct_many(
        self,
        locations: Mapping[Hashable, TensorLocation],
        expert_ids: list[int] | tuple[int, ...],
        slots: list[int],
        targets: Mapping[Hashable, tuple[mx.array, int]],
    ) -> FastExpertLoad:
        if self._fast_loader is None:
            raise RuntimeError("Fast Resource Loading is not enabled")
        return self._fast_loader.begin_direct(
            locations, expert_ids, slots, targets
        )

    def _fd(self, path: Path, *, use_file_cache: bool) -> int:
        key = (path, use_file_cache)
        fd = self._fds.get(key)
        if fd is None:
            fd = os.open(path, os.O_RDONLY)
            self._fds[key] = fd
            try:
                import fcntl

                if use_file_cache:
                    read_ahead = getattr(fcntl, "F_RDAHEAD", None)
                    if read_ahead is not None:
                        fcntl.fcntl(fd, read_ahead, 1)
                        self.readahead_descriptors += 1
                else:
                    no_cache = getattr(
                        fcntl, "F_NOCACHE", getattr(os, "F_NOCACHE", None)
                    )
                    if no_cache is not None:
                        fcntl.fcntl(fd, no_cache, 1)
                        self.no_cache_descriptors += 1
            except OSError:
                # These are advisory Darwin hints. Unsupported filesystems and
                # non-Darwin platforms should retain ordinary pread semantics.
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
        max_gap_bytes: int | None = None,
        use_file_cache: bool = False,
    ) -> mx.array:
        return self.read_many(
            {"rows": location},
            expert_ids,
            max_gap_bytes=max_gap_bytes,
            use_file_cache=use_file_cache,
        )["rows"]

    def _read_many_numpy(
        self,
        locations: Mapping[Hashable, TensorLocation],
        expert_ids: list[int] | tuple[int, ...],
        *,
        max_gap_bytes: int | None = None,
        use_file_cache: bool = False,
    ) -> dict[Hashable, np.ndarray]:
        """Read and decode components without creating thread-bound MLX arrays."""

        requested = [int(value) for value in expert_ids]
        if not requested:
            raise ValueError("At least one expert row must be requested")
        rows = sorted(set(requested))
        if max_gap_bytes is None:
            max_gap_bytes = self.cold_max_gap_bytes if use_file_cache else 0
        tasks: list[
            tuple[Hashable, TensorLocation, int, int, int, TensorRowSource | None]
        ] = []
        for key, location in locations.items():
            if min(requested) < 0 or max(requested) >= location.shape[0]:
                raise ValueError(f"Expert row outside tensor {location.name}")
            if location.row_sources is not None:
                for row in rows:
                    source = location.row_sources[row]
                    fd = self._fd(source.path, use_file_cache=use_file_cache)
                    tasks.append((key, location, fd, row, row + 1, source))
            else:
                fd = self._fd(location.path, use_file_cache=use_file_cache)
                storage_row_bytes = location.storage_row_bytes or location.row_bytes
                for first, stop in self._ranges(
                    rows, storage_row_bytes, max_gap_bytes
                ):
                    tasks.append((key, location, fd, first, stop, None))

        def read_range(task) -> tuple[Hashable, int, bytes]:
            key, location, fd, first, stop, source = task
            if source is not None:
                size = source.data_bytes
                offset = source.data_start
            else:
                storage_row_bytes = location.storage_row_bytes or location.row_bytes
                size = (stop - first) * storage_row_bytes
                offset = location.data_start + first * storage_row_bytes
            payload = os.pread(fd, size, offset)
            if len(payload) != size:
                path = source.path if source is not None else location.path
                raise OSError(f"Short expert read from {path.name}")
            return key, first, payload

        io_started = time.perf_counter()
        chunks = list(self._pool.map(read_range, tasks))
        self.io_seconds += time.perf_counter() - io_started
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

        decode_started = time.perf_counter()
        result: dict[Hashable, np.ndarray] = {}
        for key, location in locations.items():
            row_shape = location.shape[1:]
            storage_dtype = location.storage_dtype or location.dtype
            dtype = _NUMPY_DTYPES[storage_dtype]
            selected: dict[int, np.ndarray] = {}
            for first, payload in grouped[key]:
                storage_row_bytes = location.storage_row_bytes or location.row_bytes
                count = (
                    1
                    if location.row_sources is not None
                    else len(payload) // storage_row_bytes
                )
                values = np.frombuffer(payload, dtype=dtype)
                if location.output_slice is not None:
                    storage_shape = location.storage_shape
                    if storage_shape is None:
                        raise ValueError(f"Missing storage shape for {location.name}")
                    values = values.reshape((count, *storage_shape[1:]))
                    start, stop = location.output_slice
                    values = values[:, start:stop]
                else:
                    values = values.reshape((count, *row_shape))
                for row in rows:
                    if first <= row < first + count:
                        selected[row] = values[row - first]
            output = np.stack([selected[row] for row in requested])
            result[key] = output
        self.decode_seconds += time.perf_counter() - decode_started
        return result

    def materialize_many(
        self,
        locations: Mapping[Hashable, TensorLocation],
        components: Mapping[Hashable, np.ndarray],
    ) -> dict[Hashable, mx.array]:
        """Create MLX arrays on the caller's inference thread."""

        result: dict[Hashable, mx.array] = {}
        for key, output in components.items():
            location = locations[key]
            storage_dtype = location.storage_dtype or location.dtype
            array = mx.array(output)
            if storage_dtype == "BF16":
                array = array.view(mx.bfloat16)
            if storage_dtype != location.dtype:
                array = array.astype(_MLX_DTYPES[location.dtype])
            result[key] = array
        return result

    def read_many(
        self,
        locations: Mapping[Hashable, TensorLocation],
        expert_ids: list[int] | tuple[int, ...],
        *,
        max_gap_bytes: int | None = None,
        use_file_cache: bool = False,
    ) -> dict[Hashable, mx.array]:
        """Read components and materialize them on the calling thread."""

        components = self._read_many_numpy(
            locations,
            expert_ids,
            max_gap_bytes=max_gap_bytes,
            use_file_cache=use_file_cache,
        )
        return self.materialize_many(locations, components)

    def read_many_async(
        self,
        locations: Mapping[Hashable, TensorLocation],
        expert_ids: list[int] | tuple[int, ...],
        *,
        max_gap_bytes: int | None = None,
        use_file_cache: bool = False,
        use_fast_resource: bool = True,
    ) -> Future[dict[Hashable, np.ndarray]] | FastExpertLoadFuture:
        """Submit bounded CPU-only I/O and decoding for Metal overlap."""

        if (
            self.fast_resource_loading
            and use_fast_resource
            and self.fast_resource_compatible(locations)
        ):
            from .fast_resource import FastExpertLoadFuture

            return FastExpertLoadFuture(
                self.begin_fast_many(locations, expert_ids)
            )

        return self._request_pool.submit(
            self._read_many_numpy,
            locations,
            expert_ids,
            max_gap_bytes=max_gap_bytes,
            use_file_cache=use_file_cache,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._request_pool.shutdown(wait=True, cancel_futures=True)
        self._pool.shutdown(wait=True, cancel_futures=True)
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
