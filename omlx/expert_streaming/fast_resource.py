# SPDX-License-Identifier: Apache-2.0
"""Metal Fast Resource Loading plans for row-addressable expert tensors."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

from .safetensors import _NUMPY_DTYPES, TensorLocation


@dataclass
class FastExpertLoad:
    ticket: Any
    source_offsets: dict[Hashable, list[int]]
    row_bytes: dict[Hashable, int]
    native: Any
    row_conversions: dict[Hashable, str] = field(default_factory=dict)


class FastExpertLoadFuture:
    """Future-shaped handle used by the existing scratch prefetch pipeline."""

    def __init__(self, load: FastExpertLoad):
        self._load = load

    def result(self, timeout: float | None = None) -> FastExpertLoad:
        del timeout
        return self._load


class FastExpertLoader:
    """Issue MTLIO reads and blit their payloads into evaluated MLX banks."""

    def __init__(self, *, priority: str = "high"):
        from omlx.custom_kernels.fast_resource_loading import (
            FastResourceLoader,
            available,
            import_error,
        )

        if not available() or FastResourceLoader is None:
            raise RuntimeError(
                "Fast Resource Loading native extension is unavailable"
                + (f": {import_error()}" if import_error() else "")
            )
        self._native = FastResourceLoader(priority)
        self.direct_available = hasattr(self._native, "begin_direct")

    @staticmethod
    def _conversion(location: TensorLocation) -> str:
        storage_dtype = location.storage_dtype or location.dtype
        if storage_dtype == location.dtype:
            return "none"
        if storage_dtype == "BF16" and location.dtype == "F16":
            return "bf16_to_fp16"
        raise ValueError(
            "Fast Resource Loading cannot convert "
            f"{storage_dtype} to {location.dtype} for {location.name}"
        )

    @staticmethod
    def _source(location: TensorLocation, expert: int) -> tuple[str, int, int]:
        if location.row_sources is not None:
            source = location.row_sources[expert]
            if source.data_bytes != location.row_bytes:
                raise ValueError(
                    f"Fragmented expert row size mismatch for {location.name}"
                )
            return str(source.path), source.data_start, location.row_bytes

        storage_row_bytes = location.storage_row_bytes or location.row_bytes
        source_offset = location.data_start + expert * storage_row_bytes
        if location.output_slice is not None:
            if location.storage_shape is None:
                raise ValueError(f"Missing storage shape for {location.name}")
            start, _ = location.output_slice
            storage_dtype = location.storage_dtype or location.dtype
            item_size = np.dtype(_NUMPY_DTYPES[storage_dtype]).itemsize
            inner_elements = math.prod(location.storage_shape[2:])
            source_offset += start * inner_elements * item_size
        return str(location.path), source_offset, location.row_bytes

    def begin(
        self,
        locations: Mapping[Hashable, TensorLocation],
        expert_ids: list[int] | tuple[int, ...],
        *,
        max_gap_bytes: int = 64 * 1024,
    ) -> FastExpertLoad:
        requested = [int(expert) for expert in expert_ids]
        if not requested:
            raise ValueError("At least one expert row must be requested")
        requests: list[tuple[str, int, int, int]] = []
        offsets: dict[Hashable, list[int]] = {}
        row_bytes: dict[Hashable, int] = {}
        conversions: dict[Hashable, str] = {}
        staging_offset = 0
        for key, location in locations.items():
            for expert in requested:
                if not 0 <= expert < location.shape[0]:
                    raise ValueError(f"Expert row outside tensor {location.name}")
            if location.row_sources is not None:
                component_offsets = []
                for expert in requested:
                    path, source_offset, size = self._source(location, expert)
                    component_offsets.append(staging_offset)
                    requests.append((path, source_offset, size, staging_offset))
                    staging_offset += size
                offsets[key] = component_offsets
                row_bytes[key] = location.row_bytes
                conversions[key] = self._conversion(location)
                continue

            storage_row_bytes = location.storage_row_bytes or location.row_bytes
            rows = sorted(set(requested))
            ranges: list[tuple[int, int]] = []
            first = previous = rows[0]
            for row in rows[1:]:
                gap = (row - previous - 1) * storage_row_bytes
                if gap > max_gap_bytes:
                    ranges.append((first, previous + 1))
                    first = row
                previous = row
            ranges.append((first, previous + 1))

            selected_offsets: dict[int, int] = {}
            slice_offset = 0
            if location.output_slice is not None:
                if location.storage_shape is None:
                    raise ValueError(f"Missing storage shape for {location.name}")
                start, _ = location.output_slice
                storage_dtype = location.storage_dtype or location.dtype
                item_size = np.dtype(_NUMPY_DTYPES[storage_dtype]).itemsize
                slice_offset = start * math.prod(location.storage_shape[2:]) * item_size
            for first, stop in ranges:
                size = (stop - first) * storage_row_bytes
                requests.append(
                    (
                        str(location.path),
                        location.data_start + first * storage_row_bytes,
                        size,
                        staging_offset,
                    )
                )
                for row in rows:
                    if first <= row < stop:
                        selected_offsets[row] = (
                            staging_offset
                            + (row - first) * storage_row_bytes
                            + slice_offset
                        )
                staging_offset += size
            offsets[key] = [selected_offsets[expert] for expert in requested]
            row_bytes[key] = location.row_bytes
            conversions[key] = self._conversion(location)
        return FastExpertLoad(
            self._native.begin(requests),
            offsets,
            row_bytes,
            self._native,
            conversions,
        )

    def begin_direct(
        self,
        locations: Mapping[Hashable, TensorLocation],
        expert_ids: list[int] | tuple[int, ...],
        slots: list[int],
        targets: Mapping[Hashable, tuple[mx.array, int]],
    ) -> FastExpertLoad:
        """Load exact rows directly into proven-idle bank destinations."""

        if not self.direct_available:
            raise RuntimeError("Direct Fast Resource Loading is unavailable")
        requested = [int(expert) for expert in expert_ids]
        if not requested or len(requested) != len(slots):
            raise ValueError("Direct expert loads require matching experts and slots")
        requests: list[tuple[str, int, int, mx.array, int]] = []
        for key, location in locations.items():
            if self._conversion(location) != "none":
                raise ValueError(
                    "Direct Fast Resource Loading requires storage and runtime "
                    f"dtypes to match: {location.name}"
                )
            target, inner_offset = targets[key]
            target_row_bytes = int(target.nbytes) // int(target.shape[0])
            for expert, slot in zip(requested, slots, strict=True):
                if not 0 <= expert < location.shape[0]:
                    raise ValueError(f"Expert row outside tensor {location.name}")
                path, source_offset, size = self._source(location, expert)
                requests.append(
                    (
                        path,
                        source_offset,
                        size,
                        target,
                        int(slot) * target_row_bytes + int(inner_offset),
                    )
                )
        return FastExpertLoad(
            self._native.begin_direct(requests), {}, {}, self._native
        )

    def finish_into(
        self,
        load: FastExpertLoad,
        slots: list[int],
        targets: Mapping[Hashable, tuple[mx.array, int]],
        *,
        destinations_idle: bool = False,
    ) -> dict[str, int | float]:
        copies: list[tuple] = []
        if not destinations_idle:
            arrays: list[mx.array] = []
            seen: set[int] = set()
            for target, _ in targets.values():
                identity = id(target)
                if identity not in seen:
                    arrays.append(target)
                    seen.add(identity)
            mx.eval(*arrays)
        for key, source_offsets in load.source_offsets.items():
            target, inner_offset = targets[key]
            target_row_bytes = int(target.nbytes) // int(target.shape[0])
            size = load.row_bytes[key]
            conversion = load.row_conversions.get(key, "none")
            for slot, source_offset in zip(slots, source_offsets, strict=True):
                copy = (
                    target,
                    int(slot) * target_row_bytes + int(inner_offset),
                    source_offset,
                    size,
                )
                copies.append(copy if conversion == "none" else (*copy, conversion))
        return dict(load.native.finish(load.ticket, copies))
