# SPDX-License-Identifier: Apache-2.0
"""Metal Fast Resource Loading plans for row-addressable expert tensors."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from .safetensors import _NUMPY_DTYPES, TensorLocation


@dataclass
class FastExpertLoad:
    ticket: Any
    source_offsets: dict[Hashable, list[int]]
    row_bytes: dict[Hashable, int]


class FastExpertLoadFuture:
    """Future-shaped handle used by the existing scratch prefetch pipeline."""

    def __init__(self, load: FastExpertLoad):
        self._load = load

    def result(self, timeout: float | None = None) -> FastExpertLoad:
        del timeout
        return self._load


class FastExpertLoader:
    """Issue MTLIO reads and blit their payloads into evaluated MLX banks."""

    def __init__(self):
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
        self._native = FastResourceLoader()

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
            item_size = np.dtype(_NUMPY_DTYPES[location.dtype]).itemsize
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
                item_size = np.dtype(_NUMPY_DTYPES[location.dtype]).itemsize
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
        return FastExpertLoad(self._native.begin(requests), offsets, row_bytes)

    def finish_into(
        self,
        load: FastExpertLoad,
        slots: list[int],
        targets: Mapping[Hashable, tuple[mx.array, int]],
    ) -> dict[str, int | float]:
        copies: list[tuple[mx.array, int, int, int]] = []
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
            for slot, source_offset in zip(slots, source_offsets, strict=True):
                copies.append(
                    (
                        target,
                        int(slot) * target_row_bytes + int(inner_offset),
                        source_offset,
                        size,
                    )
                )
        return dict(self._native.finish(load.ticket, copies))
