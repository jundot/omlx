# SPDX-License-Identifier: Apache-2.0
"""Per-family hooks that split one prefill chunk at layer ``k``.

The helper side keeps layers ``[0, k)`` and calls :meth:`StageAdapter.run_first`
for each chunk. This Mac calls :meth:`StageAdapter.run_rest` with what came
back and then :meth:`StageAdapter.apply_state`. Arrays cross the link as flat
``{name: mx.array}`` dicts so the wire format stays family independent.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlx.core as mx

Arrays = dict[str, mx.array]
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _tensor_bytes(model_path: Path) -> dict[str, int]:
    """Bytes per tensor from the safetensors headers, without loading data."""
    index = model_path / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else ["model.safetensors"]
    )
    sizes: dict[str, int] = {}
    for name in files:
        with open(model_path / name, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(length))
        for key, spec in header.items():
            if key != "__metadata__":
                start, end = spec["data_offsets"]
                sizes[key] = end - start
    return sizes


class StageAdapter:
    """Split a model family's prefill forward between two Macs."""

    model_types: tuple[str, ...] = ()

    def language_model(self, model: Any) -> Any:
        """The module that owns ``layers`` (the scheduler holds an adapter)."""
        return getattr(model, "_language_model", None) or getattr(
            model, "language_model", model
        )

    def layer_count(self, lm: Any) -> int:
        raise NotImplementedError

    def max_split(self, lm: Any) -> int:
        """Largest ``k`` this family supports; the helper holds ``[0, k)``."""
        return self.layer_count(lm) - 1

    def keep_first(self, lm: Any, k: int) -> None:
        """Drop layers ``[k, n)`` before the helper loads weights."""
        raise NotImplementedError

    @contextmanager
    def use_first(self, lm: Any, k: int):
        """Helper: run only the first ``k`` of the layers it holds."""
        end = lm.end_idx
        lm.end_idx = k
        try:
            yield
        finally:
            lm.end_idx = end

    def layer_work(self, lm: Any) -> list[float]:
        """Relative prefill cost of each layer, used to balance the split."""
        return [1.0] * self.layer_count(lm)

    def resident(self, tensor: str) -> bool:
        """Whether the helper keeps this tensor in GPU memory."""
        return True

    def needed_outside_layers(self, tensor: str) -> bool:
        """Non-layer tensors the helper loads; all of them unless excluded."""
        return True

    def memory_plan(self, model_path: str | Path) -> tuple[int, list[int]]:
        """(bytes outside the layers, bytes per layer) the helper holds."""
        shared, per_layer = 0, {}
        for tensor, size in _tensor_bytes(Path(model_path).expanduser()).items():
            if not self.resident(tensor):
                continue
            match = _LAYER.search(tensor)
            if match:
                layer = int(match.group(1))
                per_layer[layer] = per_layer.get(layer, 0) + size
            elif self.needed_outside_layers(tensor):
                shared += size
        count = max(per_layer) + 1 if per_layer else 0
        return shared, [per_layer.get(i, 0) for i in range(count)]

    def load_first(self, path: str, k: int) -> Any:
        """Helper: load layers ``[0, k)`` of a checkpoint; returns the module
        ``run_first`` takes."""
        raise NotImplementedError

    def run_first(
        self, lm: Any, cache: list, tokens: mx.array, next_tokens: int
    ) -> tuple[Arrays, Arrays]:
        """Helper: run ``[0, k)`` on one chunk.

        Returns the tensors that cross the cut and the chunk's cache state
        for layers ``[0, k)``, both evaluated.
        """
        raise NotImplementedError

    def run_rest(
        self,
        prefill: Callable[..., Any],
        lm: Any,
        cache: list,
        tokens: mx.array,
        k: int,
        boundary: Arrays,
        state: Arrays,
        **kwargs: Any,
    ) -> Any:
        """This Mac: run ``[k, n)`` on the chunk the helper sent."""
        raise NotImplementedError

    def apply_state(self, cache: list, k: int, state: Arrays) -> None:
        """This Mac: bring layers ``[0, k)`` to the end of the chunk."""
        raise NotImplementedError

    def prefix_state(self, cache: list, k: int) -> Arrays:
        """This Mac: full state of layers ``[0, k)`` for a cached prefix."""
        raise NotImplementedError

    def load_prefix(self, cache: list, state: Arrays) -> None:
        """Helper: start from the prefix this Mac restored."""
        raise NotImplementedError
