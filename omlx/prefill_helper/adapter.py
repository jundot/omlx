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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

Arrays = dict[str, mx.array]
# Host arrays as they cross the link: (data, dtype name).
HostArrays = dict[str, tuple[np.ndarray, str]]
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def language_tensor(tensor: str) -> bool:
    """Whether a checkpoint tensor belongs to the language model, in MLX
    (``language_model.``) or Hugging Face (``model.language_model.``) layout."""
    return tensor.removeprefix("model.").startswith("language_model.")


def _tensors(model_path: Path) -> dict[str, tuple[str, int]]:
    """File and bytes of each tensor, from the safetensors headers only."""
    index = model_path / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else ["model.safetensors"]
    )
    tensors: dict[str, tuple[str, int]] = {}
    for name in files:
        with open(model_path / name, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(length))
        for key, spec in header.items():
            if key != "__metadata__":
                start, end = spec["data_offsets"]
                tensors[key] = (name, end - start)
    return tensors


@dataclass
class Capture:
    """Layer outputs a block drafter reads, from chunk row ``start`` on.

    The stage that runs a listed layer stores its output in ``out``, in the
    form the family's local prefill hands the drafter.
    """

    layers: list[int]
    start: int
    out: Arrays = field(default_factory=dict)

    def take(self, layer: int, value: mx.array) -> None:
        if layer in self.layers:
            self.out[layer] = value[:, self.start :]


class StageAdapter:
    """Split a model family's prefill forward between two Macs."""

    model_types: tuple[str, ...] = ()
    # Whether run_first/run_rest fill a Capture (block drafters need it).
    captures_layers = False

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

    def prefetch(
        self, lm: Any, cache: list, current: Any, following: Any, start: int = 0
    ) -> None:
        """Start reading what the *following* chunk looks up while *current*
        runs on layers from *start*. An empty *current* starts a prompt."""

    def layer_work(self, lm: Any) -> list[float]:
        """Relative prefill cost of each layer, used to balance the split."""
        return [1.0] * self.layer_count(lm)

    def resident(self, tensor: str) -> bool:
        """Whether the helper loads this tensor (into GPU memory)."""
        return True

    def needed_outside_layers(self, tensor: str) -> bool:
        """Non-layer tensors the helper loads; all of them unless excluded."""
        return True

    def memory_plan(self, model_path: str | Path) -> tuple[int, list[int]]:
        """(bytes outside the layers, bytes per layer) the helper holds."""
        shared, per_layer = 0, {}
        for tensor, (_, size) in _tensors(Path(model_path).expanduser()).items():
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

    def weight_files(self, model_path: str | Path, k: int) -> set[str]:
        """Weight files a helper holding layers ``[0, k)`` reads."""
        files = set()
        for tensor, (name, _) in _tensors(Path(model_path).expanduser()).items():
            if not self.resident(tensor):
                continue
            match = _LAYER.search(tensor)
            if int(match.group(1)) < k if match else self.needed_outside_layers(tensor):
                files.add(name)
        return files

    def load_first(self, path: str, k: int, a8_min_tokens: int | None = None) -> Any:
        """Helper: load layers ``[0, k)`` of a checkpoint; returns the module
        ``run_first`` takes. ``a8_min_tokens`` is the host model's A8 prefill
        setting (None when off); families without A8 ignore it."""
        raise NotImplementedError

    def job_feed(
        self, lm: Any, cache: list | None, tokens: np.ndarray, grid: list[int], k: int
    ) -> Callable[[int], HostArrays] | None:
        """This Mac: per-chunk inputs it reads for the helper's layers.

        The returned callable runs on a background thread, once per chunk in
        order, and must not use the GPU. None when the helper needs nothing.
        """
        return None

    def run_first(
        self,
        lm: Any,
        cache: list,
        tokens: mx.array,
        next_tokens: int,
        inputs: HostArrays | None = None,
        capture: Capture | None = None,
    ) -> tuple[Arrays, Arrays]:
        """Helper: run ``[0, k)`` on one chunk with its ``job_feed`` inputs.

        Returns the tensors that cross the cut and the chunk's cache state
        for layers ``[0, k)``, both evaluated, and fills ``capture``.
        ``inputs`` is None only while warming the kernels.
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
        capture: Capture | None = None,
        **kwargs: Any,
    ) -> Any:
        """This Mac: run ``[k, n)`` on the chunk the helper sent."""
        raise NotImplementedError

    def apply_state(self, cache: list, k: int, state: Arrays) -> None:
        """This Mac: bring layers ``[0, k)`` to the end of the chunk."""
        raise NotImplementedError

    def finish_chunk(self, lm: Any, tokens: mx.array, output: Any, cache: list) -> None:
        """This Mac: after ``apply_state``, with what ``run_rest`` returned."""

    def prefix_state(self, cache: list, k: int) -> Arrays:
        """This Mac: full state of layers ``[0, k)`` for a cached prefix."""
        raise NotImplementedError

    def load_prefix(self, cache: list, state: Arrays) -> None:
        """Helper: start from the prefix this Mac restored."""
        raise NotImplementedError
