"""Request-local DeepSeek-V4.1 target-cache bridge for DFly2.

The target attention path has two distinct cache segments: compressed global MLA
latents and an uncompressed local window. DFly2 consumes target *index choices*
made for anchor-1 plus the local rows immediately before the draft anchor. This
module keeps that distinction explicit; it does not misrepresent packed target
state as a flat per-token cache and owns no draft KV cache.

Integration hook (not installed in production): capture raw rows inside
`deepseek_v41.language.Attention.__call__` after the target indexer produces
`shared["idx"]`, before values are packed. Call `rollback(keep_positions)` from
the same verify-cache rollback path as the target cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mlx.core as mx


@dataclass(frozen=True)
class DSV41History:
    """History available to one DFly2 draft layer, excluding its own B6 keys."""

    indices: mx.array  # [1, index_topk + local_window], absolute target positions or -1
    valid: mx.array  # [1, index_topk + local_window]
    latents: mx.array  # [1, index_topk + local_window, D]


@dataclass(frozen=True)
class _Segment:
    start: int
    values: mx.array  # [1, length, D] or [1, length, K]

    @property
    def end(self) -> int:
        return self.start + int(self.values.shape[1])

    def truncate(self, keep_positions: int) -> "_Segment | None":
        length = max(0, min(int(self.values.shape[1]), keep_positions - self.start))
        return None if length == 0 else _Segment(self.start, self.values[:, :length])


class DSV41TargetBridge:
    """Capture and rollback target state for one request row.

    The current oMLX DeepSeek implementation evaluates request rows
    independently. Accordingly, this adapter intentionally accepts batch size 1;
    a batching owner creates one bridge per row, eliminating cross-request cache
    aliasing and making rollback local to the speculative request.
    """

    def __init__(self, *, target_layers: Iterable[int] = (28, 32, 36), index_topk: int = 512, local_window: int = 128):
        self.target_layers = tuple(int(layer) for layer in target_layers)
        self.index_topk = int(index_topk)
        self.local_window = int(local_window)
        if not self.target_layers or self.index_topk < 1 or self.local_window < 0:
            raise ValueError("target_layers must be nonempty; index_topk >= 1; local_window >= 0")
        self._window: dict[int, list[_Segment]] = {layer: [] for layer in self.target_layers}
        # Global compressed rows are sparse absolute positions, unlike the
        # contiguous decoder/window segments above.
        self._global: dict[int, list[tuple[mx.array, mx.array]]] = {layer: [] for layer in self.target_layers}
        self._index: dict[int, list[_Segment]] = {layer: [] for layer in self.target_layers}

    @staticmethod
    def _require_row(name: str, value: mx.array, dims: int) -> None:
        if value.ndim != dims or value.shape[0] != 1:
            raise ValueError(f"{name} must be [1,...] for one request row, got {value.shape}")

    def record(
        self,
        layer: int,
        *,
        start: int,
        window_latents: mx.array,
        global_positions: mx.array,
        global_latents: mx.array,
        index_topk: mx.array,
    ) -> None:
        """Record one target forward slice without reading/modifying target caches.

        `window_latents` is the raw current-window MLA latent before target packing;
        `global_*` are raw compressed source latents and their absolute positions;
        `index_topk[:, j]` is the target indexer result for absolute position
        `start + j`. Capture must happen before quantized packing loses values.
        """
        if layer not in self._window:
            raise KeyError(f"layer {layer} is not a configured DFly2 target layer")
        self._require_row("window_latents", window_latents, 3)
        self._require_row("global_latents", global_latents, 3)
        self._require_row("index_topk", index_topk, 3)
        if global_positions.ndim != 1 or global_positions.shape[0] != global_latents.shape[1]:
            raise ValueError("global_positions must be [global_length] and align with global_latents")
        if window_latents.shape[-1] != global_latents.shape[-1]:
            raise ValueError("window/global latent dimensions differ")
        if index_topk.shape[1] != window_latents.shape[1] or index_topk.shape[2] != self.index_topk:
            raise ValueError("index_topk must align with this target slice and configured top-k")
        if start < 0:
            raise ValueError("start must be non-negative")
        # Global cache may be unchanged on many target layers; callers can pass
        # an empty [1,0,D] segment for those steps.
        self._window[layer].append(_Segment(int(start), window_latents))
        self._index[layer].append(_Segment(int(start), index_topk.astype(mx.int32)))
        if int(global_latents.shape[1]):
            self._global[layer].append((global_positions.astype(mx.int32)[None], global_latents))

    @staticmethod
    def _concat(segments: list[_Segment], *, empty_shape: tuple[int, ...], dtype) -> mx.array:
        return mx.zeros(empty_shape, dtype=dtype) if not segments else mx.concatenate([part.values for part in segments], axis=1)

    def _index_for(self, layer: int, position: int) -> mx.array:
        for segment in reversed(self._index[layer]):
            if segment.start <= position < segment.end:
                return segment.values[:, position - segment.start]
        raise KeyError(f"no target index record for anchor-1 position {position} at layer {layer}")

    def history(self, *, layer: int, anchor_position: int) -> DSV41History:
        """Build target history selected at `anchor_position - 1`.

        Sparse picks resolve against global compressed rows first, then the local
        window. The local segment supplies `anchor-W .. anchor-1`; own DFly block
        keys are intentionally not represented here and are added by draft
        attention, yielding `K + W + B` attention keys (646 for 512/128/B6).
        """
        if layer not in self._window:
            raise KeyError(f"layer {layer} is not configured")
        if anchor_position <= 0:
            raise ValueError("DFly2 anchor must follow at least one committed target position")
        picks = self._index_for(layer, anchor_position - 1)
        windows = self._concat(self._window[layer], empty_shape=(1, 0, 0), dtype=mx.float32)
        if not windows.size:
            raise KeyError(f"no local target MLA rows for layer {layer}")
        d = windows.shape[-1]
        global_latents = (
            mx.zeros((1, 0, d), dtype=windows.dtype)
            if not self._global[layer]
            else mx.concatenate([values for _positions, values in self._global[layer]], axis=1)
        )
        global_pos = (
            mx.zeros((1, 0), dtype=mx.int32)
            if not self._global[layer]
            else mx.concatenate([positions for positions, _values in self._global[layer]], axis=1)
        )
        local_pos = mx.concatenate([mx.arange(part.start, part.end, dtype=mx.int32) for part in self._window[layer]])
        positions = mx.concatenate([global_pos[0], local_pos], axis=0)
        values = mx.concatenate([global_latents, windows], axis=1)

        local = mx.arange(anchor_position - self.local_window, anchor_position, dtype=mx.int32)[None]
        requested = mx.concatenate([picks, local], axis=-1)
        # First matching candidate wins. Global rows are intentionally before local
        # rows because sparse index picks address compressed/global cache positions.
        matches = requested[..., :, None] == positions[None, None, :]
        found = mx.any(matches, axis=-1) & (requested >= 0)
        source = mx.argmax(matches.astype(mx.int32), axis=-1)
        gathered = values[mx.zeros(source.shape, dtype=mx.int32), source]
        width = requested.shape[-1]
        equal_prior = requested[..., :, None] == requested[..., None, :]
        duplicate = mx.any(equal_prior & mx.tril(mx.ones((width, width), dtype=mx.bool_), k=-1), axis=-1)
        valid = found & ~duplicate
        indices = mx.where(valid, requested, mx.array(-1, mx.int32))
        latents = mx.where(valid[..., None], gathered, mx.zeros_like(gathered))
        return DSV41History(indices=indices, valid=valid, latents=latents)

    def rollback(self, *, keep_positions: int) -> None:
        """Discard all captured state at absolute positions >= `keep_positions`."""
        if keep_positions < 0:
            raise ValueError("keep_positions must be non-negative")
        for store in (self._window, self._index):
            for layer, segments in store.items():
                store[layer] = [trimmed for segment in segments if (trimmed := segment.truncate(keep_positions)) is not None]
        for layer, segments in self._global.items():
            kept = []
            for positions, values in segments:
                # Target compressed positions are chronological. This host-side
                # count only chooses a rollback boundary; no target cache values
                # are copied to host and the following slice remains an MLX view.
                count = int((positions[0] < keep_positions).sum().item())
                if count:
                    kept.append((positions[:, :count], values[:, :count]))
            self._global[layer] = kept
