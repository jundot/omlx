"""Coordinate-normalizing capture adapter for oMLX DeepSeek-V4.1 Attention.

Native `Indexer.__call__` returns *compressed-cache* coordinates (`idxs +
offset` in the pinned reference); DFly target context needs one explicit
absolute-position coordinate system.  This adapter is the only permitted place
to convert the native pick lattice: valid compressed index `i` maps to target
absolute anchor `i * compress_ratio`; native `-1` remains `-1`.
"""
from __future__ import annotations

from typing import Callable

import mlx.core as mx

from .dsv41_bridge import DSV41TargetBridge


def capture_attention_slice(
    *,
    bridge: DSV41TargetBridge,
    layer: int,
    start: int,
    ratio: int,
    window_latents: mx.array,
    compressed_latents: mx.array,
    compressed_start: int,
    native_index_topk: mx.array,
    committed_rows: int | None = None,
    compressed_rope: "Callable[[mx.array, mx.array], mx.array] | None" = None,
) -> None:
    """Capture one DSV41 attention slice in bridge coordinates.

    Call at the source seam after `shared["idx"] = self.indexer(...)` and before
    compressed/local values are packed. `compressed_start` is `kv_start` from
    the source path (not always query `start` under CED replay). The caller must
    pass only newly emitted compressor rows; this request-local bridge appends
    them and owns no oMLX cache storage.
    """
    if start < 0 or compressed_start < 0:
        raise ValueError("target positions must be non-negative")
    if ratio < 1:
        raise ValueError("DSV41 compression ratio must be >= 1")
    if window_latents.ndim != 3 or compressed_latents.ndim != 3 or native_index_topk.ndim != 3:
        raise ValueError("captured values must all be [1, sequence, ...]")
    if window_latents.shape[0] != 1 or compressed_latents.shape[0] != 1 or native_index_topk.shape[0] != 1:
        raise ValueError("capture adapter handles exactly one request row")
    if native_index_topk.shape[1] != window_latents.shape[1]:
        raise ValueError("native index rows must align with local target rows")
    if window_latents.shape[-1] != compressed_latents.shape[-1]:
        raise ValueError("local and compressed target latent widths differ")

    # The reference Indexer returns positions in the compressed lattice. Do not
    # multiply -1, because it is the invalid sentinel consumed by sparse gather.
    if start == 0:
        # Position zero is a fresh prompt: the previous request's rows would
        # otherwise be read by the first draft of this one.
        bridge.reset()

    native = native_index_topk.astype(mx.int32)
    absolute_index_topk = mx.where(native >= 0, native * ratio, mx.array(-1, mx.int32))
    compressed_positions = (
        (compressed_start // ratio) * ratio
        + mx.arange(compressed_latents.shape[1], dtype=mx.int32) * ratio
    )
    if compressed_rope is not None:
        # The reference records BOTH reusable row classes after RoPE (window rows
        # already arrive rotated from the source path). Compressed rows leave the
        # compressor unrotated, so they must be rotated here, at their own
        # absolute positions, or the draft attends to them as if position-free.
        compressed_latents = compressed_rope(compressed_latents, compressed_positions)
    bridge.record(
        layer,
        start=start,
        window_latents=window_latents,
        global_positions=compressed_positions,
        global_latents=compressed_latents,
        index_topk=absolute_index_topk,
        committed_rows=committed_rows,
    )
