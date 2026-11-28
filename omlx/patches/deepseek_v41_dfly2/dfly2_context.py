"""MLX implementation of DFly2's target-context handoff.

This is the verified portable boundary before model-specific DSA/MLA kernels:
per draft layer, gather target-layer KV latents at the target indexer's sparse
positions plus the local block window.  It deliberately owns no KV cache and
runs no indexer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mlx.core as mx


@dataclass(frozen=True)
class DFly2ContextConfig:
    """Static fields from AngelSlim/DS-V4.1-Flash-DFly2-B6 config.json."""

    block_size: int = 6
    context_local_window: int = 128
    target_kv_layers: tuple[int, ...] = (28, 32, 36)
    target_index_layers: tuple[int, ...] = (28, 32, 36)
    index_topk: int = 512

    def __post_init__(self) -> None:
        if self.block_size < 2:
            raise ValueError("block_size must include an anchor and at least one proposal")
        if self.context_local_window < 0:
            raise ValueError("context_local_window must be non-negative")
        if len(self.target_kv_layers) != len(self.target_index_layers):
            raise ValueError("each draft layer must have one target KV and index layer")


@dataclass(frozen=True)
class DFly2LayerContext:
    """Gathered target KV for one draft layer.

    indices use -1 for padded/unavailable targets. `valid` is an explicit
    attention-mask source; zero-valued latents must never imply invalidity.
    """

    indices: mx.array  # [batch, keys]
    valid: mx.array  # [batch, keys]
    latents: mx.array  # [batch, keys, latent_dim]


def _near_window(anchor: mx.array, block_size: int, local_window: int) -> mx.array:
    """Return `[anchor-local_window, ..., anchor+block_size-1]` for every row.

    The upper end includes the whole draft block.  With B6, K512 and W128 this
    creates 512 + 128 + 6 = 646 candidate slots before duplicate removal,
    matching the official model card's context-read accounting.
    """
    offsets = mx.arange(-local_window, block_size, dtype=mx.int32)
    return anchor[..., None].astype(mx.int32) + offsets


def _stable_unique(indices: mx.array, seq_len: int) -> tuple[mx.array, mx.array]:
    """Mask duplicates/invalid positions without ragged MLX outputs.

    Candidate order is preserved: sparse top-k first, then local window.
    Duplicate slots become -1 and `valid=False`, retaining a static shape for
    the later DSA/MLA attention kernel.
    """
    in_range = (indices >= 0) & (indices < seq_len)
    width = indices.shape[-1]
    # For every candidate, check whether the same valid value occurred earlier.
    equal_prior = indices[..., :, None] == indices[..., None, :]
    prior = mx.tril(mx.ones((width, width), dtype=mx.bool_), k=-1)
    duplicate = mx.any(equal_prior & prior, axis=-1)
    valid = in_range & ~duplicate
    return mx.where(valid, indices, mx.array(-1, dtype=indices.dtype)), valid


def _gather_latents(latents: mx.array, indices: mx.array, valid: mx.array) -> mx.array:
    """Gather `[B,S,D]` target latents using safely clamped padded indices."""
    if latents.ndim != 3:
        raise ValueError(f"target latents must be [B,S,D], got {latents.shape}")
    if indices.shape[0] != latents.shape[0]:
        raise ValueError("batch dimension of indices and latents must match")
    safe = mx.clip(indices, 0, latents.shape[1] - 1)
    batch = mx.arange(latents.shape[0], dtype=mx.int32)[:, None]
    gathered = latents[batch, safe]
    return mx.where(valid[..., None], gathered, mx.zeros_like(gathered))


def build_layer_context(
    target_latents: mx.array,
    index_topk: mx.array,
    anchor_positions: mx.array,
    config: DFly2ContextConfig,
) -> DFly2LayerContext:
    """Build static sparse context for a single paired target layer.

    Args:
      target_latents: target MLA latent cache `[B, sequence, latent_dim]`.
      index_topk: target indexer's *anchor-1* picks `[B, K]`; -1 pads unfilled
        positions.  The prototype receives them from the target runtime rather
        than recomputing scores.
      anchor_positions: `[B]`, position of the draft block's anchor token.
    """
    if index_topk.ndim != 2 or anchor_positions.ndim != 1:
        raise ValueError("index_topk must be [B,K] and anchor_positions must be [B]")
    if index_topk.shape[0] != target_latents.shape[0] or anchor_positions.shape[0] != target_latents.shape[0]:
        raise ValueError("batch dimensions must match")
    if index_topk.shape[-1] != config.index_topk:
        raise ValueError(f"expected {config.index_topk} target index picks, got {index_topk.shape[-1]}")
    candidates = mx.concatenate(
        [index_topk.astype(mx.int32), _near_window(anchor_positions, config.block_size, config.context_local_window)],
        axis=-1,
    )
    indices, valid = _stable_unique(candidates, target_latents.shape[1])
    return DFly2LayerContext(indices, valid, _gather_latents(target_latents, indices, valid))


def build_dfly2_context(
    target_latents_by_layer: dict[int, mx.array],
    index_topk_by_layer: dict[int, mx.array],
    anchor_positions: mx.array,
    config: DFly2ContextConfig = DFly2ContextConfig(),
) -> list[DFly2LayerContext]:
    """Return one shared-KV sparse context object for every DFly2 draft layer."""
    result: list[DFly2LayerContext] = []
    for kv_layer, index_layer in zip(config.target_kv_layers, config.target_index_layers):
        try:
            latents = target_latents_by_layer[kv_layer]
            picks = index_topk_by_layer[index_layer]
        except KeyError as exc:
            raise KeyError(f"missing target state for paired layer {exc.args[0]}") from exc
        result.append(build_layer_context(latents, picks, anchor_positions, config))
    return result
