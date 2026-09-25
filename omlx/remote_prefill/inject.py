# SPDX-License-Identifier: Apache-2.0
"""Turn pulled pages into per-layer cache updates and append them to a request's prompt cache."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from . import pages, wire
from .job import Chunk, PrefillResult
from .receiver import HandoffError

# mlx-lm cache classes that take appended keys and values through update_and_fetch.
SUPPORTED_CACHES = frozenset({"KVCache", "QuantizedKVCache"})


def _rank_tensors(mx: Any, chunks: list[Chunk]) -> tuple[Any, Any]:
    """One rank's pages of one layer as token-major keys and values, frames in row order."""
    parts = [
        pages.split_pages(
            mx,
            chunk.layer,
            chunk.data.reshape(pages.frame_shape(chunk.layer, chunk.rows)),
        )
        for chunk in sorted(chunks, key=lambda item: item.row_start)
    ]
    if len(parts) == 1:
        return parts[0]
    return (
        mx.concatenate([keys for keys, _ in parts]),
        mx.concatenate([values for _, values in parts]),
    )


def _distinct_ranks(layer: wire.LayerExport, ranks: list[int]) -> list[int]:
    """Ranks whose heads together hold the layer's heads once each, in head order."""
    if layer.kind == "mla":
        # MLA pages are replicated on every rank.
        return ranks[:1]
    if not layer.total_heads or layer.heads * len(ranks) == layer.total_heads:
        return ranks
    held = layer.heads * len(ranks)
    if held % layer.total_heads or layer.total_heads % layer.heads:
        raise HandoffError(f"layer {layer.index} heads do not tile across the ranks")
    # vLLM repeats each head on consecutive ranks when there are more ranks than heads.
    return ranks[:: held // layer.total_heads]


def layer_updates(
    mx: Any, result: PrefillResult, start: int, end: int, dtype: Any
) -> list[tuple[Any, Any]]:
    """Per layer in index order, the keys and values of prompt tokens [start, end) in cache layout."""
    first = result.manifests[0].first_token
    if not first <= start < end <= result.manifests[0].prompt_tokens:
        raise HandoffError(f"tokens [{start}, {end}) are outside the handoff")
    grouped: dict[int, dict[int, list[Chunk]]] = defaultdict(lambda: defaultdict(list))
    for chunk in result.chunks:
        grouped[chunk.layer.index][chunk.rank].append(chunk)
    if sorted(grouped) != list(range(len(grouped))):
        raise HandoffError("the handoff's layers are not numbered from 0")
    updates = []
    for index in range(len(grouped)):
        ranks = grouped[index]
        layer = ranks[min(ranks)][0].layer
        tensors = [
            _rank_tensors(mx, ranks[rank])
            for rank in _distinct_ranks(layer, sorted(ranks))
        ]
        keys = mx.concatenate([keys for keys, _ in tensors], axis=1)
        values = mx.concatenate([values for _, values in tensors], axis=1)
        if keys.shape[0] < end - first:
            raise HandoffError(
                f"layer {index} holds {keys.shape[0]} of {end - first} tokens"
            )
        keys, values = (
            keys[start - first : end - first],
            values[start - first : end - first],
        )
        if layer.kind == "mla":
            updates.append(
                (keys[None, None].astype(dtype), values[None, None].astype(dtype))
            )
        else:
            updates.append(
                (
                    keys.transpose(1, 0, 2)[None].astype(dtype),
                    values.transpose(1, 0, 2)[None].astype(dtype),
                )
            )
    return updates


def extend_caches(mx: Any, caches: list[Any], updates: list[tuple[Any, Any]]) -> None:
    """Append each layer's update to its cache and materialize them on the caller's stream."""
    if len(caches) != len(updates):
        raise HandoffError(
            f"the handoff has {len(updates)} layers; the model has {len(caches)}"
        )
    unsupported = {type(cache).__name__ for cache in caches} - SUPPORTED_CACHES
    if unsupported:
        raise HandoffError(f"cache types {sorted(unsupported)} cannot take a handoff")
    for cache, (keys, values) in zip(caches, updates):
        cache.update_and_fetch(keys, values)
    mx.eval([cache.state for cache in caches])
