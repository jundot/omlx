"""Stream-scoped mlx-lm cache operations for cold text prefill groups."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache, KVCache, make_prompt_cache


class _UnpaddedPrefillKVCache(BatchKVCache):
    """A cold-only batch whose rows always share an unpadded cache position.

    Equal-length forwards, filtering and trimming preserve this invariant.
    Padding, extension and state restoration are deliberately unsupported;
    callers must extract plain KVCache rows before handing them to decode.
    Direct mutation of cache attributes is outside this private contract.
    """

    def __init__(self, batch_size: int) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Unpadded prefill requires a positive batch size")
        super().__init__([0] * batch_size)

    def make_mask(
        self,
        token_count: int,
        return_array: bool = False,
        *,
        window_size: int | None = None,
        **kwargs,
    ):
        if return_array or window_size is not None or kwargs:
            return super().make_mask(
                token_count,
                return_array=return_array,
                window_size=window_size,
                **kwargs,
            )
        return None if token_count == 1 else "causal"

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None or right_padding is not None:
            raise ValueError("Unpadded prefill does not support padding preparation")
        super().prepare(lengths=lengths)

    def extract(self, row_index: int) -> KVCache:
        """Extract a row without reading its known-zero padding on the host."""
        cache = KVCache()
        cache.keys = mx.contiguous(self.keys[row_index : row_index + 1, :, : self._idx])
        cache.values = mx.contiguous(
            self.values[row_index : row_index + 1, :, : self._idx]
        )
        cache.offset = cache.keys.shape[2]
        return cache

    def filter(self, batch_indices: Sequence[int]) -> None:
        """Keep rows without reducing padding that is zero by construction."""
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

    def extend(self, other):
        raise ValueError("Unpadded prefill does not support cache extension")

    @BatchKVCache.state.setter
    def state(self, state):
        raise ValueError("Unpadded prefill does not support state restoration")

    @classmethod
    def merge(cls, caches):
        raise ValueError("Unpadded prefill requires cold cache construction")


def evaluate_cache(
    cache: Sequence[BatchKVCache | KVCache], *, stream: mx.Stream
) -> None:
    """Materialize cache buffers and metadata, including an empty batch."""
    with mx.stream(stream):
        arrays = []
        for layer_cache in cache:
            if layer_cache.keys is not None:
                arrays.extend((layer_cache.keys, layer_cache.values))
            if isinstance(layer_cache, BatchKVCache):
                arrays.extend((layer_cache.offset, layer_cache.left_padding))
        mx.eval(arrays)


def create_cold_batch_cache(
    model: Any, batch_size: int, *, stream: mx.Stream
) -> list[BatchKVCache]:
    """Build an unpadded batch only after validating exact cold KVCache rows."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Batched prefill requires a positive batch size")
    with mx.stream(stream):
        row_caches = [make_prompt_cache(model) for _ in range(batch_size)]
        layer_count = len(row_caches[0])
        if not layer_count or any(
            len(row_cache) != layer_count for row_cache in row_caches
        ):
            raise ValueError("Batched prefill requires matching nonempty cache layers")
        for row_cache in row_caches:
            for layer_cache in row_cache:
                if type(layer_cache) is not KVCache:
                    raise ValueError("Batched prefill requires plain KVCache layers")
                if (
                    layer_cache.keys is not None
                    or layer_cache.values is not None
                    or layer_cache.offset != 0
                ):
                    raise ValueError("Batched prefill requires cold cache layers")
        batch_cache = [_UnpaddedPrefillKVCache(batch_size) for _ in range(layer_count)]
        evaluate_cache(batch_cache, stream=stream)
        return batch_cache


def extract_row(
    cache: Sequence[BatchKVCache], row_index: int, *, stream: mx.Stream
) -> list[KVCache]:
    """Return a materialized row without removing it from the owning group."""
    with mx.stream(stream):
        extracted = [
            (
                layer_cache.extract(row_index)
                if layer_cache.keys is not None
                else KVCache()
            )
            for layer_cache in cache
        ]
        evaluate_cache(extracted, stream=stream)
        return extracted


def filter_rows(
    cache: Sequence[BatchKVCache], row_indices: Sequence[int], *, stream: mx.Stream
) -> None:
    """Keep at least one row and materialize the result before it is reused."""
    if not row_indices:
        raise ValueError("Use group shutdown to remove every cache row")
    with mx.stream(stream):
        evaluate_cache(cache, stream=stream)
        indices = list(row_indices)
        for layer_cache in cache:
            layer_cache.filter(indices)
        evaluate_cache(cache, stream=stream)
