"""Persistent unpadded batched prefill with explicit ownership."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache, KVCache

from .mlx_adapter import (
    create_cold_batch_cache,
    evaluate_cache,
    extract_row,
    filter_rows,
)


@dataclass(frozen=True)
class PrefillGroupStepResult:
    """Logical work and physical shape of a materialized forward."""

    request_ids: tuple[str, ...]
    tokens_per_request: int
    processed_tokens: int
    completed_request_ids: tuple[str, ...]
    elapsed_s: float

    @property
    def executed_tokens(self) -> int:
        return len(self.request_ids) * self.tokens_per_request

    @property
    def padding_tokens(self) -> int:
        return self.executed_tokens - self.processed_tokens

    def tokens_for(self, request_id: str) -> int:
        return self.tokens_per_request


class BatchedPrefillGroup:
    """Own cold text caches until each row is handed off or discarded.

    Rows contain prefill tokens only; the scheduler retains each last prompt
    token for decode. Every step advances all remaining rows by the same
    positive length. Completed rows must be extracted and removed before the
    next step. The group never admits
    new rows, samples, or clears MLX caches.

    Calls are serialized by the scheduler on the engine thread. A failed
    forward or in-place cache transform invalidates the group; only close()
    is safe afterward. Extraction and removal are physical cache operations;
    PrefillBatchRuntime separately tracks logical ownership and handoff commits.
    """

    def __init__(
        self,
        model: Any,
        rows: Sequence[tuple[str, Sequence[int]]],
        stream: mx.Stream,
        skip_lm_head: bool = False,
    ) -> None:
        if stream is None:
            raise ValueError("Batched prefill requires an explicit engine stream")
        if len(rows) < 2:
            raise ValueError("A new prefill group requires at least two requests")
        request_ids = tuple(request_id for request_id, _ in rows)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Prefill group request IDs must be unique")
        token_rows = tuple(tuple(tokens) for _, tokens in rows)
        if any(not tokens for tokens in token_rows):
            raise ValueError("Prefill group rows must contain prefill tokens")

        self._model = model
        self._stream = stream
        self._skip_lm_head = skip_lm_head
        self._request_ids = request_ids
        self._token_rows = token_rows
        self._row_lengths = dict(zip(request_ids, map(len, token_rows)))
        self._tokens_processed = 0
        self._invalid = False
        self._closed = False
        self._cache = create_cold_batch_cache(model, len(rows), stream=stream)

    @property
    def request_ids(self) -> tuple[str, ...]:
        return self._request_ids

    @property
    def remaining_tokens(self) -> dict[str, int]:
        return {
            request_id: len(tokens) - self.processed_tokens_for(request_id)
            for request_id, tokens in zip(self._request_ids, self._token_rows)
        }

    @property
    def remaining_min(self) -> int:
        return min(self.remaining_tokens.values(), default=0)

    @property
    def batch_size(self) -> int:
        return len(self._request_ids)

    @property
    def tokens_processed(self) -> int:
        """Physical cache position; use processed_tokens_for for request progress."""
        return self._tokens_processed

    def processed_tokens_for(self, request_id: str) -> int:
        """Retain last materialized logical progress even after invalidation."""
        return min(self._tokens_processed, self._row_lengths[request_id])

    @property
    def cache_nbytes(self) -> int:
        """Allocated KV buffers, including unused KV capacity."""
        return sum(layer_cache.nbytes for layer_cache in self._cache)

    @property
    def cache(self) -> tuple[BatchKVCache, ...]:
        """Borrow the owned cache for inspection; callers must not mutate it."""
        return tuple(self._cache)

    @property
    def valid(self) -> bool:
        return not self._invalid and not self._closed

    def _require_valid(self) -> None:
        if not self.valid:
            raise RuntimeError("Prefill group is closed or invalidated")

    def step(self, chunk_tokens: int) -> PrefillGroupStepResult:
        """Execute and materialize one equal-length chunk for every active row."""
        self._require_valid()
        if type(chunk_tokens) is not int or chunk_tokens <= 0:
            raise ValueError("Prefill chunk length must be a positive integer")
        if chunk_tokens > self.remaining_min:
            raise ValueError("Prefill chunk exceeds the shortest remaining row")

        started = time.perf_counter()
        chunk_end = self._tokens_processed + chunk_tokens
        try:
            with mx.stream(self._stream):
                chunk = mx.array(
                    [
                        tokens[self._tokens_processed : chunk_end]
                        for tokens in self._token_rows
                    ],
                    dtype=mx.int32,
                )
                if self._skip_lm_head:
                    self._model(chunk, cache=self._cache, skip_lm_head=True)
                else:
                    self._model(chunk, cache=self._cache)
                evaluate_cache(self._cache, stream=self._stream)
        except BaseException:
            self._invalid = True
            raise

        self._tokens_processed = chunk_end
        return PrefillGroupStepResult(
            request_ids=self._request_ids,
            tokens_per_request=chunk_tokens,
            processed_tokens=chunk_tokens * self.batch_size,
            completed_request_ids=tuple(
                request_id
                for request_id, remaining in self.remaining_tokens.items()
                if remaining == 0
            ),
            elapsed_s=time.perf_counter() - started,
        )

    def extract(self, request_id: str) -> list[KVCache]:
        """Extract a complete or partial row while retaining the group owner."""
        self._require_valid()
        if request_id not in self._request_ids:
            raise KeyError(request_id)
        return extract_row(
            self._cache, self._request_ids.index(request_id), stream=self._stream
        )

    def remove(self, request_ids: Sequence[str]) -> None:
        """Remove handed-off or cancelled rows at a GPU-safe boundary."""
        self._require_valid()
        removed = set(request_ids)
        unknown = removed.difference(self._request_ids)
        if unknown:
            raise KeyError(next(iter(unknown)))
        if not removed:
            return
        retained = [
            row_index
            for row_index, request_id in enumerate(self._request_ids)
            if request_id not in removed
        ]
        if not retained:
            self.close()
            return

        try:
            filter_rows(self._cache, retained, stream=self._stream)
        except BaseException:
            self._invalid = True
            raise
        self._request_ids = tuple(
            self._request_ids[row_index] for row_index in retained
        )
        self._token_rows = tuple(self._token_rows[row_index] for row_index in retained)

    def close(self) -> None:
        """Drain the engine stream before releasing even an invalidated group."""
        if self._closed:
            return
        self._invalid = True
        with mx.stream(self._stream):
            mx.synchronize(self._stream)
            self._cache.clear()
            self._request_ids = ()
            self._token_rows = ()
            self._closed = True
