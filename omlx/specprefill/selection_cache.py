# SPDX-License-Identifier: Apache-2.0
"""Memoization of SpecPrefill importance-scoring selections (#1079).

SpecPrefill's ``score -> select`` pipeline is a pure function of
``(tokens_to_score, draft model, keep_pct, chunk_size, n_lookahead, pool_kernel)``.
omlx already caches the *draft model's KV* across requests (``_draft_prefix_cache``),
but re-runs the whole lookahead + importance + chunk-select pipeline for every
request even when the prompt is byte-identical to a recent one -- wasting seconds
per request in agent retry loops that resend the same prompt.

This module caches the *output* of that pipeline (the selected token indices),
keyed on the deterministic inputs, so an identical prompt reuses the prior
selection instead of re-scoring. It is orthogonal to the draft KV cache: one
caches scoring's *input*, the other caches scoring's *result*. The draft KV cache
cannot subsume it -- the selection is what the scoring produces, so nothing keyed
on the selection can be looked up without first re-deriving it; only a cache keyed
on the raw prompt (available before scoring) can short-circuit the scoring itself.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

import mlx.core as mx
import numpy as np

from .policy import SpecPrefillScoringPlan

# Mirror the scoring constants baked into the current draft workflow so a future
# change to any of them cannot silently reuse a stale selection. See
# ``score_tokens`` (n_lookahead=8, pool_kernel=13) and ``select_chunks``
# (chunk_size=32) in ``omlx.patches.specprefill``.
SCORING_CHUNK_SIZE = 32
SCORING_N_LOOKAHEAD = 8
SCORING_POOL_KERNEL = 13

# One selection is a few hundred KB at most; 64 distinct prompts is generous for
# the agent-retry / shared-prefix workloads this targets.
DEFAULT_MAX_ENTRIES = 64

SelectionKey = tuple


def _hash_tokens(tokens: Sequence[int]) -> str:
    """Stable content hash of a token-id sequence (list / ndarray / mx.array)."""
    try:
        arr = np.asarray(tokens, dtype=np.int64)
    except Exception:
        arr = np.asarray(list(tokens), dtype=np.int64)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _indices_to_list(indices: Any) -> list[int]:
    """Convert an mx.array (or any array-like) of indices to a plain int list."""
    tolist = getattr(indices, "tolist", None)
    if callable(tolist):
        return [int(i) for i in tolist()]
    return [int(i) for i in np.asarray(indices).reshape(-1).tolist()]


def build_selection_key(
    *,
    draft_model_id: str,
    tokens_to_score: Sequence[int],
    keep_pct: float,
    chunk_size: int = SCORING_CHUNK_SIZE,
    n_lookahead: int = SCORING_N_LOOKAHEAD,
    pool_kernel: int = SCORING_POOL_KERNEL,
    scorer: str = "draft",
) -> SelectionKey:
    """Build a cache key over the deterministic inputs of the selection.

    ``tokens_to_score`` is the system-prefix-stripped suffix that scoring
    actually runs on (see ``plan_specprefill_scoring``); the selection indices
    are relative to it, so it -- not the raw prompt -- is what we key on.
    ``scorer`` reserves room for the #1078 self-layer scorer to coexist under the
    same memoization.
    """
    return (
        draft_model_id,
        scorer,
        _hash_tokens(tokens_to_score),
        round(float(keep_pct), 6),
        int(chunk_size),
        int(n_lookahead),
        int(pool_kernel),
    )


class SpecPrefillSelectionCache:
    """Bounded, thread-safe LRU mapping a selection key to selected indices.

    Values are plain python int lists (never ``mx.array``) so cached entries do
    not pin Metal buffers or capture lazy graphs.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._max_entries = max(1, int(max_entries))
        self._store: OrderedDict[SelectionKey, list[int]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: SelectionKey) -> list[int] | None:
        with self._lock:
            value = self._store.get(key)
            if value is None:
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return list(value)

    def put(self, key: SelectionKey, indices: Sequence[int]) -> None:
        with self._lock:
            self._store[key] = [int(i) for i in indices]
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


def apply_cached_selection(
    request: Any, plan: SpecPrefillScoringPlan, selected: Sequence[int]
) -> None:
    """Populate ``request`` from a cached selection, mirroring draft scoring.

    Only the selection itself is reused; the per-request position bookkeeping is
    recomputed from *this* request's ``cached_tokens`` because prefix-cache
    warmth can differ run to run. Mirrors ``run_specprefill_draft_scoring``.
    """
    request.specprefill_indices = mx.array(list(selected), dtype=mx.int32)
    request.specprefill_total_tokens = plan.n_to_score
    request.specprefill_position_offset = request.cached_tokens + plan.effective_system
    # Scheduler-owned dynamic field.
    request._specprefill_system_tokens = plan.effective_system


def memoize_scoring(
    *,
    cache: SpecPrefillSelectionCache | None,
    key: SelectionKey,
    request: Any,
    plan: SpecPrefillScoringPlan,
    run_scoring: Callable[[], None],
) -> bool:
    """Reuse a cached selection, or run scoring and cache its result.

    Returns ``True`` on a cache hit (scoring skipped), ``False`` on a miss.
    On a miss, ``run_scoring`` is expected to set ``request.specprefill_indices``
    (draft scoring sets it to ``None`` on failure -- which is deliberately not
    cached, so a transient failure never poisons future requests).
    """
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            apply_cached_selection(request, plan, cached)
            return True

    run_scoring()

    if cache is not None:
        selected = getattr(request, "specprefill_indices", None)
        if selected is not None:
            cache.put(key, _indices_to_list(selected))
    return False
