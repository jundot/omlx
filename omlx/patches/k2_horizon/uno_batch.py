# SPDX-License-Identifier: Apache-2.0
"""GenerationBatch compatibility boundary for singleton Uno decoding.

``tokens`` contains emitted rows; ``_next_tokens`` is the uncached next row.
Uno may hold verified, unconsumed rows beyond that frontier. Snapshots copy and
trim them; merges discard them; final EOS/budget rows use the ordinary lifecycle.
The scheduler calls ``step`` after accepting the current grammar token. Only
this module touches private batch/grammar state on behalf of Uno.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from time import perf_counter

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache, KVCache

from .uno_decode import UnoDecoder


@dataclass
class _State:
    uid: int
    queued: deque = field(default_factory=deque)


def reconcile(batch):
    """Discard unconsumed proposals before ordinary decoding or a batch merge."""
    state = getattr(batch, "_omlx_uno_state", None)
    if state is None:
        return
    if batch.uids != [state.uid]:
        raise RuntimeError("Uno state lost its request owner")
    length = len(batch.tokens[0])
    for cache in batch.prompt_cache:
        excess = (
            int(cache.offset[0].item())
            if isinstance(cache, BatchKVCache)
            else cache.offset
        ) - length
        if excess < 0:
            raise RuntimeError("Uno cache precedes emitted tokens")
        cache.trim(excess)
    del batch._omlx_uno_state


def _constraint(batch):
    from ...api.grammar import GrammarConstraintProcessor
    from .tool_grammar import UnoToolConstraint

    processors = batch.logits_processors[0] or []
    if not processors:
        return None
    if len(processors) != 1 or not isinstance(
        processors[0], GrammarConstraintProcessor
    ):
        raise ValueError(
            "Uno requires supported sampling settings and K2 tool constraints"
        )
    return UnoToolConstraint(
        None,
        batch.model.args.vocab_size,
        matcher=processors[0].matcher.fork(),
    )


def step(batch, ordinary_step):
    """Emit one token through GenerationBatch's existing response lifecycle."""
    enabled = getattr(batch.model, "_omlx_uno_enabled", False)
    if not enabled:
        reconcile(batch)
        return ordinary_step(batch)
    stats = getattr(batch.model, "_omlx_uno_stats", None)
    if stats is None:
        stats = batch.model._omlx_uno_stats = Counter()

    def ordinary():
        reconcile(batch)
        started = perf_counter()
        result = ordinary_step(batch)
        stats["ordinary_seconds"] += perf_counter() - started
        stats["ordinary_tokens"] += len(batch.uids)
        return result

    allowed = getattr(batch.model, "_omlx_uno_singleton", False)
    if not allowed or len(batch.uids) != 1 or not batch._next_logprobs:
        return ordinary()
    token = int(batch._next_tokens[0].item())
    # _step consumes the next token before GenerationBatch checks termination.
    # Do not speculate past EOS (whose grammar matcher is already terminated)
    # or the output budget. The ordinary step materializes the final KV row.
    if (
        token in batch.model._omlx_uno_eos
        or batch._num_tokens[0] + 1 >= batch.max_tokens[0]
    ):
        return ordinary()
    if not all(
        isinstance(cache, (BatchKVCache, KVCache)) for cache in batch.prompt_cache
    ):
        raise ValueError("Uno requires ordinary K2 KV caches")
    sampler = (batch.samplers[0] if batch.samplers else None) or batch.fallback_sampler
    state = getattr(batch, "_omlx_uno_state", None)
    if state is None:
        state = _State(batch.uids[0])
        batch._omlx_uno_state = state
    if batch.uids != [state.uid]:
        raise RuntimeError("Uno state lost its request owner")
    logprobs = batch._next_logprobs
    if not state.queued:
        caches = [
            cache.extract(0) if isinstance(cache, BatchKVCache) else cache
            for cache in batch.prompt_cache
        ]
        seed = int(mx.random.randint(0, 2**31 - 1).item())
        decoder = UnoDecoder(
            batch.model,
            eos_token_ids=batch.model._omlx_uno_eos,
            temperature=sampler.temp,
            top_p=sampler.top_p or 1.0,
            top_k=sampler.top_k or None,
            seed=seed,
            constraint=_constraint(batch),
        )
        remaining = batch.max_tokens[0] - batch._num_tokens[0] - 1
        started = perf_counter()
        cycle = decoder.cycle(
            token,
            cache=caches,
            frontier=len(batch.tokens[0]) + 1,
            max_tokens=remaining,
        )
        stats["cycle_seconds"] += perf_counter() - started
        stats["cycles"] += 1
        stats["proposals"] += min(decoder.block_size, remaining) - 1
        stats["accepted_proposals"] += cycle.accepted_proposals
        state.queued.extend(cycle.tokens)
        batch.prompt_cache = [
            BatchKVCache.merge([cache]) if isinstance(original, BatchKVCache) else cache
            for original, cache in zip(batch.prompt_cache, caches)
        ]
    stats["uno_tokens"] += 1
    batch._current_tokens = batch._next_tokens
    batch._current_logprobs = logprobs
    batch.tokens[0].append(token)
    if any(batch.logits_processors):
        batch._token_context[0].update_and_fetch(batch._current_tokens)
    batch._next_tokens = mx.array([state.queued.popleft()], dtype=mx.uint32)
    # Uno does not expose output logprobs. Do not fabricate a distribution.
    batch._next_logprobs = [None]
    for processor in batch.logits_processors[0] or []:
        processor._pending = True
    return [token], logprobs


def install_cache_hooks():
    from mlx_lm.generate import GenerationBatch

    if getattr(GenerationBatch, "_omlx_uno_cache_hooks", False):
        return
    original_extend = GenerationBatch.extend
    original_extract = GenerationBatch.extract_cache
    original_filter = GenerationBatch.filter

    def extend(self, other):
        reconcile(self)
        reconcile(other)
        return original_extend(self, other)

    def extract(self, index):
        caches = original_extract(self, index)
        state = getattr(self, "_omlx_uno_state", None)
        if state is not None:
            snapshots = []
            for cache in caches:
                snapshot = KVCache()
                snapshot.keys = cache.keys[..., : cache.offset, :]
                snapshot.values = cache.values[..., : cache.offset, :]
                snapshot.offset = cache.offset
                snapshots.append(snapshot)
            caches = snapshots
            length = len(self.tokens[index])
            for cache in caches:
                excess = cache.offset - length
                if excess < 0:
                    raise RuntimeError("Uno snapshot precedes emitted tokens")
                cache.trim(excess)
        return caches

    def filter_rows(self, keep):
        if not keep and hasattr(self, "_omlx_uno_state"):
            del self._omlx_uno_state
        return original_filter(self, keep)

    GenerationBatch.extend = extend
    GenerationBatch.extract_cache = extract
    GenerationBatch.filter = filter_rows
    GenerationBatch._omlx_uno_cache_hooks = True
