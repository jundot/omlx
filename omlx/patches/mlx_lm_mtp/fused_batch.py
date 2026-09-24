# SPDX-License-Identifier: Apache-2.0
"""Shared verification with request-local Lightning MTP acceptance and history."""

from __future__ import annotations

import copy
import logging
import time
from collections import defaultdict

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList

# Text-only distributed ranks can run without mlx-vlm installed.
try:
    from mlx_vlm.models import cache as vlm_cache
    from mlx_vlm.speculative.cache_state import SpeculativeCacheTransaction
except ImportError:
    vlm_cache = None
    SpeculativeCacheTransaction = ()

from . import batch_generator as bg
from . import batched_head

logger = logging.getLogger(__name__)


def _supports_batch_rollback(cache):
    if type(cache) is CacheList or (
        vlm_cache is not None and type(cache) is vlm_cache.CacheList
    ):
        return all(_supports_batch_rollback(part) for part in cache.caches)
    return (
        type(cache) in (ArraysCache, BatchKVCache)
        or (
            vlm_cache is not None
            and type(cache) in (vlm_cache.ArraysCache, vlm_cache.BatchKVCache)
        )
        or getattr(type(cache), "_omlx_mtp_batch_rollback_cache", False)
    )


def _independent_verify(model):
    """These backbones keep a singleton-only context for the draft head."""
    for host in (
        model,
        getattr(model, "language_model", None),
        getattr(model, "_language_model", None),
    ):
        if host is None:
            continue
        if getattr(host, "_omlx_mtp_independent_verify", False):
            return True
    return False


def advance(batch, batch_state):
    """Advance empty row queues, sharing equal-depth target verification.

    Models with vector rollback keep the complete target cache in place.
    Other models use a private view per distinct accepted length and their
    existing scalar rollback contract, including QSA caches.
    Sampling, processors and head caches always belong to an individual UID.
    """
    states = [batch_state.states[uid] for uid in batch.uids]
    if (
        len(states) > 1
        and not _independent_verify(batch.model)
        and all(
            state.chain
            and not state.queue
            and state.next_main is not None
            and state.drafts is not None
            for state in states
        )
        and len({int(state.drafts.shape[0]) for state in states}) == 1
    ):
        # The complete batch already has the required row order and padding.
        # Keep it intact until acceptance determines each committed cache.
        rows = [
            (
                index,
                bg._make_row_batch(
                    batch, index, prompt_cache=batch.prompt_cache, state=state
                ),
                state,
            )
            for index, state in enumerate(states)
        ]
        replacements = {}
        retained = _advance_group(
            batch,
            int(states[0].drafts.shape[0]),
            rows,
            replacements,
            cache=batch.prompt_cache,
        )
        if not retained:
            bg._replace_cache_rows(batch, replacements)
        return

    batched_head.flush(batch_state)
    groups = defaultdict(list)
    replacements = {}
    for index, uid in enumerate(batch.uids):
        state = batch_state.states[uid]
        if state.queue:
            continue
        row = bg._make_row_batch(batch, index, state=state)
        if not state.chain or _independent_verify(batch.model):
            bg._set_singleton_mrope_delta(row)
            bg._run_verify_cycle(row, state)
            replacements[index] = row.prompt_cache
            batch._token_context[index] = row._token_context[0]
            continue
        if state.next_main is None or state.drafts is None:
            raise bg._MtpStepFallback(f"missing draft state for uid={uid}")
        groups[int(state.drafts.shape[0])].append((index, row, state))

    for depth, rows in groups.items():
        _advance_group(batch, depth, rows, replacements)

    bg._replace_cache_rows(batch, replacements)


def _advance_group(batch, depth, rows, replacements, *, cache=None):
    batch_state = getattr(batch, "_omlx_mtp_batch_state", None)
    drafter = bg._drafter_for(batch.model)
    use_head_batch = (
        drafter is None and cache is not None and batched_head.eligible(batch, rows)
    )
    if not use_head_batch:
        batched_head.flush(batch_state)
    # A block drafter drafts every row after the shared commit, like the
    # batched head, so its jobs are collected the same way.
    draft_jobs = [] if use_head_batch or drafter is not None else None
    if len(rows) == 1:
        index, row, state = rows[0]
        bg._set_singleton_mrope_delta(row)
        bg._run_verify_cycle_chain(row, state)
        replacements[index] = row.prompt_cache
        batch._token_context[index] = row._token_context[0]
        return

    whole_batch = cache is not None
    if cache is None:
        cache = bg._merge_row_caches([row.prompt_cache for _, row, _ in rows])
    uids = [state.uid for _, _, state in rows]
    bg._set_batched_mrope_deltas(batch, uids)
    inputs = mx.stack(
        [mx.concatenate([state.next_main, state.drafts]) for _, _, state in rows]
    )
    logger.debug("Lightning MTP shared verify: rows=%d depth=%d", len(rows), depth)
    started = time.perf_counter()
    logits, hidden, gdn, captured = bg._call_backbone_captured(
        batch.model,
        inputs,
        cache,
        n_confirmed=1,
        capture_layer_ids=bg._drafter_capture_ids(batch.model),
    )
    greedy_results = None
    stochastic_results = None
    if depth > 0 and all(
        bg._is_greedy(row) and bg._proc_list(row) is None for _, row, _ in rows
    ):
        # Resolve all acceptance counts and token IDs in one host transfer.
        # Stateful processors and stochastic samplers retain their row path.
        targets = mx.argmax(logits, axis=-1).astype(mx.int32)
        drafts = inputs[:, 1:].astype(mx.int32)
        matches = (targets[:, :-1] == drafts).astype(mx.int32)
        accepted = mx.cumprod(matches, axis=1).sum(axis=1, keepdims=True)
        greedy_results = mx.concatenate([accepted, targets, drafts], axis=1).tolist()
    elif depth > 0 and all(
        not bg._is_greedy(row) and bg._proc_list(row) is None for _, row, _ in rows
    ):
        stochastic_results = mx.stack(
            [
                bg._stochastic_verify_tokens(
                    bg._resolve_sampler(row),
                    bg._logprobs(logits[index]),
                    state.drafts,
                    state.draft_accept_lps,
                )
                for index, (_, row, state) in enumerate(rows)
            ]
        ).tolist()
    else:
        mx.eval(logits, hidden)
    verify_ms = (time.perf_counter() - started) * 1000 / len(rows)
    # mlx-lm backbones return no rollback state; a model-level per-row
    # rollback keeps their merged cache in place the same way. Boundary rows
    # are extracted after the shared rollback, so commit alignment is safe.
    lm_batch_rollback = getattr(batch.model, "mtp_batch_rollback", None)
    lm_vector = (
        whole_batch
        and gdn is None
        and callable(lm_batch_rollback)
        and all(_supports_batch_rollback(layer) for layer in cache)
    )
    vector_rollback = (
        lm_vector
        or isinstance(gdn, SpeculativeCacheTransaction)
        or (
            whole_batch
            and gdn is not None
            and not getattr(batch.model, "_omlx_mtp_commit_align", 0)
            and all(_supports_batch_rollback(layer) for layer in cache)
            and any(
                getattr(host, "_omlx_mtp_batch_rollback", False)
                for host in (batch.model, getattr(batch.model, "_language_model", None))
                if host is not None
            )
        )
    )
    deferred = []
    committed = {depth: cache}

    def commit(accepted, row_index):
        if accepted not in committed:
            view = copy.deepcopy(cache)
            if not bg._chain_rollback(batch.model, view, accepted, depth, gdn):
                raise bg._MtpStepFallback("batched cache rejects scalar rollback")
            committed[accepted] = view
        result = [layer.extract(row_index) for layer in committed[accepted]]
        bg._clear_rollback(result)
        return result

    for row_index, (index, row, state) in enumerate(rows):
        # Cache capability clamps inspect the actual verify undo state.
        # commit() replaces this shared view before the request's head runs.
        row.prompt_cache = cache
        bg._set_singleton_mrope_delta(row)
        result = bg._run_verify_cycle_chain(
            row,
            state,
            verify_result=(
                logits[row_index : row_index + 1],
                hidden[row_index : row_index + 1],
                None,
                bg._slice_captured(captured, row_index),
            ),
            commit_cache=(
                (
                    lambda accepted, i=row_index, s=state: (
                        cache
                        if whole_batch and not s.boundary_emit_pending
                        else [c.extract(i) for c in cache]
                    )
                )
                if vector_rollback
                else (lambda accepted, i=row_index: commit(accepted, i))
            ),
            verify_ms=verify_ms,
            defer_commit=vector_rollback,
            draft_jobs=draft_jobs,
            greedy_result=None if greedy_results is None else greedy_results[row_index],
            stochastic_result=(
                None if stochastic_results is None else stochastic_results[row_index]
            ),
        )
        if vector_rollback:
            deferred.append(result)
            continue
        replacements[index] = row.prompt_cache
        batch._token_context[index] = row._token_context[0]
    if vector_rollback:
        # The model updates KV padding and selects each row's GDN state in
        # place. No extraction or merge is needed for the next target call.
        started = time.perf_counter()
        accepted_rows = [accepted for accepted, _ in deferred]
        if lm_vector:
            if not lm_batch_rollback(cache, accepted_rows, depth):
                raise bg._MtpStepFallback("batched cache rejects vector rollback")
        else:
            batch.model.rollback_speculative_cache(cache, gdn, accepted_rows, depth + 1)
        commit_ms = (time.perf_counter() - started) * 1000 / len(rows)
        for (index, row, _), (_, finish) in zip(rows, deferred):
            bg._set_singleton_mrope_delta(row)
            finish(commit_ms)
            # Boundary forwards advance private row caches that must be merged back.
            if not whole_batch or row.prompt_cache is not cache:
                replacements[index] = row.prompt_cache
                batch._token_context[index] = row._token_context[0]
        if whole_batch:
            batch.prompt_cache = cache
    if draft_jobs is not None:
        if drafter is not None:
            drafter.draft(draft_jobs)
        else:
            batched_head.draft(batch, draft_jobs)
    bg._clear_rollback(cache)
    return vector_rollback and whole_batch and not replacements
