# SPDX-License-Identifier: Apache-2.0
"""Conditional MTP dispatch inside ``mlx_lm.generate.GenerationBatch``.

This is the integration point that lets the existing oMLX scheduler /
paged cache / prefix cache / SSD cache stack drive MTP without touching any
of those layers. ``GenerationBatch`` is mlx-lm's per-step decoder for the
active set of sequences in continuous batching. We patch:

- ``GenerationBatch.__init__`` — after the standard ``_step()`` has run
  the prompt's last token through the backbone, we add an MTP "post-init"
  step that runs one more 1-token backbone forward (with hidden) and one
  MTP-head forward. Two confirmed tokens are queued for emission and a
  draft is stashed for the first verify cycle.

- ``GenerationBatch.next`` — when the batch holds exactly one MTP-capable
  sequence we emit from the per-batch queue first; once empty, we run a
  2-token verify forward over ``[next_main, draft]`` with
  ``n_confirmed=1`` and a single MTP-head forward at the bonus position
  (accept) or confirmed position (reject), refilling the queue from the
  verify outputs.

The throughput math (greedy, accept rate p):
  - Cost per *cycle*: 1× backbone (2-token verify) + 1× MTP head ≈ 1.15
  - Tokens per cycle: 1 + p (accept emits draft+bonus; reject emits verify_pred only)
  - At p≈1: 0.575 cost/token → ~1.74× throughput
  - At p≈0.5: ~0.77 cost/token → ~1.30× throughput

Greedy identity (sampler is None): the patched dispatch produces the same
tokens as the standard step. PR 990's ``test_mtp_generate_identity``
encodes this contract; the oMLX-side equivalent lives in
``tests/test_mlx_lm_mtp_patch.py``.

Stochastic acceptance (sampler is not None): we use ``min(1, p_target / p_draft)``
(Leviathan & Chen 2023). On rejection we sample from the residual
``max(p_target - p_draft, 0) / Z`` so the marginal output distribution
equals the target distribution exactly.

PagedCacheManager interaction
-----------------------------
``cache.trim(1)`` on a ``BatchKVCache`` only updates ``self._idx``; the
underlying paged blocks are untouched. ``ArraysCache.rollback_state``
holds ``(conv_snap, ssm_snap)`` snapshots produced by the patched
``GatedDeltaNet.__call__`` and is restored on reject. Because both code
paths only mutate cache *length* (not block ownership), oMLX's
``PagedCacheManager`` is oblivious to the trim — its block_table is
unaffected and prefix-cache lookups continue to work normally.

TokenBuffer interaction
-----------------------
``GenerationBatch._token_context[0]`` is a ``TokenBuffer`` accumulating
the prompt + every forward-input token. We update it in lock-step with
each forward-input position so that ``logits_processors`` see the same
token sequence the standard step would see. On reject we shrink the
buffer's ``_size`` by 1 to discard the rejected draft (mirroring PR 990's
``prev_tokens = prev_tokens[:-1]``).
"""

from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PATCHED = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply() -> bool:
    """Wrap ``GenerationBatch.__init__`` + ``GenerationBatch.next``."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from mlx_lm.generate import GenerationBatch
    except ImportError:
        logger.debug("mlx_lm.generate.GenerationBatch not importable")
        return False

    if hasattr(GenerationBatch, "_omlx_mtp_patched"):
        _PATCHED = True
        return True

    original_init = GenerationBatch.__init__
    original_next = GenerationBatch.next
    original_filter = GenerationBatch.filter
    original_extend = GenerationBatch.extend

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if _is_mtp_eligible(self):
            try:
                _post_init_mtp(self)
                logger.info(
                    "MTP path activated for uid=%s (model has mtp_forward, batch=1)",
                    getattr(self, "uids", ["?"])[0],
                )
            except _MtpStepFallback as exc:
                logger.warning("MTP post-init fallback: %s", exc)
        else:
            # The empty-batch case is BatchGenerator.__init__ pre-creating
            # ``self._generation_batch = GenerationBatch.empty(...)`` and is
            # always part of normal startup — silence it. Only log when the
            # batch is genuinely populated (e.g. continuous batching with
            # batch>1) so the message points at a real misconfiguration.
            uids = getattr(self, "uids", None)
            if uids:
                reason = _ineligibility_reason(self)
                if reason:
                    logger.debug("MTP path not active: %s", reason)

    def patched_next(self, *args, **kwargs):
        if _is_mtp_eligible(self):
            state = getattr(self, "_omlx_mtp_state", None)
            if state is not None:
                try:
                    return _mtp_next(self, state)
                except _MtpStepFallback as exc:
                    logger.debug(
                        "MTP next() fallback to standard step: %s", exc
                    )
                    # Best-effort: drop state so subsequent calls don't try
                    # to resume a half-built MTP cycle from a stale snapshot.
                    if hasattr(self, "_omlx_mtp_state"):
                        try:
                            delattr(self, "_omlx_mtp_state")
                        except AttributeError:
                            pass
        return original_next(self, *args, **kwargs)

    def patched_extend(self, batch, *args, **kwargs):
        # ``BatchGenerator._next()`` builds a fresh single-sequence
        # ``GenerationBatch`` via ``prompt_batch.split(...).generate(...)``
        # then merges it into ``self._generation_batch`` via extend(). The
        # MTP post-init runs on the fresh batch (since that's the one whose
        # __init__ fires with uids=[0]); without this transfer the state
        # would die with the donor instance.
        donor_state = getattr(batch, "_omlx_mtp_state", None)
        result = original_extend(self, batch, *args, **kwargs)
        if donor_state is not None and not hasattr(self, "_omlx_mtp_state"):
            self._omlx_mtp_state = donor_state
            try:
                delattr(batch, "_omlx_mtp_state")
            except AttributeError:
                pass
            logger.debug(
                "MTP state transferred from donor batch to host batch (uid=%s)",
                getattr(self, "uids", ["?"])[0] if getattr(self, "uids", None) else "?",
            )
        return result

    def patched_filter(self, keep, *args, **kwargs):
        # When the outer scheduler retires this sequence (e.g. EOS detected
        # outside our finish path), it calls filter([]) to drop everything.
        # Surface stats here so the user sees them even when the standard
        # _emit_response finish path doesn't fire.
        state = getattr(self, "_omlx_mtp_state", None)
        result = original_filter(self, keep, *args, **kwargs)
        if state is not None and not getattr(self, "uids", None):
            # Batch is now empty — log + drop state.
            try:
                _log_mtp_stats(
                    "?", state.stats, getattr(state, "_finish_reason", "external")
                )
            except Exception:
                pass
            try:
                delattr(self, "_omlx_mtp_state")
            except AttributeError:
                pass
        return result

    GenerationBatch.__init__ = patched_init
    GenerationBatch.next = patched_next
    GenerationBatch.filter = patched_filter
    GenerationBatch.extend = patched_extend
    GenerationBatch._omlx_mtp_patched = True
    _PATCHED = True
    return True


def _model_has_mtp_module(model: Any) -> bool:
    """Check whether the model actually has an MTP head attached.

    The ``mtp_forward`` method is added to the class unconditionally by
    the patch, but the per-instance ``mtp`` module is only attached when
    ``mtp_enabled`` was True at load time (see qwen35_model._patch_model
    and deepseek_v4_model._patch_model). Without the inner module the
    ``mtp_forward`` call would AttributeError, so we gate eligibility on
    the actual module's presence.
    """
    inner = getattr(model, "language_model", model)
    return hasattr(inner, "mtp") and getattr(inner, "mtp", None) is not None


def _is_mtp_eligible(gen_batch: Any) -> bool:
    """``__init__`` and ``next`` only engage MTP for single-sequence batches
    when the model exposes ``mtp_forward`` *and* has an attached MTP head."""
    if not hasattr(gen_batch, "model"):
        return False
    if not hasattr(gen_batch.model, "mtp_forward"):
        return False
    if not _model_has_mtp_module(gen_batch.model):
        return False
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) != 1:
        return False
    return True


def _ineligibility_reason(gen_batch: Any) -> str:
    """Return a short human-readable reason for why the MTP path isn't active.

    Only used for debug logging — the patched_init / patched_next paths
    don't act on this string.
    """
    if not hasattr(gen_batch, "model"):
        return "GenerationBatch has no .model attribute"
    if not hasattr(gen_batch.model, "mtp_forward"):
        return (
            f"model {type(gen_batch.model).__module__}.{type(gen_batch.model).__name__} "
            "has no mtp_forward (qwen35 patch may not have applied to this class)"
        )
    if not _model_has_mtp_module(gen_batch.model):
        return "model has no attached mtp head (mtp_enabled was False at load time)"
    uids = getattr(gen_batch, "uids", None)
    if uids is None:
        return "GenerationBatch has no uids"
    if len(uids) != 1:
        return f"batch size {len(uids)} != 1 (continuous batching, MTP off by design)"
    return ""


class _MtpStepFallback(RuntimeError):
    """Raised inside the MTP path to signal a clean fallback to the standard step."""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class _MtpStats:
    """Acceptance / throughput counters for one MTP-active sequence.

    Logged at INFO when the sequence finishes (length / stop / filter)
    so the operator can see whether the draft+verify cycle is actually
    productive on this model + sampler combo.
    """

    cycles: int = 0  # number of verify cycles run
    accepts: int = 0  # cycles where the draft was accepted
    rejects: int = 0  # cycles where the draft was rejected
    init_emits: int = 0  # tokens emitted from the post-init queue (always 2)
    draft_emits: int = 0  # tokens emitted as accepted drafts
    bonus_emits: int = 0  # tokens emitted as bonus (accepted + emit_bonus)
    verify_emits: int = 0  # tokens emitted as verify-position correction (reject path)
    # Component-level timings. Help diagnose where MTP overhead comes from
    # when accept rate is healthy but wall-clock throughput isn't.
    backbone_ms: float = 0.0  # cumulative time inside the 2-token verify forward
    mtp_head_ms: float = 0.0  # cumulative time inside MTP-head forwards
    sample_ms: float = 0.0  # cumulative time in sampling + acceptance check
    cache_ops_ms: float = 0.0  # cumulative time in trim / rollback restore


@dataclass
class _MtpState:
    """Per-batch MTP state stashed on the GenerationBatch instance."""

    # Pending tokens to emit in upcoming next() calls. Each entry is
    # (token_id_int, logprobs_1d, source_label). source_label is one of
    # "init", "draft", "bonus", "verify" — used to bucket stats correctly
    # when the queue is drained.
    queue: Deque[Tuple[int, Any, str]] = field(default_factory=deque)

    # Cache for the MTP head (separate from gen_batch.prompt_cache).
    mtp_cache: Optional[List[Any]] = None

    # First input token of the next verify forward. Tracked as a 1-element
    # mx.array (uint32) so it can be concatenated with `draft_tok` cheaply.
    next_main: Optional[Any] = None

    # Draft logprobs (vocab,) needed by stochastic acceptance / residual sampling.
    draft_tok: Optional[Any] = None  # (1,) uint32
    draft_lp: Optional[Any] = None  # (vocab,) float
    # Filtered (sampler-applied) draft logprobs reused by the next cycle's
    # acceptance ratio + residual sampling. Mirrors PR 990's accept_lp,
    # adapted to oMLX's callable-sampler contract via metadata-introspection.
    # None when the sampler exposes no metadata (raw-lp fallback path).
    draft_accept_lp: Optional[Any] = None  # (vocab,) float
    # Host-side int copy of draft_tok. Cached at draft creation time so the
    # verify cycle can compare draft vs verify ids without a separate
    # GPU→CPU sync (`int(draft_tok.tolist()[0])` would force a stall).
    draft_id: int = -1

    # Accept-rate / throughput counters. Surfaced via logger.info on finish.
    stats: _MtpStats = field(default_factory=_MtpStats)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_generation_stream():
    """Return the ``mlx_lm.generate`` module-level generation stream.

    The standard ``GenerationBatch._step`` runs all forward passes inside
    ``mx.stream(generation_stream)``; the MTP cycle does the same so the
    paged cache writes land on the same stream and ordering is preserved.
    The stream lives on the *outer* ``BatchGenerator``, not on
    ``GenerationBatch``, so we read it from the module.

    Note: ``mlx_lm.__init__`` re-exports a ``generate`` *function*, so
    ``import mlx_lm.generate as mlg`` resolves to the function, not the
    module. We use ``sys.modules`` to grab the actual module.
    """
    import sys

    return sys.modules["mlx_lm.generate"].generation_stream


def _resolve_sampler(gen_batch: Any):
    """Match ``GenerationBatch._step``'s per-sequence sampler resolution (batch=1)."""
    if gen_batch.samplers and gen_batch.samplers[0] is not None:
        return gen_batch.samplers[0]
    return gen_batch.fallback_sampler


def _is_greedy(gen_batch: Any) -> bool:
    """Heuristic mirroring PR 990's ``sampler is None``."""
    if gen_batch.samplers and gen_batch.samplers[0] is not None:
        return False
    return True


def _proc_list(gen_batch: Any) -> Optional[List[Any]]:
    if gen_batch.logits_processors and gen_batch.logits_processors[0]:
        return gen_batch.logits_processors[0]
    return None


def _apply_processors(processors, prev_tokens, logits_2d):
    if not processors:
        return logits_2d
    for proc in processors:
        logits_2d = proc(prev_tokens, logits_2d)
    return logits_2d


def _logprobs(logits_2d):
    import mlx.core as mx

    return logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)


def _accept_lp_for(sampler, lp):
    """Reproduce the sampler's filter+temperature pipeline on `lp` so the
    acceptance ratio (and residual distribution) match the distribution the
    sampler actually drew from.

    Reads sampling params off the callable as function attributes (set by
    ``omlx.utils.sampling.make_sampler``). For samplers without metadata —
    e.g. mlx-lm stock callables, fallback samplers — returns `lp` unchanged
    so behavior matches the pre-PR-990 raw-lp acceptance.
    """
    import mlx.core as mx

    from omlx.utils.sampling import apply_min_p, apply_top_k, apply_top_p

    temp = float(getattr(sampler, "temp", 0.0) or 0.0)
    if temp == 0.0:
        # Greedy / unknown sampler — raw lp is the acceptance distribution.
        return lp

    out = lp
    top_p = float(getattr(sampler, "top_p", 0.0) or 0.0)
    if 0.0 < top_p < 1.0:
        out = apply_top_p(out, top_p)
    min_p = float(getattr(sampler, "min_p", 0.0) or 0.0)
    if min_p != 0.0:
        min_keep = int(getattr(sampler, "min_tokens_to_keep", 1) or 1)
        out = apply_min_p(out, min_p, min_keep)
    top_k = int(getattr(sampler, "top_k", 0) or 0)
    if top_k > 0:
        out = apply_top_k(out, top_k)

    # Temperature scale + renormalize so the output is a proper logprob
    # distribution that can be indexed by token id for the acceptance check.
    scaled = out * (1.0 / temp)
    return scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)


def _trim_token_buffer(gen_batch: Any, n: int) -> None:
    """Shrink ``_token_context[0]`` by ``n`` (mirrors PR 990 ``prev[:-n]``)."""
    if n <= 0:
        return
    procs = _proc_list(gen_batch)
    if procs is None:
        return
    buf = gen_batch._token_context[0]
    buf._size = max(0, buf._size - n)


def _restore_or_trim_caches(prompt_cache: List[Any]) -> bool:
    """Roll back one token from each layer cache after a draft rejection.

    SSM / linear-attention layers expose ``rollback_state`` populated by the
    patched ``GatedDeltaNet.__call__``; we restore that snapshot. Standard
    KV cache layers (full-attention) expose ``trim`` and ``is_trimmable``;
    we trim by 1. Layers that support neither cause the entire MTP step to
    fall back to the standard path.
    """
    for c in prompt_cache:
        rollback = getattr(c, "rollback_state", None)
        if rollback is not None:
            conv_snap, ssm_snap = rollback
            c[0] = conv_snap
            c[1] = ssm_snap
            c.rollback_state = None
            continue
        if hasattr(c, "is_trimmable") and c.is_trimmable():
            c.trim(1)
            continue
        return False
    return True


def _rollback_after_reject(
    model: Any,
    prompt_cache: List[Any],
    gdn_states: Optional[list],
    accepted: int = 0,
    block_size: int = 2,
) -> bool:
    """Roll back per-layer cache state after a rejected MTP draft token.

    Two mechanisms are supported, dispatched on the model's capability:

    1. **mlx-vlm path** — when the model exposes ``rollback_speculative_cache``
       (Qwen3.5 LanguageModel ships with it upstream) AND ``gdn_states`` is
       populated, we delegate to that method. It batches the per-layer SSM
       replay into a single ``gated_delta_update`` call and trims KV
       caches by ``block_size - (accepted + 1)``. The backbone forward was
       run with both confirmed and draft tokens; the rollback replays only
       the accepted prefix through the original pre-update state.

    2. **mlx-lm path** (PR 990) — per-layer ``cache.rollback_state`` snapshot
       written by the patched ``GatedDeltaNet.__call__`` during the
       confirmed/draft split. We restore the snapshot for SSM layers and
       trim KV layers by 1. ``gdn_states`` is None in this path.

    Returns True on success. False means a cache layer in the list supports
    neither mechanism, in which case the caller falls back to the standard
    non-MTP step.
    """
    if gdn_states is not None and hasattr(model, "rollback_speculative_cache"):
        model.rollback_speculative_cache(
            prompt_cache, gdn_states, accepted, block_size
        )
        return True
    return _restore_or_trim_caches(prompt_cache)


def _call_backbone(
    model: Any,
    inputs: Any,
    cache: List[Any],
    n_confirmed: int = 0,
) -> Tuple[Any, Any, Optional[list]]:
    """Run the backbone with ``return_hidden=True`` and normalise the result.

    Returns ``(logits, hidden_pre_norm, gdn_states_or_None)``:

    - mlx-lm path returns the 2-tuple ``(logits, hidden)``; ``gdn_states``
      is ``None`` and rollback uses ``cache.rollback_state``.
    - mlx-vlm path returns the 3-tuple ``(logits, hidden, gdn_states)`` so
      a rejected draft can be rolled back via
      ``rollback_speculative_cache``.

    ``n_confirmed`` is forwarded so the mlx-lm path can split its
    GatedDeltaNet forward into confirmed and draft chunks. mlx-vlm
    discards it (irrelevant — rollback is post-hoc, not splitwise).
    """
    kwargs = {"cache": cache, "return_hidden": True}
    if n_confirmed:
        kwargs["n_confirmed"] = n_confirmed
    result = model(inputs, **kwargs)
    if isinstance(result, tuple):
        if len(result) == 3:
            return result
        if len(result) == 2:
            return result[0], result[1], None
    raise TypeError(
        f"backbone returned unexpected shape: {type(result).__name__}"
    )


def _clear_rollback(prompt_cache: List[Any]) -> None:
    """Drop ``rollback_state`` snapshots after a draft is accepted."""
    for c in prompt_cache:
        if hasattr(c, "rollback_state") and c.rollback_state is not None:
            c.rollback_state = None


def _ensure_uint32(arr):
    """Ensure a 1-element mx.array is uint32 (cache update_and_fetch expects it)."""
    import mlx.core as mx

    if arr.dtype == mx.uint32:
        return arr
    return arr.astype(mx.uint32)


# ---------------------------------------------------------------------------
# Post-init: run one extra backbone forward + MTP forward; queue the two
# emitted tokens; stash a draft for the first verify cycle.
# ---------------------------------------------------------------------------

def _post_init_mtp(gen_batch: Any) -> None:
    """Bridge from standard ``__init__``'s ``_step()`` into PR 990's cycle 1.

    State on entry (after standard ``__init__``):
      - cache contains the prompt up to ``prompt[-1]`` inclusive
      - ``_next_tokens`` = ``main_tok`` (token sampled from ``prompt[-1]``'s logits)
      - ``_next_logprobs[0]`` = main_tok's distribution
      - ``tokens[0]`` = original prompt list

    We perform one more 1-token backbone forward (so the cache also includes
    ``main_tok`` and we obtain the hidden state at that position), run the
    MTP head to produce a draft for the next verify cycle, and seed
    ``state.queue`` with two confirmed tokens — ``main_tok`` and the
    standard-sample at the next position. After this, the queue handles
    the first two emit calls and the third call enters the verify cycle.

    If the batch was empty when ``__init__`` ran, ``_next_tokens`` is
    ``None`` — we leave MTP inactive and the standard path runs unchanged.
    """
    import mlx.core as mx

    if gen_batch._next_tokens is None or not gen_batch.uids:
        # Nothing was sampled in the standard _step (empty batch). The
        # next() call will be a no-op anyway; leave the patch inert.
        return

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)

    main_tok = _ensure_uint32(gen_batch._next_tokens)  # (1,)
    main_lp = gen_batch._next_logprobs[0]  # (vocab,)

    if procs is not None:
        prev_buf = gen_batch._token_context[0].update_and_fetch(main_tok)
    else:
        prev_buf = None

    # 1-token backbone forward at main_tok with hidden state. No draft yet,
    # so no rollback is possible — discard gdn_states.
    with mx.stream(_get_generation_stream()):
        logits, hidden, _ = _call_backbone(
            gen_batch.model, main_tok[:, None], gen_batch.prompt_cache
        )

    next_main_logits = logits[:, -1, :]  # (1, vocab) — distribution after main_tok
    next_main_logits = _apply_processors(procs, prev_buf, next_main_logits)
    next_main_lp = _logprobs(next_main_logits)
    next_main_tok = sampler(next_main_lp)  # (1,)

    # MTP head sees (hidden_at_main, next_main_tok) and proposes the draft
    # that the *next* verify cycle will check against forward([next_main, draft]).
    mtp_cache = gen_batch.model.make_mtp_cache()
    hidden_at_main = hidden[:, -1:, :]  # (1, 1, H)
    next_ids = next_main_tok.reshape(1, 1)
    with mx.stream(_get_generation_stream()):
        mtp_logits = gen_batch.model.mtp_forward(hidden_at_main, next_ids, mtp_cache)
    mtp_logits_2d = mtp_logits[:, -1, :]
    if procs is not None:
        prev_with_main_and_next = mx.concatenate(
            [prev_buf, _ensure_uint32(next_main_tok)]
        )
        mtp_logits_2d = _apply_processors(
            procs, prev_with_main_and_next, mtp_logits_2d
        )
    draft_lp_2d = _logprobs(mtp_logits_2d)
    draft_tok = sampler(draft_lp_2d)
    # Filtered draft lp — what the sampler actually drew from. The next
    # cycle's acceptance ratio uses this so the math matches the
    # sampling distribution rather than the raw softmax.
    draft_accept_lp_2d = _accept_lp_for(sampler, draft_lp_2d)

    mx.eval(main_tok, next_main_tok, draft_tok)

    # Queue the two confirmed tokens (main_tok + next_main_tok); their
    # logprobs come from the standard / patched samplers. Cache draft_id
    # while the array is already evaluated to avoid re-syncing in cycle 1.
    state = _MtpState()
    state.mtp_cache = mtp_cache
    state.next_main = _ensure_uint32(next_main_tok)
    state.draft_tok = _ensure_uint32(draft_tok)
    state.draft_lp = draft_lp_2d.squeeze(0)
    state.draft_accept_lp = draft_accept_lp_2d.squeeze(0)
    state.draft_id = int(draft_tok.tolist()[0])
    state.queue.append((int(main_tok.tolist()[0]), main_lp, "init"))
    state.queue.append(
        (int(next_main_tok.tolist()[0]), next_main_lp.squeeze(0), "init")
    )

    gen_batch._omlx_mtp_state = state


# ---------------------------------------------------------------------------
# next() dispatch
# ---------------------------------------------------------------------------

def _mtp_next(gen_batch: Any, state: _MtpState) -> Any:
    """Emit one token; run a verify cycle if the queue is empty."""
    if state.queue:
        token_id, logprobs_1d, source = state.queue.popleft()
        _bump_emit_stat(state, source)
        return _emit_response(gen_batch, token_id, logprobs_1d, state.stats)

    _run_verify_cycle(gen_batch, state)
    if not state.queue:
        # Verify cycle should always populate the queue with at least the
        # rejected-verify token; if it didn't, fall back to the standard
        # step rather than yield an undefined response.
        raise _MtpStepFallback("verify cycle produced no emit tokens")

    token_id, logprobs_1d, source = state.queue.popleft()
    _bump_emit_stat(state, source)
    return _emit_response(gen_batch, token_id, logprobs_1d, state.stats)


def _log_mtp_stats(uid: Any, stats: "_MtpStats", finish_reason: str) -> None:
    """Emit a one-line summary of MTP draft/verify activity for a finished sequence.

    Format chosen to match PR 990's headline metrics, plus component timings
    that make wall-clock vs. accept-rate gaps debuggable:
      MTP[<uid>] finish=<reason> tokens=<N> cycles=<C> accept=<A>/<C> (<rate>%)
        emits[init=<i>,draft=<d>,bonus=<b>,verify=<v>]
        timing[backbone=<X>ms mtp=<Y>ms sample=<S>ms cache=<C>ms]
    """
    total_emits = (
        stats.init_emits + stats.draft_emits + stats.bonus_emits + stats.verify_emits
    )
    if stats.cycles > 0:
        rate_str = f"{stats.accepts / stats.cycles * 100:.1f}%"
    else:
        rate_str = "n/a"
    logger.info(
        "MTP[%s] finish=%s tokens=%d cycles=%d accept=%d/%d (%s) "
        "emits[init=%d,draft=%d,bonus=%d,verify=%d] "
        "timing[backbone=%.1fms mtp=%.1fms sample=%.1fms cache=%.1fms]",
        uid,
        finish_reason,
        total_emits,
        stats.cycles,
        stats.accepts,
        stats.cycles,
        rate_str,
        stats.init_emits,
        stats.draft_emits,
        stats.bonus_emits,
        stats.verify_emits,
        stats.backbone_ms,
        stats.mtp_head_ms,
        stats.sample_ms,
        stats.cache_ops_ms,
    )


def _bump_emit_stat(state: _MtpState, source: str) -> None:
    if source == "init":
        state.stats.init_emits += 1
    elif source == "draft":
        state.stats.draft_emits += 1
    elif source == "bonus":
        state.stats.bonus_emits += 1
    elif source == "verify":
        state.stats.verify_emits += 1


# ---------------------------------------------------------------------------
# Verify cycle: 2-token forward + accept/reject + MTP forward for next draft.
# ---------------------------------------------------------------------------

def _run_verify_cycle(gen_batch: Any, state: _MtpState) -> None:
    """Run one verify cycle. Populates ``state.queue`` with 1 (reject) or 2
    (accept) tokens for upcoming emit calls. Updates ``state.next_main`` and
    ``state.draft_tok`` / ``state.draft_lp`` for the cycle after that.
    """
    import time

    import mlx.core as mx

    if state.next_main is None or state.draft_tok is None:
        raise _MtpStepFallback("verify cycle entered without next_main / draft")

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)
    is_greedy = _is_greedy(gen_batch)

    inputs = mx.concatenate([state.next_main, state.draft_tok])  # (2,)

    # Update the token buffer per-position (mirrors PR 990 _step_backbone).
    prev_main = None
    prev_draft = None
    if procs is not None:
        prev_main = gen_batch._token_context[0].update_and_fetch(state.next_main)
        prev_draft = gen_batch._token_context[0].update_and_fetch(state.draft_tok)

    # --- backbone forward + sample (single eval point) ---
    # Dispatch backbone, processors, logprobs, and sampler all on stream
    # without forcing intermediate evaluation. The single ``mx.eval`` after
    # sampling resolves the whole graph in one stall instead of two.
    # Tradeoff: backbone_ms / sample_ms split is no longer wall-clock
    # accurate (everything lands in sample_ms), but cumulative timing is.
    t0 = time.perf_counter()
    with mx.stream(_get_generation_stream()):
        logits, hidden, gdn_states = _call_backbone(
            gen_batch.model,
            inputs[None, :],
            gen_batch.prompt_cache,
            n_confirmed=1,
        )
        verify_logits = logits[:, 0, :]
        bonus_logits = logits[:, 1, :]
    state.stats.backbone_ms += (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if procs is not None:
        verify_logits = _apply_processors(procs, prev_main, verify_logits)
        bonus_logits = _apply_processors(procs, prev_draft, bonus_logits)
    # Batched logprobs: one logsumexp over (2, vocab) instead of two over
    # (1, vocab). Shaves one reduction per cycle on the vocab dimension.
    combined_logits = mx.concatenate(
        [verify_logits, bonus_logits], axis=0
    )  # (2, vocab)
    combined_lp = combined_logits - mx.logsumexp(
        combined_logits, axis=-1, keepdims=True
    )
    verify_lp_2d = combined_lp[0:1]
    bonus_lp_2d = combined_lp[1:2]
    verify_tok = sampler(verify_lp_2d)
    bonus_tok = sampler(bonus_lp_2d)
    mx.eval(verify_tok, bonus_tok)

    # ``draft_id`` was cached when the draft was sampled (post_init or the
    # prior _step_mtp); skip the GPU→CPU sync that ``state.draft_tok.tolist()``
    # would impose on every cycle.
    draft_id = state.draft_id
    verify_id = int(verify_tok.tolist()[0])
    bonus_id = int(bonus_tok.tolist()[0])

    # Filtered logprobs — distribution the sampler actually drew from.
    # Used for acceptance ratio + residual sampling so they match the
    # sampling distribution rather than raw softmax (PR 990 alignment).
    verify_accept_lp = _accept_lp_for(sampler, verify_lp_2d)
    draft_accept_lp = (
        state.draft_accept_lp
        if state.draft_accept_lp is not None
        else _accept_lp_for(sampler, state.draft_lp)
    )

    if is_greedy:
        accept = verify_id == draft_id
    else:
        log_accept = (
            verify_accept_lp[0, draft_id].item()
            - draft_accept_lp[draft_id].item()
        )
        accept = log_accept >= 0 or random.random() < math.exp(log_accept)
    state.stats.sample_ms += (time.perf_counter() - t0) * 1000

    hidden_at_confirmed = hidden[:, 0:1, :]
    hidden_at_draft = hidden[:, 1:2, :]

    state.stats.cycles += 1
    if accept:
        state.stats.accepts += 1
        # --- cache cleanup (timed) ---
        t0 = time.perf_counter()
        _clear_rollback(gen_batch.prompt_cache)
        state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

        # --- MTP head forward for next draft (timed inside _step_mtp) ---
        new_draft, new_draft_lp = _step_mtp(
            gen_batch,
            hidden_at_draft,
            _ensure_uint32(bonus_tok),
            prev_buf=prev_draft if procs is not None else None,
            stats=state.stats,
        )
        # Queue the two emitted tokens. Per PR 990: the accepted draft uses
        # the *MTP head's* original draft distribution as its logprobs; the
        # bonus uses the verify forward's bonus distribution.
        state.queue.append((draft_id, state.draft_lp, "draft"))
        state.queue.append((bonus_id, bonus_lp_2d.squeeze(0), "bonus"))
        state.next_main = _ensure_uint32(bonus_tok)
        state.draft_tok = new_draft
        state.draft_lp = new_draft_lp
        return

    # Reject path.
    state.stats.rejects += 1
    t0 = time.perf_counter()
    # accepted=0 means only the confirmed token (verify position) is kept;
    # block_size=2 covers both the confirmed and the rejected draft.
    if not _rollback_after_reject(
        gen_batch.model, gen_batch.prompt_cache, gdn_states,
        accepted=0, block_size=2,
    ):
        if procs is not None:
            _trim_token_buffer(gen_batch, 1)
        raise _MtpStepFallback("cache layer rejects rollback")
    if procs is not None:
        _trim_token_buffer(gen_batch, 1)
    state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

    # Pick the verify-position emit token: residual sample for stochastic.
    # Residual is computed on the *filtered* distributions so the sample
    # comes from `max(p_target_filt - p_draft_filt, 0)` — matching what the
    # sampler would have produced if it had drawn directly from the verify
    # position. emit_lp returned to the caller stays as the raw verify lp
    # so downstream logprobs reporting is consistent with non-MTP paths.
    if is_greedy:
        emit_id = verify_id
        emit_lp = verify_lp_2d.squeeze(0)
    else:
        emit_id, _ = _residual_sample(verify_accept_lp, draft_accept_lp)
        emit_lp = verify_lp_2d.squeeze(0)

    emit_tok = mx.array([emit_id], dtype=mx.uint32)
    new_draft, new_draft_lp = _step_mtp(
        gen_batch,
        hidden_at_confirmed,
        emit_tok,
        prev_buf=prev_main if procs is not None else None,
        stats=state.stats,
    )

    state.queue.append((emit_id, emit_lp, "verify"))
    state.next_main = emit_tok
    state.draft_tok = new_draft
    state.draft_lp = new_draft_lp


# ---------------------------------------------------------------------------
# Helpers used by the verify cycle.
# ---------------------------------------------------------------------------

def _step_mtp(
    gen_batch: Any,
    hidden_at_position: Any,
    next_main_tok: Any,
    prev_buf: Optional[Any],
    stats: Optional["_MtpStats"] = None,
) -> Tuple[Any, Any]:
    """Run one MTP-head forward + sample. Returns ``(draft_tok, draft_lp)``.

    Side effect: caches the host-side int copy of the new draft on
    ``gen_batch._omlx_mtp_state.draft_id`` so the next verify cycle's
    accept check is sync-free.
    """
    import time

    import mlx.core as mx

    state = gen_batch._omlx_mtp_state
    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)

    t0 = time.perf_counter()
    next_ids = next_main_tok.reshape(1, 1)
    with mx.stream(_get_generation_stream()):
        mtp_logits = gen_batch.model.mtp_forward(
            hidden_at_position, next_ids, state.mtp_cache
        )
        mtp_logits_2d = mtp_logits[:, -1, :]
    if procs is not None and prev_buf is not None:
        prev_with_next = mx.concatenate(
            [prev_buf, _ensure_uint32(next_main_tok)]
        )
        mtp_logits_2d = _apply_processors(procs, prev_with_next, mtp_logits_2d)
    new_lp = _logprobs(mtp_logits_2d)
    new_tok = sampler(new_lp)
    # Filtered draft lp — what the sampler actually drew from. The next
    # verify cycle's acceptance ratio uses this so the math matches the
    # sampling distribution rather than raw softmax (PR 990 alignment).
    new_accept_lp = _accept_lp_for(sampler, new_lp)
    # ``.tolist()`` forces evaluation; replaces the explicit ``mx.eval`` and
    # piggybacks the host-side int caching on the same sync.
    draft_id_int = int(new_tok.tolist()[0])
    state.draft_id = draft_id_int
    state.draft_accept_lp = new_accept_lp.squeeze(0)
    if stats is not None:
        stats.mtp_head_ms += (time.perf_counter() - t0) * 1000
    return _ensure_uint32(new_tok), new_lp.squeeze(0)


def _residual_sample(verify_lp_2d: Any, draft_lp_1d: Any) -> Tuple[int, Any]:
    """Sample from ``max(p_target - p_draft, 0)`` (Leviathan et al. 2022).

    On degenerate input (residual all zero) falls back to the target
    distribution rather than the verify-position argmax — keeps the sample
    drawn from a proper distribution and stays in-graph (no host sync).
    Mirrors mlx-lm PR 990 commit 6594348.

    Returns ``(token_id_int, verify_lp_1d)``.
    """
    import mlx.core as mx

    p_target = mx.exp(verify_lp_2d.squeeze(0))
    p_draft = mx.exp(draft_lp_1d)
    residual = mx.maximum(p_target - p_draft, 0.0)
    # Keep z in graph; mx.where switches to the target distribution when
    # the residual mass is zero. ``categorical`` treats log(0) = -inf as
    # p=0 so no safety epsilon is needed.
    z = residual.sum(keepdims=True)
    dist = mx.where(z > 0, residual, p_target)
    sample = mx.random.categorical(mx.log(dist).reshape(1, -1))
    return int(sample.item()), verify_lp_2d.squeeze(0)


# ---------------------------------------------------------------------------
# Response builder — mirrors GenerationBatch.next()'s per-sequence epilogue.
# ---------------------------------------------------------------------------

def _emit_response(
    gen_batch: Any,
    token_id: int,
    logprobs_1d: Any,
    stats: Optional["_MtpStats"] = None,
) -> List[Any]:
    """Produce a single-element response list, applying the standard
    epilogue (token append + max_tokens / matcher checks) so external
    callers (BatchGenerator, scheduler, response stream) see the same
    contract as the unmodified next().
    """
    Response = type(gen_batch).Response

    finish_reason: Optional[str] = None
    match_sequence = None

    gen_batch.tokens[0].append(token_id)
    gen_batch._num_tokens[0] += 1
    if gen_batch._num_tokens[0] >= gen_batch.max_tokens[0]:
        finish_reason = "length"

    new_state, match_sequence, current_state = gen_batch.state_machines[0].match(
        gen_batch._matcher_states[0], token_id
    )
    gen_batch._matcher_states[0] = new_state
    if match_sequence is not None and current_state is None:
        finish_reason = "stop"

    if finish_reason is not None:
        prompt_cache = gen_batch.extract_cache(0)
        all_tokens = gen_batch.tokens[0]
        response = Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=finish_reason,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=prompt_cache,
            all_tokens=all_tokens,
        )
        if stats is not None:
            _log_mtp_stats(gen_batch.uids[0], stats, finish_reason)
        # Drop state *before* filter([]) so the patched_filter epilogue
        # doesn't double-log when the standard finish path already logged.
        if hasattr(gen_batch, "_omlx_mtp_state"):
            try:
                delattr(gen_batch, "_omlx_mtp_state")
            except AttributeError:
                pass
        gen_batch.filter([])
        return [response]

    return [
        Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=None,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=None,
            all_tokens=None,
        )
    ]
