# SPDX-License-Identifier: Apache-2.0
"""MTP-head prompt priming: fold the prompt into the head cache during prefill.

Without priming the MTP head starts generation with an empty KV cache — its
first drafts see none of the prompt and acceptance starts context-starved,
recovering only as committed generation tokens accumulate (MTPLX measured
committed-history priming at 0.90 acceptance vs 0.26 unprimed on depth-1
real-code prompts). This module rides the existing prefill forwards: every
backbone chunk forward already computes the trunk-normed hidden for all chunk
positions, so the (hidden[t], token[t+1]) pairs the head history needs are
available for free. Each chunk is folded into a head cache immediately and
the chunk hidden is discarded — only a single (1, 1, H) pending row carries
across chunks.

Transport: the priming context is attached as an attribute on the first
prompt-cache entry that exposes an integer ``offset`` (the first
full-attention ``KVCache``). The scheduler's singleton merge passthrough
preserves cache-entry identity from external prefill through
``GenerationBatch.prompt_cache``, so ``_post_init_mtp`` can pop the primed
cache at MTP activation with no registry and no lifetime bookkeeping — the
context dies with the request's cache.

Fail-safe invariant: every capture verifies the anchor offset advanced
contiguously since the previous capture (``expected_offset``). Any rewind,
trim, or unknown cache path breaks the equality and invalidates the context,
degrading to the current unprimed behaviour — never to a wrong history.

Capture sites (both call :func:`maybe_capture` after the backbone forward):

- mlx-lm text path: the patched ``qwen3_5.TextModel.__call__``
  (``qwen35_model._patch_text_model``), which computes the trunk-normed
  hidden inline.
- mlx-vlm path: a wrap on the inner ``Qwen3_5Model.__call__``
  (``qwen35_vlm_runtime``), whose return value *is* the trunk-normed hidden.
  The outer ``LanguageModel`` is reached via a weakref stamped at init.

Both sites skip ``return_hidden=True`` forwards (MTP verify cycles and the
activation forward in ``_post_init_mtp``); the final (hidden[prompt[-1]],
main_tok) pair is folded by :func:`take_primed` at activation instead.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# The MTP head is fed the trunk's *post-norm* hidden and chains on its own
# post-norm output. Measured on Qwen3.6-27B this accepts a few points higher
# than PR 990's pre-norm at every depth. Draft-side only, so output identity
# is unaffected regardless. Priming folds must use the same variant as the
# decode-time history folds in batch_generator, hence the single definition
# here.
HEAD_HIDDEN_POST_NORM = True

_CTX_ATTR = "_omlx_mtp_prime_ctx"

_SUPPRESS = threading.local()


def priming_enabled() -> bool:
    """Prompt priming is on by default for MTP-enabled models."""
    return os.environ.get("OMLX_MTP_PROMPT_PRIMING", "1").strip().lower() not in (
        "0",
        "false",
        "off",
    )


def prime_window() -> int:
    """Max prompt length (tokens) to prime; 0 = unlimited.

    Escape hatch for the head-cache memory cost on very long prompts (one
    full-attention layer of KV over the prompt). Prompts longer than the
    window run unprimed — a coarse cap, not a suffix window, because the
    prompt length is unknown while chunks stream through the model.
    """
    try:
        return max(0, int(os.environ.get("OMLX_MTP_PRIME_WINDOW", "0")))
    except ValueError:
        return 0


@contextmanager
def suppress_capture():
    """Disable capture on this thread for the duration of the block."""
    _SUPPRESS.value = True
    try:
        yield
    finally:
        _SUPPRESS.value = False


def _suppressed() -> bool:
    return bool(getattr(_SUPPRESS, "value", False))


@dataclass
class _PrimeCtx:
    """Streaming priming state riding a request's prompt cache."""

    mtp_cache: List[Any] = field(default_factory=list)
    # Trunk-normed hidden of the newest seen token, (1, 1, H) — pairs with
    # the first token of the next chunk (or with main_tok at activation).
    pending_hidden: Optional[Any] = None
    # Folded (hidden, next_token) pairs == head-cache offset.
    folded: int = 0
    # Anchor cache offset observed after the last captured forward. The next
    # capture requires offset_now - S == expected_offset (contiguity).
    expected_offset: int = 0
    valid: bool = True

    def __deepcopy__(self, memo):
        # PromptProcessingBatch._copy() deep-copies caches for partial prompt
        # splits; a duplicated priming timeline cannot stay contiguous with
        # either copy, so the clone starts over (invalid, no tensors).
        clone = _PrimeCtx()
        clone.valid = False
        memo[id(self)] = clone
        return clone


def _anchor(cache: Optional[List[Any]]) -> Optional[Any]:
    """First cache entry with a plain-int offset (first full-attention KVCache)."""
    if not cache:
        return None
    for c in cache:
        if type(getattr(c, "offset", None)) is int:
            return c
    return None


def _activation_offset(cache: Optional[List[Any]]) -> Optional[int]:
    """Attention-layer offset at MTP activation, tolerant of batch caches.

    Between the last capture and activation, ``insert()`` runs mlx-lm's
    cache merge: scalar ``KVCache`` entries without singleton passthrough
    become batch caches whose ``offset`` is a 1-element ``mx.array``. The
    one-off ``int()`` sync here is activation-time only, never per-chunk.
    """
    if not cache:
        return None
    for c in cache:
        offset = getattr(c, "offset", None)
        if type(offset) is int:
            return offset
        if offset is not None and getattr(offset, "size", 0) == 1:
            try:
                return int(offset.reshape(()).item())
            except Exception:
                continue
    return None


def _attach_target(cache: List[Any]) -> Any:
    """Cache entry the context rides on.

    Must survive the request's whole prefill→insert lifetime. The int-offset
    KVCache entries do NOT qualify: TurboQuant conversion replaces them
    (``prompt_cache[i] = TurboQuantKVCache.from_cache(...)``) at the end of
    prefill, which would silently discard the context. The recurrent-layer
    entries (ArraysCache family, no int offset) are never replaced, so prefer
    the first of those; an all-attention model falls back to the first entry
    and degrades to unprimed if that entry gets swapped.
    """
    for c in cache:
        if type(getattr(c, "offset", None)) is not int:
            return c
    return cache[0]


def _store_ctx(cache: List[Any], ctx: _PrimeCtx) -> None:
    target = _attach_target(cache)
    for c in cache:
        if c is not target and getattr(c, _CTX_ATTR, None) is not None:
            try:
                delattr(c, _CTX_ATTR)
            except AttributeError:
                pass
    setattr(target, _CTX_ATTR, ctx)


def _find_ctx(cache: Optional[List[Any]]) -> Optional[_PrimeCtx]:
    if not cache:
        return None
    for c in cache:
        ctx = getattr(c, _CTX_ATTR, None)
        if ctx is not None:
            return ctx
    return None


def drop_ctx(cache: Optional[List[Any]]) -> None:
    """Remove any priming context from a request's prompt cache."""
    if not cache:
        return
    for c in cache:
        if getattr(c, _CTX_ATTR, None) is not None:
            try:
                delattr(c, _CTX_ATTR)
            except AttributeError:
                pass


def _host_eligible(host: Any) -> bool:
    return bool(
        getattr(host, "_omlx_mtp_decode_enabled", False)
        and getattr(host, "_omlx_mtp_chain", False)
        and getattr(host, "mtp", None) is not None
    )


def maybe_capture(
    host: Any, inputs: Any, normed: Any, cache: Optional[List[Any]]
) -> None:
    """Fold this forward's (hidden, next_token) pairs into the priming cache.

    ``host`` is the patched language model (mlx-lm ``TextModel`` or mlx-vlm
    ``LanguageModel``) exposing ``mtp`` / ``model.embed_tokens`` /
    ``make_mtp_cache``. ``normed`` is the trunk-normed hidden for all
    positions of ``inputs`` (1, S, H). Host-side bookkeeping only — the head
    forward is dispatched lazily and no GPU sync happens here.

    Call sites guard the cheap negatives (return_hidden / n_confirmed /
    inputs_embeds / batch>1) before calling; everything here re-checks what
    is load-bearing and bails silently, so a miss degrades to unprimed.
    """
    if _suppressed() or not priming_enabled():
        return
    if cache is None or not _host_eligible(host):
        return
    if inputs is None or getattr(inputs, "ndim", 0) != 2 or inputs.shape[0] != 1:
        return
    anchor = _anchor(cache)
    if anchor is None:
        return

    import mlx.core as mx

    seq_len = int(inputs.shape[1])
    offset_after = anchor.offset  # forward already ran; offset includes S

    ctx = _find_ctx(cache)
    if ctx is not None and (
        not ctx.valid or ctx.expected_offset != offset_after - seq_len
    ):
        # Rewind / trim / unknown path between captures: never guess.
        drop_ctx(cache)
        ctx = None
    window = prime_window()
    if window and offset_after > window:
        if ctx is not None:
            drop_ctx(cache)
        return
    if ctx is None:
        if seq_len <= 1:
            # A lone decode step cannot start a prompt timeline.
            return
        ctx = _PrimeCtx(mtp_cache=host.make_mtp_cache())
        if not ctx.mtp_cache:
            return
        _store_ctx(cache, ctx)

    if ctx.pending_hidden is not None:
        if seq_len > 1:
            pairs_hidden = mx.concatenate(
                [ctx.pending_hidden, normed[:, :-1]], axis=1
            )
        else:
            pairs_hidden = ctx.pending_hidden
        pairs_tokens = inputs
    else:
        if seq_len <= 1:
            ctx.pending_hidden = normed[:, -1:]
            ctx.expected_offset = offset_after
            return
        pairs_hidden = normed[:, :-1]
        pairs_tokens = inputs[:, 1:]

    host.mtp(pairs_hidden, pairs_tokens, host.model.embed_tokens, ctx.mtp_cache)
    ctx.folded += int(pairs_tokens.shape[1])
    ctx.pending_hidden = normed[:, -1:]
    ctx.expected_offset = offset_after
    # Materialize the head-cache buffers per chunk so the fold graph never
    # accumulates across a long prefill; the (1,1,H) pending row is evaluated
    # alongside so the chunk's full hidden can be freed.
    evals = [ctx.pending_hidden]
    for c in ctx.mtp_cache:
        keys = getattr(c, "keys", None)
        values = getattr(c, "values", None)
        if keys is not None:
            evals.append(keys)
        if values is not None:
            evals.append(values)
    mx.async_eval(evals)


def take_primed(
    model: Any,
    cache: Optional[List[Any]],
    main_tok: Any,
) -> Optional[tuple]:
    """Pop the priming context at MTP activation and finish the seam.

    Called from ``_post_init_mtp`` after its 1-token backbone forward at
    ``main_tok`` (which capture skipped — it runs with return_hidden=True).
    Validates that the context is contiguous up to exactly that forward,
    folds the final (hidden[prompt[-1]], main_tok) pair through the public
    ``mtp_forward`` (adapter/outer-model level), and returns
    ``(mtp_cache, hist_offset)`` — or None, in which case the caller keeps
    the current unprimed behaviour.
    """
    ctx = _find_ctx(cache)
    if ctx is None:
        return None
    drop_ctx(cache)
    if not (ctx.valid and ctx.folded > 0 and ctx.pending_hidden is not None):
        return None
    offset = _activation_offset(cache)
    if offset is None or ctx.expected_offset != offset - 1:
        logger.debug(
            "MTP priming discarded: seam offset mismatch (ctx=%s cache=%s)",
            ctx.expected_offset,
            offset,
        )
        return None
    try:
        model.mtp_forward(
            ctx.pending_hidden,
            main_tok.reshape(1, 1),
            ctx.mtp_cache,
            logits_keep=1,
        )
    except Exception as exc:
        logger.debug("MTP priming discarded: seam fold failed: %s", exc)
        return None
    return ctx.mtp_cache, ctx.folded + 1


def prime_ctx_stats(cache: Optional[List[Any]]) -> Optional[int]:
    """Folded pair count of a live context (introspection / tests)."""
    ctx = _find_ctx(cache)
    return ctx.folded if ctx is not None else None


__all__ = [
    "HEAD_HIDDEN_POST_NORM",
    "priming_enabled",
    "prime_window",
    "suppress_capture",
    "maybe_capture",
    "take_primed",
    "drop_ctx",
    "prime_ctx_stats",
]
