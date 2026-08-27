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

Transport: the context lives in a single slot on the patched language-model
instance (the ``host``). Cache-entry attributes cannot carry it — mlx-lm's
insert merge rebuilds every layer cache that lacks filter/extract support
(all of DeepSeek-V4's and GLM-5.2's CacheList entries, and TurboQuant
replaces KVCache entries at end of prefill) — while the model instance is
the one object every forward and the activation both see. The engine thread
serializes forwards, and the offset-contiguity invariant below makes the
single slot safe across interleaved requests: a chunk from a different
request can never look contiguous with another request's timeline (its
first forward starts at offset 0), so it invalidates or restarts the slot,
and the activating request is always the slot's last writer.

Fail-safe invariant: every capture verifies the anchor offset advanced
contiguously since the previous capture (``expected_offset``). Any rewind,
trim, request switch, or unknown cache path breaks the equality and
invalidates the context, degrading to the current unprimed behaviour —
never to a wrong history. A batched (B>1) forward advances the anchor
without capture seeing its tokens, so it drops the context outright rather
than let a later singleton chunk read as contiguous across it.

Capture sites (each calls :func:`maybe_capture` after the backbone forward):

- mlx-lm qwen3_5 text path: the patched ``TextModel.__call__``
  (``qwen35_model``), which computes the trunk-normed hidden inline.
- mlx-vlm qwen3_5 path: a wrap on the inner ``Qwen3_5Model.__call__``
  (``qwen35_vlm_runtime``), whose return value *is* the trunk-normed
  hidden; the MoE inner model inherits it. The outer ``LanguageModel`` is
  reached via a weakref stamped at init.
- DeepSeek-V4 (``deepseek_v4_model``): the patched ``Model.__call__``
  requests ``return_raw_hidden`` and passes the raw 4D Hyper-stream hidden
  (the head input variant; no trunk norm).
- GLM-5.2 (``glm_moe_dsa_model``): the patched ``Model.__call__`` passes
  the post-final-norm hidden it already computes.

All sites skip ``return_hidden=True`` forwards (MTP verify cycles and the
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
    """Max tokens to fold into one prime context; 0 = unlimited.

    Escape hatch for the head-cache memory cost of priming (one
    full-attention layer of KV over the folded span). The cap is measured
    against the span actually folded this request — with a warm prefix cache
    that is only the boundary remainder, not the full prompt — so a
    long-context request with a small remainder still primes. A remainder
    larger than the window runs unprimed.
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
    """Streaming priming state in the host model's single slot."""

    mtp_cache: List[Any] = field(default_factory=list)
    # Head-input hidden of the newest seen token, (1, 1, ..., H) — pairs
    # with the first token of the next chunk (or main_tok at activation).
    pending_hidden: Optional[Any] = None
    # Folded (hidden, next_token) pairs == head-cache offset.
    folded: int = 0
    # Anchor cache offset observed after the last captured forward. The next
    # capture requires offset_now - S == expected_offset (contiguity).
    expected_offset: int = 0
    valid: bool = True
    # The current contiguous timeline exceeded OMLX_MTP_PRIME_WINDOW. Keep a
    # lightweight marker so later small chunks cannot restart priming.
    window_exceeded: bool = False


def _read_offset(entry: Any) -> Optional[int]:
    """``entry.offset`` as a plain int, unwrapping size-1 array offsets.

    Batch caches (``BatchKVCache`` / ``BatchRotatingKVCache``) hold their
    offset as a 1-element ``mx.array``. Reading it costs one sync, so
    callers do it once per forward at most.
    """
    offset = getattr(entry, "offset", None)
    if type(offset) is int:
        return offset
    if offset is not None and getattr(offset, "size", 0) == 1:
        try:
            return int(offset.reshape(()).item())
        except Exception:
            return None
    return None


def _offset_readable(entry: Any) -> bool:
    """Whether :func:`_read_offset` can serve this entry — no sync."""
    offset = getattr(entry, "offset", None)
    return type(offset) is int or (
        offset is not None and getattr(offset, "size", 0) == 1
    )


class _IntOffsetAnchor:
    """Anchor view exposing a scalar-or-size-1-array offset as an int.

    Under ``BatchGenerator`` every request's caches are merged into
    ``Batch*`` entries at ``PromptProcessingBatch.__init__``, whose
    ``offset`` is a 1-element ``mx.array`` **even for a single request**
    (B==1). The plain-int probe this replaces therefore found no anchor on
    any batch-engine prefill, so ``maybe_capture`` bailed silently and
    priming never activated there (#3079).

    The unwrap is unambiguous because :func:`maybe_capture` only captures
    ``(1, S)`` forwards — a singleton timeline. It does cost one ``int()``
    sync per captured forward, which is what the contiguity invariant is
    built on; ``BatchRotatingKVCache._offset`` would be sync-free but
    counts buffer slots rather than tokens.
    """

    __slots__ = ("_cache",)

    def __init__(self, cache: Any) -> None:
        self._cache = cache

    @property
    def offset(self) -> Optional[int]:
        return _read_offset(self._cache)


def _anchor(cache: Optional[List[Any]]) -> Optional[Any]:
    """First cache entry whose offset can be read as an int, as a view.

    Container layers (``CacheList``-style, exposing ``.caches`` — DeepSeek-V4
    and GLM-5.2 backbones) are searched one level deep: the container itself
    has no offset but its first sub-cache (RotatingKVCache / KVCache) does.
    """
    if not cache:
        return None
    for c in cache:
        if _offset_readable(c):
            return _IntOffsetAnchor(c)
        for sub in getattr(c, "caches", ()) or ():
            if _offset_readable(sub):
                return _IntOffsetAnchor(sub)
    return None


def _activation_offset(cache: Optional[List[Any]]) -> Optional[int]:
    """Attention-layer offset at MTP activation, tolerant of batch caches.

    Between the last capture and activation, ``insert()`` runs mlx-lm's
    cache merge: scalar ``KVCache`` entries without singleton passthrough
    become batch caches whose ``offset`` is a 1-element ``mx.array``.
    """
    if not cache:
        return None
    for c in cache:
        got = _read_offset(c)
        if got is not None:
            return got
        for sub in getattr(c, "caches", ()) or ():
            got = _read_offset(sub)
            if got is not None:
                return got
    return None


def _host_candidates(model: Any):
    """The model itself plus the wrapped language model, if any.

    Mirrors ``batch_generator._resolve_mtp_chain_depth``: the host that
    carries the slot is the patched language-model instance — the outer
    adapter / VLM wrapper for qwen paths, the Model itself for DeepSeek/GLM.
    """
    yield model
    for attr in ("language_model", "_language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            yield inner


def _find_ctx(model: Any) -> Optional[_PrimeCtx]:
    for host in _host_candidates(model):
        ctx = getattr(host, _CTX_ATTR, None)
        if ctx is not None:
            return ctx
    return None


def drop_ctx(model: Any) -> None:
    """Remove any priming context from the model's host slot."""
    if model is None:
        return
    for host in _host_candidates(model):
        if getattr(host, _CTX_ATTR, None) is not None:
            try:
                delattr(host, _CTX_ATTR)
            except AttributeError:
                pass


def _host_eligible(host: Any) -> bool:
    get_mtp = getattr(host, "get_mtp_module", None)
    mtp = get_mtp() if callable(get_mtp) else getattr(host, "mtp", None)
    return bool(
        getattr(host, "_omlx_mtp_decode_enabled", False)
        and getattr(host, "_omlx_mtp_chain", False)
        and mtp is not None
    )


def capture_eligible(host: Any, cache: Optional[List[Any]]) -> bool:
    """Cheap pre-check for capture sites that must decide the forward shape.

    The DeepSeek/GLM backbones only expose the head-input hidden when asked
    (``return_raw_hidden``), so their patched ``__call__`` consults this
    before choosing the call form. Everything here is re-checked inside
    :func:`maybe_capture`; this exists purely to keep the ineligible path
    identical to stock.
    """
    return (
        not _suppressed()
        and priming_enabled()
        and cache is not None
        and _host_eligible(host)
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
    inputs_embeds) before calling; everything here re-checks what is
    load-bearing and bails silently, so a miss degrades to unprimed.
    """
    if _suppressed() or not priming_enabled():
        return
    if cache is None or not _host_eligible(host):
        return
    if inputs is None or getattr(inputs, "ndim", 0) != 2:
        return
    if inputs.shape[0] != 1:
        # A B>1 forward advances the anchor invisibly to capture, so a later
        # singleton chunk could look contiguous with a timeline it never
        # belonged to (chunk boundaries are aligned across requests). Drop
        # the slot rather than risk a wrong history.
        drop_ctx(host)
        return
    anchor = _anchor(cache)
    if anchor is None:
        return

    import mlx.core as mx

    seq_len = int(inputs.shape[1])
    offset_after = anchor.offset  # forward already ran; offset includes S
    if offset_after is None:
        return

    ctx = getattr(host, _CTX_ATTR, None)
    if ctx is not None and (
        not ctx.valid or ctx.expected_offset != offset_after - seq_len
    ):
        # Rewind / trim / request switch / unknown path: never guess.
        drop_ctx(host)
        ctx = None
    if ctx is not None and ctx.window_exceeded:
        ctx.expected_offset = offset_after
        return
    window = prime_window()
    if window:
        # Cap by the primed span (the head-KV the window exists to bound),
        # not the absolute prompt offset: on a warm prefix cache only the
        # boundary remainder is ever folded, so a long-context request with a
        # small remainder is exactly the cheap case priming is for (#2909).
        folded = ctx.folded if ctx is not None else 0
        if folded + seq_len > window:
            setattr(
                host,
                _CTX_ATTR,
                _PrimeCtx(
                    expected_offset=offset_after,
                    window_exceeded=True,
                ),
            )
            return
    if ctx is None:
        if seq_len <= 1:
            # A lone decode step cannot start a prompt timeline.
            return
        ctx = _PrimeCtx(mtp_cache=host.make_mtp_cache())
        if not ctx.mtp_cache:
            return
        setattr(host, _CTX_ATTR, ctx)

    if ctx.pending_hidden is not None:
        if seq_len > 1:
            pairs_hidden = mx.concatenate([ctx.pending_hidden, normed[:, :-1]], axis=1)
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

    # Fold through the public mtp_forward so every family's head layout
    # (module, block list, CacheList head caches) is handled by its own
    # model patch. The returned logits are never evaluated — nothing pulls
    # on them, so the lm_head tail costs nothing.
    host.mtp_forward(pairs_hidden, pairs_tokens, ctx.mtp_cache, logits_keep=1)
    ctx.folded += int(pairs_tokens.shape[1])
    ctx.pending_hidden = normed[:, -1:]
    ctx.expected_offset = offset_after
    # Materialize the head-cache buffers per chunk so the fold graph never
    # accumulates across a long prefill; the (1,1,H) pending row is evaluated
    # alongside so the chunk's full hidden can be freed.
    evals = [ctx.pending_hidden]
    flat = []
    for c in ctx.mtp_cache:
        subs = getattr(c, "caches", None)
        flat.extend(subs if subs else (c,))
    for c in flat:
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
    # Hosts with their own priming shape (inkling's sliding-window
    # multi-block fold) own the whole activation seam.
    for host in _host_candidates(model):
        hook = getattr(host, "mtp_take_primed", None)
        if callable(hook):
            primed = hook(cache, main_tok)
            if primed is not None:
                return primed
            # None means the hook declined ownership, not "no priming": the
            # DeepSeek-V4 patch registers ``mtp_take_primed`` on the class
            # but only DSpark builds answer it, so legacy single-head MTP
            # models could never reach the generic seam below and priming
            # was structurally dead for them (#3079). Every hook pops its
            # own context before declining, so the fallthrough cannot adopt
            # a foreign timeline.
            break
    ctx = _find_ctx(model)
    if not isinstance(ctx, _PrimeCtx):
        # No context, or a host-owned one sharing the slot (inkling's) whose
        # hook declined without popping it — not ours to consume.
        return None
    drop_ctx(model)
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


def prime_ctx_stats(model: Any) -> Optional[int]:
    """Folded pair count of a live context (introspection / tests)."""
    ctx = _find_ctx(model)
    return ctx.folded if ctx is not None and not ctx.window_exceeded else None


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
