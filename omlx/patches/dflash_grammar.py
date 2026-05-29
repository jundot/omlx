# SPDX-License-Identifier: Apache-2.0
"""Sampling- and grammar-aware decoding for the DFlash speculative path.

Upstream ``dflash_mlx`` (through v0.1.7) is greedy-only and grammar-blind: the
verify loop in ``engine/spec_epoch.py`` builds its ``posterior`` with a single
``greedy_tokens_with_mask`` argmax over the target logits and accepts a drafted
token iff it equals that argmax. There is no temperature sampling and no
xgrammar / structured-output hook. So a request routed to the DFlash path
either (a) silently drops its xgrammar ``compiled_grammar`` -- emitting
schema-violating JSON that never stops -- or (b) decodes greedily regardless of
``temperature``, which makes free-text fields collapse into repetition loops.

This module wires BOTH grammar masking and rejection sampling into the verify
loop without vendoring the 600-line ``SpeculativeSession``. It monkey-patches
the two small symbols the verify loop binds at module scope:

- ``spec_epoch.greedy_tokens_with_mask`` -- produces the per-position
  ``posterior`` from the target logits. We replace argmax with either a
  grammar-masked walk, temperature sampling, or both.

- ``spec_epoch._match_acceptance_length`` -- counts how many drafted tokens
  matched the posterior. Untouched for the sampling-only path (the stock
  equality count already yields the correct stochastic acceptance length); for
  grammar it additionally rolls the matcher back from its speculative
  over-advance to the committed prefix.

Why sampling works with the stock equality-based acceptance loop
----------------------------------------------------------------
DFlash drafts greedily, so its proposal ``q`` is a point mass on the drafted
token ``d``. Standard speculative sampling accepts ``d`` with probability
``min(1, p(d)/q(d)) = p(d)`` and otherwise emits a sample from the residual
``norm(max(0, p - q))``, which for a point-mass ``q`` is just ``p`` with ``d``
removed and renormalized. That is exactly equivalent to: draw an independent
sample ``s ~ p`` at each position, accept ``d`` iff ``s == d``, and on the first
mismatch emit ``s`` (whose conditional distribution given ``s != d`` IS the
residual). So if we make ``posterior[j]`` a per-position sample from the target
distribution instead of the argmax, the loop's existing
``posterior[acceptance_len]`` bonus and its ``drafted == posterior`` equality
count produce a distribution-exact target sample -- no need to thread the
drafted tokens into this hook, and no explicit residual arithmetic. This
identity holds ONLY because the draft is deterministic (point-mass ``q``); see
vLLM PR #14702 / the speculative-sampling correctness proof.

``temperature == 0`` builds no session at all and falls through to the stock
greedy argmax, so deterministic requests are byte-for-byte unchanged. The
session is created per request and held in a thread-local that the dflash
decode thread sets around iteration.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

_local = threading.local()
_installed = False
_orig_greedy = None
_orig_match = None


def _to_logprobs(logits):
    """Normalize raw (possibly grammar-masked, -inf-bearing) logits to
    log-probabilities before handing them to ``make_sampler``.

    ``make_sampler``'s filters expect log-PROBABILITIES, not raw logits:
    ``apply_top_p`` does ``probs = exp(input)`` and compares a cumulative mass
    against ``1 - top_p``. Feeding raw logits breaks that threshold -- and under
    a grammar mask it breaks catastrophically: when every grammar-allowed token
    has a low absolute logit (the model wanted a now-masked token), their
    unnormalized ``exp`` never reaches the cutoff, so top_p filters ALL of them
    to ``-inf``, leaving an all-``-inf`` row whose categorical sample collapses
    to token 0 (``<pad>``). ``log_softmax`` makes the mass sum to 1 so the
    threshold is meaningful again; ``-inf`` entries stay ``-inf`` (prob 0)."""
    import mlx.core as mx

    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


class _DecodeSession:
    """Per-request decode state for one DFlash run.

    Holds an optional xgrammar matcher and an optional sampler callable. A
    session exists only when grammar is active OR ``temperature > 0`` -- the
    plain greedy, no-grammar case never builds one and the patched hooks defer
    to the stock dflash symbols.
    """

    def __init__(self, compiled_grammar: Any, sampler: Callable[[Any], Any] | None):
        self.matcher = None
        self._apply_mask = None
        self._vocab_size: int | None = None
        self._width: int | None = None
        # callable(logits)->token_ids, or None to keep greedy argmax.
        self.sampler = sampler
        # None until prefill produces the first token; afterwards holds the
        # committed staged_first to seed the next decode block (grammar path).
        self.pending_staged_first: int | None = None
        self.last_posterior: list[int] = []
        self.last_accepts: int = 0
        self.terminated: bool = False

        if compiled_grammar is not None:
            import xgrammar as xgr
            from xgrammar.kernels.apply_token_bitmask_mlx import (
                apply_token_bitmask_mlx,
            )

            # max_rollback_tokens=-1 -> unlimited rollback (verified empirically).
            self.matcher = xgr.GrammarMatcher(compiled_grammar)
            self._apply_mask = apply_token_bitmask_mlx

    @property
    def sampling(self) -> bool:
        return self.sampler is not None

    def _ensure_vocab(self, vocab_size: int) -> None:
        if self._vocab_size is None:
            self._vocab_size = int(vocab_size)
            self._width = (self._vocab_size + 31) // 32

    def _sample_block(self, logits, suppress_token_mask=None):
        """Vectorized sampling-only path (no grammar): sample one token per row
        of a ``[n, vocab]`` (or ``[1, vocab]`` prefill) logits block after
        applying the static suppress mask. Output shape/dtype matches the stock
        ``greedy_tokens_with_mask``."""
        import mlx.core as mx

        if suppress_token_mask is not None:
            floor = mx.array(-1e9, dtype=logits.dtype)
            logits = mx.where(suppress_token_mask, floor, logits)
        return self.sampler(_to_logprobs(logits)).astype(mx.uint32)

    def _masked_pick(self, logits_row) -> int:
        """Apply the current matcher bitmask to a ``[1, vocab]`` logits row and
        return the chosen token id (sampled if a sampler is set, else greedy)."""
        import mlx.core as mx
        import numpy as np

        bitmask = np.full((1, self._width), -1, dtype=np.int32)
        self.matcher.fill_next_token_bitmask(bitmask, index=0)
        masked = self._apply_mask(mx.array(bitmask), logits_row, self._vocab_size)
        if self.sampler is not None:
            return int(self.sampler(_to_logprobs(masked)).astype(mx.uint32).item())
        return int(mx.argmax(masked, axis=-1).astype(mx.uint32).item())

    def _accept(self, token_id: int) -> bool:
        """Advance the matcher; return True if the token was consumed.

        Sets ``terminated`` when the matcher rejects the token (state diverged)
        or when accepting it completes the grammar (stop token consumed). The
        stop-token case must flip the flag so the decode walk stops calling
        ``fill_next_token_bitmask`` on an already-terminated matcher (xgrammar
        raises if asked for a mask after the stop token is accepted).
        """
        if self.terminated:
            return False
        if not self.matcher.accept_token(int(token_id)):
            # A masked token should always be accepted; a rejection means the
            # matcher state diverged. Stop constraining rather than corrupt
            # further positions.
            logger.warning("dflash grammar: matcher rejected token %d", token_id)
            self.terminated = True
            return False
        if self.matcher.is_terminated():
            self.terminated = True
        return True


def get_session() -> _DecodeSession | None:
    return getattr(_local, "session", None)


def set_session(
    compiled_grammar: Any = None,
    *,
    temperature: float = 0.0,
    top_p: float = 0.0,
    top_k: int = 0,
    min_p: float = 0.0,
) -> None:
    """Activate grammar and/or sampling for the current (dflash decode) thread.

    A session is built only when a grammar is present OR ``temperature > 0``;
    otherwise decoding stays on the stock greedy argmax path. Never raises: a
    bad grammar / sampler degrades to the pre-patch behavior rather than
    killing the request.
    """
    want_sampling = bool(temperature and float(temperature) > 0.0)
    if compiled_grammar is None and not want_sampling:
        clear_session()
        return
    try:
        sampler = None
        if want_sampling:
            from ..utils.sampling import make_sampler

            sampler = make_sampler(
                temp=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                min_p=float(min_p),
            )
        _local.session = _DecodeSession(compiled_grammar, sampler)
    except Exception as exc:
        logger.warning(
            "dflash decode: session init failed, decoding greedy/unconstrained: %s",
            exc,
        )
        _local.session = None


def clear_session() -> None:
    _local.session = None


def _patched_greedy(logits, suppress_token_mask=None):
    sess = get_session()
    if sess is None or sess.terminated:
        return _orig_greedy(logits, suppress_token_mask)

    # Sampling-only fast path (no grammar): one independent sample per position,
    # vectorized over the whole verify block. The stock equality-based
    # ``_match_acceptance_length`` then turns this into the correct stochastic
    # acceptance length and residual bonus (see module docstring).
    if sess.matcher is None:
        return sess._sample_block(logits, suppress_token_mask)

    import mlx.core as mx

    sess._ensure_vocab(int(logits.shape[-1]))

    # Prefill: a single [1, vocab] row, fresh matcher state. Produce the first
    # generated token under the grammar but do NOT advance the matcher -- it is
    # accepted as the seed of the first decode block.
    if sess.pending_staged_first is None:
        tok = sess._masked_pick(logits[-1:])
        sess.pending_staged_first = tok
        return mx.array([tok], dtype=mx.uint32)

    # Decode: walk the grammar-masked posterior over the verify block, sampling
    # (or argmax) per position and advancing the matcher as we go.
    n = int(logits.shape[0])
    # Seed the matcher with the committed staged_first (= previous bonus).
    accepts = 1 if sess._accept(int(sess.pending_staged_first)) else 0
    posterior: list[int] = []
    for j in range(n):
        if sess.terminated:
            # Grammar completed mid-block: pad so downstream shapes hold; these
            # positions are past the stop and get rolled back / rejected.
            posterior.append(posterior[-1] if posterior else int(sess.pending_staged_first))
            continue
        tok = sess._masked_pick(logits[j : j + 1])
        posterior.append(tok)
        if sess._accept(tok):
            accepts += 1
    sess.last_accepts = accepts
    sess.last_posterior = posterior
    return mx.array(posterior, dtype=mx.uint32)


def _patched_match(drafted_tokens, posterior_tokens):
    acceptance = _orig_match(drafted_tokens, posterior_tokens)
    sess = get_session()
    # Sampling-only / no session: the stock equality count is already correct,
    # and posterior[acceptance_len] is already the residual/bonus sample.
    if sess is None or not sess.last_posterior:
        return acceptance

    acc = int(acceptance.item())
    # The decode walk accepted ``last_accepts`` tokens (staged_first seed + each
    # masked posterior, stopping early if the grammar terminated). The loop
    # commits exactly 1 + acc (staged_first + matched drafts), so roll the
    # matcher back from its speculative over-advance to that committed prefix.
    rollback = sess.last_accepts - (1 + acc)
    rollback_ok = True
    if rollback > 0:
        try:
            sess.matcher.rollback(rollback)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("dflash grammar: rollback(%d) failed: %s", rollback, exc)
            rollback_ok = False

    # Bonus token (posterior[acc]) becomes the next block's staged_first.
    if 0 <= acc < len(sess.last_posterior):
        sess.pending_staged_first = int(sess.last_posterior[acc])
    sess.last_posterior = []

    # Re-derive termination from the rolled-back committed state: a stop token
    # accepted only speculatively (beyond the committed prefix) was just undone,
    # so the matcher may no longer be terminated.
    sess.terminated = True if not rollback_ok else sess.matcher.is_terminated()
    return acceptance


def install_dflash_grammar_patches() -> None:
    """Idempotently patch dflash_mlx's verify-loop sampling for grammar +
    temperature support."""
    global _installed, _orig_greedy, _orig_match
    if _installed:
        return
    try:
        from dflash_mlx.engine import spec_epoch
    except Exception as exc:  # pragma: no cover - dflash optional
        logger.debug("dflash decode patch skipped (no dflash_mlx): %s", exc)
        return

    _orig_greedy = spec_epoch.greedy_tokens_with_mask
    _orig_match = spec_epoch._match_acceptance_length
    spec_epoch.greedy_tokens_with_mask = _patched_greedy
    spec_epoch._match_acceptance_length = _patched_match
    _installed = True
    logger.info("dflash decode patches installed (verify-loop xgrammar mask + sampling)")
