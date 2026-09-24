# SPDX-License-Identifier: Apache-2.0
"""Grammar-constrained decoding via xgrammar.

Provides a logits processor that enforces grammar constraints by masking
invalid tokens at sampling time.  Follows the same ``__call__(tokens, logits)``
interface used by :class:`ThinkingBudgetProcessor`.

Phase-awareness (thinking vs. output) is handled by the *grammar itself*
via xgrammar's structural tag API, not by this processor.  For thinking
models the grammar is compiled as a ``sequence`` of
``[tag(<think>, any_text, </think>), constrained_schema]`` so that the
bitmask is permissive during reasoning and constrained during output.
This keeps the processor simple and enables uniform batched bitmask
computation (parallel model forward || bitmask fill).

The processor has two ways of learning which token was sampled from its
bitmask:

1. **Deferred** (default, the BatchGenerator path): ``__call__`` only fills
   and applies the bitmask and marks the row :attr:`pending`; the scheduler's
   monkey-patched ``GenerationBatch._step`` calls :meth:`accept_token` with
   the sampled id at the top of the following step, where the id has to be
   evaluated anyway.  Advancing the matcher from ``tokens`` here would force a
   host sync per step, which is what constrained decoding's throughput was
   lost to before the accept moved (#2561).

2. **Speculative** (the MTP verify walks, entered via
   :meth:`begin_speculative`): the caller hands ``__call__`` the concrete
   history the logits are conditioned on (the committed prefix plus the draft
   tokens before this position) and the processor advances the matcher to
   that history itself, accepting new tokens and rolling back when the
   history is shorter than the last one it saw.  :meth:`snapshot_state` /
   :meth:`restore_state` checkpoint the matcher position so a rejected draft
   suffix can be undone with ``GrammarMatcher.rollback``.  The row is never
   marked pending in this mode; :meth:`end_speculative` hands the request
   back to the deferred protocol with a consistent matcher.

A third, unused batched mode (``advance`` + ``BatchGrammarMatcher``) is kept
for compatibility.
"""

import logging
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def create_grammar_compiler(tokenizer, model, *, cache_limit_bytes=-1):
    """Create an xgrammar GrammarCompiler for the given tokenizer and model.

    Returns None if vocab_size cannot be determined.
    """
    from .._torch_stub import install as _install_torch_stub

    _install_torch_stub()
    import xgrammar as xgr

    from ..utils.tokenizer import resolve_vocab_size, unwrap_tokenizer

    hf_tokenizer = unwrap_tokenizer(tokenizer)
    vocab_size = resolve_vocab_size(model)
    kwargs = {}
    if vocab_size is not None:
        kwargs["vocab_size"] = vocab_size

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(hf_tokenizer, **kwargs)
    return xgr.GrammarCompiler(tokenizer_info, cache_limit_bytes=cache_limit_bytes)


class GrammarConstraintProcessor:
    """Logits processor that enforces grammar constraints via xgrammar bitmask.

    Args:
        compiled_grammar: An ``xgrammar.CompiledGrammar`` instance.  For
            thinking models this should already encode the thinking phase
            (compiled from a structural tag).
        vocab_size: Model vocabulary size (from model config, not tokenizer).
    """

    def __init__(self, compiled_grammar, vocab_size: int):
        from .._torch_stub import install as _install_torch_stub

        _install_torch_stub()
        import xgrammar as xgr
        from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

        # ``max_rollback_tokens`` is unlimited by default on xgrammar >= 0.2.3
        # (the argument is deprecated); passing -1 keeps older matchers able
        # to rewind a full speculative window too.
        self._matcher = xgr.GrammarMatcher(compiled_grammar, max_rollback_tokens=-1)
        self._vocab_size = vocab_size
        self._apply_mask = apply_token_bitmask_mlx

        bitmask_width = (vocab_size + 31) // 32
        self._bitmask = np.full((1, bitmask_width), -1, dtype=np.int32)
        self._terminated = False
        self._first_call = True
        self._pending = False

        # Speculative-mode bookkeeping (see begin_speculative). History
        # lengths count prompt + generated tokens, the way the callers'
        # token buffers do.
        self._speculative = False
        self._base = 0  # history length when speculative mode began
        self._seen = 0  # history length the matcher currently reflects
        self._matcher_rel = 0  # tokens accepted into the matcher since _base
        # History index past which the matcher stopped advancing: one past
        # the terminating token, or the index of a token the grammar
        # rejected (``_dead``). None while the matcher is live.
        self._stop: int | None = None
        self._dead = False

    # ------------------------------------------------------------------
    # Per-request mode (original interface)
    # ------------------------------------------------------------------

    def __call__(self, tokens, logits: mx.array) -> mx.array:
        """Fill bitmask and apply to logits.

        In deferred mode accept is handled by the monkey-patched
        ``GenerationBatch._step()``, which is the only place that can see
        which token was sampled from the result; this method only fills the
        bitmask, applies it, and records that a token is now in flight for
        this row (see :attr:`pending`).  In speculative mode the matcher is
        first advanced (or rolled back) to ``tokens``.
        """
        if self._speculative:
            self._sync_to_history(tokens)
        if self._terminated or self._dead:
            return logits

        if not self._speculative:
            self._pending = True
        self._bitmask.fill(-1)
        self._matcher.fill_next_token_bitmask(self._bitmask)

        mx_bitmask = mx.array(self._bitmask)
        return self._apply_mask(mx_bitmask, logits, self._vocab_size)

    def accept_token(self, token_id: int) -> None:
        """Accept a generated token to advance matcher state (deferred mode)."""
        self._pending = False
        if self._terminated:
            return
        if not self._matcher.accept_token(token_id):
            logger.warning("GrammarMatcher rejected token %d", token_id)
        if self._matcher.is_terminated():
            self._terminated = True

    @property
    def pending(self) -> bool:
        """True when a token was sampled for this row and not yet accepted.

        The flag lives on the processor rather than on the generation batch
        because ``GenerationBatch.extend()`` merges rows primed by a
        *different* batch instance, each with its own token already sampled;
        only per-row state survives that merge and the matching ``filter()``.
        Speculative mode never sets it: the walk advances the matcher itself.
        """
        return self._pending

    # ------------------------------------------------------------------
    # Speculative mode (MTP verify walks)
    # ------------------------------------------------------------------

    @property
    def speculative(self) -> bool:
        """True while the processor advances the matcher from its ``tokens``."""
        return self._speculative

    def begin_speculative(self, history_len: int) -> None:
        """Switch to self-advancing mode.

        ``history_len`` is the length of the history (prompt + generated) the
        matcher currently reflects, i.e. every token accepted so far and no
        token in flight.  The deferred protocol guarantees exactly that
        between two steps: the accept of ``_next_tokens`` runs at the top of
        the step that pushes it into the token buffer, so the buffer length
        is the right value and any pending token is about to be pushed and
        will be accepted by the first speculative ``__call__``.
        """
        self._speculative = True
        self._base = int(history_len)
        self._seen = self._base
        self._matcher_rel = 0
        self._dead = False
        self._stop = self._base if self._terminated else None
        self._pending = False

    def end_speculative(self, history, pending: bool) -> None:
        """Return to the deferred protocol.

        ``history`` is the committed token history (a host list or an array;
        None to skip the final sync); the matcher is moved to it, accepting
        or rolling back as needed.  ``pending`` says whether the caller has a
        sampled-but-unaccepted token in flight (``_next_tokens`` set) that the
        scheduler's deferred accept must feed to :meth:`accept_token`.
        """
        if not self._speculative:
            return
        if history is not None:
            self._sync_to_history(history)
        self._speculative = False
        self._pending = bool(pending) and not self._terminated

    def snapshot_state(self) -> dict:
        """Checkpoint the matcher position for position-keyed rewind."""
        return {
            "seen": self._seen,
            "matcher_rel": self._matcher_rel,
            "stop": self._stop,
            "terminated": self._terminated,
            "dead": self._dead,
        }

    def restore_state(self, state: dict) -> None:
        """Rewind to a checkpoint produced by :meth:`snapshot_state`.

        Only backwards moves are possible: a checkpoint is always taken
        before the positions it undoes were accepted.
        """
        target = int(state["matcher_rel"])
        if target > self._matcher_rel:
            raise RuntimeError(
                "GrammarConstraintProcessor.restore_state cannot move the matcher "
                f"forward (at {self._matcher_rel}, checkpoint {target})"
            )
        self._rollback_matcher_to(target)
        self._seen = int(state["seen"])
        self._stop = state["stop"]
        self._terminated = bool(state["terminated"])
        self._dead = bool(state["dead"])

    # -- speculative-mode internals ---------------------------------------

    def _sync_to_history(self, tokens) -> None:
        n = len(tokens)
        if n > self._seen:
            new = tokens[self._seen : n]
            ids = new.tolist() if hasattr(new, "tolist") else list(new)
            for offset, token_id in enumerate(ids):
                self._accept_speculative(int(token_id), self._seen + offset)
            self._seen = n
        elif n < self._seen:
            self._rewind_history(n)

    def _accept_speculative(self, token_id: int, index: int) -> None:
        if self._stop is not None:
            # Terminated or dead: the matcher does not move; the history
            # still grows so a later rewind knows where it stands.
            return
        if self._matcher.accept_token(token_id):
            self._matcher_rel += 1
            if self._matcher.is_terminated():
                self._terminated = True
                self._stop = index + 1
            return
        # A speculative prefix left the grammar (a draft that was not
        # masked against this state). Rows conditioned on it are never
        # committed: the target's sample at the same position is masked and
        # therefore differs from the offending draft.
        self._dead = True
        self._stop = index
        logger.debug(
            "GrammarMatcher rejected speculative token %d at history index %d",
            token_id,
            index,
        )

    def _matcher_rel_at(self, history_len: int) -> int:
        effective = history_len if self._stop is None else min(history_len, self._stop)
        return max(0, effective - self._base)

    def _rewind_history(self, history_len: int) -> None:
        self._rollback_matcher_to(self._matcher_rel_at(history_len))
        if self._stop is not None:
            if self._terminated and history_len < self._stop:
                self._terminated = False
                self._stop = None
            elif self._dead and history_len <= self._stop:
                self._dead = False
                self._stop = None
        self._seen = history_len

    def _rollback_matcher_to(self, target_rel: int) -> None:
        steps = self._matcher_rel - target_rel
        if steps > 0:
            self._matcher.rollback(steps)
            self._matcher_rel = target_rel

    # ------------------------------------------------------------------
    # Batched mode helpers
    # ------------------------------------------------------------------

    @property
    def matcher(self):
        """Return the underlying ``xgrammar.GrammarMatcher``."""
        return self._matcher

    @property
    def is_terminated(self) -> bool:
        return self._terminated

    def advance(self, tokens: mx.array) -> bool:
        """Accept the previous token and advance grammar state.

        Call this *instead of* ``__call__`` when using batched bitmask
        filling.  Returns ``True`` if the matcher is still active (not
        terminated) and should participate in the next
        ``batch_fill_next_token_bitmask`` call.
        """
        if self._terminated:
            return False

        if self._first_call:
            self._first_call = False
        elif len(tokens) > 0:
            last_token = int(tokens[-1])
            if not self._matcher.accept_token(last_token):
                logger.warning("GrammarMatcher rejected token %d", last_token)
            if self._matcher.is_terminated():
                self._terminated = True
                return False

        return True


def grammar_processors(processors) -> list[Any]:
    """Return the :class:`GrammarConstraintProcessor` instances in a row's list."""
    return [p for p in (processors or []) if isinstance(p, GrammarConstraintProcessor)]
