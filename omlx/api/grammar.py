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

The processor supports two usage modes:

1. **Per-request** (original): call ``processor(tokens, logits)`` directly.
   Handles accept + bitmask fill + mask application in one call.

2. **Batched**: call ``processor.advance(tokens)`` to accept the previous
   token, then use ``BatchGrammarMatcher.batch_fill_next_token_bitmask``
   with the exposed ``matcher`` property to fill bitmasks in parallel
   across the batch, and apply the combined bitmask externally.
"""

import logging
from typing import List, Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def create_grammar_compiler(tokenizer, model):
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
    return xgr.GrammarCompiler(tokenizer_info)


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

        self._matcher = xgr.GrammarMatcher(compiled_grammar)
        self._vocab_size = vocab_size
        self._apply_mask = apply_token_bitmask_mlx

        bitmask_width = (vocab_size + 31) // 32
        self._bitmask = np.full((1, bitmask_width), -1, dtype=np.int32)
        self._terminated = False
        self._first_call = True
        self._pending = False
        # Count of history entries (prompt + generated) this matcher has
        # already consumed.  ``None`` until the first ``__call__`` records the
        # prompt-length baseline.  Both accept paths — the top-of-step fast
        # path and the ``__call__`` history resync below — advance it in
        # lockstep so every generated token is accepted exactly once (same
        # counter pattern as ``ThinkingBudgetProcessor``).
        self._accepted_up_to: Optional[int] = None

    # ------------------------------------------------------------------
    # Per-request mode (original interface)
    # ------------------------------------------------------------------

    def __call__(self, tokens, logits: mx.array) -> mx.array:
        """Fill bitmask and apply to logits.

        Accept normally happens in the monkey-patched
        ``GenerationBatch._step()`` top-of-step fast path, which is the
        cheapest place to read the sampled ids.  That path bails out
        whenever the batch was reshaped between sampling and the next step
        (``extend``/``filter`` break the positional mapping, ``filter``
        leaves ``_next_tokens`` as ``None``) — silently dropping the
        sampled token from the matcher and desyncing it for the rest of
        the generation.  With a structural-tag grammar
        (``[tag(<think>, any_text, </think>), schema]``) that desync made
        the matcher skip the ``</think>`` transition: the whole answer
        streamed as ``reasoning_content`` while staying schema-valid
        server-side (observed 2026-08-15 with 6 concurrent json_schema
        requests — the longest-running row lost its close tag).

        The row's own ``tokens`` history is ground truth, so before
        filling the bitmask we feed the matcher exactly the tokens it has
        not seen yet.  The counter keeps this idempotent with the fast
        path; when nothing was missed the loop body never runs.
        """
        if self._terminated:
            return logits

        n = len(tokens)
        if getattr(self, "_accepted_up_to", None) is None:
            self._accepted_up_to = n  # first call: history is just the prompt
        elif n > self._accepted_up_to:
            for i in range(self._accepted_up_to, n):
                self._advance_matcher(int(tokens[i]))
            self._accepted_up_to = n
            self._pending = False
            if self._terminated:
                return logits

        self._pending = True
        self._bitmask.fill(-1)
        self._matcher.fill_next_token_bitmask(self._bitmask)

        mx_bitmask = mx.array(self._bitmask)
        return self._apply_mask(mx_bitmask, logits, self._vocab_size)

    def _advance_matcher(self, token_id: int) -> None:
        """Feed one token to the matcher, tracking termination."""
        if self._terminated:
            return
        if not self._matcher.accept_token(token_id):
            logger.warning("GrammarMatcher rejected token %d", token_id)
        if self._matcher.is_terminated():
            self._terminated = True

    def accept_token(self, token_id: int) -> None:
        """Accept a generated token to advance matcher state (fast path).

        Accepts unconditionally — the counter only records the position when
        a ``__call__`` has established the history baseline.  Without one
        (never happens in the real flow, but the row-advance tests build bare
        instances via ``__new__``), a later ``__call__`` initialises its
        baseline from the full history, which already includes this token,
        so nothing is double-accepted either way.
        """
        self._pending = False
        if self._terminated:
            return
        self._advance_matcher(token_id)
        if getattr(self, "_accepted_up_to", None) is not None:
            self._accepted_up_to += 1

    @property
    def pending(self) -> bool:
        """True when a token was sampled for this row and not yet accepted.

        The flag lives on the processor rather than on the generation batch
        because ``GenerationBatch.extend()`` merges rows primed by a
        *different* batch instance, each with its own token already sampled;
        only per-row state survives that merge and the matching ``filter()``.
        """
        return self._pending

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
