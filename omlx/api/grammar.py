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
        import xgrammar as xgr
        from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

        self._matcher = xgr.GrammarMatcher(compiled_grammar)
        self._vocab_size = vocab_size
        self._apply_mask = apply_token_bitmask_mlx

        bitmask_width = (vocab_size + 31) // 32
        self._bitmask = np.full((1, bitmask_width), -1, dtype=np.int32)
        self._terminated = False
        self._first_call = True

    # ------------------------------------------------------------------
    # Per-request mode (original interface)
    # ------------------------------------------------------------------

    def __call__(self, tokens, logits: mx.array) -> mx.array:
        """Fill bitmask and apply to logits.

        Accept is handled by the monkey-patched GenerationBatch._step()
        which reads _next_tokens after sampling and calls accept_token().
        This method only fills the bitmask and applies it.
        """
        if self._terminated:
            return logits

        self._bitmask.fill(-1)
        self._matcher.fill_next_token_bitmask(self._bitmask)

        mx_bitmask = mx.array(self._bitmask)
        return self._apply_mask(mx_bitmask, logits, self._vocab_size)

    def accept_token(self, token_id: int) -> None:
        """Accept a generated token to advance matcher state."""
        if self._terminated:
            return
        if not self._matcher.accept_token(token_id):
            logger.warning("GrammarMatcher rejected token %d", token_id)
        if self._matcher.is_terminated():
            self._terminated = True

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
