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
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _GrammarCompletionObserver:
    """Keep a private completion observer beside its enforcing grammar.

    Use this carrier only for the narrow server-built maxLength workaround.
    Do not expose it as a request API or use its completion grammar to mask
    logits, admit tokens, or replace the primary grammar.

    Args:
        primary: The only grammar allowed to constrain generated tokens.
        completion: The maxLength-free grammar used solely to observe a root end.
        validator: A validator constructed from the exact original schema
            before scheduling.
    """

    primary: Any
    completion: Any
    validator: Any


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

        primary_grammar = compiled_grammar
        completion_grammar = None
        completion_validator = None
        if isinstance(compiled_grammar, _GrammarCompletionObserver):
            primary_grammar = compiled_grammar.primary
            completion_grammar = compiled_grammar.completion
            completion_validator = compiled_grammar.validator

        self._matcher = xgr.GrammarMatcher(primary_grammar)
        self._completion_matcher = (
            xgr.GrammarMatcher(
                completion_grammar,
                terminate_without_stop_token=True,
            )
            if completion_grammar is not None
            else None
        )
        self._completion_validator = completion_validator
        self._completion_terminated = False
        self._completion_diverged = False
        self._vocab_size = vocab_size
        self._apply_mask = apply_token_bitmask_mlx

        bitmask_width = (vocab_size + 31) // 32
        self._bitmask = np.full((1, bitmask_width), -1, dtype=np.int32)
        self._terminated = False
        self._must_end = False
        self._first_call = True
        self._pending = False

    # ------------------------------------------------------------------
    # Per-request mode (original interface)
    # ------------------------------------------------------------------

    def __call__(self, tokens, logits: mx.array) -> mx.array:
        """Fill bitmask and apply to logits.

        Accept is handled by the monkey-patched GenerationBatch._step(),
        which is the only place that can see which token was sampled from
        the result.  This method only fills the bitmask, applies it, and
        records that a token is now in flight for this row (see
        :attr:`pending`).
        """
        if self._terminated:
            return logits

        self._pending = True
        self._bitmask.fill(-1)
        self._matcher.fill_next_token_bitmask(self._bitmask)
        self._must_end = False
        if self._matcher.is_completed():
            valid_mask = self._bitmask.view(np.uint32).copy()
            trailing_bits = self._vocab_size % 32
            if trailing_bits:
                valid_mask[0, -1] &= np.uint32((1 << trailing_bits) - 1)

            stop_token_ids = np.asarray(self._matcher.stop_token_ids, dtype=np.intp)
            stop_token_ids = stop_token_ids[
                (stop_token_ids >= 0) & (stop_token_ids < self._vocab_size)
            ]
            if stop_token_ids.size:
                stop_words = stop_token_ids // 32
                stop_bits = np.left_shift(np.uint32(1), stop_token_ids % 32)
                if np.any(valid_mask[0, stop_words] & stop_bits):
                    np.bitwise_and.at(
                        valid_mask[0], stop_words, np.bitwise_not(stop_bits)
                    )
                    self._must_end = not np.any(valid_mask)

        mx_bitmask = mx.array(self._bitmask)
        return self._apply_mask(mx_bitmask, logits, self._vocab_size)

    def _accept_token(self, token_id: int) -> bool:
        """Advance primary admission before observing the accepted token.

        Use this transition from both per-request and batched decoding paths.
        Do not call the completion observer when the primary grammar rejects a
        token, and do not let observer state mutate primary termination.

        Args:
            token_id: The sampled token to admit through the primary matcher.

        Returns:
            True when the primary matcher accepted the token; otherwise False.
        """
        if not self._matcher.accept_token(token_id):
            logger.warning("GrammarMatcher rejected token %d", token_id)
            return False

        completion_matcher = getattr(self, "_completion_matcher", None)
        if completion_matcher is not None:
            if not completion_matcher.accept_token(token_id):
                logger.warning("Completion observer rejected token %d", token_id)
                self._completion_matcher = None
                self._completion_validator = None
                self._completion_diverged = True
            elif completion_matcher.is_terminated():
                self._completion_terminated = True

        if self._matcher.is_terminated():
            self._terminated = True
        return True

    def accept_token(self, token_id: int) -> None:
        """Accept a generated token to advance matcher state.

        Use after a token was selected by the primary grammar bitmask.
        Do not call this method to test whether the observer would admit a
        token; only the primary matcher owns admission.

        Args:
            token_id: The generated token id selected after primary masking.
        """
        self._pending = False
        if self._terminated:
            return
        self._accept_token(token_id)

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

    @property
    def must_end(self) -> bool:
        """Return whether the just-filled mask permits only matcher stop tokens.

        Use when turning completed grammar output into a logical scheduler stop.
        Do not use for token admission; xgrammar's bitmask remains authoritative.

        Returns:
            True when completed output has at least one allowed matcher stop
            token and no allowed non-stop token in the model vocabulary.
        """
        return getattr(self, "_must_end", False)

    @property
    def completion_terminated(self) -> bool:
        """Return whether the private observer reached a self-delimiting root.

        Use this only to request host validation at the scheduler response
        boundary. Do not use it for logits masking, token admission, or the
        primary matcher's terminal state.

        Returns:
            True after an observer has accepted a complete root object.
        """
        return getattr(self, "_completion_terminated", False)

    def completion_validates(self, instance: Any) -> bool:
        """Validate a completed observer value against the exact original schema.

        Use only after :attr:`completion_terminated` is true and immediately
        before returning a logical stop. Do not treat a validator error or a
        missing validator as a successful completion.

        Args:
            instance: JSON-decoded output to validate against the original schema.

        Returns:
            True when the exact original-schema validator accepts ``instance``.
        """
        validator = getattr(self, "_completion_validator", None)
        if (
            not self.completion_terminated
            or getattr(self, "_completion_diverged", False)
            or validator is None
        ):
            return False
        try:
            validator.validate(instance)
        except Exception as error:
            logger.warning(
                "Completion observer host validation failed (%s)",
                type(error).__name__,
            )
            return False
        return True

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
            self._accept_token(last_token)

        return not self._terminated
