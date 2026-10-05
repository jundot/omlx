# SPDX-License-Identifier: Apache-2.0
"""DRY ("Don't Repeat Yourself") repetition penalty as a logits processor.

DRY penalizes the token that would *extend* a sequence the model has already
produced, in proportion to how long the repeated sequence already is::

    penalty = multiplier * base ** (match_length - allowed_length)

Unlike repetition/presence penalties it never penalizes a token for merely
having appeared, only for continuing a verbatim run. The algorithm follows
text-generation-webui #5677 and llama.cpp's ``dry`` sampler, with two
deliberate differences:

- **Scope.** Only tokens generated in the current response are searched, and
  for thinking models only those after the last close-think token. A loop is
  the model repeating its own output; quoting the prompt (an edit tool's old
  string, a file path, a line of code) or copying a draft out of the reasoning
  trace is legitimate and must stay free. llama.cpp searches the whole
  context. ``exclude_reasoning`` goes further and applies no penalty at all
  while the model is inside its thinking block: careful reasoning restates
  numbers, expressions and code, which DRY cannot tell from a loop.
- **Shape.** The match is a fixed ``max_match x window`` comparison done in
  lazy mlx ops, not a host-side scan. Match length is capped a few tokens
  above ``allowed_length`` — past that the penalty is already a ban — so the
  cost has no worst case and nothing forces a GPU sync.

The processor is a pure function of ``(tokens, logits)``. The speculative
decode paths (Lightning MTP, vlm_mtp) call processors on draft prefixes that
may be rejected and re-sampled; a processor that accumulated its own history
would be corrupted by them. ``snapshot_state`` / ``restore_state`` are no-ops
that declare exactly that, which is what lets the vlm_mtp path keep the
request instead of falling back to BatchGenerator.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# ``penalty_last_n = -1`` means "everything generated so far"; bound it so the
# comparison matrix stays small on very long generations.
MAX_PENALTY_LAST_N = 32768
# Matches are counted up to this many tokens past ``allowed_length``. At the
# default base that is a penalty of ~2000 logits, far beyond any real margin.
MATCH_HEADROOM = 14


_warned_unsupported: set = set()


def warn_dry_unsupported(engine: str) -> None:
    """Log once per engine that a request asked for DRY it cannot apply."""
    if engine not in _warned_unsupported:
        _warned_unsupported.add(engine)
        logger.warning(
            "DRY sampling was requested but the %s engine does not apply it; "
            "generating without it.",
            engine,
        )


def find_breaker_token_ids(tokenizer: Any, breakers: Iterable[str]) -> list[int]:
    """Token ids whose text contains any of ``breakers``.

    A breaker resets sequence matching. Tokenizers merge punctuation into
    neighbouring text (``":\\n"``, ``'":'``), so matching on substring rather
    than on the breaker's own encoding is what keeps structural boilerplate
    from chaining into long matches.
    """
    breakers = [b for b in breakers if b]
    if not breakers:
        return []
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    vocab_size = len(inner.get_vocab()) if hasattr(inner, "get_vocab") else None
    if vocab_size is None:
        vocab_size = int(inner.vocab_size)
    ids = list(range(vocab_size))
    try:
        pieces = inner.batch_decode([[i] for i in ids])
    except Exception:
        pieces = [inner.decode([i]) for i in ids]
    return [i for i, piece in zip(ids, pieces) if any(b in piece for b in breakers)]


class DryProcessor:
    """Stateless DRY logits processor. See the module docstring."""

    def __init__(
        self,
        multiplier: float,
        base: float = 1.75,
        allowed_length: int = 2,
        penalty_last_n: int = 4096,
        breaker_token_ids: Sequence[int] | None = None,
        prompt_length: int = 0,
        think_end_token_id: int | None = None,
        exclude_reasoning: bool = False,
        think_start_token_id: int | None = None,
        starts_in_reasoning: bool = False,
    ) -> None:
        if multiplier < 0:
            raise ValueError(f"dry_multiplier must be non-negative, got {multiplier}")
        if base < 1:
            raise ValueError(f"dry_base must be >= 1, got {base}")
        if allowed_length < 1:
            raise ValueError(f"dry_allowed_length must be >= 1, got {allowed_length}")
        self.multiplier = float(multiplier)
        self.base = float(base)
        self.allowed_length = int(allowed_length)
        if penalty_last_n < 0:
            penalty_last_n = MAX_PENALTY_LAST_N
        self.window = min(int(penalty_last_n), MAX_PENALTY_LAST_N)
        self.max_match = self.allowed_length + MATCH_HEADROOM
        self.prompt_length = max(int(prompt_length), 0)
        self.think_end_token_id = think_end_token_id
        # Exclusion needs the close-think token to find where reasoning ends;
        # without it there is no boundary and DRY applies throughout.
        self.exclude_reasoning = (
            bool(exclude_reasoning) and think_end_token_id is not None
        )
        self.think_start_token_id = think_start_token_id
        self.starts_in_reasoning = bool(starts_in_reasoning)
        self._breaker_ids = sorted(
            {int(t) for t in (breaker_token_ids or ()) if t >= 0}
        )
        self._breaker_mask: mx.array | None = None
        self._offsets = mx.arange(self.max_match)[:, None]

    @property
    def enabled(self) -> bool:
        return self.multiplier > 0 and self.window > 0

    # Stateless: nothing to checkpoint when a speculative draft is rejected.
    def snapshot_state(self) -> dict:
        return {}

    def restore_state(self, state: dict) -> None:
        return None

    def _breakers(self, vocab_size: int) -> mx.array:
        # Sized from the logits, not the tokenizer: the two can disagree.
        mask = self._breaker_mask
        if mask is None or mask.shape[0] != vocab_size:
            mask = mx.zeros(vocab_size, dtype=mx.bool_)
            ids = [t for t in self._breaker_ids if t < vocab_size]
            if ids:
                mask[mx.array(ids)] = True
            self._breaker_mask = mask
        return mask

    def _in_reasoning(self, generated: mx.array) -> mx.array:
        """Whether the response is currently inside a thinking block.

        Reasoning is open after the last think-start (or from the first token
        when the prompt opened it) until a later close-think.
        """
        positions = mx.arange(generated.shape[0])
        last_close = mx.where(generated == self.think_end_token_id, positions, -2).max()
        last_open = mx.array(-1 if self.starts_in_reasoning else -3)
        if self.think_start_token_id is not None:
            last_open = mx.maximum(
                last_open,
                mx.where(generated == self.think_start_token_id, positions, -3).max(),
            )
        return last_open > last_close

    def __call__(self, tokens, logits: mx.array) -> mx.array:
        if not self.enabled:
            return logits
        total = len(tokens)
        start = max(self.prompt_length, total - self.window)
        width = total - start
        if width <= self.allowed_length:
            return logits

        if self.exclude_reasoning:
            # The reasoning state depends on the whole response, not just the
            # window: a long answer can push its close-think out of it.
            generated = tokens[self.prompt_length :]
            if not isinstance(generated, mx.array):
                generated = mx.array(generated)
            generated = generated.astype(mx.int32)
            in_reasoning = self._in_reasoning(generated)
            recent = generated[-width:]
        else:
            in_reasoning = None
            recent = tokens[start:]
            if not isinstance(recent, mx.array):
                recent = mx.array(recent)
            recent = recent.astype(mx.int32)

        cap = self.max_match
        vocab_size = logits.shape[-1]
        # Left-pad with an id no token has, so walking back past the start of
        # the window ends the match instead of wrapping around.
        padded = mx.concatenate([mx.full(cap, -1, dtype=mx.int32), recent])
        ends = mx.arange(width - 1)[None, :]
        # history[k, i] is the token k steps before earlier position i;
        # tail[k] is the token k steps before the current end.
        history = padded[cap + ends - self._offsets]
        tail = padded[cap + (width - 1) - self._offsets]
        tail_ids = mx.clip(tail, 0, vocab_size - 1)
        extends = (
            (history == tail) & (tail >= 0) & ~self._breakers(vocab_size)[tail_ids]
        )
        match_length = mx.cumprod(extends.astype(mx.int32), axis=0).sum(axis=0)

        applies = match_length >= self.allowed_length
        if self.think_end_token_id is not None:
            # Search only after the last close-think token. A match cannot
            # reach back across it either: the tail holds no close-think.
            positions = mx.arange(width)
            last_close = mx.where(
                recent == self.think_end_token_id, positions, -1
            ).max()
            applies = applies & (positions[:-1] > last_close)

        if in_reasoning is not None:
            applies = applies & ~in_reasoning
        penalty = mx.where(
            applies,
            self.multiplier
            * mx.power(
                self.base, (match_length - self.allowed_length).astype(mx.float32)
            ),
            0.0,
        )
        # The penalized token is the one that followed each earlier match;
        # several matches naming the same token keep the longest.
        followers = mx.clip(recent[1:], 0, vocab_size - 1)
        per_token = (
            mx.zeros(vocab_size, dtype=mx.float32).at[followers].maximum(penalty)
        )
        return logits - per_token.astype(logits.dtype)[None, :]
