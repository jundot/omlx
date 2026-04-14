# SPDX-License-Identifier: Apache-2.0
"""
Thinking/reasoning content parser for separating <think>...</think> blocks.

Provides both streaming (ThinkingParser) and non-streaming (extract_thinking)
interfaces for separating reasoning content from regular response content.

Used by reasoning models like DeepSeek R1, Qwen3/3.5, MiniMax that wrap
their chain-of-thought reasoning in <think>...</think> tags.
"""

import re
from typing import List, Optional, Tuple


# Tags used for thinking blocks
_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"
_OPEN_LEN = len(_OPEN_TAG)   # 7
_CLOSE_LEN = len(_CLOSE_TAG)  # 8

# Regex for non-streaming extraction (complete text)
_THINKING_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
# Handle case where <think> is missing but </think> is present
# (scheduler prepends <think>\n but the tag may be split)
_THINKING_TAIL_PATTERN = re.compile(r'^(.*?)</think>', re.DOTALL)


def extract_thinking(text: str) -> Tuple[str, str]:
    """Extract thinking and content from complete text.

    Handles:
    - Normal: ``<think>reasoning</think>answer`` → ``("reasoning", "answer")``
    - No thinking: ``just answer`` → ``("", "just answer")``
    - Partial (no open tag): ``reasoning</think>answer`` → ``("reasoning", "answer")``
    - Empty think: ``<think></think>answer`` → ``("", "answer")``
    - Think only: ``<think>reasoning</think>`` → ``("reasoning", "")``

    Args:
        text: Complete model output text.

    Returns:
        Tuple of (thinking_content, regular_content).
    """
    if not text:
        return ("", "")

    thinking_parts = []
    remaining = text

    # Extract all <think>...</think> blocks
    while True:
        match = _THINKING_PATTERN.search(remaining)
        if not match:
            break
        thinking_parts.append(match.group(1))
        remaining = remaining[:match.start()] + remaining[match.end():]

    if thinking_parts:
        thinking = "\n".join(thinking_parts).strip()
        return (thinking, remaining.strip())

    # Handle partial: content before </think> without <think> tag
    if '</think>' in text and '<think>' not in text:
        match = _THINKING_TAIL_PATTERN.match(text)
        if match:
            thinking = match.group(1).strip()
            remaining = text[match.end():].strip()
            return (thinking, remaining)

    return ("", text)


class ThinkingParser:
    """Stateful streaming parser for separating <think>...</think> from content.

    Handles streaming chunks where tags may span multiple chunks.
    Returns (thinking_delta, content_delta) tuples for each feed() call.

    Example::

        parser = ThinkingParser()

        # Chunk 1: "<think>Let me"
        t, c = parser.feed("<think>Let me")
        # t = "Let me", c = ""

        # Chunk 2: " think</think>Answer"
        t, c = parser.feed(" think</think>Answer")
        # t = " think", c = "Answer"

        # Flush remaining
        t, c = parser.finish()
    """

    def __init__(self):
        self._in_thinking: bool = False
        self._buffer: str = ""  # Buffer for potential partial tags

    def feed(self, text: str) -> Tuple[str, str]:
        """Feed a text chunk, return (thinking_delta, content_delta).

        Args:
            text: New text chunk from model output.

        Returns:
            Tuple of (thinking_text, content_text) extracted from this chunk.
        """
        if not text:
            return ("", "")

        # Prepend any buffered partial tag content
        text = self._buffer + text
        self._buffer = ""

        thinking_out = []
        content_out = []

        i = 0
        while i < len(text):
            if text[i] == '<':
                # Check if this could be a tag start
                remaining = text[i:]

                # Try to match <think>
                if remaining.startswith(_OPEN_TAG):
                    self._in_thinking = True
                    i += _OPEN_LEN
                    continue

                # Try to match </think>
                if remaining.startswith(_CLOSE_TAG):
                    self._in_thinking = False
                    i += _CLOSE_LEN
                    continue

                # Check if it could be a partial tag (not enough chars yet)
                if self._could_be_tag(remaining):
                    # Buffer the rest and wait for more data
                    self._buffer = remaining
                    break

                # Not a tag, emit the '<' as regular content
                if self._in_thinking:
                    thinking_out.append('<')
                else:
                    content_out.append('<')
                i += 1
            else:
                if self._in_thinking:
                    thinking_out.append(text[i])
                else:
                    content_out.append(text[i])
                i += 1

        return ("".join(thinking_out), "".join(content_out))

    def finish(self) -> Tuple[str, str]:
        """Flush any remaining buffered content.

        Should be called when the stream is complete to emit any
        buffered characters that were waiting for potential tag completion.

        Returns:
            Tuple of (thinking_text, content_text) from remaining buffer.
        """
        if not self._buffer:
            return ("", "")

        # The buffer contains a partial tag that never completed.
        # Emit it as-is in the current mode.
        buf = self._buffer
        self._buffer = ""

        if self._in_thinking:
            return (buf, "")
        else:
            return ("", buf)

    @staticmethod
    def _could_be_tag(text: str) -> bool:
        """Check if text could be the start of a <think> or </think> tag.

        Returns True if text is a proper prefix of either tag but not
        yet a complete match.
        """
        length = len(text)
        if length >= _CLOSE_LEN:
            # Long enough to determine - not a partial tag
            return False

        # Check against both tags
        if _OPEN_TAG[:length] == text:
            return True
        if _CLOSE_TAG[:length] == text:
            return True

        return False


class ThinkingBudgetProcessor:
    """Logits processor that enforces a thinking token budget.

    Counts tokens generated while in thinking mode.  When the budget is
    exceeded, forces the close-think token(s) one at a time, then becomes
    a no-op for the rest of generation.

    Handles both single-token and multi-token close-think sequences, and
    supports alternative think markers (e.g. ``<longcat_think>``).

    Args:
        think_end_token_ids: Token ID(s) for the close-think tag.
        budget: Maximum number of thinking tokens before forcing close.
        think_start_token_id: Token ID for the open-think tag (re-entry detection).
    """

    def __init__(
        self,
        think_end_token_ids: List[int],
        budget: int,
        think_start_token_id: Optional[int] = None,
        leading_token_ids: Optional[List[int]] = None,
        trailing_token_ids: Optional[List[int]] = None,
    ):
        self._think_end_ids = think_end_token_ids
        # Full force sequence: \n + </think> + \n\n (matches training pattern)
        self._force_sequence = (
            (leading_token_ids or [])
            + list(think_end_token_ids)
            + (trailing_token_ids or [])
        )
        self._budget = budget
        self._think_start_id = think_start_token_id

        # State
        self._thinking_tokens: int = 0
        self._in_thinking: bool = True  # Starts True (prompt ends with <think>)
        self._forcing: bool = False
        self._force_idx: int = 0
        self._done: bool = False
        self._first_call: bool = True
        # After forced sequence, suppress duplicate </think> tokens
        self._suppress_end: bool = False
        # Sliding window for multi-token end detection
        self._recent_tokens: List[int] = []
        # Flat set for fast single-token suppression check
        self._end_id_set = set(think_end_token_ids)

    def __call__(self, tokens, logits):
        """mlx-lm logits processor: (tokens, logits) -> logits."""
        import mlx.core as mx

        if self._done:
            return logits

        # Post-forcing phase: suppress duplicate </think> tokens
        if self._suppress_end:
            return self._suppress_end_tokens(logits, mx)

        # In new mlx-lm API, tokens is the full history list.
        # Accept each genuinely generated token exactly once (see grammar.py).
        n = len(tokens)
        if not hasattr(self, "_accepted_up_to"):
            self._accepted_up_to = n  # skip prompt tokens
        elif n > self._accepted_up_to:
            for i in range(self._accepted_up_to, n):
                self._update_state(int(tokens[i]))
            self._accepted_up_to = n

        # If state changed by _update_state, handle immediately
        if self._done:
            return logits
        if self._suppress_end:
            return self._suppress_end_tokens(logits, mx)

        if self._forcing:
            return self._force_next_token(logits, mx)

        if self._in_thinking:
            self._thinking_tokens += 1
            if self._thinking_tokens >= self._budget:
                self._forcing = True
                self._force_idx = 0
                return self._force_next_token(logits, mx)

        return logits

    def _update_state(self, token_id: int) -> None:
        """Update thinking state based on the last generated token."""
        if self._forcing:
            self._force_idx += 1
            if self._force_idx >= len(self._force_sequence):
                self._in_thinking = False
                self._forcing = False
                # Don't set _done — enter suppression mode to prevent
                # the model from generating a duplicate </think>.
                self._suppress_end = True
            return

        # Detect natural close-think via sliding window
        if len(self._think_end_ids) == 1:
            if token_id == self._think_end_ids[0]:
                self._in_thinking = False
                self._done = True
                return
        else:
            self._recent_tokens.append(token_id)
            if len(self._recent_tokens) > len(self._think_end_ids):
                self._recent_tokens.pop(0)
            if self._recent_tokens == self._think_end_ids:
                self._in_thinking = False
                self._done = True
                return

        # Detect re-entry into thinking (rare but possible)
        if not self._in_thinking and self._think_start_id and token_id == self._think_start_id:
            self._in_thinking = True

    def _force_next_token(self, logits, mx):
        """Force the next token in the close-think + trailing sequence."""
        target_id = self._force_sequence[self._force_idx]
        forced = mx.full(logits.shape, float("-inf"))
        forced[..., target_id] = 0.0
        return forced

    def _suppress_end_tokens(self, logits, mx):
        """Suppress duplicate </think> tokens after forced close."""
        # Set logits of all end-token IDs to -inf so the model
        # cannot produce another </think>.
        for tid in self._end_id_set:
            logits[..., tid] = float("-inf")
        return logits
