# SPDX-License-Identifier: Apache-2.0
"""Stable home for ``MockTokenizer``.

A test that does ``from conftest import MockTokenizer`` is fragile: pytest
imports every ``conftest.py`` under the bare module name ``conftest``, so
``tests/cluster_assistant/conftest.py`` can shadow the root one in
``sys.modules`` (that made ``tests/test_scheduler.py`` fail only when the
cluster-assistant tests were collected first). Importing from this uniquely
named module removes the collision.
"""

from __future__ import annotations

import zlib
from typing import Any, Dict, List, Optional


class MockTokenizer:
    """Mock tokenizer for testing without loading real models."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids (simple simulation)."""
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        for word in text.split():
            # crc32, unlike hash(), is stable across processes.
            token_id = (zlib.crc32(word.encode()) % (self.vocab_size - 10)) + 10
            tokens.append(token_id)
        return tokens

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token ids to text (simple simulation)."""
        if skip_special_tokens:
            token_ids = [
                t
                for t in token_ids
                if t not in (self.eos_token_id, self.pad_token_id, self.bos_token_id)
            ]
        return f"<decoded:{len(token_ids)} tokens>"

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Tokenize text and return dict with input_ids."""
        input_ids = self.encode(text)
        return {"input_ids": input_ids}
