# SPDX-License-Identifier: Apache-2.0
"""Reuse the tokens of a recent prompt that shares a long text prefix with a new one.

Each request encodes its whole prompt, and the VLM engine encodes a text-only
prompt twice (once to count it, once to build the model inputs). For a new
question on a cached 128K-token document that is ~85 ms per encode on an M5
Pro, while the prefix cache restores the KV of the shared document.

Byte-level BPE behind a regex pre-tokenizer is local: once a re-encode from a
cut point reproduces the tokens a previous encode produced for the same text,
the rest of the shared prefix tokenizes the same way. A new text that shares a
long prefix with a recent one is therefore encoded from a cut a margin before
the end of the shared prefix, and the re-encoded margin must equal the cached
tokens exactly. Anything else falls back to a full encode, so the result is
always the tokenizer's own.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Shorter texts encode in a few ms; below this every call is a plain encode.
MIN_CHARS = 32_768
# Chars before the end of the shared prefix where the re-encode starts at the
# latest, and cached tokens before that start it must reproduce.
_MARGIN_CHARS = 256
_CHECK_TOKENS = 64
_ENTRIES = 4


class PrefixTokenCache:
    """Encodes texts without special tokens, reusing a recent shared prefix."""

    def __init__(self, backend: Any, entries: int = _ENTRIES):
        self._backend = backend
        self._entries = entries
        self._items: list[tuple[str, list[int], list[tuple[int, int]]]] = []
        self._lock = threading.Lock()
        self.reused = 0
        self.full = 0

    # A copied cache starts empty with its own lock.
    def __deepcopy__(self, memo):
        return PrefixTokenCache(self._backend, self._entries)

    def __reduce__(self):
        return (PrefixTokenCache, (self._backend, self._entries))

    @staticmethod
    def _shared_chars(a: str, b: str) -> int:
        lo, hi = 0, min(len(a), len(b))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if a[:mid] == b[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _reuse(self, text: str, item, shared: int):
        _, ids, offsets = item
        limit = shared - _MARGIN_CHARS
        lo, hi = 0, len(offsets)
        while lo < hi:  # first cached token that ends past the limit
            mid = (lo + hi) // 2
            if offsets[mid][1] <= limit:
                lo = mid + 1
            else:
                hi = mid
        end = lo
        start_token = max(0, end - _CHECK_TOKENS)
        if end - start_token <= 0 or start_token >= len(offsets):
            return None
        start = offsets[start_token][0]
        enc = self._backend.encode(text[start:], add_special_tokens=False)
        if list(enc.ids[: end - start_token]) != ids[start_token:end]:
            return None
        return (
            ids[:start_token] + list(enc.ids),
            offsets[:start_token] + [(a + start, b + start) for a, b in enc.offsets],
        )

    def encode(self, text: str) -> list[int]:
        """``backend.encode(text, add_special_tokens=False).ids``, faster on shared prefixes."""
        with self._lock:
            items = list(self._items)
        best, shared = None, 0
        for item in items:
            n = self._shared_chars(item[0], text)
            if n > shared:
                best, shared = item, n
        result = self._reuse(text, best, shared) if best is not None and shared >= MIN_CHARS else None
        if result is None:
            enc = self._backend.encode(text, add_special_tokens=False)
            result = (list(enc.ids), list(enc.offsets))
            self.full += 1
        else:
            self.reused += 1
        with self._lock:
            self._items.insert(0, (text, result[0], result[1]))
            del self._items[self._entries :]
        return list(result[0])


def _byte_level_bpe(backend: Any) -> bool:
    """A BPE model behind a pre-tokenizer with a ByteLevel step (GPT-2 style)."""
    try:
        spec = json.loads(backend.to_str())
    except Exception:  # noqa: BLE001
        return False

    def has_byte_level(node) -> bool:
        if not isinstance(node, dict):
            return False
        if node.get("type") == "ByteLevel":
            return True
        return any(has_byte_level(n) for n in node.get("pretokenizers") or [])

    return (spec.get("model") or {}).get("type") == "BPE" and has_byte_level(
        spec.get("pre_tokenizer")
    )


def prefix_token_cache_for(tokenizer: Any) -> PrefixTokenCache | None:
    """A cache for ``tokenizer``, or None when it does not qualify.

    Only fast (``tokenizers``-backed) byte-level BPE tokenizers whose ``encode``
    adds no special tokens qualify, so the cache's ids equal
    ``tokenizer.encode(text)`` and the tokenization is local.
    """
    hf = tokenizer if hasattr(tokenizer, "backend_tokenizer") else getattr(tokenizer, "_tokenizer", None)
    backend = getattr(hf, "backend_tokenizer", None)
    if backend is None or not _byte_level_bpe(backend):
        return None
    try:
        probe = "Probe text,\n  with spaces and a tab\t."
        if list(hf.encode(probe)) != list(backend.encode(probe, add_special_tokens=False).ids):
            return None
    except Exception:  # noqa: BLE001
        return None
    return PrefixTokenCache(backend)


def encode_prompt(tokenizer: Any, cache: PrefixTokenCache | None, prompt: str) -> list[int]:
    """``tokenizer.encode(prompt)``, through ``cache`` for long prompts."""
    if cache is not None and isinstance(prompt, str) and len(prompt) >= MIN_CHARS:
        return cache.encode(prompt)
    return list(tokenizer.encode(prompt))
