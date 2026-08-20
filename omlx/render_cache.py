# SPDX-License-Identifier: Apache-2.0
"""Exact-key memo for chat-template rendering and prompt tokenization.

One streaming chat request renders the same conversation up to three times
(admission preflight, thinking-state detection, the engine's own render) and
BPE-encodes the rendered prompt at least twice (preflight count, request
tokenization). Each render is a full Jinja pass and each encode a full BPE
pass over the conversation — for a 30K-token agentic prompt that is repeated
work inside TTFT, after the KV prefix cache has already removed the prefill.

The memo is exact-key: a SHA-256 over the serialized inputs. No prefix
composition and no template semantics — a hit returns byte-identical output,
so it cannot change what any caller sees. An identical retry of a request
(same conversation, same kwargs) hits as well; a conversation extended by a
new turn misses and pays one full render, exactly as before.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

# A rendered 30K-token prompt is ~120KB; token ids ~250KB. 32 entries bounds
# the memo to a few MB per engine while covering the duplicate calls of every
# in-flight request plus immediate retries.
_DEFAULT_MAX_ENTRIES = 32


class RenderMemo:
    """Small thread-safe LRU keyed by a digest of the serialized inputs."""

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._lock = threading.Lock()
        self._entries: OrderedDict[bytes, Any] = OrderedDict()
        self._max_entries = max_entries
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(*parts: Any) -> bytes | None:
        """Digest the parts; None when a part cannot be serialized.

        A None key means "do not cache this call" — callers fall through to
        the uncached path, so exotic non-JSON message payloads (VLM image
        handles, custom objects) never break rendering.
        """
        digest = hashlib.sha256()
        for part in parts:
            digest.update(b"\x1f")
            if isinstance(part, str):
                payload = part.encode("utf-8", "surrogatepass")
            elif isinstance(part, bytes):
                payload = part
            else:
                try:
                    payload = json.dumps(
                        part, sort_keys=True, ensure_ascii=False
                    ).encode("utf-8", "surrogatepass")
                except (TypeError, ValueError):
                    return None
            digest.update(payload)
        return digest.digest()

    def get(self, key: bytes | None) -> Any | None:
        if key is None:
            return None
        with self._lock:
            value = self._entries.get(key)
            if value is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: bytes | None, value: Any) -> None:
        if key is None or value is None:
            return
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "hits": self.hits,
                "misses": self.misses,
            }


# Shared across the engine's preflight encode and the scheduler's
# add_request encode — the same prompt string, the same tokenizer, and the
# same bare ``encode(prompt)`` call shape, so a hit is byte-identical to
# what the second caller would have computed itself.
#
# Bounded in real memory, not just entries: a Python int tuple costs ~30B
# per token, so unbounded entries at million-token prompts would pin
# gigabytes. Oversized prompts bypass the memo entirely — the duplicate
# BPE pass is cheaper than holding the ids resident.
_ENCODE_MEMO = RenderMemo(max_entries=8)
_ENCODE_MEMO_MAX_TOKENS = 131_072
# Rendered prompt strings are ~4B/token; the same reasoning caps them.
RENDER_MEMO_MAX_CHARS = 4_000_000


def encode_cached(tokenizer: Any, prompt: str) -> list[int]:
    """``tokenizer.encode(prompt)`` with an exact-key memo; returns a copy.

    The scope prefers ``_omlx_cache_id`` (stamped at engine load, survives
    the Scheduler's tokenizer deepcopy) over ``id()``, which can never
    match across that copy.
    """
    key = RenderMemo.key(
        str(getattr(tokenizer, "_omlx_cache_id", "") or id(tokenizer)),
        str(getattr(tokenizer, "name_or_path", "") or ""),
        prompt,
    )
    ids = _ENCODE_MEMO.get(key)
    if ids is None:
        ids = tuple(tokenizer.encode(prompt))
        if len(ids) <= _ENCODE_MEMO_MAX_TOKENS:
            _ENCODE_MEMO.put(key, ids)
    return list(ids)


def encode_memo_stats() -> dict[str, int]:
    return _ENCODE_MEMO.stats()
