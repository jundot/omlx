# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SpecPrefillSelectionKey:
    token_digest: str
    token_count: int
    draft_model_name: str
    keep_pct: float
    chunk_size: int
    prefill_step_size: int


class SpecPrefillSelectionCache:
    def __init__(self, max_entries: int = 128):
        self.max_entries = max(0, max_entries)
        self._entries: OrderedDict[SpecPrefillSelectionKey, tuple[int, ...]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def digest_tokens(tokens: Iterable[int]) -> tuple[str, int]:
        digest = hashlib.blake2b(digest_size=16)
        count = 0
        for token in tokens:
            digest.update(int(token).to_bytes(8, byteorder="little", signed=True))
            count += 1
        return digest.hexdigest(), count

    @classmethod
    def make_key(
        cls,
        tokens: Iterable[int],
        *,
        draft_model_name: str,
        keep_pct: float,
        chunk_size: int,
        prefill_step_size: int,
    ) -> SpecPrefillSelectionKey:
        token_digest, token_count = cls.digest_tokens(tokens)
        return SpecPrefillSelectionKey(
            token_digest=token_digest,
            token_count=token_count,
            draft_model_name=draft_model_name,
            keep_pct=float(keep_pct),
            chunk_size=int(chunk_size),
            prefill_step_size=int(prefill_step_size),
        )

    def get(self, key: SpecPrefillSelectionKey) -> tuple[int, ...] | None:
        selected = self._entries.get(key)
        if selected is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return selected

    def put(self, key: SpecPrefillSelectionKey, selected: Iterable[int]) -> None:
        if self.max_entries <= 0:
            return
        values = tuple(int(i) for i in selected)
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = values
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }
