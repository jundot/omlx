# SPDX-License-Identifier: Apache-2.0
"""Context-copy drafts for Lightning MTP (prompt lookup).

When the text being written repeats something already in the context (a file
being edited, quoted code, a table), the tokens that followed the earlier
occurrence are a better draft than the MTP head can make. The proposer finds
the longest earlier match of the context tail and proposes the tokens that
came after it. The target model still verifies every draft, so greedy output
is unchanged; only the number of tokens per verify forward changes.

The matching rule follows TensorFold's ``SuffixLookupProposer`` (MIT): a
3-gram index over prompt + generated tokens, the longest backward match up to
64 tokens (the most recent occurrence on ties), and a pause after repeated
zero-accept rounds. Differences, measured on Qwen3.8-Flash-Next: copies
need a 12-token match (TensorFold uses 8; 8-11 token matches in chat and
long-context replies accepted fewer tokens than the MTP chain they replace);
only the 32 most recent occurrences are compared, which bounds the host time
a cycle spends on repetitive text; and a copy proposes 7 tokens, or, on a
target that verifies 16-row windows, 15 when the tail repeats the source
verbatim for the whole 64-token match and the previous copy was accepted whole
(text being copied out, like an edited file, rather than a repeated pattern
whose details change every few lines).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

# One verify window carries the pending token plus the drafts. Qwen4-Exp's
# row-exact verify kernels (MoE routed window, router, row-exact qmv, fused
# attention, GDN verify, hyper-connections) cover 16 rows, so there a copy
# proposes at most 15 tokens. An 8-row window costs about 55% of a 16-row one
# (M5 Ultra), so the wide window is kept for verbatim runs, where it is
# accepted whole. Other targets keep 8-row windows (GLM-5.3's sparse caches
# cannot undo a longer verify block), so their copies stay at 7 tokens.
MAX_COPY = 15
_NARROW_COPY = 7
_MAX_MATCH = 64
_ENTER_MATCH = 12
_MIN_COPY = 2
_CANDIDATES = 32
_MISSES_BEFORE_QUIET = 4
_QUIET_CYCLES = 8
# Token ids fit in 20 bits (Qwen vocabularies are ~250K), so a 3-gram packs
# into one int.
_KEY_BITS = 20


def _key(a: int, b: int, c: int) -> int:
    return (a << (2 * _KEY_BITS)) | (b << _KEY_BITS) | c


class ContextCopy:
    """Suffix lookup over one request's committed tokens.

    The prompt (everything known when the index is built) is indexed once
    with numpy: 3-gram keys sorted stably, so each key's end positions are a
    contiguous ascending run. Tokens appended later go into a dict.
    """

    def __init__(self, wide_window: bool = False) -> None:
        """``wide_window``: the target verifies 16-row windows (MAX_COPY + 1)."""
        self._wide_window = wide_window
        self._ids: List[int] = []
        self._sorted_keys = np.zeros(0, dtype=np.int64)
        self._sorted_ends = np.zeros(0, dtype=np.int64)
        self._recent: Dict[int, List[int]] = {}
        self._misses = 0
        self._quiet = 0
        self._wide = False

    def _rebuild(self, history: Sequence[int]) -> None:
        self._ids = list(history)
        self._recent = {}
        if len(history) < 3:
            self._sorted_keys = np.zeros(0, dtype=np.int64)
            self._sorted_ends = np.zeros(0, dtype=np.int64)
            return
        t = np.asarray(history, dtype=np.int64)
        keys = (t[:-2] << (2 * _KEY_BITS)) | (t[1:-1] << _KEY_BITS) | t[2:]
        order = np.argsort(keys, kind="stable")
        self._sorted_keys = keys[order]
        self._sorted_ends = order + 2

    def extend(self, history: Sequence[int], committed: Sequence[int]) -> bool:
        """Index ``history + committed``; True when the index was rebuilt.

        ``history`` is the request's token list before this cycle's emits.
        The index normally already holds exactly it (the previous cycle
        appended the same committed tokens); anything else — the first
        cycle, a standard step, a re-entry, a fallback — rebuilds from
        ``history`` (about 3 ms per 44K prompt tokens on M5 Ultra).
        """
        ids = self._ids
        rebuilt = len(ids) != len(history) or (ids and ids[-1] != history[-1])
        if rebuilt:
            self._rebuild(history)
            ids = self._ids
        recent = self._recent
        for token in committed:
            ids.append(token)
            end = len(ids) - 1
            if end >= 2:
                key = _key(ids[end - 2], ids[end - 1], token)
                bucket = recent.get(key)
                if bucket is None:
                    recent[key] = [end]
                else:
                    bucket.append(end)
        return bool(rebuilt)

    def propose(self, limit: int) -> List[int]:
        """Tokens that followed the longest earlier match of the tail."""
        if self._quiet:
            self._quiet -= 1
            return []
        ids = self._ids
        n = len(ids)
        if limit < _MIN_COPY or n <= _ENTER_MATCH:
            return []
        key = _key(ids[-3], ids[-2], ids[-1])
        # Newest first; the tail's own entry is skipped below.
        candidates = list(reversed(self._recent.get(key, ())))
        if len(candidates) < _CANDIDATES:
            lo = int(np.searchsorted(self._sorted_keys, key, "left"))
            hi = int(np.searchsorted(self._sorted_keys, key, "right"))
            lo = max(lo, hi - (_CANDIDATES - len(candidates)))
            candidates.extend(reversed(self._sorted_ends[lo:hi].tolist()))
        best_end, best_len = -1, 0
        tail = n - 1
        for end in candidates[:_CANDIDATES]:
            if end >= tail:
                continue
            length = 3
            cap = min(_MAX_MATCH, end + 1)
            while length < cap and ids[end - length] == ids[tail - length]:
                length += 1
            if length > best_len:
                best_end, best_len = end, length
                if length == _MAX_MATCH:
                    break
        if best_len < _ENTER_MATCH:
            return []
        verbatim = self._wide_window and self._wide and best_len == _MAX_MATCH
        take = min(limit, MAX_COPY if verbatim else _NARROW_COPY)
        copied = ids[best_end + 1 : best_end + 1 + take]
        return copied if len(copied) >= _MIN_COPY else []

    def observe(self, accepted: int, drafted: int) -> None:
        """Allow a wide copy after one accepted whole; pause copies after
        several rounds whose first token missed."""
        self._wide = accepted == drafted
        if accepted:
            self._misses = 0
            return
        self._misses += 1
        if self._misses >= _MISSES_BEFORE_QUIET:
            self._misses = 0
            self._quiet = _QUIET_CYCLES
