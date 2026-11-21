# SPDX-License-Identifier: Apache-2.0
"""Suffix-lookup drafter: proposes the tokens that followed an earlier repeat.

When the output quotes its context (an edited file, repeated code, JSON keys,
a table), the tokens after the longest earlier occurrence of the current
suffix are usually the continuation, and one verify pass accepts many of them.
It needs no weights and no GPU; the target model still checks every token.

The matching rule is Basalt's ``SuffixDrafter`` (``basalt/spec/suffix_drafter``):
every trigram of the history (prompt + accepted output) keeps its ``WAYS`` most
recent end positions, and so does every ``LONG_ORDER``-gram, so a common
trigram's few slots hold recent repeats while the longer key still finds an old
source of the text being quoted. A proposal extends each candidate backwards up
to ``max_match`` tokens and takes the longest (the long key's candidates first,
the most recent on ties). Matches shorter than ``min_match`` propose nothing.

The layout is not Basalt's open-addressing table, because a Python loop over a
98K-token prompt would cost more than the first token: a bulk append is
indexed with numpy (sorted keys, so a key's end positions are one ascending
run), and the tokens accepted afterwards go into two small dicts. ``reset``
drops the index and clears only those dicts, so nothing proportional to a
table's capacity is touched between requests. The history is a compact
``array``, which the garbage collector does not walk. Key collisions are harmless:
a candidate only counts for the tokens that really match.
"""

from __future__ import annotations

from array import array
from typing import Dict, List, Sequence, Tuple

import numpy as np

_TOKEN_BITS = 21
_TOKEN_MASK = (1 << _TOKEN_BITS) - 1
_U64 = (1 << 64) - 1
# Table keys keep 63 bits of the hash, so they fit numpy's int64.
_KEY = (1 << 63) - 1
# Polynomial hash of the last LONG_ORDER tokens, rolled per appended token.
_BASE = 0x9E3779B97F4A7C15
# Appends at least this long are indexed in bulk.
_BULK = 256


class SuffixDrafter:
    """Longest-suffix-match continuation over one request's tokens."""

    WAYS = 4
    ORDER = 3
    LONG_ORDER = 8

    def __init__(
        self, min_match: int = 3, max_match: int = 32, periodic: bool = True
    ) -> None:
        """``periodic``: a continuation that reaches the end of the history
        goes on through its own tokens (an overlapping copy), so a run or a
        repeating record is proposed past one period. Basalt stops there."""
        self.min_match = max(self.ORDER, int(min_match))
        self.max_match = max(self.min_match, int(max_match))
        self.periodic = bool(periodic)
        self._top = pow(_BASE, self.LONG_ORDER - 1, 1 << 64)
        self._hist = array("q")
        # Tokens appended one by one: key -> end positions, oldest first.
        self._short: Dict[int, List[int]] = {}
        self._long: Dict[int, List[int]] = {}
        # The bulk index: sorted keys and their end positions.
        self._bulk_short = self._bulk_long = None
        self._hash = 0
        self.last_match = 0

    def __len__(self) -> int:
        return len(self._hist)

    def reset(self) -> None:
        """Forget the history. Costs what was indexed, not a table's size."""
        self._hist = array("q")
        self._short.clear()
        self._long.clear()
        self._bulk_short = self._bulk_long = None
        self._hash = 0
        self.last_match = 0

    def sync(self, history: Sequence[int]) -> bool:
        """Make the index hold exactly ``history``; True when it was rebuilt.

        The usual call finds the index already holding a prefix of ``history``
        and appends the rest. Anything else (the first call, a history that
        diverged) starts over from ``history``.
        """
        held = len(self._hist)
        n = len(history)
        grown = (
            0 < held <= n
            and self._hist[-1] == history[held - 1]
            and self._hist[0] == history[0]
        )
        if grown:
            if held < n:
                self.append(history[held:])
            return False
        self.reset()
        self.append(history)
        return True

    def append(self, tokens: Sequence[int]) -> None:
        """Add tokens to the history (the prompt, then every accepted token)."""
        if len(tokens) >= _BULK:
            self._hist.extend(tokens)
            self._reindex()
            return
        hist = self._hist
        short, long_ = self._short, self._long
        ways, top, h = self.WAYS, self._top, self._hash
        full = self.LONG_ORDER - 1
        for token in tokens:
            token = int(token)
            end = len(hist)
            if end > full:
                h = (h - hist[end - full - 1] * top) & _U64
            h = (h * _BASE + token) & _U64
            hist.append(token)
            if end < 2:
                continue
            key = (
                ((hist[end - 2] & _TOKEN_MASK) << (2 * _TOKEN_BITS))
                | ((hist[end - 1] & _TOKEN_MASK) << _TOKEN_BITS)
                | (token & _TOKEN_MASK)
            )
            slot = short.get(key)
            if slot is None:
                short[key] = [end]
            else:
                slot.append(end)
                if len(slot) > ways:
                    del slot[0]
            if end >= full:
                slot = long_.get(h & _KEY)
                if slot is None:
                    long_[h & _KEY] = [end]
                else:
                    slot.append(end)
                    if len(slot) > ways:
                        del slot[0]
        self._hash = h

    def _reindex(self) -> None:
        """Index the whole history in bulk and empty the per-token dicts."""
        self._short.clear()
        self._long.clear()
        self._bulk_short = self._bulk_long = None
        self._hash = 0
        n = len(self._hist)
        if n < self.ORDER:
            self._hash = self._rolled(n)
            return
        t = np.frombuffer(self._hist, dtype=np.int64)
        m = t & _TOKEN_MASK
        keys = (m[:-2] << (2 * _TOKEN_BITS)) | (m[1:-1] << _TOKEN_BITS) | m[2:]
        order = np.argsort(keys, kind="stable")
        self._bulk_short = (keys[order], order + 2)
        if n >= self.LONG_ORDER:
            u = t.astype(np.uint64)
            width = n - self.LONG_ORDER + 1
            acc = np.zeros(width, dtype=np.uint64)
            base = np.uint64(_BASE)
            with np.errstate(over="ignore"):
                for j in range(self.LONG_ORDER):
                    acc = acc * base + u[j : j + width]
            self._hash = int(acc[-1])
            keys = (acc & np.uint64(_KEY)).astype(np.int64)
            order = np.argsort(keys, kind="stable")
            self._bulk_long = (keys[order], order + self.LONG_ORDER - 1)
        else:
            self._hash = self._rolled(n)

    def _rolled(self, n: int) -> int:
        h = 0
        for token in self._hist[max(0, n - self.LONG_ORDER) : n]:
            h = (h * _BASE + token) & _U64
        return h

    def _candidates(self, live, bulk, key: int) -> List[int]:
        """The key's WAYS most recent end positions, newest first."""
        found = live.get(key)
        ends = found[::-1] if found else []
        need = self.WAYS - len(ends)
        if bulk is not None and need > 0:
            keys, positions = bulk
            hi = int(keys.searchsorted(key, "right"))
            lo = max(0, hi - need)
            if hi > lo:
                run = keys[lo:hi].tolist()
                at = positions[lo:hi].tolist()
                for i in range(len(run) - 1, -1, -1):
                    if run[i] != key:
                        break
                    ends.append(at[i])
        return ends

    def propose(self, max_k: int = 8) -> Tuple[List[int], int]:
        """Up to ``max_k`` next tokens and the length of the match behind them.

        ``([], 0)`` when no earlier occurrence matches ``min_match`` tokens.
        """
        self.last_match = 0
        hist = self._hist
        n = len(hist)
        if n < self.ORDER + 1 or max_k <= 0:
            return [], 0
        cur = n - 1
        key = (
            ((hist[cur - 2] & _TOKEN_MASK) << (2 * _TOKEN_BITS))
            | ((hist[cur - 1] & _TOKEN_MASK) << _TOKEN_BITS)
            | (hist[cur] & _TOKEN_MASK)
        )
        # The long key first: on equal lengths its candidates win.
        candidates = (
            self._candidates(self._long, self._bulk_long, self._hash & _KEY)
            if n >= self.LONG_ORDER
            else []
        )
        candidates += self._candidates(self._short, self._bulk_short, key)
        best_end, best_len = -1, 0
        max_match = self.max_match
        for p in candidates:
            if p >= cur:
                continue  # the current suffix itself
            cap = max_match if p + 1 >= max_match else p + 1
            if cap <= best_len:
                continue
            if hist[p - cap + 1 : p + 1] == hist[cur - cap + 1 : n]:
                length = cap
            else:
                length = 0
                while hist[p - length] == hist[cur - length]:
                    length += 1
            if length > best_len:
                best_end, best_len = p, length
                if length == max_match:
                    break
        if best_len < self.min_match:
            return [], 0
        self.last_match = best_len
        # The continuation may run into the current suffix (periodic text).
        out = hist[best_end + 1 : best_end + 1 + max_k].tolist()
        if self.periodic:
            period = cur - best_end
            while len(out) < max_k:
                out.append(out[-period])
        return out, best_len
