# SPDX-License-Identifier: Apache-2.0
"""Self-speculative n-gram draft source (prompt/history lookup).

Drafts are produced by string matching over the request's own token
history — the prompt plus everything generated so far — with no model
forward involved. When a recent suffix has appeared before, the tokens
that followed that earlier occurrence are proposed as the draft and
verified by the target model in one wide forward. Verification makes
the scheme lossless regardless of draft quality.

v3 upgrades (plan 2026-08-27-ngram-v3-aprimoramentos):
- cascading windows: the lookup tries a long suffix first (high
  precision), then shorter ones (recall); the copy length is capped by
  the window that matched, so weak 8-token matches cannot request the
  widest — and most expensive — verify.
- per-key freeze: a key whose copies died twice below draft_min stops
  volunteering, while productive keys keep serving. Replaces the v2
  global cooloff, which silenced the whole source over one bad region.

v4 upgrade (plan 2026-08-27-ngram-v4-frequencia-e-sistema): frequency
rule. ``extend`` counts how many times each gram occurred; with
``freq_rule=True`` the copy cap follows that count — a suffix repeated
3+ times buys the full draft_max even through a short window, one seen
a single time is halved. SuffixDecoding's per-step scoring, without the
suffix tree. Off (the default) reproduces v3 byte for byte; the freeze
memory stays a separate mechanism (a brake, never a length signal).

Not to be confused with Qwen4-Exp's PLE "n-gram embedding", a trained
table inside the model. This module never touches weights.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# llama.cpp's ngram-mod lookup length — the F1.3 candidate the A/B flips
# against the v3 default of 16. match_len=24 yields windows (36, 24, 12).
NGRAM_MOD_MATCH_LEN = 24


class NGramDraftSource:
    """Cascading hash lookup of recent suffixes in the token history.

    ``match_len`` seeds the cascade: windows are ``(3n/2, n, n/2)``
    deduped — the default 16 yields (24, 16, 8), llama.cpp's documented
    operating point for MoE models. ``extend`` registers, per window,
    the gram ending at each new position; ``lookup`` walks windows long
    to short and returns the continuation after the most recent earlier
    occurrence, or ``None``.
    """

    def __init__(
        self,
        match_len: int = 16,
        draft_max: int = 16,
        draft_min: int = 4,
        freq_rule: bool = False,
    ):
        if match_len < 2:
            raise ValueError("match_len must be >= 2")
        if draft_min < 1 or draft_max < draft_min:
            raise ValueError("need 1 <= draft_min <= draft_max")
        n = int(match_len)
        self.windows: Tuple[int, ...] = tuple(
            sorted({n * 3 // 2, n, max(2, n // 2)}, reverse=True)
        )
        self.draft_max = int(draft_max)
        self.draft_min = int(draft_min)
        self.freq_rule = bool(freq_rule)
        self._tokens: List[int] = []
        # per window size: gram-ending-at-p -> p, plus the previous holder
        # so the suffix's own registration never shadows what lookup needs
        self._last: Dict[int, dict] = {n: {} for n in self.windows}
        self._prev: Dict[int, dict] = {n: {} for n in self.windows}
        # per window size: gram -> occurrence count over the whole history.
        # Rebuilt histories (the pool's trim) re-derive it by re-extending,
        # so the count always reflects the tokens actually held.
        self._count: Dict[int, Dict[tuple, int]] = {n: {} for n in self.windows}
        # per-key acceptance memory: key -> consecutive sub-min failures;
        # at _FREEZE_AT the key stops volunteering (thawed by one success)
        self._fails: Dict[tuple, int] = {}
        self._last_key: Optional[tuple] = None
        self._last_region: Tuple[tuple, ...] = ()
        # counters for the stats line / admin telemetry
        self.hits = 0
        self.misses = 0
        self.drafted_tokens = 0
        self.accepted_tokens = 0
        self.frozen_keys = 0

    _FREEZE_AT = 2
    # freq_rule: this many earlier occurrences of the suffix buy draft_max
    _STRONG_AT = 3

    def __len__(self) -> int:
        return len(self._tokens)

    def extend(self, tokens) -> None:
        toks = self._tokens
        for t in tokens:
            toks.append(int(t))
            p = len(toks) - 1
            for n in self.windows:
                if p >= n - 1:
                    key = tuple(toks[p - n + 1 : p + 1])
                    last = self._last[n]
                    old = last.get(key)
                    if old is not None:
                        self._prev[n][key] = old
                    last[key] = p
                    cnt = self._count[n]
                    cnt[key] = cnt.get(key, 0) + 1

    def _copy_cap(self, window: int, reps: Optional[int] = None) -> int:
        # A short-window match is weaker evidence: cap its copy at the
        # window size so it cannot buy the widest verify forward.
        if window >= self.windows[len(self.windows) // 2]:
            base = self.draft_max
        else:
            base = min(self.draft_max, max(self.draft_min, window))
        if not self.freq_rule or reps is None:
            return base
        # Frequency rule (v4): boost-only. Strong repetition buys the full
        # draft even through a short window; weak evidence keeps the v3 cap.
        # The first cut also HALVED single-sighting copies and lost 3.4% on
        # the rewrite A/B (freq_on 25.84 vs freq_off 26.74) — never punish.
        if reps >= self._STRONG_AT:
            return self.draft_max
        return base

    def _reps(self, window: int, key: tuple) -> int:
        # Earlier occurrences of the gram, excluding the registration of
        # the suffix now being looked up (extend already counted it).
        return max(0, self._count[window].get(key, 0) - 1)

    def lookup(self) -> Optional[List[int]]:
        toks = self._tokens
        end = len(toks) - 1
        for n in self.windows:
            if len(toks) < n:
                continue
            key = tuple(toks[end - n + 1 :])
            if self._fails.get(key, 0) >= self._FREEZE_AT:
                continue
            p = self._last[n].get(key)
            if p == end:
                p = self._prev[n].get(key)
            if p is None:
                continue
            cont = toks[p + 1 : p + 1 + self._copy_cap(n, self._reps(n, key))]
            if len(cont) < self.draft_min:
                continue
            self.hits += 1
            self.drafted_tokens += len(cont)
            self._last_key = key
            # Freezing must silence the REGION, not just the serving key:
            # otherwise the cascade re-offers the same bad copy through a
            # shorter window. Remember every window's key for this suffix.
            self._last_region = tuple(
                tuple(toks[end - w + 1 :]) for w in self.windows if len(toks) >= w
            )
            return cont
        self.misses += 1
        self._last_key = None
        return None

    def feedback(self, accepted: int) -> None:
        """Report how many tokens of the last served copy were accepted."""
        self.accepted_tokens += int(accepted)
        if self._last_key is None:
            return
        region = getattr(self, "_last_region", (self._last_key,))
        if accepted < self.draft_min:
            for key in region:
                fails = self._fails.get(key, 0) + 1
                self._fails[key] = fails
                if fails == self._FREEZE_AT:
                    self.frozen_keys += 1
        else:
            for key in region:
                if key in self._fails:
                    if self._fails[key] >= self._FREEZE_AT:
                        self.frozen_keys -= 1
                    del self._fails[key]


class SharedNGramPool:
    """Cross-request n-gram pool (plan v3, F4.1) — llama.cpp's ngram-mod idea.

    One lock-guarded history shared by every request of the process: an
    agent loop that re-sends the same file gets its copies from the first
    request onward. Feeds from different requests are separated by a
    negative FENCE token, which can never equal a real token id — keys
    spanning a fence never match a real suffix, and continuations are cut
    at the fence, so no copy ever splices two unrelated requests.
    """

    FENCE = -1

    def __init__(self, match_len: int = 16, draft_max: int = 16,
                 draft_min: int = 4, freq_rule: bool = False,
                 max_tokens: int = 65536):
        # ponytail: global lock + full rebuild on trim; per-shard locks and
        # incremental eviction only if contention ever shows up in profiles
        import threading

        self._lock = threading.Lock()
        self._args = (match_len, draft_max, draft_min, freq_rule)
        self._src = NGramDraftSource(*self._args)
        self.max_tokens = int(max_tokens)

    def feed(self, tokens, new_segment: bool = True) -> None:
        """Append tokens; ``new_segment`` fences off the previous request.

        A request feeds its prompt with the fence and its incremental
        commits without it, so its own text stays contiguous while text
        from different requests can never splice.
        """
        with self._lock:
            src = self._src
            if new_segment and len(src) > 0:
                src.extend([self.FENCE])
            src.extend(tokens)
            if len(src) > self.max_tokens:
                tail = src._tokens[-self.max_tokens // 2 :]
                fresh = NGramDraftSource(*self._args)
                fresh.extend(tail)
                self._src = fresh

    def lookup_suffix(self, suffix) -> "Optional[List[int]]":
        with self._lock:
            src = self._src
            toks = src._tokens
            for n in src.windows:
                if len(suffix) < n or len(toks) < n:
                    continue
                key = tuple(int(t) for t in suffix[-n:])
                p = src._last[n].get(key)
                if p is None or p == len(toks) - 1:
                    p = src._prev[n].get(key)
                if p is None:
                    continue
                # Same frequency rule as the per-request path (F1.5). The
                # querying request usually feeds the pool too, so its own
                # registration is excluded the same way.
                cont = toks[p + 1 : p + 1 + src._copy_cap(n, src._reps(n, key))]
                if self.FENCE in cont:
                    cont = cont[: cont.index(self.FENCE)]
                if len(cont) < src.draft_min:
                    continue
                src.hits += 1
                src.drafted_tokens += len(cont)
                return cont
            src.misses += 1
            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._src)
