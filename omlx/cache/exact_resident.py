# SPDX-License-Identifier: Apache-2.0
"""Bounded, exact-token resident prompt-cache handoff.

The paged prefix cache is the durable/general fallback.  This small tier keeps
the most recently detached *live* cache object so a following chat turn can
take ownership without serializing, loading, or concatenating its blocks.

Correctness is deliberately simple: an entry is reusable only when every
stored token is an exact prefix of the new scheduler-owned prompt.  Entries
are removed on acquisition, so a mutable cache object is never shared by two
requests.  Callers remain responsible for validating model/cache offsets and
for excluding media-keyed requests.
"""

from __future__ import annotations

import threading
from array import array
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExactResidentHit:
    """Exclusive ownership transfer returned by :meth:`acquire_prefix`."""

    cache: list[Any]
    cached_tokens: int
    cache_nbytes: int
    durable_tokens: int
    terminal_proof: str | None = None


@dataclass
class _ExactResidentEntry:
    tokens: array
    cache: list[Any]
    cache_nbytes: int
    durable_tokens: int
    terminal_proof: str | None


class ExactResidentPrefixCache:
    """A tiny LRU of exact terminal cache objects.

    The scheduler never hands a retained entry to the asynchronous durable
    writer.  It is therefore claimable immediately after the response, with
    no shared-array reader race. ``durable_tokens`` records the independently
    published paged/SSD prompt boundary that remains the crash, eviction, and
    concurrent-claim fallback for that mutable terminal state.
    """

    def __init__(
        self,
        max_entries: int = 1,
        max_bytes: int = 8 * 1024**3,
    ) -> None:
        self.max_entries = max(0, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self._entries: OrderedDict[int, _ExactResidentEntry] = OrderedDict()
        self._size_bytes = 0
        self._next_id = 0
        self._generation = 0
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.oversize_rejections = 0
        self.protected_rejections = 0

    @staticmethod
    def _tokens_equal_prefix(stored: array, prompt: list[int]) -> bool:
        if len(stored) >= len(prompt):
            # A non-empty suffix is required.  Generation from cache state at
            # exactly N tokens needs an N-1 trim/kickoff, which recurrent cache
            # families cannot perform generically.
            return False
        return all(saved == current for saved, current in zip(stored, prompt))

    @staticmethod
    def _stored_extends_candidate(stored: array, candidate: array) -> bool:
        return bool(
            len(stored) > len(candidate)
            and all(saved == current for saved, current in zip(stored, candidate))
        )

    def _newest_extending_entry(
        self,
        candidate: array,
    ) -> tuple[int, _ExactResidentEntry] | None:
        """Return the newest terminal extending a shared stable boundary.

        A current turn can leave multiple terminal branches that all extend the
        same coarse boundary. Only the newest is the active conversation; older
        branches remain recoverable through the durable tier and are LRU
        victims. OrderedDict insertion order makes recency unambiguous, so
        terminal length is deliberately not used to override it.
        """

        for entry_id, entry in reversed(self._entries.items()):
            if self._stored_extends_candidate(entry.tokens, candidate):
                return entry_id, entry
        return None

    def can_fit_protected_candidate(
        self,
        tokens: Iterable[int],
        *,
        estimated_cache_nbytes: int,
    ) -> bool:
        """Preflight a fallback without sacrificing the current terminal.

        Unrelated and older terminal entries are evictable. The newest entry
        extending this exact candidate is protected, and its measured bytes
        calibrate a conservative same-cache-family estimate for the fallback.
        """

        try:
            token_array = array("I", [int(token) for token in tokens])
        except (OverflowError, TypeError, ValueError):
            return False
        if not token_array or token_array.itemsize != 4:
            return False
        estimate = max(0, int(estimated_cache_nbytes))
        with self._lock:
            protected = self._newest_extending_entry(token_array)
            if protected is None:
                return True
            _, protected_entry = protected
            if len(protected_entry.tokens) > 0:
                estimate = max(
                    estimate,
                    (
                        protected_entry.cache_nbytes * len(token_array)
                        + len(protected_entry.tokens)
                        - 1
                    )
                    // len(protected_entry.tokens),
                )
            return bool(
                self.max_entries >= 2
                and protected_entry.cache_nbytes + estimate <= self.max_bytes
            )

    def put(
        self,
        tokens: Iterable[int],
        cache: list[Any],
        *,
        cache_nbytes: int = 0,
        durable_tokens: int = 0,
        protect_longer_prefix: bool = False,
        expected_generation: int | None = None,
        terminal_proof: str | None = None,
    ) -> bool:
        """Retain one detached cache, evicting oldest entries as needed."""

        if self.max_entries <= 0 or not isinstance(cache, list) or not cache:
            return False
        token_values = [int(token) for token in tokens]
        if any(token < 0 or token > 0xFFFFFFFF for token in token_values):
            return False
        token_array = array("I", token_values)
        if token_array.itemsize != 4:
            # The durable/on-wire scheduler token contract is uint32.  Refuse
            # a platform whose native unsigned-int array is not 32 bits.
            return False
        if not token_array:
            return False
        cache_nbytes = max(0, int(cache_nbytes))
        durable_tokens = int(durable_tokens)
        if durable_tokens < 0 or durable_tokens > len(token_array):
            return False
        if self.max_bytes <= 0 or cache_nbytes > self.max_bytes:
            with self._lock:
                self.oversize_rejections += 1
            return False
        entry = _ExactResidentEntry(
            tokens=token_array,
            cache=cache,
            cache_nbytes=cache_nbytes,
            durable_tokens=durable_tokens,
            terminal_proof=(
                str(terminal_proof) if terminal_proof is not None else None
            ),
        )
        with self._lock:
            if (
                expected_generation is not None
                and int(expected_generation) != self._generation
            ):
                return False
            if protect_longer_prefix:
                protected = self._newest_extending_entry(token_array)
                protected_ids = {protected[0]} if protected is not None else set()
                if protected_ids:
                    prospective_count = len(self._entries) + 1
                    prospective_bytes = self._size_bytes + cache_nbytes
                    victims: list[int] = []
                    for entry_id, existing in self._entries.items():
                        if (
                            prospective_count <= self.max_entries
                            and prospective_bytes <= self.max_bytes
                        ):
                            break
                        if entry_id in protected_ids:
                            continue
                        victims.append(entry_id)
                        prospective_count -= 1
                        prospective_bytes -= existing.cache_nbytes
                    if (
                        prospective_count > self.max_entries
                        or prospective_bytes > self.max_bytes
                    ):
                        # Prompt-tail prewarm is a fallback for transcript
                        # divergence, never a reason to evict the newest
                        # extending terminal state for the active conversation.
                        self.protected_rejections += 1
                        return False
                    for entry_id in victims:
                        evicted = self._entries.pop(entry_id)
                        self._size_bytes -= evicted.cache_nbytes
                        self.evictions += 1
            self._next_id += 1
            self._entries[self._next_id] = entry
            self._size_bytes += cache_nbytes
            while (
                len(self._entries) > self.max_entries
                or self._size_bytes > self.max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._size_bytes -= evicted.cache_nbytes
                self.evictions += 1
        return True

    def generation(self) -> int:
        """Return the lifecycle generation used by deferred publishers."""

        with self._lock:
            return self._generation

    def contains_exact(self, tokens: Iterable[int]) -> bool:
        """Whether an independently owned entry has this exact token ledger."""

        try:
            token_array = array("I", [int(token) for token in tokens])
        except (OverflowError, TypeError, ValueError):
            return False
        with self._lock:
            return any(entry.tokens == token_array for entry in self._entries.values())

    def contains_prefix(
        self,
        tokens: Iterable[int],
        *,
        minimum_tokens: int = 1,
    ) -> bool:
        """Whether a sufficiently long resident entry exactly prefixes tokens."""

        try:
            token_array = array("I", [int(token) for token in tokens])
            minimum = max(1, int(minimum_tokens))
        except (OverflowError, TypeError, ValueError):
            return False
        with self._lock:
            return any(
                minimum <= len(entry.tokens) <= len(token_array)
                and entry.tokens == token_array[: len(entry.tokens)]
                for entry in self._entries.values()
            )

    def acquire_prefix(
        self,
        prompt_tokens: list[int],
        *,
        allowed_terminal_proofs: set[str] | None = None,
    ) -> ExactResidentHit | None:
        """Pop the longest ready entry that exactly prefixes ``prompt_tokens``."""

        if self.max_entries <= 0 or not prompt_tokens:
            self.misses += 1
            return None
        with self._lock:
            best_id = None
            best_len = -1
            for entry_id, entry in reversed(self._entries.items()):
                if (
                    allowed_terminal_proofs is not None
                    and entry.terminal_proof not in allowed_terminal_proofs
                ):
                    continue
                if len(entry.tokens) <= best_len:
                    continue
                if self._tokens_equal_prefix(entry.tokens, prompt_tokens):
                    best_id = entry_id
                    best_len = len(entry.tokens)

            if best_id is None:
                self.misses += 1
                return None

            entry = self._entries.pop(best_id)
            self._size_bytes -= entry.cache_nbytes
            self.hits += 1
            return ExactResidentHit(
                cache=entry.cache,
                cached_tokens=len(entry.tokens),
                cache_nbytes=entry.cache_nbytes,
                durable_tokens=entry.durable_tokens,
                terminal_proof=entry.terminal_proof,
            )

    def snapshot_entries(self) -> list[tuple[list[int], list[Any], str | None]]:
        """Return newest-to-oldest entries for lifecycle persistence.

        The returned cache objects remain owned by this tier until the caller
        explicitly clears it.  This is used only during serialized engine
        shutdown, after request/store workers have drained.
        """

        with self._lock:
            return [
                (
                    list(entry.tokens),
                    entry.cache,
                    entry.terminal_proof,
                )
                for entry in reversed(self._entries.values())
            ]

    def clear(self) -> int:
        """Drop resident references without touching the durable cache tier."""

        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._size_bytes = 0
            self._generation += 1
            return count

    def resize(self, max_entries: int) -> int:
        """Apply a live entry limit and evict oldest entries to fit."""

        max_entries = max(0, int(max_entries))
        evicted_count = 0
        with self._lock:
            if max_entries != self.max_entries:
                self._generation += 1
            self.max_entries = max_entries
            while len(self._entries) > self.max_entries:
                _, evicted = self._entries.popitem(last=False)
                self._size_bytes -= evicted.cache_nbytes
                self.evictions += 1
                evicted_count += 1
        return evicted_count

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "size_bytes": self._size_bytes,
                "max_token_count": max(
                    (len(entry.tokens) for entry in self._entries.values()),
                    default=0,
                ),
                "max_durable_token_count": max(
                    (entry.durable_tokens for entry in self._entries.values()),
                    default=0,
                ),
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes if self.max_entries > 0 else 0,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "oversize_rejections": self.oversize_rejections,
                "protected_rejections": self.protected_rejections,
                "generation": self._generation,
            }
