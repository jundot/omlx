# SPDX-License-Identifier: Apache-2.0
"""
Protected Multi-Turn Interactive Cache for oMLX.

Stores trailing KV state from priority-zero completions for exact-prefix
transfer on the next turn. Bounded by TTL, session count, and byte limit.
"""

from __future__ import annotations

import copy
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InteractiveCacheEntry:
    """A single cached trailing KV state from a priority-zero completion."""

    token_ids: tuple[int, ...]
    cache_data: list[Any]
    model_cache_config: Any | None
    size_bytes: int
    expires_at: float
    last_access: float


class InteractiveSessionCache:
    """Bounded LRU cache for interactive multi-turn trailing KV state.

    Only priority-zero completions store entries. Entries are keyed by
    exact token prefix. On hit, the entry is *transferred* (removed before
    return) so the caller owns the cache data exclusively.

    Bounds:
        ttl_secs: entries expire after this duration
        max_sessions: maximum number of entries
        max_bytes: maximum total size in bytes
    """

    def __init__(
        self,
        ttl_secs: float = 600.0,
        max_sessions: int = 64,
        max_bytes: int = 2 * 1024 * 1024 * 1024,  # 2 GB
        clock: Any = time.monotonic,
    ):
        self._entries: OrderedDict[tuple[int, ...], InteractiveCacheEntry] = OrderedDict()
        self._ttl = ttl_secs
        self._max_sessions = max_sessions
        self._max_bytes = max_bytes
        self._clock = clock
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_bytes = 0

    def put(
        self,
        token_ids: list[int],
        cache_data: list[Any],
        model_cache_config: Any | None = None,
    ) -> bool:
        """Store a trailing KV state from a priority-zero completion.

        Returns True if stored, False if rejected (empty/oversized).
        """
        if not token_ids or not cache_data:
            return False

        key = tuple(token_ids)

        # Estimate size — use len(cache_data) as proxy if no size_bytes available
        size_bytes = self._estimate_size(cache_data)
        if size_bytes > self._max_bytes:
            logger.warning(
                f"Interactive cache: entry too large ({size_bytes} bytes), rejecting"
            )
            return False

        now = self._clock()
        entry = InteractiveCacheEntry(
            token_ids=key,
            cache_data=cache_data,
            model_cache_config=model_cache_config,
            size_bytes=size_bytes,
            expires_at=now + self._ttl,
            last_access=now,
        )

        # Replace if same key
        if key in self._entries:
            old = self._entries.pop(key)
            self._total_bytes -= old.size_bytes

        self._entries[key] = entry
        self._total_bytes += size_bytes

        # Evict LRU until within bounds
        self._evict_lru()

        return True

    def take_longest_prefix(
        self, token_ids: list[int]
    ) -> tuple[list[Any], Any | None, int] | None:
        """Find and take the longest exact-prefix entry.

        Returns (cache_data, model_cache_config, prefix_len) or None on miss.
        The entry is removed before return (transfer semantics).
        """
        self._evict_expired()

        key = tuple(token_ids)
        best_entry: InteractiveCacheEntry | None = None
        best_len = 0

        # Check all entries for prefix match
        for entry_key, entry in self._entries.items():
            elen = len(entry_key)
            if elen <= best_len:
                continue
            if elen > len(key):
                continue
            if key[:elen] == entry_key:
                best_entry = entry
                best_len = elen

        if best_entry is None:
            self._misses += 1
            return None

        # Transfer: remove before returning, deep-copy for exclusive ownership
        self._entries.pop(best_entry.token_ids)
        self._total_bytes -= best_entry.size_bytes
        self._hits += 1

        return copy.deepcopy(best_entry.cache_data), best_entry.model_cache_config, best_len

    def evict_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        return self._evict_expired()

    def _evict_expired(self) -> int:
        now = self._clock()
        expired = [k for k, v in self._entries.items() if v.expires_at <= now]
        for k in expired:
            entry = self._entries.pop(k)
            self._total_bytes -= entry.size_bytes
            self._evictions += 1
        return len(expired)

    def shrink_to(self, target_bytes: int, force: bool = False) -> int:
        """Evict LRU entries until byte usage reaches target.

        When force=False, retains unexpired protected entries when possible.
        Returns bytes released.
        """
        released = 0
        while self._total_bytes > target_bytes and self._entries:
            # Find LRU entry
            oldest_key = next(iter(self._entries))
            entry = self._entries[oldest_key]

            # Don't evict unexpired entries in non-force mode
            if not force and entry.expires_at > self._clock():
                break

            self._entries.pop(oldest_key)
            self._total_bytes -= entry.size_bytes
            released += entry.size_bytes
            self._evictions += 1

        return released

    def clear(self) -> int:
        """Remove all entries. Returns bytes released."""
        released = self._total_bytes
        count = len(self._entries)
        self._entries.clear()
        self._total_bytes = 0
        self._evictions += count
        return released

    def get_stats(self) -> dict[str, int | float]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "bytes": self._total_bytes,
            "sessions": len(self._entries),
        }

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def session_count(self) -> int:
        return len(self._entries)

    def _evict_lru(self) -> None:
        """Evict LRU entries until within bounds."""
        while len(self._entries) > self._max_sessions:
            oldest_key = next(iter(self._entries))
            entry = self._entries.pop(oldest_key)
            self._total_bytes -= entry.size_bytes
            self._evictions += 1

        while self._total_bytes > self._max_bytes and self._entries:
            oldest_key = next(iter(self._entries))
            entry = self._entries.pop(oldest_key)
            self._total_bytes -= entry.size_bytes
            self._evictions += 1

    def _estimate_size(self, cache_data: list[Any]) -> int:
        """Estimate byte size of cache data."""
        # Each entry is roughly 128 bytes per layer * num_layers
        # Use a conservative estimate based on list length
        return len(cache_data) * 128
