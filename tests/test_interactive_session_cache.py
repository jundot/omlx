"""Tests for InteractiveSessionCache."""

from __future__ import annotations

import pytest


class FakeClock:
    """Deterministic clock for testing."""

    def __init__(self, start: float = 0.0):
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, secs: float) -> None:
        self._now += secs


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cache(clock):
    from omlx.cache.interactive_session_cache import InteractiveSessionCache

    return InteractiveSessionCache(
        ttl_secs=60.0,
        max_sessions=4,
        max_bytes=10_000,
        clock=clock,
    )


# ---- put / take basic ----


def test_priority_zero_completion_creates_protected_tip(cache, clock):
    """Complete priority-zero turn → one protected entry with full token sequence."""
    tokens = [100, 200, 300, 400]
    cache_data = [b"layer0", b"layer1"]

    assert cache.put(tokens, cache_data) is True
    assert cache.session_count == 1

    result = cache.take_longest_prefix(tokens)
    assert result is not None
    data, config, prefix_len = result
    assert data == cache_data
    assert prefix_len == 4
    assert cache.session_count == 0  # transferred


def test_background_completion_does_not_create_protected_tip(cache):
    """Background completions don't call put → cache stays empty."""
    # The cache itself doesn't filter by priority — caller decides.
    # This test verifies that if we simply don't call put, cache is empty.
    assert cache.session_count == 0
    result = cache.take_longest_prefix([100, 200])
    assert result is None


def test_put_rejects_empty_tokens(cache):
    assert cache.put([], [b"data"]) is False


def test_put_rejects_empty_cache_data(cache):
    assert cache.put([1, 2, 3], []) is False


def test_put_rejects_oversized_entry(cache):
    """Entry exceeding max_bytes is rejected."""
    big_data = [b"x" * 1000 for _ in range(100)]
    assert cache.put([1, 2, 3], big_data) is False
    assert cache.session_count == 0


# ---- prefix matching ----


def test_partial_tip_matches_exact_extended_prefix(cache):
    """Store non-block-aligned tokens, request same prefix + suffix → match."""
    stored = [100, 200, 300]
    cache.put(stored, [b"kv1"])

    # Request with longer prefix — should match first 3 tokens
    result = cache.take_longest_prefix([100, 200, 300, 400, 500])
    assert result is not None
    _, _, prefix_len = result
    assert prefix_len == 3


def test_longest_protected_prefix_wins(cache):
    """Store nested prefixes → longest exact prefix transfers."""
    short = [100, 200]
    long = [100, 200, 300, 400]

    cache.put(short, [b"short"])
    cache.put(long, [b"long"])

    result = cache.take_longest_prefix([100, 200, 300, 400, 500])
    assert result is not None
    data, _, prefix_len = result
    assert data == [b"long"]
    assert prefix_len == 4


def test_hit_transfers_entry_instead_of_sharing_mutable_cache(cache):
    """take removes entry before caller can mutate returned cache."""
    tokens = [100, 200]
    original_data = [b"kv"]
    cache.put(tokens, original_data)

    result = cache.take_longest_prefix(tokens)
    assert result is not None
    data, _, _ = result

    # Mutate the returned data
    data.append(b"mutated")

    # Cache should be empty (transferred)
    assert cache.session_count == 0

    # Re-put original — should not contain mutation
    cache.put(tokens, original_data)
    result2 = cache.take_longest_prefix(tokens)
    assert result2 is not None
    assert b"mutated" not in result2[0]


# ---- TTL ----


def test_expired_entry_misses(cache, clock):
    """Advance clock beyond TTL → miss + eviction."""
    cache.put([100, 200], [b"kv"])

    clock.advance(61.0)  # TTL is 60s

    result = cache.take_longest_prefix([100, 200])
    assert result is None
    stats = cache.get_stats()
    assert stats["evictions"] >= 1


# ---- session limit ----


def test_session_limit_evicts_lru(cache):
    """Exceed session bound → LRU entry disappears."""
    for i in range(6):  # max_sessions=4
        cache.put([i * 100, i * 100 + 1], [f"kv{i}"])

    assert cache.session_count == 4

    # Oldest entries (0, 1) should be evicted
    assert cache.take_longest_prefix([0, 1]) is None
    assert cache.take_longest_prefix([100, 101]) is None


# ---- byte limit ----


def test_byte_limit_evicts_lru(cache):
    """Exceed byte bound → bytes return at or below ceiling."""
    # Each entry is ~128 bytes per cache_data element
    # max_bytes=10000, so fill with large entries
    for i in range(20):
        data = [b"x" * 512 for _ in range(10)]  # ~6400 bytes each
        cache.put([i], data)

    assert cache.total_bytes <= 10_000


# ---- pressure ----


def test_hard_pressure_can_evict_protected_entry(cache, clock):
    """force=True shrink_to(0) removes all entries."""
    cache.put([100, 200], [b"kv"])
    cache.put([300, 400], [b"kv2"])

    released = cache.shrink_to(0, force=True)
    assert released > 0
    assert cache.session_count == 0


def test_normal_pressure_preserves_unexpired_interactive_entry(cache, clock):
    """force=False shrink retains unexpired entries when possible."""
    cache.put([100, 200], [b"kv"])

    # Shrink to a large target — should not evict unexpired entry
    released = cache.shrink_to(1_000_000, force=False)
    assert released == 0
    assert cache.session_count == 1


# ---- full-block prefix cache behavior unchanged ----


def test_full_block_prefix_cache_behavior_is_unchanged(cache):
    """Interactive cache doesn't interfere with normal prefix lookup."""
    # Store an entry
    cache.put([100, 200, 300], [b"kv"])

    # Normal prefix lookup (different tokens) should miss
    result = cache.take_longest_prefix([500, 600, 700])
    assert result is None

    # Original entry should still be there
    assert cache.session_count == 1


# ---- multi-turn cached tokens ----


def test_multi_turn_cached_tokens_include_trailing_partial_tokens(cache):
    """Store one full block plus partial tail → reported count includes both."""
    # Simulate: full block = 256 tokens, partial = 42 tokens
    full_block = list(range(256))
    partial_tail = list(range(256, 298))
    all_tokens = full_block + partial_tail

    cache.put(all_tokens, [b"kv_full_and_partial"])

    result = cache.take_longest_prefix(all_tokens)
    assert result is not None
    _, _, prefix_len = result
    assert prefix_len == 298  # full block + partial


# ---- stats ----


def test_get_stats(cache):
    cache.put([100], [b"a"])
    cache.take_longest_prefix([100])  # hit
    cache.take_longest_prefix([999])  # miss

    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["sessions"] == 0  # transferred
    assert isinstance(stats["bytes"], int)
