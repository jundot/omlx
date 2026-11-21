"""Suffix-lookup drafter: the matching rule, its index layouts, and its cost."""

from __future__ import annotations

import random
import time

from omlx.speculative.suffix_drafter import SuffixDrafter


def _reference(hist, max_k, min_match=3, max_match=32, ways=4, long_order=8):
    """The rule, by scanning: each key's ``ways`` most recent ends (the current
    suffix's own among them), the long key first, longest match, newest on ties."""
    n = len(hist)
    if n < 4:
        return [], 0
    cur = n - 1
    best_end, best_len = -1, 0
    for order in (long_order, 3):
        if n < order:
            continue
        key = hist[n - order :]
        ends = [e for e in range(order - 1, n) if hist[e - order + 1 : e + 1] == key]
        for p in reversed(ends[-ways:]):
            if p >= cur:
                continue
            length = 0
            while (
                length < min(max_match, p + 1)
                and hist[p - length] == hist[cur - length]
            ):
                length += 1
            if length > best_len:
                best_end, best_len = p, length
    if best_len < min_match:
        return [], 0
    out = hist[best_end + 1 : best_end + 1 + max_k]
    while len(out) < max_k:  # an overlapping copy: the period goes on
        out.append(out[best_end - cur])
    return out, best_len


def test_proposals_follow_the_rule_through_bulk_and_single_appends():
    """Prompts indexed in bulk, tokens appended one by one, and quoted stretches
    give the proposals of the scanned rule, for alphabets from 3 tokens (every
    slot overflows) to 40."""
    rng = random.Random(7)
    proposed = 0
    for _ in range(120):
        vocab = rng.choice([3, 6, 40])
        hist = [rng.randrange(vocab) for _ in range(rng.choice([0, 5, 300, 700]))]
        drafter = SuffixDrafter()
        drafter.append(hist)
        for _ in range(40):
            add = [rng.randrange(vocab) for _ in range(rng.choice([1, 1, 2, 5]))]
            if rng.random() < 0.3 and len(hist) > 40:
                start = rng.randrange(len(hist) - 20)
                add = hist[start : start + rng.randrange(3, 20)]
            hist += add
            drafter.append(add)
            tokens, match = drafter.propose(8)
            assert (tokens, match) == _reference(hist, 8), len(hist)
            proposed += bool(tokens)
    assert proposed > 1000


def test_long_key_keeps_a_source_the_trigram_slots_lost():
    """Five later repeats of a trigram push the quoted stretch out of the
    trigram's four slots; the 8-gram still finds it and the whole stretch."""
    source = list(range(100, 120))
    noise = []
    for i in range(5):
        noise += [900 + i, 105, 106, 107, 950 + i]
    drafter = SuffixDrafter()
    drafter.append(source + noise + source[:8])
    tokens, match = drafter.propose(6)
    assert tokens == source[8:14] and match == 8

    trigram_only = SuffixDrafter()
    trigram_only.append(source + noise + [1, 2] + source[5:8])
    tokens, match = trigram_only.propose(6)
    # The newest repeat, behind a 3-token match, not the source.
    assert match == 3 and tokens[0] == 954


def test_match_bounds_and_periodic_continuation():
    drafter = SuffixDrafter(min_match=4)
    drafter.append([1, 2, 3, 9, 1, 2, 3])
    assert drafter.propose(8) == ([], 0)  # a 3-token match, below min_match
    drafter = SuffixDrafter()
    drafter.append([5, 6, 7] * 30)
    tokens, match = drafter.propose(8)
    # The continuation runs through the current suffix and on through its own
    # period; the match is capped.
    assert tokens == [5, 6, 7, 5, 6, 7, 5, 6] and match == 32
    assert drafter.propose(0) == ([], 0)
    basalt = SuffixDrafter(periodic=False)
    basalt.append([5, 6, 7] * 30)
    assert basalt.propose(8) == ([5, 6, 7], 32)


def test_reset_and_sync_leave_nothing_of_the_old_request():
    first = [3, 1, 4, 1, 5, 9, 2, 6] * 60
    drafter = SuffixDrafter()
    assert drafter.sync(first) is True
    assert drafter.propose(4)[0]
    assert drafter.sync(first + [3, 1]) is False and len(drafter) == len(first) + 2
    # Another request: nothing of the first one is proposed or kept.
    second = [7, 7, 8]
    assert drafter.sync(second) is True
    assert len(drafter) == 3 and drafter.propose(4) == ([], 0)
    drafter.append([3, 1, 4, 1])
    assert drafter.propose(4) == ([], 0)
    drafter.reset()
    assert len(drafter) == 0 and not drafter._short and not drafter._long
    assert drafter._bulk_short is None and drafter._bulk_long is None


def test_lookup_time_does_not_grow_with_the_history():
    """A proposal on a 32K-token history costs what one on 1K does, far under
    0.1 ms (medians: a loaded runner moves single samples, not these)."""
    rng = random.Random(11)

    def median_us(n):
        record = [rng.randrange(50_000) for _ in range(40)]
        hist = []
        while len(hist) < n:
            hist += record[:20] + [rng.randrange(1000) for _ in range(4)] + record[20:]
        drafter = SuffixDrafter()
        drafter.append(hist[:n])
        samples = []
        for token in hist[1000:1400]:
            drafter.append([token])
            started = time.perf_counter()
            drafter.propose(8)
            samples.append((time.perf_counter() - started) * 1e6)
        return sorted(samples)[len(samples) // 2]

    small, large = median_us(1024), median_us(32768)
    assert large < 100.0
    assert large < 4.0 * small + 5.0
