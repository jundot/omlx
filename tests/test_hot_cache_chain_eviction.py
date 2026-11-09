# SPDX-License-Identifier: Apache-2.0
"""Tests for hot_cache_eviction_policy="chain_aware_lru".

Plain LRU evicts a prefix chain's oldest block first: every restore touches
a chain from its root, so the root is always the least recently used entry
of the chain, and prefix matching stops at the first missing block. One
eviction therefore makes the whole rest of the chain unreachable while it
keeps occupying the hot tier. chain_aware_lru only evicts entries no other
hot entry builds on (leaves), unreachable orphans first, and refuses to
admit a block rather than break a chain.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omlx.cache.paged_ssd_cache import PagedSSDCacheManager, SharedHotCacheBudget
from omlx.settings import GlobalSettings

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

SIZE = 100  # bytes per synthetic entry


def _entry(parent: bytes | None = None, size: int = SIZE) -> dict:
    return {
        "tensors_raw": {"layer_0": (b"\0" * size, "uint8", [size])},
        "block_metadata": SimpleNamespace(parent_hash=parent, tail_terminal=False),
        "dirty": False,
    }


def _manager(tmp_path, policy, cap_entries=5, budget=False, name="m"):
    cap = cap_entries * SIZE
    shared = SharedHotCacheBudget(cap, eviction_policy=policy) if budget else None
    mgr = PagedSSDCacheManager(
        cache_dir=tmp_path / f"ssd_{name}",
        max_size_bytes=1 << 20,
        hot_cache_max_bytes=cap,
        hot_cache_only=True,
        hot_cache_budget=shared,
        hot_cache_eviction_policy=policy,
    )
    evicted: list[bytes] = []
    original = mgr._handle_hot_cache_eviction

    def record(block_hash, entry):
        evicted.append(block_hash)
        original(block_hash, entry)

    mgr._handle_hot_cache_eviction = record
    mgr.evicted = evicted
    return mgr


def _put(mgr, key: bytes, parent: bytes | None = None, size: int = SIZE):
    mgr._hot_cache_put(key, _entry(parent, size))


def _chain(prefix: str, n: int, root_parent: bytes | None = None) -> list[tuple[bytes, bytes | None]]:
    out, parent = [], root_parent
    for i in range(n):
        key = f"{prefix}{i}".encode()
        out.append((key, parent))
        parent = key
    return out


@pytest.fixture(params=[False, True], ids=["local", "budget"])
def budget(request):
    return request.param


class TestVictimOrder:
    def _forest(self, mgr):
        # Shared prefix S0 <- S1; chain A: S1 <- A1 <- A2; chain B: S1 <- B1.
        for key, parent in [
            (b"S0", None),
            (b"S1", b"S0"),
            (b"A1", b"S1"),
            (b"A2", b"A1"),
            (b"B1", b"S1"),
        ]:
            _put(mgr, key, parent)

    def test_lru_evicts_oldest_even_if_it_breaks_a_chain(self, tmp_path, budget):
        mgr = _manager(tmp_path, "lru", budget=budget)
        self._forest(mgr)
        _put(mgr, b"X")
        assert mgr.evicted == [b"S0"]  # the shared root: every chain now unreachable

    def test_chain_aware_evicts_lru_leaf(self, tmp_path, budget):
        mgr = _manager(tmp_path, "chain_aware_lru", budget=budget)
        self._forest(mgr)
        _put(mgr, b"X")
        assert mgr.evicted == [b"A2"]  # oldest entry nothing builds on
        _put(mgr, b"Y")
        assert mgr.evicted == [b"A2", b"A1"]  # A1 became a leaf
        _put(mgr, b"Z")
        assert mgr.evicted == [b"A2", b"A1", b"B1"]
        assert {b"S0", b"S1"} <= set(mgr._hot_cache)

    def test_orphans_go_before_older_leaves(self, tmp_path, budget):
        mgr = _manager(tmp_path, "chain_aware_lru", budget=budget)
        _put(mgr, b"Z")  # oldest, a leaf
        _put(mgr, b"P")
        _put(mgr, b"C", b"P")
        mgr._hot_cache_remove(b"P")  # C is now unreachable
        for k in (b"W1", b"W2", b"W3"):
            _put(mgr, k)
        assert mgr.evicted == []
        _put(mgr, b"W4")
        assert mgr.evicted == [b"C"]

    def test_shared_prefix_survives_pressure(self, tmp_path, budget):
        mgr = _manager(tmp_path, "chain_aware_lru", cap_entries=6, budget=budget)
        _put(mgr, b"SYS")
        for key, parent in _chain("A", 3, b"SYS") + _chain("B", 2, b"SYS"):
            _put(mgr, key, parent)
        for key, parent in _chain("C", 12, b"SYS"):
            _put(mgr, key, parent)
            assert b"SYS" in mgr._hot_cache


class TestOverCapacityChain:
    def test_lru_keeps_only_the_unreachable_tail(self, tmp_path, budget):
        mgr = _manager(tmp_path, "lru", cap_entries=4, budget=budget)
        for key, parent in _chain("A", 10):
            _put(mgr, key, parent)
        assert list(mgr._hot_cache) == [b"A6", b"A7", b"A8", b"A9"]
        assert mgr._stats["hot_cache_admission_refusals"] == 0

    def test_chain_aware_keeps_the_prefix_and_refuses_the_rest(self, tmp_path, budget):
        mgr = _manager(tmp_path, "chain_aware_lru", cap_entries=4, budget=budget)
        for key, parent in _chain("A", 10):
            _put(mgr, key, parent)
        assert list(mgr._hot_cache) == [b"A0", b"A1", b"A2", b"A3"]
        assert mgr._stats["hot_cache_admission_refusals"] == 6

    def test_lru_keeps_lone_oversized_entry(self, tmp_path, budget):
        mgr = _manager(tmp_path, "lru", cap_entries=1, budget=budget)
        _put(mgr, b"big", size=5 * SIZE)
        assert b"big" in mgr._hot_cache  # historical behavior, unchanged

    def test_chain_aware_refuses_lone_oversized_entry(self, tmp_path, budget):
        mgr = _manager(tmp_path, "chain_aware_lru", cap_entries=1, budget=budget)
        _put(mgr, b"big", size=5 * SIZE)
        assert b"big" not in mgr._hot_cache
        assert mgr._hot_cache_total_bytes == 0


class TestShrink:
    def test_shrink_respects_chains_and_protection(self, tmp_path, budget):
        mgr = _manager(tmp_path, "chain_aware_lru", cap_entries=10, budget=budget)
        for key, parent in _chain("A", 4):
            _put(mgr, key, parent)
        _put(mgr, b"X")
        mgr.shrink_hot_cache_to(2 * SIZE, protected_hashes={b"X"})
        assert list(mgr._hot_cache) == [b"A0", b"X"]
        assert mgr.evicted == [b"A3", b"A2", b"A1"]


def _check_invariants(mgr):
    entries = mgr._hot_cache
    assert mgr._hot_cache_total_bytes == sum(
        mgr._hot_cache_entry_size(e) for e in entries.values()
    )
    expected_children: dict[bytes, int] = {}
    for e in entries.values():
        p = e["block_metadata"].parent_hash
        if p is not None:
            expected_children[p] = expected_children.get(p, 0) + 1
    assert mgr._hot_children == expected_children
    if mgr._hot_cache_budget is not None:
        assert mgr._hot_cache_budget.total_bytes == mgr._hot_cache_total_bytes


@pytest.mark.parametrize("policy", ["lru", "chain_aware_lru"])
def test_random_operation_sequences(tmp_path, policy, budget):
    """Seeded random put/get/remove/shrink/clear sequences keep all invariants.

    Under chain_aware_lru no eviction may remove a *reachable* entry (its
    whole ancestor chain is hot) that still has a hot child; evicting an
    unreachable subtree is fine. A failure names the seed and step for replay.
    """
    for seed in range(60):
        rng = random.Random(seed)
        mgr = _manager(tmp_path, policy, cap_entries=8, budget=budget, name=f"{seed}")
        violations = []
        original = mgr._handle_hot_cache_eviction

        def reachable(entry, mgr=mgr):
            parent = entry["block_metadata"].parent_hash
            seen = set()
            while parent is not None:
                if parent not in mgr._hot_cache or parent in seen:
                    return False
                seen.add(parent)
                parent = mgr._hot_cache[parent]["block_metadata"].parent_hash
            return True

        def checked(block_hash, entry, mgr=mgr, original=original):
            if (
                policy == "chain_aware_lru"
                and mgr._hot_children.get(block_hash, 0)
                and reachable(entry)
            ):
                violations.append(block_hash)
            original(block_hash, entry)

        mgr._handle_hot_cache_eviction = checked
        keys: list[bytes] = []
        ops = []
        for step in range(150):
            op = rng.choices(["put", "get", "remove", "shrink", "clear"], [60, 25, 8, 5, 2])[0]
            ops.append(op)
            if op == "put":
                key = f"k{step}".encode()
                parent = rng.choice(keys + [None, b"gone"]) if keys else None
                _put(mgr, key, parent, size=rng.choice([SIZE // 2, SIZE, 2 * SIZE]))
                keys.append(key)
            elif op == "get" and keys:
                mgr._hot_cache_get(rng.choice(keys))
            elif op == "remove" and keys:
                mgr._hot_cache_remove(rng.choice(keys))
            elif op == "shrink":
                mgr.shrink_hot_cache_to(rng.randint(0, 8) * SIZE)
            elif op == "clear":
                mgr.clear_hot_cache()
            try:
                _check_invariants(mgr)
            except AssertionError as exc:
                raise AssertionError(f"seed={seed} step={step} ops={ops}") from exc
            assert not violations, f"seed={seed} step={step}: evicted parent {violations}"
            assert mgr._hot_cache_total_bytes <= mgr._effective_hot_cache_max_bytes()
        mgr.close()


def test_budget_and_local_paths_pick_identical_victims(tmp_path):
    rng = random.Random(7)
    ops = []
    keys: list[bytes] = []
    for step in range(200):
        key = f"k{step}".encode()
        parent = rng.choice(keys) if keys and rng.random() < 0.8 else None
        ops.append((key, parent))
        keys.append(key)
    results = []
    for use_budget in (False, True):
        mgr = _manager(tmp_path, "chain_aware_lru", cap_entries=7, budget=use_budget, name=str(use_budget))
        for key, parent in ops:
            _put(mgr, key, parent)
        results.append((mgr.evicted, list(mgr._hot_cache)))
        mgr.close()
    assert results[0] == results[1]


class TestConfig:
    def test_invalid_policy_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            SharedHotCacheBudget(100, eviction_policy="arc")
        with pytest.raises(ValueError):
            PagedSSDCacheManager(
                cache_dir=tmp_path / "x",
                max_size_bytes=1 << 20,
                hot_cache_max_bytes=100,
                hot_cache_only=True,
                hot_cache_eviction_policy="lfu",
            )

    def test_settings_roundtrip_and_validation(self, tmp_path):
        s = GlobalSettings(base_path=tmp_path)
        assert s.cache.hot_cache_eviction_policy == "lru"
        s.cache.hot_cache_eviction_policy = "chain_aware_lru"
        s.save()
        loaded = GlobalSettings.load(base_path=tmp_path)
        assert loaded.cache.hot_cache_eviction_policy == "chain_aware_lru"
        assert not [e for e in loaded.validate() if "eviction_policy" in e]
        loaded.cache.hot_cache_eviction_policy = "mru"
        assert any("hot_cache_eviction_policy" in e for e in loaded.validate())
        assert (
            loaded.to_scheduler_config().hot_cache_eviction_policy == "mru"
        )


# --- end to end with a real prefix cache -------------------------------------

BLOCK_SIZE = 4


class _MockModel:
    def __init__(self, num_layers: int = 2):
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        a = MagicMock()
        a.num_hidden_layers = len(self.layers)
        return a


def _kv_data(seq_len):
    return [
        {
            "state": (mx.ones((1, 2, seq_len, 8)), mx.ones((1, 2, seq_len, 8))),
            "cache_type": "KVCache",
            "class_name": "KVCache",
            "meta_state": (str(seq_len),),
        }
        for _ in range(2)
    ]


def _prefix_cache(tmp_path, policy, hot_bytes):
    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    paged = PagedCacheManager(
        block_size=BLOCK_SIZE, max_blocks=200, model_name="test-model", initial_blocks=200
    )
    ssd = PagedSSDCacheManager(
        cache_dir=tmp_path / f"e2e_{policy}",
        max_size_bytes=100 * 1024**2,
        hot_cache_max_bytes=hot_bytes,
        hot_cache_only=True,
        hot_cache_budget=SharedHotCacheBudget(hot_bytes, eviction_policy=policy),
        hot_cache_eviction_policy=policy,
    )
    cache = BlockAwarePrefixCache(
        model=_MockModel(), paged_cache_manager=paged, paged_ssd_cache_manager=ssd
    )
    return cache, ssd


def _restorable_tokens(cache, request_id, tokens) -> int:
    table, _remaining = cache.fetch_cache(request_id, tokens)
    if table is None:
        return 0
    restored = cache.reconstruct_cache(table)
    return table.num_tokens if restored is not None else 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not available")
@pytest.mark.parametrize("policy,expected_blocks", [("lru", 0), ("chain_aware_lru", 4)])
def test_over_capacity_chain_end_to_end(tmp_path, policy, expected_blocks):
    """A 10-block conversation in a 4-block tier: LRU restores nothing,
    chain_aware_lru restores the first 4 blocks."""
    # Measure one block's hot size, then size the tier for ~4.5 blocks.
    probe, probe_ssd = _prefix_cache(tmp_path / "probe", "lru", 100 * 1024**2)
    probe.store_cache("probe", list(range(BLOCK_SIZE)), _kv_data(BLOCK_SIZE))
    block_bytes = probe_ssd._hot_cache_total_bytes
    assert block_bytes > 0
    probe_ssd.close()

    cache, ssd = _prefix_cache(tmp_path, policy, int(4.5 * block_bytes))
    tokens = list(range(10 * BLOCK_SIZE))
    assert cache.store_cache("turn-1", tokens, _kv_data(len(tokens))) is not None

    query = tokens + [9999]  # next turn: same history plus one new token
    assert _restorable_tokens(cache, "turn-2", query) == expected_blocks * BLOCK_SIZE
    ssd.close()
