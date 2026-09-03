# SPDX-License-Identifier: Apache-2.0
"""Tests for cache-aware request scheduler."""

from __future__ import annotations

import logging

import pytest

from omlx.cluster.cache_aware_scheduler import (
    CacheAwareScheduler,
    CacheAwareSchedulerConfig,
    RoutingDecision,
)
from omlx.cluster.prefix_cache_api import NodeCacheState


def _make_node(
    node_id: str,
    prefix_hashes: list[str] | None = None,
    hit_rate: float = 0.0,
    tokens_saved: int = 0,
) -> NodeCacheState:
    return NodeCacheState(
        node_id=node_id,
        hostname=f"{node_id}.local",
        total_blocks=100,
        used_blocks=50,
        hit_rate=hit_rate,
        prefix_hashes=prefix_hashes or [],
        tokens_saved=tokens_saved,
        total_queries=100,
        evictions=0,
    )


class TestCacheAwareSchedulerSelectsBestNode:
    """Node with most prefix overlap is selected."""

    def test_selects_node_with_highest_overlap(self):
        node_a = _make_node("node-a", prefix_hashes=["h1", "h2", "h3"])
        node_b = _make_node("node-b", prefix_hashes=["h2", "h3", "h4", "h5"])
        node_c = _make_node("node-c", prefix_hashes=["h5", "h6"])

        scheduler = CacheAwareScheduler()
        decision = scheduler.select_node(
            request_prefix_hashes=["h1", "h2", "h3"],
            available_nodes=[node_a, node_b, node_c],
        )

        assert decision.node_id == "node-a"
        assert decision.cache_overlap == 3
        assert decision.total_prefixes == 3
        assert "3/3" in decision.reason

    def test_ties_go_to_first_encountered(self):
        node_a = _make_node("node-a", prefix_hashes=["h1", "h2"])
        node_b = _make_node("node-b", prefix_hashes=["h1", "h2"])

        scheduler = CacheAwareScheduler()
        decision = scheduler.select_node(
            request_prefix_hashes=["h1", "h2"],
            available_nodes=[node_a, node_b],
        )

        assert decision.node_id == "node-a"
        assert decision.cache_overlap == 2

    def test_partial_overlap_prefers_higher(self):
        node_a = _make_node("node-a", prefix_hashes=["h1"])
        node_b = _make_node("node-b", prefix_hashes=["h1", "h2", "h3"])

        scheduler = CacheAwareScheduler()
        decision = scheduler.select_node(
            request_prefix_hashes=["h1", "h2", "h3"],
            available_nodes=[node_a, node_b],
        )

        assert decision.node_id == "node-b"
        assert decision.cache_overlap == 3

    def test_no_overlap_selects_first_by_score(self):
        node_a = _make_node("node-a", prefix_hashes=["x1"])
        node_b = _make_node("node-b", prefix_hashes=["x2"])

        scheduler = CacheAwareScheduler()
        decision = scheduler.select_node(
            request_prefix_hashes=["h1", "h2"],
            available_nodes=[node_a, node_b],
        )

        assert decision.cache_overlap == 0
        assert "0/2" in decision.reason

    def test_single_node_returns_that_node(self):
        node = _make_node("only-node", prefix_hashes=["h1"])
        scheduler = CacheAwareScheduler()
        decision = scheduler.select_node(
            request_prefix_hashes=["h1"],
            available_nodes=[node],
        )

        assert decision.node_id == "only-node"
        assert decision.cache_overlap == 1


class TestCacheAwareFallbackToRoundRobin:
    """When no cache match, falls back to round-robin."""

    def test_round_robin_no_prefixes(self):
        node_a = _make_node("node-a", prefix_hashes=["h1"])
        node_b = _make_node("node-b", prefix_hashes=["h2"])

        scheduler = CacheAwareScheduler()
        d1 = scheduler.select_node([], [node_a, node_b])
        d2 = scheduler.select_node([], [node_a, node_b])

        assert d1.node_id == "node-a"
        assert d2.node_id == "node_b" or d2.node_id == "node-b"
        assert "round-robin" in d1.reason

    def test_round_robin_cycles(self):
        nodes = [_make_node(f"node-{i}") for i in range(3)]
        scheduler = CacheAwareScheduler()

        results = [
            scheduler.select_node([], nodes).node_id
            for _ in range(6)
        ]

        assert results == ["node-0", "node-1", "node-2", "node-0", "node-1", "node-2"]

    def test_fallback_below_overlap_threshold(self):
        node_a = _make_node("node-a", prefix_hashes=["h1"])

        scheduler = CacheAwareScheduler(
            CacheAwareSchedulerConfig(min_overlap_ratio=0.5)
        )
        decision = scheduler.select_node(
            request_prefix_hashes=["h1", "h2", "h3", "h4"],
            available_nodes=[node_a],
        )

        assert "round-robin" in decision.reason
        assert "below threshold" in decision.reason


class TestCacheAwareDisabledPassthrough:
    """When disabled, behaves like normal scheduler."""

    def test_disabled_returns_first_node(self):
        node_a = _make_node("node-a", prefix_hashes=["h1", "h2", "h3"])
        node_b = _make_node("node-b", prefix_hashes=["h4", "h5"])

        scheduler = CacheAwareScheduler(CacheAwareSchedulerConfig(enabled=False))
        decision = scheduler.select_node(
            request_prefix_hashes=["h1", "h2", "h3"],
            available_nodes=[node_a, node_b],
        )

        assert decision.node_id == "node-a"
        assert decision.cache_overlap == 0
        assert "disabled" in decision.reason

    def test_disabled_ignores_prefixes(self):
        node = _make_node("node-x", prefix_hashes=[])
        scheduler = CacheAwareScheduler(CacheAwareSchedulerConfig(enabled=False))
        decision = scheduler.select_node(
            request_prefix_hashes=["h1", "h2"],
            available_nodes=[node],
        )

        assert decision.node_id == "node-x"
        assert "disabled" in decision.reason


class TestCacheAwareRoutingLogged:
    """Routing decision includes cache reasoning."""

    def test_decision_contains_reason(self):
        node = _make_node("node-1", prefix_hashes=["h1"])
        scheduler = CacheAwareScheduler()
        decision = scheduler.select_node(
            request_prefix_hashes=["h1"],
            available_nodes=[node],
        )

        assert isinstance(decision, RoutingDecision)
        assert len(decision.reason) > 0
        assert decision.node_id == "node-1"
        assert decision.cache_overlap == 1
        assert decision.total_prefixes == 1

    def test_disabled_decision_logged(self):
        node = _make_node("node-1")
        scheduler = CacheAwareScheduler(CacheAwareSchedulerConfig(enabled=False))
        decision = scheduler.select_node(
            request_prefix_hashes=["h1"],
            available_nodes=[node],
        )

        assert "disabled" in decision.reason

    def test_round_robin_fallback_logged(self):
        node = _make_node("node-1")
        scheduler = CacheAwareScheduler()
        decision = scheduler.select_node([], [node])

        assert "round-robin" in decision.reason


class TestCacheAwareNoRegression:
    """Existing behavior unchanged when disabled."""

    def test_disabled_always_returns_first_node(self):
        nodes = [_make_node(f"n-{i}", prefix_hashes=[f"h{i}"]) for i in range(5)]
        scheduler = CacheAwareScheduler(CacheAwareSchedulerConfig(enabled=False))

        for _ in range(10):
            decision = scheduler.select_node(
                request_prefix_hashes=["h0", "h1", "h2"],
                available_nodes=nodes,
            )
            assert decision.node_id == "n-0"

    def test_enabled_considers_all_nodes(self):
        nodes = [
            _make_node("n-0", prefix_hashes=["h10", "h11"]),
            _make_node("n-1", prefix_hashes=["h0", "h1", "h2"]),
            _make_node("n-2", prefix_hashes=["h5", "h6"]),
        ]
        scheduler = CacheAwareScheduler()
        decision = scheduler.select_node(
            request_prefix_hashes=["h0", "h1", "h2"],
            available_nodes=nodes,
        )

        assert decision.node_id == "n-1"
        assert decision.cache_overlap == 3

    def test_single_node_works_in_both_modes(self):
        node = _make_node("solo", prefix_hashes=["h1"])
        for enabled in (True, False):
            scheduler = CacheAwareScheduler(
                CacheAwareSchedulerConfig(enabled=enabled)
            )
            decision = scheduler.select_node(["h1"], [node])
            assert decision.node_id == "solo"


class TestCacheAwareEdgeCases:
    """Edge cases and error handling."""

    def test_raises_on_empty_nodes(self):
        scheduler = CacheAwareScheduler()
        with pytest.raises(ValueError, match="no available nodes"):
            scheduler.select_node(["h1"], [])

    def test_raises_on_empty_nodes_disabled(self):
        scheduler = CacheAwareScheduler(CacheAwareSchedulerConfig(enabled=False))
        with pytest.raises(ValueError, match="no available nodes"):
            scheduler.select_node(["h1"], [])

    def test_config_update(self):
        scheduler = CacheAwareScheduler(CacheAwareSchedulerConfig(enabled=False))
        node = _make_node("n1", prefix_hashes=["h1"])

        decision = scheduler.select_node(["h1"], [node])
        assert "disabled" in decision.reason

        scheduler.update_config(CacheAwareSchedulerConfig(enabled=True))
        decision = scheduler.select_node(["h1"], [node])
        assert "1/1" in decision.reason

    def test_topology_summary(self):
        from omlx.cluster.prefix_cache_api import PrefixCacheTopology

        topology = PrefixCacheTopology(
            nodes=[
                _make_node("n1", prefix_hashes=["h1", "h2"], hit_rate=0.8, tokens_saved=100),
                _make_node("n2", prefix_hashes=["h3"], hit_rate=0.6, tokens_saved=50),
            ]
        )
        scheduler = CacheAwareScheduler()
        summary = scheduler.get_topology(topology)

        assert summary["node_count"] == 2
        assert summary["total_prefix_hashes"] == 3
        assert summary["total_tokens_saved"] == 150
        assert abs(summary["avg_hit_rate"] - 0.7) < 1e-9

    def test_empty_prefix_hashes_handled(self):
        node = _make_node("n1", prefix_hashes=["h1"])
        scheduler = CacheAwareScheduler()
        decision = scheduler.select_node([], [node])

        assert decision.node_id == "n1"
        assert "round-robin" in decision.reason
