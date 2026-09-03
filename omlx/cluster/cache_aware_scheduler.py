# SPDX-License-Identifier: Apache-2.0
"""Cache-aware request scheduler for cluster-level routing.

Routes requests to nodes with the best prefix-cache hit potential,
reducing cold-start latency by reusing already-cached prefixes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .prefix_cache_api import NodeCacheState, PrefixCacheTopology

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of a cache-aware routing decision."""

    node_id: str
    cache_overlap: int
    total_prefixes: int
    reason: str


@dataclass
class CacheAwareSchedulerConfig:
    """Configuration for cache-aware scheduling."""

    enabled: bool = True
    # Minimum overlap ratio (0.0-1.0) to consider a node cache-aware.
    # Below this threshold, falls back to round-robin.
    min_overlap_ratio: float = 0.0


class CacheAwareScheduler:
    """Routes requests to nodes with the best prefix-cache overlap.

    When enabled, this scheduler scores each available node by how many
    of the request's prefix hashes are already cached on that node,
    selecting the node with the highest overlap. When no node has cached
    prefixes or the scheduler is disabled, it falls back to round-robin
    selection.

    Args:
        config: Configuration controlling enabled state and thresholds.
    """

    def __init__(self, config: CacheAwareSchedulerConfig | None = None) -> None:
        self._config = config or CacheAwareSchedulerConfig()
        self._round_robin_index = 0

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def select_node(
        self,
        request_prefix_hashes: list[str],
        available_nodes: list[NodeCacheState],
    ) -> RoutingDecision:
        """Select the best node for a request based on prefix-cache overlap.

        Args:
            request_prefix_hashes: Prefix hashes the request will process.
            available_nodes: Current cache state of each available node.

        Returns:
            RoutingDecision with the selected node and reasoning.
        """
        if not self._config.enabled:
            return self._passthrough(available_nodes)

        if not available_nodes:
            raise ValueError("no available nodes for routing")

        if not request_prefix_hashes:
            decision = self._round_robin(available_nodes)
            decision.reason = "no request prefixes provided, round-robin"
            return decision

        best_node: NodeCacheState | None = None
        best_overlap = -1

        for node in available_nodes:
            node_hashes = set(node.prefix_hashes)
            overlap = sum(1 for h in request_prefix_hashes if h in node_hashes)
            if overlap > best_overlap:
                best_overlap = overlap
                best_node = node

        assert best_node is not None

        overlap_ratio = (
            best_overlap / len(request_prefix_hashes)
            if request_prefix_hashes
            else 0.0
        )

        if overlap_ratio < self._config.min_overlap_ratio:
            decision = self._round_robin(available_nodes)
            decision.reason = (
                f"best overlap {overlap_ratio:.2f} below threshold "
                f"{self._config.min_overlap_ratio:.2f}, round-robin"
            )
            return decision

        return RoutingDecision(
            node_id=best_node.node_id,
            cache_overlap=best_overlap,
            total_prefixes=len(request_prefix_hashes),
            reason=(
                f"cache hit: {best_overlap}/{len(request_prefix_hashes)} "
                f"prefixes cached on {best_node.node_id}"
            ),
        )

    def _passthrough(self, available_nodes: list[NodeCacheState]) -> RoutingDecision:
        """Return the first node when cache-aware routing is disabled."""
        if not available_nodes:
            raise ValueError("no available nodes for routing")
        node = available_nodes[0]
        return RoutingDecision(
            node_id=node.node_id,
            cache_overlap=0,
            total_prefixes=0,
            reason="cache-aware routing disabled, passthrough",
        )

    def _round_robin(self, available_nodes: list[NodeCacheState]) -> RoutingDecision:
        """Select the next node in round-robin order."""
        if not available_nodes:
            raise ValueError("no available nodes for routing")
        index = self._round_robin_index % len(available_nodes)
        node = available_nodes[index]
        self._round_robin_index += 1
        return RoutingDecision(
            node_id=node.node_id,
            cache_overlap=0,
            total_prefixes=0,
            reason="round-robin fallback",
        )

    def update_config(self, config: CacheAwareSchedulerConfig) -> None:
        """Update the scheduler configuration at runtime."""
        self._config = config

    def get_topology(self, topology: PrefixCacheTopology) -> dict[str, Any]:
        """Summarize cache topology for logging/observability."""
        return {
            "node_count": len(topology.nodes),
            "total_prefix_hashes": sum(
                len(n.prefix_hashes) for n in topology.nodes
            ),
            "total_tokens_saved": sum(n.tokens_saved for n in topology.nodes),
            "avg_hit_rate": (
                sum(n.hit_rate for n in topology.nodes) / len(topology.nodes)
                if topology.nodes
                else 0.0
            ),
        }
