# SPDX-License-Identifier: Apache-2.0
"""How fast each link actually is, so placement stops guessing from the label.

Rank placement asked one question of a link: is its ``kind`` in
``{"rdma", "thunderbolt"}``? That is a poor proxy. On this fabric a Mac-to-Mac
Thunderbolt link carries ~6.6 GB/s while a USB-NCM link presenting as ordinary
networking carries ~0.53 GB/s — a 12x difference invisible to a kind check.
Two links can also share a kind and differ by a factor of four (TB4 vs TB5), and
a negotiated-down cable reports the same ``kind`` as a healthy one.

So bandwidth is taken from the best evidence available, in order:

1. **measured** — oMLX's own collective probe. ``NodePerformanceProfile``
   already records ``collective_bandwidth_bytes_per_second`` from a real
   all-reduce, which is exactly the traffic tensor parallelism generates. A
   link is bounded by its slower endpoint, so a pair takes the minimum.
2. **nominal** — the negotiated link speed from ``system_profiler``. Real, but
   a ceiling: no fabric delivers its line rate to a collective.
3. **assumed** — a conservative constant per kind, when nothing else is known.

Keeping the source alongside the number matters as much as the number. "These
two Macs were grouped because we measured 6.6 GB/s between them" is a different
claim from "because the cable says Thunderbolt", and the interface should be
able to say which one it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

GB = 1000**3

# What a link of this kind is worth when we have measured nothing. Deliberately
# pessimistic: under-promising costs a suboptimal grouping, over-promising puts
# an all-reduce on a link that cannot carry it.
_ASSUMED_BYTES_PER_SECOND = {
    "rdma": 5.0 * GB,
    "thunderbolt": 2.0 * GB,
    "ethernet": 0.1 * GB,
    "unknown": 0.05 * GB,
}

# A negotiated line rate is a ceiling, not a delivery. Collectives see roughly
# this fraction of it in practice.
_NOMINAL_EFFICIENCY = 0.35

# Below this, a link cannot carry per-layer all-reduces at a useful rate and a
# tensor-parallel group should not straddle it.
FAST_LINK_FLOOR_BYTES_PER_SECOND = 1.0 * GB

# Bandwidth is not sufficient. Tensor parallelism issues two all-reduces per
# layer per token, so it is bound by round-trip latency as much as throughput:
# on this fabric the *same* Thunderbolt cable gave 28.6 tok/s under RDMA and
# 6.6 tok/s under the TCP ring — 4.3x from the transport alone, at identical
# bandwidth. So a line rate quoted for an Ethernet-class link is not evidence
# that it can carry a collective; only these kinds may be called fast on
# nominal evidence. A real measurement overrides this, because a measured
# all-reduce has already answered the question the kind is a proxy for.
_COLLECTIVE_CAPABLE_KINDS = {"rdma", "thunderbolt"}


@dataclass(frozen=True)
class LinkBandwidth:
    """What a link between two nodes is worth, and how we know."""

    source_node_id: str
    peer_node_id: str
    bytes_per_second: float
    source: str  # "measured" | "nominal" | "assumed"
    kind: str = "unknown"

    @property
    def gigabytes_per_second(self) -> float:
        return self.bytes_per_second / GB

    @property
    def fast(self) -> bool:
        """Can this link carry a tensor-parallel all-reduce?

        Fast enough *and* of a kind that can sustain the round trips — unless
        we measured it, in which case the measurement settles it.
        """

        if self.bytes_per_second < FAST_LINK_FLOOR_BYTES_PER_SECOND:
            return False
        return self.source == "measured" or self.kind in _COLLECTIVE_CAPABLE_KINDS

    def describe(self) -> str:
        return (
            f"{self.gigabytes_per_second:.2f} GB/s {self.source} "
            f"({self.kind})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "peer_node_id": self.peer_node_id,
            "bytes_per_second": self.bytes_per_second,
            "gigabytes_per_second": self.gigabytes_per_second,
            "source": self.source,
            "kind": self.kind,
            "fast": self.fast,
        }


def _measured_by_node(profiles: Sequence[Any] | None) -> dict[str, float]:
    """Collective bandwidth per node, from oMLX's own probe."""

    measured: dict[str, float] = {}
    for profile in profiles or ():
        node_id = getattr(profile, "node_id", "") or ""
        rate = getattr(profile, "collective_bandwidth_bytes_per_second", 0) or 0
        if node_id and rate > 0:
            # Several ranks can share a node; the slowest is what it delivers.
            existing = measured.get(node_id)
            measured[node_id] = min(existing, float(rate)) if existing else float(rate)
    return measured


def link_bandwidth(
    transport: Any,
    *,
    measured_by_node: dict[str, float] | None = None,
) -> LinkBandwidth:
    """The best available bandwidth estimate for one link."""

    kind = getattr(transport, "kind", "unknown") or "unknown"
    source_id = getattr(transport, "source_node_id", "") or ""
    peer_id = getattr(transport, "peer_node_id", "") or ""
    measured = measured_by_node or {}

    # A link is only as fast as its slower end, and only measured if both ends
    # were measured — one endpoint's number says nothing about the pair.
    if source_id in measured and peer_id in measured:
        return LinkBandwidth(
            source_id,
            peer_id,
            min(measured[source_id], measured[peer_id]),
            "measured",
            kind,
        )

    gbps = getattr(transport, "link_speed_gbps", None)
    if gbps:
        return LinkBandwidth(
            source_id,
            peer_id,
            (float(gbps) * 1000**3 / 8) * _NOMINAL_EFFICIENCY,
            "nominal",
            kind,
        )

    return LinkBandwidth(
        source_id,
        peer_id,
        _ASSUMED_BYTES_PER_SECOND.get(kind, _ASSUMED_BYTES_PER_SECOND["unknown"]),
        "assumed",
        kind,
    )


def bandwidth_graph(
    transports: Sequence[Any],
    profiles: Sequence[Any] | None = None,
) -> dict[tuple[str, str], LinkBandwidth]:
    """Undirected fabric weighted by bandwidth, keyed by sorted node pair.

    When the same pair appears more than once — two cables, or a link seen from
    both ends — the better evidence wins, then the faster number.
    """

    measured = _measured_by_node(profiles)
    graph: dict[tuple[str, str], LinkBandwidth] = {}
    rank = {"measured": 2, "nominal": 1, "assumed": 0}

    for transport in transports:
        link = link_bandwidth(transport, measured_by_node=measured)
        if not link.source_node_id or not link.peer_node_id:
            continue
        if link.source_node_id == link.peer_node_id:
            continue
        key = tuple(sorted((link.source_node_id, link.peer_node_id)))
        current = graph.get(key)  # type: ignore[arg-type]
        if current is None or (
            rank[link.source],
            link.bytes_per_second,
        ) > (rank[current.source], current.bytes_per_second):
            graph[key] = link  # type: ignore[index]
    return graph


def bandwidth_between(
    graph: dict[tuple[str, str], LinkBandwidth],
    a: str,
    b: str,
) -> float:
    """Bandwidth between two nodes; 0 when they are not directly linked."""

    link = graph.get(tuple(sorted((a, b))))  # type: ignore[arg-type]
    return link.bytes_per_second if link else 0.0


def slowest_link_in(
    graph: dict[tuple[str, str], LinkBandwidth],
    members: Sequence[str],
) -> LinkBandwidth | None:
    """The weakest link inside a group — what its all-reduce actually runs at.

    Returns None when a pair in the group has no link at all, which is a
    different failure from "slow" and must not be reported as a bandwidth.
    """

    worst: LinkBandwidth | None = None
    for index, first in enumerate(members):
        for second in members[index + 1 :]:
            link = graph.get(tuple(sorted((first, second))))  # type: ignore[arg-type]
            if link is None:
                return None
            if worst is None or link.bytes_per_second < worst.bytes_per_second:
                worst = link
    return worst
