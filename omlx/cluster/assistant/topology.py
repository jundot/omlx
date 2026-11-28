# SPDX-License-Identifier: Apache-2.0
"""The measured fabric as a pipeline line.

A pipeline is a chain: every rank talks to the rank before and after it. For a
2–4 Mac cluster the validation fabric is a **line**, so a plan is only runnable
when every consecutive pair in its rank order has a measured link. A full mesh
is stronger than the product needs, and a ring is not the deployed topology.

``fabric_links`` is evidence, not a guess: an empty/absent set means "the fabric
was not measured", never "there are no links". Callers pass ``None`` for an
unmeasured fabric and keep the documented nominal-ring fallback.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

LinkPair = tuple[str, str]


def _normalized(pair: Sequence[str]) -> LinkPair:
    first, second = str(pair[0]), str(pair[1])
    return (first, second) if first <= second else (second, first)


def link_pairs(links: Iterable[Sequence[str]]) -> frozenset[LinkPair]:
    """A set of undirected links, ignoring self-links and blank endpoints."""

    pairs: set[LinkPair] = set()
    for pair in links:
        if len(pair) != 2:
            continue
        normalized = _normalized(pair)
        if normalized[0] and normalized[1] and normalized[0] != normalized[1]:
            pairs.add(normalized)
    return frozenset(pairs)


def missing_line_links(
    rank_order: Sequence[str], links: Iterable[Sequence[str]]
) -> tuple[LinkPair, ...]:
    """The consecutive rank pairs that have no measured link, in rank order.

    One Mac (or none) is trivially a line. Every longer order needs each
    neighbour pair; the returned pairs are the exact cables to check.
    """

    available = link_pairs(links)
    return tuple(
        (str(before), str(after))
        for before, after in zip(rank_order, rank_order[1:])
        if _normalized((str(before), str(after))) not in available
    )
