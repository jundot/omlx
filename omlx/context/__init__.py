# SPDX-License-Identifier: Apache-2.0
"""Context compaction strategies for oMLX.

Public API::

    from omlx.context import TieredCompact, get_strategy

    strategy = get_strategy("tiered")
    compacted, tokens = strategy.compact(messages, budget_tokens=4096)
"""
from __future__ import annotations

from omlx.context.compaction import (
    TRUNCATED_MARKER,
    CompactStrategy,
    NoCompact,
    SlidingWindowCompact,
    TieredCompact,
    get_strategy,
)

__all__ = [
    "CompactStrategy",
    "NoCompact",
    "SlidingWindowCompact",
    "TieredCompact",
    "TRUNCATED_MARKER",
    "get_strategy",
]
