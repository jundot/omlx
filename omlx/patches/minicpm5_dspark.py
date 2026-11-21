# SPDX-License-Identifier: Apache-2.0
"""MiniCPM5-2B DSpark speculative decoding adaptation (llama backbone).

This module now contains **only the pure-logic DSpark adaptive block-width
functions** used by the oMLX DFlash block-drafter path.

Background
----------
The MiniCPM5-2B target is a text-only ``llama`` model.  It runs DFlash /
DSpark as a **batched** block drafter directly (no VLM engine forcing):
``utils/model_loading.py`` adds ``"llama"`` to
``_DFLASH_BATCHED_MODEL_TYPES["batched"]``, so the generic DFlash path
activates for llama targets without any engine conversion.

The earlier VLM force-hook monkey-patch (``install_engine_force_hook``),
the mlx-vlm llama ``LanguageModel`` capture/rollback hooks, the
``_DepthController`` extension, and the ``attach_drafter`` wrapper were all
part of the removed VLM route.  They are no longer needed because the
batched path uses the generic DFlash drafter machinery.

The two functions below are pure ports of the official DSpark adaptive
block-width logic (``_dflash_initial_depth`` / ``_dflash_next_block_size``)
and are exercised by the unit test suite.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# DSpark keeps proposing down to this minimum block width (official default).
_MIN_BLOCK_SIZE = 4

def dspark_initial_depth(block_total: int, min_block_size: int = _MIN_BLOCK_SIZE) -> int:
    """DSpark's initial proposal count (official initial block 4 -> 3)."""
    if block_total <= 1:
        return max(0, block_total - 1)
    return max(1, min(block_total, max(2, int(min_block_size))) - 1)


def dspark_next_block_size(
    block_total: int,
    accept_lens: List[int],
    draft_lens: List[int],
    min_block_size: int = _MIN_BLOCK_SIZE,
    log: Optional[logging.Logger] = None,
) -> int:
    """Official DSpark adaptive block width (pure port of ``_dflash_next_block_size``).

    ``accept_lens`` / ``draft_lens`` are **block widths** (the official
    convention: ``draft_lens`` records how many tokens were verified this
    cycle, not how many were proposed).  Returns the next **block width**;
    the controller converts to depth (``block - 1``).  Mirrors mlx-vlm
    ``_dflash_next_block_size`` exactly: back off below 0.30 accept or
    mean_accept < 2.0, hold at 0.50-0.85, grow at >=0.85 with >=0.75
    full-hit rate, never below ``min_block_size``.
    """
    if block_total <= 1:
        return block_total
    recent: List[Tuple[float, int]] = [
        (float(a), int(d))
        for a, d in zip(accept_lens[-8:], draft_lens[-8:])
        if int(d) > 0
    ]
    if not recent:
        if min_block_size >= 2:
            # No history: start at the configured floor (official initial).
            return min(block_total, max(2, int(min_block_size)))
        return block_total
    current = min(block_total, max(2, recent[-1][1] + 1))
    min_total = min(block_total, max(2, int(min_block_size)))
    drafted = sum(d for _, d in recent)
    accepted = sum(a for a, _ in recent)
    accept_rate = accepted / max(1e-6, drafted)
    mean_accept = accepted / len(recent)
    if accept_rate < 0.30 or mean_accept < 2.0:
        if current >= 8:
            return max(min_total, min(block_total, current // 2))
        result = max(min_total, min(block_total, current - 2))
        if log is not None:
            log.debug(
                "_dspark_next_block_size shrink: recent=%s rate=%.2f current=%d "
                "min_total=%d -> %d",
                recent, accept_rate, current, min_total, result,
            )
        return result
    if accept_rate < 0.50:
        return max(min_total, min(block_total, current - 2))
    full_hits = sum(1 for a, d in recent if a >= d)
    full_hit_rate = full_hits / len(recent)
    if accept_rate >= 0.85 and full_hit_rate >= 0.75:
        return min(block_total, current + 2)
    return min(block_total, current)


