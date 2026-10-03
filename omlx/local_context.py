# SPDX-License-Identifier: Apache-2.0
"""Largest context window that fits on this Mac alone, for local serving.

The cluster deployment UI already answers "how much context can this
model actually hold?" via :func:`omlx.cluster.catalogue.largest_context_that_fits`,
which bisects over the real planner instead of guessing bytes-per-token.
Local (single-node) model settings only had a blind numeric input. This
module reuses the same planner arithmetic with one synthetic node whose
capacity is the live memory ceiling the rest of oMLX admits against.

Everything here is advisory: it feeds an "Auto" button, so it returns
zeros for anything it cannot answer rather than raising.
"""

from __future__ import annotations

from pathlib import Path

from omlx.cluster.catalogue import largest_context_that_fits
from omlx.cluster.planner import ModelLayout, NodeBudget

__all__ = ["auto_context_for_layout", "local_auto_context"]


def auto_context_for_layout(
    layout: ModelLayout,
    *,
    capacity_bytes: int,
    declared_context_tokens: int | None = None,
) -> int:
    """Biggest context that plans on one local node of ``capacity_bytes``.

    Capped at the model's own declared context length so "auto" never
    proposes more than the model was trained for. Returns 0 when the
    layout has no known KV growth (synthetic/undownloaded models) or when
    nothing fits — same contract as ``largest_context_that_fits``.
    """

    if capacity_bytes <= 0:
        return 0
    node = NodeBudget(node_id="local", capacity_bytes=int(capacity_bytes), rank=0)
    ceiling = (
        int(declared_context_tokens)
        if declared_context_tokens and declared_context_tokens > 0
        else None
    )
    return largest_context_that_fits(layout, [node], ceiling_tokens=ceiling)


def local_auto_context(
    model_path: str | Path,
    *,
    capacity_bytes: int | None = None,
) -> dict[str, int]:
    """Auto-context assessment for a local model path. Never raises.

    Returns ``{"max_context_tokens": int, "declared_context_tokens": int}``
    with 0 for either value when it is unknown — an unreadable model, a
    layout with no KV information, or a model that simply does not fit.
    ``capacity_bytes`` defaults to the same live ceiling the memory-budget
    UI and the process memory enforcer admit against.
    """

    root = Path(model_path)

    declared = 0
    try:
        from omlx.model_discovery import _read_model_context_length

        declared = int(_read_model_context_length(root) or 0)
    except Exception:
        declared = 0

    try:
        # Layout first: an unreadable model bails out here without ever
        # touching the memory guard.
        from omlx.cluster.planner import complete_model_layout

        layout = complete_model_layout(root)

        if capacity_bytes is None:
            from omlx.cluster.memory_guard import ceiling_breakdown

            capacity_bytes = int(ceiling_breakdown().get("hard_limit", 0))
        maximum = auto_context_for_layout(
            layout,
            capacity_bytes=capacity_bytes,
            declared_context_tokens=declared or None,
        )
    except Exception:
        maximum = 0

    return {
        "max_context_tokens": int(maximum),
        "declared_context_tokens": declared,
    }
