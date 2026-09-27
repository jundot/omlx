# SPDX-License-Identifier: Apache-2.0
"""Static-prefix boundary detection for SpecPrefill.

The scheduler uses the boundary as a prefix length, so it must be measured on
the rendered prompt. Subtracting a non-system render does not work when the
template adds a default system block to that render. Instead, render the
static messages with two different user turns and keep the prefix that both
renders and the real prompt share.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

# Roles whose content SpecPrefill must never sparsify.
STATIC_PREFIX_ROLES = ("system", "developer")

# Two throwaway user turns; where their renders diverge, conversation begins.
_PROBE_A = "specprefill-boundary-probe-a"
_PROBE_B = "specprefill-boundary-probe-b-differs"

RenderTokens = Callable[[list[dict[str, Any]]], Sequence[int]]


def common_prefix_length(a: Sequence[int], b: Sequence[int]) -> int:
    """Return how many leading elements ``a`` and ``b`` share."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def resolve_static_prefix_end(
    messages: list[dict[str, Any]],
    prompt_token_ids: Sequence[int],
    render_tokens: RenderTokens,
) -> int:
    """Return the length of the prompt's static system/developer/tool prefix.

    ``render_tokens`` must render with the same tools and template kwargs as
    the real prompt. The result can include the first turn's header tokens.
    Returns 0 when there is nothing to protect.
    """
    static_messages = []
    for message in messages:
        if message.get("role") not in STATIC_PREFIX_ROLES:
            break
        static_messages.append(message)
    if not static_messages or len(static_messages) == len(messages):
        return 0

    probe_a = render_tokens(static_messages + [{"role": "user", "content": _PROBE_A}])
    probe_b = render_tokens(static_messages + [{"role": "user", "content": _PROBE_B}])

    scaffolding = common_prefix_length(probe_a, probe_b)
    return min(common_prefix_length(prompt_token_ids, probe_a), scaffolding)
