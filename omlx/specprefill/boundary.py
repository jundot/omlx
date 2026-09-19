# SPDX-License-Identifier: Apache-2.0
"""Static-prefix boundary detection for SpecPrefill.

SpecPrefill never drops tokens from the static prefix of a prompt — the
system/developer material plus whatever tool-instruction scaffolding the chat
template emits ahead of the first conversation turn.  The scheduler consumes
the boundary as a plain prefix length (``prompt_token_ids[:system_end]``), so
it has to be the index where the *rendered* prompt stops being static, not a
count of the system messages' own tokens.

Deriving that index by subtracting a re-render of the non-system messages from
the full prompt is unsound: a chat template is free to emit content into the
non-system render that never appears in the full prompt.  Qwen3.5-family
templates, for instance, inject a default system block whenever no system
message is supplied, which makes the subtraction under-report the boundary by
the size of that block and leaves real system text unprotected.

Instead, measure the boundary directly.  Re-render the system/developer
messages twice, each followed by a different throwaway conversation turn, and
keep the token prefix that (a) both probes agree on and (b) the real prompt
also starts with.  Tokens the two probes agree on cannot depend on conversation
content, and tokens the real prompt shares with them are by construction part
of its own static prefix.  The result needs no per-template constants and can
only ever under-report, never over-report.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

# Roles whose content SpecPrefill must never sparsify.
STATIC_PREFIX_ROLES = ("system", "developer")

# Two throwaway user turns used to fingerprint the template scaffolding. They
# only have to differ from each other and to be implausible as a real prefix of
# user content; nothing about their text reaches the model.
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

    ``render_tokens`` renders a message list through the caller's own chat
    template — same tools, same template kwargs as the real prompt — and
    returns its token ids. Returns 0 when there is no static prefix to protect
    or when the boundary cannot be established, which leaves SpecPrefill in its
    existing "no protected prefix" behaviour rather than guessing.
    """
    static_messages = [m for m in messages if m.get("role") in STATIC_PREFIX_ROLES]
    if not static_messages or len(static_messages) == len(messages):
        return 0

    probe_a = render_tokens(static_messages + [{"role": "user", "content": _PROBE_A}])
    probe_b = render_tokens(static_messages + [{"role": "user", "content": _PROBE_B}])

    # Where the probes diverge is where conversation content begins; the real
    # prompt can only be protected up to that point, and only as far as it
    # actually matches.
    scaffolding = common_prefix_length(probe_a, probe_b)
    return min(common_prefix_length(prompt_token_ids, probe_a), scaffolding)
