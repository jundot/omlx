"""Retention policy for per-boundary recurrent-state checkpoints.

Hybrid models (GDN / Mamba / KDA linear-attention layers) can only resume a
shared prefix from a sequence boundary where the recurrent state was captured,
so oMLX captures a checkpoint at every block boundary during prefill. That makes
the retained prefix cost grow linearly with context *on top of* the attention
KV, and the checkpoint is by far the larger term: measured on
Qwen3.8-Flash-Next (36 GDN layers, block_size 2048), one 234,070-token prefill
captured 113 checkpoints of 115.6MB each -- 12.06GB resident for the whole
prefill, 52.5KB per prompt token against the 27KB of the entire live attention
cache. On a 128GB machine that resident term is what pushes long-context
requests into the memory guard.

Restores do not need that density. A restore lands on the deepest retained
checkpoint at or below the depth it needs and re-prefills the gap: the store path
already refuses to persist a non-sliceable block without its checkpoint ("storing
live non-sliceable state would corrupt later prefix hits"), so checkpoints only
trade memory against gap recompute -- which makes them the right thing to budget.

The budget is spent as an even lattice over the prompt, because uniform spacing
minimizes the worst-case recompute gap for a fixed budget. It is applied while
prefilling (the final length is unknown, so the lattice is re-spread over the
range reached so far each time it overflows) and always keeps the newest
boundary: an append-only multi-turn conversation resumes from that one alone,
while the interior ones serve requests that diverge inside the stored prefix.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

#: Retaining more than this many checkpoints is past the point where the
#: placement argument holds: the budget exists to bound a per-request resident
#: cost, and 64 checkpoints of a hybrid model are already gigabytes.
MAX_RETAINED_BOUNDARIES = 64


def select_spaced_boundaries(
    recorded: Sequence[int], limit: int, block_size: int, keep: int
) -> set[int]:
    """Return the boundaries of ``recorded`` that keep a checkpoint.

    ``recorded`` is the ascending boundary token counts captured so far for one
    request. ``limit`` is the depth the request's cache can still be *stored*
    and resumed at: the end of prefill while prefilling, and the cacheable
    prefix while decoding (see the call sites). ``block_size`` is the block size
    the boundaries are aligned to. The budget ``keep`` counts interior
    checkpoints; in addition to those the function always retains two
    boundaries that the caller cannot afford to lose:

    * the deepest recorded boundary at or below ``limit`` -- the depth the next
      turn of a growing conversation resumes from, and the only boundary the
      store path can use for its deepest block;
    * the deepest recorded boundary overall, which is the one the store reads as
      ``latest_tc``.

    ``keep <= 0`` retains everything, i.e. the pre-existing behavior.

    ``limit`` rather than the newest capture is the reference point because a
    decoding request keeps capturing boundaries past the prompt, and while
    speculative decoding and reasoning output push that newest capture further
    out, anchoring the lattice on it widens the stride and evicts the in-range
    boundary the next turn needs -- which presents not as a shallower hit but as
    no hit at all, and on a store whose only surviving boundary is out of range,
    as a refused store for the whole request.
    """
    budget = 0 if keep < 0 else min(int(keep), MAX_RETAINED_BOUNDARIES)
    if budget == 0 or not recorded or block_size <= 0:
        return set(recorded)

    # Widening by powers of two keeps the lattices nested: a point on the
    # ``2 * stride`` lattice was always on the ``stride`` lattice, so growing the
    # prompt never deletes a checkpoint the wider lattice still needs. Any other
    # widening leaves holes -- measured, one gap spanning 98% of the prompt.
    stride = 1
    while stride * block_size * budget < limit:
        stride *= 2
    step = block_size * stride

    within = [tc for tc in recorded if tc <= limit]
    retained = {tc for tc in within if tc % step == 0}
    if within:
        retained.add(within[-1])
    retained.add(recorded[-1])
    return retained
