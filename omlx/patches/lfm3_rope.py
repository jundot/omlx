# SPDX-License-Identifier: Apache-2.0
"""One-launch RoPE for LFM3 (mlx-vlm ``models/lfm3``).

mlx-vlm builds the rotation from about fifteen array ops per call. It is
half-split RoPE whose pairs past ``partial_rotary_factor`` have zero
frequency, so ``mx.fast.rope`` with an infinite period for those pairs
(angle 0, identity) gives the same rotation in one launch. The fast kernel
forms the angle and rotates in fp32 with one rounding, so outputs differ
from stock by rounding only.
"""

from __future__ import annotations

import logging
import math
from functools import cache

import mlx.core as mx

logger = logging.getLogger(__name__)


@cache
def rope_periods(dim: int, base: float, angles: int) -> mx.array:
    """``base ** (2 i / dim)`` for the first ``angles`` pairs, inf after."""
    i = mx.arange(dim // 2, dtype=mx.float32)
    periods = mx.where(
        i < angles, base ** (2 * i / dim), mx.array(math.inf, dtype=mx.float32)
    )
    mx.eval(periods)
    return periods


def apply_lfm3_rope_patch() -> bool:
    """Route ``RotaryEmbedding`` calls through ``mx.fast.rope``.

    Integer offsets and the per-row offset arrays of batch caches both map
    to ``mx.fast.rope``; explicit ``position_ids`` keep the original body.
    """
    try:
        from mlx_vlm.models.lfm3 import language as lfm3
    except ImportError:
        return False
    cls = lfm3.RotaryEmbedding
    if getattr(cls, "_omlx_lfm3_fast_rope", False):
        return True
    orig_call = cls.__call__

    def patched_call(self, x, offset=0, position_ids=None):
        if position_ids is not None or x.ndim != 4:
            return orig_call(self, x, offset, position_ids)
        return mx.fast.rope(
            x,
            self.dim,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=rope_periods(self.dim, float(self.base), int(self.angles)),
        )

    cls.__call__ = patched_call
    cls._omlx_lfm3_fast_rope = True
    cls._omlx_lfm3_original_call = orig_call
    logger.info("LFM3 fast RoPE patch applied")
    return True
