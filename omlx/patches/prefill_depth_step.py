# SPDX-License-Identifier: Apache-2.0
"""Bound mlx-lm's prefill chunk by the KV depth already in the cache (Metal GPU watchdog).

One prefill chunk is ONE command buffer, and its duration grows with chunk x KV depth: the GPU watchdog
(``kIOGPUCommandBufferCallbackErrorTimeout``) kills a rank once it exceeds a few seconds. A fixed small
chunk protects the deep end of a long prompt but wastes the shallow part (measured: a 12.5k prompt prefills
at ~380 tok/s with a 512 chunk and ~680 tok/s with 1024 on the same pipeline), and a fixed large chunk dies
near 245k. The loop in ``PromptProcessingBatch.prompt`` re-reads ``self.prefill_step_size`` every iteration,
so a property that answers min(configured step, budget / (depth + step)) makes the chunk shrink as the
prompt gets deeper, with no change to mlx-lm.

Budget (chunk x depth, tokens): step 1024 passed a 124k prompt and died near the end of a 245k one, so the
default 1.6e8 gives 1024 up to ~150k, 512 up to ~310k. The configured step is still the ceiling.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_BUDGET = 160_000_000
_FLOOR = 128
_MARKER = "_omlx_depth_bounded_prefill_step"


def bounded_step(base: int, depth: int, budget: int = DEFAULT_BUDGET) -> int:
    """Largest power-of-two step <= base with step x (depth + step) kept inside the budget."""

    base = int(base)
    depth = max(0, int(depth))
    allowed = max(_FLOOR, budget // max(1, depth + base))
    step = _FLOOR
    while step * 2 <= min(base, allowed):
        step *= 2
    return min(base, step) if step < base else base


def cache_depth(prompt_cache) -> int:
    """Tokens already in the deepest growing KV cache of this stage (0 when none exposes an integer offset)."""

    depth = 0
    for entry in prompt_cache or ():
        offset = getattr(entry, "offset", None)
        try:
            depth = max(depth, int(offset))
        except (TypeError, ValueError):
            try:  # a batched cache keeps one offset per row
                depth = max(depth, int(max(offset)))
            except (TypeError, ValueError):
                continue
    return depth


def install(budget: int = DEFAULT_BUDGET) -> bool:
    """Make ``PromptProcessingBatch.prefill_step_size`` depth-bounded. Idempotent."""

    import importlib

    generate = importlib.import_module(
        "mlx_lm.generate"
    )  # not ``from mlx_lm import generate`` (a function)

    cls = getattr(generate, "PromptProcessingBatch", None)
    if cls is None:
        return False
    current = cls.__dict__.get("prefill_step_size")
    if isinstance(current, property) and getattr(current.fget, _MARKER, False):
        return True

    def getter(self):
        base = self.__dict__.get("_omlx_configured_prefill_step", 2048)
        return bounded_step(
            base, cache_depth(getattr(self, "prompt_cache", ())), budget
        )

    def setter(self, value):
        self.__dict__["_omlx_configured_prefill_step"] = int(value)

    setattr(getter, _MARKER, True)
    cls.prefill_step_size = property(getter, setter)
    logger.info("prefill chunk bounded by KV depth (budget %s)", budget)
    return True
