# SPDX-License-Identifier: Apache-2.0
"""Cache-safe model-shape warmups shared by local and cluster engines."""

from __future__ import annotations

import os
import time
from typing import Any

MAX_PREFILL_SHAPE_WARMUP_TOKENS = 4096
_LOCAL_PREFILL_SHAPE_WARMUP_ENV = "OMLX_PREFILL_SHAPE_WARMUP"
_DS4_PREFILL_SHAPE_WARMUP_TOKENS = 1024


def _enabled(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "on", "yes"}


def planned_local_prefill_shape_warmup_tokens(
    model_type: str | None,
    *,
    environ: Any = os.environ,
) -> int:
    """Return a bounded warmup shape for a qualified local model.

    This is deliberately capability-gated rather than universal. A 1K
    hidden-state-only forward is known to visit DS4's long-prefill Metal
    variants and materially reduces its first-request compile penalty. New
    model families can join this table after the same cache-safety and live
    latency gates pass for them.
    """

    if not _enabled(str(environ.get(_LOCAL_PREFILL_SHAPE_WARMUP_ENV, "1"))):
        return 0
    if isinstance(model_type, str) and model_type.startswith("deepseek_v4"):
        return _DS4_PREFILL_SHAPE_WARMUP_TOKENS
    return 0


def run_prefill_shape_warmup(
    mx: Any,
    model: Any,
    *,
    tokens: int,
    max_kv_size: int | None,
    cache_factory: Any = None,
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    """Execute one cache-safe prefill forward and retain only kernel state.

    The temporary KV cache and allocator scratch are released after evaluation;
    Metal's in-process compute-pipeline cache remains available to subsequent
    user requests.
    """

    if not 1 <= int(tokens) <= MAX_PREFILL_SHAPE_WARMUP_TOKENS:
        raise ValueError("prefill shape warmup token count is out of bounds")
    if cache_factory is None:
        from mlx_lm.models.cache import make_prompt_cache

        cache_factory = make_prompt_cache

    prompt_cache = cache_factory(model, max_kv_size=max_kv_size)
    token_batch = mx.zeros((1, int(tokens)), dtype=mx.int32)
    mx.eval(token_batch)
    mx.synchronize()
    started = float(clock())
    output = None
    cache_states = []
    try:
        output = model(
            token_batch,
            cache=prompt_cache,
            skip_lm_head=True,
        )
        cache_states = [cache.state for cache in prompt_cache]
        if output is None:
            mx.eval(cache_states)
        else:
            mx.eval(output, cache_states)
        mx.synchronize()
        elapsed = max(0.0, float(clock()) - started)
    finally:
        del output, cache_states, prompt_cache, token_batch
        mx.clear_cache()
    return {
        "active": True,
        "tokens": int(tokens),
        "elapsed_seconds": elapsed,
    }
