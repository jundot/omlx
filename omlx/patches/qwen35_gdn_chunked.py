# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803, N806
"""Route Qwen3.5/3.6 Gated DeltaNet prefill to an optimized Metal kernel.

Default route: ``gated_delta_blocked_seq`` — the exact sequential recurrence
restructured for Apple GPUs (threadgroup-staged k/q/v blocks, register-resident
state, Dv/32 split). ~2x faster than mlx_lm's stock sequential kernel at 16k
(14.9ms vs 29.7ms per layer call) with fp32-exact state (rel-err ~5e-8).

Optional route (``OMLX_GDN_IMPL=chunked``): the FLA chunked WY-representation
kernels — accuracy-validated but slower than the stock kernel E2E; kept for
future iteration.

This rebinds ``gated_delta_update`` in ``mlx_vlm.models.qwen3_5.language`` for
scalar-gated prefill with T >= OMLX_GDN_MIN_T. Decode (T==1) and
masked/vectorized paths keep the original kernel.

Toggles:
  OMLX_GDN_KERNEL=0    disable the patch entirely
  OMLX_GDN_IMPL=...    blocked_seq (default) | chunked
  OMLX_GDN_BLOCK_T=N   blocked_seq time block: 16 | 32 | 48
                         (default 16 for float32, 32 otherwise)
  OMLX_GDN_MIN_T=N     minimum prefill length to engage (default 64)
  OMLX_GDN_FUSED_G_BETA=1
                         use mlx-vlm's Metal g/beta precompute helper
  OMLX_GDN_STUB=1      debug only: skip the GDN op to measure its E2E share
"""

import logging
import os

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_qwen35_gdn_prefill_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if os.environ.get("OMLX_GDN_KERNEL", "1") == "0":
        return False
    if not mx.metal.is_available():
        return False

    try:
        from mlx_vlm.models.qwen3_5 import gated_delta as gd
        from mlx_vlm.models.qwen3_5 import language as lang
    except ImportError:
        logger.debug("mlx_vlm qwen3_5 not importable; GDN prefill patch skipped")
        return False

    min_t = int(os.environ.get("OMLX_GDN_MIN_T", "64"))
    fused_g_beta = os.environ.get("OMLX_GDN_FUSED_G_BETA", "0") == "1"
    stub = os.environ.get("OMLX_GDN_STUB", "0") == "1"
    original = gd.gated_delta_update

    from omlx.custom_kernels.qwen35_prefill import (
        gated_delta_blocked_seq,
        gated_delta_chunked_metal,
    )

    impl = os.environ.get("OMLX_GDN_IMPL", "blocked_seq")
    fast_prefill = (
        gated_delta_chunked_metal if impl == "chunked" else gated_delta_blocked_seq
    )

    def gated_delta_update_metal(
        q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True
    ):
        # Debug-only: skip the GDN op entirely to measure its E2E share.
        # Output is garbage; never enable outside profiling.
        if stub and q.shape[1] > 1:
            if state is None:
                B_, Hv_, Dv_ = v.shape[0], v.shape[-2], v.shape[-1]
                state = mx.zeros((B_, Hv_, Dv_, q.shape[-1]), dtype=mx.float32)
            return v, state
        if (
            use_kernel
            and mask is None
            and q.shape[1] >= min_t
            and q.shape[-1] % 16 == 0
            and v.shape[-1] % 32 == 0
            and a.ndim == 3  # scalar per-head gating
        ):
            if fused_g_beta and hasattr(gd, "_compute_g_beta_prefill"):
                g, beta = gd._compute_g_beta_prefill(A_log, a, b, dt_bias)
            else:
                g, beta = gd._compute_g_beta(A_log, a, b, dt_bias)
            return fast_prefill(q, k, v, g, beta, state)
        return original(
            q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel
        )

    lang.gated_delta_update = gated_delta_update_metal
    gd.gated_delta_update = gated_delta_update_metal
    _PATCHED = True
    logger.info(
        "Qwen3.5/3.6 GDN prefill kernel patch applied (Metal, impl=%s, min_t=%d)",
        impl, min_t,
    )
    return True


def apply_qwen35_gdn_chunked_patch() -> bool:
    """Backward-compatible name for older callers/configuration."""
    return apply_qwen35_gdn_prefill_patch()
