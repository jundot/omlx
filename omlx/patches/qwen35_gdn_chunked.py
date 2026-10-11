# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N803, N806
"""Route Qwen3.5/3.6 Gated DeltaNet prefill to an optimized Metal kernel.

Default route: ``gated_delta_pipelined`` — the exact sequential recurrence
with 8 lanes per value row, 16-row threadgroups and a software-pipelined,
unrolled 12-token block (Qwen3.8 16/48 heads, T=8191: 3.0 ms vs 4.9 ms per
layer call for ``gated_delta_blocked_seq`` on M5 Ultra). Layouts it does not
cover run ``gated_delta_blocked_seq``: threadgroup-staged k/q/v blocks,
register-resident state, Dv/32 split, fp32-exact state (rel-err ~5e-8). Both
kernels assume 128-wide heads, so the route only engages for Dk = Dv = 128.

Optional route (``_IMPL = "chunked"``): the FLA chunked WY-representation
kernels — accuracy-validated but slower than the stock kernel E2E; kept for
future iteration.

This rebinds ``gated_delta_update`` in ``mlx_vlm.models.qwen3_5.language`` for
scalar-gated prefill with T >= ``_MIN_T``. Decode (T==1) and
masked/vectorized paths keep the original kernel.
"""

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
# Prefill kernel: "pipelined" | "blocked_seq" | "chunked".
_IMPL = "pipelined"
# Minimum prefill length that takes the Metal kernel.
_MIN_T = 64


def apply_qwen35_gdn_prefill_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    if not mx.metal.is_available():
        return False

    try:
        from mlx_vlm.models.qwen3_5 import gated_delta as gd
        from mlx_vlm.models.qwen3_5 import language as lang
    except ImportError:
        logger.debug("mlx_vlm qwen3_5 not importable; GDN prefill patch skipped")
        return False

    original = gd.gated_delta_update

    from omlx.custom_kernels.qwen35_prefill import (
        gated_delta_blocked_seq,
        gated_delta_chunked_metal,
        gated_delta_pipelined,
    )

    kernels = {
        "pipelined": gated_delta_pipelined,
        "chunked": gated_delta_chunked_metal,
        "blocked_seq": gated_delta_blocked_seq,
    }
    fast_prefill = kernels[_IMPL]

    def gated_delta_update_metal(
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        state=None,
        mask=None,
        use_kernel=True,
        state_steps=None,
        cache=None,
        cache_index=1,
    ):
        if cache is not None or state_steps is not None:
            return original(
                q,
                k,
                v,
                a,
                b,
                A_log,
                dt_bias,
                state,
                mask,
                use_kernel=use_kernel,
                state_steps=state_steps,
                cache=cache,
                cache_index=cache_index,
            )
        if (
            use_kernel
            and mask is None
            and q.shape[1] >= _MIN_T
            # Both the chunked kernel (A) and the default blocked_seq
            # kernel (S) hard-assume Dk=128/Dv=128 internally; this gate
            # used to admit any multiple of 16/32, which would silently
            # misbehave rather than error on other head dims that satisfy
            # the modulus but not the hard-coded 128 assumption.
            # See docs/qwen35-hardening-and-optimization.md E2.
            and q.shape[-1] == 128
            and v.shape[-1] == 128
            and a.ndim == 3  # scalar per-head gating
        ):
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
        _IMPL,
        _MIN_T,
    )
    return True


def apply_qwen35_gdn_chunked_patch() -> bool:
    """Backward-compatible name for older callers/configuration."""
    return apply_qwen35_gdn_prefill_patch()
