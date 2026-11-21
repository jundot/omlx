# SPDX-License-Identifier: Apache-2.0
"""Row-chunked verify attention for Flint.

Flint has 32 query heads on 4 KV heads. MLX's SDPA vector kernel serves
``q_len * gqa <= 32``, so it takes at most four Flint query rows; a DFlash
verify block of five to eight rows falls to MLX's other paths. On M1 Max
(bf16, with Flint's attention sinks), 49 layers of 5-8 rows take 6.9 ms at
512 keys against 3.3 ms for 4 rows, and ~37 ms against 13 ms at 4096 keys.

This patch runs multi-row calls through ``row_exact_causal_sdpa``: chunks of
at most four rows on the vector kernel, each row on the kernel plan of its
own one-row decode, so verify rows also match plain decode. Prefill-width
calls, quantized KV caches and non-causal blocks keep mlx-vlm's path.
"""

from __future__ import annotations

import logging

import mlx.core as mx

from .qwen35_verify_sdpa_split import _VECTOR_ROW_BUDGET, row_exact_causal_sdpa

logger = logging.getLogger(__name__)

_MIN_ROWS = 2
# Verify blocks; prompt chunks are far wider and stay on the stock kernels.
_MAX_ROWS = 16


def _routed(queries, keys, cache, mask) -> bool:
    rows = queries.shape[-2]
    return (
        _MIN_ROWS <= rows <= _MAX_ROWS
        and queries.shape[0] == 1
        and not hasattr(cache, "bits")
        and isinstance(keys, mx.array)
        and keys.dtype == queries.dtype
        # The chunks rebuild causality from the block's bottom-right
        # alignment, or slice an array keep-mask that already holds it.
        and (isinstance(mask, mx.array) or mask == "causal")
    )


def apply_flint_verify_attention_patch() -> bool:
    """Route Flint's multi-row attention through row-chunked vector SDPA."""
    if not mx.metal.is_available():
        return False
    try:
        from mlx_vlm.models.flint import language as flint
    except ImportError:
        return False
    if getattr(flint, "_omlx_flint_verify_attention", False):
        return True
    original = flint.scaled_dot_product_attention

    def scaled_dot_product_attention(
        queries, keys, values, cache, scale, mask, sinks=None
    ):
        if not _routed(queries, keys, cache, mask):
            return original(
                queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks
            )
        group = queries.shape[-3] // keys.shape[-3]
        return row_exact_causal_sdpa(
            queries,
            keys,
            values,
            scale,
            max(1, _VECTOR_ROW_BUDGET // group),
            mask=mask if isinstance(mask, mx.array) else None,
            sinks=sinks,
        )

    flint.scaled_dot_product_attention = scaled_dot_product_attention
    flint._omlx_flint_verify_attention = True
    logger.info("Flint row-chunked verify attention patch applied")
    return True
