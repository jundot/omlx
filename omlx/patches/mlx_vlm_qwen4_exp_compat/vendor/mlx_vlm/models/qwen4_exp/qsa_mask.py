# SPDX-License-Identifier: Apache-2.0
"""Exact expansion of selected QSA blocks and the incomplete causal tail."""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)
_KERNEL = None
_PROVEN = False
_FAILED = False


def _launch_mask(hits, counts, ends, ratio, topk, key_len):
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_qwen4_block_mask",
            input_names=["hits", "counts", "ends", "length"],
            output_names=["out"],
            source=r"""
                const uint i = thread_position_in_grid.x;
                const uint n = length[0];
                const uint s = hits_shape[1];
                const uint blocks = hits_shape[2];
                if (i >= uint(hits_shape[0]) * s * n) return;
                const uint row = i / n;
                const uint q = row % s;
                const uint token = i % n;
                const uint end = uint(ends[q]);
                const uint complete = uint(counts[q]);
                if (complete <= TOPK) {
                    out[i] = token < end;
                } else {
                    const uint block = token / RATIO;
                    const bool hit = block < blocks && hits[row * blocks + block];
                    const bool tail = token >= complete * RATIO && token < end;
                    out[i] = hit || tail;
                }
            """,
            ensure_row_contiguous=True,
        )
    batch, seq, _ = hits.shape
    return _KERNEL(
        inputs=[hits, counts, ends, mx.array([key_len], dtype=mx.uint32)],
        template=[("RATIO", ratio), ("TOPK", topk)],
        grid=(batch * seq * key_len, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, 1, seq, key_len)],
        output_dtypes=[mx.bool_],
    )[0]


def fused_block_mask(hits, counts, ends, ratio, topk, key_len):
    """Expand selected blocks and the causal tail in one Metal kernel.

    Selection is already complete. This only replaces repeat/pad/tail/where
    operations, with the same boolean result even when the hits contain blocks
    beyond a query's causal endpoint. Key length is a runtime input so growing
    the cache cannot create a new shader specialization for every token.
    """
    global _PROVEN, _FAILED
    if (
        _FAILED
        or hits.ndim != 3
        or hits.shape[0] != 1
        or not 1 <= hits.shape[1] <= 6
        or hits.dtype != mx.bool_
        or counts.dtype != mx.int32
        or ends.dtype != mx.int32
        or counts.shape != (hits.shape[1],)
        or ends.shape != counts.shape
        or ratio != 4
        or topk != 512
        or not 2048 < key_len <= 32768
        or hits.shape[-1] != key_len // ratio
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
    ):
        return None
    try:
        output = _launch_mask(hits, counts, ends, ratio, topk, key_len)
        if not _PROVEN:
            mx.eval(output)
            _PROVEN = True
        return output
    except Exception as exc:
        # The kernel writes only a fresh mask. The caller can reuse its hits
        # in the general implementation without updating any cache again.
        _FAILED = True
        logger.warning("Qwen4 fused mask disabled after kernel failure: %s", exc)
        return None
