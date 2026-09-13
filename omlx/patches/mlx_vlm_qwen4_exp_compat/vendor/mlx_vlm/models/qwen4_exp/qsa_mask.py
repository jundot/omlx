# SPDX-License-Identifier: Apache-2.0
"""Exact expansion of selected QSA blocks and the incomplete causal tail."""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)
_KERNEL = None
_PROVEN = False
_FAILED = False
_INELIGIBLE_LOGGED: set[str] = set()


def _ineligible(reason):
    if reason not in _INELIGIBLE_LOGGED:
        _INELIGIBLE_LOGGED.add(reason)
        logger.debug("Qwen4 fused mask not used (%s); using general expansion", reason)
    return None


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
                    // Selection currently excludes future blocks. Also enforce
                    // causality here so a selection change cannot expose them.
                    const bool hit = token < end && block < blocks
                        && hits[row * blocks + block];
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
    operations for causal block selections. Future hits are additionally
    clamped to the query endpoint. Key length is a runtime input so growing
    the cache cannot create a new shader specialization for every token.
    """
    global _PROVEN, _FAILED
    if _FAILED:
        return None
    if hits.ndim != 3 or hits.shape[0] != 1:
        return _ineligible("hits must have shape (1, queries, blocks)")
    if not 1 <= hits.shape[1] <= 6:
        return _ineligible("query count must be 1..6")
    if hits.dtype != mx.bool_ or counts.dtype != mx.int32 or ends.dtype != mx.int32:
        return _ineligible("hits/counts/ends require bool/int32/int32")
    if counts.shape != (hits.shape[1],) or ends.shape != counts.shape:
        return _ineligible("counts and ends must match the query count")
    if ratio != 4 or topk != 512:
        return _ineligible("compression ratio/top-k must be 4/512")
    if not 2048 < key_len <= 32768:
        return _ineligible("key length must be 2049..32768")
    if hits.shape[-1] != key_len // ratio:
        return _ineligible("hit width must match the complete key blocks")
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return _ineligible("requires a Metal GPU device")
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
        logger.warning(
            "Qwen4 fused mask disabled for this process after kernel failure: %s", exc
        )
        return None
