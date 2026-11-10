# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-invariant short-row attention for embedded DeepSeek DSpark.

DeepSeek-V4 uses 64 query heads and one KV head.  MLX collapses those heads
into GEMM's M dimension for ordinary one-token decode, but a DSpark verify
batch with one physical KV view per row takes the batched GEMV route instead.
The two routes are mathematically equivalent but not bitwise equivalent.

These kernels keep the per-head reduction fixed for both decode and verify.
Multiple verify rows share one dispatch while retaining independent KV views.
"""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx


_ROWWISE_KERNEL_EXACT: bool | None = None


def _rowwise_fallback(lhs: mx.array, rhs: mx.array, transpose_rhs: bool) -> mx.array:
    return mx.concatenate(
        [
            lhs[idx : idx + 1]
            @ (
                rhs[idx : idx + 1].swapaxes(-1, -2)
                if transpose_rhs
                else rhs[idx : idx + 1]
            )
            for idx in range(lhs.shape[0])
        ]
    )


def _rowwise_kernel_is_exact(fast) -> bool:
    """Probe once whether the packed kernel reproduces MLX's one-row matmul bitwise.

    The verify path promises decode-exact rows. On some GPUs (seen on the
    80-core M5 Ultra, ``applegpu_g17d``) the Steel grid kernel and the stock
    one-row matmul pick different reduction orders and differ by one bf16
    ulp in ~0.1-0.4 % of elements; there the fallback is the exact path.
    """
    global _ROWWISE_KERNEL_EXACT
    if _ROWWISE_KERNEL_EXACT is None:
        try:
            state = mx.random.key(811)
            k1, k2 = mx.random.split(state)
            lhs = mx.random.normal((3, 64, 512), key=k1).astype(mx.bfloat16)
            rhs = mx.random.normal((3, 156, 512), key=k2).astype(mx.bfloat16)
            expected = _rowwise_fallback(lhs, rhs, True)
            got = fast.rowwise_gemm(mx.contiguous(lhs), mx.contiguous(rhs), True)
            _ROWWISE_KERNEL_EXACT = bool(mx.array_equal(expected, got).item())
        except Exception:
            _ROWWISE_KERNEL_EXACT = False
    return _ROWWISE_KERNEL_EXACT


_RING_KERNEL_EXACT: bool | None = None


def ring_kernel_is_exact(fast) -> bool:
    """Probe once whether ``fast.ring_gemm`` equals the exact materialized rowwise path bitwise.

    Same contract as ``_rowwise_kernel_is_exact``: the ring kernel is only used
    where it reproduces one-row decode arithmetic. Must be called outside
    ``mx.compile`` tracing (it evaluates the probe).
    """
    global _RING_KERNEL_EXACT
    if _RING_KERNEL_EXACT is None:
        try:
            if not fast.has_symbol("ring_gemm"):
                _RING_KERNEL_EXACT = False
            else:
                rows = 3
                k1, k2 = mx.random.split(mx.random.key(83))
                source = mx.random.normal((128 + rows, 512), key=k1).astype(mx.bfloat16)
                index_rows = []
                for row in range(rows):
                    snapshot = list(range(128))
                    for update in range(row + 1):
                        snapshot[(123 + update) % 128] = 128 + update
                    index_rows.append(snapshot)
                indices = mx.array(index_rows, dtype=mx.uint32)
                queries = mx.random.normal((rows, 64, 512), key=k2).astype(mx.bfloat16)
                gathered = mx.take(source, indices, axis=0)
                expected = rowwise_gemm(queries, gathered, True)
                got = fast.ring_gemm(
                    mx.contiguous(queries), mx.contiguous(source), mx.contiguous(indices), True
                )
                _RING_KERNEL_EXACT = bool(mx.array_equal(expected, got).item())
        except Exception:
            _RING_KERNEL_EXACT = False
    return _RING_KERNEL_EXACT


def ring_gemm(lhs: mx.array, source: mx.array, indices: mx.array, transpose_rhs: bool) -> mx.array:
    """Per-row GEMM against ring snapshots ``source[indices[row]]``, decode-exact.

    Uses the packed ring kernel only where it is bitwise equal to the exact
    rowwise path; otherwise materializes the snapshots and runs ``rowwise_gemm``.
    """
    try:
        from omlx.custom_kernels.attention import fast

        if ring_kernel_is_exact(fast):
            return fast.ring_gemm(
                mx.contiguous(lhs), mx.contiguous(source), mx.contiguous(indices), transpose_rhs
            )
    except Exception:
        pass
    return rowwise_gemm(lhs, mx.take(source, indices, axis=0), transpose_rhs)


def steel_rowwise_available() -> bool:
    """True when the native Steel rowwise GEMM is bitwise-safe here.

    The Steel rowwise kernel reproduces stock ``@`` bits on the GPU
    generations the suffix gate names (``d``/``g``/``p``), but not on
    newer matrix-math routes (e.g. applegpu_g17s, where MLX bf16 matmul
    accumulates in a different order).  Every kernel that promises bitwise
    parity with the materialized rowwise path — the DSpark ring GEMM
    included — must stay off wherever this returns False, or verify rows
    drift from decode bits by one ULP. A one-time probe also checks the bits,
    since the 80-core M5 Ultra (applegpu_g17d) passes the suffix gate but
    not the probe.
    """
    try:
        architecture = str(mx.device_info().get("architecture", ""))
        if not architecture.endswith(("d", "g", "p")):
            return False
        from omlx.custom_kernels.attention import fast

        return fast.has_symbol("rowwise_gemm") and _rowwise_kernel_is_exact(fast)
    except Exception:
        return False


def rowwise_gemm(lhs: mx.array, rhs: mx.array, transpose_rhs: bool) -> mx.array:
    """Dispatch independent M=64 Steel GEMMs in one grid when available."""
    if steel_rowwise_available():
        from omlx.custom_kernels.attention import fast

        # The native kernel rejects other layouts, such as TP-sharded heads.
        try:
            return fast.rowwise_gemm(
                mx.contiguous(lhs),
                mx.contiguous(rhs),
                transpose_rhs,
            )
        except Exception:
            pass
    return mx.concatenate(
        [
            lhs[idx : idx + 1]
            @ (
                rhs[idx : idx + 1].swapaxes(-1, -2)
                if transpose_rhs
                else rhs[idx : idx + 1]
            )
            for idx in range(lhs.shape[0])
        ]
    )


def exact_attention(
    queries: mx.array,
    key_rows: Sequence[mx.array],
    scale: float,
    sinks: mx.array | None,
) -> mx.array:
    """Attend M query rows to M same-length KV snapshots.

    ``queries`` is ``[1, H, M, D]`` and every key row is ``[1, 1, S, D]``.
    The return layout is ``[1, H, M, D]``.  Different key lengths are kept
    correct with small same-length groups rather than padding the softmax.
    """
    if len(key_rows) != int(queries.shape[2]):
        raise ValueError("one DeepSeek KV view is required for each query row")

    scaled_rows = queries * mx.array(scale, dtype=queries.dtype)
    outputs = []
    start = 0
    while start < len(key_rows):
        stop = start + 1
        group_length = int(key_rows[start].shape[2])
        while (
            stop < len(key_rows)
            and int(key_rows[stop].shape[2]) == group_length
        ):
            stop += 1

        query_batch = scaled_rows[0, :, start:stop].transpose(1, 0, 2)
        keys = mx.concatenate(key_rows[start:stop], axis=0)[:, 0]
        scores = rowwise_gemm(query_batch, keys, True)

        if sinks is None:
            weights = mx.softmax(scores, axis=-1, precise=True)
        else:
            sink_scores = mx.broadcast_to(
                sinks.astype(scores.dtype)[None, :, None],
                (*scores.shape[:-1], 1),
            )
            weights = mx.softmax(
                mx.concatenate([sink_scores, scores], axis=-1),
                axis=-1,
                precise=True,
            )[..., 1:]
        group_output = rowwise_gemm(weights, keys, False)
        outputs.append(group_output.transpose(1, 0, 2)[None])
        start = stop

    return outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)


def exact_local_scores(
    queries: mx.array,
    key_rows: Sequence[mx.array],
    scale: float,
) -> mx.array:
    """Return batch-invariant local scores for sparse split attention."""
    if len(key_rows) != int(queries.shape[2]):
        raise ValueError("one DeepSeek KV view is required for each query row")
    if any(int(row.shape[2]) != int(key_rows[0].shape[2]) for row in key_rows):
        raise ValueError("sparse local KV views must have equal lengths")
    query_batch = (
        queries * mx.array(scale, dtype=queries.dtype)
    )[0].transpose(1, 0, 2)
    keys = mx.concatenate(key_rows, axis=0)[:, 0]
    return rowwise_gemm(query_batch, keys, True).transpose(1, 0, 2)[None]


def exact_local_values(weights: mx.array, key_rows: Sequence[mx.array]) -> mx.array:
    """Apply sparse local weights with the same decode/verify reduction."""
    weight_rows = weights[0].transpose(1, 0, 2)
    keys = mx.concatenate(key_rows, axis=0)[:, 0]
    return rowwise_gemm(weight_rows, keys, False).transpose(1, 0, 2)[None]


__all__ = [
    "exact_attention",
    "exact_local_scores",
    "exact_local_values",
    "rowwise_gemm",
]
