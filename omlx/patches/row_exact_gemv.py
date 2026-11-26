# SPDX-License-Identifier: Apache-2.0
"""Multi-row dense projection with one-row decode arithmetic per row.

A decode step's ``x @ w.T`` runs MLX 0.32.4's one-row ``gemv`` kernel, while
more rows take ``gemv_wide`` or a GEMM tile that sums in another order. Here
every row follows the one-row ``gemv`` traversal of a row-major weight (BN 1,
SN 32, TM 4, TN 4): each lane accumulates 4 consecutive inputs of every
128-wide K block into 4 output columns, then a ``simd_shuffle_down`` tree adds
the 32 lanes. ``gemv`` uses that traversal when K is above 64 and below 16 N;
this kernel also needs K to be a multiple of 128 (no partial block). Row ``r``
then equals ``x[r:r+1] @ w.T`` bit for bit.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

MAX_ROWS = 16

# A simdgroup computes 4 output columns for up to 4 rows: 16 accumulators,
# inside the envelope where unrolled row-exact tiles compiled correctly (see
# row_exact_qmv). The row groups of a column tile share a threadgroup, so the
# weight tile is fetched from memory once for all rows. gemv shifts a partial
# last column tile inward (out_row = N - 4); the overlap recomputes the same
# columns the same way.
_SOURCE = r"""
    const int n = w_shape[0];
    const int k = w_shape[1];
    const int rows = x_shape[0];
    const int lane = int(thread_index_in_simdgroup);
    const int r0 = int(thread_position_in_grid.z) * 4;
    int out_row = int(thread_position_in_grid.y) * 4;
    if (out_row >= n) {
        return;
    }
    out_row = out_row + 4 <= n ? out_row : n - 4;
    const device T* mat = w + size_t(out_row) * k;
    const device T* xr = x + size_t(r0) * k;

    float result[4][4];
    UNROLL for (int r = 0; r < 4; r++) {
        UNROLL for (int tm = 0; tm < 4; tm++) {
            result[r][tm] = 0;
        }
    }
    for (int bn = lane * 4; bn < k; bn += 128) {
        T inter[4][4];
        UNROLL for (int tm = 0; tm < 4; tm++) {
            UNROLL for (int tn = 0; tn < 4; tn++) {
                inter[tm][tn] = mat[tm * k + bn + tn];
            }
        }
        UNROLL for (int r = 0; r < 4; r++) {
            if (r0 + r < rows) {
                float v[4];
                UNROLL for (int tn = 0; tn < 4; tn++) {
                    v[tn] = static_cast<float>(xr[r * k + bn + tn]);
                }
                UNROLL for (int tm = 0; tm < 4; tm++) {
                    UNROLL for (int tn = 0; tn < 4; tn++) {
                        result[r][tm] += inter[tm][tn] * v[tn];
                    }
                }
            }
        }
    }
    UNROLL for (int r = 0; r < 4; r++) {
        if (r0 + r < rows) {
            UNROLL for (int tm = 0; tm < 4; tm++) {
                for (ushort sn = 16; sn >= 1; sn >>= 1) {
                    result[r][tm] += simd_shuffle_down(result[r][tm], sn);
                }
                if (lane == 0) {
                    y[(r0 + r) * n + out_row + tm] = static_cast<T>(result[r][tm]);
                }
            }
        }
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="omlx_row_exact_gemv",
        input_names=["x", "w"],
        output_names=["y"],
        header='#define UNROLL _Pragma("clang loop unroll(full)")\n',
        source=_SOURCE,
    )


def linear(weight: mx.array, x: mx.array) -> mx.array | None:
    """``x @ weight.T`` for ``x`` [..., K] and a row-major ``weight`` [N, K],
    each row with the bits of its own one-row product. None for layouts the
    kernel does not cover."""
    n, k = weight.shape
    rows = x.size // k
    if (
        weight.dtype not in (mx.bfloat16, mx.float16)
        or x.dtype != weight.dtype
        or k % 128
        or k >= 16 * n
        or n < 4
        or not 1 < rows <= MAX_ROWS
        or mx.default_device() != mx.gpu
    ):
        return None
    groups = (rows + 3) // 4
    y = _kernel()(
        inputs=[x.reshape(rows, k), weight],
        template=[("T", x.dtype)],
        grid=(32, (n + 3) // 4, groups),
        # 8 simdgroups (6 for three row groups) per threadgroup.
        threadgroup=(32, 8 // groups, groups),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )[0]
    return y.reshape(*x.shape[:-1], n)
