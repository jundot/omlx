"""Tensor-unit (NAX) DSA indexer scores for GLM-5.3 prefill.

The GLM-5.3 lightning indexer scores every query against every pooled key:

    score[s, p] = sum_h relu(q[s, h] . k[p]) * w[s, h]      (32 heads x 128)

The native ``dsa_indexer_scores`` kernel runs this on the classic SIMD
matrix units, reloading each key tile once per head, at ~13 TFLOPS; its
cost grows with the context (O(L x P)) and is a main part of GLM's
long-context prefill taper. This module runs the same math on the M5
tensor units: per 32x32 output block a simdgroup computes each head's
[32 x 128] x [128 x 32] product with NAX matmuls (fp32 accumulation),
applies relu and the head weight in fp32 in head order, and writes the
bf16 score once, with the pooled causal mask folded into the epilogue
(masked -> bf16(-1e30), exactly what the call site's ``mx.where`` wrote).
Blocks that are entirely masked skip the matmuls.

Only the fp32 summation order of each 128-wide dot product differs from
the native kernel, so scores agree to bf16 rounding (a few ulp at most),
and top-k selection can only differ at exact near-ties.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import mlx.core as mx

_ENV = os.environ.get("OMLX_GLM_DSA_INDEXER_NAX", "1").strip().lower()
_ENABLED = _ENV not in {"0", "false", "off"}

# Rows per call: bounds the [rows, P] bf16 score buffer (256 MiB).
_MAX_SCORE_ELEMENTS = 1 << 27

_HEAD_DIM = 128

_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
"""

# One simdgroup owns a 32 (queries) x 32 (pooled keys) output block as 2x2
# 16x16 fragments in the MLX NAX fragment layout: lane -> rows fm, fm + 8
# and columns fn .. fn + 3 of each fragment.
_SOURCE = """
    constexpr int D = 128;
    const int S = params[0];
    const int P = params[1];
    const int before = params[2];
    const int pool_len = params[3];
    const int ratio = params[4];
    const int H = params[5];
    const int q_row_stride = H * D;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int sm = int(threadgroup_position_in_grid.y) * (32 * WM) + int(sg / WN) * 32;
    const int sn = int(threadgroup_position_in_grid.x) * (32 * WN) + int(sg % WN) * 32;

    const short qid = short(lane >> 2);
    const short fm = short((qid & 4) | ((lane >> 1) & 3));
    const short fn = short(((qid & 2) | (lane & 1)) * 4);

    // Every score of the block is masked when its first pooled key is not
    // yet complete for the block's last query row.
    const int s_last = min(sm + 31, S - 1);
    const bool live = sm < S && sn < P && sn < pool_len &&
        (sn + 1) * ratio - 1 <= before + s_last;

    float acc[2][16];
    for (short a = 0; a < 2; ++a) {
        for (short e = 0; e < 16; ++e) {
            acc[a][e] = 0.0f;
        }
    }

    int qrow[2][2];
    for (short tm = 0; tm < 2; ++tm) {
        for (short i = 0; i < 2; ++i) {
            qrow[tm][i] = min(sm + tm * 16 + fm + i * 8, S - 1);
        }
    }

    if (live) {
        constexpr auto desc = matmul2d_descriptor(
            16, 32, 16, false, true, true,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroup> op;

        int krow[2][2];
        for (short tn = 0; tn < 2; ++tn) {
            for (short i = 0; i < 2; ++i) {
                krow[tn][i] = min(sn + tn * 16 + fm + i * 8, P - 1);
            }
        }
        const device T* kbase[2][2];
        for (short tn = 0; tn < 2; ++tn) {
            for (short i = 0; i < 2; ++i) {
                kbase[tn][i] = k + ulong(krow[tn][i]) * D + fn;
            }
        }

        for (int h = 0; h < H; ++h) {
            auto ct_a0 = op.template get_left_input_cooperative_tensor<T, T, float>();
            auto ct_a1 = op.template get_left_input_cooperative_tensor<T, T, float>();
            auto ct_b = op.template get_right_input_cooperative_tensor<T, T, float>();
            auto c0 = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ct_a0)>,
                metal::remove_addrspace_t<decltype(ct_b)>, float>();
            auto c1 = op.template get_destination_cooperative_tensor<
                metal::remove_addrspace_t<decltype(ct_a0)>,
                metal::remove_addrspace_t<decltype(ct_b)>, float>();
            for (short e = 0; e < 16; ++e) {
                c0[e] = 0.0f;
                c1[e] = 0.0f;
            }
            const device T* a00 = q + ulong(qrow[0][0]) * q_row_stride + h * D + fn;
            const device T* a01 = q + ulong(qrow[0][1]) * q_row_stride + h * D + fn;
            const device T* a10 = q + ulong(qrow[1][0]) * q_row_stride + h * D + fn;
            const device T* a11 = q + ulong(qrow[1][1]) * q_row_stride + h * D + fn;
            for (short kk = 0; kk < D; kk += 16) {
                for (short j = 0; j < 4; ++j) {
                    ct_a0[j] = a00[kk + j];
                    ct_a0[4 + j] = a01[kk + j];
                    ct_a1[j] = a10[kk + j];
                    ct_a1[4 + j] = a11[kk + j];
                }
                for (short tn = 0; tn < 2; ++tn) {
                    for (short i = 0; i < 2; ++i) {
                        for (short j = 0; j < 4; ++j) {
                            ct_b[tn * 8 + i * 4 + j] = kbase[tn][i][kk + j];
                        }
                    }
                }
                op.run(ct_a0, ct_b, c0);
                op.run(ct_a1, ct_b, c1);
            }
            // relu * head weight in fp32, heads accumulated in order.
            for (short i = 0; i < 2; ++i) {
                const float w0 = float(w[ulong(qrow[0][i]) * H + h]);
                const float w1 = float(w[ulong(qrow[1][i]) * H + h]);
                for (short tn = 0; tn < 2; ++tn) {
                    for (short j = 0; j < 4; ++j) {
                        const short e = tn * 8 + i * 4 + j;
                        acc[0][e] += max(float(c0[e]), 0.0f) * w0;
                        acc[1][e] += max(float(c1[e]), 0.0f) * w1;
                    }
                }
            }
        }
    }

    const T masked = T(-1e30f);
    for (short tm = 0; tm < 2; ++tm) {
        for (short i = 0; i < 2; ++i) {
            const int s = sm + tm * 16 + fm + i * 8;
            if (s >= S) {
                continue;
            }
            for (short tn = 0; tn < 2; ++tn) {
                for (short j = 0; j < 4; ++j) {
                    const int p = sn + tn * 16 + fn + j;
                    if (p >= P) {
                        continue;
                    }
                    const bool valid = p < pool_len && (p + 1) * ratio - 1 <= before + s;
                    out[ulong(s) * P + p] = valid ? T(acc[tm][tn * 8 + i * 4 + j]) : masked;
                }
            }
        }
    }
"""

_KERNEL = None
_WM = 2
_WN = 2


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="omlx_glm_dsa_indexer_scores_nax",
            input_names=["q", "k", "w", "params"],
            output_names=["out"],
            header=_HEADER,
            source=_SOURCE,
        )
    return _KERNEL


@lru_cache(maxsize=1)
def nax_indexer_available() -> bool:
    if not _ENABLED:
        return False
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


def max_rows_per_call(pool_width: int) -> int:
    """Largest multiple of 64 query rows whose score buffer fits the cap."""
    rows = _MAX_SCORE_ELEMENTS // max(int(pool_width), 1)
    return max(64, (rows // 64) * 64)


def indexer_scores_nax(
    q: mx.array,
    pool_keys: mx.array,
    weights: mx.array,
    before: int,
    pool_len: int,
    ratio: int,
) -> Optional[mx.array]:
    """Masked head-summed indexer scores on the tensor units.

    q: [S, H, 128] (row-major), pool_keys: [P, 128], weights: [S, H] already
    scaled, all bf16. Query row ``s`` sits at absolute position
    ``before + s``; pooled key ``p`` is visible to it iff ``p < pool_len``
    and ``(p + 1) * ratio - 1 <= before + s``. Returns [S, P] bf16 with
    masked entries set to bf16(-1e30), or None for unsupported inputs.
    """
    if (
        q.ndim != 3
        or pool_keys.ndim != 2
        or weights.ndim != 2
        or q.shape[-1] != _HEAD_DIM
        or pool_keys.shape[-1] != _HEAD_DIM
        or weights.shape != q.shape[:2]
        or q.dtype != mx.bfloat16
        or pool_keys.dtype != mx.bfloat16
        or weights.dtype != mx.bfloat16
        or ratio < 1
    ):
        return None
    S, H, _ = q.shape
    P = pool_keys.shape[0]
    if S == 0 or P == 0:
        return None
    params = mx.array([S, P, int(before), int(pool_len), int(ratio), H], dtype=mx.int32)
    tg_x = (P + 32 * _WN - 1) // (32 * _WN)
    tg_y = (S + 32 * _WM - 1) // (32 * _WM)
    return _kernel()(
        inputs=[q, pool_keys, weights, params],
        template=[("T", mx.bfloat16), ("WM", _WM), ("WN", _WN)],
        grid=(tg_x * _WM * _WN * 32, tg_y, 1),
        threadgroup=(_WM * _WN * 32, 1, 1),
        output_shapes=[(S, P)],
        output_dtypes=[mx.bfloat16],
    )[0]
