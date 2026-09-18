# SPDX-License-Identifier: Apache-2.0
"""Causal multi-token verification without splitting padded batch rows.

The kernel extends mlx-vlm Qwen3.5's one-pass ragged SDPA: each threadgroup
handles one (request, head, query token), retaining its FP32 reduction order.
The physical KV stride stays unchanged while the causal endpoint varies with
query position. Padding remains on the GPU after vector cache rollback.
"""

from functools import cache

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache as LMBatchKVCache
from mlx_lm.models.cache import KVCache as LMKVCache
from mlx_vlm.models.cache import BatchKVCache, KVCache

_SOURCE = r"""

    uint q_batch_head_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D_SIZE / BD;
    constexpr int v_per_thread = V_SIZE / BD;

    typedef float U;
    thread U q[qk_per_thread];
    thread U k[qk_per_thread];
    thread U o[v_per_thread];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    int K_SIZE = int(k_size[0]);
    int query_token = int(q_batch_head_idx) % QUERY_T;
    int batch_idx = int(q_batch_head_idx) / (NUM_Q_HEADS * QUERY_T);
    int q_head_idx = (int(q_batch_head_idx) / QUERY_T) % NUM_Q_HEADS;
    int kv_head_idx = q_head_idx / GQA_FACTOR;
    int pad = int(pads[batch_idx]);
    int N = K_SIZE - pad - QUERY_T + query_token + 1;

    const device T* qptr =
        queries + int(q_batch_head_idx) * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* kptr =
        keys + (batch_idx * NUM_KV_HEADS + kv_head_idx) * K_SIZE * D_SIZE +
        (pad + int(simd_gid)) * D_SIZE + int(simd_lid) * qk_per_thread;
    const device T* vptr =
        values + (batch_idx * NUM_KV_HEADS + kv_head_idx) * K_SIZE * V_SIZE +
        (pad + int(simd_gid)) * V_SIZE + int(simd_lid) * v_per_thread;
    device T* optr =
        out + int(q_batch_head_idx) * V_SIZE + int(simd_gid) * v_per_thread;

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; i++) {
        q[i] = s * qptr[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
        o[i] = 0;
    }

    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    for (int i = int(simd_gid); i < N; i += BN) {
        if constexpr (HAS_MASK) {
            int mask_idx = (((MASK_B == 1 ? 0 : batch_idx) * MASK_H +
                             (MASK_H == 1 ? 0 : q_head_idx)) * MASK_T +
                            (MASK_T == 1 ? 0 : query_token)) * K_SIZE + pad + i;
            if (!mask[mask_idx]) {
                kptr += BN * D_SIZE;
                vptr += BN * V_SIZE;
                continue;
            }
        }
        for (int j = 0; j < qk_per_thread; j++) {
            k[j] = kptr[j];
        }

        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * k[j];
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * vptr[j];
        }

        kptr += BN * D_SIZE;
        vptr += BN * V_SIZE;
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (int i = 0; i < v_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; i++) {
            optr[i] = static_cast<T>(o[i]);
        }
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="omlx_qwen_verify_ragged_sdpa",
        input_names=["queries", "keys", "values", "pads", "scale", "k_size", "mask"],
        output_names=["out"],
        header="#include <metal_simdgroup>\nusing namespace metal;\n",
        source=_SOURCE,
    )


def verify_attention(queries, keys, values, *, cache, scale, mask):
    """Return bounded causal attention, or None for other contracts."""
    if (
        not (
            type(cache) in (BatchKVCache, KVCache, LMBatchKVCache, LMKVCache)
            or getattr(type(cache), "_omlx_mtp_verify_attention_cache", False)
        )
        or queries.ndim != 4
        or keys.ndim != 4
        or values.ndim != 4
        or queries.shape[2] <= 1
        or queries.dtype not in (mx.float16, mx.bfloat16)
        or keys.dtype != queries.dtype
        or values.dtype != queries.dtype
    ):
        return None
    batch, heads, length, dim = queries.shape
    kv_batch, kv_heads, size, kv_dim = keys.shape
    # Keep singleton verification and contexts beyond the tested range on
    # the existing upstream attention path.
    if (
        (size >= 1024 and (batch == 1 or size > 8192))
        or size < length
        or dim not in (64, 96, 128, 256)
        or kv_dim != dim
        or kv_batch != batch
        or kv_heads == 0
        or heads % kv_heads
        or values.shape != keys.shape
    ):
        return None
    pads = getattr(cache, "left_padding", None)
    if pads is None:
        pads = mx.zeros((batch,), dtype=mx.int32)
    if pads.shape != (batch,):
        return None

    has_mask = isinstance(mask, mx.array)
    if has_mask:
        if mask.dtype != mx.bool_ or not 2 <= mask.ndim <= 4:
            return None
        mask = mask.reshape((1,) * (4 - mask.ndim) + mask.shape)
        if any(m not in (1, q) for m, q in zip(mask.shape[:3], queries.shape[:3])):
            return None
        if mask.shape[-1] != size:
            return None
        mask_dims = mask.shape[:3]
    else:
        if mask is not None and mask != "causal":
            return None
        mask_dims = (1, 1, 1)
        mask = mx.array([True])

    return _kernel()(
        inputs=[
            mx.contiguous(queries),
            mx.contiguous(keys),
            mx.contiguous(values),
            pads.astype(mx.int32),
            mx.array([scale]),
            mx.array([size], dtype=mx.int32),
            mx.contiguous(mask),
        ],
        template=[
            ("T", queries.dtype),
            ("D_SIZE", dim),
            ("V_SIZE", dim),
            ("NUM_Q_HEADS", heads),
            ("NUM_KV_HEADS", kv_heads),
            ("GQA_FACTOR", heads // kv_heads),
            ("QUERY_T", length),
            ("HAS_MASK", has_mask),
            ("MASK_B", mask_dims[0]),
            ("MASK_H", mask_dims[1]),
            ("MASK_T", mask_dims[2]),
        ],
        grid=(1024, batch * heads * length, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]


def apply(language):
    if getattr(language, "_omlx_verify_attention", False):
        return
    original = language._qwen3_5_left_padded_attention

    def attention(queries, keys, values, *, cache, scale, mask):
        result = verify_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )
        if result is not None:
            return result
        # The upstream ragged helper ignores explicit masks.
        if isinstance(mask, mx.array):
            return None
        return original(queries, keys, values, cache=cache, scale=scale, mask=mask)

    language._qwen3_5_left_padded_attention = attention
    language._omlx_verify_attention = True
