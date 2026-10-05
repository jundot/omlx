# SPDX-License-Identifier: Apache-2.0
"""Metal kernels for Qwen4 dense-arm attention rows (decode and verify).

Below the QSA block budget a decode step (and every row of a row-exact
Lightning MTP verify window) runs plain attention: four projections, q/k
RMS norms, MRoPE, the cache appends, one MLX SDPA per row, the sigmoid gate
and ``o_proj`` -- about twenty dependent launches per layer. The kernels here
replace most of them with the same float operations in the same order:

* ``prep_qk``: the q and k RMS norms (``mx.fast.rms_norm`` with the FP32
  ``1 + weight`` scale, its ``rms_single_row`` reduction lane for lane) and
  mlx-vlm's fused MRoPE kernel, for every head and row in one launch.
* ``dense_sdpa_gate``: MLX 0.32.2's vector SDPA plan of each row -- the
  one-pass kernel below 1024 keys, the two-pass kernel with 128 partitions
  from 1024 keys (Apple 'd' GPUs) -- then MLX's multiply by the gate's
  sigmoid (``mx.sigmoid`` itself: a transcribed exp/divide rounds
  differently). The one-pass kernel's 32 per-simdgroup key chains of a head
  run in separate simdgroups spread over the GPU (MLX keeps them in one
  threadgroup on one core), scoring four keys ahead of the state updates;
  their states meet in a second launch for MLX's closing reduction. Every
  row of a verify window takes the plan its own serial decode step would,
  in one launch pair.

``ready`` runs each specialization once on tiny inputs first; a compile or
launch failure turns the kernels off for the process and callers keep the
MLX path. Disable with OMLX_QWEN4_ATTN_FUSED=0.
"""

from __future__ import annotations

import functools
import logging

import mlx.core as mx

from .hc_projection import env_enabled

logger = logging.getLogger(__name__)

_DISABLED = not env_enabled("OMLX_QWEN4_ATTN_FUSED")
MAX_ROWS = 8
# MLX's vector SDPA: one pass below this many keys on 'd'/'s' GPUs.
_TWO_PASS_KEYS = 1024
# MLX's two-pass partition count for 12 query heads per KV head below 16K keys.
_TWO_PASS_BLOCKS = 128
_TWO_PASS_MAX_KEYS = 16384
# One-pass chains: query heads per simdgroup, simdgroups per threadgroup
# (both chains load four keys ahead: _CHAIN_MACROS).
_ONE_PASS_HPT = 2
_ONE_PASS_SG = 4

_KERNELS: dict[str, object] = {}
_READY: set[tuple] = set()
_SCALARS: dict[tuple, mx.array] = {}

# mlx-vlm's MRoPE kernel (half-split pairs) and MLX's rms_single_row for one
# head of one row: 64 threads, four values each, simd_sum per simdgroup and
# across the two simdgroups' sums (zeros elsewhere), exactly as MLX's
# rms_norm dispatch lays out an axis of 256.
_PREP_SOURCE = r"""
    constexpr int SIMD_SIZE = 32;
    constexpr int N_READS = 4;
    const uint axis_size = D;
    const uint lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const int h = threadgroup_position_in_grid.x;
    const int t = threadgroup_position_in_grid.y;
    const int q_len = threadgroups_per_grid.y;
    const bool is_q = h < QH;

    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[SIMD_SIZE];
    threadgroup T normed[D];

    const device T* x = is_q
        ? q_proj + (size_t)t * (QH * 2 * D) + h * 2 * D
        : k_proj + (size_t)t * (KH * D) + (h - QH) * D;
    const device float* w = is_q ? q_scale : k_scale;

    // mx.fast.rms_norm on the FP32 cast of the projection.
    float acc = 0;
    float thread_x[N_READS];
    x += lid * N_READS;
    w += lid * N_READS;
    for (int i = 0; i < N_READS; i++) {
        thread_x[i] = static_cast<float>(x[i]);
        acc += thread_x[i] * thread_x[i];
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_READS; i++) {
        // The FP32 norm output, then its cast back to the activation dtype.
        const float y = w[i] * static_cast<float>(thread_x[i] * local_inv_mean[0]);
        normed[lid * N_READS + i] = static_cast<T>(y);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // mlx-vlm's MRoPE kernel over [1, heads, q_len, D].
    const int half_dim = ROT / 2;
    device T* x_out = is_q
        ? queries + ((size_t)h * q_len + t) * D
        : keys + ((size_t)(h - QH) * q_len + t) * D;
    for (int slot = int(lid); slot < half_dim + D - ROT; slot += 64) {
        if (slot >= half_dim) {
            const int pass_d = ROT + slot - half_dim;
            x_out[pass_d] = normed[pass_d];
            continue;
        }
        int freq_idx = slot;
        int d = freq_idx;
        int pair_d = d + half_dim;
#if POS_NDIM == 3
        int axis = int(position_selector[freq_idx]);
        float pos = static_cast<float>(position_ids[axis * q_len + t]);
#else
        float pos = static_cast<float>(position_ids[t]);
#endif
        float angle = pos * static_cast<float>(inv_freq[freq_idx]);
        float c = metal::cos(angle);
        float s = metal::sin(angle);
        float xv = static_cast<float>(normed[d]);
        float xp = static_cast<float>(normed[pair_d]);
        x_out[d] = static_cast<T>(xv * c - xp * s);
        x_out[pair_d] = static_cast<T>(xp * c + xv * s);
    }
"""

# A chain visits its keys four at a time, held in registers with fixed names
# (no spills): the four scores (independent of the running state) first, then
# sdpa_vector's online-softmax updates key by key in its order, while the
# next four keys load. Every value is sdpa_vector's arithmetic.
_CHAIN_MACROS = r"""
#define LOAD_K(slot, idx)                                                   \
    if ((idx) < N) {                                                        \
        const device T* kq = kp + ((idx) - first) / STRIDE * k_inner;        \
        for (int e = 0; e < PT; e++) {                                      \
            ks##slot[e] = kq[e];                                            \
        }                                                                   \
    }
#define LOAD_V(slot, idx)                                                   \
    if ((idx) < N) {                                                        \
        const device T* vq = vp + ((idx) - first) / STRIDE * v_inner;        \
        for (int e = 0; e < PT; e++) {                                      \
            vs##slot[e] = vq[e];                                            \
        }                                                                   \
    }
#define SCORE(slot)                                                         \
    for (int j = 0; j < HPT; j++) {                                         \
        U score = 0;                                                        \
        for (int e = 0; e < PT; e++) {                                      \
            score += q[j][e] * static_cast<U>(ks##slot[e]);                 \
        }                                                                   \
        scores[j][slot] = simd_sum(score);                                  \
    }
#define UPDATE(slot)                                                        \
    for (int j = 0; j < HPT; j++) {                                         \
        U score = scores[j][slot];                                          \
        U new_max = max(max_score[j], score);                               \
        U factor = fast::exp(max_score[j] - new_max);                       \
        U exp_score = fast::exp(score - new_max);                           \
        max_score[j] = new_max;                                             \
        sum_exp_score[j] = sum_exp_score[j] * factor + exp_score;           \
        for (int e = 0; e < PT; e++) {                                      \
            o[j][e] = o[j][e] * factor + exp_score * static_cast<U>(vs##slot[e]); \
        }                                                                   \
    }
#define CHAIN(first_key)                                                    \
    T ks0[PT], vs0[PT], ks1[PT], vs1[PT], ks2[PT], vs2[PT], ks3[PT], vs3[PT]; \
    LOAD_K(0, first_key) LOAD_V(0, first_key)                               \
    LOAD_K(1, first_key + STRIDE) LOAD_V(1, first_key + STRIDE)             \
    LOAD_K(2, first_key + 2 * STRIDE) LOAD_V(2, first_key + 2 * STRIDE)     \
    LOAD_K(3, first_key + 3 * STRIDE) LOAD_V(3, first_key + 3 * STRIDE)     \
    for (int i = first_key; i < N; i += 4 * STRIDE) {                       \
        U scores[HPT][4];                                                   \
        SCORE(0)                                                            \
        if (i + STRIDE < N) { SCORE(1) }                                    \
        if (i + 2 * STRIDE < N) { SCORE(2) }                                \
        if (i + 3 * STRIDE < N) { SCORE(3) }                                \
        LOAD_K(0, i + 4 * STRIDE)                                           \
        LOAD_K(1, i + 5 * STRIDE)                                           \
        LOAD_K(2, i + 6 * STRIDE)                                           \
        LOAD_K(3, i + 7 * STRIDE)                                           \
        UPDATE(0)                                                           \
        LOAD_V(0, i + 4 * STRIDE)                                           \
        if (i + STRIDE >= N) break;                                         \
        UPDATE(1)                                                           \
        LOAD_V(1, i + 5 * STRIDE)                                           \
        if (i + 2 * STRIDE >= N) break;                                     \
        UPDATE(2)                                                           \
        LOAD_V(2, i + 6 * STRIDE)                                           \
        if (i + 3 * STRIDE >= N) break;                                     \
        UPDATE(3)                                                           \
        LOAD_V(3, i + 7 * STRIDE)                                           \
    }
"""

# sdpa_vector (one pass): simdgroup s of a head's threadgroup visits keys
# s, s + 32, ... below the row's causal end. Here each simdgroup runs chain s
# of HPT query heads sharing a KV head and stores its state for the combine.
_ONE_PASS_CHAINS = r"""
    constexpr int BN = 32;
    constexpr int PT = D / 32;
    constexpr int STRIDE = BN;
    typedef float U;
    const uint lane = thread_index_in_simdgroup;
    const uint f = threadgroup_position_in_grid.x * SG + simdgroup_index_in_threadgroup;
    const int s = int(f % BN);
    const int hg = int(f / BN);
    const int kv = threadgroup_position_in_grid.y;
    const int row = threadgroup_position_in_grid.z;
    const int rows = threadgroups_per_grid.z;
    const int H = GQA * int(threadgroups_per_grid.y);
    const int N = int(n_keys[0]) - rows + row + 1;
    const float sc = scale[0];

    U q[HPT][PT];
    U o[HPT][PT];
    U max_score[HPT];
    U sum_exp_score[HPT];
    for (int j = 0; j < HPT; j++) {
        const int head = kv * GQA + hg * HPT + j;
        const device T* qp = queries + ((size_t)head * rows + row) * D + lane * PT;
        for (int e = 0; e < PT; e++) {
            q[j][e] = static_cast<U>(sc) * qp[e];
            o[j][e] = 0;
        }
        max_score[j] = Limits<U>::finite_min;
        sum_exp_score[j] = 0;
    }
    const int first = s;
    const int k_inner = STRIDE * int(keys_strides[2]);
    const int v_inner = STRIDE * int(values_strides[2]);
    const device T* kp = keys + kv * keys_strides[1] + first * keys_strides[2] + lane * PT;
    const device T* vp = values + kv * values_strides[1] + first * values_strides[2] + lane * PT;
    CHAIN(first)

    for (int j = 0; j < HPT; j++) {
        const int head = kv * GQA + hg * HPT + j;
        const size_t c = (size_t)(row * H + head) * BN + s;
        if (lane == 0) {
            maxs[c] = max_score[j];
            sums[c] = sum_exp_score[j];
        }
        // [row, head, dim, chain]: the combine reads a dim's 32 chains at once.
        for (int e = 0; e < PT; e++) {
            outs[((size_t)(row * H + head) * D + lane * PT + e) * BN + s] = o[j][e];
        }
    }
"""

# sdpa_vector's closing reduction for one (head, row), then the gate.
_ONE_PASS_COMBINE = r"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int PT = D / BD;
    typedef float U;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int head = threadgroup_position_in_grid.x;
    const int row = threadgroup_position_in_grid.y;
    const int H = threadgroups_per_grid.x;
    const size_t c0 = (size_t)(row * H + head) * BN;

    U max_score = maxs[c0 + simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    U sum_exp_score = simd_sum(sums[c0 + simd_lid] * factor);
    U o[PT];
    for (int i = 0; i < PT; i++) {
        // MLX's outputs[simd_gid * BD + simd_lid]: dim simd_gid * PT + i of
        // chain simd_lid.
        o[i] = simd_sum(
            outs[((size_t)(row * H + head) * D + simd_gid * PT + i) * BN + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    }
    if (simd_lid == 0) {
        // MLX's multiply by the gate's sigmoid, both in the activation dtype.
        const size_t base = (size_t)row * (H * D) + head * D + simd_gid * PT;
        for (int i = 0; i < PT; i++) {
            out[base + i] = static_cast<T>(o[i]) * gate[base + i];
        }
    }
"""

# sdpa_vector_2pass_1: the simdgroup of query head g (of a KV head's group)
# and partition b visits keys b, b + BLOCKS, ... below the row's causal end.
_TWO_PASS_CHAINS = r"""
    constexpr int PT = D / 32;
    constexpr int HPT = 1;
    constexpr int STRIDE = BLOCKS;
    typedef float U;
    const uint lane = thread_index_in_simdgroup;
    const int kv = threadgroup_position_in_grid.x;
    const int block = int(threadgroup_position_in_grid.z) % BLOCKS;
    const int row = int(threadgroup_position_in_grid.z) / BLOCKS;
    const int rows = int(threadgroups_per_grid.z) / BLOCKS;
    const int H = GQA * int(threadgroups_per_grid.x);
    const int head = GQA * kv + int(thread_position_in_threadgroup.y);
    const int N = int(n_keys[0]) - rows + row + 1;
    const float sc = scale[0];

    U q[HPT][PT];
    U o[HPT][PT];
    U max_score[HPT];
    U sum_exp_score[HPT];
    const device T* qp = queries + ((size_t)head * rows + row) * D + lane * PT;
    for (int e = 0; e < PT; e++) {
        q[0][e] = static_cast<U>(sc) * qp[e];
        o[0][e] = 0;
    }
    max_score[0] = Limits<U>::finite_min;
    sum_exp_score[0] = 0;
    const int first = block;
    const int k_inner = STRIDE * int(keys_strides[2]);
    const int v_inner = STRIDE * int(values_strides[2]);
    const device T* kp = keys + kv * keys_strides[1] + first * keys_strides[2] + lane * PT;
    const device T* vp = values + kv * values_strides[1] + first * values_strides[2] + lane * PT;
    CHAIN(first)

    const size_t c = (size_t)(row * H + head) * BLOCKS + block;
    if (lane == 0) {
        sums[c] = sum_exp_score[0];
        maxs[c] = max_score[0];
    }
    for (int e = 0; e < PT; e++) {
        partials[c * D + lane * PT + e] = static_cast<T>(o[0][e]);
    }
"""

# sdpa_vector_2pass_2 for one (head, row), then the gate.
_TWO_PASS_COMBINE = r"""
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int elem_per_thread = D / BD;
    typedef float U;

    thread U o[elem_per_thread] = {0};
    threadgroup U outputs[BN * BD];

    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int head = threadgroup_position_in_grid.x;
    const int row = threadgroup_position_in_grid.y;
    const int H = threadgroups_per_grid.x;
    const size_t q_offset = (size_t)row * H + head;
    const device T* pp = partials + q_offset * BLOCKS * D + simd_gid * D
        + simd_lid * elem_per_thread;
    const device float* sp = sums + q_offset * BLOCKS;
    const device float* mp = maxs + q_offset * BLOCKS;

    U sum_exp_score = 0.0;
    U max_score = Limits<U>::finite_min;

    for (int b = 0; b < BLOCKS / BN; ++b) {
        max_score = max(max_score, mp[simd_lid + BN * b]);
    }
    max_score = simd_max(max_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(mp[simd_lid + BN * b] - max_score);
        sum_exp_score += factor * sp[simd_lid + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);

    for (int b = 0; b < BLOCKS / BN; ++b) {
        U factor = fast::exp(mp[simd_gid] - max_score);
        for (int i = 0; i < elem_per_thread; i++) {
            o[i] += factor * static_cast<U>(pp[i]);
        }
        mp += BN;
        sp += BN;
        pp += BN * D;
    }

    for (int i = 0; i < elem_per_thread; i++) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        // MLX's multiply by the gate's sigmoid, both in the activation dtype.
        const size_t base = (size_t)row * (H * D) + head * D + simd_gid * elem_per_thread;
        for (int i = 0; i < elem_per_thread; i++) {
            out[base + i] = static_cast<T>(o[i]) * gate[base + i];
        }
    }
"""


def _kernel(name: str):
    kernel = _KERNELS.get(name)
    if kernel is not None:
        return kernel
    if name == "prep":
        kernel = mx.fast.metal_kernel(
            name="omlx_qwen4_attn_prep_qk",
            input_names=[
                "q_proj",
                "k_proj",
                "q_scale",
                "k_scale",
                "eps",
                "position_ids",
                "inv_freq",
                "position_selector",
            ],
            output_names=["queries", "keys"],
            source=_PREP_SOURCE,
        )
    elif name == "one_chains":
        kernel = mx.fast.metal_kernel(
            name="omlx_qwen4_attn_sdpa1_chains",
            input_names=["queries", "keys", "values", "n_keys", "scale"],
            output_names=["maxs", "sums", "outs"],
            header=_CHAIN_MACROS,
            source=_ONE_PASS_CHAINS,
            ensure_row_contiguous=False,
        )
    elif name == "one_combine":
        kernel = mx.fast.metal_kernel(
            name="omlx_qwen4_attn_sdpa1_gate",
            input_names=["maxs", "sums", "outs", "gate"],
            output_names=["out"],
            source=_ONE_PASS_COMBINE,
        )
    elif name == "two_chains":
        kernel = mx.fast.metal_kernel(
            name="omlx_qwen4_attn_sdpa2_chains",
            input_names=["queries", "keys", "values", "n_keys", "scale"],
            output_names=["partials", "sums", "maxs"],
            header=_CHAIN_MACROS,
            source=_TWO_PASS_CHAINS,
            ensure_row_contiguous=False,
        )
    else:
        kernel = mx.fast.metal_kernel(
            name="omlx_qwen4_attn_sdpa2_gate",
            input_names=["partials", "sums", "maxs", "gate"],
            output_names=["out"],
            source=_TWO_PASS_COMBINE,
        )
    _KERNELS[name] = kernel
    return kernel


def _scalar(value, dtype) -> mx.array:
    key = (value, dtype)
    array = _SCALARS.get(key)
    if array is None:
        # MLX hands its kernels the FP32 value of the Python float.
        array = mx.array([value], dtype=dtype)
        mx.eval(array)
        _SCALARS[key] = array
    return array


@functools.lru_cache(maxsize=None)
def _gpu_class() -> str:
    try:
        return str(mx.device_info().get("architecture", ""))[-1:]
    except Exception:
        return ""


def rows_available() -> bool:
    """Whether the fused decode/verify attention rows can run on this GPU at
    all (layers still check their own shapes): off after a kernel failure."""
    return not _DISABLED and _gpu_class() == "d"


def row_plan(first_keys: int, last_keys: int) -> int | None:
    """MLX 0.32.2's vector SDPA plan shared by rows seeing ``first_keys`` to
    ``last_keys`` keys (12 query heads per KV head, one row per call, 'd'
    GPU): 1 (one pass), 2 (two passes, 128 partitions), or None when the rows
    straddle a plan boundary or leave the transcribed range."""
    if last_keys < _TWO_PASS_KEYS:
        return 1
    if first_keys >= _TWO_PASS_KEYS and last_keys < _TWO_PASS_MAX_KEYS:
        return 2
    return None


def prep_qk(
    q_proj: mx.array,
    k_proj: mx.array,
    q_scale: mx.array,
    k_scale: mx.array,
    eps: float,
    position_ids: mx.array,
    inv_freq: mx.array,
    position_selector: mx.array,
    *,
    heads: int,
    kv_heads: int,
    rotary_dim: int,
):
    """``(queries [1, heads, R, D], keys [1, kv_heads, R, D])``: the q/k norms
    and MRoPE of ``R`` projected rows -- ``q_proj`` [R, heads * 2 * D] holds
    each head's query then gate, ``k_proj`` [R, kv_heads * D]; ``D`` = 256."""
    rows = q_proj.shape[0]
    head_dim = k_proj.shape[-1] // kv_heads
    dtype = q_proj.dtype
    return _kernel("prep")(
        inputs=[
            q_proj,
            k_proj,
            q_scale,
            k_scale,
            _scalar(eps, mx.float32),
            position_ids,
            inv_freq,
            position_selector,
        ],
        template=[
            ("T", dtype),
            ("QH", heads),
            ("KH", kv_heads),
            ("D", head_dim),
            ("ROT", rotary_dim),
            ("POS_NDIM", position_ids.ndim),
        ],
        grid=(64 * (heads + kv_heads), rows, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(1, heads, rows, head_dim), (1, kv_heads, rows, head_dim)],
        output_dtypes=[dtype, dtype],
    )


def dense_sdpa_gate(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    gate: mx.array,
    scale: float,
    plan: int,
) -> mx.array:
    """``sdpa(row) * gate`` for the ``R`` causal rows of ``queries``
    ([1, H, R, D], row-contiguous) over the cache views ``keys``/``values``
    ([1, KVH, N, D], unit stride along D; row ``r`` sees ``N - R + r + 1``
    keys), as ``[R, H * D]``; ``gate`` is ``[R, H * D]`` (MLX's sigmoid of the
    gate projection) and ``plan`` is ``row_plan``'s."""
    _, heads, rows, head_dim = queries.shape
    kv_heads = keys.shape[1]
    gqa = heads // kv_heads
    dtype = queries.dtype
    n_keys = mx.array([keys.shape[2]], dtype=mx.int32)
    scale_array = _scalar(scale, mx.float32)
    if plan == 1:
        hpt, sg = _ONE_PASS_HPT, _ONE_PASS_SG
        chains = 32 * (gqa // hpt)
        maxs, sums, outs = _kernel("one_chains")(
            inputs=[queries, keys, values, n_keys, scale_array],
            template=[("T", dtype), ("D", head_dim), ("GQA", gqa), ("HPT", hpt), ("SG", sg)],
            grid=(32 * chains, kv_heads, rows),
            threadgroup=(32 * sg, 1, 1),
            output_shapes=[
                (rows * heads * 32,),
                (rows * heads * 32,),
                (rows * heads * head_dim * 32,),
            ],
            output_dtypes=[mx.float32] * 3,
        )
        return _kernel("one_combine")(
            inputs=[maxs, sums, outs, gate],
            template=[("T", dtype), ("D", head_dim)],
            grid=(heads * 1024, rows, 1),
            threadgroup=(1024, 1, 1),
            output_shapes=[(rows, heads * head_dim)],
            output_dtypes=[dtype],
        )[0]
    blocks = _TWO_PASS_BLOCKS
    partials, sums, maxs = _kernel("two_chains")(
        inputs=[queries, keys, values, n_keys, scale_array],
        template=[("T", dtype), ("D", head_dim), ("GQA", gqa), ("BLOCKS", blocks)],
        grid=(32 * kv_heads, gqa, blocks * rows),
        threadgroup=(32, gqa, 1),
        output_shapes=[
            (rows * heads * blocks * head_dim,),
            (rows * heads * blocks,),
            (rows * heads * blocks,),
        ],
        output_dtypes=[dtype, mx.float32, mx.float32],
    )
    return _kernel("two_combine")(
        inputs=[partials, sums, maxs, gate],
        template=[("T", dtype), ("D", head_dim), ("BLOCKS", blocks)],
        grid=(heads * 1024, rows, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(rows, heads * head_dim)],
        output_dtypes=[dtype],
    )[0]


def ready(dtype, heads: int, kv_heads: int, head_dim: int, rotary_dim: int, pos_ndim: int) -> bool:
    """Whether the kernels serve this layout: the one-pass chains need the
    query heads of a KV head in pairs, the transcribed plans 12+ of them and
    head dim 256 (the RMS layout too). Each specialization runs once on tiny
    inputs first; a compile or launch failure turns the kernels off."""
    global _DISABLED
    if _DISABLED or _gpu_class() != "d":
        return False
    gqa = heads // kv_heads if kv_heads else 0
    if (
        head_dim != 256
        or not kv_heads
        or heads % kv_heads
        or gqa < 6
        or gqa % _ONE_PASS_HPT
        or rotary_dim % 2
        or not 0 < rotary_dim <= head_dim
    ):
        return False
    signature = (dtype, heads, kv_heads, head_dim, rotary_dim, pos_ndim)
    if signature in _READY:
        return True
    try:
        q_proj = mx.zeros((1, heads * 2 * head_dim), dtype=dtype)
        k_proj = mx.ones((1, kv_heads * head_dim), dtype=dtype)
        scale = mx.ones((head_dim,), dtype=mx.float32)
        positions = mx.zeros((3, 1, 1) if pos_ndim == 3 else (1, 1), dtype=mx.int32)
        half = rotary_dim // 2
        queries, keys = prep_qk(
            q_proj,
            k_proj,
            scale,
            scale,
            1e-6,
            positions,
            mx.ones((half,), dtype=mx.float32),
            mx.zeros((half,), dtype=mx.int32),
            heads=heads,
            kv_heads=kv_heads,
            rotary_dim=rotary_dim,
        )
        cache = mx.zeros((1, kv_heads, 4, head_dim), dtype=dtype)
        gate = mx.ones((1, heads * head_dim), dtype=dtype)
        outs = [
            dense_sdpa_gate(queries, cache[..., :2, :], cache[..., :2, :], gate, 0.0625, plan)
            for plan in (1, 2)
        ]
        mx.eval(queries, keys, *outs)
    except Exception:
        _DISABLED = True
        logger.warning("Qwen4 fused attention kernels failed; using the MLX path", exc_info=True)
        return False
    _READY.add(signature)
    return True
