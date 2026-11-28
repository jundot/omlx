# SPDX-License-Identifier: Apache-2.0
"""Multi-row (speculative verify) attention on simdgroup 8x8 matrices.

MLX's vector SDPA kernel serves q_len <= 8 with one simd_sum, two exps and
an output update per key *per row*, so a verify call costs roughly T times
a one-row decode. Here the gqa x T query rows that share one KV head form
the M side of an 8x8 matrix multiply: scores and the P@V update cost the
same for one row or eight, and K/V are read once per row tile.

Semantics: row j attends keys [0 .. n_keys - T + j] (end-aligned causal);
the mask argument of the stock call is not needed. Pass 1 splits the keys
over up to 128 blocks (8-key tiles, strided), pass 2 combines the partials.

Shape contract: queries [B, Hq, T, 128], K/V buffers [B, Hkv, cap, 128],
gqa * T <= 32.
"""

from __future__ import annotations

import functools

import mlx.core as mx

_MAX_BLOCKS = 128  # most key blocks (pass-1 threadgroups per KV head)
_TARGET_GROUPS = 256  # threadgroups per dispatch: layers run back to back, so a
# small grid leaves the GPU idle (measured: 32 blocks x 8 KV heads beat 16 and 64)
_HEAD_DIM = 128
_MAX_ROWS = 32  # four 8-row tiles, one simdgroup each

_HEADER = """
#include <metal_simdgroup_matrix>
constant constexpr float NEG_MAX = -3.402823466e38f;
"""

# Threadgroup (32, MT, 1): simdgroup rt owns rows [8*rt, 8*rt+8) of the
# gqa*T rows of one KV head. Grid threadgroups (Hkv, B, BLOCKS).
# Fragment layout of an 8x8 simdgroup matrix: lane holds row fm, columns
# fn and fn+1; the four lanes of a row differ in lane bits 0 and 3.
# params = [N, cap_k, cap_v, blocks] (int32). qscale[0] is the softmax
# scale (fp32); it is applied to q in fp32 here, as stock SDPA does.
_PASS1_SRC = """
    const int BLOCKS = params[3];
    constexpr int D = {head_dim};
    constexpr int ND = D / 8;
    constexpr int M = G * TR;
    constexpr int LD = D + 8;  // padded tgmem row: no bank conflicts

    const int N = params[0];
    const int cap_k = params[1];
    const int cap_v = params[2];

    threadgroup float ks[8 * LD];
    threadgroup float vs[8 * LD];
    const uint lane = thread_index_in_simdgroup;
    const int nthreads = threads_per_threadgroup.y * 32;
    const int tid = thread_position_in_threadgroup.y * 32 + lane;
    const int kv = threadgroup_position_in_grid.x;
    const int b = threadgroup_position_in_grid.y;
    const int block = threadgroup_position_in_grid.z;
    const int rt = thread_position_in_threadgroup.y;
    const int num_kv = threadgroups_per_grid.x;
    const int num_q = num_kv * G;

    const int qid = lane >> 2;
    const int fm = (qid & 4) + ((lane >> 1) & 3);
    const int fn = ((qid & 2) << 1) + ((lane & 1) << 1);
    const int r = rt * 8 + fm;
    const bool row_ok = r < M;
    const int horizon = N - TR + (r % TR);

    const size_t base_row = ((size_t)b * num_q + (size_t)kv * G) * TR;
    const device T* qp = q + (base_row + r) * D;
    const device T* kp = k + (size_t)(b * num_kv + kv) * cap_k * D;
    const device T* vp = v + (size_t)(b * num_kv + kv) * cap_v * D;

    simdgroup_float8x8 qf[ND];
    simdgroup_float8x8 o[ND];
    _Pragma("clang loop unroll(full)")
    for (int d8 = 0; d8 < ND; d8++) {{
        qf[d8].thread_elements()[0] = row_ok ? static_cast<float>(qp[d8 * 8 + fn]) * qscale[0] : 0.0f;
        qf[d8].thread_elements()[1] = row_ok ? static_cast<float>(qp[d8 * 8 + fn + 1]) * qscale[0] : 0.0f;
        o[d8] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }}
    float m = NEG_MAX;
    float lsum = 0.0f;

    for (int tile = block; tile * 8 < N; tile += BLOCKS) {{
        const int k0 = tile * 8;
        const int key0 = k0 + fn;
        const int key1 = key0 + 1;

        // Whole threadgroup stages the 8 K rows and 8 V rows (coalesced,
        // fp32) so the row-tile simdgroups share one device read.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < 8 * (D / 4); i += nthreads) {{
            const int kr = i / (D / 4);
            const int c4 = (i % (D / 4)) * 4;
            const int key = k0 + kr;
            float4 kv4 = 0.0f;
            float4 vv4 = 0.0f;
            if (key < N) {{
                kv4 = float4(*(const device vec<T, 4>*)(kp + (size_t)key * D + c4));
                vv4 = float4(*(const device vec<T, 4>*)(vp + (size_t)key * D + c4));
            }}
            *(threadgroup float4*)(ks + kr * LD + c4) = kv4;
            *(threadgroup float4*)(vs + kr * LD + c4) = vv4;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        _Pragma("clang loop unroll(full)")
    for (int d8 = 0; d8 < ND; d8++) {{
            simdgroup_float8x8 kt;
            simdgroup_load(kt, ks + d8 * 8, LD, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s, qf[d8], kt, s);
        }}

        const bool ok0 = key0 <= horizon;
        const bool ok1 = key1 <= horizon;
        const float s0 = ok0 ? s.thread_elements()[0] : NEG_MAX;
        const float s1 = ok1 ? s.thread_elements()[1] : NEG_MAX;
        float tmax = max(s0, s1);
        tmax = max(tmax, simd_shuffle_xor(tmax, 1));
        tmax = max(tmax, simd_shuffle_xor(tmax, 8));
        const float new_m = max(m, tmax);
        const float factor = metal::fast::exp(m - new_m);
        const float p0 = ok0 ? metal::fast::exp(s0 - new_m) : 0.0f;
        const float p1 = ok1 ? metal::fast::exp(s1 - new_m) : 0.0f;
        m = new_m;
        lsum = lsum * factor + p0 + p1;

        simdgroup_float8x8 p;
        p.thread_elements()[0] = p0;
        p.thread_elements()[1] = p1;
        _Pragma("clang loop unroll(full)")
    for (int d8 = 0; d8 < ND; d8++) {{
            o[d8].thread_elements()[0] *= factor;
            o[d8].thread_elements()[1] *= factor;
            simdgroup_float8x8 vt;
            simdgroup_load(vt, vs + d8 * 8, LD);
            simdgroup_multiply_accumulate(o[d8], p, vt, o[d8]);
        }}
    }}

    float lt = lsum;
    lt += simd_shuffle_xor(lt, 1);
    lt += simd_shuffle_xor(lt, 8);
    if (row_ok) {{
        const size_t o_off = base_row + r;
        // fp32 partials, as in the gemma4 verify kernel (no bf16 rounding
        // of per-block accumulators).
        device float* pp = partials + (o_off * BLOCKS + block) * D;
        _Pragma("clang loop unroll(full)")
    for (int d8 = 0; d8 < ND; d8++) {{
            pp[d8 * 8 + fn] = o[d8].thread_elements()[0];
            pp[d8 * 8 + fn + 1] = o[d8].thread_elements()[1];
        }}
        if (fn == 0) {{
            sums[o_off * BLOCKS + block] = lt;
            maxs[o_off * BLOCKS + block] = m;
        }}
    }}
"""

# Threadgroup (D, 1, 1); grid threadgroups (B * Hq, T). One lane per output
# element; every lane re-reads the (small) per-block max / sum arrays.
_PASS2_SRC = """
    const int BLOCKS = params[3];
    constexpr int D = {head_dim};
    const int o_off = threadgroup_position_in_grid.x * threadgroups_per_grid.y
        + threadgroup_position_in_grid.y;
    const int d = thread_position_in_threadgroup.x;
    auto mp = maxs + (size_t)o_off * BLOCKS;
    auto sp = sums + (size_t)o_off * BLOCKS;
    auto pp = partials + (size_t)o_off * BLOCKS * D + d;

    float gmax = NEG_MAX;
    for (int i = 0; i < BLOCKS; i++) {{
        gmax = max(gmax, mp[i]);
    }}
    float total = 0.0f;
    float acc = 0.0f;
    for (int i = 0; i < BLOCKS; i++) {{
        const float w = metal::fast::exp(mp[i] - gmax);
        total += w * sp[i];
        acc += w * pp[(size_t)i * D];
    }}
    out[(size_t)o_off * D + d] = static_cast<T>(total == 0.0f ? acc : acc / total);
"""

_kernel = None
_kernel2 = None


def _pass1():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name=f"omlx_mma_verify_sdpa_p1_d{_HEAD_DIM}",
            input_names=["q", "k", "v", "params", "qscale"],
            output_names=["partials", "sums", "maxs"],
            header=_HEADER,
            source=_PASS1_SRC.format(head_dim=_HEAD_DIM),
        )
    return _kernel


@functools.lru_cache(maxsize=16)
def _params(n_keys: int, cap_k: int, cap_v: int, blocks: int):
    # Every layer of a forward shares these four ints: build the device
    # array once per (n_keys, ...) rather than once per layer.
    return mx.array([n_keys, cap_k, cap_v, blocks], dtype=mx.int32)


@functools.lru_cache(maxsize=16)
def _scale_array(scale: float):
    return mx.array([scale], dtype=mx.float32)


def _pass2():
    global _kernel2
    if _kernel2 is None:
        _kernel2 = mx.fast.metal_kernel(
            name=f"omlx_mma_verify_sdpa_p2_d{_HEAD_DIM}",
            input_names=["partials", "sums", "maxs", "params"],
            output_names=["out"],
            header=_HEADER,
            source=_PASS2_SRC.format(head_dim=_HEAD_DIM),
        )
    return _kernel2


def mma_verify_sdpa(queries, k_buf, v_buf, n_keys: int, scale: float):
    """Multi-row causal SDPA over a linear KV buffer.

    queries [B, Hq, T, 128]; k_buf/v_buf [B, Hkv, capacity, 128] with the
    first ``n_keys`` positions valid (the last T are the block's own keys).
    Row j attends keys [0 .. n_keys - T + j]. Returns [B, Hq, T, 128].
    """
    B, hq, T, head_dim = queries.shape
    hkv = k_buf.shape[1]
    gqa = hq // hkv
    if head_dim != _HEAD_DIM:
        raise ValueError(f"head_dim {head_dim} unsupported (need {_HEAD_DIM})")
    if gqa * T > _MAX_ROWS:
        raise ValueError(f"gqa * rows = {gqa * T} exceeds {_MAX_ROWS}")
    tiles = -(-gqa * T // 8)
    q = queries
    qscale = _scale_array(scale)

    blocks = max(1, min(_MAX_BLOCKS, -(-_TARGET_GROUPS // (hkv * B)), -(-n_keys // 16)))
    params = _params(n_keys, k_buf.shape[2], v_buf.shape[2], blocks)
    partials, sums, maxs = _pass1()(
        inputs=[q, k_buf, v_buf, params, qscale],
        template=[("T", q.dtype), ("TR", T), ("G", gqa)],
        grid=(32 * hkv, tiles * B, blocks),
        threadgroup=(32, tiles, 1),
        output_shapes=[
            (B, hq, T, blocks, head_dim),
            (B, hq, T, blocks),
            (B, hq, T, blocks),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    (out,) = _pass2()(
        inputs=[partials, sums, maxs, params],
        template=[("T", q.dtype)],
        grid=(head_dim * B * hq, T, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[(B, hq, T, head_dim)],
        output_dtypes=[q.dtype],
    )
    return out
