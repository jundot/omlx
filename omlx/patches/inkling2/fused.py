# SPDX-License-Identifier: Apache-2.0
"""Long-context full-attention decode and verify kernel for Inkling 2."""

from __future__ import annotations

import mlx.core as mx

KEYS_PER_SPLIT = 256
_NSG = 8
_ROWS = 4
_MERGE_NSG = 8
HEAD_DIM = 128
# Metal: a threadgroup holds at most 32 KB of threadgroup memory.
_THREADGROUP_BYTES = 32768

# Invariant: splits are fixed KEYS_PER_SPLIT ranges counted from key 0.
# - query t of L sits at key S - L + t and sees keys [lp, S - L + t].
# - a masked key has weight 0, so a split without a visible key leaves its partial unchanged.
# - the merge folds split g in simdgroup g % _MERGE_NSG in ascending g, so a row has the same bits in any block.
_ATTN_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int RG = HALVES == 1 ? L : (L + 1) / 2;
    constexpr int ROWS = GROUP * RG;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint tid = thread_position_in_threadgroup.x;
    const int nsplit = dims[0];
    const int lp = dims[1];
    const int hh = HALVES == 1 ? 0 : int(sg % 2);
    const int split = int(threadgroup_position_in_grid.y) * (NSG / HALVES) + int(sg / HALVES);
    const int t0 = hh * RG;
    const uint hk = threadgroup_position_in_grid.z;
    const int S = k_shape[2];
    const float qscale = scale[0] * LOG2E;
    threadgroup float4 qs[GROUP * L][D / 4];
    threadgroup float ps[NSG][ROWS][32];
    for (uint i = tid; i < GROUP * L * D / 4; i += 32 * NSG) {
        const uint row = i / (D / 4), c = i % (D / 4);
        const uint g = row / L, t = row % L;
        const device T* qp = q + (hk * GROUP + g) * q_strides[1] + t * q_strides[2]
            + (c * 4) * q_strides[3];
        qs[row][c] = float4(float(qp[0]), float(qp[q_strides[3]]),
                            float(qp[2 * q_strides[3]]), float(qp[3 * q_strides[3]])) * qscale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (split >= nsplit) return;
    const int s0 = split * SPLIT;
    const int s1 = min(S, s0 + SPLIT);
    const device T* kb = k + hk * k_strides[1];
    const device T* vb = v + hk * v_strides[1];
    float m[ROWS], l[ROWS], acc[ROWS][4];
    for (int r = 0; r < ROWS; ++r) {
        m[r] = -INFINITY; l[r] = 0.0f;
        for (int i = 0; i < 4; ++i) acc[r][i] = 0.0f;
    }
    for (int tile = s0; tile < s1; tile += 32) {
        const int j = tile + int(lane);
        float sc[ROWS];
        if (j < s1 && j >= lp) {
            float kd[ROWS];
            for (int r = 0; r < ROWS; ++r) kd[r] = 0.0f;
            const device T* kp = kb + (size_t)j * k_strides[2];
            for (int c = 0; c < D / 4; ++c) {
                const float4 kv = float4(float(kp[4 * c]), float(kp[4 * c + 1]),
                                         float(kp[4 * c + 2]), float(kp[4 * c + 3]));
                for (int r = 0; r < ROWS; ++r) {
                    const int t = min(t0 + r % RG, L - 1);
                    kd[r] += dot(qs[(r / RG) * L + t][c], kv);
                }
            }
            for (int r = 0; r < ROWS; ++r) {
                const int g = r / RG, t = t0 + r % RG;
                const int d = S - L + t - j;
                float s = (d < 0 || t >= L) ? -INFINITY : kd[r];
                if (t < L && d >= 0 && d < R) s += float(rel[(t * HQ + hk * GROUP + g) * R + d]) * LOG2E;
                sc[r] = s;
            }
        } else {
            for (int r = 0; r < ROWS; ++r) sc[r] = -INFINITY;
        }
        float p[ROWS];
        for (int r = 0; r < ROWS; ++r) {
            const float mn = max(m[r], simd_max(sc[r]));
            if (mn == -INFINITY) { p[r] = 0.0f; continue; }
            const float f = m[r] == -INFINITY ? 0.0f : fast::exp2(m[r] - mn);
            p[r] = sc[r] == -INFINITY ? 0.0f : fast::exp2(sc[r] - mn);
            l[r] = l[r] * f + simd_sum(p[r]);
            for (int i = 0; i < 4; ++i) acc[r][i] *= f;
            m[r] = mn;
        }
        const int nk = min(32, s1 - tile);
        for (int r = 0; r < ROWS; ++r) ps[sg][r][lane] = p[r];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < nk; ++kk) {
            const device T* vp = vb + (size_t)(tile + kk) * v_strides[2] + lane * 4;
            const float4 vv = float4(float(vp[0]), float(vp[1]), float(vp[2]), float(vp[3]));
            for (int r = 0; r < ROWS; ++r) {
                const float pk = ps[sg][r][kk];
                if (pk != 0.0f) {
                    for (int i = 0; i < 4; ++i) acc[r][i] += pk * vv[i];
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int r = 0; r < ROWS; ++r) {
        const int g = r / RG, t = t0 + r % RG;
        if (t >= L) continue;
        device float* pp = part + (((size_t)(hk * GROUP + g) * L + t) * nsplit + split) * (D + 2);
        for (int i = 0; i < 4; ++i) pp[lane * 4 + i] = acc[r][i];
        if (lane == 0) { pp[D] = m[r]; pp[D + 1] = l[r]; }
    }
"""

_MERGE_SOURCE = """
    constexpr int DPL = D / 32;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;
    const int L = part_shape[2];
    const int nsplit = part_shape[3];
    const int t = row % L;
    const int h = row / L;
    const device float* pp = part + (size_t)row * nsplit * (D + 2);
    const float sk = float(sink[t * sink_strides[1] + h * sink_strides[2]]) * LOG2E;
    threadgroup float tm[NSG];
    threadgroup float to[NSG][D + 1];
    float mt = sk;
    for (int g = int(sg * 32 + lane); g < nsplit; g += NSG * 32) mt = max(mt, pp[g * (D + 2) + D]);
    mt = simd_max(mt);
    if (lane == 0) tm[sg] = mt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < NSG; ++i) mt = max(mt, tm[i]);
    float lt = 0.0f;
    float o[DPL];
    for (int i = 0; i < DPL; ++i) o[i] = 0.0f;
    for (int g = int(sg); g < nsplit; g += NSG) {
        const float mg = pp[g * (D + 2) + D];
        if (mg == -INFINITY) continue;
        const float f = fast::exp2(mg - mt);
        lt += pp[g * (D + 2) + D + 1] * f;
        for (int i = 0; i < DPL; ++i) o[i] += pp[g * (D + 2) + lane + 32 * i] * f;
    }
    for (int i = 0; i < DPL; ++i) to[sg][lane + 32 * i] = o[i];
    if (lane == 0) to[sg][D] = lt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    lt = fast::exp2(sk - mt);
    for (int i = 0; i < NSG; ++i) lt += to[i][D];
    const float inv = 1.0f / lt;
    device T* op = out + (size_t)row * D;
    for (int i = 0; i < DPL; ++i) {
        float o_sum = 0.0f;
        for (int j = 0; j < NSG; ++j) o_sum += to[j][lane + 32 * i];
        op[lane + 32 * i] = T(o_sum * inv);
    }
"""

_kernels: dict = {}


def _kernel(name: str, source: str, inputs: list, outputs: list):
    kern = _kernels.get(name)
    if kern is None:
        kern = _kernels[name] = mx.fast.metal_kernel(
            name=f"inkling2_{name}",
            input_names=inputs,
            output_names=outputs,
            source=source,
            ensure_row_contiguous=False,
        )
    return kern


def _threadgroup_bytes(group: int, length: int) -> int:
    rows = length if length <= _ROWS else -(-length // 2)
    return 4 * group * (length * HEAD_DIM + _NSG * rows * 32)


def _rows_per_pass(group: int) -> int:
    return max(
        n
        for n in range(1, 2 * _ROWS + 1)
        if _threadgroup_bytes(group, n) <= _THREADGROUP_BYTES
    )


def long_attention_supported(q: mx.array, k: mx.array, v: mx.array) -> bool:
    """Shapes and dtypes ``long_decode_attention`` covers: one sequence,
    128-wide heads with contiguous rows, BF16 activations on the GPU."""
    return (
        mx.default_device() == mx.gpu
        and q.shape[0] == 1
        and q.shape[-1] == HEAD_DIM
        and q.dtype == mx.bfloat16
    )


def long_decode_attention(q, k, v, rel, sink, *, scale, left_padding=None):
    """``kernels.banded_sink_attention`` for full-attention layers (no window).

    q: [1, Hq, L, D]; k, v: [1, Hkv, S, D] whose last dimension is contiguous
    (cache views); rel: [1, L, Hq, R]; sink: [1, L, Hq]; ``left_padding``: [1]
    array or None. Query t sits at key S - L + t. Returns [1, Hq, L, D].
    """
    length, s = q.shape[2], k.shape[2]
    step = _rows_per_pass(q.shape[1] // k.shape[1])
    outs = []
    for t0 in range(0, length, step):
        t1 = min(length, t0 + step)
        end = s - length + t1
        outs.append(
            _attention_block(
                q[:, :, t0:t1],
                k[:, :, :end],
                v[:, :, :end],
                rel[:, t0:t1],
                sink[:, t0:t1],
                scale,
                left_padding,
            )
        )
    return mx.concatenate(outs, axis=2)


def _attention_block(q, k, v, rel, sink, scale, left_padding):
    _, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    r = rel.shape[-1]
    nsplit = -(-s // KEYS_PER_SPLIT)
    halves = 1 if length <= _ROWS else 2
    lp = (
        left_padding.astype(mx.int32)
        if left_padding is not None
        else mx.zeros((1,), dtype=mx.int32)
    )
    part = _kernel(
        "long_attention",
        _ATTN_SOURCE,
        ["q", "k", "v", "rel", "dims", "scale"],
        ["part"],
    )(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            mx.concatenate([mx.array([nsplit], dtype=mx.int32), lp]),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("D", d),
            ("GROUP", hq // hkv),
            ("R", r),
            ("NSG", _NSG),
            ("SPLIT", KEYS_PER_SPLIT),
            ("L", length),
            ("HALVES", halves),
            ("HQ", hq),
        ],
        grid=(32 * _NSG, -(-nsplit // (_NSG // halves)), hkv),
        threadgroup=(32 * _NSG, 1, 1),
        output_shapes=[(1, hq, length, nsplit, d + 2)],
        output_dtypes=[mx.float32],
    )[
        0
    ]
    return _kernel("long_attention_merge", _MERGE_SOURCE, ["part", "sink"], ["out"])(
        inputs=[part, sink],
        template=[("T", q.dtype), ("D", d), ("NSG", _MERGE_NSG)],
        grid=(32 * _MERGE_NSG * hq * length, 1, 1),
        threadgroup=(32 * _MERGE_NSG, 1, 1),
        output_shapes=[(1, hq, length, d)],
        output_dtypes=[q.dtype],
    )[0]
