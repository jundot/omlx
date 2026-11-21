# SPDX-License-Identifier: Apache-2.0
"""Attention and attention-residual primitives for Inkling 2."""

from __future__ import annotations

import mlx.core as mx

from ...custom_kernels.nax import is_nax_available

# Upper bound on one additive mask block (elements) in the general attention
# path; queries are split into sub-blocks so the mask stays below it.
_MASK_BUDGET = 1 << 29


_ATTN_RES_TG = 256
_attn_res_kernels: dict = {}


def _attn_res_kernel(blocks: int, queries: int):
    """Fused depth mixing for ``blocks`` completed blocks plus the partial sum.

    One threadgroup per token row: dot products of every source with every
    query, softmax over sources, then the weighted sum, without stacking the
    sources.
    """
    key = (blocks, queries)
    kernel = _attn_res_kernels.get(key)
    if kernel is not None:
        return kernel
    srcs = [f"b{i}" for i in range(blocks)] + ["p"]
    loads = "\n".join(
        f"        float s{i} = {name}[base + h];" for i, name in enumerate(srcs)
    )
    dots = "\n".join(
        f"        acc[{i} * Q + qi] += s{i} * qv;" for i in range(len(srcs))
    )
    mix = " + ".join(f"w[{i} * Q + qi] * s{i}" for i in range(len(srcs)))
    source = f"""
    constexpr uint NS = {len(srcs)};
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    size_t base = (size_t)row * H;
    float acc[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) acc[i] = 0.0f;
    for (uint h = tid; h < H; h += TG) {{
{loads}
        for (uint qi = 0; qi < Q; ++qi) {{
            float qv = query[qi * H + h];
{dots}
        }}
    }}
    threadgroup float red[NS * Q][TG / 32];
    for (uint i = 0; i < NS * Q; ++i) {{
        float v = simd_sum(acc[i]);
        if (lane == 0) red[i][sg] = v;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float w[NS * Q];
    for (uint i = 0; i < NS * Q; ++i) {{
        float t = 0.0f;
        for (uint j = 0; j < TG / 32; ++j) t += red[i][j];
        w[i] = t * scale[0];
    }}
    for (uint qi = 0; qi < Q; ++qi) {{
        float m = -INFINITY;
        for (uint n = 0; n < NS; ++n) m = max(m, w[n * Q + qi]);
        float z = 0.0f;
        for (uint n = 0; n < NS; ++n) {{
            w[n * Q + qi] = metal::exp(w[n * Q + qi] - m);
            z += w[n * Q + qi];
        }}
        for (uint n = 0; n < NS; ++n) w[n * Q + qi] /= z;
    }}
    for (uint h = tid; h < H; h += TG) {{
{loads}
        for (uint qi = 0; qi < Q; ++qi) {{
            out[((size_t)row * Q + qi) * H + h] = {mix};
        }}
    }}
"""
    kernel = mx.fast.metal_kernel(
        name=f"omlx_inkling2_attn_res_{blocks}_{queries}",
        input_names=[*srcs, "query", "scale"],
        output_names=["out"],
        source=source,
    )
    _attn_res_kernels[key] = kernel
    return kernel


def attn_res_mix(blocks: list, partial_n: mx.array, query: mx.array, scale: float):
    """Mix unit-RMS depth sources with learned queries.

    ``blocks`` holds [B, L, H] FP32 sources, ``partial_n`` is the current
    partial sum (also unit-RMS) and ``query`` is [Q, H]. Returns [B, L, Q, H].
    """
    if mx.default_device() != mx.gpu:
        src = mx.stack([*blocks, partial_n], axis=-2)
        logits = (src @ query.T) * scale
        w = mx.softmax(logits, axis=-2, precise=True)
        return w.swapaxes(-1, -2) @ src
    b, length, hidden = partial_n.shape
    nq = query.shape[0]
    rows = b * length
    kernel = _attn_res_kernel(len(blocks), nq)
    out = kernel(
        inputs=[
            *[x.astype(mx.float32) for x in blocks],
            partial_n.astype(mx.float32),
            query.astype(mx.float32),
            mx.array([scale], dtype=mx.float32),
        ],
        template=[("H", hidden), ("Q", nq), ("TG", _ATTN_RES_TG)],
        grid=(rows * _ATTN_RES_TG, 1, 1),
        threadgroup=(_ATTN_RES_TG, 1, 1),
        output_shapes=[(b, length, nq, hidden)],
        output_dtypes=[mx.float32],
    )[0]
    return out


def _decode_attention(q, k, v, rel, sink, scale, window):
    s = k.shape[2]
    lo = max(0, s - window) if window else 0
    if lo:
        k = k[:, :, lo:]
        v = v[:, :, lo:]
    sv = s - lo
    r = rel.shape[-1]
    n = min(r, sv)
    hq = q.shape[1]
    tail = rel[0, 0, :, :n][:, ::-1]
    if n < sv:
        tail = mx.concatenate([mx.zeros((hq, sv - n), dtype=tail.dtype), tail], -1)
    mask = tail.reshape(1, hq, 1, sv).astype(q.dtype)
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask, sinks=sink[0, 0].astype(q.dtype)
    )


# Flash attention with the relative-bias band and per-query sinks, for
# prefill-sized BF16 query blocks. One threadgroup runs 32 queries of one head
# in four simdgroups (8 rows each) over 16-key tiles staged in half. Scores
# live in the log2 domain; each row's online softmax starts from its sink
# logit, so the zero-value sink is exact without an extra key.
_FLASH_SOURCE = """
    constexpr int BQ = 32;
    constexpr int BK = 16;
    constexpr int NT = 128;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int qt = threadgroup_position_in_grid.x;
    const int h = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int L = dims[0];
    const int S = dims[1];
    const int lp = dims[2 + b];
    const int hk = h / (HQ / HKV);
    const int q0 = qt * BQ;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int a_row = S - L + row;
    const float qscale = scale[0] * LOG2E;

    // Q, K and V are BF16 values within half's range (L2-clipped q/k,
    // unit-RMS v), so staging them in half is exact; scores accumulate in
    // FP32 and the softmax scale is applied there. K is stored transposed
    // and both tiles are padded by 16 bytes per row against bank conflicts.
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 Qf[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + b * q_strides[0] + h * q_strides[1]
                + row * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = half(qv.x);
        Qf[db].thread_elements()[1] = half(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = row_ok ? float(sink[(b * L + row) * HQ + h]) * LOG2E : 0.0f;
    float l = 1.0f;
    const device T* relp = rel + ((size_t)(b * L + (row_ok ? row : 0)) * HQ + h) * R;

    const int a_first = S - L + q0;
    const int a_last = S - L + min(q0 + BQ, L) - 1;
    int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int k_begin = (lo / BK) * BK;
    const device T* kb_ptr = k + b * k_strides[0] + hk * k_strides[1];
    const device T* vb_ptr = v + b * v_strides[0] + hk * v_strides[1];
    const bool vec_kv = k_strides[3] == 1 && v_strides[3] == 1;

    for (int k0 = k_begin; k0 <= a_last; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vec_kv) {
            // 8 BF16 values (16 bytes) per load.
            for (int c = tid; c < BK * D / 8; c += NT) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                uint4 kw = uint4(0);
                uint4 vw = uint4(0);
                if (key < S) {
                    kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                    vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                }
                threadgroup half* kd = Ks + col * LDK + r;
                threadgroup half* vd = Vs + r * LDV + col;
                for (int j = 0; j < 4; ++j) {
                    kd[(2 * j) * LDK] = half(as_type<float>(kw[j] << 16));
                    kd[(2 * j + 1) * LDK] = half(as_type<float>(kw[j] & 0xFFFF0000u));
                    vd[2 * j] = half(as_type<float>(vw[j] << 16));
                    vd[2 * j + 1] = half(as_type<float>(vw[j] & 0xFFFF0000u));
                }
            }
        } else {
            for (int c = tid; c < BK * D; c += NT) {
                const int r = c / D;
                const int col = c % D;
                const int key = k0 + r;
                float kv = 0.0f;
                float vv = 0.0f;
                if (key < S) {
                    kv = float(kb_ptr[key * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[key * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = half(kv);
                Vs[r * LDV + col] = half(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 Sf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }

        float rmax = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = a_row - key;
                const bool vis = row_ok && d >= 0 && key >= lp
                    && (WINDOW == 0 || d < WINDOW);
                float s = Sf[kb].thread_elements()[e] * qscale;
                if (vis) {
                    if (d < R) {
                        s += float(relp[d]) * LOG2E;
                    }
                } else {
                    s = -INFINITY;
                }
                Sf[kb].thread_elements()[e] = s;
                rmax = max(rmax, s);
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const float m_new = max(m, rmax);
        const float factor = fast::exp2(m - m_new);
        float rsum = 0.0f;
        simdgroup_half8x8 Pf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(Sf[kb].thread_elements()[e] - m_new);
                Pf[kb].thread_elements()[e] = half(p);
                rsum += p;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * factor + rsum;
        m = m_new;
        for (int db = 0; db < D / 8; ++db) {
            Of[db].thread_elements()[0] *= factor;
            Of[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }

    if (row_ok) {
        const float inv = 1.0f / l;
        device T* op = out + ((size_t)(b * L + row) * HQ + h) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(Of[db].thread_elements()[0] * inv);
            op[db * 8 + fn + 1] = T(Of[db].thread_elements()[1] * inv);
        }
    }
"""
_flash_kernel = None
# Query blocks shorter than this keep the SDPA path (decode and verify).
_FLASH_MIN_QUERIES = 32


def _use_flash(length: int, window: int) -> bool:
    """Prefill blocks take the flash kernel on the GPU.

    On hosts with tensor units (NAX) MLX's masked SDPA runs on them, which
    this kernel does not; there only sliding layers, whose cost is dominated
    by building the mask, switch to it.
    """
    if length < _FLASH_MIN_QUERIES or mx.default_device() != mx.gpu:
        return False
    return bool(window) or not is_nax_available()


def flash_banded_attention(q, k, v, rel, sink, *, scale, window, left_padding=None):
    """``banded_sink_attention`` for BF16 prefill blocks; same arguments and result."""
    global _flash_kernel
    b, hq, length, d = q.shape
    hkv, s = k.shape[1], k.shape[2]
    if _flash_kernel is None:
        _flash_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_flash_banded",
            input_names=["q", "k", "v", "rel", "sink", "dims", "scale"],
            output_names=["out"],
            source=_FLASH_SOURCE,
            ensure_row_contiguous=False,
        )
    lp = (
        left_padding.astype(mx.int32)
        if left_padding is not None
        else mx.zeros((b,), dtype=mx.int32)
    )
    dims = mx.concatenate([mx.array([length, s], dtype=mx.int32), lp])
    out = _flash_kernel(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            mx.contiguous(sink.astype(mx.float32)),
            dims,
            mx.array([scale], dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("D", d),
            ("HQ", hq),
            ("HKV", hkv),
            ("R", rel.shape[-1]),
            ("WINDOW", int(window)),
        ],
        grid=(((length + 31) // 32) * 128, hq, b),
        threadgroup=(128, 1, 1),
        output_shapes=[(b, length, hq, d)],
        output_dtypes=[q.dtype],
    )[0]
    return out.transpose(0, 2, 1, 3)


def banded_sink_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sink: mx.array,
    *,
    scale: float,
    window: int = 0,
    left_padding=None,
) -> mx.array:
    """Attention with a relative bias band and per-query sink logits.

    q: [B, Hq, L, D]; k, v: [B, Hkv, S, D] (queries are the last L keys);
    rel: [B, L, Hq, R] bias by backward distance d (applied for d < R);
    sink: [B, L, Hq] logits of a zero-value sink. ``window`` > 0 limits
    keys to d < window (sliding layers use window == R).
    """
    b, length = q.shape[0], q.shape[2]
    if b == 1 and length == 1 and left_padding is None:
        return _decode_attention(q, k, v, rel, sink, scale, window)
    if q.dtype == mx.bfloat16 and _use_flash(length, window):
        return flash_banded_attention(
            q, k, v, rel, sink, scale=scale, window=window, left_padding=left_padding
        )
    return _masked_attention(q, k, v, rel, sink, scale, window, left_padding)


def _masked_attention(q, k, v, rel, sink, scale, window, left_padding):
    """MLX SDPA over an additive mask that carries the bias band and sinks."""
    b, hq, length, d = q.shape
    s = k.shape[2]
    r = rel.shape[-1]
    if window:
        assert window == r, "sliding layers expect rel_extent == window"
    lo = max(0, s - length - window + 1) if window else 0
    hkv = k.shape[1]
    # A zero key/value row carries the sink: its logit comes from the mask.
    zero = mx.zeros((b, hkv, 1, d), dtype=k.dtype)
    kz = mx.concatenate([zero, k[:, :, lo:]], axis=2)
    vz = mx.concatenate([zero, v[:, :, lo:]], axis=2)
    nk = s - lo
    qb = max(16, min(length, _MASK_BUDGET // max(1, b * hq * (nk + 1))))
    neg = mx.array(-mx.inf, dtype=q.dtype)
    lp_ok = None
    if left_padding is not None:
        lp_ok = (lo + mx.arange(nk))[None, :] >= left_padding[:, None]
        lp_ok = mx.concatenate([mx.ones((b, 1), dtype=mx.bool_), lp_ok], axis=1)
        lp_ok = lp_ok[:, None, None, :]

    outs = []
    for i0 in range(0, length, qb):
        i1 = min(length, i0 + qb)
        nq = i1 - i0
        a0 = s - length + i0
        hi = s - length + i1
        f = max(lo, a0 - r + 1)
        near = hi - f
        dist = (a0 - f) + mx.arange(nq)[:, None] - mx.arange(near)[None, :]
        relq = rel[:, i0:i1].transpose(0, 2, 1, 3)
        gidx = mx.broadcast_to(mx.clip(dist, 0, r - 1), (b, hq, nq, near))
        bias = mx.take_along_axis(relq, gidx, axis=-1)
        vis = dist >= 0
        if window:
            vis = vis & (dist < window)
        near_mask = mx.where(vis, mx.where(dist < r, bias, 0), neg)
        parts = [sink[:, i0:i1].transpose(0, 2, 1)[..., None].astype(q.dtype)]
        if f > lo:
            fill = neg if window else mx.array(0, dtype=q.dtype)
            parts.append(mx.broadcast_to(fill, (b, hq, nq, f - lo)))
        parts.append(near_mask.astype(q.dtype))
        if hi < s:
            parts.append(mx.broadcast_to(neg, (b, hq, nq, s - hi)))
        mask = mx.concatenate(parts, axis=-1)
        if lp_ok is not None:
            mask = mx.where(lp_ok, mask, neg)
        out = mx.fast.scaled_dot_product_attention(
            q[:, :, i0:i1], kz, vz, scale=scale, mask=mask
        )
        if qb < length:
            # A lazy graph would hold every block's mask at once (12 GiB at
            # 64K keys); evaluated block by block, one mask is alive.
            mx.eval(out)
        outs.append(out)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


_ROUTER_SOURCE = """
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    constexpr uint PER = E / 32;
    float sc[PER];
    float key[PER];
    for (uint j = 0; j < PER; ++j) {
        uint e = lane + 32 * j;
        float s = 1.0f / (1.0f + metal::exp(-logits[row * E + e]));
        sc[j] = s;
        key[j] = s + bias[e];
    }
    float picked[K];
    uint chosen[K];
    float total = 0.0f;
    for (uint k = 0; k < K; ++k) {
        float best = -INFINITY;
        uint bj = 0;
        for (uint j = 0; j < PER; ++j) {
            if (key[j] > best) {
                best = key[j];
                bj = j;
            }
        }
        float top = simd_max(best);
        uint mine = (best == top) ? (lane + 32 * bj) : 0xFFFFFFFFu;
        uint win = simd_min(mine);
        float s = (lane == win % 32) ? sc[win / 32] : 0.0f;
        s = simd_sum(s);
        if (lane == win % 32) key[win / 32] = -INFINITY;
        picked[k] = s;
        chosen[k] = win;
        total += s;
    }
    if (lane == 0) {
        float norm = route_scale[0] / (total + 1e-20f);
        for (uint k = 0; k < K; ++k) {
            inds[row * K + k] = chosen[k];
            weights[row * K + k] = picked[k] * norm;
        }
    }
"""
_router_kernel = None


def route_topk(logits: mx.array, bias: mx.array, top_k: int, route_scale: float):
    """Sigmoid scores, top-k by ``score + bias``, renormalized unbiased weights.

    ``logits``: [N, E] FP32. Returns ([N, K] uint32 indices, [N, K] FP32).
    """
    global _router_kernel
    rows, experts = logits.shape
    if mx.default_device() != mx.gpu or experts % 32:
        scores = mx.sigmoid(logits)
        inds = mx.argpartition(-(scores + bias), kth=top_k - 1, axis=-1)[..., :top_k]
        w = mx.take_along_axis(scores, inds, axis=-1)
        w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20) * route_scale
        return inds, w
    if _router_kernel is None:
        _router_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_route_topk",
            input_names=["logits", "bias", "route_scale"],
            output_names=["inds", "weights"],
            source=_ROUTER_SOURCE,
        )
    inds, weights = _router_kernel(
        inputs=[
            logits.astype(mx.float32),
            bias.astype(mx.float32),
            mx.array([route_scale], dtype=mx.float32),
        ],
        template=[("E", experts), ("K", top_k)],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return inds, weights


# Fused loop-controller kernels for ``_mix_messages``:
# - ``_NORM_PAIR_SOURCE`` computes unit-RMS of ``x`` and ``msg``, rounds to
#   BF16, and writes the stacked [N, 2, ROWS, H] array in either BF16 (small-M
#   fused path) or FP32 (large-M GEMM path) in one pass.
# - ``_LOOP_DOWN_SOURCE`` projects [N, 2, ROWS, H] BF16 against transposed
#   ``down_t`` [N, 2, R, H] (BF16 or FP32), accumulating both halves in FP32
#   and reading ``down_t`` once across all ``ROWS`` (exact chunk invariance).
# - ``_LOOP_GMIX_SOURCE`` projects ``low`` [N, ROWS, R] BF16 against transposed
#   ``gf_t`` [N, H, 2, R] (BF16 or FP32), applies ``tanh`` to gain/forget in
#   registers, and writes ``(1 - forget) * x + gain * msg`` directly without
#   materializing ``gates``.
# - ``_TANH_MIX_SOURCE`` fuses ``tanh(gain/forget)`` + residual mix for the
#   large-M GEMM path.
_NORM_PAIR_SOURCE = """
    uint row_idx = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint total_rows = N * ROWS;
    uint is_msg = row_idx >= total_rows ? 1u : 0u;
    uint nr = is_msg ? (row_idx - total_rows) : row_idx;
    uint n = nr / ROWS;
    uint r = nr % ROWS;
    const device float* src = (is_msg ? msg : x) + (size_t)nr * H;
    float ss = 0.0f;
    for (uint h = tid; h < H; h += TG) {
        float v = src[h];
        ss += v * v;
    }
    threadgroup float red[TG / 32];
    ss = simd_sum(ss);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < TG / 32; ++i) total += red[i];
    float inv = metal::rsqrt(total / float(H) + 1e-6f);
    device OT* dst = out + ((size_t)(n * 2 + is_msg) * ROWS + r) * H;
    for (uint h = tid; h < H; h += TG) {
        bfloat bv = bfloat(src[h] * inv);
        dst[h] = OT(bv);
    }
"""

_LOOP_DOWN_SOURCE = """
    uint nr = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint n = nr / R;
    uint r = nr % R;
    float acc[ROWS];
    for (uint i = 0; i < ROWS; ++i) acc[i] = 0.0f;
    for (uint part = 0; part < 2; ++part) {
        const device WT* wp = down_t + ((size_t)(n * 2 + part) * R + r) * H;
        const device bfloat* zp = z + (size_t)(n * 2 + part) * ROWS * H;
        for (uint h = lane * 4; h < H; h += 128) {
            float w0 = float(wp[h + 0]);
            float w1 = float(wp[h + 1]);
            float w2 = float(wp[h + 2]);
            float w3 = float(wp[h + 3]);
            for (uint row = 0; row < ROWS; ++row) {
                const device bfloat* zr = zp + (size_t)row * H + h;
                acc[row] += float(zr[0]) * w0 + float(zr[1]) * w1
                    + float(zr[2]) * w2 + float(zr[3]) * w3;
            }
        }
    }
    for (uint row = 0; row < ROWS; ++row) {
        float s = simd_sum(acc[row]);
        if (lane == 0) {
            low[((size_t)n * ROWS + row) * R + r] = bfloat(s);
        }
    }
"""

_LOOP_GMIX_SOURCE = """
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint nh = threadgroup_position_in_grid.x * 4 + sg;
    if (nh >= N * H) return;
    uint n = nh / H;
    uint h = nh % H;
    constexpr uint PER = (R + 31) / 32;
    float wg[PER], wf[PER];
    const device WT* gp = gf_t + ((size_t)nh * 2 + 0) * R;
    const device WT* fp = gf_t + ((size_t)nh * 2 + 1) * R;
    for (uint j = 0; j < PER; ++j) {
        uint idx = lane + j * 32;
        wg[j] = idx < R ? float(gp[idx]) : 0.0f;
        wf[j] = idx < R ? float(fp[idx]) : 0.0f;
    }
    for (uint row = 0; row < ROWS; ++row) {
        const device bfloat* lp = low + ((size_t)n * ROWS + row) * R;
        float dg = 0.0f;
        float df = 0.0f;
        for (uint j = 0; j < PER; ++j) {
            uint idx = lane + j * 32;
            float lv = idx < R ? float(lp[idx]) : 0.0f;
            dg += lv * wg[j];
            df += lv * wf[j];
        }
        dg = simd_sum(dg);
        df = simd_sum(df);
        if (lane == 0) {
            float g = metal::tanh(dg);
            float f = metal::tanh(df);
            size_t pos = ((size_t)n * ROWS + row) * H + h;
            out[pos] = (1.0f - f) * x[pos] + g * msg[pos];
        }
    }
"""

_TANH_MIX_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx >= N * ROWS * H) return;
    uint nr = idx / H;
    uint h = idx % H;
    size_t gbase = (size_t)nr * (2 * H);
    float g = metal::tanh(float(gates[gbase + h]));
    float f = metal::tanh(float(gates[gbase + H + h]));
    out[idx] = (1.0f - f) * x[idx] + g * msg[idx];
"""

_norm_pair_kernel = None
_loop_down_kernel = None
_loop_gmix_kernel = None
_tanh_mix_kernel = None
_LOOP_FUSED_MAX_ROWS = 16


def pack_loop_controller(down: mx.array, gain_forget: mx.array):
    """Transpose loop controller weights for coalesced decode/verify reads.

    Real checkpoints widen BF16 controller weights to FP32 at load; when the
    values are exact BF16, pack as BF16 to halve memory bandwidth with zero
    numerical loss.
    """
    n, _, h, r = down.shape
    exact_bf16 = (
        mx.array_equal(down.astype(mx.bfloat16).astype(mx.float32), down).item()
        and mx.array_equal(
            gain_forget.astype(mx.bfloat16).astype(mx.float32), gain_forget
        ).item()
    )
    dtype = mx.bfloat16 if exact_bf16 else down.dtype
    down_t = mx.contiguous(down.astype(dtype).transpose(0, 1, 3, 2))
    gf_t = mx.contiguous(
        gain_forget.astype(dtype).reshape(n, r, 2, h).transpose(0, 3, 2, 1)
    )
    mx.eval(down_t, gf_t)
    return down_t, gf_t


def mix_messages_gpu(
    x: mx.array,
    message: mx.array,
    down: mx.array,
    gain_forget: mx.array,
    down_t: mx.array | None = None,
    gf_t: mx.array | None = None,
) -> mx.array:
    """Fused GPU loop-controller message mixing; exact FP32 accumulation."""
    global _norm_pair_kernel, _loop_down_kernel, _loop_gmix_kernel, _tanh_mix_kernel
    n, b, length, hidden = x.shape
    rows = b * length
    rank = down.shape[-1]
    if _norm_pair_kernel is None:
        _norm_pair_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_norm_pair",
            input_names=["x", "msg"],
            output_names=["out"],
            source=_NORM_PAIR_SOURCE,
        )
    if (
        rows <= _LOOP_FUSED_MAX_ROWS
        and hidden % 4 == 0
        and down_t is not None
        and gf_t is not None
    ):
        if _loop_down_kernel is None:
            _loop_down_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_loop_down",
                input_names=["z", "down_t"],
                output_names=["low"],
                source=_LOOP_DOWN_SOURCE,
            )
            _loop_gmix_kernel = mx.fast.metal_kernel(
                name="omlx_inkling2_loop_gmix",
                input_names=["x", "msg", "low", "gf_t"],
                output_names=["out"],
                source=_LOOP_GMIX_SOURCE,
            )
        z = _norm_pair_kernel(
            inputs=[x.astype(mx.float32), message.astype(mx.float32)],
            template=[
                ("OT", mx.bfloat16),
                ("N", n),
                ("ROWS", rows),
                ("H", hidden),
                ("TG", 256),
            ],
            grid=(2 * n * rows * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(n, 2, rows, hidden)],
            output_dtypes=[mx.bfloat16],
        )[0]
        low = _loop_down_kernel(
            inputs=[z, down_t],
            template=[
                ("WT", down_t.dtype),
                ("R", rank),
                ("H", hidden),
                ("ROWS", rows),
            ],
            grid=(n * rank * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(n, rows, rank)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return _loop_gmix_kernel(
            inputs=[x.astype(mx.float32), message.astype(mx.float32), low, gf_t],
            template=[
                ("WT", gf_t.dtype),
                ("N", n),
                ("H", hidden),
                ("R", rank),
                ("ROWS", rows),
            ],
            grid=(((n * hidden + 3) // 4) * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(n, b, length, hidden)],
            output_dtypes=[mx.float32],
        )[0]

    # Large-M (prefill) path: keep exact FP32 GEMMs (chunk-invariant) while
    # fusing norm+BF16-round and tanh+residual-mix into single-pass kernels.
    if _tanh_mix_kernel is None:
        _tanh_mix_kernel = mx.fast.metal_kernel(
            name="omlx_inkling2_tanh_mix",
            input_names=["x", "msg", "gates"],
            output_names=["out"],
            source=_TANH_MIX_SOURCE,
        )
    xf = x.astype(mx.float32)
    mf = message.astype(mx.float32)
    z = _norm_pair_kernel(
        inputs=[xf, mf],
        template=[
            ("OT", mx.float32),
            ("N", n),
            ("ROWS", rows),
            ("H", hidden),
            ("TG", 256),
        ],
        grid=(2 * n * rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(2 * n, rows, hidden)],
        output_dtypes=[mx.float32],
    )[0]
    low = (z @ down.reshape(2 * n, hidden, rank)).reshape(n, 2, rows, rank).sum(axis=1)
    low = low.astype(mx.bfloat16).astype(mx.float32)
    gates = low @ gain_forget
    return _tanh_mix_kernel(
        inputs=[xf, mf, gates],
        template=[("N", n), ("ROWS", rows), ("H", hidden)],
        grid=(n * rows * hidden, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n, b, length, hidden)],
        output_dtypes=[mx.float32],
    )[0]
