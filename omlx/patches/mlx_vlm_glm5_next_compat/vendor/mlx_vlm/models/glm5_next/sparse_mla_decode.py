# SPDX-License-Identifier: Apache-2.0
"""Split-K NoPE sparse MLA attention for decode and MTP verify (GLM-5.3-Flash DSA layers).

Inputs are the indexer's top-k *window* ids, not expanded token ids: each
threadgroup expands its 128 selected slots in-kernel (window w -> tokens
4w..4w+3, then the incomplete-window tail), checks validity (window complete
and causal for this row), and runs NAX fragment attention (MLX
steel nax.h) over that single key block. A second kernel merges the
per-block (max, sum, O) partials. This replaces the index expansion ops, the
gathered KV copy and the unfused 512-dim SDPA of the general decode path.
"""

import os
import re

import mlx.core as mx

# The NAX attention tile spans four 32-key column fragments. Each head
# fragment has sixteen rows; latent dimensions divide into float4 packs
# across four column fragments and a 32-lane simdgroup.
_NAX_COLUMN_FRAGMENTS = 4
_NAX_FRAGMENT_HEAD_ROWS = 16
_METAL_SIMDGROUP_LANES = 32
_SPLIT_K_KEY_TILE_ROWS = _NAX_COLUMN_FRAGMENTS * _METAL_SIMDGROUP_LANES
_NAX_LATENT_TILE_ELEMENTS = 4 * _METAL_SIMDGROUP_LANES
_NAX_MAX_HEAD_FRAGMENTS = 4
_MERGE_THREADGROUP_LANES = 8 * _METAL_SIMDGROUP_LANES

_INC = os.path.join(os.path.dirname(mx.__file__), "include")


def _inline(rel, seen):
    """Inline an MLX Metal header and its mlx/ includes for metal_kernel."""
    if rel in seen:
        return ""
    seen.add(rel)
    out = []
    with open(os.path.join(_INC, rel)) as source:
        lines = source.readlines()
    for line in lines:
        m = re.match(r'\s*#include\s+"(mlx/[^"]+)"', line)
        if m:
            out.append(_inline(m.group(1), seen))
        elif line.strip() != "#pragma once":
            out.append(line)
    return "".join(out)


_SRC = """
    using namespace mlx::steel;
    using Frag = BaseNAXFrag;
    typedef typename Frag::template dtype_frag_t<float> ff;
    typedef typename Frag::template dtype_frag_t<T> ft;
    typedef metal::vec<T, 4> T4;

    constexpr int DW = D / WN;
    constexpr int OT = DW / 32;
    constexpr int KPS = BK / WN;
    constexpr int HALF = BK / 2;
    constexpr int NSG = WM * WN;
    constexpr float NEG = -3.402823466e+38f;

    threadgroup int sel[BK];
    threadgroup float Pb[H * HALF];
    threadgroup float rmax[NSG * 16];
    threadgroup float rsum[NSG * 16];

    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const int wm = sg / WN;
    const int wn = sg % WN;
    const int kb = int(threadgroup_position_in_grid.x);   // key block
    const int s = int(threadgroup_position_in_grid.y);    // query row
    const int S = dims[0];
    const int KVL = dims[1];
    const int NSEL = dims[2];     // selected windows per row
    const int P = dims[3];        // pooled windows available
    const int before = dims[4];   // tokens processed before this block
    const int NB = dims[5];       // key blocks
    const int KP = KPOOL;
    const int qpos = before + s;
    const int total = NSEL * KP + (KP - 1);
    const float scale2 = scale_in[0] * 1.44269504089f;

    for (int i = sg * 32 + lane; i < BK; i += NSG * 32) {
        const int slot = kb * BK + i;
        int tok = -1;
        if (slot < NSEL * KP) {
            const int w = win[s * NSEL + slot / KP];
            if (w >= 0 && w < P && (w + 1) * KP - 1 <= qpos) tok = w * KP + slot % KP;
        } else if (slot < total) {
            const int j = slot - NSEL * KP;
            const int cnt = (qpos + 1) % KP;
            if (j < cnt) tok = qpos + 1 - cnt + j;
        }
        sel[i] = (tok >= 0 && tok < KVL && (!HAS_MASK || attn_mask[s * KVL + tok])) ? tok : -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short2 sc = Frag::get_coord();
    const size_t qstr = size_t(S) * D;
    const device T* Qb = Q + (size_t(wm * 16) * S + s) * D;

    ff O[OT * 2];
    for (int i = 0; i < OT * 2; ++i) O[i] = ff(0);

    const threadgroup int* ks = sel + wn * KPS;
    const int k00 = ks[sc.y], k01 = ks[sc.y + 8];
    const int k10 = ks[16 + sc.y], k11 = ks[16 + sc.y + 8];
    ff S0 = ff(0), S1 = ff(0);
    for (int id = 0; id < D / 16; ++id) {
        const int dc = id * 16 + sc.x;
        const T4 qa = *(const device T4*)(Qb + size_t(sc.y) * qstr + dc);
        const T4 qb = *(const device T4*)(Qb + size_t(sc.y + 8) * qstr + dc);
        const T4 z = T4(0);
        const T4 x00 = k00 >= 0 ? *(const device T4*)(KV + size_t(k00) * D + dc) : z;
        const T4 x01 = k01 >= 0 ? *(const device T4*)(KV + size_t(k01) * D + dc) : z;
        const T4 x10 = k10 >= 0 ? *(const device T4*)(KV + size_t(k10) * D + dc) : z;
        const T4 x11 = k11 >= 0 ? *(const device T4*)(KV + size_t(k11) * D + dc) : z;
        ft a, b0, b1;
        for (int j = 0; j < 4; ++j) {
            a[j] = qa[j]; a[4 + j] = qb[j];
            b0[j] = x00[j]; b0[4 + j] = x01[j];
            b1[j] = x10[j]; b1[4 + j] = x11[j];
        }
        Frag::mma(S0, S1, a, metal::false_type{}, b0, b1, metal::true_type{});
    }
    float mloc[2];
    for (int i = 0; i < 2; ++i) {
        float v = NEG;
        for (int j = 0; j < 4; ++j) {
            const int c = sc.x + j;
            float s0 = ks[c] >= 0 ? S0[i * 4 + j] * scale2 : NEG;
            float s1 = ks[16 + c] >= 0 ? S1[i * 4 + j] * scale2 : NEG;
            S0[i * 4 + j] = s0; S1[i * 4 + j] = s1;
            v = metal::max(v, metal::max(s0, s1));
        }
        v = metal::max(v, simd_shuffle_xor(v, ushort(1)));
        v = metal::max(v, simd_shuffle_xor(v, ushort(8)));
        mloc[i] = v;
    }
    if ((lane & 9) == 0) {
        rmax[sg * 16 + sc.y] = mloc[0];
        rmax[sg * 16 + sc.y + 8] = mloc[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mrow[2], slocal[2];
    for (int i = 0; i < 2; ++i) {
        const int row = sc.y + i * 8;
        float mn = NEG;
        for (int w = 0; w < WN; ++w) mn = metal::max(mn, rmax[(wm * WN + w) * 16 + row]);
        mrow[i] = mn;
        float sm = 0.0f;
        for (int j = 0; j < 4; ++j) {
            const float p0 = S0[i * 4 + j] == NEG ? 0.0f : fast::exp2(S0[i * 4 + j] - mn);
            const float p1 = S1[i * 4 + j] == NEG ? 0.0f : fast::exp2(S1[i * 4 + j] - mn);
            S0[i * 4 + j] = p0; S1[i * 4 + j] = p1;
            sm += p0 + p1;
        }
        sm += simd_shuffle_xor(sm, ushort(1));
        sm += simd_shuffle_xor(sm, ushort(8));
        slocal[i] = sm;
    }
    if ((lane & 9) == 0) {
        rsum[sg * 16 + sc.y] = slocal[0];
        rsum[sg * 16 + sc.y + 8] = slocal[1];
    }
    for (int ph = 0; ph < 2; ++ph) {
        if (wn / 2 == ph) {
            const int koff = (wn % 2) * 32;
            for (int i = 0; i < 2; ++i) {
                const int row = wm * 16 + sc.y + i * 8;
                for (int j = 0; j < 4; ++j) {
                    Pb[row * HALF + koff + sc.x + j] = S0[i * 4 + j];
                    Pb[row * HALF + koff + 16 + sc.x + j] = S1[i * 4 + j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kf = 0; kf < HALF / 16; ++kf) {
            ff a;
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 4; ++j)
                    a[i * 4 + j] = Pb[(wm * 16 + sc.y + i * 8) * HALF + kf * 16 + sc.x + j];
            const int kv0 = sel[ph * HALF + kf * 16 + sc.y];
            const int kv1 = sel[ph * HALF + kf * 16 + sc.y + 8];
            for (int o = 0; o < OT; ++o) {
                const int dc = wn * DW + o * 32 + sc.x;
                const T4 z = T4(0);
                const T4 v00 = kv0 >= 0 ? *(const device T4*)(KV + size_t(kv0) * D + dc) : z;
                const T4 v01 = kv1 >= 0 ? *(const device T4*)(KV + size_t(kv1) * D + dc) : z;
                const T4 v10 = kv0 >= 0 ? *(const device T4*)(KV + size_t(kv0) * D + dc + 16) : z;
                const T4 v11 = kv1 >= 0 ? *(const device T4*)(KV + size_t(kv1) * D + dc + 16) : z;
                ft b0, b1;
                for (int j = 0; j < 4; ++j) {
                    b0[j] = v00[j]; b0[4 + j] = v01[j];
                    b1[j] = v10[j]; b1[4 + j] = v11[j];
                }
                Frag::mma(O[2 * o], O[2 * o + 1], a, metal::false_type{}, b0, b1, metal::false_type{});
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // partials: pm/pl [S, NB, H], po [S, NB, H, D] (unnormalized, relative to pm)
    for (int i = 0; i < 2; ++i) {
        const int row = sc.y + i * 8;
        const int h = wm * 16 + row;
        const size_t pidx = (size_t(s) * NB + kb) * H + h;
        if (wn == 0 && (lane & 9) == 0) {
            float l = 0.0f;
            for (int w = 0; w < WN; ++w) l += rsum[(wm * WN + w) * 16 + row];
            pm[pidx] = mrow[i];
            pl[pidx] = l;
        }
        device float* orow = po + pidx * D + wn * DW;
        for (int o = 0; o < OT; ++o) {
            for (int j = 0; j < 4; ++j) {
                orow[o * 32 + sc.x + j] = O[2 * o][i * 4 + j];
                orow[o * 32 + 16 + sc.x + j] = O[2 * o + 1][i * 4 + j];
            }
        }
    }
"""

_MERGE_SRC = """
    // one thread per (s, h, d4)
    const uint gid = thread_position_in_grid.x;
    const int d4 = gid % (D / 4);
    const int h = (gid / (D / 4)) % H;
    const int s = gid / (D / 4) / H;
    const int NB = dims[5];
    float M = -3.402823466e+38f;
    for (int b = 0; b < NB; ++b) M = metal::max(M, pm[(size_t(s) * NB + b) * H + h]);
    float4 acc = 0.0f;
    float L = 0.0f;
    for (int b = 0; b < NB; ++b) {
        const size_t pidx = (size_t(s) * NB + b) * H + h;
        const float m = pm[pidx];
        if (m == -3.402823466e+38f) continue;
        const float f = fast::exp2(m - M);
        L += pl[pidx] * f;
        acc += *(const device float4*)(po + pidx * D + d4 * 4) * f;
    }
    const float inv = L > 0.0f ? 1.0f / L : 0.0f;
    const float4 r = acc * inv;
    device T* o = out + (size_t(h) * dims[0] + s) * D + d4 * 4;
    o[0] = static_cast<T>(r.x); o[1] = static_cast<T>(r.y); o[2] = static_cast<T>(r.z); o[3] = static_cast<T>(r.w);
"""

_kernels = None


def _get():
    global _kernels
    if _kernels is None:
        header = _inline("mlx/backend/metal/kernels/steel/attn/nax.h", set())
        _kernels = (
            mx.fast.metal_kernel(
                name="glm53_sparse_mla_decode_blocks",
                input_names=["Q", "KV", "win", "dims", "scale_in", "attn_mask"],
                output_names=["pm", "pl", "po"],
                source=_SRC,
                header=header,
            ),
            mx.fast.metal_kernel(
                name="glm53_sparse_mla_decode_merge",
                input_names=["pm", "pl", "po", "dims"],
                output_names=["out"],
                source=_MERGE_SRC,
            ),
        )
    return _kernels


def supports_sparse_mla_mask(mask, batch_size, query_rows, kv_length):
    if mask is None:
        return True
    if mask.dtype != mx.bool_ or mask.ndim not in (2, 3, 4):
        return False
    shape = mask.shape
    if mask.ndim == 4:
        if shape[1] != 1:
            return False
        shape = (shape[0], shape[2], shape[3])
    elif mask.ndim == 2:
        shape = (1, shape[0], shape[1])
    return all(
        value in (1, target)
        for value, target in zip(shape, (batch_size, query_rows, kv_length))
    )


def _attention_mask(mask, batch_size, query_rows, kv_length):
    if not supports_sparse_mla_mask(mask, batch_size, query_rows, kv_length):
        raise ValueError("Unsupported GLM split-K attention mask")
    if mask is None:
        return mx.zeros((1,), mx.bool_)
    if mask.ndim == 4:
        mask = mask[:, 0]
    return mx.broadcast_to(mask, (batch_size, query_rows, kv_length)).reshape(-1)


def _validate_attention(q, kv, windows, batch_size, query_rows):
    if q.ndim != 4 or kv.ndim != 4:
        raise ValueError("GLM split-K attention requires rank-four q and KV")
    if (
        q.shape[0] != batch_size
        or q.shape[2] != query_rows
        or q.shape[1] % _NAX_FRAGMENT_HEAD_ROWS
        or not _NAX_FRAGMENT_HEAD_ROWS
        <= q.shape[1]
        <= _NAX_MAX_HEAD_FRAGMENTS * _NAX_FRAGMENT_HEAD_ROWS
        or q.shape[-1] % _NAX_LATENT_TILE_ELEMENTS
        or q.shape[-1] < _NAX_LATENT_TILE_ELEMENTS
        or query_rows < 1
        or kv.shape[:2] != (batch_size, 1)
        or kv.shape[-1] != q.shape[-1]
        or kv.shape[2] < 1
        or q.dtype not in (mx.float16, mx.bfloat16)
        or kv.dtype != q.dtype
        or windows.shape[-1] < 1
        or windows.dtype != mx.int32
    ):
        raise ValueError("Unsupported GLM split-K attention inputs")


def sparse_mla_decode(
    q_latent, kv_latent, windows, before, n_pool, scale, kpool=4, mask=None
):
    """q_latent [1,H,S,D]; kv_latent [1,1,K,D]; windows [S,NSEL] int32 -> [1,H,S,D]."""
    if q_latent.ndim != 4 or windows.ndim != 2:
        raise ValueError("GLM split-K scalar input rank mismatch")
    if windows.shape[0] != q_latent.shape[2] or kpool < 1 or before < 0 or n_pool < 0:
        raise ValueError("GLM split-K scalar metadata mismatch")
    _validate_attention(q_latent, kv_latent, windows, 1, q_latent.shape[2])
    if (
        n_pool * kpool > kv_latent.shape[2]
        or before + q_latent.shape[2] > kv_latent.shape[2]
    ):
        raise ValueError("GLM split-K scalar positions exceed KV")
    _, heads, query_rows, latent_dim = q_latent.shape
    kv_length = kv_latent.shape[2]
    selected_count = windows.shape[-1]
    key_tile_rows = _SPLIT_K_KEY_TILE_ROWS
    key_blocks = (
        selected_count * kpool + kpool - 1 + key_tile_rows - 1
    ) // key_tile_rows
    dims = mx.array(
        [query_rows, kv_length, selected_count, n_pool, before, key_blocks],
        dtype=mx.int32,
    )
    attn_mask = _attention_mask(mask, 1, query_rows, kv_length)
    blocks, merge = _get()
    pm, pl, po = blocks(
        inputs=[
            q_latent,
            kv_latent,
            windows.astype(mx.int32),
            dims,
            mx.array([scale], dtype=mx.float32),
            attn_mask,
        ],
        template=[
            ("T", q_latent.dtype),
            ("H", heads),
            ("D", latent_dim),
            ("BK", key_tile_rows),
            ("WM", heads // _NAX_FRAGMENT_HEAD_ROWS),
            ("WN", _NAX_COLUMN_FRAGMENTS),
            ("KPOOL", kpool),
            ("HAS_MASK", mask is not None),
        ],
        grid=(key_blocks * heads * 8, query_rows, 1),
        threadgroup=(heads * 8, 1, 1),  # WM * WN simdgroups of 32 lanes
        output_shapes=[
            (query_rows, key_blocks, heads),
            (query_rows, key_blocks, heads),
            (query_rows, key_blocks, heads, latent_dim),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return merge(
        inputs=[pm, pl, po, dims],
        template=[("T", q_latent.dtype), ("H", heads), ("D", latent_dim)],
        grid=(query_rows * heads * (latent_dim // 4), 1, 1),
        threadgroup=(_MERGE_THREADGROUP_LANES, 1, 1),
        output_shapes=[(1, heads, query_rows, latent_dim)],
        output_dtypes=[q_latent.dtype],
    )[0]


# ---------------------------------------------------------------------------
# Batched (continuous batching) variant.
#
# Query rows are flattened to r = b * S + s. Per batch row b the kernel reads
# ``rows[b] = (n_pool, before, left_pad)``: windows are validated against the
# row's own pooled count and causal position ``before + s`` exactly as in the
# single-row kernel, and the resulting row-relative token t is read from KV
# index ``t + left_pad`` (mlx BatchKVCache left-pads, the pooling cache never
# does). KV is addressed through an explicit batch stride so the padded
# BatchKVCache buffer can be passed without copying its unused capacity.
# The math is textually the single-row kernel's; only addressing differs.
# ---------------------------------------------------------------------------


def _batched_source(src, edits):
    for old, new, count in edits:
        assert src.count(old) == count, (old, src.count(old))
        src = src.replace(old, new)
    return src


_B_SRC = _batched_source(
    _SRC,
    [
        (
            """    const int s = int(threadgroup_position_in_grid.y);    // query row
    const int S = dims[0];
    const int KVL = dims[1];
    const int NSEL = dims[2];     // selected windows per row
    const int P = dims[3];        // pooled windows available
    const int before = dims[4];   // tokens processed before this block
    const int NB = dims[5];       // key blocks
""",
            """    const int r = int(threadgroup_position_in_grid.y);    // batch row * S + query row
    const int S = dims[0];
    const int KVL = dims[1];
    const int NSEL = dims[2];     // selected windows per row
    const int NB = dims[5];       // key blocks
    const int bi = r / S;
    const int s = r - bi * S;
    const int P = rows[bi * 3];           // this row's pooled windows
    const int before = rows[bi * 3 + 1];  // row tokens processed before this block
    const int lpad = rows[bi * 3 + 2];    // KV left padding of this row
    const device T* KVb = KV + size_t(bi) * size_t(dims[6]);
""",
            1,
        ),
        ("win[s * NSEL + slot / KP]", "win[r * NSEL + slot / KP]", 1),
        (
            "sel[i] = (tok >= 0 && tok < KVL && (!HAS_MASK || attn_mask[s * KVL + tok])) ? tok : -1;",
            "sel[i] = (tok >= 0 && tok + lpad >= 0 && tok + lpad < KVL && (!HAS_MASK || attn_mask[r * KVL + tok + lpad])) ? tok + lpad : -1;",
            1,
        ),
        (
            "const device T* Qb = Q + (size_t(wm * 16) * S + s) * D;",
            "const device T* Qb = Q + size_t(bi) * H * qstr + (size_t(wm * 16) * S + s) * D;",
            1,
        ),
        ("(KV + size_t(", "(KVb + size_t(", 8),
        (
            "const size_t pidx = (size_t(s) * NB + kb) * H + h;",
            "const size_t pidx = (size_t(r) * NB + kb) * H + h;",
            1,
        ),
    ],
)

_B_MERGE_SRC = _batched_source(
    _MERGE_SRC,
    [
        (
            "const int s = gid / (D / 4) / H;",
            "const int qr = gid / (D / 4) / H;\n    const int bi = qr / dims[0];\n    const int s = qr - bi * dims[0];",
            1,
        ),
        ("(size_t(s) * NB + b)", "(size_t(qr) * NB + b)", 2),
        (
            "out + (size_t(h) * dims[0] + s) * D",
            "out + ((size_t(bi) * H + h) * dims[0] + s) * D",
            1,
        ),
    ],
)

_b_kernels = None


def _get_batched():
    global _b_kernels
    if _b_kernels is None:
        header = _inline("mlx/backend/metal/kernels/steel/attn/nax.h", set())
        _b_kernels = (
            mx.fast.metal_kernel(
                name="glm53_sparse_mla_decode_blocks_batched",
                input_names=["Q", "KV", "win", "dims", "scale_in", "rows", "attn_mask"],
                output_names=["pm", "pl", "po"],
                source=_B_SRC,
                header=header,
            ),
            mx.fast.metal_kernel(
                name="glm53_sparse_mla_decode_merge_batched",
                input_names=["pm", "pl", "po", "dims"],
                output_names=["out"],
                source=_B_MERGE_SRC,
            ),
        )
    return _b_kernels


def sparse_mla_decode_batched(
    q_latent, kv, kv_len, windows, rows, scale, kpool=4, mask=None
):
    """Batched window-index sparse MLA decode.

    q_latent [B,H,S,D]; kv [B,1,C,D] with C >= kv_len (a BatchKVCache buffer,
    only the first ``kv_len`` positions are addressed); windows [B,S,NSEL];
    rows [B,3] int32 = (n_pool, before, left_pad) per batch row -> [B,H,S,D].
    """
    if q_latent.ndim != 4 or windows.ndim != 3 or rows.ndim != 2:
        raise ValueError("GLM split-K batch input rank mismatch")
    batch_size, heads, query_rows, latent_dim = q_latent.shape
    if (
        windows.shape[:2] != (batch_size, query_rows)
        or rows.shape != (batch_size, 3)
        or rows.dtype != mx.int32
    ):
        raise ValueError("GLM split-K batch metadata mismatch")
    _validate_attention(q_latent, kv, windows, batch_size, query_rows)
    if kpool < 1 or not 1 <= kv_len <= kv.shape[2]:
        raise ValueError("GLM split-K batch KV extent mismatch")
    capacity = kv.shape[2]
    selected_count = windows.shape[-1]
    key_tile_rows = _SPLIT_K_KEY_TILE_ROWS
    key_blocks = (
        selected_count * kpool + kpool - 1 + key_tile_rows - 1
    ) // key_tile_rows
    dims = mx.array(
        [query_rows, kv_len, selected_count, 0, 0, key_blocks, capacity * latent_dim],
        dtype=mx.int32,
    )
    attn_mask = _attention_mask(mask, batch_size, query_rows, kv_len)
    blocks, merge = _get_batched()
    pm, pl, po = blocks(
        inputs=[
            q_latent,
            kv,
            windows.astype(mx.int32),
            dims,
            mx.array([scale], dtype=mx.float32),
            rows.astype(mx.int32),
            attn_mask,
        ],
        template=[
            ("T", q_latent.dtype),
            ("H", heads),
            ("D", latent_dim),
            ("BK", key_tile_rows),
            ("WM", heads // _NAX_FRAGMENT_HEAD_ROWS),
            ("WN", _NAX_COLUMN_FRAGMENTS),
            ("KPOOL", kpool),
            ("HAS_MASK", mask is not None),
        ],
        grid=(key_blocks * heads * 8, batch_size * query_rows, 1),
        threadgroup=(heads * 8, 1, 1),
        output_shapes=[
            (batch_size * query_rows, key_blocks, heads),
            (batch_size * query_rows, key_blocks, heads),
            (batch_size * query_rows, key_blocks, heads, latent_dim),
        ],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return merge(
        inputs=[pm, pl, po, dims],
        template=[("T", q_latent.dtype), ("H", heads), ("D", latent_dim)],
        grid=(batch_size * query_rows * heads * (latent_dim // 4), 1, 1),
        threadgroup=(_MERGE_THREADGROUP_LANES, 1, 1),
        output_shapes=[(batch_size, heads, query_rows, latent_dim)],
        output_dtypes=[q_latent.dtype],
    )[0]
