# ruff: noqa
# Vendored verbatim from KRAFTON/Raon-1-mini-preview-mlx (raon_mlx_model.py,
# revision c558ec0f8050925e44a80e7ee357f41fbcf8cfe5, sha256 82a5c6bc...5296).
# Apache-2.0 (KRAFTON) with MIT portions from Apple mlx-lm; see the notice below.
# Portions Copyright © 2025-2026 Apple Inc., MIT License:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Copyright © 2026 KRAFTON Inc.
"""Metal kernels for Raon single-token bfloat16 inference."""

import mlx.core as mx


def cached_array(module, name, make):
    """Per-module derived constant, computed once and kept out of the parameter tree."""
    if module.training:
        return make()
    value = module.__dict__.get(name)
    if value is None:
        value = make()
        mx.eval(value)
        object.__setattr__(module, name, value)
    return value


_FUSED_TG = 512


_FUSED_BLK = 1024


_FUSED_SRC1 = r"""
    const uint row = threadgroup_position_in_grid.y;
    const uint blk = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint T = threads_per_threadgroup.x;
    const int H = H_DIM;
    const int S = 4;
    const int K = 17;
    const int D = S * H;
    const int B = BLK;
    threadgroup float red[32][K + 1];
    float acc[K + 1];
    for (int k = 0; k <= K; ++k) acc[k] = 0.0f;
    const device bfloat16_t* rin = rows + row * D;
    device bfloat16_t* rout = new_rows + row * D;
    const device bfloat16_t* br = branch + row * H;
    const int start = blk * B;
    const int stop = metal::min(start + B, D);
    for (int idx = start + tid; idx < stop; idx += T) {
        const int s = idx / H;
        const int h = idx - s * H;
        float v;
        if (DO_MIX) {
            float m = 0.0f;
            for (int j = 0; j < S; ++j) m += res_in[row * 16 + s * 4 + j] * (float)rin[j * H + h];
            m += post_in[row * 4 + s] * (float)br[h];
            bfloat16_t mb = (bfloat16_t)m;
            rout[idx] = mb;
            v = (float)mb;
        } else {
            v = (float)rin[idx];
            rout[idx] = rin[idx];
        }
        acc[K] += v * v;
        for (int k = 0; k < K; ++k) acc[k] += v * (float)phi[k * D + idx];
    }
    const uint lane = tid % 32;
    const uint sg = tid / 32;
    for (int k = 0; k <= K; ++k) {
        float t = simd_sum(acc[k]);
        if (lane == 0) red[sg][k] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <= (uint)K) {
        float t = 0.0f;
        for (uint g = 0; g < (T + 31) / 32; ++g) t += red[g][tid];
        partial[(row * NBLK + blk) * (K + 1) + tid] = t;
    }
"""


_FUSED_SRC2 = r"""
    const uint row = threadgroup_position_in_grid.y;
    const uint hb = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint T = threads_per_threadgroup.x;
    const int H = H_DIM;
    const int S = 4;
    const int K = 17;
    const int D = S * H;
    threadgroup float tail[4 + 4 + 16];
    if (tid == 0) {
        float tot[K + 1];
        for (int k = 0; k <= K; ++k) {
            float t = 0.0f;
            for (int b = 0; b < NBLK; ++b) t += partial[(row * NBLK + b) * (K + 1) + k];
            tot[k] = t;
        }
        const float inv = metal::rsqrt(tot[K] / (float)D + prm[0]);
        float z[K];
        for (int k = 0; k < K; ++k) z[k] = (float)(bfloat16_t)tot[k] * inv;
        for (int i = 0; i < 4; ++i) {
            tail[i] = 1.0f / (1.0f + metal::exp(-(prm[2] * z[i] + prm[3 + i])));
            tail[4 + i] = prm[1] / (1.0f + metal::exp(-(prm[7] * z[4 + i] + prm[8 + i])));
        }
        float c[9];
        for (int k = 0; k < 9; ++k) c[k] = prm[12] * z[8 + k] + prm[13 + k];
        float e[16];
        float rho = 0.0f;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) {
                float t = 0.0f;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        t += prm[23 + i * 3 + a] * c[a * 3 + b] * prm[23 + j * 3 + b];
                e[i * 4 + j] = t;
                rho = metal::max(rho, -t / prm[35 + i * 4 + j]);
            }
        const float g = prm[22];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                tail[8 + i * 4 + j] = (1.0f - g) * (i == j ? 1.0f : 0.0f)
                                      + g * (prm[35 + i * 4 + j] + e[i * 4 + j] / (1.0f + rho));
        if (hb == 0) {
            for (int i = 0; i < 4; ++i) post_out[row * 4 + i] = tail[4 + i];
            for (int k = 0; k < 16; ++k) res_out[row * 16 + k] = tail[8 + k];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int h = hb * T + tid;
    if (h < H) {
        float a = 0.0f;
        for (int s = 0; s < S; ++s) a += (float)new_rows[row * D + s * H + h] * tail[s];
        agg[row * H + h] = (bfloat16_t)a;
    }
"""


_FUSED_KERNELS = {}


def _fused_kernels():
    if not _FUSED_KERNELS:
        _FUSED_KERNELS["k1"] = mx.fast.metal_kernel(
            name="raon_mhc_pass1",
            input_names=["rows", "branch", "post_in", "res_in", "phi"],
            output_names=["new_rows", "partial"],
            source=_FUSED_SRC1,
        )
        _FUSED_KERNELS["k2"] = mx.fast.metal_kernel(
            name="raon_mhc_pass2",
            input_names=["partial", "new_rows", "prm"],
            output_names=["agg", "post_out", "res_out"],
            source=_FUSED_SRC2,
        )
    return _FUSED_KERNELS["k1"], _FUSED_KERNELS["k2"]


def fused_mhc(rows, branch, post, res, mhc, do_mix):
    """rows (N,4,H) bf16; branch (N,H) bf16 or None; post (N,4) f32; res (N,4,4) f32.
    Returns new_rows, agg (N,H), post_next, res_next for `mhc`."""
    n, s, h = rows.shape
    if branch is None:
        branch = mx.zeros((n, h), mx.bfloat16)
        post = mx.zeros((n, 4), mx.float32)
        res = mx.zeros((n, 4, 4), mx.float32)
    d = s * h
    nblk = (d + _FUSED_BLK - 1) // _FUSED_BLK
    k1, k2 = _fused_kernels()
    new_rows, partial = k1(
        inputs=[
            rows,
            branch,
            post.astype(mx.float32),
            res.astype(mx.float32),
            mhc.fused_phi(),
        ],
        template=[
            ("H_DIM", h),
            ("DO_MIX", int(bool(do_mix))),
            ("BLK", _FUSED_BLK),
            ("NBLK", nblk),
        ],
        grid=(_FUSED_TG * nblk, n, 1),
        threadgroup=(_FUSED_TG, 1, 1),
        output_shapes=[(n, s, h), (n, nblk, 18)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    hb = (h + _FUSED_TG - 1) // _FUSED_TG
    agg, post_out, res_out = k2(
        inputs=[partial, new_rows, mhc.fused_params()],
        template=[("H_DIM", h), ("NBLK", nblk)],
        grid=(_FUSED_TG * hb, n, 1),
        threadgroup=(_FUSED_TG, 1, 1),
        output_shapes=[(n, h), (n, 4), (n, 4, 4)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32],
    )
    return new_rows, agg, post_out, res_out


_FUSED_SRC2_NORM = r"""
    const uint row = threadgroup_position_in_grid.y;
    const uint hb = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint T = threads_per_threadgroup.x;
    const int H = H_DIM;
    const int S = 4;
    const int K = 17;
    const int D = S * H;
    threadgroup float tail[4 + 4 + 16];
    threadgroup float tots[K + 1];
    if (tid <= (uint)K) {
        float t = 0.0f;
        for (int b = 0; b < NBLK; ++b) t += partial[(row * NBLK + b) * (K + 1) + tid];
        tots[tid] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float zs[K];
    threadgroup float cs[9];
    threadgroup float es[16];
    threadgroup float rhos[1];
    const float inv_t = metal::rsqrt(tots[K] / (float)D + prm[0]);
    if (tid < (uint)K) zs[tid] = (float)(bfloat16_t)tots[tid] * inv_t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 4) {
        tail[tid] = 1.0f / (1.0f + metal::exp(-(prm[2] * zs[tid] + prm[3 + tid])));
    } else if (tid < 8) {
        const int i = tid - 4;
        tail[4 + i] = prm[1] / (1.0f + metal::exp(-(prm[7] * zs[4 + i] + prm[8 + i])));
    } else if (tid < 17) {
        const int k = tid - 8;
        cs[k] = prm[12] * zs[8 + k] + prm[13 + k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) {
        const int i = tid / 4, j = tid % 4;
        float t = 0.0f;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                t += prm[23 + i * 3 + a] * cs[a * 3 + b] * prm[23 + j * 3 + b];
        es[tid] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float rho = 0.0f;
        for (int ij = 0; ij < 16; ++ij) rho = metal::max(rho, -es[ij] / prm[35 + ij]);
        rhos[0] = rho;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 16) {
        const int i = tid / 4, j = tid % 4;
        const float g = prm[22];
        tail[8 + tid] = (1.0f - g) * (i == j ? 1.0f : 0.0f) + g * (prm[35 + tid] + es[tid] / (1.0f + rhos[0]));
        res_out[row * 16 + tid] = tail[8 + tid];
        if (tid < 4) post_out[row * 4 + tid] = tail[4 + tid];
    }
    threadgroup float local_sums[32];
    threadgroup float local_inv[1];
    const uint lane = tid % 32;
    const uint sg = tid / 32;
    if (sg == 0) local_sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float xs[4];
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int h = tid * 4 + i;
        float x = 0.0f;
        if (h < H) {
            float a = 0.0f;
            for (int s = 0; s < S; ++s) a += (float)new_rows[row * D + s * H + h] * tail[s];
            x = (float)(bfloat16_t)a;
        }
        xs[i] = x;
        acc += x * x;
    }
    acc = simd_sum(acc);
    if (lane == 0) local_sums[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv[0] = metal::precise::rsqrt(acc / (uint)H + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 4; ++i) {
        const int h = tid * 4 + i;
        if (h < H) {
            const bfloat16_t o = w[h] * static_cast<bfloat16_t>(xs[i] * local_inv[0]);
            normed[row * H + h] = o;
            if (WITH_F32) normed32[row * H + h] = static_cast<float>(o);
        }
    }
"""


_FUSED_SRC1_MOE = r"""
    const uint row = threadgroup_position_in_grid.y;
    const uint blk = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint T = threads_per_threadgroup.x;
    const int H = H_DIM;
    const int S = 4;
    const int K = 17;
    const int D = S * H;
    const int B = BLK;
    threadgroup float red[32][K + 1];
    float acc[K + 1];
    for (int k = 0; k <= K; ++k) acc[k] = 0.0f;
    const device bfloat16_t* rin = rows + row * D;
    device bfloat16_t* rout = new_rows + row * D;
    const int start = blk * B;
    const int stop = metal::min(start + B, D);
    for (int idx = start + tid; idx < stop; idx += T) {
        const int s = idx / H;
        const int h = idx - s * H;
        float v;
        if (DO_MIX) {
            float m = 0.0f;
            for (int j = 0; j < S; ++j) m += res_in[row * 16 + s * 4 + j] * (float)rin[j * H + h];
            float t = 0.0f;
            for (int kk = 0; kk < TOPK; ++kk) {
                const float pr = (float)mo[(row * TOPK + kk) * H + h] * mw[row * TOPK + kk];
                t += as_type<float>(as_type<uint>(pr));
            }
            const bfloat16_t bv = static_cast<bfloat16_t>(t) + sh[row * H + h];
            m += post_in[row * 4 + s] * (float)bv;
            bfloat16_t mb = (bfloat16_t)m;
            rout[idx] = mb;
            v = (float)mb;
        } else {
            v = (float)rin[idx];
            rout[idx] = rin[idx];
        }
        acc[K] += v * v;
        for (int k = 0; k < K; ++k) acc[k] += v * (float)phi[k * D + idx];
    }
    const uint lane = tid % 32;
    const uint sg = tid / 32;
    for (int k = 0; k <= K; ++k) {
        float t = simd_sum(acc[k]);
        if (lane == 0) red[sg][k] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid <= (uint)K) {
        float t = 0.0f;
        for (uint g = 0; g < (T + 31) / 32; ++g) t += red[g][tid];
        partial[(row * NBLK + blk) * (K + 1) + tid] = t;
    }
"""


_FINAL_SRC = r"""
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = tid % 32;
    const uint sg = tid / 32;
    const int H = H_DIM;
    const int S = 4;
    threadgroup float local_sums[32];
    threadgroup float local_inv[1];
    if (sg == 0) local_sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device bfloat16_t* rin = rows + row * S * H;
    float xs[4];
    float acc = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int h = tid * 4 + i;
        float x = 0.0f;
        if (h < H) {
            float t = 0.0f;
            for (int kk = 0; kk < TOPK; ++kk) {
                const float pr = (float)mo[(row * TOPK + kk) * H + h] * mw[row * TOPK + kk];
                t += as_type<float>(as_type<uint>(pr));
            }
            const bfloat16_t bv = static_cast<bfloat16_t>(t) + sh[row * H + h];
            float sm = 0.0f;
            for (int s = 0; s < S; ++s) {
                float m = 0.0f;
                for (int j = 0; j < S; ++j) m += res_in[row * 16 + s * 4 + j] * (float)rin[j * H + h];
                const float pb = post_in[row * 4 + s] * (float)bv;
                m += as_type<float>(as_type<uint>(pb));
                sm += (float)(bfloat16_t)m;
            }
            x = (float)static_cast<bfloat16_t>(sm * 0.25f);
        }
        xs[i] = x;
        acc += x * x;
    }
    acc = simd_sum(acc);
    if (lane == 0) local_sums[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = simd_sum(local_sums[lane]);
        if (lane == 0) local_inv[0] = metal::precise::rsqrt(acc / (uint)H + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 4; ++i) {
        const int h = tid * 4 + i;
        if (h < H) normed[row * H + h] = w[h] * static_cast<bfloat16_t>(xs[i] * local_inv[0]);
    }
"""


_ROUTER_SRC = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint e = thread_position_in_threadgroup.x;
    const int E = N_EXPERTS;
    const int K = TOP_K;
    threadgroup float sv[N_EXPERTS];
    threadgroup float bv[32];
    threadgroup int bi[32];
    threadgroup int chosen[TOP_K];
    auto sigmoid = [](float z) { float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(z))); return (z < 0.0f) ? y : 1.0f - y; };
    const float s = sigmoid(logits[row * E + e]);
    float r = -(s + bias[e]);
    sv[e] = s;
    const uint lane = e % 32, sg = e / 32;
    for (int j = 0; j < K; ++j) {
        float v = r;
        int i = (int)e;
        for (uint off = 16; off > 0; off >>= 1) {
            float v2 = simd_shuffle_down(v, off);
            int i2 = simd_shuffle_down(i, off);
            if (v2 < v || (v2 == v && i2 < i)) { v = v2; i = i2; }
        }
        if (lane == 0) { bv[sg] = v; bi[sg] = i; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (e == 0) {
            float bvv = bv[0];
            int bii = bi[0];
            for (int g = 1; g < E / 32; ++g) {
                if (bv[g] < bvv || (bv[g] == bvv && bi[g] < bii)) { bvv = bv[g]; bii = bi[g]; }
            }
            chosen[j] = bii;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((int)e == chosen[j]) r = INFINITY;
    }
    if (e == 0) {
        float total = 0.0f;
        for (int j = 0; j < K; ++j) total = sv[chosen[j]] + total;
        const float denom = total + 1e-20f;
        for (int j = 0; j < K; ++j) {
            indices[row * K + j] = (uint32_t)chosen[j];
            weights[row * K + j] = metal::precise::divide(sv[chosen[j]], denom) * scale[0];
        }
    }
"""


# Adapted from the gated delta-rule kernel in mlx-lm (MIT License), with the output RMS norm and
# sigmoid gate fused in.
_RDA_OUT_SRC = r"""
    auto n = threadgroup_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;
    threadgroup float ys[Dv];
    auto q_ = q + b_idx * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * Hv * Dv + hv_idx * Dv;
    auto g_ = g + (b_idx * Hv + hv_idx) * Dk;
    const float bt = beta[b_idx * Hv + hv_idx];
    auto dk_idx = thread_position_in_threadgroup.x;
    auto rg = thread_position_in_threadgroup.y;
    float qv[n_per_t], kv[n_per_t], gv[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      qv[i] = q_[s_idx];
      kv[i] = k_[s_idx];
      gv[i] = g_[s_idx];
    }
    for (int r = 0; r < ROWS; ++r) {
      auto dv_idx = rg * ROWS + r;
      auto i_state = state_in + (n * Dv + dv_idx) * Dk;
      auto o_state = state_out + (n * Dv + dv_idx) * Dk;
      float state[n_per_t];
      for (int i = 0; i < n_per_t; ++i)
        state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        state[i] = state[i] * gv[i];
        kv_mem += state[i] * kv[i];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (v_[dv_idx] - kv_mem) * bt;
      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        state[i] = state[i] + kv[i] * delta;
        out += state[i] * qv[i];
      }
      out = simd_sum(out);
      if (dk_idx == 0)
        ys[dv_idx] = static_cast<float>(static_cast<InT>(out));
      for (int i = 0; i < n_per_t; ++i)
        o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rg == 0) {
      float xs[4];
      float acc = 0.0f;
      for (int i = 0; i < 4; ++i) {
        const int h = dk_idx * 4 + i;
        xs[i] = (h < Dv) ? ys[h] : 0.0f;
        acc += xs[i] * xs[i];
      }
      acc = simd_sum(acc);
      acc = simd_sum(dk_idx == 0 ? acc : 0.0f);
      const float inv = metal::precise::rsqrt(acc / (uint)Dv + eps[0]);
      auto gate_ = gate + (b_idx * Hv + hv_idx) * Dv;
      auto o_ = gated + (b_idx * Hv + hv_idx) * Dv;
      for (int i = 0; i < 4; ++i) {
        const int h = dk_idx * 4 + i;
        if (h < Dv) {
          const float nv = norm_w[h] * (xs[i] * inv);
          const float z = static_cast<float>(gate_[h]);
          const float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(z)));
          o_[h] = static_cast<bfloat16_t>(nv * ((z < 0.0f) ? e : 1.0f - e));
        }
      }
    }
"""


_FUSION_KERNELS = {}


def decode_kernel(name):
    if name not in _FUSION_KERNELS:
        spec = {
            "pass1_moe": (
                ["rows", "mo", "mw", "sh", "post_in", "res_in", "phi"],
                ["new_rows", "partial"],
                _FUSED_SRC1_MOE,
            ),
            "pass2_norm": (
                ["partial", "new_rows", "prm", "w", "eps"],
                ["normed", "post_out", "res_out", "normed32"],
                _FUSED_SRC2_NORM,
            ),
            "final": (
                ["rows", "mo", "mw", "sh", "post_in", "res_in", "w", "eps"],
                ["normed"],
                _FINAL_SRC,
            ),
            "router": (
                ["logits", "bias", "scale"],
                ["indices", "weights"],
                _ROUTER_SRC,
            ),
            "rda_out": (
                ["q", "k", "v", "g", "beta", "state_in", "norm_w", "gate", "eps"],
                ["gated", "state_out"],
                _RDA_OUT_SRC,
            ),
        }[name]
        _FUSION_KERNELS[name] = mx.fast.metal_kernel(
            name=f"raon_{name}",
            input_names=spec[0],
            output_names=spec[1],
            source=spec[2],
        )
    return _FUSION_KERNELS[name]


def _norm_threads(h):
    return 32 * ((h + 127) // 128)


def fused_mhc_norm(rows, branch, post, res, mhc, do_mix, norm, with_f32=False):
    """fused_mhc followed by `norm` on the aggregate. `branch` is a bf16 (N,H) array, or the MoE parts
    (routed out (N,K,H) bf16, weights (N,K) f32, shared (N,H) bf16) to combine in pass1.
    Returns new_rows, normed (N,H) bf16, post_next, res_next and, with `with_f32`, the f32 copy of normed.
    """
    n, s, h = rows.shape
    if branch is None:
        branch = mx.zeros((n, h), mx.bfloat16)
        post = mx.zeros((n, 4), mx.float32)
        res = mx.zeros((n, 4, 4), mx.float32)
    nblk = (s * h + _FUSED_BLK - 1) // _FUSED_BLK
    template = [
        ("H_DIM", h),
        ("DO_MIX", int(bool(do_mix))),
        ("BLK", _FUSED_BLK),
        ("NBLK", nblk),
    ]
    tail = [post.astype(mx.float32), res.astype(mx.float32), mhc.fused_phi()]
    if isinstance(branch, tuple):
        mo, mw, sh = branch
        k = mw.shape[-1]
        kernel, inputs = decode_kernel("pass1_moe"), [
            rows,
            mo.reshape(n, k, h),
            mw.reshape(n, k),
            sh.reshape(n, h),
        ]
        template.append(("TOPK", k))
    else:
        kernel, inputs = _fused_kernels()[0], [rows, branch]
    new_rows, partial = kernel(
        inputs=inputs + tail,
        template=template,
        grid=(_FUSED_TG * nblk, n, 1),
        threadgroup=(_FUSED_TG, 1, 1),
        output_shapes=[(n, s, h), (n, nblk, 18)],
        output_dtypes=[mx.bfloat16, mx.float32],
    )
    eps = cached_array(norm, "_eps32", lambda: mx.array([norm.eps], mx.float32))
    tg = _norm_threads(h)
    normed, post_out, res_out, normed32 = decode_kernel("pass2_norm")(
        inputs=[partial, new_rows, mhc.fused_params(), norm.weight, eps],
        template=[("H_DIM", h), ("NBLK", nblk), ("WITH_F32", int(with_f32))],
        grid=(tg, n, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n, h), (n, 4), (n, 4, 4), (n, h) if with_f32 else (1,)],
        output_dtypes=[mx.bfloat16, mx.float32, mx.float32, mx.float32],
    )
    return (new_rows, normed, post_out, res_out) + ((normed32,) if with_f32 else ())


def fused_final(rows, parts, post, res, norm):
    """Last MoE combine, last mHC mix, mean over the streams and `norm`: (N,H) bf16."""
    mo, mw, sh = parts
    n, s, h = rows.shape
    k = mw.shape[-1]
    eps = cached_array(norm, "_eps32", lambda: mx.array([norm.eps], mx.float32))
    tg = _norm_threads(h)
    return decode_kernel("final")(
        inputs=[
            rows,
            mo.reshape(n, k, h),
            mw.reshape(n, k),
            sh.reshape(n, h),
            post.astype(mx.float32),
            res.astype(mx.float32),
            norm.weight,
            eps,
        ],
        template=[("H_DIM", h), ("TOPK", k)],
        grid=(tg, n, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(n, h)],
        output_dtypes=[mx.bfloat16],
    )[0]


_RDA_PREP_SOURCE = r"""

    const uint b = threadgroup_position_in_grid.y;
    const uint h = threadgroup_position_in_grid.x;
    const uint d = thread_position_in_threadgroup.x;
    const int D = HEAD_DIM;
    const int NH = N_HEADS;
    const int P = NH * D;
    const int KS = KSIZE;
    const int KEEP = KS - 1;

    float outv[3];
    for (int part = 0; part < 3; ++part) {
        const int c = part * P + h * D + d;               // channel in the merged q|k|v conv
        float acc = 0.0f;
        for (int t = 0; t < KEEP; ++t)
            acc += (float)state[(b * KEEP + t) * 3 * P + c] * w32[c * KS + t];
        const float xin = (float)x[b * 3 * P + c];
        acc += xin * w32[c * KS + KEEP];
        for (int t = 0; t < KEEP - 1; ++t)
            new_state[(b * KEEP + t) * 3 * P + c] = state[(b * KEEP + t + 1) * 3 * P + c];
        new_state[(b * KEEP + KEEP - 1) * 3 * P + c] = x[b * 3 * P + c];
        const float ys = 1.0f / (1.0f + metal::precise::exp(metal::abs(acc)));
        const float s = acc * ((acc < 0.0f) ? ys : 1.0f - ys);
        outv[part] = (float)(bfloat16_t)s;
    }
    // Sum in the order of the MLX row reduction (32 lanes, D / 32 values each) to match _rda_qk.
    threadgroup float sq[2][HEAD_DIM];
    threadgroup float tot[2];
    sq[0][d] = outv[0] * outv[0];
    sq[1][d] = outv[1] * outv[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d < 32) {
      float pq = 0.0f, pk = 0.0f;
      for (int i = 0; i < D / 32; ++i) { pq = sq[0][(D / 32) * d + i] + pq; pk = sq[1][(D / 32) * d + i] + pk; }
      pq = simd_sum(pq);
      pk = simd_sum(pk);
      if (d == 0) { tot[0] = pq; tot[1] = pk; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float qs = tot[0], ks = tot[1];
    const float qn = (float)(bfloat16_t)metal::precise::divide(outv[0], metal::max(metal::precise::sqrt(qs), 1e-6f));
    const float kn = (float)(bfloat16_t)metal::precise::divide(outv[1], metal::max(metal::precise::sqrt(ks), 1e-6f));
    const int o = (b * NH + h) * D + d;
    q_out[o] = (bfloat16_t)(qn * (float)(bfloat16_t)fp[0]);
    k_out[o] = (bfloat16_t)kn;
    const float br = (float)braw[b * NH + h];
    const float dl = (float)delta[b * NH + h];
    // Same formulas as the MLX ops, so the results match _rda_gates.
    auto sigmoid = [](float z) { float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(z))); return (z < 0.0f) ? y : 1.0f - y; };
    auto log1p_mlx = [](float z) { float xp1 = 1.0f + z; if (xp1 == 1.0f) return z; return z * (metal::log(xp1) / (xp1 - 1.0f)); };
    auto softplus = [&](float z) {
      float mx_ = metal::max(z, 0.0f), mn_ = metal::min(z, 0.0f);
      return (metal::isinf(mn_) || metal::isinf(mx_)) ? mx_ : mx_ + log1p_mlx(metal::exp(mn_ - mx_)); };
    const float ratio = metal::precise::exp(-softplus(-br - dl) + softplus(-br));
    v_out[o] = (bfloat16_t)(outv[2] * ratio);
    const float dec = ((float)decay[b * P + h * D + d] + dt32[h * D + d]) * a_exp[h];
    g_out[o] = metal::precise::exp(fp[1] * sigmoid(dec));
    if (d == 0) beta_out[b * NH + h] = (bfloat16_t)sigmoid(br);
"""


_RDA_PREP_KERNEL = {}


def rda_prep_kernel():
    if "k" not in _RDA_PREP_KERNEL:
        _RDA_PREP_KERNEL["k"] = mx.fast.metal_kernel(
            name="raon_rda_decode_prep",
            input_names=[
                "x",
                "state",
                "w32",
                "braw",
                "delta",
                "decay",
                "dt32",
                "a_exp",
                "fp",
            ],
            output_names=["q_out", "k_out", "v_out", "g_out", "beta_out", "new_state"],
            source=_RDA_PREP_SOURCE,
        )
    return _RDA_PREP_KERNEL["k"]


# Copyright © 2026 KRAFTON Inc.
"""Raon with merged projections and Metal decode kernels.

Prefill and unsupported devices or dtypes use MLX operations. The Metal
mHC path changes float32 accumulation order relative to the operation path.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_ops
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
    _gather_sort,
    _scatter_unsort,
)


# rda_config keys this model reads; any other key is an error.
_RDA_CONFIG_KEYS = frozenset(
    {"num_heads", "head_dim", "conv_kernel_size", "full_rank_gate", "gate_lower_bound", "rda_layers",
     "mla_layers"}
)

# Adapted from mlx-lm (MIT License).
# SiTU activation: beta * tanh(gate / beta) * sigmoid(gate) times linear_beta * tanh(x / linear_beta), computed in float32
# and rounded once to the input dtype, as one compiled kernel.
@partial(mx.compile, shapeless=True)
def _situ(x, gate, beta, linear_beta):
    dtype = x.dtype
    gate = gate.astype(mx.float32)
    x = x.astype(mx.float32)
    a = beta * mx.tanh(gate / beta) * mx.sigmoid(gate)
    if linear_beta is not None:
        x = linear_beta * mx.tanh(x / linear_beta)
    return (a * x).astype(dtype)


class SiTU(nn.Module):
    def __init__(self, beta: float = 1.0, linear_beta: Optional[float] = None):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        return _situ(x, gate, self.beta, self.linear_beta)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "raon1_mini"
    hidden_size: int = 2560
    num_hidden_layers: int = 25
    num_attention_heads: int = 40
    head_dim: int = 128
    vocab_size: int = 259840
    intermediate_size: int = 12032
    rms_norm_eps: float = 1e-5
    mla_latent_norm_eps: float = 1e-6
    raon_mhc_selector_eps: float = 1e-5
    raon_mhc_streams: int = 4
    raon_mhc_post_factor: float = 2.0
    hidden_act: str = "situ"
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    num_experts: Optional[int] = 128
    num_experts_per_token: int = 6
    moe_intermediate_size: int = 1088
    num_shared_experts: int = 1
    first_k_dense_replace: int = 1
    moe_layer_freq: int = 1
    moe_router_activation_func: str = "sigmoid"
    moe_renormalize: bool = True
    routed_scaling_factor: float = 1.0
    use_grouped_topk: bool = False
    routed_expert_hidden_size: Optional[int] = None
    latent_moe_use_norm: bool = False
    q_lora_rank: int = 576
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True
    rda_config: Optional[dict] = None
    rda_decay_rank: int = 128
    num_nextn_predict_layers: int = 1
    attn_res_block_size: Optional[int] = None
    tie_word_embeddings: bool = False
    raon_fp32_export_head: bool = False
    raon_fused_mhc_decode: bool = True
    raon_global_scaled_linears: Optional[list] = None
    raon_merged_gate_up: bool = True

    def __post_init__(self):
        if self.hidden_act != "situ" or self.moe_router_activation_func != "sigmoid":
            raise ValueError("RAON requires SiTU activation and sigmoid routing")
        if (
            self.use_grouped_topk
            or self.routed_expert_hidden_size is not None
            or self.latent_moe_use_norm
        ):
            raise ValueError("Grouped routing and latent MoE are not supported")
        if not self.mla_use_nope or self.attn_res_block_size is not None:
            raise ValueError("RAON requires NoPE MLA and mHC residual streams")
        if self.raon_mhc_streams < 2 or self.moe_layer_freq < 1:
            raise ValueError("Invalid stream count or MoE layer frequency")
        if self.num_experts and not 0 < self.num_experts_per_token <= self.num_experts:
            raise ValueError("Invalid number of active experts")
        if self.rda_config is None:
            raise ValueError("config must set rda_config")
        unknown = set(self.rda_config) - _RDA_CONFIG_KEYS
        missing = {"rda_layers", "mla_layers"} - set(self.rda_config)
        if unknown or missing:
            raise ValueError(f"rda_config keys: unknown {sorted(unknown)}, missing {sorted(missing)}")
        if not self.rda_config.get("full_rank_gate", True):
            raise ValueError("Only full-rank RDA gates are supported")
        linear = self.rda_config["rda_layers"]
        full = self.rda_config["mla_layers"]
        if sorted(linear + full) != list(range(1, self.num_hidden_layers + 1)):
            raise ValueError("Attention layer lists must partition all layers")


_DECODE_ROWS = 8


def _qparams(m):
    """(kind, group_size, bits, mode) of a quantized linear without bias, or None."""
    if isinstance(m, GlobalScaledLinear):
        return ("gs", 16, 4, "nvfp4")
    if isinstance(m, nn.QuantizedLinear) and "bias" not in m:
        return ("q", m.group_size, m.bits, m.mode)
    return None


def _merged_decode(module, names):
    """A cached merged copy of `names` (same input, same quantization) for decode, or None."""
    d = module.__dict__
    if "_raon_merged" not in d:
        mods = [getattr(module, n) for n in names]
        kinds = {_qparams(m) for m in mods}
        merged = None
        if len(kinds) == 1 and None not in kinds:
            kind, gs, bits, mode = kinds.pop()
            w = mx.concatenate([m["weight"] for m in mods], axis=0)
            sc = mx.concatenate([m["scales"] for m in mods], axis=0)
            b = (
                mx.concatenate([m["biases"] for m in mods], axis=0)
                if all(m.get("biases") is not None for m in mods)
                else None
            )
            col = None
            if kind == "gs":
                col = mx.concatenate(
                    [
                        mx.broadcast_to(
                            m["global_scale"].astype(mx.float32),
                            (m["weight"].shape[0],),
                        )
                        for m in mods
                    ]
                )
            sizes, off = [], 0
            for m in mods[:-1]:
                off += m["weight"].shape[0]
                sizes.append(off)
            mx.eval(w, sc, *[a for a in (b, col) if a is not None])
            merged = (w, sc, b, col, gs, bits, mode, sizes)
            _share_parts(mods, merged)
        object.__setattr__(module, "_raon_merged", merged)
    return d["_raon_merged"]


def _share_parts(mods, merged):
    """Decode fusion: point the part linears at row slices of the merged arrays, so the original copies are freed.
    The slices hold the same bytes with the same shapes, so the per-part (prefill) products do not change.
    """
    w, sc, b, _, _, _, _, sizes = merged
    bounds = [0] + list(sizes) + [w.shape[0]]
    parts = []
    for m, lo, hi in zip(mods, bounds[:-1], bounds[1:]):
        m.weight, m.scales = w[lo:hi], sc[lo:hi]
        parts += [m.weight, m.scales]
        if b is not None:
            m.biases = b[lo:hi]
            parts.append(m.biases)
    mx.eval(*parts)


def _project(module, names, x):
    """Outputs of the linears `names` on x; one merged quantized matmul for decode-sized inputs."""
    if not module.training and x.size // x.shape[-1] <= _DECODE_ROWS:
        merged = _merged_decode(module, names)
        if merged is not None:
            w, sc, b, col, gs, bits, mode, sizes = merged
            y = mx.quantized_matmul(
                x, w, sc, b, transpose=True, group_size=gs, bits=bits, mode=mode
            )
            if col is not None:
                y = _scale_cast(y, col)
            return mx.split(y, sizes, axis=-1)
    return [getattr(module, n)(x) for n in names]


@mx.compile
def _scale_cast(y, g):
    return (y.astype(mx.float32) * g).astype(y.dtype)


@partial(mx.compile, shapeless=False)
def _route_tail(
    z,
    alpha_pre,
    b_pre,
    alpha_post,
    b_post,
    alpha_z,
    b_z,
    eta,
    helmert,
    chart_c,
    eye,
    post_factor,
):
    n = eye.shape[0]
    f32 = mx.float32
    pre = mx.sigmoid(alpha_pre.astype(f32) * z[:, :n] + b_pre.astype(f32))
    post = post_factor * mx.sigmoid(
        alpha_post.astype(f32) * z[:, n : 2 * n] + b_post.astype(f32)
    )
    chart = alpha_z.astype(f32) * z[:, 2 * n :] + b_z.astype(f32)
    e = helmert @ chart.reshape(-1, n - 1, n - 1) @ helmert.T
    rho = mx.maximum((-e / chart_c).max(axis=(-2, -1)), 0)
    phi = chart_c + e / (1 + rho[:, None, None])
    gamma = mx.sigmoid(eta.astype(f32))
    return pre, post, (1 - gamma) * eye + gamma * phi


@mx.compile
def _route_head(flat, weights, eps):
    inv = mx.rsqrt(mx.square(flat.astype(mx.float32)).mean(-1, keepdims=True) + eps)
    return (flat @ weights.astype(flat.dtype).T).astype(mx.float32) * inv


@mx.compile
def _aggregate(stream, pre):
    return (stream.astype(mx.float32) * pre[..., None]).sum(-2).astype(stream.dtype)


@mx.compile
def _mix(stream, branch, post, residual):
    return (
        residual @ stream.astype(mx.float32)
        + post[..., None] * branch.astype(mx.float32)[:, None]
    ).astype(stream.dtype)


class StreamMix(nn.Module):
    def __init__(self, streams, hidden):
        super().__init__()
        size = (streams - 1) ** 2
        self.phi_z = mx.zeros((size, streams * hidden))
        self.alpha_z = mx.array(1.0)
        self.b_z = mx.zeros((size,))
        self.eta = mx.array(0.0)


class MHC(nn.Module):
    def __init__(self, args):
        super().__init__()
        n, hidden = args.raon_mhc_streams, args.hidden_size
        self.num_streams = n
        self.eps = args.raon_mhc_selector_eps
        self.post_factor = args.raon_mhc_post_factor
        self.phi_pre = mx.zeros((n, n * hidden))
        self.phi_post = mx.zeros((n, n * hidden))
        self.alpha_pre = mx.array(1.0)
        self.alpha_post = mx.array(1.0)
        self.b_pre = mx.zeros((n,))
        self.b_post = mx.zeros((n,))
        self.res = StreamMix(n, hidden)
        self._helmert = mx.array(
            [
                [
                    (1.0 if i < k else -float(k) if i == k else 0.0)
                    / math.sqrt(k * (k + 1))
                    for k in range(1, n)
                ]
                for i in range(n)
            ]
        )
        self._eye = mx.eye(n)
        self._chart = (self._eye + mx.ones((n, n)) / n) * 0.5

    def phi(self):
        return cached_array(
            self,
            "_phi_cat",
            lambda: mx.concatenate([self.phi_pre, self.phi_post, self.res.phi_z]),
        )

    def route(self, stream):
        n = self.num_streams
        flat = stream.reshape(-1, n * stream.shape[-1])
        z = _route_head(flat, self.phi(), mx.array(self.eps, mx.float32))
        return _route_tail(
            z,
            self.alpha_pre,
            self.b_pre,
            self.alpha_post,
            self.b_post,
            self.res.alpha_z,
            self.res.b_z,
            self.res.eta,
            self._helmert,
            self._chart,
            self._eye,
            mx.array(self.post_factor, mx.float32),
        )

    aggregate = staticmethod(_aggregate)
    mix = staticmethod(_mix)

    # fused mHC kernel inputs
    def fused_phi(self):
        return cached_array(self, "_fused_phi", lambda: self.phi().astype(mx.bfloat16))

    def fused_params(self):
        def make():
            f = lambda a: a.astype(mx.float32).reshape(-1)
            gamma = mx.sigmoid(self.res.eta.astype(mx.float32)).reshape(-1)
            out = mx.concatenate(
                [
                    mx.array([self.eps, self.post_factor], mx.float32),
                    f(self.alpha_pre),
                    f(self.b_pre),
                    f(self.alpha_post),
                    f(self.b_post),
                    f(self.res.alpha_z),
                    f(self.res.b_z),
                    gamma,
                    f(self._helmert),
                    f(self._chart),
                ]
            )
            if out.size != 51:
                raise ValueError("fused mHC kernel expects 4 streams")
            return out

        return cached_array(self, "_fused_prm", make)


_CACHE_EVAL_EVERY = 64


def _bound_cache_graph(cache, names):
    """Evaluate batched cache counters periodically to bound lazy graph growth."""
    d = cache.__dict__
    n = d.get("_raon_cf_n", 0) + 1
    d["_raon_cf_n"] = n
    if n % _CACHE_EVAL_EVERY == 0:
        arrs = [
            a
            for a in (getattr(cache, k, None) for k in names)
            if isinstance(a, mx.array)
        ]
        if arrs:
            mx.eval(*arrs)


class ShortConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.weight = mx.zeros((channels, kernel_size, 1))
        self.kernel_size = kernel_size

    def __call__(self, x, state, mask, lengths):
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)
        keep = self.kernel_size - 1
        if state is None:
            state = mx.zeros((x.shape[0], keep, x.shape[-1]), dtype=x.dtype)
        signal = mx.concatenate([state, x], axis=1)
        w32 = cached_array(self, "_w32", lambda: self.weight.astype(mx.float32))
        out = nn.silu(
            mx.conv1d(signal.astype(mx.float32), w32, groups=x.shape[-1])
        ).astype(x.dtype)
        if lengths is not None:
            ends = mx.clip(lengths, 0, x.shape[1])
            indices = (ends[:, None] + mx.arange(keep))[..., None]
            state = mx.take_along_axis(signal, indices, axis=1)
        else:
            state = mx.contiguous(signal[:, -keep:, :]) if keep else signal[:, :0, :]
        return out, state


@mx.compile
def _rda_qk(q, k, inv_sqrt_d):
    def normalize(value):
        v32 = value.astype(mx.float32)
        return (
            v32 / mx.maximum(mx.sqrt(mx.square(v32).sum(-1, keepdims=True)), 1e-6)
        ).astype(value.dtype)

    return normalize(q) * inv_sqrt_d.astype(q.dtype), normalize(k)


@mx.compile
def _rda_gates(beta_raw, delta, v, decay, dt32, a_exp, lower_bound):
    f32 = mx.float32
    beta_raw, delta = beta_raw.astype(f32), delta.astype(f32)
    ratio = mx.exp(-nn.softplus(-beta_raw - delta) + nn.softplus(-beta_raw))
    v = (v.astype(f32) * ratio[..., None]).astype(v.dtype)
    decay = (decay.astype(f32) + dt32) * a_exp[:, None]
    log_decay = (
        -nn.softplus(decay) if lower_bound is None else lower_bound * mx.sigmoid(decay)
    )
    g = mx.exp(log_decay)
    return v, g, mx.sigmoid(beta_raw)


@mx.compile
def _rda_out(out, norm_w, gate_raw, eps):
    normed = mx.fast.rms_norm(out.astype(mx.float32), norm_w.astype(mx.float32), eps)
    return normed * mx.sigmoid(gate_raw.astype(mx.float32))


_RDA_PARTS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "g_proj",
    "f_a_proj",
    "b_proj",
    "rda_w_proj",
)


class RDA(nn.Module):
    def __init__(self, args):
        super().__init__()
        cfg = args.rda_config
        self.num_heads = cfg.get("num_heads", args.num_attention_heads)
        self.head_dim = cfg.get("head_dim", args.head_dim)
        self.proj_size = self.num_heads * self.head_dim
        self.lower_bound = cfg.get("gate_lower_bound", -5.0)
        self.eps = args.rms_norm_eps
        hidden, size = args.hidden_size, self.proj_size
        for name in ["q_proj", "k_proj", "v_proj", "g_proj"]:
            setattr(self, name, nn.Linear(hidden, size, bias=False))
        self.qkv_conv1d = ShortConv(3 * size, cfg.get("conv_kernel_size", 4))
        self.f_a_proj = nn.Linear(hidden, args.rda_decay_rank, bias=False)
        self.f_b_proj = nn.Linear(args.rda_decay_rank, size, bias=False)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.rda_w_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.A_log = mx.zeros((self.num_heads,))
        self.dt_bias = mx.zeros((size,))
        self.o_norm = nn.RMSNorm(self.head_dim, eps=self.eps)
        self.o_proj = nn.Linear(size, hidden, bias=False)

    def _fused_decode(self, x, cache):
        return (
            x.shape[1] == 1
            and cache is not None
            and cache.lengths is None
            and x.dtype == mx.bfloat16
            and self.head_dim == 128
            and self.lower_bound is not None
            and not self.training
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        )

    def _fused_prep(self, qkv, state, braw, delta, decay, dt32, a_exp):
        batch = qkv.shape[0]
        nh, hd, conv = self.num_heads, self.head_dim, self.qkv_conv1d
        keep = conv.kernel_size - 1
        if state is None:
            state = mx.zeros((batch, keep, qkv.shape[-1]), dtype=qkv.dtype)
        w32 = cached_array(conv, "_w32", lambda: conv.weight.astype(mx.float32))
        fp = cached_array(
            self, "_prep_fp", lambda: mx.array([hd**-0.5, self.lower_bound], mx.float32)
        )
        q, k, v, g, beta, new_state = rda_prep_kernel()(
            inputs=[
                qkv.reshape(batch, -1),
                state,
                w32.reshape(-1),
                braw.reshape(batch, nh),
                delta.reshape(batch, nh),
                decay.reshape(batch, -1),
                dt32.reshape(-1),
                a_exp,
                fp,
            ],
            template=[("HEAD_DIM", hd), ("N_HEADS", nh), ("KSIZE", conv.kernel_size)],
            grid=(nh * hd, batch, 1),
            threadgroup=(hd, 1, 1),
            output_shapes=[(batch, nh, hd)] * 4 + [(batch, nh), state.shape],
            output_dtypes=[
                mx.bfloat16,
                mx.bfloat16,
                mx.bfloat16,
                mx.float32,
                mx.bfloat16,
                qkv.dtype,
            ],
        )
        shape = (batch, 1, nh, hd)
        return (
            q.reshape(shape),
            k.reshape(shape),
            v.reshape(shape),
            g.reshape(shape),
            beta.reshape(batch, 1, nh),
            new_state,
        )

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        dtype = x.dtype
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)
        states = [None, None] if cache is None else cache.cache
        lengths = None if cache is None else cache.lengths
        qp, kp, vp, gp, fa, braw, delta = _project(self, _RDA_PARTS, x)
        shape = (batch, length, self.num_heads, self.head_dim)
        a_exp = cached_array(
            self, "_a_exp", lambda: mx.exp(self.A_log.astype(mx.float32))
        )
        dt32 = cached_array(
            self,
            "_dt32",
            lambda: self.dt_bias.astype(mx.float32).reshape(
                self.num_heads, self.head_dim
            ),
        )
        if self._fused_decode(x, cache):
            q, k, v, g, beta, conv_state = self._fused_prep(
                mx.concatenate([qp, kp, vp], axis=-1),
                states[0],
                braw,
                delta,
                self.f_b_proj(fa),
                dt32,
                a_exp,
            )
        else:
            qkv, conv_state = self.qkv_conv1d(
                mx.concatenate([qp, kp, vp], axis=-1), states[0], mask, lengths
            )
            q, k, v = mx.split(qkv, 3, axis=-1)
            q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
            q, k = _rda_qk(
                q,
                k,
                cached_array(
                    self,
                    "_inv_sqrt_d",
                    lambda: mx.array(self.head_dim**-0.5, mx.float32),
                ),
            )
            # RDA may exceed float16 range before beta compensates it.
            update_dtype = mx.float32 if dtype == mx.float16 else dtype
            q, k, v = (
                q.astype(update_dtype),
                k.astype(update_dtype),
                v.astype(update_dtype),
            )
            decay = self.f_b_proj(fa).reshape(shape)
            v, g, beta = _rda_gates(
                braw,
                delta,
                v,
                decay,
                dt32,
                a_exp,
                (
                    None
                    if self.lower_bound is None
                    else cached_array(
                        self, "_lower32", lambda: mx.array(self.lower_bound, mx.float32)
                    )
                ),
            )
            beta = beta.astype(update_dtype)
        state = states[1]
        if state is None:
            state = mx.zeros(
                (batch, self.num_heads, self.head_dim, self.head_dim), dtype=mx.float32
            )
        if mask is None and self._fused_decode(x, cache):
            gated, state = self._fused_out(q, k, v, g, beta, state, gp.reshape(shape))
        else:
            use_kernel = (
                not self.training
                and mx.default_device() == mx.gpu
                and mx.metal.is_available()
                and self.head_dim >= 32
                and self.head_dim % 32 == 0
            )
            update = gated_delta_kernel if use_kernel else gated_delta_ops
            out, state = update(q, k, v, g, beta, state, mask)
            gated = _rda_out(
                out, self.o_norm.weight, gp.reshape(shape), self.eps
            ).astype(dtype)
        if cache is not None:
            cache.cache = [conv_state, state]
            cache.advance(length)
            _bound_cache_graph(cache, ("lengths", "left_padding"))
        return self.o_proj(gated.reshape(batch, length, self.proj_size))

    def _fused_out(self, q, k, v, g, beta, state, gate):
        """Decode fusion: recurrence, output norm and gate in one kernel; returns bf16 gated output and state."""
        batch, _, nh, hd = q.shape
        rows = 4
        norm_w = cached_array(
            self, "_norm_w32", lambda: self.o_norm.weight.astype(mx.float32)
        )
        eps = cached_array(self, "_eps32", lambda: mx.array([self.eps], mx.float32))
        return decode_kernel("rda_out")(
            inputs=[q, k, v, g, beta, state, norm_w, gate, eps],
            template=[
                ("InT", q.dtype),
                ("StT", state.dtype),
                ("Dk", hd),
                ("Dv", hd),
                ("Hk", nh),
                ("Hv", nh),
                ("ROWS", rows),
            ],
            grid=(32, hd // rows, batch * nh),
            threadgroup=(32, hd // rows, 1),
            output_shapes=[q.shape, state.shape],
            output_dtypes=[mx.bfloat16, state.dtype],
        )


class MLA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.nope = args.qk_nope_head_dim
        self.extra = args.qk_rope_head_dim
        self.value_dim = args.v_head_dim
        self.latent_dim = args.kv_lora_rank
        self.scale = (self.nope + self.extra) ** -0.5
        self.use_gate = args.mla_use_output_gate
        hidden = args.hidden_size
        self.q_a_proj = nn.Linear(hidden, args.q_lora_rank, bias=False)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.mla_latent_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, self.num_heads * (self.nope + self.extra), bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden, self.latent_dim + self.extra, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.latent_dim, eps=args.mla_latent_norm_eps)
        self.embed_q = MultiLinear(self.nope, self.latent_dim, self.num_heads)
        self.unembed_out = MultiLinear(self.latent_dim, self.value_dim, self.num_heads)
        self.o_proj = nn.Linear(self.num_heads * self.value_dim, hidden, bias=False)
        if self.use_gate:
            self.g_proj = nn.Linear(hidden, self.num_heads * self.value_dim, bias=False)

    def __call__(self, x, mask=None, cache=None):
        batch, length, _ = x.shape
        if self.use_gate:
            qa, kva, gate = _project(
                self, ("q_a_proj", "kv_a_proj_with_mqa", "g_proj"), x
            )
        else:
            qa, kva = _project(self, ("q_a_proj", "kv_a_proj_with_mqa"), x)
        q = self.q_b_proj(self.q_a_layernorm(qa))
        q = q.reshape(batch, length, self.num_heads, self.nope + self.extra).transpose(
            0, 2, 1, 3
        )
        q_nope, q_extra = mx.split(q, [self.nope], axis=-1)
        latent, extra = mx.split(kva, [self.latent_dim], axis=-1)
        latent = self.kv_a_layernorm(latent)[:, None]
        extra = extra[:, None]
        if cache is not None:
            latent, extra = cache.update_and_fetch(latent, extra)
            _bound_cache_graph(cache, ("offset", "left_padding"))
        quantized = hasattr(cache, "bits")
        if quantized:
            pe_scores = mx.quantized_matmul(
                q_extra * self.scale,
                *extra,
                transpose=True,
                group_size=cache.group_size,
                bits=cache.bits,
            )
        else:
            pe_scores = (q_extra * self.scale) @ extra.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(mask, pe_scores, mx.finfo(pe_scores.dtype).min)
        attn_cache = cache
        if length == 1:
            q_nope = self.embed_q(q_nope)
            key = value = latent
        else:
            if quantized:
                latent = mx.dequantize(
                    *latent, group_size=cache.group_size, bits=cache.bits
                )
                attn_cache = None
            key = self.embed_q(latent, transpose=False)
            value = self.unembed_out(latent)
        if quantized and length == 1:
            scores = mx.quantized_matmul(
                q_nope * self.scale,
                *key,
                transpose=True,
                group_size=cache.group_size,
                bits=cache.bits,
            )
            scores = mx.softmax(scores + pe_scores, axis=-1, precise=True)
            out = mx.quantized_matmul(
                scores,
                *value,
                transpose=False,
                group_size=cache.group_size,
                bits=cache.bits,
            )
        else:
            out = scaled_dot_product_attention(
                q_nope, key, value, cache=attn_cache, scale=self.scale, mask=pe_scores
            )
        if length == 1:
            out = self.unembed_out(out)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        if self.use_gate:
            out = out * mx.sigmoid(gate)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, args, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)
        self.activation = SiTU(
            args.activation_situ_beta, args.activation_situ_linear_beta
        )

    def __call__(self, x):
        gate, up = _project(self, ("gate_proj", "up_proj"), x)
        return self.down_proj(self.activation(up, gate))


@mx.compile
def _router_scores(x, weight, bias):
    scores = mx.sigmoid(x.astype(mx.float32) @ weight.T)
    return scores, scores + bias.astype(mx.float32)


@mx.compile
def _router_weights(scores, indices, scale):
    weights = mx.take_along_axis(scores, indices, axis=-1)
    return weights / (weights.sum(-1, keepdims=True) + 1e-20) * scale


class Router(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros((args.num_experts,))
        self.top_k = args.num_experts_per_token
        self.renormalize = args.moe_renormalize
        self.scale = args.routed_scaling_factor

    def __call__(self, x):
        w32 = cached_array(self, "_w32", lambda: self.weight.astype(mx.float32))
        scores, ranked = _router_scores(x, w32, self.e_score_correction_bias)
        indices = mx.stop_gradient(mx.argsort(-ranked, axis=-1)[..., : self.top_k])
        if self.renormalize:
            return indices, _router_weights(
                scores,
                indices,
                cached_array(
                    self, "_scale32", lambda: mx.array(self.scale, mx.float32)
                ),
            )
        return indices, mx.take_along_axis(scores, indices, axis=-1) * self.scale

    def route_f32(self, x32):
        """Decode fusion: renormalized routing from the f32 input in one tail kernel (same result as __call__)."""
        w32 = cached_array(self, "_w32", lambda: self.weight.astype(mx.float32))
        bias = cached_array(
            self, "_bias32", lambda: self.e_score_correction_bias.astype(mx.float32)
        )
        scale = cached_array(
            self, "_scale1", lambda: mx.array([self.scale], mx.float32)
        )
        logits = x32 @ w32.T
        e = logits.shape[-1]
        rows = logits.size // e
        indices, weights = decode_kernel("router")(
            inputs=[logits.reshape(rows, e), bias, scale],
            template=[("N_EXPERTS", e), ("TOP_K", self.top_k)],
            grid=(rows * e, 1, 1),
            threadgroup=(e, 1, 1),
            output_shapes=[(rows, self.top_k)] * 2,
            output_dtypes=[mx.uint32, mx.float32],
        )
        shape = (*x32.shape[:-1], self.top_k)
        return indices.reshape(shape), weights.reshape(shape)


@mx.compile
def _moe_combine(out, weights, shared):
    return (out.astype(mx.float32) * weights[..., None]).sum(-2).astype(
        shared.dtype
    ) + shared


@mx.compile
def _moe_sum(out, weights, like):
    return (out.astype(mx.float32) * weights[..., None]).sum(-2).astype(like.dtype)


class MergedSwitchGLU(nn.Module):
    """SwitchGLU with up|gate in one switch linear (`gate_up`, up rows first)."""

    def __init__(self, input_dims, hidden_dims, num_experts, activation):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.gate_up = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=False
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=False)
        self.activation = activation

    def _half(self, x, idx, sort, lo, hi):
        gu = self.gate_up
        if isinstance(gu, QuantizedSwitchLinear):
            return mx.gather_qmm(
                x,
                gu["weight"][:, lo:hi],
                gu["scales"][:, lo:hi],
                None if gu.get("biases") is None else gu["biases"][:, lo:hi],
                rhs_indices=idx,
                transpose=True,
                group_size=gu.group_size,
                bits=gu.bits,
                mode=gu.mode,
                global_scale=gu.global_scale,
                sorted_indices=sort,
            )
        return mx.gather_mm(
            x,
            gu["weight"][:, lo:hi].swapaxes(-1, -2),
            rhs_indices=idx,
            sorted_indices=sort,
        )

    def __call__(self, x, indices):
        decode = x.size // x.shape[-1] <= _DECODE_ROWS
        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        sort = indices.size >= 64
        idx, inv_order = indices, None
        if sort:
            x, idx, inv_order = _gather_sort(x, indices)
        h = self.hidden_dims
        if decode:
            x_up, x_gate = mx.split(
                self.gate_up(x, idx, sorted_indices=sort), 2, axis=-1
            )
        else:
            # the matrix kernels tile by output width: keep the shapes of the separate products
            x_up, x_gate = self._half(x, idx, sort, 0, h), self._half(
                x, idx, sort, h, 2 * h
            )
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=sort)
        if sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class MoE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gate = Router(args)
        glu = MergedSwitchGLU if args.raon_merged_gate_up else SwitchGLU
        self.switch_mlp = glu(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            activation=SiTU(
                args.activation_situ_beta, args.activation_situ_linear_beta
            ),
        )
        self.shared_experts = (
            MLP(args, args.moe_intermediate_size * args.num_shared_experts)
            if args.num_shared_experts
            else None
        )

    def __call__(self, x):
        indices, weights = self.gate(x)
        out = self.switch_mlp(x, indices)
        if self.shared_experts is not None:
            return _moe_combine(out, weights, self.shared_experts(x))
        return _moe_sum(out, weights, x)

    def fusable(self):
        return (
            self.shared_experts is not None
            and self.gate.renormalize
            and self.gate.weight.shape[0] % 32 == 0
            and self.gate.weight.shape[0] <= 1024
        )

    def parts(self, x, x32):
        """Decode fusion: (routed out, weights, shared out) before the combine, which the next kernel does."""
        indices, weights = self.gate.route_f32(x32)
        return self.switch_mlp(x, indices), weights, self.shared_experts(x)


class GlobalScaledLinear(nn.Module):
    """NVFP4 weight-only linear from the converted checkpoint: quantized_matmul(x, codes, E4M3 block scales) * FP32 global.
    quantized_matmul has no global_scale argument, so the per-tensor scale is applied to the output (exact).
    """

    def __init__(self, input_dims, output_dims):
        super().__init__()
        self.weight = mx.zeros((output_dims, input_dims // 8), mx.uint32)
        self.scales = mx.zeros((output_dims, input_dims // 16), mx.uint8)
        self.global_scale = mx.array(1.0, mx.float32)
        self.freeze()

    def __call__(self, x):
        y = mx.quantized_matmul(
            x,
            self["weight"],
            self["scales"],
            None,
            transpose=True,
            group_size=16,
            bits=4,
            mode="nvfp4",
        )
        return _scale_cast(y, self["global_scale"])


class DecoderLayer(nn.Module):
    def __init__(self, args, index):
        super().__init__()
        self.is_linear = index + 1 in args.rda_config["rda_layers"]
        self.is_moe = (
            bool(args.num_experts)
            and index >= args.first_k_dense_replace
            and index % args.moe_layer_freq == 0
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.self_attn = RDA(args) if self.is_linear else MLA(args)
        self.self_attn_raon_mhc = MHC(args)
        self.mlp_raon_mhc = MHC(args)
        if self.is_moe:
            self.moe = MoE(args)
        else:
            self.mlp = MLP(args, args.intermediate_size)

    def __call__(self, stream, mask=None, cache=None):
        batch, length, streams, hidden = stream.shape
        rows = stream.reshape(-1, streams, hidden)
        pre, post, residual = self.self_attn_raon_mhc.route(stream)
        x = self.input_layernorm(MHC.aggregate(rows, pre)).reshape(
            batch, length, hidden
        )
        branch = self.self_attn(x, mask, cache).reshape(-1, hidden)
        rows = MHC.mix(rows, branch, post, residual)
        pre, post, residual = self.mlp_raon_mhc.route(rows.reshape(stream.shape))
        x = self.post_attention_layernorm(MHC.aggregate(rows, pre)).reshape(
            batch, length, hidden
        )
        branch = (self.moe(x) if self.is_moe else self.mlp(x)).reshape(
            -1, hidden
        )
        return MHC.mix(rows, branch, post, residual).reshape(stream.shape)


class RaonModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.streams = args.raon_mhc_streams
        self._linear_index = next(
            (i for i, layer in enumerate(self.layers) if layer.is_linear), None
        )
        self._full_index = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_linear), None
        )

    def _masks(self, h, cache):
        linear_mask = (
            create_ssm_mask(h, cache[self._linear_index])
            if self._linear_index is not None
            else None
        )
        full_mask = (
            create_attention_mask(h, cache[self._full_index], return_array=True)
            if self._full_index is not None
            else None
        )
        return linear_mask, full_mask

    def __call__(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        cache = [None] * len(self.layers) if cache is None else cache
        if len(cache) != len(self.layers):
            raise ValueError("Cache must have one entry per layer")
        linear_mask, full_mask = self._masks(h, cache)
        # the fused kernels take bf16 rows/branch: require bf16 activations end to end (norm weights bf16)
        fused = (
            not self.training
            and self.args.raon_fused_mhc_decode
            and inputs.shape[1] == 1
            and h.dtype == mx.bfloat16
            and self.layers[0].input_layernorm.weight.dtype == mx.bfloat16
            and self.streams == 4
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        )
        if fused and self.norm.weight.dtype == mx.bfloat16:
            return self._decode_fusion(h, cache, linear_mask, full_mask)
        if fused:
            return self._decode_fused(h, cache, linear_mask, full_mask)
        stream = mx.repeat(h[:, :, None], self.streams, axis=2)
        for layer, layer_cache in zip(self.layers, cache):
            stream = layer(
                stream, linear_mask if layer.is_linear else full_mask, layer_cache
            )
        return self.norm(stream.astype(mx.float32).mean(axis=2).astype(h.dtype))

    def _decode_fused(self, h, cache, linear_mask, full_mask):
        batch, length, hidden = h.shape
        rows = mx.repeat(h[:, :, None], self.streams, axis=2).reshape(
            -1, self.streams, hidden
        )
        branch = post = res = None
        for layer, lc in zip(self.layers, cache):
            rows, agg, post, res = fused_mhc(
                rows, branch, post, res, layer.self_attn_raon_mhc, branch is not None
            )
            x = layer.input_layernorm(agg).reshape(batch, length, hidden)
            branch = layer.self_attn(
                x, linear_mask if layer.is_linear else full_mask, lc
            ).reshape(-1, hidden)
            rows, agg, post, res = fused_mhc(
                rows, branch, post, res, layer.mlp_raon_mhc, True
            )
            x = layer.post_attention_layernorm(agg).reshape(batch, length, hidden)
            branch = (
                layer.moe(x) if layer.is_moe else layer.mlp(x)
            ).reshape(-1, hidden)
        rows = MHC.mix(rows, branch, post, res)
        stream = rows.reshape(batch, length, self.streams, hidden)
        return self.norm(stream.astype(mx.float32).mean(axis=2).astype(h.dtype))

    def _decode_fusion(self, h, cache, linear_mask, full_mask):
        """Fuse norms and MoE combination into the mHC decode kernels."""
        batch, length, hidden = h.shape
        rows = mx.repeat(h[:, :, None], self.streams, axis=2).reshape(
            -1, self.streams, hidden
        )
        branch = post = res = None
        for layer, lc in zip(self.layers, cache):
            rows, x, post, res = fused_mhc_norm(
                rows,
                branch,
                post,
                res,
                layer.self_attn_raon_mhc,
                branch is not None,
                layer.input_layernorm,
            )
            mask = linear_mask if layer.is_linear else full_mask
            branch = layer.self_attn(
                x.reshape(batch, length, hidden), mask, lc
            ).reshape(-1, hidden)
            if layer.is_moe and layer.moe.fusable():
                rows, x, post, res, x32 = fused_mhc_norm(
                    rows,
                    branch,
                    post,
                    res,
                    layer.mlp_raon_mhc,
                    True,
                    layer.post_attention_layernorm,
                    with_f32=True,
                )
                shape = (batch, length, hidden)
                branch = layer.moe.parts(
                    x.reshape(shape), x32.reshape(shape)
                )
            else:
                rows, x, post, res = fused_mhc_norm(
                    rows,
                    branch,
                    post,
                    res,
                    layer.mlp_raon_mhc,
                    True,
                    layer.post_attention_layernorm,
                )
                x = x.reshape(batch, length, hidden)
                branch = (
                    layer.moe(x) if layer.is_moe else layer.mlp(x)
                ).reshape(-1, hidden)
        if isinstance(branch, tuple):
            return fused_final(rows, branch, post, res, self.norm).reshape(
                batch, length, hidden
            )
        rows = MHC.mix(rows, branch, post, res)
        stream = rows.reshape(batch, length, self.streams, hidden)
        return self.norm(stream.astype(mx.float32).mean(axis=2).astype(h.dtype))


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = RaonModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        for path in args.raon_global_scaled_linears or []:
            self._install_global_scaled(path)

    def _install_global_scaled(self, path):
        parent = self
        parts = path.split(".")
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        lin = getattr(parent, parts[-1])
        if not isinstance(lin, nn.Linear):
            raise ValueError(f"{path} is not a Linear")
        out_dims, in_dims = lin.weight.shape
        setattr(parent, parts[-1], GlobalScaledLinear(in_dims, out_dims))

    def __call__(self, inputs, cache=None):
        out = self.model(inputs, cache)
        return (
            self.model.embed_tokens.as_linear(out)
            if self.args.tie_word_embeddings
            else self.lm_head(out)
        )

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            ArraysCache(2) if layer.is_linear else KVCache() for layer in self.layers
        ]

    def sanitize(self, weights):
        weights = dict(weights)
        args = self.args
        for k, v in weights.items():
            if k.endswith("self_attn.f_a_proj.weight") and v.shape[0] != args.rda_decay_rank:
                raise ValueError(f"{k} has {v.shape[0]} rows but rda_decay_rank is {args.rda_decay_rank}")
        for i in range(
            args.num_hidden_layers,
            args.num_hidden_layers + args.num_nextn_predict_layers,
        ):
            prefix = f"model.layers.{i}."
            weights = {k: v for k, v in weights.items() if not k.startswith(prefix)}
        for i, layer in enumerate(self.layers):
            prefix = f"model.layers.{i}."
            if layer.is_linear:
                convs = [
                    prefix + f"self_attn.{n}_conv1d.weight" for n in ("q", "k", "v")
                ]
                if all(c in weights for c in convs):
                    ws = [weights.pop(c) for c in convs]
                    ws = [w[..., None] if w.ndim == 2 else w for w in ws]
                    weights[prefix + "self_attn.qkv_conv1d.weight"] = mx.concatenate(
                        ws, axis=0
                    )
                key = prefix + "self_attn.qkv_conv1d.weight"
                if key in weights and weights[key].ndim == 2:
                    weights[key] = weights[key][..., None]
            else:
                key = prefix + "self_attn.kv_b_proj.weight"
                if key in weights:
                    w = weights.pop(key).reshape(
                        args.num_attention_heads,
                        args.qk_nope_head_dim + args.v_head_dim,
                        args.kv_lora_rank,
                    )
                    weights[prefix + "self_attn.embed_q.weight"] = w[
                        :, : args.qk_nope_head_dim
                    ].swapaxes(-1, -2)
                    weights[prefix + "self_attn.unembed_out.weight"] = w[
                        :, args.qk_nope_head_dim :
                    ]
            if layer.is_moe:
                moe = prefix + "moe."
                if moe + "experts.0.gate_proj.weight" in weights:
                    for old, new in [
                        ("gate_proj", "gate_proj"),
                        ("up_proj", "up_proj"),
                        ("down_proj", "down_proj"),
                    ]:
                        weights[moe + f"switch_mlp.{new}.weight"] = mx.stack(
                            [
                                weights.pop(moe + f"experts.{e}.{old}.weight")
                                for e in range(args.num_experts)
                            ]
                        )
            if layer.is_moe and args.raon_merged_gate_up:
                sw = prefix + "moe.switch_mlp."
                if sw + "up_proj.weight" in weights:
                    gs_up = weights.pop(sw + "up_proj.global_scale", None)
                    gs_gate = weights.pop(sw + "gate_proj.global_scale", None)
                    if (gs_up is None) != (gs_gate is None) or (
                        gs_up is not None and not mx.array_equal(gs_up, gs_gate)
                    ):
                        raise ValueError(
                            f"{sw}: up and gate need the same global scale to merge"
                        )
                    for name in ("weight", "scales", "biases"):
                        up, gate = sw + f"up_proj.{name}", sw + f"gate_proj.{name}"
                        if up in weights:
                            weights[sw + f"gate_up.{name}"] = mx.concatenate(
                                [weights.pop(up), weights.pop(gate)], axis=1
                            )
                    if gs_up is not None:
                        weights[sw + "gate_up.global_scale"] = gs_up
        if args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def cast_predicate(self):
        return lambda path: not (
            path.endswith(
                (
                    "A_log",
                    "dt_bias",
                    "e_score_correction_bias",
                    "res.eta",
                    "global_scale",
                )
            )
            or (self.args.raon_fp32_export_head and path == "lm_head.weight")
        )

    @property
    def quant_predicate(self):
        return lambda path, module: not (
            path.endswith("moe.gate")
            or isinstance(module, GlobalScaledLinear)
            or (self.args.raon_fp32_export_head and path == "lm_head")
        )
