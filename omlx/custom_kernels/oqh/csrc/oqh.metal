// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// oQh kernels.
//
// A weight row holds uniform codes q in [0, 2^bits) in MLX affine packing and one
// scale: w = scale * (q - (2^bits - 1) / 2). Weights are stored in a basis rotated
// by a signed block Walsh-Hadamard on the input axis, so their inputs must be
// rotated the same way. The producer kernels (norm, activation, gated norm) apply
// that rotation while they write their output.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

// ---------------------------------------------------------------------------
// Matrix-vector product (decode) and its routed-expert form.
// Same tiling as MLX qmv_fast: 2 simdgroups x 4 output rows per threadgroup.
// The per-group bias term of affine qmv becomes one -center * sum(x) per row.
// PAIR reads two matrices of N rows (gate and up) into one [.., 2N] output; a
// simdgroup's 4 rows never straddle them because N % 8 == 0.
// R x rows share each threadgroup, so a weight block comes from DRAM once for
// all of them (batched decode, MTP verify). Every x row keeps the one-row
// arithmetic and order, so its result does not depend on R.
// ---------------------------------------------------------------------------

template <typename T, int bits, bool gather, bool shared_x, bool pair, int R>
METAL_FUNC void oqh_qmv_impl(
    const device uint32_t* w,
    const device T* rscale,
    const device uint32_t* w_b,
    const device T* rscale_b,
    const device T* x,
    device T* y,
    const device uint32_t* indices,
    const int in_vec_size,
    const int out_vec_size,
    const int n_rows,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
  constexpr int packs_per_thread = bits <= 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * SIMD_SIZE;
  constexpr float center = 0.5f * float((1 << bits) - 1);
  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[R][results_per_simdgroup];
  thread U xsum[R];
  for (int r = 0; r < R; r++) {
    xsum[r] = 0;
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[r][row] = 0;
    }
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int y_cols = pair ? 2 * out_vec_size : out_vec_size;
  int w_row = out_row;
  if (pair && out_row >= out_vec_size) {
    w = w_b;
    rscale = rscale_b;
    w_row -= out_vec_size;
  }
  const device uint8_t* ws = (const device uint8_t*)w;
  int xrow = tid.x * R;
  int yrow = tid.x * R;
  if (gather) {
    const size_t e = indices[tid.z];
    ws += e * size_t(out_vec_size) * size_t(in_vec_size_w);
    rscale += e * size_t(out_vec_size);
    xrow = shared_x ? 0 : int(tid.z);
    yrow = tid.z;
  }
  // The same for the whole threadgroup, so the early exits keep simd_sum uniform.
  const int nr = gather ? 1 : min(R, n_rows - xrow);
  ws += w_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  x += size_t(xrow) * in_vec_size + simd_lid * values_per_thread;
  y += size_t(yrow) * y_cols + out_row;

  const int aligned_end = (in_vec_size / block_size) * block_size;
  for (int k = 0; k < aligned_end; k += block_size) {
    for (int r = 0; r < R && r < nr; r++) {
      U sum = load_vector<T, U, values_per_thread, bits>(x + r * in_vec_size, x_thread);
      xsum[r] += sum;
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        result[r][row] += qdot<U, values_per_thread, bits>(wl, x_thread, U(1), U(0), sum);
      }
    }
    ws += block_size * bytes_per_pack / pack_factor;
    x += block_size;
  }
  // Tail for widths that are a multiple of values_per_thread but not of block_size.
  if (aligned_end < in_vec_size &&
      aligned_end + int(simd_lid) * values_per_thread < in_vec_size) {
    for (int r = 0; r < R && r < nr; r++) {
      U sum = load_vector<T, U, values_per_thread, bits>(x + r * in_vec_size, x_thread);
      xsum[r] += sum;
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        result[r][row] += qdot<U, values_per_thread, bits>(wl, x_thread, U(1), U(0), sum);
      }
    }
  }

  for (int r = 0; r < R && r < nr; r++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      U v = simd_sum(result[r][row] - center * xsum[r]);
      if (simd_lid == 0) {
        y[r * y_cols + row] = static_cast<T>(v * U(rscale[w_row + row]));
      }
    }
  }
}

// w_b / rscale_b are bound only for pair kernels. n_rows: rows of x.
template <typename T, int bits, bool pair, int R>
[[kernel]] void oqh_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* rscale [[buffer(1)]],
    const device T* x [[buffer(2)]],
    device T* y [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const device uint32_t* w_b [[buffer(6)]],
    const device T* rscale_b [[buffer(7)]],
    const constant int& n_rows [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  oqh_qmv_impl<T, bits, false, false, pair, R>(
      w, rscale, w_b, rscale_b, x, y, nullptr, in_vec_size, out_vec_size, n_rows, tid,
      simd_gid, simd_lid);
}

template <typename T, int bits, bool shared_x, bool pair>
[[kernel]] void oqh_gather_qmv(
    const device uint32_t* w [[buffer(0)]],
    const device T* rscale [[buffer(1)]],
    const device T* x [[buffer(2)]],
    device T* y [[buffer(3)]],
    const device uint32_t* indices [[buffer(4)]],
    const constant int& in_vec_size [[buffer(5)]],
    const constant int& out_vec_size [[buffer(6)]],
    const device uint32_t* w_b [[buffer(7)]],
    const device T* rscale_b [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  oqh_qmv_impl<T, bits, true, shared_x, pair, 1>(
      w, rscale, w_b, rscale_b, x, y, indices, in_vec_size, out_vec_size, 1, tid, simd_gid,
      simd_lid);
}

#define oqh_instantiate_qmv_rows(type, bits, pair, sfx)                                            \
  instantiate_kernel("oqh_qmv_" #type "_b" #bits sfx "_r1", oqh_qmv, type, bits, pair, 1)          \
  instantiate_kernel("oqh_qmv_" #type "_b" #bits sfx "_r4", oqh_qmv, type, bits, pair, 4)          \
  instantiate_kernel("oqh_qmv_" #type "_b" #bits sfx "_r8", oqh_qmv, type, bits, pair, 8)

#define oqh_instantiate_qmv_pair(type, bits, pair, sfx)                                          \
  oqh_instantiate_qmv_rows(type, bits, pair, sfx)                                                \
  instantiate_kernel("oqh_gather_qmv_" #type "_b" #bits "_s1" sfx, oqh_gather_qmv, type, bits, true, pair) \
  instantiate_kernel("oqh_gather_qmv_" #type "_b" #bits "_s0" sfx, oqh_gather_qmv, type, bits, false, pair)

#define oqh_instantiate_qmv(type, bits)                \
  oqh_instantiate_qmv_pair(type, bits, false, "")      \
  oqh_instantiate_qmv_pair(type, bits, true, "_pair")

#define oqh_instantiate_qmv_bits(type) \
  oqh_instantiate_qmv(type, 2)         \
  oqh_instantiate_qmv(type, 3)         \
  oqh_instantiate_qmv(type, 4)         \
  oqh_instantiate_qmv(type, 5)         \
  oqh_instantiate_qmv(type, 6)         \
  oqh_instantiate_qmv(type, 8)

oqh_instantiate_qmv_bits(float16_t)
oqh_instantiate_qmv_bits(bfloat16_t)

// ---------------------------------------------------------------------------
// Block producers: one threadgroup rotates one BLK-wide block (128 or 256) with
// BLK / 4 threads, 4 values per thread at e = tid * 4 + j. The output is
// normalized by 1 / sqrt(BLK).
// ---------------------------------------------------------------------------

METAL_FUNC void oqh_bfly4(thread float* v) {
  float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
  v[0] = a0 + a2;
  v[1] = a1 + a3;
  v[2] = a0 - a2;
  v[3] = a1 - a3;
}

template <int BLK>
METAL_FUNC void oqh_fwht4(thread float* v, threadgroup float* sh, uint tid, uint lane) {
  oqh_bfly4(v);
  for (ushort s = 1; s < 32 && s * 4 < BLK; s <<= 1) {
    bool up = (lane & s) != 0;
    for (int j = 0; j < 4; ++j) {
      float p = simd_shuffle_xor(v[j], s);
      v[j] = up ? (p - v[j]) : (v[j] + p);
    }
  }
  if (BLK == 256) {
    for (int j = 0; j < 4; ++j) {
      sh[tid * 4 + j] = v[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = 0; j < 4; ++j) {
      uint e = tid * 4 + j;
      float p = sh[e ^ 128];
      v[j] = (e & 128) ? (p - v[j]) : (v[j] + p);
    }
  }
}

// MODE 0: silu(a) * b, 2: silu(min(a, lim)) * clamp(b, -lim, lim) (clamped
// SwiGLU), 3: a (rotation only).
// SPLIT: a and b are the two halves of one [rows, 2L] input (paired gate | up
// GEMV); otherwise two [rows, L] inputs (gate and up matmuls).
template <typename T, int BLK, int MODE, bool SPLIT>
[[kernel]] void oqh_act(
    const device T* a [[buffer(0)]],
    const device T* b [[buffer(1)]],
    const device T* sg [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int& L [[buffer(4)]],
    const constant float& lim [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh[BLK];
  const size_t base = size_t(tg) * BLK + tid * 4;
  const int col = (int(tg) % (L / BLK)) * BLK + tid * 4;
  const size_t abase = SPLIT ? size_t(tg / (L / BLK)) * 2 * L + col : base;
  const size_t bbase = SPLIT ? abase + L : base;
  float v[4];
  for (int j = 0; j < 4; ++j) {
    const float av = float(a[abase + j]);
    float f;
    if (MODE == 0) {
      f = (av / (1.0f + metal::exp(-av))) * float(b[bbase + j]);
    } else if (MODE == 2) {
      const float g = metal::min(av, lim);
      f = (g / (1.0f + metal::exp(-g))) * metal::clamp(float(b[bbase + j]), -lim, lim);
    } else {
      f = av;
    }
    v[j] = f * float(sg[col + j]);
  }
  oqh_fwht4<BLK>(v, sh, tid, lane);
  const float scale = metal::precise::rsqrt(float(BLK));
  for (int j = 0; j < 4; ++j) {
    out[base + j] = static_cast<T>(v[j] * scale);
  }
}

// Per-head (128) RMSNorm with weight, times silu(z), then the block rotation.
// Matches the Qwen3-Next gated norm: the normed value is rounded to Tin first.
// plain gets the gated output before the rotation.
template <typename Tin, typename T, int BLK>
[[kernel]] void oqh_gnorm(
    const device Tin* x [[buffer(0)]],
    const device T* z [[buffer(1)]],
    const device T* w [[buffer(2)]],
    const device T* sg [[buffer(3)]],
    device T* out [[buffer(4)]],
    const constant int& L [[buffer(5)]],
    const constant float& eps [[buffer(6)]],
    device T* plain [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sh[BLK];
  const size_t base = size_t(tg) * BLK + tid * 4;
  const int col = (int(tg) % (L / BLK)) * BLK + tid * 4;
  float v[4];
  float ss = 0.0f;
  for (int j = 0; j < 4; ++j) {
    v[j] = float(x[base + j]);
    ss += v[j] * v[j];
  }
  ss = simd_sum(ss);
  const float inv = metal::precise::rsqrt(ss / 128.0f + eps);
  for (int j = 0; j < 4; ++j) {
    const float zv = float(z[base + j]);
    const float n = static_cast<float>(static_cast<Tin>(v[j] * inv * float(w[(col + j) % 128])));
    const float g = n * (zv / (1.0f + metal::exp(-zv)));
    plain[base + j] = static_cast<T>(g);
    v[j] = g * float(sg[col + j]);
  }
  oqh_fwht4<BLK>(v, sh, tid, lane);
  const float scale = metal::precise::rsqrt(float(BLK));
  for (int j = 0; j < 4; ++j) {
    out[base + j] = static_cast<T>(v[j] * scale);
  }
}

#define oqh_instantiate_block(type, blk)                                                  \
  instantiate_kernel("oqh_act_" #type "_k" #blk "_m3", oqh_act, type, blk, 3, false)      \
  instantiate_kernel("oqh_act_" #type "_k" #blk "_m0", oqh_act, type, blk, 0, false)      \
  instantiate_kernel("oqh_act_" #type "_k" #blk "_m2", oqh_act, type, blk, 2, false)      \
  instantiate_kernel("oqh_act_" #type "_k" #blk "_m0_s", oqh_act, type, blk, 0, true)     \
  instantiate_kernel("oqh_act_" #type "_k" #blk "_m2_s", oqh_act, type, blk, 2, true)     \
  instantiate_kernel("oqh_gnorm_" #type "_" #type "_k" #blk, oqh_gnorm, type, type, blk) \
  instantiate_kernel("oqh_gnorm_float_" #type "_k" #blk, oqh_gnorm, float, type, blk)

#define oqh_instantiate_blocks(type) \
  oqh_instantiate_block(type, 128)   \
  oqh_instantiate_block(type, 256)

oqh_instantiate_blocks(float16_t)
oqh_instantiate_blocks(bfloat16_t)

// ---------------------------------------------------------------------------
// Row norm with one threadgroup per row: D / 8 threads, 8 values per thread at
// e = tid * 8 + i, so a 256-wide rotation block is one simdgroup (3 register
// stages and 5 lane shuffles).
// ---------------------------------------------------------------------------

METAL_FUNC void oqh_fwht256_simd(thread float* v, uint lane) {
  for (int h = 1; h < 8; h <<= 1) {
    for (int i = 0; i < 8; ++i) {
      if ((i & h) == 0) {
        float a = v[i];
        float b = v[i + h];
        v[i] = a + b;
        v[i + h] = a - b;
      }
    }
  }
  for (ushort st = 1; st < 32; st <<= 1) {
    bool up = (lane & st) != 0;
    for (int i = 0; i < 8; ++i) {
      float p = simd_shuffle_xor(v[i], st);
      v[i] = up ? (p - v[i]) : (v[i] + p);
    }
  }
}

METAL_FUNC float oqh_row_sum(float v, threadgroup float* red, uint lane, uint sgid, uint nsg) {
  v = simd_sum(v);
  if (lane == 0) {
    red[sgid] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = lane < nsg ? red[lane] : 0.0f;
  t = simd_sum(t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

// plain is MLX rms_norm's output, w * T(x * inv) in T; the rotation reads it, so
// the rotated output is exactly the rotated plain value.
template <typename T>
[[kernel]] void oqh_norm(
    const device T* x [[buffer(0)]],
    const device T* w [[buffer(1)]],
    const device T* sg [[buffer(2)]],
    device T* out [[buffer(3)]],
    device T* plain [[buffer(4)]],
    const constant int& D [[buffer(5)]],
    const constant float& eps [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]) {
  threadgroup float red[32];
  const size_t base = size_t(row) * D + tid * 8;
  float v[8];
  float ss = 0.0f;
  for (int i = 0; i < 8; ++i) {
    v[i] = float(x[base + i]);
    ss += v[i] * v[i];
  }
  const float inv = metal::precise::rsqrt(oqh_row_sum(ss, red, lane, sgid, nsg) / float(D) + eps);
  for (int i = 0; i < 8; ++i) {
    const int e = tid * 8 + i;
    const T pv = w[e] * static_cast<T>(v[i] * inv);
    plain[base + i] = pv;
    v[i] = float(pv) * float(sg[e]);
  }
  oqh_fwht256_simd(v, lane);
  const float scale = metal::precise::rsqrt(256.0f);
  for (int i = 0; i < 8; ++i) {
    out[base + i] = static_cast<T>(v[i] * scale);
  }
}

instantiate_kernel("oqh_norm_float16_t", oqh_norm, float16_t)
instantiate_kernel("oqh_norm_bfloat16_t", oqh_norm, bfloat16_t)

// clang-format on
