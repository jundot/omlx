// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// Instantiates only the Bonsai-specific kernel variants that are not present
// in the stock mlx 0.32.0 metallib:
//   - affine_qmv_fast  for bits=1 (1-bit path added by Bonsai MLX fork)
//   - affine_qmv_wide  for bits=1 and bits=2 (new kernel from Bonsai fork)
//
// The vendored quantized.h in this directory is the Bonsai-patched version.
// It must shadow the mlx-installed copy; CMake sets -I for this directory
// before the system mlx include path.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "quantized.h"   // Bonsai-patched: 1-bit qmv_fast + qmv_wide

// ---- instantiation helpers ------------------------------------------------

#define bonsai_instantiate_qmv_fast(type, group_size, bits, batched)       \
  instantiate_kernel(                                                       \
      "affine_qmv_fast_" #type "_gs_" #group_size "_b_" #bits              \
          "_batch_" #batched,                                               \
      affine_qmv_fast, type, group_size, bits, batched)

#define bonsai_instantiate_qmv(type, group_size, bits, batched)            \
  instantiate_kernel(                                                       \
      "affine_qmv_" #type "_gs_" #group_size "_b_" #bits                   \
          "_batch_" #batched,                                               \
      affine_qmv, type, group_size, bits, batched)

#define bonsai_instantiate_qmv_wide(type, group_size, bits, nv, kl, batch) \
  instantiate_kernel(                                                       \
      "affine_qmv_wide_" #type "_gs_" #group_size "_b_" #bits              \
          "_nv_" #nv "_kl_" #kl "_batch_" #batch,                          \
      affine_qmv_wide, type, group_size, bits, nv, kl, batch)

// ---- qmv_fast for bits=1 and bits=2 (Bonsai fork) -------------------------
// group_sizes 32, 64, 128; types float, float16_t, bfloat16_t

#define bonsai_qmv_fast_bits(type, gs, bits) \
  bonsai_instantiate_qmv_fast(type, gs, bits, 0) \
  bonsai_instantiate_qmv(type, gs, bits, 0)

#define bonsai_qmv_fast_types(gs, bits) \
  bonsai_qmv_fast_bits(float, gs, bits) \
  bonsai_qmv_fast_bits(float16_t, gs, bits) \
  bonsai_qmv_fast_bits(bfloat16_t, gs, bits)

// bits=1
bonsai_qmv_fast_types(32, 1)
bonsai_qmv_fast_types(64, 1)
bonsai_qmv_fast_types(128, 1)

// bits=2: qmv_fast for single-row (M=1) decode, with uint32 load optimizations
bonsai_qmv_fast_types(32, 2)
bonsai_qmv_fast_types(64, 2)
bonsai_qmv_fast_types(128, 2)

// ---- qmv_wide: bits=1 and bits=2 -----------------------------------------
// vecs_per_tg 2..5; k_lanes=8 for affine mode; batch 0 and 1

#define bonsai_qmv_wide_bit(type, gs, bits) \
  bonsai_instantiate_qmv_wide(type, gs, bits, 2, 8, 0) \
  bonsai_instantiate_qmv_wide(type, gs, bits, 3, 8, 0) \
  bonsai_instantiate_qmv_wide(type, gs, bits, 4, 8, 0) \
  bonsai_instantiate_qmv_wide(type, gs, bits, 5, 8, 0)

#define bonsai_qmv_wide_types(gs, bits) \
  bonsai_qmv_wide_bit(float, gs, bits) \
  bonsai_qmv_wide_bit(float16_t, gs, bits) \
  bonsai_qmv_wide_bit(bfloat16_t, gs, bits)

// bits=1
bonsai_qmv_wide_types(32, 1)
bonsai_qmv_wide_types(64, 1)
bonsai_qmv_wide_types(128, 1)

// bits=2
bonsai_qmv_wide_types(32, 2)
bonsai_qmv_wide_types(64, 2)
bonsai_qmv_wide_types(128, 2)

// ---- symmetric variants (identity I-B: bias = -scale*ratio, no DRAM load) --
// Only instantiated for bits=1 and bits=2 (the only Bonsai quantization widths).

#define bonsai_instantiate_qmv_fast_sym(type, group_size, bits, batched)     \
  instantiate_kernel(                                                         \
      "affine_qmv_fast_sym_" #type "_gs_" #group_size "_b_" #bits            \
          "_batch_" #batched,                                                 \
      affine_qmv_fast_sym, type, group_size, bits, batched)

#define bonsai_instantiate_qmv_wide_sym(type, group_size, bits, nv, kl, batch) \
  instantiate_kernel(                                                          \
      "affine_qmv_wide_sym_" #type "_gs_" #group_size "_b_" #bits             \
          "_nv_" #nv "_kl_" #kl "_batch_" #batch,                             \
      affine_qmv_wide_sym, type, group_size, bits, nv, kl, batch)

#define bonsai_qmv_fast_sym_bits(type, gs, bits) \
  bonsai_instantiate_qmv_fast_sym(type, gs, bits, 0)

#define bonsai_qmv_fast_sym_types(gs, bits) \
  bonsai_qmv_fast_sym_bits(float, gs, bits) \
  bonsai_qmv_fast_sym_bits(float16_t, gs, bits) \
  bonsai_qmv_fast_sym_bits(bfloat16_t, gs, bits)

#define bonsai_qmv_wide_sym_bit(type, gs, bits) \
  bonsai_instantiate_qmv_wide_sym(type, gs, bits, 2, 8, 0) \
  bonsai_instantiate_qmv_wide_sym(type, gs, bits, 3, 8, 0) \
  bonsai_instantiate_qmv_wide_sym(type, gs, bits, 4, 8, 0) \
  bonsai_instantiate_qmv_wide_sym(type, gs, bits, 5, 8, 0)

#define bonsai_qmv_wide_sym_types(gs, bits) \
  bonsai_qmv_wide_sym_bit(float, gs, bits) \
  bonsai_qmv_wide_sym_bit(float16_t, gs, bits) \
  bonsai_qmv_wide_sym_bit(bfloat16_t, gs, bits)

// qmv_fast_sym: bits=1 and bits=2
bonsai_qmv_fast_sym_types(32, 1)
bonsai_qmv_fast_sym_types(64, 1)
bonsai_qmv_fast_sym_types(128, 1)
bonsai_qmv_fast_sym_types(32, 2)
bonsai_qmv_fast_sym_types(64, 2)
bonsai_qmv_fast_sym_types(128, 2)

// qmv_wide_sym: bits=1 and bits=2
bonsai_qmv_wide_sym_types(32, 1)
bonsai_qmv_wide_sym_types(64, 1)
bonsai_qmv_wide_sym_types(128, 1)
bonsai_qmv_wide_sym_types(32, 2)
bonsai_qmv_wide_sym_types(64, 2)
bonsai_qmv_wide_sym_types(128, 2)

// ---- t5: base-3 ternary packing (Identity I-D) ----------------------------
// ~1.585 bpw vs 2.0 bpw; always symmetric (no bias tensor).
// Only gs=64 and gs=128 (the two Bonsai group sizes).

// t5 qmv kernels have no batch dimension — always B=1 in practice and the
// batched template parameter was never read by the implementations.
// A single instantiation per (type, gs) avoids ~60 dead kernel variants.
template <typename T, int group_size>
[[kernel]] void affine_qmv_fast_t5(
    const device uint8_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* x [[buffer(2)]],
    device T* y [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
    // Load the T5_TO_B4 LUT into threadgroup memory once per threadgroup
    // (128 threads: 32 lanes x 4 simdgroups cover all 256 entries in 2 steps).
    threadgroup uint t5_lut[256];
    for (uint i = simd_lid + simd_gid * 32; i < 256; i += 128) {
        t5_lut[i] = T5_TO_B4[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    qmv_fast_t5_impl<T, group_size, /*USE_SIGMA=*/false>(
        w, scales, x, y, in_vec_size, out_vec_size, nullptr, t5_lut,
        tid, simd_gid, simd_lid);
}

template <typename T, int group_size, int vecs_per_tg, int k_lanes>
[[kernel]] void affine_qmv_wide_t5(
    const device uint8_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* x [[buffer(2)]],
    device T* y [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]])
{
    // Same cooperative T5_TO_B4 preload as affine_qmv_fast_t5
    // (128 threads: 32 lanes x 4 simdgroups).
    threadgroup uint t5_lut[256];
    for (uint i = simd_lid + simd_gid * 32; i < 256; i += 128) {
        t5_lut[i] = T5_TO_B4[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    qmv_wide_t5_impl<T, group_size, vecs_per_tg, k_lanes>(
        w, scales, x, y, in_vec_size, out_vec_size, M, t5_lut,
        tid, simd_gid, simd_lid);
}

#define bonsai_instantiate_qmv_fast_t5(type, gs)                            \
  instantiate_kernel(                                                        \
      "affine_qmv_fast_t5_" #type "_gs_" #gs,                               \
      affine_qmv_fast_t5, type, gs)

#define bonsai_instantiate_qmv_wide_t5(type, gs, nv, kl)                   \
  instantiate_kernel(                                                        \
      "affine_qmv_wide_t5_" #type "_gs_" #gs "_nv_" #nv "_kl_" #kl,       \
      affine_qmv_wide_t5, type, gs, nv, kl)

#define bonsai_qmv_fast_t5_types(gs)                                        \
  bonsai_instantiate_qmv_fast_t5(float, gs)                                 \
  bonsai_instantiate_qmv_fast_t5(float16_t, gs)                             \
  bonsai_instantiate_qmv_fast_t5(bfloat16_t, gs)

#define bonsai_qmv_wide_t5_gs(type, gs)                                     \
  bonsai_instantiate_qmv_wide_t5(type, gs, 2, 8)                           \
  bonsai_instantiate_qmv_wide_t5(type, gs, 3, 8)                           \
  bonsai_instantiate_qmv_wide_t5(type, gs, 4, 8)                           \
  bonsai_instantiate_qmv_wide_t5(type, gs, 5, 8)

#define bonsai_qmv_wide_t5_types(gs)                                        \
  bonsai_qmv_wide_t5_gs(float, gs)                                          \
  bonsai_qmv_wide_t5_gs(float16_t, gs)                                      \
  bonsai_qmv_wide_t5_gs(bfloat16_t, gs)

// qmv_fast_t5: gs=64 and gs=128
bonsai_qmv_fast_t5_types(64)
bonsai_qmv_fast_t5_types(128)

// qmv_wide_t5: gs=64 and gs=128
bonsai_qmv_wide_t5_types(64)
bonsai_qmv_wide_t5_types(128)

// ---- t5 MMA GEMM (Identity I-M): fused dequant + simdgroup matmul for prefill ----
// w(0)=uint8 t5 bytes, scales(1), x(2), out(3), M(4), N(5), K(6)
// Grid: (ceil(N/32), ceil(M/32), B)  TG: (32, 4, 1)

template <typename T, int group_size>
[[kernel]] void affine_qmm_t5(
    const device uint8_t* w  [[buffer(0)]],
    const device T* scales   [[buffer(1)]],
    const device T* x        [[buffer(2)]],
    device T* out            [[buffer(3)]],
    const constant int& M    [[buffer(4)]],
    const constant int& N    [[buffer(5)]],
    const constant int& K    [[buffer(6)]],
    uint2 tgid               [[threadgroup_position_in_grid]],
    uint  lane               [[thread_index_in_simdgroup]],
    uint  sg_id              [[simdgroup_index_in_threadgroup]])
{
    // Threadgroup tiles declared here (Metal requires [[kernel]] scope for threadgroup vars)
    threadgroup T xs[32 * (group_size + 4)];
    threadgroup T ws[32 * (group_size + 4)];
    qmm_t5_impl<T, group_size>(w, scales, x, out, M, N, K, xs, ws, tgid, lane, sg_id);
}

#define bonsai_instantiate_qmm_t5(type, gs) \
  instantiate_kernel(                        \
      "affine_qmm_t5_" #type "_gs_" #gs,    \
      affine_qmm_t5, type, gs)

#define bonsai_qmm_t5_types(gs)                \
  bonsai_instantiate_qmm_t5(float, gs)         \
  bonsai_instantiate_qmm_t5(float16_t, gs)     \
  bonsai_instantiate_qmm_t5(bfloat16_t, gs)

bonsai_qmm_t5_types(64)
bonsai_qmm_t5_types(128)
// clang-format on
