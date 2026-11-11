// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0
//
// Instantiates the Bonsai t5 (base-3 ternary) kernels for the native extension.
// Stock MLX serves the uint32 1-bit and 2-bit affine formats.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "t5_quantized.h"

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
