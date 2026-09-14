// Stage-A activation quantization and the weight-decode reference kernel for
// the oQ mixed-bit QxA8 path.
//
// These compile into the classic metallib: neither kernel touches the M5
// tensor units, so they must keep working on hardware that has no NAX. Only
// the GEMM in qwen35_oq_a8_nax.metal needs the 26.2 deployment floor.

// clang-format off
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

#include "oq_a8_decode.h"
// clang-format on

using namespace metal;
using namespace omlx::oq_a8;

// Simdgroups per Stage-A threadgroup. One threadgroup handles one row; each
// simdgroup owns whole affine groups, so the group sums Ra never straddle a
// simdgroup and reduce with a single simd_sum.
// The host launches 32 * kQuantSimdgroups threads per group; see
// kQuantSimdgroups in qwen35_oq_a8.cpp.
constant constexpr int kQuantSimdgroups = 8;

// ACT_MODE 0: one scale per row. ACT_MODE 1: one scale per
// group of 64, aligned with the weight groups.
//
// Both modes emit the same Qa/Ra so the GEMM shares its integer path; only
// the shape of Sa and the point at which it is applied differ.
template <typename T, int ACT_MODE>
[[kernel]] void oq_a8_quantize(
    const device T* x [[buffer(0)]],
    device int8_t* qa [[buffer(1)]],
    device float* sa [[buffer(2)]],
    device short* ra [[buffer(3)]],
    const constant int& K [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int row = int(tgid);
  const int groups = K / kGroupSize;

  const device T* xr = x + size_t(row) * size_t(K);
  device int8_t* qr = qa + size_t(row) * size_t(K);
  device short* rr = ra + size_t(row) * size_t(groups);

  threadgroup float partial_amax[kQuantSimdgroups];
  float row_inv_scale = 0.0f;

  if (ACT_MODE == 0) {
    // Pass 1: row-wide max(|x|). Each simdgroup folds its own groups, then
    // the threadgroup folds the per-simdgroup results.
    float amax = 0.0f;
    for (int g = int(simd_gid); g < groups; g += kQuantSimdgroups) {
      const int base = g * kGroupSize + int(simd_lid);
      amax = max(amax, abs(float(xr[base])));
      amax = max(amax, abs(float(xr[base + 32])));
    }
    amax = simd_max(amax);
    if (simd_lid == 0) {
      partial_amax[simd_gid] = amax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float row_amax = 0.0f;
#pragma clang loop unroll(full)
    for (int i = 0; i < kQuantSimdgroups; ++i) {
      row_amax = max(row_amax, partial_amax[i]);
    }
    // An all-zero row has no representable scale; emit zeros rather than a
    // NaN and let the affine bias term carry the output.
    row_inv_scale = row_amax > 0.0f ? (127.0f / row_amax) : 0.0f;
    if (simd_gid == 0 && simd_lid == 0) {
      sa[row] = row_amax > 0.0f ? (row_amax / 127.0f) : 0.0f;
    }
  }

  // Pass 2: quantize and accumulate the group sums Ra. Ra must be built from
  // the rounded codes, not from x, or the affine bias correction stops being
  // exact.
  for (int g = int(simd_gid); g < groups; g += kQuantSimdgroups) {
    const int base = g * kGroupSize + int(simd_lid);
    const float x0 = float(xr[base]);
    const float x1 = float(xr[base + 32]);

    float inv_scale;
    if (ACT_MODE == 0) {
      inv_scale = row_inv_scale;
    } else {
      float amax = max(abs(x0), abs(x1));
      amax = simd_max(amax);
      inv_scale = amax > 0.0f ? (127.0f / amax) : 0.0f;
      if (simd_lid == 0) {
        sa[size_t(row) * size_t(groups) + g] =
            amax > 0.0f ? (amax / 127.0f) : 0.0f;
      }
    }

    // rint() is roundTiesToEven, which numpy's rint reproduces exactly, so
    // the correctness tests can compare codes rather than tolerances.
    const int q0 = int(clamp(rint(x0 * inv_scale), -127.0f, 127.0f));
    const int q1 = int(clamp(rint(x1 * inv_scale), -127.0f, 127.0f));

    qr[base] = int8_t(q0);
    qr[base + 32] = int8_t(q1);

    // |Ra| <= 64 * 127 = 8128, so INT16 is sufficient.
    const int group_sum = simd_sum(q0 + q1);
    if (simd_lid == 0) {
      rr[g] = short(group_sum);
    }
  }
}

#define instantiate_oq_a8_quantize(type, act_mode)                            \
  instantiate_kernel(                                                         \
      "oq_a8_quantize_" #type "_am" #act_mode,                                \
      oq_a8_quantize,                                                         \
      type,                                                                   \
      act_mode)

instantiate_oq_a8_quantize(float16_t, 0);
instantiate_oq_a8_quantize(float16_t, 1);
instantiate_oq_a8_quantize(bfloat16_t, 0);
instantiate_oq_a8_quantize(bfloat16_t, 1);

// Weight-decode reference kernel.
//
// Test-only: it writes unpacked INT8 to device memory, which is exactly what
// the production path must never do, so the bit-exactness tests have something
// to compare against MLX's dequantize. The GEMM never calls it.
template <int BITS>
[[kernel]] void oq_a8_decode_weights(
    const device uint32_t* w [[buffer(0)]],
    device int8_t* out [[buffer(1)]],
    const constant int& K [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  const int group = int(gid.x);
  const int row = int(gid.y);
  const int groups = K / kGroupSize;
  if (group >= groups) {
    return;
  }

  const device uint32_t* words =
      w + (size_t(row) * size_t(groups) + size_t(group)) * oq_group_words(BITS);

  int8_t codes[kGroupSize];
  oq_decode_group<BITS>(words, codes);

  device int8_t* dst = out + size_t(row) * size_t(K) + size_t(group) * kGroupSize;
#pragma clang loop unroll(full)
  for (int i = 0; i < kGroupSize; ++i) {
    dst[i] = codes[i];
  }
}

#define instantiate_oq_a8_decode_weights(bits)                                \
  instantiate_kernel(                                                         \
      "oq_a8_decode_weights_q" #bits, oq_a8_decode_weights, bits)

instantiate_oq_a8_decode_weights(4);
instantiate_oq_a8_decode_weights(5);
