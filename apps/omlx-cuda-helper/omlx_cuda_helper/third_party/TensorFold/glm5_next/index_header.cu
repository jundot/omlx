// Adapted from TensorFold zig/kernels/cuda/common/cp_async.cuh,
// common/qmm_frag.cuh and glm_index_scores.cuh (Apache-2.0, Copyright 2026
// TensorFold contributors). Changes: only the primitives the index score
// kernel uses, renamed and gathered in one header; the mma wrapper adds each
// 16-product sum to the accumulator rounded to nearest.
#include <cuda_bf16.h>
__device__ __forceinline__ uint32_t tfi_smem(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void tfi_cp16(void* dst, const void* src, bool read) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(tfi_smem(dst)), "l"(src), "r"(read ? 16 : 0));
}
__device__ __forceinline__ void tfi_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void tfi_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
__device__ __forceinline__ void tfi_ldm4(uint32_t* r, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(tfi_smem(p)));
}
// The tensor core sums the 16 products from zero; the sum is added to the
// accumulator rounded to nearest, which its own accumulation is not.
__device__ __forceinline__ void tfi_mma(float* c, const uint32_t* a, const uint32_t* b) {
  const float zero = 0.f;
  float p[4];
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};\n"
      : "=f"(p[0]), "=f"(p[1]), "=f"(p[2]), "=f"(p[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(zero));
  c[0] = __fadd_rn(c[0], p[0]); c[1] = __fadd_rn(c[1], p[1]); c[2] = __fadd_rn(c[2], p[2]); c[3] = __fadd_rn(c[3], p[3]);
}
// Element offset of 16-byte chunk u of key row `row` (0..63) in a staged
// tile: rows are rotated so that ldmatrix reads without bank conflicts.
__device__ __forceinline__ int tfi_tile_at(int row, int u) { return row * 128 + ((u ^ (row & 7)) << 3); }
