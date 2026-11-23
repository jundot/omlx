// Adapted from TensorFold zig/kernels/cuda/common/qmm_frag.cuh and
// common/cp_async.cuh (Apache-2.0, Copyright 2026 TensorFold contributors).
// Changes: only cp.async with zero fill, ldmatrix x4 (plain and transposed)
// and the bf16 m16n8k16 mma are kept, renamed with a qsa_ prefix; qsa_mma_rn
// and qsa_split are added.
#include <cuda_bf16.h>
#include <math_constants.h>

typedef __nv_bfloat16 qsa_bf16;

__device__ __forceinline__ unsigned qsa_smem(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void qsa_cp16(void* dst, const void* src, bool read) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(qsa_smem(dst)), "l"(src),
               "r"(read ? 16 : 0));
}
__device__ __forceinline__ void qsa_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void qsa_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void qsa_ldm4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(qsa_smem(p)));
}
__device__ __forceinline__ void qsa_ldm4t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(qsa_smem(p)));
}
__device__ __forceinline__ void qsa_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                        uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// d += a @ b with the 16 products summed from zero and added with a correctly
// rounded float add: the tensor core truncates when it adds to an accumulator.
__device__ __forceinline__ void qsa_mma_rn(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                           uint32_t b1) {
  const float zero = 0.f;
  float p[4];
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
      "{%8, %9}, {%10, %10, %10, %10};\n"
      : "=f"(p[0]), "=f"(p[1]), "=f"(p[2]), "=f"(p[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(zero));
  d[0] = __fadd_rn(d[0], p[0]);
  d[1] = __fadd_rn(d[1], p[1]);
  d[2] = __fadd_rn(d[2], p[2]);
  d[3] = __fadd_rn(d[3], p[3]);
}
// Two float32 values as a bf16 pair and the bf16 pair of what rounding lost.
__device__ __forceinline__ void qsa_split(float x, float y, __nv_bfloat162& high,
                                          __nv_bfloat162& low) {
  high = __floats2bfloat162_rn(x, y);
  const float2 back = __bfloat1622float2(high);
  low = __floats2bfloat162_rn(x - back.x, y - back.y);
}
