// Adapted from mlx-serve src/kernels/moe/moe_prefill_cuda_reduce.cu (MIT,
// Copyright (c) 2026 David Dalcu). Changes: the routing weights are float32
// and the weighted rows sum in float32 (each product rounded, then added in
// expert order), rounded once to bfloat16, as MiniMax combines its routed
// experts.
//
// Inputs: down [R, H] bf16 routed rows sorted by expert, inverse [T * TOPK]
// uint32 (the sorted row of each token's k-th expert), weights [T, TOPK]
// float32.
// Output: y [T, H] bf16 = sum_k weights[t, k] * down[inverse[t, k]].
// Grid: one block per token, a thread per 8 output columns.
const int t = blockIdx.x;
const int d = threadIdx.x * 8;
float acc[8];
#pragma unroll
for (int i = 0; i < 8; ++i) acc[i] = 0.f;
for (int j = 0; j < TOPK; ++j) {
  const uint32_t r = inverse[t * TOPK + j];
  const float weight = weights[t * TOPK + j];
  const uint4 v = *reinterpret_cast<const uint4*>(down + (long)r * H + d);
  const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(&v);
  // The product rounds before the add, as the host's separate multiply and
  // sum do; a fused multiply-add would round once.
  #pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = __fadd_rn(acc[i], __fmul_rn(__bfloat162float(vb[i]), weight));
}
uint4 o;
__nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
for (int i = 0; i < 8; ++i) ob[i] = __float2bfloat16_rn(acc[i]);
*reinterpret_cast<uint4*>(y + (long)t * H + d) = o;
