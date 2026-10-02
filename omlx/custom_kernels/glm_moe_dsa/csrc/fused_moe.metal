#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "kernels/quantized_moe.h"

#define instantiate_quantized_head_flat(name, type, group_size, bits, aligned) \
  instantiate_kernel(                                                          \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,         \
      name,                                                                    \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      aligned,                                                                 \
      true)

#define instantiate_moe_weighted_sum_tiled(type, score_type, topk, threads)    \
  instantiate_kernel(                                                          \
      "moe_weighted_sum_tiled_" #type "_score_" #score_type "_topk_" #topk     \
      "_t_" #threads,                                                          \
      moe_weighted_sum_tiled,                                                  \
      type,                                                                    \
      score_type,                                                              \
      topk,                                                                    \
      threads)

instantiate_quantized_head_flat(affine_qmm_t_head_flat, float16_t, 64, 8, true);
instantiate_quantized_head_flat(
    affine_qmm_t_head_flat,
    bfloat16_t,
    64,
    8,
    true);

instantiate_moe_weighted_sum_tiled(float16_t, float, 8, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 8, 256);
instantiate_moe_weighted_sum_tiled(float16_t, float, 6, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 6, 256);
