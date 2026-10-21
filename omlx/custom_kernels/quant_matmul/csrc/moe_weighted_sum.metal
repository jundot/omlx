#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "affine_quantized.h"

#define instantiate_moe_weighted_sum_tiled(type, score_type, topk, threads)    \
  instantiate_kernel(                                                          \
      "moe_weighted_sum_tiled_" #type "_score_" #score_type "_topk_" #topk     \
      "_t_" #threads,                                                          \
      moe_weighted_sum_tiled,                                                  \
      type,                                                                    \
      score_type,                                                              \
      topk,                                                                    \
      threads)

instantiate_moe_weighted_sum_tiled(float16_t, float, 8, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 8, 256);
instantiate_moe_weighted_sum_tiled(float16_t, float, 6, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 6, 256);
instantiate_moe_weighted_sum_tiled(float16_t, float, 10, 256);
instantiate_moe_weighted_sum_tiled(bfloat16_t, float, 10, 256);
