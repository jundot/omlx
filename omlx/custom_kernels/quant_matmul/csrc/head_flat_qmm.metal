#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "affine_quantized.h"

#define instantiate_quantized_head_flat(name, type, group_size, bits, aligned) \
  instantiate_kernel(                                                          \
      #name "_" #type "_gs_" #group_size "_b_" #bits "_alN_" #aligned,         \
      name,                                                                    \
      type,                                                                    \
      group_size,                                                              \
      bits,                                                                    \
      aligned,                                                                 \
      true)

instantiate_quantized_head_flat(affine_qmm_t_head_flat, float16_t, 64, 8, true);
instantiate_quantized_head_flat(
    affine_qmm_t_head_flat,
    bfloat16_t,
    64,
    8,
    true);
