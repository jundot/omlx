#include "mlx/backend/metal/kernels/utils.h"
#include "kernels/steel_attention_block_token.h"

#define instantiate_qwen35_fa256_attn(tname, dtype, bq, bk, bd, wm, wn)       \
  instantiate_kernel(                                                         \
      "omlx_qwen35_fa256_attention_" #tname "_bq" #bq "_bk" #bk "_bd" #bd    \
      "_wm" #wm "_wn" #wn "_mask" #tname,                                    \
      attention,                                                              \
      dtype,                                                                  \
      bq,                                                                     \
      bk,                                                                     \
      bd,                                                                     \
      wm,                                                                     \
      wn,                                                                     \
      dtype,                                                                  \
      float)

instantiate_qwen35_fa256_attn(float16, half, 16, 8, 256, 2, 1);
instantiate_qwen35_fa256_attn(float16, half, 16, 16, 256, 2, 1);
instantiate_qwen35_fa256_attn(float16, half, 32, 8, 256, 4, 1);
instantiate_qwen35_fa256_attn(float16, half, 32, 16, 256, 4, 1);

instantiate_qwen35_fa256_attn(bfloat16, bfloat16_t, 16, 8, 256, 2, 1);
instantiate_qwen35_fa256_attn(bfloat16, bfloat16_t, 16, 16, 256, 2, 1);
instantiate_qwen35_fa256_attn(bfloat16, bfloat16_t, 32, 8, 256, 4, 1);
instantiate_qwen35_fa256_attn(bfloat16, bfloat16_t, 32, 16, 256, 4, 1);

instantiate_kernel(
    "omlx_qwen35_fa256_chunk_reduce_float16",
    attention_chunk_reduce,
    half);
instantiate_kernel(
    "omlx_qwen35_fa256_chunk_reduce_bfloat16",
    attention_chunk_reduce,
    bfloat16_t);
