#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"
#include "kernels/steel_sparse_mla.h"

#define instantiate_sparse_mla(tname, dtype, bk, dc, h, d, pe, wm)      \
  instantiate_kernel(                                                   \
      "steel_sparse_mla_" #tname "_bk" #bk "_dc" #dc "_h" #h           \
      "_d" #d "_pe" #pe "_wm" #wm,                                     \
      sparse_mla_attention,                                             \
      dtype,                                                            \
      bk,                                                               \
      dc,                                                               \
      h,                                                                \
      d,                                                                \
      pe,                                                               \
      wm,                                                               \
      uint,                                                             \
      float)

instantiate_sparse_mla(float16, half, 256, 32, 64, 512, 64, 8);
instantiate_sparse_mla(bfloat16, bfloat16_t, 256, 32, 64, 512, 64, 8);
