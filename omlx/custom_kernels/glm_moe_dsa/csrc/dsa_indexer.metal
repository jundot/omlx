#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"

namespace mlx {
namespace steel {

struct OMLXDSATopKParams {
  int rows;
  int L;
  int K;
  int topk;
  bool causal_valid_prefix;
};

} // namespace steel
} // namespace mlx

#define DSATopKParams OMLXDSATopKParams
#include "kernels/steel_dsa_indexer_score.h"
#undef DSATopKParams

#define instantiate_dsa_indexer_score(iname, itype, bm, bn, bk, wm, wn) \
  instantiate_kernel(                                                   \
      "steel_dsa_indexer_score_" #iname                                 \
      "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn,              \
      dsa_indexer_score, itype, bm, bn, bk, wm, wn)

#define instantiate_dsa_topk_indices(iname, itype, topk, threads)       \
  instantiate_kernel(                                                   \
      "steel_dsa_topk_indices_" #iname "_topk" #topk "_t" #threads,    \
      dsa_topk_indices_16bit,                                           \
      itype,                                                            \
      uint,                                                             \
      topk,                                                             \
      threads)

instantiate_dsa_indexer_score(float16, half, 64, 64, 16, 2, 2);
instantiate_dsa_indexer_score(bfloat16, bfloat16_t, 64, 64, 16, 2, 2);

instantiate_dsa_topk_indices(float16, half, 2048, 1024);
instantiate_dsa_topk_indices(bfloat16, bfloat16_t, 2048, 1024);
instantiate_dsa_topk_indices(float16, half, 512, 1024);
instantiate_dsa_topk_indices(bfloat16, bfloat16_t, 512, 1024);
