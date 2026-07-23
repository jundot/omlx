#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "dsa_indexer.h"
#include "deepseek_v4_sparse_attention.h"
#include "exact_block_attention.h"
#include "fused_moe.h"
#include "sparse_mla.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native GLM kernels for oMLX";

  // ABI canary: when the extension is built with a nanobind whose ABI tag
  // differs from the one the mlx wheel was built with, the NB_DOMAIN is
  // isolated and every mx.array argument is rejected with "incompatible
  // function arguments" (issue #2139). fast.py calls this probe once at
  // import and disables the native symbols when it fails.
  m.def(
      "abi_probe",
      [](const mlx::core::array& a) {
        return static_cast<int64_t>(a.size());
      },
      "a"_a);

  m.def(
      "dsa_indexer_scores",
      &omlx::glm_kernels::dsa_indexer_scores,
      "queries"_a,
      "keys"_a,
      "weights"_a,
      "causal"_a = true,
      "unused_causal_prefix_topk"_a = 0,
      "skip_causal_future_store"_a = false,
      "causal_q_offset"_a = -1,
      "stream"_a = nb::none());
  m.def(
      "dsa_topk_indices",
      &omlx::glm_kernels::dsa_topk_indices,
      "scores"_a,
      "topk"_a,
      "bucketed"_a = false,
      "causal_valid_prefix"_a = false,
      "stream"_a = nb::none());
  m.def(
      "glm_dsa_sparse_mla_attention",
      &omlx::glm_kernels::glm_dsa_sparse_mla_attention,
      "q_latent"_a,
      "q_pe"_a,
      "kv_latent"_a,
      "k_pe"_a,
      "topk_indices"_a,
      "scale"_a,
      "causal"_a = true,
      "topk_valid_prefix"_a = false,
      "causal_prefix_indices"_a = false,
      "topk_length"_a = nb::none(),
      "causal_prefix_rows"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "glm_dsa_exact_block_attention",
      &omlx::glm_kernels::glm_dsa_exact_block_attention,
      "q"_a,
      "k"_a,
      "v"_a,
      "block_mask"_a,
      "block_token_mask"_a,
      "scale"_a,
      "causal"_a = true,
      "stream"_a = nb::none());
  m.def(
      "deepseek_v4_sparse_attention",
      &omlx::glm_kernels::deepseek_v4_sparse_attention,
      "q"_a,
      "local_kv"_a,
      "pooled"_a,
      "topk_indices"_a,
      "sinks"_a,
      "scale"_a,
      "q_offset"_a,
      "compress_ratio"_a,
      "local_window"_a,
      "stream"_a = nb::none());
  m.def(
      "glm_dsa_q8_vup_flat",
      &omlx::glm_kernels::glm_dsa_q8_vup_flat,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "stream"_a = nb::none());
  m.def(
      "glm_moe_weighted_sum",
      &omlx::glm_kernels::glm_moe_weighted_sum,
      "x_sorted"_a,
      "inv_order"_a,
      "scores"_a,
      "stream"_a = nb::none());
  m.def(
      "deepseek_mxfp4_gather_qmm_blocks",
      &omlx::glm_kernels::deepseek_mxfp4_gather_qmm_blocks,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "block_meta"_a,
      "block_count"_a,
      "variant"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "deepseek_mxfp4_gather_qmm_pair_blocks",
      &omlx::glm_kernels::deepseek_mxfp4_gather_qmm_pair_blocks,
      "x"_a,
      "weight0"_a,
      "scales0"_a,
      "weight1"_a,
      "scales1"_a,
      "block_meta"_a,
      "block_count"_a,
      "variant"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "deepseek_mxfp4_gather_qmm_pair_concat_blocks",
      &omlx::glm_kernels::deepseek_mxfp4_gather_qmm_pair_concat_blocks,
      "x"_a,
      "weight0"_a,
      "scales0"_a,
      "weight1"_a,
      "scales1"_a,
      "block_meta"_a,
      "block_count"_a,
      "variant"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "deepseek_affine_gather_qmm_blocks",
      &omlx::glm_kernels::deepseek_affine_gather_qmm_blocks,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "block_meta"_a,
      "block_count"_a,
      "group_size"_a,
      "bits"_a,
      "variant"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "deepseek_affine_gather_qmm_pair_concat_blocks",
      &omlx::glm_kernels::deepseek_affine_gather_qmm_pair_concat_blocks,
      "x"_a,
      "weight0"_a,
      "scales0"_a,
      "biases0"_a,
      "weight1"_a,
      "scales1"_a,
      "biases1"_a,
      "block_meta"_a,
      "block_count"_a,
      "group_size"_a,
      "bits"_a,
      "variant"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "deepseek_mxfp4_gather_qmm_expert",
      &omlx::glm_kernels::deepseek_mxfp4_gather_qmm_expert,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "indices"_a,
      "variant"_a = 0,
      "stream"_a = nb::none());
}
