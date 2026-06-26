#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "dsa_indexer.h"
#include "exact_block_attention.h"
#include "fused_moe.h"
#include "sparse_mla.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native GLM kernels for oMLX";

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
}
