#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>

#include "dsa_indexer.h"
#include "msa_block_topk.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native sparse-attention index kernels for oMLX";

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
      &omlx::sparse_index::dsa_indexer_scores,
      "queries"_a,
      "keys"_a,
      "weights"_a,
      "causal"_a = true,
      "unused_causal_prefix_topk"_a = 0,
      "skip_causal_future_store"_a = false,
      "causal_q_offset"_a = -1,
      "mask_ratio"_a = 0,
      "mask_q_offset"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "dsa_indexer_scores_mma",
      &omlx::sparse_index::dsa_indexer_scores_mma,
      "queries"_a,
      "keys"_a,
      "weights"_a,
      "mask_ratio"_a = 0,
      "mask_q_offset"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "qsa_indexer_scores",
      &omlx::sparse_index::qsa_indexer_scores,
      "queries"_a,
      "pooled_keys"_a,
      "mask_ratio"_a = 4,
      "mask_q_offset"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "qsa_topk_indices",
      &omlx::sparse_index::qsa_topk_indices,
      "scores"_a,
      "topk"_a = 512,
      "stream"_a = nb::none());
  m.def(
      "dsa_topk_indices",
      &omlx::sparse_index::dsa_topk_indices,
      "scores"_a,
      "topk"_a,
      "bucketed"_a = false,
      "causal_valid_prefix"_a = false,
      "stream"_a = nb::none());
  m.def(
      "fp32_topk_indices",
      &omlx::sparse_index::fp32_topk_indices,
      "scores"_a,
      "topk"_a = 512,
      "stream"_a = nb::none());
  m.def(
      "dsa_decode_scores",
      &omlx::sparse_index::dsa_decode_scores,
      "queries"_a,
      "keys"_a,
      "weights"_a,
      "fp32_scores"_a = false,
      "stream"_a = nb::none());
  m.def(
      "msa_block_topk",
      &omlx::sparse_index::msa_block_topk,
      "idx_queries"_a,
      "idx_keys"_a,
      "q_start"_a,
      "scale"_a,
      "block_size"_a,
      "topk"_a,
      "init_blocks"_a,
      "local_blocks"_a,
      "stream"_a = nb::none());
}
