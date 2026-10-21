#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "block_mask_attention.h"
#include "compressed_sparse_attention.h"
#include "fa256_attention.h"
#include "packed_sparse_attention.h"
#include "sdpa_decode.h"
#include "sparse_gqa_attention.h"
#include "sparse_mla.h"
#include "verify_gemm.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native attention kernels for oMLX";

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
      "sparse_gqa_attention",
      &omlx::attention::sparse_gqa_attention,
      "queries"_a,
      "keys"_a,
      "values"_a,
      "selected_blocks"_a,
      "scale"_a,
      "q_offset"_a,
      "key_tile"_a = 128,
      "dimension_tile"_a = 32,
      "stream"_a = nb::none());
  m.def(
      "sparse_mla_attention",
      &omlx::attention::sparse_mla_attention,
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
      "block_mask_attention",
      &omlx::attention::block_mask_attention,
      "q"_a,
      "k"_a,
      "v"_a,
      "block_mask"_a,
      "block_token_mask"_a,
      "scale"_a,
      "causal"_a = true,
      "stream"_a = nb::none());
  m.def(
      "rowwise_gemm",
      &omlx::attention::rowwise_gemm,
      "lhs"_a,
      "rhs"_a,
      "transpose_rhs"_a,
      "stream"_a = nb::none());
  m.def(
      "ring_gemm",
      &omlx::attention::ring_gemm,
      "lhs"_a,
      "source"_a,
      "indices"_a,
      "transpose_rhs"_a,
      "stream"_a = nb::none());
  m.def(
      "compressed_sparse_attention",
      &omlx::attention::compressed_sparse_attention,
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
      "packed_sparse_attention",
      &omlx::attention::packed_sparse_attention,
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
      "fa256_attention",
      &omlx::attention::fa256_attention,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "causal"_a = true,
      "q_block"_a = 32,
      "k_block"_a = 8,
      "dispatch_budget"_a = 0,
      "stream"_a = nb::none());
  // Capability probe for fast.py: older extensions reject the
  // dispatch_budget kwarg, so the wrapper only forwards it when present.
  m.attr("FA256_HAS_DISPATCH_BUDGET") = true;
  m.def(
      "sdpa_decode_supported",
      &omlx::attention::sdpa_decode_supported,
      "q"_a,
      "k"_a,
      "v"_a,
      "stream"_a = nb::none());
  m.def(
      "sdpa_decode",
      &omlx::attention::sdpa_decode,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "causal"_a,
      "mask"_a = nb::none(),
      "sinks"_a = nb::none(),
      "stream"_a = nb::none());
}
