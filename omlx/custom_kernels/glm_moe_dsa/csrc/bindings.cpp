#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include "dsa_indexer.h"
#include "ds4_attention_finalizer.h"
#include "ds4_prefill_moe_nax.h"
#include "ds4_output_chain.h"
#include "ds4_projection_qmm.h"
#include "ds4_prefill_moe.h"
#include "ds4_qkv_bundle.h"
#include "deepseek_v4_sparse_attention.h"
#include "dspark_gemm.h"
#include "dspark_qmv.h"
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
      "mask_ratio"_a = 0,
      "mask_q_offset"_a = 0,
      "stream"_a = nb::none(),
      "use_nax"_a = false);
  m.def(
      "dsa_indexer_nax_kernels_built",
      &omlx::glm_kernels::dsa_indexer_nax_kernels_built);
  m.def(
      "dsa_indexer_nax_runtime_active",
      &omlx::glm_kernels::dsa_indexer_nax_runtime_active);
  m.def(
      "dsa_indexer_scores_mma",
      &omlx::glm_kernels::dsa_indexer_scores_mma,
      "queries"_a,
      "keys"_a,
      "weights"_a,
      "mask_ratio"_a = 0,
      "mask_q_offset"_a = 0,
      "stream"_a = nb::none(),
      "use_wm4_wn1"_a = false);
  m.def(
      "dsa_topk_indices",
      &omlx::glm_kernels::dsa_topk_indices,
      "scores"_a,
      "topk"_a,
      "bucketed"_a = false,
      "causal_valid_prefix"_a = false,
      "stream"_a = nb::none());
  m.def(
      "dspark_fp32_topk_indices",
      &omlx::glm_kernels::dspark_fp32_topk_indices,
      "scores"_a,
      "topk"_a = 512,
      "stream"_a = nb::none());
  m.def(
      "ds4_router_topk_indices",
      &omlx::glm_kernels::ds4_router_topk_indices,
      "scores"_a,
      "stream"_a = nb::none());
  m.def(
      "dsa_decode_scores",
      &omlx::glm_kernels::dsa_decode_scores,
      "queries"_a,
      "keys"_a,
      "weights"_a,
      "fp32_scores"_a = false,
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
      "dspark_rowwise_gemm",
      &omlx::glm_kernels::dspark_rowwise_gemm,
      "lhs"_a,
      "rhs"_a,
      "transpose_rhs"_a,
      "stream"_a = nb::none());
  m.def(
      "dspark_ring_gemm",
      &omlx::glm_kernels::dspark_ring_gemm,
      "lhs"_a,
      "source"_a,
      "indices"_a,
      "transpose_rhs"_a,
      "stream"_a = nb::none());
  m.def(
      "dspark_exact_mxfp8_qmv_pair",
      &omlx::glm_kernels::dspark_exact_mxfp8_qmv_pair,
      "input"_a,
      "weight_a"_a,
      "scales_a"_a,
      "weight_b"_a,
      "scales_b"_a,
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
      "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks",
      &omlx::glm_kernels::deepseek_mxfp4_gather_qmm_pair_swiglu_blocks,
      "x"_a,
      "up_weight"_a,
      "up_scales"_a,
      "gate_weight"_a,
      "gate_scales"_a,
      "block_meta"_a,
      "block_count"_a,
      "activation_limit"_a = 10.0f,
      "variant"_a = 2,
      "stream"_a = nb::none());
  m.def(
      "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8",
      &omlx::glm_kernels::deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8,
      "x"_a,
      "up_weight"_a,
      "up_scales"_a,
      "gate_weight"_a,
      "gate_scales"_a,
      "block_meta"_a,
      "block_count"_a,
      "activation_limit"_a = 10.0f,
      "variant"_a = 2,
      "stream"_a = nb::none());
  m.def(
      "deepseek_mxfp4_gather_qmm_blocks_tail8",
      &omlx::glm_kernels::deepseek_mxfp4_gather_qmm_blocks_tail8,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "block_meta"_a,
      "block_count"_a,
      "variant"_a = 2,
      "stream"_a = nb::none());
  m.def(
      "ds4_q_head_rms_rope",
      &omlx::glm_kernels::ds4_q_head_rms_rope,
      "q"_a,
      "freqs"_a,
      "offset"_a,
      "eps"_a,
      "return_normalized"_a = false,
      "stream"_a = nb::none());
  m.def(
      "ds4_kv_rms_rope",
      &omlx::glm_kernels::ds4_kv_rms_rope,
      "kv"_a,
      "weight"_a,
      "freqs"_a,
      "offset"_a,
      "eps"_a,
      "return_normalized"_a = false,
      "stream"_a = nb::none());
  m.def(
      "ds4_projection_mxfp8_qmm",
      &omlx::glm_kernels::ds4_projection_mxfp8_qmm,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "variant"_a = 0,
      "use_nax"_a = false,
      "nax_variant"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "ds4_output_oa_interleaved",
      &omlx::glm_kernels::ds4_output_oa_interleaved,
      "x"_a,
      "o_a_weight"_a,
      "o_a_scales"_a,
      "variant"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "ds4_output_projection_chain",
      &omlx::glm_kernels::ds4_output_projection_chain,
      "x"_a,
      "o_a_weight"_a,
      "o_a_scales"_a,
      "o_b_weight"_a,
      "o_b_scales"_a,
      "variant"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "ds4_projection_nax_kernels_built",
      &omlx::glm_kernels::ds4_projection_nax_kernels_built);
  m.def(
      "ds4_projection_nax_device_available",
      &omlx::glm_kernels::ds4_projection_nax_device_available);
  m.def(
      "deepseek_mxfp4_gather_qmm_blocks_nax",
      &omlx::glm_kernels::deepseek_mxfp4_gather_qmm_blocks_nax,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "block_meta"_a,
      "block_count"_a,
      "stream"_a = nb::none());
  m.def(
      "deepseek_mxfp4_gather_qmm_pair_blocks_nax",
      &omlx::glm_kernels::deepseek_mxfp4_gather_qmm_pair_blocks_nax,
      "x"_a,
      "weight0"_a,
      "scales0"_a,
      "weight1"_a,
      "scales1"_a,
      "block_meta"_a,
      "block_count"_a,
      "stream"_a = nb::none());
  m.def(
      "deepseek_v4_qkv_pair_b1",
      &omlx::glm_kernels::deepseek_v4_qkv_pair_b1,
      "x"_a,
      "wq_a_weight"_a,
      "wq_a_scales"_a,
      "wkv_weight"_a,
      "wkv_scales"_a,
      "stream"_a = nb::none());
  m.def(
      "deepseek_v4_qkv_compressor128_bundle_b1",
      &omlx::glm_kernels::deepseek_v4_qkv_compressor128_bundle_b1,
      "x"_a,
      "wq_a_weight"_a,
      "wq_a_scales"_a,
      "wkv_weight"_a,
      "wkv_scales"_a,
      "compressor_wkv"_a,
      "compressor_wgate"_a,
      "stream"_a = nb::none());
  m.def(
      "deepseek_v4_qkv_compressor_bundle_b1",
      &omlx::glm_kernels::deepseek_v4_qkv_compressor_bundle_b1,
      "x"_a,
      "wq_a_weight"_a,
      "wq_a_scales"_a,
      "wkv_weight"_a,
      "wkv_scales"_a,
      "compressor_wkv"_a,
      "compressor_wgate"_a,
      "index_compressor_wkv"_a,
      "index_compressor_wgate"_a,
      "stream"_a = nb::none());
  m.def(
      "deepseek_v4_qkv_compressor_bundle_b1_dispatches",
      &omlx::glm_kernels::deepseek_v4_qkv_compressor_bundle_b1_dispatches);
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
  m.def(
      "deepseek_mxfp4_full_decode",
      &omlx::glm_kernels::deepseek_mxfp4_full_decode,
      "x"_a,
      "up_weight"_a,
      "up_scales"_a,
      "gate_weight"_a,
      "gate_scales"_a,
      "down_weight"_a,
      "down_scales"_a,
      "indices"_a,
      "scores"_a,
      "activation_limit"_a,
      "stream"_a = nb::none());
}
