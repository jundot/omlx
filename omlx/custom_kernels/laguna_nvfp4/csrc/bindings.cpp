// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/variant.h>

#include "laguna_nvfp4_kernels.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "Native NVFP4 (E4M3) decode kernels for oMLX, ported from "
              "Layr-Labs/mlxfast-challenge";

    // ABI canary — see qwen35_prefill/csrc/bindings.cpp for rationale.
    m.def(
        "abi_probe",
        [](const mlx::core::array& a) {
            return static_cast<int64_t>(a.size());
        },
        "a"_a);

    m.def(
        "shared_nvfp4_swiglu_qmv",
        &omlx::laguna_nvfp4::shared_nvfp4_swiglu_qmv,
        "x"_a, "w"_a, "scales"_a,
        "stream"_a = nb::none());

    m.def(
        "routed_nvfp4_swiglu_qmv",
        &omlx::laguna_nvfp4::routed_nvfp4_swiglu_qmv,
        "input"_a, "fused_weight"_a, "fused_scales"_a, "indices"_a,
        "stream"_a = nb::none());

    m.def(
        "routed_nvfp4_down_reduce",
        &omlx::laguna_nvfp4::routed_nvfp4_down_reduce,
        "activated"_a, "down_weight"_a, "down_scales"_a,
        "indices"_a, "router_weights"_a,
        "stream"_a = nb::none());

    m.def(
        "full_qk_norm_yarn",
        &omlx::laguna_nvfp4::full_qk_norm_yarn,
        "raw_queries"_a, "raw_keys"_a, "query_weight"_a, "key_weight"_a,
        "angles"_a,
        "stream"_a = nb::none());

    m.def(
        "sliding_qk_norm_rope",
        &omlx::laguna_nvfp4::sliding_qk_norm_rope,
        "raw_queries"_a, "raw_keys"_a, "query_weight"_a, "key_weight"_a,
        "angles"_a,
        "stream"_a = nb::none());

    m.def(
        "decode_nvfp4_qkv_r1",
        &omlx::laguna_nvfp4::decode_nvfp4_qkv_r1,
        "normalized"_a, "weight_codes"_a, "weight_scales"_a, "heads"_a,
        "stream"_a = nb::none());

    m.def(
        "oproj_act",
        &omlx::laguna_nvfp4::oproj_act,
        "attention_output"_a, "gate_values"_a, "weight_codes"_a,
        "weight_scales"_a, "heads"_a,
        "stream"_a = nb::none());

    m.def(
        "residual_rms",
        &omlx::laguna_nvfp4::residual_rms,
        "residual"_a, "branch"_a, "weight"_a,
        "stream"_a = nb::none());

    m.def(
        "decode_router_top8",
        &omlx::laguna_nvfp4::decode_router_top8,
        "logits"_a, "correction_bias"_a, "normalizing"_a,
        "stream"_a = nb::none());

    m.def(
        "decode_router_top8_ordinal",
        &omlx::laguna_nvfp4::decode_router_top8_ordinal,
        "logits"_a, "correction_bias"_a, "normalizing"_a, "score_table"_a,
        "stream"_a = nb::none());

    m.def(
        "sliding_fused_attn_ring",
        &omlx::laguna_nvfp4::sliding_fused_attn_ring,
        "raw_queries"_a, "raw_keys"_a, "raw_values"_a,
        "query_weight"_a, "key_weight"_a, "angles"_a,
        "k_cache"_a, "v_cache"_a, "params"_a, "scale_arr"_a,
        "stream"_a = nb::none());

    m.def(
        "residual_rms_router",
        &omlx::laguna_nvfp4::residual_rms_router,
        "residual"_a, "branch"_a, "weight"_a, "router_weight"_a,
        "correction_bias"_a,
        "stream"_a = nb::none());

    m.def(
        "prefill_moe_tail",
        &omlx::laguna_nvfp4::prefill_moe_tail,
        "expert_outputs"_a, "router_weights"_a, "shared_output"_a,
        "residual"_a,
        "stream"_a = nb::none());

    m.def(
        "prefill_router_top8",
        &omlx::laguna_nvfp4::prefill_router_top8,
        "logits"_a, "correction_bias"_a, "normalizing"_a,
        "stream"_a = nb::none());

    m.def(
        "prefill_router_tournament",
        &omlx::laguna_nvfp4::prefill_router_tournament,
        "logits"_a, "correction_bias"_a, "normalizing"_a = false,
        "stream"_a = nb::none());

    m.def(
        "prefill_router_tournament_ordinal",
        &omlx::laguna_nvfp4::prefill_router_tournament_ordinal,
        "logits"_a, "correction_bias"_a, "normalizing"_a,
        "stream"_a = nb::none());

    m.def(
        "lm_head_prune",
        &omlx::laguna_nvfp4::lm_head_prune,
        "x"_a, "codes_lo"_a, "codes_hi"_a, "scales"_a, "lm_head"_a,
        "stream"_a = nb::none());

    m.def(
        "inject_empty_dispatch",
        &omlx::laguna_nvfp4::inject_empty_dispatch,
        "control"_a, "prev"_a, "stream"_a = nb::none());

    m.def(
        "inject_dram_sweep",
        &omlx::laguna_nvfp4::inject_dram_sweep,
        "pool"_a, "control"_a, "stream"_a = nb::none());

    m.def(
        "decode_embedding_rope_atlas",
        &omlx::laguna_nvfp4::decode_embedding_rope_atlas,
        "tokens"_a, "embedding_weight"_a, "full_atlas"_a, "sliding_atlas"_a,
        "atlas_position"_a, "stream"_a = nb::none());

    m.def(
        "full_fused_attn_grow",
        &omlx::laguna_nvfp4::full_fused_attn_grow,
        "raw_queries"_a, "raw_keys"_a, "raw_values"_a,
        "query_weight"_a, "key_weight"_a, "angles"_a,
        "k_cache"_a, "v_cache"_a, "params"_a, "scale_arr"_a,
        "stream"_a = nb::none());

    m.def(
        "lm_head_int5_base_coarse",
        &omlx::laguna_nvfp4::lm_head_int5_base_coarse,
        "x"_a, "codes_base"_a, "scales"_a,
        "stream"_a = nb::none());

    m.def(
        "lm_head_exact_sparse_refine",
        &omlx::laguna_nvfp4::lm_head_exact_sparse_refine,
        "coarse"_a, "delta"_a, "thr"_a, "lm_head"_a, "x"_a,
        "codes_bit"_a, "scales"_a,
        "stream"_a = nb::none());

    m.def(
        "shared_nvfp4_down_residual",
        &omlx::laguna_nvfp4::shared_nvfp4_down_residual,
        "activated"_a, "down_weight"_a, "down_scales"_a,
        "routed"_a, "residual"_a,
        "stream"_a = nb::none());

    m.def(
        "shared_nvfp4_swiglu_qmv_rows1",
        &omlx::laguna_nvfp4::shared_nvfp4_swiglu_qmv_rows1,
        "x"_a, "w"_a, "scales"_a,
        "stream"_a = nb::none());

    m.def(
        "shared_nvfp4_swiglu_qmv_rows1_halved",
        &omlx::laguna_nvfp4::shared_nvfp4_swiglu_qmv_rows1_halved,
        "x"_a, "w"_a, "scales"_a,
        "stream"_a = nb::none());

    m.def(
        "shared_nvfp4_swiglu_qmv_rows1_halved_wide",
        &omlx::laguna_nvfp4::shared_nvfp4_swiglu_qmv_rows1_halved_wide,
        "x"_a, "w"_a, "scales"_a,
        "stream"_a = nb::none());

    m.def(
        "shared_nvfp4_down_residual_halved",
        &omlx::laguna_nvfp4::shared_nvfp4_down_residual_halved,
        "activated"_a, "down_weight"_a, "down_scales"_a,
        "routed"_a, "residual"_a,
        "stream"_a = nb::none());

    m.def(
        "routed_nvfp4_swiglu_qmv_rows1",
        &omlx::laguna_nvfp4::routed_nvfp4_swiglu_qmv_rows1,
        "input"_a, "fused_weight"_a, "fused_scales"_a, "indices"_a,
        "stream"_a = nb::none());

    m.def(
        "routed_nvfp4_swiglu_qmv_packed",
        &omlx::laguna_nvfp4::routed_nvfp4_swiglu_qmv_packed,
        "input"_a, "fused_weight"_a, "packed_scales"_a, "indices"_a,
        "stream"_a = nb::none());

    m.def(
        "routed_nvfp4_swiglu_qmv_packed_top8keys",
        &omlx::laguna_nvfp4::routed_nvfp4_swiglu_qmv_packed_top8keys,
        "input"_a, "fused_weight"_a, "packed_scales"_a, "router_keys"_a,
        "stream"_a = nb::none());

    m.def(
        "routed_nvfp4_swiglu_qmv_packed_top8keys_r1",
        &omlx::laguna_nvfp4::routed_nvfp4_swiglu_qmv_packed_top8keys_r1,
        "input"_a, "fused_weight"_a, "packed_scales"_a, "router_keys"_a,
        "stream"_a = nb::none());
}
