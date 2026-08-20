// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
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
        "shared_nvfp4_down_residual",
        &omlx::laguna_nvfp4::shared_nvfp4_down_residual,
        "activated"_a, "down_weight"_a, "down_scales"_a,
        "routed"_a, "residual"_a,
        "stream"_a = nb::none());
}
