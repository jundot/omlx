// Copyright © 2026 oMLX contributors
// SPDX-License-Identifier: Apache-2.0

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/variant.h>

#include "bonsai_kernels.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
    m.doc() = "Native Bonsai 1-bit / 2-bit decode kernels for oMLX";

    // ABI canary — see qwen35_prefill/csrc/bindings.cpp for rationale.
    m.def(
        "abi_probe",
        [](const mlx::core::array& a) { return static_cast<int64_t>(a.size()); },
        "a"_a);

    m.def("is_nax_available", &omlx::bonsai_kernels::is_nax_available);

    m.def(
        "bonsai_q1_affine_qmv",
        &omlx::bonsai_kernels::bonsai_q1_affine_qmv,
        "x"_a, "w"_a, "scales"_a, "biases"_a,
        "stream"_a = nb::none());

    m.def(
        "bonsai_q2_affine_qmv",
        &omlx::bonsai_kernels::bonsai_q2_affine_qmv,
        "x"_a, "w"_a, "scales"_a, "biases"_a,
        "stream"_a = nb::none());

    m.def(
        "bonsai_q1_affine_qmv_wide",
        &omlx::bonsai_kernels::bonsai_q1_affine_qmv_wide,
        "x"_a, "w"_a, "scales"_a, "biases"_a,
        "stream"_a = nb::none());

    m.def(
        "bonsai_q2_affine_qmv_wide",
        &omlx::bonsai_kernels::bonsai_q2_affine_qmv_wide,
        "x"_a, "w"_a, "scales"_a, "biases"_a,
        "stream"_a = nb::none());

    m.def(
        "bonsai_q1_affine_qmv_sym",
        &omlx::bonsai_kernels::bonsai_q1_affine_qmv_sym,
        "x"_a, "w"_a, "scales"_a, "biases"_a,
        "stream"_a = nb::none());

    m.def(
        "bonsai_q2_affine_qmv_sym",
        &omlx::bonsai_kernels::bonsai_q2_affine_qmv_sym,
        "x"_a, "w"_a, "scales"_a, "biases"_a,
        "stream"_a = nb::none());

    m.def(
        "bonsai_q1_affine_qmv_wide_sym",
        &omlx::bonsai_kernels::bonsai_q1_affine_qmv_wide_sym,
        "x"_a, "w"_a, "scales"_a, "biases"_a,
        "stream"_a = nb::none());

    m.def(
        "bonsai_q2_affine_qmv_wide_sym",
        &omlx::bonsai_kernels::bonsai_q2_affine_qmv_wide_sym,
        "x"_a, "w"_a, "scales"_a, "biases"_a,
        "stream"_a = nb::none());

    m.def(
        "bonsai_t5_qmv",
        &omlx::bonsai_kernels::bonsai_t5_qmv,
        "x"_a, "w"_a, "scales"_a,
        "stream"_a = nb::none());

    m.def(
        "bonsai_t5_qmv_wide",
        &omlx::bonsai_kernels::bonsai_t5_qmv_wide,
        "x"_a, "w"_a, "scales"_a,
        "stream"_a = nb::none());

    m.def(
        "bonsai_t5_qmm",
        &omlx::bonsai_kernels::bonsai_t5_qmm,
        "x"_a, "w"_a, "scales"_a,
        "stream"_a = nb::none());

    m.def(
        "bonsai_spec_decode_verify",
        &omlx::bonsai_kernels::bonsai_spec_decode_verify,
        "draft"_a, "target"_a,
        "stream"_a = nb::none());
}
