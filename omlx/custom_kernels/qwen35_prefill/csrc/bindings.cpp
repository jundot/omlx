#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "qwen35_prefill.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native Qwen3.5/3.6 prefill kernels for oMLX";

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
      "qwen35_fa256_attention",
      &omlx::qwen35_prefill_kernels::qwen35_fa256_attention,
      "q"_a,
      "k"_a,
      "v"_a,
      "scale"_a,
      "causal"_a = true,
      "q_block"_a = 32,
      "k_block"_a = 8,
      "stream"_a = nb::none());
  m.def(
      "qwen35_q4_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_q4_affine_qmm_t,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8,
      "stream"_a = nb::none());
  m.def(
      "qwen35_q5_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_q5_affine_qmm_t,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8,
      "stream"_a = nb::none());
  m.def(
      "qwen35_q6_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_q6_affine_qmm_t,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8,
      "stream"_a = nb::none());
  m.def(
      "qwen35_q8_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_q8_affine_qmm_t,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8,
      "stream"_a = nb::none());
  m.def(
      "qwen35_moe_weighted_sum",
      &omlx::qwen35_prefill_kernels::qwen35_moe_weighted_sum,
      "x_sorted"_a,
      "inv_order"_a,
      "scores"_a,
      "stream"_a = nb::none());
}
