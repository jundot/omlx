#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "qwen35_ane.h"
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
      "is_nax_available",
      &omlx::qwen35_prefill_kernels::is_nax_available);
  m.def(
      "nax_qmm_kernels_built",
      &omlx::qwen35_prefill_kernels::nax_qmm_kernels_built);
  m.def(
      "nax_qmm_runtime_active",
      &omlx::qwen35_prefill_kernels::nax_qmm_runtime_active);
  m.def(
      "qwen35_ane_available",
      &omlx::qwen35_prefill_kernels::qwen35_ane_available);
  m.def(
      "qwen35_ane_hybrid_nax_enabled",
      &omlx::qwen35_prefill_kernels::qwen35_ane_hybrid_nax_enabled);
  m.def(
      "qwen35_ane_profile_set_enabled",
      &omlx::qwen35_prefill_kernels::qwen35_ane_profile_set_enabled,
      "enabled"_a);
  m.def(
      "qwen35_ane_profile_reset",
      &omlx::qwen35_prefill_kernels::qwen35_ane_profile_reset);
  m.def(
      "qwen35_ane_profile_snapshot",
      &omlx::qwen35_prefill_kernels::qwen35_ane_profile_snapshot);
  nb::class_<omlx::qwen35_prefill_kernels::AneLinearModel>(m, "AneLinearModel")
      .def_prop_ro(
          "input_dim",
          &omlx::qwen35_prefill_kernels::AneLinearModel::input_dim)
      .def_prop_ro(
          "output_dim",
          &omlx::qwen35_prefill_kernels::AneLinearModel::output_dim)
      .def_prop_ro(
          "sequence_length",
          &omlx::qwen35_prefill_kernels::AneLinearModel::sequence_length)
      .def(
          "warmup",
          &omlx::qwen35_prefill_kernels::AneLinearModel::warmup,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "has_error",
          &omlx::qwen35_prefill_kernels::AneLinearModel::has_error);
  nb::class_<omlx::qwen35_prefill_kernels::AneLinearBankBuilder>(
      m, "AneLinearBankBuilder")
      .def(nb::init<int>(), "sequence_length"_a)
      .def(
          "add",
          &omlx::qwen35_prefill_kernels::AneLinearBankBuilder::add,
          "weight"_a,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "size",
          &omlx::qwen35_prefill_kernels::AneLinearBankBuilder::size)
      .def(
          "compile",
          &omlx::qwen35_prefill_kernels::AneLinearBankBuilder::compile,
          "ane_instance"_a,
          "start"_a,
          "stop"_a,
          nb::call_guard<nb::gil_scoped_release>());
  nb::class_<omlx::qwen35_prefill_kernels::AneFusedBankBuilder>(
      m, "AneFusedBankBuilder")
      .def(nb::init<int>(), "sequence_length"_a)
      .def(
          "add",
          &omlx::qwen35_prefill_kernels::AneFusedBankBuilder::add,
          "gate_weight"_a,
          "up_weight"_a,
          "down_weight"_a,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "size",
          &omlx::qwen35_prefill_kernels::AneFusedBankBuilder::size)
      .def(
          "compile",
          &omlx::qwen35_prefill_kernels::AneFusedBankBuilder::compile,
          "ane_instance"_a,
          "start"_a,
          "stop"_a,
          nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "qwen35_ane_compile_linear",
      static_cast<std::shared_ptr<
          omlx::qwen35_prefill_kernels::AneLinearModel> (*)(
          const mlx::core::array &, int, int)>(
      &omlx::qwen35_prefill_kernels::qwen35_ane_compile_linear),
      "weight"_a,
      "sequence_length"_a,
      "ane_instance"_a = 0,
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "qwen35_ane_compile_linear_bank",
      &omlx::qwen35_prefill_kernels::qwen35_ane_compile_linear_bank,
      "weights"_a,
      "sequence_length"_a,
      "ane_instance"_a,
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "qwen35_ane_compile_fp16_linear",
      &omlx::qwen35_prefill_kernels::qwen35_ane_compile_fp16_linear,
      "weight"_a,
      "sequence_length"_a,
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "qwen35_ane_compile_swiglu_down",
      &omlx::qwen35_prefill_kernels::qwen35_ane_compile_swiglu_down,
      "gate_weight"_a,
      "up_weight"_a,
      "down_weight"_a,
      "sequence_length"_a,
      "ane_instance"_a = 0,
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "qwen35_ane_compile_swiglu_down_bank",
      &omlx::qwen35_prefill_kernels::qwen35_ane_compile_swiglu_down_bank,
      "gate_weights"_a,
      "up_weights"_a,
      "down_weights"_a,
      "sequence_length"_a,
      "ane_instance"_a,
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "qwen35_ane_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_ane_affine_qmm_t,
      "x"_a,
      "gpu_weight"_a,
      "gpu_scales"_a,
      "gpu_biases"_a,
      "ane_model"_a,
      "bits"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "profile_category"_a = 1,
      "stream"_a = nb::none());
  m.def("qwen35_ane_cpu_fp16_affine_qmm_t",
        &omlx::qwen35_prefill_kernels::qwen35_ane_cpu_fp16_affine_qmm_t, "x"_a,
        "cpu_weight"_a, "gpu_weight"_a, "gpu_scales"_a, "gpu_biases"_a,
        "ane_model"_a, "bits"_a, "variant"_a = 8, "group_size"_a = 128,
        "profile_category"_a = 1, "cpu_threads"_a = 0,
        "cpu_shared_resource"_a = false, "stream"_a = nb::none());
  m.def(
      "qwen35_cpu_shared_resource_available",
      &omlx::qwen35_prefill_kernels::qwen35_cpu_shared_resource_available);
  m.def(
      "qwen35_cpu_fp16_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_cpu_fp16_affine_qmm_t,
      "x"_a,
      "cpu_weight"_a,
      "gpu_weight"_a,
      "gpu_scales"_a,
      "gpu_biases"_a,
      "bits"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "cpu_threads"_a = 0,
      "cpu_shared_resource"_a = false,
      "stream"_a = nb::none());
  m.def(
      "qwen35_ane_q4_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_ane_q4_affine_qmm_t,
      "x"_a,
      "gpu_weight"_a,
      "gpu_scales"_a,
      "gpu_biases"_a,
      "ane_model"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "profile_category"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "qwen35_ane_q4_swiglu_t",
      &omlx::qwen35_prefill_kernels::qwen35_ane_q4_swiglu_t,
      "x"_a,
      "gpu_weight"_a,
      "gpu_scales"_a,
      "gpu_biases"_a,
      "ane_model"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "stream"_a = nb::none());
  m.def(
      "qwen35_ane_affine_swiglu_t",
      &omlx::qwen35_prefill_kernels::qwen35_ane_affine_swiglu_t,
      "x"_a,
      "gpu_weight"_a,
      "gpu_scales"_a,
      "gpu_biases"_a,
      "ane_model"_a,
      "bits"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "stream"_a = nb::none());
  m.def("qwen35_ane_cpu_fp16_swiglu_t",
        &omlx::qwen35_prefill_kernels::qwen35_ane_cpu_fp16_swiglu_t, "x"_a,
        "cpu_weight"_a, "gpu_weight"_a, "gpu_scales"_a, "gpu_biases"_a,
        "ane_model"_a, "bits"_a, "variant"_a = 8, "group_size"_a = 128,
        "cpu_threads"_a = 0, "cpu_shared_resource"_a = false,
        "stream"_a = nb::none());
  m.def("qwen35_ane_cpu_fp16_q4_swiglu_t",
        &omlx::qwen35_prefill_kernels::qwen35_ane_cpu_fp16_q4_swiglu_t, "x"_a,
        "cpu_weight"_a, "gpu_weight"_a, "gpu_scales"_a, "gpu_biases"_a,
        "ane_model"_a, "variant"_a = 8, "group_size"_a = 128,
        "cpu_threads"_a = 0, "cpu_shared_resource"_a = false,
        "stream"_a = nb::none());
  m.def(
      "qwen35_ane_dual_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_ane_dual_affine_qmm_t,
      "x"_a,
      "gpu_weight"_a,
      "gpu_scales"_a,
      "gpu_biases"_a,
      "ane_model0"_a,
      "ane_model1"_a,
      "bits"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "profile_category"_a = 1,
      "stream"_a = nb::none());
  m.def(
      "qwen35_ane_dual_cpu_fp16_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_ane_dual_cpu_fp16_affine_qmm_t,
      "x"_a,
      "cpu_weight"_a,
      "gpu_weight"_a,
      "gpu_scales"_a,
      "gpu_biases"_a,
      "ane_model0"_a,
      "ane_model1"_a,
      "bits"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "profile_category"_a = 1,
      "cpu_threads"_a = 0,
      "cpu_shared_resource"_a = false,
      "stream"_a = nb::none());
  m.def(
      "qwen35_ane_dual_cpu_fp16_swiglu_t",
      &omlx::qwen35_prefill_kernels::qwen35_ane_dual_cpu_fp16_swiglu_t,
      "x"_a,
      "cpu_weight"_a,
      "gpu_weight"_a,
      "gpu_scales"_a,
      "gpu_biases"_a,
      "ane_model0"_a,
      "ane_model1"_a,
      "bits"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "cpu_threads"_a = 0,
      "cpu_shared_resource"_a = false,
      "stream"_a = nb::none());
  m.def(
      "qwen35_ane_dual_q4_swiglu_t",
      &omlx::qwen35_prefill_kernels::qwen35_ane_dual_q4_swiglu_t,
      "x"_a,
      "gpu_weight"_a,
      "gpu_scales"_a,
      "gpu_biases"_a,
      "ane_model0"_a,
      "ane_model1"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "stream"_a = nb::none());
  m.def(
      "qwen35_ane_dual_affine_swiglu_t",
      &omlx::qwen35_prefill_kernels::qwen35_ane_dual_affine_swiglu_t,
      "x"_a,
      "gpu_weight"_a,
      "gpu_scales"_a,
      "gpu_biases"_a,
      "ane_model0"_a,
      "ane_model1"_a,
      "bits"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "stream"_a = nb::none());
  m.def(
      "qwen35_ane_dual_cpu_fp16_q4_swiglu_t",
      &omlx::qwen35_prefill_kernels::qwen35_ane_dual_cpu_fp16_q4_swiglu_t,
      "x"_a,
      "cpu_weight"_a,
      "gpu_weight"_a,
      "gpu_scales"_a,
      "gpu_biases"_a,
      "ane_model0"_a,
      "ane_model1"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "cpu_threads"_a = 0,
      "cpu_shared_resource"_a = false,
      "stream"_a = nb::none());
  m.def(
      "qwen35_ane_q4_swiglu_down_t",
      &omlx::qwen35_prefill_kernels::qwen35_ane_q4_swiglu_down_t,
      "x"_a,
      "gpu_gate_up_weight"_a,
      "gpu_gate_up_scales"_a,
      "gpu_gate_up_biases"_a,
      "gpu_down_weight"_a,
      "gpu_down_scales"_a,
      "gpu_down_biases"_a,
      "ane_model"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "stream"_a = nb::none());
  m.def(
      "qwen35_ane_dual_q4_swiglu_down_t",
      &omlx::qwen35_prefill_kernels::qwen35_ane_dual_q4_swiglu_down_t,
      "x"_a,
      "gpu_gate_up_weight"_a,
      "gpu_gate_up_scales"_a,
      "gpu_gate_up_biases"_a,
      "gpu_down_weight"_a,
      "gpu_down_scales"_a,
      "gpu_down_biases"_a,
      "ane_model0"_a,
      "ane_model1"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "stream"_a = nb::none());
  m.def(
      "qwen35_ane_dual_cpu_fp16_q4_swiglu_down_t",
      &omlx::qwen35_prefill_kernels::
          qwen35_ane_dual_cpu_fp16_q4_swiglu_down_t,
      "x"_a,
      "cpu_gate_up_weight"_a,
      "cpu_down_weight"_a,
      "gpu_gate_up_weight"_a,
      "gpu_gate_up_scales"_a,
      "gpu_gate_up_biases"_a,
      "gpu_down_weight"_a,
      "gpu_down_scales"_a,
      "gpu_down_biases"_a,
      "ane_model0"_a,
      "ane_model1"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "cpu_threads"_a = 0,
      "cpu_shared_resource"_a = false,
      "stream"_a = nb::none());
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
      "dispatch_budget"_a = 0,
      "stream"_a = nb::none());
  // Capability probe for fast.py: older extensions reject the
  // dispatch_budget kwarg, so the wrapper only forwards it when present.
  m.attr("FA256_HAS_DISPATCH_BUDGET") = true;
  m.def(
      "qwen35_q2_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_q2_affine_qmm_t,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8,
      "use_nax"_a = false,
      "nax_variant"_a = 0,
      "group_size"_a = 64,
      "stream"_a = nb::none());
  m.def(
      "qwen35_q4_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_q4_affine_qmm_t,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8,
      "use_nax"_a = false,
      "nax_variant"_a = 0,
      "group_size"_a = 64,
      "stream"_a = nb::none());
  m.def(
      "qwen35_q5_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_q5_affine_qmm_t,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8,
      "use_nax"_a = false,
      "nax_variant"_a = 0,
      "group_size"_a = 64,
      "stream"_a = nb::none());
  m.def(
      "qwen35_q6_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_q6_affine_qmm_t,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8,
      "use_nax"_a = false,
      "nax_variant"_a = 0,
      "group_size"_a = 64,
      "stream"_a = nb::none());
  m.def(
      "qwen35_q8_affine_qmm_t",
      &omlx::qwen35_prefill_kernels::qwen35_q8_affine_qmm_t,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "variant"_a = 8,
      "use_nax"_a = false,
      "nax_variant"_a = 0,
      "group_size"_a = 64,
      "stream"_a = nb::none());
  m.def(
      "qwen35_moe_weighted_sum",
      &omlx::qwen35_prefill_kernels::qwen35_moe_weighted_sum,
      "x_sorted"_a,
      "inv_order"_a,
      "scores"_a,
      "stream"_a = nb::none());
}
