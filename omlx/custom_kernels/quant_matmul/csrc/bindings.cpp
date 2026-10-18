#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "affine_qmm.h"
#include "ane/ane_hybrid.h"
#include "command_buffer_caps.h"
#include "command_queue.h"
#include "gather_qmm.h"
#include "grouped_expert.h"
#include "head_flat_qmm.h"
#include "moe_weighted_sum.h"
#include "mxfp8_qmv.h"
#include "nax.h"
#include "oq_a8.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace omlx::moe_staging {
void register_bindings(nb::module_& m);
}

NB_MODULE(_ext, m) {
  m.doc() = "Native quantized matmul and ANE offload kernels for oMLX";

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
  m.def("grouped_expert",
        &omlx::quant_matmul::grouped_expert,
        "gate"_a, "up"_a, "activation"_a, "down"_a);
  m.def(
      "mxfp8_qmv_pair",
      &omlx::quant_matmul::mxfp8_qmv_pair,
      "input"_a,
      "weight_a"_a,
      "scales_a"_a,
      "weight_b"_a,
      "scales_b"_a,
      "stream"_a = nb::none());
  m.def(
      "q8_head_flat_qmm",
      &omlx::quant_matmul::q8_head_flat_qmm,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "stream"_a = nb::none());
  m.def(
      "mxfp4_gather_qmm_blocks",
      &omlx::quant_matmul::mxfp4_gather_qmm_blocks,
      "x"_a,
      "weight"_a,
      "scales"_a,
      "block_meta"_a,
      "block_count"_a,
      "variant"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "mxfp4_gather_qmm_pair_blocks",
      &omlx::quant_matmul::mxfp4_gather_qmm_pair_blocks,
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
      "mxfp4_gather_qmm_pair_concat_blocks",
      &omlx::quant_matmul::mxfp4_gather_qmm_pair_concat_blocks,
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
      "affine_gather_qmm_blocks",
      &omlx::quant_matmul::affine_gather_qmm_blocks,
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
      "affine_gather_qmm_pair_concat_blocks",
      &omlx::quant_matmul::affine_gather_qmm_pair_concat_blocks,
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
      "is_nax_available",
      &omlx::quant_matmul::is_nax_available);

  omlx::moe_staging::register_bindings(m);

  m.def(
      "set_command_buffer_caps",
      &omlx::quant_matmul::set_command_buffer_caps,
      "ops"_a,
      "mb"_a);
  m.def(
      "share_command_queue",
      &omlx::quant_matmul::share_command_queue,
      "stream"_a,
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "nax_qmm_kernels_built",
      &omlx::quant_matmul::nax_qmm_kernels_built);
  m.def(
      "nax_qmm_runtime_active",
      &omlx::quant_matmul::nax_qmm_runtime_active);
  m.def(
      "ane_available",
      &omlx::quant_matmul::ane_available);
  m.def(
      "ane_hybrid_nax_enabled",
      &omlx::quant_matmul::ane_hybrid_nax_enabled);
  m.def(
      "ane_profile_set_enabled",
      &omlx::quant_matmul::ane_profile_set_enabled,
      "enabled"_a);
  m.def(
      "ane_profile_reset",
      &omlx::quant_matmul::ane_profile_reset);
  m.def(
      "ane_profile_snapshot",
      &omlx::quant_matmul::ane_profile_snapshot);
  nb::class_<omlx::quant_matmul::AneLinearModel>(m, "AneLinearModel")
      .def_prop_ro(
          "input_dim",
          &omlx::quant_matmul::AneLinearModel::input_dim)
      .def_prop_ro(
          "output_dim",
          &omlx::quant_matmul::AneLinearModel::output_dim)
      .def_prop_ro(
          "sequence_length",
          &omlx::quant_matmul::AneLinearModel::sequence_length)
      .def(
          "warmup",
          &omlx::quant_matmul::AneLinearModel::warmup,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "has_error",
          &omlx::quant_matmul::AneLinearModel::has_error);
  nb::class_<omlx::quant_matmul::AneLinearBankBuilder>(
      m, "AneLinearBankBuilder")
      .def(nb::init<int>(), "sequence_length"_a)
      .def(
          "add",
          &omlx::quant_matmul::AneLinearBankBuilder::add,
          "weight"_a,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "size",
          &omlx::quant_matmul::AneLinearBankBuilder::size)
      .def(
          "compile",
          &omlx::quant_matmul::AneLinearBankBuilder::compile,
          "ane_instance"_a,
          "start"_a,
          "stop"_a,
          nb::call_guard<nb::gil_scoped_release>());
  nb::class_<omlx::quant_matmul::AneFusedBankBuilder>(
      m, "AneFusedBankBuilder")
      .def(nb::init<int>(), "sequence_length"_a)
      .def(
          "add",
          &omlx::quant_matmul::AneFusedBankBuilder::add,
          "gate_weight"_a,
          "up_weight"_a,
          "down_weight"_a,
          nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro(
          "size",
          &omlx::quant_matmul::AneFusedBankBuilder::size)
      .def(
          "compile",
          &omlx::quant_matmul::AneFusedBankBuilder::compile,
          "ane_instance"_a,
          "start"_a,
          "stop"_a,
          nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "ane_compile_linear",
      static_cast<std::shared_ptr<
          omlx::quant_matmul::AneLinearModel> (*)(
          const mlx::core::array &, int, int)>(
      &omlx::quant_matmul::ane_compile_linear),
      "weight"_a,
      "sequence_length"_a,
      "ane_instance"_a = 0,
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "ane_compile_linear_bank",
      &omlx::quant_matmul::ane_compile_linear_bank,
      "weights"_a,
      "sequence_length"_a,
      "ane_instance"_a,
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "ane_compile_fp16_linear",
      &omlx::quant_matmul::ane_compile_fp16_linear,
      "weight"_a,
      "sequence_length"_a,
      nb::call_guard<nb::gil_scoped_release>());
  m.def("ane_planar", &omlx::quant_matmul::ane_planar, "x"_a, "model"_a);
  m.def("ane_compile_program", &omlx::quant_matmul::ane_compile_program,
        "mil"_a, "weight_blob"_a, "input_dim"_a, "output_dim"_a,
        "sequence_length"_a, nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "ane_compile_swiglu_down",
      &omlx::quant_matmul::ane_compile_swiglu_down,
      "gate_weight"_a,
      "up_weight"_a,
      "down_weight"_a,
      "sequence_length"_a,
      "ane_instance"_a = 0,
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "ane_compile_swiglu_down_bank",
      &omlx::quant_matmul::ane_compile_swiglu_down_bank,
      "gate_weights"_a,
      "up_weights"_a,
      "down_weights"_a,
      "sequence_length"_a,
      "ane_instance"_a,
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "ane_affine_qmm_t",
      &omlx::quant_matmul::ane_affine_qmm_t,
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
  m.def("ane_cpu_fp16_affine_qmm_t",
        &omlx::quant_matmul::ane_cpu_fp16_affine_qmm_t, "x"_a,
        "cpu_weight"_a, "gpu_weight"_a, "gpu_scales"_a, "gpu_biases"_a,
        "ane_model"_a, "bits"_a, "variant"_a = 8, "group_size"_a = 128,
        "profile_category"_a = 1, "cpu_threads"_a = 0,
        "cpu_shared_resource"_a = false, "stream"_a = nb::none());
  m.def(
      "cpu_shared_resource_available",
      &omlx::quant_matmul::cpu_shared_resource_available);
  m.def(
      "cpu_fp16_affine_qmm_t",
      &omlx::quant_matmul::cpu_fp16_affine_qmm_t,
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
      "ane_q4_affine_qmm_t",
      &omlx::quant_matmul::ane_q4_affine_qmm_t,
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
      "ane_q4_swiglu_t",
      &omlx::quant_matmul::ane_q4_swiglu_t,
      "x"_a,
      "gpu_weight"_a,
      "gpu_scales"_a,
      "gpu_biases"_a,
      "ane_model"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "stream"_a = nb::none());
  m.def(
      "ane_affine_swiglu_t",
      &omlx::quant_matmul::ane_affine_swiglu_t,
      "x"_a,
      "gpu_weight"_a,
      "gpu_scales"_a,
      "gpu_biases"_a,
      "ane_model"_a,
      "bits"_a,
      "variant"_a = 8,
      "group_size"_a = 128,
      "stream"_a = nb::none());
  m.def("ane_cpu_fp16_swiglu_t",
        &omlx::quant_matmul::ane_cpu_fp16_swiglu_t, "x"_a,
        "cpu_weight"_a, "gpu_weight"_a, "gpu_scales"_a, "gpu_biases"_a,
        "ane_model"_a, "bits"_a, "variant"_a = 8, "group_size"_a = 128,
        "cpu_threads"_a = 0, "cpu_shared_resource"_a = false,
        "stream"_a = nb::none());
  m.def("ane_cpu_fp16_q4_swiglu_t",
        &omlx::quant_matmul::ane_cpu_fp16_q4_swiglu_t, "x"_a,
        "cpu_weight"_a, "gpu_weight"_a, "gpu_scales"_a, "gpu_biases"_a,
        "ane_model"_a, "variant"_a = 8, "group_size"_a = 128,
        "cpu_threads"_a = 0, "cpu_shared_resource"_a = false,
        "stream"_a = nb::none());
  m.def(
      "ane_dual_affine_qmm_t",
      &omlx::quant_matmul::ane_dual_affine_qmm_t,
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
      "ane_dual_cpu_fp16_affine_qmm_t",
      &omlx::quant_matmul::ane_dual_cpu_fp16_affine_qmm_t,
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
      "ane_dual_cpu_fp16_swiglu_t",
      &omlx::quant_matmul::ane_dual_cpu_fp16_swiglu_t,
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
      "ane_dual_q4_swiglu_t",
      &omlx::quant_matmul::ane_dual_q4_swiglu_t,
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
      "ane_dual_affine_swiglu_t",
      &omlx::quant_matmul::ane_dual_affine_swiglu_t,
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
      "ane_dual_cpu_fp16_q4_swiglu_t",
      &omlx::quant_matmul::ane_dual_cpu_fp16_q4_swiglu_t,
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
      "ane_q4_swiglu_down_t",
      &omlx::quant_matmul::ane_q4_swiglu_down_t,
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
      "ane_dual_q4_swiglu_down_t",
      &omlx::quant_matmul::ane_dual_q4_swiglu_down_t,
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
      "ane_dual_cpu_fp16_q4_swiglu_down_t",
      &omlx::quant_matmul::
          ane_dual_cpu_fp16_q4_swiglu_down_t,
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
      "affine_q2_qmm_t",
      &omlx::quant_matmul::affine_q2_qmm_t,
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
      "affine_q4_qmm_t",
      &omlx::quant_matmul::affine_q4_qmm_t,
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
      "affine_q5_qmm_t",
      &omlx::quant_matmul::affine_q5_qmm_t,
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
      "affine_q6_qmm_t",
      &omlx::quant_matmul::affine_q6_qmm_t,
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
      "affine_q8_qmm_t",
      &omlx::quant_matmul::affine_q8_qmm_t,
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
      "oq_a8_kernels_available",
      &omlx::quant_matmul::oq_a8_kernels_available);
  m.def(
      "oq_a8_quantize",
      &omlx::quant_matmul::oq_a8_quantize,
      "x"_a,
      "act_mode"_a = 0,
      "stream"_a = nb::none());
  m.def(
      "oq_a8_qmm_t",
      &omlx::quant_matmul::oq_a8_qmm_t,
      "qa"_a,
      "sa"_a,
      "ra"_a,
      "weight"_a,
      "scales"_a,
      "biases"_a,
      "bits"_a,
      "act_mode"_a = 0,
      "variant"_a = 800,
      "packed"_a = false,
      "stream"_a = nb::none());
  m.def(
      "oq_a8_decode_weights",
      &omlx::quant_matmul::oq_a8_decode_weights,
      "weight"_a,
      "bits"_a,
      "group_count"_a,
      "stream"_a = nb::none());
  m.def(
      "moe_weighted_sum",
      &omlx::quant_matmul::moe_weighted_sum,
      "x_sorted"_a,
      "inv_order"_a,
      "scores"_a,
      "stream"_a = nb::none());
}
