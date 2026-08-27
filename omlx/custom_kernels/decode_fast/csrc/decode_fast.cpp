// SPDX-License-Identifier: Apache-2.0
#include "decode_fast.h"

#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace omlx::decode_fast_kernels {

namespace {

using namespace mlx::core;

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get omlx_decode_fast binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

std::string rms_type_name(Dtype dtype) {
  if (dtype == float32) {
    return "float32";
  }
  if (dtype == float16) {
    return "float16";
  }
  if (dtype == bfloat16) {
    return "bfloat16";
  }
  std::ostringstream msg;
  msg << "Unsupported rms_norm_residual dtype: " << dtype << ".";
  throw std::invalid_argument(msg.str());
}

// Fused residual-add + RMS norm. Mirrors the eval_gpu dispatch of mlx's
// RMSNorm fused branch (ml-explore/mlx#4295) against the omlx-prefixed
// kernels in rms_residual.metal.
class RmsNormResidualPrimitive : public Primitive {
 public:
  RmsNormResidualPrimitive(Stream stream, float eps)
      : Primitive(stream), eps_(eps) {}

  // Shared composed fallback: add -> rms_norm via regular ops. Used for CPU
  // streams and for GPU inputs whose realized layout is not dense (lazy
  // array flags are unreliable pre-eval, so the final layout check happens
  // here on the realized inputs).
  void eval_composed(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) {
    auto& s = stream();
    auto summed = add(inputs[0], inputs[1], s);
    auto xf = astype(summed, float32, s);
    auto out = multiply(
        xf,
        rsqrt(
            add(
                mean(square(xf, s), -1, /* keepdims */ true, s),
                array(eps_, float32),
                s),
            s),
        s);
    out = astype(out, summed.dtype(), s);
    out = multiply(out, inputs[2], s);
    outputs[0].copy_shared_buffer(out);
    outputs[1].copy_shared_buffer(summed);
  }

  void eval_cpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    eval_composed(inputs, outputs);
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);

    auto& out = outputs[0];
    auto& out_sum = outputs[1];

    // The kernel indexes rows densely. Layout cannot be vetted reliably at
    // graph-construction time (flags on lazy arrays do not reflect the
    // realized view), and neither a nested eval nor a contiguous copy is
    // legal inside eval_gpu, so a non-dense realized input is a hard error
    // with a clear message. Call sites must pass dense rows (hidden states
    // straight out of matmul/attention/add always are); anything else must
    // use the composed fallback in fast.py.
    auto dense_rows = [](const array& a) {
      return a.flags().row_contiguous && a.strides(-1) == 1;
    };
    if (!dense_rows(inputs[0]) || !dense_rows(inputs[1])) {
      throw std::runtime_error(
          "[omlx_decode_fast.rms_norm_residual] realized input is not "
          "row-contiguous with a unit-stride last axis; use the composed "
          "fallback (fast._composed_rms_norm_residual) for strided views.");
    }

    // Multi-output primitive outputs arrive unallocated; each gets a fresh
    // dense buffer matching its input's layout.
    auto set_output = [](const array& a, array& out) {
      out.set_data(
          allocator::malloc(a.data_size() * a.itemsize()),
          a.data_size(),
          a.strides(),
          a.flags());
      return a;
    };
    const array& x = set_output(inputs[0], out);
    const array& r = set_output(inputs[1], out_sum);
    const array& w = inputs[2];

    auto axis_size = static_cast<uint32_t>(x.shape().back());
    int n_rows = x.data_size() / axis_size;

    const int simd_size = 32;
    const int looped_limit = 4096; // RMS_LOOPED_LIMIT
    int n_reads = 4; // RMS_N_READS
    std::string op_name = "omlx_rms_residual";
    // NOTE: upstream #4295 used a 16-read variant for axis_size <= 3072, but
    // stock (non-residual) rms_norm always uses N_READS=4 there. The 16-read
    // tiling changes the fp32 sum-of-squares accumulation order, which breaks
    // bit-exactness with the composed add -> rms_norm path (observed: greedy
    // decode divergence on Qwen3-30B-A3B at hidden=2048). omlx always uses
    // N_READS=4 so fused output is bit-identical to the composed path.
    if (axis_size > looped_limit) {
      op_name += "_looped";
    }

    std::string kname;
    concatenate(kname, op_name, rms_type_name(x.dtype()));

    auto lib = d.get_library("omlx_decode_fast_kernels", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);

    auto& compute_encoder = metal::get_command_encoder(s);
    MTL::Size grid_dims, group_dims;
    if (axis_size <= looped_limit) {
      size_t threadgroup_needed = (axis_size + n_reads - 1) / n_reads;
      size_t simds_needed = (threadgroup_needed + simd_size - 1) / simd_size;
      size_t threadgroup_size = simd_size * simds_needed;
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    } else {
      size_t threadgroup_size = kernel->maxTotalThreadsPerThreadgroup();
      size_t n_threads = n_rows * threadgroup_size;
      grid_dims = MTL::Size(n_threads, 1, 1);
      group_dims = MTL::Size(threadgroup_size, 1, 1);
    }

    uint32_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_input_array(r, 1);
    compute_encoder.set_input_array(w, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_output_array(out_sum, 4);
    compute_encoder.set_bytes(eps_, 5);
    compute_encoder.set_bytes(axis_size, 6);
    compute_encoder.set_bytes(w_stride, 7);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  DEFINE_NAME(OMLXRmsNormResidual)

  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const RmsNormResidualPrimitive&>(other);
    return eps_ == rhs.eps_;
  }

 private:
  float eps_;
};

} // namespace

bool rms_norm_residual_supported(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& residual,
    mx::StreamOrDevice s_) {
  auto s = to_stream(s_);
  if (s.device == Device::cpu) {
    return false;
  }
  if (x.ndim() < 1 || x.shape() != residual.shape()) {
    return false;
  }
  if (weight.ndim() != 1 || weight.size() != x.shape(-1)) {
    return false;
  }
  // NOTE: layout (contiguity) is intentionally not checked here — flags on
  // lazy arrays are unreliable pre-eval. The primitive re-checks the
  // realized inputs in eval_gpu and composes a fallback for odd layouts.
  auto t = result_type(x, residual, weight);
  return t == float32 || t == float16 || t == bfloat16;
}

std::vector<mx::array> rms_norm_residual(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& residual,
    float eps,
    mx::StreamOrDevice s_) {
  if (x.ndim() < 1) {
    throw std::invalid_argument(
        "[omlx_decode_fast.rms_norm_residual] x must have at least 1 "
        "dimension.");
  }
  if (residual.shape() != x.shape()) {
    std::ostringstream msg;
    msg << "[omlx_decode_fast.rms_norm_residual] residual shape "
        << residual.shape() << " does not match x shape " << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight.ndim() != 1 || weight.size() != x.shape(-1)) {
    std::ostringstream msg;
    msg << "[omlx_decode_fast.rms_norm_residual] weight must be 1-D with "
        << x.shape(-1) << " elements.";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);
  auto out_type = result_type(x, residual, weight);
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[omlx_decode_fast.rms_norm_residual] unsupported type "
        << out_type << ".";
    throw std::invalid_argument(msg.str());
  }

  std::vector<array> inputs = {
      astype(x, out_type, s),
      astype(residual, out_type, s),
      astype(weight, out_type, s)};
  return array::make_arrays(
      {x.shape(), x.shape()},
      {out_type, out_type},
      std::make_shared<RmsNormResidualPrimitive>(s, eps),
      std::move(inputs));
}

} // namespace omlx::decode_fast_kernels
