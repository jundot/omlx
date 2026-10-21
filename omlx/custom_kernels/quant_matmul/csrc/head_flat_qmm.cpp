#include "head_flat_qmm.h"
#include "omlx/binary_dir.h"

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <string>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::quant_matmul {

namespace {

using namespace mlx::core;

bool row_contiguous(const array& arr) {
  return arr.flags().row_contiguous && arr.strides(-1) == 1;
}

std::string metal_type_name(Dtype dtype) {
  if (dtype == float16) {
    return "float16_t";
  }
  if (dtype == bfloat16) {
    return "bfloat16_t";
  }
  std::ostringstream msg;
  msg << "Unsupported GLM fused kernel dtype: " << dtype << ".";
  throw std::invalid_argument(msg.str());
}

class Q8HeadFlatQmmPrimitive : public Primitive {
 public:
  explicit Q8HeadFlatQmmPrimitive(Stream stream) : Primitive(stream) {}

  static bool unsupported(
      const array& x,
      const array& weight,
      const array& scales,
      const array& biases,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (x.dtype() != float16 && x.dtype() != bfloat16) {
      return true;
    }
    if (weight.dtype() != uint32 || scales.dtype() != x.dtype() ||
        biases.dtype() != x.dtype()) {
      return true;
    }
    if (x.ndim() != 4 || weight.ndim() != 3 || scales.ndim() != 3 ||
        biases.ndim() != 3) {
      return true;
    }
    if (!row_contiguous(x) || !row_contiguous(weight) ||
        !row_contiguous(scales) || !row_contiguous(biases)) {
      return true;
    }

    constexpr int bits = 8;
    constexpr int group_size = 64;
    constexpr int pack_factor = 32 / bits;
    const int H = x.shape(1);
    const int K = x.shape(3);
    const int N = weight.shape(1);
    if (H != 64 || K != 512 || N != 256) {
      return true;
    }
    if (weight.shape(0) != H || scales.shape(0) != H ||
        biases.shape(0) != H || scales.shape(1) != N ||
        biases.shape(1) != N || weight.shape(2) * pack_factor != K ||
        scales.shape(2) != K / group_size ||
        biases.shape(2) != K / group_size) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("Q8HeadFlatQmmPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scales = inputs[2];
    const auto& biases = inputs[3];

    out.set_data(allocator::malloc(out.nbytes()));

    constexpr int group_size = 64;
    constexpr int bits = 8;
    constexpr int bm = 32;
    constexpr int bn = 32;

    const int B = x.shape(0);
    const int H = x.shape(1);
    const int M = x.shape(2);
    const int K = x.shape(3);
    const int N = weight.shape(1);

    std::string kname;
    concatenate(
        kname,
        "affine_qmm_t_head_flat_",
        metal_type_name(x.dtype()),
        "_gs_",
        group_size,
        "_b_",
        bits,
        "_alN_true");

    auto lib = d.get_library("omlx_quant_matmul_kernels", binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(weight, 0);
    compute_encoder.set_input_array(scales, 1);
    compute_encoder.set_input_array(biases, 2);
    compute_encoder.set_input_array(x, 3);
    compute_encoder.set_output_array(out, 4);
    compute_encoder.set_bytes(K, 5);
    compute_encoder.set_bytes(N, 6);
    compute_encoder.set_bytes(M, 7);
    compute_encoder.set_bytes(H, 8);

    MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B * H);
    MTL::Size group_dims(32, 2, 2);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(Q8HeadFlatQmmPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& /* other */) const override {
    return true;
  }
  auto state() const {
    return std::make_tuple(nullptr);
  }

};

} // namespace

array q8_head_flat_qmm(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    StreamOrDevice s /* = {} */) {
  if (x.ndim() != 4 || weight.ndim() != 3 || scales.ndim() != 3 ||
      biases.ndim() != 3) {
    std::ostringstream msg;
    msg << "[quant_matmul.q8_head_flat_qmm] expected x rank 4 and "
        << "quantized weights rank 3, got " << x.shape() << ", "
        << weight.shape() << ", " << scales.shape() << ", " << biases.shape()
        << ".";
    throw std::invalid_argument(msg.str());
  }

  const int B = x.shape(0);
  const int H = x.shape(1);
  const int L = x.shape(2);
  constexpr int bits = 8;
  constexpr int group_size = 64;
  constexpr int pack_factor = 32 / bits;
  const int V = weight.shape(1);
  const int K = weight.shape(2) * pack_factor;
  if (H != weight.shape(0) || H != scales.shape(0) ||
      H != biases.shape(0) || V != scales.shape(1) ||
      V != biases.shape(1) || x.shape(3) != K ||
      scales.shape(2) != K / group_size ||
      biases.shape(2) != K / group_size) {
    std::ostringstream msg;
    msg << "[quant_matmul.q8_head_flat_qmm] incompatible shapes: "
        << x.shape() << ", " << weight.shape() << ", " << scales.shape()
        << ", " << biases.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[quant_matmul.q8_head_flat_qmm] expected float16 or "
        << "bfloat16 input, got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight.dtype() != uint32 || scales.dtype() != x.dtype() ||
      biases.dtype() != x.dtype()) {
    std::ostringstream msg;
    msg << "[quant_matmul.q8_head_flat_qmm] expected uint32 weight and "
        << "scale/bias dtype " << x.dtype() << ", got " << weight.dtype()
        << ", " << scales.dtype() << ", " << biases.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {x, weight, scales, biases};
  if (Q8HeadFlatQmmPrimitive::unsupported(x, weight, scales, biases, stream)) {
    throw std::invalid_argument(
        "[quant_matmul.q8_head_flat_qmm] unsupported M3 GLM shape.");
  }

  Shape out_shape{B, L, H * V};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<Q8HeadFlatQmmPrimitive>(stream),
      std::move(inputs));
}

} // namespace omlx::quant_matmul
