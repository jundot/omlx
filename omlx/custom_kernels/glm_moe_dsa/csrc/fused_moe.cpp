#include "fused_moe.h"

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

namespace omlx::glm_kernels {

namespace {

using namespace mlx::core;

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get omlx_glm_kernels binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

bool row_contiguous(const array& arr) {
  return arr.flags().row_contiguous && arr.strides(-1) == 1;
}

std::string glm_type_name(Dtype dtype) {
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

class GlmDsaQ8VupFlatPrimitive : public Primitive {
 public:
  explicit GlmDsaQ8VupFlatPrimitive(Stream stream) : Primitive(stream) {}

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
    throw std::runtime_error("GlmDsaQ8VupFlatPrimitive has no CPU path.");
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
        glm_type_name(x.dtype()),
        "_gs_",
        group_size,
        "_b_",
        bits,
        "_alN_true");

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
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

  DEFINE_NAME(GlmDsaQ8VupFlatPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& /* other */) const override {
    return true;
  }
  auto state() const {
    return std::make_tuple(nullptr);
  }

};

class GlmMoeWeightedSumPrimitive : public Primitive {
 public:
  explicit GlmMoeWeightedSumPrimitive(Stream stream) : Primitive(stream) {}

  static bool unsupported(
      const array& x_sorted,
      const array& inv_order,
      const array& scores,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (x_sorted.dtype() != float16 && x_sorted.dtype() != bfloat16) {
      return true;
    }
    if (scores.dtype() != float32 || inv_order.dtype() != uint32) {
      return true;
    }
    if (x_sorted.ndim() != 3 || x_sorted.shape(-2) != 1 ||
        scores.ndim() < 2 || inv_order.ndim() != 1) {
      return true;
    }
    if (!row_contiguous(x_sorted) || !row_contiguous(inv_order) ||
        !row_contiguous(scores)) {
      return true;
    }
    if (scores.shape(-1) != 8 || x_sorted.shape(0) != scores.size() ||
        inv_order.size() != scores.size()) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("GlmMoeWeightedSumPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& x_sorted = inputs[0];
    const auto& inv_order = inputs[1];
    const auto& scores = inputs[2];

    out.set_data(allocator::malloc(out.nbytes()));

    const int topk = scores.shape(-1);
    const int tokens = scores.size() / topk;
    const int D = x_sorted.shape(-1);

    const bool use_tiled = true;
    const int tiled_threads = 256;
    const int vec = (D % 4 == 0) ? 4 : 1;

    std::string kname;
    if (use_tiled) {
      concatenate(
          kname,
          "moe_weighted_sum_tiled_",
          glm_type_name(x_sorted.dtype()),
          "_score_float_topk_",
          topk,
          "_t_",
          tiled_threads);
    } else {
      concatenate(
          kname,
          vec == 1 ? "moe_weighted_sum_" : "moe_weighted_sum_vec",
          vec == 1 ? "" : std::to_string(vec),
          vec == 1 ? "" : "_",
          glm_type_name(x_sorted.dtype()),
          "_score_float_topk_",
          topk);
    }

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x_sorted, 0);
    compute_encoder.set_input_array(inv_order, 1);
    compute_encoder.set_input_array(scores, 2);
    compute_encoder.set_output_array(out, 3);
    compute_encoder.set_bytes(tokens, 4);
    compute_encoder.set_bytes(D, 5);

    const int threads = use_tiled ? tiled_threads : 256;
    const int total = vec == 1 ? tokens * D : tokens * ((D + vec - 1) / vec);
    MTL::Size group_dims(threads, 1, 1);
    MTL::Size grid_dims(
        use_tiled ? tokens : (total + threads - 1) / threads, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(GlmMoeWeightedSumPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& /* other */) const override {
    return true;
  }
  auto state() const {
    return std::make_tuple(nullptr);
  }

};

} // namespace

array glm_dsa_q8_vup_flat(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    StreamOrDevice s /* = {} */) {
  if (x.ndim() != 4 || weight.ndim() != 3 || scales.ndim() != 3 ||
      biases.ndim() != 3) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_dsa_q8_vup_flat] expected x rank 4 and "
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
    msg << "[omlx_glm_kernels.glm_dsa_q8_vup_flat] incompatible shapes: "
        << x.shape() << ", " << weight.shape() << ", " << scales.shape()
        << ", " << biases.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_dsa_q8_vup_flat] expected float16 or "
        << "bfloat16 input, got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight.dtype() != uint32 || scales.dtype() != x.dtype() ||
      biases.dtype() != x.dtype()) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_dsa_q8_vup_flat] expected uint32 weight and "
        << "scale/bias dtype " << x.dtype() << ", got " << weight.dtype()
        << ", " << scales.dtype() << ", " << biases.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {x, weight, scales, biases};
  if (GlmDsaQ8VupFlatPrimitive::unsupported(x, weight, scales, biases, stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.glm_dsa_q8_vup_flat] unsupported M3 GLM shape.");
  }

  Shape out_shape{B, L, H * V};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<GlmDsaQ8VupFlatPrimitive>(stream),
      std::move(inputs));
}

array glm_moe_weighted_sum(
    const array& x_sorted,
    const array& inv_order,
    const array& scores,
    StreamOrDevice s /* = {} */) {
  if (x_sorted.ndim() != 3 || x_sorted.shape(-2) != 1) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_moe_weighted_sum] expected x_sorted shape "
        << "[N, 1, D], got " << x_sorted.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (scores.ndim() < 2) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_moe_weighted_sum] expected scores rank >= 2, "
        << "got " << scores.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (inv_order.ndim() != 1 || inv_order.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_moe_weighted_sum] expected uint32 inv_order "
        << "rank 1, got " << inv_order.shape() << " dtype "
        << inv_order.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  const int topk = scores.shape(-1);
  const int64_t routed_rows = scores.size();
  const int D = x_sorted.shape(-1);
  if (x_sorted.shape(0) != routed_rows || inv_order.size() != routed_rows) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_moe_weighted_sum] incompatible shapes: "
        << x_sorted.shape() << ", " << inv_order.shape() << ", "
        << scores.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (topk <= 0 || D <= 0) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_moe_weighted_sum] invalid topk or hidden "
        << "dim: topk=" << topk << ", D=" << D << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(x_sorted.dtype(), floating)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_moe_weighted_sum] expected floating "
        << "x_sorted, got " << x_sorted.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {x_sorted, inv_order, scores};
  Shape out_shape = scores.shape();
  out_shape.pop_back();
  out_shape.push_back(D);
  if (GlmMoeWeightedSumPrimitive::unsupported(
          x_sorted, inv_order, scores, stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.glm_moe_weighted_sum] unsupported M3 GLM shape.");
  }
  return array(
      std::move(out_shape),
      x_sorted.dtype(),
      std::make_shared<GlmMoeWeightedSumPrimitive>(stream),
      std::move(inputs));
}

} // namespace omlx::glm_kernels
