#include "dspark_qmv.h"

#include <dlfcn.h>
#include <filesystem>
#include <sstream>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"

namespace omlx::glm_kernels {

namespace {

using namespace mlx::core;

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to locate DSpark QMV kernel binary.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

struct DSparkQMVParams {
  int K;
  int N;
};

class DSparkExactMXFP8QMVPairPrimitive : public Primitive {
 public:
  explicit DSparkExactMXFP8QMVPairPrimitive(Stream stream) : Primitive(stream) {}

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("DSpark exact MXFP8 QMV pair has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& stream = this->stream();
    auto& device = metal::device(stream.device);
    const auto& input = inputs[0];
    const auto& weight_a = inputs[1];
    const auto& scales_a = inputs[2];
    const auto& weight_b = inputs[3];
    const auto& scales_b = inputs[4];
    auto& output = outputs[0];

    const int rows = input.shape(0);
    const int K = input.shape(1);
    const int N = scales_a.shape(0);
    output.set_data(allocator::malloc(output.nbytes()));

    std::ostringstream name;
    name << "omlx_dspark_exact_mxfp8_qmv_pair_" << type_to_name(input)
         << "_m" << rows;
    auto library = device.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = device.get_kernel(name.str(), library);
    auto& encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(input, 0);
    encoder.set_input_array(weight_a, 1);
    encoder.set_input_array(scales_a, 2);
    encoder.set_input_array(weight_b, 3);
    encoder.set_input_array(scales_b, 4);
    encoder.set_output_array(output, 5);
    DSparkQMVParams params{K, N};
    encoder.set_bytes(params, 6);
    encoder.dispatch_threadgroups(
        MTL::Size(1, N / 8, 2), MTL::Size(32, 2, 1));
  }

  DEFINE_NAME(OMLXDSparkExactMXFP8QMVPair)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& /* other */) const override {
    return true;
  }
  auto state() const {
    return std::make_tuple(nullptr);
  }
};

} // namespace

array dspark_exact_mxfp8_qmv_pair(
    const array& input,
    const array& weight_a,
    const array& scales_a,
    const array& weight_b,
    const array& scales_b,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  if (!metal::is_available() || stream.device == Device::cpu) {
    throw std::invalid_argument("DSpark exact MXFP8 QMV pair requires Metal.");
  }
  if (input.ndim() != 2 || input.shape(0) < 2 || input.shape(0) > 6 ||
      (input.dtype() != float16 && input.dtype() != bfloat16) ||
      !input.flags().row_contiguous || weight_a.ndim() != 2 ||
      weight_b.ndim() != 2 || scales_a.ndim() != 2 || scales_b.ndim() != 2 ||
      weight_a.dtype() != uint32 || weight_b.dtype() != uint32 ||
      scales_a.dtype() != uint8 || scales_b.dtype() != uint8 ||
      weight_a.shape() != weight_b.shape() ||
      scales_a.shape() != scales_b.shape() ||
      !weight_a.flags().row_contiguous || !weight_b.flags().row_contiguous ||
      !scales_a.flags().row_contiguous || !scales_b.flags().row_contiguous) {
    throw std::invalid_argument("Unsupported DSpark exact MXFP8 QMV pair layout.");
  }

  const int K = input.shape(1);
  const int N = scales_a.shape(0);
  if (K % 512 != 0 || N % 8 != 0 || weight_a.shape(0) != N ||
      weight_a.shape(1) * 4 != K || scales_a.shape(1) * 32 != K) {
    throw std::invalid_argument("Invalid DSpark exact MXFP8 QMV pair shape.");
  }

  Shape shape{2, input.shape(0), N};
  return array(
      std::move(shape),
      input.dtype(),
      std::make_shared<DSparkExactMXFP8QMVPairPrimitive>(stream),
      std::vector<array>{input, weight_a, scales_a, weight_b, scales_b});
}

} // namespace omlx::glm_kernels
