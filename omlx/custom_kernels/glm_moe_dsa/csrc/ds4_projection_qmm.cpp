#include "ds4_projection_qmm.h"

#include <dlfcn.h>

#include <filesystem>
#include <sstream>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"

namespace omlx::glm_kernels {

namespace {

using namespace mlx::core;

constexpr const char *kNaxMetallibName = "omlx_glm_kernels_nax";

struct ProjectionTile {
  int bm;
  int bk;
  int bn;
};

struct NaxProjectionTile {
  int bm;
  int bk;
  int bn;
  int wm;
  int wn;
};

ProjectionTile projection_tile(int variant) {
  switch (variant) {
  case 0:
    return {32, 32, 32};
  case 1:
    return {64, 32, 32};
  case 2:
    return {128, 32, 32};
  case 3:
    return {32, 32, 64};
  case 4:
    return {64, 32, 64};
  case 5:
    return {128, 32, 64};
  case 6:
    return {32, 64, 64};
  case 7:
    return {64, 64, 64};
  case 8:
    return {128, 64, 64};
  case 9:
    return {64, 32, 128};
  default: {
    std::ostringstream msg;
    msg << "Unsupported DS4 projection MXFP8 variant " << variant << ".";
    throw std::invalid_argument(msg.str());
  }
  }
}

NaxProjectionTile nax_projection_tile(int variant) {
  switch (variant) {
  case 0:
    return {64, 64, 64, 2, 2};
  case 1:
    return {32, 64, 32, 1, 1};
  case 2:
    return {32, 64, 64, 1, 2};
  case 3:
    return {64, 64, 32, 2, 1};
  case 4:
    return {64, 64, 128, 2, 4};
  case 5:
    return {128, 64, 64, 4, 2};
  case 6:
    return {128, 64, 128, 4, 4};
  case 7:
    return {32, 64, 128, 1, 4};
  case 8:
    return {128, 64, 32, 4, 1};
  case 9:
    return {64, 32, 64, 2, 2};
  default: {
    std::ostringstream msg;
    msg << "Unsupported DS4 projection NAX variant " << variant << ".";
    throw std::invalid_argument(msg.str());
  }
  }
}

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void *>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get omlx_glm_kernels binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

bool row_contiguous(const array &value) {
  return value.flags().row_contiguous && value.strides(-1) == 1;
}

std::string type_name(Dtype dtype) {
  if (dtype == float16) {
    return "float16_t";
  }
  if (dtype == bfloat16) {
    return "bfloat16_t";
  }
  std::ostringstream msg;
  msg << "Unsupported DS4 projection dtype " << dtype << ".";
  throw std::invalid_argument(msg.str());
}

class DS4ProjectionMxfp8QmmPrimitive : public Primitive {
public:
  DS4ProjectionMxfp8QmmPrimitive(Stream stream, int variant, bool use_nax,
                                 int nax_variant)
      : Primitive(stream), variant_(variant), use_nax_(use_nax),
        nax_variant_(nax_variant) {
    (void)projection_tile(variant_);
    (void)nax_projection_tile(nax_variant_);
  }

  static bool unsupported(const array &x, const array &weight,
                          const array &scales, int variant, bool use_nax,
                          int nax_variant, Stream stream) {
    if (stream.device == Device::cpu || x.ndim() < 2 ||
        (weight.ndim() != 2 && weight.ndim() != 3) ||
        scales.ndim() != weight.ndim() || !row_contiguous(x) ||
        !row_contiguous(weight) || !row_contiguous(scales)) {
      return true;
    }
    if ((x.dtype() != float16 && x.dtype() != bfloat16) ||
        weight.dtype() != uint32 || scales.dtype() != uint8 ||
        (use_nax && x.dtype() != bfloat16)) {
      return true;
    }
    const auto classic = projection_tile(variant);
    const auto nax = nax_projection_tile(nax_variant);
    const int bm = use_nax ? nax.bm : classic.bm;
    const int bk = use_nax ? nax.bk : classic.bk;
    const int bn = use_nax ? nax.bn : classic.bn;
    const int K = x.shape(-1);
    const int B = weight.ndim() == 3 ? weight.shape(0) : 1;
    const int N = weight.shape(-2);
    const int64_t M = x.size() / B / K;
    return K <= 0 || N <= 0 || M <= 0 || K % bk != 0 || N % bn != 0 ||
           M % bm != 0 || weight.shape(-1) * 4 != K || scales.shape(-2) != N ||
           scales.shape(-1) != K / 32 ||
           (weight.ndim() == 3 && scales.shape(0) != B);
  }

  void eval_cpu(const std::vector<array> & /* inputs */,
                std::vector<array> & /* outputs */) override {
    throw std::runtime_error("DS4ProjectionMxfp8QmmPrimitive has no CPU path.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    const auto &x = inputs[0];
    const auto &weight = inputs[1];
    const auto &scales = inputs[2];
    auto &output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));

    const auto classic = projection_tile(variant_);
    const auto nax = nax_projection_tile(nax_variant_);
    const int bm = use_nax_ ? nax.bm : classic.bm;
    const int bk = use_nax_ ? nax.bk : classic.bk;
    const int bn = use_nax_ ? nax.bn : classic.bn;
    const int wm = use_nax_ ? nax.wm : 2;
    const int wn = use_nax_ ? nax.wn : 2;
    const int K = x.shape(-1);
    const int B = weight.ndim() == 3 ? weight.shape(0) : 1;
    const int N = weight.shape(-2);
    const int M = x.size() / B / K;

    std::string kernel_name;
    concatenate(kernel_name,
                use_nax_ ? "ds4_projection_mxfp8_qmm_t_nax_"
                         : "ds4_projection_mxfp8_qmm_t_",
                type_name(x.dtype()), "_bm", bm, "_bk", bk, "_bn", bn);
    if (use_nax_) {
      concatenate(kernel_name, "_wm", wm, "_wn", wn);
    }

    auto library = device.get_library(
        use_nax_ ? kNaxMetallibName : "omlx_glm_kernels", current_binary_dir());
    auto kernel = device.get_kernel(kernel_name, library);
    auto &encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(weight, 0);
    encoder.set_input_array(scales, 1);
    encoder.set_input_array(x, 2);
    encoder.set_output_array(output, 3);
    encoder.set_bytes(K, 4);
    encoder.set_bytes(N, 5);
    encoder.set_bytes(M, 6);
    encoder.dispatch_threadgroups(MTL::Size(N / bn, M / bm, B),
                                  MTL::Size(32, wn, wm));
  }

  DEFINE_NAME(DS4ProjectionMxfp8QmmPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs =
        static_cast<const DS4ProjectionMxfp8QmmPrimitive &>(other);
    return variant_ == rhs.variant_ && use_nax_ == rhs.use_nax_ &&
           nax_variant_ == rhs.nax_variant_;
  }
  auto state() const {
    return std::make_tuple(nullptr, variant_, use_nax_, nax_variant_);
  }

private:
  int variant_;
  bool use_nax_;
  int nax_variant_;
};

} // namespace

array ds4_projection_mxfp8_qmm(const array &x, const array &weight,
                               const array &scales, int variant, bool use_nax,
                               int nax_variant, StreamOrDevice s) {
  auto stream = to_stream(s);
  if (DS4ProjectionMxfp8QmmPrimitive::unsupported(
          x, weight, scales, variant, use_nax, nax_variant, stream)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.ds4_projection_mxfp8_qmm] unsupported shape; "
           "requires row-contiguous BF16/FP16 x [...,K], U32 weight [N,K/4], "
           "U8 scales [N,K/32], and a tile-divisible M/N/K; got "
        << x.shape() << ", " << weight.shape() << ", " << scales.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  Shape output_shape = x.shape();
  output_shape.back() = weight.shape(-2);
  std::vector<array> inputs = {x, weight, scales};
  return array(std::move(output_shape), x.dtype(),
               std::make_shared<DS4ProjectionMxfp8QmmPrimitive>(
                   stream, variant, use_nax, nax_variant),
               std::move(inputs));
}

bool ds4_projection_nax_kernels_built() {
  std::error_code error;
  return std::filesystem::exists(
      std::filesystem::path(current_binary_dir()) /
          (std::string(kNaxMetallibName) + ".metallib"),
      error);
}

bool ds4_projection_nax_device_available() {
  if (!metal::is_available()) {
    return false;
  }
  bool os_ok = false;
  if (__builtin_available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *)) {
    os_ok = true;
  }
  if (!os_ok) {
    return false;
  }
  auto &device = metal::device(Device::gpu);
  const auto &architecture = device.get_architecture();
  if (architecture.empty()) {
    return false;
  }
  const char suffix = architecture.back();
  return device.get_architecture_gen() >= (suffix == 'p' ? 18 : 17);
}

} // namespace omlx::glm_kernels
