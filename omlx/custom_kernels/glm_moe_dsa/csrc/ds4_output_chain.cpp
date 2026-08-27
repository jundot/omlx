#include "ds4_output_chain.h"

#include <dlfcn.h>

#include <filesystem>
#include <sstream>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"

namespace omlx::glm_kernels {

namespace {

using namespace mlx::core;

constexpr int kGroups = 8;
constexpr int kORank = 1024;
constexpr int kHidden = 4096;
constexpr int kOBInput = kGroups * kORank;
constexpr int kGroupSize = 32;
constexpr int kValuesPerU32 = 4;
constexpr int kBK = 32;
constexpr int kBN = 32;

struct ChainTile {
  int o_a_bm;
  int o_b_bm;
};

ChainTile chain_tile(int variant) {
  switch (variant) {
  case 0:
    return {32, 32};
  case 1:
    return {64, 64};
  case 2:
    return {64, 32};
  case 3:
    return {32, 64};
  default: {
    std::ostringstream msg;
    msg << "Unsupported DS4 output-chain variant " << variant << ".";
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

int o_a_bm(int variant) {
  if (variant == 0) {
    return 32;
  }
  if (variant == 1) {
    return 64;
  }
  std::ostringstream msg;
  msg << "Unsupported DS4 interleaved O-A variant " << variant << ".";
  throw std::invalid_argument(msg.str());
}

bool unsupported_o_a(const array &x, const array &o_a_weight,
                     const array &o_a_scales, int variant, Stream stream) {
  (void)o_a_bm(variant);
  if (stream.device == Device::cpu || x.ndim() != 4 || o_a_weight.ndim() != 3 ||
      o_a_scales.ndim() != 3 || !row_contiguous(x) ||
      !row_contiguous(o_a_weight) || !row_contiguous(o_a_scales) ||
      x.dtype() != bfloat16 || o_a_weight.dtype() != uint32 ||
      o_a_scales.dtype() != uint8) {
    return true;
  }
  const int K = x.shape(3);
  const int tokens = x.shape(2);
  return (K != 1536 && K != 2048 && K != 2560 && K != 4096) ||
         (tokens != 1024 && tokens != 2048) ||
         x.shape() != Shape{1, kGroups, tokens, K} ||
         o_a_weight.shape() != Shape{kGroups, kORank, K / kValuesPerU32} ||
         o_a_scales.shape() != Shape{kGroups, kORank, K / kGroupSize};
}

class DS4OutputOAInterleavedPrimitive : public Primitive {
public:
  DS4OutputOAInterleavedPrimitive(Stream stream, int variant)
      : Primitive(stream), variant_(variant) {
    (void)o_a_bm(variant_);
  }

  void eval_cpu(const std::vector<array> & /* inputs */,
                std::vector<array> & /* outputs */) override {
    throw std::runtime_error(
        "DS4OutputOAInterleavedPrimitive has no CPU path.");
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

    const int K = x.shape(3);
    const int tokens = x.shape(2);
    const int bm = o_a_bm(variant_);
    auto library = device.get_library("omlx_glm_kernels", current_binary_dir());
    std::string kernel_name;
    concatenate(kernel_name, "ds4_output_oa_interleaved_bfloat16_t_bm", bm,
                "_bk", kBK, "_bn", kBN);
    auto kernel = device.get_kernel(kernel_name, library);
    auto &encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(weight, 0);
    encoder.set_input_array(scales, 1);
    encoder.set_input_array(x, 2);
    encoder.set_output_array(output, 3);
    encoder.set_bytes(K, 4);
    encoder.set_bytes(kORank, 5);
    encoder.set_bytes(tokens, 6);
    encoder.set_bytes(kGroups, 7);
    encoder.dispatch_threadgroups(
        MTL::Size(kORank / kBN, tokens / bm, kGroups), MTL::Size(32, 2, 2));
  }

  DEFINE_NAME(DS4OutputOAInterleavedPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs =
        static_cast<const DS4OutputOAInterleavedPrimitive &>(other);
    return variant_ == rhs.variant_;
  }
  auto state() const { return std::make_tuple(nullptr, variant_); }

private:
  int variant_;
};

class DS4OutputProjectionChainPrimitive : public Primitive {
public:
  DS4OutputProjectionChainPrimitive(Stream stream, int variant)
      : Primitive(stream), variant_(variant) {
    (void)chain_tile(variant_);
  }

  static bool unsupported(const array &x, const array &o_a_weight,
                          const array &o_a_scales, const array &o_b_weight,
                          const array &o_b_scales, int variant, Stream stream) {
    (void)chain_tile(variant);
    if (stream.device == Device::cpu || x.ndim() != 4 ||
        o_a_weight.ndim() != 3 || o_a_scales.ndim() != 3 ||
        o_b_weight.ndim() != 2 || o_b_scales.ndim() != 2 ||
        !row_contiguous(x) || !row_contiguous(o_a_weight) ||
        !row_contiguous(o_a_scales) || !row_contiguous(o_b_weight) ||
        !row_contiguous(o_b_scales)) {
      return true;
    }
    if (x.dtype() != bfloat16 || o_a_weight.dtype() != uint32 ||
        o_a_scales.dtype() != uint8 || o_b_weight.dtype() != uint32 ||
        o_b_scales.dtype() != uint8) {
      return true;
    }
    const int K = x.shape(3);
    const int tokens = x.shape(2);
    if (K != 1536 && K != 2048 && K != 2560 && K != 4096) {
      return true;
    }
    return (tokens != 1024 && tokens != 2048) ||
           x.shape() != Shape{1, kGroups, tokens, K} ||
           o_a_weight.shape() != Shape{kGroups, kORank, K / kValuesPerU32} ||
           o_a_scales.shape() != Shape{kGroups, kORank, K / kGroupSize} ||
           o_b_weight.shape() != Shape{kHidden, kOBInput / kValuesPerU32} ||
           o_b_scales.shape() != Shape{kHidden, kOBInput / kGroupSize};
  }

  void eval_cpu(const std::vector<array> & /* inputs */,
                std::vector<array> & /* outputs */) override {
    throw std::runtime_error(
        "DS4OutputProjectionChainPrimitive has no CPU path.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    const auto &x = inputs[0];
    const auto &o_a_weight = inputs[1];
    const auto &o_a_scales = inputs[2];
    const auto &o_b_weight = inputs[3];
    const auto &o_b_scales = inputs[4];
    auto &output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));

    // This allocation is the mandatory BF16 rounding boundary, not a second
    // transposed copy.  The command encoder owns it until both dispatches
    // finish, so no graph-visible O-A tensor survives the primitive.
    const int tokens = x.shape(2);
    array o_mid({1, tokens, kOBInput}, bfloat16, nullptr, {});
    o_mid.set_data(allocator::malloc(o_mid.nbytes()));

    const auto tile = chain_tile(variant_);
    const int o_a_k = x.shape(3);
    const int o_a_n = kORank;
    const int o_a_m = tokens;
    const int o_b_k = kOBInput;
    const int o_b_n = kHidden;
    const int o_b_m = tokens;

    auto library = device.get_library("omlx_glm_kernels", current_binary_dir());
    auto &encoder = metal::get_command_encoder(stream);
    encoder.add_temporary(o_mid);

    std::string kernel_name;
    concatenate(kernel_name, "ds4_output_oa_interleaved_bfloat16_t_bm",
                tile.o_a_bm, "_bk", kBK, "_bn", kBN);
    auto kernel = device.get_kernel(kernel_name, library);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(o_a_weight, 0);
    encoder.set_input_array(o_a_scales, 1);
    encoder.set_input_array(x, 2);
    encoder.set_output_array(o_mid, 3);
    encoder.set_bytes(o_a_k, 4);
    encoder.set_bytes(o_a_n, 5);
    encoder.set_bytes(o_a_m, 6);
    encoder.set_bytes(kGroups, 7);
    encoder.dispatch_threadgroups(
        MTL::Size(o_a_n / kBN, o_a_m / tile.o_a_bm, kGroups),
        MTL::Size(32, 2, 2));

    kernel_name.clear();
    concatenate(kernel_name, "ds4_projection_mxfp8_qmm_t_bfloat16_t_bm",
                tile.o_b_bm, "_bk", kBK, "_bn", kBN);
    kernel = device.get_kernel(kernel_name, library);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(o_b_weight, 0);
    encoder.set_input_array(o_b_scales, 1);
    encoder.set_input_array(o_mid, 2);
    encoder.set_output_array(output, 3);
    encoder.set_bytes(o_b_k, 4);
    encoder.set_bytes(o_b_n, 5);
    encoder.set_bytes(o_b_m, 6);
    encoder.dispatch_threadgroups(
        MTL::Size(o_b_n / kBN, o_b_m / tile.o_b_bm, 1), MTL::Size(32, 2, 2));
  }

  DEFINE_NAME(DS4OutputProjectionChainPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs =
        static_cast<const DS4OutputProjectionChainPrimitive &>(other);
    return variant_ == rhs.variant_;
  }
  auto state() const { return std::make_tuple(nullptr, variant_); }

private:
  int variant_;
};

} // namespace

mx::array ds4_output_oa_interleaved(const mx::array &x,
                                    const mx::array &o_a_weight,
                                    const mx::array &o_a_scales, int variant,
                                    mx::StreamOrDevice s) {
  auto stream = to_stream(s);
  if (unsupported_o_a(x, o_a_weight, o_a_scales, variant, stream)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.ds4_output_oa_interleaved] unsupported shape; "
           "requires BF16 x [1,8,M,K], M in {1024,2048}, K in "
           "{1536,2048,2560,4096}, and MXFP8 "
           "O-A [8,1024,K/4]; got "
        << x.shape() << ", " << o_a_weight.shape() << ", " << o_a_scales.shape()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  std::vector<array> inputs = {x, o_a_weight, o_a_scales};
  return array(
      Shape{1, x.shape(2), kOBInput}, bfloat16,
      std::make_shared<DS4OutputOAInterleavedPrimitive>(stream, variant),
      std::move(inputs));
}

mx::array ds4_output_projection_chain(const mx::array &x,
                                      const mx::array &o_a_weight,
                                      const mx::array &o_a_scales,
                                      const mx::array &o_b_weight,
                                      const mx::array &o_b_scales, int variant,
                                      mx::StreamOrDevice s) {
  auto stream = to_stream(s);
  if (DS4OutputProjectionChainPrimitive::unsupported(
          x, o_a_weight, o_a_scales, o_b_weight, o_b_scales, variant, stream)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.ds4_output_projection_chain] unsupported "
           "shape; requires BF16 x [1,8,M,K], M in {1024,2048}, K in "
           "{1536,2048,2560,4096}, "
           "MXFP8 O-A [8,1024,K/4], and MXFP8 O-B [4096,2048]; got "
        << x.shape() << ", " << o_a_weight.shape() << ", " << o_a_scales.shape()
        << ", " << o_b_weight.shape() << ", " << o_b_scales.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  std::vector<array> inputs = {x, o_a_weight, o_a_scales, o_b_weight,
                               o_b_scales};
  return array(
      Shape{1, x.shape(2), kHidden}, bfloat16,
      std::make_shared<DS4OutputProjectionChainPrimitive>(stream, variant),
      std::move(inputs));
}

} // namespace omlx::glm_kernels
