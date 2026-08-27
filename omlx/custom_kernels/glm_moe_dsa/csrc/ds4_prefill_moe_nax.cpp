#include "ds4_prefill_moe_nax.h"

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
constexpr int kRoutes = 6144;
constexpr int kExperts = 256;
constexpr int kMaxBlocks = 448;
constexpr int kBlockRows = 32;
constexpr int kBlockCols = 64;
constexpr int kBlockDepth = 64;
constexpr int kWarpRows = 1;
constexpr int kWarpCols = 2;

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

bool supported_projection(int K, int N) {
  return (K == 4096 && (N == 1024 || N == 1280)) ||
         ((K == 1024 || K == 1280) && N == 4096);
}

class DS4Mxfp4GatherQmmBlocksNaxPrimitive : public Primitive {
public:
  explicit DS4Mxfp4GatherQmmBlocksNaxPrimitive(Stream stream)
      : Primitive(stream) {}

  static bool unsupported(const array &x, const array &weight,
                          const array &scales, const array &block_meta,
                          const array &block_count, Stream stream) {
    if (stream.device == Device::cpu || x.ndim() != 3 ||
        weight.ndim() != 3 || scales.ndim() != 3 ||
        block_meta.ndim() != 2 || block_count.ndim() != 1 ||
        !row_contiguous(x) || !row_contiguous(weight) ||
        !row_contiguous(scales) || !row_contiguous(block_meta) ||
        !row_contiguous(block_count)) {
      return true;
    }
    if (x.dtype() != bfloat16 || weight.dtype() != uint32 ||
        scales.dtype() != uint8 || block_meta.dtype() != int32 ||
        block_count.dtype() != int32) {
      return true;
    }
    if (x.shape(0) != kRoutes || x.shape(1) != 1 ||
        weight.shape(0) != kExperts || scales.shape(0) != kExperts ||
        block_meta.shape(0) != kMaxBlocks || block_meta.shape(1) != 3 ||
        block_count.shape(0) != 1) {
      return true;
    }
    const int K = x.shape(2);
    const int N = weight.shape(1);
    return !supported_projection(K, N) || K % kBlockDepth != 0 ||
           N % kBlockCols != 0 || weight.shape(2) * 8 != K ||
           scales.shape(1) != N || scales.shape(2) * 32 != K;
  }

  void eval_cpu(const std::vector<array> & /* inputs */,
                std::vector<array> & /* outputs */) override {
    throw std::runtime_error(
        "DS4Mxfp4GatherQmmBlocksNaxPrimitive has no CPU path.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    const auto &x = inputs[0];
    const auto &weight = inputs[1];
    const auto &scales = inputs[2];
    const auto &block_meta = inputs[3];
    const auto &block_count = inputs[4];
    auto &output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));

    const int K = x.shape(2);
    const int N = weight.shape(1);
    std::string kernel_name;
    concatenate(kernel_name,
                "ds4_mxfp4_gather_qmm_blocks_nax_bfloat16_t_bm",
                kBlockRows, "_bn", kBlockCols, "_bk", kBlockDepth, "_wm",
                kWarpRows, "_wn", kWarpCols);

    auto library = device.get_library(kNaxMetallibName, current_binary_dir());
    auto kernel = device.get_kernel(kernel_name, library);
    auto &encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(weight, 1);
    encoder.set_input_array(scales, 2);
    encoder.set_input_array(block_meta, 3);
    encoder.set_input_array(block_count, 4);
    encoder.set_output_array(output, 5);
    encoder.set_bytes(kMaxBlocks, 6);
    encoder.set_bytes(K, 7);
    encoder.set_bytes(N, 8);
    encoder.dispatch_threadgroups(
        MTL::Size(N / kBlockCols, kMaxBlocks, 1),
        MTL::Size(32, kWarpCols, kWarpRows));
  }

  DEFINE_NAME(DS4Mxfp4GatherQmmBlocksNaxPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive & /* other */) const override {
    return true;
  }
  auto state() const { return std::make_tuple(nullptr); }
};

class DS4Mxfp4GatherQmmPairBlocksNaxPrimitive : public Primitive {
public:
  explicit DS4Mxfp4GatherQmmPairBlocksNaxPrimitive(Stream stream)
      : Primitive(stream) {}

  static bool unsupported(
      const array &x, const array &weight0, const array &scales0,
      const array &weight1, const array &scales1, const array &block_meta,
      const array &block_count, Stream stream) {
    return DS4Mxfp4GatherQmmBlocksNaxPrimitive::unsupported(
               x, weight0, scales0, block_meta, block_count, stream) ||
        weight1.shape() != weight0.shape() ||
        scales1.shape() != scales0.shape() || weight1.dtype() != uint32 ||
        scales1.dtype() != uint8 || !row_contiguous(weight1) ||
        !row_contiguous(scales1) || x.shape(2) != 4096 ||
        (weight0.shape(1) != 1024 && weight0.shape(1) != 1280);
  }

  void eval_cpu(const std::vector<array> & /* inputs */,
                std::vector<array> & /* outputs */) override {
    throw std::runtime_error(
        "DS4Mxfp4GatherQmmPairBlocksNaxPrimitive has no CPU path.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    const auto &x = inputs[0];
    const auto &weight0 = inputs[1];
    const auto &scales0 = inputs[2];
    const auto &weight1 = inputs[3];
    const auto &scales1 = inputs[4];
    const auto &block_meta = inputs[5];
    const auto &block_count = inputs[6];
    auto &output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));

    const int K = x.shape(2);
    const int N = weight0.shape(1);
    std::string kernel_name;
    concatenate(
        kernel_name,
        "ds4_mxfp4_gather_qmm_pair_blocks_nax_bfloat16_t_bm",
        kBlockRows, "_bn", kBlockCols, "_bk", kBlockDepth, "_wm",
        kWarpRows, "_wn", kWarpCols);

    auto library = device.get_library(kNaxMetallibName, current_binary_dir());
    auto kernel = device.get_kernel(kernel_name, library);
    auto &encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(weight0, 1);
    encoder.set_input_array(scales0, 2);
    encoder.set_input_array(weight1, 3);
    encoder.set_input_array(scales1, 4);
    encoder.set_input_array(block_meta, 5);
    encoder.set_input_array(block_count, 6);
    encoder.set_output_array(output, 7);
    encoder.set_bytes(kMaxBlocks, 8);
    encoder.set_bytes(K, 9);
    encoder.set_bytes(N, 10);
    encoder.dispatch_threadgroups(
        MTL::Size(N / kBlockCols, kMaxBlocks, 1),
        MTL::Size(32, kWarpCols, kWarpRows));
  }

  DEFINE_NAME(DS4Mxfp4GatherQmmPairBlocksNaxPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive & /* other */) const override {
    return true;
  }
  auto state() const { return std::make_tuple(nullptr); }
};

} // namespace

array deepseek_mxfp4_gather_qmm_blocks_nax(
    const array &x, const array &weight, const array &scales,
    const array &block_meta, const array &block_count, StreamOrDevice s) {
  auto stream = to_stream(s);
  if (DS4Mxfp4GatherQmmBlocksNaxPrimitive::unsupported(
          x, weight, scales, block_meta, block_count, stream)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_blocks_nax] "
           "isolated symbol requires BF16 x [6144,1,K], MXFP4 U32 weight "
           "[256,N,K/8], U8 scales [256,N,K/32], BM32 block_meta [448,3], "
           "and (K,N) in {(4096,1024),(1024,4096),(4096,1280),"
           "(1280,4096)}; got "
        << x.shape() << ", " << weight.shape() << ", " << scales.shape()
        << ", " << block_meta.shape() << ", " << block_count.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  Shape output_shape = x.shape();
  output_shape.back() = weight.shape(1);
  std::vector<array> inputs = {x, weight, scales, block_meta, block_count};
  return array(std::move(output_shape), bfloat16,
               std::make_shared<DS4Mxfp4GatherQmmBlocksNaxPrimitive>(stream),
               std::move(inputs));
}

array deepseek_mxfp4_gather_qmm_pair_blocks_nax(
    const array &x, const array &weight0, const array &scales0,
    const array &weight1, const array &scales1, const array &block_meta,
    const array &block_count, StreamOrDevice s) {
  auto stream = to_stream(s);
  if (DS4Mxfp4GatherQmmPairBlocksNaxPrimitive::unsupported(
          x, weight0, scales0, weight1, scales1, block_meta, block_count,
          stream)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_pair_blocks_nax] "
           "requires BF16 x [6144,1,4096], two MXFP4 U32 weights "
           "[256,N,512] with N in {1024,1280}, matching U8 scales, and the BM32 "
           "block plan; got "
        << x.shape() << ", " << weight0.shape() << ", " << scales0.shape()
        << ", " << weight1.shape() << ", " << scales1.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  Shape output_shape = {2, x.shape(0), x.shape(1), weight0.shape(1)};
  std::vector<array> inputs = {x,       weight0,    scales0, weight1,
                               scales1, block_meta, block_count};
  return array(
      std::move(output_shape), bfloat16,
      std::make_shared<DS4Mxfp4GatherQmmPairBlocksNaxPrimitive>(stream),
      std::move(inputs));
}

} // namespace omlx::glm_kernels
