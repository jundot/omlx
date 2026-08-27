#include "ds4_qkv_bundle.h"

#include <dlfcn.h>
#include <filesystem>
#include <sstream>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"

namespace omlx::glm_kernels {

namespace {

using namespace mlx::core;

constexpr int kHidden = 4096;
constexpr int kQRows = 1024;
constexpr int kKVRows = 512;
constexpr int kCompressorRows = 1024;
constexpr int kIndexRows = 256;
constexpr int kPackedRows = 4096;
constexpr int kMXFP8ValuesPerU32 = 4;
constexpr int kGroupSize = 32;
constexpr int kDispatches = 1;

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

bool row_contiguous(const array& value) {
  return value.flags().row_contiguous && value.strides(-1) == 1;
}

class DS4QKVCompressorBundleB1Primitive : public Primitive {
 public:
  explicit DS4QKVCompressorBundleB1Primitive(Stream stream)
      : Primitive(stream) {}

  static bool unsupported(const std::vector<array>& values, Stream s) {
    if (s.device == Device::cpu || values.size() != 9) {
      return true;
    }
    const auto& x = values[0];
    const auto& wq = values[1];
    const auto& sq = values[2];
    const auto& wkv = values[3];
    const auto& skv = values[4];
    const auto& ckv = values[5];
    const auto& cgate = values[6];
    const auto& ikv = values[7];
    const auto& igate = values[8];
    if (x.dtype() != bfloat16 || wq.dtype() != uint32 ||
        wkv.dtype() != uint32 || sq.dtype() != uint8 ||
        skv.dtype() != uint8 || ckv.dtype() != bfloat16 ||
        cgate.dtype() != bfloat16 || ikv.dtype() != bfloat16 ||
        igate.dtype() != bfloat16) {
      return true;
    }
    for (const auto& value : values) {
      if (!row_contiguous(value)) {
        return true;
      }
    }
    return x.shape() != Shape{1, kHidden} ||
        wq.shape() != Shape{kQRows, kHidden / kMXFP8ValuesPerU32} ||
        sq.shape() != Shape{kQRows, kHidden / kGroupSize} ||
        wkv.shape() != Shape{kKVRows, kHidden / kMXFP8ValuesPerU32} ||
        skv.shape() != Shape{kKVRows, kHidden / kGroupSize} ||
        ckv.shape() != Shape{kCompressorRows, kHidden} ||
        cgate.shape() != Shape{kCompressorRows, kHidden} ||
        ikv.shape() != Shape{kIndexRows, kHidden} ||
        igate.shape() != Shape{kIndexRows, kHidden};
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "DS4QKVCompressorBundleB1Primitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    auto lib = d.get_library(
        "omlx_glm_kernels_decode", current_binary_dir());
    auto& encoder = metal::get_command_encoder(s);

    auto kernel = d.get_kernel("ds4_qkv_bundle_all_b1", lib);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_input_array(inputs[3], 3);
    encoder.set_input_array(inputs[4], 4);
    encoder.set_input_array(inputs[5], 5);
    encoder.set_input_array(inputs[6], 6);
    encoder.set_input_array(inputs[7], 7);
    encoder.set_input_array(inputs[8], 8);
    encoder.set_output_array(out, 9);
    constexpr int quantized_rows = kQRows + kKVRows;
    constexpr int dense_rows = 2 * kCompressorRows + 2 * kIndexRows;
    constexpr int quantized_physical_groups = quantized_rows / 16;
    encoder.dispatch_threadgroups(
        MTL::Size(quantized_physical_groups + dense_rows / 16, 1, 1),
        MTL::Size(32, 1, 4));
  }

  DEFINE_NAME(DS4QKVCompressorBundleB1)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& /* other */) const override {
    return true;
  }
  auto state() const {
    return std::make_tuple(nullptr);
  }
};

class DS4QKVPairB1Primitive : public Primitive {
 public:
  explicit DS4QKVPairB1Primitive(Stream stream) : Primitive(stream) {}

  static bool unsupported(const std::vector<array>& v, Stream s) {
    if (s.device == Device::cpu || v.size() != 5) return true;
    for (const auto& x : v) if (!row_contiguous(x)) return true;
    return v[0].dtype() != bfloat16 || v[0].shape() != Shape{1, kHidden} ||
        v[1].dtype() != uint32 ||
        v[1].shape() != Shape{kQRows, kHidden / kMXFP8ValuesPerU32} ||
        v[2].dtype() != uint8 ||
        v[2].shape() != Shape{kQRows, kHidden / kGroupSize} ||
        v[3].dtype() != uint32 ||
        v[3].shape() != Shape{kKVRows, kHidden / kMXFP8ValuesPerU32} ||
        v[4].dtype() != uint8 ||
        v[4].shape() != Shape{kKVRows, kHidden / kGroupSize};
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("DS4QKVPairB1Primitive has no CPU path.");
  }
  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
    auto& s = stream(); auto& d = metal::device(s.device); auto& y = out[0];
    y.set_data(allocator::malloc(y.nbytes()));
    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = d.get_kernel("ds4_qkv_bundle_mxfp8_b1", lib);
    auto& encoder = metal::get_command_encoder(s);
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 5; ++i) encoder.set_input_array(in[i], i);
    encoder.set_output_array(y, 5);
    encoder.dispatch_threadgroups(
        MTL::Size(1, kQRows / 8, 2), MTL::Size(32, 2, 1));
  }
  DEFINE_NAME(DS4QKVPairB1Primitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive&) const override { return true; }
  auto state() const { return std::make_tuple(nullptr); }
};

class DS4QKVCompressor128BundleB1Primitive : public Primitive {
 public:
  explicit DS4QKVCompressor128BundleB1Primitive(Stream stream)
      : Primitive(stream) {}

  static bool unsupported(const std::vector<array>& v, Stream s) {
    if (s.device == Device::cpu || v.size() != 7) return true;
    if (DS4QKVPairB1Primitive::unsupported(
            std::vector<array>(v.begin(), v.begin() + 5), s)) return true;
    return v[5].dtype() != bfloat16 ||
        v[5].shape() != Shape{512, kHidden} || !row_contiguous(v[5]) ||
        v[6].dtype() != bfloat16 ||
        v[6].shape() != Shape{512, kHidden} || !row_contiguous(v[6]);
  }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "DS4QKVCompressor128BundleB1Primitive has no CPU path.");
  }
  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
    auto& s = stream(); auto& d = metal::device(s.device); auto& y = out[0];
    y.set_data(allocator::malloc(y.nbytes()));
    auto lib = d.get_library("omlx_glm_kernels_decode", current_binary_dir());
    auto kernel = d.get_kernel("ds4_qkv_bundle128_all_b1", lib);
    auto& encoder = metal::get_command_encoder(s);
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 7; ++i) encoder.set_input_array(in[i], i);
    encoder.set_output_array(y, 7);
    encoder.dispatch_threadgroups(MTL::Size(160, 1, 1), MTL::Size(32, 1, 4));
  }
  DEFINE_NAME(DS4QKVCompressor128BundleB1Primitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive&) const override { return true; }
  auto state() const { return std::make_tuple(nullptr); }
};

} // namespace

array deepseek_v4_qkv_compressor_bundle_b1(
    const array& x,
    const array& wq_a_weight,
    const array& wq_a_scales,
    const array& wkv_weight,
    const array& wkv_scales,
    const array& compressor_wkv,
    const array& compressor_wgate,
    const array& index_compressor_wkv,
    const array& index_compressor_wgate,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  std::vector<array> inputs = {
      x,
      wq_a_weight,
      wq_a_scales,
      wkv_weight,
      wkv_scales,
      compressor_wkv,
      compressor_wgate,
      index_compressor_wkv,
      index_compressor_wgate};
  if (!metal::is_available() ||
      DS4QKVCompressorBundleB1Primitive::unsupported(inputs, stream)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_v4_qkv_compressor_bundle_b1] "
           "unsupported configuration; isolated symbol requires the exact "
           "ratio-4 B1 contract (BF16 x [1,4096], MXFP8 Q-A/raw-KV, four "
           "BF16 compressor banks); got x="
        << x.shape() << ", wq=" << wq_a_weight.shape()
        << ", wkv=" << wkv_weight.shape()
        << ", compressor_wkv=" << compressor_wkv.shape()
        << ", index_wkv=" << index_compressor_wkv.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  return array(
      Shape{1, kPackedRows},
      bfloat16,
      std::make_shared<DS4QKVCompressorBundleB1Primitive>(stream),
      std::move(inputs));
}

array deepseek_v4_qkv_pair_b1(
    const array& x, const array& wq_a_weight, const array& wq_a_scales,
    const array& wkv_weight, const array& wkv_scales, StreamOrDevice s) {
  auto stream = to_stream(s);
  std::vector<array> inputs = {
      x, wq_a_weight, wq_a_scales, wkv_weight, wkv_scales};
  if (DS4QKVPairB1Primitive::unsupported(inputs, stream)) {
    throw std::invalid_argument("unsupported DS4 ratio-0 B1 QKV bundle");
  }
  return array(
      Shape{1, kQRows + kKVRows}, bfloat16,
      std::make_shared<DS4QKVPairB1Primitive>(stream), std::move(inputs));
}

array deepseek_v4_qkv_compressor128_bundle_b1(
    const array& x, const array& wq_a_weight, const array& wq_a_scales,
    const array& wkv_weight, const array& wkv_scales,
    const array& compressor_wkv, const array& compressor_wgate,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  std::vector<array> inputs = {x, wq_a_weight, wq_a_scales, wkv_weight,
                               wkv_scales, compressor_wkv, compressor_wgate};
  if (DS4QKVCompressor128BundleB1Primitive::unsupported(inputs, stream)) {
    throw std::invalid_argument("unsupported DS4 ratio-128 B1 QKV bundle");
  }
  return array(
      Shape{1, kQRows + kKVRows + 1024}, bfloat16,
      std::make_shared<DS4QKVCompressor128BundleB1Primitive>(stream),
      std::move(inputs));
}

int deepseek_v4_qkv_compressor_bundle_b1_dispatches() {
  return kDispatches;
}

} // namespace omlx::glm_kernels
