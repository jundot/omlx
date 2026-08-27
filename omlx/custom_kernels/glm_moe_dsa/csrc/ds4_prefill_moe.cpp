#include "ds4_prefill_moe.h"

#include <dlfcn.h>
#include <filesystem>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/utils.h"

namespace omlx::glm_kernels {

namespace {

using namespace mlx::core;

constexpr int kRoutes = 6144;
constexpr int kExperts = 256;
constexpr int kHidden = 4096;
constexpr int kIntermediate = 1024;
constexpr int kGroupSize = 32;
constexpr int kBits = 4;
constexpr int kValuesPerU32 = 32 / kBits;
constexpr int kBM = 32;
constexpr int kBN = 32;
constexpr int kBK = 32;
constexpr int kWM = 1;
constexpr int kWN = 2;
constexpr int kVariant = 2;
constexpr int kMaxBlocks = (kRoutes + kBM - 1) / kBM + kExperts;
constexpr float kActivationLimit = 10.0f;

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

class DS4Mxfp4PairSwiGLUBlocksPrimitive : public Primitive {
 public:
  explicit DS4Mxfp4PairSwiGLUBlocksPrimitive(
      Stream stream,
      bool tail8 = false,
      int intermediate = kIntermediate)
      : Primitive(stream), tail8_(tail8), intermediate_(intermediate) {}

  static bool unsupported(
      const array& x,
      const array& up_weight,
      const array& up_scales,
      const array& gate_weight,
      const array& gate_scales,
      const array& block_meta,
      const array& block_count,
      float activation_limit,
      int variant,
      int intermediate,
      Stream s) {
    if (intermediate != 768 && intermediate != 1024 && intermediate != 1280) {
      return true;
    }
    if (s.device == Device::cpu || x.dtype() != float16 ||
        up_weight.dtype() != uint32 || gate_weight.dtype() != uint32 ||
        up_scales.dtype() != uint8 || gate_scales.dtype() != uint8 ||
        block_meta.dtype() != int32 || block_count.dtype() != int32) {
      return true;
    }
    if (x.ndim() != 3 || up_weight.ndim() != 3 || up_scales.ndim() != 3 ||
        gate_weight.ndim() != 3 || gate_scales.ndim() != 3 ||
        block_meta.ndim() != 2 || block_count.ndim() != 1) {
      return true;
    }
    if (!row_contiguous(x) || !row_contiguous(up_weight) ||
        !row_contiguous(up_scales) || !row_contiguous(gate_weight) ||
        !row_contiguous(gate_scales) || !row_contiguous(block_meta) ||
        !row_contiguous(block_count)) {
      return true;
    }
    if (x.shape() != Shape{kRoutes, 1, kHidden} ||
        up_weight.shape() !=
            Shape{kExperts, intermediate, kHidden / kValuesPerU32} ||
        up_scales.shape() !=
            Shape{kExperts, intermediate, kHidden / kGroupSize} ||
        gate_weight.shape() != up_weight.shape() ||
        gate_scales.shape() != up_scales.shape() ||
        block_meta.shape() != Shape{kMaxBlocks, 3} ||
        block_count.shape() != Shape{1}) {
      return true;
    }
    return activation_limit != kActivationLimit || variant != kVariant;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "DS4Mxfp4PairSwiGLUBlocksPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    const auto& x = inputs[0];
    const auto& up_weight = inputs[1];
    const auto& up_scales = inputs[2];
    const auto& gate_weight = inputs[3];
    const auto& gate_scales = inputs[4];
    const auto& block_meta = inputs[5];
    const auto& block_count = inputs[6];

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = d.get_kernel(
        tail8_
            ? "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8_"
              "float16_t_bm32_bn32_bk32_wm1_wn2"
            : "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_"
              "float16_t_bm32_bn32_bk32_wm1_wn2",
        lib);
    auto& encoder = metal::get_command_encoder(s);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(up_weight, 1);
    encoder.set_input_array(up_scales, 2);
    encoder.set_input_array(gate_weight, 3);
    encoder.set_input_array(gate_scales, 4);
    encoder.set_input_array(block_meta, 5);
    encoder.set_input_array(block_count, 6);
    encoder.set_output_array(out, 7);
    encoder.set_bytes(kMaxBlocks, 8);
    encoder.set_bytes(kRoutes, 9);
    encoder.set_bytes(intermediate_, 10);
    encoder.set_bytes(kHidden, 11);
    encoder.set_bytes(kActivationLimit, 12);
    encoder.dispatch_threadgroups(
        MTL::Size(intermediate_ / kBN, kMaxBlocks, 1),
        MTL::Size(kWM * kWN * 32, 1, 1));
  }

  DEFINE_NAME(DS4Mxfp4PairSwiGLUBlocks)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const DS4Mxfp4PairSwiGLUBlocksPrimitive&>(other);
    return tail8_ == rhs.tail8_ && intermediate_ == rhs.intermediate_;
  }
  auto state() const {
    return std::make_tuple(tail8_, intermediate_);
  }

 private:
  bool tail8_;
  int intermediate_;
};

class DS4Mxfp4DownTail8BlocksPrimitive : public Primitive {
 public:
  explicit DS4Mxfp4DownTail8BlocksPrimitive(
      Stream stream,
      int intermediate = kIntermediate)
      : Primitive(stream), intermediate_(intermediate) {}

  static bool unsupported(
      const array& x,
      const array& weight,
      const array& scales,
      const array& block_meta,
      const array& block_count,
      int variant,
      int intermediate,
      Stream s) {
    if (intermediate != 768 && intermediate != 1024 && intermediate != 1280) {
      return true;
    }
    if (s.device == Device::cpu || x.dtype() != float16 ||
        weight.dtype() != uint32 || scales.dtype() != uint8 ||
        block_meta.dtype() != int32 || block_count.dtype() != int32) {
      return true;
    }
    if (x.ndim() != 3 || weight.ndim() != 3 || scales.ndim() != 3 ||
        block_meta.ndim() != 2 || block_count.ndim() != 1) {
      return true;
    }
    if (!row_contiguous(x) || !row_contiguous(weight) ||
        !row_contiguous(scales) || !row_contiguous(block_meta) ||
        !row_contiguous(block_count)) {
      return true;
    }
    if (x.shape() != Shape{kRoutes, 1, intermediate} ||
        weight.shape() !=
            Shape{kExperts, kHidden, intermediate / kValuesPerU32} ||
        scales.shape() !=
            Shape{kExperts, kHidden, intermediate / kGroupSize} ||
        block_meta.shape() != Shape{kMaxBlocks, 3} ||
        block_count.shape() != Shape{1}) {
      return true;
    }
    return variant != kVariant;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "DS4Mxfp4DownTail8BlocksPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];
    out.set_data(allocator::malloc(out.nbytes()));

    const auto& x = inputs[0];
    const auto& weight = inputs[1];
    const auto& scales = inputs[2];
    const auto& block_meta = inputs[3];
    const auto& block_count = inputs[4];

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = d.get_kernel(
        "deepseek_mxfp4_gather_qmm_blocks_tail8_"
        "float16_t_bm32_bn32_bk32_wm1_wn2",
        lib);
    auto& encoder = metal::get_command_encoder(s);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(weight, 1);
    encoder.set_input_array(scales, 2);
    encoder.set_input_array(block_meta, 3);
    encoder.set_input_array(block_count, 4);
    encoder.set_output_array(out, 5);
    encoder.set_bytes(kMaxBlocks, 6);
    encoder.set_bytes(kRoutes, 7);
    encoder.set_bytes(kHidden, 8);
    encoder.set_bytes(intermediate_, 9);
    encoder.dispatch_threadgroups(
        MTL::Size(kHidden / kBN, kMaxBlocks, 1),
        MTL::Size(kWM * kWN * 32, 1, 1));
  }

  DEFINE_NAME(DS4Mxfp4DownTail8Blocks)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const DS4Mxfp4DownTail8BlocksPrimitive&>(other);
    return intermediate_ == rhs.intermediate_;
  }
  auto state() const {
    return std::make_tuple(intermediate_);
  }

 private:
  int intermediate_;
};

} // namespace

array deepseek_mxfp4_gather_qmm_pair_swiglu_blocks(
    const array& x,
    const array& up_weight,
    const array& up_scales,
    const array& gate_weight,
    const array& gate_scales,
    const array& block_meta,
    const array& block_count,
    float activation_limit,
    int variant,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  if (DS4Mxfp4PairSwiGLUBlocksPrimitive::unsupported(
          x,
          up_weight,
          up_scales,
          gate_weight,
          gate_scales,
          block_meta,
          block_count,
          activation_limit,
          variant,
          kIntermediate,
          stream)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels."
           "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks] unsupported "
           "configuration; isolated symbol requires FP16 x [6144,1,4096], "
           "MXFP4 up/gate [256,1024,512] with scales [256,1024,128], "
           "block_meta [448,3], limit=10, variant=2; got x="
        << x.shape() << ", up=" << up_weight.shape()
        << ", scales=" << up_scales.shape()
        << ", block_meta=" << block_meta.shape()
        << ", activation_limit=" << activation_limit
        << ", variant=" << variant << ".";
    throw std::invalid_argument(msg.str());
  }

  std::vector<array> inputs = {
      x,
      up_weight,
      up_scales,
      gate_weight,
      gate_scales,
      block_meta,
      block_count};
  return array(
      Shape{kRoutes, 1, kIntermediate},
      float16,
      std::make_shared<DS4Mxfp4PairSwiGLUBlocksPrimitive>(stream),
      std::move(inputs));
}

array deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8(
    const array& x,
    const array& up_weight,
    const array& up_scales,
    const array& gate_weight,
    const array& gate_scales,
    const array& block_meta,
    const array& block_count,
    float activation_limit,
    int variant,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  const int intermediate = up_weight.ndim() >= 2 ? up_weight.shape(1) : -1;
  if (DS4Mxfp4PairSwiGLUBlocksPrimitive::unsupported(
          x,
          up_weight,
          up_scales,
          gate_weight,
          gate_scales,
          block_meta,
          block_count,
          activation_limit,
          variant,
          intermediate,
          stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels."
        "deepseek_mxfp4_gather_qmm_pair_swiglu_blocks_tail8] isolated "
        "symbol requires the fixed M=1024 DS4 TP intermediate widths "
        "{768,1024,1280}.");
  }
  std::vector<array> inputs = {
      x,
      up_weight,
      up_scales,
      gate_weight,
      gate_scales,
      block_meta,
      block_count};
  return array(
      Shape{kRoutes, 1, intermediate},
      float16,
      std::make_shared<DS4Mxfp4PairSwiGLUBlocksPrimitive>(
          stream, true, intermediate),
      std::move(inputs));
}

array deepseek_mxfp4_gather_qmm_blocks_tail8(
    const array& x,
    const array& weight,
    const array& scales,
    const array& block_meta,
    const array& block_count,
    int variant,
    StreamOrDevice s) {
  auto stream = to_stream(s);
  const int intermediate = x.ndim() >= 3 ? x.shape(2) : -1;
  if (DS4Mxfp4DownTail8BlocksPrimitive::unsupported(
          x,
          weight,
          scales,
          block_meta,
          block_count,
          variant,
          intermediate,
          stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.deepseek_mxfp4_gather_qmm_blocks_tail8] "
        "isolated symbol requires M=1024 DS4 TP intermediate widths "
        "{768,1024,1280}, matching MXFP4 down/scales, block_meta [448,3], "
        "and variant=2.");
  }
  std::vector<array> inputs = {x, weight, scales, block_meta, block_count};
  return array(
      Shape{kRoutes, 1, kHidden},
      float16,
      std::make_shared<DS4Mxfp4DownTail8BlocksPrimitive>(
          stream, intermediate),
      std::move(inputs));
}

} // namespace omlx::glm_kernels
