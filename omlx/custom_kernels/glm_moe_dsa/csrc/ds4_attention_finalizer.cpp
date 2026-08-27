#include "ds4_attention_finalizer.h"

#include <dlfcn.h>

#include <filesystem>
#include <sstream>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"

namespace omlx::glm_kernels {

namespace {

using namespace mlx::core;

constexpr int kHeadDim = 512;

enum class FinalizerKind { QHead, KV };

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

class DS4AttentionFinalizerPrimitive : public Primitive {
public:
  DS4AttentionFinalizerPrimitive(Stream stream, FinalizerKind kind, int offset,
                                 float eps, bool return_normalized)
      : Primitive(stream), kind_(kind), offset_(offset), eps_(eps),
        return_normalized_(return_normalized) {}

  void eval_cpu(const std::vector<array> & /* inputs */,
                std::vector<array> & /* outputs */) override {
    throw std::runtime_error("DS4AttentionFinalizerPrimitive has no CPU path.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    auto &output = outputs[0];
    output.set_data(allocator::malloc(output.nbytes()));

    auto library = device.get_library("omlx_glm_kernels", current_binary_dir());
    auto &encoder = metal::get_command_encoder(stream);
    const uint32_t write_normalized = return_normalized_ ? 1u : 0u;

    if (kind_ == FinalizerKind::QHead) {
      const auto &q = inputs[0];
      const auto &freqs = inputs[1];
      const int tokens = q.shape(1);
      const int heads = q.shape(2);
      auto kernel = device.get_kernel("ds4_q_head_rms_rope_bf16", library);

      array rotated({1, heads, tokens, kHeadDim}, bfloat16, nullptr, {});
      if (return_normalized_) {
        rotated.set_data(allocator::malloc(rotated.nbytes()));
        encoder.add_temporary(rotated);
      }
      array &normalized = output;
      array &rotated_target = return_normalized_ ? rotated : output;

      encoder.set_compute_pipeline_state(kernel);
      encoder.set_input_array(q, 0);
      encoder.set_input_array(freqs, 1);
      encoder.set_output_array(normalized, 2);
      encoder.set_output_array(rotated_target, 3);
      encoder.set_bytes(offset_, 4);
      encoder.set_bytes(eps_, 5);
      encoder.set_bytes(heads, 6);
      encoder.set_bytes(write_normalized, 7);
      encoder.set_bytes(tokens, 8);
      encoder.dispatch_threadgroups(MTL::Size(heads / 4, tokens, 1),
                                    MTL::Size(512, 1, 1));
      return;
    }

    const auto &kv = inputs[0];
    const int tokens = kv.shape(1);
    const auto &weight = inputs[1];
    const auto &freqs = inputs[2];
    auto kernel = device.get_kernel("ds4_kv_rms_rope_bf16", library);

    array rotated({1, 1, tokens, kHeadDim}, bfloat16, nullptr, {});
    if (return_normalized_) {
      rotated.set_data(allocator::malloc(rotated.nbytes()));
      encoder.add_temporary(rotated);
    }
    array &normalized = output;
    array &rotated_target = return_normalized_ ? rotated : output;

    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(kv, 0);
    encoder.set_input_array(weight, 1);
    encoder.set_input_array(freqs, 2);
    encoder.set_output_array(normalized, 3);
    encoder.set_output_array(rotated_target, 4);
    encoder.set_bytes(offset_, 5);
    encoder.set_bytes(eps_, 6);
    encoder.set_bytes(write_normalized, 7);
    encoder.dispatch_threadgroups(MTL::Size(tokens, 1, 1),
                                  MTL::Size(128, 1, 1));
  }

  DEFINE_NAME(DS4AttentionFinalizerPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs =
        static_cast<const DS4AttentionFinalizerPrimitive &>(other);
    return kind_ == rhs.kind_ && offset_ == rhs.offset_ && eps_ == rhs.eps_ &&
           return_normalized_ == rhs.return_normalized_;
  }
  auto state() const {
    return std::make_tuple(nullptr, int(kind_), offset_, eps_,
                           return_normalized_);
  }

private:
  FinalizerKind kind_;
  int offset_;
  float eps_;
  bool return_normalized_;
};

bool common_supported(const array &input, const array &freqs, int offset,
                      Stream stream) {
  return !(stream.device == Device::cpu) && input.dtype() == bfloat16 &&
         freqs.dtype() == float32 && row_contiguous(input) &&
         row_contiguous(freqs) && freqs.shape() == Shape{256} && offset >= 0;
}

} // namespace

array ds4_q_head_rms_rope(const array &q, const array &freqs, int offset,
                          float eps, bool return_normalized, StreamOrDevice s) {
  auto stream = to_stream(s);
  const bool heads_supported =
      q.ndim() == 4 &&
      (q.shape(2) == 24 || q.shape(2) == 32 || q.shape(2) == 40 ||
       q.shape(2) == 64);
  const bool tokens_supported =
      q.ndim() == 4 &&
      (q.shape(1) == 6 || q.shape(1) == 1024 || q.shape(1) == 2048);
  if (!common_supported(q, freqs, offset, stream) || !heads_supported ||
      !tokens_supported || q.shape(0) != 1 || q.shape(3) != kHeadDim) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.ds4_q_head_rms_rope] unsupported shape; "
           "requires BF16 q [1,M,H,512], M in {6,1024,2048}, "
           "H in {24,32,40,64}, FP32 freqs "
           "[256], non-negative scalar offset; got "
        << q.shape() << ", " << freqs.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  Shape output_shape =
      return_normalized ? q.shape()
                        : Shape{1, q.shape(2), q.shape(1), kHeadDim};
  std::vector<array> inputs = {q, freqs};
  return array(
      std::move(output_shape), bfloat16,
      std::make_shared<DS4AttentionFinalizerPrimitive>(
          stream, FinalizerKind::QHead, offset, eps, return_normalized),
      std::move(inputs));
}

array ds4_kv_rms_rope(const array &kv, const array &weight, const array &freqs,
                      int offset, float eps, bool return_normalized,
                      StreamOrDevice s) {
  auto stream = to_stream(s);
  const bool shape_supported = kv.ndim() == 3 && kv.shape(0) == 1 &&
      (kv.shape(1) == 6 || kv.shape(1) == 1024 || kv.shape(1) == 2048) &&
      kv.shape(2) == kHeadDim;
  if (!common_supported(kv, freqs, offset, stream) || !shape_supported ||
      weight.shape() != Shape{kHeadDim} || weight.dtype() != bfloat16 ||
      !row_contiguous(weight)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.ds4_kv_rms_rope] unsupported shape; requires "
           "BF16 kv [1,M,512], M in {6,1024,2048}, BF16 weight [512], "
           "FP32 freqs [256], "
           "non-negative scalar offset; got "
        << kv.shape() << ", " << weight.shape() << ", " << freqs.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  Shape output_shape =
      return_normalized ? kv.shape() : Shape{1, 1, kv.shape(1), kHeadDim};
  std::vector<array> inputs = {kv, weight, freqs};
  return array(std::move(output_shape), bfloat16,
               std::make_shared<DS4AttentionFinalizerPrimitive>(
                   stream, FinalizerKind::KV, offset, eps, return_normalized),
               std::move(inputs));
}

} // namespace omlx::glm_kernels
