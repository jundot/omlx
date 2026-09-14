// Host side of the oQ mixed-bit QxA8 prefill path.
//
// Three ops:
//
//   qwen35_oq_a8_quantize        BF16/FP16 X -> (Qa, Sa, Ra)          [Stage A]
//   qwen35_oq_a8_qmm_t           (Qa, Sa, Ra) x packed Q4/Q5 -> Y     [GEMM]
//   qwen35_oq_a8_decode_weights  packed Q4/Q5 -> INT8 codes           [tests]
//
// Stage A is separate so Python can share quantized activations between
// projections of the same input.

#include "qwen35_prefill.h"

#include <dlfcn.h>
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::qwen35_prefill_kernels {

namespace {

using namespace mlx::core;

constexpr int kGroupSize = 64;
constexpr const char* kOqClassicMetallib = "omlx_qwen35_prefill_kernels";
constexpr const char* kOqNaxMetallib = "omlx_qwen35_prefill_kernels_nax";

// Simdgroups per Stage-A threadgroup; must match kQuantSimdgroups in
// qwen35_oq_a8.metal.
constexpr int kQuantSimdgroups = 8;

std::string oq_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&oq_binary_dir), &info)) {
      throw std::runtime_error("Unable to get omlx_qwen35_prefill binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

bool oq_row_contiguous(const array& arr) {
  // A trailing dimension of size 1 carries an arbitrary stride -- nothing
  // indexes past it -- so requiring stride 1 there rejects arrays that are
  // laid out exactly as the kernels want. It shows up for real: at M = 1 the
  // group-major Ra is [K/64, 1].
  return arr.flags().row_contiguous &&
      (arr.strides(-1) == 1 || arr.shape(-1) == 1);
}

std::string oq_type_name(Dtype dtype) {
  if (dtype == float16) {
    return "float16_t";
  }
  if (dtype == bfloat16) {
    return "bfloat16_t";
  }
  std::ostringstream msg;
  msg << "Unsupported oQ A8 dtype: " << dtype << ".";
  throw std::invalid_argument(msg.str());
}

struct OqA8NaxVariant {
  int bm;
  int bn;
  int wm;
  int wn;
};

// Tile grids must match the v8 Metal instantiations.
OqA8NaxVariant oq_a8_nax_variant(int variant) {
  switch (variant - 800) {
    case 0:
      return {64, 64, 2, 2};
    case 1:
      return {128, 64, 4, 2};
    case 2:
      return {64, 128, 2, 4};
    case 3:
      return {128, 128, 4, 4};
    case 4:
      return {32, 128, 1, 4};
    case 5:
      return {256, 64, 8, 2};
    case 6:
      return {32, 64, 1, 2};
    default: {
      std::ostringstream msg;
      msg << "Unsupported oQ A8 variant " << variant << ".";
      throw std::invalid_argument(msg.str());
    }
  }
}

bool oq_a8_bits_supported(int bits) {
  return bits == 4 || bits == 5;
}

// Packed layout check, identical to the one oMLX already applies to affine
// QMM: a row of K codes occupies K * bits / 32 uint32 words.
bool oq_a8_packed_shape_matches(int packed_dim, int K, int bits) {
  return K > 0 && packed_dim > 0 &&
      static_cast<int64_t>(packed_dim) * 32 == static_cast<int64_t>(K) * bits;
}

std::atomic<bool> oq_nax_runtime_ok{true};

// ---------------------------------------------------------------------------
// Stage A
// ---------------------------------------------------------------------------

class Qwen35OqA8QuantizePrimitive : public Primitive {
 public:
  Qwen35OqA8QuantizePrimitive(Stream stream, int act_mode)
      : Primitive(stream), act_mode_(act_mode) {
    if (act_mode_ != 0 && act_mode_ != 1) {
      std::ostringstream msg;
      msg << "Unsupported oQ A8 activation mode " << act_mode_ << ".";
      throw std::invalid_argument(msg.str());
    }
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("Qwen35OqA8QuantizePrimitive has no CPU path.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    const auto& x = inputs[0];

    auto& qa = outputs[0];
    auto& sa = outputs[1];
    auto& ra = outputs[2];
    qa.set_data(allocator::malloc(qa.nbytes()));
    sa.set_data(allocator::malloc(sa.nbytes()));
    ra.set_data(allocator::malloc(ra.nbytes()));

    const int K = x.shape(-1);
    const int M = x.size() / K;

    std::string kname;
    concatenate(
        kname,
        "oq_a8_quantize_",
        oq_type_name(x.dtype()),
        "_am",
        act_mode_);

    auto lib = d.get_library(kOqClassicMetallib, oq_binary_dir());
    auto kernel = d.get_kernel(kname, lib);

    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(x, 0);
    compute_encoder.set_output_array(qa, 1);
    compute_encoder.set_output_array(sa, 2);
    compute_encoder.set_output_array(ra, 3);
    compute_encoder.set_bytes(K, 4);

    // One threadgroup per row: the row-wide max reduction for mode 0 and the
    // group sums for both modes stay inside a single threadgroup.
    MTL::Size grid_dims(M, 1, 1);
    MTL::Size group_dims(32 * kQuantSimdgroups, 1, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(Qwen35OqA8QuantizePrimitive)
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto qa_shape = inputs[0].shape();
    auto ra_shape = qa_shape;
    ra_shape.back() = qa_shape.back() / kGroupSize;
    Shape sa_shape = act_mode_ == 0
        ? Shape{static_cast<ShapeElem>(inputs[0].size() / qa_shape.back())}
        : ra_shape;
    return {std::move(qa_shape), std::move(sa_shape), std::move(ra_shape)};
  }
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Qwen35OqA8QuantizePrimitive&>(other);
    return act_mode_ == rhs.act_mode_;
  }
  auto state() const {
    return act_mode_;
  }

 private:
  int act_mode_;
};

// ---------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------

class Qwen35OqA8QmmTPrimitive : public Primitive {
 public:
  Qwen35OqA8QmmTPrimitive(Stream stream, int bits, int act_mode, int variant)
      : Primitive(stream), bits_(bits), act_mode_(act_mode), variant_(variant) {
    if (!oq_a8_bits_supported(bits_)) {
      std::ostringstream msg;
      msg << "Unsupported oQ A8 bits " << bits_ << " (expected 4 or 5).";
      throw std::invalid_argument(msg.str());
    }
    if (act_mode_ != 0 && act_mode_ != 1) {
      std::ostringstream msg;
      msg << "Unsupported oQ A8 activation mode " << act_mode_ << ".";
      throw std::invalid_argument(msg.str());
    }
    (void)oq_a8_nax_variant(variant_);
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("Qwen35OqA8QmmTPrimitive has no CPU path.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];

    const auto& qa = inputs[0];
    const auto& sa = inputs[1];
    const auto& ra = inputs[2];
    const auto& weight = inputs[3];
    const auto& scales = inputs[4];
    const auto& biases = inputs[5];

    out.set_data(allocator::malloc(out.nbytes()));

    const int K = qa.shape(-1);
    const int N = weight.shape(0);
    const int M = qa.size() / K;

    const auto cfg = oq_a8_nax_variant(variant_);
    std::string kname;
    concatenate(kname, "oq_a8_qmm_t_nax_v8_q", bits_, "_am", act_mode_,
                "_", oq_type_name(out.dtype()), "_wm_", cfg.wm, "_wn_", cfg.wn);

    auto lib = d.get_library(kOqNaxMetallib, oq_binary_dir());
    auto kernel = d.get_kernel(kname, lib);

    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(qa, 0);
    compute_encoder.set_input_array(sa, 1);
    compute_encoder.set_input_array(ra, 2);
    compute_encoder.set_input_array(weight, 3);
    compute_encoder.set_input_array(scales, 4);
    compute_encoder.set_input_array(biases, 5);
    compute_encoder.set_output_array(out, 6);
    compute_encoder.set_bytes(K, 7);
    compute_encoder.set_bytes(N, 8);
    compute_encoder.set_bytes(M, 9);

    MTL::Size grid_dims(N / cfg.bn, (M + cfg.bm - 1) / cfg.bm, 1);
    MTL::Size group_dims(32, cfg.wm, cfg.wn);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(Qwen35OqA8QmmTPrimitive)
  // Not DEFINE_INPUT_OUTPUT_SHAPE(): that macro returns the first input's
  // shape, and this output is [..., N] while Qa is [..., K].
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    auto shape = inputs[0].shape();
    shape.back() = inputs[3].shape(0);
    return {std::move(shape)};
  }
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Qwen35OqA8QmmTPrimitive&>(other);
    return bits_ == rhs.bits_ && act_mode_ == rhs.act_mode_ &&
        variant_ == rhs.variant_;
  }
  auto state() const {
    return std::make_tuple(bits_, act_mode_, variant_);
  }

 private:
  int bits_;
  int act_mode_;
  int variant_;
};

// ---------------------------------------------------------------------------
// Weight decode (tests only)
// ---------------------------------------------------------------------------

class Qwen35OqA8DecodeWeightsPrimitive : public Primitive {
 public:
  Qwen35OqA8DecodeWeightsPrimitive(Stream stream, int bits, int k)
      : Primitive(stream), bits_(bits), k_(k) {
    if (!oq_a8_bits_supported(bits_)) {
      std::ostringstream msg;
      msg << "Unsupported oQ A8 bits " << bits_ << " (expected 4 or 5).";
      throw std::invalid_argument(msg.str());
    }
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "Qwen35OqA8DecodeWeightsPrimitive has no CPU path.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    auto& s = stream();
    auto& d = metal::device(s.device);
    auto& out = outputs[0];
    const auto& weight = inputs[0];

    out.set_data(allocator::malloc(out.nbytes()));

    const int K = out.shape(-1);
    const int N = out.shape(0);
    const int groups = K / kGroupSize;

    std::string kname;
    concatenate(kname, "oq_a8_decode_weights_q", bits_);

    auto lib = d.get_library(kOqClassicMetallib, oq_binary_dir());
    auto kernel = d.get_kernel(kname, lib);

    auto& compute_encoder = metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(weight, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_bytes(K, 2);

    MTL::Size grid_dims(groups, N, 1);
    const int tg = std::min(groups, 32);
    MTL::Size group_dims(tg, 1, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  DEFINE_NAME(Qwen35OqA8DecodeWeightsPrimitive)
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override {
    return {Shape{inputs[0].shape(0), k_}};
  }
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const Qwen35OqA8DecodeWeightsPrimitive&>(other);
    return bits_ == rhs.bits_ && k_ == rhs.k_;
  }
  auto state() const {
    return std::make_tuple(bits_, k_);
  }

 private:
  int bits_;
  int k_;
};

} // namespace

bool oq_a8_kernels_available() {
  return is_nax_available() && nax_qmm_kernels_built() &&
      oq_nax_runtime_ok.load(std::memory_order_relaxed);
}

std::vector<array> qwen35_oq_a8_quantize(
    const array& x,
    int act_mode,
    StreamOrDevice s) {
  if (x.ndim() < 2) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_oq_a8_quantize] expected rank >= 2.");
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_oq_a8_quantize] expected float16 or "
        << "bfloat16 input, got " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!oq_row_contiguous(x)) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_oq_a8_quantize] expected a row-contiguous "
        "input.");
  }

  const int K = x.shape(-1);
  if (K <= 0 || K % kGroupSize != 0) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_oq_a8_quantize] K=" << K
        << " must be a positive multiple of " << kGroupSize << ".";
    throw std::invalid_argument(msg.str());
  }
  const int M = x.size() / K;
  const int groups = K / kGroupSize;

  auto stream = to_stream(s);

  Shape qa_shape = x.shape();
  Shape ra_shape = x.shape();
  ra_shape.back() = groups;
  Shape sa_shape = act_mode == 0 ? Shape{M} : ra_shape;

  return array::make_arrays(
      {std::move(qa_shape), std::move(sa_shape), std::move(ra_shape)},
      {int8, float32, int16},
      std::make_shared<Qwen35OqA8QuantizePrimitive>(stream, act_mode),
      {x});
}

array qwen35_oq_a8_qmm_t(
    const array& qa,
    const array& sa,
    const array& ra,
    const array& weight,
    const array& scales,
    const array& biases,
    int bits,
    int act_mode,
    int variant,
    StreamOrDevice s) {
  if (!oq_a8_bits_supported(bits)) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] bits " << bits
        << " unsupported (expected 4 or 5).";
    throw std::invalid_argument(msg.str());
  }
  if (qa.dtype() != int8 || sa.dtype() != float32 || ra.dtype() != int16) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] expected int8 Qa, "
        << "float32 Sa and int16 Ra, got " << qa.dtype() << ", " << sa.dtype()
        << ", " << ra.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] expected uint32 packed "
        << "weight, got " << weight.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  const auto out_dtype = scales.dtype();
  if (out_dtype != float16 && out_dtype != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] expected float16 or "
        << "bfloat16 scales, got " << out_dtype << ".";
    throw std::invalid_argument(msg.str());
  }
  if (biases.dtype() != out_dtype) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] scales and biases must share "
        "a dtype.");
  }
  if (qa.ndim() < 2 || weight.ndim() != 2 || scales.ndim() != 2 ||
      biases.ndim() != 2) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] unexpected input ranks.");
  }
  for (const auto& t : {qa, sa, ra, weight, scales, biases}) {
    if (!oq_row_contiguous(t)) {
      throw std::invalid_argument(
          "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] every input must be "
          "row-contiguous.");
    }
  }

  const int K = qa.shape(-1);
  const int N = weight.shape(0);
  const int M = qa.size() / K;
  const int groups = K / kGroupSize;

  if (K <= 0 || N <= 0 || M <= 0 || K % kGroupSize != 0) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] bad shape M=" << M
        << " N=" << N << " K=" << K << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!oq_a8_packed_shape_matches(weight.shape(1), K, bits)) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] packed weight "
        << weight.shape() << " does not match K=" << K << " at " << bits
        << " bits.";
    throw std::invalid_argument(msg.str());
  }
  // Metadata is group-major: scales/biases [K/64, N], Ra [K/64, M].
  (void)oq_a8_nax_variant(variant);
  const int sc_rows = groups;
  const int sc_cols = N;
  if (scales.shape(0) != sc_rows || scales.shape(1) != sc_cols ||
      biases.shape() != scales.shape()) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] scales " << scales.shape()
        << " incompatible with N=" << N << " groups=" << groups
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (ra.size() != static_cast<size_t>(M) * static_cast<size_t>(groups)) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] Ra does not match [M, K/64].");
  }
  const size_t expected_sa = act_mode == 0
      ? static_cast<size_t>(M)
      : static_cast<size_t>(M) * static_cast<size_t>(groups);
  if (sa.size() != expected_sa) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] Sa does not match the "
        "activation scale mode.");
  }

  // N must tile exactly so the weight decoder can stay bounds-check free; M
  // is the token count and is handled with masked loads and stores.
  const auto cfg = oq_a8_nax_variant(variant);
  if (N % cfg.bn != 0) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] N=" << N
        << " is not a multiple of the tile BN=" << cfg.bn << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  if (stream.device == Device::cpu) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] requires the GPU stream.");
  }
  if (!oq_a8_kernels_available()) {
    throw std::runtime_error(
        "[omlx_qwen35_prefill.qwen35_oq_a8_qmm_t] the NAX metallib is "
        "unavailable on this runtime.");
  }

  Shape out_shape = qa.shape();
  out_shape.back() = N;
  return array(
      std::move(out_shape),
      out_dtype,
      std::make_shared<Qwen35OqA8QmmTPrimitive>(stream, bits, act_mode, variant),
      {qa, sa, ra, weight, scales, biases});
}

array qwen35_oq_a8_decode_weights(
    const array& weight,
    int bits,
    int group_count,
    StreamOrDevice s) {
  if (!oq_a8_bits_supported(bits)) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_oq_a8_decode_weights] bits " << bits
        << " unsupported (expected 4 or 5).";
    throw std::invalid_argument(msg.str());
  }
  if (weight.dtype() != uint32 || weight.ndim() != 2 ||
      !oq_row_contiguous(weight)) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_oq_a8_decode_weights] expected a "
        "row-contiguous 2D uint32 weight.");
  }
  const int K = group_count * kGroupSize;
  if (group_count <= 0 || !oq_a8_packed_shape_matches(weight.shape(1), K, bits)) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_oq_a8_decode_weights] packed weight "
        << weight.shape() << " does not hold " << group_count << " groups at "
        << bits << " bits.";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  return array(
      Shape{weight.shape(0), K},
      int8,
      std::make_shared<Qwen35OqA8DecodeWeightsPrimitive>(stream, bits, K),
      {weight});
}

} // namespace omlx::qwen35_prefill_kernels
