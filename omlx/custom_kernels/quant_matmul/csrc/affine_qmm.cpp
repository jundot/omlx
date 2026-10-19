#include "affine_qmm.h"
#include "omlx/binary_dir.h"
#include "nax.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::quant_matmul {

namespace {

using namespace mlx::core;

bool last_dim_contiguous(const array& arr) {
  return arr.strides(-1) == 1;
}

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
  msg << "Unsupported affine qmm dtype: " << dtype << ".";
  throw std::invalid_argument(msg.str());
}

struct AffineQmmVariant {
  int bm;
  int bk;
  int bn;
};

struct AffineQmmNaxVariant {
  int bm;
  int bk;
  int bn;
  int wm;
  int wn;
};

bool affine_qmm_bits_supported(int bits) {
  return bits == 2 || bits == 4 || bits == 5 || bits == 6 || bits == 8;
}

bool affine_qmm_packed_shape_matches(int packed_dim, int K, int bits) {
  return K > 0 && packed_dim > 0 &&
      static_cast<int64_t>(packed_dim) * 32 == static_cast<int64_t>(K) * bits;
}

constexpr const char* kNaxMetallibName = "omlx_quant_matmul_kernels_nax";

// Set to false once loading the NAX metallib (or one of its pipelines) fails
// so every later call degrades to the classic kernels without re-probing.
std::atomic<bool> nax_qmm_runtime_ok{true};

AffineQmmVariant affine_qmm_variant(int variant) {
  switch (variant) {
    case 0:
      return {/* bm = */ 32, /* bk = */ 32, /* bn = */ 32};
    case 1:
      return {/* bm = */ 32, /* bk = */ 64, /* bn = */ 32};
    case 2:
      return {/* bm = */ 32, /* bk = */ 64, /* bn = */ 64};
    case 3:
      return {/* bm = */ 64, /* bk = */ 64, /* bn = */ 64};
    case 4:
      return {/* bm = */ 16, /* bk = */ 64, /* bn = */ 64};
    case 5:
      return {/* bm = */ 64, /* bk = */ 64, /* bn = */ 128};
    case 6:
      return {/* bm = */ 128, /* bk = */ 64, /* bn = */ 64};
    case 7:
      return {/* bm = */ 128, /* bk = */ 64, /* bn = */ 32};
    case 8:
      return {/* bm = */ 64, /* bk = */ 32, /* bn = */ 64};
    case 9:
      return {/* bm = */ 128, /* bk = */ 32, /* bn = */ 64};
    default: {
      std::ostringstream msg;
      msg << "Unsupported affine qmm variant " << variant << ".";
      throw std::invalid_argument(msg.str());
    }
  }
}

// Must stay in sync with the instantiations in affine_qmm_nax.metal.
// Variant 0 matches the tile MLX ships for affine_qmm_t_nax. BK stays at or
// below the group size (64): QuantizedBlockLoader rejects larger columns.
AffineQmmNaxVariant affine_qmm_nax_variant(int variant) {
  switch (variant) {
    case 0:
      return {/* bm = */ 64, /* bk = */ 64, /* bn = */ 64, 2, 2};
    case 1:
      return {/* bm = */ 32, /* bk = */ 64, /* bn = */ 64, 2, 2};
    case 2:
      return {/* bm = */ 128, /* bk = */ 64, /* bn = */ 64, 2, 2};
    case 3:
      return {/* bm = */ 64, /* bk = */ 64, /* bn = */ 128, 2, 2};
    case 4:
      return {/* bm = */ 64, /* bk = */ 32, /* bn = */ 64, 2, 2};
    case 5:
      return {/* bm = */ 64, /* bk = */ 64, /* bn = */ 64, 4, 1};
    // (64, 64, 64, wm=1, wn=4) was variant 6: a 16-column per-simdgroup N
    // tile that returns wrong output on M5 Max / mlx 0.32.2 (max abs err 5-8
    // against stock at every shape and bit width; the other tiles match
    // exactly). Rejected rather than kept as an opt-in footgun.
    default: {
      std::ostringstream msg;
      msg << "Unsupported affine qmm NAX variant " << variant << ".";
      throw std::invalid_argument(msg.str());
    }
  }
}


class AffineQmmTPrimitive : public Primitive {
 public:
  AffineQmmTPrimitive(
      Stream stream,
      int bits,
      int variant,
      bool use_nax,
      int nax_variant,
      int group_size)
      : Primitive(stream),
        bits_(bits),
        variant_(variant),
        use_nax_(use_nax),
        nax_variant_(nax_variant),
        group_size_(group_size) {
    if (!affine_qmm_bits_supported(bits_)) {
      std::ostringstream msg;
      msg << "Unsupported affine qmm bits " << bits_ << ".";
      throw std::invalid_argument(msg.str());
    }
    if (group_size_ != 64 && group_size_ != 128) {
      std::ostringstream msg;
      msg << "Unsupported affine qmm group_size " << group_size_ << ".";
      throw std::invalid_argument(msg.str());
    }
    (void)affine_qmm_variant(variant_);
    if (use_nax_) {
      (void)affine_qmm_nax_variant(nax_variant_);
    }
  }

  static bool unsupported(
      const array& x,
      const array& weight,
      const array& scales,
      const array& biases,
      int bits,
      int variant,
      int group_size,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (!affine_qmm_bits_supported(bits)) {
      return true;
    }
    if (group_size != 64 && group_size != 128) {
      return true;
    }
    if (x.dtype() != float16 && x.dtype() != bfloat16) {
      return true;
    }
    if (weight.dtype() != uint32 || scales.dtype() != x.dtype() ||
        biases.dtype() != x.dtype()) {
      return true;
    }
    if (x.ndim() < 2 || weight.ndim() != 2 || scales.ndim() != 2 ||
        biases.ndim() != 2) {
      return true;
    }
    if (!row_contiguous(x) || !row_contiguous(weight) ||
        !row_contiguous(scales) || !row_contiguous(biases)) {
      return true;
    }

    const auto cfg = affine_qmm_variant(variant);
    const int K = x.shape(-1);
    const int N = weight.shape(0);
    if (K <= 0 || N <= 0 || x.size() <= 0 || K % group_size != 0 ||
        K % cfg.bk != 0 || N % cfg.bn != 0) {
      return true;
    }
    if (!affine_qmm_packed_shape_matches(weight.shape(1), K, bits) ||
        scales.shape(0) != N || scales.shape(1) != K / group_size ||
        biases.shape() != scales.shape()) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("AffineQmmTPrimitive has no CPU path.");
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

    const int K = x.shape(-1);
    const int N = weight.shape(0);
    const int M = x.size() / K;

    auto& compute_encoder = metal::get_command_encoder(s);
    auto encode = [&](MTL::ComputePipelineState* kernel,
                      int bm,
                      int bn,
                      int wm,
                      int wn) {
      compute_encoder.set_compute_pipeline_state(kernel);
      compute_encoder.set_input_array(weight, 0);
      compute_encoder.set_input_array(scales, 1);
      compute_encoder.set_input_array(biases, 2);
      compute_encoder.set_input_array(x, 3);
      compute_encoder.set_output_array(out, 4);
      compute_encoder.set_bytes(K, 5);
      compute_encoder.set_bytes(N, 6);
      compute_encoder.set_bytes(M, 7);

      MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);
      MTL::Size group_dims(32, wm, wn);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    };

    if (use_nax_ && nax_qmm_runtime_ok.load(std::memory_order_relaxed)) {
      const auto cfg = affine_qmm_nax_variant(nax_variant_);
      std::string kname;
      concatenate(
          kname,
          "qwen35_q",
          bits_,
          group_size_ == 128 ? "_affine_qmm128_t_nax_"
                             : "_affine_qmm_t_nax_",
          metal_type_name(x.dtype()),
          "_bm_",
          cfg.bm,
          "_bk_",
          cfg.bk,
          "_bn_",
          cfg.bn,
          "_wm_",
          cfg.wm,
          "_wn_",
          cfg.wn);
      try {
        auto lib = d.get_library(kNaxMetallibName, binary_dir());
        auto kernel = d.get_kernel(kname, lib);
        encode(kernel, cfg.bm, cfg.bn, cfg.wm, cfg.wn);
        return;
      } catch (const std::exception&) {
        // The metallib next to the extension predates the NAX kernels (or
        // pipeline creation was rejected); disable NAX for the process and
        // fall through to the classic kernel, which unsupported() already
        // validated for these shapes.
        nax_qmm_runtime_ok.store(false, std::memory_order_relaxed);
      }
    }

    const auto cfg = affine_qmm_variant(variant_);
    std::string kname;
    concatenate(
        kname,
        "qwen35_q",
        bits_,
        group_size_ == 128 ? "_affine_qmm128_t_" : "_affine_qmm_t_",
        metal_type_name(x.dtype()),
        "_bm_",
        cfg.bm,
        "_bk_",
        cfg.bk,
        "_bn_",
        cfg.bn);

    auto lib = d.get_library("omlx_quant_matmul_kernels", binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    encode(kernel, cfg.bm, cfg.bn, 2, 2);
  }

  DEFINE_NAME(AffineQmmTPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const AffineQmmTPrimitive&>(other);
    return bits_ == rhs.bits_ && variant_ == rhs.variant_ &&
        use_nax_ == rhs.use_nax_ && nax_variant_ == rhs.nax_variant_ &&
        group_size_ == rhs.group_size_;
  }
  auto state() const {
    return std::make_tuple(bits_, variant_, use_nax_, nax_variant_, group_size_);
  }

 private:
  int bits_;
  int variant_;
  bool use_nax_;
  int nax_variant_;
  int group_size_;
};


} // namespace

bool nax_qmm_kernels_built() {
  static bool built = []() {
    std::error_code ec;
    return std::filesystem::exists(
        std::filesystem::path(binary_dir()) /
            (std::string(kNaxMetallibName) + ".metallib"),
        ec);
  }();
  return built;
}

bool nax_qmm_runtime_active() {
  return nax_qmm_runtime_ok.load(std::memory_order_relaxed);
}


array affine_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int bits,
    int variant,
    bool use_nax,
    int nax_variant,
    int group_size,
    StreamOrDevice s) {
  (void)affine_qmm_variant(variant);
  if (!affine_qmm_bits_supported(bits)) {
    std::ostringstream msg;
    msg << "[quant_matmul.affine_q" << bits
        << "_qmm_t] unsupported bits.";
    throw std::invalid_argument(msg.str());
  }
  if (group_size != 64 && group_size != 128) {
    std::ostringstream msg;
    msg << "[quant_matmul.affine_q" << bits
        << "_qmm_t] unsupported group_size " << group_size << ".";
    throw std::invalid_argument(msg.str());
  }

  if (x.ndim() < 2 || weight.ndim() != 2 || scales.ndim() != 2 ||
      biases.ndim() != 2) {
    std::ostringstream msg;
    msg << "[quant_matmul.affine_q" << bits
        << "_qmm_t] expected x [...,K], packed weight, "
        << "scales/biases [N,K/" << group_size << "], got " << x.shape()
        << ", " << weight.shape() << ", " << scales.shape() << ", "
        << biases.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  const int K = x.shape(-1);
  const int N = weight.shape(0);
  if (K <= 0 || N <= 0 || K % group_size != 0 ||
      !affine_qmm_packed_shape_matches(weight.shape(1), K, bits) ||
      scales.shape(0) != N || scales.shape(1) != K / group_size ||
      biases.shape() != scales.shape()) {
    std::ostringstream msg;
    msg << "[quant_matmul.affine_q" << bits
        << "_qmm_t] incompatible shapes: " << x.shape() << ", "
        << weight.shape() << ", " << scales.shape() << ", " << biases.shape()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[quant_matmul.affine_q" << bits
        << "_qmm_t] expected float16 or bfloat16 input, got "
        << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight.dtype() != uint32 || scales.dtype() != x.dtype() ||
      biases.dtype() != x.dtype()) {
    std::ostringstream msg;
    msg << "[quant_matmul.affine_q" << bits
        << "_qmm_t] expected uint32 weight and scale/bias dtype "
        << x.dtype() << ", got " << weight.dtype() << ", " << scales.dtype()
        << ", " << biases.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  if (AffineQmmTPrimitive::unsupported(
          x, weight, scales, biases, bits, variant, group_size, stream)) {
    throw std::invalid_argument(
        "[quant_matmul.affine_qmm_t] unsupported shape.");
  }

  // Demote rather than throwing when the NAX tile does not fit or the runtime
  // lacks tensor units / the separately built NAX metallib.
  bool nax = use_nax && is_nax_available() && nax_qmm_kernels_built() &&
      nax_qmm_runtime_ok.load(std::memory_order_relaxed);
  if (nax) {
    const auto nax_cfg = affine_qmm_nax_variant(nax_variant);
    if (K % nax_cfg.bk != 0 || N % nax_cfg.bn != 0) {
      nax = false;
    }
  }

  Shape out_shape = x.shape();
  out_shape.back() = N;
  std::vector<array> inputs = {x, weight, scales, biases};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<AffineQmmTPrimitive>(
          stream, bits, variant, nax, nax_variant, group_size),
      std::move(inputs));
}

array affine_q2_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    bool use_nax,
    int nax_variant,
    int group_size,
    StreamOrDevice s) {
  return affine_qmm_t(
      x, weight, scales, biases, 2, variant, use_nax, nax_variant, group_size, s);
}

array affine_q4_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    bool use_nax,
    int nax_variant,
    int group_size,
    StreamOrDevice s) {
  return affine_qmm_t(
      x, weight, scales, biases, 4, variant, use_nax, nax_variant, group_size, s);
}

array affine_q5_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    bool use_nax,
    int nax_variant,
    int group_size,
    StreamOrDevice s) {
  return affine_qmm_t(
      x, weight, scales, biases, 5, variant, use_nax, nax_variant, group_size, s);
}

array affine_q6_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    bool use_nax,
    int nax_variant,
    int group_size,
    StreamOrDevice s) {
  return affine_qmm_t(
      x, weight, scales, biases, 6, variant, use_nax, nax_variant, group_size, s);
}

array affine_q8_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    bool use_nax,
    int nax_variant,
    int group_size,
    StreamOrDevice s) {
  return affine_qmm_t(
      x, weight, scales, biases, 8, variant, use_nax, nax_variant, group_size, s);
}


} // namespace omlx::quant_matmul
