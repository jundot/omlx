#include "qwen35_prefill.h"

#include <dlfcn.h>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::qwen35_prefill_kernels {

namespace {

using namespace mlx::core;
using namespace mlx::steel;

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get omlx_qwen35_prefill binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

bool last_dim_contiguous(const array& arr) {
  return arr.strides(-1) == 1;
}

bool row_contiguous(const array& arr) {
  return arr.flags().row_contiguous && arr.strides(-1) == 1;
}

std::string qwen_type_name(Dtype dtype) {
  if (dtype == float16) {
    return "float16_t";
  }
  if (dtype == bfloat16) {
    return "bfloat16_t";
  }
  std::ostringstream msg;
  msg << "Unsupported Qwen prefill kernel dtype: " << dtype << ".";
  throw std::invalid_argument(msg.str());
}

struct QwenQAffineVariant {
  int bm;
  int bk;
  int bn;
};

bool qwen_q_affine_bits_supported(int bits) {
  return bits == 4 || bits == 5 || bits == 6 || bits == 8;
}

bool qwen_q_affine_packed_shape_matches(int packed_dim, int K, int bits) {
  return K > 0 && packed_dim > 0 &&
      static_cast<int64_t>(packed_dim) * 32 == static_cast<int64_t>(K) * bits;
}

QwenQAffineVariant qwen_q_affine_variant(int variant) {
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
      msg << "Unsupported Qwen affine qmm variant " << variant << ".";
      throw std::invalid_argument(msg.str());
    }
  }
}

class Qwen35Fa256AttentionPrimitive : public Primitive {
 public:
  Qwen35Fa256AttentionPrimitive(
      Stream stream,
      float scale,
      bool causal,
      int q_block,
      int k_block)
      : Primitive(stream),
        scale_(scale),
        causal_(causal),
        q_block_(q_block),
        k_block_(k_block) {}

  static bool unsupported(
      const array& q,
      const array& k,
      const array& v,
      bool causal,
      int q_block,
      int k_block,
      Stream s) {
    if (s.device == Device::cpu || !causal) {
      return true;
    }
    if (q.dtype() != k.dtype() || q.dtype() != v.dtype()) {
      return true;
    }
    if (q.dtype() != float16 && q.dtype() != bfloat16) {
      return true;
    }
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4) {
      return true;
    }
    if (!last_dim_contiguous(q) || !last_dim_contiguous(k) ||
        !last_dim_contiguous(v)) {
      return true;
    }
    if (!((q_block == 16 || q_block == 32) &&
          (k_block == 8 || k_block == 16))) {
      return true;
    }
    if (q.shape(0) != k.shape(0) || q.shape(0) != v.shape(0) ||
        k.shape(0) != v.shape(0) || q.shape(1) % k.shape(1) != 0 ||
        k.shape(1) != v.shape(1) || k.shape(2) != v.shape(2) ||
        q.shape(2) > k.shape(2) || q.shape(2) <= 1 ||
        q.shape(3) != k.shape(3) || q.shape(3) != v.shape(3) ||
        q.shape(3) != 256) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("Qwen35Fa256AttentionPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);

    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    auto& o = outputs[0];

    const int bq = q_block_;
    const int bk = k_block_;
    const int wm = bq == 16 ? 2 : 4;
    constexpr int wn = 1;
    const int bd = q.shape(-1);

    const int B = q.shape(0);
    const int H = q.shape(1);
    const int qL = q.shape(2);
    const int kL = k.shape(2);
    const int gqa_factor = q.shape(1) / k.shape(1);

    const bool align_Q = (qL % bq) == 0;
    const bool align_K = (kL % bk) == 0;
    const bool has_mask = false;
    const bool has_sinks = false;
    const bool has_block_mask = false;
    const bool has_block_token_mask = false;
    const bool has_block_indices = false;
    const bool do_causal = causal_;

    metal::MTLFCList func_consts = {
        {&align_Q, MTL::DataType::DataTypeBool, 200},
        {&align_K, MTL::DataType::DataTypeBool, 201},
        {&has_mask, MTL::DataType::DataTypeBool, 300},
        {&do_causal, MTL::DataType::DataTypeBool, 301},
        {&has_sinks, MTL::DataType::DataTypeBool, 302},
        {&has_block_mask, MTL::DataType::DataTypeBool, 303},
        {&has_block_token_mask, MTL::DataType::DataTypeBool, 304},
        {&has_block_indices, MTL::DataType::DataTypeBool, 305}};

    std::string base_name;
    concatenate(
        base_name,
        "omlx_qwen35_fa256_attention_",
        type_to_name(q),
        "_bq",
        bq,
        "_bk",
        bk,
        "_bd",
        bd,
        "_wm",
        wm,
        "_wn",
        wn,
        "_mask",
        type_to_name(q));

    std::string hash_name;
    concatenate(
        hash_name,
        "omlx_qwen35_fa256_",
        type_to_name(q),
        "_bq",
        bq,
        "_bk",
        bk,
        "_bd",
        bd,
        "_align_Q_",
        (align_Q ? 't' : 'n'),
        "_align_K_",
        (align_K ? 't' : 'n'),
        "_causal_",
        (do_causal ? 't' : 'n'));

    int64_t str_oD = 1;
    int64_t str_oH = o.shape(3);
    int64_t str_oL = o.shape(1) * str_oH;
    int64_t str_oB = o.shape(2) * str_oL;
    size_t data_size = o.shape(0) * str_oB;
    array::Flags flags{
        /* bool contiguous = */ 1,
        /* bool row_contiguous = */ 0,
        /* bool col_contiguous = */ 0,
    };
    o.set_data(
        allocator::malloc(o.nbytes()),
        data_size,
        {str_oB, str_oH, str_oL, str_oD},
        flags);

    auto lib = d.get_library("omlx_qwen35_prefill_kernels", current_binary_dir());
    auto& compute_encoder = metal::get_command_encoder(s);
    auto kernel = d.get_kernel(base_name, lib, hash_name, func_consts);
    compute_encoder.set_compute_pipeline_state(kernel);

    const int NQ = (qL + bq - 1) / bq;
    const int NK = (kL + bk - 1) / bk;
    const int NQ_aligned = qL / bq;
    const int NK_aligned = kL / bk;

    AttnParams params{
        /* int B = */ B,
        /* int H = */ H,
        /* int D = */ bd,
        /* int qL = */ qL,
        /* int kL = */ kL,
        /* int gqa_factor = */ gqa_factor,
        /* float scale = */ scale_,
        /* int NQ = */ NQ,
        /* int NK = */ NK,
        /* int NQ_aligned = */ NQ_aligned,
        /* int NK_aligned = */ NK_aligned,
        /* int qL_rem = */ (qL - NQ_aligned * bq),
        /* int kL_rem = */ (kL - NK_aligned * bk),
        /* int qL_off = */ (kL - qL),
        /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
        /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
        /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
        /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)}};

    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_output_array(o, 3);
    compute_encoder.set_bytes(params, 4);

    MTL::Size grid_dims = MTL::Size(NQ, H, B);
    MTL::Size group_dims = MTL::Size(32, wm, wn);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(OMLXQwen35Fa256Attention)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Qwen35Fa256AttentionPrimitive&>(other);
    return scale_ == rhs.scale_ && causal_ == rhs.causal_ &&
        q_block_ == rhs.q_block_ && k_block_ == rhs.k_block_;
  }
  auto state() const {
    return std::make_tuple(nullptr, scale_, causal_, q_block_, k_block_);
  }

 private:
  float scale_;
  bool causal_;
  int q_block_;
  int k_block_;
};

class Qwen35QAffineQmmTPrimitive : public Primitive {
 public:
  Qwen35QAffineQmmTPrimitive(Stream stream, int bits, int variant)
      : Primitive(stream), bits_(bits), variant_(variant) {
    if (!qwen_q_affine_bits_supported(bits_)) {
      std::ostringstream msg;
      msg << "Unsupported Qwen affine qmm bits " << bits_ << ".";
      throw std::invalid_argument(msg.str());
    }
    (void)qwen_q_affine_variant(variant_);
  }

  static bool unsupported(
      const array& x,
      const array& weight,
      const array& scales,
      const array& biases,
      int bits,
      int variant,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (!qwen_q_affine_bits_supported(bits)) {
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

    constexpr int group_size = 64;
    const auto cfg = qwen_q_affine_variant(variant);
    const int K = x.shape(-1);
    const int N = weight.shape(0);
    if (K <= 0 || N <= 0 || x.size() <= 0 || K % group_size != 0 ||
        K % cfg.bk != 0 || N % cfg.bn != 0) {
      return true;
    }
    if (!qwen_q_affine_packed_shape_matches(weight.shape(1), K, bits) ||
        scales.shape(0) != N || scales.shape(1) != K / group_size ||
        biases.shape() != scales.shape()) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("Qwen35QAffineQmmTPrimitive has no CPU path.");
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

    const auto cfg = qwen_q_affine_variant(variant_);
    const int K = x.shape(-1);
    const int N = weight.shape(0);
    const int M = x.size() / K;

    std::string kname;
    concatenate(
        kname,
        "qwen35_q",
        bits_,
        "_affine_qmm_t_",
        qwen_type_name(x.dtype()),
        "_bm_",
        cfg.bm,
        "_bk_",
        cfg.bk,
        "_bn_",
        cfg.bn);

    auto lib = d.get_library("omlx_qwen35_prefill_kernels", current_binary_dir());
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

    MTL::Size grid_dims(
        (N + cfg.bn - 1) / cfg.bn, (M + cfg.bm - 1) / cfg.bm, 1);
    MTL::Size group_dims(32, 2, 2);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(Qwen35QAffineQmmTPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const Qwen35QAffineQmmTPrimitive&>(other);
    return bits_ == rhs.bits_ && variant_ == rhs.variant_;
  }
  auto state() const {
    return std::make_tuple(bits_, variant_);
  }

 private:
  int bits_;
  int variant_;
};

class Qwen35MoeWeightedSumPrimitive : public Primitive {
 public:
  explicit Qwen35MoeWeightedSumPrimitive(Stream stream) : Primitive(stream) {}

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
    const int topk = scores.shape(-1);
    if ((topk != 6 && topk != 8) || x_sorted.shape(0) != scores.size() ||
        inv_order.size() != scores.size()) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("Qwen35MoeWeightedSumPrimitive has no CPU path.");
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

    constexpr bool use_tiled = true;
    constexpr int tiled_threads = 256;
    const int vec = (D % 4 == 0) ? 4 : 1;

    std::string kname;
    if (use_tiled) {
      concatenate(
          kname,
          "moe_weighted_sum_tiled_",
          qwen_type_name(x_sorted.dtype()),
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
          qwen_type_name(x_sorted.dtype()),
          "_score_float_topk_",
          topk);
    }

    auto lib = d.get_library("omlx_qwen35_prefill_kernels", current_binary_dir());
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

  DEFINE_NAME(Qwen35MoeWeightedSumPrimitive)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& /* other */) const override {
    return true;
  }
  auto state() const {
    return std::make_tuple(nullptr);
  }
};

} // namespace

array qwen35_fa256_attention(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    bool causal,
    int q_block,
    int k_block,
    StreamOrDevice s) {
  for (const auto& tensor : {q, k, v}) {
    if (tensor.ndim() != 4) {
      std::ostringstream msg;
      msg << "[omlx_qwen35_prefill.qwen35_fa256_attention] input with shape "
          << tensor.shape() << " expected rank 4.";
      throw std::invalid_argument(msg.str());
    }
  }
  auto stream = to_stream(s);
  auto final_type = result_type(std::vector<array>{q, k, v});
  if (final_type != float16 && final_type != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_fa256_attention] expected fp16 or "
        << "bf16 inputs, got " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto q_cast = astype(q, final_type, stream);
  auto k_cast = astype(k, final_type, stream);
  auto v_cast = astype(v, final_type, stream);
  if (Qwen35Fa256AttentionPrimitive::unsupported(
          q_cast, k_cast, v_cast, causal, q_block, k_block, stream)) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_fa256_attention] unsupported Qwen FA-256 shape.");
  }

  Shape out_shape{
      q_cast.shape(0), q_cast.shape(1), q_cast.shape(2), v_cast.shape(3)};
  std::vector<array> inputs = {q_cast, k_cast, v_cast};
  return array(
      std::move(out_shape),
      final_type,
      std::make_shared<Qwen35Fa256AttentionPrimitive>(
          stream, scale, causal, q_block, k_block),
      std::move(inputs));
}

array qwen35_q_affine_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int bits,
    int variant,
    StreamOrDevice s) {
  (void)qwen_q_affine_variant(variant);
  if (!qwen_q_affine_bits_supported(bits)) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_q" << bits
        << "_affine_qmm_t] unsupported bits.";
    throw std::invalid_argument(msg.str());
  }

  if (x.ndim() < 2 || weight.ndim() != 2 || scales.ndim() != 2 ||
      biases.ndim() != 2) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_q" << bits
        << "_affine_qmm_t] expected x [...,K], packed weight, "
        << "scales/biases [N,K/64], got " << x.shape() << ", "
        << weight.shape() << ", " << scales.shape() << ", "
        << biases.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  constexpr int group_size = 64;
  const int K = x.shape(-1);
  const int N = weight.shape(0);
  if (K <= 0 || N <= 0 || K % group_size != 0 ||
      !qwen_q_affine_packed_shape_matches(weight.shape(1), K, bits) ||
      scales.shape(0) != N || scales.shape(1) != K / group_size ||
      biases.shape() != scales.shape()) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_q" << bits
        << "_affine_qmm_t] incompatible shapes: " << x.shape() << ", "
        << weight.shape() << ", " << scales.shape() << ", " << biases.shape()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_q" << bits
        << "_affine_qmm_t] expected float16 or bfloat16 input, got "
        << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (weight.dtype() != uint32 || scales.dtype() != x.dtype() ||
      biases.dtype() != x.dtype()) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_q" << bits
        << "_affine_qmm_t] expected uint32 weight and scale/bias dtype "
        << x.dtype() << ", got " << weight.dtype() << ", " << scales.dtype()
        << ", " << biases.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  if (Qwen35QAffineQmmTPrimitive::unsupported(
          x, weight, scales, biases, bits, variant, stream)) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_q_affine_qmm_t] unsupported shape.");
  }

  Shape out_shape = x.shape();
  out_shape.back() = N;
  std::vector<array> inputs = {x, weight, scales, biases};
  return array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<Qwen35QAffineQmmTPrimitive>(stream, bits, variant),
      std::move(inputs));
}

array qwen35_q4_affine_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    StreamOrDevice s) {
  return qwen35_q_affine_qmm_t(x, weight, scales, biases, 4, variant, s);
}

array qwen35_q5_affine_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    StreamOrDevice s) {
  return qwen35_q_affine_qmm_t(x, weight, scales, biases, 5, variant, s);
}

array qwen35_q6_affine_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    StreamOrDevice s) {
  return qwen35_q_affine_qmm_t(x, weight, scales, biases, 6, variant, s);
}

array qwen35_q8_affine_qmm_t(
    const array& x,
    const array& weight,
    const array& scales,
    const array& biases,
    int variant,
    StreamOrDevice s) {
  return qwen35_q_affine_qmm_t(x, weight, scales, biases, 8, variant, s);
}

array qwen35_moe_weighted_sum(
    const array& x_sorted,
    const array& inv_order,
    const array& scores,
    StreamOrDevice s) {
  if (x_sorted.ndim() != 3 || x_sorted.shape(-2) != 1) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] expected "
        << "x_sorted shape [N, 1, D], got " << x_sorted.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (scores.ndim() < 2) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] expected scores "
        << "rank >= 2, got " << scores.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (inv_order.ndim() != 1 || inv_order.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] expected uint32 "
        << "inv_order rank 1, got " << inv_order.shape() << " dtype "
        << inv_order.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  const int topk = scores.shape(-1);
  const int64_t routed_rows = scores.size();
  const int D = x_sorted.shape(-1);
  if (x_sorted.shape(0) != routed_rows || inv_order.size() != routed_rows) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] incompatible "
        << "shapes: " << x_sorted.shape() << ", " << inv_order.shape()
        << ", " << scores.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (topk <= 0 || D <= 0) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] invalid topk or "
        << "hidden dim: topk=" << topk << ", D=" << D << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(x_sorted.dtype(), floating)) {
    std::ostringstream msg;
    msg << "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] expected floating "
        << "x_sorted, got " << x_sorted.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  std::vector<array> inputs = {x_sorted, inv_order, scores};
  Shape out_shape = scores.shape();
  out_shape.pop_back();
  out_shape.push_back(D);
  if (Qwen35MoeWeightedSumPrimitive::unsupported(
          x_sorted, inv_order, scores, stream)) {
    throw std::invalid_argument(
        "[omlx_qwen35_prefill.qwen35_moe_weighted_sum] unsupported Qwen shape.");
  }
  return array(
      std::move(out_shape),
      x_sorted.dtype(),
      std::make_shared<Qwen35MoeWeightedSumPrimitive>(stream),
      std::move(inputs));
}

} // namespace omlx::qwen35_prefill_kernels
