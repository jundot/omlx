#include "exact_block_attention.h"

#include <dlfcn.h>
#include <filesystem>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::glm_kernels {

namespace {

using namespace mlx::core;
using namespace mlx::steel;

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

inline int64_t bcast_stride(const array& a, int axis) {
  return a.shape(axis) == 1 ? 0 : a.strides(axis);
}

bool last_dim_contiguous(const array& arr) {
  return arr.strides(-1) == 1;
}

class GlmDsaExactBlockAttentionPrimitive : public Primitive {
 public:
  GlmDsaExactBlockAttentionPrimitive(Stream stream, float scale, bool causal)
      : Primitive(stream), scale_(scale), causal_(causal) {}

  static bool unsupported(
      const array& q,
      const array& k,
      const array& v,
      const array& block_mask,
      const array& block_token_mask,
      bool causal,
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
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 ||
        block_mask.ndim() != 4 || block_token_mask.ndim() != 4) {
      return true;
    }
    if (!last_dim_contiguous(q) || !last_dim_contiguous(k) ||
        !last_dim_contiguous(v) || !last_dim_contiguous(block_mask) ||
        !last_dim_contiguous(block_token_mask)) {
      return true;
    }
    if (q.shape(0) != k.shape(0) || q.shape(0) != v.shape(0) ||
        k.shape(0) != v.shape(0) || q.shape(1) % k.shape(1) != 0 ||
        k.shape(1) != v.shape(1) || k.shape(2) != v.shape(2) ||
        q.shape(3) != k.shape(3) || q.shape(3) != v.shape(3) ||
        q.shape(3) != 256) {
      return true;
    }
    if (block_mask.dtype() != bool_ || block_token_mask.dtype() != uint32) {
      return true;
    }

    const int qL = q.shape(2);
    const int kL = k.shape(2);
    const int q_blocks16 = (qL + 15) / 16;
    const int q_blocks32 = (qL + 31) / 32;
    const int k_blocks8 = (kL + 7) / 8;
    const int k_blocks16 = (kL + 15) / 16;
    const bool q_block_ok =
        block_mask.shape(-2) == q_blocks16 || block_mask.shape(-2) == q_blocks32;
    const bool k_block_ok = block_mask.shape(-1) == k_blocks8 ||
        block_mask.shape(-1) == k_blocks16;
    if (!q_block_ok || !k_block_ok ||
        block_token_mask.shape(-2) != qL ||
        block_token_mask.shape(-1) != block_mask.shape(-1)) {
      return true;
    }
    if ((block_mask.shape(0) != 1 && block_mask.shape(0) != q.shape(0)) ||
        (block_mask.shape(1) != 1 && block_mask.shape(1) != q.shape(1)) ||
        (block_token_mask.shape(0) != 1 &&
         block_token_mask.shape(0) != q.shape(0)) ||
        (block_token_mask.shape(1) != 1 &&
         block_token_mask.shape(1) != q.shape(1))) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error("GlmDsaExactBlockAttentionPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);

    const auto& q = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    const auto& block_mask = inputs[3];
    const auto& block_token_mask = inputs[4];
    auto& o = outputs[0];

    int wm = 4;
    constexpr int wn = 1;
    const int bd = q.shape(-1);
    int bk = block_token_mask.shape(-1) == (k.shape(2) + 7) / 8 ? 8 : 16;
    int bq = 32;
    if (block_mask.shape(-2) == (q.shape(2) + 15) / 16) {
      bq = 16;
      wm = 2;
    }

    const int B = q.shape(0);
    const int H = q.shape(1);
    const int qL = q.shape(2);
    const int kL = k.shape(2);
    const int gqa_factor = q.shape(1) / k.shape(1);

    const bool align_Q = (qL % bq) == 0;
    const bool align_K = (kL % bk) == 0;
    const bool has_mask = false;
    const bool has_sinks = false;
    const bool has_block_mask = true;
    const bool has_block_token_mask = true;
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
        "omlx_glm_exact_attention_",
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
        base_name,
        "_align_Q_",
        (align_Q ? 't' : 'n'),
        "_align_K_",
        (align_K ? 't' : 'n'),
        "_has_mask_n_do_causal_",
        (do_causal ? 't' : 'n'),
        "_has_sinks_n_has_block_mask_t_has_block_token_mask_t_has_block_indices_n");

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

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
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
    AttnBlockMaskParams block_mask_params{/* int64_t BM_strides[3] = */ {
        bcast_stride(block_mask, 0),
        bcast_stride(block_mask, 1),
        bcast_stride(block_mask, 2)}};
    AttnBlockTokenMaskParams block_token_mask_params{
        /* int64_t BTM_strides[3] = */ {
            bcast_stride(block_token_mask, 0),
            bcast_stride(block_token_mask, 1),
            bcast_stride(block_token_mask, 2)}};

    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(k, 1);
    compute_encoder.set_input_array(v, 2);
    compute_encoder.set_output_array(o, 3);
    compute_encoder.set_bytes(params, 4);
    compute_encoder.set_bytes(block_mask_params, 8);
    compute_encoder.set_input_array(block_mask, 9);
    compute_encoder.set_bytes(block_token_mask_params, 10);
    compute_encoder.set_input_array(block_token_mask, 11);

    MTL::Size grid_dims = MTL::Size(NQ, H, B);
    MTL::Size group_dims = MTL::Size(32, wm, wn);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(OMLXGlmDsaExactBlockAttention)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const GlmDsaExactBlockAttentionPrimitive&>(other);
    return scale_ == rhs.scale_ && causal_ == rhs.causal_;
  }
  auto state() const {
    return std::make_tuple(nullptr, scale_, causal_);
  }

 private:
  float scale_;
  bool causal_;
};

} // namespace

array glm_dsa_exact_block_attention(
    const array& q,
    const array& k,
    const array& v,
    const array& block_mask,
    const array& block_token_mask,
    float scale,
    bool causal,
    StreamOrDevice s) {
  for (const auto& tensor : {q, k, v}) {
    if (tensor.ndim() != 4) {
      std::ostringstream msg;
      msg << "[omlx_glm_kernels.glm_dsa_exact_block_attention] input with "
          << "shape " << tensor.shape() << " expected rank 4.";
      throw std::invalid_argument(msg.str());
    }
  }
  auto stream = to_stream(s);
  auto final_type = result_type(std::vector<array>{q, k, v});
  if (final_type != float16 && final_type != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_dsa_exact_block_attention] expected fp16 "
        << "or bf16 inputs, got " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto q_cast = astype(q, final_type, stream);
  auto k_cast = astype(k, final_type, stream);
  auto v_cast = astype(v, final_type, stream);
  if (GlmDsaExactBlockAttentionPrimitive::unsupported(
          q_cast, k_cast, v_cast, block_mask, block_token_mask, causal, stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.glm_dsa_exact_block_attention] unsupported GLM exact block SDPA shape.");
  }

  Shape out_shape{q_cast.shape(0), q_cast.shape(1), q_cast.shape(2), v_cast.shape(3)};
  std::vector<array> inputs = {q_cast, k_cast, v_cast, block_mask, block_token_mask};
  return array(
      std::move(out_shape),
      final_type,
      std::make_shared<GlmDsaExactBlockAttentionPrimitive>(
          stream, scale, causal),
      std::move(inputs));
}

} // namespace omlx::glm_kernels
