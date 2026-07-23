#include "sparse_mla.h"

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <sstream>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::glm_kernels {

namespace {

using namespace mlx::core;

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

struct GlmDsaSparseMlaParams {
  int B;
  int H;
  int qL;
  int kL;
  int topk;
  int topk_valid_prefix;
  int causal_prefix_indices;
  int has_topk_length;
  int causal_prefix_rows;

  float scale;
  int qL_off;

  int64_t Q_latent_strides[3];
  int64_t Q_pe_strides[3];
  int64_t KV_latent_strides[3];
  int64_t K_pe_strides[3];
  int64_t Topk_strides[3];
  int64_t TopkLength_strides[2];
  int64_t O_strides[3];
};

class GlmDsaSparseMlaAttentionPrimitive : public Primitive {
 public:
  GlmDsaSparseMlaAttentionPrimitive(
      Stream stream,
      float scale,
      bool do_causal,
      bool topk_valid_prefix,
      bool causal_prefix_indices,
      bool has_topk_length,
      int causal_prefix_rows)
      : Primitive(stream),
        scale_(scale),
        do_causal_(do_causal),
        topk_valid_prefix_(topk_valid_prefix),
        causal_prefix_indices_(causal_prefix_indices),
        has_topk_length_(has_topk_length),
        causal_prefix_rows_(causal_prefix_rows) {}

  static bool unsupported(
      const array& q_latent,
      const array& q_pe,
      const array& kv_latent,
      const array& k_pe,
      const array& topk_indices,
      const std::optional<array>& topk_length,
      bool topk_valid_prefix,
      bool causal_prefix_indices,
      int causal_prefix_rows,
      bool do_causal,
      Stream s) {
    if (s.device == Device::cpu || !do_causal) {
      return true;
    }
    if (q_latent.dtype() != q_pe.dtype() ||
        q_latent.dtype() != kv_latent.dtype() ||
        q_latent.dtype() != k_pe.dtype()) {
      return true;
    }
    if (q_latent.dtype() != float16 && q_latent.dtype() != bfloat16) {
      return true;
    }
    if (q_latent.ndim() != 4 || q_pe.ndim() != 4 || kv_latent.ndim() != 4 ||
        k_pe.ndim() != 4 || topk_indices.ndim() != 4) {
      return true;
    }
    if (!last_dim_contiguous(q_latent) || !last_dim_contiguous(q_pe) ||
        !last_dim_contiguous(kv_latent) || !last_dim_contiguous(k_pe) ||
        !last_dim_contiguous(topk_indices)) {
      return true;
    }
    if ((q_latent.shape(1) != 64 && q_latent.shape(1) != 32) ||
        kv_latent.shape(1) != 1 || k_pe.shape(1) != 1 ||
        topk_indices.shape(1) != 1) {
      return true;
    }
    if (q_latent.shape(3) != 512 || kv_latent.shape(3) != 512 ||
        q_pe.shape(3) != 64 || k_pe.shape(3) != 64) {
      return true;
    }
    if (q_latent.shape(2) <= 1 || topk_indices.shape(3) < 16 ||
        topk_indices.dtype() != uint32) {
      return true;
    }
    if (causal_prefix_rows < 0 || causal_prefix_rows > q_latent.shape(2)) {
      return true;
    }
    const bool compact_prefix =
        causal_prefix_rows > 0 && topk_indices.shape(2) != q_latent.shape(2);
    if (compact_prefix) {
      if (!causal_prefix_indices || !topk_valid_prefix ||
          topk_indices.shape(2) + causal_prefix_rows != q_latent.shape(2)) {
        return true;
      }
    } else if (topk_indices.shape(2) != q_latent.shape(2)) {
      return true;
    }
    if (topk_length.has_value() &&
        (topk_length->dtype() != uint32 ||
         (topk_length->ndim() != 2 && topk_length->ndim() != 3))) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "GlmDsaSparseMlaAttentionPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);

    const auto& q_latent = inputs[0];
    const auto& q_pe = inputs[1];
    const auto& kv_latent = inputs[2];
    const auto& k_pe = inputs[3];
    const auto& topk = inputs[4];
    const bool has_topk_length = inputs.size() > 5;
    const array& topk_length = has_topk_length ? inputs[5] : topk;
    auto& o = outputs[0];

    constexpr int bk = 256;
    constexpr int dc = 32;
    constexpr int d_latent = 512;
    constexpr int d_pe = 64;

    const int B = q_latent.shape(0);
    const int H = q_latent.shape(1);
    const int h = H;                  // head count selects the kernel instantiation
    const int wm = (H == 64) ? 8 : 4; // TQ = H / (wm * 8) must be >= 1: 64->8, 32->4
    const int qL = q_latent.shape(2);
    const int kL = kv_latent.shape(2);
    int64_t topk_length_strides[2] = {0, 0};
    if (has_topk_length) {
      topk_length_strides[0] = topk_length.strides(0);
      topk_length_strides[1] =
          topk_length.ndim() == 3 ? topk_length.strides(2)
                                  : topk_length.strides(1);
    }

    int64_t str_oD = 1;
    int64_t str_oL = o.shape(3);
    int64_t str_oH = o.shape(2) * str_oL;
    int64_t str_oB = o.shape(1) * str_oH;
    size_t data_size = o.shape(0) * str_oB;

    array::Flags flags{
        /* bool contiguous = */ 1,
        /* bool row_contiguous = */ 1,
        /* bool col_contiguous = */ 0,
    };

    o.set_data(
        allocator::malloc(o.nbytes()),
        data_size,
        {str_oB, str_oH, str_oL, str_oD},
        flags);

    const bool do_causal = do_causal_;
    metal::MTLFCList func_consts = {
        {&do_causal, MTL::DataType::DataTypeBool, 301},
    };

    std::string base_name;
    concatenate(
        base_name,
        "steel_sparse_mla_",
        type_to_name(q_latent),
        "_bk",
        bk,
        "_dc",
        dc,
        "_h",
        h,
        "_d",
        d_latent,
        "_pe",
        d_pe,
        "_wm",
        wm);

    std::string hash_name;
    concatenate(
        hash_name,
        base_name,
        "_do_causal_",
        (do_causal ? 't' : 'n'));

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto& compute_encoder = metal::get_command_encoder(s);
    auto kernel = d.get_kernel(base_name, lib, hash_name, func_consts);
    compute_encoder.set_compute_pipeline_state(kernel);

    GlmDsaSparseMlaParams params{
        /* int B = */ B,
        /* int H = */ H,
        /* int qL = */ qL,
        /* int kL = */ kL,
        /* int topk = */ topk.shape(3),
        /* int topk_valid_prefix = */ topk_valid_prefix_,
        /* int causal_prefix_indices = */ causal_prefix_indices_,
        /* int has_topk_length = */ has_topk_length,
        /* int causal_prefix_rows = */ causal_prefix_rows_,

        /* float scale = */ scale_,
        /* int qL_off = */ kL - qL,

        /* int64_t Q_latent_strides[3] = */ {
            q_latent.strides(0), q_latent.strides(1), q_latent.strides(2)},
        /* int64_t Q_pe_strides[3] = */ {
            q_pe.strides(0), q_pe.strides(1), q_pe.strides(2)},
        /* int64_t KV_latent_strides[3] = */ {
            kv_latent.strides(0),
            bcast_stride(kv_latent, 1),
            kv_latent.strides(2)},
        /* int64_t K_pe_strides[3] = */ {
            k_pe.strides(0), bcast_stride(k_pe, 1), k_pe.strides(2)},
        /* int64_t Topk_strides[3] = */ {
            topk.strides(0), bcast_stride(topk, 1), topk.strides(2)},
        /* int64_t TopkLength_strides[2] = */ {
            topk_length_strides[0], topk_length_strides[1]},
        /* int64_t O_strides[3] = */ {
            o.strides(0), o.strides(1), o.strides(2)}};

    compute_encoder.set_input_array(q_latent, 0);
    compute_encoder.set_input_array(q_pe, 1);
    compute_encoder.set_input_array(kv_latent, 2);
    compute_encoder.set_input_array(k_pe, 3);
    compute_encoder.set_input_array(topk, 4);
    compute_encoder.set_input_array(topk_length, 5);
    compute_encoder.set_output_array(o, 6);
    compute_encoder.set_bytes(params, 7);

    MTL::Size grid_dims = MTL::Size(qL, B, 1);
    MTL::Size group_dims = MTL::Size(32, wm, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(OMLXGlmDsaSparseMlaAttention)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const GlmDsaSparseMlaAttentionPrimitive&>(other);
    return scale_ == rhs.scale_ && do_causal_ == rhs.do_causal_ &&
        topk_valid_prefix_ == rhs.topk_valid_prefix_ &&
        causal_prefix_indices_ == rhs.causal_prefix_indices_ &&
        has_topk_length_ == rhs.has_topk_length_ &&
        causal_prefix_rows_ == rhs.causal_prefix_rows_;
  }
  auto state() const {
    return std::make_tuple(
        nullptr,
        scale_,
        do_causal_,
        topk_valid_prefix_,
        causal_prefix_indices_,
        has_topk_length_,
        causal_prefix_rows_);
  }

 private:
  float scale_;
  bool do_causal_;
  bool topk_valid_prefix_;
  bool causal_prefix_indices_;
  bool has_topk_length_;
  int causal_prefix_rows_;
};

} // namespace

array glm_dsa_sparse_mla_attention(
    const array& q_latent,
    const array& q_pe,
    const array& kv_latent,
    const array& k_pe,
    const array& topk_indices,
    float scale,
    bool causal,
    bool topk_valid_prefix,
    bool causal_prefix_indices,
    const std::optional<array>& topk_length,
    int causal_prefix_rows,
    StreamOrDevice s) {
  for (const auto& tensor : {q_latent, q_pe, kv_latent, k_pe}) {
    if (tensor.ndim() != 4) {
      std::ostringstream msg;
      msg << "[omlx_glm_kernels.glm_dsa_sparse_mla_attention] input with shape "
          << tensor.shape() << " expected to be rank 4.";
      throw std::invalid_argument(msg.str());
    }
  }
  if (topk_indices.ndim() != 4) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_dsa_sparse_mla_attention] topk_indices with "
        << "shape " << topk_indices.shape() << " expected to be rank 4.";
    throw std::invalid_argument(msg.str());
  }
  if (causal_prefix_rows < 0 || causal_prefix_rows > q_latent.shape(2)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_dsa_sparse_mla_attention] "
        << "causal_prefix_rows must be in [0, L], got "
        << causal_prefix_rows << " for L=" << q_latent.shape(2) << ".";
    throw std::invalid_argument(msg.str());
  }
  const bool compact_prefix_topk =
      causal_prefix_rows > 0 && topk_indices.shape(2) != q_latent.shape(2);
  const bool topk_length_ok = topk_indices.shape(2) == q_latent.shape(2) ||
      (compact_prefix_topk &&
       topk_indices.shape(2) + causal_prefix_rows == q_latent.shape(2) &&
       causal_prefix_indices && topk_valid_prefix);
  if (q_latent.shape(0) != q_pe.shape(0) ||
      q_latent.shape(0) != kv_latent.shape(0) ||
      q_latent.shape(0) != k_pe.shape(0) ||
      q_latent.shape(0) != topk_indices.shape(0) ||
      q_latent.shape(1) != q_pe.shape(1) || kv_latent.shape(1) != 1 ||
      k_pe.shape(1) != 1 || topk_indices.shape(1) != 1 ||
      q_latent.shape(2) != q_pe.shape(2) || !topk_length_ok ||
      kv_latent.shape(2) != k_pe.shape(2) ||
      q_latent.shape(3) != kv_latent.shape(3) ||
      q_pe.shape(3) != k_pe.shape(3)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_dsa_sparse_mla_attention] incompatible "
        << "shapes: " << q_latent.shape() << ", " << q_pe.shape() << ", "
        << kv_latent.shape() << ", " << k_pe.shape() << ", "
        << topk_indices.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (topk_indices.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_dsa_sparse_mla_attention] topk_indices must "
        << "be uint32, got " << topk_indices.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (topk_length.has_value()) {
    const auto& lengths = *topk_length;
    const bool rank_ok = lengths.ndim() == 2 || lengths.ndim() == 3;
    const bool shape_ok = rank_ok && lengths.shape(0) == q_latent.shape(0) &&
        lengths.shape(lengths.ndim() - 1) == q_latent.shape(2) &&
        (lengths.ndim() == 2 || lengths.shape(1) == 1);
    if (!shape_ok) {
      std::ostringstream msg;
      msg << "[omlx_glm_kernels.glm_dsa_sparse_mla_attention] topk_length "
          << "with shape " << lengths.shape() << " expected [B, L] or "
          << "[B, 1, L].";
      throw std::invalid_argument(msg.str());
    }
    if (lengths.dtype() != uint32) {
      std::ostringstream msg;
      msg << "[omlx_glm_kernels.glm_dsa_sparse_mla_attention] topk_length must "
          << "be uint32, got " << lengths.dtype() << ".";
      throw std::invalid_argument(msg.str());
    }
  }

  auto final_type =
      result_type(std::vector<array>{q_latent, q_pe, kv_latent, k_pe});
  if (final_type != float16 && final_type != bfloat16 && final_type != float32) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.glm_dsa_sparse_mla_attention] expected floating "
        << "inputs, got " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  auto ql = astype(q_latent, final_type, stream);
  auto qp = astype(q_pe, final_type, stream);
  auto kv = astype(kv_latent, final_type, stream);
  auto kp = astype(k_pe, final_type, stream);

  std::vector<array> inputs = {ql, qp, kv, kp, topk_indices};
  if (topk_length.has_value()) {
    inputs.push_back(*topk_length);
  }
  if (GlmDsaSparseMlaAttentionPrimitive::unsupported(
          ql,
          qp,
          kv,
          kp,
          topk_indices,
          topk_length,
          topk_valid_prefix,
          causal_prefix_indices,
          causal_prefix_rows,
          causal,
          stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.glm_dsa_sparse_mla_attention] unsupported M3 GLM shape.");
  }

  Shape out_shape{ql.shape(0), ql.shape(1), ql.shape(2), kv.shape(3)};
  return array(
      std::move(out_shape),
      final_type,
      std::make_shared<GlmDsaSparseMlaAttentionPrimitive>(
          stream,
          scale,
          causal,
          topk_valid_prefix,
          causal_prefix_indices,
          topk_length.has_value(),
          causal_prefix_rows),
      std::move(inputs));
}

} // namespace omlx::glm_kernels
