#include "deepseek_v4_sparse_attention.h"

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

class DeepseekV4SparseAttentionPrimitive : public Primitive {
 public:
  DeepseekV4SparseAttentionPrimitive(
      Stream stream,
      float scale,
      int q_offset,
      int compress_ratio,
      int local_window)
      : Primitive(stream),
        scale_(scale),
        q_offset_(q_offset),
        compress_ratio_(compress_ratio),
        local_window_(local_window) {}

  static bool unsupported(
      const array& q,
      const array& local_kv,
      const array& pooled,
      const array& topk_indices,
      const array& sinks,
      int q_offset,
      int compress_ratio,
      int local_window,
      Stream s) {
    if (s.device == Device::cpu) {
      return true;
    }
    if (q.dtype() != local_kv.dtype() || q.dtype() != pooled.dtype() ||
        q.dtype() != sinks.dtype()) {
      return true;
    }
    if (q.dtype() != float16 && q.dtype() != bfloat16) {
      return true;
    }
    if (q.ndim() != 4 || local_kv.ndim() != 4 || pooled.ndim() != 3 ||
        topk_indices.ndim() != 4 || sinks.ndim() != 1) {
      return true;
    }
    if (!last_dim_contiguous(q) || !last_dim_contiguous(local_kv) ||
        !last_dim_contiguous(pooled) || !last_dim_contiguous(topk_indices) ||
        !last_dim_contiguous(sinks)) {
      return true;
    }
    if (q.shape(0) != local_kv.shape(0) || q.shape(0) != pooled.shape(0) ||
        q.shape(0) != topk_indices.shape(0) || q.shape(1) != 64 ||
        q.shape(3) != 512 || local_kv.shape(1) != 1 ||
        local_kv.shape(3) != 512 || pooled.shape(2) != 512 ||
        topk_indices.shape(1) != 1 || topk_indices.shape(2) != q.shape(2) ||
        sinks.shape(0) != q.shape(1)) {
      return true;
    }
    if (q.shape(2) <= 1 || local_kv.shape(2) < q.shape(2) ||
        pooled.shape(1) <= 0 || topk_indices.shape(3) <= 0 ||
        topk_indices.dtype() != uint32) {
      return true;
    }
    if (q_offset < 0 || compress_ratio <= 0 || local_window <= 0) {
      return true;
    }
    return false;
  }

  void eval_cpu(
      const std::vector<array>& /* inputs */,
      std::vector<array>& /* outputs */) override {
    throw std::runtime_error(
        "DeepseekV4SparseAttentionPrimitive has no CPU path.");
  }

  void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) override {
    auto& s = stream();
    auto& d = metal::device(s.device);

    const auto& q = inputs[0];
    const auto& local_kv = inputs[1];
    const auto& pooled = inputs[2];
    const auto& topk = inputs[3];
    const auto& sinks = inputs[4];
    auto& o = outputs[0];

    constexpr int bk = 256;
    constexpr int dc = 32;
    constexpr int h = 64;
    constexpr int dim = 512;
    constexpr int wm = 8;

    const int B = q.shape(0);
    const int H = q.shape(1);
    const int qL = q.shape(2);
    const int localL = local_kv.shape(2);
    const int pooledL = pooled.shape(1);
    const int topkN = topk.shape(3);

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

    std::string base_name;
    concatenate(
        base_name,
        "deepseek_v4_sparse_attention_",
        type_to_name(q),
        "_bk",
        bk,
        "_dc",
        dc,
        "_h",
        h,
        "_d",
        dim,
        "_wm",
        wm);

    auto lib = d.get_library("omlx_glm_kernels", current_binary_dir());
    auto& compute_encoder = metal::get_command_encoder(s);
    auto kernel = d.get_kernel(base_name, lib);
    compute_encoder.set_compute_pipeline_state(kernel);

    DeepseekV4SparseAttentionParams params{
        /* int B = */ B,
        /* int H = */ H,
        /* int qL = */ qL,
        /* int localL = */ localL,
        /* int pooledL = */ pooledL,
        /* int topk = */ topkN,
        /* int local_window = */ local_window_,
        /* int compress_ratio = */ compress_ratio_,
        /* int q_offset = */ q_offset_,

        /* float scale = */ scale_,

        /* int64_t Q_strides[3] = */ {
            q.strides(0), q.strides(1), q.strides(2)},
        /* int64_t Local_strides[3] = */ {
            local_kv.strides(0),
            bcast_stride(local_kv, 1),
            local_kv.strides(2)},
        /* int64_t Pooled_strides[2] = */ {
            pooled.strides(0), pooled.strides(1)},
        /* int64_t Topk_strides[3] = */ {
            topk.strides(0), bcast_stride(topk, 1), topk.strides(2)},
        /* int64_t O_strides[3] = */ {
            o.strides(0), o.strides(1), o.strides(2)}};

    compute_encoder.set_input_array(q, 0);
    compute_encoder.set_input_array(local_kv, 1);
    compute_encoder.set_input_array(pooled, 2);
    compute_encoder.set_input_array(topk, 3);
    compute_encoder.set_input_array(sinks, 4);
    compute_encoder.set_output_array(o, 5);
    compute_encoder.set_bytes(params, 6);

    MTL::Size grid_dims = MTL::Size(qL, B, 1);
    MTL::Size group_dims = MTL::Size(32, wm, 1);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  DEFINE_NAME(OMLXDeepseekV4SparseAttention)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs =
        static_cast<const DeepseekV4SparseAttentionPrimitive&>(other);
    return scale_ == rhs.scale_ && q_offset_ == rhs.q_offset_ &&
        compress_ratio_ == rhs.compress_ratio_ &&
        local_window_ == rhs.local_window_;
  }
  auto state() const {
    return std::make_tuple(
        nullptr, scale_, q_offset_, compress_ratio_, local_window_);
  }

 private:
  float scale_;
  int q_offset_;
  int compress_ratio_;
  int local_window_;
};

} // namespace

array deepseek_v4_sparse_attention(
    const array& q,
    const array& local_kv,
    const array& pooled,
    const array& topk_indices,
    const array& sinks,
    float scale,
    int q_offset,
    int compress_ratio,
    int local_window,
    StreamOrDevice s) {
  if (q.ndim() != 4 || local_kv.ndim() != 4 || pooled.ndim() != 3 ||
      topk_indices.ndim() != 4 || sinks.ndim() != 1) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_v4_sparse_attention] incompatible "
        << "ranks: " << q.shape() << ", " << local_kv.shape() << ", "
        << pooled.shape() << ", " << topk_indices.shape() << ", "
        << sinks.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (q.shape(0) != local_kv.shape(0) || q.shape(0) != pooled.shape(0) ||
      q.shape(0) != topk_indices.shape(0) || q.shape(1) != 64 ||
      q.shape(3) != 512 || local_kv.shape(1) != 1 ||
      local_kv.shape(3) != 512 || pooled.shape(2) != 512 ||
      topk_indices.shape(1) != 1 || topk_indices.shape(2) != q.shape(2) ||
      sinks.shape(0) != q.shape(1)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_v4_sparse_attention] incompatible "
        << "shapes: " << q.shape() << ", " << local_kv.shape() << ", "
        << pooled.shape() << ", " << topk_indices.shape() << ", "
        << sinks.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (topk_indices.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_v4_sparse_attention] topk_indices "
        << "must be uint32, got " << topk_indices.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto final_type = result_type(std::vector<array>{q, local_kv, pooled});
  if (final_type != float16 && final_type != bfloat16) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.deepseek_v4_sparse_attention] expected fp16 or "
        << "bf16 inputs, got " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto stream = to_stream(s);
  auto q_cast = astype(q, final_type, stream);
  auto local_cast = astype(local_kv, final_type, stream);
  auto pooled_cast = astype(pooled, final_type, stream);
  auto sinks_cast = astype(sinks, final_type, stream);

  if (DeepseekV4SparseAttentionPrimitive::unsupported(
          q_cast,
          local_cast,
          pooled_cast,
          topk_indices,
          sinks_cast,
          q_offset,
          compress_ratio,
          local_window,
          stream)) {
    throw std::invalid_argument(
        "[omlx_glm_kernels.deepseek_v4_sparse_attention] unsupported DeepSeek V4 sparse attention shape.");
  }

  Shape out_shape{q_cast.shape(0), q_cast.shape(1), q_cast.shape(2), q_cast.shape(3)};
  std::vector<array> inputs = {
      q_cast, local_cast, pooled_cast, topk_indices, sinks_cast};
  return array(
      std::move(out_shape),
      final_type,
      std::make_shared<DeepseekV4SparseAttentionPrimitive>(
          stream, scale, q_offset, compress_ratio, local_window),
      std::move(inputs));
}

} // namespace omlx::glm_kernels
