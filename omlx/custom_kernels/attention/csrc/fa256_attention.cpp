#include "fa256_attention.h"
#include "attn_params.h"
#include "omlx/binary_dir.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/backend/common/utils.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

namespace omlx::attention {

namespace {

using namespace mlx::core;
using namespace mlx::steel;

bool last_dim_contiguous(const array& arr) {
  return arr.strides(-1) == 1;
}

bool row_contiguous(const array& arr) {
  return arr.flags().row_contiguous && arr.strides(-1) == 1;
}

class Fa256AttentionPrimitive : public Primitive {
 public:
  Fa256AttentionPrimitive(
      Stream stream,
      float scale,
      bool causal,
      int q_block,
      int k_block,
      int64_t dispatch_budget)
      : Primitive(stream),
        scale_(scale),
        causal_(causal),
        q_block_(q_block),
        k_block_(k_block),
        dispatch_budget_(dispatch_budget) {}

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
    throw std::runtime_error("Fa256AttentionPrimitive has no CPU path.");
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

    auto lib = d.get_library("omlx_attention_kernels", binary_dir());
    auto& compute_encoder = metal::get_command_encoder(s);

    const int NQ = (qL + bq - 1) / bq;
    const int NQ_aligned = qL / bq;

    MTL::Size grid_dims = MTL::Size(NQ, H, B);
    MTL::Size group_dims = MTL::Size(32, wm, wn);

    // The kernel scans its whole key range inside one Metal dispatch, so the
    // per-dispatch wallclock grows linearly with kL. Past the macOS IOGPU
    // interactivity threshold the OS demotes (or kills) the command buffer
    // and long-context prefill collapses on pre-NAX GPUs (issue #2225,
    // mlx#3302). Bound the per-dispatch work by splitting the keys into
    // chunks of at most chunk_keys, each its own preemptible dispatch, and
    // fold the partials with logsumexp weights afterwards (mlx#3307).
    int64_t chunk_keys = kL;
    if (dispatch_budget_ > 0) {
      const int64_t work = int64_t(B) * H * qL * kL;
      if (work > dispatch_budget_) {
        // Very short chunks would re-dispatch the full query grid per sliver
        // of keys; 4 * bq keys is plenty to amortize the dead threadgroups.
        const int64_t min_chunk_keys = 4LL * bq;
        // The partial slab costs B*H*qL*D per chunk, so huge-qL calls (one
        // shot square prefill) cap the chunk count on memory instead of
        // honoring the dispatch budget exactly.
        const int64_t max_slab_bytes = 2LL << 30;
        const int64_t chunk_bytes = int64_t(B) * H * qL * bd * q.itemsize();
        const int64_t n_mem_cap =
            std::max<int64_t>(1, max_slab_bytes / std::max<int64_t>(chunk_bytes, 1));
        int64_t n_target = (work + dispatch_budget_ - 1) / dispatch_budget_;
        n_target = std::min(n_target, n_mem_cap);
        chunk_keys = (kL + n_target - 1) / n_target;
        chunk_keys = ((chunk_keys + bk - 1) / bk) * bk; // align to K tile
        chunk_keys = std::max(chunk_keys, min_chunk_keys);
      }
    }

    const int n_chunks = int((kL + chunk_keys - 1) / chunk_keys);

    if (n_chunks <= 1) {
      const int NK = (kL + bk - 1) / bk;
      const int NK_aligned = kL / bk;

      auto kernel = d.get_kernel(base_name, lib, hash_name, func_consts);
      compute_encoder.set_compute_pipeline_state(kernel);

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
          /* int64_t Q_strides[3] = */
          {q.strides(0), q.strides(1), q.strides(2)},
          /* int64_t K_strides[3] = */
          {k.strides(0), k.strides(1), k.strides(2)},
          /* int64_t V_strides[3] = */
          {v.strides(0), v.strides(1), v.strides(2)},
          /* int64_t O_strides[3] = */
          {o.strides(0), o.strides(1), o.strides(2)}};

      compute_encoder.set_input_array(q, 0);
      compute_encoder.set_input_array(k, 1);
      compute_encoder.set_input_array(v, 2);
      compute_encoder.set_output_array(o, 3);
      compute_encoder.set_bytes(params, 4);
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
      return;
    }

    // Chunked path: per-chunk normalized partials (input dtype) plus fp32
    // logsumexp rows, folded by the reduce kernel below.
    const int64_t o_chunk_stride = int64_t(B) * H * qL * bd;
    const int64_t lse_chunk_stride = int64_t(B) * H * qL;

    array o_part(
        {n_chunks, B, H, qL, bd}, o.dtype(), nullptr, std::vector<array>{});
    o_part.set_data(allocator::malloc(o_part.nbytes()));
    array lse_part(
        {n_chunks, B, H, qL}, float32, nullptr, std::vector<array>{});
    lse_part.set_data(allocator::malloc(lse_part.nbytes()));
    compute_encoder.add_temporary(o_part);
    compute_encoder.add_temporary(lse_part);

    const bool partials_true = true;
    for (int c = 0; c < n_chunks; ++c) {
      const int64_t k_start = int64_t(c) * chunk_keys;
      const int kL_c = int(std::min<int64_t>(chunk_keys, kL - k_start));
      const int NK_c = (kL_c + bk - 1) / bk;
      const int NK_aligned_c = kL_c / bk;
      const bool align_K_c = (kL_c % bk) == 0;

      metal::MTLFCList chunk_consts = {
          {&align_Q, MTL::DataType::DataTypeBool, 200},
          {&align_K_c, MTL::DataType::DataTypeBool, 201},
          {&has_mask, MTL::DataType::DataTypeBool, 300},
          {&do_causal, MTL::DataType::DataTypeBool, 301},
          {&has_sinks, MTL::DataType::DataTypeBool, 302},
          {&has_block_mask, MTL::DataType::DataTypeBool, 303},
          {&has_block_token_mask, MTL::DataType::DataTypeBool, 304},
          {&has_block_indices, MTL::DataType::DataTypeBool, 305},
          {&partials_true, MTL::DataType::DataTypeBool, 306}};

      std::string chunk_hash;
      concatenate(
          chunk_hash,
          "omlx_qwen35_fa256_part_",
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
          (align_K_c ? 't' : 'n'),
          "_causal_",
          (do_causal ? 't' : 'n'));

      auto kernel = d.get_kernel(base_name, lib, chunk_hash, chunk_consts);
      compute_encoder.set_compute_pipeline_state(kernel);

      AttnParams params{
          /* int B = */ B,
          /* int H = */ H,
          /* int D = */ bd,
          /* int qL = */ qL,
          /* int kL = */ kL_c,
          /* int gqa_factor = */ gqa_factor,
          /* float scale = */ scale_,
          /* int NQ = */ NQ,
          /* int NK = */ NK_c,
          /* int NQ_aligned = */ NQ_aligned,
          /* int NK_aligned = */ NK_aligned_c,
          /* int qL_rem = */ (qL - NQ_aligned * bq),
          /* int kL_rem = */ (kL_c - NK_aligned_c * bk),
          // Global position of local query row 0 relative to this chunk's
          // first key; negative once the chunk starts past early rows.
          /* int qL_off = */ int((int64_t(kL) - qL) - k_start),
          /* int64_t Q_strides[3] = */
          {q.strides(0), q.strides(1), q.strides(2)},
          /* int64_t K_strides[3] = */
          {k.strides(0), k.strides(1), k.strides(2)},
          /* int64_t V_strides[3] = */
          {v.strides(0), v.strides(1), v.strides(2)},
          // Partial slab is contiguous (B, H, qL, D) per chunk.
          /* int64_t O_strides[3] = */
          {int64_t(H) * qL * bd, int64_t(qL) * bd, int64_t(bd)}};

      compute_encoder.set_input_array(q, 0);
      compute_encoder.set_input_array(
          k, 1, k_start * k.strides(2) * k.itemsize());
      compute_encoder.set_input_array(
          v, 2, k_start * v.strides(2) * v.itemsize());
      compute_encoder.set_output_array(o, 3);
      compute_encoder.set_bytes(params, 4);
      compute_encoder.set_output_array(
          o_part, 14, c * o_chunk_stride * o_part.itemsize());
      compute_encoder.set_output_array(
          lse_part, 15, c * lse_chunk_stride * lse_part.itemsize());
      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }

    std::string reduce_name;
    concatenate(
        reduce_name, "omlx_qwen35_fa256_chunk_reduce_", type_to_name(q));
    auto reduce_kernel = d.get_kernel(reduce_name, lib);
    compute_encoder.set_compute_pipeline_state(reduce_kernel);

    AttnChunkReduceParams reduce_params{
        /* int C = */ n_chunks,
        /* int H = */ H,
        /* int qL = */ qL,
        /* int D = */ bd,
        /* int64_t o_chunk_stride = */ o_chunk_stride,
        /* int64_t lse_chunk_stride = */ lse_chunk_stride,
        /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)}};

    compute_encoder.set_input_array(o_part, 0);
    compute_encoder.set_input_array(lse_part, 1);
    compute_encoder.set_output_array(o, 2);
    compute_encoder.set_bytes(reduce_params, 3);

    MTL::Size reduce_grid = MTL::Size(bd / 4, qL, int64_t(B) * H);
    MTL::Size reduce_group = MTL::Size(bd / 4, std::max(1, 256 / (bd / 4)), 1);
    compute_encoder.dispatch_threads(reduce_grid, reduce_group);
  }

  DEFINE_NAME(OMLXFa256Attention)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive& other) const override {
    const auto& rhs = static_cast<const Fa256AttentionPrimitive&>(other);
    return scale_ == rhs.scale_ && causal_ == rhs.causal_ &&
        q_block_ == rhs.q_block_ && k_block_ == rhs.k_block_ &&
        dispatch_budget_ == rhs.dispatch_budget_;
  }
  auto state() const {
    return std::make_tuple(
        nullptr, scale_, causal_, q_block_, k_block_, dispatch_budget_);
  }

 private:
  float scale_;
  bool causal_;
  int q_block_;
  int k_block_;
  int64_t dispatch_budget_;
};


} // namespace

array fa256_attention(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    bool causal,
    int q_block,
    int k_block,
    int64_t dispatch_budget,
    StreamOrDevice s) {
  for (const auto& tensor : {q, k, v}) {
    if (tensor.ndim() != 4) {
      std::ostringstream msg;
      msg << "[attention.fa256_attention] input with shape "
          << tensor.shape() << " expected rank 4.";
      throw std::invalid_argument(msg.str());
    }
  }
  auto stream = to_stream(s);
  auto final_type = result_type(std::vector<array>{q, k, v});
  if (final_type != float16 && final_type != bfloat16) {
    std::ostringstream msg;
    msg << "[attention.fa256_attention] expected fp16 or "
        << "bf16 inputs, got " << final_type << ".";
    throw std::invalid_argument(msg.str());
  }

  auto q_cast = astype(q, final_type, stream);
  auto k_cast = astype(k, final_type, stream);
  auto v_cast = astype(v, final_type, stream);
  if (Fa256AttentionPrimitive::unsupported(
          q_cast, k_cast, v_cast, causal, q_block, k_block, stream)) {
    throw std::invalid_argument(
        "[attention.fa256_attention] unsupported FA-256 shape.");
  }

  Shape out_shape{
      q_cast.shape(0), q_cast.shape(1), q_cast.shape(2), v_cast.shape(3)};
  std::vector<array> inputs = {q_cast, k_cast, v_cast};
  return array(
      std::move(out_shape),
      final_type,
      std::make_shared<Fa256AttentionPrimitive>(
          stream, scale, causal, q_block, k_block, dispatch_budget),
      std::move(inputs));
}


} // namespace omlx::attention
