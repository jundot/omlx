// SPDX-License-Identifier: Apache-2.0

#include "qwen4_qsa_sparse_gqa_tq.h"

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

std::string tq_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void *>(&tq_binary_dir), &info)) {
      throw std::runtime_error("Unable to get omlx_glm_kernels binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

bool last_dim_contiguous(const array &arr) { return arr.strides(-1) == 1; }

// MSE bit width from a codebook's entry count: exact log2 for the
// instantiated widths {2, 3, 4, 6, 8}; -1 rejects everything else.
int codebook_bits(const array &cb) {
  if (cb.ndim() != 1) {
    return -1;
  }
  const size_t n = cb.shape(0);
  if (n < 4 || n > 256 || (n & (n - 1)) != 0) {
    return -1;
  }
  int bits = 0;
  while ((size_t(1) << bits) < n) {
    ++bits;
  }
  switch (bits) {
  case 2:
  case 3:
  case 4:
  case 6:
  case 8:
    return bits;
  default:
    return -1;
  }
}

// Must match Qwen4QSATQParams in the shared Steel params header exactly.
struct Qwen4QSATQParams {
  int B;
  int q_heads;
  int kv_heads;
  int qL;
  int kL;
  int topk;
  int gqa_factor;
  int q_offset;

  float scale;

  int64_t Q_strides[3];
  int64_t Topk_strides[3];
  int64_t O_strides[3];
  int64_t KNorm_strides[2];
  int64_t KPacked_strides[3];
  int64_t VNorm_strides[2];
  int64_t VPacked_strides[3];
};

class Qwen4QSATQPrimitive : public Primitive {
public:
  Qwen4QSATQPrimitive(Stream stream, float scale, int q_offset, int bits_k,
                      int bits_v, int key_tile, int dimension_tile)
      : Primitive(stream), scale_(scale), q_offset_(q_offset), bits_k_(bits_k),
        bits_v_(bits_v), key_tile_(key_tile), dimension_tile_(dimension_tile) {}

  static bool unsupported(const array &q, const array &kn, const array &kp,
                          const array &vn, const array &vp, const array &cbk,
                          const array &cbv, const array &selected, int q_offset,
                          int key_tile, int dimension_tile, Stream stream) {
    if (stream.device == Device::cpu) {
      return true;
    }
    if (q.dtype() != float16 && q.dtype() != bfloat16) {
      return true;
    }
    // MSE widths {2,3,4,6,8} per side: fp16 norms over [B, H, T], uint32
    // packed words over [B, H, T, 256*bits/32], and 2^bits-entry fp32
    // codebooks (the bit widths are inferred from the codebook sizes).
    if (kn.dtype() != float16 || vn.dtype() != float16 ||
        kp.dtype() != uint32 || vp.dtype() != uint32 ||
        cbk.dtype() != float32 || cbv.dtype() != float32) {
      return true;
    }
    const int bits_k = codebook_bits(cbk);
    const int bits_v = codebook_bits(cbv);
    if (bits_k < 0 || bits_v < 0) {
      return true;
    }
    if (q.ndim() != 4 || kn.ndim() != 3 || kp.ndim() != 4 || vn.ndim() != 3 ||
        vp.ndim() != 4 || cbk.ndim() != 1 || cbv.ndim() != 1 ||
        selected.ndim() != 4 || selected.dtype() != uint32) {
      return true;
    }
    if (!last_dim_contiguous(q) || !last_dim_contiguous(kn) ||
        !last_dim_contiguous(kp) || !last_dim_contiguous(vn) ||
        !last_dim_contiguous(vp) || !last_dim_contiguous(cbk) ||
        !last_dim_contiguous(cbv) || !last_dim_contiguous(selected)) {
      return true;
    }
    if (q.shape(0) != 1 || q.shape(1) != 24 || q.shape(3) != 256 ||
        q.shape(2) <= 0) {
      return true;
    }
    if (kn.shape(0) != 1 || kn.shape(1) != 2 || kp.shape(0) != 1 ||
        kp.shape(1) != 2 || vn.shape(0) != 1 || vn.shape(1) != 2 ||
        vp.shape(0) != 1 || vp.shape(1) != 2) {
      return true;
    }
    // D=256 is enforced by the query shape check above.
    if (kn.shape(2) != vn.shape(2) || kp.shape(2) != vp.shape(2) ||
        kn.shape(2) != kp.shape(2) || kp.shape(3) != 256 * bits_k / 32 ||
        vp.shape(3) != 256 * bits_v / 32) {
      return true;
    }
    if (selected.shape(0) != 1 || selected.shape(1) != 1 ||
        selected.shape(2) != q.shape(2) || selected.shape(3) != 512) {
      return true;
    }
    if (q_offset < 0 || q_offset + q.shape(2) > kn.shape(2)) {
      return true;
    }
    // The (128,32) tuning variant is instantiated for the 4-bit pair only.
    if (!((key_tile == 64 && dimension_tile == 64) ||
          (key_tile == 128 && dimension_tile == 32 && bits_k == 4 &&
           bits_v == 4))) {
      return true;
    }
    return false;
  }

  void eval_cpu(const std::vector<array> & /* inputs */,
                std::vector<array> & /* outputs */) override {
    throw std::runtime_error("Qwen4QSATQPrimitive has no CPU path.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    const auto &q = inputs[0];
    const auto &kn = inputs[1];
    const auto &kp = inputs[2];
    const auto &vn = inputs[3];
    const auto &vp = inputs[4];
    const auto &cbk = inputs[5];
    const auto &cbv = inputs[6];
    const auto &selected = inputs[7];
    auto &out = outputs[0];

    constexpr int gqa = 12;
    constexpr int dim = 256;
    constexpr int wm = 2;
    constexpr int hpad = 16;

    out.set_data(allocator::malloc(out.nbytes()));
    Qwen4QSATQParams params{
        /* int B = */ 1,
        /* int q_heads = */ 24,
        /* int kv_heads = */ 2,
        /* int qL = */ q.shape(2),
        /* int kL = */ kn.shape(2),
        /* int topk = */ selected.shape(3),
        /* int gqa_factor = */ gqa,
        /* int q_offset = */ q_offset_,
        /* float scale = */ scale_,
        /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
        /* int64_t Topk_strides[3] = */
        {selected.strides(0), selected.strides(1), selected.strides(2)},
        /* int64_t O_strides[3] = */
        {out.strides(0), out.strides(1), out.strides(2)},
        /* int64_t KNorm_strides[2] = */ {kn.strides(0), kn.strides(1)},
        /* int64_t KPacked_strides[3] = */
        {kp.strides(0), kp.strides(1), kp.strides(2)},
        /* int64_t VNorm_strides[2] = */ {vn.strides(0), vn.strides(1)},
        /* int64_t VPacked_strides[3] = */
        {vp.strides(0), vp.strides(1), vp.strides(2)}};

    std::string kernel_name;
    concatenate(kernel_name, "qwen4_qsa_sparse_gqa_tq_", type_to_name(q),
                "_kb", bits_k_, "_vb", bits_v_, "_bk", key_tile_, "_dc",
                dimension_tile_, "_gqa", gqa, "_hp", hpad, "_d", dim, "_wm",
                wm);

    auto library = device.get_library("omlx_glm_kernels", tq_binary_dir());
    auto kernel = device.get_kernel(kernel_name, library);
    auto &encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(q, 0);
    encoder.set_input_array(kn, 1);
    encoder.set_input_array(kp, 2);
    encoder.set_input_array(vn, 3);
    encoder.set_input_array(vp, 4);
    encoder.set_input_array(cbk, 5);
    encoder.set_input_array(cbv, 6);
    encoder.set_input_array(selected, 7);
    encoder.set_output_array(out, 8);
    encoder.set_bytes(params, 9);
    encoder.dispatch_threadgroups(MTL::Size(q.shape(2), kn.shape(1), 1),
                                  MTL::Size(32, wm, 1));
  }

  DEFINE_NAME(OMLXQwen4QSASparseGQAAttentionTQ)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs = static_cast<const Qwen4QSATQPrimitive &>(other);
    return scale_ == rhs.scale_ && q_offset_ == rhs.q_offset_ &&
           bits_k_ == rhs.bits_k_ && bits_v_ == rhs.bits_v_ &&
           key_tile_ == rhs.key_tile_ &&
           dimension_tile_ == rhs.dimension_tile_;
  }
  auto state() const {
    return std::make_tuple(nullptr, scale_, q_offset_, bits_k_, bits_v_,
                           key_tile_, dimension_tile_);
  }

private:
  float scale_;
  int q_offset_;
  int bits_k_;
  int bits_v_;
  int key_tile_;
  int dimension_tile_;
};

} // namespace

array qwen4_qsa_sparse_gqa_attention_tq(
    const array &queries, const array &key_norms, const array &key_packed,
    const array &value_norms, const array &value_packed,
    const array &codebook_k, const array &codebook_v,
    const array &selected_blocks, float scale, int q_offset, int key_tile,
    int dimension_tile, StreamOrDevice s) {
  auto stream = to_stream(s);
  if (Qwen4QSATQPrimitive::unsupported(queries, key_norms, key_packed,
                                       value_norms, value_packed, codebook_k,
                                       codebook_v, selected_blocks, q_offset,
                                       key_tile, dimension_tile, stream)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.qwen4_qsa_sparse_gqa_attention_tq] expected "
        << "q=[1,24,M,256] fp16/bf16, norms=[1,2,T] fp16, packed "
        << "[1,2,T,256*bits/32] uint32, codebooks=[2^bits] fp32 with bits in "
        << "{2,3,4,6,8} per side, uint32 selected blocks=[1,1,M,512], "
        << "q_offset>=0, (BK,DC)=(64,64) or (128,32) at 4 bits; got "
        << queries.shape() << ", " << key_norms.shape() << ", "
        << key_packed.shape() << ", " << selected_blocks.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  // Validated by unsupported(): both codebooks carry instantiated widths.
  const int bits_k = codebook_bits(codebook_k);
  const int bits_v = codebook_bits(codebook_v);

  Shape out_shape{queries.shape(0), queries.shape(1), queries.shape(2),
                  queries.shape(3)};
  return array(
      std::move(out_shape), float32,
      std::make_shared<Qwen4QSATQPrimitive>(stream, scale, q_offset, bits_k,
                                            bits_v, key_tile, dimension_tile),
      std::vector<array>{queries, key_norms, key_packed, value_norms,
                         value_packed, codebook_k, codebook_v,
                         selected_blocks});
}

} // namespace omlx::glm_kernels
