// SPDX-License-Identifier: Apache-2.0

#include "qwen4_qsa_sparse_gqa.h"

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
    if (!dladdr(reinterpret_cast<void *>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get omlx_glm_kernels binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

bool last_dim_contiguous(const array &arr) { return arr.strides(-1) == 1; }

struct Qwen4QSASparseGQAParams {
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
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t Topk_strides[3];
  int64_t O_strides[3];
};

class Qwen4QSASparseGQAPrimitive : public Primitive {
public:
  Qwen4QSASparseGQAPrimitive(Stream stream, float scale, int q_offset,
                             int key_tile, int dimension_tile)
      : Primitive(stream), scale_(scale), q_offset_(q_offset),
        key_tile_(key_tile), dimension_tile_(dimension_tile) {}

  static bool unsupported(const array &q, const array &k, const array &v,
                          const array &selected, int q_offset, int key_tile,
                          int dimension_tile, Stream stream) {
    if (stream.device == Device::cpu || q.dtype() != k.dtype() ||
        q.dtype() != v.dtype()) {
      return true;
    }
    if (q.dtype() != float16 && q.dtype() != bfloat16) {
      return true;
    }
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 ||
        selected.ndim() != 4 || selected.dtype() != uint32) {
      return true;
    }
    if (!last_dim_contiguous(q) || !last_dim_contiguous(k) ||
        !last_dim_contiguous(v) || !last_dim_contiguous(selected)) {
      return true;
    }
    if (q.shape(0) != 1 || k.shape(0) != 1 || v.shape(0) != 1 ||
        selected.shape(0) != 1 || q.shape(1) != 24 || k.shape(1) != 2 ||
        v.shape(1) != 2 || q.shape(3) != 256 || k.shape(3) != 256 ||
        v.shape(3) != 256 || k.shape(2) != v.shape(2) || q.shape(2) <= 0 ||
        selected.shape(1) != 1 || selected.shape(2) != q.shape(2) ||
        selected.shape(3) != 512) {
      return true;
    }
    if (q_offset < 0 || q_offset + q.shape(2) > k.shape(2) ||
        !((dimension_tile == 32 && (key_tile == 128 || key_tile == 256)) ||
          (dimension_tile == 64 && (key_tile == 64 || key_tile == 128)))) {
      return true;
    }
    return false;
  }

  void eval_cpu(const std::vector<array> & /* inputs */,
                std::vector<array> & /* outputs */) override {
    throw std::runtime_error("Qwen4QSASparseGQAPrimitive has no CPU path.");
  }

  void eval_gpu(const std::vector<array> &inputs,
                std::vector<array> &outputs) override {
    auto &stream = this->stream();
    auto &device = metal::device(stream.device);
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &selected = inputs[3];
    auto &out = outputs[0];

    constexpr int gqa = 12;
    constexpr int dim = 256;
    constexpr int wm = 2;
    constexpr int hpad = 16;

    out.set_data(allocator::malloc(out.nbytes()));
    Qwen4QSASparseGQAParams params{
        /* int B = */ 1,
        /* int q_heads = */ 24,
        /* int kv_heads = */ 2,
        /* int qL = */ q.shape(2),
        /* int kL = */ k.shape(2),
        /* int topk = */ selected.shape(3),
        /* int gqa_factor = */ gqa,
        /* int q_offset = */ q_offset_,
        /* float scale = */ scale_,
        /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
        /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
        /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
        /* int64_t Topk_strides[3] = */
        {selected.strides(0), selected.strides(1), selected.strides(2)},
        /* int64_t O_strides[3] = */
        {out.strides(0), out.strides(1), out.strides(2)}};

    std::string kernel_name;
    concatenate(kernel_name, "qwen4_qsa_sparse_gqa_", type_to_name(q), "_bk",
                key_tile_, "_dc", dimension_tile_, "_gqa", gqa, "_hp", hpad,
                "_d", dim, "_wm", wm);

    auto library = device.get_library("omlx_glm_kernels", current_binary_dir());
    auto kernel = device.get_kernel(kernel_name, library);
    auto &encoder = metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(q, 0);
    encoder.set_input_array(k, 1);
    encoder.set_input_array(v, 2);
    encoder.set_input_array(selected, 3);
    encoder.set_output_array(out, 4);
    encoder.set_bytes(params, 5);
    encoder.dispatch_threadgroups(MTL::Size(q.shape(2), k.shape(1), 1),
                                  MTL::Size(32, wm, 1));
  }

  DEFINE_NAME(OMLXQwen4QSASparseGQAAttention)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive &other) const override {
    const auto &rhs = static_cast<const Qwen4QSASparseGQAPrimitive &>(other);
    return scale_ == rhs.scale_ && q_offset_ == rhs.q_offset_ &&
           key_tile_ == rhs.key_tile_ && dimension_tile_ == rhs.dimension_tile_;
  }
  auto state() const {
    return std::make_tuple(nullptr, scale_, q_offset_, key_tile_,
                           dimension_tile_);
  }

private:
  float scale_;
  int q_offset_;
  int key_tile_;
  int dimension_tile_;
};

} // namespace

array qwen4_qsa_sparse_gqa_attention(const array &queries, const array &keys,
                                     const array &values,
                                     const array &selected_blocks, float scale,
                                     int q_offset, int key_tile,
                                     int dimension_tile, StreamOrDevice s) {
  auto stream = to_stream(s);
  if (Qwen4QSASparseGQAPrimitive::unsupported(
          queries, keys, values, selected_blocks, q_offset, key_tile,
          dimension_tile, stream)) {
    std::ostringstream msg;
    msg << "[omlx_glm_kernels.qwen4_qsa_sparse_gqa_attention] expected "
        << "q=[1,24,M,256], k/v=[1,2,K,256], uint32 selected blocks="
        << "[1,1,M,512], q_offset>=0, (BK,DC) in "
        << "{(128,32),(256,32),(64,64),(128,64)}; got " << queries.shape()
        << ", " << keys.shape() << ", " << values.shape() << ", "
        << selected_blocks.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  Shape out_shape{queries.shape(0), queries.shape(1), queries.shape(2),
                  queries.shape(3)};
  return array(std::move(out_shape), queries.dtype(),
               std::make_shared<Qwen4QSASparseGQAPrimitive>(
                   stream, scale, q_offset, key_tile, dimension_tile),
               std::vector<array>{queries, keys, values, selected_blocks});
}

} // namespace omlx::glm_kernels
