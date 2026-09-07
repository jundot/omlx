// Copyright © 2026 IndenScale
#include <dlfcn.h>
#include <filesystem>
#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include "mlx/mlx.h"
#include "mlx/primitives.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
namespace mx = mlx::core;
using namespace mlx::core;
std::string binary_dir() {
  Dl_info info;
  if (!dladdr(reinterpret_cast<void*>(&binary_dir), &info)) throw std::runtime_error("Cannot locate extension");
  return std::filesystem::path(info.dli_fname).parent_path().string();
}
class EcoPagedAttention : public UnaryPrimitive {
 public:
  EcoPagedAttention(Stream s, float scale, int partitions)
      : UnaryPrimitive(s), scale_(scale), partitions_(partitions) {}
  void eval_cpu(const std::vector<array>&, array&) override { throw std::runtime_error("GPU required"); }
  void eval_gpu(const std::vector<array>& inputs, array& out) override;
  const char* name() const override { return "EcoPagedAttention"; }
  bool is_equivalent(const Primitive& p) const override {
    auto other = dynamic_cast<const EcoPagedAttention*>(&p);
    return other && other->scale_ == scale_ && other->partitions_ == partitions_;
  }
 private:
  float scale_;
  int partitions_;
};
void EcoPagedAttention::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& q = inputs[0];
  auto& key_pages = inputs[1];
  auto& value_pages = inputs[2];
  auto& block_tables = inputs[3];
  auto& lengths = inputs[4];
  auto s = stream();
  auto& d = metal::device(s.device);

  const int batch_size = q.shape(0);
  const int num_query_heads = q.shape(1);
  const int num_kv_heads = key_pages.shape(1);
  const int gqa_factor = num_query_heads / num_kv_heads;
  const int page_size = key_pages.shape(2);
  const int max_blocks = block_tables.shape(1);
  const int context_capacity = page_size * max_blocks;

  int blocks = 32;
  if (context_capacity > 32768) {
    blocks = 512;
  } else if (context_capacity > 8192) {
    blocks = 256;
  } else if (context_capacity > 2048) {
    blocks = 128;
  }
  if (partitions_ > 0) blocks = partitions_;

  Shape partial_shape = {batch_size, num_query_heads, 1, blocks, q.shape(-1)};
  array partials(partial_shape, q.dtype(), nullptr, {});
  partial_shape.pop_back();
  array sums(partial_shape, float32, nullptr, {});
  array maxs(std::move(partial_shape), float32, nullptr, {});
  partials.set_data(allocator::malloc(partials.nbytes()));
  sums.set_data(allocator::malloc(sums.nbytes()));
  maxs.set_data(allocator::malloc(maxs.nbytes()));
  out.set_data(allocator::malloc(out.nbytes()));

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.add_temporary(partials);
  compute_encoder.add_temporary(sums);
  compute_encoder.add_temporary(maxs);

  std::string kernel_name = "paged_sdpa_vector_2pass_1_gqa_";
  kernel_name += std::to_string(gqa_factor);
  kernel_name += "_";
  kernel_name += (q.dtype() == float32 ? "float" : q.dtype() == float16 ? "float16_t" : "bfloat16_t");
  kernel_name += "_" + std::to_string(q.shape(-1));

  auto kernel = d.get_kernel(kernel_name, d.get_library("eco_paged", binary_dir()));
  MTL::Size group_dims(32, gqa_factor, 1);
  MTL::Size grid_dims(num_kv_heads, batch_size, blocks);
  check_kernel_threadgroup_size(kernel, group_dims, kernel_name);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(key_pages, 1);
  compute_encoder.set_input_array(value_pages, 2);
  compute_encoder.set_input_array(block_tables, 3);
  compute_encoder.set_input_array(lengths, 4);
  compute_encoder.set_output_array(partials, 5);
  compute_encoder.set_output_array(sums, 6);
  compute_encoder.set_output_array(maxs, 7);
  compute_encoder.set_bytes(page_size, 8);
  compute_encoder.set_bytes(max_blocks, 9);
  compute_encoder.set_bytes(num_kv_heads, 10);
  compute_encoder.set_bytes(scale_, 11);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  kernel_name = "eco_sdpa_reduce_";
  kernel_name += (q.dtype() == float32 ? "float" : q.dtype() == float16 ? "float16_t" : "bfloat16_t");
  kernel_name += "_" + std::to_string(q.shape(-1));
  kernel = d.get_kernel(kernel_name, d.get_library("eco_paged", binary_dir()));
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(partials, 0);
  compute_encoder.set_input_array(sums, 1);
  compute_encoder.set_input_array(maxs, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(blocks, 4);

  group_dims = MTL::Size(1024, 1, 1);
  grid_dims = MTL::Size(batch_size * num_query_heads, 1, 1);
  check_kernel_threadgroup_size(kernel, group_dims, kernel_name);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

array paged_attention(const array& q, const array& k, const array& v,
                      const array& tables, const array& lengths, float scale, int partitions,
                      StreamOrDevice stream = {}) {
  auto s = to_stream(stream);
  return array(q.shape(), q.dtype(), std::make_shared<EcoPagedAttention>(s, scale, partitions),
               {contiguous(q, false, s), contiguous(k, false, s), contiguous(v, false, s),
                contiguous(tables, false, s), contiguous(lengths, false, s)});
}
namespace nb = nanobind;
using namespace nb::literals;
NB_MODULE(_ext, m) {
  m.def("paged_attention", [](nb::handle q, nb::handle k, nb::handle v,
       nb::handle tables, nb::handle lengths, float scale, nb::handle out, nb::handle stream, int partitions) {
    auto result = paged_attention(*nb::inst_ptr<array>(q), *nb::inst_ptr<array>(k),
      *nb::inst_ptr<array>(v), *nb::inst_ptr<array>(tables), *nb::inst_ptr<array>(lengths),
      scale, partitions, *nb::inst_ptr<Stream>(stream));
    nb::inst_ptr<array>(out)->overwrite_descriptor(result);
  });
}
