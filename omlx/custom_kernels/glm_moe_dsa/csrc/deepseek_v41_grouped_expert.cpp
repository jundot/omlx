#include "deepseek_v41_grouped_expert.h"

#include <stdexcept>
#include <string>
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

namespace omlx::glm_kernels {
namespace {
using namespace mlx::core;

class GroupedExpert : public Primitive {
 public:
  GroupedExpert(Stream s, const array& gate, const array& up,
                const array& activation, const array& down)
      : Primitive(s), operations_{gate.primitive_ptr(), up.primitive_ptr(),
                                 activation.primitive_ptr(), down.primitive_ptr()},
        intermediate_shape_(gate.shape()),
        gate_inputs_(gate.inputs().size()), up_inputs_(up.inputs().size()) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("DeepSeek V4.1 grouped expert requires GPU execution");
  }

  void eval_gpu(const std::vector<array>& in, std::vector<array>& out) override {
    std::vector<array> gate{array(intermediate_shape_, bfloat16, nullptr, {})};
    std::vector<array> up{array(intermediate_shape_, bfloat16, nullptr, {})};
    std::vector<array> activation{array(intermediate_shape_, bfloat16, nullptr, {})};
    auto& encoder = metal::get_command_encoder(stream());
    // Reuse MLX kernels without evaluating a nested lazy graph. Register each
    // temporary before a later stage can throw or release its dependencies.
    const auto up_begin = gate_inputs_;
    const auto activation_begin = gate_inputs_ + up_inputs_;
    const auto down_begin = activation_begin + 2;
    operations_[0]->eval_gpu(
        std::vector<array>(in.begin(), in.begin() + up_begin), gate);
    encoder.add_temporary(gate[0]);
    operations_[1]->eval_gpu(
        std::vector<array>(in.begin() + up_begin, in.begin() + activation_begin), up);
    encoder.add_temporary(up[0]);
    operations_[2]->eval_gpu(
        {gate[0], up[0], in[activation_begin], in[activation_begin + 1]}, activation);
    encoder.add_temporary(activation[0]);
    std::vector<array> down_inputs{activation[0]};
    down_inputs.insert(down_inputs.end(), in.begin() + down_begin, in.end());
    operations_[3]->eval_gpu(down_inputs, out);
  }

  DEFINE_NAME(GroupedExpert)
  DEFINE_INPUT_OUTPUT_SHAPE()
  bool is_equivalent(const Primitive&) const override { return false; }

 private:
  std::vector<std::shared_ptr<Primitive>> operations_;
  Shape intermediate_shape_;
  size_t gate_inputs_;
  size_t up_inputs_;
};
} // namespace

mlx::core::array deepseek_v41_grouped_expert(
    const mlx::core::array& gate, const mlx::core::array& up,
    const mlx::core::array& activation, const mlx::core::array& down) {
  using namespace mlx::core;
  for (const auto* node : {&gate, &up, &down}) {
    if (!node->has_primitive() ||
        std::string(node->primitive().name()) != "GatherQMM" ||
        (node->inputs().size() != 5 && node->inputs().size() != 6) ||
        node->dtype() != bfloat16) {
      throw std::invalid_argument("Expected unevaluated BF16 MXFP or affine GatherQMM nodes");
    }
  }
  if (!activation.has_primitive() ||
      std::string(activation.primitive().name()) != "CustomKernel" ||
      activation.inputs().size() != 4 || activation.dtype() != bfloat16 ||
      activation.inputs()[0].id() != gate.id() ||
      activation.inputs()[1].id() != up.id() ||
      down.inputs()[0].id() != activation.id() ||
      gate.shape() != up.shape() || gate.shape() != activation.shape() ||
      gate.ndim() != 3 || gate.shape(0) < 1 || gate.shape(1) != 1 ||
      gate.shape(2) % 32 != 0 || down.ndim() != 3 ||
      down.shape(0) != gate.shape(0) || down.shape(1) != 1) {
    throw std::invalid_argument("Unsupported DeepSeek V4.1 expert graph");
  }
  auto s = gate.primitive().stream();
  if (s.device != Device::gpu) {
    throw std::invalid_argument("DeepSeek V4.1 grouped expert requires a GPU stream");
  }
  for (const auto* node : {&up, &activation, &down}) {
    if (node->primitive().stream() != s) {
      throw std::invalid_argument("Expert nodes must share a GPU stream");
    }
  }
  std::vector<array> inputs = gate.inputs();
  inputs.insert(inputs.end(), up.inputs().begin(), up.inputs().end());
  inputs.push_back(activation.inputs()[2]);
  inputs.push_back(activation.inputs()[3]);
  inputs.insert(inputs.end(), down.inputs().begin() + 1, down.inputs().end());
  return array(down.shape(), down.dtype(),
               std::make_shared<GroupedExpert>(s, gate, up, activation, down),
               std::move(inputs));
}
} // namespace omlx::glm_kernels
