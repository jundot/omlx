#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace MTL {
class Buffer;
class CommandBuffer;
} // namespace MTL

namespace omlx::qwen35_prefill_kernels {

// Opaque owner for one private-ANE, fixed-shape INT8 linear slice.
// The public Python feature is opt-in because these APIs are undocumented and
// may change between macOS releases.
class AneLinearModel : public std::enable_shared_from_this<AneLinearModel> {
public:
  ~AneLinearModel();
  AneLinearModel(const AneLinearModel &) = delete;
  AneLinearModel &operator=(const AneLinearModel &) = delete;

  int input_dim() const;
  int output_dim() const;
  int sequence_length() const;
  MTL::Buffer *input_buffer() const;
  MTL::Buffer *output_buffer() const;

  struct Ticket {
    uint64_t ready;
    uint64_t done;
  };

  Ticket begin(MTL::CommandBuffer *command_buffer);
  void execute(Ticket ticket);
  void wait(Ticket ticket);
  // Run one throwaway evaluation to pay the first-run compilation cost at
  // load time instead of inside the first user request. Input contents are
  // irrelevant; the output is discarded.
  void warmup();
  void end(MTL::CommandBuffer *command_buffer, Ticket ticket);

private:
  class Impl;
  explicit AneLinearModel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend std::shared_ptr<AneLinearModel>
  qwen35_ane_compile_linear(const mlx::core::array &, int, int);
  friend std::vector<std::shared_ptr<AneLinearModel>>
  qwen35_ane_compile_linear_bank(const std::vector<mlx::core::array> &, int,
                                 int);
  friend std::shared_ptr<AneLinearModel>
  qwen35_ane_compile_fp16_linear(const mlx::core::array &, int);
  friend std::shared_ptr<AneLinearModel> qwen35_ane_compile_swiglu_down(
      const mlx::core::array &, const mlx::core::array &,
      const mlx::core::array &, int);
};

bool qwen35_ane_available();
void qwen35_ane_profile_set_enabled(bool enabled);
void qwen35_ane_profile_reset();
std::vector<double> qwen35_ane_profile_snapshot();

std::shared_ptr<AneLinearModel>
qwen35_ane_compile_linear(const mlx::core::array &weight, int sequence_length);

std::shared_ptr<AneLinearModel>
qwen35_ane_compile_linear(const mlx::core::array &weight, int sequence_length,
                          int ane_instance);

std::vector<std::shared_ptr<AneLinearModel>> qwen35_ane_compile_linear_bank(
    const std::vector<mlx::core::array> &weights, int sequence_length,
    int ane_instance);

mlx::core::array qwen35_ane_affine_qmm_t(
    const mlx::core::array &x, const mlx::core::array &gpu_weight,
    const mlx::core::array &gpu_scales, const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model, int bits,
    int variant = 8, int group_size = 128, int profile_category = 1,
    mlx::core::StreamOrDevice s = {});

std::shared_ptr<AneLinearModel> qwen35_ane_compile_fp16_linear(
    const mlx::core::array &weight, int sequence_length);

std::shared_ptr<AneLinearModel> qwen35_ane_compile_swiglu_down(
    const mlx::core::array &gate_weight,
    const mlx::core::array &up_weight,
    const mlx::core::array &down_weight,
    int sequence_length);

mlx::core::array qwen35_ane_q4_affine_qmm_t(
    const mlx::core::array &x, const mlx::core::array &gpu_weight,
    const mlx::core::array &gpu_scales, const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model, int variant = 8,
    int group_size = 128, int profile_category = 0,
    mlx::core::StreamOrDevice s = {});

mlx::core::array qwen35_ane_q4_swiglu_t(
    const mlx::core::array &x, const mlx::core::array &gpu_weight,
    const mlx::core::array &gpu_scales, const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model, int variant = 8,
    int group_size = 128, mlx::core::StreamOrDevice s = {});

mlx::core::array qwen35_ane_affine_swiglu_t(
    const mlx::core::array &x, const mlx::core::array &gpu_weight,
    const mlx::core::array &gpu_scales, const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model, int bits,
    int variant = 8, int group_size = 128,
    mlx::core::StreamOrDevice s = {});

mlx::core::array qwen35_ane_dual_affine_qmm_t(
    const mlx::core::array &x, const mlx::core::array &gpu_weight,
    const mlx::core::array &gpu_scales, const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int bits,
    int variant = 8, int group_size = 128, int profile_category = 1,
    mlx::core::StreamOrDevice s = {});

mlx::core::array qwen35_ane_dual_q4_swiglu_t(
    const mlx::core::array &x, const mlx::core::array &gpu_weight,
    const mlx::core::array &gpu_scales, const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int variant = 8,
    int group_size = 128, mlx::core::StreamOrDevice s = {});

mlx::core::array qwen35_ane_dual_affine_swiglu_t(
    const mlx::core::array &x, const mlx::core::array &gpu_weight,
    const mlx::core::array &gpu_scales, const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int bits,
    int variant = 8, int group_size = 128,
    mlx::core::StreamOrDevice s = {});

mlx::core::array qwen35_ane_q4_swiglu_down_t(
    const mlx::core::array &x,
    const mlx::core::array &gpu_gate_up_weight,
    const mlx::core::array &gpu_gate_up_scales,
    const mlx::core::array &gpu_gate_up_biases,
    const mlx::core::array &gpu_down_weight,
    const mlx::core::array &gpu_down_scales,
    const mlx::core::array &gpu_down_biases,
    const std::shared_ptr<AneLinearModel> &ane_model, int variant = 8,
    int group_size = 128, mlx::core::StreamOrDevice s = {});

} // namespace omlx::qwen35_prefill_kernels
