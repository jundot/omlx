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
  // Retire a ticket whose producer command buffer failed before execute()
  // was ever called: balances the submission counters and wakes waiters
  // without latching the program (the failure is per-submission, typically
  // a request abort, not a program fault).
  void cancel_ticket(Ticket ticket);
  // True once an evaluation has failed or timed out on this program; the
  // program is latched and every later begin() will throw. Lets the Python
  // layer detect a wedged program at graph-construction time and fall back.
  bool has_error();
  // Run one throwaway evaluation to pay the first-run compilation cost at
  // load time instead of inside the first user request. Input contents are
  // irrelevant; the output is discarded.
  void warmup();
  void end(MTL::CommandBuffer *command_buffer, Ticket ticket);

private:
  class Impl;
  explicit AneLinearModel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;

  friend class AneLinearBankBuilder;
  friend class AneFusedBankBuilder;
  friend std::shared_ptr<AneLinearModel>
  qwen35_ane_compile_linear(const mlx::core::array &, int, int);
  friend std::vector<std::shared_ptr<AneLinearModel>>
  qwen35_ane_compile_linear_bank(const std::vector<mlx::core::array> &, int,
                                 int);
  friend std::shared_ptr<AneLinearModel>
  qwen35_ane_compile_fp16_linear(const mlx::core::array &, int);
  friend std::shared_ptr<AneLinearModel> qwen35_ane_compile_swiglu_down(
      const mlx::core::array &, const mlx::core::array &,
      const mlx::core::array &, int, int);
  friend std::vector<std::shared_ptr<AneLinearModel>>
  qwen35_ane_compile_swiglu_down_bank(
      const std::vector<mlx::core::array> &,
      const std::vector<mlx::core::array> &,
      const std::vector<mlx::core::array> &, int, int);
};

// Incremental builder for one instance-pinned procedure bank. add() converts
// a fp32 slice to the INT8 program format immediately, so the caller can
// release each staging array before preparing the next layer instead of
// holding every slice until compile time (issue #2781, ~16 GiB on a 27B).
// compile() assembles and loads one program from the stored chunks
// [start, stop), which lets the split-ladder retry without the fp32 sources.
class AneLinearBankBuilder {
public:
  explicit AneLinearBankBuilder(int sequence_length);
  ~AneLinearBankBuilder();
  AneLinearBankBuilder(const AneLinearBankBuilder &) = delete;
  AneLinearBankBuilder &operator=(const AneLinearBankBuilder &) = delete;

  void add(const mlx::core::array &weight);
  int size() const;
  std::vector<std::shared_ptr<AneLinearModel>>
  compile(int ane_instance, int start, int stop);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class AneFusedBankBuilder {
public:
  explicit AneFusedBankBuilder(int sequence_length);
  ~AneFusedBankBuilder();
  AneFusedBankBuilder(const AneFusedBankBuilder &) = delete;
  AneFusedBankBuilder &operator=(const AneFusedBankBuilder &) = delete;

  void add(const mlx::core::array &gate_weight,
           const mlx::core::array &up_weight,
           const mlx::core::array &down_weight);
  int size() const;
  std::vector<std::shared_ptr<AneLinearModel>>
  compile(int ane_instance, int start, int stop);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

bool qwen35_ane_available();
bool qwen35_ane_hybrid_nax_enabled();
bool qwen35_cpu_shared_resource_available();
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

mlx::core::array qwen35_cpu_fp16_affine_qmm_t(
    const mlx::core::array &x, const mlx::core::array &cpu_weight,
    const mlx::core::array &gpu_weight, const mlx::core::array &gpu_scales,
    const mlx::core::array &gpu_biases, int bits, int variant = 8,
    int group_size = 128, int cpu_threads = 0,
    bool cpu_shared_resource = false,
    mlx::core::StreamOrDevice s = {});

mlx::core::array qwen35_ane_affine_qmm_t(
    const mlx::core::array &x, const mlx::core::array &gpu_weight,
    const mlx::core::array &gpu_scales, const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model, int bits,
    int variant = 8, int group_size = 128, int profile_category = 1,
    mlx::core::StreamOrDevice s = {});

mlx::core::array qwen35_ane_cpu_fp16_affine_qmm_t(
    const mlx::core::array &x, const mlx::core::array &cpu_weight,
    const mlx::core::array &gpu_weight, const mlx::core::array &gpu_scales,
    const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model, int bits, int variant = 8,
    int group_size = 128, int profile_category = 1, int cpu_threads = 0,
    bool cpu_shared_resource = false, mlx::core::StreamOrDevice s = {});

std::shared_ptr<AneLinearModel> qwen35_ane_compile_fp16_linear(
    const mlx::core::array &weight, int sequence_length);

std::shared_ptr<AneLinearModel> qwen35_ane_compile_swiglu_down(
    const mlx::core::array &gate_weight,
    const mlx::core::array &up_weight,
    const mlx::core::array &down_weight,
    int sequence_length, int ane_instance = 0);

std::vector<std::shared_ptr<AneLinearModel>>
qwen35_ane_compile_swiglu_down_bank(
    const std::vector<mlx::core::array> &gate_weights,
    const std::vector<mlx::core::array> &up_weights,
    const std::vector<mlx::core::array> &down_weights,
    int sequence_length, int ane_instance);

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

mlx::core::array qwen35_ane_cpu_fp16_swiglu_t(
    const mlx::core::array &x, const mlx::core::array &cpu_weight,
    const mlx::core::array &gpu_weight, const mlx::core::array &gpu_scales,
    const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model, int bits, int variant = 8,
    int group_size = 128, int cpu_threads = 0, bool cpu_shared_resource = false,
    mlx::core::StreamOrDevice s = {});

mlx::core::array qwen35_ane_cpu_fp16_q4_swiglu_t(
    const mlx::core::array &x, const mlx::core::array &cpu_weight,
    const mlx::core::array &gpu_weight, const mlx::core::array &gpu_scales,
    const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model, int variant = 8,
    int group_size = 128, int cpu_threads = 0, bool cpu_shared_resource = false,
    mlx::core::StreamOrDevice s = {});

mlx::core::array qwen35_ane_dual_affine_qmm_t(
    const mlx::core::array &x, const mlx::core::array &gpu_weight,
    const mlx::core::array &gpu_scales, const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int bits,
    int variant = 8, int group_size = 128, int profile_category = 1,
    mlx::core::StreamOrDevice s = {});

mlx::core::array qwen35_ane_dual_cpu_fp16_affine_qmm_t(
    const mlx::core::array &x, const mlx::core::array &cpu_weight,
    const mlx::core::array &gpu_weight, const mlx::core::array &gpu_scales,
    const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int bits,
    int variant = 8, int group_size = 128, int profile_category = 1,
    int cpu_threads = 0, bool cpu_shared_resource = false,
    mlx::core::StreamOrDevice s = {});

mlx::core::array qwen35_ane_dual_cpu_fp16_swiglu_t(
    const mlx::core::array &x, const mlx::core::array &cpu_weight,
    const mlx::core::array &gpu_weight, const mlx::core::array &gpu_scales,
    const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int bits,
    int variant = 8, int group_size = 128, int cpu_threads = 0,
    bool cpu_shared_resource = false,
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

mlx::core::array qwen35_ane_dual_cpu_fp16_q4_swiglu_t(
    const mlx::core::array &x, const mlx::core::array &cpu_weight,
    const mlx::core::array &gpu_weight, const mlx::core::array &gpu_scales,
    const mlx::core::array &gpu_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int variant = 8,
    int group_size = 128, int cpu_threads = 0,
    bool cpu_shared_resource = false,
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

mlx::core::array qwen35_ane_dual_q4_swiglu_down_t(
    const mlx::core::array &x,
    const mlx::core::array &gpu_gate_up_weight,
    const mlx::core::array &gpu_gate_up_scales,
    const mlx::core::array &gpu_gate_up_biases,
    const mlx::core::array &gpu_down_weight,
    const mlx::core::array &gpu_down_scales,
    const mlx::core::array &gpu_down_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int variant = 8,
    int group_size = 128, mlx::core::StreamOrDevice s = {});

mlx::core::array qwen35_ane_dual_cpu_fp16_q4_swiglu_down_t(
    const mlx::core::array &x,
    const mlx::core::array &cpu_gate_up_weight,
    const mlx::core::array &cpu_down_weight,
    const mlx::core::array &gpu_gate_up_weight,
    const mlx::core::array &gpu_gate_up_scales,
    const mlx::core::array &gpu_gate_up_biases,
    const mlx::core::array &gpu_down_weight,
    const mlx::core::array &gpu_down_scales,
    const mlx::core::array &gpu_down_biases,
    const std::shared_ptr<AneLinearModel> &ane_model0,
    const std::shared_ptr<AneLinearModel> &ane_model1, int variant = 8,
    int group_size = 128, int cpu_threads = 0,
    bool cpu_shared_resource = false,
    mlx::core::StreamOrDevice s = {});

} // namespace omlx::qwen35_prefill_kernels
