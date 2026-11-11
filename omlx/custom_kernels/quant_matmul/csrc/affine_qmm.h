#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::quant_matmul {

// True when the NAX metallib was built next to the extension. Kernel launch
// still degrades to the classic kernels if loading it fails at runtime.
bool nax_qmm_kernels_built();

// False once a NAX kernel launch failed and the op fell back to the classic
// kernels for the rest of the process (diagnostics for the M5 sweep).
bool nax_qmm_runtime_active();

mx::array affine_q2_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    int group_size = 64,
    mx::StreamOrDevice s = {});

mx::array affine_q4_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    int group_size = 64,
    mx::StreamOrDevice s = {});

mx::array affine_q5_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    int group_size = 64,
    mx::StreamOrDevice s = {});

mx::array affine_q6_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    int group_size = 64,
    mx::StreamOrDevice s = {});

mx::array affine_q8_qmm_t(
    const mx::array& x,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int variant = 8,
    bool use_nax = false,
    int nax_variant = 0,
    int group_size = 64,
    mx::StreamOrDevice s = {});

} // namespace omlx::quant_matmul
