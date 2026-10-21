#pragma once

#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace omlx::quant_matmul {

// oQ mixed-bit QxA8 path (Q4/Q5/Q8, GS64, affine) on the M5 tensor units.

// True when the INT8 NAX GEMM can actually run here: tensor units present,
// the NAX metallib built next to the extension, and no earlier launch failure.
bool oq_a8_kernels_available();

// Stage A. Returns {Qa int8 [..., K], Sa float32, Ra int16 [..., K/64]}.
// act_mode 0 gives one scale per row (Sa is [M]); act_mode 1 gives one scale
// per group of 64 (Sa is [..., K/64]).
std::vector<mx::array> oq_a8_quantize(
    const mx::array& x,
    int act_mode = 0,
    mx::StreamOrDevice s = {});

// INT8 x INT8 -> INT32 GEMM against packed affine Q4/Q5/Q8 weights, with the
// affine correction applied at every GS64 boundary. Output dtype follows
// `scales`. Q4/Q5 read [K/64, N] metadata, or with `packed` Q4 weights and
// metadata in the PackedLinear tile layout. Q8 reads scales/biases in the
// checkpoint layout [N, K/64]. Q4/Q5 read Qa in Stage A v8's permuted K order;
// Q8 reads it in checkpoint order. `variant` selects the Q4/Q5 tile; Q8 picks
// its tile from the row count.
mx::array oq_a8_qmm_t(
    const mx::array& qa,
    const mx::array& sa,
    const mx::array& ra,
    const mx::array& weight,
    const mx::array& scales,
    const mx::array& biases,
    int bits,
    int act_mode = 0,
    int variant = 800,
    bool packed = false,
    mx::StreamOrDevice s = {});

// Test helper: unpack Q4/Q5/Q8 codes to INT8 [N, group_count * 64] (Q8 is
// centered to q - 128). Production code never materializes this.
mx::array oq_a8_decode_weights(
    const mx::array& weight,
    int bits,
    int group_count,
    mx::StreamOrDevice s = {});

} // namespace omlx::quant_matmul
