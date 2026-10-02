// Copyright © 2024 Apple Inc.
// Originally from the Bonsai MLX fork (github.com/PrismML-Eng/Bonsai-demo).
// SPDX-License-Identifier: MIT

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// Greedy speculative-decoding verify, one thread per batch row.
//   draft   [B, K]    int32  — drafted tokens
//   target  [B, K+1]  int32  — target argmax tokens over [last] + draft
// Writes:
//   n_accepted [B]      — longest matching prefix length (0..K)
//   committed  [B, K+1] — draft prefix (< n) then the corrected token at n.
[[kernel]] void spec_decode_verify(
    const device int* draft [[buffer(0)]],
    const device int* target [[buffer(1)]],
    device int* n_accepted [[buffer(2)]],
    device int* committed [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& B [[buffer(5)]],
    uint b [[thread_position_in_grid]]) {
  if (b >= uint(B)) {
    return;
  }
  int row = int(b);
  const device int* d = draft + row * K;
  const device int* t = target + row * (K + 1);
  device int* c = committed + row * (K + 1);

  int n = K;
  for (int j = 0; j < K; ++j) {
    if (d[j] != t[j]) {
      n = j;
      break;
    }
  }
  n_accepted[row] = n;
  for (int j = 0; j < K + 1; ++j) {
    c[j] = (j < n) ? d[j] : ((j == n) ? t[n] : 0);
  }
}
