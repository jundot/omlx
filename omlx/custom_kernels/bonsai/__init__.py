"""Bonsai 1-bit / 2-bit decode kernels for oMLX.

Provides two primitives used by Bonsai (1-bit) and Ternary-Bonsai (2-bit) models:

* **qmv / qmv_wide** — fast affine quantized matrix-vector kernels for decode
  (M = 1..5 input rows).  1-bit uses a dedicated qmv_fast kernel ported from
  the Bonsai MLX fork; 2-bit routes to qmv_wide at M >= 3 on gen-15+ hardware.

* **spec_decode_verify** — fused greedy speculative-decoding verify: given
  draft_tokens [B, K] and target_logits [B, K+1, V] it returns (n_accepted [B],
  committed [B, K+1]) in one pass.  Falls back to a pure-mlx op composition when
  the Metal kernel is unavailable.

The native C++ extension (``_ext``) is optional; without it every call falls
back to stock mlx kernels or pure-Python compositions (slower but correct).
"""

from . import fast

__all__ = ["fast"]
