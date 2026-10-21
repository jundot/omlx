"""Low-bit decode kernels for the Bonsai 1-bit / 2-bit formats.

Used by Bonsai (1-bit) and Ternary-Bonsai (2-bit) models:

* **qmv / qmv_wide** — fast affine quantized matrix-vector kernels for decode
  (M = 1..5 input rows).  1-bit uses a dedicated qmv_fast kernel ported from
  the Bonsai MLX fork; 2-bit routes to qmv_wide at M >= 3 on gen-15+ hardware.

The native C++ extension (``_ext``) is optional; without it every call falls
back to stock mlx kernels or pure-Python compositions (slower but correct).
"""

from . import fast

__all__ = ["fast"]
