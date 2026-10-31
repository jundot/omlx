"""Kernels for the Bonsai t5 (base-3 ternary, ~1.585 bpw) weight format.

Stock MLX serves the uint32 1-bit and 2-bit affine formats.
Without the native extension, t5 layers fall back to dequantize + matmul.
"""

from . import fast

__all__ = ["fast"]
