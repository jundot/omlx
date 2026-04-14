# SPDX-License-Identifier: Apache-2.0
"""PlanarQuant3 KV cache — 2D Givens rotation + 3-bit Lloyd-Max quantization.

Port of the llama.cpp fork feature/planarquant-kv-cache branch to MLX.
Upstream reference: https://github.com/scrya-com/rotorquant (MIT)
Bit-exact source: https://github.com/johndpope/llama-cpp-turboquant
"""

from .constants import (
    PLANAR_BITS,
    PLANAR_CENTROIDS_3BIT,
    PLANAR_COS_64,
    PLANAR_D,
    PLANAR_PAIRS,
    PLANAR_SIN_64,
    centroids_mx,
    cos_sin_mx,
)
from .reference import dequantize_block, quantize_block, roundtrip

__all__ = [
    "PLANAR_D",
    "PLANAR_PAIRS",
    "PLANAR_BITS",
    "PLANAR_CENTROIDS_3BIT",
    "PLANAR_COS_64",
    "PLANAR_SIN_64",
    "centroids_mx",
    "cos_sin_mx",
    "quantize_block",
    "dequantize_block",
    "roundtrip",
]
