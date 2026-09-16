# SPDX-License-Identifier: Apache-2.0
"""FP8 E4M3 / UE8M0 helpers for DeepSeek-V4.1 Engram tables."""
from __future__ import annotations

from typing import Optional

import numpy as np

_E4M3_LUT: Optional[np.ndarray] = None


def _ue8m0_to_float32(scale_u8: np.ndarray) -> np.ndarray:
    """Decode UE8M0 / float8_e8m0fnu bytes to float32 powers of two."""
    return np.ldexp(1.0, scale_u8.astype(np.int32) - 127).astype(np.float32)


def _e4m3_to_float32(weight_u8: np.ndarray) -> np.ndarray:
    """Decode float8_e4m3fn bytes to float32."""
    try:
        import ml_dtypes

        return weight_u8.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    except Exception:
        # Manual E4M3: 1 sign, 4 exp, 3 mant; bias 7; max finite 448.
        u = weight_u8.astype(np.uint8)
        sign = ((u >> 7) & 1).astype(np.float32)
        exp = (u >> 3) & 0xF
        mant = u & 0x7
        out = np.empty(u.shape, dtype=np.float32)
        # zeros
        zero = exp == 0
        # subnormals: (-1)^s * 2^(1-bias) * (mant/8)
        sub = zero & (mant != 0)
        out[zero & (mant == 0)] = 0.0
        out[sub] = (mant[sub].astype(np.float32) / 8.0) * np.ldexp(1.0, 1 - 7)
        # normals
        norm = ~zero
        out[norm] = (1.0 + mant[norm].astype(np.float32) / 8.0) * np.ldexp(
            1.0, exp[norm].astype(np.int32) - 7
        )
        out[sign == 1] = -out[sign == 1]
        return out

def _e4m3_lut() -> np.ndarray:
    global _E4M3_LUT
    if _E4M3_LUT is None:
        _E4M3_LUT = _e4m3_to_float32(np.arange(256, dtype=np.uint8))
    return _E4M3_LUT
