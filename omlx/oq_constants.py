# SPDX-License-Identifier: Apache-2.0
"""Shared constants and helpers for quantization.

This module holds values and helpers that are needed by both ``oq.py``
and ``oq_policies.py`` so that neither module imports the other at
import-time (which would cause a circular-import error).
"""

from __future__ import annotations

from typing import Callable


# Quantization level configuration
OQ_LEVELS = {2, 3, 3.5, 4, 5, 6, 8}
OQ_DTYPES: tuple[str, ...] = ("bfloat16", "float16")
_OQ_DEFAULT_GROUP_SIZE = 64
_MAX_MODEL_RAM_FRACTION = 0.8
_PROXY_QUANT_BITS = 4
_PROXY_QUANT_GROUP_SIZE = 64

_LEVEL_BITS: dict[float, int] = {2: 2, 3: 3, 3.5: 3, 4: 4, 5: 5, 6: 6, 8: 8}
_LEVEL_PROTECTION: dict[float, str] = {
    2: "full", 3: "full", 3.5: "full",
    4: "full", 5: "full", 6: "full", 8: "full",
}
_OQ_BPW_TARGETS: dict[float, tuple[float, float]] = {
    2: (2.8, 3.0),
    3: (3.5, 3.7),
    3.5: (3.8, 4.0),
    4: (4.6, 4.7),
    5: (5.5, 5.7),
    6: (6.5, 6.7),
}

# --- helpers ---


def _extract_layer_index(path: str) -> int:
    """Extract transformer layer index from module path. Returns -1 if absent."""
    import re
    m = re.search(r"layers\.(\d+)\.", path)
    return int(m.group(1)) if m else -1


def _mode_for_bits(bits: int) -> str:
    """Select quantization mode. Always affine to minimize kernel combos."""
    return "affine"


def _gs_for_mode(bits: int, default_gs: int) -> int:
    """Get group_size. Always default to minimize kernel combos."""
    return default_gs


def _bits_fn_factory(base_bits: int) -> Callable[[int], dict]:
    """Create a bits-calculating function that closes over base_bits."""
    def _bits(n: int) -> dict:
        effective = int(max(n, base_bits))
        return {
            "bits": effective,
            "group_size": _OQ_DEFAULT_GROUP_SIZE,
            "mode": _mode_for_bits(effective),
        }
    return _bits
