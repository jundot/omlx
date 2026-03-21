# SPDX-License-Identifier: Apache-2.0
"""
Hardware detection and system information for oMLX.

This module provides:
- Hardware detection for Apple Silicon
- System memory detection

Note: mlx-lm already includes optimized implementations internally:
- Flash Attention via mx.fast.scaled_dot_product_attention
- Efficient memory management
- Optimized Metal kernels

No additional optimization is needed - mlx-lm is already fast out of the box.

Usage:
    from omlx.optimizations import (
        detect_hardware,
        get_optimization_status,
    )
"""

import logging

import mlx.core as mx

# Re-export from hardware module for backward compatibility
from omlx.utils.hardware import (
    HardwareInfo,
    detect_hardware,
    get_total_memory_gb as get_system_memory_gb,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HardwareInfo",
    "detect_hardware",
    "get_system_memory_gb",
    "get_optimization_status",
]



def get_optimization_status() -> dict:
    """
    Get current hardware and MLX status.

    Returns:
        dict with hardware info and MLX configuration
    """
    hw = detect_hardware()
    device_info = mx.device_info()
    flash_available = hasattr(mx, "fast") and hasattr(
        mx.fast, "scaled_dot_product_attention"
    )

    return {
        "hardware": {
            "chip": hw.chip_name,
            "total_memory_gb": hw.total_memory_gb,
            "device_name": device_info.get("device_name", "Unknown"),
        },
        "mlx_memory": {
            "active_bytes": mx.get_active_memory(),
            "cache_bytes": mx.get_cache_memory(),
            "peak_bytes": mx.get_peak_memory(),
        },
        "mlx_lm_features": {
            "flash_attention": "built-in" if flash_available else "not available",
            "metal_kernels": "optimized for Apple Silicon",
            "kv_cache": "managed by mlx-lm",
            "quantization": "4-bit and 8-bit supported",
        },
    }
