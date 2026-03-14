# SPDX-License-Identifier: Apache-2.0
"""
Utility modules for oMLX.

This package contains shared utility functions and helpers used across
the oMLX codebase.
"""

from .tokenizer import get_tokenizer_config, apply_qwen3_fix
from .formatting import format_bytes as format_bytes_util
from .install import get_cli_prefix, is_app_bundle
from .hardware import (
    HardwareInfo,
    detect_hardware,
    get_chip_name,
    get_total_memory_bytes,
    get_total_memory_gb,
    get_max_working_set_bytes,
    get_mlx_device_name,
    is_mlx_available,
    is_apple_silicon,
    get_mlx_version,
    get_mlx_lm_version,
    get_mlx_vlm_version,
    format_bytes,
    DEFAULT_MEMORY_BYTES,
)

__all__ = [
    # Tokenizer utilities
    "get_tokenizer_config",
    "apply_qwen3_fix",
    # Hardware utilities
    "HardwareInfo",
    "detect_hardware",
    "get_chip_name",
    "get_total_memory_bytes",
    "get_total_memory_gb",
    "get_max_working_set_bytes",
    "get_mlx_device_name",
    "is_mlx_available",
    "is_apple_silicon",
    "get_mlx_version",
    "get_mlx_lm_version",
    "get_mlx_vlm_version",
    "format_bytes",
    "DEFAULT_MEMORY_BYTES",
    # Install detection
    "get_cli_prefix",
    "is_app_bundle",
]
