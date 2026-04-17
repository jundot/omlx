# SPDX-License-Identifier: Apache-2.0
"""Utility exports for oMLX.

Keep package initialization lightweight so importing submodules like
``omlx.utils.install`` does not eagerly trigger hardware/MLX detection.
"""

from importlib import import_module

_LAZY_EXPORTS = {
    "get_tokenizer_config": ("omlx.utils.tokenizer", "get_tokenizer_config"),
    "apply_qwen3_fix": ("omlx.utils.tokenizer", "apply_qwen3_fix"),
    "format_bytes_util": ("omlx.utils.formatting", "format_bytes"),
    "get_cli_prefix": ("omlx.utils.install", "get_cli_prefix"),
    "get_install_method": ("omlx.utils.install", "get_install_method"),
    "is_app_bundle": ("omlx.utils.install", "is_app_bundle"),
    "is_homebrew": ("omlx.utils.install", "is_homebrew"),
    "HardwareInfo": ("omlx.utils.hardware", "HardwareInfo"),
    "detect_hardware": ("omlx.utils.hardware", "detect_hardware"),
    "get_chip_name": ("omlx.utils.hardware", "get_chip_name"),
    "get_total_memory_bytes": ("omlx.utils.hardware", "get_total_memory_bytes"),
    "get_total_memory_gb": ("omlx.utils.hardware", "get_total_memory_gb"),
    "get_max_working_set_bytes": (
        "omlx.utils.hardware",
        "get_max_working_set_bytes",
    ),
    "get_mlx_device_name": ("omlx.utils.hardware", "get_mlx_device_name"),
    "is_mlx_available": ("omlx.utils.hardware", "is_mlx_available"),
    "is_apple_silicon": ("omlx.utils.hardware", "is_apple_silicon"),
    "get_mlx_version": ("omlx.utils.hardware", "get_mlx_version"),
    "get_mlx_lm_version": ("omlx.utils.hardware", "get_mlx_lm_version"),
    "get_mlx_vlm_version": ("omlx.utils.hardware", "get_mlx_vlm_version"),
    "format_bytes": ("omlx.utils.hardware", "format_bytes"),
    "DEFAULT_MEMORY_BYTES": ("omlx.utils.hardware", "DEFAULT_MEMORY_BYTES"),
}

__all__ = [
    "get_tokenizer_config",
    "apply_qwen3_fix",
    "format_bytes_util",
    "get_cli_prefix",
    "get_install_method",
    "is_app_bundle",
    "is_homebrew",
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
]


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'omlx.utils' has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
