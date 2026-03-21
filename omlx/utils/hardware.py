# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-mlx (https://github.com/vllm-project/vllm-mlx).
"""
Unified hardware detection for Apple Silicon.

Single source of truth for:
- Chip identification (M1, M2, M3, M4 series)
- Memory detection (total, available, max working set)
- MLX availability checks
"""

from __future__ import annotations

import hashlib
import logging
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)

# Default fallback value for all memory functions (conservative)
DEFAULT_MEMORY_BYTES = 8 * 1024 * 1024 * 1024  # 8GB


@dataclass
class HardwareInfo:
    """Hardware information for Apple Silicon."""

    chip_name: str
    total_memory_gb: float
    max_working_set_bytes: int
    mlx_device_name: Optional[str] = None


# =============================================================================
# Core Detection Functions
# =============================================================================


def get_chip_name() -> str:
    """
    Get Apple Silicon chip name via sysctl.

    Returns:
        Chip name (e.g., "Apple M4 Pro") or "Apple Silicon" as fallback.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "Apple Silicon"


def get_total_memory_bytes() -> int:
    """
    Get total unified memory in bytes.

    Fallback chain:
    1. sysctl hw.memsize (most reliable)
    2. mlx.metal.device_info()["memory_size"]
    3. DEFAULT_MEMORY_BYTES (8GB)

    Returns:
        Total memory in bytes.
    """
    # Primary: sysctl
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip())
    except Exception:
        pass

    # Fallback: MLX Metal
    if HAS_MLX:
        try:
            if mx.metal.is_available():
                device_info = mx.device_info()
                if "memory_size" in device_info:
                    return int(device_info["memory_size"])
        except Exception:
            pass

    # Last resort: default
    logger.warning(f"Using default memory size: {DEFAULT_MEMORY_BYTES // (1024**3)} GB")
    return DEFAULT_MEMORY_BYTES


def get_total_memory_gb() -> float:
    """Get total unified memory in GB."""
    return get_total_memory_bytes() / (1024**3)


def get_max_working_set_bytes() -> int:
    """
    Get max_recommended_working_set_size from MLX Metal.

    Fallback chain:
    1. mlx.metal.device_info()["max_recommended_working_set_size"]
    2. psutil.virtual_memory().total * 0.75
    3. DEFAULT_MEMORY_BYTES (8GB)

    Returns:
        Maximum working set size in bytes.
    """
    # Primary: MLX Metal
    if HAS_MLX:
        try:
            if mx.metal.is_available():
                device_info = mx.device_info()
                max_working_set = device_info.get("max_recommended_working_set_size", 0)
                if max_working_set > 0:
                    return max_working_set
        except Exception:
            pass

    # Fallback: psutil with 75% heuristic
    try:
        import psutil

        total_ram = psutil.virtual_memory().total
        return int(total_ram * 0.75)
    except ImportError:
        pass

    # Last resort: default
    logger.warning(
        f"Using default max working set: {DEFAULT_MEMORY_BYTES // (1024**3)} GB"
    )
    return DEFAULT_MEMORY_BYTES


def get_mlx_device_name() -> Optional[str]:
    """Get raw device name from MLX Metal API."""
    if HAS_MLX:
        try:
            if mx.metal.is_available():
                device_info = mx.device_info()
                return device_info.get("device_name")
        except Exception:
            pass
    return None


def detect_hardware() -> HardwareInfo:
    """
    Detect Apple Silicon hardware and return complete info.

    Returns:
        HardwareInfo with all hardware specifications.
    """
    return HardwareInfo(
        chip_name=get_chip_name(),
        total_memory_gb=get_total_memory_gb(),
        max_working_set_bytes=get_max_working_set_bytes(),
        mlx_device_name=get_mlx_device_name(),
    )


# =============================================================================
# MLX Availability Checks
# =============================================================================


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (arm64 macOS)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def is_mlx_available() -> bool:
    """Check if MLX is available and working."""
    if not is_apple_silicon():
        return False
    if not HAS_MLX:
        return False

    try:
        # Verify we can actually use MLX
        _ = mx.array([1.0, 2.0, 3.0])
        return True
    except Exception:
        return False


# =============================================================================
# Version Information
# =============================================================================


def get_mlx_version() -> str:
    """Get MLX version string."""
    try:
        import mlx

        return getattr(mlx, "__version__", "Unknown")
    except Exception:
        return "Unknown"


def get_mlx_lm_version() -> str:
    """Get mlx-lm version string."""
    try:
        import mlx_lm

        return getattr(mlx_lm, "__version__", "Unknown")
    except Exception:
        return "Unknown"


def get_mlx_vlm_version() -> str:
    """Get mlx-vlm version string."""
    try:
        import mlx_vlm

        return getattr(mlx_vlm, "__version__", "Unknown")
    except Exception:
        return "Unknown"


# =============================================================================
# Benchmark / omlx.ai Integration
# =============================================================================

_OWNER_HASH_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def get_gpu_core_count() -> Optional[int]:
    """Get GPU core count via system_profiler."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if "Total Number of Cores" in line:
                match = re.search(r"(\d+)", line)
                if match:
                    return int(match.group(1))
    except Exception:
        pass
    return None


def get_io_platform_uuid() -> Optional[str]:
    """Get IOPlatformUUID from ioreg (unique per device)."""
    try:
        result = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.splitlines():
            if "IOPlatformUUID" in line:
                match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', line)
                if match:
                    return match.group(1)
    except Exception:
        pass
    return None


def parse_chip_info(chip_string: str) -> tuple[str, str]:
    """Parse chip name and variant from sysctl brand string.

    Args:
        chip_string: e.g. "Apple M4 Pro", "Apple M3 Max", "Apple M2"

    Returns:
        (chip_name, chip_variant) e.g. ("M4", "Pro"), ("M3", "Max"), ("M2", "")
    """
    match = re.search(r"M(\d+)\s*(Pro|Max|Ultra)?", chip_string)
    if not match:
        return ("M1", "")
    chip_name = f"M{match.group(1)}"
    chip_variant = match.group(2) or ""
    return (chip_name, chip_variant)


def compute_owner_hash(
    uuid: str, chip_name: str, gpu_cores: Optional[int], memory_gb: int
) -> str:
    """Compute owner_hash for omlx.ai benchmark submissions.

    Format: SHA-256(uuid + chip_name + gpu_cores + memory_gb) + verify_char
    The verify_char is ALPHABET[sum(charCodes of hash) % 36].

    Returns:
        Full owner_hash including verify character.
    """
    raw = f"{uuid}{chip_name}{gpu_cores}{memory_gb}"
    hash_hex = hashlib.sha256(raw.encode()).hexdigest()
    verify_sum = sum(ord(c) for c in hash_hex)
    verify_char = _OWNER_HASH_ALPHABET[verify_sum % 36]
    return hash_hex + verify_char


def get_os_version() -> str:
    """Get macOS version string (e.g. 'macOS 15.2')."""
    try:
        mac_ver = platform.mac_ver()[0]
        if mac_ver:
            return f"macOS {mac_ver}"
    except Exception:
        pass
    return "macOS"


# =============================================================================
# Utility Functions
# =============================================================================


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string (e.g., '16.00 GB')."""
    if bytes_value >= 1024**3:
        return f"{bytes_value / 1024**3:.2f} GB"
    elif bytes_value >= 1024**2:
        return f"{bytes_value / 1024**2:.2f} MB"
    elif bytes_value >= 1024:
        return f"{bytes_value / 1024:.2f} KB"
    else:
        return f"{bytes_value} B"
