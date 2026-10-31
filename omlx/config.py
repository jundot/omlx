# SPDX-License-Identifier: Apache-2.0
"""Size parsing shared by the CLI, settings and admin routes."""

import math


def validate_decode_fair_share(value: float) -> float:
    """Validate the global decode-debt multiplier without importing MLX."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("decode_fair_share must be a finite number between 0 and 1")
    return float(value)


def parse_size(size_str: str) -> int:
    """
    Parse a human-readable size string to bytes.

    Args:
        size_str: Size string like "100GB", "50MB", "1TB".

    Returns:
        Size in bytes.
    """
    size_str = size_str.strip().upper()

    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }

    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                value = float(size_str[: -len(unit)])
                byte_value = value * multiplier
                if not math.isfinite(byte_value):
                    raise ValueError
                return int(byte_value)
            except ValueError:
                pass

    # Try parsing as plain number (bytes)
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size string: {size_str}")

