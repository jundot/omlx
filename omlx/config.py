# SPDX-License-Identifier: Apache-2.0
"""Size parsing shared by the CLI, settings and admin routes."""

import math

# Ceiling for the "auto" SSD cache budget. Without it the budget is half of
# free space plus existing cache, so it grows with the size of the volume.
AUTO_SSD_CACHE_MAX_BYTES = 100 * 1024**3


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

