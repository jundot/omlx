# SPDX-License-Identifier: Apache-2.0
"""Formatting utilities for oMLX."""


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string.

    Args:
        bytes_value: Number of bytes to format

    Returns:
        Human-readable string (e.g., "1.50 GB", "256.00 MB")
    """
    if bytes_value >= 1024**3:
        return f"{bytes_value / 1024**3:.2f} GB"
    elif bytes_value >= 1024**2:
        return f"{bytes_value / 1024**2:.2f} MB"
    elif bytes_value >= 1024:
        return f"{bytes_value / 1024:.2f} KB"
    else:
        return f"{bytes_value} B"


def make_json_safe(obj: object) -> object:
    """Recursively convert non-JSON-serializable objects (like Exceptions) to strings."""
    if isinstance(obj, Exception):
        return str(obj)
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    return obj
