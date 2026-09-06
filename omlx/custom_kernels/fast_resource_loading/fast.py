"""Availability facade used by the native-kernel status endpoint."""

from __future__ import annotations

from . import available, import_error


def is_native_available() -> bool:
    return available()


__all__ = ["import_error", "is_native_available"]
