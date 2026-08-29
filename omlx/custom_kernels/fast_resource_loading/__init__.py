"""Apple Metal Fast Resource Loading bridge for expert streaming."""

from __future__ import annotations

try:
    from ._ext import FastResourceLoader, abi_probe

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on optional native build
    FastResourceLoader = None  # type: ignore[assignment,misc]
    abi_probe = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def available() -> bool:
    """Return whether the native bridge is built and ABI-compatible."""

    if FastResourceLoader is None or abi_probe is None:
        return False
    try:
        import mlx.core as mx

        return abi_probe(mx.zeros((1,), dtype=mx.uint8)) == 1
    except Exception:
        return False


def import_error() -> str | None:
    return str(_IMPORT_ERROR) if _IMPORT_ERROR is not None else None


__all__ = ["FastResourceLoader", "available", "import_error"]
