"""Apple Metal Fast Resource Loading bridge for expert streaming."""

from __future__ import annotations

try:
    from ._ext import (
        FastResourceLoader,
        abi_probe,
        check_async_route_error,
        eval_with_gil_released,
        publish_route_maps,
        resolve_route_async,
        synchronize_with_gil_released,
    )

    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on optional native build
    FastResourceLoader = None  # type: ignore[assignment,misc]
    abi_probe = None  # type: ignore[assignment]
    check_async_route_error = None  # type: ignore[assignment]
    eval_with_gil_released = None  # type: ignore[assignment]
    publish_route_maps = None  # type: ignore[assignment]
    resolve_route_async = None  # type: ignore[assignment]
    synchronize_with_gil_released = None  # type: ignore[assignment]
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


__all__ = [
    "FastResourceLoader",
    "available",
    "check_async_route_error",
    "eval_with_gil_released",
    "import_error",
    "publish_route_maps",
    "resolve_route_async",
    "synchronize_with_gil_released",
]
