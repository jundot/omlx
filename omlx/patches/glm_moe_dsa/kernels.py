# SPDX-License-Identifier: Apache-2.0
"""Fast-kernel dispatch for the GLM MoE DSA monkey patch."""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


def _detach_import_error(exc: Exception) -> Exception:
    """Keep the diagnostic message without retaining import caller frames."""
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None
    return exc


try:
    from omlx.custom_kernels.glm_moe_dsa import fast as _native_fast
except Exception as exc:  # pragma: no cover - depends on native extension build
    _native_fast = None
    _native_import_error = _detach_import_error(exc)
else:
    _native_import_error = None


class _FastDispatch:
    def __getattr__(self, name: str) -> Any:
        if _native_fast is not None and _native_fast.has_symbol(name):
            try:
                return getattr(_native_fast, name)
            except AttributeError:
                pass
        return getattr(mx.fast, name)

    def __dir__(self) -> list[str]:
        names = set(dir(mx.fast))
        if _native_fast is not None:
            names.update(dir(_native_fast))
        return sorted(names)

    def has(self, name: str) -> bool:
        return (
            (_native_fast is not None and _native_fast.has_symbol(name))
            or hasattr(mx.fast, name)
        )

    def missing(self, required: tuple[str, ...]) -> list[str]:
        return [name for name in required if not self.has(name)]

    def native_available(self) -> bool:
        return _native_fast is not None and _native_fast.is_native_available()

    def native_import_error(self) -> Exception | None:
        if _native_fast is not None:
            return _native_fast.import_error()
        return _native_import_error


fast = _FastDispatch()

__all__ = ["fast"]
