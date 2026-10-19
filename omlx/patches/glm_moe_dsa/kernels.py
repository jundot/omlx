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
    from omlx.custom_kernels.attention import fast as _attention_fast
    from omlx.custom_kernels.quant_matmul import fast as _quant_fast
    from omlx.custom_kernels.sparse_index import fast as _index_fast
except Exception as exc:  # pragma: no cover - depends on native extension build
    _native_modules: tuple = ()
    _native_import_error = _detach_import_error(exc)
else:
    _native_modules = (_attention_fast, _index_fast, _quant_fast)
    _native_import_error = None


def _native_module(name: str):
    for module in _native_modules:
        if module.has_symbol(name):
            return module
    return None


class _FastDispatch:
    def __getattr__(self, name: str) -> Any:
        module = _native_module(name)
        if module is not None:
            try:
                return getattr(module, name)
            except AttributeError:
                pass
        return getattr(mx.fast, name)

    def __dir__(self) -> list[str]:
        names = set(dir(mx.fast))
        for module in _native_modules:
            names.update(dir(module))
        return sorted(names)

    def has(self, name: str) -> bool:
        return _native_module(name) is not None or hasattr(mx.fast, name)

    def missing(self, required: tuple[str, ...]) -> list[str]:
        return [name for name in required if not self.has(name)]

    def native_available(self) -> bool:
        return bool(_native_modules) and all(
            module.is_native_available() for module in _native_modules
        )

    def native_import_error(self) -> Exception | None:
        for module in _native_modules:
            error = module.import_error()
            if error is not None:
                return error
        return _native_import_error


fast = _FastDispatch()

__all__ = ["fast"]
