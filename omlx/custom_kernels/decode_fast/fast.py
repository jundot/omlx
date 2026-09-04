# SPDX-License-Identifier: Apache-2.0
"""Exact decode-mode SDPA with a fail-closed MLX fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def _detach_import_error(exc: Exception) -> Exception:
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None
    return exc


try:
    from . import _ext
except Exception as exc:  # pragma: no cover - depends on local native build
    _ext = None
    _IMPORT_ERROR = _detach_import_error(exc)
    if any(Path(__file__).parent.glob("_ext*.so")):
        logger.warning(
            "%s: native extension failed to load; using exact MLX SDPA: %s",
            __name__,
            _IMPORT_ERROR,
        )
else:
    _IMPORT_ERROR = None


def _verify_abi(ext, import_error):
    """Disable the native symbol when nanobind rejects MLX arrays."""
    if ext is None:
        return ext, import_error
    probe = getattr(ext, "abi_probe", None)
    if probe is None:
        return ext, import_error
    try:
        probe(mx.zeros((1,)))
    except TypeError as exc:
        logger.warning(
            "%s: native SDPA disabled after nanobind ABI mismatch",
            __name__,
        )
        return None, _detach_import_error(exc)
    return ext, import_error


_ext, _IMPORT_ERROR = _verify_abi(_ext, _IMPORT_ERROR)
NATIVE_AVAILABLE = _ext is not None


def is_native_available() -> bool:
    return NATIVE_AVAILABLE


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def sdpa_decode(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool = False,
    mask: Optional[mx.array] = None,
    sinks: Optional[mx.array] = None,
    *,
    stream: Optional[mx.Stream] = None,
    force_fallback: bool = False,
) -> mx.array:
    """Run native decode SDPA when its exact ABI accepts the inputs."""
    if (
        not force_fallback
        and _ext is not None
        and _ext.sdpa_decode_supported(q, k, v, stream)
    ):
        return _ext.sdpa_decode(q, k, v, scale, causal, mask, sinks, stream)
    return mx.fast.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=scale,
        mask=mask,
        sinks=sinks,
    )


__all__ = ["NATIVE_AVAILABLE", "is_native_available", "import_error", "sdpa_decode"]
