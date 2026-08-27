# SPDX-License-Identifier: Apache-2.0
"""Decode fast-path kernels (fused residual+RMS norm, ...) with fallback.

Ports of the user's closed-unmerged mlx core PRs so omlx ships the fusion
without waiting on an mlx release. Every public symbol degrades to the
composed mlx ops when the native extension is absent, ABI-mismatched, or
the shape/dtype is unsupported — callers can use these unconditionally.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

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
            "%s: native extension is present but failed to load; falling "
            "back to the composed path: %s",
            __name__,
            _IMPORT_ERROR,
        )
else:
    _IMPORT_ERROR = None


def _verify_abi(ext, import_error):
    """Disable native symbols when the extension rejects mlx arrays (#2139)."""
    if ext is None:
        return ext, import_error
    probe = getattr(ext, "abi_probe", None)
    if probe is None:
        return ext, import_error
    try:
        probe(mx.zeros((1,)))
    except TypeError as exc:
        logger.warning(
            "%s: native kernels disabled — nanobind ABI mismatch with the "
            "installed mlx wheel; rebuild against the installed mlx.",
            __name__,
        )
        return None, _detach_import_error(exc)
    return ext, import_error


_ext, _IMPORT_ERROR = _verify_abi(_ext, _IMPORT_ERROR)

NATIVE_AVAILABLE = _ext is not None


def _composed_rms_norm_residual(
    x: mx.array, weight: mx.array, residual: mx.array, eps: float
) -> Tuple[mx.array, mx.array]:
    summed = x + residual
    out = mx.fast.rms_norm(summed, weight, eps)
    return out, summed


def rms_norm_residual(
    x: mx.array,
    weight: mx.array,
    residual: mx.array,
    eps: float,
    *,
    stream: Optional[mx.Stream] = None,
    force_composed: bool = False,
) -> Tuple[mx.array, mx.array]:
    """Return (rms_norm(x + residual) * weight, x + residual).

    Single fused Metal dispatch when the native extension applies; composed
    add + mx.fast.rms_norm otherwise. NOTE: the fused kernel requires dense
    rows (row-contiguous, unit-stride last axis — always true for hidden
    states produced by matmul/attention/add). Layout cannot be vetted on
    lazy arrays, so do not route arbitrary strided views through the native
    path; use force_composed=True for those.
    """
    if not force_composed and _ext is not None and _ext.rms_norm_residual_supported(
        x, weight, residual, stream
    ):
        out, summed = _ext.rms_norm_residual(x, weight, residual, eps, stream)
        return out, summed
    return _composed_rms_norm_residual(x, weight, residual, eps)


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
    """Decode-mode SDPA (query length <= 8) with the ported #4294 kernels.

    Falls back to mx.fast.scaled_dot_product_attention when the native
    extension is unavailable or the shapes/dtypes are unsupported. As with
    rms_norm_residual, realized layouts must satisfy the vectorized-load
    stride predicates (always true for omlx KV caches); exotic views should
    use force_fallback=True.
    """
    if (
        not force_fallback
        and _ext is not None
        and _ext.sdpa_decode_supported(q, k, v, stream)
    ):
        return _ext.sdpa_decode(q, k, v, scale, causal, mask, sinks, stream)
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=mask, sinks=sinks
    )


def rope_kv_append(
    keys: mx.array,
    values: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    offset: int,
    dims: int,
    traditional: bool = False,
    base: Optional[float] = None,
    scale: float = 1.0,
    freqs: Optional[mx.array] = None,
    *,
    stream: Optional[mx.Stream] = None,
    force_fallback: bool = False,
) -> Tuple[mx.array, mx.array]:
    """Fused RoPE(K) + KV cache append for single-token decode (#4297 port).

    Returns (key_cache, value_cache) with the rotated K written directly
    into the donated K-cache buffer (no rope-temp round-trip) and V appended
    via slice update. Falls back to composed rope + slice assignment.
    """
    if (
        not force_fallback
        and _ext is not None
        and _ext.rope_kv_append_supported(
            keys, values, key_cache, value_cache, offset, dims, stream
        )
    ):
        kc, vc = _ext.rope_kv_append(
            keys,
            values,
            key_cache,
            value_cache,
            offset,
            dims,
            traditional,
            base,
            scale,
            freqs,
            stream,
        )
        return kc, vc
    k_rot = mx.fast.rope(
        keys, dims, traditional=traditional, base=base, scale=scale, offset=offset, freqs=freqs
    )
    # In-place slice assignment mirrors the native update semantics.
    key_cache[:, :, offset : offset + keys.shape[-2], :] = k_rot
    value_cache[:, :, offset : offset + values.shape[-2], :] = values
    return key_cache, value_cache


__all__ = ["NATIVE_AVAILABLE", "rms_norm_residual", "sdpa_decode", "rope_kv_append"]
