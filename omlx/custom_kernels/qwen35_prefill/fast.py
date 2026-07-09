"""Fast Qwen3.5/3.6 prefill kernels with optional native dispatch."""

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
    from . import _ext
except Exception as exc:  # pragma: no cover - depends on local native build
    _ext = None
    _IMPORT_ERROR = _detach_import_error(exc)
else:
    _IMPORT_ERROR = None


def _verify_abi(ext, import_error):
    """Disable the native symbols when the extension rejects mlx arrays.

    An extension built with a nanobind whose ABI tag differs from the mlx
    wheel's imports cleanly and lists every symbol, but its type casters
    live in an isolated NB_DOMAIN, so every call raises ``TypeError:
    incompatible function arguments`` (issue #2139). Probe once at import
    and degrade with a single warning instead of failing per call; builds
    predating the ``abi_probe`` binding are assumed compatible.
    """
    if ext is None:
        return ext, import_error
    probe = getattr(ext, "abi_probe", None)
    if probe is None:
        return ext, import_error
    try:
        probe(mx.zeros((1,)))
    except TypeError as exc:
        logger.warning(
            "%s: native kernels disabled — the extension was built with a "
            "nanobind ABI that does not match this mlx wheel; rebuild it "
            "against the installed mlx (see pyproject build-system pins).",
            __name__,
        )
        return None, _detach_import_error(exc)
    return ext, import_error


_ext, _IMPORT_ERROR = _verify_abi(_ext, _IMPORT_ERROR)


NATIVE_SYMBOLS = (
    "qwen35_fa256_attention",
    "qwen35_q4_affine_qmm_t",
    "qwen35_q5_affine_qmm_t",
    "qwen35_q6_affine_qmm_t",
    "qwen35_q8_affine_qmm_t",
    "qwen35_moe_weighted_sum",
)


def is_native_available() -> bool:
    return _ext is not None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def _has_weighted_sum() -> bool:
    return hasattr(_ext, "qwen35_moe_weighted_sum") or hasattr(
        mx.fast, "qwen35_moe_weighted_sum"
    )


def has_symbol(name: str) -> bool:
    if name == "qwen35_moe_weighted_sum":
        return _has_weighted_sum()
    return hasattr(_ext, name) or hasattr(mx.fast, name)


def native_symbols() -> tuple[str, ...]:
    symbols: list[str] = []
    if _ext is not None:
        symbols.extend(name for name in NATIVE_SYMBOLS if hasattr(_ext, name))
    if _has_weighted_sum() and "qwen35_moe_weighted_sum" not in symbols:
        symbols.append("qwen35_moe_weighted_sum")
    return tuple(symbols)


def missing_symbols(required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not has_symbol(name)]


def _native_stream_kwargs(stream) -> dict[str, object]:
    """Accept the same stream shorthand that mlx.fast kernels accept."""
    if isinstance(stream, mx.DeviceType):
        stream = None
    return {"stream": stream}


def qwen35_fa256_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool = True,
    q_block: int = 32,
    k_block: int = 8,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "qwen35_fa256_attention"):
        return _ext.qwen35_fa256_attention(
            q,
            k,
            v,
            scale,
            causal=causal,
            q_block=q_block,
            k_block=k_block,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("qwen35_fa256_attention native kernel is unavailable")


def qwen35_q4_affine_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "qwen35_q4_affine_qmm_t"):
        return _ext.qwen35_q4_affine_qmm_t(
            x,
            weight,
            scales,
            biases,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("qwen35_q4_affine_qmm_t native kernel is unavailable")


def qwen35_q5_affine_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "qwen35_q5_affine_qmm_t"):
        return _ext.qwen35_q5_affine_qmm_t(
            x,
            weight,
            scales,
            biases,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("qwen35_q5_affine_qmm_t native kernel is unavailable")


def qwen35_q6_affine_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "qwen35_q6_affine_qmm_t"):
        return _ext.qwen35_q6_affine_qmm_t(
            x,
            weight,
            scales,
            biases,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("qwen35_q6_affine_qmm_t native kernel is unavailable")


def qwen35_q8_affine_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "qwen35_q8_affine_qmm_t"):
        return _ext.qwen35_q8_affine_qmm_t(
            x,
            weight,
            scales,
            biases,
            variant,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("qwen35_q8_affine_qmm_t native kernel is unavailable")


def qwen35_moe_weighted_sum(
    x_sorted: mx.array,
    inv_order: mx.array,
    scores: mx.array,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "qwen35_moe_weighted_sum"):
        return _ext.qwen35_moe_weighted_sum(
            x_sorted,
            inv_order,
            scores,
            **_native_stream_kwargs(stream),
        )
    if hasattr(mx.fast, "qwen35_moe_weighted_sum"):
        return mx.fast.qwen35_moe_weighted_sum(
            x_sorted,
            inv_order,
            scores,
            stream=stream or mx.gpu,
        )
    raise RuntimeError("qwen35_moe_weighted_sum native kernel is unavailable")


def __getattr__(name: str) -> Any:
    if _ext is not None and hasattr(_ext, name):
        return getattr(_ext, name)
    return getattr(mx.fast, name)


def __dir__() -> list[str]:
    names = set(globals())
    names.update(NATIVE_SYMBOLS)
    names.update(dir(mx.fast))
    if _ext is not None:
        names.update(dir(_ext))
    return sorted(names)
