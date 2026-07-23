"""Fast MiniMax M3 kernels with optional native extension dispatch."""

from __future__ import annotations

import logging
from pathlib import Path
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
    # Default installs ship no extension; warn only when a built _ext fails
    # to load (e.g. unresolved @rpath/libmlx.dylib, issue #2233) so the
    # silent-slow-path fallback leaves a trace in the server log.
    if any(Path(__file__).parent.glob("_ext*.so")):
        logger.warning(
            "%s: native extension is present but failed to load; falling "
            "back to the slow path: %s",
            __name__,
            _IMPORT_ERROR,
        )
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


NATIVE_SYMBOLS = ("minimax_msa_topk",)


def is_native_available() -> bool:
    return _ext is not None


def import_error() -> Exception | None:
    return _IMPORT_ERROR


def has_symbol(name: str) -> bool:
    return hasattr(_ext, name) or hasattr(mx.fast, name)


def native_symbols() -> tuple[str, ...]:
    if _ext is None:
        return ()
    return tuple(name for name in NATIVE_SYMBOLS if hasattr(_ext, name))


def _native_stream_kwargs(stream) -> dict[str, object]:
    if isinstance(stream, mx.DeviceType):
        stream = None
    return {"stream": stream}


def minimax_msa_topk(
    idx_queries: mx.array,
    idx_keys: mx.array,
    *,
    q_start: int,
    scale: float,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    stream=None,
) -> mx.array:
    if _ext is not None:
        return _ext.minimax_msa_topk(
            idx_queries,
            idx_keys,
            q_start,
            scale,
            block_size,
            topk,
            init_blocks,
            local_blocks,
            **_native_stream_kwargs(stream),
        )
    if not hasattr(mx.fast, "minimax_msa_topk"):
        raise AttributeError("mx.fast.minimax_msa_topk is unavailable")
    return mx.fast.minimax_msa_topk(
        idx_queries,
        idx_keys,
        q_start=q_start,
        scale=scale,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        stream=stream or mx.gpu,
    )


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
