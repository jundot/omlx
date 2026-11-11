# SPDX-License-Identifier: Apache-2.0
"""Import and ABI checks shared by the native kernel packages."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import ModuleType

import mlx.core as mx

logger = logging.getLogger(__name__)


def detach_import_error(exc: Exception) -> Exception:
    """Keep the diagnostic message without retaining import caller frames."""
    exc.__traceback__ = None
    exc.__cause__ = None
    exc.__context__ = None
    return exc


def verify_abi(
    ext: ModuleType | None, import_error: Exception | None, package: str
) -> tuple[ModuleType | None, Exception | None]:
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
            package,
        )
        return None, detach_import_error(exc)
    return ext, import_error


def load_extension(
    package: str, package_dir: Path
) -> tuple[ModuleType | None, Exception | None]:
    """Import ``<package>._ext`` and return ``(ext, import_error)``."""
    try:
        ext = importlib.import_module(f"{package}._ext")
    except Exception as exc:  # pragma: no cover - depends on local native build
        error = detach_import_error(exc)
        # Default installs ship no extension; warn only when a built _ext fails
        # to load (e.g. unresolved @rpath/libmlx.dylib, issue #2233) so the
        # silent-slow-path fallback leaves a trace in the server log.
        if any(package_dir.glob("_ext*.so")):
            logger.warning(
                "%s: native extension is present but failed to load; falling "
                "back to the slow path: %s",
                package,
                error,
            )
        return None, error
    return verify_abi(ext, None, package)


def native_stream_kwargs(stream) -> dict[str, object]:
    """Accept the same stream shorthand that mlx.fast kernels accept."""
    if isinstance(stream, mx.DeviceType):
        stream = None
    return {"stream": stream}
