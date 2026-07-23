"""Fast Qwen3.5/3.6 prefill kernels with optional native dispatch."""

from __future__ import annotations

import logging
import os
import platform
import re
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


NATIVE_SYMBOLS = (
    "qwen35_fa256_attention",
    "qwen35_q2_affine_qmm_t",
    "qwen35_q4_affine_qmm_t",
    "qwen35_q5_affine_qmm_t",
    "qwen35_q6_affine_qmm_t",
    "qwen35_q8_affine_qmm_t",
    "qwen35_moe_weighted_sum",
)

# Extensions built before the NAX split reject the use_nax/nax_variant kwargs,
# so only pass them when the rebuilt binding is present.
_EXT_HAS_NAX = _ext is not None and hasattr(_ext, "is_nax_available")

# Extensions built before the chunked-dispatch fix (issue #2225) reject the
# dispatch_budget kwarg, so only forward it when the rebuilt binding is
# present; those older builds keep the single-dispatch behavior.
_EXT_HAS_FA256_DISPATCH_BUDGET = _ext is not None and hasattr(
    _ext, "FA256_HAS_DISPATCH_BUDGET"
)

# Extensions built before the Bonsai group_size=128 support reject the
# group_size kwarg on the qmm bindings, so only forward it when the rebuilt
# binding is present.  The q2 binding ships in the same rebuild, so its
# presence marks the new ABI.  Older builds only carry gs=64 kernels; callers
# must keep gs!=64 layers on stock mlx (see qmm_supports_group_size).
_EXT_HAS_QMM_GROUP_SIZE = _ext is not None and hasattr(
    _ext, "qwen35_q2_affine_qmm_t"
)


def _qmm_group_size_kwargs(group_size: int) -> dict:
    if _EXT_HAS_QMM_GROUP_SIZE:
        return {"group_size": group_size}
    return {}


def qmm_supports_group_size(group_size: int) -> bool:
    """True if the compiled extension can run qmm at this group size.

    gs=64 always works (the pre-group_size builds are gs=64-only).  Other
    group sizes require the rebuilt binding; routing a gs=128 layer through
    an old build would silently run the gs=64 kernel on gs=128 data.
    """
    return group_size == 64 or _EXT_HAS_QMM_GROUP_SIZE

_NAX_ARCH_RE = re.compile(r"applegpu_g(\d+)([a-z])")
_NAX_KERNEL_NEEDLE = b"affine_qmm_t_nax"

_nax_available_cache: bool | None = None
_stock_nax_cache: bool | None = None
_qmm_nax_cache: bool | None = None

QMM_NAX_VARIANT = int(os.environ.get("OMLX_QWEN35_QMM_NAX_VARIANT", "0"))


def _nax_available_fallback(
    version: str | None = None, arch: str | None = None
) -> bool:
    """Python mirror of mlx metal::is_nax_available() for pre-NAX extensions."""
    if version is None or arch is None:
        if not mx.metal.is_available():
            return False
        if version is None:
            version = platform.mac_ver()[0]
        if arch is None:
            arch = str(mx.device_info().get("architecture", ""))
    try:
        release = tuple(int(part) for part in version.split(".")[:2])
    except ValueError:
        return False
    if len(release) < 2 or release < (26, 2):
        return False
    match = _NAX_ARCH_RE.fullmatch(arch)
    if match is None:
        return False
    gen = int(match.group(1))
    suffix = match.group(2)
    return gen >= (18 if suffix == "p" else 17)


def _stock_mlx_has_nax(lib_path: Path | None = None) -> bool:
    """True when the installed mlx wheel ships the NAX kernels.

    Wheels built for macOS < 26.2 are compiled with MLX_METAL_NO_NAX (e.g.
    the macosx_15 wheel in the sequoia app bundle): on those installs stock
    stays on the classic kernels even on M5 hardware, so route-to-stock
    decisions must not fire. Unreadable/absent metallibs (JIT builds) fall
    back to True, which reduces to the hardware-only gate.
    """
    global _stock_nax_cache
    if lib_path is None and _stock_nax_cache is not None:
        return _stock_nax_cache
    path = lib_path
    if path is None:
        core_file = getattr(mx, "__file__", None)
        if core_file is None:
            return True
        path = Path(core_file).parent / "lib" / "mlx.metallib"
    found = True
    try:
        if path.is_file():
            overlap = len(_NAX_KERNEL_NEEDLE) - 1
            tail = b""
            found = False
            with open(path, "rb") as f:
                while chunk := f.read(1 << 23):
                    if _NAX_KERNEL_NEEDLE in tail + chunk:
                        found = True
                        break
                    tail = chunk[-overlap:]
    except OSError:
        found = True
    if lib_path is None:
        _stock_nax_cache = found
    return found


def is_nax_available() -> bool:
    """True when stock MLX will dispatch to the M5 tensor-unit (NAX) kernels.

    Requires both NAX hardware (mirroring mlx metal::is_nax_available) and an
    mlx install whose metallib actually ships the NAX kernels. OMLX_NAX=0/1
    overrides detection (testing only; the native op still refuses NAX
    pipelines on hardware without tensor units).
    """
    global _nax_available_cache
    env = os.environ.get("OMLX_NAX", "").strip().lower()
    if env in ("0", "false", "off"):
        return False
    if env in ("1", "true", "on"):
        return True
    if _nax_available_cache is None:
        if _EXT_HAS_NAX:
            hardware = bool(_ext.is_nax_available())
        else:
            hardware = _nax_available_fallback()
        _nax_available_cache = hardware and _stock_mlx_has_nax()
    return _nax_available_cache


def nax_qmm_kernels_built() -> bool:
    if not _EXT_HAS_NAX:
        return False
    return bool(_ext.nax_qmm_kernels_built())


def _qmm_use_nax() -> bool:
    global _qmm_nax_cache
    if _qmm_nax_cache is None:
        if os.environ.get("OMLX_QWEN35_QMM_NAX", "").strip().lower() in (
            "0",
            "false",
            "off",
        ):
            _qmm_nax_cache = False
        else:
            _qmm_nax_cache = (
                _EXT_HAS_NAX
                and bool(_ext.is_nax_available())
                and bool(_ext.nax_qmm_kernels_built())
            )
        if _qmm_nax_cache:
            logger.info(
                "Qwen qmm NAX dispatch enabled (nax_variant=%d)",
                QMM_NAX_VARIANT,
            )
    return _qmm_nax_cache


def _qmm_nax_kwargs() -> dict[str, object]:
    if not _EXT_HAS_NAX:
        return {}
    return {"use_nax": _qmm_use_nax(), "nax_variant": QMM_NAX_VARIANT}


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


def fa256_supports_dispatch_budget() -> bool:
    return _EXT_HAS_FA256_DISPATCH_BUDGET


def qwen35_fa256_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool = True,
    q_block: int = 32,
    k_block: int = 8,
    dispatch_budget: int = 0,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "qwen35_fa256_attention"):
        budget_kwargs: dict[str, int] = {}
        if _EXT_HAS_FA256_DISPATCH_BUDGET:
            budget_kwargs["dispatch_budget"] = int(dispatch_budget)
        return _ext.qwen35_fa256_attention(
            q,
            k,
            v,
            scale,
            causal=causal,
            q_block=q_block,
            k_block=k_block,
            **budget_kwargs,
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("qwen35_fa256_attention native kernel is unavailable")


def qwen35_q2_affine_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    group_size: int = 64,
    *,
    stream=None,
) -> mx.array:
    if _ext is not None and hasattr(_ext, "qwen35_q2_affine_qmm_t"):
        return _ext.qwen35_q2_affine_qmm_t(
            x,
            weight,
            scales,
            biases,
            variant,
            **_qmm_nax_kwargs(),
            **_qmm_group_size_kwargs(group_size),
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("qwen35_q2_affine_qmm_t native kernel is unavailable")


def qwen35_q4_affine_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    group_size: int = 64,
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
            **_qmm_nax_kwargs(),
            **_qmm_group_size_kwargs(group_size),
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("qwen35_q4_affine_qmm_t native kernel is unavailable")


def qwen35_q5_affine_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    group_size: int = 64,
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
            **_qmm_nax_kwargs(),
            **_qmm_group_size_kwargs(group_size),
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("qwen35_q5_affine_qmm_t native kernel is unavailable")


def qwen35_q6_affine_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    group_size: int = 64,
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
            **_qmm_nax_kwargs(),
            **_qmm_group_size_kwargs(group_size),
            **_native_stream_kwargs(stream),
        )
    raise RuntimeError("qwen35_q6_affine_qmm_t native kernel is unavailable")


def qwen35_q8_affine_qmm_t(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    variant: int = 8,
    group_size: int = 64,
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
            **_qmm_nax_kwargs(),
            **_qmm_group_size_kwargs(group_size),
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
