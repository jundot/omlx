# SPDX-License-Identifier: Apache-2.0
"""Minimal ``torch`` stub for the DMG bundle.

xgrammar 0.2.3 declares ``torch>=1.10.0`` as a runtime dep, but oMLX never
exercises its torch-backed code paths: bitmasks are allocated as numpy
``int32`` buffers, the C++ binding fills them, and the MLX kernel applies the
mask. The torch dep is load-bearing only at *import time* — module-level code
in ``xgrammar.matcher``, ``xgrammar.testing``, ``xgrammar.contrib.hf`` and
``tvm_ffi.core`` does ``import torch`` plus a handful of attribute lookups.

Real torch is ~500 MB unpacked on macOS arm64 — too heavy to ship in the DMG.
This stub provides just enough of the torch surface for those modules to
finish loading. Code paths that would actually call into torch raise
``RuntimeError`` from the helpers below; oMLX never reaches them.

When a real torch is installed (pip / Homebrew flow) the stub is a no-op:
``install()`` checks ``importlib.util.find_spec('torch')`` first.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import os
import sys
import threading
import types

logger = logging.getLogger(__name__)

# xgrammar / tvm-ffi versions this stub is known to cover.
# This module is the *single source of truth* — packaging/build.py imports
# these constants to keep the DMG install pin in sync with the stub. Update
# both tuples here when bumping; the build script auto-tracks.
#
# Reachable-but-stubbed torch surface to be aware of when upgrading:
#   - ``torch.full``: ``xgrammar.allocate_token_bitmask`` calls it. oMLX
#     never invokes ``allocate_token_bitmask`` (we use the MLX kernel
#     path), but the symbol is re-exported from ``xgrammar.__init__``.
#     Any future caller that touches it will hit ``_unsupported("full")``
#     and surface a clear RuntimeError.
#   - ``torch.tensor`` returns a ``_StubTensor`` whose attribute access
#     raises a stub-identifying RuntimeError. Module-level
#     ``_FULL_MASK = torch.tensor(-1, ...)`` patterns succeed at import
#     time; any subsequent method call (.fill_, .item, ...) fails.
_TARGET_XGRAMMAR_VERSIONS = ("0.2.3",)
_TARGET_TVM_FFI_VERSIONS = ("0.1.11",)

# Serialize install() across threads. Without this, two threads that both
# pass the "torch" in sys.modules check race to build modules and overwrite
# each other's sys.modules['torch'] entry, leaving threads that already
# dereferenced the loser's module with stale references. Reachable today
# from concurrent HTTP handlers that call install() on first xgrammar use.
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False


class _StubTensor:
    """Placeholder for ``torch.Tensor`` (annotations + isinstance checks).

    Any attribute access raises a clear RuntimeError so runtime use of a
    stubbed tensor (e.g. ``some_tensor.fill_(...)``) fails loudly with a
    pointer to the cause, rather than at the AttributeError level with a
    generic ``has no attribute 'fill_'`` message.
    """

    def __getattr__(self, name: str):
        # Let dunder probes (pickle, copy.deepcopy, descriptor lookups,
        # `hasattr` chains in third-party libs) fall through cleanly as
        # AttributeError — that's the documented `__getattr__` contract.
        # Real torch tensors lack many of these probed dunders anyway, so
        # raising AttributeError is the correct, distinguishable signal.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise RuntimeError(
            f"_StubTensor.{name} is not implemented: oMLX ships a torch "
            "stub for xgrammar's import-time needs only. Reaching a real "
            "tensor method means a code path that needs real torch was "
            "exercised — install torch via pip/Homebrew or report this as "
            "a bug if the call originated inside oMLX."
        )


class _StubDtype:
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"torch.{self._name}"

    # Some xgrammar/tvm-ffi paths convert dtype to string via ``str(dt)``
    # rather than ``repr(dt)`` (e.g. ``to_cpp_dtype`` strips the "torch."
    # prefix). Match real torch's behaviour where ``str(torch.int32)`` is
    # ``"torch.int32"`` so those paths keep working.
    def __str__(self) -> str:
        return f"torch.{self._name}"


def _stub_tensor_factory(*args, **kwargs) -> _StubTensor:
    """torch.tensor(...) stub: returns a _StubTensor instance.

    Returning a real object (rather than None) means module-globals like
    xgrammar.matcher._FULL_MASK = torch.tensor(-1, dtype=...) succeed at
    import time. Any subsequent method call on the result (.fill_, .item,
    etc.) raises with a clear pointer via _StubTensor.__getattr__.
    """
    return _StubTensor()


def _false(*args, **kwargs) -> bool:
    return False


def _true(*args, **kwargs) -> bool:
    return True


def _zero(*args, **kwargs) -> int:
    return 0


def _unsupported(qualname: str):
    def _fn(*args, **kwargs):
        raise RuntimeError(
            f"torch.{qualname} is not available: this oMLX build ships a "
            "torch stub for xgrammar's import-time needs only. Install "
            "real torch via pip/Homebrew if you need this code path."
        )

    return _fn


# (canonical, alias) pairs — real torch aliases torch.int to torch.int32,
# torch.long to torch.int64, etc.; preserve those identities so code that
# does ``torch.int is torch.int32`` keeps working.
_DTYPE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("int32", ("int",)),
    ("int16", ("short",)),
    ("int64", ("long",)),
    ("float16", ("half",)),
    ("float32", ("float",)),
    ("float64", ("double",)),
    ("int8", ()),
    ("uint8", ()),
    ("bfloat16", ()),
    ("bool", ()),
)

_TENSOR_ALIASES = (
    "Tensor", "LongTensor", "FloatTensor", "IntTensor", "ByteTensor",
    "DoubleTensor", "HalfTensor", "BoolTensor", "ShortTensor",
)


# Names that xgrammar / tvm_ffi probe via getattr(torch, name) for
# feature-detection — they catch AttributeError and fall back gracefully.
# Logging WARNING for these floods the log on every model load (one per
# name per process) with diagnostics that aren't actually actionable.
# Demote known-probed names to DEBUG; everything else stays WARNING so
# genuinely-missing attributes surface in operator logs.
_KNOWN_PROBE_NAMES: frozenset[str] = frozenset({
    # Integer dtypes added post-torch-2.0 that tvm_ffi.dtypes enumerates
    "uint16", "uint32", "uint64",
    # FP8 / FP4 dtypes (probed by tvm_ffi.dtypes' dtype-mapping table)
    "float8_e4m3fn", "float8_e4m3fnuz",
    "float8_e5m2", "float8_e5m2fnuz",
    "float8_e8m0fnu",
    "float4_e2m1fn_x2",
})


def _make_top_level_torch_getattr() -> "callable":
    """Return a ``__getattr__`` for the stub's top-level torch module.

    Real-torch users who reach an unset attribute would get an
    ``AttributeError``; consumers that probe with ``hasattr`` rely on that.
    But we *also* want a clearly-identifiable message when downstream
    libraries (transformers, accelerate, etc.) reach for a torch surface
    we never stubbed — so this raises ``AttributeError`` whose message
    pinpoints the omlx stub. ``pkgutil.iter_modules(torch.__path__)`` and
    similar discovery paths see the empty ``__path__`` and short-circuit
    before hitting this.
    """

    _missing_attr_logged: set[str] = set()

    def __getattr__(name: str):  # noqa: N807
        # Surface the miss at WARNING level so a future xgrammar release
        # reaching for a new torch attribute is diagnosable from logs
        # before the AttributeError surfaces in a request handler. Rate-
        # limit per name so repeated probes (e.g. hasattr() under a
        # loop) don't flood the journal — once per name per process is
        # enough to identify the gap. Known-probed dtype names log at
        # DEBUG because xgrammar / tvm_ffi catch the AttributeError and
        # the WARNING is pure noise on every model load.
        if name not in _missing_attr_logged:
            _missing_attr_logged.add(name)
            level = logging.DEBUG if name in _KNOWN_PROBE_NAMES else logging.WARNING
            logger.log(
                level,
                "oMLX torch stub missing attribute: torch.%s "
                "(install real torch if this is load-bearing)",
                name,
            )
        # Dunder probes always fall through as AttributeError so pickling,
        # copy.deepcopy, and similar Python machinery work as expected.
        raise AttributeError(
            f"torch.{name!s} is not provided by the oMLX torch stub. "
            "Install real torch via pip/Homebrew if this attribute is "
            "actually needed."
        )

    return __getattr__


def _build_modules() -> dict[str, types.ModuleType]:
    torch = types.ModuleType("torch")
    for alias in _TENSOR_ALIASES:
        setattr(torch, alias, _StubTensor)
    torch.dtype = _StubDtype
    torch.__version__ = "0.0.0+omlx-stub"
    # Pin the stub as the source of truth for the xgrammar version it
    # targets; packaging/build.py imports this constant to stay in sync.
    # (Module-level constant lives at the top of this file.)
    for canonical, aliases in _DTYPE_ALIASES:
        dt = _StubDtype(canonical)
        setattr(torch, canonical, dt)
        for a in aliases:
            setattr(torch, a, dt)
    torch.tensor = _stub_tensor_factory
    torch.full = _unsupported("full")
    torch.zeros = _unsupported("zeros")
    torch.from_dlpack = _unsupported("from_dlpack")

    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = _false
    cuda.device_count = _zero

    cuda_amp_common = types.ModuleType("torch.cuda.amp.common")
    cuda_amp_common.amp_definitely_not_available = _true
    cuda_amp = types.ModuleType("torch.cuda.amp")
    cuda_amp.common = cuda_amp_common
    cuda.amp = cuda_amp

    class _Stream:
        pass

    cuda.Stream = _Stream
    torch.cuda = cuda

    backends_mps = types.ModuleType("torch.backends.mps")
    backends_mps.is_available = _false
    backends_mps.is_built = _false
    backends_cudnn = types.ModuleType("torch.backends.cudnn")
    backends_cudnn.deterministic = False
    backends_cudnn.benchmark = False
    backends = types.ModuleType("torch.backends")
    backends.mps = backends_mps
    backends.cudnn = backends_cudnn
    torch.backends = backends

    version = types.ModuleType("torch.version")
    version.cuda = None
    version.hip = None
    torch.version = version

    nn_functional = types.ModuleType("torch.nn.functional")
    nn_functional.pad = _unsupported("nn.functional.pad")
    nn = types.ModuleType("torch.nn")
    nn.functional = nn_functional
    torch.nn = nn

    utils_dlpack = types.ModuleType("torch.utils.dlpack")
    utils_dlpack.to_dlpack = _unsupported("utils.dlpack.to_dlpack")
    utils = types.ModuleType("torch.utils")
    utils.dlpack = utils_dlpack
    torch.utils = utils

    # Top-level __getattr__ so a future xgrammar that reaches into a
    # torch surface we never stubbed (e.g. ``torch.compile``,
    # ``torch.distributed``) fails with a stub-identifying message rather
    # than a cryptic ``AttributeError: module 'torch' has no attribute…``.
    torch.__getattr__ = _make_top_level_torch_getattr()

    return {
        "torch": torch,
        "torch.cuda": cuda,
        "torch.cuda.amp": cuda_amp,
        "torch.cuda.amp.common": cuda_amp_common,
        "torch.backends": backends,
        "torch.backends.mps": backends_mps,
        "torch.backends.cudnn": backends_cudnn,
        "torch.version": version,
        "torch.nn": nn,
        "torch.nn.functional": nn_functional,
        "torch.utils": utils,
        "torch.utils.dlpack": utils_dlpack,
    }


def install() -> bool:
    """Install the stub into ``sys.modules`` if no real torch is available.

    Returns True if the stub was installed (or had been installed previously),
    False if a real torch was found and left alone.

    Thread-safe — concurrent callers (e.g. multiple FastAPI handlers hitting
    the xgrammar entry points in parallel) serialize on _INSTALL_LOCK.
    """
    global _INSTALLED
    needs_version_check = False
    with _INSTALL_LOCK:
        if _INSTALLED:
            return True

        if "torch" in sys.modules:
            already_stub = getattr(
                sys.modules["torch"], "__version__", ""
            ).endswith("+omlx-stub")
            _INSTALLED = already_stub
            return already_stub

        try:
            if importlib.util.find_spec("torch") is not None:
                # Real torch is on the path — leave it alone, install() is
                # a no-op. Don't mark _INSTALLED so a future sys.modules
                # reset (e.g. in tests) re-evaluates. Crucially, also DO
                # NOT touch ``TVM_FFI_DISABLE_TORCH_C_DLPACK`` — the user
                # has real torch and the tvm-ffi/torch-C-DLPack JIT path
                # may be their preferred fast path.
                return False
        except Exception:
            # find_spec can raise on broken parent packages, partial
            # installs, or weird import hooks. Treat as "no torch" — the
            # stub is the safe fallback.
            pass

        # No real torch — disable tvm_ffi's JIT torch-C-DLPack extension
        # before any tvm-ffi / xgrammar import. Without this,
        # tvm_ffi/_optional_torch_c_dlpack tries to JIT a C extension
        # against our stub at first import, spawns a doomed Python
        # subprocess that fails to ``import torch.utils.cpp_extension``
        # (the stub does not provide it), and surfaces a misleading
        # "Failed to JIT torch c dlpack extension" warning to users on
        # every cold start. The guard inside that module honours this
        # env var and skips the JIT path entirely.
        os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

        for name, mod in _build_modules().items():
            # ``__spec__`` must be a real ModuleSpec (not None) so that
            # ``importlib.util.find_spec`` succeeds when called by
            # transformers and other consumers. ``__version__`` is a
            # clearly-fake value so transformers refuses to take the
            # torch-modeling path.
            mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            mod.__loader__ = None
            if "." not in name:
                mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = mod
        _INSTALLED = True
        needs_version_check = True

    # Fire the version-drift check OUTSIDE the install lock. xgrammar's
    # C++ extension load can be slow on a cold disk; running it under
    # the lock would block every concurrent install() caller behind one
    # cold import. install() is idempotent at this point — _INSTALLED is
    # set and any racing caller short-circuits at the top of the lock.
    if needs_version_check:
        try:
            warn_if_unexpected_versions()
        except Exception:  # pragma: no cover — defensive
            pass
    return True


def warn_if_unexpected_versions() -> None:
    """Log a warning when bundled xgrammar / tvm-ffi versions drift past the
    versions this stub was tested against. Best-effort: silent if the
    imports themselves haven't happened yet, since the stub is installed
    eagerly at startup.
    """
    try:
        import xgrammar  # type: ignore[import-not-found]

        v = getattr(xgrammar, "__version__", None)
        if v and v not in _TARGET_XGRAMMAR_VERSIONS:
            logger.warning(
                "xgrammar %s is not in the torch-stub target set %s; "
                "structured output may fail at runtime. Update the stub "
                "or pin xgrammar back.",
                v,
                _TARGET_XGRAMMAR_VERSIONS,
            )
    except Exception:
        pass
    try:
        import tvm_ffi  # type: ignore[import-not-found]

        v = getattr(tvm_ffi, "__version__", None)
        if v and v not in _TARGET_TVM_FFI_VERSIONS:
            logger.warning(
                "apache-tvm-ffi %s is not in the torch-stub target set %s; "
                "structured output may fail at runtime.",
                v,
                _TARGET_TVM_FFI_VERSIONS,
            )
    except Exception:
        pass
