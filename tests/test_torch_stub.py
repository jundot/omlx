# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx._torch_stub.

The stub is load-bearing for the DMG flow: it satisfies xgrammar /
tvm_ffi's import-time torch references without the real ~500 MB torch
wheel. Direct tests here catch the realistic regression where a future
xgrammar / tvm_ffi version starts touching a new torch attribute at
import.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
import sys
import textwrap
import threading
import tomllib
import types
import unittest.mock as mock
from pathlib import Path

import pytest

# Save modules touched by install() so each test starts clean.
_TOUCHED = (
    "torch",
    "torch.cuda",
    "torch.cuda.amp",
    "torch.cuda.amp.common",
    "torch.backends",
    "torch.backends.mps",
    "torch.backends.cudnn",
    "torch.version",
    "torch.nn",
    "torch.nn.functional",
    "torch.utils",
    "torch.utils.dlpack",
)


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    saved = {k: sys.modules[k] for k in _TOUCHED if k in sys.modules}
    # Clear any leftover stub state from a previous test so each starts clean.
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    yield
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    sys.modules.update(saved)


@pytest.fixture
def stub_module():
    """Import a fresh copy of the stub module so its module-level state
    doesn't leak between tests."""
    if "omlx._torch_stub" in sys.modules:
        importlib.reload(sys.modules["omlx._torch_stub"])
        return sys.modules["omlx._torch_stub"]
    import omlx._torch_stub as m
    return m


def test_install_returns_true_and_populates_sys_modules(stub_module):
    # Force "no real torch": remove any existing torch import.
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    with mock.patch(
        "importlib.util.find_spec", side_effect=lambda name: None
    ):
        applied = stub_module.install()
    assert applied is True
    for k in _TOUCHED:
        assert k in sys.modules, f"{k} not installed in sys.modules"
    torch = sys.modules["torch"]
    assert torch.__version__.endswith("+omlx-stub")
    # The dtype set xgrammar/tvm_ffi look up at import time.
    for dt in (
        "int8", "int16", "int32", "int", "int64", "long", "uint8",
        "float16", "half", "float32", "float", "float64", "double",
        "bfloat16", "bool", "short",
    ):
        assert hasattr(torch, dt), f"torch.{dt} missing"
    # Tensor aliases that xgrammar's contrib/hf.py uses in annotations.
    for alias in ("Tensor", "LongTensor", "FloatTensor", "IntTensor"):
        assert hasattr(torch, alias)
    # Submodules tvm_ffi reaches into.
    assert sys.modules["torch.cuda"].is_available() is False
    assert sys.modules["torch.cuda"].device_count() == 0
    assert (
        sys.modules["torch.cuda.amp.common"].amp_definitely_not_available() is True
    )
    assert sys.modules["torch.backends.mps"].is_available() is False
    assert sys.modules["torch.backends.mps"].is_built() is False
    assert sys.modules["torch.version"].cuda is None


def test_install_is_idempotent(stub_module):
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    with mock.patch("importlib.util.find_spec", side_effect=lambda name: None):
        first = stub_module.install()
        second = stub_module.install()
    assert first is True
    # Second call sees the stub already in sys.modules and reports it.
    assert second is True


def test_install_no_op_when_real_torch_present(stub_module):
    # Simulate a previously-imported real torch module.
    real = types.ModuleType("torch")
    real.__version__ = "2.4.0"
    real.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    sys.modules["torch"] = real
    applied = stub_module.install()
    assert applied is False
    # We must not have replaced the real torch.
    assert sys.modules["torch"] is real
    # And we must not have added stub submodules on top of real torch.
    assert "torch.cuda" not in sys.modules


def test_install_no_op_when_torch_findable_via_spec(stub_module):
    # No torch in sys.modules, but importlib can find a spec for it.
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    fake_spec = importlib.machinery.ModuleSpec("torch", loader=None)
    with mock.patch(
        "importlib.util.find_spec",
        side_effect=lambda name: fake_spec if name == "torch" else None,
    ):
        applied = stub_module.install()
    assert applied is False
    assert "torch" not in sys.modules


def test_stub_dtype_works_as_dict_key(stub_module):
    """tvm_ffi.cython.dtype.pxi builds a dict keyed by torch.int8,
    torch.bfloat16, etc. — verify the stub dtypes are hashable and
    distinct."""
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    with mock.patch("importlib.util.find_spec", side_effect=lambda name: None):
        stub_module.install()
    torch = sys.modules["torch"]
    table = {
        torch.int8: 1,
        torch.short: 2,
        torch.int32: 3,
        torch.int64: 4,
        torch.bfloat16: 5,
        torch.bool: 6,
        torch.float32: 7,
    }
    # All distinct keys.
    assert len(table) == 7
    assert table[torch.int32] == 3


def test_stub_tensor_isinstance_check(stub_module):
    """xgrammar/tvm_ffi use isinstance(value, torch.Tensor) to gate
    torch-specific paths. Our values (numpy arrays, mx.array) must
    correctly fail that check."""
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    with mock.patch("importlib.util.find_spec", side_effect=lambda name: None):
        stub_module.install()
    torch = sys.modules["torch"]
    assert isinstance(torch.Tensor(), torch.Tensor)  # stub instance is its own tensor
    # Non-stub values cleanly fail.
    assert not isinstance(42, torch.Tensor)
    assert not isinstance([1, 2, 3], torch.Tensor)
    assert not isinstance("hello", torch.Tensor)
    # torch.dtype is also a class for isinstance checks.
    assert isinstance(torch.int32, torch.dtype)
    assert not isinstance(42, torch.dtype)


def test_unsupported_helpers_raise_runtime_error(stub_module):
    """torch.full / torch.zeros / torch.nn.functional.pad are stubbed to
    raise RuntimeError so a future caller gets a clear error instead of
    a cryptic None-attribute traceback."""
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    with mock.patch("importlib.util.find_spec", side_effect=lambda name: None):
        stub_module.install()
    torch = sys.modules["torch"]
    with pytest.raises(RuntimeError, match="torch.full"):
        torch.full((1,), 0)
    with pytest.raises(RuntimeError, match="torch.zeros"):
        torch.zeros((1,))
    with pytest.raises(RuntimeError, match="nn.functional.pad"):
        torch.nn.functional.pad(None, (0, 1))


def test_torch_tensor_returns_stub_instance_with_loud_method_failure(
    stub_module,
):
    """torch.tensor(...) returns a _StubTensor instance so module-globals
    like ``_FULL_MASK = torch.tensor(-1, dtype=...)`` survive import time.
    Subsequent method calls (e.g. ``.fill_()``) raise a clear RuntimeError
    rather than the prior silent-None path.
    """
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    with mock.patch("importlib.util.find_spec", side_effect=lambda name: None):
        stub_module.install()
    torch = sys.modules["torch"]
    t = torch.tensor(-1, dtype=torch.int32)
    assert isinstance(t, torch.Tensor)
    with pytest.raises(RuntimeError, match="_StubTensor.fill_"):
        t.fill_(0)


def test_dtype_aliases_share_identity(stub_module):
    """Real torch has ``torch.int is torch.int32`` — preserve that identity
    so code doing ``assert x.dtype is torch.int32`` against ``torch.int``
    works identically against the stub."""
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    with mock.patch("importlib.util.find_spec", side_effect=lambda name: None):
        stub_module.install()
    torch = sys.modules["torch"]
    assert torch.int is torch.int32
    assert torch.long is torch.int64
    assert torch.short is torch.int16
    assert torch.half is torch.float16
    assert torch.float is torch.float32
    assert torch.double is torch.float64


def test_dtype_str_returns_torch_prefix(stub_module):
    """tvm_ffi.cpp.dtype.to_cpp_dtype calls ``str(dtype)`` and strips
    a ``torch.`` prefix; our dtypes must serialize that way."""
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    with mock.patch("importlib.util.find_spec", side_effect=lambda name: None):
        stub_module.install()
    torch = sys.modules["torch"]
    assert str(torch.int32) == "torch.int32"
    assert str(torch.bfloat16) == "torch.bfloat16"


def test_install_sets_tvm_ffi_dlpack_env_var(stub_module):
    """install() must set TVM_FFI_DISABLE_TORCH_C_DLPACK so tvm-ffi skips
    the doomed JIT extension build that otherwise spawns a Python
    subprocess and surfaces a misleading warning at every cold start.
    """
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    os.environ.pop("TVM_FFI_DISABLE_TORCH_C_DLPACK", None)
    try:
        with mock.patch(
            "importlib.util.find_spec", side_effect=lambda name: None
        ):
            stub_module.install()
        assert os.environ.get("TVM_FFI_DISABLE_TORCH_C_DLPACK") == "1"
    finally:
        os.environ.pop("TVM_FFI_DISABLE_TORCH_C_DLPACK", None)


def test_install_does_not_touch_env_var_when_real_torch_present(stub_module):
    """The opposite of the previous test: when real torch is detected via
    find_spec, install() must NOT mutate TVM_FFI_DISABLE_TORCH_C_DLPACK.
    A user with real torch installed may want the tvm-ffi/torch-C-DLPack
    fast path; the stub should not silently disable it.
    """
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    os.environ.pop("TVM_FFI_DISABLE_TORCH_C_DLPACK", None)
    try:
        fake_spec = importlib.util.spec_from_loader("torch", loader=None)
        with mock.patch(
            "importlib.util.find_spec",
            side_effect=lambda name: fake_spec if name == "torch" else None,
        ):
            result = stub_module.install()
        assert result is False
        assert "TVM_FFI_DISABLE_TORCH_C_DLPACK" not in os.environ, (
            "real-torch path must leave the env var alone"
        )
    finally:
        os.environ.pop("TVM_FFI_DISABLE_TORCH_C_DLPACK", None)


def test_missing_top_level_attribute_raises_attributeerror_and_logs(
    stub_module, caplog
):
    """``torch.<unknown>`` must raise ``AttributeError`` (so ``hasattr``
    consumers behave correctly) AND log a one-shot WARNING that names
    the missing attribute. The log is the operator-facing diagnostic
    when a future xgrammar / tvm-ffi release reaches for a torch
    surface the stub doesn't cover; without it, the AttributeError
    surfaces only if the caller logs it themselves.
    """
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    with mock.patch("importlib.util.find_spec", side_effect=lambda name: None):
        stub_module.install()
    torch = sys.modules["torch"]
    with caplog.at_level("WARNING", logger="omlx._torch_stub"):
        with pytest.raises(AttributeError, match="torch.compile"):
            torch.compile  # noqa: B018
    assert any(
        "missing attribute: torch.compile" in rec.message
        for rec in caplog.records
    ), caplog.records
    # ``hasattr`` must continue to return False (i.e. the AttributeError
    # path is reachable) — regression for replacing the raise with a
    # log-and-return.
    assert not hasattr(torch, "another_missing_attr")


def test_known_probe_names_log_at_debug_not_warning(stub_module, caplog):
    """xgrammar / tvm_ffi probe a fixed set of dtype names via
    ``getattr(torch, name)`` for feature detection. They catch the
    AttributeError and fall back, so a per-probe WARNING is pure noise.
    Known-probed names log at DEBUG instead.

    Regression for #1453 review feedback (fry69): 9 WARNING entries per
    model load flagged as actionable when they aren't.
    """
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    with mock.patch("importlib.util.find_spec", side_effect=lambda name: None):
        stub_module.install()
    torch = sys.modules["torch"]

    # Probe one known dtype + one genuinely-missing attribute. Capture at
    # DEBUG so both log calls land in caplog.records and we can compare
    # their levels.
    with caplog.at_level("DEBUG", logger="omlx._torch_stub"):
        with pytest.raises(AttributeError):
            torch.float8_e4m3fn  # noqa: B018
        with pytest.raises(AttributeError):
            torch.totally_unknown_attr  # noqa: B018

    dtype_records = [
        rec for rec in caplog.records
        if "torch.float8_e4m3fn" in rec.message
    ]
    unknown_records = [
        rec for rec in caplog.records
        if "torch.totally_unknown_attr" in rec.message
    ]
    assert dtype_records, "known-probe name should still log at DEBUG"
    assert unknown_records, "unknown name should still log"
    assert all(rec.levelname == "DEBUG" for rec in dtype_records), (
        f"known probe must log at DEBUG, got {[r.levelname for r in dtype_records]}"
    )
    assert all(rec.levelname == "WARNING" for rec in unknown_records), (
        f"unknown attr must log at WARNING, got {[r.levelname for r in unknown_records]}"
    )


def test_stub_modules_have_real_spec_and_loader(stub_module):
    """Every stub module in sys.modules must have a real ``__spec__``
    (a ``ModuleSpec`` instance, not ``None``) so ``importlib.util.
    find_spec`` succeeds for downstream consumers — transformers /
    accelerate / huggingface_hub all probe torch via find_spec at
    import time, and ``None`` here trips their fallback paths into
    incorrect behavior.
    """
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    with mock.patch("importlib.util.find_spec", side_effect=lambda name: None):
        stub_module.install()
    for name in (
        "torch",
        "torch.cuda",
        "torch.cuda.amp",
        "torch.cuda.amp.common",
        "torch.backends",
        "torch.backends.mps",
        "torch.backends.cudnn",
        "torch.version",
        "torch.nn",
        "torch.nn.functional",
        "torch.utils",
        "torch.utils.dlpack",
    ):
        mod = sys.modules[name]
        assert mod.__spec__ is not None, f"{name} missing __spec__"
        assert isinstance(mod.__spec__, importlib.machinery.ModuleSpec), (
            f"{name}.__spec__ wrong type: {type(mod.__spec__)}"
        )
        assert mod.__spec__.name == name


def test_utils_dlpack_to_dlpack_raises(stub_module):
    """``torch.utils.dlpack.to_dlpack`` is a separately-exposed helper
    (not in ``torch.nn.functional``). If a future tvm-ffi reaches for
    it under the stub it must raise loudly rather than silently return
    None — calls into this path mean the caller assumed real torch and
    will produce wrong results downstream.
    """
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    with mock.patch("importlib.util.find_spec", side_effect=lambda name: None):
        stub_module.install()
    import torch  # type: ignore

    with pytest.raises(RuntimeError, match="utils.dlpack.to_dlpack"):
        torch.utils.dlpack.to_dlpack(object())


def test_install_is_thread_safe(stub_module):
    """Concurrent install() calls must serialize and produce a single
    consistent stub. Regression for a race where two threads both passed
    the ``"torch" in sys.modules`` check, both built modules, and
    overwrote each other in sys.modules — leaving threads with stale
    references to the loser's module objects.
    """
    for k in _TOUCHED:
        sys.modules.pop(k, None)
    results: list[bool] = []
    barrier = threading.Barrier(8)
    errors: list[Exception] = []

    def worker():
        try:
            barrier.wait(timeout=2.0)
            with mock.patch(
                "importlib.util.find_spec", side_effect=lambda name: None
            ):
                results.append(stub_module.install())
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
    assert not errors, errors
    assert len(results) == 8
    assert all(r is True for r in results)
    # All threads see the same single torch module instance.
    torch = sys.modules["torch"]
    assert torch.__version__.endswith("+omlx-stub")


@pytest.mark.skipif(
    not (importlib.util.find_spec("xgrammar") and importlib.util.find_spec("tvm_ffi")),
    reason="xgrammar / tvm_ffi not installed",
)
def test_xgrammar_imports_against_stub_only(stub_module, tmp_path):
    """Realistic regression: spawn a subprocess that blocks real torch and
    asserts ``import xgrammar`` and the modules oMLX touches still load
    against the stub. This is the test that gates xgrammar / tvm-ffi
    version bumps — if a new release reaches for a torch attribute the
    stub doesn't cover, this fails loudly at the import step.
    """
    script = tmp_path / "probe.py"
    script.write_text(textwrap.dedent("""
        import sys

        # Block real torch end-to-end without touching sys.path (which
        # would also strip xgrammar in the common pip layout where both
        # live in the same site-packages). A meta-path finder that
        # returns None just delegates to the next finder; raising
        # ImportError aborts the import before PathFinder runs.
        for k in list(sys.modules):
            if k == "torch" or k.startswith("torch."):
                del sys.modules[k]

        import importlib.abc

        class _BlockTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "torch" or fullname.startswith("torch."):
                    raise ImportError(
                        f"{fullname} blocked by test probe to force "
                        "the stub-only path"
                    )
                return None

        sys.meta_path.insert(0, _BlockTorch())

        # install()'s own `importlib.util.find_spec('torch')` check
        # also needs to see no torch.
        import importlib.util
        _orig_find_spec = importlib.util.find_spec
        def _no_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                return None
            return _orig_find_spec(name, *args, **kwargs)
        importlib.util.find_spec = _no_torch

        from omlx._torch_stub import install
        assert install() is True, (
            "stub install returned False — real torch was reachable "
            "despite meta-path / find_spec blocking"
        )

        import xgrammar
        from xgrammar import contrib  # noqa: F401
        from xgrammar.kernels.apply_token_bitmask_mlx import (  # noqa: F401
            apply_token_bitmask_mlx,
        )
        print("OK")
    """))
    env = dict(os.environ)
    env.pop("TVM_FFI_DISABLE_TORCH_C_DLPACK", None)
    out = subprocess.check_output(
        [sys.executable, str(script)],
        stderr=subprocess.STDOUT,
        env=env,
        timeout=30,
    )
    assert b"OK" in out, out


def _fake_metadata_version(xgrammar_v, tvm_ffi_v):
    versions = {"xgrammar": xgrammar_v, "apache-tvm-ffi": tvm_ffi_v}

    def fake(dist):
        v = versions[dist]
        if v is None:
            raise importlib.metadata.PackageNotFoundError(dist)
        return v

    return fake


def test_warn_fires_on_version_drift(stub_module, caplog):
    fake = _fake_metadata_version("9.9.9", "8.8.8")
    with (
        mock.patch("importlib.metadata.version", side_effect=fake),
        caplog.at_level("WARNING", logger="omlx._torch_stub"),
    ):
        stub_module.warn_if_unexpected_versions()
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("xgrammar 9.9.9" in m for m in messages), messages
    assert any("apache-tvm-ffi 8.8.8" in m for m in messages), messages


def test_warn_silent_when_versions_match_targets(stub_module, caplog):
    fake = _fake_metadata_version(
        stub_module._TARGET_XGRAMMAR_VERSIONS[0],
        stub_module._TARGET_TVM_FFI_VERSIONS[0],
    )
    with (
        mock.patch("importlib.metadata.version", side_effect=fake),
        caplog.at_level("WARNING", logger="omlx._torch_stub"),
    ):
        stub_module.warn_if_unexpected_versions()
    assert not caplog.records, [rec.getMessage() for rec in caplog.records]


def test_warn_silent_when_distributions_missing(stub_module, caplog):
    fake = _fake_metadata_version(None, None)
    with (
        mock.patch("importlib.metadata.version", side_effect=fake),
        caplog.at_level("WARNING", logger="omlx._torch_stub"),
    ):
        stub_module.warn_if_unexpected_versions()
    assert not caplog.records, [rec.getMessage() for rec in caplog.records]


def _pyproject_dev_pins(package):
    """Collect ``package==X`` pins from both dev dependency lists in
    pyproject.toml (the [dev] extra and the PEP 735 dependency group)."""
    root = Path(__file__).resolve().parents[1]
    with open(root / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    specs = list(data["project"]["optional-dependencies"]["dev"])
    specs += [s for s in data["dependency-groups"]["dev"] if isinstance(s, str)]
    prefix = f"{package}=="
    return {s[len(prefix) :] for s in specs if s.startswith(prefix)}


def test_pyproject_dev_pins_match_stub_targets(stub_module):
    """Dependabot bumps the pyproject dev pins but cannot touch this stub,
    and packaging/build.py ships _TARGET_*_VERSIONS[0] in the DMG. Without
    this check a bare pyproject bump silently makes dev/CI test a version
    the bundle does not ship. Bump _TARGET_XGRAMMAR_VERSIONS /
    _TARGET_TVM_FFI_VERSIONS in omlx/_torch_stub.py alongside the pin.
    """
    for package, targets in (
        ("xgrammar", stub_module._TARGET_XGRAMMAR_VERSIONS),
        ("apache-tvm-ffi", stub_module._TARGET_TVM_FFI_VERSIONS),
    ):
        pins = _pyproject_dev_pins(package)
        assert len(pins) == 1, (
            f"{package}: expected one identical pin across both pyproject "
            f"dev lists, got {sorted(pins) or 'none'}"
        )
        assert pins == {targets[0]}, (
            f"{package}: pyproject dev pin {sorted(pins)} != stub target "
            f"{targets[0]} — update _TARGET_*_VERSIONS in omlx/_torch_stub.py"
        )
