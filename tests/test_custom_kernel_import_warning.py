"""A built _ext that fails to load must warn; a missing one stays silent.

Regression test for issue #2233: custom kernel binaries with an unresolved
@rpath/libmlx.dylib dependency silently fell back to the slow path. Each
kernel's ``fast.py`` is imported from a scratch package so the module-level
import block runs against a controlled ``_ext`` state.
"""

import importlib
import logging
import shutil
import sys
from pathlib import Path

import pytest

import omlx.custom_kernels as custom_kernels

KERNELS = ["glm_moe_dsa", "minimax_m3", "qwen35_prefill"]


def _import_fast_copy(tmp_path, kernel, name, with_broken_ext):
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    src = Path(custom_kernels.__file__).parent / kernel / "fast.py"
    shutil.copy(src, pkg / "fast.py")
    if with_broken_ext:
        (pkg / "_ext.cpython-311-darwin.so").write_bytes(b"\x00" * 64)
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module(f"{name}.fast")
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(f"{name}.fast", None)
        sys.modules.pop(name, None)


@pytest.mark.parametrize("kernel", KERNELS)
def test_broken_native_extension_warns(tmp_path, caplog, kernel):
    name = f"fakepkg_{kernel}_broken"
    with caplog.at_level(logging.WARNING):
        fast = _import_fast_copy(tmp_path, kernel, name, with_broken_ext=True)
    assert not fast.is_native_available()
    assert fast.import_error() is not None
    warnings = [r for r in caplog.records if r.name == f"{name}.fast"]
    assert len(warnings) == 1
    assert "failed to load" in warnings[0].getMessage()


@pytest.mark.parametrize("kernel", KERNELS)
def test_missing_native_extension_stays_silent(tmp_path, caplog, kernel):
    name = f"fakepkg_{kernel}_missing"
    with caplog.at_level(logging.WARNING):
        fast = _import_fast_copy(tmp_path, kernel, name, with_broken_ext=False)
    assert not fast.is_native_available()
    assert [r for r in caplog.records if r.name == f"{name}.fast"] == []
