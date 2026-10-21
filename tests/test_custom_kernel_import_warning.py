"""A built _ext that fails to load must warn; a missing one stays silent.

Regression test for issue #2233: custom kernel binaries with an unresolved
@rpath/libmlx.dylib dependency silently fell back to the slow path. The shared
loader runs against a scratch package so the import runs against a controlled
``_ext`` state.
"""

import importlib
import importlib.util
import logging
import re
import subprocess
import sys

import pytest

import omlx.custom_kernels as custom_kernels
from omlx.custom_kernels import _loader


def _load_scratch_extension(tmp_path, name, with_broken_ext):
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    if with_broken_ext:
        (pkg / "_ext.cpython-311-darwin.so").write_bytes(b"\x00" * 64)
    sys.path.insert(0, str(tmp_path))
    try:
        return _loader.load_extension(name, pkg)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(f"{name}._ext", None)
        sys.modules.pop(name, None)


def test_broken_native_extension_warns(tmp_path, caplog):
    name = "fakepkg_broken"
    with caplog.at_level(logging.WARNING):
        ext, error = _load_scratch_extension(tmp_path, name, with_broken_ext=True)
    assert ext is None
    assert error is not None
    warnings = [r for r in caplog.records if r.name == _loader.logger.name]
    assert len(warnings) == 1
    assert "failed to load" in warnings[0].getMessage()
    assert name in warnings[0].getMessage()


def test_missing_native_extension_stays_silent(tmp_path, caplog):
    name = "fakepkg_missing"
    with caplog.at_level(logging.WARNING):
        ext, error = _load_scratch_extension(tmp_path, name, with_broken_ext=False)
    assert ext is None
    assert error is not None
    assert [r for r in caplog.records if r.name == _loader.logger.name] == []


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS binaries")
@pytest.mark.parametrize("kernel", custom_kernels.NATIVE_KERNEL_PACKAGES)
def test_kernel_links_relative_mlx_rpath(kernel):
    spec = importlib.util.find_spec(f"omlx.custom_kernels.{kernel}._ext")
    if spec is None:
        pytest.skip("native extension not built")
    output = subprocess.check_output(["otool", "-l", spec.origin], text=True)
    rpaths = re.findall(r"path (.+) \(offset \d+\)", output)
    assert "@loader_path/../../../mlx/lib" in rpaths
