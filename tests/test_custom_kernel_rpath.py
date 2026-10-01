# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the native custom kernels' build-time rpath entries.

Every custom kernel extension links against the mlx wheel's `libmlx.dylib`
through `@rpath/libmlx.dylib`. A single `LC_RPATH` pointing at the link-time
absolute path into the build environment is dead once the extension is
installed, so the installed wheel must also carry a relative rpath that
resolves from the packaged layout:

    site-packages/omlx/custom_kernels/<pkg>/_ext...so
      -> ../../..            = site-packages
      -> mlx/lib             = site-packages/mlx/lib

Issue #2233 added that entry to every kernel except `bonsai`, which then
failed to import with `dlopen: Library not loaded: @rpath/libmlx.dylib` on
venv installs and silently reported `available: false` through
`GET /api/status` (issue #2822).

CMake is not required to check this: the assertion is text-level, so it runs
in CI on any platform and does not need a `--with-custom-kernel` build.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from omlx.custom_kernels import NATIVE_KERNEL_PACKAGES

CUSTOM_KERNELS_DIR = Path(__file__).resolve().parents[1] / "omlx" / "custom_kernels"

# The relative rpath every installed extension needs to find the mlx wheel.
MLX_LIB_RPATH = "@loader_path/../../../mlx/lib"


def _cmake_source(package: str) -> str:
    path = CUSTOM_KERNELS_DIR / package / "csrc" / "CMakeLists.txt"
    if not path.is_file():
        pytest.skip(f"{package} ships no CMake project")
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("package", NATIVE_KERNEL_PACKAGES)
def test_kernel_links_mlx_wheel_relative_rpath(package: str) -> None:
    """Each kernel extension must keep the relative libmlx rpath (#2822)."""
    assert MLX_LIB_RPATH in _cmake_source(package), (
        f"{package} does not link -Wl,-rpath,{MLX_LIB_RPATH}; the built "
        f"extension would fail dlopen with 'Library not loaded: "
        f"@rpath/libmlx.dylib' once installed outside the build environment"
    )


@pytest.mark.parametrize("package", NATIVE_KERNEL_PACKAGES)
def test_kernel_rpath_is_scoped_to_shared_builds(package: str) -> None:
    """The rpath block must stay inside the BUILD_SHARED_LIBS guard.

    Static builds have no install-time loader indirection to fix, and the
    flags are only valid where a shared extension is actually produced.
    """
    source = _cmake_source(package)
    match = re.search(
        r"if\(BUILD_SHARED_LIBS\)(.*?)endif\(\)",
        source,
        re.DOTALL,
    )
    assert match, f"{package} no longer guards its link options on BUILD_SHARED_LIBS"
    assert "target_link_options" in match.group(1)
