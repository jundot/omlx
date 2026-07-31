"""Regression tests for native-kernel cleanup in the macOS app build."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path


BUILD_SCRIPT = Path("apps/omlx-mac/Scripts/build.sh")
SETUP_PY = Path("setup.py")


def _kernel_names_from_setup() -> set[str]:
    return set(
        re.findall(
            r'CMakeExtension\(\s*"omlx\.custom_kernels\.([^."]+)\._ext"',
            SETUP_PY.read_text(),
        )
    )


def _kernel_names_from_build_script(script: str) -> set[str]:
    match = re.search(r"CUSTOM_KERNEL_DIRS=\((.*?)\n\)", script, re.DOTALL)
    assert match is not None
    return set(re.findall(r"custom_kernels/([a-z0-9_]+)\"", match.group(1)))


def _cleanup_function(script: str) -> str:
    match = re.search(
        r"(_clean_custom_kernel_build_artifacts\(\) \{.*?\n\})"
        r"\n\n_sdk_supports_nax\(\)",
        script,
        re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def test_build_cleanup_tracks_every_cmake_extension():
    script = BUILD_SCRIPT.read_text()
    assert _kernel_names_from_build_script(script) == _kernel_names_from_setup()


def test_build_cleanup_removes_stale_abi_outputs(tmp_path):
    script = BUILD_SCRIPT.read_text()
    kernel_names = sorted(_kernel_names_from_setup())
    source_dirs = []

    for name in kernel_names:
        source_dir = tmp_path / "omlx" / "custom_kernels" / name
        source_dir.mkdir(parents=True)
        source_dirs.append(source_dir)
        (source_dir / "_ext.cpython-39-darwin.so").touch()
        (source_dir / "_ext.cpython-311-darwin.so").touch()
        (source_dir / "libomlx_test.dylib").touch()
        (source_dir / "omlx_test_kernels.metallib").touch()
        (source_dir / "keep.py").write_text("# source sentinel\n")

        cmake_dir = (
            tmp_path
            / "build"
            / "temp.macosx-arm64-cpython-311"
            / f"omlx.custom_kernels.{name}._ext"
        )
        cmake_dir.mkdir(parents=True)
        (cmake_dir / "CMakeCache.txt").touch()

        build_lib_dir = (
            tmp_path
            / "build"
            / "lib.macosx-arm64-cpython-311"
            / "omlx"
            / "custom_kernels"
            / name
        )
        build_lib_dir.mkdir(parents=True)
        (build_lib_dir / "_ext.cpython-39-darwin.so").touch()
        (build_lib_dir / "_ext.cpython-311-darwin.so").touch()
        (build_lib_dir / "libomlx_test.dylib").touch()
        (build_lib_dir / "omlx_test_kernels.metallib").touch()

    build_sentinel = tmp_path / "build" / "keep.txt"
    build_sentinel.write_text("build sentinel\n")

    shell = "\n".join(
        [
            f"REPO_ROOT={shlex.quote(str(tmp_path))}",
            "CUSTOM_KERNEL_DIRS=(",
            *(f"  {shlex.quote(str(path))}" for path in source_dirs),
            ")",
            _cleanup_function(script),
            "_clean_custom_kernel_build_artifacts",
        ]
    )
    subprocess.run(["bash", "-c", shell], check=True)

    for source_dir in source_dirs:
        assert (source_dir / "keep.py").exists()
        assert not list(source_dir.glob("*.so"))
        assert not list(source_dir.glob("*.dylib"))
        assert not list(source_dir.glob("*.metallib"))

    assert build_sentinel.exists()
    assert not list(tmp_path.glob("build/**/omlx.custom_kernels.*._ext"))
    assert not list(tmp_path.glob("build/**/omlx/custom_kernels/**/*.so"))
    assert not list(tmp_path.glob("build/**/omlx/custom_kernels/**/*.dylib"))
    assert not list(tmp_path.glob("build/**/omlx/custom_kernels/**/*.metallib"))
