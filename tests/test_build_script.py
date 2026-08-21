# SPDX-License-Identifier: Apache-2.0
"""Static contracts for the macOS app build script."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "apps" / "omlx-mac" / "Scripts" / "build.sh"


def test_custom_kernel_build_uses_donor_python_and_pep517_wheel():
    script = BUILD_SCRIPT.read_text()

    assert 'local donor_python="$DONOR_LAYERS/cpython-3.11/bin/python3"' in script
    assert '"$build_python" -m pip wheel' in script
    assert "--no-build-isolation" in script
    assert "setup.py build_ext" not in script


def test_packaging_driver_rejects_unsupported_python_versions():
    script = BUILD_SCRIPT.read_text()

    assert "_python_is_supported" in script
    assert "(3, 11) <= sys.version_info[:2] < (3, 14)" in script
    assert '"$REPO_ROOT/.venv/bin/python"' in script


def test_custom_kernel_build_reads_declared_build_requirements():
    script = BUILD_SCRIPT.read_text()

    assert 'custom_kernel_wheel.py" requirements' in script
    assert '"$build_python" -m pip install' in script
    assert "_check_custom_kernel_nanobind" not in script


def test_source_native_artifacts_are_always_excluded_then_overlaid():
    script = BUILD_SCRIPT.read_text()

    assert "--exclude='custom_kernels/*/*.so'" in script
    assert "--exclude='custom_kernels/*/*.dylib'" in script
    assert "--exclude='custom_kernels/*/*.metallib'" in script
    assert 'rsync -a "$CUSTOM_KERNEL_PAYLOAD/omlx/" "$RESOURCES_DIR/omlx/"' in script
