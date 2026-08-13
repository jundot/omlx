"""Regression coverage for the app bundle's custom-kernel CPython ABI guard."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CHECKER = (
    Path(__file__).resolve().parents[1]
    / "apps/omlx-mac/Scripts/check_custom_kernel_python_abi.py"
)
BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "apps/omlx-mac/Scripts/build.sh"


def _write_fake_python(path: Path, *, descriptor: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if descriptor is None:
        body = "#!/bin/sh\necho 'query intentionally failed' >&2\nexit 42\n"
    else:
        body = f"#!/bin/sh\nprintf '%s\\n' '{descriptor}'\n"
    path.write_text(body)
    path.chmod(0o755)


def _run_checker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": ""},
    )


def test_matching_build_and_donor_python_abis_pass(tmp_path):
    descriptor = (
        '{"implementation":"cpython","cache_tag":"cpython-311",'
        '"version":[3,11],"extension_suffix":".cpython-311-darwin.so"}'
    )
    build = tmp_path / "build-python"
    donor = tmp_path / "donor-python"
    _write_fake_python(build, descriptor=descriptor)
    _write_fake_python(donor, descriptor=descriptor)

    result = _run_checker("--build-python", str(build), "--donor-python", str(donor))

    assert result.returncode == 0, result.stderr


def test_mismatched_build_and_donor_python_abis_fail_closed(tmp_path):
    build = tmp_path / "build-python"
    donor = tmp_path / "donor-python"
    _write_fake_python(
        build,
        descriptor=(
            '{"implementation":"cpython","cache_tag":"cpython-313",'
            '"version":[3,13],"extension_suffix":".cpython-313-darwin.so"}'
        ),
    )
    _write_fake_python(
        donor,
        descriptor=(
            '{"implementation":"cpython","cache_tag":"cpython-311",'
            '"version":[3,11],"extension_suffix":".cpython-311-darwin.so"}'
        ),
    )

    result = _run_checker("--build-python", str(build), "--donor-python", str(donor))

    assert result.returncode == 1
    assert "does not match the donor app" in result.stderr
    assert "cpython-313" in result.stderr
    assert "cpython-311" in result.stderr


def test_donor_abi_query_failure_fails_closed(tmp_path):
    build = tmp_path / "build-python"
    donor = tmp_path / "donor-python"
    _write_fake_python(
        build,
        descriptor=(
            '{"implementation":"cpython","cache_tag":"cpython-311",'
            '"version":[3,11],"extension_suffix":".cpython-311-darwin.so"}'
        ),
    )
    _write_fake_python(donor)

    result = _run_checker("--build-python", str(build), "--donor-python", str(donor))

    assert result.returncode == 1
    assert "could not query donor app Python ABI" in result.stderr


def test_staged_extension_suffix_mismatch_fails_closed(tmp_path):
    donor = tmp_path / "donor-python"
    _write_fake_python(
        donor,
        descriptor=(
            '{"implementation":"cpython","cache_tag":"cpython-311",'
            '"version":[3,11],"extension_suffix":".cpython-311-darwin.so"}'
        ),
    )
    extension = tmp_path / "custom_kernels/glm_moe_dsa/_ext.cpython-313-darwin.so"
    extension.parent.mkdir(parents=True)
    extension.touch()

    result = _run_checker(
        "--donor-python",
        str(donor),
        "--extension-directory",
        str(extension.parents[1]),
    )

    assert result.returncode == 1
    assert "extension suffix does not match the donor app ABI" in result.stderr


def test_app_build_wires_the_guard_before_build_and_after_staging():
    script = BUILD_SCRIPT.read_text()

    build_function = script.index("_build_custom_kernels()")
    prebuild_guard = script.index(
        "    _validate_custom_kernel_python_abi", build_function
    )
    build = script.index("    _check_custom_kernel_nanobind", build_function)
    copy = script.index('"$REPO_ROOT/omlx/" "$RESOURCES_DIR/omlx/"')
    post_staging_guard = script.index(
        "    _validate_packaged_custom_kernel_extensions", copy
    )

    assert prebuild_guard < build
    assert post_staging_guard > copy
