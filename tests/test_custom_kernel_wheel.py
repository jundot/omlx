# SPDX-License-Identifier: Apache-2.0
"""Tests for app custom-kernel wheel preparation."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "packaging" / "custom_kernel_wheel.py"
SPEC = importlib.util.spec_from_file_location("custom_kernel_wheel", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
custom_kernel_wheel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(custom_kernel_wheel)


ARTIFACTS = {
    "bonsai": (
        "_ext.cpython-311-darwin.so",
        "libomlx_bonsai_kernel_ops.dylib",
        "omlx_bonsai_kernels.metallib",
    ),
    "glm_moe_dsa": (
        "_ext.cpython-311-darwin.so",
        "libomlx_glm_kernel_ops.dylib",
        "omlx_glm_kernels.metallib",
    ),
    "minimax_m3": (
        "_ext.cpython-311-darwin.so",
        "libomlx_minimax_m3_kernel_ops.dylib",
        "omlx_minimax_m3_kernels.metallib",
    ),
    "qwen35_prefill": (
        "_ext.cpython-311-darwin.so",
        "libomlx_qwen35_prefill_kernel_ops.dylib",
        "omlx_qwen35_prefill_kernels.metallib",
    ),
}


def _write_wheel(
    path: Path,
    *,
    include_nax: bool = False,
    omit: str | None = None,
    extra: str | None = None,
) -> None:
    with zipfile.ZipFile(path, "w") as wheel:
        for package, names in ARTIFACTS.items():
            for name in names:
                member = f"omlx/custom_kernels/{package}/{name}"
                if member != omit:
                    wheel.writestr(member, b"native")
        if include_nax:
            wheel.writestr(
                "omlx/custom_kernels/qwen35_prefill/"
                "omlx_qwen35_prefill_kernels_nax.metallib",
                b"nax",
            )
        if extra:
            wheel.writestr(extra, b"unexpected")


def test_write_build_requirements_uses_pyproject_build_system(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    output = tmp_path / "requirements.txt"
    pyproject.write_text(
        '[build-system]\nrequires = ["setuptools>=61", "nanobind==2.13.0"]\n'
    )

    custom_kernel_wheel.write_build_requirements(pyproject, output)

    assert output.read_text() == "setuptools>=61\nnanobind==2.13.0\n"


def test_extract_native_artifacts_accepts_complete_wheel(tmp_path):
    wheel = tmp_path / "omlx.whl"
    destination = tmp_path / "payload"
    _write_wheel(wheel, include_nax=True)

    extracted = custom_kernel_wheel.extract_native_artifacts(
        wheel, destination, require_nax=True
    )

    assert len(extracted) == 13
    assert all(path.is_file() for path in extracted)
    assert (
        destination
        / "omlx/custom_kernels/qwen35_prefill/"
        "omlx_qwen35_prefill_kernels_nax.metallib"
    ).is_file()


def test_extract_native_artifacts_allows_no_nax_on_older_sdk(tmp_path):
    wheel = tmp_path / "omlx.whl"
    _write_wheel(wheel)

    extracted = custom_kernel_wheel.extract_native_artifacts(
        wheel, tmp_path / "payload", require_nax=False
    )

    assert len(extracted) == 12


def test_extract_native_artifacts_requires_nax_when_requested(tmp_path):
    wheel = tmp_path / "omlx.whl"
    _write_wheel(wheel)

    with pytest.raises(ValueError, match="NAX|nax"):
        custom_kernel_wheel.extract_native_artifacts(
            wheel, tmp_path / "payload", require_nax=True
        )


def test_extract_native_artifacts_rejects_missing_required_file(tmp_path):
    wheel = tmp_path / "omlx.whl"
    _write_wheel(
        wheel,
        omit="omlx/custom_kernels/minimax_m3/omlx_minimax_m3_kernels.metallib",
    )

    with pytest.raises(ValueError, match="minimax_m3"):
        custom_kernel_wheel.extract_native_artifacts(
            wheel, tmp_path / "payload", require_nax=False
        )


def test_extract_native_artifacts_rejects_unexpected_native_file(tmp_path):
    wheel = tmp_path / "omlx.whl"
    _write_wheel(
        wheel,
        extra="omlx/custom_kernels/qwen35_prefill/stale.metallib",
    )

    with pytest.raises(ValueError, match="unexpected native artifacts"):
        custom_kernel_wheel.extract_native_artifacts(
            wheel, tmp_path / "payload", require_nax=False
        )
