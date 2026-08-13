#!/usr/bin/env python3
"""Fail closed when app-bundled native kernels use the wrong CPython ABI.

The macOS app build can use a donor application's embedded Python runtime
while ``PYTHON_BIN`` selects the interpreter that compiles native extensions.
Those interpreters must have the same CPython extension ABI: a successful
build-time import under ``PYTHON_BIN`` does not prove that the bundled app can
load the resulting ``_ext`` module.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_DESCRIPTOR_PROGRAM = r"""
import json
import sys
import sysconfig

print(json.dumps({
    "implementation": sys.implementation.name,
    "cache_tag": sys.implementation.cache_tag,
    "version": list(sys.version_info[:2]),
    "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
}))
"""


class ABIQueryError(RuntimeError):
    """An interpreter could not report its extension ABI."""


def python_abi_descriptor(interpreter: Path, role: str) -> dict[str, object]:
    """Query an interpreter without importing the app package or MLX."""
    try:
        result = subprocess.run(
            [str(interpreter), "-c", _DESCRIPTOR_PROGRAM],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        raise ABIQueryError(
            f"could not run {role} Python {interpreter}: {exc}"
        ) from exc

    if result.returncode:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"exit {result.returncode}"
        )
        raise ABIQueryError(
            f"could not query {role} Python ABI at {interpreter}: {detail}"
        )

    try:
        descriptor = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ABIQueryError(
            f"could not query {role} Python ABI at {interpreter}: invalid JSON output"
        ) from exc

    required = ("implementation", "cache_tag", "version", "extension_suffix")
    if not all(descriptor.get(field) for field in required):
        raise ABIQueryError(
            f"could not query {role} Python ABI at {interpreter}: incomplete descriptor"
        )
    if not isinstance(descriptor["version"], list) or len(descriptor["version"]) != 2:
        raise ABIQueryError(
            f"could not query {role} Python ABI at {interpreter}: invalid version"
        )
    return descriptor


def _format_descriptor(descriptor: dict[str, object]) -> str:
    version = descriptor["version"]
    assert isinstance(version, list)
    return (
        f"{descriptor['implementation']} {version[0]}.{version[1]} "
        f"({descriptor['cache_tag']}, {descriptor['extension_suffix']})"
    )


def validate_build_python(build: dict[str, object], donor: dict[str, object]) -> None:
    """Require every extension-ABI field needed by this package to match."""
    fields = ("implementation", "cache_tag", "version", "extension_suffix")
    if any(build[field] != donor[field] for field in fields):
        raise ValueError(
            "custom kernel build Python ABI does not match the donor app: "
            f"build={_format_descriptor(build)}; donor={_format_descriptor(donor)}. "
            "Set PYTHON_BIN to a Python with the donor app's exact CPython ABI."
        )


def validate_extension_directory(directory: Path, donor: dict[str, object]) -> None:
    """Reject a staged bundle unless every native extension has donor's suffix."""
    if not directory.is_dir():
        raise ValueError(f"custom kernel extension directory is missing: {directory}")

    extensions = sorted(directory.rglob("_ext*.so"))
    if not extensions:
        raise ValueError(f"no native _ext modules found under {directory}")

    extension_suffix = donor["extension_suffix"]
    assert isinstance(extension_suffix, str)
    expected_name = f"_ext{extension_suffix}"
    mismatched = [path for path in extensions if path.name != expected_name]
    if mismatched:
        found = ", ".join(str(path) for path in mismatched)
        raise ValueError(
            "staged custom kernel extension suffix does not match the donor app "
            f"ABI ({expected_name} expected): {found}"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--donor-python", required=True, type=Path, help="bundled donor interpreter"
    )
    parser.add_argument(
        "--build-python", type=Path, help="interpreter compiling native extensions"
    )
    parser.add_argument(
        "--extension-directory",
        type=Path,
        help="staged omlx/custom_kernels directory to validate",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        donor = python_abi_descriptor(args.donor_python, "donor app")
        if args.build_python is not None:
            build = python_abi_descriptor(args.build_python, "build")
            validate_build_python(build, donor)
        if args.extension_directory is not None:
            validate_extension_directory(args.extension_directory, donor)
    except (ABIQueryError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
