#!/usr/bin/env python3
"""Helpers for building and unpacking the app's custom-kernel wheel."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tomllib
import zipfile
from pathlib import Path, PurePosixPath


NATIVE_SUFFIXES = (".so", ".dylib", ".metallib")
REQUIRED_ARTIFACTS = {
    "bonsai": (
        re.compile(r"_ext\.[^.]+(?:\.[^.]+)*\.so$"),
        "libomlx_bonsai_kernel_ops.dylib",
        "omlx_bonsai_kernels.metallib",
    ),
    "glm_moe_dsa": (
        re.compile(r"_ext\.[^.]+(?:\.[^.]+)*\.so$"),
        "libomlx_glm_kernel_ops.dylib",
        "omlx_glm_kernels.metallib",
    ),
    "minimax_m3": (
        re.compile(r"_ext\.[^.]+(?:\.[^.]+)*\.so$"),
        "libomlx_minimax_m3_kernel_ops.dylib",
        "omlx_minimax_m3_kernels.metallib",
    ),
    "qwen35_prefill": (
        re.compile(r"_ext\.[^.]+(?:\.[^.]+)*\.so$"),
        "libomlx_qwen35_prefill_kernel_ops.dylib",
        "omlx_qwen35_prefill_kernels.metallib",
    ),
}
NAX_ARTIFACT = "omlx_qwen35_prefill_kernels_nax.metallib"


def write_build_requirements(pyproject: Path, output: Path) -> None:
    with pyproject.open("rb") as stream:
        data = tomllib.load(stream)
    requirements = data.get("build-system", {}).get("requires")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError(f"{pyproject} has no [build-system].requires entries")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{requirement}\n" for requirement in requirements))


def _matches(expected: str | re.Pattern[str], name: str) -> bool:
    if isinstance(expected, str):
        return name == expected
    return expected.fullmatch(name) is not None


def extract_native_artifacts(
    wheel: Path,
    destination: Path,
    *,
    require_nax: bool,
) -> list[Path]:
    prefix = PurePosixPath("omlx/custom_kernels")
    found: dict[str, list[tuple[str, str]]] = {
        package: [] for package in REQUIRED_ARTIFACTS
    }

    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or len(path.parts) != 4:
                continue
            if PurePosixPath(*path.parts[:2]) != prefix:
                continue
            package, filename = path.parts[2], path.parts[3]
            if not filename.endswith(NATIVE_SUFFIXES):
                continue
            if package not in REQUIRED_ARTIFACTS:
                raise ValueError(f"unexpected custom-kernel package in wheel: {path}")
            found[package].append((filename, info.filename))

        selected: list[tuple[str, str]] = []
        for package, expected_items in REQUIRED_ARTIFACTS.items():
            package_files = found[package]
            for expected in expected_items:
                matches = [item for item in package_files if _matches(expected, item[0])]
                label = expected if isinstance(expected, str) else "_ext.*.so"
                if len(matches) != 1:
                    raise ValueError(
                        f"expected one {package}/{label} in {wheel}, found {len(matches)}"
                    )
                selected.append(matches[0])

            expected_names = {
                item[0]
                for item in package_files
                if any(_matches(expected, item[0]) for expected in expected_items)
            }
            allowed_names = set(expected_names)
            if package == "qwen35_prefill":
                nax_matches = [item for item in package_files if item[0] == NAX_ARTIFACT]
                if require_nax and len(nax_matches) != 1:
                    raise ValueError(
                        f"expected one {package}/{NAX_ARTIFACT} in {wheel}, "
                        f"found {len(nax_matches)}"
                    )
                if len(nax_matches) > 1:
                    raise ValueError(
                        f"expected at most one {package}/{NAX_ARTIFACT} in {wheel}, "
                        f"found {len(nax_matches)}"
                    )
                if nax_matches:
                    selected.append(nax_matches[0])
                    allowed_names.add(NAX_ARTIFACT)

            unexpected = sorted(name for name, _ in package_files if name not in allowed_names)
            if unexpected:
                raise ValueError(
                    f"unexpected native artifacts in {package}: {', '.join(unexpected)}"
                )

        if destination.exists():
            shutil.rmtree(destination)
        extracted: list[Path] = []
        for _, member in selected:
            target = destination / member
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            if target.stat().st_size == 0:
                raise ValueError(f"empty native artifact in wheel: {member}")
            extracted.append(target)
    return extracted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    requirements = subparsers.add_parser("requirements")
    requirements.add_argument("pyproject", type=Path)
    requirements.add_argument("output", type=Path)

    extract = subparsers.add_parser("extract")
    extract.add_argument("wheel", type=Path)
    extract.add_argument("destination", type=Path)
    extract.add_argument("--require-nax", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "requirements":
            write_build_requirements(args.pyproject, args.output)
        else:
            extracted = extract_native_artifacts(
                args.wheel,
                args.destination,
                require_nax=args.require_nax,
            )
            for path in extracted:
                print(path)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"custom-kernel wheel error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
