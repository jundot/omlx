"""Tests for staging bundled DS4 support resources."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

from ds4_support_fixtures import PINNED_DS4_METAL_FILES
from omlx.ds4_support import DS4_REQUIRED_CLI_FLAGS, DS4_SERVER_BINARY

_SCRIPT = Path("scripts/build-ds4-support.sh")


def _write_support_scaffold(source: Path, *, with_binary: bool = False) -> None:
    source.mkdir(parents=True)
    (source / "LICENSE").write_text("DS4 license\n", encoding="utf-8")
    (source / "README.md").write_text("# DS4\n", encoding="utf-8")
    metal = source / "metal"
    metal.mkdir()
    for name in PINNED_DS4_METAL_FILES:
        (metal / name).write_text("// metal\n", encoding="utf-8")
    if with_binary:
        binary = source / DS4_SERVER_BINARY
        flag_lines = "\n".join(f"echo '{flag}'" for flag in DS4_REQUIRED_CLI_FLAGS)
        binary.write_text(
            f'#!/bin/sh\nif [ "$1" = "--help" ]; then\n{flag_lines}\nfi\n',
            encoding="utf-8",
        )
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)


def _commit_support_scaffold(source: Path) -> str:
    subprocess.run(["git", "init"], cwd=source, capture_output=True, check=True)
    subprocess.run(["git", "add", "."], cwd=source, capture_output=True, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=oMLX Tests",
            "-c",
            "user.email=omlx-tests@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        cwd=source,
        capture_output=True,
        check=True,
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        text=True,
    ).strip()


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHON_BIN"] = sys.executable
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def test_build_ds4_support_script_copies_existing_support_tree(tmp_path):
    """--skip-build stages an already-built ds4-server support tree."""
    source = tmp_path / "ds4-source"
    output = tmp_path / "DS4Support"
    _write_support_scaffold(source, with_binary=True)
    commit = _commit_support_scaffold(source)

    result = _run_script(
        "--skip-build",
        "--source",
        str(source),
        "--commit",
        commit,
        "--out",
        str(output),
    )

    assert "Skipping DS4 build" in result.stdout
    assert "DS4 support tree ready" in result.stdout
    assert (output / DS4_SERVER_BINARY).is_file()
    assert os.access(output / DS4_SERVER_BINARY, os.X_OK)
    assert (output / "LICENSE").read_text(encoding="utf-8") == "DS4 license\n"
    assert (output / "metal" / PINNED_DS4_METAL_FILES[0]).is_file()


def test_build_ds4_support_script_builds_ds4_server_with_make(tmp_path):
    """Default mode runs the local ds4 Makefile before staging resources."""
    source = tmp_path / "ds4-source"
    output = tmp_path / "DS4Support"
    _write_support_scaffold(source)
    (source / "Makefile").write_text(
        "ds4-server:\n"
        '\tprintf \'#!/bin/sh\\nif [ "$$1" = "--help" ]; then\\n'
        + "\\n".join(f"echo {flag}" for flag in DS4_REQUIRED_CLI_FLAGS)
        + "\\nfi\\n' > ds4-server\n"
        "\tchmod 755 ds4-server\n"
        "\ttouch built.marker\n",
        encoding="utf-8",
    )
    commit = _commit_support_scaffold(source)

    result = _run_script(
        "--source",
        str(source),
        "--commit",
        commit,
        "--out",
        str(output),
    )

    assert "Building ds4-server" in result.stdout
    assert (source / "built.marker").is_file()
    assert (output / DS4_SERVER_BINARY).is_file()
    assert os.access(output / DS4_SERVER_BINARY, os.X_OK)


def test_build_ds4_support_artifacts_are_gitignored():
    """Generated release binaries are not accidentally committed."""
    ignore_text = Path("packaging/.gitignore").read_text(encoding="utf-8")

    assert "DS4Support/" in ignore_text


def test_vendor_ds4_runtime_artifacts_are_gitignored():
    """Generated vendor DS4 runtime files are not accidentally re-added."""
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "omlx/vendor/ds4/darwin-arm64/ds4-server",
            "omlx/vendor/ds4/darwin-arm64/metal/flash_attn.metal",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    ignored = set(result.stdout.splitlines())
    assert "omlx/vendor/ds4/darwin-arm64/ds4-server" in ignored
    assert "omlx/vendor/ds4/darwin-arm64/metal/flash_attn.metal" in ignored
