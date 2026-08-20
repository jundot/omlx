# SPDX-License-Identifier: Apache-2.0
"""Support-file checks for the managed DS4/GGUF backend.

The DS4 process backend is launched from a user support directory rather than
from package resources.  This module owns the small, testable pieces that
validate, copy, or lazily build those support files before later process-launch
code consumes those paths.
"""

from __future__ import annotations

import os
import platform
import json
import shutil
import shlex
import subprocess
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .settings import DEFAULT_BASE_PATH, DS4Settings

DS4_SERVER_BINARY = "ds4-server"
BUNDLED_DS4_SUPPORT_ENV = "OMLX_BUNDLED_DS4_SUPPORT_DIR"
BUNDLED_DS4_SUPPORT_DIR_NAME = "DS4Support"
VENDORED_DS4_SUPPORT_RELATIVE_DIR = Path("vendor") / "ds4" / "darwin-arm64"
DS4_SUPPORT_MANIFEST = "manifest.json"
DS4_SUPPORT_FILES: tuple[str, ...] = (
    "LICENSE",
    "README.md",
)
DS4_REQUIRED_CLI_FLAGS: tuple[str, ...] = (
    "--ssd-streaming",
    "--mtp",
    "--dspark",
)
_DS4_AUTO_BUILD_FAILURES: dict[tuple[str, str, str], str] = {}
_DS4_SUPPORT_LOCKS: dict[str, threading.Lock] = {}
_DS4_SUPPORT_LOCKS_GUARD = threading.Lock()


class DS4SupportError(RuntimeError):
    """Raised when DS4 support files are unavailable or incomplete."""


def clear_ds4_auto_build_failures() -> None:
    """Clear remembered DS4 auto-build failures.

    Intended for tests and long-running callers that deliberately change the
    build environment before retrying in the same process.
    """
    _DS4_AUTO_BUILD_FAILURES.clear()


def _ds4_auto_build_failure_key(
    settings: DS4Settings,
    base_path: Path,
) -> tuple[str, str, str]:
    """Key auto-build failures by destination and explicit source override."""
    return (
        str(settings.get_support_dir(base_path)),
        settings.source_repo or "",
        settings.source_commit or "",
    )


def _ds4_support_lock(settings: DS4Settings, base_path: Path) -> threading.Lock:
    key = str(settings.get_support_dir(base_path))
    with _DS4_SUPPORT_LOCKS_GUARD:
        lock = _DS4_SUPPORT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DS4_SUPPORT_LOCKS[key] = lock
        return lock


@dataclass(frozen=True)
class DS4SupportManifest:
    """Pinned DS4 source/build metadata shipped with oMLX."""

    name: str
    source_repo: str
    source_commit: str
    platform: str
    binary: str
    build_command: str
    required_cli_flags: tuple[str, ...]
    binary_sha256: str | None = None
    metal_files: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "DS4SupportManifest":
        """Build a typed manifest from parsed JSON."""
        required = (
            "name",
            "source_repo",
            "source_commit",
            "platform",
            "binary",
            "build_command",
        )
        missing = [key for key in required if not str(data.get(key) or "").strip()]
        if missing:
            raise DS4SupportError(
                "DS4 support manifest is missing required field(s): "
                + ", ".join(missing)
            )
        flags = data.get("required_cli_flags", ())
        if not isinstance(flags, list):
            raise DS4SupportError(
                "DS4 support manifest required_cli_flags must be a list"
            )
        metal_files = data.get("metal_files")
        if metal_files is None:
            metal_files = []
        elif not isinstance(metal_files, list):
            raise DS4SupportError("DS4 support manifest metal_files must be a list")
        return cls(
            name=str(data["name"]),
            source_repo=str(data["source_repo"]),
            source_commit=str(data["source_commit"]),
            platform=str(data["platform"]),
            binary=str(data["binary"]),
            build_command=str(data["build_command"]),
            required_cli_flags=tuple(str(flag) for flag in flags),
            binary_sha256=(
                str(data["binary_sha256"])
                if str(data.get("binary_sha256") or "").strip()
                else None
            ),
            metal_files=tuple(
                rel
                for rel in (
                    _normalize_ds4_metal_relative_path(item) for item in metal_files
                )
                if rel is not None
            ),
        )

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-serializable manifest mapping."""
        data: dict[str, object] = {
            "name": self.name,
            "source_repo": self.source_repo,
            "source_commit": self.source_commit,
            "platform": self.platform,
            "binary": self.binary,
            "build_command": self.build_command,
            "required_cli_flags": list(self.required_cli_flags),
        }
        if self.binary_sha256:
            data["binary_sha256"] = self.binary_sha256
        if self.metal_files:
            data["metal_files"] = list(self.metal_files)
        return data


@dataclass(frozen=True)
class DS4SupportStatus:
    """Result of inspecting the configured DS4 support directory."""

    support_dir: Path
    binary_path: Path
    missing_files: tuple[str, ...]
    binary_missing: bool
    binary_not_executable: bool
    unsupported_platform: bool
    platform_name: str
    binary_capability_error: str | None = None

    @property
    def ready(self) -> bool:
        """True when all required support files are present and launchable."""
        return not (
            self.missing_files
            or self.binary_missing
            or self.binary_not_executable
            or self.unsupported_platform
            or self.binary_capability_error
        )

    def error_message(self) -> str | None:
        """Return a clear user-facing error, or ``None`` when ready."""
        problems: list[str] = []
        if self.unsupported_platform:
            problems.append(
                "DS4 backend is supported only on macOS Apple Silicon "
                f"(detected {self.platform_name})"
            )
        if self.binary_missing:
            problems.append(f"missing DS4 binary: {self.binary_path}")
        elif self.binary_not_executable:
            problems.append(f"DS4 binary is not executable: {self.binary_path}")
        elif self.binary_capability_error:
            problems.append(
                f"DS4 binary is incompatible: {self.binary_path}: "
                f"{self.binary_capability_error}"
            )
        if self.missing_files:
            missing = ", ".join(self.missing_files)
            problems.append(
                f"missing DS4 support files under {self.support_dir}: {missing}"
            )
        if not problems:
            return None
        return "; ".join(problems)


@dataclass(frozen=True)
class DS4SupportCopyResult:
    """Files copied into the DS4 support directory."""

    source_dir: Path
    destination_dir: Path
    copied_files: tuple[Path, ...]


def _platform_name(system: str | None = None, machine: str | None = None) -> str:
    system = system or platform.system()
    machine = machine or platform.machine()
    return f"{system} {machine}".strip()


def is_ds4_supported_platform(
    system: str | None = None, machine: str | None = None
) -> bool:
    """Return True for the v1 DS4 target platform: macOS Apple Silicon."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    return system == "darwin" and machine in {"arm64", "aarch64"}


def _base_ds4_support_relative_paths(*, include_binary: bool = True) -> list[str]:
    """Return non-Metal support paths relative to the DS4 support directory."""
    paths: list[str] = []
    if include_binary:
        paths.append(DS4_SERVER_BINARY)
    paths.extend(DS4_SUPPORT_FILES)
    return paths


def _normalize_ds4_metal_relative_path(value: object) -> str | None:
    rel = str(value).strip().replace("\\", "/")
    parts = rel.split("/")
    if (
        len(parts) < 2
        or parts[0] != "metal"
        or not rel.endswith(".metal")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        return None
    return rel


def _discover_ds4_metal_relative_paths(root: Path) -> tuple[str, ...]:
    metal_dir = root / "metal"
    if not metal_dir.is_dir():
        return ()
    return tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in metal_dir.rglob("*.metal")
            if path.is_file()
        )
    )


def _remove_stale_ds4_metal_files(
    destination: Path,
    keep_relative_paths: Iterable[str],
) -> None:
    keep = set(keep_relative_paths)
    for rel in _discover_ds4_metal_relative_paths(destination):
        if rel not in keep:
            (destination / rel).unlink()


def _manifest_ds4_metal_relative_paths(root: Path) -> tuple[str, ...]:
    manifest_path = root / DS4_SUPPORT_MANIFEST
    if not manifest_path.is_file():
        return ()
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return ()
    if not isinstance(data, dict):
        return ()
    metal_files = data.get("metal_files", ())
    if not isinstance(metal_files, list):
        return ()
    return tuple(
        rel
        for rel in (_normalize_ds4_metal_relative_path(item) for item in metal_files)
        if rel is not None
    )


def _missing_relative_paths(
    root: Path, relative_paths: Iterable[str]
) -> tuple[str, ...]:
    return tuple(rel for rel in relative_paths if not (root / rel).is_file())


def default_ds4_support_manifest_path(*, module_file: str | Path | None = None) -> Path:
    """Return the in-tree DS4 source pin manifest path."""
    module_path = Path(module_file or __file__).expanduser().resolve()
    return module_path.parent / VENDORED_DS4_SUPPORT_RELATIVE_DIR / DS4_SUPPORT_MANIFEST


def load_ds4_support_manifest(
    manifest_path: str | Path | None = None,
) -> DS4SupportManifest:
    """Load the pinned DS4 source/build manifest."""
    path = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else default_ds4_support_manifest_path()
    )
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise DS4SupportError(f"DS4 support manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DS4SupportError(
            f"DS4 support manifest is invalid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise DS4SupportError(
            f"DS4 support manifest must contain a JSON object: {path}"
        )
    return DS4SupportManifest.from_mapping(data)


def _xcode_select_available() -> bool:
    try:
        completed = subprocess.run(
            ["xcode-select", "-p"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def ds4_build_environment_errors(
    *,
    system: str | None = None,
    machine: str | None = None,
    require_git: bool = True,
    require_make: bool = True,
) -> tuple[str, ...]:
    """Return missing prerequisite messages for building DS4 locally."""
    errors: list[str] = []
    if not is_ds4_supported_platform(system, machine):
        errors.append(
            "DS4 support can be built only on macOS Apple Silicon "
            f"(detected {_platform_name(system, machine)})"
        )
    if require_git and shutil.which("git") is None:
        errors.append("missing git")
    if require_make and shutil.which("make") is None:
        errors.append("missing make")
    if (
        require_make
        and (system or platform.system()).lower() == "darwin"
        and not _xcode_select_available()
    ):
        errors.append("missing Apple Command Line Tools")
    return tuple(errors)


def ds4_build_environment_hint(errors: Iterable[str]) -> str:
    """Format an actionable DS4 build-prerequisite hint."""
    problem_list = ", ".join(errors)
    hint = (
        "Install Apple Command Line Tools with `xcode-select --install`, "
        "then retry. If you already have a prebuilt DS4 tree, set "
        "OMLX_DS4_SUPPORT_DIR or ds4.support_dir; for a custom binary, set "
        "OMLX_DS4_BINARY_PATH or ds4.binary_path."
    )
    return f"Cannot build DS4 support files: {problem_list}. {hint}"


def _raise_if_ds4_build_environment_unavailable(
    *,
    system: str | None = None,
    machine: str | None = None,
    require_git: bool = True,
    require_make: bool = True,
) -> None:
    errors = ds4_build_environment_errors(
        system=system,
        machine=machine,
        require_git=require_git,
        require_make=require_make,
    )
    if errors:
        raise DS4SupportError(ds4_build_environment_hint(errors))


def _inspect_ds4_binary_capabilities(binary_path: Path) -> str | None:
    """Return a compatibility error when the DS4 binary lacks required flags."""
    try:
        completed = subprocess.run(
            [str(binary_path), "--help", "all"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "timed out while probing --help"
    except OSError as exc:
        return f"failed to run --help: {exc}"

    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        return f"--help exited with status {completed.returncode}"
    missing = tuple(flag for flag in DS4_REQUIRED_CLI_FLAGS if flag not in output)
    if missing:
        return "missing required CLI option(s): " + ", ".join(missing)
    return None


def _source_is_local_dir(source: str | Path) -> bool:
    return Path(source).expanduser().is_dir()


def _git_head(source_dir: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _clone_ds4_source(repo: str, commit: str | None, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if commit:
        destination.mkdir(parents=True, exist_ok=True)
        commands = (
            ["git", "init", str(destination)],
            ["git", "-C", str(destination), "remote", "add", "origin", repo],
            ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", commit],
            ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
        )
    else:
        commands = (["git", "clone", "--depth", "1", repo, str(destination)],)
    for command in commands:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            output = (completed.stderr or completed.stdout).strip()
            source_label = f"{repo}@{commit}" if commit else repo
            raise DS4SupportError(
                "Failed to clone DS4 source "
                f"{source_label}: {' '.join(command)}: {output}"
            )
    return destination


def _run_ds4_build_command(source_dir: Path, build_command: str) -> None:
    argv = shlex.split(build_command)
    if not argv:
        raise DS4SupportError("DS4 support manifest build_command is empty")
    try:
        completed = subprocess.run(
            argv,
            cwd=source_dir,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except OSError as exc:
        raise DS4SupportError(
            f"Failed to run DS4 build command {build_command!r}: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DS4SupportError(
            f"Timed out while running DS4 build command {build_command!r}"
        ) from exc
    if completed.returncode != 0:
        output = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part and part.strip()
        )
        raise DS4SupportError(
            f"DS4 build command failed with status {completed.returncode}: "
            f"{build_command}\n{output}"
        )


def _write_staged_manifest(
    destination: Path,
    manifest: DS4SupportManifest,
    *,
    source_repo: str,
    source_commit: str,
    source_dir: Path,
) -> Path:
    data = manifest.to_mapping()
    data["source_repo"] = source_repo
    data["source_commit"] = source_commit
    if source_repo == str(source_dir):
        data["source_path"] = str(source_dir)
    data["metal_files"] = list(_discover_ds4_metal_relative_paths(destination))
    manifest_path = destination / DS4_SUPPORT_MANIFEST
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    return manifest_path


def _build_and_copy_ds4_support(
    source_dir: Path,
    destination: Path,
    manifest: DS4SupportManifest,
    *,
    source_repo: str,
    source_commit: str,
    skip_build: bool,
    overwrite: bool,
) -> DS4SupportCopyResult:
    if not skip_build:
        _run_ds4_build_command(source_dir, manifest.build_command)
    result = copy_ds4_support_files(source_dir, destination, overwrite=overwrite)
    manifest_path = _write_staged_manifest(
        result.destination_dir,
        manifest,
        source_repo=source_repo,
        source_commit=source_commit,
        source_dir=source_dir,
    )
    copied = tuple(result.copied_files) + (
        (manifest_path,) if manifest_path not in result.copied_files else ()
    )
    return DS4SupportCopyResult(
        source_dir=result.source_dir,
        destination_dir=result.destination_dir,
        copied_files=copied,
    )


def build_ds4_support_from_source(
    settings: DS4Settings | None = None,
    *,
    base_path: str | Path | None = None,
    destination_dir: str | Path | None = None,
    source: str | Path | None = None,
    commit: str | None = None,
    manifest_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    skip_build: bool = False,
    overwrite: bool = True,
    validate_environment: bool = True,
    system: str | None = None,
    machine: str | None = None,
) -> DS4SupportCopyResult:
    """Build DS4 from a pinned/custom source and stage the support tree."""
    settings = settings or DS4Settings()
    base = Path(base_path).expanduser().resolve() if base_path else DEFAULT_BASE_PATH
    destination = (
        Path(destination_dir).expanduser().resolve()
        if destination_dir is not None
        else settings.get_support_dir(base)
    )
    manifest = load_ds4_support_manifest(manifest_path)
    source_value = (
        str(source)
        if source is not None
        else (settings.source_repo or manifest.source_repo)
    )
    source_override = source is not None or bool(settings.source_repo)
    requested_commit = commit or settings.source_commit
    source_commit = (
        requested_commit
        if requested_commit
        else (None if source_override else manifest.source_commit)
    )
    source_is_local = _source_is_local_dir(source_value)

    if validate_environment:
        _raise_if_ds4_build_environment_unavailable(
            system=system,
            machine=machine,
            require_git=True,
            require_make=not skip_build,
        )

    if source_is_local:
        source_dir = Path(source_value).expanduser().resolve()
        current_head = _git_head(source_dir)
        if not current_head:
            raise DS4SupportError(
                f"Cannot verify local DS4 source {source_dir}; use a git checkout "
                "so oMLX can record the built source commit"
            )
        if source_commit:
            if current_head != source_commit:
                raise DS4SupportError(
                    f"Local DS4 source {source_dir} is at {current_head}, "
                    f"not requested commit {source_commit}"
                )
        return _build_and_copy_ds4_support(
            source_dir,
            destination,
            manifest,
            source_repo=str(source_dir),
            source_commit=current_head,
            skip_build=skip_build,
            overwrite=overwrite,
        )

    repo = source_value
    if work_dir is not None:
        checkout = Path(work_dir).expanduser().resolve() / "ds4-source"
        source_dir = _clone_ds4_source(repo, source_commit, checkout)
        current_head = _git_head(source_dir)
        if not current_head:
            raise DS4SupportError(
                f"Cannot verify cloned DS4 source {repo}; git did not report HEAD"
            )
        return _build_and_copy_ds4_support(
            source_dir,
            destination,
            manifest,
            source_repo=repo,
            source_commit=current_head,
            skip_build=skip_build,
            overwrite=overwrite,
        )

    with tempfile.TemporaryDirectory(prefix="omlx-ds4-build-") as temp_dir:
        source_dir = _clone_ds4_source(repo, source_commit, Path(temp_dir) / "ds4")
        current_head = _git_head(source_dir)
        if not current_head:
            raise DS4SupportError(
                f"Cannot verify cloned DS4 source {repo}; git did not report HEAD"
            )
        return _build_and_copy_ds4_support(
            source_dir,
            destination,
            manifest,
            source_repo=repo,
            source_commit=current_head,
            skip_build=skip_build,
            overwrite=overwrite,
        )


def inspect_ds4_support(
    settings: DS4Settings | None = None,
    *,
    base_path: str | Path | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> DS4SupportStatus:
    """Inspect configured DS4 support files without mutating the filesystem."""
    settings = settings or DS4Settings()
    base = Path(base_path).expanduser().resolve() if base_path else DEFAULT_BASE_PATH
    support_dir = settings.get_support_dir(base)
    binary_path = settings.get_binary_path(base)
    binary_override = settings.binary_path is not None
    missing_files = list(
        _missing_relative_paths(
            support_dir,
            _base_ds4_support_relative_paths(include_binary=not binary_override),
        )
    )
    metal_files = _manifest_ds4_metal_relative_paths(
        support_dir
    ) or _discover_ds4_metal_relative_paths(support_dir)
    if metal_files:
        missing_files.extend(_missing_relative_paths(support_dir, metal_files))
    else:
        missing_files.append("metal/*.metal")
    binary_missing = not binary_path.is_file()
    binary_not_executable = binary_path.is_file() and not os.access(
        binary_path, os.X_OK
    )
    platform_name = _platform_name(system, machine)
    unsupported_platform = not is_ds4_supported_platform(system, machine)
    binary_capability_error = None
    if not (binary_missing or binary_not_executable or unsupported_platform):
        binary_capability_error = _inspect_ds4_binary_capabilities(binary_path)

    return DS4SupportStatus(
        support_dir=support_dir,
        binary_path=binary_path,
        missing_files=tuple(missing_files),
        binary_missing=binary_missing,
        binary_not_executable=binary_not_executable,
        unsupported_platform=unsupported_platform,
        platform_name=platform_name,
        binary_capability_error=binary_capability_error,
    )


def require_ds4_support(
    settings: DS4Settings | None = None,
    *,
    base_path: str | Path | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> DS4SupportStatus:
    """Return support status or raise ``DS4SupportError`` with a clear message."""
    status = inspect_ds4_support(
        settings,
        base_path=base_path,
        system=system,
        machine=machine,
    )
    if not status.ready:
        raise DS4SupportError(status.error_message() or "DS4 support is unavailable")
    return status


def ensure_ds4_support(
    settings: DS4Settings | None = None,
    *,
    base_path: str | Path | None = None,
    system: str | None = None,
    machine: str | None = None,
    source_dir: str | Path | None = None,
) -> DS4SupportStatus:
    """Ensure bundled DS4 support files are installed, then validate them.

    This is the transparent provisioning path used by managed DS4 launches.  A
    ready configured directory is returned unchanged.  Otherwise, when the user
    has not configured a custom ``ds4.support_dir``, bundled/vendor support
    files are copied into the default oMLX support directory, or the pinned
    source is built on first DS4 launch when auto-build is enabled.  Explicit
    custom support directories remain user-managed.
    """
    settings = settings or DS4Settings()
    base = Path(base_path).expanduser().resolve() if base_path else DEFAULT_BASE_PATH
    status = inspect_ds4_support(
        settings,
        base_path=base,
        system=system,
        machine=machine,
    )
    if status.ready:
        return status
    if status.unsupported_platform:
        raise DS4SupportError(status.error_message() or "DS4 support is unavailable")

    # Keep explicit custom support directories fully user-managed.  The default
    # support dir is oMLX-owned and can be repaired from bundled resources or
    # built from the pinned source commit.
    if settings.support_dir is None:
        with _ds4_support_lock(settings, base):
            status = inspect_ds4_support(
                settings,
                base_path=base,
                system=system,
                machine=machine,
            )
            if status.ready:
                return status
            if status.unsupported_platform:
                raise DS4SupportError(
                    status.error_message() or "DS4 support is unavailable"
                )
            install_bundled_ds4_support_files(
                settings,
                base_path=base,
                source_dir=source_dir,
                overwrite=True,
            )
            status = inspect_ds4_support(
                settings,
                base_path=base,
                system=system,
                machine=machine,
            )
            if status.ready:
                return status
            if settings.auto_build:
                failure_key = _ds4_auto_build_failure_key(settings, base)
                if failure_key in _DS4_AUTO_BUILD_FAILURES:
                    raise DS4SupportError(
                        (status.error_message() or "DS4 support is unavailable")
                        + "; DS4 auto-build previously failed: "
                        + _DS4_AUTO_BUILD_FAILURES[failure_key]
                    )
                try:
                    build_ds4_support_from_source(
                        settings,
                        base_path=base,
                        system=system,
                        machine=machine,
                    )
                except DS4SupportError as exc:
                    _DS4_AUTO_BUILD_FAILURES[failure_key] = str(exc)
                    raise
                status = inspect_ds4_support(
                    settings,
                    base_path=base,
                    system=system,
                    machine=machine,
                )
                if status.ready:
                    _DS4_AUTO_BUILD_FAILURES.pop(failure_key, None)
                    return status
                failure_message = (
                    status.error_message() or "built DS4 support is unavailable"
                )
                _DS4_AUTO_BUILD_FAILURES[failure_key] = failure_message
                raise DS4SupportError(failure_message)
            else:
                raise DS4SupportError(
                    (status.error_message() or "DS4 support is unavailable")
                    + "; DS4 auto-build is disabled. Run `omlx ds4 install` or "
                    "configure ds4.support_dir / OMLX_DS4_SUPPORT_DIR."
                )

    raise DS4SupportError(status.error_message() or "DS4 support is unavailable")


def find_bundled_ds4_support_dir(
    *,
    env: Mapping[str, str] | None = None,
    module_file: str | Path | None = None,
) -> Path | None:
    """Locate bundled DS4 support files shipped next to the app resources.

    The Swift app bundle copies Python sources into ``Contents/Resources/omlx``
    and DS4 runtime files into ``Contents/Resources/DS4Support``.  A hidden env
    override keeps tests and alternate packagers deterministic.
    """
    environment = os.environ if env is None else env
    if bundled_override := environment.get(BUNDLED_DS4_SUPPORT_ENV):
        return Path(bundled_override).expanduser().resolve()

    module_path = Path(module_file or __file__).expanduser().resolve()
    try:
        resources_dir = module_path.parents[1]
    except IndexError:
        return None

    for name in (BUNDLED_DS4_SUPPORT_DIR_NAME, "ds4"):
        candidate = resources_dir / name
        if candidate.is_dir():
            return candidate

    package_candidate = module_path.parent / VENDORED_DS4_SUPPORT_RELATIVE_DIR
    if (package_candidate / DS4_SERVER_BINARY).is_file():
        return package_candidate
    return None


def install_bundled_ds4_support_files(
    settings: DS4Settings | None = None,
    *,
    base_path: str | Path | None = None,
    source_dir: str | Path | None = None,
    overwrite: bool = False,
) -> DS4SupportCopyResult | None:
    """Copy bundled DS4 support files into the default user support dir.

    Custom ``ds4.support_dir`` values are treated as an explicit user choice and
    are left untouched.  Returning ``None`` means no bundled source was present
    or the user configured a custom support directory.
    """
    settings = settings or DS4Settings()
    if settings.support_dir is not None:
        return None

    base = Path(base_path).expanduser().resolve() if base_path else DEFAULT_BASE_PATH
    source = (
        Path(source_dir).expanduser().resolve()
        if source_dir is not None
        else find_bundled_ds4_support_dir()
    )
    if source is None:
        return None
    return copy_ds4_support_files(
        source,
        settings.get_support_dir(base),
        overwrite=overwrite,
    )


def copy_ds4_support_files(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    overwrite: bool = False,
) -> DS4SupportCopyResult:
    """Copy required DS4 support files from a bundled resource directory.

    The copy is intentionally deterministic: only the required binary,
    license, README, and discovered Metal source files are copied. Missing
    bundled inputs raise a clear error instead of attempting to rebuild or fetch
    DS4.
    """
    source = Path(source_dir).expanduser().resolve()
    destination = Path(destination_dir).expanduser().resolve()
    required = _base_ds4_support_relative_paths(include_binary=True)
    missing = _missing_relative_paths(source, required)
    metal_files = _discover_ds4_metal_relative_paths(source)
    if metal_files:
        required.extend(metal_files)
    else:
        missing = missing + ("metal/*.metal",)
    if missing:
        raise DS4SupportError(
            "Bundled DS4 support files are incomplete under "
            f"{source}: {', '.join(missing)}"
        )
    capability_error = _inspect_ds4_binary_capabilities(source / DS4_SERVER_BINARY)
    if capability_error:
        raise DS4SupportError(
            f"Bundled DS4 binary is incompatible under {source}: {capability_error}"
        )

    if overwrite:
        _remove_stale_ds4_metal_files(destination, metal_files)

    copied: list[Path] = []
    for rel in required:
        src = source / rel
        dst = destination / rel
        if dst.exists() and not overwrite:
            if rel == DS4_SERVER_BINARY:
                dst.chmod(dst.stat().st_mode | 0o111)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if rel == DS4_SERVER_BINARY:
            dst.chmod(dst.stat().st_mode | 0o111)
        copied.append(dst)

    manifest_src = source / DS4_SUPPORT_MANIFEST
    manifest_dst = destination / DS4_SUPPORT_MANIFEST
    if manifest_src.is_file():
        if overwrite or not manifest_dst.exists():
            shutil.copy2(manifest_src, manifest_dst)
            copied.append(manifest_dst)
    elif overwrite and manifest_dst.exists():
        manifest_dst.unlink()

    return DS4SupportCopyResult(
        source_dir=source,
        destination_dir=destination,
        copied_files=tuple(copied),
    )
