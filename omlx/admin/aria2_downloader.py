# SPDX-License-Identifier: Apache-2.0
"""Optional aria2 process support for ModelScope downloads."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_ARIA2_PATHS = ("/opt/homebrew/bin/aria2c", "/usr/local/bin/aria2c")
_BREW_PATHS = ("/opt/homebrew/bin/brew", "/usr/local/bin/brew")
_PROXY_KEYS = {
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "ftp_proxy",
    "no_proxy",
}


class Aria2UnavailableError(RuntimeError):
    """Raised when an aria2 runner is requested without aria2 installed."""


class Aria2DownloadError(RuntimeError):
    """Raised when aria2 exits unsuccessfully."""


@dataclass(frozen=True)
class Aria2File:
    """One repository file ready for transfer by aria2."""

    url: str
    relative_path: str
    size: int = 0
    headers: tuple[str, ...] = ()


def _find_executable(name: str, candidates: Sequence[str]) -> str | None:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    for candidate in candidates:
        if os.access(candidate, os.X_OK):
            return candidate
    return None


def find_aria2c() -> str | None:
    """Return the aria2c executable path, including common Homebrew paths."""

    return _find_executable("aria2c", _ARIA2_PATHS)


def find_brew() -> str | None:
    """Return the Homebrew executable path on Apple Silicon or Intel Macs."""

    return _find_executable("brew", _BREW_PATHS)


def configured_aria2_runner() -> Aria2Runner | None:
    """Build a configured runner, or return None when aria2 is unavailable."""

    try:
        from ..settings import get_settings

        settings = get_settings().aria2
    except (RuntimeError, AttributeError):
        settings = None
    try:
        return Aria2Runner(
            proxy=settings.proxy if settings else "",
            connections_per_file=settings.connections_per_file if settings else 8,
            concurrent_files=settings.concurrent_files if settings else 4,
        )
    except Aria2UnavailableError:
        return None


async def get_aria2_status() -> dict[str, object]:
    """Return executable availability and the first version line."""

    executable = find_aria2c()
    if not executable:
        return {"installed": False, "path": None, "version": None}
    process = await asyncio.create_subprocess_exec(
        executable,
        "--version",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout=5)
    except TimeoutError:
        process.kill()
        await process.wait()
        return {"installed": True, "path": executable, "version": None}
    first_line = output.decode("utf-8", errors="replace").splitlines()
    return {
        "installed": True,
        "path": executable,
        "version": first_line[0] if first_line else None,
    }


class Aria2Runner:
    """Build safe aria2 manifests and commands for short-lived downloads."""

    def __init__(
        self,
        executable: str | None = None,
        *,
        proxy: str = "",
        connections_per_file: int = 8,
        concurrent_files: int = 4,
    ):
        self.executable = executable or find_aria2c()
        if not self.executable:
            raise Aria2UnavailableError("aria2 is not installed")
        self.proxy = proxy.strip()
        self.connections_per_file = max(1, min(16, int(connections_per_file)))
        self.concurrent_files = max(1, min(16, int(concurrent_files)))

    @staticmethod
    def build_environment(
        environ: Mapping[str, str], *, bypass_proxies: bool
    ) -> dict[str, str]:
        """Copy an environment, optionally removing every proxy spelling."""

        if not bypass_proxies:
            return dict(environ)
        return {
            key: value
            for key, value in environ.items()
            if key.lower() not in _PROXY_KEYS
        }

    def build_command(self, manifest: Path, *, bypass_proxies: bool) -> list[str]:
        """Return an argv vector; secrets live in the protected manifest."""

        args = [
            self.executable,
            f"--input-file={manifest}",
            "--continue=true",
            "--allow-overwrite=true",
            "--auto-file-renaming=false",
            "--file-allocation=none",
            f"--max-connection-per-server={self.connections_per_file}",
            f"--split={self.connections_per_file}",
            f"--max-concurrent-downloads={self.concurrent_files}",
            "--min-split-size=1M",
            "--max-tries=5",
            "--retry-wait=2",
            "--connect-timeout=30",
            "--timeout=60",
            "--summary-interval=1",
            "--download-result=hide",
            "--console-log-level=warn",
        ]
        if bypass_proxies:
            args.extend(
                [
                    "--all-proxy=",
                    "--http-proxy=",
                    "--https-proxy=",
                    "--ftp-proxy=",
                    "--no-proxy=*",
                ]
            )
        elif self.proxy:
            args.append(f"--all-proxy={self.proxy}")
        return args

    def write_manifest(self, files: Sequence[Aria2File], target_dir: Path) -> Path:
        """Write a mode-0600 aria2 input file for repository payload files."""

        target_dir.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for item in files:
            relative = PurePosixPath(item.relative_path)
            if relative.is_absolute() or ".." in relative.parts or not relative.name:
                raise ValueError(
                    f"Model file must use a safe relative path: {item.relative_path!r}"
                )
            destination_dir = target_dir.joinpath(*relative.parts[:-1])
            destination_dir.mkdir(parents=True, exist_ok=True)
            lines.append(item.url)
            lines.append(f"  dir={destination_dir}")
            lines.append(f"  out={relative.name}")
            for header in item.headers:
                if "\n" in header or "\r" in header:
                    raise ValueError("aria2 headers must not contain newlines")
                lines.append(f"  header={header}")

        fd, raw_path = tempfile.mkstemp(
            prefix=".omlx-aria2-", suffix=".txt", dir=target_dir
        )
        manifest = Path(raw_path)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines))
                handle.write("\n")
        except Exception:
            os.close(fd)
            manifest.unlink(missing_ok=True)
            raise
        return manifest

    async def download(
        self,
        files: Sequence[Aria2File],
        target_dir: Path,
        *,
        bypass_proxies: bool,
    ) -> None:
        """Download into hidden staging and publish files only after success."""

        staging_dir = target_dir / "._____temp"
        manifest = self.write_manifest(files, staging_dir)
        command = self.build_command(manifest, bypass_proxies=bypass_proxies)
        environment = self.build_environment(os.environ, bypass_proxies=bypass_proxies)
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output, _ = await process.communicate()
        except asyncio.CancelledError:
            if process and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=3)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            raise
        finally:
            manifest.unlink(missing_ok=True)

        if process.returncode:
            message = output.decode("utf-8", errors="replace")[-8000:]
            for item in files:
                for header in item.headers:
                    message = message.replace(header, "[REDACTED]")
                    if ":" in header:
                        secret = header.split(":", 1)[1].strip()
                        if secret:
                            message = message.replace(secret, "[REDACTED]")
            raise Aria2DownloadError(
                message.strip() or f"aria2 exited with status {process.returncode}."
            )

        for item in files:
            relative = PurePosixPath(item.relative_path)
            staged = staging_dir.joinpath(*relative.parts)
            destination = target_dir.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, destination)
        with contextlib.suppress(OSError):
            staging_dir.rmdir()


class Aria2Installer:
    """Install the fixed aria2 Homebrew package without shell interpolation."""

    _lock = asyncio.Lock()

    async def _run(self, *args: str, env: Mapping[str, str]) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            *args,
            env=dict(env),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await process.communicate()
        text = output.decode("utf-8", errors="replace")[-8000:]
        return process.returncode or 0, text

    async def install(self) -> dict[str, object]:
        """Install aria2 with Homebrew and return a bounded status payload."""

        brew = find_brew()
        if not brew:
            return {
                "installed": False,
                "error": "Homebrew is required to install aria2 automatically.",
            }

        async with self._lock:
            code, output = await self._run(
                brew, "install", "aria2", env=os.environ.copy()
            )
        if code == 0:
            return {"installed": True, "path": find_aria2c(), "output": output}
        return {
            "installed": False,
            "error": output.strip() or f"Homebrew exited with status {code}.",
        }
