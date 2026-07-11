# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omlx.admin.aria2_downloader import (
    Aria2DownloadError,
    Aria2File,
    Aria2Installer,
    Aria2Runner,
    Aria2UnavailableError,
    configured_aria2_runner,
    find_aria2c,
    get_aria2_status,
)


def test_find_aria2c_uses_path() -> None:
    with patch("omlx.admin.aria2_downloader.shutil.which", return_value="/bin/aria2c"):
        assert find_aria2c() == "/bin/aria2c"


def test_mirror_environment_removes_every_proxy_spelling() -> None:
    environ = {
        "HTTP_PROXY": "http://app-proxy",
        "http_proxy": "http://lower-proxy",
        "HTTPS_PROXY": "http://secure-proxy",
        "https_proxy": "http://lower-secure-proxy",
        "ALL_PROXY": "socks5://all-proxy",
        "all_proxy": "socks5://lower-all-proxy",
        "FTP_PROXY": "http://ftp-proxy",
        "ftp_proxy": "http://lower-ftp-proxy",
        "NO_PROXY": "localhost",
        "no_proxy": "localhost",
        "PATH": "/usr/bin",
    }

    result = Aria2Runner.build_environment(environ, bypass_proxies=True)

    assert result == {"PATH": "/usr/bin"}


def test_default_environment_preserves_proxy_configuration() -> None:
    environ = {"HTTPS_PROXY": "http://proxy", "PATH": "/usr/bin"}

    assert Aria2Runner.build_environment(environ, bypass_proxies=False) == environ


def test_command_explicitly_disables_aria2_proxies_for_mirror(tmp_path: Path) -> None:
    runner = Aria2Runner(
        executable="/opt/homebrew/bin/aria2c",
        proxy="http://127.0.0.1:7897",
        connections_per_file=12,
        concurrent_files=3,
    )

    args = runner.build_command(tmp_path / "manifest.txt", bypass_proxies=True)

    assert "--all-proxy=" in args
    assert "--http-proxy=" in args
    assert "--https-proxy=" in args
    assert "--ftp-proxy=" in args
    assert "--no-proxy=*" in args
    assert "--all-proxy=http://127.0.0.1:7897" not in args
    assert "--split=12" in args
    assert "--max-connection-per-server=12" in args
    assert "--max-concurrent-downloads=3" in args


def test_command_uses_explicit_aria2_proxy_without_mirror(tmp_path: Path) -> None:
    runner = Aria2Runner(
        executable="/opt/homebrew/bin/aria2c",
        proxy="http://127.0.0.1:7897",
    )

    args = runner.build_command(tmp_path / "manifest.txt", bypass_proxies=False)

    assert "--all-proxy=http://127.0.0.1:7897" in args


def test_manifest_keeps_token_out_of_command_line_and_rejects_path_escape(
    tmp_path: Path,
) -> None:
    runner = Aria2Runner(executable="/opt/homebrew/bin/aria2c")
    token = "hf_secret_token"
    files = [
        Aria2File(
            url="https://huggingface.co/owner/model/resolve/main/model.safetensors",
            relative_path="weights/model.safetensors",
            size=123,
            headers=(f"Authorization: Bearer {token}",),
        )
    ]

    manifest = runner.write_manifest(files, tmp_path)
    command = runner.build_command(manifest, bypass_proxies=False)

    assert token not in " ".join(command)
    assert token in manifest.read_text()
    assert oct(manifest.stat().st_mode & 0o777) == "0o600"

    with pytest.raises(ValueError, match="relative path"):
        runner.write_manifest(
            [Aria2File(url="https://example.test/file", relative_path="../escape")],
            tmp_path,
        )


def test_runner_requires_aria2() -> None:
    with (
        patch("omlx.admin.aria2_downloader.find_aria2c", return_value=None),
        pytest.raises(Aria2UnavailableError, match="aria2"),
    ):
        Aria2Runner()


def test_configured_runner_reads_global_transfer_settings() -> None:
    settings = MagicMock()
    settings.aria2.proxy = "http://127.0.0.1:7897"
    settings.aria2.connections_per_file = 6
    settings.aria2.concurrent_files = 2

    with (
        patch("omlx.settings.get_settings", return_value=settings),
        patch("omlx.admin.aria2_downloader.find_aria2c", return_value="/bin/aria2c"),
    ):
        runner = configured_aria2_runner()

    assert runner.proxy == "http://127.0.0.1:7897"
    assert runner.connections_per_file == 6
    assert runner.concurrent_files == 2


@pytest.mark.asyncio
async def test_download_removes_protected_manifest_after_success(
    tmp_path: Path,
) -> None:
    runner = Aria2Runner(executable="/usr/bin/true")
    files = [Aria2File(url="https://example.test/model", relative_path="model.bin")]

    await runner.download(files, tmp_path, bypass_proxies=True)

    assert list(tmp_path.glob(".omlx-aria2-*.txt")) == []


@pytest.mark.asyncio
async def test_download_redacts_authentication_from_aria2_error(tmp_path: Path) -> None:
    executable = tmp_path / "failing-aria2"
    executable.write_text(
        "#!/bin/sh\n" "manifest=${1#--input-file=}\n" 'cat "$manifest"\n' "exit 7\n"
    )
    executable.chmod(0o755)
    token = "hf_do_not_leak"
    runner = Aria2Runner(executable=str(executable))
    files = [
        Aria2File(
            url="https://example.test/model",
            relative_path="model.bin",
            headers=(f"Authorization: Bearer {token}",),
        )
    ]

    with pytest.raises(Aria2DownloadError) as exc_info:
        await runner.download(files, tmp_path, bypass_proxies=False)

    assert token not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)
    assert list(tmp_path.glob(".omlx-aria2-*.txt")) == []


@pytest.mark.asyncio
async def test_installer_runs_homebrew_without_a_shell() -> None:
    installer = Aria2Installer()

    with (
        patch(
            "omlx.admin.aria2_downloader.find_brew",
            return_value="/opt/homebrew/bin/brew",
        ),
        patch.object(installer, "_run", return_value=(0, "installed")) as run,
    ):
        result = await installer.install()

    assert result["installed"] is True
    run.assert_awaited_once_with(
        "/opt/homebrew/bin/brew", "install", "aria2", env=os.environ.copy()
    )


@pytest.mark.asyncio
async def test_installer_reports_missing_homebrew() -> None:
    installer = Aria2Installer()

    with patch("omlx.admin.aria2_downloader.find_brew", return_value=None):
        result = await installer.install()

    assert result == {
        "installed": False,
        "error": "Homebrew is required to install aria2 automatically.",
    }


@pytest.mark.asyncio
async def test_status_reports_missing_aria2() -> None:
    with patch("omlx.admin.aria2_downloader.find_aria2c", return_value=None):
        assert await get_aria2_status() == {
            "installed": False,
            "path": None,
            "version": None,
        }
