# SPDX-License-Identifier: Apache-2.0
"""Tests for DS4 support-file validation and copy helpers."""

import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import omlx.ds4_support as ds4_support
from ds4_support_fixtures import (
    PINNED_DS4_METAL_FILES,
    pinned_ds4_support_relative_paths,
)
from omlx.ds4_support import (
    BUNDLED_DS4_SUPPORT_DIR_NAME,
    BUNDLED_DS4_SUPPORT_ENV,
    DS4_REQUIRED_CLI_FLAGS,
    DS4_SERVER_BINARY,
    DS4SupportError,
    build_ds4_support_from_source,
    clear_ds4_auto_build_failures,
    copy_ds4_support_files,
    ensure_ds4_support,
    find_bundled_ds4_support_dir,
    inspect_ds4_support,
    install_bundled_ds4_support_files,
    is_ds4_supported_platform,
    load_ds4_support_manifest,
    require_ds4_support,
)
from omlx.settings import DS4Settings


def _write_complete_support_tree(
    root: Path,
    *,
    executable: bool = True,
    supports_required_flags: bool = True,
    metal_files: tuple[str, ...] = PINNED_DS4_METAL_FILES,
) -> None:
    binary = root / DS4_SERVER_BINARY
    binary.parent.mkdir(parents=True, exist_ok=True)
    if supports_required_flags:
        flag_lines = "\n".join(f"echo '{flag}'" for flag in DS4_REQUIRED_CLI_FLAGS)
        binary.write_text(
            f'#!/bin/sh\nif [ "$1" = "--help" ]; then\n{flag_lines}\nfi\n'
        )
    else:
        binary.write_text(
            '#!/bin/sh\nif [ "$1" = "--help" ]; then\necho stale-ds4\nfi\n'
        )
    binary.chmod(0o755 if executable else 0o644)
    (root / "LICENSE").write_text("MIT\n")
    (root / "README.md").write_text("DS4\n")
    metal_dir = root / "metal"
    metal_dir.mkdir()
    for name in metal_files:
        metal_path = metal_dir / name
        metal_path.parent.mkdir(parents=True, exist_ok=True)
        metal_path.write_text("// metal\n")


def _write_buildable_ds4_source(
    root: Path,
    *,
    metal_files: tuple[str, ...] = PINNED_DS4_METAL_FILES,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "LICENSE").write_text("MIT\n")
    (root / "README.md").write_text("DS4\n")
    metal_dir = root / "metal"
    metal_dir.mkdir()
    for name in metal_files:
        metal_path = metal_dir / name
        metal_path.parent.mkdir(parents=True, exist_ok=True)
        metal_path.write_text("// metal\n")
    (root / "Makefile").write_text(
        "\n".join(
            [
                "ds4-server:",
                "\tprintf '%s\\n' '#!/bin/sh' "
                '\'if [ "$$1" = "--help" ]; then echo --ssd-streaming --mtp --dspark; fi\' '
                "> ds4-server",
                "\tchmod +x ds4-server",
                "",
            ]
        )
    )


def _write_manifest(path: Path, *, source_repo: str, build_command: str) -> None:
    path.write_text(
        json.dumps(
            {
                "name": "ds4",
                "source_repo": source_repo,
                "source_commit": "local-test",
                "platform": "darwin-arm64",
                "binary": "ds4-server",
                "build_command": build_command,
                "required_cli_flags": list(DS4_REQUIRED_CLI_FLAGS),
            }
        )
    )


class TestDS4SupportInspection:
    """Tests for DS4 support status inspection."""

    def test_required_paths_include_binary_license_readme_and_metal(self):
        """Required relative paths match unpatched DS4 runtime needs."""
        paths = pinned_ds4_support_relative_paths()

        assert DS4_SERVER_BINARY in paths
        assert "LICENSE" in paths
        assert "README.md" in paths
        assert "metal/flash_attn.metal" in paths
        assert "metal/set_rows.metal" in paths

    def test_supported_platform_detection(self):
        """V1 DS4 backend is macOS Apple Silicon only."""
        assert is_ds4_supported_platform("Darwin", "arm64") is True
        assert is_ds4_supported_platform("Darwin", "aarch64") is True
        assert is_ds4_supported_platform("Linux", "aarch64") is False
        assert is_ds4_supported_platform("Darwin", "x86_64") is False

    def test_complete_support_tree_is_ready(self, tmp_path):
        """A complete support dir is ready for later process launch."""
        support = tmp_path / "support" / "ds4"
        _write_complete_support_tree(support)
        settings = DS4Settings(support_dir=str(support))

        status = inspect_ds4_support(
            settings,
            base_path=tmp_path,
            system="Darwin",
            machine="arm64",
        )

        assert status.ready is True
        assert status.error_message() is None
        assert status.support_dir == support.resolve()
        assert status.binary_path == (support / DS4_SERVER_BINARY).resolve()

    def test_custom_support_tree_can_use_discovered_metal_files(self, tmp_path):
        """Custom DS4 forks can change the Metal kernel file set."""
        support = tmp_path / "support" / "ds4"
        _write_complete_support_tree(
            support,
            metal_files=("fork_attention.metal", "experimental/block.metal"),
        )
        settings = DS4Settings(support_dir=str(support))

        status = inspect_ds4_support(
            settings,
            base_path=tmp_path,
            system="Darwin",
            machine="arm64",
        )

        assert status.ready is True
        assert status.missing_files == ()

    def test_incompatible_binary_missing_required_flags_is_not_ready(self, tmp_path):
        """Stale ds4-server binaries are rejected before launch."""
        support = tmp_path / "support" / "ds4"
        _write_complete_support_tree(support, supports_required_flags=False)
        settings = DS4Settings(support_dir=str(support))

        status = inspect_ds4_support(
            settings,
            base_path=tmp_path,
            system="Darwin",
            machine="arm64",
        )

        assert status.ready is False
        assert status.binary_capability_error is not None
        assert "--ssd-streaming" in status.binary_capability_error
        assert "incompatible" in (status.error_message() or "")

    def test_missing_support_files_report_clear_error(self, tmp_path):
        """Missing support files produce a user-facing reinstall error message."""
        support = tmp_path / "support" / "ds4"
        support.mkdir(parents=True)
        settings = DS4Settings(support_dir=str(support))

        status = inspect_ds4_support(
            settings,
            base_path=tmp_path,
            system="Darwin",
            machine="arm64",
        )

        assert status.ready is False
        assert status.binary_missing is True
        assert "LICENSE" in status.missing_files
        assert "metal/*.metal" in status.missing_files
        message = status.error_message()
        assert message is not None
        assert "missing DS4 binary" in message
        assert "missing DS4 support files" in message
        assert str(support.resolve()) in message

    def test_binary_override_does_not_require_support_dir_binary(self, tmp_path):
        """Advanced binary override still validates Metal/support files."""
        support = tmp_path / "support" / "ds4"
        _write_complete_support_tree(support)
        (support / DS4_SERVER_BINARY).unlink()
        override = tmp_path / "custom" / "ds4-server"
        override.parent.mkdir()
        flag_lines = "\n".join(f"echo '{flag}'" for flag in DS4_REQUIRED_CLI_FLAGS)
        override.write_text(
            f'#!/bin/sh\nif [ "$1" = "--help" ]; then\n{flag_lines}\nfi\n'
        )
        override.chmod(0o755)
        settings = DS4Settings(
            support_dir=str(support),
            binary_path=str(override),
        )

        status = inspect_ds4_support(
            settings,
            base_path=tmp_path,
            system="Darwin",
            machine="arm64",
        )

        assert status.ready is True
        assert status.binary_path == override.resolve()
        assert DS4_SERVER_BINARY not in status.missing_files

    def test_non_executable_binary_is_not_ready(self, tmp_path):
        """Existing but non-executable ds4-server is reported distinctly."""
        support = tmp_path / "support" / "ds4"
        _write_complete_support_tree(support, executable=False)
        settings = DS4Settings(support_dir=str(support))

        status = inspect_ds4_support(
            settings,
            base_path=tmp_path,
            system="Darwin",
            machine="arm64",
        )

        assert status.ready is False
        assert status.binary_not_executable is True
        assert "not executable" in (status.error_message() or "")

    def test_unsupported_platform_is_reported(self, tmp_path):
        """Support inspection enforces macOS Apple Silicon v1 target."""
        support = tmp_path / "support" / "ds4"
        _write_complete_support_tree(support)
        settings = DS4Settings(support_dir=str(support))

        status = inspect_ds4_support(
            settings,
            base_path=tmp_path,
            system="Linux",
            machine="arm64",
        )

        assert status.ready is False
        assert status.unsupported_platform is True
        assert "macOS Apple Silicon" in (status.error_message() or "")

    def test_require_ds4_support_raises_clear_error(self, tmp_path):
        """Callers can raise a clear error instead of probing status manually."""
        settings = DS4Settings(support_dir=str(tmp_path / "missing"))

        with pytest.raises(DS4SupportError) as exc_info:
            require_ds4_support(
                settings,
                base_path=tmp_path,
                system="Darwin",
                machine="arm64",
            )

        assert "missing DS4 binary" in str(exc_info.value)


class TestDS4BundledSupport:
    """Tests for bundled app-resource DS4 support discovery/install."""

    def test_find_bundled_support_dir_from_env(self, tmp_path):
        source = tmp_path / "bundle" / BUNDLED_DS4_SUPPORT_DIR_NAME
        source.mkdir(parents=True)

        found = find_bundled_ds4_support_dir(
            env={BUNDLED_DS4_SUPPORT_ENV: str(source)},
            module_file=tmp_path / "Resources" / "omlx" / "ds4_support.py",
        )

        assert found == source.resolve()

    def test_find_bundled_support_dir_next_to_app_resources(self, tmp_path):
        resources = tmp_path / "oMLX.app" / "Contents" / "Resources"
        source = resources / BUNDLED_DS4_SUPPORT_DIR_NAME
        source.mkdir(parents=True)
        module_file = resources / "omlx" / "ds4_support.py"
        module_file.parent.mkdir()
        module_file.write_text("# module\n")

        found = find_bundled_ds4_support_dir(env={}, module_file=module_file)

        assert found == source.resolve()

    def test_find_bundled_support_dir_from_package_vendor(self, tmp_path):
        package_dir = tmp_path / "site-packages" / "omlx"
        source = package_dir / "vendor" / "ds4" / "darwin-arm64"
        _write_complete_support_tree(source)
        module_file = package_dir / "ds4_support.py"
        module_file.write_text("# module\n")

        found = find_bundled_ds4_support_dir(env={}, module_file=module_file)

        assert found == source.resolve()

    def test_find_bundled_support_dir_ignores_metadata_only_package_vendor(
        self, tmp_path
    ):
        package_dir = tmp_path / "site-packages" / "omlx"
        source = package_dir / "vendor" / "ds4" / "darwin-arm64"
        source.mkdir(parents=True)
        (source / "manifest.json").write_text("{}\n")
        module_file = package_dir / "ds4_support.py"
        module_file.write_text("# module\n")

        found = find_bundled_ds4_support_dir(env={}, module_file=module_file)

        assert found is None

    def test_install_bundled_support_files_to_default_dir(self, tmp_path):
        source = tmp_path / "Resources" / BUNDLED_DS4_SUPPORT_DIR_NAME
        _write_complete_support_tree(source)
        base_path = tmp_path / "base"

        result = install_bundled_ds4_support_files(
            DS4Settings(),
            base_path=base_path,
            source_dir=source,
        )

        assert result is not None
        assert result.destination_dir == (base_path / "support" / "ds4").resolve()
        assert (base_path / "support" / "ds4" / DS4_SERVER_BINARY).is_file()
        assert (base_path / "support" / "ds4" / "metal" / "dense.metal").is_file()

    def test_install_bundled_support_skips_custom_support_dir(self, tmp_path):
        source = tmp_path / "Resources" / BUNDLED_DS4_SUPPORT_DIR_NAME
        _write_complete_support_tree(source)
        custom = tmp_path / "custom-support"

        result = install_bundled_ds4_support_files(
            DS4Settings(support_dir=str(custom)),
            base_path=tmp_path / "base",
            source_dir=source,
        )

        assert result is None
        assert not custom.exists()

    def test_ensure_ds4_support_installs_bundled_files_transparently(self, tmp_path):
        source = tmp_path / "Resources" / BUNDLED_DS4_SUPPORT_DIR_NAME
        _write_complete_support_tree(source)
        base_path = tmp_path / "base"

        status = ensure_ds4_support(
            DS4Settings(),
            base_path=base_path,
            source_dir=source,
            system="Darwin",
            machine="arm64",
        )

        assert status.ready is True
        assert status.support_dir == (base_path / "support" / "ds4").resolve()
        assert (status.support_dir / DS4_SERVER_BINARY).is_file()
        assert os.access(status.support_dir / DS4_SERVER_BINARY, os.X_OK)

    def test_ensure_ds4_support_rejects_incomplete_bundled_tree_without_build(
        self, tmp_path, monkeypatch
    ):
        """Broken app/Homebrew bundled support fails instead of runtime-building."""
        source = tmp_path / "Resources" / BUNDLED_DS4_SUPPORT_DIR_NAME
        source.mkdir(parents=True)
        (source / "LICENSE").write_text("MIT\n")
        calls = 0

        def fake_build(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("invalid bundled support should fail before build")

        monkeypatch.setattr(ds4_support, "build_ds4_support_from_source", fake_build)

        with pytest.raises(DS4SupportError) as exc_info:
            ensure_ds4_support(
                DS4Settings(auto_build=True),
                base_path=tmp_path / "base",
                source_dir=source,
                system="Darwin",
                machine="arm64",
            )

        assert "Bundled DS4 support files are incomplete" in str(exc_info.value)
        assert calls == 0


class TestDS4SourceBuild:
    """Tests for build-from-source DS4 staging."""

    def setup_method(self):
        clear_ds4_auto_build_failures()

    def teardown_method(self):
        clear_ds4_auto_build_failures()

    def test_load_manifest_reads_source_pin_without_binary_hash(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        _write_manifest(
            manifest,
            source_repo="https://example.invalid/ds4.git",
            build_command="make ds4-server",
        )

        loaded = load_ds4_support_manifest(manifest)

        assert loaded.source_repo == "https://example.invalid/ds4.git"
        assert loaded.source_commit == "local-test"
        assert loaded.binary_sha256 is None
        assert "--ssd-streaming" in loaded.required_cli_flags

    def test_build_from_local_source_stages_validated_support_tree(
        self, tmp_path, monkeypatch
    ):
        source = tmp_path / "ds4-source"
        destination = tmp_path / "support" / "ds4"
        manifest = tmp_path / "manifest.json"
        _write_buildable_ds4_source(source)
        _write_manifest(
            manifest, source_repo=str(source), build_command="make ds4-server"
        )
        monkeypatch.setattr(ds4_support, "_git_head", lambda _source_dir: "local-test")

        result = build_ds4_support_from_source(
            destination_dir=destination,
            source=source,
            manifest_path=manifest,
            validate_environment=False,
        )

        assert result.destination_dir == destination.resolve()
        assert (destination / DS4_SERVER_BINARY).is_file()
        assert os.access(destination / DS4_SERVER_BINARY, os.X_OK)
        assert (destination / "metal" / "flash_attn.metal").is_file()
        staged_manifest = json.loads((destination / "manifest.json").read_text())
        assert staged_manifest["source_repo"] == str(source.resolve())
        assert staged_manifest["source_path"] == str(source.resolve())

    def test_build_from_settings_source_override_stages_support_tree(
        self, tmp_path, monkeypatch
    ):
        source = tmp_path / "ds4-source"
        destination = tmp_path / "support" / "ds4"
        manifest = tmp_path / "manifest.json"
        _write_buildable_ds4_source(source)
        _write_manifest(
            manifest,
            source_repo="https://example.invalid/manifest-default.git",
            build_command="make ds4-server",
        )
        source_commit = "settings-pin"
        monkeypatch.setattr(ds4_support, "_git_head", lambda _source_dir: source_commit)

        result = build_ds4_support_from_source(
            DS4Settings(source_repo=str(source), source_commit=source_commit),
            destination_dir=destination,
            manifest_path=manifest,
            validate_environment=False,
        )

        assert result.destination_dir == destination.resolve()
        assert (destination / DS4_SERVER_BINARY).is_file()
        staged_manifest = json.loads((destination / "manifest.json").read_text())
        assert staged_manifest["source_repo"] == str(source.resolve())
        assert staged_manifest["source_commit"] == source_commit
        assert staged_manifest["source_path"] == str(source.resolve())

    def test_build_from_custom_remote_without_commit_uses_default_head(
        self, tmp_path, monkeypatch
    ):
        checkout_source = tmp_path / "checkout-source"
        destination = tmp_path / "support" / "ds4"
        manifest = tmp_path / "manifest.json"
        repo = "https://example.invalid/fork.git"
        _write_buildable_ds4_source(checkout_source)
        _write_manifest(
            manifest,
            source_repo="https://example.invalid/manifest-default.git",
            build_command="make ds4-server",
        )
        clone_calls: list[tuple[str, str | None]] = []

        def fake_clone(clone_repo, clone_commit, checkout):
            clone_calls.append((clone_repo, clone_commit))
            shutil.copytree(checkout_source, checkout)
            return checkout

        monkeypatch.setattr(ds4_support, "_clone_ds4_source", fake_clone)
        monkeypatch.setattr(ds4_support, "_git_head", lambda _source_dir: "fork-head")

        build_ds4_support_from_source(
            DS4Settings(source_repo=repo),
            destination_dir=destination,
            manifest_path=manifest,
            work_dir=tmp_path,
            validate_environment=False,
        )

        staged_manifest = json.loads((destination / "manifest.json").read_text())
        assert clone_calls == [(repo, None)]
        assert staged_manifest["source_repo"] == repo
        assert staged_manifest["source_commit"] == "fork-head"

    def test_build_from_custom_source_stages_discovered_metal_files(
        self, tmp_path, monkeypatch
    ):
        source = tmp_path / "ds4-source"
        destination = tmp_path / "support" / "ds4"
        manifest = tmp_path / "manifest.json"
        metal_files = ("fork_attention.metal", "nested/custom_kernel.metal")
        _write_buildable_ds4_source(source, metal_files=metal_files)
        _write_complete_support_tree(destination)
        _write_manifest(
            manifest, source_repo=str(source), build_command="make ds4-server"
        )
        monkeypatch.setattr(ds4_support, "_git_head", lambda _source_dir: "fork-head")

        build_ds4_support_from_source(
            destination_dir=destination,
            source=source,
            manifest_path=manifest,
            validate_environment=False,
        )

        staged_manifest = json.loads((destination / "manifest.json").read_text())
        assert (destination / "metal" / "fork_attention.metal").is_file()
        assert (destination / "metal" / "nested" / "custom_kernel.metal").is_file()
        assert not (destination / "metal" / "flash_attn.metal").exists()
        assert staged_manifest["metal_files"] == [
            "metal/fork_attention.metal",
            "metal/nested/custom_kernel.metal",
        ]

    def test_build_from_local_source_rejects_unverified_commit(self, tmp_path):
        source = tmp_path / "ds4-source"
        destination = tmp_path / "support" / "ds4"
        manifest = tmp_path / "manifest.json"
        _write_buildable_ds4_source(source)
        _write_manifest(
            manifest, source_repo=str(source), build_command="make ds4-server"
        )

        with pytest.raises(DS4SupportError) as exc_info:
            build_ds4_support_from_source(
                destination_dir=destination,
                source=source,
                manifest_path=manifest,
                validate_environment=False,
            )

        assert "Cannot verify local DS4 source" in str(exc_info.value)
        assert not destination.exists()

    def test_ensure_ds4_support_builds_on_first_launch_when_enabled(
        self, tmp_path, monkeypatch
    ):
        """Managed DS4 launch can build missing default support files once."""
        base_path = tmp_path / "base"
        calls = 0

        def fake_build(settings, **kwargs):
            nonlocal calls
            calls += 1
            destination = settings.get_support_dir(kwargs["base_path"])
            _write_complete_support_tree(destination)
            return None

        monkeypatch.setattr(ds4_support, "build_ds4_support_from_source", fake_build)

        status = ensure_ds4_support(
            DS4Settings(auto_build=True),
            base_path=base_path,
            system="Darwin",
            machine="arm64",
        )

        assert status.ready is True
        assert calls == 1

    def test_ensure_ds4_support_single_flights_concurrent_auto_build(
        self, tmp_path, monkeypatch
    ):
        """Concurrent first loads share one default support-dir auto-build."""
        base_path = tmp_path / "base"
        calls = 0
        build_started = threading.Event()
        release_build = threading.Event()
        second_build_started = threading.Event()

        def fake_build(settings, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                second_build_started.set()
            build_started.set()
            assert release_build.wait(timeout=2)
            destination = settings.get_support_dir(kwargs["base_path"])
            _write_complete_support_tree(destination)
            return None

        monkeypatch.setattr(ds4_support, "build_ds4_support_from_source", fake_build)
        settings = DS4Settings(auto_build=True)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                ensure_ds4_support,
                settings,
                base_path=base_path,
                system="Darwin",
                machine="arm64",
            )
            assert build_started.wait(timeout=1)
            second = executor.submit(
                ensure_ds4_support,
                settings,
                base_path=base_path,
                system="Darwin",
                machine="arm64",
            )
            assert not second_build_started.wait(timeout=0.2)
            release_build.set()

            assert first.result(timeout=2).ready is True
            assert second.result(timeout=2).ready is True

        assert calls == 1

    def test_ensure_ds4_support_auto_build_disabled_fails_without_build(
        self, tmp_path, monkeypatch
    ):
        """Operators can disable first-launch source builds."""
        calls = 0

        def fake_build(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("auto-build should not run")

        monkeypatch.setattr(ds4_support, "build_ds4_support_from_source", fake_build)

        with pytest.raises(DS4SupportError) as exc_info:
            ensure_ds4_support(
                DS4Settings(auto_build=False),
                base_path=tmp_path,
                system="Darwin",
                machine="arm64",
            )

        assert calls == 0
        assert "auto-build is disabled" in str(exc_info.value)

    def test_ensure_ds4_support_failed_auto_build_is_not_retried(
        self, tmp_path, monkeypatch
    ):
        """A failed auto-build is remembered until support is externally fixed."""
        calls = 0

        def fake_build(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise DS4SupportError("missing make")

        monkeypatch.setattr(ds4_support, "build_ds4_support_from_source", fake_build)
        settings = DS4Settings(auto_build=True)

        for expected in ("missing make", "previously failed"):
            with pytest.raises(DS4SupportError) as exc_info:
                ensure_ds4_support(
                    settings,
                    base_path=tmp_path,
                    system="Darwin",
                    machine="arm64",
                )
            assert expected in str(exc_info.value)

        assert calls == 1

    def test_ensure_ds4_support_retries_failed_auto_build_for_new_source(
        self, tmp_path, monkeypatch
    ):
        """Changing admin source overrides gets a fresh first-launch build attempt."""
        base_path = tmp_path / "base"
        calls: list[str | None] = []

        def fake_build(settings, **kwargs):
            calls.append(settings.source_repo)
            if settings.source_repo == "bad":
                raise DS4SupportError("bad source")
            destination = settings.get_support_dir(kwargs["base_path"])
            _write_complete_support_tree(destination)
            return None

        monkeypatch.setattr(ds4_support, "build_ds4_support_from_source", fake_build)
        settings = DS4Settings(auto_build=True, source_repo="bad")

        with pytest.raises(DS4SupportError) as exc_info:
            ensure_ds4_support(
                settings,
                base_path=base_path,
                system="Darwin",
                machine="arm64",
            )
        assert "bad source" in str(exc_info.value)

        settings.source_repo = "good"
        status = ensure_ds4_support(
            settings,
            base_path=base_path,
            system="Darwin",
            machine="arm64",
        )

        assert status.ready is True
        assert calls == ["bad", "good"]


class TestDS4SupportCopy:
    """Tests for copying bundled DS4 support files."""

    def test_copy_required_support_files(self, tmp_path):
        """Only required support files are copied into the destination."""
        source = tmp_path / "resources" / "ds4"
        destination = tmp_path / "support" / "ds4"
        _write_complete_support_tree(source)
        (source / "ignored.txt").write_text("ignore me")

        result = copy_ds4_support_files(source, destination)

        assert result.source_dir == source.resolve()
        assert result.destination_dir == destination.resolve()
        assert len(result.copied_files) == len(pinned_ds4_support_relative_paths())
        assert (destination / DS4_SERVER_BINARY).is_file()
        assert os.access(destination / DS4_SERVER_BINARY, os.X_OK)
        assert (destination / "metal" / "flash_attn.metal").is_file()
        assert not (destination / "ignored.txt").exists()

    def test_copy_discovers_custom_metal_files(self, tmp_path):
        """Custom support trees copy their actual Metal kernels."""
        source = tmp_path / "resources" / "ds4"
        destination = tmp_path / "support" / "ds4"
        _write_complete_support_tree(
            source,
            metal_files=("fork_attention.metal", "nested/custom_kernel.metal"),
        )

        copy_ds4_support_files(source, destination)

        assert (destination / "metal" / "fork_attention.metal").is_file()
        assert (destination / "metal" / "nested" / "custom_kernel.metal").is_file()
        assert not (destination / "metal" / "flash_attn.metal").exists()

    def test_copy_skips_existing_files_without_overwrite(self, tmp_path):
        """Existing files are preserved unless overwrite is requested."""
        source = tmp_path / "resources" / "ds4"
        destination = tmp_path / "support" / "ds4"
        _write_complete_support_tree(source)
        _write_complete_support_tree(destination)
        (destination / "README.md").write_text("keep\n")

        result = copy_ds4_support_files(source, destination)

        assert result.copied_files == ()
        assert (destination / "README.md").read_text() == "keep\n"

    def test_copy_overwrites_when_requested(self, tmp_path):
        """Explicit overwrite refreshes already-copied support files."""
        source = tmp_path / "resources" / "ds4"
        destination = tmp_path / "support" / "ds4"
        _write_complete_support_tree(source, metal_files=("fork_attention.metal",))
        _write_complete_support_tree(destination)
        (source / "README.md").write_text("new\n")
        (destination / "README.md").write_text("old\n")

        result = copy_ds4_support_files(source, destination, overwrite=True)

        assert result.copied_files
        assert (destination / "README.md").read_text() == "new\n"
        assert (destination / "metal" / "fork_attention.metal").is_file()
        assert not (destination / "metal" / "flash_attn.metal").exists()

    def test_copy_overwrite_removes_stale_manifest_when_source_has_none(self, tmp_path):
        """Manifest-less prebuilt support should not inherit old Metal metadata."""
        source = tmp_path / "resources" / "ds4"
        destination = tmp_path / "support" / "ds4"
        _write_complete_support_tree(source, metal_files=("fork_attention.metal",))
        _write_complete_support_tree(destination)
        (destination / "manifest.json").write_text(
            json.dumps({"metal_files": ["metal/flash_attn.metal"]})
        )

        copy_ds4_support_files(source, destination, overwrite=True)
        status = inspect_ds4_support(
            DS4Settings(support_dir=str(destination)),
            base_path=tmp_path,
            system="Darwin",
            machine="arm64",
        )

        assert not (destination / "manifest.json").exists()
        assert status.ready is True
        assert status.missing_files == ()

    def test_copy_rejects_incomplete_source_tree(self, tmp_path):
        """Missing bundled files fail clearly; no fetch/build is attempted."""
        source = tmp_path / "resources" / "ds4"
        source.mkdir(parents=True)
        destination = tmp_path / "support" / "ds4"

        with pytest.raises(DS4SupportError) as exc_info:
            copy_ds4_support_files(source, destination)

        assert "Bundled DS4 support files are incomplete" in str(exc_info.value)
        assert not destination.exists()

    def test_copy_rejects_incompatible_binary(self, tmp_path):
        """Bundled/staged DS4 binaries must expose current launch flags."""
        source = tmp_path / "resources" / "ds4"
        destination = tmp_path / "support" / "ds4"
        _write_complete_support_tree(source, supports_required_flags=False)

        with pytest.raises(DS4SupportError) as exc_info:
            copy_ds4_support_files(source, destination)

        assert "Bundled DS4 binary is incompatible" in str(exc_info.value)
        assert "--ssd-streaming" in str(exc_info.value)
        assert not destination.exists()
