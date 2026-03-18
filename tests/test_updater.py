# SPDX-License-Identifier: Apache-2.0
"""
Tests for oMLX auto-update module (packaging/omlx_app/updater.py).
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent / "packaging"))
from omlx_app.updater import AppUpdater, UpdateError


class TestAppUpdater:
    """Tests for AppUpdater class."""

    def test_init(self):
        """Test AppUpdater initialization."""
        updater = AppUpdater(
            dmg_url="https://example.com/oMLX-1.0.0.dmg",
            version="1.0.0",
        )
        assert updater.dmg_url == "https://example.com/oMLX-1.0.0.dmg"
        assert updater.version == "1.0.0"
        assert updater._cancelled is False

    def test_cancel(self):
        """Test cancel sets the flag."""
        updater = AppUpdater(dmg_url="https://x.com/a.dmg", version="1.0")
        updater.cancel()
        assert updater._cancelled is True

    def test_get_app_bundle_path(self):
        """Test getting app bundle path from NSBundle."""
        mock_bundle = MagicMock()
        mock_bundle.bundlePath.return_value = "/Applications/oMLX.app"
        mock_ns_bundle = MagicMock()
        mock_ns_bundle.mainBundle.return_value = mock_bundle

        with patch.dict("sys.modules", {"AppKit": MagicMock(NSBundle=mock_ns_bundle)}):
            # Force re-import inside the method
            result = AppUpdater.get_app_bundle_path()
            assert result == Path("/Applications/oMLX.app")

    def test_is_writable_true(self, tmp_path):
        """Test is_writable returns True for writable directory."""
        app_path = tmp_path / "oMLX.app"
        app_path.mkdir()
        assert AppUpdater.is_writable(app_path) is True

    def test_is_writable_false(self):
        """Test is_writable returns False for non-writable directory."""
        # /System is not writable
        app_path = Path("/System/oMLX.app")
        assert AppUpdater.is_writable(app_path) is False

    @patch("omlx_app.updater.AppUpdater.get_app_bundle_path")
    def test_cleanup_staged_app(self, mock_path, tmp_path):
        """Test cleanup removes leftover staged app."""
        mock_path.return_value = tmp_path / "oMLX.app"
        staged = tmp_path / ".oMLX-update.app"
        staged.mkdir()
        (staged / "Contents").mkdir()

        AppUpdater.cleanup_staged_app()
        assert not staged.exists()

    @patch("omlx_app.updater.AppUpdater.get_app_bundle_path")
    def test_cleanup_staged_app_no_leftover(self, mock_path, tmp_path):
        """Test cleanup does nothing when no staged app exists."""
        mock_path.return_value = tmp_path / "oMLX.app"
        AppUpdater.cleanup_staged_app()  # Should not raise

    def test_find_app_in_volume_default(self, tmp_path):
        """Test finding oMLX.app in mounted volume."""
        app = tmp_path / "oMLX.app"
        app.mkdir()
        result = AppUpdater._find_app_in_volume(tmp_path)
        assert result == app

    def test_find_app_in_volume_other_name(self, tmp_path):
        """Test finding any .app bundle in mounted volume."""
        app = tmp_path / "SomeOther.app"
        app.mkdir()
        result = AppUpdater._find_app_in_volume(tmp_path)
        assert result == app

    def test_find_app_in_volume_not_found(self, tmp_path):
        """Test UpdateError when no .app found."""
        with pytest.raises(UpdateError, match="not found"):
            AppUpdater._find_app_in_volume(tmp_path)

    @patch("omlx_app.updater.subprocess.run")
    def test_mount_dmg_success(self, mock_run):
        """Test successful DMG mounting."""
        # Realistic hdiutil output: first lines have empty mount point,
        # only the last line has the actual mount path
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "/dev/disk5\tApple_partition_scheme\t\n"
                "/dev/disk5s1\tApple_partition_map\t\n"
                "/dev/disk5s2\tApple_HFS\t/tmp/dmg-XXXX\n"
            ),
        )

        updater = AppUpdater(dmg_url="https://x.com/a.dmg", version="1.0")

        with patch.object(Path, "is_dir", return_value=True):
            result = updater._mount_dmg(Path("/tmp/test.dmg"))

        assert result == Path("/tmp/dmg-XXXX")

    @patch("omlx_app.updater.subprocess.run")
    def test_mount_dmg_skips_empty_mount_points(self, mock_run):
        """Test that empty mount point strings (from partition lines) are skipped."""
        # Lines with empty last column should NOT match as mount points
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=(
                "/dev/disk5\tApple_partition_scheme\t\n"
                "/dev/disk5s1\tApple_partition_map\t\n"
                "/dev/disk5s2\tApple_HFS\t/tmp/dmg-REAL\n"
            ),
        )

        updater = AppUpdater(dmg_url="https://x.com/a.dmg", version="1.0")

        with patch.object(Path, "is_dir", return_value=True):
            result = updater._mount_dmg(Path("/tmp/test.dmg"))

        # Should return the actual mount point, not empty string from first lines
        assert str(result) == "/tmp/dmg-REAL"
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "hdiutil" in args
        assert "attach" in args

    @patch("omlx_app.updater.subprocess.run")
    def test_mount_dmg_failure(self, mock_run):
        """Test DMG mount failure raises UpdateError."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="hdiutil: attach failed",
        )

        updater = AppUpdater(dmg_url="https://x.com/a.dmg", version="1.0")
        with pytest.raises(UpdateError, match="Failed to mount"):
            updater._mount_dmg(Path("/tmp/test.dmg"))

    @patch("omlx_app.updater.subprocess.run")
    def test_unmount_dmg(self, mock_run):
        """Test DMG unmounting."""
        mock_run.return_value = MagicMock(returncode=0)
        updater = AppUpdater(dmg_url="https://x.com/a.dmg", version="1.0")
        updater._unmount_dmg(Path("/tmp/dmg-mount"))

        args = mock_run.call_args[0][0]
        assert "hdiutil" in args
        assert "detach" in args
        assert "-force" in args

    @patch("omlx_app.updater.requests.get")
    def test_download_dmg(self, mock_get, tmp_path):
        """Test DMG download with progress callback."""
        # Simulate streaming response
        chunk_data = b"x" * 1024
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(chunk_data))}
        mock_response.iter_content.return_value = [chunk_data]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        progress_calls = []
        updater = AppUpdater(
            dmg_url="https://x.com/a.dmg",
            version="1.0",
            on_progress=lambda msg: progress_calls.append(msg),
        )

        dest = tmp_path / "test.dmg"
        updater._download_dmg(dest)

        assert dest.exists()
        assert dest.read_bytes() == chunk_data
        assert any("100%" in msg for msg in progress_calls)

    @patch("omlx_app.updater.requests.get")
    def test_download_dmg_cancelled(self, mock_get, tmp_path):
        """Test download cancellation stops writing."""
        chunks = [b"x" * 512, b"y" * 512]
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "1024"}
        mock_response.raise_for_status = MagicMock()

        def cancel_on_second_chunk():
            for i, chunk in enumerate(chunks):
                yield chunk
                if i == 0:
                    updater.cancel()

        mock_response.iter_content.return_value = cancel_on_second_chunk()
        mock_get.return_value = mock_response

        updater = AppUpdater(dmg_url="https://x.com/a.dmg", version="1.0")
        dest = tmp_path / "test.dmg"
        updater._download_dmg(dest)

        # First chunk written, second chunk written before cancel check
        assert dest.exists()

    @patch("omlx_app.updater.subprocess.Popen")
    @patch("omlx_app.updater.AppUpdater.get_app_bundle_path")
    def test_perform_swap_and_relaunch(self, mock_path, mock_popen, tmp_path):
        """Test swap script is spawned with correct parameters."""
        app_path = tmp_path / "oMLX.app"
        app_path.mkdir()
        staged = tmp_path / ".oMLX-update.app"
        staged.mkdir()

        mock_path.return_value = app_path

        result = AppUpdater.perform_swap_and_relaunch()
        assert result is True

        # Verify Popen was called with bash -c
        mock_popen.assert_called_once()
        args = mock_popen.call_args
        assert args[0][0][0] == "bash"
        assert args[0][0][1] == "-c"
        assert args[1]["start_new_session"] is True

        # Verify script contains key operations
        script = args[0][0][2]
        assert "rm -rf" in script
        assert str(app_path) in script
        assert str(staged) in script
        assert "xattr -rd com.apple.quarantine" in script
        assert f"open \"{app_path}\"" in script

    @patch("omlx_app.updater.AppUpdater.get_app_bundle_path")
    def test_perform_swap_no_staged_app(self, mock_path, tmp_path):
        """Test swap fails gracefully when staged app doesn't exist."""
        mock_path.return_value = tmp_path / "oMLX.app"
        result = AppUpdater.perform_swap_and_relaunch()
        assert result is False

    @patch("omlx_app.updater.AppUpdater._unmount_dmg")
    @patch("omlx_app.updater.AppUpdater._mount_dmg")
    @patch("omlx_app.updater.AppUpdater._download_dmg")
    @patch("omlx_app.updater.AppUpdater.is_writable", return_value=True)
    @patch("omlx_app.updater.AppUpdater.get_app_bundle_path")
    def test_run_full_flow(
        self, mock_path, mock_writable, mock_download, mock_mount, mock_unmount, tmp_path
    ):
        """Test the full background update flow."""
        app_path = tmp_path / "oMLX.app"
        app_path.mkdir()
        mock_path.return_value = app_path

        # Set up mounted volume with app
        mount_dir = tmp_path / "mount"
        mount_dir.mkdir()
        new_app = mount_dir / "oMLX.app"
        new_app.mkdir()
        (new_app / "Contents").mkdir()
        (new_app / "Contents" / "Info.plist").write_text("test")
        mock_mount.return_value = mount_dir

        on_ready = MagicMock()
        on_error = MagicMock()

        updater = AppUpdater(
            dmg_url="https://x.com/oMLX-2.0.dmg",
            version="2.0.0",
            on_ready=on_ready,
            on_error=on_error,
        )
        updater._run()

        # Verify staged app was created
        staged = app_path.parent / ".oMLX-update.app"
        assert staged.exists()
        assert (staged / "Contents" / "Info.plist").read_text() == "test"

        on_ready.assert_called_once()
        on_error.assert_not_called()
        mock_unmount.assert_called_once_with(mount_dir)

    @patch("omlx_app.updater.AppUpdater.is_writable", return_value=False)
    @patch("omlx_app.updater.AppUpdater.get_app_bundle_path")
    def test_run_no_write_permission(self, mock_path, mock_writable, tmp_path):
        """Test error callback when write permission denied."""
        mock_path.return_value = tmp_path / "oMLX.app"

        on_error = MagicMock()
        updater = AppUpdater(
            dmg_url="https://x.com/a.dmg",
            version="1.0",
            on_error=on_error,
        )
        updater._run()

        on_error.assert_called_once()
        assert "write" in on_error.call_args[0][0].lower()


class TestVersionComparison:
    """Tests for version comparison logic used in macOS app update checking.

    Replicates _is_newer_version from omlx_app/app.py to avoid PyObjC dependency.
    """

    @staticmethod
    def _is_newer_version(latest: str, current: str) -> bool:
        try:
            from packaging.version import Version

            latest_ver = Version(latest)
            return latest_ver > Version(current) and not latest_ver.is_prerelease
        except Exception:
            return False

    def test_stable_newer(self):
        assert self._is_newer_version("0.2.19", "0.2.18") is True

    def test_dev_not_shown(self):
        assert self._is_newer_version("0.2.19.dev1", "0.2.18") is False

    def test_dev10_not_shown(self):
        assert self._is_newer_version("0.2.19.dev10", "0.2.18") is False

    def test_rc_not_shown(self):
        assert self._is_newer_version("0.2.19rc1", "0.2.18") is False

    def test_stable_from_dev_current(self):
        assert self._is_newer_version("0.2.19", "0.2.19.dev1") is True

    def test_same_version(self):
        assert self._is_newer_version("0.2.18", "0.2.18") is False

    def test_older_version(self):
        assert self._is_newer_version("0.2.17", "0.2.18") is False
