"""
Auto-update module for oMLX macOS app.

Downloads DMG from GitHub releases, stages the new app bundle,
and performs swap + relaunch via a detached shell script.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)


class UpdateError(Exception):
    """Auto-update failure."""

    pass


class AppUpdater:
    """Handles downloading, installing, and relaunching the app."""

    STAGED_APP_NAME = ".oMLX-update.app"

    def __init__(
        self,
        dmg_url: str,
        version: str,
        on_progress: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_ready: Optional[Callable[[], None]] = None,
    ):
        self.dmg_url = dmg_url
        self.version = version
        self._on_progress = on_progress
        self._on_error = on_error
        self._on_ready = on_ready
        self._cancelled = False

    @staticmethod
    def get_app_bundle_path() -> Path:
        """Get the path of the currently running .app bundle."""
        from AppKit import NSBundle

        bundle = NSBundle.mainBundle()
        return Path(bundle.bundlePath())

    @staticmethod
    def is_writable(app_path: Path) -> bool:
        """Check if we can write to the app's parent directory."""
        return os.access(str(app_path.parent), os.W_OK)

    @staticmethod
    def cleanup_staged_app():
        """Remove leftover staged app from a previous update attempt."""
        try:
            app_path = AppUpdater.get_app_bundle_path()
            staged = app_path.parent / AppUpdater.STAGED_APP_NAME
            if staged.exists():
                shutil.rmtree(staged)
                logger.info("Cleaned up leftover staged update")
        except Exception as e:
            logger.debug(f"Staged app cleanup failed: {e}")

    def start(self):
        """Start the update process in a background thread."""
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def cancel(self):
        """Cancel the update process."""
        self._cancelled = True

    def _progress(self, msg: str):
        if self._on_progress:
            self._on_progress(msg)

    def _error(self, msg: str):
        if self._on_error:
            self._on_error(msg)

    def _run(self):
        """Background thread: download -> mount -> stage -> signal ready."""
        tmp_dir = None
        mount_point = None
        try:
            app_path = self.get_app_bundle_path()

            if not self.is_writable(app_path):
                raise UpdateError(
                    f"Cannot write to {app_path.parent}. "
                    "Please move oMLX.app to a writable location."
                )

            # Download DMG
            self._progress("Downloading update...")
            tmp_dir = Path(tempfile.mkdtemp(prefix="omlx-update-"))
            dmg_path = tmp_dir / f"oMLX-{self.version}.dmg"
            self._download_dmg(dmg_path)

            if self._cancelled:
                return

            # Mount DMG
            self._progress("Preparing update...")
            mount_point = self._mount_dmg(dmg_path)

            # Find oMLX.app in mounted volume
            new_app = self._find_app_in_volume(mount_point)

            # Stage: copy new app next to current app
            staged_app = app_path.parent / self.STAGED_APP_NAME
            if staged_app.exists():
                shutil.rmtree(staged_app)
            shutil.copytree(str(new_app), str(staged_app), symlinks=True)

            # Unmount and clean up
            self._unmount_dmg(mount_point)
            mount_point = None
            shutil.rmtree(tmp_dir)
            tmp_dir = None

            # Signal ready for swap
            if self._on_ready:
                self._on_ready()

        except UpdateError as e:
            logger.error(f"Update failed: {e}")
            self._error(str(e))
        except Exception as e:
            logger.error(f"Update failed unexpectedly: {e}", exc_info=True)
            self._error(f"Unexpected error: {e}")
        finally:
            if mount_point:
                try:
                    self._unmount_dmg(mount_point)
                except Exception:
                    pass
            if tmp_dir and tmp_dir.exists():
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass

    def _download_dmg(self, dest: Path):
        """Download DMG with progress tracking."""
        resp = requests.get(self.dmg_url, stream=True, timeout=30)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=256 * 1024):
                if self._cancelled:
                    return
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded * 100 / total)
                    self._progress(f"Downloading... {pct}%")

    def _mount_dmg(self, dmg_path: Path) -> Path:
        """Mount DMG and return mount point path."""
        result = subprocess.run(
            [
                "hdiutil",
                "attach",
                "-nobrowse",
                "-noverify",
                "-noautoopen",
                "-mountrandom",
                "/tmp",
                str(dmg_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise UpdateError(f"Failed to mount DMG: {result.stderr}")

        # Parse mount point from hdiutil output (last column of last line)
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t")
            if len(parts) >= 3:
                mount_point = parts[-1].strip()
                if mount_point and Path(mount_point).is_dir():
                    return Path(mount_point)

        raise UpdateError("Could not determine DMG mount point")

    def _unmount_dmg(self, mount_point: Path):
        """Unmount DMG volume."""
        subprocess.run(
            ["hdiutil", "detach", str(mount_point), "-force"],
            capture_output=True,
            timeout=15,
        )

    @staticmethod
    def _find_app_in_volume(mount_point: Path) -> Path:
        """Find the .app bundle inside a mounted DMG volume."""
        # Try oMLX.app first
        default = mount_point / "oMLX.app"
        if default.exists():
            return default

        # Search for any .app bundle
        for item in mount_point.iterdir():
            if item.name.endswith(".app") and item.is_dir():
                return item

        raise UpdateError("oMLX.app not found in DMG")

    @staticmethod
    def perform_swap_and_relaunch() -> bool:
        """Replace current app with staged update and relaunch.

        Spawns a detached shell script that waits for this process to exit,
        then swaps the app bundles and relaunches. Must be called right
        before app termination.

        Returns True if the swap script was spawned successfully.
        """
        app_path = AppUpdater.get_app_bundle_path()
        staged_app = app_path.parent / AppUpdater.STAGED_APP_NAME

        if not staged_app.exists():
            logger.error("Staged update not found")
            return False

        pid = os.getpid()

        # Shell script that waits for exit, swaps, removes quarantine, relaunches
        script = f"""\
#!/bin/bash
while kill -0 {pid} 2>/dev/null; do
    sleep 0.2
done
sleep 0.5
rm -rf "{app_path}"
mv "{staged_app}" "{app_path}"
xattr -rd com.apple.quarantine "{app_path}" 2>/dev/null
open "{app_path}"
"""
        subprocess.Popen(
            ["bash", "-c", script],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
