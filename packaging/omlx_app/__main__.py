"""Entry point for oMLX menubar app.

Wraps the app launch in comprehensive error handling so that failures
(missing dependencies, PyObjC issues, etc.) are shown to the user
via a native dialog instead of being silently swallowed.
"""

import logging
import sys
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _get_crash_log_path() -> Path:
    """Get crash log file path in Application Support."""
    app_support = Path.home() / "Library" / "Application Support" / "oMLX"
    app_support.mkdir(parents=True, exist_ok=True)
    return app_support / "crash.log"


def _configure_file_logging() -> None:
    """Route menubar app logs to ~/Library/Application Support/oMLX/logs/menubar.log.

    The menubar process has no terminal to print to under LaunchServices, so
    without a file handler every logger.info/warning call is discarded. The
    file is what `omlx diagnose menubar` reads when troubleshooting hidden
    icon reports.
    """
    log_dir = Path.home() / "Library" / "Application Support" / "oMLX" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "menubar.log", maxBytes=1_000_000, backupCount=3
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    root = logging.getLogger()
    # Only install once per process.
    if not any(
        isinstance(h, RotatingFileHandler)
        and getattr(h, "baseFilename", "").endswith("menubar.log")
        for h in root.handlers
    ):
        root.addHandler(handler)
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)


def _write_crash_log(exc_text: str) -> Path:
    """Append crash info to the crash log file."""
    crash_log = _get_crash_log_path()
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(crash_log, "a") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Crash at {timestamp}\n")
            f.write(f"{'=' * 60}\n")
            f.write(exc_text)
            f.write("\n")
    except Exception:
        pass
    return crash_log


def _show_error_dialog(title: str, message: str) -> None:
    """Show error dialog via osascript (works without PyObjC).

    Uses AppleScript 'return' character for newlines in the dialog text.
    """
    import subprocess

    # Escape backslashes and double quotes for AppleScript string literal
    escaped = message.replace("\\", "\\\\").replace('"', '\\"')
    title_escaped = title.replace("\\", "\\\\").replace('"', '\\"')
    # AppleScript: use 'return' for newlines inside display dialog
    script = (
        f'set msg to "{escaped}"\n'
        f'display dialog msg buttons {{"OK"}} '
        f'default button 1 with icon stop with title "{title_escaped}"'
    )
    try:
        subprocess.run(["osascript", "-e", script], timeout=60)
    except Exception:
        pass


def _check_os_version() -> None:
    """Verify macOS version is 15.0+ (required by MLX >= 0.29.2)."""
    import platform

    mac_ver = platform.mac_ver()[0]
    if not mac_ver:
        return
    parts = mac_ver.split(".")
    try:
        major = int(parts[0])
    except (ValueError, IndexError):
        return
    if major < 15:
        _show_error_dialog(
            "macOS 15.0 Required",
            f"oMLX requires macOS 15.0 (Sequoia) or later.\n\n"
            f"Your system is running macOS {mac_ver}.\n\n"
            f"MLX >= 0.29.2 requires macOS 15.0+. "
            f"Please update your operating system to use oMLX.",
        )
        sys.exit(1)


_check_os_version()
_configure_file_logging()

try:
    from .app import main

    main()
except Exception as e:
    exc_text = traceback.format_exc()
    crash_log = _write_crash_log(exc_text)

    error_str = str(e)
    if "mlx" in error_str.lower() or "libmlx" in exc_text.lower():
        _show_error_dialog(
            "MLX Compatibility Error",
            "oMLX failed to load the MLX framework.\n\n"
            "This usually means your macOS version is too old. "
            "oMLX requires macOS 15.0 (Sequoia) or later.\n\n"
            f"Crash log: {crash_log}",
        )
    else:
        # Build a concise message for the dialog (full traceback is in crash log)
        error_line = error_str.replace("\n", " ")[:200]
        _show_error_dialog(
            "oMLX Launch Error",
            f"{type(e).__name__}: {error_line}\n\nCrash log: {crash_log}",
        )
    sys.exit(1)
