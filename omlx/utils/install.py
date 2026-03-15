"""Installation method detection."""

from pathlib import Path

_APP_BUNDLE_CLI = "/Applications/oMLX.app/Contents/MacOS/omlx-cli"
_PATH_CLI = "omlx"


def is_app_bundle() -> bool:
    """Return True if running inside the macOS .app bundle."""
    here = Path(__file__).resolve()
    return ".app/Contents/" in str(here)


def get_cli_prefix() -> str:
    """Return the correct CLI command prefix for the current installation."""
    if is_app_bundle():
        return _APP_BUNDLE_CLI
    return _PATH_CLI
