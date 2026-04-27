"""Installation method detection."""

import sys
from pathlib import Path

_APP_BUNDLE_CLI = "/Applications/oMLX.app/Contents/MacOS/omlx-cli"
_PATH_CLI = "omlx"


def is_app_bundle() -> bool:
    """Return True if running inside the macOS .app bundle."""
    here = Path(__file__).resolve()
    return ".app/Contents/" in str(here)


def is_homebrew() -> bool:
    """Return True if running inside a Homebrew-installed virtualenv."""
    prefix = sys.prefix
    return "/Cellar/" in prefix or "/homebrew/" in prefix


def get_install_method() -> str:
    """Return the installation method: 'dmg', 'homebrew', or 'pip'."""
    if is_app_bundle():
        return "dmg"
    if is_homebrew():
        return "homebrew"
    return "pip"


def get_cli_prefix() -> str:
    """Return the correct CLI command prefix for the current installation."""
    if is_app_bundle():
        return _APP_BUNDLE_CLI
    return _PATH_CLI
