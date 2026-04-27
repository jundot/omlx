"""Tests for installation method detection."""

from unittest.mock import patch

from omlx.utils.install import (
    get_cli_prefix,
    get_install_method,
    is_app_bundle,
    is_homebrew,
)


class TestInstallDetection:
    def test_not_app_bundle_in_dev(self):
        """Dev/pip install should not detect as app bundle."""
        assert not is_app_bundle()
        assert get_cli_prefix() == "omlx"

    def test_app_bundle_detected(self):
        """Simulate running inside .app bundle."""
        fake = "/Applications/oMLX.app/Contents/Resources/omlx/utils/install.py"
        with patch("omlx.utils.install.__file__", fake):
            assert is_app_bundle()
            assert get_cli_prefix() == "/Applications/oMLX.app/Contents/MacOS/omlx-cli"

    def test_custom_app_location(self):
        """App bundle installed in non-standard location."""
        fake = "/Users/me/Apps/oMLX.app/Contents/Resources/omlx/utils/install.py"
        with patch("omlx.utils.install.__file__", fake):
            assert is_app_bundle()


class TestIsHomebrew:
    def test_not_homebrew_in_dev(self):
        """Dev/pip install should not detect as Homebrew."""
        assert not is_homebrew()

    def test_cellar_prefix(self):
        """Homebrew Cellar path detected."""
        with patch("omlx.utils.install.sys") as mock_sys:
            mock_sys.prefix = "/opt/homebrew/Cellar/omlx/0.3.0/libexec"
            assert is_homebrew()

    def test_homebrew_prefix(self):
        """Generic /homebrew/ path detected."""
        with patch("omlx.utils.install.sys") as mock_sys:
            mock_sys.prefix = "/usr/local/homebrew/opt/omlx/libexec"
            assert is_homebrew()

    def test_non_homebrew_prefix(self):
        """Regular venv should not detect as Homebrew."""
        with patch("omlx.utils.install.sys") as mock_sys:
            mock_sys.prefix = "/Users/me/.venv"
            assert not is_homebrew()


class TestGetInstallMethod:
    def test_dmg_takes_priority(self):
        """App bundle detection takes priority over Homebrew."""
        fake = "/Applications/oMLX.app/Contents/Resources/omlx/utils/install.py"
        with patch("omlx.utils.install.__file__", fake):
            assert get_install_method() == "dmg"

    def test_homebrew_detected(self):
        with patch("omlx.utils.install.sys") as mock_sys:
            mock_sys.prefix = "/opt/homebrew/Cellar/omlx/0.3.0/libexec"
            assert get_install_method() == "homebrew"

    def test_pip_default(self):
        """Default install method is pip."""
        assert get_install_method() == "pip"
