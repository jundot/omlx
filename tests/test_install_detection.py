"""Tests for installation method detection."""

from unittest.mock import patch

from omlx.utils.install import is_app_bundle, get_cli_prefix


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
