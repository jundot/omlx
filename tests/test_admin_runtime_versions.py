# SPDX-License-Identifier: Apache-2.0
"""Tests for the installed MLX runtime versions on the dashboard.

Covers ``GET /admin/api/stats``'s ``runtime`` section: the installed mlx /
mlx-lm / mlx-vlm versions reported by :mod:`omlx.utils.hardware`, and the
guarantee that a server running without the MLX stack still renders.
"""

import asyncio
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import omlx.admin.routes as admin_routes

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_TEMPLATE = ROOT / "omlx/admin/templates/dashboard/blocks/_engine_versions.html"
I18N_DIR = ROOT / "omlx/admin/i18n"

RUNTIME_KEYS = ("mlx", "mlx_lm", "mlx_vlm")


def _version_getters():
    """Map the runtime payload keys to the hardware getter each one uses."""
    return {
        "mlx": "get_mlx_version",
        "mlx_lm": "get_mlx_lm_version",
        "mlx_vlm": "get_mlx_vlm_version",
    }


def _all_getters(value):
    """Patch every hardware version getter at once.

    ``value`` is either a return value or an exception instance/class to
    raise. Patching all three together is what a machine without the MLX
    stack actually looks like; patching one at a time leaves the other two
    reporting their real versions.
    """
    is_exception = isinstance(value, BaseException) or (
        isinstance(value, type) and issubclass(value, BaseException)
    )
    kwargs = {"side_effect": value} if is_exception else {"return_value": value}

    stack = ExitStack()
    for attr in _version_getters().values():
        stack.enter_context(patch.object(admin_routes, attr, **kwargs))
    return stack


class TestGetRuntimeVersions:
    """Unit coverage for the ``runtime`` payload builder."""

    def test_reports_installed_versions(self):
        """Each MLX package is reported with its installed version string."""
        with (
            patch.object(admin_routes, "get_mlx_version", return_value="0.32.2"),
            patch.object(
                admin_routes,
                "get_mlx_lm_version",
                return_value="0.31.4.dev132+g94cdcae13",
            ),
            patch.object(admin_routes, "get_mlx_vlm_version", return_value="0.6.3"),
        ):
            runtime = admin_routes._get_runtime_versions()

        assert set(runtime) == set(RUNTIME_KEYS)
        assert runtime["mlx"] == {"name": "mlx", "version": "0.32.2"}
        assert runtime["mlx_lm"] == {
            "name": "mlx-lm",
            "version": "0.31.4.dev132+g94cdcae13",
        }
        assert runtime["mlx_vlm"] == {"name": "mlx-vlm", "version": "0.6.3"}

    def test_real_hardware_helpers_are_usable(self):
        """The getters this payload relies on exist and return strings."""
        from omlx.utils.hardware import (
            get_mlx_lm_version,
            get_mlx_version,
            get_mlx_vlm_version,
        )

        for getter in (get_mlx_version, get_mlx_lm_version, get_mlx_vlm_version):
            assert isinstance(getter(), str)

    def test_import_error_degrades_to_none(self):
        """A server without the MLX stack reports None instead of raising."""
        with _all_getters(ImportError("no mlx")):
            runtime = admin_routes._get_runtime_versions()

        assert all(entry["version"] is None for entry in runtime.values())

    def test_one_failing_getter_does_not_poison_the_others(self):
        """One broken import must not blank out the packages that did load."""
        with (
            patch.object(admin_routes, "get_mlx_version", return_value="0.32.2"),
            patch.object(
                admin_routes, "get_mlx_lm_version", side_effect=ImportError("no mlx_lm")
            ),
            patch.object(admin_routes, "get_mlx_vlm_version", return_value="0.6.3"),
        ):
            runtime = admin_routes._get_runtime_versions()

        assert runtime["mlx"]["version"] == "0.32.2"
        assert runtime["mlx_lm"]["version"] is None
        assert runtime["mlx_vlm"]["version"] == "0.6.3"

    @pytest.mark.parametrize("bad", [None, "", "   ", "Unknown", "unknown"])
    def test_sentinel_versions_become_none(self, bad):
        """hardware.py's "Unknown" sentinel and blank values mean "not installed"."""
        with _all_getters(bad):
            runtime = admin_routes._get_runtime_versions()

        assert all(entry["version"] is None for entry in runtime.values())

    def test_non_string_version_is_coerced(self):
        """A non-str ``__version__`` must not leak into the payload as-is."""
        with patch.object(admin_routes, "get_mlx_version", return_value=(0, 29)):
            runtime = admin_routes._get_runtime_versions()

        assert isinstance(runtime["mlx"]["version"], str)

    def test_very_long_version_is_preserved(self):
        """Absurdly long version strings are passed through, not truncated."""
        long_version = "9" * 5000
        with patch.object(
            admin_routes, "get_mlx_lm_version", return_value=long_version
        ):
            runtime = admin_routes._get_runtime_versions()

        assert runtime["mlx_lm"]["version"] == long_version

    def test_getter_raising_arbitrary_exception_is_contained(self):
        """Non-ImportError failures are contained too, never re-raised."""
        with patch.object(
            admin_routes,
            "get_mlx_vlm_version",
            side_effect=RuntimeError("boom"),
        ):
            runtime = admin_routes._get_runtime_versions()

        assert runtime["mlx_vlm"]["version"] is None

    def test_names_are_stable(self):
        """Display names are the pip distribution names, not the payload keys."""
        with patch.object(admin_routes, "get_mlx_version", return_value="0.32.2"):
            runtime = admin_routes._get_runtime_versions()

        assert runtime["mlx"]["name"] == "mlx"
        assert runtime["mlx_lm"]["name"] == "mlx-lm"
        assert runtime["mlx_vlm"]["name"] == "mlx-vlm"


class TestStatsPayload:
    """The ``runtime`` section is part of the /admin/api/stats payload."""

    def _stats(self):
        mock_settings = MagicMock()
        mock_settings.server.host = "127.0.0.1"
        mock_settings.server.port = 9981
        mock_settings.auth.api_key = "secret"

        mock_metrics = MagicMock()
        mock_metrics.get_snapshot.return_value = {"total_requests": 0}

        with (
            patch.object(
                admin_routes, "_get_global_settings", return_value=mock_settings
            ),
            patch("omlx.server_metrics.get_server_metrics", return_value=mock_metrics),
            patch.object(admin_routes, "_get_engine_info", return_value={}),
            patch.object(
                admin_routes, "_build_active_models_data", return_value={"models": []}
            ),
            patch.object(
                admin_routes,
                "_build_runtime_cache_observability",
                return_value={"models": []},
            ),
        ):
            return asyncio.run(admin_routes.get_server_stats(is_admin=True))

    def test_stats_payload_has_runtime_section(self):
        result = self._stats()

        assert "runtime" in result
        assert set(result["runtime"]) == set(RUNTIME_KEYS)

    def test_stats_payload_survives_missing_mlx_stack(self):
        """The whole point: a cosmetic card must never 500 the stats endpoint."""
        with _all_getters(ImportError("no mlx")):
            result = self._stats()

        assert set(result["runtime"]) == set(RUNTIME_KEYS)
        assert all(entry["version"] is None for entry in result["runtime"].values())

    def test_stats_payload_keeps_engines_section(self):
        """Adding runtime must not displace the existing engines block."""
        result = self._stats()

        assert "engines" in result


class TestEngineVersionsCard:
    """The Engine Versions card renders the runtime rows."""

    def test_template_renders_runtime_section(self):
        template = RUNTIME_TEMPLATE.read_text(encoding="utf-8")

        assert "stats.runtime" in template
        assert "status.runtime.section_label" in template
        assert "status.runtime.not_installed" in template

    def test_runtime_keys_exist_in_every_locale(self):
        keys = {"status.runtime.section_label", "status.runtime.not_installed"}
        locales = sorted(I18N_DIR.glob("*.json"))
        assert len(locales) >= 10

        for locale_path in locales:
            locale = json.loads(locale_path.read_text(encoding="utf-8"))
            missing = keys - locale.keys()
            assert not missing, f"{locale_path.name} is missing {sorted(missing)}"
            for key in keys:
                assert locale[key].strip(), f"{locale_path.name} has empty {key}"
