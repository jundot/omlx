# SPDX-License-Identifier: Apache-2.0
"""Tests for POST /admin/api/ssd-cache/clear's filesystem fallback.

Covers the design-doc §A4 bug: with no model loaded, the fallback swept
only the 16 hex main-KV subdirs and left GDN sidecars (the majority of the
cache by bytes) and the vision-feature cache behind, while still reporting
success.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import omlx.server  # noqa: F401 — triggers set_admin_getters
import omlx.admin.routes as admin_routes


def _run_clear():
    return asyncio.run(admin_routes.clear_ssd_cache(is_admin=True))


def _empty_pool():
    pool = MagicMock(spec=[])
    pool.get_status = MagicMock(return_value={"models": []})
    pool._entries = {}
    return pool


def _settings_for(cache_dir: Path):
    return SimpleNamespace(
        base_path=cache_dir.parent,
        cache=SimpleNamespace(get_ssd_cache_dir=lambda base_path: cache_dir),
    )


class TestSsdCacheClearFallbackSweep:
    def test_sweeps_main_blocks_sidecars_and_vision_features(self, tmp_path):
        cache_dir = tmp_path / "cache"
        main_file = cache_dir / "a" / "aaaa.safetensors"
        main_file.parent.mkdir(parents=True)
        main_file.write_bytes(b"main")

        sidecar_file = cache_dir / "_gdn_sidecars" / "deadbeef" / "sidecar.safetensors"
        sidecar_file.parent.mkdir(parents=True)
        sidecar_file.write_bytes(b"sidecar")

        vision_file = cache_dir / "vision_features" / "b" / "bbbb.safetensors"
        vision_file.parent.mkdir(parents=True)
        vision_file.write_bytes(b"vision")

        with patch.object(
            admin_routes, "_get_engine_pool", return_value=_empty_pool()
        ), patch.object(
            admin_routes, "_get_global_settings", return_value=_settings_for(cache_dir)
        ):
            result = _run_clear()

        assert result["total_deleted"] == 3
        assert not main_file.exists()
        assert not sidecar_file.exists()
        assert not vision_file.exists()

    def test_refuses_to_follow_symlinked_sidecar_digest_dir(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        escaped_file = outside / "escaped.safetensors"
        escaped_file.write_bytes(b"do not delete me")

        sidecar_root = cache_dir / "_gdn_sidecars"
        sidecar_root.mkdir()
        (sidecar_root / "deadbeef").symlink_to(outside)

        with patch.object(
            admin_routes, "_get_engine_pool", return_value=_empty_pool()
        ), patch.object(
            admin_routes, "_get_global_settings", return_value=_settings_for(cache_dir)
        ):
            result = _run_clear()

        assert result["total_deleted"] == 0
        assert escaped_file.exists()

    def test_refuses_to_follow_symlinked_vision_subdir(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        escaped_file = outside / "escaped.safetensors"
        escaped_file.write_bytes(b"do not delete me")

        vision_root = cache_dir / "vision_features"
        vision_root.mkdir()
        (vision_root / "b").symlink_to(outside)

        with patch.object(
            admin_routes, "_get_engine_pool", return_value=_empty_pool()
        ), patch.object(
            admin_routes, "_get_global_settings", return_value=_settings_for(cache_dir)
        ):
            result = _run_clear()

        assert result["total_deleted"] == 0
        assert escaped_file.exists()

    def test_no_files_present_is_a_no_op(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)

        with patch.object(
            admin_routes, "_get_engine_pool", return_value=_empty_pool()
        ), patch.object(
            admin_routes, "_get_global_settings", return_value=_settings_for(cache_dir)
        ):
            result = _run_clear()

        assert result == {"status": "ok", "total_deleted": 0}
