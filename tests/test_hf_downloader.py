# SPDX-License-Identifier: Apache-2.0
"""Tests for the HuggingFace model downloader."""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.admin.hf_downloader import DownloadStatus, DownloadTask, HFDownloader


# =============================================================================
# DownloadTask Tests
# =============================================================================


class TestDownloadTask:
    """Test DownloadTask dataclass."""

    def test_default_values(self):
        task = DownloadTask(task_id="test-id", repo_id="owner/model")
        assert task.task_id == "test-id"
        assert task.repo_id == "owner/model"
        assert task.status == DownloadStatus.PENDING
        assert task.progress == 0.0
        assert task.total_size == 0
        assert task.downloaded_size == 0
        assert task.error == ""
        assert task.started_at == 0.0
        assert task.completed_at == 0.0

    def test_to_dict(self):
        task = DownloadTask(
            task_id="abc-123",
            repo_id="mlx-community/Llama-3-8B",
            status=DownloadStatus.DOWNLOADING,
            progress=45.67,
            total_size=1000000,
            downloaded_size=456700,
            created_at=1700000000.0,
        )
        d = task.to_dict()
        assert d["task_id"] == "abc-123"
        assert d["repo_id"] == "mlx-community/Llama-3-8B"
        assert d["status"] == "downloading"
        assert d["progress"] == 45.7  # rounded to 1 decimal
        assert d["total_size"] == 1000000
        assert d["downloaded_size"] == 456700

    def test_to_dict_status_values(self):
        for status in DownloadStatus:
            task = DownloadTask(task_id="t", repo_id="o/m", status=status)
            assert task.to_dict()["status"] == status.value


# =============================================================================
# HFDownloader Tests
# =============================================================================


class TestHFDownloader:
    """Test HFDownloader class."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        return tmp_path / "models"

    @pytest.fixture
    def downloader(self, model_dir):
        model_dir.mkdir(parents=True, exist_ok=True)
        return HFDownloader(model_dir=str(model_dir))

    # --- Start Download ---

    @pytest.mark.asyncio
    async def test_start_download_creates_task(self, downloader):
        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download"
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")

            assert task.repo_id == "owner/model"
            assert task.status in (
                DownloadStatus.PENDING,
                DownloadStatus.DOWNLOADING,
            )
            assert task.task_id in [t["task_id"] for t in downloader.get_tasks()]

            # Cleanup
            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_start_download_invalid_repo_id(self, downloader):
        with pytest.raises(ValueError, match="Invalid repository ID"):
            await downloader.start_download("no-slash")

    @pytest.mark.asyncio
    async def test_start_download_invalid_repo_id_too_many_parts(self, downloader):
        with pytest.raises(ValueError, match="Invalid repository ID"):
            await downloader.start_download("a/b/c")

    @pytest.mark.asyncio
    async def test_start_download_strips_whitespace(self, downloader):
        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download"
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("  owner/model  ")
            assert task.repo_id == "owner/model"

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_start_download_duplicate(self, downloader):
        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download"
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            await downloader.start_download("owner/model")
            with pytest.raises(ValueError, match="already in progress"):
                await downloader.start_download("owner/model")

            await downloader.shutdown()

    # --- Download Success/Failure ---

    @pytest.mark.asyncio
    async def test_download_success_calls_callback(self, model_dir, tmp_path):
        model_dir.mkdir(parents=True, exist_ok=True)
        callback = AsyncMock()
        downloader = HFDownloader(
            model_dir=str(model_dir), on_complete=callback
        )

        # Create a fake model directory to simulate download
        target_dir = model_dir / "model"
        target_dir.mkdir()
        (target_dir / "config.json").write_text("{}")

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download"
        ) as mock_download:
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")

            # Wait for task to complete
            await asyncio.sleep(0.5)

            assert task.status == DownloadStatus.COMPLETED
            assert task.progress == 100.0
            callback.assert_awaited_once()

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_download_failure_sets_error(self, model_dir):
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=Exception("Network error"),
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")

            # Wait for task to fail
            await asyncio.sleep(0.5)

            assert task.status == DownloadStatus.FAILED
            assert "Network error" in task.error

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_download_repo_not_found(self, model_dir):
        from huggingface_hub.utils import RepositoryNotFoundError
        from unittest.mock import Mock

        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.headers = {}
        mock_response.url = "https://huggingface.co/api/models/owner/nonexistent"

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=RepositoryNotFoundError(
                "Not found", response=mock_response
            ),
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/nonexistent")

            await asyncio.sleep(0.5)

            assert task.status == DownloadStatus.FAILED
            assert "not found" in task.error.lower()

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_download_gated_repo(self, model_dir):
        from huggingface_hub.utils import GatedRepoError
        from unittest.mock import Mock

        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {}
        mock_response.url = "https://huggingface.co/api/models/owner/gated-model"

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=GatedRepoError(
                "Gated", response=mock_response
            ),
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/gated-model")

            await asyncio.sleep(0.5)

            assert task.status == DownloadStatus.FAILED
            assert "gated" in task.error.lower()

            await downloader.shutdown()

    # --- Cancel Download ---

    @pytest.mark.asyncio
    async def test_cancel_download(self, downloader):
        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=lambda **kwargs: time.sleep(10),
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")

            # Give it a moment to start
            await asyncio.sleep(0.2)

            success = await downloader.cancel_download(task.task_id)
            assert success is True
            assert task.status == DownloadStatus.CANCELLED

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_returns_false(self, downloader):
        result = await downloader.cancel_download("nonexistent-id")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_completed_returns_false(self, model_dir):
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download"
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")
            await asyncio.sleep(0.5)
            assert task.status == DownloadStatus.COMPLETED

            result = await downloader.cancel_download(task.task_id)
            assert result is False

            await downloader.shutdown()

    # --- Task Management ---

    def test_get_tasks_empty(self, downloader):
        assert downloader.get_tasks() == []

    @pytest.mark.asyncio
    async def test_get_tasks_returns_all(self, downloader):
        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download"
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            await downloader.start_download("owner/model-a")
            await downloader.start_download("owner/model-b")

            tasks = downloader.get_tasks()
            assert len(tasks) == 2
            repo_ids = [t["repo_id"] for t in tasks]
            assert "owner/model-a" in repo_ids
            assert "owner/model-b" in repo_ids

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_remove_completed_task(self, model_dir):
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download"
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")
            await asyncio.sleep(0.5)
            assert task.status == DownloadStatus.COMPLETED

            result = downloader.remove_task(task.task_id)
            assert result is True
            assert downloader.get_tasks() == []

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_remove_active_task_fails(self, downloader):
        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=lambda **kwargs: time.sleep(10),
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")
            await asyncio.sleep(0.2)

            result = downloader.remove_task(task.task_id)
            assert result is False

            await downloader.shutdown()

    def test_remove_nonexistent_returns_false(self, downloader):
        result = downloader.remove_task("nonexistent-id")
        assert result is False

    # --- Model Directory ---

    def test_update_model_dir(self, downloader, tmp_path):
        new_dir = tmp_path / "new_models"
        downloader.update_model_dir(str(new_dir))
        assert downloader.model_dir == new_dir

    # --- Shutdown ---

    @pytest.mark.asyncio
    async def test_shutdown_cancels_active_tasks(self, downloader):
        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=lambda **kwargs: time.sleep(10),
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")
            await asyncio.sleep(0.2)

            await downloader.shutdown()
            assert task.status == DownloadStatus.CANCELLED

    # --- Directory Size ---

    def test_get_dir_size(self, tmp_path):
        d = tmp_path / "test_model"
        d.mkdir()
        (d / "file1.bin").write_bytes(b"x" * 100)
        (d / "file2.bin").write_bytes(b"y" * 200)
        sub = d / "subdir"
        sub.mkdir()
        (sub / "file3.bin").write_bytes(b"z" * 50)

        assert HFDownloader._get_dir_size(d) == 350

    def test_get_dir_size_nonexistent(self, tmp_path):
        assert HFDownloader._get_dir_size(tmp_path / "nonexistent") == 0

    # --- Cleanup ---

    @pytest.mark.asyncio
    async def test_cleanup_partial_removes_directory(self, model_dir):
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        # Create a partial download directory
        target = model_dir / "model"
        target.mkdir()
        (target / "partial.bin").write_bytes(b"x" * 100)

        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._cleanup_partial(task)

        assert not target.exists()


# =============================================================================
# API Routes Tests
# =============================================================================


class TestHFDownloaderRoutes:
    """Test admin API endpoints for the HF downloader."""

    @pytest.fixture
    def model_dir_with_models(self, tmp_path):
        """Create a model directory with some fake models."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        # Model A
        model_a = model_dir / "model-a"
        model_a.mkdir()
        (model_a / "config.json").write_text('{"architectures": ["LlamaForCausalLM"]}')
        (model_a / "model.safetensors").write_bytes(b"x" * 1024)

        # Model B
        model_b = model_dir / "model-b"
        model_b.mkdir()
        (model_b / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
        (model_b / "model.safetensors").write_bytes(b"y" * 2048)

        # Directory without config.json (should be excluded)
        (model_dir / "not-a-model").mkdir()

        # Hidden directory (should be excluded)
        (model_dir / ".hidden").mkdir()
        (model_dir / ".hidden" / "config.json").write_text("{}")

        return model_dir

    @pytest.mark.asyncio
    async def test_list_models(self, model_dir_with_models):
        """Test the list_hf_models endpoint logic."""
        from omlx.admin.routes import list_hf_models, _get_global_settings

        # Create a mock global settings
        mock_settings = MagicMock()
        mock_settings.model.model_dir = str(model_dir_with_models)
        mock_settings.model.get_model_dirs.return_value = [model_dir_with_models]

        import omlx.admin.routes as routes_module

        original = routes_module._get_global_settings
        routes_module._get_global_settings = lambda: mock_settings

        try:
            # Mock require_admin dependency
            result = await list_hf_models(is_admin=True)
            models = result["models"]

            assert len(models) == 2
            names = [m["name"] for m in models]
            assert "model-a" in names
            assert "model-b" in names
            assert "not-a-model" not in names
            assert ".hidden" not in names

            for m in models:
                assert "size" in m
                assert "size_formatted" in m
                assert m["size"] > 0
        finally:
            routes_module._get_global_settings = original

    @pytest.mark.asyncio
    async def test_delete_model(self, model_dir_with_models):
        """Test the delete_hf_model endpoint logic."""
        from omlx.admin.routes import delete_hf_model

        import omlx.admin.routes as routes_module

        mock_settings = MagicMock()
        mock_settings.model.model_dir = str(model_dir_with_models)
        mock_settings.model.get_model_dirs.return_value = [model_dir_with_models]

        mock_pool = MagicMock()
        mock_pool.get_loaded_model_ids.return_value = []
        mock_pool._entries = {}
        mock_pool.discover_models = MagicMock()

        mock_settings_mgr = MagicMock()
        mock_settings_mgr.get_pinned_model_ids.return_value = []

        orig_settings = routes_module._get_global_settings
        orig_pool = routes_module._get_engine_pool
        orig_mgr = routes_module._get_settings_manager

        routes_module._get_global_settings = lambda: mock_settings
        routes_module._get_engine_pool = lambda: mock_pool
        routes_module._get_settings_manager = lambda: mock_settings_mgr

        try:
            assert (model_dir_with_models / "model-a").exists()

            result = await delete_hf_model(model_name="model-a", is_admin=True)
            assert result["success"] is True

            assert not (model_dir_with_models / "model-a").exists()
            mock_pool.discover_models.assert_called_once()
        finally:
            routes_module._get_global_settings = orig_settings
            routes_module._get_engine_pool = orig_pool
            routes_module._get_settings_manager = orig_mgr

    @pytest.mark.asyncio
    async def test_delete_model_path_traversal(self, model_dir_with_models):
        """Test that path traversal is blocked."""
        from fastapi import HTTPException
        from omlx.admin.routes import delete_hf_model

        import omlx.admin.routes as routes_module

        mock_settings = MagicMock()
        mock_settings.model.model_dir = str(model_dir_with_models)
        mock_settings.model.get_model_dirs.return_value = [model_dir_with_models]

        orig = routes_module._get_global_settings
        routes_module._get_global_settings = lambda: mock_settings
        routes_module._get_engine_pool = lambda: MagicMock()

        try:
            with pytest.raises(HTTPException) as exc_info:
                await delete_hf_model(
                    model_name="../../../etc/passwd", is_admin=True
                )
            # Path traversal is blocked: returns 404 (not found) since the
            # traversal path won't match any model in the directories
            assert exc_info.value.status_code in (400, 404)
        finally:
            routes_module._get_global_settings = orig

    @pytest.mark.asyncio
    async def test_delete_nonexistent_model(self, model_dir_with_models):
        """Test deleting a model that doesn't exist."""
        from fastapi import HTTPException
        from omlx.admin.routes import delete_hf_model

        import omlx.admin.routes as routes_module

        mock_settings = MagicMock()
        mock_settings.model.model_dir = str(model_dir_with_models)
        mock_settings.model.get_model_dirs.return_value = [model_dir_with_models]

        orig = routes_module._get_global_settings
        routes_module._get_global_settings = lambda: mock_settings
        routes_module._get_engine_pool = lambda: MagicMock()

        try:
            with pytest.raises(HTTPException) as exc_info:
                await delete_hf_model(
                    model_name="nonexistent-model", is_admin=True
                )
            assert exc_info.value.status_code == 404
        finally:
            routes_module._get_global_settings = orig


# =============================================================================
# Recommended Models Tests
# =============================================================================


def _make_mock_model(
    repo_id: str,
    disk_size_bytes: int = None,
    downloads: int = 0,
    likes: int = 0,
    trending_score: float = 0,
):
    """Create a mock HF model with safetensors info.

    disk_size_bytes is the desired on-disk size. We fake a BF16 parameters
    entry so that _calc_safetensors_disk_size returns exactly this value
    (BF16 = 2 bytes per parameter, so param_count = disk_size_bytes / 2).
    """
    m = MagicMock()
    m.id = repo_id
    m.downloads = downloads
    m.likes = likes
    m.trending_score = trending_score
    if disk_size_bytes is not None:
        param_count = disk_size_bytes // 2
        m.safetensors = {"parameters": {"BF16": param_count}, "total": param_count}
    else:
        m.safetensors = None
    return m


class TestGetRecommendedModels:
    """Test HFDownloader.get_recommended_models static method."""

    @pytest.mark.asyncio
    async def test_returns_trending_and_popular(self):
        """Verify both 'trending' and 'popular' keys exist in the result."""
        mock_models = [
            _make_mock_model(
                "mlx-community/model-a",
                disk_size_bytes=1_000_000_000,
                downloads=500,
                trending_score=5,
            ),
        ]

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = mock_models
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=16 * 1024**3
            )

        assert "trending" in result
        assert "popular" in result
        assert len(result["trending"]) == 1
        assert len(result["popular"]) == 1

    @pytest.mark.asyncio
    async def test_filters_by_memory(self):
        """Only models that fit in the given memory should be returned."""
        small_model = _make_mock_model(
            "mlx-community/small",
            disk_size_bytes=4 * 1024**3,  # 4 GB
            downloads=200,
        )
        large_model = _make_mock_model(
            "mlx-community/large",
            disk_size_bytes=32 * 1024**3,  # 32 GB
            downloads=200,
        )

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [small_model, large_model]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=16 * 1024**3  # 16 GB limit
            )

        # Only the small model should pass
        for category in ("trending", "popular"):
            names = [m["name"] for m in result[category]]
            assert "small" in names
            assert "large" not in names

    @pytest.mark.asyncio
    async def test_excludes_models_without_safetensors(self):
        """Models with no safetensors info should be excluded."""
        good_model = _make_mock_model(
            "mlx-community/good",
            disk_size_bytes=2 * 1024**3,
            downloads=200,
        )
        no_safetensors = _make_mock_model(
            "mlx-community/no-st",
            disk_size_bytes=None,
            downloads=200,
        )

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [good_model, no_safetensors]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=64 * 1024**3
            )

        for category in ("trending", "popular"):
            names = [m["name"] for m in result[category]]
            assert "good" in names
            assert "no-st" not in names

    @pytest.mark.asyncio
    async def test_excludes_low_download_models(self):
        """Models with fewer than 100 downloads should be excluded."""
        popular = _make_mock_model(
            "mlx-community/popular",
            disk_size_bytes=2 * 1024**3,
            downloads=500,
        )
        unpopular = _make_mock_model(
            "mlx-community/unpopular",
            disk_size_bytes=2 * 1024**3,
            downloads=50,
        )

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [popular, unpopular]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=64 * 1024**3
            )

        for category in ("trending", "popular"):
            names = [m["name"] for m in result[category]]
            assert "popular" in names
            assert "unpopular" not in names

    @pytest.mark.asyncio
    async def test_model_dict_format(self):
        """Verify returned dicts have the expected keys."""
        model = _make_mock_model(
            "mlx-community/test-model-4bit",
            disk_size_bytes=5_000_000_000,
            downloads=1234,
            likes=56,
            trending_score=3.5,
        )

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [model]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=64 * 1024**3
            )

        item = result["trending"][0]
        assert item["repo_id"] == "mlx-community/test-model-4bit"
        assert item["name"] == "test-model-4bit"
        assert item["downloads"] == 1234
        assert item["likes"] == 56
        assert item["trending_score"] == 3.5
        assert item["size"] == 5_000_000_000
        assert "GB" in item["size_formatted"]

    @pytest.mark.asyncio
    async def test_limits_to_10_per_category(self):
        """Each category should return at most 10 models."""
        models = [
            _make_mock_model(
                f"mlx-community/model-{i}",
                disk_size_bytes=1_000_000_000,
                downloads=200 + i,
                trending_score=i,
            )
            for i in range(20)
        ]

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = models
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=64 * 1024**3
            )

        assert len(result["trending"]) == 10
        assert len(result["popular"]) == 10


class TestCalcSafetensorsDiskSize:
    """Test _calc_safetensors_disk_size helper."""

    def test_bf16_only(self):
        from omlx.admin.hf_downloader import _calc_safetensors_disk_size

        st = {"parameters": {"BF16": 1_000_000}, "total": 1_000_000}
        assert _calc_safetensors_disk_size(st) == 2_000_000  # BF16 = 2 bytes

    def test_mixed_dtypes(self):
        from omlx.admin.hf_downloader import _calc_safetensors_disk_size

        st = {"parameters": {"BF16": 100, "U32": 200, "F32": 50}, "total": 350}
        # BF16: 100*2=200, U32: 200*4=800, F32: 50*4=200 → 1200
        assert _calc_safetensors_disk_size(st) == 1200

    def test_empty_parameters(self):
        from omlx.admin.hf_downloader import _calc_safetensors_disk_size

        assert _calc_safetensors_disk_size({"parameters": {}}) == 0
        assert _calc_safetensors_disk_size({}) == 0
