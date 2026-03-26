# SPDX-License-Identifier: Apache-2.0
"""Tests for the HuggingFace model downloader."""

import asyncio
import json
import shutil
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

    def test_default_retry_count(self):
        task = DownloadTask(task_id="test-id", repo_id="owner/model")
        assert task.retry_count == 0

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
        assert d["retry_count"] == 0

    def test_to_dict_retry_count(self):
        task = DownloadTask(task_id="t", repo_id="o/m", retry_count=3)
        assert task.to_dict()["retry_count"] == 3

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
    async def test_cancel_download(self, downloader, model_dir):
        # Create partial files to verify they are preserved after cancel
        target = model_dir / "model"
        target.mkdir(exist_ok=True)
        (target / "partial.bin").write_bytes(b"x" * 100)

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

            # Partial files should be preserved for resume
            assert target.exists()
            assert (target / "partial.bin").exists()

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

    @pytest.mark.asyncio
    async def test_delete_model_resource_fork_ignored(self, model_dir_with_models):
        """._* resource fork files vanishing mid-deletion should not abort the delete."""
        from omlx.admin.routes import delete_hf_model

        import omlx.admin.routes as routes_module

        mock_settings = MagicMock()
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
            # Simulate the onerror/onexc callback firing for a vanishing ._* file
            # inside the model directory (which is the real behavior of shutil.rmtree)
            original_rmtree = shutil.rmtree

            def rmtree_with_vanishing_fork(path, **kwargs):
                import sys

                handler = kwargs.get("onexc") or kwargs.get("onerror")
                if handler:
                    rf_path = str(model_dir_with_models / "model-a" / "._config.json")
                    err = FileNotFoundError(rf_path)
                    if sys.version_info >= (3, 12):
                        handler(None, rf_path, err)
                    else:
                        handler(None, rf_path, (FileNotFoundError, err, None))
                original_rmtree(path)

            with patch("shutil.rmtree", side_effect=rmtree_with_vanishing_fork):
                result = await delete_hf_model(model_name="model-a", is_admin=True)

            assert result["success"] is True
        finally:
            routes_module._get_global_settings = orig_settings
            routes_module._get_engine_pool = orig_pool
            routes_module._get_settings_manager = orig_mgr

    @pytest.mark.asyncio
    async def test_delete_model_real_error_still_raises(self, model_dir_with_models):
        """Non-resource-fork errors during deletion must propagate as HTTP 500."""
        from fastapi import HTTPException
        from omlx.admin.routes import delete_hf_model

        import omlx.admin.routes as routes_module

        mock_settings = MagicMock()
        mock_settings.model.get_model_dirs.return_value = [model_dir_with_models]

        mock_pool = MagicMock()
        mock_pool.get_loaded_model_ids.return_value = []

        orig_settings = routes_module._get_global_settings
        orig_pool = routes_module._get_engine_pool
        orig_mgr = routes_module._get_settings_manager

        routes_module._get_global_settings = lambda: mock_settings
        routes_module._get_engine_pool = lambda: mock_pool
        routes_module._get_settings_manager = lambda: None

        try:
            with patch("shutil.rmtree", side_effect=PermissionError("Access denied")):
                with pytest.raises(HTTPException) as exc_info:
                    await delete_hf_model(model_name="model-a", is_admin=True)
            assert exc_info.value.status_code == 500
        finally:
            routes_module._get_global_settings = orig_settings
            routes_module._get_engine_pool = orig_pool
            routes_module._get_settings_manager = orig_mgr

    @pytest.mark.asyncio
    async def test_delete_model_dot_underscore_in_dir_name_not_skipped(
        self, model_dir_with_models
    ):
        """FileNotFoundError on a regular file whose parent dir contains ._ should NOT be ignored."""
        from fastapi import HTTPException
        from omlx.admin.routes import delete_hf_model

        import omlx.admin.routes as routes_module

        mock_settings = MagicMock()
        mock_settings.model.get_model_dirs.return_value = [model_dir_with_models]

        mock_pool = MagicMock()
        mock_pool.get_loaded_model_ids.return_value = []

        orig_settings = routes_module._get_global_settings
        orig_pool = routes_module._get_engine_pool
        orig_mgr = routes_module._get_settings_manager

        routes_module._get_global_settings = lambda: mock_settings
        routes_module._get_engine_pool = lambda: mock_pool
        routes_module._get_settings_manager = lambda: None

        try:
            # e.g. /volumes/my._drive/model/config.json — filename is "config.json",
            # not a resource fork, so the error should propagate
            def rmtree_error_on_normal_file(path, **kwargs):
                import sys

                handler = kwargs.get("onexc") or kwargs.get("onerror")
                if handler:
                    regular_file = "/volumes/my._drive/model/config.json"
                    err = FileNotFoundError(regular_file)
                    if sys.version_info >= (3, 12):
                        handler(None, regular_file, err)
                    else:
                        handler(None, regular_file, (FileNotFoundError, err, None))

            with patch("shutil.rmtree", side_effect=rmtree_error_on_normal_file):
                with pytest.raises(HTTPException) as exc_info:
                    await delete_hf_model(model_name="model-a", is_admin=True)
            assert exc_info.value.status_code == 500
        finally:
            routes_module._get_global_settings = orig_settings
            routes_module._get_engine_pool = orig_pool
            routes_module._get_settings_manager = orig_mgr


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
    async def test_respects_result_limit(self):
        """Each category should respect the result_limit parameter."""
        models = [
            _make_mock_model(
                f"mlx-community/model-{i}",
                disk_size_bytes=1_000_000_000,
                downloads=200 + i,
                trending_score=i,
            )
            for i in range(60)
        ]

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = models
            mock_api_cls.return_value = mock_api

            # Default result_limit is 50
            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=64 * 1024**3
            )

        assert len(result["trending"]) == 50
        assert len(result["popular"]) == 50

    @pytest.mark.asyncio
    async def test_custom_result_limit(self):
        """Test custom result_limit parameter."""
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
                max_memory_bytes=64 * 1024**3,
                result_limit=5,
            )

        assert len(result["trending"]) == 5
        assert len(result["popular"]) == 5

    @pytest.mark.asyncio
    async def test_model_dict_includes_params(self):
        """Verify returned dicts include params and params_formatted."""
        model = _make_mock_model(
            "mlx-community/test-model",
            disk_size_bytes=14_000_000_000,  # BF16: 7B params
            downloads=200,
        )

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [model]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=64 * 1024**3
            )

        item = result["trending"][0]
        assert item["params"] == 7_000_000_000
        assert item["params_formatted"] == "7.0B"


# =============================================================================
# Search Models Tests
# =============================================================================


class TestSearchModels:
    """Test HFDownloader.search_models static method."""

    @pytest.mark.asyncio
    async def test_returns_models_and_total(self):
        """Verify search returns models list and total count."""
        mock_models = [
            _make_mock_model(
                "org/model-a",
                disk_size_bytes=4_000_000_000,
                downloads=500,
            ),
            _make_mock_model(
                "org/model-b",
                disk_size_bytes=8_000_000_000,
                downloads=200,
            ),
        ]

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = mock_models
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(query="model")

        assert "models" in result
        assert "total" in result
        assert len(result["models"]) == 2
        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_search_passes_mlx_filter(self):
        """Verify list_models is called with filter='mlx' to restrict results."""
        mock_models = [
            _make_mock_model("org/model-a", disk_size_bytes=4_000_000_000, downloads=500),
        ]

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = mock_models
            mock_api_cls.return_value = mock_api

            await HFDownloader.search_models(query="test", sort="trending", limit=50)

            call_kwargs = mock_api.list_models.call_args[1]
            assert call_kwargs["filter"] == "mlx"
            assert call_kwargs["search"] == "test"
            assert call_kwargs["limit"] == 50

    @pytest.mark.asyncio
    async def test_search_result_format(self):
        """Verify search results have full repo_id as name."""
        model = _make_mock_model(
            "some-org/cool-model-4bit",
            disk_size_bytes=6_000_000_000,
            downloads=1000,
            likes=42,
        )

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [model]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(query="cool")

        item = result["models"][0]
        assert item["repo_id"] == "some-org/cool-model-4bit"
        assert item["name"] == "some-org/cool-model-4bit"  # Full name for search
        assert item["downloads"] == 1000
        assert item["likes"] == 42
        assert item["params"] == 3_000_000_000  # 6GB BF16 = 3B params
        assert item["params_formatted"] == "3.0B"

    @pytest.mark.asyncio
    async def test_search_handles_no_safetensors(self):
        """Models without safetensors should still appear with size=0."""
        model = _make_mock_model("org/model", disk_size_bytes=None, downloads=100)

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [model]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(query="model")

        item = result["models"][0]
        assert item["size"] == 0
        assert item["params"] is None

    @pytest.mark.asyncio
    async def test_search_most_params_sort(self):
        """Test most_params sorting works correctly."""
        small = _make_mock_model("org/small", disk_size_bytes=2_000_000_000, downloads=100)
        large = _make_mock_model("org/large", disk_size_bytes=20_000_000_000, downloads=100)

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [small, large]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(
                query="model", sort="most_params"
            )

        # Large should come first
        assert result["models"][0]["repo_id"] == "org/large"
        assert result["models"][1]["repo_id"] == "org/small"

    @pytest.mark.asyncio
    async def test_search_least_params_sort(self):
        """Test least_params sorting works correctly."""
        small = _make_mock_model("org/small", disk_size_bytes=2_000_000_000, downloads=100)
        large = _make_mock_model("org/large", disk_size_bytes=20_000_000_000, downloads=100)

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [small, large]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(
                query="model", sort="least_params"
            )

        # Small should come first
        assert result["models"][0]["repo_id"] == "org/small"
        assert result["models"][1]["repo_id"] == "org/large"

    @pytest.mark.asyncio
    async def test_search_respects_limit(self):
        """Test limit parameter is respected."""
        models = [
            _make_mock_model(
                f"org/model-{i}", disk_size_bytes=1_000_000_000, downloads=100
            )
            for i in range(20)
        ]

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = models
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(query="model", limit=5)

        assert len(result["models"]) == 5


# =============================================================================
# Get Model Info Tests
# =============================================================================


class TestGetModelInfo:
    """Test HFDownloader.get_model_info static method."""

    @pytest.mark.asyncio
    async def test_returns_model_info(self):
        """Verify model info returns expected fields."""
        mock_info = MagicMock()
        mock_info.id = "org/test-model"
        mock_info.downloads = 5000
        mock_info.likes = 100
        mock_info.tags = ["text-generation", "mlx"]
        mock_info.pipeline_tag = "text-generation"
        mock_info.created_at = None
        mock_info.last_modified = None
        mock_info.safetensors = {"parameters": {"BF16": 7_000_000_000}, "total": 7_000_000_000}
        mock_info.card_data = None

        mock_sibling = MagicMock()
        mock_sibling.rfilename = "model.safetensors"
        mock_sibling.size = 14_000_000_000
        mock_info.siblings = [mock_sibling]

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls, \
             patch("omlx.admin.hf_downloader.hf_hub_download", side_effect=Exception("no readme")):
            mock_api = MagicMock()
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_model_info("org/test-model")

        assert result["repo_id"] == "org/test-model"
        assert result["downloads"] == 5000
        assert result["likes"] == 100
        assert result["params"] == 7_000_000_000
        assert result["params_formatted"] == "7.0B"
        assert len(result["files"]) == 1
        assert result["files"][0]["name"] == "model.safetensors"
        assert "text-generation" in result["tags"]
        assert result["model_card"] == ""  # No README available
        assert result["is_adapter"] is False

    @pytest.mark.asyncio
    async def test_detects_lora_adapter(self):
        """Verify is_adapter=True when adapter_config.json is in file list."""
        mock_info = MagicMock()
        mock_info.id = "user/lora-adapter"
        mock_info.downloads = 50
        mock_info.likes = 5
        mock_info.tags = ["lora", "mlx"]
        mock_info.pipeline_tag = "text-generation"
        mock_info.created_at = None
        mock_info.last_modified = None
        mock_info.safetensors = None
        mock_info.card_data = None

        siblings = []
        for name in ["adapter_config.json", "adapters.safetensors", "config.json"]:
            s = MagicMock()
            s.rfilename = name
            s.size = 1000
            siblings.append(s)
        mock_info.siblings = siblings

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls, \
             patch("omlx.admin.hf_downloader.hf_hub_download", side_effect=Exception("no readme")):
            mock_api = MagicMock()
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_model_info("user/lora-adapter")

        assert result["is_adapter"] is True

    @pytest.mark.asyncio
    async def test_returns_model_card(self, tmp_path):
        """Verify model card content is fetched and front matter stripped."""
        mock_info = MagicMock()
        mock_info.id = "org/test-model"
        mock_info.downloads = 100
        mock_info.likes = 10
        mock_info.tags = []
        mock_info.pipeline_tag = "text-generation"
        mock_info.created_at = None
        mock_info.last_modified = None
        mock_info.safetensors = None
        mock_info.card_data = None
        mock_info.siblings = []

        # Create a fake README file with YAML front matter
        readme_path = tmp_path / "README.md"
        readme_path.write_text("---\nlicense: mit\n---\n# My Model\n\nThis is a great model.")

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls, \
             patch("omlx.admin.hf_downloader.hf_hub_download", return_value=str(readme_path)):
            mock_api = MagicMock()
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_model_info("org/test-model")

        assert "# My Model" in result["model_card"]
        assert "This is a great model." in result["model_card"]
        assert "license: mit" not in result["model_card"]


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestFormatParamCount:
    """Test _format_param_count helper."""

    def test_billions(self):
        from omlx.admin.hf_downloader import _format_param_count

        assert _format_param_count(7_000_000_000) == "7.0B"
        assert _format_param_count(13_500_000_000) == "13.5B"

    def test_millions(self):
        from omlx.admin.hf_downloader import _format_param_count

        assert _format_param_count(125_000_000) == "125.0M"

    def test_trillions(self):
        from omlx.admin.hf_downloader import _format_param_count

        assert _format_param_count(1_500_000_000_000) == "1.5T"

    def test_small(self):
        from omlx.admin.hf_downloader import _format_param_count

        assert _format_param_count(500) == "500"


class TestGetParamCount:
    """Test _get_param_count helper."""

    def test_single_dtype(self):
        from omlx.admin.hf_downloader import _get_param_count

        assert _get_param_count({"parameters": {"BF16": 7_000_000_000}}) == 7_000_000_000

    def test_mixed_dtypes(self):
        from omlx.admin.hf_downloader import _get_param_count

        assert _get_param_count({"parameters": {"BF16": 100, "F32": 200}}) == 300

    def test_empty(self):
        from omlx.admin.hf_downloader import _get_param_count

        assert _get_param_count({"parameters": {}}) == 0
        assert _get_param_count({}) == 0


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


# =============================================================================
# Timeout Tests
# =============================================================================


class TestHFAPITimeouts:
    """Test that HF API calls respect timeouts when HuggingFace is unreachable."""

    @pytest.mark.asyncio
    async def test_get_recommended_models_timeout(self):
        """get_recommended_models should raise TimeoutError when HF is unreachable."""
        import time as time_mod

        def slow_list_models(**kwargs):
            time_mod.sleep(5)
            return []

        with patch("omlx.admin.hf_downloader._HF_API_TIMEOUT", 0.5), \
             patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.side_effect = slow_list_models
            mock_api_cls.return_value = mock_api

            with pytest.raises(asyncio.TimeoutError):
                await HFDownloader.get_recommended_models(
                    max_memory_bytes=16 * 1024**3
                )

    @pytest.mark.asyncio
    async def test_search_models_timeout(self):
        """search_models should raise TimeoutError when HF is unreachable."""
        import time as time_mod

        def slow_list_models(**kwargs):
            time_mod.sleep(5)
            return []

        with patch("omlx.admin.hf_downloader._HF_API_TIMEOUT", 0.5), \
             patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.side_effect = slow_list_models
            mock_api_cls.return_value = mock_api

            with pytest.raises(asyncio.TimeoutError):
                await HFDownloader.search_models(query="test")

    @pytest.mark.asyncio
    async def test_get_model_info_timeout(self):
        """get_model_info should raise TimeoutError when HF is unreachable."""
        import time as time_mod

        def slow_model_info(*args, **kwargs):
            time_mod.sleep(5)

        with patch("omlx.admin.hf_downloader._HF_API_TIMEOUT", 0.5), \
             patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.model_info.side_effect = slow_model_info
            mock_api_cls.return_value = mock_api

            with pytest.raises(asyncio.TimeoutError):
                await HFDownloader.get_model_info("org/model")


class TestHFEndpointPassthrough:
    """Verify that custom HF endpoint is passed to snapshot_download and hf_hub_download."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        return d

    @pytest.mark.asyncio
    async def test_snapshot_download_receives_endpoint(self, model_dir):
        """snapshot_download should receive endpoint= when mirror is configured."""
        target_dir = model_dir / "model"
        target_dir.mkdir()
        (target_dir / "config.json").write_text("{}")

        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_info.siblings = []
        mock_api.model_info.return_value = mock_info

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(mock_api, "https://hf-mirror.com"),
        ), patch("omlx.admin.hf_downloader.snapshot_download") as mock_download:
            downloader = HFDownloader(model_dir=str(model_dir))
            task = await downloader.start_download("owner/model")
            await asyncio.sleep(0.5)

            mock_download.assert_called_once()
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["endpoint"] == "https://hf-mirror.com"

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_snapshot_download_endpoint_none_without_mirror(self, model_dir):
        """snapshot_download should receive endpoint=None when no mirror is configured."""
        target_dir = model_dir / "model"
        target_dir.mkdir()
        (target_dir / "config.json").write_text("{}")

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls, \
             patch("omlx.admin.hf_downloader.snapshot_download") as mock_download:
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            downloader = HFDownloader(model_dir=str(model_dir))
            task = await downloader.start_download("owner/model")
            await asyncio.sleep(0.5)

            mock_download.assert_called_once()
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["endpoint"] is None

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_hf_hub_download_receives_endpoint(self):
        """hf_hub_download for README should receive endpoint= when mirror is configured."""
        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_info.id = "org/test-model"
        mock_info.downloads = 100
        mock_info.likes = 10
        mock_info.tags = []
        mock_info.pipeline_tag = "text-generation"
        mock_info.created_at = None
        mock_info.last_modified = None
        mock_info.safetensors = None
        mock_info.card_data = None
        mock_info.siblings = []
        mock_api.model_info.return_value = mock_info

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(mock_api, "https://hf-mirror.com"),
        ), patch("omlx.admin.hf_downloader.hf_hub_download") as mock_hf_download:
            mock_hf_download.side_effect = Exception("no readme")

            await HFDownloader.get_model_info("org/test-model")

            mock_hf_download.assert_called_once()
            call_kwargs = mock_hf_download.call_args[1]
            assert call_kwargs["endpoint"] == "https://hf-mirror.com"


# =============================================================================
# Retry Download Tests
# =============================================================================


class TestRetryDownload:
    """Test download retry functionality."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        return d

    @pytest.fixture
    def downloader(self, model_dir):
        return HFDownloader(model_dir=str(model_dir))

    @pytest.mark.asyncio
    async def test_retry_failed_download(self, downloader, model_dir):
        """Retry a failed download should create a new task with incremented retry_count."""
        # Create partial files that should be preserved
        target = model_dir / "model"
        target.mkdir()
        (target / "partial.bin").write_bytes(b"x" * 100)

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

            # Start and fail a download
            task = await downloader.start_download("owner/model")
            task.status = DownloadStatus.FAILED
            task.error = "Network error"
            old_task_id = task.task_id

            # Retry
            new_task = await downloader.retry_download(old_task_id)
            assert new_task.repo_id == "owner/model"
            assert new_task.retry_count == 1
            assert new_task.task_id != old_task_id
            # Old task should be removed
            assert old_task_id not in {t["task_id"] for t in downloader.get_tasks()}
            # Partial files should still exist
            assert (target / "partial.bin").exists()

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_retry_cancelled_download(self, downloader):
        """Retry a cancelled download should work."""
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
            task.status = DownloadStatus.CANCELLED
            old_task_id = task.task_id

            new_task = await downloader.retry_download(old_task_id)
            assert new_task.repo_id == "owner/model"
            assert new_task.retry_count == 1

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_retry_increments_count(self, downloader):
        """Multiple retries should increment retry_count."""
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
            task.status = DownloadStatus.FAILED

            task2 = await downloader.retry_download(task.task_id)
            assert task2.retry_count == 1
            task2.status = DownloadStatus.FAILED

            task3 = await downloader.retry_download(task2.task_id)
            assert task3.retry_count == 2

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_retry_active_download_raises(self, downloader):
        """Retrying an active download should raise ValueError."""
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

            with pytest.raises(ValueError, match="not retryable"):
                await downloader.retry_download(task.task_id)

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_retry_nonexistent_raises(self, downloader):
        """Retrying a nonexistent task should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            await downloader.retry_download("nonexistent-id")


# =============================================================================
# Stall Detection Tests
# =============================================================================


class TestStallDetection:
    """Test download stall detection in _poll_progress."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        return d

    @pytest.mark.asyncio
    async def test_stall_detection_marks_task_failed(self, model_dir, monkeypatch):
        """Download should be marked failed when stalled for _STALL_TIMEOUT seconds."""
        import omlx.admin.hf_downloader as dl_module

        # Use a very short stall timeout for testing
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 2)

        target = model_dir / "model"
        target.mkdir()
        # Create a file so current_size > 0 (needed to trigger stall detection)
        (target / "partial.bin").write_bytes(b"x" * 1000)

        downloader = HFDownloader(model_dir=str(model_dir))

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=lambda **kwargs: time.sleep(30),
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.safetensors = {"parameters": {"BF16": 5000}}
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")

            # Wait for stall detection to kick in (2s timeout + polling intervals)
            await asyncio.sleep(8)

            assert task.status == DownloadStatus.FAILED
            assert "stalled" in task.error.lower()

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_no_stall_when_size_zero(self, model_dir, monkeypatch):
        """Stall detection should not trigger when current_size is 0."""
        import omlx.admin.hf_downloader as dl_module

        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 1)

        # Empty directory - no files yet
        target = model_dir / "model"
        target.mkdir()

        downloader = HFDownloader(model_dir=str(model_dir))

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=lambda **kwargs: time.sleep(10),
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.safetensors = {"parameters": {"BF16": 5000}}
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")
            await asyncio.sleep(5)

            # Should still be downloading, not stalled
            assert task.status in (
                DownloadStatus.DOWNLOADING,
                DownloadStatus.COMPLETED,
            )

            await downloader.shutdown()


# =============================================================================
# Sequential Download Queue Tests
# =============================================================================


class TestSequentialDownloadQueue:
    """Test that only one download runs at a time."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        return d

    @pytest.mark.asyncio
    async def test_second_download_stays_pending(self, model_dir):
        """When two downloads are started, only the first should be DOWNLOADING."""
        downloader = HFDownloader(model_dir=str(model_dir))

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=lambda **kwargs: time.sleep(30),
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.safetensors = {"parameters": {"BF16": 5000}}
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task1 = await downloader.start_download("owner/model-a")
            task2 = await downloader.start_download("owner/model-b")

            # Give first task time to acquire semaphore
            await asyncio.sleep(1)

            assert task1.status == DownloadStatus.DOWNLOADING
            assert task2.status == DownloadStatus.PENDING

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_queued_download_starts_after_first_completes(self, model_dir):
        """Second download should start after first one finishes."""
        downloader = HFDownloader(model_dir=str(model_dir))

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
        ) as mock_download:
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.safetensors = {"parameters": {"BF16": 5000}}
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task1 = await downloader.start_download("owner/model-a")
            task2 = await downloader.start_download("owner/model-b")

            # Let both tasks finish (snapshot_download returns immediately)
            await asyncio.sleep(2)

            assert task1.status == DownloadStatus.COMPLETED
            assert task2.status == DownloadStatus.COMPLETED

            await downloader.shutdown()


# =============================================================================
# Mtime-based Activity Detection Tests
# =============================================================================


class TestMtimeActivityDetection:
    """Test that file mtime changes prevent false stall detection."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        return d

    @pytest.mark.asyncio
    async def test_mtime_prevents_false_stall(self, model_dir, monkeypatch):
        """Download should not stall if file mtimes are updating."""
        import omlx.admin.hf_downloader as dl_module

        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 3)

        target = model_dir / "model"
        target.mkdir()
        # Create a file (size won't change, but mtime will)
        test_file = target / "partial.bin"
        test_file.write_bytes(b"x" * 1000)

        downloader = HFDownloader(model_dir=str(model_dir))

        # Simulate mtime advancing on each call
        call_count = 0
        original_get_latest_mtime = HFDownloader._get_latest_mtime

        @staticmethod
        def mock_get_latest_mtime(path):
            nonlocal call_count
            call_count += 1
            # Return current time to simulate active writes
            return time.time()

        monkeypatch.setattr(
            HFDownloader, "_get_latest_mtime", mock_get_latest_mtime
        )

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=lambda **kwargs: time.sleep(30),
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.safetensors = {"parameters": {"BF16": 5000}}
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")
            # Wait longer than stall timeout
            await asyncio.sleep(8)

            # Should still be downloading because mtime keeps updating
            assert task.status == DownloadStatus.DOWNLOADING

            await downloader.shutdown()


# =============================================================================
# Etag Timeout Tests
# =============================================================================


class TestEtagTimeout:
    """Verify etag_timeout is passed to snapshot_download."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        return d

    @pytest.mark.asyncio
    async def test_etag_timeout_passed(self, model_dir):
        """snapshot_download should receive etag_timeout=30."""
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

            downloader = HFDownloader(model_dir=str(model_dir))
            await downloader.start_download("owner/model")
            await asyncio.sleep(0.5)

            mock_download.assert_called_once()
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["etag_timeout"] == 30

            await downloader.shutdown()
