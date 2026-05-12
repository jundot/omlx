# SPDX-License-Identifier: Apache-2.0
"""Tests for the ModelScope model downloader."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omlx.admin.hf_downloader import DownloadStatus, DownloadTask
from omlx.admin.ms_downloader import (
    MSDownloader,
    _extract_model_size_from_files,
    _get_ms_endpoint,
    _parse_ms_model_entry,
)


# =============================================================================
# Helper function tests
# =============================================================================


class TestGetMsEndpoint:
    """Test _get_ms_endpoint helper."""

    def test_default_endpoint(self):
        with patch.dict("os.environ", {}, clear=True):
            # When no env var and settings import fails, return default
            endpoint = _get_ms_endpoint()
            assert endpoint == "https://modelscope.cn"

    def test_env_var_override(self):
        with patch.dict(
            "os.environ", {"MODELSCOPE_DOMAIN": "https://custom.modelscope.cn/"}
        ):
            endpoint = _get_ms_endpoint()
            assert endpoint == "https://custom.modelscope.cn"

    def test_env_var_strips_trailing_slash(self):
        with patch.dict(
            "os.environ", {"MODELSCOPE_DOMAIN": "https://example.com///"}
        ):
            endpoint = _get_ms_endpoint()
            # rstrip("/") removes all trailing slashes
            assert endpoint == "https://example.com"


class TestExtractModelSizeFromFiles:
    """Test _extract_model_size_from_files helper."""

    def test_empty_list(self):
        assert _extract_model_size_from_files([]) == 0

    def test_with_size_key(self):
        files = [{"Size": 100}, {"Size": 200}, {"Size": 300}]
        assert _extract_model_size_from_files(files) == 600

    def test_with_lowercase_size_key(self):
        files = [{"size": 500}, {"size": 1000}]
        assert _extract_model_size_from_files(files) == 1500

    def test_with_missing_size(self):
        files = [{"name": "file.bin"}, {"Size": 100}]
        assert _extract_model_size_from_files(files) == 100

    def test_with_non_numeric_size(self):
        files = [{"Size": "not_a_number"}, {"Size": 100}]
        assert _extract_model_size_from_files(files) == 100


class TestParseMsModelEntry:
    """Test _parse_ms_model_entry helper."""

    def test_basic_entry(self):
        entry = {
            "Path": "qwen",
            "Name": "Qwen2.5-7B",
            "Downloads": 5000,
            "Likes": 100,
        }
        result = _parse_ms_model_entry(entry)
        assert result["repo_id"] == "qwen/Qwen2.5-7B"
        assert result["name"] == "Qwen2.5-7B"
        assert result["downloads"] == 5000
        assert result["likes"] == 100
        assert result["size"] == 0
        assert result["trending_score"] == 0

    def test_entry_with_stars(self):
        entry = {"Path": "o", "Name": "m", "Stars": 42}
        result = _parse_ms_model_entry(entry)
        assert result["likes"] == 42

    def test_entry_with_missing_fields(self):
        result = _parse_ms_model_entry({})
        assert result["repo_id"] == ""
        assert result["name"] == ""
        assert result["downloads"] == 0
        assert result["likes"] == 0

    def test_name_fallback_from_path(self):
        entry = {"Path": "owner/my-model"}
        result = _parse_ms_model_entry(entry)
        assert result["name"] == "my-model"


# =============================================================================
# MSDownloader Tests
# =============================================================================


class TestMSDownloader:
    """Test MSDownloader class."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        return tmp_path / "models"

    @pytest.fixture
    def downloader(self, model_dir):
        model_dir.mkdir(parents=True, exist_ok=True)
        return MSDownloader(model_dir=str(model_dir))

    # --- Start Download ---

    @pytest.mark.asyncio
    async def test_start_download_creates_task(self, downloader):
        with patch(
            "omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True
        ), patch(
            "omlx.admin.ms_downloader._get_ms_api"
        ) as mock_get_api, patch(
            "omlx.admin.ms_downloader.ms_snapshot_download"
        ):
            mock_api = MagicMock()
            mock_api.get_model_files.return_value = []
            mock_get_api.return_value = mock_api

            task = await downloader.start_download("owner/model")

            assert task.repo_id == "owner/model"
            assert task.status in (
                DownloadStatus.PENDING,
                DownloadStatus.DOWNLOADING,
            )
            assert task.task_id in [t["task_id"] for t in downloader.get_tasks()]

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_start_download_invalid_model_id_no_slash(self, downloader):
        with patch("omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True):
            with pytest.raises(ValueError, match="Invalid model ID"):
                await downloader.start_download("no-slash")

    @pytest.mark.asyncio
    async def test_start_download_invalid_model_id_too_many_parts(self, downloader):
        with patch("omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True):
            with pytest.raises(ValueError, match="Invalid model ID"):
                await downloader.start_download("a/b/c")

    @pytest.mark.asyncio
    async def test_start_download_strips_whitespace(self, downloader):
        with patch(
            "omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True
        ), patch(
            "omlx.admin.ms_downloader._get_ms_api"
        ) as mock_get_api, patch(
            "omlx.admin.ms_downloader.ms_snapshot_download"
        ):
            mock_api = MagicMock()
            mock_api.get_model_files.return_value = []
            mock_get_api.return_value = mock_api

            task = await downloader.start_download("  owner/model  ")
            assert task.repo_id == "owner/model"

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_start_download_sdk_not_available(self, downloader):
        with patch("omlx.admin.ms_downloader.MS_SDK_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="ModelScope SDK not installed"):
                await downloader.start_download("owner/model")

    @pytest.mark.asyncio
    async def test_start_download_duplicate(self, downloader):
        with patch(
            "omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True
        ), patch(
            "omlx.admin.ms_downloader._get_ms_api"
        ) as mock_get_api, patch(
            "omlx.admin.ms_downloader.ms_snapshot_download",
            side_effect=lambda **kwargs: asyncio.sleep(10),
        ):
            mock_api = MagicMock()
            mock_api.get_model_files.return_value = []
            mock_get_api.return_value = mock_api

            await downloader.start_download("owner/model")

            with pytest.raises(ValueError, match="already in progress"):
                await downloader.start_download("owner/model")

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_download_uses_owner_model_layout(self, model_dir):
        """ms_snapshot_download must receive local_dir under the org subfolder."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = MSDownloader(model_dir=str(model_dir))

        with patch(
            "omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True
        ), patch(
            "omlx.admin.ms_downloader._get_ms_api"
        ) as mock_get_api, patch(
            "omlx.admin.ms_downloader.ms_snapshot_download"
        ) as mock_download:
            mock_api = MagicMock()
            mock_api.get_model_files.return_value = []
            mock_get_api.return_value = mock_api

            await downloader.start_download("qwen/Qwen2.5-7B-Instruct-MLX")
            await asyncio.sleep(0.5)

            assert mock_download.called
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["local_dir"] == str(
                model_dir / "qwen" / "Qwen2.5-7B-Instruct-MLX"
            )

            await downloader.shutdown()

    # --- Cancel Download ---

    @pytest.mark.asyncio
    async def test_cancel_download(self, downloader):
        with patch(
            "omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True
        ), patch(
            "omlx.admin.ms_downloader._get_ms_api"
        ) as mock_get_api, patch(
            "omlx.admin.ms_downloader.ms_snapshot_download",
            side_effect=lambda **kwargs: time.sleep(10),
        ):
            mock_api = MagicMock()
            mock_api.get_model_files.return_value = []
            mock_get_api.return_value = mock_api

            task = await downloader.start_download("owner/model")
            # Allow task to start
            await asyncio.sleep(0.1)

            result = await downloader.cancel_download(task.task_id)
            assert result is True
            assert task.status == DownloadStatus.CANCELLED

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, downloader):
        result = await downloader.cancel_download("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_completed_task(self, downloader):
        task = DownloadTask(
            task_id="t1",
            repo_id="o/m",
            status=DownloadStatus.COMPLETED,
        )
        downloader._tasks["t1"] = task
        result = await downloader.cancel_download("t1")
        assert result is False

    # --- Remove Task ---

    def test_remove_completed_task(self, downloader):
        task = DownloadTask(
            task_id="t1",
            repo_id="o/m",
            status=DownloadStatus.COMPLETED,
        )
        downloader._tasks["t1"] = task
        assert downloader.remove_task("t1") is True
        assert "t1" not in downloader._tasks

    def test_remove_failed_task(self, downloader):
        task = DownloadTask(
            task_id="t1",
            repo_id="o/m",
            status=DownloadStatus.FAILED,
        )
        downloader._tasks["t1"] = task
        assert downloader.remove_task("t1") is True

    def test_remove_active_task_fails(self, downloader):
        task = DownloadTask(
            task_id="t1",
            repo_id="o/m",
            status=DownloadStatus.DOWNLOADING,
        )
        downloader._tasks["t1"] = task
        assert downloader.remove_task("t1") is False
        assert "t1" in downloader._tasks

    def test_remove_nonexistent_task(self, downloader):
        assert downloader.remove_task("nonexistent") is False

    # --- Retry Download ---

    @pytest.mark.asyncio
    async def test_retry_failed_download(self, downloader):
        with patch(
            "omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True
        ), patch(
            "omlx.admin.ms_downloader._get_ms_api"
        ) as mock_get_api, patch(
            "omlx.admin.ms_downloader.ms_snapshot_download"
        ):
            mock_api = MagicMock()
            mock_api.get_model_files.return_value = []
            mock_get_api.return_value = mock_api

            # Setup failed task
            old_task = DownloadTask(
                task_id="t-old",
                repo_id="owner/model",
                status=DownloadStatus.FAILED,
                retry_count=1,
            )
            downloader._tasks["t-old"] = old_task

            new_task = await downloader.retry_download("t-old")
            assert new_task.repo_id == "owner/model"
            assert new_task.retry_count == 2
            assert "t-old" not in downloader._tasks

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_retry_nonexistent_task(self, downloader):
        with pytest.raises(ValueError, match="Task not found"):
            await downloader.retry_download("nonexistent")

    @pytest.mark.asyncio
    async def test_retry_active_task_fails(self, downloader):
        task = DownloadTask(
            task_id="t1",
            repo_id="o/m",
            status=DownloadStatus.DOWNLOADING,
        )
        downloader._tasks["t1"] = task
        with pytest.raises(ValueError, match="not retryable"):
            await downloader.retry_download("t1")

    # --- Get Tasks ---

    def test_get_tasks_empty(self, downloader):
        assert downloader.get_tasks() == []

    def test_get_tasks_ordered_by_creation(self, downloader):
        task1 = DownloadTask(task_id="t1", repo_id="o/m1", created_at=100.0)
        task2 = DownloadTask(task_id="t2", repo_id="o/m2", created_at=50.0)
        task3 = DownloadTask(task_id="t3", repo_id="o/m3", created_at=150.0)
        downloader._tasks = {"t1": task1, "t2": task2, "t3": task3}

        tasks = downloader.get_tasks()
        assert [t["task_id"] for t in tasks] == ["t2", "t1", "t3"]

    # --- Model Dir ---

    def test_model_dir_property(self, downloader, model_dir):
        assert downloader.model_dir == model_dir

    def test_update_model_dir(self, downloader, tmp_path):
        new_dir = tmp_path / "new_models"
        downloader.update_model_dir(str(new_dir))
        assert downloader.model_dir == new_dir

    # --- Shutdown ---

    @pytest.mark.asyncio
    async def test_shutdown(self, downloader):
        await downloader.shutdown()
        assert len(downloader._active_tasks) == 0
        assert len(downloader._progress_tasks) == 0

    # --- Static Helper Methods ---

    def test_get_dir_size_empty(self, tmp_path):
        assert MSDownloader._get_dir_size(tmp_path) == 0

    def test_get_dir_size_with_files(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"x" * 100)
        (tmp_path / "b.bin").write_bytes(b"y" * 200)
        assert MSDownloader._get_dir_size(tmp_path) == 300

    def test_get_dir_size_nonexistent(self, tmp_path):
        assert MSDownloader._get_dir_size(tmp_path / "nonexistent") == 0

    def test_get_latest_mtime_empty(self, tmp_path):
        assert MSDownloader._get_latest_mtime(tmp_path) == 0.0

    def test_get_latest_mtime_with_files(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"x")
        mtime = MSDownloader._get_latest_mtime(tmp_path)
        assert mtime > 0

    def test_get_latest_mtime_nonexistent(self, tmp_path):
        assert MSDownloader._get_latest_mtime(tmp_path / "nonexistent") == 0.0


# =============================================================================
# MSDownloader Static API Methods Tests
# =============================================================================


class TestMSDownloaderStaticMethods:
    """Test MSDownloader static methods for API interaction."""

    @pytest.mark.asyncio
    async def test_search_models(self):
        """Test search_models using SDK list_models."""
        mock_api = MagicMock()
        mock_api.list_models.return_value = {
            "Models": [
                {
                    "Path": "mlx-community",
                    "Name": "Qwen2.5-7B-MLX",
                    "Downloads": 1000,
                    "Likes": 50,
                },
                {
                    "Path": "mlx-community",
                    "Name": "Qwen2.5-14B-MLX",
                    "Downloads": 500,
                    "Likes": 30,
                },
            ],
        }

        with patch(
            "omlx.admin.ms_downloader._get_ms_api",
            return_value=mock_api,
        ):
            result = await MSDownloader.search_models("qwen")
            assert len(result["models"]) == 2
            assert result["total"] == 2
            assert result["models"][0]["repo_id"] == "mlx-community/Qwen2.5-7B-MLX"

    @pytest.mark.asyncio
    async def test_search_models_empty(self):
        """Test search with no matching results."""
        mock_api = MagicMock()
        mock_api.list_models.return_value = {
            "Models": [
                {"Path": "mlx-community", "Name": "other-model", "Downloads": 100},
            ],
        }

        with patch(
            "omlx.admin.ms_downloader._get_ms_api",
            return_value=mock_api,
        ):
            result = await MSDownloader.search_models("nonexistent")
            assert result["models"] == []
            assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_search_models_api_error(self):
        """Test search handles SDK errors gracefully."""
        with patch(
            "omlx.admin.ms_downloader._get_ms_api",
            return_value=None,
        ):
            result = await MSDownloader.search_models("test")
            assert result["models"] == []
            assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_get_recommended_models(self):
        mock_api = MagicMock()
        mock_api.list_models.return_value = {
            "Models": [
                {
                    "Path": "qwen",
                    "Name": "model",
                    "Downloads": 1000,
                    "Likes": 50,
                },
            ],
        }

        with patch(
            "omlx.admin.ms_downloader._get_ms_api",
            return_value=mock_api,
        ):
            result = await MSDownloader.get_recommended_models(
                max_memory_bytes=32 * 1024**3,
            )
            assert "trending" in result
            assert "popular" in result
            # Verify the model is returned
            assert len(result["trending"]) == 1
            assert result["trending"][0]["repo_id"] == "qwen/model"

    @pytest.mark.asyncio
    async def test_get_recommended_models_filters_low_downloads(self):
        mock_api = MagicMock()
        mock_api.list_models.return_value = {
            "Models": [
                {"Path": "o", "Name": "low", "Downloads": 10, "Likes": 1},
                {"Path": "o", "Name": "high", "Downloads": 1000, "Likes": 50},
            ],
        }

        with patch(
            "omlx.admin.ms_downloader._get_ms_api",
            return_value=mock_api,
        ):
            result = await MSDownloader.get_recommended_models(
                max_memory_bytes=32 * 1024**3,
            )
            # Only the model with downloads >= _MIN_DOWNLOADS (50) should be included
            for category in ["trending", "popular"]:
                for m in result[category]:
                    assert m["downloads"] >= 50
            # Verify only the high-download model is returned
            assert len(result["trending"]) == 1
            assert result["trending"][0]["repo_id"] == "o/high"

    @pytest.mark.asyncio
    async def test_get_model_info(self):
        mock_api = MagicMock()
        mock_api.get_model.return_value = {
            "Name": "test-model",
            "Downloads": 5000,
            "Likes": 200,
            "Tags": ["mlx", "text-generation"],
            "Task": "text-generation",
            "Description": "A test model",
        }
        mock_api.get_model_files.return_value = [
            {"Name": "model.safetensors", "Size": 1000000},
            {"Name": "config.json", "Size": 500},
        ]

        mock_readme_response = MagicMock()
        mock_readme_response.status_code = 200
        mock_readme_response.text = "# Test Model\nThis is a test."

        with patch(
            "omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True
        ), patch(
            "omlx.admin.ms_downloader._get_ms_api",
            return_value=mock_api,
        ), patch(
            "omlx.admin.ms_downloader.requests.get",
            return_value=mock_readme_response,
        ):
            result = await MSDownloader.get_model_info("owner/test-model")
            assert result["repo_id"] == "owner/test-model"
            assert result["name"] == "test-model"
            assert result["downloads"] == 5000
            assert result["likes"] == 200
            assert result["size"] == 1000500
            assert len(result["files"]) == 2
            assert result["model_card"] == "# Test Model\nThis is a test."
            assert "mlx" in result["tags"]

    @pytest.mark.asyncio
    async def test_get_model_info_sdk_not_available(self):
        with patch("omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True), patch(
            "omlx.admin.ms_downloader._get_ms_api",
            return_value=None,
        ):
            with pytest.raises(RuntimeError, match="SDK not available"):
                await MSDownloader.get_model_info("owner/model")

    @pytest.mark.asyncio
    async def test_get_model_info_strips_yaml_front_matter(self):
        mock_api = MagicMock()
        mock_api.get_model.return_value = {"Name": "m"}
        mock_api.get_model_files.return_value = []

        mock_readme_response = MagicMock()
        mock_readme_response.status_code = 200
        mock_readme_response.text = "---\ntitle: Test\n---\n# Model Card"

        with patch(
            "omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True
        ), patch(
            "omlx.admin.ms_downloader._get_ms_api",
            return_value=mock_api,
        ), patch(
            "omlx.admin.ms_downloader.requests.get",
            return_value=mock_readme_response,
        ):
            result = await MSDownloader.get_model_info("owner/model")
            assert result["model_card"] == "# Model Card"

    @pytest.mark.asyncio
    async def test_get_model_info_tags_as_string(self):
        mock_api = MagicMock()
        mock_api.get_model.return_value = {"Name": "m", "Tags": "mlx,nlp,text"}
        mock_api.get_model_files.return_value = []

        mock_readme_response = MagicMock()
        mock_readme_response.status_code = 404
        mock_readme_response.text = ""

        with patch(
            "omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True
        ), patch(
            "omlx.admin.ms_downloader._get_ms_api",
            return_value=mock_api,
        ), patch(
            "omlx.admin.ms_downloader.requests.get",
            return_value=mock_readme_response,
        ):
            result = await MSDownloader.get_model_info("owner/model")
            assert result["tags"] == ["mlx", "nlp", "text"]
