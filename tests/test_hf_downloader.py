# SPDX-License-Identifier: Apache-2.0
"""Tests for the HuggingFace model downloader."""

import asyncio
import json
import logging
import os
import shutil
import stat
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from omlx._hf_download_worker import _download_without_xet
from omlx.admin import hf_downloader as hf_downloader_mod
from omlx.admin.hf_downloader import (
    DownloadStatus,
    DownloadTask,
    HFDownloader,
    _DownloadActivity,
    _DownloadCancelled,
    _calc_safetensors_disk_size,
    _histogram_has_packed_u32,
    _is_xet_transport_error,
    _make_cancellable_tqdm,
    _sum_safetensors_blob_bytes,
)


@pytest.fixture(autouse=True)
def _clear_blob_size_cache():
    hf_downloader_mod._blob_size_cache.clear()
    yield
    hf_downloader_mod._blob_size_cache.clear()


@pytest.fixture
async def blocked_worker():
    """Hold worker-thread work until teardown, even if its awaiter is cancelled."""
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    finished = asyncio.Event()
    release = threading.Event()

    def call(*args, **kwargs):
        if kwargs.get("dry_run"):
            return []
        loop.call_soon_threadsafe(started.set)
        try:
            assert release.wait(5), "Test did not release the worker"
            return []
        finally:
            loop.call_soon_threadsafe(finished.set)

    try:
        yield SimpleNamespace(call=call, started=started)
    finally:
        release.set()
        if started.is_set():
            await asyncio.wait_for(finished.wait(), timeout=5)


async def _wait_for_downloads(downloader):
    """Wait for the scheduled downloads instead of guessing their duration."""
    await asyncio.wait_for(
        asyncio.gather(*downloader._active_tasks.values()), timeout=5
    )


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
        assert task.speed_bps == 0.0
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
            speed_bps=4534000.56,
            created_at=1700000000.0,
        )
        d = task.to_dict()
        assert d["task_id"] == "abc-123"
        assert d["repo_id"] == "mlx-community/Llama-3-8B"
        assert d["status"] == "downloading"
        assert d["progress"] == 45.7  # rounded to 1 decimal
        assert d["total_size"] == 1000000
        assert d["downloaded_size"] == 456700
        assert d["speed_bps"] == 4534000.6  # rounded to 1 decimal
        assert d["retry_count"] == 0

    def test_to_dict_speed_defaults_to_zero(self):
        task = DownloadTask(task_id="t", repo_id="o/m")
        assert task.to_dict()["speed_bps"] == 0.0

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
            await _wait_for_downloads(downloader)

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
            await _wait_for_downloads(downloader)

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

            await _wait_for_downloads(downloader)

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

            await _wait_for_downloads(downloader)

            assert task.status == DownloadStatus.FAILED
            assert "gated" in task.error.lower()

            await downloader.shutdown()

    # --- Cancel Download ---

    @pytest.mark.asyncio
    async def test_cancel_download(self, downloader, model_dir, blocked_worker):
        # In-progress shards live under ._____temp and must be removed,
        # while finalized shards outside it stay for resume on retry.
        target = model_dir / "owner" / "model"
        target.mkdir(parents=True, exist_ok=True)
        (target / "model-00001-of-00002.safetensors").write_bytes(b"finalized")
        temp_dir = target / "._____temp"
        temp_dir.mkdir()
        (temp_dir / "model-00002-of-00002.safetensors").write_bytes(b"in-progress")

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=blocked_worker.call,
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")

            # Wait until the download thread is running
            await asyncio.wait_for(blocked_worker.started.wait(), timeout=5)

            active_task = downloader._active_tasks[task.task_id]
            success = await downloader.cancel_download(task.task_id)
            assert success is True
            assert task.status == DownloadStatus.CANCELLED
            await active_task

            assert not temp_dir.exists()
            assert (target / "model-00001-of-00002.safetensors").exists()
            assert target.exists()

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_cancelled_download_cleans_up_temp_dir_only(
        self, downloader, model_dir
    ):
        target = model_dir / "owner" / "model"
        target.mkdir(parents=True)
        (target / "model-00001-of-00002.safetensors").write_bytes(b"finalized")
        temp_dir = target / "._____temp"
        temp_dir.mkdir()
        (temp_dir / "model-00002-of-00002.safetensors").write_bytes(b"x")

        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_info.safetensors = {}
        mock_api.model_info.return_value = mock_info

        def fake_snapshot_download(**kwargs):
            if kwargs.get("dry_run"):
                return []
            raise asyncio.CancelledError()

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(mock_api, None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            await downloader._run_download(task.task_id, "")

        assert task.status == DownloadStatus.CANCELLED
        assert not temp_dir.exists()
        assert (target / "model-00001-of-00002.safetensors").exists()

    @pytest.mark.asyncio
    async def test_cancelled_download_logs_cleanup_failure(self, downloader, caplog):
        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_info.safetensors = {}
        mock_api.model_info.return_value = mock_info

        def fake_snapshot_download(**kwargs):
            if kwargs.get("dry_run"):
                return []
            raise asyncio.CancelledError()

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(mock_api, None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fake_snapshot_download,
        ), patch.object(
            downloader, "_cleanup_partial", side_effect=Exception("boom")
        ):
            await downloader._run_download(task.task_id, "")

        assert task.status == DownloadStatus.CANCELLED
        assert "Failed to clean up cancelled download owner/model: boom" in caplog.text

    def test_xet_not_disabled_on_import(self):
        """Importing the downloader must leave the xet fast path enabled.

        The old #1322 force-off is gone: cancellation on the xet path is now
        driven by ``abort_xet_session()`` instead of the tqdm raise, so the
        module no longer flips ``HF_HUB_DISABLE_XET``.
        """
        import huggingface_hub.constants as hc

        assert hc.HF_HUB_DISABLE_XET is False

    @pytest.mark.asyncio
    async def test_cancel_active_download_aborts_xet_session(self, downloader):
        """Cancelling the in-flight task must abort the global xet session.

        The tqdm raise never interrupts the xet path (the Rust side defers
        the exception until the transfer completes), so cancel has to reap
        the thread via abort_xet_session().
        """
        task = DownloadTask(
            task_id="t1", repo_id="owner/model", status=DownloadStatus.DOWNLOADING
        )
        downloader._tasks[task.task_id] = task
        active = asyncio.create_task(asyncio.sleep(10))
        downloader._active_tasks[task.task_id] = active

        with patch(
            "omlx.admin.hf_downloader.abort_xet_session"
        ) as mock_abort:
            assert await downloader.cancel_download(task.task_id) is True

        mock_abort.assert_called_once()
        assert task.status == DownloadStatus.CANCELLED
        with pytest.raises(asyncio.CancelledError):
            await active

    @pytest.mark.asyncio
    async def test_cancel_pending_download_does_not_abort_xet(self, downloader):
        """Cancelling a queued task must not kill another task's transfer.

        Only the DOWNLOADING task owns the semaphore and the xet session;
        aborting on a PENDING cancel would tear down the active download.
        """
        task = DownloadTask(
            task_id="t1", repo_id="owner/model", status=DownloadStatus.PENDING
        )
        downloader._tasks[task.task_id] = task

        with patch(
            "omlx.admin.hf_downloader.abort_xet_session"
        ) as mock_abort:
            assert await downloader.cancel_download(task.task_id) is True

        mock_abort.assert_not_called()
        assert task.status == DownloadStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_shutdown_aborts_xet_session(self, downloader):
        """shutdown() must reap any in-flight xet transfer thread."""
        with patch(
            "omlx.admin.hf_downloader.abort_xet_session"
        ) as mock_abort:
            await downloader.shutdown()

        mock_abort.assert_called_once()

    def test_cancellable_tqdm_raises_only_after_cancel(self):
        """The injected tqdm aborts on update() once the cancel flag is set."""
        cancelled = {"v": False}
        tqdm_cls = _make_cancellable_tqdm(lambda: cancelled["v"])
        bar = tqdm_cls(total=100, disable=True)

        # Not cancelled yet: update is a normal no-op.
        bar.update(10)

        cancelled["v"] = True
        with pytest.raises(_DownloadCancelled):
            bar.update(10)

    @pytest.mark.asyncio
    async def test_cancel_aborts_in_progress_download(self, downloader, model_dir):
        """A download cancelled mid-flight is interrupted via the tqdm callback.

        snapshot_download runs in a worker thread that can't be force-killed,
        so cancel must propagate through the per-chunk progress callback.
        """
        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_info.safetensors = {}
        mock_api.model_info.return_value = mock_info

        seen = {"tqdm_class": None}

        def fake_snapshot_download(**kwargs):
            if kwargs.get("dry_run"):
                return []
            # Simulate huggingface_hub http_get: build the progress bar and
            # call update() per chunk. The user cancels after the first chunk.
            tqdm_cls = kwargs["tqdm_class"]
            seen["tqdm_class"] = tqdm_cls
            bar = tqdm_cls(total=100, disable=True)
            bar.update(10)
            downloader._cancelled.add(task.task_id)
            bar.update(10)  # raises _DownloadCancelled
            raise AssertionError("download should have been interrupted")

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(mock_api, None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            await downloader._run_download(task.task_id, "")

        assert seen["tqdm_class"] is not None
        assert task.status == DownloadStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_shutdown_marks_tasks_cancelled_for_thread_abort(
        self, downloader, blocked_worker
    ):
        """shutdown() flags active tasks so in-flight threads abort via tqdm."""
        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=blocked_worker.call,
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")
            await asyncio.wait_for(blocked_worker.started.wait(), timeout=5)

            await downloader.shutdown()
            assert task.task_id in downloader._cancelled

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
            await _wait_for_downloads(downloader)
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
            await _wait_for_downloads(downloader)
            assert task.status == DownloadStatus.COMPLETED

            result = downloader.remove_task(task.task_id)
            assert result is True
            assert downloader.get_tasks() == []

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_remove_active_task_fails(self, downloader, blocked_worker):
        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=blocked_worker.call,
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")
            await asyncio.wait_for(blocked_worker.started.wait(), timeout=5)

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
    async def test_shutdown_cancels_active_tasks(self, downloader, blocked_worker):
        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=blocked_worker.call,
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")
            await asyncio.wait_for(blocked_worker.started.wait(), timeout=5)

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
    async def test_cleanup_partial_removes_temp_dir_only(self, model_dir):
        """Cleanup deletes the hidden ._____temp dir, finalized shards stay."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        org_dir = model_dir / "owner"
        target = org_dir / "model"
        target.mkdir(parents=True)
        (target / "model-00001-of-00002.safetensors").write_bytes(b"finalized")
        temp_dir = target / "._____temp"
        temp_dir.mkdir()
        (temp_dir / "model-00002-of-00002.safetensors").write_bytes(b"in-progress")

        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._cleanup_partial(task)

        # In-progress shards gone, finalized shards and dirs preserved
        # so snapshot_download can resume on retry.
        assert not temp_dir.exists()
        assert (target / "model-00001-of-00002.safetensors").exists()
        assert target.exists()
        assert org_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_partial_is_noop_when_no_temp_dir(self, model_dir):
        """With nothing in ._____temp, cleanup leaves the dir untouched."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        org_dir = model_dir / "owner"
        target = org_dir / "model"
        target.mkdir(parents=True)
        (target / "config.json").write_text("{}")

        sibling = org_dir / "other-model"
        sibling.mkdir()
        (sibling / "config.json").write_text("{}")

        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._cleanup_partial(task)

        assert (target / "config.json").exists()
        assert sibling.exists()
        assert org_dir.exists()

    @pytest.mark.asyncio
    async def test_download_uses_owner_model_layout(self, model_dir):
        """snapshot_download must receive local_dir under the org subfolder."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

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

            await downloader.start_download("Jundot/Qwen3.6-27B-oQ8-mtp")
            await _wait_for_downloads(downloader)

            # The actual download call (last call; the first is dry_run).
            call_kwargs = mock_download.call_args[1]
            assert call_kwargs["local_dir"] == str(
                model_dir / "Jundot" / "Qwen3.6-27B-oQ8-mtp"
            )

            await downloader.shutdown()

    @pytest.mark.asyncio
    async def test_dry_run_failure_falls_back_to_safetensors_size(self, model_dir):
        """When dry_run raises, total_size is estimated from safetensors metadata."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        task = DownloadTask(task_id="t-fallback", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        mock_api = MagicMock()
        # 7B BF16 model: 7_000_000_000 params * 2 bytes = 14_000_000_000 bytes
        mock_info = MagicMock()
        mock_info.safetensors = {
            "parameters": {"BF16": 7_000_000_000},
            "total": 7_000_000_000,
        }
        mock_info.siblings = None
        mock_api.model_info.return_value = mock_info

        def fake_snapshot_download(**kwargs):
            if kwargs.get("dry_run"):
                raise RuntimeError("dry_run not supported")
            # actual download succeeds immediately (no-op)

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(mock_api, None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            await downloader._run_download(task.task_id, "")

        # Fallback estimate: 7B BF16 params * 2 bytes/param = 14 GB
        assert task.total_size == 14_000_000_000
        assert task.status == DownloadStatus.COMPLETED
        # On completion the estimate is dropped in favor of the measured
        # dir size (nothing was written here, so 0), not the 14 GB guess.
        assert task.downloaded_size == 0

    @pytest.mark.asyncio
    async def test_run_download_model_info_omits_expand(self, model_dir):
        """files_metadata and expand together raise in huggingface_hub."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        task = DownloadTask(task_id="t-no-expand", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        mock_info = MagicMock()
        mock_info.safetensors = {
            "parameters": {"BF16": 7_000_000_000},
            "total": 7_000_000_000,
        }
        mock_info.siblings = None

        def model_info(*args, **kwargs):
            if kwargs.get("expand") and (
                kwargs.get("files_metadata") or kwargs.get("securityStatus")
            ):
                raise ValueError(
                    "`expand` cannot be used if `securityStatus` or "
                    "`files_metadata` are set."
                )
            return mock_info

        mock_api = MagicMock()
        mock_api.model_info.side_effect = model_info
        snapshot_calls = []

        def fake_snapshot_download(**kwargs):
            snapshot_calls.append(kwargs)
            if kwargs.get("dry_run"):
                raise RuntimeError("dry_run not supported")

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(mock_api, None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            await downloader._run_download(task.task_id, "")

        info_kwargs = mock_api.model_info.call_args.kwargs
        assert info_kwargs.get("files_metadata") is True
        assert "expand" not in info_kwargs
        assert snapshot_calls
        assert snapshot_calls[0]["ignore_patterns"] == [
            "*.bin",
            "original/**",
            "consolidated.*.pth",
        ]
        assert task.status == DownloadStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_dry_run_failure_u32_uses_sibling_blob_size(self, model_dir):
        """U32 histograms must not be billed at 4 bytes/param for the estimate."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        task = DownloadTask(task_id="t-u32-fallback", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_info.safetensors = {
            "parameters": {"U32": 25_235_685_376, "BF16": 570_250_830},
            "total": 25_805_936_206,
        }
        weight = MagicMock()
        weight.rfilename = "model.safetensors"
        weight.size = 15_400_000_000
        mock_info.siblings = [weight]
        mock_api.model_info.return_value = mock_info

        def fake_snapshot_download(**kwargs):
            if kwargs.get("dry_run"):
                raise RuntimeError("dry_run not supported")

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(mock_api, None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            await downloader._run_download(task.task_id, "")

        assert task.total_size == 15_400_000_000
        assert task.status == DownloadStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_dry_run_failure_no_safetensors_leaves_total_size_zero(
        self, model_dir
    ):
        """When dry_run raises and model_info has no safetensors, total_size stays 0."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        task = DownloadTask(task_id="t-no-st", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_info.safetensors = None
        mock_api.model_info.return_value = mock_info

        def fake_snapshot_download(**kwargs):
            if kwargs.get("dry_run"):
                raise RuntimeError("dry_run not supported")

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(mock_api, None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            await downloader._run_download(task.task_id, "")

        # The download itself must still proceed and complete; only the
        # progress denominator is unavailable. Pinning status/error here
        # keeps this test from passing vacuously if the fallback handler
        # ever raised (which would set FAILED while total_size stays 0).
        assert task.total_size == 0
        assert task.status == DownloadStatus.COMPLETED
        assert task.error == ""

    @pytest.mark.asyncio
    async def test_malformed_safetensors_metadata_does_not_fail_download(
        self, model_dir
    ):
        """A non-int parameters count must not escalate to a FAILED task."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        task = DownloadTask(task_id="t-malformed", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        mock_api = MagicMock()
        mock_info = MagicMock()
        # Malformed count: the size estimate raises TypeError internally
        mock_info.safetensors = {"parameters": {"BF16": None}}
        mock_api.model_info.return_value = mock_info

        def fake_snapshot_download(**kwargs):
            if kwargs.get("dry_run"):
                raise RuntimeError("dry_run not supported")

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(mock_api, None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            await downloader._run_download(task.task_id, "")

        # The bad estimate degrades to no estimate; the download proceeds.
        assert task.total_size == 0
        assert task.status == DownloadStatus.COMPLETED
        assert task.error == ""


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

        # Mixed-case models to verify case-insensitive sort: "Zebra-Model" must sort after "apple-model".
        model_z = model_dir / "Zebra-Model"
        model_z.mkdir()
        (model_z / "config.json").write_text('{"architectures": ["TestZ"]}')
        (model_z / "model.safetensors").write_bytes(b"z" * 512)

        model_apple = model_dir / "apple-model"
        model_apple.mkdir()
        (model_apple / "config.json").write_text('{"architectures": ["TestA"]}')
        (model_apple / "model.safetensors").write_bytes(b"a" * 256)

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

        nested_model = (
            model_dir_with_models / "deepsweet" / "Qwen3.6-27B-MLX-oQ5-FP16"
        )
        nested_model.mkdir(parents=True)
        (nested_model / "config.json").write_text(
            '{"architectures": ["Qwen2ForCausalLM"]}'
        )
        (nested_model / "model.safetensors").write_bytes(b"q" * 4096)

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

            assert len(models) == 5
            names = [m["name"] for m in models]
            assert "model-a" in names
            assert "model-b" in names
            assert "Zebra-Model" in names
            assert "apple-model" in names
            assert "Qwen3.6-27B-MLX-oQ5-FP16" in names
            assert "not-a-model" not in names
            assert ".hidden" not in names

            display_names = {m["name"]: m["display_name"] for m in models}
            assert (
                display_names["Qwen3.6-27B-MLX-oQ5-FP16"]
                == "deepsweet/Qwen3.6-27B-MLX-oQ5-FP16"
            )
            assert display_names["model-a"] == "model-a"

            for m in models:
                assert "size" in m
                assert "size_formatted" in m
                assert m["size"] > 0

            # Models must be returned case-insensitive ascending by display name.
            displays = [m["display_name"] for m in models]
            expected = sorted(displays, key=str.lower)
            assert displays == expected, (
                f"Expected case-insensitive ascending order. "
                f"Got {displays}, expected {expected}"
            )
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
            # Deleted model's settings (alias etc.) must be released (issue #1321)
            mock_settings_mgr.delete_settings.assert_called_once_with("model-a")
        finally:
            routes_module._get_global_settings = orig_settings
            routes_module._get_engine_pool = orig_pool
            routes_module._get_settings_manager = orig_mgr

    @pytest.mark.asyncio
    async def test_delete_model_organized_drops_empty_org_folder(self, tmp_path):
        """Deleting the last model in an org folder should drop the empty org dir."""
        from omlx.admin.routes import delete_hf_model

        import omlx.admin.routes as routes_module

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        org_dir = model_dir / "Jundot"
        model_path = org_dir / "Qwen-only-child"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text(
            '{"architectures": ["Qwen2ForCausalLM"]}'
        )
        (model_path / "model.safetensors").write_bytes(b"x" * 8)

        mock_settings = MagicMock()
        mock_settings.model.get_model_dirs.return_value = [model_dir]

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
            result = await delete_hf_model(
                model_name="Qwen-only-child", is_admin=True
            )
            assert result["success"] is True
            assert not model_path.exists()
            assert not org_dir.exists()
        finally:
            routes_module._get_global_settings = orig_settings
            routes_module._get_engine_pool = orig_pool
            routes_module._get_settings_manager = orig_mgr

    @pytest.mark.asyncio
    async def test_delete_model_organized_keeps_org_with_siblings(self, tmp_path):
        """Deleting one model in an org folder should keep the org dir if siblings remain."""
        from omlx.admin.routes import delete_hf_model

        import omlx.admin.routes as routes_module

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        org_dir = model_dir / "Jundot"
        org_dir.mkdir()

        target = org_dir / "Qwen-to-delete"
        target.mkdir()
        (target / "config.json").write_text(
            '{"architectures": ["Qwen2ForCausalLM"]}'
        )
        (target / "model.safetensors").write_bytes(b"x" * 8)

        sibling = org_dir / "Qwen-keeper"
        sibling.mkdir()
        (sibling / "config.json").write_text(
            '{"architectures": ["Qwen2ForCausalLM"]}'
        )

        mock_settings = MagicMock()
        mock_settings.model.get_model_dirs.return_value = [model_dir]

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
            result = await delete_hf_model(
                model_name="Qwen-to-delete", is_admin=True
            )
            assert result["success"] is True
            assert not target.exists()
            assert org_dir.exists()
            assert sibling.exists()
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
    m.siblings = None
    if disk_size_bytes is not None:
        param_count = disk_size_bytes // 2
        m.safetensors = {"parameters": {"BF16": param_count}, "total": param_count}
    else:
        m.safetensors = None
    return m


def _make_mock_u32_model(
    repo_id: str,
    *,
    downloads: int = 200,
    likes: int = 0,
    trending_score: float = 0,
    u32_count: int = 25_235_685_376,
    bf16_count: int = 570_250_830,
    sibling_bytes: int | None = None,
):
    """HF list row for a U32-packed MLX quant (logical param counts under U32)."""
    m = MagicMock()
    m.id = repo_id
    m.downloads = downloads
    m.likes = likes
    m.trending_score = trending_score
    total = u32_count + bf16_count
    m.safetensors = {
        "parameters": {"U32": u32_count, "BF16": bf16_count},
        "total": total,
    }
    if sibling_bytes is None:
        m.siblings = None
    else:
        weight = MagicMock()
        weight.rfilename = "model.safetensors"
        weight.size = sibling_bytes
        m.siblings = [weight]
    return m


def _blob_info(repo_id: str, size: int, safetensors=None):
    """model_info(files_metadata=True) result with one weight shard."""
    info = MagicMock()
    info.id = repo_id
    info.safetensors = safetensors
    sibling = MagicMock()
    sibling.rfilename = "model.safetensors"
    sibling.size = size
    extra = MagicMock()
    extra.rfilename = "tokenizer.json"
    extra.size = 1_000_000
    info.siblings = [sibling, extra]
    return info


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

        mock_api.model_info.assert_not_called()
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
        mock_api.model_info.assert_not_called()

    @pytest.mark.asyncio
    async def test_u32_quant_uses_blob_size_not_dtype_histogram(self):
        """U32-packed 4-bit repos must not be billed at 4 bytes/param (#3401)."""
        blob_bytes = 15_400_000_000
        model = _make_mock_u32_model(
            "mlx-community/gemma-4-26B-A4B-it-4bit",
            downloads=500,
            trending_score=5,
        )
        inflated = _calc_safetensors_disk_size(model.safetensors)
        assert inflated > 90 * 1024**3

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [model]
            mock_api.model_info.return_value = _blob_info(
                model.id, blob_bytes, safetensors=model.safetensors
            )
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=96 * 1024**3
            )

        mock_api.model_info.assert_called()
        assert mock_api.model_info.call_args.kwargs.get("files_metadata") is True
        item = result["trending"][0]
        assert item["size"] == blob_bytes
        assert item["size"] != inflated
        assert "GB" in item["size_formatted"]

    @pytest.mark.asyncio
    async def test_u32_quant_skips_model_info_when_siblings_have_sizes(self):
        blob_bytes = 15_400_000_000
        model = _make_mock_u32_model(
            "mlx-community/gemma-4-26B-A4B-it-4bit",
            downloads=500,
            sibling_bytes=blob_bytes,
        )

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [model]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=96 * 1024**3
            )

        mock_api.model_info.assert_not_called()
        assert result["trending"][0]["size"] == blob_bytes

    @pytest.mark.asyncio
    async def test_u32_blob_fetch_failure_excludes_from_recommended(self):
        """Unknown size must not appear on Recommended (memory-fit list)."""
        model = _make_mock_u32_model(
            "mlx-community/gemma-4-26B-A4B-it-4bit",
            downloads=500,
        )

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [model]
            mock_api.model_info.side_effect = RuntimeError("hub down")
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=16 * 1024**3
            )

        assert result["trending"] == []
        assert result["popular"] == []

    @pytest.mark.asyncio
    async def test_malformed_histogram_does_not_fail_recommended(self):
        """A None dtype count must not 500 the Recommended listing."""
        bad = _make_mock_model(
            "mlx-community/broken",
            disk_size_bytes=2 * 1024**3,
            downloads=200,
        )
        bad.safetensors = {"parameters": {"BF16": None}, "total": None}
        good = _make_mock_model(
            "mlx-community/ok",
            disk_size_bytes=2 * 1024**3,
            downloads=200,
        )

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [bad, good]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=64 * 1024**3
            )

        names = [m["name"] for m in result["trending"]]
        assert "ok" in names
        assert "broken" not in names


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
        mock_api.model_info.assert_not_called()

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

    @pytest.mark.asyncio
    async def test_search_largest_sort(self):
        """Test largest sorting works correctly."""
        small = _make_mock_model("org/small", disk_size_bytes=2_000_000_000, downloads=100)
        large = _make_mock_model("org/large", disk_size_bytes=20_000_000_000, downloads=100)

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [small, large]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(
                query="model", sort="largest"
            )

        # Large should come first
        assert result["models"][0]["repo_id"] == "org/large"
        assert result["models"][1]["repo_id"] == "org/small"

    @pytest.mark.asyncio
    async def test_search_smallest_sort(self):
        """Test smallest sorting works correctly."""
        small = _make_mock_model("org/small", disk_size_bytes=2_000_000_000, downloads=100)
        large = _make_mock_model("org/large", disk_size_bytes=20_000_000_000, downloads=100)

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [small, large]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(
                query="model", sort="smallest"
            )

        # Small should come first
        assert result["models"][0]["repo_id"] == "org/small"
        assert result["models"][1]["repo_id"] == "org/large"

    @pytest.mark.asyncio
    async def test_search_sort_by_size(self):
        """Test sort_by_size parameter works correctly."""
        small = _make_mock_model("org/small", disk_size_bytes=2_000_000_000, downloads=100)
        large = _make_mock_model("org/large", disk_size_bytes=20_000_000_000, downloads=100)

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [small, large]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(
                query="model",
                sort="downloads",  # base sort
                sort_by_size=True,
                sort_ascending=True,  # smallest first
            )

        # Small should come first when ascending
        assert result["models"][0]["repo_id"] == "org/small"

    @pytest.mark.asyncio
    async def test_search_filter_by_min_max_params(self):
        """Test filtering by parameter count range."""
        small = _make_mock_model("org/small", disk_size_bytes=4_000_000_000, downloads=100)
        medium = _make_mock_model("org/medium", disk_size_bytes=14_000_000_000, downloads=100)
        large = _make_mock_model("org/large", disk_size_bytes=28_000_000_000, downloads=100)

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [small, medium, large]
            mock_api_cls.return_value = mock_api

            # Filter: 3B-8B params (BF16: 4GB=2B, 14GB=7B, 28GB=14B)
            result = await HFDownloader.search_models(
                query="model",
                min_params=3_000_000_000,
                max_params=8_000_000_000,
            )

        # Only medium model should be included
        assert len(result["models"]) == 1
        assert result["models"][0]["repo_id"] == "org/medium"

    @pytest.mark.asyncio
    async def test_search_filter_by_min_max_size(self):
        """Test filtering by model size range."""
        small = _make_mock_model("org/small", disk_size_bytes=2_000_000_000, downloads=100)
        medium = _make_mock_model("org/medium", disk_size_bytes=8_000_000_000, downloads=100)
        large = _make_mock_model("org/large", disk_size_bytes=20_000_000_000, downloads=100)

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [small, medium, large]
            mock_api_cls.return_value = mock_api

            # Filter: 5GB-15GB
            result = await HFDownloader.search_models(
                query="model",
                min_size=5_000_000_000,
                max_size=15_000_000_000,
            )

        # Only medium model should be included
        assert len(result["models"]) == 1
        assert result["models"][0]["repo_id"] == "org/medium"

    @pytest.mark.asyncio
    async def test_search_u32_quant_uses_blob_size(self):
        blob_bytes = 15_400_000_000
        model = _make_mock_u32_model("mlx-community/gemma-4-26B-A4B-it-4bit")

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [model]
            mock_api.model_info.return_value = _blob_info(
                model.id, blob_bytes, safetensors=model.safetensors
            )
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(query="gemma")

        mock_api.model_info.assert_called()
        item = result["models"][0]
        assert item["size"] == blob_bytes
        assert item["size"] < _calc_safetensors_disk_size(model.safetensors)

    @pytest.mark.asyncio
    async def test_search_skips_blob_fetch_for_param_filtered_u32(self):
        """min/max params run before model_info so oversize U32 rows stay off Hub."""
        huge = _make_mock_u32_model("mlx-community/huge-u32")
        small = _make_mock_model(
            "org/small-bf16", disk_size_bytes=2_000_000_000, downloads=100
        )

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [huge, small]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(
                query="model",
                max_params=8_000_000_000,
            )

        mock_api.model_info.assert_not_called()
        assert [m["repo_id"] for m in result["models"]] == ["org/small-bf16"]

    @pytest.mark.asyncio
    async def test_search_skips_blob_fetch_when_param_count_is_zero(self):
        """0 from a malformed histogram is unknown, not a size that passes max_params."""
        model = _make_mock_u32_model("mlx-community/unknown-u32")
        model.safetensors = {
            "parameters": {"U32": 25_235_685_376, "BF16": None},
            "total": None,
        }

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [model]
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(
                query="gemma",
                max_params=8_000_000_000,
            )

        mock_api.model_info.assert_not_called()
        assert result["models"] == []

    @pytest.mark.asyncio
    async def test_search_reuses_cached_blob_size(self):
        blob_bytes = 15_400_000_000
        model = _make_mock_u32_model("mlx-community/gemma-4-26B-A4B-it-4bit")
        info = _blob_info(model.id, blob_bytes, safetensors=model.safetensors)

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = [model]
            mock_api.model_info.return_value = info
            mock_api_cls.return_value = mock_api

            first = await HFDownloader.search_models(query="gemma")
            second = await HFDownloader.search_models(query="gemma")

        assert mock_api.model_info.call_count == 1
        assert first["models"][0]["size"] == blob_bytes
        assert second["models"][0]["size"] == blob_bytes


# =============================================================================
# Stale Token Fallback Tests
# =============================================================================


def _make_401_error() -> HfHubHTTPError:
    """Build the 401 the Hub returns for a stale stored token (#2276, #2310)."""
    request = httpx.Request("GET", "https://huggingface.co/api/models")
    response = httpx.Response(401, request=request)
    return HfHubHTTPError(
        "Client error '401 Unauthorized' for url "
        "'https://huggingface.co/api/models': OAuth token signature "
        "verification failed",
        response=response,
    )


def _stale_token_list_models(models):
    """list_models double that rejects the implicit token but allows anonymous."""

    def side_effect(**kwargs):
        if kwargs.get("token") is not False:
            raise _make_401_error()
        return models

    return side_effect


class TestStaleTokenFallback:
    """Browse calls must survive a stale stored HF token (#2276, #2310)."""

    @pytest.mark.asyncio
    async def test_search_retries_anonymously_on_401(self):
        """A 401 from the stored token retries with token=False and flags it."""
        models = [
            _make_mock_model("org/model-a", disk_size_bytes=4_000_000_000, downloads=500),
        ]

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.side_effect = _stale_token_list_models(models)
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(query="model")

        assert result["total"] == 1
        assert result["hf_token_invalid"] is True
        assert mock_api.list_models.call_count == 2
        assert mock_api.list_models.call_args[1]["token"] is False

    @pytest.mark.asyncio
    async def test_search_valid_token_not_flagged(self):
        """The flag stays False when the listing succeeds first try."""
        models = [
            _make_mock_model("org/model-a", disk_size_bytes=4_000_000_000, downloads=500),
        ]

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = models
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(query="model")

        assert result["hf_token_invalid"] is False
        assert mock_api.list_models.call_count == 1

    @pytest.mark.asyncio
    async def test_search_non_401_propagates(self):
        """Only 401 triggers the anonymous retry; other HTTP errors raise."""
        request = httpx.Request("GET", "https://huggingface.co/api/models")
        response = httpx.Response(503, request=request)
        error = HfHubHTTPError("Service unavailable", response=response)

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.side_effect = error
            mock_api_cls.return_value = mock_api

            with pytest.raises(HfHubHTTPError):
                await HFDownloader.search_models(query="model")

        assert mock_api.list_models.call_count == 1

    @pytest.mark.asyncio
    async def test_recommended_retries_anonymously_on_401(self):
        """Recommended lists survive a stale token and set the flag."""
        models = [
            _make_mock_model(
                "mlx-community/model-a",
                disk_size_bytes=1_000_000_000,
                downloads=500,
                trending_score=5,
            ),
        ]

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.side_effect = _stale_token_list_models(models)
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=16 * 1024**3
            )

        assert len(result["trending"]) == 1
        assert len(result["popular"]) == 1
        assert result["hf_token_invalid"] is True

    @pytest.mark.asyncio
    async def test_recommended_valid_token_not_flagged(self):
        """The flag stays False when both recommended fetches succeed."""
        models = [
            _make_mock_model(
                "mlx-community/model-a",
                disk_size_bytes=1_000_000_000,
                downloads=500,
            ),
        ]

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.return_value = models
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_recommended_models(
                max_memory_bytes=16 * 1024**3
            )

        assert result["hf_token_invalid"] is False


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
        assert result["size"] == 14_000_000_000
        assert len(result["files"]) == 1
        assert result["files"][0]["name"] == "model.safetensors"
        assert "text-generation" in result["tags"]
        assert result["model_card"] == ""  # No README available
        assert result["is_adapter"] is False

    @pytest.mark.asyncio
    async def test_u32_size_uses_sibling_blobs_not_histogram(self):
        mock_info = MagicMock()
        mock_info.id = "mlx-community/gemma-4-26B-A4B-it-4bit"
        mock_info.downloads = 1000
        mock_info.likes = 10
        mock_info.tags = ["mlx"]
        mock_info.pipeline_tag = "text-generation"
        mock_info.created_at = None
        mock_info.last_modified = None
        mock_info.safetensors = {
            "parameters": {"U32": 25_235_685_376, "BF16": 570_250_830},
            "total": 25_805_936_206,
        }
        mock_info.card_data = None
        weight = MagicMock()
        weight.rfilename = "model.safetensors"
        weight.size = 15_400_000_000
        tokenizer = MagicMock()
        tokenizer.rfilename = "tokenizer.json"
        tokenizer.size = 1_000_000
        mock_info.siblings = [weight, tokenizer]

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls, \
             patch("omlx.admin.hf_downloader.hf_hub_download", side_effect=Exception("no readme")):
            mock_api = MagicMock()
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.get_model_info(mock_info.id)

        assert result["size"] == 15_400_000_000
        assert result["size"] != _calc_safetensors_disk_size(mock_info.safetensors)
        assert result["params"] == 25_805_936_206

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

    def test_non_int_count_returns_zero(self):
        from omlx.admin.hf_downloader import _get_param_count

        assert _get_param_count({"parameters": {"BF16": None}}) == 0


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

    def test_non_int_count_returns_zero(self):
        from omlx.admin.hf_downloader import _calc_safetensors_disk_size

        assert _calc_safetensors_disk_size({"parameters": {"BF16": None}}) == 0
        assert (
            _calc_safetensors_disk_size({"parameters": {"BF16": 100, "F32": None}})
            == 0
        )


class TestSafetensorsBlobSize:
    """Blob-size helpers for U32-packed MLX quants (#3401)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reject_token", [False, True])
    async def test_http_timeout_leaves_size_retryable(self, reject_token):
        requests = []

        def respond(request):
            requests.append(request)
            if reject_token and request.headers.get("authorization"):
                return httpx.Response(401)
            raise httpx.ReadTimeout("Hub stalled", request=request)

        api = HfApi(token="test-token" if reject_token else False)
        with httpx.Client(transport=httpx.MockTransport(respond)) as client, patch(
            "huggingface_hub.hf_api.get_session", return_value=client
        ), patch.object(hf_downloader_mod, "_HF_API_TIMEOUT", 0.1):
            for _ in range(2):
                sizes = await hf_downloader_mod._blob_bytes_for_repos(
                    api, ["owner/model"]
                )
                assert sizes == {"owner/model": 0}
                assert hf_downloader_mod._cached_blob_size("owner/model") is None

        assert len(requests) == (4 if reject_token else 2)
        assert all(request.extensions["timeout"]["read"] == 0.1 for request in requests)
        if reject_token:
            assert "authorization" not in requests[1].headers
            assert "authorization" not in requests[3].headers

    def test_empty_or_name_only_siblings(self):
        assert _sum_safetensors_blob_bytes(None) is None
        assert _sum_safetensors_blob_bytes([]) is None
        nameless = MagicMock()
        nameless.rfilename = "model.safetensors"
        nameless.size = None
        assert _sum_safetensors_blob_bytes([nameless]) is None

    def test_sums_safetensors_and_ignores_tokenizer(self):
        weight = MagicMock()
        weight.rfilename = "model-00001-of-00002.safetensors"
        weight.size = 10_000_000_000
        weight2 = MagicMock()
        weight2.rfilename = "model-00002-of-00002.safetensors"
        weight2.size = 5_400_000_000
        tokenizer = MagicMock()
        tokenizer.rfilename = "tokenizer.json"
        tokenizer.size = 1_000_000
        assert _sum_safetensors_blob_bytes([weight, weight2, tokenizer]) == 15_400_000_000

    def test_issue_3401_histogram_is_packed_u32(self):
        st = {
            "parameters": {"U32": 25_235_685_376, "BF16": 570_250_830},
            "total": 25_805_936_206,
        }
        assert _histogram_has_packed_u32(st) is True
        assert _histogram_has_packed_u32({"parameters": {"BF16": 1_000}}) is False
        inflated = _calc_safetensors_disk_size(st)
        assert inflated > 90 * 1024**3

    def test_store_blob_size_drops_expired_entries(self):
        now = time.monotonic()
        expired_at = now - hf_downloader_mod._BLOB_SIZE_CACHE_TTL - 1
        hf_downloader_mod._blob_size_cache["old/a"] = (100, expired_at)
        hf_downloader_mod._blob_size_cache["old/b"] = (200, expired_at)

        hf_downloader_mod._store_blob_size("fresh/c", 15_400_000_000)

        assert set(hf_downloader_mod._blob_size_cache) == {"fresh/c"}
        assert hf_downloader_mod._cached_blob_size("fresh/c") == 15_400_000_000

    def test_store_blob_size_caps_cache_length(self):
        with patch.object(hf_downloader_mod, "_BLOB_SIZE_CACHE_MAX", 3):
            for i in range(5):
                hf_downloader_mod._store_blob_size(f"org/m{i}", 1000 + i)

            assert len(hf_downloader_mod._blob_size_cache) == 3
            assert "org/m0" not in hf_downloader_mod._blob_size_cache
            assert "org/m1" not in hf_downloader_mod._blob_size_cache
            assert hf_downloader_mod._cached_blob_size("org/m4") == 1004

    def test_store_blob_size_ignores_non_positive(self):
        hf_downloader_mod._store_blob_size("org/zero", 0)
        hf_downloader_mod._store_blob_size("org/neg", -1)
        assert hf_downloader_mod._blob_size_cache == {}


# =============================================================================
# Timeout Tests
# =============================================================================


class TestHFAPITimeouts:
    """Test that HF API calls respect timeouts when HuggingFace is unreachable."""

    @pytest.mark.asyncio
    async def test_get_recommended_models_timeout(self, blocked_worker):
        """get_recommended_models should raise TimeoutError when HF is unreachable."""

        def slow_list_models(**kwargs):
            blocked_worker.call()
            return []

        with patch("omlx.admin.hf_downloader._HF_API_TIMEOUT", 0.1), \
             patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.side_effect = slow_list_models
            mock_api_cls.return_value = mock_api

            with pytest.raises(asyncio.TimeoutError):
                await HFDownloader.get_recommended_models(
                    max_memory_bytes=16 * 1024**3
                )

            assert blocked_worker.started.is_set()

    @pytest.mark.asyncio
    async def test_search_models_timeout(self, blocked_worker):
        """search_models should raise TimeoutError when HF is unreachable."""

        def slow_list_models(**kwargs):
            blocked_worker.call()
            return []

        with patch("omlx.admin.hf_downloader._HF_API_TIMEOUT", 0.1), \
             patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.side_effect = slow_list_models
            mock_api_cls.return_value = mock_api

            with pytest.raises(asyncio.TimeoutError):
                await HFDownloader.search_models(query="test")

            assert blocked_worker.started.is_set()

    @pytest.mark.asyncio
    async def test_get_model_info_timeout(self, blocked_worker):
        """get_model_info should raise TimeoutError when HF is unreachable."""

        def slow_model_info(*args, **kwargs):
            blocked_worker.call()

        with patch("omlx.admin.hf_downloader._HF_API_TIMEOUT", 0.1), \
             patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.model_info.side_effect = slow_model_info
            mock_api_cls.return_value = mock_api

            with pytest.raises(asyncio.TimeoutError):
                await HFDownloader.get_model_info("org/model")

            assert blocked_worker.started.is_set()

    @pytest.mark.asyncio
    async def test_search_models_timeout_on_lazy_iteration(self, blocked_worker):
        """list_models returns a lazy generator; a hang during iteration
        (not the call itself) must still hit the timeout instead of
        blocking the event loop (issue #2325)."""

        def lazy_hanging_list_models(**kwargs):
            def gen():
                blocked_worker.call()
                yield None

            return gen()

        with patch("omlx.admin.hf_downloader._HF_API_TIMEOUT", 0.1), \
             patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.side_effect = lazy_hanging_list_models
            mock_api_cls.return_value = mock_api

            with pytest.raises(asyncio.TimeoutError):
                await HFDownloader.search_models(query="test")

            assert blocked_worker.started.is_set()

    @pytest.mark.asyncio
    async def test_get_recommended_models_timeout_on_lazy_iteration(
        self, blocked_worker
    ):
        """Same lazy-iteration hang, via get_recommended_models."""

        def lazy_hanging_list_models(**kwargs):
            def gen():
                blocked_worker.call()
                yield None

            return gen()

        with patch("omlx.admin.hf_downloader._HF_API_TIMEOUT", 0.1), \
             patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.side_effect = lazy_hanging_list_models
            mock_api_cls.return_value = mock_api

            with pytest.raises(asyncio.TimeoutError):
                await HFDownloader.get_recommended_models(
                    max_memory_bytes=16 * 1024**3
                )

            assert blocked_worker.started.is_set()

    @pytest.mark.asyncio
    async def test_search_models_drains_generator_off_event_loop(self):
        """The lazy generator must be consumed in a worker thread, never
        on the event loop thread."""
        seen_threads = []

        def lazy_list_models(**kwargs):
            def gen():
                seen_threads.append(threading.current_thread())
                yield _make_mock_model(
                    "mlx-community/model-a",
                    disk_size_bytes=1_000_000_000,
                    downloads=500,
                )

            return gen()

        with patch("omlx.admin.hf_downloader.HfApi") as mock_api_cls:
            mock_api = MagicMock()
            mock_api.list_models.side_effect = lazy_list_models
            mock_api_cls.return_value = mock_api

            result = await HFDownloader.search_models(query="test")

        assert len(result["models"]) == 1
        loop_thread = threading.current_thread()
        assert seen_threads
        assert all(t is not loop_thread for t in seen_threads)


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
            await _wait_for_downloads(downloader)

            # Called twice: dry_run + actual download
            assert mock_download.call_count == 2
            # Last call is the actual download
            call_kwargs = mock_download.call_args[1]
            assert "dry_run" not in call_kwargs
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
            await _wait_for_downloads(downloader)

            assert mock_download.call_count == 2
            call_kwargs = mock_download.call_args[1]
            assert "dry_run" not in call_kwargs
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
    async def test_retry_active_download_raises(self, downloader, blocked_worker):
        """Retrying an active download should raise ValueError."""
        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=blocked_worker.call,
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.siblings = []
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task = await downloader.start_download("owner/model")
            await asyncio.wait_for(blocked_worker.started.wait(), timeout=5)

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
    async def test_zero_byte_startup_stall_aborts_xet(self, model_dir, monkeypatch):
        """No first write must trigger the startup deadline, including at 0%."""
        import omlx.admin.hf_downloader as dl_module

        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 0.03)
        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.01)
        target = model_dir / "owner" / "model"
        target.mkdir(parents=True)
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(
            task_id="t1",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
        )
        downloader._tasks[task.task_id] = task
        calls = 0

        def zero_byte_temp(_path):
            nonlocal calls
            calls += 1
            if calls == 1:
                return _DownloadActivity()
            return _DownloadActivity(file_count=1, latest_mtime_ns=1)

        with patch.object(
            downloader,
            "_get_download_activity",
            side_effect=zero_byte_temp,
        ), patch("omlx.admin.hf_downloader.abort_xet_session") as mock_abort:
            await downloader._poll_progress(task.task_id, target)

        stalled = downloader._stalled[task.task_id]
        assert stalled.phase == "startup"
        assert stalled.transport == "Xet"
        mock_abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_active_stall_uses_longer_timeout(self, model_dir, monkeypatch):
        """After the first write, the active-transfer timeout must apply."""
        import omlx.admin.hf_downloader as dl_module

        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 1)
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 0.03)
        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.01)
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(
            task_id="t1",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
        )
        downloader._tasks[task.task_id] = task
        empty = _DownloadActivity()
        writing = _DownloadActivity(
            file_count=1,
            logical_size=10,
            allocated_size=4096,
            latest_mtime_ns=1,
        )

        with patch.object(
            downloader,
            "_get_download_activity",
            side_effect=[empty, writing, writing, writing, writing, writing],
        ), patch("omlx.admin.hf_downloader.abort_xet_session") as mock_abort:
            await downloader._poll_progress(task.task_id, model_dir)

        stalled = downloader._stalled[task.task_id]
        assert stalled.phase == "active"
        assert stalled.timeout == 0.03
        mock_abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_wire_activity_prevents_false_stall(
        self, model_dir, monkeypatch
    ):
        """xet's fetch phase keeps the filesystem silent for minutes while
        bytes keep arriving on the wire: wire movement alone must hold the
        stall deadline open."""
        import omlx.admin.hf_downloader as dl_module
        from omlx.admin.hf_downloader import _WireCounter

        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 0.03)
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 0.03)
        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.01)
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(
            task_id="t1",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
        )
        downloader._tasks[task.task_id] = task
        counter = _WireCounter()
        downloader._wire_counters[task.task_id] = counter
        frozen = _DownloadActivity()  # fetch phase: nothing lands on disk

        with patch.object(
            downloader,
            "_get_download_activity",
            return_value=frozen,
        ), patch("omlx.admin.hf_downloader.abort_xet_session") as mock_abort:
            poll = asyncio.create_task(
                downloader._poll_progress(task.task_id, model_dir)
            )
            # Well past both 0.03s deadlines, wire bytes keep flowing.
            for _ in range(8):
                counter.add(1_000_000)
                await asyncio.sleep(0.01)
            task.status = DownloadStatus.COMPLETED
            await poll

        assert task.task_id not in downloader._stalled
        mock_abort.assert_not_called()

    @pytest.mark.asyncio
    async def test_stopped_wire_reports_an_active_stall(
        self, model_dir, monkeypatch
    ):
        """Wire bytes are payload activity, so once they stop against a
        silent disk the stall must be reported as 'active' under the longer
        deadline — not as a startup handshake hang."""
        import omlx.admin.hf_downloader as dl_module
        from omlx.admin.hf_downloader import _WireCounter

        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 0.03)
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 0.3)
        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.01)
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(
            task_id="t1",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
        )
        downloader._tasks[task.task_id] = task
        counter = _WireCounter()
        downloader._wire_counters[task.task_id] = counter
        frozen = _DownloadActivity()

        with patch.object(
            downloader,
            "_get_download_activity",
            return_value=frozen,
        ), patch("omlx.admin.hf_downloader.abort_xet_session") as mock_abort:
            poll = asyncio.create_task(
                downloader._poll_progress(task.task_id, model_dir)
            )
            for _ in range(4):  # payload past the startup window, then stops
                counter.add(1_000_000)
                await asyncio.sleep(0.01)
            for _ in range(200):  # wait up to ~2s for the active deadline
                if task.task_id in downloader._stalled:
                    break
                await asyncio.sleep(0.01)
            stalled = downloader._stalled.get(task.task_id)
            task.status = DownloadStatus.COMPLETED
            await poll

        assert stalled is not None, "a silent wire must still stall"
        assert stalled.phase == "active"
        assert stalled.timeout == 0.3
        mock_abort.assert_called_once()


# =============================================================================
# Xet HTTP Fallback Tests
# =============================================================================


class TestXetHTTPFallback:
    @pytest.fixture
    def model_dir(self, tmp_path):
        path = tmp_path / "models"
        path.mkdir()
        return path

    @staticmethod
    def _api():
        api = MagicMock()
        info = MagicMock()
        info.safetensors = {}
        api.model_info.return_value = info
        return api

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (
                RuntimeError(
                    "CAS service error: ReqwestMiddleware request failed "
                    "for /xet-read-token"
                ),
                True,
            ),
            (RuntimeError("ordinary download failure"), False),
            (OSError(28, "No space left on device"), False),
        ],
    )
    def test_xet_error_classification(self, error, expected):
        assert _is_xet_transport_error(error) is expected

    @pytest.mark.asyncio
    async def test_xet_error_retries_once_over_http(self, model_dir):
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        def fail_xet(**kwargs):
            if kwargs.get("dry_run"):
                return []
            raise RuntimeError(
                "CAS service error: ReqwestMiddleware request failed "
                "for /xet-read-token"
            )

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(self._api(), None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fail_xet,
        ), patch.object(
            downloader,
            "_run_http_fallback",
            new_callable=AsyncMock,
        ) as fallback:
            await downloader._run_download(task.task_id, "secret-token")

        fallback.assert_awaited_once()
        assert task.status == DownloadStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_zero_byte_stall_waits_for_xet_exit_before_fallback(
        self, model_dir, monkeypatch
    ):
        import omlx.admin.hf_downloader as dl_module

        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 0.03)
        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.01)
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._tasks[task.task_id] = task
        aborted = threading.Event()
        events = []

        def stalled_xet(**kwargs):
            if kwargs.get("dry_run"):
                return []
            assert aborted.wait(1)
            events.append("xet_stopped")
            raise RuntimeError("xet session aborted")

        async def finish_http(*_args, **_kwargs):
            events.append("http_started")

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(self._api(), None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=stalled_xet,
        ), patch(
            "omlx.admin.hf_downloader.abort_xet_session",
            side_effect=aborted.set,
        ) as abort, patch.object(
            downloader,
            "_run_http_fallback",
            side_effect=finish_http,
        ) as fallback:
            await downloader._run_download(task.task_id, "")

        assert events == ["xet_stopped", "http_started"]
        abort.assert_called_once()
        fallback.assert_awaited_once()
        assert task.status == DownloadStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_http_failure_preserves_both_errors(self, model_dir):
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        def fail_xet(**kwargs):
            if kwargs.get("dry_run"):
                return []
            raise RuntimeError("CAS service error from hf_xet")

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(self._api(), None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fail_xet,
        ), patch.object(
            downloader,
            "_run_http_fallback",
            new_callable=AsyncMock,
            side_effect=RuntimeError("HTTP offline"),
        ):
            await downloader._run_download(task.task_id, "")

        assert task.status == DownloadStatus.FAILED
        assert "CAS service error" in task.error
        assert "HTTP offline" in task.error

    @pytest.mark.asyncio
    async def test_http_worker_disables_xet_without_token_in_argv(self, model_dir):
        downloader = HFDownloader(model_dir=str(model_dir))
        process = MagicMock()
        process.returncode = 0
        process.communicate = AsyncMock(return_value=(b'{"ok": true}\n', b""))
        kwargs = {
            "repo_id": "owner/model",
            "local_dir": str(model_dir / "owner" / "model"),
            "token": "secret-token",
            "endpoint": None,
            "etag_timeout": 30,
        }

        with patch(
            "omlx.admin.hf_downloader.asyncio.create_subprocess_exec",
            new_callable=AsyncMock,
            return_value=process,
        ) as spawn:
            await downloader._run_http_fallback("t1", kwargs)

        argv = spawn.await_args.args
        assert argv[1:] == ("-m", "omlx._hf_download_worker")
        assert "secret-token" not in argv
        assert spawn.await_args.kwargs["env"]["HF_HUB_DISABLE_XET"] == "1"
        request = json.loads(process.communicate.await_args.kwargs["input"])
        assert request["kwargs"]["token"] == "secret-token"

    def test_worker_sets_disable_xet_before_download(self, monkeypatch):
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
        download = MagicMock()

        _download_without_xet({"repo_id": "owner/model"}, download)

        assert os.environ["HF_HUB_DISABLE_XET"] == "1"
        download.assert_called_once_with(repo_id="owner/model")

    @pytest.mark.asyncio
    async def test_http_fallback_starts_wire_progress_from_zero(
        self, model_dir
    ):
        """The HTTP worker refetches the payload (xet's chunk cache is not
        reusable), so fetch-phase wire bytes must not carry into the
        restart's progress: the replacement poll gets a fresh counter."""
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._tasks[task.task_id] = task
        seen: dict[str, object] = {}

        def fail_xet(**kwargs):
            if kwargs.get("dry_run"):
                return []
            # Bytes that arrived on the wire before the transport died.
            downloader._wire_counters["t1"].add(7_000_000)
            raise RuntimeError(
                "CAS service error: ReqwestMiddleware request failed "
                "for /xet-read-token"
            )

        async def note_fallback(*_args, **_kwargs):
            counter = downloader._wire_counters.get("t1")
            seen["value"] = counter.value if counter is not None else None

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(self._api(), None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fail_xet,
        ), patch.object(
            downloader,
            "_run_http_fallback",
            new=note_fallback,
        ):
            await downloader._run_download(task.task_id, "secret-token")

        assert seen["value"] == 0
        assert task.status == DownloadStatus.COMPLETED


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
    async def test_second_download_stays_pending(self, model_dir, blocked_worker):
        """When two downloads are started, only the first should be DOWNLOADING."""
        downloader = HFDownloader(model_dir=str(model_dir))

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=blocked_worker.call,
        ):
            mock_api = MagicMock()
            mock_info = MagicMock()
            mock_info.safetensors = {"parameters": {"BF16": 5000}}
            mock_api.model_info.return_value = mock_info
            mock_api_cls.return_value = mock_api

            task1 = await downloader.start_download("owner/model-a")
            task2 = await downloader.start_download("owner/model-b")

            # The running download holds the semaphore.
            await asyncio.wait_for(blocked_worker.started.wait(), timeout=5)

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

            # Wait for both scheduled downloads to finish.
            await _wait_for_downloads(downloader)

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

        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 0.03)
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 0.03)
        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.01)
        downloader = HFDownloader(model_dir=str(model_dir))
        call_count = 0

        def active_download(_path):
            nonlocal call_count
            call_count += 1
            return _DownloadActivity(
                file_count=1,
                logical_size=1000,
                allocated_size=4096,
                latest_mtime_ns=call_count,
            )

        task = DownloadTask(
            task_id="t1",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
        )
        downloader._tasks[task.task_id] = task

        with patch.object(
            downloader,
            "_get_download_activity",
            side_effect=active_download,
        ), patch("omlx.admin.hf_downloader.abort_xet_session") as mock_abort:
            poll = asyncio.create_task(
                downloader._poll_progress(task.task_id, model_dir)
            )
            await asyncio.sleep(0.08)
            task.status = DownloadStatus.COMPLETED
            await poll

        assert task.task_id not in downloader._stalled
        mock_abort.assert_not_called()


# =============================================================================
# Download Speed Tests
# =============================================================================


class TestDownloadSpeed:
    """The poll loop must publish a live rate and clear it at terminal states."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        return d

    @staticmethod
    def _growing_activity(step=100_000):
        """Activity scanner whose allocated blocks grow by `step` per call.

        The per-file map mirrors the aggregate so the speed meter sees the
        same growth under a single watched path.
        """
        state = {"allocated": 0}

        def scan(_path):
            state["allocated"] += step
            return _DownloadActivity(
                file_count=1,
                logical_size=state["allocated"],
                allocated_size=state["allocated"],
                latest_mtime_ns=1,
                files={"payload": state["allocated"]},
            )

        return scan

    def test_defaults_read_as_a_per_second_rate(self):
        """The readout is "bytes per second": sample at 0.5s, average over
        a 1s window, and never average fewer samples than the window holds."""
        import omlx.admin.hf_downloader as dl_module
        import omlx.admin.ms_downloader as ms_module

        assert dl_module._PROGRESS_POLL_INTERVAL == 0.5
        assert ms_module._PROGRESS_POLL_INTERVAL == 0.5
        assert dl_module._SPEED_WINDOW == 1.0

    @pytest.mark.asyncio
    async def test_poll_reports_speed_then_zeroes_it(self, model_dir, monkeypatch):
        """A live transfer publishes bytes/s; a terminal task publishes 0."""
        import omlx.admin.hf_downloader as dl_module

        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.01)
        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 5)
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 5)
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(
            task_id="t-speed",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
            total_size=10_000_000,
        )
        downloader._tasks[task.task_id] = task

        with patch.object(
            downloader,
            "_get_download_activity",
            side_effect=self._growing_activity(),
        ):
            poll = asyncio.create_task(
                downloader._poll_progress(task.task_id, model_dir)
            )
            await asyncio.sleep(0.05)
            observed_speed = task.speed_bps
            task.status = DownloadStatus.COMPLETED
            await poll

        assert observed_speed > 0, "a live download must report a rate"
        # The poll loop's finally clause clears the rate with the task.
        assert task.speed_bps == 0.0

    @pytest.mark.asyncio
    async def test_speed_smooths_steady_rate(self, model_dir, monkeypatch):
        """A constant per-interval rate must be reported near its true value."""
        import omlx.admin.hf_downloader as dl_module

        interval = 0.02
        step = 200_000
        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", interval)
        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 5)
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 5)
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(
            task_id="t-smooth",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
        )
        downloader._tasks[task.task_id] = task

        with patch.object(
            downloader,
            "_get_download_activity",
            side_effect=self._growing_activity(step),
        ):
            poll = asyncio.create_task(
                downloader._poll_progress(task.task_id, model_dir)
            )
            # Let several samples accumulate so the EMA settles.
            await asyncio.sleep(0.12)
            steady_speed = task.speed_bps
            task.status = DownloadStatus.COMPLETED
            await poll

        expected = step / interval
        assert steady_speed > 0
        # Order of magnitude of the true rate: smoothing must not lose it.
        assert 0.25 * expected <= steady_speed <= 4 * expected

    def test_speed_meter_forgets_stale_history_within_the_window(self):
        """A stopped transfer must read 0 quickly, not decay forever.

        This is the lag complaint: an exponential average keeps a fraction
        of the old rate alive indefinitely, so the number trails reality.
        The fixed window must bottom out once every sample in it is flat
        (one poll interval after the window has passed the last byte).
        """
        from omlx.admin.hf_downloader import _SpeedMeter

        window = 2.0
        interval = 1.0
        meter = _SpeedMeter(window=window)
        t = 100.0
        last_byte_at = t + 4 * interval  # 50 MB/s for 4 intervals, then stops

        # The poll loop samples every interval whether or not bytes moved.
        rates = []
        for i in range(8):
            now = t + i * interval
            bytes_written = min(i * 50_000_000, 4 * 50_000_000)
            rates.append(meter.add({"payload": bytes_written}, now=now))

        # The last byte lands on the tick `last_byte_at` names: the meter is
        # still reporting traffic there, and only bottoms out afterwards.
        assert last_byte_at == t + 4 * interval
        assert rates[4] > 0.0, "the rate must not reach 0 before the transfer stops"
        assert rates[-1] == 0.0, "speed must reach 0 within ~window of the last byte"

    def test_speed_meter_reports_true_mean_over_the_window(self):
        """The reported rate is bytes/second over the window, not a blend."""
        from omlx.admin.hf_downloader import _SpeedMeter

        meter = _SpeedMeter(window=3.0)
        t = 50.0
        # Steady 10 MB/s sampled every second.
        for i in range(6):
            rate = meter.add({"payload": i * 10_000_000}, now=t + i * 1.0)
        # Window spans 3s of history -> exactly the true rate.
        assert abs(rate - 10_000_000) < 1_000_000

    def test_speed_meter_survives_a_truncated_file(self):
        """A truncated file must not poison the window with a negative
        delta (which would freeze the display), and its refill must count
        as real growth from the new, smaller baseline."""
        from omlx.admin.hf_downloader import _SpeedMeter

        meter = _SpeedMeter(window=2.0)
        t = 10.0
        meter.add({"f": 100_000_000}, now=t)  # baseline: everything is new
        meter.add({"f": 100_000_000}, now=t + 1)  # idle at full size
        after_shrink = meter.add({"f": 50_000_000}, now=t + 2)
        assert after_shrink == 0.0, "a truncation is not transfer"
        # Refill to 150M from the 50M baseline: 100M of genuine growth
        # must surface, not be suppressed by the stale 100M size.
        recovered = meter.add({"f": 150_000_000}, now=t + 3)
        assert recovered > 0

    def test_speed_meter_ignores_bytes_that_reappear_wholesale(self):
        """Resume must not read the previously downloaded bytes as fresh
        transfer. Cleanup racing the retry can make one walk come back
        empty (temp wiped, rglob aborted); the next complete walk then
        sees the whole tree again. The old aggregate window re-anchored on
        the empty sample and reported the reappearance as the entire prior
        download arriving within one window — a multi-GB/s spike for a
        second. File-level continuity: forgotten paths start at zero."""
        from omlx.admin.hf_downloader import _SpeedMeter

        meter = _SpeedMeter(window=1.0)
        t = 100.0
        full = {"a": 27_000_000_000, "b": 1_000_000_000}
        meter.add(full, now=t)  # prime: everything is new -> no growth
        assert meter.add(dict(full), now=t + 0.5) == 0.0  # idle at full size
        assert meter.add({}, now=t + 1.0) == 0.0  # wiped / aborted walk
        reappeared = meter.add(dict(full), now=t + 1.5)  # tree back in full
        assert reappeared == 0.0, "reappearing bytes are not new transfer"

    def test_speed_meter_ignores_a_file_first_seen_at_full_size(self):
        """A file moved/copied into the tree wholesale is not transfer,
        but once it is under watch its real growth still counts."""
        from omlx.admin.hf_downloader import _SpeedMeter

        meter = _SpeedMeter(window=1.0)
        t = 0.0
        base = {"x": 1_000_000}
        meter.add(dict(base), now=t)
        assert meter.add(dict(base), now=t + 0.5) == 0.0
        moved_in = {"x": 1_000_000, "y": 5_000_000_000}
        assert meter.add(moved_in, now=t + 1.0) == 0.0, (
            "first sight at full size must not count"
        )
        grown = {"x": 1_000_000, "y": 5_000_100_000}
        assert meter.add(grown, now=t + 1.5) > 0, (
            "growth of a tracked file still counts"
        )

    def test_speed_meter_prime_after_a_partial_walk_does_not_spike(self):
        """The retry's baseline scan can abort mid-way (cleanup race).
        A prime of nothing followed by a complete walk must read 0, not
        the whole model's bytes divided by one poll interval."""
        from omlx.admin.hf_downloader import _SpeedMeter

        meter = _SpeedMeter(window=1.0)
        meter.add({}, now=0.0)  # walk aborted at prime time
        rate = meter.add({"model": 20_000_000_000}, now=0.5)
        assert rate == 0.0

    def test_only_the_transfer_bar_feeds_the_wire_counter(self):
        """Wire bytes come from xet's network-transfer bar alone.

        snapshot_download's reconstruction bar (disk bytes, has a
        denominator), the meta file-count bar, and any default-format bar
        must never feed the wire counter, or one payload would be counted
        twice and the readout could show up to 2x the real rate."""
        from huggingface_hub.utils._xet_progress_reporting import (
            XET_BYTES_BAR_FORMAT,
            XET_TRANSFER_BAR_FORMAT,
        )
        from omlx.admin.hf_downloader import _make_cancellable_tqdm as make

        seen = []
        cls = make(lambda: False, on_wire_bytes=seen.append)
        bars = [
            cls(  # snapshot_download's transfer bar: network bytes, no total
                desc="Downloading bytes",
                total=0,
                unit="B",
                unit_scale=True,
                bar_format=XET_TRANSFER_BAR_FORMAT,
                disable=True,
            ),
            cls(  # reconstruction bar: disk bytes, "{...}/{total_fmt}"
                desc="Reconstructing (incomplete total...)",
                total=0,
                unit="B",
                unit_scale=True,
                bar_format=XET_BYTES_BAR_FORMAT,
                disable=True,
            ),
            cls(desc="Fetching 7 files", total=7, disable=True),  # meta
            cls(total=100, disable=True),  # default format
        ]
        for bar in bars:
            bar.update(1_000_000)

        assert seen == [1_000_000]

    def test_cancel_still_raises_on_the_wire_bar(self):
        """The wire hook observes the increment but must not swallow the
        cancellation raise that unwinds the download thread."""
        from huggingface_hub.utils._xet_progress_reporting import (
            XET_TRANSFER_BAR_FORMAT,
        )

        seen = []
        cls = _make_cancellable_tqdm(lambda: True, on_wire_bytes=seen.append)
        bar = cls(bar_format=XET_TRANSFER_BAR_FORMAT, disable=True)

        with pytest.raises(_DownloadCancelled):
            bar.update(5)

        assert seen == [5]

    def test_wire_counter_is_thread_safe_and_ignores_junk(self):
        """Progress callbacks fire from xet's reporting thread while the poll
        loop reads: the accumulator must be exact under concurrency and skip
        non-positive increments."""
        from omlx.admin.hf_downloader import _WireCounter

        counter = _WireCounter()

        def spam():
            for _ in range(500):
                counter.add(100)

        threads = [threading.Thread(target=spam) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.value == 400_000
        counter.add(0)
        counter.add(-1)
        assert counter.value == 400_000

    def test_wire_speed_meter_window_and_settle_to_zero(self):
        """Same window semantics as the disk meter: a live mean over the
        window, reaching 0 within ~window of the last wire byte."""
        from omlx.admin.hf_downloader import _WireSpeedMeter

        meter = _WireSpeedMeter(window=1.0)
        assert meter.add(0, now=0.0) == 0.0  # a prime alone is no rate
        assert meter.add(500, now=0.5) == 1000.0  # 500B over 0.5s
        assert meter.add(500, now=1.0) == 500.0  # averaged over the window
        assert meter.add(500, now=1.6) == 0.0  # stopped -> settles to 0

    @pytest.mark.asyncio
    async def test_poll_shows_wire_speed_while_disk_is_idle(
        self, model_dir, monkeypatch
    ):
        """xet's fetch phase pulls from the network before any disk write
        (reconstruction blocks are >= 256MB): the readout must show the wire
        rate while filesystem activity stays frozen, then clear to 0."""
        import omlx.admin.hf_downloader as dl_module
        from omlx.admin.hf_downloader import _WireCounter

        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.01)
        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 5)
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 5)
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(
            task_id="t-wire",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
            total_size=10_000_000_000,
        )
        downloader._tasks[task.task_id] = task
        counter = _WireCounter()
        downloader._wire_counters[task.task_id] = counter

        frozen = _DownloadActivity()  # no byte lands on disk during fetch
        with patch.object(
            downloader, "_get_download_activity", return_value=frozen
        ):
            poll = asyncio.create_task(
                downloader._poll_progress(task.task_id, model_dir)
            )
            observed = 0.0
            for _ in range(5):
                counter.add(1_000_000)
                await asyncio.sleep(0.03)
                observed = max(observed, task.speed_bps)
            task.status = DownloadStatus.COMPLETED
            await poll

        assert frozen.allocated_size == 0, "precondition: the disk never moved"
        assert observed > 0, "wire traffic must show while the disk is idle"
        assert task.speed_bps == 0.0  # terminal tasks publish 0

    @pytest.mark.asyncio
    async def test_poll_publishes_the_larger_stage_not_the_sum(
        self, model_dir, monkeypatch
    ):
        """Fetch (wire) and reconstruction (disk) are two views of one
        payload: the published rate is the larger of the two, never their
        sum, which would count the same bytes twice."""
        import omlx.admin.hf_downloader as dl_module
        from omlx.admin.hf_downloader import _WireCounter

        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.02)
        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 5)
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 5)
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(
            task_id="t-stage",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
            total_size=10_000_000_000,
        )
        downloader._tasks[task.task_id] = task
        counter = _WireCounter()
        downloader._wire_counters[task.task_id] = counter

        disk = {"bytes": 0}

        def growing_activity(_path):
            disk["bytes"] += 2_000_000
            return _DownloadActivity(
                file_count=1,
                logical_size=disk["bytes"],
                allocated_size=disk["bytes"],
                latest_mtime_ns=1,
                files={"payload": disk["bytes"]},
            )

        observed = 0.0
        with patch.object(
            downloader, "_get_download_activity", side_effect=growing_activity
        ):
            poll = asyncio.create_task(
                downloader._poll_progress(task.task_id, model_dir)
            )
            start = time.monotonic()
            for _ in range(5):
                counter.add(2_000_000)  # same order of magnitude as the disk
                await asyncio.sleep(0.02)
                observed = max(observed, task.speed_bps)
            elapsed = max(time.monotonic() - start, 1e-6)
            task.status = DownloadStatus.COMPLETED
            await poll

        stage_rate = (
            max(disk["bytes"], 10_000_000) / elapsed
        )  # the two stages carry comparable byte counts
        assert observed > 0
        assert observed <= stage_rate * 1.5, "must not exceed the larger stage"
        assert observed < (disk["bytes"] + 10_000_000) / elapsed * 0.8, (
            "the two stages must not be summed"
        )


# =============================================================================
# Progress Reads Both Pipeline Stages (fetch = wire, reconstruction = disk)
# =============================================================================


class TestProgressFromWire:
    """Reported bytes must not freeze at the small files while xet's fetch
    phase moves the payload over the network before any disk write."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        return d

    @pytest.mark.asyncio
    async def test_fetch_phase_progress_follows_the_wire(
        self, model_dir, monkeypatch
    ):
        import omlx.admin.hf_downloader as dl_module
        from omlx.admin.hf_downloader import _WireCounter

        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.01)
        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 5)
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 5)
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(
            task_id="t-wire",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
            total_size=10_000_000,
        )
        downloader._tasks[task.task_id] = task
        counter = _WireCounter()
        downloader._wire_counters[task.task_id] = counter

        with patch.object(
            downloader,
            "_get_download_activity",
            return_value=_DownloadActivity(),  # fetch: the disk stays silent
        ):
            poll = asyncio.create_task(
                downloader._poll_progress(task.task_id, model_dir)
            )
            counter.add(2_500_000)
            await asyncio.sleep(0.05)  # several poll iterations

            # The wire stage leads: 25% instead of a frozen 0%.
            assert task.downloaded_size == 2_500_000
            assert task.progress == 25.0

            # The wire may pass the size estimate (retries, protocol
            # overhead): the report caps at the estimate, and 100% stays
            # reserved for snapshot_download's completion write.
            counter.add(9_500_000)
            await asyncio.sleep(0.05)
            assert task.downloaded_size == 10_000_000
            assert task.progress == 99.0

            task.status = DownloadStatus.COMPLETED
            await poll

    @pytest.mark.asyncio
    async def test_resume_adds_the_wire_delta_to_the_on_disk_baseline(
        self, model_dir, monkeypatch
    ):
        """A resumed download re-fetches only the missing bytes, so its wire
        counter starts at zero while the disk already holds earlier files.
        The report must add that delta on top of the on-disk baseline:
        max(disk, wire) alone parks on the disk figure for the whole fetch
        phase — the freeze this class exists to prevent."""
        import omlx.admin.hf_downloader as dl_module
        from omlx.admin.hf_downloader import _WireCounter

        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.01)
        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 5)
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 5)
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(
            task_id="t-resume",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
            total_size=10_000_000,
        )
        downloader._tasks[task.task_id] = task
        counter = _WireCounter()
        downloader._wire_counters[task.task_id] = counter
        resumed = _DownloadActivity(
            file_count=1,
            logical_size=6_000_000,
            allocated_size=6_000_000,
            latest_mtime_ns=1,
            files={"big.safetensors": 6_000_000},
        )

        with patch.object(
            downloader, "_get_download_activity", return_value=resumed
        ):
            poll = asyncio.create_task(
                downloader._poll_progress(task.task_id, model_dir)
            )
            counter.add(1_000_000)
            await asyncio.sleep(0.05)
            assert task.downloaded_size == 7_000_000
            assert task.progress == 70.0

            counter.add(2_000_000)
            await asyncio.sleep(0.05)
            # The wire adds the bytes the disk has not caught up with; the
            # on-disk reading is the floor, never regressed by stale wire
            # bytes, and the total caps the report.
            assert task.downloaded_size >= 8_000_000
            assert task.progress >= 80.0

            task.status = DownloadStatus.COMPLETED
            await poll

    @pytest.mark.asyncio
    async def test_reconstruction_phase_keeps_the_disk_reading(
        self, model_dir, monkeypatch
    ):
        """Once the transfer bar stops (fetch done) the disk leads: wire
        bytes never push the report below the on-disk size."""
        import omlx.admin.hf_downloader as dl_module
        from omlx.admin.hf_downloader import _WireCounter

        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.01)
        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 5)
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 5)
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(
            task_id="t-disk",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
            total_size=10_000_000,
        )
        downloader._tasks[task.task_id] = task
        counter = _WireCounter()
        counter.add(2_000_000)  # fetch delivered only part over the wire
        downloader._wire_counters[task.task_id] = counter
        on_disk = _DownloadActivity(
            file_count=1,
            logical_size=8_000_000,
            allocated_size=8_000_000,
            latest_mtime_ns=1,
        )

        with patch.object(
            downloader, "_get_download_activity", return_value=on_disk
        ):
            poll = asyncio.create_task(
                downloader._poll_progress(task.task_id, model_dir)
            )
            await asyncio.sleep(0.05)

            # The wire adds the bytes the disk has not caught up with; the
            # on-disk reading is the floor, never regressed by stale wire
            # bytes, and the total caps the report.
            assert task.downloaded_size >= 8_000_000
            assert task.progress >= 80.0

            task.status = DownloadStatus.COMPLETED
            await poll


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
            await _wait_for_downloads(downloader)

            assert mock_download.call_count == 2
            # Last call is the actual download
            call_kwargs = mock_download.call_args[1]
            assert "dry_run" not in call_kwargs
            assert call_kwargs["etag_timeout"] == 30

            await downloader.shutdown()


# =============================================================================
# Endpoint resolution (_resolve_endpoint)
# =============================================================================
#
# Background: `huggingface_hub` does not follow cross-origin permanent
# redirects during the HEAD probe it issues at the start of a download
# (e.g. hf-mirror.com permanently 308s to huggingface.co when accessed
# from non-CN IPs). The result is a silent download failure with a
# misleading error. `_resolve_endpoint` probes the configured endpoint
# upfront, walks the redirect chain, and pins HfApi to the final origin.
#
# These tests pin that behavior so a future refactor doesn't regress it.


class TestResolveEndpoint:
    """Pin the cross-origin redirect resolution for HF_ENDPOINT."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        # Cache is module-global; clear before/after every test so cases
        # don't bleed into each other.
        from omlx.admin.hf_downloader import _endpoint_resolution_cache
        _endpoint_resolution_cache.clear()
        yield
        _endpoint_resolution_cache.clear()

    @staticmethod
    def _response(status_code: int, location: str | None = None) -> MagicMock:
        r = MagicMock()
        r.status_code = status_code
        r.headers = {"location": location} if location else {}
        return r

    def _patch_httpx(self, responses: list):
        """Patch httpx.Client.head to walk through `responses` in order."""
        mock_client_cls = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.head = MagicMock(side_effect=responses)
        mock_client_cls.return_value = mock_client
        return patch("httpx.Client", mock_client_cls), mock_client

    def test_no_redirect_returns_endpoint_unchanged(self):
        from omlx.admin.hf_downloader import _resolve_endpoint
        ctx, _ = self._patch_httpx([self._response(200)])
        with ctx:
            assert _resolve_endpoint("https://huggingface.co") == "https://huggingface.co"

    def test_cross_origin_308_returns_redirected_origin(self):
        # The bug this whole module exists to fix: hf-mirror permanently
        # 308s to huggingface.co; downloads must resolve to the final origin.
        from omlx.admin.hf_downloader import _resolve_endpoint
        ctx, _ = self._patch_httpx([
            self._response(308, "https://huggingface.co/api/models/gpt2"),
            self._response(200),  # probe at resolved origin
        ])
        with ctx:
            assert _resolve_endpoint("https://hf-mirror.com") == "https://huggingface.co"

    def test_cross_origin_301_also_handled(self):
        # 301 (Moved Permanently) gets the same treatment as 308.
        from omlx.admin.hf_downloader import _resolve_endpoint
        ctx, _ = self._patch_httpx([
            self._response(301, "https://huggingface.co/api/models/gpt2"),
            self._response(200),
        ])
        with ctx:
            assert _resolve_endpoint("https://hf-mirror.com") == "https://huggingface.co"

    def test_same_origin_redirect_does_not_rewrite(self):
        # If the server returns a relative Location (`/foo`) we must not
        # try to rewrite the endpoint — same origin, same hostname.
        from omlx.admin.hf_downloader import _resolve_endpoint
        ctx, _ = self._patch_httpx([
            self._response(308, "/api/models/gpt2"),
        ])
        with ctx:
            assert _resolve_endpoint("https://hf-mirror.com") == "https://hf-mirror.com"

    def test_chained_redirects_walk_up_to_3_hops(self):
        # A → B → C all cross-origin permanent. Final hop wins.
        from omlx.admin.hf_downloader import _resolve_endpoint
        ctx, _ = self._patch_httpx([
            self._response(308, "https://hop2.example/api/models/gpt2"),
            self._response(308, "https://huggingface.co/api/models/gpt2"),
            self._response(200),
        ])
        with ctx:
            assert _resolve_endpoint("https://hop1.example") == "https://huggingface.co"

    def test_temporary_redirect_is_not_followed(self):
        # 302 / 307 are NOT permanent — leave the endpoint alone so the HF
        # client can handle them per-request.
        from omlx.admin.hf_downloader import _resolve_endpoint
        ctx, _ = self._patch_httpx([
            self._response(302, "https://huggingface.co/api/models/gpt2"),
        ])
        with ctx:
            assert _resolve_endpoint("https://hf-mirror.com") == "https://hf-mirror.com"

    def test_network_error_falls_back_to_original_endpoint(self):
        # Best-effort probe: any httpx exception leaves the endpoint as-is.
        from omlx.admin.hf_downloader import _resolve_endpoint
        mock_client_cls = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.head = MagicMock(side_effect=OSError("network unreachable"))
        mock_client_cls.return_value = mock_client
        with patch("httpx.Client", mock_client_cls):
            assert _resolve_endpoint("https://hf-mirror.com") == "https://hf-mirror.com"

    def test_result_is_cached_per_endpoint(self):
        # Second call for the same endpoint must not re-probe.
        from omlx.admin.hf_downloader import _resolve_endpoint
        ctx, mock_client = self._patch_httpx([
            self._response(308, "https://huggingface.co/api/models/gpt2"),
            self._response(200),
        ])
        with ctx:
            _resolve_endpoint("https://hf-mirror.com")
            _resolve_endpoint("https://hf-mirror.com")
        assert mock_client.head.call_count == 2  # one probe + one resolved probe

    def test_trailing_slash_normalized(self):
        # `https://hf-mirror.com/` and `https://hf-mirror.com` are the same
        # endpoint and must share the cache.
        from omlx.admin.hf_downloader import _resolve_endpoint
        ctx, mock_client = self._patch_httpx([
            self._response(308, "https://huggingface.co/api/models/gpt2"),
            self._response(200),
        ])
        with ctx:
            r1 = _resolve_endpoint("https://hf-mirror.com")
            r2 = _resolve_endpoint("https://hf-mirror.com/")
        assert r1 == r2 == "https://huggingface.co"
        # Second call was a cache hit — head() count unchanged from first probe.
        assert mock_client.head.call_count == 2


# =============================================================================
# Queue Persistence and Restart Resume
# =============================================================================


class TestQueuePersistence:
    """The queue persists to disk and survives a restart."""

    @pytest.fixture
    def tasks_file(self, tmp_path):
        return tmp_path / "state" / "hf_download_tasks.json"

    @pytest.fixture
    def downloader(self, tmp_path, tasks_file):
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        return HFDownloader(model_dir=str(model_dir), tasks_file=tasks_file)

    @staticmethod
    def _rows(tasks_file):
        return json.loads(tasks_file.read_text(encoding="utf-8"))

    def test_a_resumable_row_keeps_the_token_and_a_finished_one_does_not(
        self, downloader, tasks_file
    ):
        """Only a row that can still be resumed needs the credential: a
        terminal row keeps it in memory for a retry in this process but is
        written without it, so tokens do not outlive their download."""
        task = DownloadTask(task_id="t1", repo_id="private/model")
        task.token = "hf_SECRET"
        downloader._tasks["t1"] = task

        downloader._persist()
        assert self._rows(tasks_file)[0]["token"] == "hf_SECRET"

        task.status = DownloadStatus.COMPLETED
        downloader._persist()
        assert self._rows(tasks_file)[0]["token"] == ""
        # The credential survives in memory, so a retry in this process can
        # still reach a gated repository.
        assert task.token == "hf_SECRET"

    def test_the_queue_file_is_owner_only_from_the_first_write(
        self, downloader, tasks_file
    ):
        task = DownloadTask(task_id="t1", repo_id="owner/model")
        task.token = "hf_SECRET"
        downloader._tasks["t1"] = task

        downloader._persist()

        assert stat.S_IMODE(tasks_file.stat().st_mode) == 0o600
        assert not tasks_file.with_name(tasks_file.name + ".tmp").exists()

    @pytest.mark.asyncio
    async def test_start_download_persists_pending_row(
        self, downloader, tasks_file
    ):
        async def _noop(self, task_id, hf_token):
            return None

        with patch.object(HFDownloader, "_run_download", new=_noop):
            task = await downloader.start_download("owner/model")

        rows = self._rows(tasks_file)
        assert len(rows) == 1
        assert rows[0]["task_id"] == task.task_id
        assert rows[0]["repo_id"] == "owner/model"
        assert rows[0]["status"] == DownloadStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_cancel_persists_cancelled_row(self, downloader, tasks_file):
        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._tasks[task.task_id] = task
        downloader._persist()

        with patch("omlx.admin.hf_downloader.abort_xet_session"):
            assert await downloader.cancel_download(task.task_id) is True

        rows = self._rows(tasks_file)
        assert rows[0]["status"] == DownloadStatus.CANCELLED.value

    @pytest.mark.asyncio
    async def test_remove_task_persists_without_row(
        self, downloader, tasks_file
    ):
        task = DownloadTask(
            task_id="t1",
            repo_id="owner/model",
            status=DownloadStatus.COMPLETED,
        )
        downloader._tasks[task.task_id] = task
        downloader._persist()

        assert downloader.remove_task(task.task_id) is True
        assert self._rows(tasks_file) == []

    @pytest.mark.asyncio
    async def test_failed_run_persists_failed_row(self, downloader, tasks_file):
        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        mock_api = MagicMock()
        mock_api.model_info.side_effect = Exception("boom")

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(mock_api, None),
        ):
            await downloader._run_download(task.task_id, "")

        assert task.status == DownloadStatus.FAILED
        rows = self._rows(tasks_file)
        assert rows[0]["status"] == DownloadStatus.FAILED.value
        # The repo-info fallback may rewrite the raw error into a
        # repository-not-found style message; only persistence matters here
        # (error round-trip is covered by the restore test below).
        assert rows[0]["error"]

    @pytest.mark.asyncio
    async def test_shutdown_leaves_row_resumable_on_disk(
        self, downloader, tasks_file
    ):
        async def _noop(self, task_id, hf_token):
            return None

        with patch.object(HFDownloader, "_run_download", new=_noop):
            task = await downloader.start_download("owner/model")

        task.status = DownloadStatus.DOWNLOADING
        with patch("omlx.admin.hf_downloader.abort_xet_session"):
            await downloader.shutdown()

        # The in-memory row went CANCELLED, but shutdown must not write that:
        # the on-disk row keeps its queued status so the next boot resumes.
        rows = self._rows(tasks_file)
        assert rows[0]["status"] not in (
            DownloadStatus.CANCELLED.value,
            DownloadStatus.FAILED.value,
        )

    @pytest.mark.asyncio
    async def test_restore_resumes_interrupted_and_keeps_terminal(
        self, downloader, tasks_file
    ):
        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        tasks_file.write_text(
            json.dumps(
                [
                    {
                        "task_id": "done",
                        "repo_id": "owner/done",
                        "status": "completed",
                        "progress": 100.0,
                        "created_at": 100.0,
                    },
                    {
                        "task_id": "fail",
                        "repo_id": "owner/fail",
                        "status": "failed",
                        "error": "boom",
                        "created_at": 200.0,
                    },
                    {
                        "task_id": "live",
                        "repo_id": "owner/live",
                        "status": "downloading",
                        "created_at": 300.0,
                        "retry_count": 2,
                    },
                ]
            ),
            encoding="utf-8",
        )

        async def _noop(self, task_id, hf_token):
            return None

        with patch.object(HFDownloader, "_run_download", new=_noop):
            await downloader.restore_tasks()

        resumed = [
            t for t in downloader._tasks.values()
            if t.status == DownloadStatus.PENDING
        ]
        assert [t.repo_id for t in resumed] == ["owner/live"]
        assert resumed[0].created_at == 300.0
        assert resumed[0].retry_count == 2
        # Terminal rows come back as display-only entries, error text intact.
        assert downloader._tasks["done"].status == DownloadStatus.COMPLETED
        assert downloader._tasks["done"].speed_bps == 0.0
        assert downloader._tasks["fail"].error == "boom"
        # The rewritten queue records the resumed row as pending under its
        # new task id (task ids are restart-scoped; rows are matched by repo).
        live_rows = [
            r for r in self._rows(tasks_file)
            if r["repo_id"] == "owner/live"
        ]
        assert live_rows
        assert live_rows[0]["status"] == DownloadStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_restore_tolerates_missing_and_corrupt_files(
        self, downloader, tasks_file
    ):
        await downloader.restore_tasks()  # missing file: no-op

        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        tasks_file.write_text("{not json", encoding="utf-8")
        await downloader.restore_tasks()  # corrupt file: no-op, no raise
        assert downloader._tasks == {}

        tasks_file.write_text('{"not": "a list"}', encoding="utf-8")
        await downloader.restore_tasks()
        assert downloader._tasks == {}

    @pytest.mark.asyncio
    async def test_restore_warns_about_a_bad_row_without_its_token(
        self, downloader, tasks_file, caplog
    ):
        """A row the loader cannot read is skipped, not fatal — and the row may
        still carry the credential, so the warning leaves it out."""
        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        tasks_file.write_text(
            json.dumps([{
                "task_id": "t1",
                "status": "completed",
                "token": "hf_SUPERSECRET",
                # repo_id is what from_dict reads first; its absence is the
                # KeyError this path already tolerates.
            }]),
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING):
            await downloader.restore_tasks()

        assert "Skipping unpersistable download row" in caplog.text
        assert "hf_SUPERSECRET" not in caplog.text

    @pytest.mark.asyncio
    async def test_one_row_this_build_cannot_read_does_not_lose_the_queue(
        self, downloader, tasks_file
    ):
        """`restore_tasks` promises never to raise, so one field a build that
        stored it differently left behind skips that row's bookkeeping alone:
        the rows behind it come back and the healing rewrite runs."""
        rows = [
            # A queue row interrupted mid-download, carrying the two fields a
            # newer build could have changed the type of.
            {"task_id": "live", "repo_id": "owner/live",
             "status": DownloadStatus.DOWNLOADING.value,
             "created_at": "2026-09-25T00:00:00", "retry_count": "x"},
            {"task_id": "done", "repo_id": "owner/done",
             "status": DownloadStatus.COMPLETED.value,
             "created_at": 5.0, "retry_count": 1},
            # A status this build does not know is failed and display-only,
            # not a queued row this boot never starts.
            {"task_id": "no-status", "repo_id": "owner/unknown",
             "created_at": 7.0},
        ]
        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        tasks_file.write_text(json.dumps(rows), encoding="utf-8")

        async def _noop(self, task_id, hf_token):
            return None

        with patch.object(HFDownloader, "_run_download", new=_noop):
            await downloader.restore_tasks()

        by_repo = {task.repo_id: task for task in downloader._tasks.values()}
        assert set(by_repo) == {"owner/live", "owner/done", "owner/unknown"}
        assert by_repo["owner/live"].status in (
            DownloadStatus.PENDING,
            DownloadStatus.DOWNLOADING,
        )
        assert by_repo["owner/live"].retry_count == 0
        assert by_repo["owner/done"].status == DownloadStatus.COMPLETED
        assert by_repo["owner/unknown"].status == DownloadStatus.FAILED

        # The rewrite at the end of the restore ran, so the next boot reads a
        # queue this one already healed rather than failing the same way.
        written = {row["repo_id"]: row for row in self._rows(tasks_file)}
        assert set(written) == set(by_repo)
        assert written["owner/unknown"]["status"] == DownloadStatus.FAILED.value

    @pytest.mark.asyncio
    async def test_restore_resumes_duplicate_interrupted_repo_once(
        self, downloader, tasks_file
    ):
        row = {
            "task_id": "x",
            "repo_id": "owner/dup",
            "status": "downloading",
            "created_at": 100.0,
        }
        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        tasks_file.write_text(
            json.dumps([dict(row, task_id="a"), dict(row, task_id="b")]),
            encoding="utf-8",
        )

        async def _noop(self, task_id, hf_token):
            return None

        with patch.object(HFDownloader, "_run_download", new=_noop):
            await downloader.restore_tasks()  # duplicate must not raise

        active = [
            t for t in downloader._tasks.values()
            if t.status == DownloadStatus.PENDING
        ]
        assert len(active) == 1
        assert active[0].repo_id == "owner/dup"

    @pytest.mark.asyncio
    async def test_row_persists_credential_but_api_never_exposes_it(
        self, downloader, tasks_file
    ):
        async def _noop(self, task_id, hf_token):
            return None

        with patch.object(HFDownloader, "_run_download", new=_noop):
            task = await downloader.start_download("owner/model", "hf_secret")

        rows = self._rows(tasks_file)
        assert rows[0]["token"] == "hf_secret"
        # The queue API serves to_dict() output — a credential must never
        # travel back to a client.
        assert "token" not in task.to_dict()
        assert all("token" not in row for row in downloader.get_tasks())
        # The row is a credential store now: owner-only file.
        if os.name != "nt":
            assert tasks_file.stat().st_mode & 0o777 == 0o600

    @pytest.mark.asyncio
    async def test_restore_resumes_with_persisted_credential(
        self, downloader, tasks_file
    ):
        """A gated download restarts with the token its request supplied."""
        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        tasks_file.write_text(
            json.dumps(
                [
                    {
                        "task_id": "live",
                        "repo_id": "owner/gated",
                        "status": "downloading",
                        "created_at": 100.0,
                        "token": "hf_secret",
                    }
                ]
            ),
            encoding="utf-8",
        )

        seen: dict = {}

        async def _noop(self, task_id, hf_token):
            seen["token"] = hf_token

        with patch.object(HFDownloader, "_run_download", new=_noop):
            await downloader.restore_tasks()
        # restore only schedules the download coroutine — yield once so the
        # patched run body actually executes before we inspect it.
        await asyncio.sleep(0)

        assert seen["token"] == "hf_secret"
        resumed = [
            t for t in downloader._tasks.values()
            if t.status == DownloadStatus.PENDING
        ]
        assert [t.repo_id for t in resumed] == ["owner/gated"]
        assert resumed[0].token == "hf_secret"
        # The credential survives the restore rewrite for the next restart.
        live_rows = [
            r for r in self._rows(tasks_file)
            if r["repo_id"] == "owner/gated"
        ]
        assert live_rows[0]["token"] == "hf_secret"

    @pytest.mark.asyncio
    async def test_restore_without_token_row_falls_back_to_empty(
        self, downloader, tasks_file
    ):
        """Rows written before the token field keep hub's env/login lookup."""
        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        tasks_file.write_text(
            json.dumps(
                [
                    {
                        "task_id": "live",
                        "repo_id": "owner/model",
                        "status": "pending",
                        "created_at": 100.0,
                    }
                ]
            ),
            encoding="utf-8",
        )

        seen = "unset"

        async def _noop(self, task_id, hf_token):
            nonlocal seen
            seen = hf_token

        with patch.object(HFDownloader, "_run_download", new=_noop):
            await downloader.restore_tasks()
        await asyncio.sleep(0)  # let the scheduled download coroutine run

        assert seen == ""  # start_download maps "" to token=None for hub

    @pytest.mark.asyncio
    async def test_retry_recovers_credential_and_persists_bookkeeping(
        self, downloader, tasks_file
    ):
        old = DownloadTask(
            task_id="old",
            repo_id="owner/gated",
            status=DownloadStatus.FAILED,
            token="hf_secret",
        )
        downloader._tasks["old"] = old
        downloader._persist()

        async def _noop(self, task_id, hf_token):
            return None

        # Retry without a token (the app sends none): the stored credential
        # is kept instead of being wiped to "".
        with patch.object(HFDownloader, "_run_download", new=_noop):
            kept = await downloader.retry_download("old", "")
        assert kept.token == "hf_secret"
        assert kept.retry_count == 1
        rows = {r["task_id"]: r for r in self._rows(tasks_file)}
        assert rows[kept.task_id]["token"] == "hf_secret"
        # The retry bookkeeping is on disk immediately, not on some later
        # event — restarting right now must not lose the count.
        assert rows[kept.task_id]["retry_count"] == 1

        # Retry with a freshly re-entered token (the web form): the new
        # credential replaces the stale one on disk.
        kept.status = DownloadStatus.FAILED
        with patch.object(HFDownloader, "_run_download", new=_noop):
            replaced = await downloader.retry_download(kept.task_id, "hf_new")
        assert replaced.token == "hf_new"
        assert replaced.retry_count == 2
        rows = {r["task_id"]: r for r in self._rows(tasks_file)}
        assert rows[replaced.task_id]["token"] == "hf_new"
        assert rows[replaced.task_id]["retry_count"] == 2


# =============================================================================
# Xet Group Capture and Cancellation
# =============================================================================


class TestXetGroupCancellation:
    """Cancelling in-flight work aborts every recorded xet group."""

    @pytest.fixture
    def downloader(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        return HFDownloader(model_dir=str(model_dir))

    def setup_method(self):
        hf_downloader_mod._xet_groups.clear()

    def teardown_method(self):
        hf_downloader_mod._xet_groups.clear()

    def test_session_proxy_records_new_groups_and_delegates(self):
        inner = MagicMock()
        group = MagicMock()
        group.__enter__.return_value = group
        inner.new_file_download_group.return_value = group

        proxy = hf_downloader_mod._XetSessionProxy(inner)
        got = proxy.new_file_download_group(endpoint="ep")

        inner.new_file_download_group.assert_called_once_with(endpoint="ep")
        # hub gets a tracked wrapper; the real group is registered the moment
        # it is created and stays registered while its `with` block is open.
        assert isinstance(got, hf_downloader_mod._TrackedXetGroup)
        assert list(hf_downloader_mod._xet_groups) == [group]
        with got as entered:
            assert entered is group
            assert list(hf_downloader_mod._xet_groups) == [group]
        # Settled groups leave the registry: abort would find no work there.
        group.__exit__.assert_called_once_with(None, None, None)
        assert list(hf_downloader_mod._xet_groups) == []
        # Every other session attribute delegates to the real session.
        assert proxy.status is inner.status

    def test_install_is_idempotent_and_delegates(self):
        import huggingface_hub.utils._xet as hub_xet

        sentinel = object()
        original = hub_xet.get_xet_session
        try:
            # Fresh un-wrapped function, so a wrapper installed by an earlier
            # test cannot shadow this one.
            hub_xet.get_xet_session = lambda: sentinel
            hf_downloader_mod._install_xet_group_capture()
            hf_downloader_mod._install_xet_group_capture()

            proxy = hub_xet.get_xet_session()
            assert isinstance(proxy, hf_downloader_mod._XetSessionProxy)
            assert proxy._inner is sentinel
        finally:
            hub_xet.get_xet_session = original

    def test_abort_xet_transfers_aborts_recorded_group_once(self):
        group = MagicMock()
        hf_downloader_mod._register_xet_group(group)

        assert hf_downloader_mod._abort_xet_transfers() is True
        group.abort.assert_called_once()
        assert hf_downloader_mod._xet_groups == []
        # Second call finds nothing left to abort.
        assert hf_downloader_mod._abort_xet_transfers() is False
        group.abort.assert_called_once()

    def test_abort_xet_transfers_swallows_stale_group_errors(self):
        group = MagicMock()
        group.abort.side_effect = RuntimeError("stale group")
        hf_downloader_mod._register_xet_group(group)

        assert hf_downloader_mod._abort_xet_transfers() is False
        group.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_active_download_aborts_registered_group(
        self, downloader
    ):
        group = MagicMock()
        hf_downloader_mod._register_xet_group(group)
        task = DownloadTask(
            task_id="t1",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
        )
        downloader._tasks[task.task_id] = task
        active = asyncio.create_task(asyncio.sleep(10))
        downloader._active_tasks[task.task_id] = active

        with patch(
            "omlx.admin.hf_downloader.abort_xet_session"
        ) as mock_abort:
            assert await downloader.cancel_download(task.task_id) is True

        group.abort.assert_called_once()
        mock_abort.assert_called_once()
        with pytest.raises(asyncio.CancelledError):
            await active

    @pytest.mark.asyncio
    async def test_cancel_pending_download_keeps_registered_group(
        self, downloader
    ):
        group = MagicMock()
        hf_downloader_mod._register_xet_group(group)
        task = DownloadTask(
            task_id="t1",
            repo_id="owner/model",
            status=DownloadStatus.PENDING,
        )
        downloader._tasks[task.task_id] = task

        with patch("omlx.admin.hf_downloader.abort_xet_session"):
            assert await downloader.cancel_download(task.task_id) is True

        group.abort.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_aborts_registered_group(self, downloader):
        """shutdown() must stop the Rust transfer, not just flip the sigint.

        The sigint flag alone can leave a reconstruction running, and the
        interpreter would then wait for that non-daemon writer thread on
        exit — a graceful restart mid-download has to unwind promptly.
        """
        group = MagicMock()
        hf_downloader_mod._register_xet_group(group)

        with patch("omlx.admin.hf_downloader.abort_xet_session"):
            await downloader.shutdown()

        group.abort.assert_called_once()
        assert hf_downloader_mod._xet_groups == []

    @pytest.mark.asyncio
    async def test_cancel_aborts_every_concurrently_active_group(
        self, downloader
    ):
        """snapshot_download shards files across hf_thread_map workers.

        One xet_get() (hence one group) runs per file, concurrently, so the
        registry holds several live groups at once and cancel must abort
        all of them — aborting only the newest would leave the other shards
        ghost-running.
        """
        inner = MagicMock()
        groups = [MagicMock(), MagicMock()]
        inner.new_file_download_group.side_effect = list(groups)
        proxy = hf_downloader_mod._XetSessionProxy(inner)

        for _ in range(len(groups)):  # two shard downloads open at once
            proxy.new_file_download_group()
        assert list(hf_downloader_mod._xet_groups) == groups

        task = DownloadTask(
            task_id="t1",
            repo_id="owner/model",
            status=DownloadStatus.DOWNLOADING,
        )
        downloader._tasks[task.task_id] = task
        active = asyncio.create_task(asyncio.sleep(10))
        downloader._active_tasks[task.task_id] = active

        with patch(
            "omlx.admin.hf_downloader.abort_xet_session"
        ) as mock_abort:
            assert await downloader.cancel_download(task.task_id) is True

        for group in groups:
            group.abort.assert_called_once()
        assert hf_downloader_mod._xet_groups == []
        mock_abort.assert_called_once()
        with pytest.raises(asyncio.CancelledError):
            await active

    @pytest.mark.asyncio
    async def test_shutdown_aborts_every_concurrently_active_group(
        self, downloader
    ):
        """Shutdown must reap all open shard groups, then settle cleanly.

        A writer thread parked in any group's reconstruction would block
        interpreter exit, so every live group gets aborted — and the groups'
        later `with` exits must not trip over the cleared registry.
        """
        inner = MagicMock()
        groups = [MagicMock(), MagicMock()]
        inner.new_file_download_group.side_effect = list(groups)
        proxy = hf_downloader_mod._XetSessionProxy(inner)

        open_groups = [
            proxy.new_file_download_group() for _ in range(len(groups))
        ]
        for tracked in open_groups:
            tracked.__enter__()  # shards mid-transfer

        with patch("omlx.admin.hf_downloader.abort_xet_session"):
            await downloader.shutdown()

        for group in groups:
            group.abort.assert_called_once()
        assert hf_downloader_mod._xet_groups == []

        # The with-blocks settle after the abort without tripping anything.
        for tracked in open_groups:
            tracked.__exit__(None, None, None)
        assert hf_downloader_mod._xet_groups == []

    def test_group_deregisters_when_enter_fails(self):
        """A failed CAS handshake must not leak a stale registry entry.

        The `with` statement skips __exit__ when __enter__ raises, so the
        wrapper has to unregister itself on that path.
        """
        inner = MagicMock()
        group = MagicMock()
        group.__enter__.side_effect = RuntimeError("handshake failed")
        inner.new_file_download_group.return_value = group
        proxy = hf_downloader_mod._XetSessionProxy(inner)

        tracked = proxy.new_file_download_group()
        with pytest.raises(RuntimeError):
            with tracked:
                pass

        assert list(hf_downloader_mod._xet_groups) == []

    def test_late_group_is_aborted_and_the_flag_dies_with_its_call(self):
        """A group opened after the abort is stopped, a later call's is not.

        abort_xet_session() only drops the session, so a snapshot_download
        call that reached its first xet_get() after the cancel opens its
        group on a fresh session that nothing is left to abort. The call is
        flagged instead, and that flag has to die with the call rather than
        catch the download that starts after it.
        """
        inner = MagicMock()
        late, next_call = MagicMock(), MagicMock()
        inner.new_file_download_group.side_effect = [late, next_call]
        proxy = hf_downloader_mod._XetSessionProxy(inner)

        def aborted_call(**kwargs):
            # The cancel lands while this call is registered but before it has
            # opened a group: the registry it snapshots is empty.
            assert hf_downloader_mod._abort_xet_transfers() is False
            proxy.new_file_download_group()
            return []

        hf_downloader_mod._tracked_snapshot_download(
            aborted_call, lambda: False
        )
        late.abort.assert_called_once()

        def next_download(**kwargs):
            proxy.new_file_download_group()
            return []

        hf_downloader_mod._tracked_snapshot_download(
            next_download, lambda: False
        )
        next_call.abort.assert_not_called()

    def test_tracked_call_flags_itself_when_already_cancelled(self):
        """A cancel that landed before the worker registered still stops it.

        The to_thread job is submitted before its awaiter can be cancelled,
        so by the time the worker runs the abort that cleared the (empty)
        registry is history; the call has to notice the cancel itself.
        """
        inner = MagicMock()
        group = MagicMock()
        inner.new_file_download_group.return_value = group
        proxy = hf_downloader_mod._XetSessionProxy(inner)

        def call(**kwargs):
            proxy.new_file_download_group()
            return []

        hf_downloader_mod._tracked_snapshot_download(call, lambda: True)

        group.abort.assert_called_once()

    def test_abort_reaches_only_the_call_that_was_cancelled(self):
        """A download started while an aborted worker unwinds keeps its groups.

        Task cancellation does not stop the worker, so the semaphore can be
        released and the next download can reach its first xet_get() before
        the cancelled one does. That next download must not inherit the
        abort: aborting its group mid-handshake turns its xet_get() into a
        "User cancelled" failure.
        """
        inner = MagicMock()
        old_group, new_group = MagicMock(), MagicMock()
        inner.new_file_download_group.side_effect = [new_group, old_group]
        proxy = hf_downloader_mod._XetSessionProxy(inner)

        aborted_call = hf_downloader_mod._open_xet_call()
        new_call = hf_downloader_mod._open_xet_call()
        hf_downloader_mod._abort_xet_call(aborted_call)
        previous = getattr(hf_downloader_mod._xet_call_in_thread, "call", None)
        try:
            # The replacement download gets there first, then the cancelled
            # worker finally reaches its own first xet_get().
            hf_downloader_mod._mark_xet_call(new_call)
            proxy.new_file_download_group()
            hf_downloader_mod._mark_xet_call(aborted_call)
            proxy.new_file_download_group()
        finally:
            hf_downloader_mod._mark_xet_call(previous)
            hf_downloader_mod._close_xet_call(aborted_call)
            hf_downloader_mod._close_xet_call(new_call)

        new_group.abort.assert_not_called()
        old_group.abort.assert_called_once()

    def test_hub_per_file_workers_inherit_the_call_they_serve(self):
        """hub opens groups on its own file workers, so they need the mark.

        snapshot_download maps files across a ThreadPoolExecutor; without
        wrapping that map the per-file threads carry no call, the late group
        goes unattributed, and the abort that was supposed to stop it finds
        nothing (or stops the wrong download).
        """
        from huggingface_hub import _snapshot_download as hub_snapshot
        from huggingface_hub.utils.tqdm import hf_thread_map as pristine

        original = hub_snapshot.hf_thread_map
        call = hf_downloader_mod._open_xet_call()
        previous = getattr(hf_downloader_mod._xet_call_in_thread, "call", None)
        try:
            hub_snapshot.hf_thread_map = pristine
            hf_downloader_mod._install_xet_call_marking()
            hf_downloader_mod._install_xet_call_marking()  # idempotent
            marked = hub_snapshot.hf_thread_map
            assert getattr(marked, "_omlx_call_marking", False)

            hf_downloader_mod._mark_xet_call(call)

            seen = []

            def worker(_item):
                seen.append(
                    getattr(hf_downloader_mod._xet_call_in_thread, "call", None)
                )

            marked(worker, [1, 2], disable=True)

            assert seen == [call, call]
        finally:
            hub_snapshot.hf_thread_map = original
            hf_downloader_mod._mark_xet_call(previous)
            hf_downloader_mod._close_xet_call(call)

    @pytest.mark.asyncio
    async def test_group_opened_after_cancel_is_aborted(self, downloader):
        """A cancel that beats the call's first xet_get() must still abort it.

        Task cancellation does not stop the snapshot_download worker, so the
        worker can reach its first xet_get() after cancel_download() returned.
        By then the registry it emptied is still empty and abort_xet_session()
        has already replaced the session, so the group opened on that fresh
        session must be aborted the moment it appears instead of ghost-running
        for the life of the process.
        """
        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_info.safetensors = {}
        mock_info.siblings = []
        mock_api.model_info.return_value = mock_info

        group = MagicMock()
        group.__enter__.return_value = group
        inner = MagicMock()
        inner.new_file_download_group.return_value = group

        worker_started = threading.Event()
        release = threading.Event()
        group_opened = threading.Event()

        def fake_snapshot_download(**kwargs):
            if kwargs.get("dry_run"):
                return []
            worker_started.set()
            assert release.wait(5), "test never released the worker"
            # Only now does the call reach its first xet_get(): the cancel has
            # long returned and its abort_xet_session() dropped the session.
            hf_downloader_mod._XetSessionProxy(
                inner
            ).new_file_download_group(endpoint="ep")
            group_opened.set()
            return []

        try:
            with patch(
                "omlx.admin.hf_downloader._get_hf_api",
                return_value=(mock_api, None),
            ), patch(
                "omlx.admin.hf_downloader.snapshot_download",
                side_effect=fake_snapshot_download,
            ), patch(
                "omlx.admin.hf_downloader.abort_xet_session"
            ) as mock_abort:
                active = asyncio.create_task(
                    downloader._run_download(task.task_id, "")
                )
                downloader._active_tasks[task.task_id] = active
                assert await asyncio.to_thread(worker_started.wait, 5)

                assert await downloader.cancel_download(task.task_id) is True
                mock_abort.assert_called_once()
                # The abort found nothing: the call has no group open yet and
                # the group it opens next does not exist anywhere.
                assert hf_downloader_mod._xet_groups == []
                group.abort.assert_not_called()

                release.set()
                await active
                assert await asyncio.to_thread(group_opened.wait, 5)
                group.abort.assert_called_once()
        finally:
            release.set()
