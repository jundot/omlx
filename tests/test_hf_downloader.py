# SPDX-License-Identifier: Apache-2.0
"""Tests for the HuggingFace model downloader."""

import asyncio
import contextlib
import fcntl
import io
import json
import os
import shutil
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import huggingface_hub
import pytest
from fastapi import HTTPException
from huggingface_hub import HfApi
from huggingface_hub import constants as hub_constants
from huggingface_hub import file_download as hub_file_download
from huggingface_hub.utils import HfHubHTTPError

from omlx._hf_download_worker import _download_without_xet
from omlx.admin import hf_downloader as hf_downloader_mod
from omlx.admin import routes as admin_routes
from omlx.admin.hf_downloader import (
    DownloadStatus,
    DownloadTask,
    HFDownloader,
    _DownloadActivity,
    _DownloadCancelled,
    _calc_safetensors_disk_size,
    _histogram_has_packed_u32,
    _is_xet_transport_error,
    _SpeedMeter,
    _make_cancellable_tqdm,
    _sum_safetensors_blob_bytes,
)
from omlx.engine_pool import EnginePool


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
        yield SimpleNamespace(call=call, started=started, release=release)
    finally:
        release.set()
        if started.is_set():
            await asyncio.wait_for(finished.wait(), timeout=5)


async def _wait_for_downloads(downloader):
    """Wait for the scheduled downloads instead of guessing their duration."""
    await asyncio.wait_for(
        asyncio.gather(*downloader._active_tasks.values()), timeout=5
    )


def _downloading_task(model_dir, *, task_id="t1", total_size=0):
    """One running task on its own downloader, with no poll started yet."""
    downloader = HFDownloader(model_dir=str(model_dir))
    task = DownloadTask(
        task_id=task_id,
        repo_id="owner/model",
        status=DownloadStatus.DOWNLOADING,
        total_size=total_size,
    )
    downloader._tasks[task.task_id] = task
    return downloader, task


def start_poll(
    monkeypatch, downloader, task, model_dir, *, wire=None, interval=0.01
):
    """Run a poll loop for a test: fast ticks, generous stall deadlines.

    Returns the task; it starts running at the caller's next await.
    """
    monkeypatch.setattr(hf_downloader_mod, "_PROGRESS_POLL_INTERVAL", interval)
    monkeypatch.setattr(hf_downloader_mod, "_STARTUP_STALL_TIMEOUT", 5)
    monkeypatch.setattr(hf_downloader_mod, "_STALL_TIMEOUT", 5)
    return asyncio.create_task(
        downloader._poll_progress(task.task_id, model_dir, wire)
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

    @pytest.mark.parametrize(
        "total, done, rate, want",
        [
            (1000, 400, 20.0, 30),
            (1000, 400, 0.0, None),  # no transfer in the window
            (1000, 1000, 20.0, None),  # waiting for completion
            (0, 400, 20.0, None),  # size not known yet
        ],
    )
    def test_to_dict_eta_seconds(self, total, done, rate, want):
        task = DownloadTask(
            task_id="t",
            repo_id="o/m",
            total_size=total,
            downloaded_size=done,
            eta_speed_bps=rate,
        )
        assert task.to_dict()["eta_seconds"] == want

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
    @pytest.mark.parametrize("repo_id", ["a/b/c", "../..", "a/..", "/etc"])
    async def test_start_download_invalid_repo_id_parts(self, downloader, repo_id):
        with pytest.raises(ValueError, match="Invalid repository ID"):
            await downloader.start_download(repo_id)

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
            assert "valid HF token" in task.error

            await downloader.shutdown()

    # --- Cancel Download ---

    @pytest.mark.asyncio
    async def test_cancel_download(self, downloader, model_dir, blocked_worker):
        target = model_dir / "owner" / "model"
        staging = target / ".cache" / "huggingface" / "download"
        staging.mkdir(parents=True)
        (target / "model-00001-of-00002.safetensors").write_bytes(b"finalized")
        partial = staging / "model-00002-of-00002.safetensors.abc.incomplete"
        partial.write_bytes(b"in-progress")

        with patch(
            "omlx.admin.hf_downloader.HfApi"
        ) as mock_api_cls, patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=blocked_worker.call,
        ), patch(
            "omlx.admin.hf_downloader.abort_xet_session",
            side_effect=blocked_worker.release.set,
        ), patch("omlx.admin.hf_downloader._REAP_INTERVAL", 0.01):
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

            assert not partial.exists()
            assert (target / "model-00001-of-00002.safetensors").exists()

            await downloader.shutdown()

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
    async def test_cancel_keeps_aborting_until_the_worker_returns(
        self, downloader, model_dir
    ):
        """Cancel keeps aborting until the worker returns, not just once."""
        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        mock_api = MagicMock()
        mock_info = MagicMock()
        mock_info.safetensors = {}
        mock_api.model_info.return_value = mock_info

        aborts = []
        worker_started = threading.Event()
        worker_returned = threading.Event()

        def fake_snapshot_download(**kwargs):
            if kwargs.get("dry_run"):
                return []
            worker_started.set()
            # Hub starts the next file on a fresh session after one abort.
            deadline = time.monotonic() + 2
            while len(aborts) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            worker_returned.set()

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(mock_api, None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fake_snapshot_download,
        ), patch(
            "omlx.admin.hf_downloader.abort_xet_session",
            side_effect=lambda: aborts.append(1),
        ), patch("omlx.admin.hf_downloader._REAP_INTERVAL", 0.01):
            active = asyncio.create_task(
                downloader._run_download(task.task_id, "")
            )
            downloader._active_tasks[task.task_id] = active
            while not worker_started.is_set():
                await asyncio.sleep(0.01)

            assert await downloader.cancel_download(task.task_id) is True
            await asyncio.wait_for(active, timeout=5)

        assert worker_returned.is_set(), "the task returned while its writer ran on"
        assert len(aborts) >= 2, "the session the worker rebuilt was never aborted"
        assert task.status == DownloadStatus.CANCELLED

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
    async def test_cleanup_partial_removes_hub_partials_only(self, model_dir):
        """Cleanup deletes hub's .incomplete files; finished files stay for resume."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        target = model_dir / "owner" / "model"
        staging = target / ".cache" / "huggingface" / "download"
        staging.mkdir(parents=True)
        (target / "model-00001-of-00002.safetensors").write_bytes(b"finalized")
        partial = staging / "model-00002-of-00002.safetensors.abc.incomplete"
        partial.write_bytes(b"in-progress")
        metadata = staging / "model-00001-of-00002.safetensors.metadata"
        metadata.write_text("abc\ndef\n")

        task = DownloadTask(task_id="t1", repo_id="owner/model")
        downloader._cleanup_partial(task)

        assert not partial.exists()
        assert metadata.exists()
        assert (target / "model-00001-of-00002.safetensors").exists()

    @pytest.mark.asyncio
    async def test_cleanup_partial_is_noop_when_no_temp_dir(self, model_dir):
        """With no hub staging dir, cleanup leaves the dir untouched."""
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

    @staticmethod
    def _orphan_layout(model_dir):
        """A finished shard, its metadata, and a temp file left by a killed writer."""
        target = model_dir / "owner" / "model"
        staging = target / ".cache" / "huggingface" / "download"
        staging.mkdir(parents=True)
        (target / "model-00001-of-00002.safetensors").write_bytes(b"finalized")
        (staging / "model-00001-of-00002.safetensors.metadata").write_text("a\nb\n")
        (staging / "model-00001-of-00002.safetensors.lock").touch()
        lock = staging / "model-00002-of-00002.safetensors.lock"
        lock.touch()
        orphan = staging / "1a2b3c4d.etag.5e6f7a8b.incomplete"
        orphan.write_bytes(b"x" * 4096)
        return target, lock, orphan

    @pytest.mark.asyncio
    async def test_run_download_drops_orphaned_partials_before_dry_run(self, model_dir):
        """A dead writer's temp file must not be counted as progress."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))
        target, _, orphan = self._orphan_layout(model_dir)
        task = DownloadTask(task_id="t-orphan", repo_id="owner/model")
        downloader._tasks[task.task_id] = task
        orphan_at_dry_run = []

        def fake_snapshot_download(**kwargs):
            if kwargs.get("dry_run"):
                orphan_at_dry_run.append(orphan.exists())
            return []

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(MagicMock(), None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            await downloader._run_download(task.task_id, "")

        assert orphan_at_dry_run == [False]
        assert (target / "model-00001-of-00002.safetensors").exists()
        assert (
            target
            / ".cache/huggingface/download/model-00001-of-00002.safetensors.metadata"
        ).exists()
        assert task.status == DownloadStatus.COMPLETED

    def test_orphan_cleanup_waits_for_a_live_hub_writer(self, model_dir):
        """A temp file stays while another writer holds a hub lock in the folder."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))
        _, lock, partial = self._orphan_layout(model_dir)
        task = DownloadTask(task_id="t1", repo_id="owner/model")

        with open(lock) as holder:
            fcntl.flock(holder, fcntl.LOCK_EX)
            downloader._cleanup_partial(task, orphans_only=True)
            assert partial.exists()

        downloader._cleanup_partial(task, orphans_only=True)
        assert not partial.exists()

    @pytest.mark.asyncio
    async def test_failed_download_drops_partials_its_writers_left(self, model_dir):
        """A writer killed mid-file leaves a temp file no later attempt resumes."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))
        task = DownloadTask(task_id="t-failed", repo_id="owner/model")
        downloader._tasks[task.task_id] = task
        staging = model_dir / "owner" / "model" / ".cache" / "huggingface" / "download"
        partial = staging / "1a2b3c4d.etag.5e6f7a8b.incomplete"

        def fake_snapshot_download(**kwargs):
            if kwargs.get("dry_run"):
                return []
            staging.mkdir(parents=True, exist_ok=True)
            partial.write_bytes(b"x" * 4096)
            raise RuntimeError("connection reset by peer")

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(MagicMock(), None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            await downloader._run_download(task.task_id, "")

        assert task.status == DownloadStatus.FAILED
        assert not partial.exists()

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
    async def test_dry_run_failure_counts_every_fetched_file(self, model_dir):
        """The estimate covers every file the snapshot fetches, not only weights."""
        model_dir.mkdir(parents=True, exist_ok=True)
        downloader = HFDownloader(model_dir=str(model_dir))

        task = DownloadTask(task_id="t-all-files", repo_id="owner/model")
        downloader._tasks[task.task_id] = task

        mock_info = MagicMock()
        mock_info.safetensors = {"parameters": {"BF16": 5_000}, "total": 5_000}
        mock_info.siblings = [
            SimpleNamespace(rfilename=name, size=size)
            for name, size in [
                ("model.safetensors", 10_000),
                ("model-Q4_K_M.gguf", 20_000),
                ("pytorch_model.bin", 40_000),  # in ignore_patterns
                ("config.json", 100),
            ]
        ]
        mock_api = MagicMock()
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

        assert task.total_size == 30_100
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
    async def test_recommendations_report_free_space_of_the_model_folder(
        self, tmp_path
    ):
        import omlx.admin.routes as routes_module
        from fastapi import HTTPException

        downloader = MagicMock()
        # A fresh install has no model folder yet; its parent volume counts.
        downloader.model_dir = tmp_path / "models"
        orig_downloader = routes_module._hf_downloader
        try:
            routes_module._hf_downloader = None
            with pytest.raises(HTTPException) as exc:
                await routes_module.get_model_recommendations(is_admin=True)
            assert exc.value.status_code == 503

            routes_module._hf_downloader = downloader
            with patch.object(
                routes_module.model_recommender,
                "recommendations",
                AsyncMock(return_value={"sections": {}}),
            ):
                result = await routes_module.get_model_recommendations(is_admin=True)
        finally:
            routes_module._hf_downloader = orig_downloader

        assert result["disk"]["model_dir"] == str(tmp_path / "models")
        assert result["disk"]["free_bytes"] > 0

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
        mock_settings.get_effective_model_dirs.return_value = [model_dir_with_models]

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

            repo_ids = {m["name"]: m["repo_id"] for m in models}
            assert repo_ids["Qwen3.6-27B-MLX-oQ5-FP16"] == (
                "deepsweet/Qwen3.6-27B-MLX-oQ5-FP16"
            )
            assert repo_ids["model-a"] is None

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
        mock_settings.get_effective_model_dirs.return_value = [model_dir_with_models]

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
    @pytest.mark.parametrize("cancel_first", [False, True])
    async def test_delete_model_cancels_running_download(
        self, tmp_path, cancel_first
    ):
        """Delete waits for the download writer, also right after a cancel."""
        import omlx.admin.routes as routes_module
        from omlx.admin.routes import delete_hf_model

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        target = model_dir / "owner" / "model"
        target.mkdir(parents=True)
        (target / "config.json").write_text(
            '{"architectures": ["Qwen2ForCausalLM"]}'
        )

        downloader = HFDownloader(model_dir=str(model_dir))

        mock_settings = MagicMock()
        mock_settings.get_effective_model_dirs.return_value = [model_dir]

        mock_pool = MagicMock()
        mock_pool.get_loaded_model_ids.return_value = []
        mock_pool._entries = {}
        mock_pool.discover_models = MagicMock()

        mock_settings_mgr = MagicMock()
        mock_settings_mgr.get_pinned_model_ids.return_value = []

        orig_settings = routes_module._get_global_settings
        orig_pool = routes_module._get_engine_pool
        orig_mgr = routes_module._get_settings_manager
        orig_downloader = routes_module._hf_downloader

        routes_module._get_global_settings = lambda: mock_settings
        routes_module._get_engine_pool = lambda: mock_pool
        routes_module._get_settings_manager = lambda: mock_settings_mgr
        routes_module._hf_downloader = downloader

        aborts = []
        worker_started = threading.Event()
        worker_returned = threading.Event()

        def fake_snapshot_download(**kwargs):
            if kwargs.get("dry_run"):
                return []
            worker_started.set()
            deadline = time.monotonic() + 2
            while len(aborts) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            worker_returned.set()

        try:
            with patch(
                "omlx.admin.hf_downloader.HfApi"
            ) as mock_api_cls, patch(
                "omlx.admin.hf_downloader.snapshot_download",
                side_effect=fake_snapshot_download,
            ), patch(
                "omlx.admin.hf_downloader.abort_xet_session",
                side_effect=lambda: aborts.append(1),
            ), patch("omlx.admin.hf_downloader._REAP_INTERVAL", 0.05):
                mock_api = MagicMock()
                mock_info = MagicMock()
                mock_info.siblings = []
                mock_api.model_info.return_value = mock_info
                mock_api_cls.return_value = mock_api

                task = await downloader.start_download("owner/model")
                while not worker_started.is_set():
                    await asyncio.sleep(0.01)
                if cancel_first:
                    assert await downloader.cancel_download(task.task_id)

                result = await delete_hf_model(model_name="model", is_admin=True)

                assert worker_returned.is_set()
                assert result["success"] is True
                assert task.status == DownloadStatus.CANCELLED
                assert not target.exists()
        finally:
            routes_module._get_global_settings = orig_settings
            routes_module._get_engine_pool = orig_pool
            routes_module._get_settings_manager = orig_mgr
            routes_module._hf_downloader = orig_downloader
            await downloader.shutdown()

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
        mock_settings.get_effective_model_dirs.return_value = [model_dir]

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
        mock_settings.get_effective_model_dirs.return_value = [model_dir]

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
        mock_settings.get_effective_model_dirs.return_value = [model_dir_with_models]

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
        mock_settings.get_effective_model_dirs.return_value = [model_dir_with_models]

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
        mock_settings.get_effective_model_dirs.return_value = [model_dir_with_models]

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
        mock_settings.get_effective_model_dirs.return_value = [model_dir_with_models]

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
        mock_settings.get_effective_model_dirs.return_value = [model_dir_with_models]

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

    @staticmethod
    def _make_hf_cache_repo(hub, org, name):
        """Create an HF Hub cache repo whose snapshot files link into blobs/."""
        repo = hub / f"models--{org}--{name}"
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text("abc123")
        (repo / "blobs").mkdir()
        snapshot = repo / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        for filename, data in (
            ("config.json", b'{"model_type": "llama"}'),
            ("model.safetensors", b"0" * 1000),
        ):
            blob = repo / "blobs" / filename
            blob.write_bytes(data)
            (snapshot / filename).symlink_to(blob)
        return repo

    @staticmethod
    def _patch_route_state(model_dirs):
        settings = MagicMock()
        settings.get_effective_model_dirs.return_value = model_dirs
        pool = MagicMock()
        pool.get_loaded_model_ids.return_value = []
        pool._entries = {}
        settings_manager = MagicMock()
        settings_manager.get_pinned_model_ids.return_value = []
        patcher = patch.multiple(
            admin_routes,
            _get_global_settings=lambda: settings,
            _get_engine_pool=lambda: pool,
            _get_settings_manager=lambda: settings_manager,
        )
        return patcher, settings_manager

    @pytest.mark.asyncio
    async def test_hf_cache_repo_is_listed_and_deleted_whole(self, tmp_path):
        """MLX repos in the HF Hub cache list and delete like discovery sees them."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        hub = tmp_path / "hub"
        repo = self._make_hf_cache_repo(hub, "mlx-community", "Tiny-4bit")
        plain = self._make_hf_cache_repo(hub, "acme", "PlainModel")
        patcher, settings_manager = self._patch_route_state([models_dir, hub])

        with patcher:
            listed = (await admin_routes.list_hf_models(is_admin=True))["models"]
            assert [(m["name"], m["source_type"], m["repo_id"]) for m in listed] == [
                ("mlx-community--Tiny-4bit", "hf_cache", "mlx-community/Tiny-4bit")
            ]
            assert listed[0]["size"] == 1023

            result = await admin_routes.delete_hf_model(
                model_name="mlx-community--Tiny-4bit", is_admin=True
            )
            assert result["success"] is True
            assert not repo.exists()
            assert plain.exists()
            settings_manager.delete_settings.assert_called_once_with(
                "mlx-community--Tiny-4bit"
            )

            # Discovery skips non-MLX cache repos, so deletion does too.
            with pytest.raises(HTTPException) as exc_info:
                await admin_routes.delete_hf_model(
                    model_name="acme--PlainModel", is_admin=True
                )
            assert exc_info.value.status_code == 404
            assert plain.exists()

    @pytest.mark.asyncio
    async def test_gguf_models_list_and_delete_one_file_at_a_time(
        self, tmp_path, write_gguf
    ):
        models_dir = tmp_path / "models"
        repo = models_dir / "unsloth" / "Q-GGUF"
        for name, size in (("Q-Q4_K_M.gguf", 100), ("Q-Q8_0.gguf", 200)):
            write_gguf(repo / name, {"general.architecture": "qwen35"}, size=size)
        write_gguf(repo / "mmproj-BF16.gguf", {"general.architecture": "clip"}, size=10)
        # Part 1 of an unfinished split download: no model stands for it.
        write_gguf(repo / "Q-BF16-00001-of-00002.gguf", {}, size=50)
        (repo / "README.md").write_text("card")
        base = SimpleNamespace(base_path=tmp_path / "base")
        pool = EnginePool()
        settings = MagicMock()
        settings.get_effective_model_dirs.return_value = [models_dir]
        downloader = HFDownloader(model_dir=str(models_dir))
        q8_download = DownloadTask(
            task_id="t-q8",
            repo_id="unsloth/Q-GGUF",
            status=DownloadStatus.DOWNLOADING,
            files=["Q-Q8_0.gguf"],
        )
        downloader._tasks[q8_download.task_id] = q8_download
        downloader._active_tasks[q8_download.task_id] = asyncio.create_task(
            asyncio.sleep(60)
        )

        with patch(
            "omlx.engine_pool.get_global_settings", return_value=base
        ), patch("omlx.admin.hf_downloader.abort_xet_session"), patch.multiple(
            admin_routes,
            _get_global_settings=lambda: settings,
            _get_engine_pool=lambda: pool,
            _get_settings_manager=lambda: None,
            _hf_downloader=downloader,
        ):
            pool.discover_models([str(models_dir)])
            rows = (await admin_routes.list_hf_models(is_admin=True))["models"]
            listed = [
                (r["display_name"], r["model_format"], r["repo_id"]) for r in rows
            ]
            assert listed == [
                ("unsloth/Q-GGUF", "gguf", None),
                ("unsloth/Q-Q4_K_M", "gguf", None),
                ("unsloth/Q-Q8_0", "gguf", None),
            ]

            await admin_routes.delete_hf_model("Q-Q4_K_M", is_admin=True)
            assert sorted(p.name for p in repo.iterdir()) == [
                "Q-BF16-00001-of-00002.gguf",
                "Q-Q8_0.gguf",
                "README.md",
                "mmproj-BF16.gguf",
            ]
            assert set(pool._entries) == {"Q-Q8_0"}
            # A download of another file in the folder goes on.
            assert q8_download.status == DownloadStatus.DOWNLOADING

            # Its projector goes with the last model; the split part stays.
            await admin_routes.delete_hf_model("Q-Q8_0", is_admin=True)
            assert sorted(p.name for p in repo.iterdir()) == [
                "Q-BF16-00001-of-00002.gguf",
                "README.md",
            ]
            assert pool._entries == {}
            assert q8_download.status == DownloadStatus.CANCELLED
            assert list((tmp_path / "base" / "gguf").iterdir()) == []

            # The folder row deletes what is left, and the empty org folder.
            await admin_routes.delete_hf_model("Q-GGUF", is_admin=True)
            assert not (models_dir / "unsloth").exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cached, aliased", [(False, False), (True, False), (False, True)])
    async def test_split_variant_delete_preserves_other_variants(
        self, tmp_path, write_gguf, cached, aliased
    ):
        models_dir = tmp_path / "models"
        if cached:
            repo = models_dir / "models--ISTA-DASLab--Flash-Next-GGUF"
            (repo / "refs").mkdir(parents=True)
            (repo / "refs" / "main").write_text("c" * 40)
            folder = repo / "snapshots" / ("c" * 40)
        else:
            repo = folder = models_dir / "ISTA-DASLab" / "Flash-Next-GGUF"
        for variant in ("Q2_0", "Q4_K_M"):
            for part in (1, 2):
                write_gguf(
                    folder / variant / f"Flash-Next-{variant}-{part:05d}-of-00002.gguf",
                    {"general.architecture": "qwen4exp"},
                )
        write_gguf(folder / "mmproj-BF16.gguf", {"general.architecture": "clip"})
        if aliased:
            from omlx.model_discovery import discover_models

            alias = tmp_path / "model-link"
            alias.symlink_to(models_dir, target_is_directory=True)
            discover_models(alias, tmp_path / "base" / "gguf")
            alias.unlink()
        pool = EnginePool()
        settings = MagicMock()
        settings.get_effective_model_dirs.return_value = [models_dir]
        downloader = MagicMock()
        downloader.cancel_download_for_dir = AsyncMock()
        with (
            patch(
                "omlx.engine_pool.get_global_settings",
                return_value=SimpleNamespace(base_path=tmp_path / "base"),
            ),
            patch.multiple(
                admin_routes,
                _get_global_settings=lambda: settings,
                _get_engine_pool=lambda: pool,
                _get_settings_manager=lambda: None,
                _hf_downloader=downloader,
            ),
        ):
            pool.discover_models([str(models_dir)])
            assert set(pool._entries) == {"Flash-Next-Q2_0", "Flash-Next-Q4_K_M"}
            rows = (await admin_routes.list_hf_models(is_admin=True))["models"]
            assert {row["name"] for row in rows} == set(pool._entries)
            await admin_routes.delete_hf_model("Flash-Next-Q2_0", is_admin=True)
            assert not list((folder / "Q2_0").glob("*.gguf"))
            assert len(list((folder / "Q4_K_M").glob("*.gguf"))) == 2
            assert (folder / "mmproj-BF16.gguf").exists()
            assert set(pool._entries) == {"Flash-Next-Q4_K_M"}
            if not cached:
                downloader.cancel_download_for_dir.assert_awaited_once_with(
                    folder,
                    [
                        f"Q2_0/Flash-Next-Q2_0-{part:05d}-of-00002.gguf"
                        for part in (1, 2)
                    ],
                )
            await admin_routes.delete_hf_model("Flash-Next-Q4_K_M", is_admin=True)
            assert not repo.exists()
            assert pool._entries == {}

    @pytest.mark.asyncio
    async def test_gguf_delete_keeps_a_checkpoint_beside_it(self, tmp_path, write_gguf):
        models_dir = tmp_path / "models"
        org = models_dir / "unsloth"
        write_gguf(org / "Q-Q4_K_M.gguf", {"general.architecture": "qwen35"})
        checkpoint = org / "Tiny-4bit"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text('{"model_type": "llama"}')
        (checkpoint / "model.safetensors").write_bytes(b"0" * 1000)
        pool = EnginePool()
        settings = MagicMock()
        settings.get_effective_model_dirs.return_value = [models_dir]
        with (
            patch(
                "omlx.engine_pool.get_global_settings",
                return_value=SimpleNamespace(base_path=tmp_path / "base"),
            ),
            patch.multiple(
                admin_routes,
                _get_global_settings=lambda: settings,
                _get_engine_pool=lambda: pool,
                _get_settings_manager=lambda: None,
                _hf_downloader=None,
            ),
        ):
            pool.discover_models([str(models_dir)])
            await admin_routes.delete_hf_model("Q-Q4_K_M", is_admin=True)

        assert [path.name for path in org.iterdir()] == ["Tiny-4bit"]
        assert (checkpoint / "model.safetensors").exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("target_inside_hub", [False, True])
    async def test_delete_refuses_symlinked_hf_cache_repo(
        self, tmp_path, target_inside_hub
    ):
        """A cache repo entry that is a symlink is refused and its target kept."""
        hub = tmp_path / "hub"
        hub.mkdir()
        target_parent = hub / "store" if target_inside_hub else tmp_path / "other"
        target = self._make_hf_cache_repo(target_parent, "mlx-community", "Tiny-4bit")
        (hub / target.name).symlink_to(target, target_is_directory=True)
        patcher, _ = self._patch_route_state([hub])

        with patcher, pytest.raises(HTTPException) as exc_info:
            await admin_routes.delete_hf_model(
                model_name="mlx-community--Tiny-4bit", is_admin=True
            )
        assert exc_info.value.status_code == 400
        assert (target / "snapshots" / "abc123" / "model.safetensors").exists()


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
        downloader, task = _downloading_task(model_dir, task_id="t1")
        calls = 0

        def zero_byte_temp(_path, _only=None):
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
        downloader, task = _downloading_task(model_dir, task_id="t1")
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
        """Wire bytes alone keep the stall deadline open."""
        import omlx.admin.hf_downloader as dl_module
        from omlx.admin.hf_downloader import _WireCounter

        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 0.03)
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 0.03)
        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.01)
        downloader, task = _downloading_task(model_dir, task_id="t1")
        counter = _WireCounter()
        frozen = _DownloadActivity()  # fetch phase: nothing lands on disk

        with patch.object(
            downloader,
            "_get_download_activity",
            return_value=frozen,
        ), patch("omlx.admin.hf_downloader.abort_xet_session") as mock_abort:
            poll = asyncio.create_task(
                downloader._poll_progress(task.task_id, model_dir, counter)
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
        """Once wire bytes stop against a silent disk, the stall is 'active'."""
        import omlx.admin.hf_downloader as dl_module
        from omlx.admin.hf_downloader import _WireCounter

        monkeypatch.setattr(dl_module, "_STARTUP_STALL_TIMEOUT", 0.03)
        monkeypatch.setattr(dl_module, "_STALL_TIMEOUT", 0.3)
        monkeypatch.setattr(dl_module, "_PROGRESS_POLL_INTERVAL", 0.01)
        downloader, task = _downloading_task(model_dir, task_id="t1")
        counter = _WireCounter()
        frozen = _DownloadActivity()

        with patch.object(
            downloader,
            "_get_download_activity",
            return_value=frozen,
        ), patch("omlx.admin.hf_downloader.abort_xet_session") as mock_abort:
            poll = asyncio.create_task(
                downloader._poll_progress(task.task_id, model_dir, counter)
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
        # setenv, not delenv: teardown must undo the worker's own write.
        monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
        download = MagicMock()

        _download_without_xet({"repo_id": "owner/model"}, download)

        assert os.environ["HF_HUB_DISABLE_XET"] == "1"
        download.assert_called_once_with(repo_id="owner/model")

    def test_worker_writes_http_streams_before_a_full_hub_chunk(self, monkeypatch):
        """The stall watchdog sees an HTTP stream only when hub writes to disk."""
        monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
        hub_chunk = hub_constants.DOWNLOAD_CHUNK_SIZE
        monkeypatch.setattr(hub_constants, "DOWNLOAD_CHUNK_SIZE", hub_chunk)
        payload = b"x" * hub_chunk
        response = httpx.Response(
            200, content=payload, request=httpx.Request("GET", "https://hub.test/f")
        )
        monkeypatch.setattr(
            hub_file_download,
            "http_stream_backoff",
            lambda *args, **kwargs: contextlib.nullcontext(response),
        )
        writes = []

        class Sink(io.BytesIO):
            def write(self, chunk):
                writes.append(len(chunk))
                return super().write(chunk)

        def fake_snapshot_download(**kwargs):
            hub_file_download.http_get(
                "https://hub.test/f", Sink(), expected_size=len(payload)
            )

        monkeypatch.setattr(
            huggingface_hub, "snapshot_download", fake_snapshot_download
        )

        _download_without_xet({"repo_id": "owner/model"})

        assert sum(writes) == len(payload)
        assert len(writes) > 1

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
        counters: list = []
        real_counter = hf_downloader_mod._WireCounter

        def tracked_counter():
            counter = real_counter()
            counters.append(counter)
            return counter

        def fail_xet(**kwargs):
            if kwargs.get("dry_run"):
                return []
            # Bytes that arrived on the wire before the transport died.
            counters[0].add(7_000_000)
            raise RuntimeError(
                "CAS service error: ReqwestMiddleware request failed "
                "for /xet-read-token"
            )

        async def note_fallback(*_args, **_kwargs):
            # The replacement poll's own counter, not the dead xet call's.
            seen["value"] = counters[-1].value if counters else None

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(self._api(), None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fail_xet,
        ), patch.object(
            hf_downloader_mod,
            "_WireCounter",
            new=tracked_counter,
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

        def active_download(_path, _only=None):
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


class _PositiveRate:
    """A table cell for a rate that must be greater than zero."""

    def __eq__(self, other):
        return other > 0

    def __repr__(self):
        return "> 0"


_POSITIVE = _PositiveRate()


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

        def scan(_path, _only=None):
            state["allocated"] += step
            return _DownloadActivity(
                file_count=1,
                logical_size=state["allocated"],
                allocated_size=state["allocated"],
                latest_mtime_ns=1,
                files={"payload": state["allocated"]},
            )

        scan.state = state  # the running total, for assertions
        return scan

    @pytest.mark.asyncio
    async def test_poll_reports_speed_then_zeroes_it(self, model_dir, monkeypatch):
        """A live transfer publishes bytes/s while the ETA is still in warmup;
        a terminal task publishes 0."""
        downloader, task = _downloading_task(model_dir, task_id="t-speed", total_size=10_000_000)

        with patch.object(
            downloader,
            "_get_download_activity",
            side_effect=self._growing_activity(),
        ):
            poll = start_poll(monkeypatch, downloader, task, model_dir)
            await asyncio.sleep(0.05)
            observed_speed = task.speed_bps
            observed_eta = task.eta_seconds()
            task.status = DownloadStatus.COMPLETED
            await poll

        assert observed_speed > 0, "a live download must report a rate"
        assert observed_eta is None, "the slow start must not show an ETA"
        # The poll loop's finally clause clears the rate with the task.
        assert task.speed_bps == 0.0
        assert task.eta_seconds() is None

    @pytest.mark.asyncio
    async def test_eta_rate_starts_at_the_first_byte(self, model_dir, monkeypatch):
        """Idle ticks before the first byte (handshake) must not dilute the
        ETA rate, or a fresh download reads as hours too long."""
        monkeypatch.setattr(hf_downloader_mod, "_ETA_WARMUP", 0)
        downloader, task = _downloading_task(
            model_dir, task_id="t-eta", total_size=10_000_000_000
        )
        step, idle_ticks = 1_000_000, 20
        state = {"calls": 0, "allocated": 1_000_000}
        grew_at = []

        def scan(_path, _only=None):
            state["calls"] += 1
            if state["calls"] > idle_ticks:
                state["allocated"] += step
                grew_at.append(time.monotonic())
            return _DownloadActivity(
                file_count=1,
                logical_size=state["allocated"],
                allocated_size=state["allocated"],
                latest_mtime_ns=1,
                files={"payload": state["allocated"]},
            )

        with patch.object(downloader, "_get_download_activity", side_effect=scan):
            poll = start_poll(monkeypatch, downloader, task, model_dir)
            while len(grew_at) < 6:
                await asyncio.sleep(0.005)
            # Scan and meter update share one tick, so grew_at matches the rate.
            eta_bps = task.eta_speed_bps
            moving = list(grew_at)
            task.status = DownloadStatus.COMPLETED
            await poll

        # The first moving tick only primes the meters.
        expected = step * (len(moving) - 1) / (moving[-1] - moving[0])
        assert eta_bps == pytest.approx(expected, rel=0.25)

    @pytest.mark.asyncio
    async def test_eta_rate_ignores_a_disk_flush_of_fetched_bytes(
        self, model_dir, monkeypatch
    ):
        """xet writes fetched bytes to disk in large bursts. The flush moves no
        progress, so it must not raise the ETA rate above the wire rate."""
        from omlx.admin.hf_downloader import _WireCounter

        monkeypatch.setattr(hf_downloader_mod, "_ETA_WARMUP", 0)
        monkeypatch.setattr(hf_downloader_mod, "_ETA_WINDOW", 0.05)
        downloader, task = _downloading_task(
            model_dir, task_id="t-flush", total_size=10_000_000_000
        )
        counter = _WireCounter()
        step = 1_000_000
        disk = {"allocated": 0}

        def scan(_path, _only=None):
            return _DownloadActivity(
                file_count=1,
                logical_size=disk["allocated"],
                allocated_size=disk["allocated"],
                latest_mtime_ns=1,
                files={"payload": disk["allocated"]},
            )

        with patch.object(downloader, "_get_download_activity", side_effect=scan):
            poll = start_poll(monkeypatch, downloader, task, model_dir, wire=counter)
            started = time.monotonic()
            for tick in range(23):
                if tick == 20:
                    disk["allocated"] = counter.value  # flush, older than the window
                counter.add(step)
                await asyncio.sleep(0.01)
            wire_bps = counter.value / (time.monotonic() - started)
            eta_bps = task.eta_speed_bps
            task.status = DownloadStatus.COMPLETED
            await poll

        assert 0 < eta_bps < 2 * wire_bps

    @staticmethod
    def _meter_rates(window, samples):
        """Feed (timestamp, per-file allocated map) samples to a meter."""
        meter = _SpeedMeter(window=window)
        return [meter.add(files, now=now) for now, files in samples]

    @pytest.mark.parametrize(
        "window, samples, expected",
        [
            pytest.param(
                2.0,
                [(i * 1.0, {"payload": min(i * 50_000_000, 4 * 50_000_000)})
                 for i in range(8)],
                {4: _POSITIVE, 7: 0.0},
                id="a-stopped-transfer-settles-to-zero-within-the-window",
            ),
            pytest.param(
                3.0,
                [(i * 1.0, {"payload": i * 10_000_000}) for i in range(6)],
                {5: pytest.approx(10_000_000, abs=1_000_000)},
                id="the-window-reports-the-true-mean",
            ),
            pytest.param(
                2.0,
                [(0, {"f": 100_000_000}), (1, {"f": 100_000_000}),
                 (2, {"f": 50_000_000}), (3, {"f": 150_000_000})],
                {2: 0.0, 3: _POSITIVE},
                id="a-truncation-is-not-transfer",
            ),
            pytest.param(
                1.0,
                [(0, {"a": 27_000_000_000, "b": 1_000_000_000}),
                 (0.5, {"a": 27_000_000_000, "b": 1_000_000_000}),
                 (1.0, {}),  # wiped / aborted walk
                 (1.5, {"a": 27_000_000_000, "b": 1_000_000_000})],
                {1: 0.0, 2: 0.0, 3: 0.0},
                id="bytes-that-reappear-wholesale-are-not-transfer",
            ),
            pytest.param(
                1.0,
                [(0, {"x": 1_000_000}), (0.5, {"x": 1_000_000}),
                 (1.0, {"x": 1_000_000, "y": 5_000_000_000}),
                 (1.5, {"x": 1_000_000, "y": 5_000_100_000})],
                {2: 0.0, 3: _POSITIVE},
                id="a-file-first-seen-at-full-size-is-not-transfer",
            ),
            pytest.param(
                1.0,
                [(0.0, {}), (0.5, {"model": 20_000_000_000})],
                {1: 0.0},
                id="a-partial-prime-walk-does-not-spike",
            ),
        ],
    )
    def test_speed_meter(self, window, samples, expected):
        """The fixed window bottoms out at 0, reports the true mean, and
        never counts bytes it has not watched grow (first sight, truncation,
        a tree that reappeared after an aborted walk)."""
        rates = self._meter_rates(window, samples)
        for tick, want in expected.items():
            assert rates[tick] == want, f"tick {tick}"

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

    @pytest.mark.asyncio
    async def test_poll_shows_wire_speed_while_disk_is_idle(
        self, model_dir, monkeypatch
    ):
        """xet's fetch phase pulls from the network before any disk write
        (reconstruction blocks are >= 256MB): the readout must show the wire
        rate while filesystem activity stays frozen, then clear to 0."""
        from omlx.admin.hf_downloader import _WireCounter

        downloader, task = _downloading_task(model_dir, task_id="t-wire", total_size=10_000_000_000)
        counter = _WireCounter()

        frozen = _DownloadActivity()  # no byte lands on disk during fetch
        with patch.object(
            downloader, "_get_download_activity", return_value=frozen
        ):
            poll = start_poll(monkeypatch, downloader, task, model_dir, wire=counter)
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

# =============================================================================
# Progress Reads Both Pipeline Stages (fetch = wire, reconstruction = disk)
# =============================================================================


class TestProgressFromWire:
    """Reported bytes must not freeze at the small files while xet's fetch
    phase moves the payload over the network before any disk write.

    One row per contract: the wire leads a fresh transfer, a resume adds the
    wire delta on top of the on-disk baseline, and the reconstruction phase
    never reports below the disk. A step's (download, progress) pair is exact
    unless the contract is a floor (`>=`).
    """

    @pytest.fixture
    def model_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        return d

    @pytest.mark.parametrize(
        "disk, preload, steps",
        [
            pytest.param(
                _DownloadActivity(),  # fetch: the disk stays silent
                0,
                [(2_500_000, 2_500_000, 25.0, True),
                 (9_500_000, 10_000_000, 99.0, True)],
                id="the-wire-leads-a-fresh-transfer",
            ),
            pytest.param(
                # A resumed download: earlier files are already on disk.
                _DownloadActivity(
                    file_count=1,
                    logical_size=6_000_000,
                    allocated_size=6_000_000,
                    latest_mtime_ns=1,
                    files={"big.safetensors": 6_000_000},
                ),
                0,
                [(1_000_000, 7_000_000, 70.0, True),
                 (2_000_000, 8_000_000, 80.0, False)],
                id="a-resume-adds-the-wire-delta-to-the-disk-baseline",
            ),
            pytest.param(
                _DownloadActivity(
                    file_count=1,
                    logical_size=8_000_000,
                    allocated_size=8_000_000,
                    latest_mtime_ns=1,
                ),
                2_000_000,  # fetch delivered only part over the wire
                [(0, 8_000_000, 80.0, False)],
                id="reconstruction-keeps-the-disk-reading",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_reported_bytes_follow_the_leading_stage(
        self, model_dir, monkeypatch, disk, preload, steps
    ):
        from omlx.admin.hf_downloader import _WireCounter

        downloader, task = _downloading_task(model_dir, task_id="t-wire", total_size=10_000_000)
        counter = _WireCounter()
        counter.add(preload)

        with patch.object(
            downloader, "_get_download_activity", return_value=disk
        ):
            poll = start_poll(
                monkeypatch, downloader, task, model_dir, wire=counter
            )
            for added, size, progress, exact in steps:
                counter.add(added)
                await asyncio.sleep(0.05)  # several poll iterations
                # The wire may pass the size estimate (retries, protocol
                # overhead) and stale wire bytes must never pull the report
                # below the disk: the total caps it and 100% stays reserved
                # for snapshot_download's completion write.
                if exact:
                    assert task.downloaded_size == size
                    assert task.progress == progress
                else:
                    assert task.downloaded_size >= size
                    assert task.progress >= progress
            task.status = DownloadStatus.COMPLETED
            await poll


# =============================================================================
# Re-downloads Over Outdated Files
# =============================================================================


_OLD_COMMIT = "a" * 40
_NEW_COMMIT = "b" * 40


def _dry_info(local_dir, filename, *, will_download=False):
    from huggingface_hub import DryRunFileInfo

    path = local_dir / filename
    return DryRunFileInfo(
        commit_hash=_NEW_COMMIT,
        file_size=path.stat().st_size if path.exists() else 0,
        filename=filename,
        local_path=str(path),
        is_cached=path.exists(),
        will_download=will_download,
    )


def _write(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)


class TestStaleLocalFiles:
    """The dry run calls an outdated local file cached, so staleness comes
    from the commit in the metadata the dry run leaves behind."""

    def test_only_files_hub_will_replace_are_stale(self, tmp_path):
        from huggingface_hub._local_folder import write_download_metadata

        from omlx.admin.hf_downloader import _stale_local_files

        for name in ("current.json", "outdated.safetensors", "bare.safetensors"):
            _write(tmp_path / name, 100)
        write_download_metadata(tmp_path, "current.json", _NEW_COMMIT, "e1")
        write_download_metadata(tmp_path, "outdated.safetensors", _OLD_COMMIT, "e2")
        _write(tmp_path / "forced.bin", 50)

        stale = _stale_local_files(
            tmp_path,
            [
                _dry_info(tmp_path, "current.json"),
                _dry_info(tmp_path, "outdated.safetensors"),
                _dry_info(tmp_path, "bare.safetensors"),  # no metadata
                _dry_info(tmp_path, "forced.bin", will_download=True),
                _dry_info(tmp_path, "missing.safetensors", will_download=True),
            ],
        )

        assert {Path(p).name: size for p, (_, size) in stale.items()} == {
            "outdated.safetensors": 100,
            "bare.safetensors": 100,
            "forced.bin": 50,
        }


class TestRedownloadProgress:
    """Hub deletes an outdated file before refetching it. Its bytes must not
    sit in the poll's baseline, or the directory shrinks below it, disk growth
    reads zero, and every payload byte counts twice (wire, then disk)."""

    OLD = 8_000

    @pytest.fixture
    def model_dir(self, tmp_path):
        d = tmp_path / "models"
        _write(d / "config.json", 2_000)
        _write(d / "model.safetensors", self.OLD)
        return d

    def _stale(self, model_dir):
        """Outdated files, plus the bytes hub keeps (config and its own
        bookkeeping under .cache)."""
        from omlx.admin.hf_downloader import _stale_local_files

        stale = _stale_local_files(
            model_dir, [_dry_info(model_dir, "model.safetensors")]
        )
        on_disk = HFDownloader._get_download_activity(model_dir).logical_size
        return stale, on_disk - self.OLD

    @pytest.mark.asyncio
    async def test_wire_then_disk_counts_each_byte_once(self, model_dir, monkeypatch):
        from omlx.admin.hf_downloader import _WireCounter

        stale, kept = self._stale(model_dir)
        total = kept + self.OLD
        downloader, task = _downloading_task(
            model_dir, task_id="t-redl", total_size=total
        )
        wire = _WireCounter()
        monkeypatch.setattr(hf_downloader_mod, "_PROGRESS_POLL_INTERVAL", 0.01)
        poll = asyncio.create_task(
            downloader._poll_progress(task.task_id, model_dir, wire, stale)
        )
        try:
            incomplete = model_dir / ".cache/huggingface/download/x.incomplete"

            async def settle():
                await asyncio.sleep(0.05)  # several poll iterations

            # The outdated shard is still on disk: it is not progress.
            wire.add(1_000)
            await settle()
            assert task.downloaded_size == kept + 1_000

            # Hub deletes it and fetches the new payload over the wire.
            (model_dir / "model.safetensors").unlink()
            wire.add(3_000)
            await settle()
            assert task.downloaded_size == kept + 4_000

            # Reconstruction writes what the wire already counted.
            _write(incomplete, 4_000)
            await settle()
            assert task.downloaded_size == kept + 4_000
            assert task.progress == pytest.approx((kept + 4_000) / total * 100)

            # The finished file lands where the outdated one was.
            _write(incomplete, self.OLD)
            incomplete.rename(model_dir / "model.safetensors")
            wire.add(3_800)  # xet compression: fewer wire bytes than the file
            await settle()
            assert task.downloaded_size == total
        finally:
            task.status = DownloadStatus.COMPLETED
            await poll

    @pytest.mark.asyncio
    async def test_replacement_seen_in_one_poll_counts_as_progress(
        self, model_dir, monkeypatch
    ):
        """Deleted and replaced between two polls: the new inode at the same
        path is the refetched file, not the outdated one."""
        stale, kept = self._stale(model_dir)
        downloader, task = _downloading_task(
            model_dir, task_id="t-swap", total_size=2 * (kept + self.OLD)
        )
        monkeypatch.setattr(hf_downloader_mod, "_PROGRESS_POLL_INTERVAL", 0.01)
        poll = asyncio.create_task(
            downloader._poll_progress(task.task_id, model_dir, None, stale)
        )
        try:
            await asyncio.sleep(0.05)
            assert task.downloaded_size == kept

            fresh = model_dir / "fresh.tmp"
            _write(fresh, self.OLD)
            fresh.replace(model_dir / "model.safetensors")
            await asyncio.sleep(0.05)
            assert task.downloaded_size == kept + self.OLD
        finally:
            task.status = DownloadStatus.COMPLETED
            await poll

    @pytest.mark.asyncio
    async def test_run_download_hands_outdated_files_to_the_poll(self, model_dir):
        downloader = HFDownloader(model_dir=str(model_dir.parent))
        target = model_dir.parent / "owner" / "model"
        _write(target / "model.safetensors", self.OLD)
        mock_api = MagicMock()
        mock_api.model_info.return_value = MagicMock(siblings=[])
        polled = {}

        async def fake_poll(task_id, target_dir, wire=None, stale=None):
            polled["stale"] = stale

        def fake_snapshot_download(**kwargs):
            if kwargs.get("dry_run"):
                return [_dry_info(target, "model.safetensors")]
            return str(target)

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(mock_api, None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fake_snapshot_download,
        ), patch.object(downloader, "_poll_progress", side_effect=fake_poll):
            await downloader.start_download("owner/model")
            await _wait_for_downloads(downloader)

        assert list(polled["stale"]) == [str(target / "model.safetensors")]


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


def _sibling(name: str, size: int) -> SimpleNamespace:
    return SimpleNamespace(rfilename=name, size=size)


class TestFileSelection:
    """Downloads of a chosen subset of a repo's files."""

    @pytest.fixture
    def model_dir(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        return d

    @staticmethod
    def _gguf_api(siblings):
        info = MagicMock()
        info.safetensors = None
        info.siblings = siblings
        api = MagicMock()
        api.model_info.return_value = info
        return api

    @pytest.mark.asyncio
    async def test_selection_downloads_only_those_files(self, model_dir):
        downloader = HFDownloader(model_dir=str(model_dir))
        siblings = [
            _sibling("Q-UD-Q4_K_M.gguf", 100),
            _sibling("Q-Q8_0.gguf", 200),
            _sibling("mmproj-BF16.gguf", 10),
        ]
        files = ["mmproj-BF16.gguf", "Q-UD-Q4_K_M.gguf"]
        calls = []

        def fake_snapshot_download(**kwargs):
            calls.append(kwargs)
            if kwargs.get("dry_run"):
                return [
                    SimpleNamespace(
                        filename=name,
                        file_size=size,
                        will_download=True,
                        commit_hash="c",
                    )
                    for name, size in (
                        ("Q-UD-Q4_K_M.gguf", 100),
                        ("mmproj-BF16.gguf", 10),
                    )
                ]

        with patch(
            "omlx.admin.hf_downloader._get_hf_api",
            return_value=(self._gguf_api(siblings), None),
        ), patch(
            "omlx.admin.hf_downloader.snapshot_download",
            side_effect=fake_snapshot_download,
        ):
            task = await downloader.start_download("unsloth/Q-GGUF", files=files)
            await downloader._active_tasks[task.task_id]

        assert task.status == DownloadStatus.COMPLETED
        assert task.files == sorted(files)
        assert task.total_size == 110
        assert [call["allow_patterns"] for call in calls] == [sorted(files)] * 2
        assert all("ignore_patterns" not in call for call in calls)
        # The HTTP fallback worker takes the same selection.
        worker_kwargs = {
            k: v for k, v in calls[-1].items() if k not in ("tqdm_class", "dry_run")
        }
        download = MagicMock()
        _download_without_xet(worker_kwargs, download)
        assert download.call_args.kwargs["allow_patterns"] == sorted(files)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("library", "files", "error"),
        [
            (None, ["Q-UD-Q4_K_M.gguf"], "Q-UD-Q4_K_M.gguf"),
            ("splash", None, "Download mlx-community/Qwen3.8-27B-4bit instead"),
        ],
    )
    async def test_refused_download_fails_before_transfer(
        self, model_dir, library, files, error
    ):
        downloader = HFDownloader(model_dir=str(model_dir))
        api = self._gguf_api([_sibling("Q-Q8_0.gguf", 200)])
        info = api.model_info.return_value
        info.library_name = library
        info.card_data = SimpleNamespace(
            base_model=["mlx-community/Qwen3.8-27B-4bit", "incoai/Qwen3.8-27B-DFlash2"]
        )

        with patch(
            "omlx.admin.hf_downloader._get_hf_api", return_value=(api, None)
        ), patch("omlx.admin.hf_downloader.snapshot_download") as snapshot:
            task = await downloader.start_download("owner/model", files=files)
            await downloader._active_tasks[task.task_id]

        assert task.status == DownloadStatus.FAILED
        assert error in task.error
        snapshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_selection_survives_retry_and_restart(self, tmp_path, model_dir):
        tasks_file = tmp_path / "download_tasks.json"
        files = ["Q-UD-Q4_K_M.gguf", "mmproj-BF16.gguf"]

        async def no_download(self, task_id, token):
            return None

        with patch.object(HFDownloader, "_run_download", no_download):
            downloader = HFDownloader(model_dir=str(model_dir), tasks_file=tasks_file)
            task = await downloader.start_download("unsloth/Q-GGUF", files=files)
            with pytest.raises(ValueError, match="already in progress"):
                await downloader.start_download("Unsloth/q-gguf")
            task.status = DownloadStatus.FAILED
            retried = await downloader.retry_download(task.task_id)
            assert retried.files == files

            restarted = HFDownloader(model_dir=str(model_dir), tasks_file=tasks_file)
            await restarted.restore_tasks()
        assert [t.files for t in restarted._tasks.values()] == [files]

    def test_model_info_rows_describe_what_to_select(self):
        mlx = [
            {"name": name, "size": 1}
            for name in (
                "config.json",
                "model.safetensors",
                "README.md",
                "pytorch_model.bin",
            )
        ]
        assert hf_downloader_mod._describe_repo_files(mlx, None) == "safetensors"
        required = {row["name"] for row in mlx if row["required"]}
        assert required == {"config.json", "model.safetensors"}
        skipped = {row["name"] for row in mlx if not row["preselected"]}
        assert skipped == {"pytorch_model.bin"}

        gguf = [
            {"name": "Q-UD-Q4_K_M.gguf", "size": 4},
            {"name": "BF16/Q-BF16-00001-of-00002.gguf", "size": 8},
            {"name": "mmproj-F32.gguf", "size": 2},
            {"name": "mmproj-BF16.gguf", "size": 1},
        ]
        assert hf_downloader_mod._describe_repo_files(gguf, None) == "gguf"
        rows = {row["name"]: row for row in gguf}
        assert rows["Q-UD-Q4_K_M.gguf"]["mmproj"] == "mmproj-BF16.gguf"
        part = rows["BF16/Q-BF16-00001-of-00002.gguf"]
        assert part["kind"] == "gguf"
        assert part["group"] == "BF16/Q-BF16"
        assert part["mmproj"] == "mmproj-BF16.gguf"
        assert not any(row["required"] for row in gguf)

    def test_split_variants_group_their_parts_and_skip_draft_folders(self):
        files = [
            {"name": name, "size": 1}
            for name in (
                "Q2_0/Flash-Next-Q2_0-00001-of-00002.gguf",
                "Q2_0/Flash-Next-Q2_0-00002-of-00002.gguf",
                "MTP/draft.gguf",
                "MTP/draft-00001-of-00002.gguf",
                "other/Flash-Next-Q2_0-00001-of-00002.gguf",
                "nested/Q2_0/Flash-Next-Q2_0-00001-of-00002.gguf",
                "mmproj-BF16.gguf",
            )
        ]
        assert hf_downloader_mod._describe_repo_files(files, None) == "gguf"
        for row in files[:2]:
            assert row["kind"] == "gguf"
            assert row["group"] == "Q2_0/Flash-Next-Q2_0"
            assert row["mmproj"] == "mmproj-BF16.gguf"
        assert all(row["kind"] == "other" for row in files[2:6])
