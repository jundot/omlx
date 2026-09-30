# SPDX-License-Identifier: Apache-2.0
"""Restart-surviving download queue, tested for both backends."""

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omlx.admin.hf_downloader import DownloadStatus, DownloadTask, HFDownloader
from omlx.admin.ms_downloader import MSDownloader


def _write_rows(path: Path, rows) -> None:
    """Write a persisted queue file the way a previous boot left it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def _read_rows(path: Path) -> list:
    """Read back the persisted queue rows."""
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(params=[HFDownloader, MSDownloader], ids=["hf", "ms"])
def queue(request, tmp_path):
    """A downloader whose run body only records the token it was given."""
    cls = request.param
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    tasks_file = tmp_path / "state" / f"{cls.__name__}_queue.json"
    tokens: dict = {}

    async def _run(self, task_id, token):
        tokens["token"] = token

    with patch.object(cls, "_run_download", new=_run), patch(
        "omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True
    ):
        yield SimpleNamespace(
            cls=cls,
            downloader=cls(model_dir=str(model_dir), tasks_file=tasks_file),
            tasks_file=tasks_file,
            tokens=tokens,
        )


class TestQueuePersistence:
    """The queue persists to disk and survives a restart."""

    def test_a_resumable_row_keeps_the_token_and_a_finished_one_does_not(
        self, queue
    ):
        downloader, tasks_file = queue.downloader, queue.tasks_file
        task = DownloadTask(task_id="t1", repo_id="private/model")
        task.token = "SECRET"
        downloader._tasks["t1"] = task

        downloader._persist()
        assert _read_rows(tasks_file)[0]["token"] == "SECRET"

        task.status = DownloadStatus.COMPLETED
        downloader._persist()
        assert _read_rows(tasks_file)[0]["token"] == ""
        assert task.token == "SECRET"

    @pytest.mark.asyncio
    async def test_start_and_cancel_persist_rows(self, queue):
        downloader, tasks_file = queue.downloader, queue.tasks_file
        task = await downloader.start_download("owner/model")

        rows = _read_rows(tasks_file)
        assert len(rows) == 1
        assert rows[0]["task_id"] == task.task_id
        assert rows[0]["repo_id"] == "owner/model"
        assert rows[0]["status"] == DownloadStatus.PENDING.value

        assert await downloader.cancel_download(task.task_id) is True
        assert _read_rows(tasks_file)[0]["status"] == (
            DownloadStatus.CANCELLED.value
        )

    @pytest.mark.asyncio
    async def test_remove_task_persists_without_row(self, queue):
        downloader, tasks_file = queue.downloader, queue.tasks_file
        task = DownloadTask(
            task_id="t1",
            repo_id="owner/model",
            status=DownloadStatus.COMPLETED,
        )
        downloader._tasks[task.task_id] = task
        downloader._persist()

        assert downloader.remove_task(task.task_id) is True
        assert _read_rows(tasks_file) == []

    @pytest.mark.asyncio
    async def test_failed_run_persists_failed_row(self, tmp_path):
        """HF-only: the run fails through the hub API, not the SDK."""
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        tasks_file = tmp_path / "state" / "hf_queue.json"
        downloader = HFDownloader(
            model_dir=str(model_dir), tasks_file=tasks_file
        )
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
        rows = _read_rows(tasks_file)
        assert rows[0]["status"] == DownloadStatus.FAILED.value
        assert rows[0]["error"]

    @pytest.mark.asyncio
    async def test_shutdown_leaves_row_resumable_on_disk(self, queue):
        downloader, tasks_file = queue.downloader, queue.tasks_file
        task = await downloader.start_download("owner/model")

        task.status = DownloadStatus.DOWNLOADING
        with patch("omlx.admin.hf_downloader.abort_xet_session"):
            await downloader.shutdown()

        rows = _read_rows(tasks_file)
        assert rows[0]["status"] not in (
            DownloadStatus.CANCELLED.value,
            DownloadStatus.FAILED.value,
        )

    @pytest.mark.asyncio
    async def test_restore_resumes_interrupted_rows(self, queue):
        downloader, tasks_file = queue.downloader, queue.tasks_file
        _write_rows(tasks_file, [
            {"task_id": "done", "repo_id": "owner/done", "status": "completed",
             "progress": 100.0, "created_at": 100.0},
            {"task_id": "fail", "repo_id": "owner/fail", "status": "failed",
             "error": "boom", "created_at": 200.0},
            {"task_id": "live", "repo_id": "owner/live",
             "status": "downloading", "created_at": 300.0, "retry_count": 2},
        ])

        await downloader.restore_tasks()

        resumed = [
            t for t in downloader._tasks.values()
            if t.status == DownloadStatus.PENDING
        ]
        assert [t.repo_id for t in resumed] == ["owner/live"]
        assert resumed[0].created_at == 300.0
        assert resumed[0].retry_count == 2
        # Failed rows stay retryable; completed rows are dropped.
        assert downloader._tasks["fail"].error == "boom"
        assert "done" not in downloader._tasks
        live_rows = [
            r for r in _read_rows(tasks_file) if r["repo_id"] == "owner/live"
        ]
        assert live_rows
        assert live_rows[0]["status"] == DownloadStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_restore_resumes_duplicate_interrupted_repo_once(self, queue):
        downloader, tasks_file = queue.downloader, queue.tasks_file
        row = {
            "task_id": "x",
            "repo_id": "owner/dup",
            "status": "downloading",
            "created_at": 100.0,
        }
        _write_rows(
            tasks_file, [dict(row, task_id="a"), dict(row, task_id="b")]
        )

        await downloader.restore_tasks()  # duplicate must not raise

        active = [
            t for t in downloader._tasks.values()
            if t.status == DownloadStatus.PENDING
        ]
        assert len(active) == 1
        assert active[0].repo_id == "owner/dup"

    @pytest.mark.asyncio
    async def test_restore_tolerates_missing_and_corrupt_files(self, queue):
        downloader, tasks_file = queue.downloader, queue.tasks_file
        await downloader.restore_tasks()  # missing file: no-op

        tasks_file.parent.mkdir(parents=True, exist_ok=True)
        tasks_file.write_text("{not json", encoding="utf-8")
        await downloader.restore_tasks()  # corrupt file: no-op, no raise
        assert downloader._tasks == {}

        tasks_file.write_text('{"not": "a list"}', encoding="utf-8")
        await downloader.restore_tasks()
        assert downloader._tasks == {}

    @pytest.mark.asyncio
    async def test_restore_skips_a_bad_row_without_logging_its_token(
        self, queue, caplog
    ):
        downloader, tasks_file = queue.downloader, queue.tasks_file
        _write_rows(tasks_file, [
            {"task_id": "bad", "status": "failed", "token": "SUPERSECRET"},
            {"task_id": "unknown", "repo_id": "owner/x", "status": "paused?"},
            {"task_id": "fail", "repo_id": "owner/fail", "status": "failed"},
        ])

        with caplog.at_level(logging.WARNING):
            await downloader.restore_tasks()

        assert set(downloader._tasks) == {"fail"}
        assert "Skipping persisted download row" in caplog.text
        assert "SUPERSECRET" not in caplog.text

    @pytest.mark.asyncio
    async def test_credential_persists_and_restores_without_reaching_api(
        self, queue
    ):
        downloader, tasks_file = queue.downloader, queue.tasks_file
        task = await downloader.start_download("owner/model", "GEHEIM")
        await asyncio.sleep(0)  # let the scheduled download coroutine run

        # On disk: the credential that queued the download, owner-only.
        assert _read_rows(tasks_file)[0]["token"] == "GEHEIM"
        assert tasks_file.stat().st_mode & 0o777 == 0o600
        # Over the API: never (the queue serves to_dict() output).
        assert "token" not in task.to_dict()
        assert all("token" not in row for row in downloader.get_tasks())
        assert task.token == "GEHEIM"

        # Simulate a restart: the persisted row re-queues with its token.
        queue.tokens.clear()
        with patch("omlx.admin.hf_downloader.abort_xet_session"):
            await downloader.shutdown()
        fresh = queue.cls(
            model_dir=str(downloader._model_dir), tasks_file=tasks_file
        )
        await fresh.restore_tasks()
        await asyncio.sleep(0)

        assert queue.tokens["token"] == "GEHEIM"
        resumed = [
            t for t in fresh._tasks.values()
            if t.status == DownloadStatus.PENDING
        ]
        assert [t.repo_id for t in resumed] == ["owner/model"]
        assert resumed[0].token == "GEHEIM"
        # The credential survives the restore rewrite for the next restart.
        assert _read_rows(tasks_file)[0]["token"] == "GEHEIM"

    @pytest.mark.asyncio
    async def test_retry_recovers_credential_and_persists_bookkeeping(
        self, queue
    ):
        downloader, tasks_file = queue.downloader, queue.tasks_file
        old = DownloadTask(
            task_id="old",
            repo_id="owner/gated",
            status=DownloadStatus.FAILED,
            token="GEHEIM",
        )
        downloader._tasks["old"] = old
        downloader._persist()

        # The app retries without a token; the stored one is kept.
        kept = await downloader.retry_download("old", "")
        assert kept.token == "GEHEIM"
        assert kept.retry_count == 1
        rows = {r["task_id"]: r for r in _read_rows(tasks_file)}
        assert rows[kept.task_id]["token"] == "GEHEIM"
        assert rows[kept.task_id]["retry_count"] == 1

        # A re-entered token replaces the stored one.
        kept.status = DownloadStatus.FAILED
        replaced = await downloader.retry_download(kept.task_id, "NEU")
        assert replaced.token == "NEU"
        assert replaced.retry_count == 2
        rows = {r["task_id"]: r for r in _read_rows(tasks_file)}
        assert rows[replaced.task_id]["token"] == "NEU"
        assert rows[replaced.task_id]["retry_count"] == 2
