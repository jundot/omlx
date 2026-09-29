# SPDX-License-Identifier: Apache-2.0
"""The restart-surviving download queue, tested for both backends.

`_persist_queue`/`_restore_queue` and the on-disk row format are shared by
HFDownloader and MSDownloader, so every shared case runs against both; only
the seam a test has to reach differs (how an environment refuses a resume,
which failure the SDK reports), and each such seam is selected by a fixture.
"""

import asyncio
import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from omlx.admin.hf_downloader import DownloadStatus, DownloadTask, HFDownloader
from omlx.admin.ms_downloader import MSDownloader
from tests._download_helpers import _read_rows, _write_rows


@pytest.fixture(params=[HFDownloader, MSDownloader], ids=["hf", "ms"])
def queue(request, tmp_path):
    """One downloader of either kind with its run body neutralised.

    Restore only schedules each resumed row's run coroutine, so the tests
    here replace it wholesale — recording the credential it was handed in
    ``queue.tokens`` — and open the ModelScope SDK gate, the check that HF's
    start_download simply does not have.
    """
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


@pytest.fixture
def refuse_resume(queue):
    """Make the environment refuse a queued row's `start_download`.

    A refusal that is not the row's own: HF's hub may be unimportable, the
    ModelScope SDK gate is checked inside start_download.
    """
    if queue.cls is HFDownloader:
        async def _refused(self, repo_id, token):
            raise RuntimeError("hub is not importable right now")

        return patch.object(HFDownloader, "start_download", new=_refused)
    return patch("omlx.admin.ms_downloader.MS_SDK_AVAILABLE", False)


class TestQueuePersistence:
    """The queue persists to disk and survives a restart."""

    def test_a_resumable_row_keeps_the_token_and_a_finished_one_does_not(
        self, queue
    ):
        """Only a row that can still be resumed needs the credential: a
        terminal row keeps it in memory for a retry in this process but is
        written without it, so tokens do not outlive their download."""
        downloader, tasks_file = queue.downloader, queue.tasks_file
        task = DownloadTask(task_id="t1", repo_id="private/model")
        task.token = "SECRET"
        downloader._tasks["t1"] = task

        downloader._persist()
        assert _read_rows(tasks_file)[0]["token"] == "SECRET"

        task.status = DownloadStatus.COMPLETED
        downloader._persist()
        assert _read_rows(tasks_file)[0]["token"] == ""
        # The credential survives in memory, so a retry in this process can
        # still reach a gated repository.
        assert task.token == "SECRET"

    def test_the_queue_file_is_owner_only_from_the_first_write(self, queue):
        downloader, tasks_file = queue.downloader, queue.tasks_file
        task = DownloadTask(task_id="t1", repo_id="owner/model")
        task.token = "SECRET"
        downloader._tasks["t1"] = task

        downloader._persist()

        assert tasks_file.stat().st_mode & 0o777 == 0o600
        assert not tasks_file.with_name(tasks_file.name + ".tmp").exists()

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
        # The repo-info fallback may rewrite the raw error into a
        # repository-not-found style message; only persistence matters here
        # (error round-trip is covered by the restore tests below).
        assert rows[0]["error"]

    @pytest.mark.asyncio
    async def test_shutdown_leaves_row_resumable_on_disk(self, queue):
        downloader, tasks_file = queue.downloader, queue.tasks_file
        task = await downloader.start_download("owner/model")

        task.status = DownloadStatus.DOWNLOADING
        with patch("omlx.admin.hf_downloader.abort_xet_session"):
            await downloader.shutdown()

        # The in-memory row went CANCELLED, but shutdown must not write that:
        # the on-disk row keeps its queued status so the next boot resumes.
        rows = _read_rows(tasks_file)
        assert rows[0]["status"] not in (
            DownloadStatus.CANCELLED.value,
            DownloadStatus.FAILED.value,
        )

    @pytest.mark.asyncio
    async def test_a_terminal_row_with_unreadable_numbers_still_restores(
        self, queue
    ):
        """A finished row gets the same tolerance the queued branch has.

        A *queued* row carrying an ISO timestamp and a text retry count
        restores regardless (its two fields go through _read_float and
        _read_int). The same fields on a *finished* row go through
        from_dict, which parsed every number eagerly: one unparseable value
        raised, restore skipped the row, and its display, error text and
        retry entry were lost — exactly what the resumable branch is built
        to avoid.
        """
        downloader, tasks_file = queue.downloader, queue.tasks_file
        _write_rows(tasks_file, [
            {"task_id": "junk", "repo_id": "owner/junk",
             "status": DownloadStatus.FAILED.value, "error": "boom",
             "progress": "n/a", "total_size": "big", "downloaded_size": None,
             "created_at": "2026-09-25T00:00:00", "started_at": [],
             "completed_at": {}, "retry_count": "many"},
            {"task_id": "good", "repo_id": "owner/good",
             "status": DownloadStatus.COMPLETED.value, "created_at": 5.0,
             "retry_count": 1},
        ])

        await downloader.restore_tasks()

        junk = downloader._tasks["junk"]
        assert junk.status == DownloadStatus.FAILED
        assert junk.error == "boom"
        assert junk.progress == 0.0
        assert junk.total_size == 0
        assert junk.downloaded_size == 0
        assert junk.retry_count == 0
        assert junk.created_at  # the restore time, not 0.0
        # ...and the healthy row behind it still restores.
        assert downloader._tasks["good"].status == DownloadStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_a_row_the_environment_refuses_stays_queued_on_disk(
        self, queue, refuse_resume, caplog
    ):
        """`restore_tasks` promises never to raise — and a refusal that is not
        the row's own must not cost the row either: the restore keeps going,
        the healing rewrite is skipped, and the queue file stays byte-for-byte
        so the next boot retries the interrupted row.
        """
        downloader, tasks_file = queue.downloader, queue.tasks_file
        _write_rows(tasks_file, [
            {"task_id": "live", "repo_id": "owner/live",
             "status": DownloadStatus.DOWNLOADING.value, "created_at": 10.0},
            {"task_id": "done", "repo_id": "owner/done",
             "status": DownloadStatus.COMPLETED.value, "created_at": 20.0},
        ])
        before = tasks_file.read_text(encoding="utf-8")

        with refuse_resume, caplog.at_level(logging.WARNING):
            await downloader.restore_tasks()

        # The restore kept its promise and carried on: the terminal row came back.
        assert downloader._tasks["done"].status == DownloadStatus.COMPLETED
        # The row that could not start is not in memory...
        assert "owner/live" not in {
            t.repo_id for t in downloader._tasks.values()
        }
        # ...and is still queued on disk, byte for byte.
        assert tasks_file.read_text(encoding="utf-8") == before
        assert "Deferring resume of owner/live" in caplog.text

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
        # Terminal rows come back as display-only entries, error text intact.
        assert downloader._tasks["done"].status == DownloadStatus.COMPLETED
        assert downloader._tasks["done"].speed_bps == 0.0
        assert downloader._tasks["fail"].error == "boom"
        # The rewritten queue records the resumed row as pending under its
        # new task id (task ids are restart-scoped; rows are matched by repo).
        live_rows = [
            r for r in _read_rows(tasks_file) if r["repo_id"] == "owner/live"
        ]
        assert live_rows
        assert live_rows[0]["status"] == DownloadStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_one_row_this_build_cannot_read_does_not_lose_the_queue(
        self, queue
    ):
        """`restore_tasks` promises never to raise, so one field a build that
        stored it differently left behind skips that row's bookkeeping alone:
        the rows behind it come back and the healing rewrite runs."""
        downloader, tasks_file = queue.downloader, queue.tasks_file
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
        _write_rows(tasks_file, rows)

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
        written = {row["repo_id"]: row for row in _read_rows(tasks_file)}
        assert set(written) == set(by_repo)
        assert written["owner/unknown"]["status"] == DownloadStatus.FAILED.value

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
    async def test_restore_warns_about_a_bad_row_without_its_token(
        self, queue, caplog
    ):
        """A row the loader cannot read is skipped, not fatal — and the row may
        still carry the credential, so the warning leaves it out."""
        downloader, tasks_file = queue.downloader, queue.tasks_file
        _write_rows(tasks_file, [{
            "task_id": "t1",
            "status": "completed",
            "token": "SUPERSECRET",
            # repo_id is what from_dict reads first; its absence is the
            # KeyError this path already tolerates.
        }])

        with caplog.at_level(logging.WARNING):
            await downloader.restore_tasks()

        assert "Skipping unpersistable download row" in caplog.text
        assert "SUPERSECRET" not in caplog.text

    @pytest.mark.asyncio
    async def test_credential_persists_and_restores_without_reaching_api(
        self, queue
    ):
        """A gated download restarts with the token its request supplied."""
        downloader, tasks_file = queue.downloader, queue.tasks_file
        task = await downloader.start_download("owner/model", "GEHEIM")
        await asyncio.sleep(0)  # let the scheduled download coroutine run

        # On disk: the credential that queued the download, owner-only.
        assert _read_rows(tasks_file)[0]["token"] == "GEHEIM"
        if os.name != "nt":
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
    async def test_restore_without_token_row_falls_back_to_empty(self, queue):
        """Rows written before the token field keep hub's env/login lookup."""
        downloader, tasks_file = queue.downloader, queue.tasks_file
        _write_rows(
            tasks_file,
            [
                {
                    "task_id": "live",
                    "repo_id": "owner/model",
                    "status": "pending",
                    "created_at": 100.0,
                }
            ],
        )

        await downloader.restore_tasks()
        await asyncio.sleep(0)  # let the scheduled download coroutine run

        assert queue.tokens["token"] == ""  # start_download maps "" to None

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

        # Retry without a token (the app sends none): the stored credential
        # is kept instead of being wiped to "".
        kept = await downloader.retry_download("old", "")
        assert kept.token == "GEHEIM"
        assert kept.retry_count == 1
        rows = {r["task_id"]: r for r in _read_rows(tasks_file)}
        assert rows[kept.task_id]["token"] == "GEHEIM"
        # The retry bookkeeping is on disk immediately, not on some later
        # event — restarting right now must not lose the count.
        assert rows[kept.task_id]["retry_count"] == 1

        # Retry with a freshly re-entered token (the web form): the new
        # credential replaces the stale one on disk.
        kept.status = DownloadStatus.FAILED
        replaced = await downloader.retry_download(kept.task_id, "NEU")
        assert replaced.token == "NEU"
        assert replaced.retry_count == 2
        rows = {r["task_id"]: r for r in _read_rows(tasks_file)}
        assert rows[replaced.task_id]["token"] == "NEU"
        assert rows[replaced.task_id]["retry_count"] == 2
