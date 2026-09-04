# SPDX-License-Identifier: Apache-2.0
"""Tests for the log rotation backstop and transient-artifact reaper
(design doc R4). Every case exercises a `tmp_path`-rooted synthetic tree —
nothing here runs against the real `~/.omlx` log directory."""

import os
import time

from omlx.log_artifact_reaper import run_log_artifact_reaper


def _age(path, days):
    stamp = time.time() - days * 86400
    os.utime(path, (stamp, stamp))


class TestDatedServerLogRotationBackstop:
    def test_keeps_only_the_newest_seven(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        for day in range(1, 11):  # 10 dated files, e.g. server.log.2026-01-01..10
            p = log_dir / f"server.log.2026-01-{day:02d}"
            p.write_text("x")

        run_log_artifact_reaper(base_path=tmp_path, log_dir=log_dir)

        remaining = sorted(p.name for p in log_dir.glob("server.log.*"))
        assert len(remaining) == 7
        assert remaining == [f"server.log.2026-01-{d:02d}" for d in range(4, 11)]

    def test_within_bounds_is_a_no_op(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        for day in range(1, 4):
            (log_dir / f"server.log.2026-01-{day:02d}").write_text("x")

        run_log_artifact_reaper(base_path=tmp_path, log_dir=log_dir)

        assert len(list(log_dir.glob("server.log.*"))) == 3

    def test_malformed_dated_name_is_never_touched(self, tmp_path):
        """Only the exact TimedRotatingFileHandler suffix pattern counts —
        anything else wasn't created by log rotation."""
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        odd = log_dir / "server.log.not-a-date"
        odd.write_text("x")
        for day in range(1, 9):
            (log_dir / f"server.log.2026-01-{day:02d}").write_text("x")

        run_log_artifact_reaper(base_path=tmp_path, log_dir=log_dir)

        assert odd.exists()


class TestAdHocLogClass:
    def test_old_ad_hoc_logs_deleted(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        old_fork = log_dir / "fork-server-1.log"
        old_fork.write_text("x")
        _age(old_fork, 20)
        old_crash = log_dir / "crash.log"
        old_crash.write_text("x")
        _age(old_crash, 20)
        old_installed = log_dir / "installed-mini.log"
        old_installed.write_text("x")
        _age(old_installed, 20)

        result = run_log_artifact_reaper(base_path=tmp_path, log_dir=log_dir)

        assert not old_fork.exists()
        assert not old_crash.exists()
        assert not old_installed.exists()
        assert result.ad_hoc_logs_deleted == 3

    def test_recent_ad_hoc_logs_survive(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        recent = log_dir / "fork-server-1.log"
        recent.write_text("x")
        _age(recent, 2)

        run_log_artifact_reaper(base_path=tmp_path, log_dir=log_dir)

        assert recent.exists()

    def test_symlinked_ad_hoc_log_never_touched(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        outside = tmp_path / "outside.log"
        outside.write_text("do not delete me")
        _age(outside, 30)
        link = log_dir / "fork-linked.log"
        link.symlink_to(outside)

        run_log_artifact_reaper(base_path=tmp_path, log_dir=log_dir)

        assert outside.exists()


class TestWatchdogLogTruncation:
    def test_truncates_when_over_five_mib(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        watchdog = log_dir / "watchdog.log"
        watchdog.write_bytes(b"a" * (6 * 1024 * 1024) + b"TAIL_MARKER")

        result = run_log_artifact_reaper(base_path=tmp_path, log_dir=log_dir)

        assert result.watchdog_log_truncated is True
        content = watchdog.read_bytes()
        assert content.endswith(b"TAIL_MARKER")
        assert len(content) <= 1024 * 1024 + len(b"TAIL_MARKER")

    def test_leaves_small_watchdog_log_alone(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        watchdog = log_dir / "watchdog.log"
        watchdog.write_bytes(b"small")

        result = run_log_artifact_reaper(base_path=tmp_path, log_dir=log_dir)

        assert result.watchdog_log_truncated is False
        assert watchdog.read_bytes() == b"small"


class TestLogDirCap:
    def test_evicts_oldest_first_when_over_cap(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        # Live server.log — must never be evicted regardless of age.
        server_log = log_dir / "server.log"
        server_log.write_bytes(b"x" * (300 * 1024 * 1024))
        _age(server_log, 100)

        oldest = log_dir / "old-thing.log"
        oldest.write_bytes(b"x" * (150 * 1024 * 1024))
        _age(oldest, 50)

        newest = log_dir / "new-thing.log"
        newest.write_bytes(b"x" * (150 * 1024 * 1024))
        _age(newest, 1)

        result = run_log_artifact_reaper(base_path=tmp_path, log_dir=log_dir)

        assert server_log.exists()
        assert not oldest.exists()
        assert newest.exists()
        assert result.cap_evicted_count == 1

    def test_under_cap_is_a_no_op(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        small = log_dir / "small.log"
        small.write_bytes(b"x" * 1024)

        result = run_log_artifact_reaper(base_path=tmp_path, log_dir=log_dir)

        assert small.exists()
        assert result.cap_evicted_count == 0


class TestHfDownloadStaging:
    def test_old_temp_dir_removed(self, tmp_path):
        models_dir = tmp_path / "models"
        temp_dir = (
            models_dir / "org" / "model" / ".cache" / "huggingface"
            / "download" / "._____temp"
        )
        temp_dir.mkdir(parents=True)
        (temp_dir / "partial.bin").write_bytes(b"x")
        _age(temp_dir, 10)

        result = run_log_artifact_reaper(
            base_path=tmp_path, log_dir=tmp_path / "logs", models_dir=models_dir
        )

        assert not temp_dir.exists()
        assert result.hf_staging_removed >= 1

    def test_recent_temp_dir_survives_as_an_active_download(self, tmp_path):
        models_dir = tmp_path / "models"
        temp_dir = (
            models_dir / "org" / "model" / ".cache" / "huggingface"
            / "download" / "._____temp"
        )
        temp_dir.mkdir(parents=True)
        (temp_dir / "partial.bin").write_bytes(b"x")
        _age(temp_dir, 1)

        run_log_artifact_reaper(
            base_path=tmp_path, log_dir=tmp_path / "logs", models_dir=models_dir
        )

        assert temp_dir.exists()

    def test_old_lock_file_removed(self, tmp_path):
        models_dir = tmp_path / "models"
        cache_dir = models_dir / "org" / "model" / ".cache" / "huggingface"
        cache_dir.mkdir(parents=True)
        lock = cache_dir / "some.lock"
        lock.write_text("x")
        _age(lock, 10)

        run_log_artifact_reaper(
            base_path=tmp_path, log_dir=tmp_path / "logs", models_dir=models_dir
        )

        assert not lock.exists()


class TestPycache:
    def test_pycache_at_base_removed(self, tmp_path):
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "mod.cpython-311.pyc").write_bytes(b"x")

        result = run_log_artifact_reaper(base_path=tmp_path, log_dir=tmp_path / "logs")

        assert not pycache.exists()
        assert result.pycache_removed is True


class TestMissingDirsAreNoOps:
    def test_missing_log_dir_is_a_no_op(self, tmp_path):
        run_log_artifact_reaper(base_path=tmp_path, log_dir=tmp_path / "does-not-exist")

    def test_missing_models_dir_is_a_no_op(self, tmp_path):
        run_log_artifact_reaper(
            base_path=tmp_path,
            log_dir=tmp_path / "logs",
            models_dir=tmp_path / "does-not-exist",
        )
