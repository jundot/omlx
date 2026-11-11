# SPDX-License-Identifier: Apache-2.0
"""Tests for logging configuration filters."""

import logging

from omlx.logging_config import AdminStatsAccessFilter


class TestAdminStatsAccessFilter:
    """Tests for the admin polling access log filter."""

    def setup_method(self):
        self.filter = AdminStatsAccessFilter()

    def _make_record(self, msg: str) -> logging.LogRecord:
        return logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_suppresses_admin_stats(self):
        record = self._make_record('127.0.0.1 - "GET /admin/api/stats HTTP/1.1" 200')
        assert self.filter.filter(record) is False

    def test_suppresses_admin_stats_with_params(self):
        record = self._make_record(
            '127.0.0.1 - "GET /admin/api/stats?scope=alltime HTTP/1.1" 200'
        )
        assert self.filter.filter(record) is False

    def test_suppresses_admin_login(self):
        record = self._make_record(
            '127.0.0.1 - "POST /admin/api/login HTTP/1.1" 200'
        )
        assert self.filter.filter(record) is False

    def test_allows_other_requests(self):
        record = self._make_record('127.0.0.1 - "GET /v1/models HTTP/1.1" 200')
        assert self.filter.filter(record) is True

    def test_allows_health_check(self):
        record = self._make_record('127.0.0.1 - "GET /health HTTP/1.1" 200')
        assert self.filter.filter(record) is True

    def test_allows_chat_completions(self):
        record = self._make_record(
            '127.0.0.1 - "POST /v1/chat/completions HTTP/1.1" 200'
        )
        assert self.filter.filter(record) is True


class TestSupervisedConsoleLevel:
    """Under a supervisor the console keeps warnings and errors only.

    launchd (brew services) and the menubar app capture stdout/stderr into a
    file that never rotates; server.log already has every record.
    """

    def test_plain_serve_keeps_the_level(self, monkeypatch):
        from omlx.logging_config import console_log_level

        monkeypatch.delenv("OMLX_SUPERVISED", raising=False)
        assert console_log_level(logging.INFO) == logging.INFO
        assert console_log_level(5) == 5

    def test_supervised_raises_to_warning(self, monkeypatch):
        from omlx.logging_config import console_log_level

        monkeypatch.setenv("OMLX_SUPERVISED", "launchd")
        assert console_log_level(logging.INFO) == logging.WARNING
        assert console_log_level(5) == logging.WARNING
        assert console_log_level(logging.ERROR) == logging.ERROR

    def test_file_handler_is_not_a_console_handler(self, tmp_path):
        from omlx.logging_config import is_console_handler

        fh = logging.FileHandler(tmp_path / "x.log")
        try:
            assert is_console_handler(logging.StreamHandler()) is True
            assert is_console_handler(fh) is False
        finally:
            fh.close()

    def test_runtime_level_change_keeps_console_quiet(self, monkeypatch, tmp_path):
        from omlx.admin.routes import _apply_log_level_runtime

        monkeypatch.setenv("OMLX_SUPERVISED", "launchd")
        root = logging.getLogger()
        console = logging.StreamHandler()
        fh = logging.FileHandler(tmp_path / "server.log")
        old_level = root.level
        root.addHandler(console)
        root.addHandler(fh)
        try:
            _apply_log_level_runtime("debug")
            assert console.level == logging.WARNING
            assert fh.level == logging.DEBUG
        finally:
            root.removeHandler(console)
            root.removeHandler(fh)
            fh.close()
            root.setLevel(old_level)
