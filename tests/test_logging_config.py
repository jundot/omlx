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
