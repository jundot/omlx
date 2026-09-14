# SPDX-License-Identifier: Apache-2.0
"""Rolling dashboard scopes use hourly buckets, not request timestamps."""
from datetime import datetime
from unittest.mock import patch

import pytest

from omlx.server_metrics import ServerMetrics
from omlx.usage_history import UsageHistory, _hour


@pytest.fixture
def history(tmp_path):
    recorder = UsageHistory(tmp_path / "usage.sqlite3")
    yield recorder
    recorder.close()


def record(history, when, model="a"):
    history.record(model_id=model, timestamp=when, prompt_tokens=100,
                   completion_tokens=20, cached_tokens=60,
                   prefill_duration=0.5, generation_duration=2.0)


@pytest.mark.parametrize("hours", [12, 24])
def test_window_includes_oldest_hour_bucket(history, hours):
    now = datetime.now().replace(hour=14, minute=30, second=0, microsecond=0).timestamp()
    first = _hour(now - hours * 3600)
    record(history, first - 1)
    record(history, first + 1)
    record(history, now - 60, "b")
    assert history.flush()
    result = history.query(f"{hours}h", now=now, include_details=True)
    assert result["totals"]["requests"] == 2
    assert history.query(f"{hours}h", model="a", now=now)["totals"]["requests"] == 1
    assert result["totals"]["prefill_tps"] == 80.0
    assert result["totals"]["generation_tps"] == 10.0


def test_disabled_after_start_is_unavailable(history):
    metrics = ServerMetrics()
    metrics.usage_history = history
    history.set_enabled(False)
    assert metrics.get_snapshot(scope="12h")["history_available"] is False


def test_session_alltime_and_rolling_are_independent(history):
    metrics = ServerMetrics()
    metrics.usage_history = history
    record(history, datetime.now().timestamp() - 60)
    history.flush()
    assert metrics.get_snapshot(scope="24h")["total_requests"] == 1
    assert metrics.get_snapshot(scope="session")["total_requests"] == 0
    assert metrics.get_snapshot(scope="alltime")["total_requests"] == 0
    assert metrics.get_snapshot(model_id="missing", scope="24h")["total_requests"] == 0
    with patch.object(history, "query", side_effect=OSError("test failure")):
        assert metrics.get_snapshot(scope="12h")["history_available"] is False


def test_calendar_query_keeps_exclusive_midnight(history):
    now = datetime.now().timestamp()
    assert len(history.query("today", now=now)["heatmap"]) == 1
    assert len(history.query("7d", now=now)["heatmap"]) == 7
