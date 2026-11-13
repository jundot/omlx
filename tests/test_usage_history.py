# SPDX-License-Identifier: Apache-2.0
"""Historical aggregation, persistence, calendar boundaries, and failure isolation."""

import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pytest

from omlx.server_metrics import ServerMetrics
from omlx.usage_history import UsageHistory, _bounds, _hour


@pytest.fixture
def history(tmp_path):
    recorder = UsageHistory(tmp_path / "nested" / "usage.sqlite3")
    yield recorder
    recorder.close()


def record(history, model="model-a", timestamp=None, **overrides):
    values = dict(
        model_id=model,
        prompt_tokens=100,
        completion_tokens=20,
        cached_tokens=60,
        prefill_duration=0.5,
        generation_duration=2.0,
        request_duration=3.0,
        timestamp=timestamp,
    )
    values.update(overrides)
    history.record(**values)


def test_first_run_schema_and_empty_range(history):
    assert history.available
    with sqlite3.connect(history.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        columns = [
            row[1] for row in db.execute("PRAGMA table_info(model_usage_hourly)")
        ]
    assert columns == [
        "timestamp_hour",
        "model_id",
        "requests",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "prefill_seconds",
        "generation_seconds",
        "request_seconds",
        "timed_requests",
    ]
    result = history.query("7d", include_details=True)
    assert result["totals"]["requests"] == 0
    assert len(result["daily"]) == len(result["heatmap"]) == 7
    assert result["totals"]["generation_tps"] is None


def test_aggregation_models_cache_and_weighted_speed(history):
    record(history)
    record(history, generation_duration=8.0)
    record(history, "model-b", cached_tokens=0)
    assert history.flush()
    result = history.query(include_details=True)
    totals = result["totals"]
    assert totals["requests"] == 3
    assert totals["prompt_tokens"] == 300
    assert totals["completion_tokens"] == 60
    assert totals["total_tokens"] == 360
    assert totals["cached_tokens"] == 120
    assert totals["cache_efficiency"] == 0.4
    assert totals["generation_tps"] == 5
    assert totals["prefill_tps"] == 120
    assert totals["average_request_seconds"] == 3
    assert sum(sum(day["tokens"]) for day in result["heatmap"]) == 360
    assert sum(day["total_tokens"] for day in result["daily"]) == 360
    assert sum(hour["total_tokens"] for hour in result["hourly"]) == 360
    assert history.query(model="model-a")["totals"]["generation_tps"] == 4
    assert history.query(model="missing")["totals"]["requests"] == 0
    with sqlite3.connect(history.path) as db:
        assert db.execute("SELECT count(*) FROM model_usage_hourly").fetchone()[0] == 2


def test_no_sqlite_on_record_path(history):
    with patch.object(history, "_connect", side_effect=AssertionError("inline I/O")):
        for _ in range(50):
            record(history)
    assert len(history._pending) == 1
    assert history.flush()
    assert history.query()["totals"]["requests"] == 50


def test_shutdown_flush_and_restart(history):
    record(history)
    history.close()
    restarted = UsageHistory(history.path)
    try:
        assert restarted.query()["totals"]["requests"] == 1
        record(restarted, "model-b")
        restarted.flush()
        assert len(restarted.query()["models"]) == 2
    finally:
        restarted.close()


def test_concurrent_records_and_flushes(history):
    def writer(index):
        for n in range(200):
            record(history, f"model-{index % 3}")
            if n % 50 == 0:
                history.flush()
                history.query()

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(writer, range(8)))
    assert history.flush()
    result = history.query()
    assert result["totals"]["requests"] == 1600
    assert result["totals"]["total_tokens"] == 192000
    assert len(result["models"]) == 3


def test_background_writer(tmp_path, monkeypatch):
    monkeypatch.setattr("omlx.usage_history.FLUSH_SECONDS", 0.02)
    history = UsageHistory(tmp_path / "usage.sqlite3")
    try:
        record(history)
        deadline = time.monotonic() + 3
        while (
            history.query()["totals"]["requests"] == 0 and time.monotonic() < deadline
        ):
            threading.Event().wait(0.01)
        assert history.query()["totals"]["requests"] == 1
    finally:
        history.close()


def test_deleted_database_recreated(history):
    record(history)
    history.flush()
    history.path.unlink()
    record(history, "model-b")
    assert history.flush()
    assert history.query()["totals"]["requests"] == 1


def test_corrupt_storage_preserved_and_recreated(tmp_path):
    path = tmp_path / "usage.sqlite3"
    path.write_bytes(b"not a sqlite database")
    history = UsageHistory(path)
    try:
        assert history.available
        assert (
            path.with_suffix(".sqlite3.corrupt").read_bytes()
            == b"not a sqlite database"
        )
        record(history)
        assert history.flush()
        assert history.query()["totals"]["requests"] == 1
    finally:
        history.close()


def test_future_schema_not_replaced(tmp_path):
    path = tmp_path / "usage.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=99")
    history = UsageHistory(path)
    try:
        assert not history.available
        record(history)
        assert not history.flush()
        with sqlite3.connect(path) as db:
            assert db.execute("PRAGMA user_version").fetchone()[0] == 99
        assert not path.with_suffix(".sqlite3.corrupt").exists()
    finally:
        history.close()


def test_invalid_storage_location_does_not_break_metrics(tmp_path):
    parent = tmp_path / "file"
    parent.write_text("a file cannot be a directory")
    history = UsageHistory(parent / "usage.sqlite3")
    try:
        metrics = ServerMetrics()
        metrics.usage_history = history
        metrics.record_request_complete(100, 20, model_id="model-a")
        assert metrics.get_snapshot()["total_requests"] == 1
        assert not history.flush()
    finally:
        history.close()


def test_recording_exception_does_not_break_serving_stats():
    metrics = ServerMetrics()
    metrics.usage_history = Mock()
    metrics.usage_history.record.side_effect = RuntimeError("analytics failed")
    metrics.record_request_complete(100, 20, model_id="canonical-model")
    assert metrics.get_snapshot()["total_tokens_served"] == 120
    assert (
        metrics.usage_history.record.call_args.kwargs["model_id"] == "canonical-model"
    )


def test_failed_transaction_retried_once(history):
    record(history)
    with patch.object(
        history, "_connect", side_effect=sqlite3.OperationalError("locked")
    ):
        assert not history.flush()
        record(history)
    assert history.flush()
    assert history.flush()
    assert history.query()["totals"]["requests"] == 2


def test_pending_buffer_bounded(history, monkeypatch):
    monkeypatch.setattr("omlx.usage_history._MAX_PENDING_BUCKETS", 2)
    for i in range(10):
        record(history, f"model-{i}")
    assert len(history._pending) == 2
    assert history.dropped_requests == 8


def test_retention(history):
    now = time.time()
    record(history, timestamp=now - 401 * 86400)
    record(history, timestamp=now - 399 * 86400)
    record(history, timestamp=now)
    assert history.flush()
    with sqlite3.connect(history.path) as db:
        assert (
            db.execute("SELECT sum(requests) FROM model_usage_hourly").fetchone()[0]
            == 2
        )


def test_unknown_audio_timing_and_invalid_cache(history):
    record(
        history,
        "audio",
        prompt_tokens=0,
        completion_tokens=0,
        cached_tokens=0,
        request_duration=None,
        prefill_duration=0,
        generation_duration=0,
    )
    record(history, cached_tokens=101)
    record(history, generation_duration=float("nan"))
    history.flush()
    result = history.query()["totals"]
    assert result["requests"] == 1
    assert result["total_tokens"] == 0
    assert result["timed_requests"] == 0
    assert result["average_request_seconds"] is None


def test_sql_model_id_is_only_data(history):
    model = "model'); DROP TABLE model_usage_hourly;--"
    record(history, model)
    history.flush()
    assert history.query(model=model)["totals"]["requests"] == 1
    assert history.query()["totals"]["requests"] == 1


@pytest.fixture
def timezone_env():
    previous = os.environ.get("TZ")

    def set_timezone(name):
        os.environ["TZ"] = name
        time.tzset()

    yield set_timezone
    if previous is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = previous
    time.tzset()


@pytest.mark.parametrize("zone", ["America/New_York", "Asia/Kathmandu"])
def test_calendar_ranges(history, timezone_env, zone):
    timezone_env(zone)
    now = time.time()
    midnight = datetime.fromtimestamp(now).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    for days_ago in (0, 1, 6, 7, 29, 30, 89, 90):
        record(history, timestamp=(midnight - timedelta(days=days_ago)).timestamp())
    history.flush()
    for period, expected in [
        ("today", 1),
        ("yesterday", 1),
        ("7d", 3),
        ("30d", 5),
        ("90d", 7),
    ]:
        assert history.query(period, now=now)["totals"]["requests"] == expected
    start, _ = _bounds("month", now)
    assert start.day == 1
    with pytest.raises(ValueError):
        history.query("forever")


@pytest.mark.parametrize("date,hours", [("2026-03-08", 23), ("2026-11-01", 25)])
def test_dst_day_lengths(timezone_env, date, hours):
    timezone_env("America/New_York")
    now = datetime.fromisoformat(date + "T12:00").timestamp()
    start, end = _bounds("today", now)
    assert end.timestamp() - start.timestamp() == hours * 3600


def test_repeated_dst_hour_separate_storage_combined_heatmap(history, timezone_env):
    timezone_env("America/New_York")
    zone = ZoneInfo("America/New_York")
    first = datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=0).timestamp()
    second = datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=1).timestamp()
    assert _hour(second) - _hour(first) == 3600
    record(history, timestamp=first)
    record(history, timestamp=second)
    history.flush()
    result = history.query(now=second, include_details=True)
    assert len(result["hourly"]) == 2
    assert result["heatmap"][0]["tokens"][1] == 240


def test_locked_sqlite_does_not_hold_recording_lock(history):
    record(history)
    with sqlite3.connect(history.path) as db:
        db.execute("BEGIN IMMEDIATE")
        with ThreadPoolExecutor(max_workers=2) as executor:
            flushing = executor.submit(history.flush)
            # Recording must finish while SQLite's writer lock is still held.
            executor.submit(record, history).result(timeout=0.5)
            assert not flushing.result(timeout=3)
        db.rollback()
    assert history.flush()
    assert history.query()["totals"]["requests"] == 2


def test_server_metrics_initializes_and_closes_history(tmp_path):
    from omlx.server_metrics import get_server_metrics, reset_server_metrics

    path = tmp_path / "stats.json"
    try:
        reset_server_metrics(stats_path=path)
        old = get_server_metrics()
        old.record_request_complete(100, 20, model_id="model-a")
        reset_server_metrics(stats_path=path)
        assert not old.usage_history._thread.is_alive()
        current = get_server_metrics()
        assert current.usage_history.query()["totals"]["requests"] == 1
        assert current.get_snapshot()["total_requests"] == 0
        assert current.get_snapshot(scope="alltime")["total_requests"] == 1
    finally:
        reset_server_metrics()


def test_runtime_toggle_flushes_then_stops_recording_and_resumes(history):
    record(history)
    history.set_enabled(False)  # flushes the pending aggregate synchronously
    assert history.enabled is False
    with sqlite3.connect(history.path) as db:
        assert (
            db.execute("SELECT SUM(requests) FROM model_usage_hourly").fetchone()[0]
            == 1
        )
    record(history)  # ignored while off
    assert not history._pending
    assert history.flush()
    disabled = history.query("7d")
    assert disabled["enabled"] is False
    assert disabled["totals"]["requests"] == 0
    assert disabled["models"] == []
    assert len(disabled["heatmap"]) == 7
    with sqlite3.connect(history.path) as db:
        assert (
            db.execute("SELECT SUM(requests) FROM model_usage_hourly").fetchone()[0]
            == 1
        )
    history.set_enabled(True)
    record(history)
    assert history.flush()
    result = history.query("7d")
    assert result["enabled"] is True
    assert result["totals"]["requests"] == 2


@pytest.mark.parametrize("storage", ["missing", "valid", "corrupt"])
def test_disabled_startup_leaves_storage_untouched(tmp_path, storage):
    path = tmp_path / "usage.sqlite3"
    if storage == "valid":
        existing = UsageHistory(path)
        record(existing)
        existing.close()
    elif storage == "corrupt":
        path.write_bytes(b"not a sqlite database")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with patch("omlx.usage_history.sqlite3.connect") as connect:
        history = UsageHistory(path, enabled=False)
        try:
            record(history)
            assert history.flush()
            assert history.query()["enabled"] is False
            history.set_enabled(False)
        finally:
            history.close()
        connect.assert_not_called()
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before

    history = UsageHistory(path, enabled=False)
    try:
        history.set_enabled(True)
        assert history.available
        record(history)
        assert history.flush()
        assert history.query()["totals"]["requests"] == (2 if storage == "valid" else 1)
        if storage == "corrupt":
            assert (
                path.with_suffix(".sqlite3.corrupt").read_bytes() == before[path.name]
            )
    finally:
        history.close()


def test_idle_flush_skips_database_until_retention_due(history):
    assert history.flush()
    with patch.object(history, "_connect", wraps=history._connect) as connect:
        for _ in range(3):
            assert history.flush()
        connect.assert_not_called()
        record(history)
        assert history.flush()
        assert connect.call_count == 1

    record(history, timestamp=time.time() - 401 * 86400)
    assert history.flush()
    history._last_prune -= 86400
    assert not history._pending
    assert history.flush()
    with sqlite3.connect(history.path) as connection:
        assert (
            connection.execute(
                "SELECT SUM(requests) FROM model_usage_hourly"
            ).fetchone()[0]
            == 1
        )


def test_disabled_failed_flush_retries_pending_requests(history):
    record(history)
    with patch.object(
        history, "_connect", side_effect=sqlite3.OperationalError("locked")
    ):
        history.set_enabled(False)
    assert history._pending
    assert history.flush()
    assert not history._pending
    history.set_enabled(True)
    assert history.query()["totals"]["requests"] == 1


def test_server_metrics_respects_disabled_recorder(tmp_path):
    metrics = ServerMetrics()
    metrics.usage_history = UsageHistory(tmp_path / "usage.sqlite3", enabled=False)
    try:
        metrics.record_request_complete(100, 20, 60, 0.5, 1.0, "model-a", 2.0)
        assert metrics.get_snapshot()["total_requests"] == 1
        metrics.usage_history.set_enabled(True)
        metrics.record_request_complete(100, 20, 60, 0.5, 1.0, "model-a", 2.0)
        assert metrics.usage_history.flush()
        assert metrics.usage_history.query()["totals"]["requests"] == 1
    finally:
        metrics.close()


# Per-client history (opt-in)

EDITOR_DESK = ("sub_key", "Editor", "192.168.1.20")
EDITOR_LOCAL = ("sub_key", "Editor", "127.0.0.1")
MAIN_LOCAL = ("main_key", "", "127.0.0.1")
KEYLESS_LOCAL = ("none", "", "127.0.0.1")


def _pairs(history, **query):
    return {
        (row["key_kind"], row["key_id"], row["client_ip"]): row
        for row in history.query(**query)["clients"]
    }


def _by_key(history, **query):
    return {
        (row["key_kind"], row["key_id"]): row
        for row in history.query(**query)["clients_by_key"]
    }


def _by_ip(history, **query):
    return {row["client_ip"]: row for row in history.query(**query)["clients_by_ip"]}


def test_client_attribution_off_by_default(history):
    assert history.by_client is False
    record(history, client=EDITOR_DESK)
    assert history.flush()
    result = history.query()
    assert result["by_client"] is False
    assert result["clients"] == result["clients_by_key"] == []
    assert result["clients_by_ip"] == []
    assert result["totals"]["requests"] == 1
    with sqlite3.connect(history.path) as db:
        assert db.execute("SELECT count(*) FROM client_usage_hourly").fetchone()[0] == 0


def test_client_views_pair_key_and_ip(history):
    history.set_by_client(True)
    record(history, "model-a", client=EDITOR_DESK)
    record(history, "model-b", client=EDITOR_DESK)
    record(history, "model-a", client=EDITOR_LOCAL)
    record(history, "model-a", client=MAIN_LOCAL, completion_tokens=500)
    record(history, "model-a", client=KEYLESS_LOCAL)
    record(history, "model-a")  # no request context: model totals only
    assert history.flush()
    result = history.query()
    assert result["totals"]["requests"] == 6

    pairs = _pairs(history)
    assert set(pairs) == {EDITOR_DESK, EDITOR_LOCAL, MAIN_LOCAL, KEYLESS_LOCAL}
    assert pairs[EDITOR_DESK]["requests"] == 2
    assert result["clients"][0]["key_kind"] == "main_key"  # most tokens first

    keys = _by_key(history)
    assert keys[("sub_key", "Editor")]["requests"] == 3
    assert keys[("sub_key", "Editor")]["total_tokens"] == 360
    assert keys[("main_key", "")]["requests"] == 1
    assert keys[("none", "")]["requests"] == 1

    ips = _by_ip(history)
    # Every request is grouped by machine, whichever key it used.
    assert ips["127.0.0.1"]["requests"] == 3
    assert ips["192.168.1.20"]["requests"] == 2
    assert ips["192.168.1.20"]["generation_tps"] == 10.0

    assert list(_by_key(history, model="model-b")) == [("sub_key", "Editor")]
    assert list(_by_ip(history, model="model-b")) == ["192.168.1.20"]


def test_client_rows_kept_after_tracking_turned_off(history):
    history.set_by_client(True)
    record(history, client=EDITOR_DESK)
    assert history.flush()
    history.set_by_client(False)
    record(history, client=EDITOR_DESK)
    assert history.flush()
    result = history.query()
    assert result["by_client"] is False
    assert _pairs(history)[EDITOR_DESK]["requests"] == 1
    assert result["totals"]["requests"] == 2


@pytest.mark.parametrize(
    "client",
    [
        ("ip", "", "127.0.0.1"),
        ("sub_key", 7, "127.0.0.1"),
        ("sub_key", "x" * 257, "127.0.0.1"),
        ("sub_key", "Editor", "x" * 300),
        ("none", "", None),
    ],
)
def test_invalid_client_label_ignored_but_model_counted(history, client):
    history.set_by_client(True)
    record(history, client=client)
    assert history.flush()
    assert history.query()["clients"] == []
    assert history.query()["totals"]["requests"] == 1


def test_client_buffer_bounded_separately(history, monkeypatch):
    monkeypatch.setattr("omlx.usage_history._MAX_PENDING_BUCKETS", 2)
    history.set_by_client(True)
    for i in range(5):
        record(history, client=("none", "", f"10.0.0.{i}"))
    # One model bucket, two client buckets: overflowing clients never drops models.
    assert len(history._pending) == 1
    assert len(history._pending_clients) == 2
    assert history.dropped_requests == 0
    assert history.dropped_client_requests == 3


def test_client_batch_retried_after_failed_flush(history):
    history.set_by_client(True)
    record(history, client=EDITOR_DESK)
    with patch.object(
        history, "_connect", side_effect=sqlite3.OperationalError("locked")
    ):
        assert not history.flush()
        record(history, client=EDITOR_DESK)
    assert history.flush()
    assert _pairs(history)[EDITOR_DESK]["requests"] == 2


def test_client_retention(history):
    history.set_by_client(True)
    now = time.time()
    record(history, timestamp=now - 401 * 86400, client=EDITOR_DESK)
    record(history, timestamp=now, client=EDITOR_DESK)
    assert history.flush()
    with sqlite3.connect(history.path) as db:
        assert (
            db.execute("SELECT sum(requests) FROM client_usage_hourly").fetchone()[0]
            == 1
        )


def test_client_label_is_only_data(history):
    history.set_by_client(True)
    label = "x'); DROP TABLE model_usage_hourly; --"
    record(history, client=("sub_key", label, "127.0.0.1"))
    assert history.flush()
    assert _by_key(history)[("sub_key", label)]["requests"] == 1
    assert history.query()["totals"]["requests"] == 1


def test_existing_v1_database_gains_client_table_without_version_bump(tmp_path):
    path = tmp_path / "usage.sqlite3"
    first = UsageHistory(path)
    record(first)
    first.close()
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE client_usage_hourly")
    # A file from an older build: queries degrade to no clients until the
    # writer adds the table, and model history is untouched.
    reader = UsageHistory(path, enabled=False)
    reader.enabled = True
    assert reader.query()["clients"] == []
    assert reader.query()["clients_by_ip"] == []
    assert reader.query()["totals"]["requests"] == 1
    reader.close()
    upgraded = UsageHistory(path, by_client=True)
    record(upgraded, client=EDITOR_DESK)
    assert upgraded.flush()
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
    assert upgraded.query()["totals"]["requests"] == 2
    assert _pairs(upgraded)[EDITOR_DESK]["requests"] == 1
    upgraded.close()


def test_server_metrics_passes_request_client(tmp_path):
    from omlx import client_identity

    metrics = ServerMetrics()
    metrics.usage_history = UsageHistory(tmp_path / "usage.sqlite3", by_client=True)
    slot = client_identity._ClientSlot("192.168.1.20")
    slot.key_kind, slot.key_label = "sub_key", "Editor"
    token = client_identity._current.set(slot)
    try:
        metrics.record_request_complete(10, 2, 0, 0.1, 0.1, "model-a", 0.2)
    finally:
        client_identity._current.reset(token)
    metrics.record_request_complete(10, 2, 0, 0.1, 0.1, "model-a", 0.2)
    metrics.usage_history.flush()
    pairs = _pairs(metrics.usage_history)
    assert list(pairs) == [EDITOR_DESK]
    assert pairs[EDITOR_DESK]["requests"] == 1
    metrics.close()
