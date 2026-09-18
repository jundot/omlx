# SPDX-License-Identifier: Apache-2.0

import threading
import time
from pathlib import Path

import pytest

from omlx.cluster.sample_store import SampleStore
from omlx.server_metrics import ServerMetrics


def _make_store(tmp_path: Path, **kwargs) -> SampleStore:
    return SampleStore(tmp_path / "sample-store.json", **kwargs)


def test_sample_store_roundtrip(tmp_path):
    store = _make_store(tmp_path)
    ts = 1_700_000_000.0
    store.record(ts=ts, request_count=5, error_count=1, ttft_sum=2.5, tpot_sum=1.2)
    store.record(ts=ts + 60, request_count=3, error_count=0, ttft_sum=1.0, tpot_sum=0.5)
    store.flush()

    samples = store.query(ts - 1, ts + 120)
    assert len(samples) == 2
    assert samples[0]["request_count"] == 5
    assert samples[1]["error_count"] == 0


def test_sample_store_survives_restart(tmp_path):
    ts = 1_700_000_000.0

    store = _make_store(tmp_path)
    store.record(ts=ts, request_count=7, error_count=2, ttft_sum=3.0, tpot_sum=1.0)
    store.record(ts=ts + 60, request_count=4, error_count=0, ttft_sum=1.5, tpot_sum=0.8)
    store.flush()

    store2 = _make_store(tmp_path)
    samples = store2.query(ts - 1, ts + 120)
    assert len(samples) == 2
    assert samples[0]["request_count"] == 7
    assert samples[1]["tpot_sum"] == 0.8


def test_retention_evicts_old_samples(tmp_path):
    store = _make_store(tmp_path, retention_days=2)
    now = time.time()

    store.record(ts=now - 5 * 86400, request_count=1)
    store.record(ts=now, request_count=2)

    samples = store.query(now - 6 * 86400, now + 1)
    assert len(samples) == 1
    assert samples[0]["request_count"] == 2


def test_bounded_disk_growth(tmp_path):
    store = _make_store(tmp_path, retention_days=1, sample_interval=60)
    now = time.time()

    for i in range(1000):
        store.record(
            ts=now - 3 * 86400 + i * (3 * 86400 / 1000),
            request_count=1,
            error_count=0,
            ttft_sum=0.0,
            tpot_sum=0.0,
        )
    store.flush()

    assert store.sample_count() < 500
    file_size = tmp_path.joinpath("sample-store.json").stat().st_size
    assert file_size < 200_000


def test_range_query(tmp_path):
    store = _make_store(tmp_path)
    for i in range(10):
        store.record(ts=1_700_000_000.0 + i * 60, request_count=i + 1)

    samples = store.query(1_700_000_000.0, 1_700_000_000.0 + 4 * 60)
    assert len(samples) == 5

    samples = store.query(1_700_000_000.0 + 2 * 60, 1_700_000_000.0 + 2 * 60)
    assert len(samples) == 1


def test_downsampling(tmp_path):
    samples = [
        {"ts": float(i), "request_count": i, "error_count": 0, "ttft_sum": 0.0, "tpot_sum": 0.0}
        for i in range(100)
    ]
    down = SampleStore.downsample(samples, 10)
    assert len(down) <= 10
    total = sum(d["request_count"] for d in down)
    assert total == sum(s["request_count"] for s in samples)

    assert SampleStore.downsample([], 10) == []
    assert SampleStore.downsample(samples, 0) == []


def test_thread_safety(tmp_path):
    store = _make_store(tmp_path)
    errors: list[Exception] = []

    def writer(start: float):
        try:
            for i in range(200):
                store.record(
                    ts=start + i * 0.001,
                    request_count=1,
                    error_count=0,
                    ttft_sum=0.0,
                    tpot_sum=0.0,
                )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i * 1000.0,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert store.sample_count() == 800


def test_server_metrics_unchanged(tmp_path):
    sm = ServerMetrics()
    sm.record_request_complete(
        prompt_tokens=100,
        completion_tokens=50,
        cached_tokens=10,
        prefill_duration=0.1,
        generation_duration=0.2,
        model_id="test-model",
    )
    sm.record_request_complete(
        prompt_tokens=200,
        completion_tokens=100,
        cached_tokens=20,
        prefill_duration=0.3,
        generation_duration=0.4,
        model_id="test-model",
    )

    snap = sm.get_snapshot(scope="session")
    assert snap["total_prompt_tokens"] == 300
    assert snap["total_completion_tokens"] == 150
    assert snap["total_requests"] == 2

    snap_at = sm.get_snapshot(scope="alltime")
    assert snap_at["total_prompt_tokens"] == 300
    assert snap_at["total_requests"] == 2
