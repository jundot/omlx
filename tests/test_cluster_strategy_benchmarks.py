# SPDX-License-Identifier: Apache-2.0
"""Durable end-to-end strategy measurement tests."""

import json

import pytest

from omlx.cluster.strategy_benchmarks import (
    StrategyBenchmark,
    StrategyBenchmarkStore,
    context_bucket,
)


def _measurement(*, tp=1, context=8000, prefill=100.0, decode=20.0, at="2026-07-29T12:00:00+00:00"):
    return StrategyBenchmark(
        model="org/model",
        node_ids=("mbp", "studio"),
        backend="jaccl",
        tensor_parallel_size=tp,
        context_tokens=context,
        prompt_tokens_per_second=prefill,
        decode_tokens_per_second=decode,
        time_to_first_token_seconds=2.0,
        measured_at=at,
    )


def test_context_buckets_match_the_hardware_gate_lengths():
    assert [context_bucket(value) for value in (1, 1024, 1025, 8192, 8193, 32768, 32769, 262144)] == [
        1024,
        1024,
        8192,
        8192,
        32768,
        32768,
        262144,
        262144,
    ]


def test_store_round_trips_and_uses_recent_medians(tmp_path):
    store = StrategyBenchmarkStore(tmp_path)
    store.record(_measurement(prefill=10, decode=4))
    store.record(_measurement(prefill=100, decode=40, at="2026-07-29T12:01:00+00:00"))
    store.record(_measurement(prefill=30, decode=12, at="2026-07-29T12:02:00+00:00"))
    store.record(_measurement(tp=2, prefill=50, decode=25))

    restored = StrategyBenchmarkStore(tmp_path)
    results = restored.measurements(
        model="org/model",
        node_ids=("mbp", "studio"),
        backend="jaccl",
        target_context_tokens=8192,
    )

    assert set(results) == {1, 2}
    assert results[1].prompt_tokens_per_second == 30
    assert results[1].decode_tokens_per_second == 12
    assert results[1].samples == 3
    assert json.loads(store.path.read_text())["schema_version"] == 1
    assert store.path.stat().st_mode & 0o777 == 0o600


def test_store_does_not_reuse_measurements_for_reversed_rank_order(tmp_path):
    store = StrategyBenchmarkStore(tmp_path)
    store.record(_measurement())

    results = store.measurements(
        model="org/model",
        node_ids=("studio", "mbp"),
        backend="jaccl",
        target_context_tokens=8192,
    )

    assert results == {}


def test_corrupt_store_fails_closed_without_losing_server_startup(tmp_path):
    path = tmp_path / "cluster" / "strategy-benchmarks.json"
    path.parent.mkdir()
    path.write_text("{broken")

    store = StrategyBenchmarkStore(tmp_path)

    assert store.load_error
    assert store.to_dict()["benchmarks"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_tokens_per_second", 0),
        ("decode_tokens_per_second", float("nan")),
        ("time_to_first_token_seconds", -1),
    ],
)
def test_measurements_reject_unusable_rates(field, value):
    payload = _measurement().to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match="finite and positive"):
        StrategyBenchmark.from_dict(payload)
