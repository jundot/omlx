# SPDX-License-Identifier: Apache-2.0
"""The persisted measurement record.

A measurement is only as useful as the plan built from it, so these tests pin
the qualities that make a stored record trustworthy:

* it survives a restart with every number intact;
* a corrupt file cannot stop oMLX from starting, and cannot be quietly planned
  against either;
* records are atomic, bounded, opaque and free of secrets;
* staleness and mismatch live in one function, because two places deciding
  "this measurement still applies" is how a plan gets approved against a
  cluster that changed an hour ago.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta

import pytest

from omlx.cluster.assistant.measurement import (
    MeasurementWarning,
    measure_budget_payloads,
)
from omlx.cluster.assistant.state import (
    MAX_MEASUREMENT_RECORDS,
    MEASUREMENT_FAILED,
    MEASUREMENT_READY,
    MEASUREMENT_RUNNING,
    MEASUREMENT_TTL_SECONDS,
    REFUSAL_MODEL,
    REFUSAL_MTP,
    REFUSAL_NODE_SET,
    REFUSAL_NOT_READY,
    REFUSAL_STALE,
    MeasurementEndpoint,
    MeasurementStore,
    ModelIdentity,
    measurement_refusal,
)

GIB = 1024**3
NOW = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
MEASURED_AT = NOW.isoformat()
MODEL = ModelIdentity(model_size_bytes=40 * GIB, layer_count=20)


def profile(node_id: str, decode_gib_per_s: float) -> dict:
    return {
        "node_id": node_id,
        "rank": 0,
        "decode_weight_bytes_per_second": decode_gib_per_s * GIB,
        "prefill_weight_bytes_per_second": decode_gib_per_s * 2 * GIB,
        "collective_latency_seconds": 0.0003,
        "collective_bandwidth_bytes_per_second": 6.6e9,
        "backend": "ring",
        "measured_at": MEASURED_AT,
        "samples": 5,
    }


def payload(node_id: str, *, capacity_gib: float, usable_gib: float, host: str) -> dict:
    return {
        "node_id": node_id,
        "capacity_bytes": int(capacity_gib * GIB),
        "reserve_bytes": int((capacity_gib - usable_gib) * GIB),
        "usable_bytes": int(usable_gib * GIB),
        "role": "headless",
        "capacity_source": "admission_ceiling",
        "endpoint_host": host,
        "endpoint_port": 8099,
        "disk_free_bytes": 300 * GIB,
        "performance": profile(node_id, 2.0 if node_id == "m5" else 0.8),
    }


def measurement(*, measured_at: str = MEASURED_AT):
    return measure_budget_payloads(
        [
            payload("m5", capacity_gib=107.5, usable_gib=96.75, host="172.16.1.11"),
            payload("m1", capacity_gib=60, usable_gib=54, host="172.16.1.14"),
        ],
        measured_at=measured_at,
    )


def finished_store(tmp_path, *args, **kwargs):
    store = MeasurementStore(tmp_path)
    record = store.start(model=MODEL, node_ids=["m5", "m1"], now=NOW)
    store.finish(
        record.measurement_id,
        measurement=measurement(*args, **kwargs),
        endpoints={
            "m5": MeasurementEndpoint(host="172.16.1.11", port=8099),
            "m1": MeasurementEndpoint(host="172.16.1.14", port=8099),
        },
        now=NOW,
    )
    return store, store.get(record.measurement_id)


# ---------------------------------------------------------------------------
# Round trip and durability
# ---------------------------------------------------------------------------


def test_a_finished_measurement_survives_a_restart(tmp_path) -> None:
    store, record = finished_store(tmp_path)
    assert record is not None
    assert store.path == tmp_path / "cluster" / "assistant-measurements.json"

    reopened = MeasurementStore(tmp_path)
    assert reopened.load_error is None
    loaded = reopened.get(record.measurement_id)
    assert loaded is not None
    assert loaded.state == MEASUREMENT_READY
    assert loaded.measured_at == MEASURED_AT
    assert loaded.model == MODEL
    assert loaded.node_ids == ("m5", "m1")
    assert loaded.node_count == 2
    assert loaded.endpoint_for("m5") == MeasurementEndpoint("172.16.1.11", 8099)
    assert loaded.warnings == ()

    facts = loaded.measurement_set().nodes
    assert [node.node_id for node in facts] == ["m5", "m1"]
    assert facts[0].admission_p05_gib == pytest.approx(96.75)
    assert facts[0].compute is not None
    assert facts[0].endpoint_host == "172.16.1.11"


def test_a_running_job_does_not_survive_a_restart_as_running(tmp_path) -> None:
    """A probe cannot outlive its process; a ghost job would poll forever."""

    store = MeasurementStore(tmp_path)
    record = store.start(model=MODEL, node_ids=["m5"])
    assert record.state == MEASUREMENT_RUNNING
    assert store.active() is not None

    reopened = MeasurementStore(tmp_path)
    recovered = reopened.get(record.measurement_id)
    assert recovered is not None
    assert recovered.state == MEASUREMENT_FAILED
    assert recovered.error is not None
    assert recovered.error.code == "measurement_interrupted"
    assert recovered.error.action == "measure"
    assert reopened.active() is None
    assert reopened.recovered == 1


def test_a_corrupt_file_loads_empty_and_reports_the_error(tmp_path) -> None:
    path = tmp_path / "cluster" / "assistant-measurements.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")

    store = MeasurementStore(tmp_path)

    assert store.get("ms-whatever") is None
    assert store.list() == ()
    assert store.load_error is not None
    assert "could not read" in store.load_error
    # No measurement is available, but the server path still works.
    assert store.start(model=MODEL, node_ids=["m5"]).state == MEASUREMENT_RUNNING


def test_an_unsupported_schema_is_refused_without_raising(tmp_path) -> None:
    path = tmp_path / "cluster" / "assistant-measurements.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 99, "measurements": []}))

    store = MeasurementStore(tmp_path)

    assert store.load_error is not None
    assert "schema" in store.load_error


def test_a_record_whose_facts_are_corrupt_fails_closed(tmp_path) -> None:
    """Stored facts are validated on load, not when a plan already trusts them."""

    path = tmp_path / "cluster" / "assistant-measurements.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "measurements": [
                    {
                        "measurement_id": "ms-20261103T120000Z-0000abcd",
                        "state": "ready",
                        "created_at": MEASURED_AT,
                        "updated_at": MEASURED_AT,
                        "measured_at": MEASURED_AT,
                        "model": MODEL.to_dict(),
                        "requested_node_ids": ["m5"],
                        "endpoints": {},
                        # admission above capacity: NodeFacts refuses it.
                        "nodes": [
                            {
                                "node_id": "m5",
                                "capacity_gib": 10,
                                "admission_p05_gib": 90,
                            }
                        ],
                        "warnings": [],
                        "error": None,
                    }
                ],
            }
        )
    )

    store = MeasurementStore(tmp_path)

    assert store.list() == ()
    assert store.load_error is not None


def test_an_atomic_write_leaves_no_partial_file(tmp_path) -> None:
    store, record = finished_store(tmp_path)
    files = sorted(path.name for path in (tmp_path / "cluster").iterdir())

    assert files == ["assistant-measurements.json"]
    serialized = json.loads(store.path.read_text(encoding="utf-8"))
    assert serialized["schema_version"] == 1
    assert [item["measurement_id"] for item in serialized["measurements"]] == [
        record.measurement_id
    ]


# ---------------------------------------------------------------------------
# Bounds, IDs and content
# ---------------------------------------------------------------------------


def test_pruning_keeps_the_newest_records(tmp_path) -> None:
    store = MeasurementStore(tmp_path)
    ids = [
        store.start(model=MODEL, node_ids=["m5"]).measurement_id
        for _ in range(MAX_MEASUREMENT_RECORDS + 3)
    ]

    kept = [record.measurement_id for record in store.list()]

    assert kept == ids[-MAX_MEASUREMENT_RECORDS:]
    assert MeasurementStore(tmp_path).list()[0].measurement_id == kept[0]


def test_ids_are_unique_opaque_ordered_and_never_reused(tmp_path) -> None:
    store = MeasurementStore(tmp_path)
    ids = [store.start(model=MODEL, node_ids=["m5"]).measurement_id for _ in range(12)]

    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)
    assert all(re.fullmatch(r"ms-\d{8}T\d{6}Z-[0-9a-f]{8}", item) for item in ids)
    # A restarted process must not hand out an ID that is already on disk.
    reopened = MeasurementStore(tmp_path)
    fresh = reopened.start(model=MODEL, node_ids=["m5"]).measurement_id
    assert fresh not in ids


def test_the_file_holds_no_secrets_and_no_ssh_targets(tmp_path) -> None:
    store, record = finished_store(tmp_path)
    text = store.path.read_text(encoding="utf-8")

    stored = json.loads(text)["measurements"][0]
    assert set(stored) == {
        "measurement_id",
        "state",
        "created_at",
        "updated_at",
        "measured_at",
        "model",
        "requested_node_ids",
        "endpoints",
        "nodes",
        "warnings",
        "error",
        # The checkpoint layout the measurement was taken against: shape numbers
        # only, so plan/apply/start cannot disagree when a shard file changes.
        "layout",
        # The per-Mac model lists (id, path, size) the free-space affordance
        # shows. Paths and sizes only — still no SSH target, no key.
        "inventories",
        # The saved per-model levers the plan starts from. Flags, a fraction and
        # a context window — no credential, no path.
        "model_settings",
        # The measured fabric as node-ID pairs and whether the probe answered.
        # Node IDs only — no address and no credential.
        "fabric_links",
        "fabric_measured",
    }
    assert "ssh" not in text
    assert "password" not in text.lower()
    assert "api_key" not in text
    assert record.measurement_id in text


# ---------------------------------------------------------------------------
# Staleness and mismatch
# ---------------------------------------------------------------------------


def test_a_fresh_measurement_answers_its_own_question(tmp_path) -> None:
    _, record = finished_store(tmp_path)
    assert record is not None

    refusal = measurement_refusal(
        record,
        node_ids=["m5", "m1"],
        model=MODEL,
        now=NOW + timedelta(minutes=5),
    )

    assert refusal is None


def test_age_beyond_the_ttl_is_stale(tmp_path) -> None:
    _, record = finished_store(tmp_path)
    assert record is not None

    fresh = measurement_refusal(
        record,
        model=MODEL,
        now=NOW + timedelta(seconds=MEASUREMENT_TTL_SECONDS - 1),
    )
    stale = measurement_refusal(
        record,
        model=MODEL,
        now=NOW + timedelta(seconds=MEASUREMENT_TTL_SECONDS + 1),
    )

    assert fresh is None
    assert stale is not None
    assert stale.reason == REFUSAL_STALE
    assert "15 minutes" in stale.message


def test_a_plan_for_another_node_set_is_refused(tmp_path) -> None:
    _, record = finished_store(tmp_path)
    assert record is not None

    # A subset of the measured Macs is the new normal: the cheap pair is chosen
    # while the measurement still covers every paired Mac.
    assert measurement_refusal(record, node_ids=["m5"], model=MODEL, now=NOW) is None

    refusal = measurement_refusal(
        record, node_ids=["m5", "m9"], model=MODEL, now=NOW
    )

    assert refusal is not None
    assert refusal.reason == REFUSAL_NODE_SET
    assert "m9" in refusal.message


def test_a_plan_for_another_model_is_refused(tmp_path) -> None:
    _, record = finished_store(tmp_path)
    assert record is not None

    refusal = measurement_refusal(
        record,
        model=ModelIdentity(model_size_bytes=50 * GIB, layer_count=20),
        now=NOW,
    )

    assert refusal is not None
    assert refusal.reason == REFUSAL_MODEL
    assert "40.0 GiB" in refusal.message


def test_a_plan_for_another_path_is_refused(tmp_path) -> None:
    store = MeasurementStore(tmp_path)
    record = store.start(
        model=ModelIdentity(model_path="/models/qwen"), node_ids=["m5"], now=NOW
    )
    store.finish(
        record.measurement_id,
        measurement=measurement(),
        endpoints={"m5": MeasurementEndpoint("172.16.1.11", 8099)},
        now=NOW,
    )

    refusal = measurement_refusal(
        store.get(record.measurement_id),
        model=ModelIdentity(model_path="/models/other"),
        now=NOW,
    )

    assert refusal is not None
    assert refusal.reason == REFUSAL_MODEL


def test_a_plan_with_another_mtp_setting_is_refused(tmp_path) -> None:
    _, record = finished_store(tmp_path)
    assert record is not None

    refusal = measurement_refusal(
        record,
        model=ModelIdentity(model_size_bytes=40 * GIB, layer_count=20, mtp=True),
        now=NOW,
    )

    assert refusal is not None
    assert refusal.reason == REFUSAL_MTP
    assert "MTP" in refusal.message


def test_a_running_or_failed_measurement_is_not_planable(tmp_path) -> None:
    store = MeasurementStore(tmp_path)
    record = store.start(model=MODEL, node_ids=["m5"], now=NOW)

    running = measurement_refusal(
        store.get(record.measurement_id), model=MODEL, now=NOW
    )
    store.fail(record.measurement_id, code="probe_failed", message="no route")
    failed = measurement_refusal(store.get(record.measurement_id), model=MODEL, now=NOW)

    assert running is not None and running.reason == REFUSAL_NOT_READY
    assert failed is not None and failed.reason == REFUSAL_NOT_READY
    assert "no route" in failed.message


def test_a_record_with_warnings_keeps_them(tmp_path) -> None:
    store = MeasurementStore(tmp_path)
    record = store.start(model=MODEL, node_ids=["m5"], now=NOW)
    measured = measure_budget_payloads(
        [payload("m5", capacity_gib=107.5, usable_gib=96.75, host="172.16.1.11")],
        measured_at=MEASURED_AT,
    )
    from dataclasses import replace

    measured = replace(
        measured,
        warnings=(
            MeasurementWarning(
                code="link_unmeasured",
                message="the fabric could not be measured",
            ),
        ),
    )
    store.finish(
        record.measurement_id,
        measurement=measured,
        endpoints={"m5": MeasurementEndpoint("172.16.1.11", 8099)},
        now=NOW,
    )

    loaded = MeasurementStore(tmp_path).get(record.measurement_id)

    assert loaded is not None
    assert [warning.code for warning in loaded.warnings] == ["link_unmeasured"]
    assert loaded.measurement_set().warning_messages() == (
        "the fabric could not be measured",
    )
